"""带状态的策略包装，给"停滞来自哪里"的对照实验用（notes/07）。包在 openpi 的 Policy 外面，加四个开关：

1. noise_scale：初始噪声乘一个系数。1 = 原样；0 = 完全确定性（每次从零向量出发去噪）。
2. noise_mode=episode：一集之内每次推理用同一份初始噪声（开头抽一次）。
3. rtc：Real-Time Chunking 的推理时引导（arXiv 2506.07339，算法 1）。新 chunk 的前 H-k 步被拉向上一段没执行完的部分。
   仿真里没有推理延迟，所以 d=0：没有硬冻结的前缀，只有软引导。
4. probe：对同一个观测额外采 N 条 chunk 返回（不执行、用独立的随机数流），用来看"这个状态下的动作分布长什么样"。

全部关闭时与原 Policy 逐位一致：噪声用同一个 key、同一个 shape 生成，采样仍走 model.sample_actions。

client 在观测里带一个 "ctrl" 字典告诉 server 一集的边界：
  {"reset": 一集的第一次推理, "executed": 上一段执行了几步, "probe": 额外采几条}
"""
import time

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi_client import base_policy as _base_policy


def rtc_weights(horizon: int, executed: int, delay: int = 0) -> np.ndarray:
    """RTC 的软掩码 W（论文式 5）。下标 i 是新 chunk 里的第 i 步。
    i < delay：1（推理期间必然执行，硬约束）；delay <= i < H-s：指数衰减；i >= H-s：0（上一段没有对应动作）。
    """
    i = np.arange(horizon)
    n = horizon - executed                      # 上一段剩下几步可以对齐
    c = (n - i) / (n - delay + 1)
    w = c * np.expm1(c) / (np.e - 1)
    w = np.where(i < delay, 1.0, w)
    return np.where(i >= n, 0.0, w).astype(np.float32)


def sample_actions_rtc(model, observation, noise, prev, weights, *, num_steps=10, beta=5.0):
    """pi0.py sample_actions 加上 RTC 引导项。时间约定沿用 openpi：t=1 是噪声，t=0 是动作（论文里 τ = 1 - t）。

    每个去噪步：
      v      = 模型速度（≈ noise - actions）
      x0_hat = x_t - t * v                          # 从当前点一步外推到的干净动作
      g      = (W ⊙ (prev - x0_hat))ᵀ ∂x0_hat/∂x_t  # 让外推结果靠近上一段的方向（一次 VJP）
      v'     = v - min(beta, (τ² + t²) / (τ t)) * g
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def velocity(x_t, time):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=pos, kv_cache=kv_cache, adarms_cond=[None, adarms_cond]
        )
        return model.action_out_proj(suffix_out[:, -model.action_horizon :])

    def step(carry):
        x_t, time = carry

        def denoise(x):
            v = velocity(x, time)
            return x - time * v, v

        x0_hat, vjp_fn, v_t = jax.vjp(denoise, x_t, has_aux=True)
        (g,) = vjp_fn(((prev - x0_hat) * weights).astype(x0_hat.dtype))
        tau = 1.0 - time
        gain = jnp.minimum(beta, (tau**2 + time**2) / jnp.maximum(tau * time, 1e-6))
        return x_t + dt * (v_t - gain * g).astype(x_t.dtype), time + dt

    def cond(carry):
        return carry[1] >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0


class PolicyExt(_base_policy.BasePolicy):
    def __init__(self, policy, *, noise_scale=1.0, noise_mode="fresh", rtc=False, rtc_beta=5.0, rtc_diag=False, env_action_dim=7):
        assert noise_mode in ("fresh", "episode"), noise_mode
        self._p = policy
        model = policy._model
        self._H, self._D = model.action_horizon, model.action_dim
        self._noise_scale, self._noise_mode = float(noise_scale), noise_mode
        self._rtc, self._rtc_beta, self._rtc_diag = bool(rtc), float(rtc_beta), bool(rtc_diag)
        self._env_dim = env_action_dim
        # 引导只作用在真实动作维度上（LIBERO 7 维；模型输出 32 维，其余是 padding）
        self._dim_mask = (np.arange(self._D) < env_action_dim).astype(np.float32)
        self._probe_rng = jax.random.key(12345)   # probe 用独立的流，不影响被执行动作的随机数
        self._prev = None        # 上一次的模型空间输出 (1, H, D)
        self._ep_noise = None
        if self._rtc:
            graphdef, self._state = nnx.split(model)   # 和 nnx_utils.module_jit 同一个做法：冻结参数后用纯 jax.jit
            self._sample_rtc = jax.jit(lambda state, *a, **k: sample_actions_rtc(nnx.merge(graphdef, state), *a, **k))

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        ctrl = obs.pop("ctrl", None) or {}
        p = self._p
        if ctrl.get("reset"):
            self._prev, self._ep_noise = None, None

        # --- 以下到 Observation.from_dict 与 Policy.infer 相同 ---
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = p._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        p._rng, key = jax.random.split(p._rng)
        observation = _model.Observation.from_dict(inputs)

        noise = jax.random.normal(key, (1, self._H, self._D))   # sample_actions 内部就是这一行
        if self._noise_mode == "episode":
            if self._ep_noise is None:
                self._ep_noise = noise
            noise = self._ep_noise
        if self._noise_scale != 1.0:
            noise = noise * self._noise_scale

        start = time.monotonic()
        k = int(ctrl.get("executed", 0))
        guided = self._rtc and self._prev is not None and 0 < k < self._H
        if guided:
            prev = jnp.concatenate([self._prev[:, k:], jnp.zeros((1, k, self._D), self._prev.dtype)], axis=1)
            w = rtc_weights(self._H, k)[None, :, None] * self._dim_mask[None, None, :]
            raw = self._sample_rtc(
                self._state, observation, noise, prev, jnp.asarray(w),
                num_steps=p._sample_kwargs.get("num_steps", 10), beta=self._rtc_beta,
            )
        else:
            raw = p._sample_actions(key, observation, noise=noise, **p._sample_kwargs)
        raw.block_until_ready()
        infer_ms = (time.monotonic() - start) * 1000
        self._prev = raw

        state0 = np.asarray(inputs["state"][0])
        outputs = p._output_transform({"state": state0.copy(), "actions": np.asarray(raw[0])})
        outputs["policy_timing"] = {"infer_ms": infer_ms}
        outputs["guided"] = bool(guided)
        if guided and self._rtc_diag:
            # 同一观测、同一份噪声，不加引导再采一次：两者到上一段的距离之差就是引导的净效果
            plain = np.asarray(p._sample_actions(key, observation, noise=noise, **p._sample_kwargs))[0]
            n_ov, tgt, got = self._H - k, np.asarray(prev)[0], np.asarray(raw)[0]
            def dist(x, steps):   # 平移+旋转 6 维上的均方根误差（模型的归一化空间）
                return float(np.sqrt(np.mean((x[:steps, :6] - tgt[:steps, :6]) ** 2)))
            def agree(x):         # 夹爪维符号一致的比例
                return float(np.mean(np.sign(x[:n_ov, self._env_dim - 1]) == np.sign(tgt[:n_ov, self._env_dim - 1])))
            outputs["rtc_diag"] = np.array(
                [dist(got, n_ov), dist(plain, n_ov), dist(got, 1), dist(plain, 1), agree(got), agree(plain)], np.float32
            )

        n = int(ctrl.get("probe", 0))
        if n > 0:
            self._probe_rng, pk = jax.random.split(self._probe_rng)
            # from_dict 已经把 inputs 里的图像原地转成 float，这里直接沿 batch 维复制
            obs_n = _model.Observation.from_dict(jax.tree.map(lambda x: jnp.repeat(x, n, axis=0), inputs))
            raw_n = np.asarray(p._sample_actions(pk, obs_n, **p._sample_kwargs))
            outputs["probe_actions"] = np.stack(
                [p._output_transform({"state": state0.copy(), "actions": raw_n[i]})["actions"] for i in range(n)]
            )
        return outputs

    @property
    def metadata(self):
        return self._p.metadata
