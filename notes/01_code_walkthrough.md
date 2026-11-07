# notes/01 openpi 代码走读：从观测到动作

> 读者：lwang。顺着一次 `client.infer(obs)` 的调用链读 openpi，每一站给代码位置、摘录和解释。
> 文件路径相对 `external/openpi/`，行号对应 2026-11-07 clone 的 main 分支。

## 0. 总览

```
eval_libero.py                 obs dict: image(224,224,3) uint8, wrist_image, state(8,), prompt
    │ websocket
serve_policy_ext.py → Policy.infer()                                   [src/openpi/policies/policy.py:66]
    ├─ 输入 transforms（按顺序）                                        [policy_config.py:78-90]
    │    LiberoInputs        键名对齐、补第三张空图                        [policies/libero_policy.py:31]
    │    Normalize           state 用 q01/q99 归一到 [-1,1]               [transforms.py:115]
    │    ResizeImages(224)   已经是 224，跳过
    │    TokenizePrompt      prompt + 离散化 state → token ids            [transforms.py:248, models/tokenizer.py:20]
    │    PadStatesAndActions state 8 维 → 32 维                           [transforms.py:328]
    ├─ Observation.from_dict  uint8 图像 → [-1,1] float                   [models/model.py:110]
    ├─ model.sample_actions   flow matching 10 步去噪 → (1,10,32)          [models/pi0.py:217]
    └─ 输出 transforms
         Unnormalize         [-1,1] → 原始动作尺度                        [transforms.py:149]
         LiberoOutputs       取前 7 维                                    [libero_policy.py:94]
```

## 1. 配置：`pi05_libero` 是什么

`src/openpi/training/config.py:743-763`：

```python
TrainConfig(
    name="pi05_libero",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
)
```

要点：
- **action_horizon=10**，不是 π0 默认的 50（`pi0_config.py:26`）。LIBERO 20 Hz，所以一个 chunk 是 0.5 秒。客户端默认每 5 步重规划。
- **discrete_state_input=False**：π0.5 原生把 state 离散化塞进文本 prompt（`pi0_config.py:30-32` 的注释），但 LIBERO 这个配置关掉了。所以 state 在哪里进模型？答案：**哪儿都不进**。看 §4：pi05 分支的 `embed_suffix` 不加 state token，`TokenizePrompt` 的 `discrete_state_input=False` 也不把 state 放进 prompt。这个 checkpoint 只看两张图和语言。这是个值得记住的事实，第 2 周看本体感知相关失败时要想到。
- **这是全量微调，不是 LoRA**：`paligemma_variant` 用默认 `"gemma_2b"`。v1 计划以为官方是 LoRA，错了。LoRA 版本要照 `config.py:679-699` 的 `pi0_libero_low_mem_finetune` 另写一个配置，见 §6。
- batch 256、30k 步、lr 5e-5 常数（warmup 10k 后 decay 到同一个值，等于没 decay）、EMA 0.999。
- 从 `pi05_base` 初始化，不是从头训。

## 2. 数据变换：LiberoInputs / LiberoOutputs

`src/openpi/policies/libero_policy.py:31-91`。核心就是把 client 发来的键名对到模型的键名，并把不存在的第三张图补零：

```python
inputs = {
    "state": data["observation/state"],
    "image": {
        "base_0_rgb": base_image,
        "left_wrist_0_rgb": wrist_image,
        "right_wrist_0_rgb": np.zeros_like(base_image),     # 没有右腕相机，补零
    },
    "image_mask": {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,   # pi0/pi05: mask 掉；pi0-FAST: 不 mask
    },
}
```

模型固定吃三路图像（`pi0_config.py:62-66`），mask=False 的那路在 attention 里被屏蔽（§4 `embed_prefix` 的 `input_mask`）。

输出端 `LiberoOutputs` 只做一件事：`data["actions"][..., :7]`。模型内部动作维度是 32（`action_dim=32`），LIBERO 只用前 7 维，其余是 padding。

训练时还多一层 `RepackTransform`（`config.py:301-313`），把 LeRobot 数据集的列名（`image`, `wrist_image`, `state`, `actions`）改成和推理时一样的 `observation/image` 等。目的是让训练和推理走完全相同的 transform 链。

## 3. 归一化：state 和 action 怎么缩放

`src/openpi/transforms.py:115-181`。π0.5 用**分位数归一化**（`config.py:187`：`use_quantile_norm = model_type != PI0`）：

```python
def _normalize_quantile(self, x, stats):
    q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

把 1% 和 99% 分位映射到 -1 和 +1，超出的部分不裁剪。反归一化 `_unnormalize_quantile` 是逆运算，并且如果 stats 维度比 x 少，多出来的维度原样返回（`transforms.py:179-180`），这就是 32 维里 padding 部分不受影响的原因。

norm stats 从哪来：
- 训练前 `uv run scripts/compute_norm_stats.py --config-name pi05_libero` 遍历数据集算均值、方差、分位数，存到 `assets/<asset_id>/norm_stats.json`。
- **推理时从 checkpoint 目录的 `assets/` 读**，不从配置读（`policy_config.py:66-70`），保证用的是训练时那一份。
- 换数据源最常见的错：忘了重算，或者 asset_id 对不上。`docs/norm_stats.md` 讲了怎么复用预训练的 stats。

### 3.1 `discrete_state_input=False` 到底意味着什么

`state` 是本体感知：末端位置 3 + 轴角 3 + 夹爪 2 = 8 维，由 `LiberoInputs` 从 `observation/state` 取出（`policies/libero_policy.py:57`）。它进模型有三种可能：

| 配置 | 路径 | 代码 |
|---|---|---|
| π0 | 连续向量 → `state_proj` 线性层 → 1 个 token 放在 suffix 最前面 | `models/pi0.py:152-155` |
| π0.5 默认 | 每维量化到 256 个 bin → 文字 `Task: ..., State: 128 97 ...;` → 进 prompt | `models/tokenizer.py:24-28`；`Pi0Config.__post_init__` 让 `discrete_state_input` 默认等于 `pi05`（`pi0_config.py:40-41`） |
| `pi05_libero` | **不进网络**：pi05 没有 `state_proj`，`TokenizePrompt` 收到 `state=None` | `transforms.py:256-260`，`config.py:745` 显式 `discrete_state_input=False` |

所以这个 checkpoint 是纯视觉 + 语言策略。state 仍会被 `Normalize` 归一化、在 `Policy.infer` 里原样放进 outputs（`policy.py:93`），但那只是透传。想验证：把 client 发的 state 全置零，成功率应该不变——这是一个 10 分钟的实验。

## 4. 模型：π0.5 的 forward

`src/openpi/models/pi0.py`。结构是**两个 Gemma 共享一条 attention 序列**：PaliGemma（2B，吃图像和文本）+ action expert（300M，吃噪声动作）。

### 4.1 构造（`pi0.py:66-104`）

```python
llm = nnx_bridge.ToNNX(_gemma.Module(configs=[paligemma_config, action_expert_config], adarms=config.pi05))
img = nnx_bridge.ToNNX(_siglip.Module(variant="So400m/14", ...))
self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width)   # 32 → 1024
if config.pi05:
    self.time_mlp_in  = nnx.Linear(width, width)   # 时间步 → adaRMSNorm 条件
    self.time_mlp_out = nnx.Linear(width, width)
else:
    self.state_proj = nnx.Linear(config.action_dim, width)   # π0 才有 state token
self.action_out_proj = nnx.Linear(width, config.action_dim)  # 1024 → 32
```

π0 和 π0.5 的两处差别（`pi0_config.py:30-32`）：π0.5 没有 state token，时间步通过 adaRMSNorm 注入而不是拼到 token 上。

### 4.2 prefix：图像 + 文本（`pi0.py:106-138`）

```python
for name in obs.images:
    image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)   # SigLIP，每张图 256 个 token
    tokens.append(image_tokens)
    input_mask.append(repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
    ar_mask += [False] * image_tokens.shape[1]      # 图像 token 之间全连接
tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
ar_mask += [False] * tokenized_inputs.shape[1]      # 文本和图像之间也全连接
```

`ar_mask` 的语义在 `make_attn_mask`（`pi0.py:19-45`）：False 表示和前一个 token 同一个块，True 表示开新块。prefix 全是 False，所以图像和文本之间是双向注意力。

### 4.3 suffix：噪声动作 + 时间步（`pi0.py:140-187`）

```python
action_tokens = self.action_in_proj(noisy_actions)                       # (b, 10, 1024)
time_emb = posemb_sincos(timestep, 1024, min_period=4e-3, max_period=4.0)
if self.pi05:
    time_emb = swish(self.time_mlp_out(swish(self.time_mlp_in(time_emb))))
    adarms_cond = time_emb                 # 送给 action expert 每层的 adaRMSNorm
ar_mask += [True] + [False] * (self.action_horizon - 1)   # 动作块是一个新块：prefix 看不到它，它看得到 prefix
```

关键的 mask 设计：`[True, False × 9]` 意味着 10 个动作 token 互相全连接、能看到全部 prefix，但 prefix 看不到动作。所以推理时 prefix 的 KV 可以缓存，只有 suffix 重复算。

### 4.4 训练 loss：flow matching（`pi0.py:189-215`）

```python
noise = jax.random.normal(noise_rng, actions.shape)
time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001   # 偏向 t 接近 1（噪声多的一端）
x_t = time * noise + (1 - time) * actions          # 线性插值：t=1 纯噪声，t=0 真动作
u_t = noise - actions                              # 目标速度场
...
v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
return jnp.mean(jnp.square(v_t - u_t), axis=-1)   # MSE，每个时间步一个数
```

注意注释（`pi0.py:224-225`）：这里 t=1 是噪声、t=0 是数据，和 π0 论文相反。

### 4.5 采样（`pi0.py:217-279`）

```python
dt = -1.0 / num_steps                                    # 默认 10 步
noise = jax.random.normal(rng, (batch, action_horizon, action_dim))
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)   # prefix 只算一次
def step(carry):
    x_t, time = carry
    ... suffix 过 action expert，用 kv_cache ...
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
    return x_t + dt * v_t, time + dt                     # 欧拉积分
x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
```

10 步欧拉，从 t=1 走到 t=0。`num_steps` 就是 §1.3 要扫的"去噪步数"，`serve_policy_ext.py` 通过 `sample_kwargs` 传进去。随机性只来自 `noise`，`Policy.infer` 每次调用 split 一次 rng（`policy.py:71`），所以同一观测重复调用结果不同。这是 §1.5 随机性实验的依据。

## 5. 推理封装：Policy.infer 和 client 主循环

`src/openpi/policies/policy.py:66-110`：
```python
inputs = self._input_transform(inputs)                    # §2 §3 的链
inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)   # 加 batch 维
self._rng, sample_rng = jax.random.split(self._rng)
outputs = {"state": inputs["state"], "actions": self._sample_actions(sample_rng, observation, **sample_kwargs)}
outputs = self._output_transform(outputs)                 # Unnormalize + LiberoOutputs
outputs["policy_timing"] = {"infer_ms": ...}
```

`scripts/eval_libero.py` 的主循环（改自 `examples/libero/main.py:97-150`）：
```python
if not action_plan:                       # 上一个 chunk 用完了
    chunk = client.infer(element)["actions"]        # (10, 7)
    action_plan.extend(chunk[:args.replan_steps])   # 只排队前 replan_steps 步
action = action_plan.popleft()
obs, reward, done, info = env.step(action.tolist())
```

`replan_steps=5` 时：预测 10 步，执行 5 步，丢掉后 5 步，重新观测。`replan_steps=10` 就是整个 chunk 开环执行。这是 §1.3 的第一个旋钮。

## 6. LoRA：第 2 周要用的部分

`pi05_libero` 本身是全量微调。LoRA 版本要照 `config.py:679-699` 写一个 `pi05_libero_lora`：
- `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")`
- `freeze_filter=<同一个 Pi0Config>.get_freeze_filter()`：`pi0_config.py:88-117`，用正则 `.*llm.*` 冻结 Gemma 参数，再用 `nnx.Not(".*lora.*")` 把 LoRA 参数排除出冻结集。
- `ema_decay=None`。

LoRA 挂在哪：`gemma.py:88-107`。2B 主干 attn 和 ffn 各 rank 16，alpha 16；300M action expert rank 32，alpha 32。实现在 `models/lora.py`，是对 einsum 权重加 `A @ B` 低秩项。

训练循环 `scripts/train.py:137-192`：冻结参数转 bf16 只读（`:104`），每步算 loss、更新、EMA；`save_interval=1000`、`keep_period=5000` 控制存哪些 checkpoint（`config.py:516-518`）。

## 7. 环境侧：LIBERO 怎么判成功、初始状态是什么

`third_party/libero/libero/libero/envs/env_wrapper.py`：
- `ControlEnv.__init__`（:17-27）：`controller="OSC_POSE"`, `control_freq=20`, 两个相机 `agentview` 和 `robot0_eye_in_hand`。
- `set_init_state(init_state)`（:136-145）：直接把一个 MuJoCo 状态向量写进 sim，`sim.forward()`，然后重新生成观测。50 个初始状态存在 `init_files/<suite>/<task>.pruned_init`，是 benchmark 发布时固定的，所有论文的 50 episode 用的是同一批。
- `check_success()`（:103）→ `env._check_success()`：按 BDDL 文件里的 goal 谓词判断（例如 `On(bowl, plate)`）。`env.step` 返回的 `done` 就是它。

## 8. 换数据源 / 真机时的坑（本文档小结）

1. `action_horizon` 和控制频率绑定；换 10 Hz 机器人要重新想 chunk 长度。
2. norm stats 必须用新数据重算，并确认 checkpoint 里 `assets/` 下是新的。
3. 这个 LIBERO 配置不用 state。换真机如果需要本体感知，要开 `discrete_state_input=True` 并保证 `max_token_len`（pi05 默认 200）够放。
4. 三路图像的 mask：没有的相机补零并 mask=False，不要用随机图。
5. 训练和推理必须走同一条 transform 链，`RepackTransform` 只负责改键名。
