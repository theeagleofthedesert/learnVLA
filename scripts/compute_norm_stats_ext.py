"""openpi scripts/compute_norm_stats.py 的包装：带上 train_configs 的配置和子集 patch。
  cd $OPENPI && uv run python $ROOT/scripts/compute_norm_stats_ext.py --config-name pi05_libero10_lora_n10
同时把子集用到的 episode 索引写到 assets/<config>/<asset_id>/episodes.json 留档。
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import train_configs  # noqa: F401
import openpi.training.config as _config
import tyro
OPENPI = pathlib.Path(__file__).resolve().parents[1] / "external/openpi"
sys.path.insert(0, str(OPENPI / "scripts"))
import compute_norm_stats


def main(config_name: str, max_frames: int | None = None):
    cfg = _config.get_config(config_name)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    eps = getattr(data_cfg, "episodes", None)
    out = cfg.assets_dirs / data_cfg.asset_id
    out.mkdir(parents=True, exist_ok=True)
    if eps is not None:
        json.dump({"repo_id": data_cfg.repo_id, "n_episodes": len(eps), "episodes": list(eps)}, open(out / "episodes.json", "w"))
        print(f"{config_name}: {len(eps)} episodes -> {out / 'episodes.json'}")
    compute_norm_stats.main(config_name, max_frames)


if __name__ == "__main__":
    tyro.cli(main)
