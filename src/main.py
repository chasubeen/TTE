"""src — entry point for Stable Test-Time Memory Expansion (Score-then-Expansion).

Each test batch is scored against the pre-absorption memory, then absorbed:
  ① Scoring    Ranking-Preserving Adaptation   (Req-C, src/scorer/)
  ② Selection  Memory-independent Gate e<τ_low (Req-B, src/selector/)
  ③ Expansion  reservoir / append              (Req-A, src/memory/)

Run from the repository root (the directory that contains `src/`):

  # Current method (reservoir + Memory-independent gate + Ranking-Preserving Adaptation)
  python -m src.main --seeds 1 2 3 --memory-policy reservoir --scoring-mode context_discount
  python -m src.main --seeds 1 2 3 --dataset VisA_pytorch \
      --memory-policy reservoir --scoring-mode context_discount

  # Static AnomalyDINO baseline (omit the two policy flags)
  python -m src.main --seeds 1 2 3

  # Point to the dataset root explicitly (default: <repo>/data)
  python -m src.main --data-path /path/to/data --categories bottle cable

Config defaults live in src/configs/default.yaml (CLI flags override).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path

from src.configs import DEFAULT_CFG, flatten_config, load_config_file
from src.pipeline.runner import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_yaml_overrides(path=None):
    """Load YAML config and return flat runtime/dataset/pipeline values."""
    return flatten_config(load_config_file(path))


def build_parser():
    p = argparse.ArgumentParser(
        description="src: selective test-time memory expansion + single-memory scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Run-time
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--shot", type=int, default=None)
    p.add_argument("--dataset", default=None,
                   choices=["MVTecAD", "VisA_pytorch"])
    p.add_argument("--data-path", default=None,
                   help="Default: <project>/data")
    p.add_argument("--categories", nargs="+", default=None,
                   help="Default: all categories of the dataset")
    p.add_argument("--device", default=None)
    p.add_argument("--config", default=None,
                   help="Optional YAML config (CLI flags override)")

    # Pipeline knobs (all default to DEFAULT_CFG; CLI override allowed)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--tau-ratio-low", type=float, default=None)
    p.add_argument("--novelty-ratio", type=float, default=None,
                   help="Req-A novelty filter ratio (0=disabled). Default: 0.0")
    p.add_argument("--scoring-mode", default=None,
                   choices=["baseline", "context_discount"],
                   help="Scoring: baseline (d(q,M)) or context_discount (CURRENT — "
                        "Ranking-Preserving Adaptation; use with --memory-policy reservoir).")
    p.add_argument("--memory-policy", default=None,
                   choices=["append", "reservoir"],
                   help="Memory policy: append (legacy gap-priority) or reservoir (CURRENT method).")
    return p


def apply_runtime_cfg(args, yaml_cfg):
    """Merge runtime/dataset YAML defaults into argparse namespace."""
    if args.dataset is None:
        args.dataset = yaml_cfg.get("name", "MVTecAD")
    if args.shot is None:
        args.shot = int(yaml_cfg.get("shot", 1))
    if args.device is None:
        args.device = yaml_cfg.get("device", "cuda:0")
    if args.data_path is None and yaml_cfg.get("data_path") is not None:
        args.data_path = yaml_cfg["data_path"]


def merge_pipeline_cfg(args, yaml_cfg=None):
    """Merge precedence: DEFAULT_CFG → YAML → CLI flags."""
    cfg = dict(DEFAULT_CFG)

    # YAML overrides
    if yaml_cfg is None:
        yaml_cfg = _load_yaml_overrides(args.config) if args.config else {}
    if yaml_cfg:
        for k in DEFAULT_CFG:
            if k in yaml_cfg:
                cfg[k] = yaml_cfg[k]

    # CLI overrides
    cli_keys = {
        "batch_size": args.batch_size,
        "budget": args.budget,
        "tau_ratio_low": args.tau_ratio_low,
        "novelty_ratio": args.novelty_ratio,
        "scoring_mode": args.scoring_mode,
        "memory_policy": args.memory_policy,
    }
    for k, v in cli_keys.items():
        if v is not None:
            cfg[k] = v
    return cfg


def main():
    args = build_parser().parse_args()
    yaml_cfg = _load_yaml_overrides(args.config)
    apply_runtime_cfg(args, yaml_cfg)

    # Resolve defaults relative to project root
    project_root = Path(os.path.dirname(os.path.abspath(__file__)))
    if args.data_path is None:
        args.data_path = str(project_root / "data")

    # Load + merge pipeline config (DEFAULT_CFG → YAML → CLI)
    pipeline_cfg = merge_pipeline_cfg(args, yaml_cfg)

    logger.info(f"Pipeline config: {pipeline_cfg}")
    run(args, cfg=pipeline_cfg)


if __name__ == "__main__":
    main()
