"""YAML-backed configuration access for src.

Configuration values live in `src/configs/*.yaml`.  This package only loads
those files and exposes compatibility names used by the pipeline and analysis
scripts.
"""
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"


def load_config_file(path=None):
    """Load a src YAML config file."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def flatten_config(config):
    """Flatten dataset/pipeline/runtime sections for CLI override handling."""
    flat = {}
    for section in ("dataset", "pipeline", "runtime"):
        flat.update(config.get(section, {}) or {})
    return flat


_CONFIG = load_config_file()

MODEL_CFG = dict(_CONFIG.get("model", {}) or {})
BACKBONE = MODEL_CFG["backbone"]
IMG_SIZE = int(MODEL_CFG["img_size"])
RESIZE_MASK = int(MODEL_CFG["resize_mask"])
ROTATION_ANGLES = list(MODEL_CFG["rotation_angles"])

METRICS = list((_CONFIG.get("evaluation", {}) or {})["metrics"])
DATASET_CATEGORIES = {
    name: list(categories)
    for name, categories in (_CONFIG.get("categories", {}) or {}).items()
}

MEMORY_BANK_KWARGS = dict(_CONFIG.get("memory_bank", {}) or {})

SCORING_POLICY = dict(_CONFIG.get("scoring_policy", {}) or {})
GATE_Q_LOW = float(SCORING_POLICY["gate_q_low"])
GATE_Q_HIGH = float(SCORING_POLICY["gate_q_high"])
RESIDUAL_GAMMA = float(SCORING_POLICY["residual_gamma"])

DEFAULT_CFG = dict(_CONFIG.get("pipeline", {}) or {})
MLP_CFG = dict(_CONFIG.get("mlp", {}) or {})
NOVELTY_RATIO = float(DEFAULT_CFG.get("novelty_ratio", 0.0))


def make_memory_bank(device):
    """Create a canonical MemoryBank from YAML memory-bank settings."""
    from src.memory.bank import MemoryBank

    return MemoryBank(device=device, **MEMORY_BANK_KWARGS)
