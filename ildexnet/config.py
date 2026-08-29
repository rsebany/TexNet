"""Central configuration for ILD-TexNet.

Every value can be overridden with an ``ILDEXNET_*`` environment variable.
"""

from __future__ import annotations

import os

# Architecture defaults
NUM_CLASSES = int(os.environ.get("ILDEXNET_NUM_CLASSES", "6"))
IN_CHANNELS = int(os.environ.get("ILDEXNET_IN_CHANNELS", "1"))
PATCH_SIZE = int(os.environ.get("ILDEXNET_PATCH_SIZE", "32"))

STEM_CH = int(os.environ.get("ILDEXNET_STEM_CH", "48"))
GROWTH = int(os.environ.get("ILDEXNET_GROWTH", "24"))


def _layers():
    raw = os.environ.get("ILDEXNET_LAYERS", "")
    if raw:
        return tuple(int(v) for v in raw.split(",") if v.strip())
    return (4, 4, 4)


LAYERS = _layers()
EXPAND = float(os.environ.get("ILDEXNET_EXPAND", "4"))
REDUCTION = float(os.environ.get("ILDEXNET_REDUCTION", "0.5"))
DROPOUT = float(os.environ.get("ILDEXNET_DROPOUT", "0.2"))

# Ablation knobs keep the design decisions auditable.
USE_MULTISCALE = os.environ.get("ILDEXNET_USE_MULTISCALE", "1") == "1"
USE_ATTN_POOL = os.environ.get("ILDEXNET_USE_ATTN_POOL", "1") == "1"
BLOCK = os.environ.get("ILDEXNET_BLOCK", "se_mbconv")  # se_mbconv | bottleneck

SEED = int(os.environ.get("ILDEXNET_SEED", "42"))


def build_kwargs(command_line: dict | None = None) -> dict:
    """Model constructor kwargs from environment + explicit CLI overrides."""
    kw = {
        "num_classes": NUM_CLASSES,
        "in_ch": IN_CHANNELS,
        "stem_ch": STEM_CH,
        "growth": GROWTH,
        "layers": LAYERS,
        "expand": EXPAND,
        "reduction": REDUCTION,
        "dropout": DROPOUT,
        "use_multiscale": USE_MULTISCALE,
        "block": BLOCK,
        "use_attn_pool": USE_ATTN_POOL,
    }
    if command_line:
        kw.update({k: v for k, v in command_line.items() if v is not None})
    return kw


def device(name: str = "auto"):
    """Resolve compute device, falling back to CPU when CUDA is absent."""
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)