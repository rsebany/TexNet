"""Central configuration for ILD-TexNet.

Only the architecture-level settings and the synthetic training-demo settings
live here. The full training schedule used for the published benchmark is
maintained in the companion research repository, not in this package. Every
value can be overridden with an ``ILDEXNET_*`` environment variable.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Architecture defaults (as reported for the proposed model)
# ---------------------------------------------------------------------------
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

# Ablation knobs kept so the design decisions remain auditable.
USE_MULTISCALE = os.environ.get("ILDEXNET_USE_MULTISCALE", "1") == "1"
USE_ATTN_POOL = os.environ.get("ILDEXNET_USE_ATTN_POOL", "1") == "1"
BLOCK = os.environ.get("ILDEXNET_BLOCK", "se_mbconv")  # se_mbconv | bottleneck

# ---------------------------------------------------------------------------
# Synthetic training demo (no real data is bundled with this package). The
# demo only proves the forward/backward/optimiser path works end to end.
# ---------------------------------------------------------------------------
SEED = int(os.environ.get("ILDEXNET_SEED", "42"))
EPOCHS = int(os.environ.get("ILDEXNET_EPOCHS", "30"))
BATCH_SIZE = int(os.environ.get("ILDEXNET_BATCH", "128"))
LR = float(os.environ.get("ILDEXNET_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("ILDEXNET_WD", "1e-4"))
LABEL_SMOOTHING = float(os.environ.get("ILDEXNET_LABEL_SMOOTHING", "0.03"))
CLASS_WEIGHT_POWER = float(os.environ.get("ILDEXNET_CLASS_WEIGHT_POWER", "0.75"))
DEMO_SAMPLES = int(os.environ.get("ILDEXNET_DEMO_SAMPLES", "2048"))

OUT_DIR = os.environ.get(
    "ILDEXNET_OUT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "outputs"),
)


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
    """Resolve the compute device, falling back to CPU when CUDA is absent."""
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)