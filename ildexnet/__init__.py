"""ILD-TexNet: a compact texture network for interstitial lung disease (ILD)
pattern classification from 32x32 high-resolution CT texture patches.

This package ships only the *proposed architecture*: the multi-scale dilated
stem, the dense SE-MBConv stages, and the attention pooling head. Training,
evaluation and the patient-disjoint leakage benchmark that motivated the
design live in the companion repository, not here.

The model is implemented from scratch in PyTorch. No pretrained weights are
loaded anywhere in this package.
"""

__version__ = "0.1.0"

from ildexnet.models.ildexnet import ILDTexNet  # noqa: E402,F401

__all__ = ["ILDTexNet", "__version__"]