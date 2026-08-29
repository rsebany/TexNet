"""ILD-TexNet: compact texture network for interstitial lung disease (ILD)
pattern classification from 32x32 high-resolution CT texture patches.

This package ships only the proposed architecture. Training, evaluation and
the patient-disjoint leakage benchmark live in the companion repository.
No pretrained weights are loaded anywhere.
"""

__version__ = "0.1.0"

from ildexnet.models.ildexnet import ILDTexNet  # noqa: E402,F401

__all__ = ["ILDTexNet", "__version__"]