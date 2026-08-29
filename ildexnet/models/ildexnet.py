"""ILD-TexNet: the proposed compact texture network.

Three components, each answering a failure measured in the patient-disjoint
confusion matrix of the first benchmark run rather than chosen for novelty:

* **Multi-scale dilated stem** -- parallel 3x3 convolutions at dilations 1,
  2, 3 (receptive fields 3, 5, 7 px). Patterns such as Consolidation collapse
  into Fibrosis/GroundGlass when the classifier commits to a single
  first-layer receptive field; those patterns differ by the spatial frequency
  of their texture, not by mean attenuation.

* **Dense SE-MBConv stages** -- each block concatenates its input to its
  output (DenseNet) while computing the output with an inverted-residual
  depthwise-separable convolution carrying squeeze-excitation (MobileNetV3).
  Dense concatenation keeps the high-frequency stem features reachable by the
  classifier at every depth, and the separable form keeps that reuse
  affordable.

* **Attention pooling head** -- a learned spatial weighting so the classifier
  can ignore the off-class tissue inside low-purity training patches.

``use_multiscale``, ``block`` and ``use_attn_pool`` exist so the design can be
ablated one component at a time.
"""

from __future__ import annotations

import math

import torch.nn as nn

from ildexnet.models.components import (
    AttentionPool2d,
    DenseLayer,
    DenseSEMBConv,
    MultiScaleStem,
    _init_weights,
)


class ILDTexNet(nn.Module):
    """Multi-scale dilated stem, dense SE-MBConv stages, attention pooling.

    Args:
        num_classes: number of output classes.
        in_ch: input channels (``1`` for single-slice HU-normalised patches).
        stem_ch: channels produced by the stem.
        growth: per-block channel growth of the dense stages.
        layers: number of blocks in each of the three stages.
        expand: expansion factor of the inverted-residual blocks.
        reduction: channel compression applied between stages.
        dropout: dropout applied before the classifier head.
        use_multiscale: replace the dilated stem with a plain 3x3 conv.
        block: dense block type -- ``"se_mbconv"`` (proposed) or
            ``"bottleneck"`` (control that drops the separable convolution and
            squeeze-excitation while keeping dense connectivity).
        use_attn_pool: replace the attention pooling with adaptive avg pooling.
    """

    def __init__(self, num_classes=6, in_ch=1, stem_ch=48, growth=24,
                 layers=(4, 4, 4), expand=4, reduction=0.5, dropout=0.2,
                 use_multiscale=True, block="se_mbconv", use_attn_pool=True):
        super().__init__()
        self.stem = (MultiScaleStem(in_ch, stem_ch) if use_multiscale
                     else nn.Sequential(
                         nn.Conv2d(in_ch, stem_ch, 3, padding=1, bias=False),
                         nn.BatchNorm2d(stem_ch),
                         nn.LeakyReLU(0.1, inplace=True)))
        stages, cin = [], stem_ch
        for si, n in enumerate(layers):
            for _ in range(n):
                stages.append(DenseSEMBConv(cin, growth, expand)
                              if block == "se_mbconv"
                              else DenseLayer(cin, growth))
                cin += growth
            if si < len(layers) - 1:
                cout = int(math.floor(cin * reduction))
                stages += [nn.Conv2d(cin, cout, 1, bias=False),
                           nn.BatchNorm2d(cout),
                           nn.LeakyReLU(0.1, inplace=True),
                           nn.AvgPool2d(2)]
                cin = cout
        self.stages = nn.Sequential(*stages)
        self.pool = (AttentionPool2d(cin) if use_attn_pool
                     else nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()))
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(cin, num_classes))
        self.apply(_init_weights)

    def forward(self, x):
        return self.head(self.pool(self.stages(self.stem(x))))