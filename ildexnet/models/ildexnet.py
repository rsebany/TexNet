"""ILD-TexNet: the proposed compact texture network.

* **Multi-scale dilated stem** -- parallel 3x3 convolutions at dilations
  1, 2, 3 (receptive fields 3, 5, 7 px), so the first layer does not commit
  to a single spatial-frequency band.
* **Dense SE-MBConv stages** -- each block concatenates input to output
  (DenseNet) while computing the output with an inverted-residual
  depthwise-separable convolution carrying squeeze-excitation (MobileNetV3).
* **Attention pooling head** -- learned spatial weighting so the classifier
  can ignore off-class tissue inside low-purity patches.

``use_multiscale``, ``block`` and ``use_attn_pool`` allow ablating one
component at a time.
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
    """Multi-scale dilated stem, dense SE-MBConv stages, attention pooling."""

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