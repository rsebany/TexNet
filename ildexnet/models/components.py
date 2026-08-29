"""Shared building blocks for ILD-TexNet.

Every module is implemented from scratch in PyTorch and initialised with a
uniform scheme (:func:`_init_weights`), so the network trains from scratch
with no pretrained weights anywhere.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module):
    """Conv2d -> Kaiming normal (fan-out); BN/GN -> ones + zeros;
    Linear -> truncated normal (std 0.02)."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SqueezeExcite(nn.Module):
    """Channel recalibration: global pooling -> two 1x1 convs -> sigmoid gate."""

    def __init__(self, ch, reduction=4):
        super().__init__()
        hidden = max(8, ch // reduction)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        return x * torch.sigmoid(self.fc2(s))


class MultiScaleStem(nn.Module):
    """Parallel dilated 3x3 convolutions: receptive fields 3, 5 and 7 px."""

    def __init__(self, in_ch, out_ch, dilations=(1, 2, 3)):
        super().__init__()
        per = out_ch // len(dilations)
        chs = [per] * len(dilations)
        chs[-1] += out_ch - sum(chs)
        self.branches = nn.ModuleList(
            [nn.Conv2d(in_ch, c, 3, padding=d, dilation=d, bias=False)
             for c, d in zip(chs, dilations)])
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.norm(
            torch.cat([b(x) for b in self.branches], 1)))


class DenseLayer(nn.Module):
    """Bottleneck dense block (1x1 -> 3x3, BN+ReLU between), growth ``growth``.
    Used by the ``block="bottleneck"`` ablation control."""

    def __init__(self, cin, growth, bn_size=4):
        super().__init__()
        mid = bn_size * growth
        self.norm1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, mid, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, growth, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        return torch.cat([x, out], 1)


class DenseSEMBConv(nn.Module):
    """MBConv with squeeze-excitation whose output is concatenated, not added."""

    def __init__(self, cin, growth, expand=4, kernel=3):
        super().__init__()
        hidden = int(round(growth * expand))
        self.block = nn.Sequential(
            nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, kernel, 1, kernel // 2, groups=hidden,
                      bias=False), nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.1, inplace=True),
            SqueezeExcite(hidden),
            nn.Conv2d(hidden, growth, 1, bias=False), nn.BatchNorm2d(growth))

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)


class AttentionPool2d(nn.Module):
    """Learned spatial weighting before the classifier.

    Patches can be only ~70% class-pure, so a learned weighting lets the
    classifier concentrate on the region that carries the label.
    """

    def __init__(self, ch, hidden=None):
        super().__init__()
        hidden = hidden or max(8, ch // 8)
        self.score = nn.Sequential(
            nn.Conv2d(ch, hidden, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 1, 1))

    def forward(self, x):
        B, C = x.shape[:2]
        w = torch.softmax(self.score(x).view(B, 1, -1), dim=-1)
        return (x.view(B, C, -1) * w).sum(-1)