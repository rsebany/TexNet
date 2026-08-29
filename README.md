<div align="center">

# TexNet

**A compact multi-scale texture network for six-class ILD pattern classification on 32x32 HRCT patches**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c)

</div>

TexNet is a multi-scale texture network for interstitial lung disease (ILD) pattern classification on high-resolution CT patches. ~0.25M parameters, trained from scratch, no pretraining. Manuscript and benchmark: [TexNet](https://github.com/rsebany/TexNet).

| Component | Role |
|-----------|------|
| Multi-scale dilated stem | Parallel 3x3 convolutions (dilations 1, 2, 3) |
| Dense SE-MBConv stages | DenseNet-style reuse with MobileNetV3-style inverted residuals + squeeze-excitation |
| Attention pooling | Learned spatial weighting before the classifier |

## Install

```bash
pip install -e .   # or: pip install -r requirements.txt
```

Requires Python >= 3.9 and PyTorch >= 2.0.

## Usage

Build and inspect the model (layer table, parameters, MACs, latency, forward sanity check):

```bash
python main.py
# or, after install: ildexnet [--device cuda --json report.json]
```

Architecture flags (`--stem-ch`, `--growth`, `--layers`, `--no-multiscale`, `--block bottleneck`, `--no-attn-pool`, ...) mirror `ILDEXNET_*` environment variables in `ildexnet/config.py`.

## Layout

```
ildexnet/
  models/ildexnet.py    # ILDTexNet module
  models/components.py  # stem, blocks, attention pool
  cli.py                # command-line interface
  complexity.py         # profiling helpers
  config.py             # defaults (enabled via ILDEXNET_* env vars)
```

## Dataset

Dataset used: **MedGIFT**, the public ILD database of [Depeursinge et al. (2012)](https://doi.org/10.1016/j.compmedimag.2011.07.003) · [Google Scholar](https://scholar.google.com/scholar?q=%22Building+a+reference+multimedia+database+for+interstitial+lung+diseases%22). Patch extraction, splits, and the full patient-disjoint evaluation live in the [TexNet](https://github.com/rsebany/TexNet) repository.

## License

MIT, see [LICENSE](LICENSE).