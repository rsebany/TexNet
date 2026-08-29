# ILD-TexNet

> References: [MedGIFT ILD database](https://doi.org/10.1016/j.compmedimag.2011.07.003) (Depeursinge et al., 2012) · [TexNet (manuscript, benchmark, results)](https://github.com/rsebany/TexNet)

Compact PyTorch implementation of **ILD-TexNet** — a multi-scale texture network for six-class ILD pattern classification on 32×32 HRCT patches (~0.25M parameters, trained from scratch, no pretraining).

| Component | Role |
|-----------|------|
| Multi-scale dilated stem | Parallel 3×3 convolutions (dilations 1, 2, 3) |
| Dense SE-MBConv stages | DenseNet-style reuse with MobileNetV3-style inverted residuals + squeeze-excitation |
| Attention pooling | Learned spatial weighting before the classifier |

## Install

```bash
pip install -e .   # or: pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

## Usage

Build and inspect the model (layer table, parameters, MACs, latency, forward sanity check):

```bash
python main.py
# or, after install: ildexnet [--device cuda --json report.json]
```

Architecture flags (`--stem-ch`, `--growth`, `--layers`, `--no-multiscale`, `--block bottleneck`, `--no-attn-pool`, …) mirror `ILDEXNET_*` environment variables in `ildexnet/config.py`.

## Layout

```
ildexnet/
  models/ildexnet.py    # ILDTexNet module
  models/components.py  # stem, blocks, attention pool
  cli.py                # command-line interface
  complexity.py         # profiling helpers
  config.py             # defaults (enabled via ILDEXNET_* env vars)
```

## License

MIT — see [LICENSE](LICENSE).