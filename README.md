# ILD-TexNet

**Requirements:** [Python](https://www.python.org/) ≥ 3.9 · [PyTorch](https://pytorch.org/) ≥ 2.0

Compact PyTorch implementation of **ILD-TexNet** — a multi-scale texture network for six-class ILD pattern classification on 32×32 HRCT patches (~0.25M parameters, trained from scratch, no pretraining).

| Component | Role |
|-----------|------|
| Multi-scale dilated stem | Parallel 3×3 convolutions (dilations 1, 2, 3) |
| Dense SE-MBConv stages | DenseNet-style reuse with MobileNet-style inverted-residual blocks and squeeze-excitation |
| Attention pooling | Learned spatial weighting before the classifier |

Patient-disjoint evaluation, benchmark results, and the manuscript live in the companion repository: [TexNet](https://github.com/rsebany/TexNet).

## Installation

```bash
pip install -e .
# or: pip install -r requirements.txt
```

See [python.org](https://www.python.org/) and [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for install instructions.

## Usage

**Inspect the model** (default — layer table, parameters, MACs, latency):

```bash
python main.py
ildexnet --device cuda --json report.json
```

**Synthetic training demo** (no dataset bundled):

```bash
python main.py --train-demo
```

Architecture flags (`--stem-ch`, `--growth`, `--layers`, `--no-multiscale`, `--block bottleneck`, `--no-attn-pool`, etc.) mirror `ILDEXNET_*` environment variables in `ildexnet/config.py`.

## Tests

```bash
pytest
```

## Layout

```
ildexnet/
  models/ildexnet.py    # ILDTexNet module
  models/components.py  # stem, blocks, attention pool
  cli.py                # command-line interface
  complexity.py         # profiling helpers
  train.py              # synthetic demo trainer
```

## Citation

Please cite the paper in [TexNet](https://github.com/rsebany/TexNet) if you use this code.

## License

MIT — see [LICENSE](LICENSE).
