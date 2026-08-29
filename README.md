# ILD-TexNet

**ILD-TexNet** is a compact texture network for classifying interstitial lung
disease (ILD) patterns on 32x32 high-resolution CT (HRCT) texture patches.
This repository ships **the architecture only**, implemented from scratch in
PyTorch with no pretrained weights. The patient-disjoint evaluation benchmark
that motivated and validated the design lives in the companion repository.

> Evaluation protocol, not architecture, dominates reported ILD performance:
> a leaky patch-level split reproduces the published high-accuracy band while
> patient-disjoint evaluation converges to about 70% accuracy. ILD-TexNet is
> the compact architecture designed inside that rigorous benchmark -- it
> matches the strongest baselines at a fraction of the parameters.

## Architecture

Three components, each answering a failure measured in the benchmark rather
than chosen for novelty:

| Component | Idea | Why |
|---|---|---|
| **Multi-scale dilated stem** | Parallel 3x3 convolutions at dilations 1, 2, 3 (receptive fields 3, 5, 7 px) concatenated at the first layer | ILD patterns differ by texture *spatial frequency*, not mean attenuation; committing to one first-layer receptive field destroys the evidence the task depends on |
| **Dense SE-MBConv stages** | Each block concatenates its input to its output (DenseNet) while computing the output with an inverted-residual depthwise-separable convolution carrying squeeze-excitation (MobileNetV3) | Dense reuse keeps high-frequency stem features reachable at every depth; the separable form keeps that reuse affordable |
| **Attention pooling head** | A learned spatial weighting replaces global average pooling before the classifier | Patches are only ~70% class-pure, so average pooling blends off-class tissue into the descriptor |

The paper-reported configuration is ~0.25M parameters. Ablation controls
(`--no-multiscale`, `--block bottleneck`, `--no-attn-pool`) keep the design
auditable one component at a time.

## Installation

```bash
cd ILD-TexNet
pip install --upgrade pip
pip install -e .          # installs the ildexnet package + CLI
# or just: pip install -r requirements.txt
```

Requires Python >= 3.9, PyTorch >= 2.0 (CPU works; CUDA accelerates the
timing report and the demo).

## Usage

`main.py` is a thin wrapper around the `ildexnet` command. Two modes:

### 1. Inspect the model (default)

```bash
python main.py                          # or: ildexnet
python main.py --device cuda --json report.json
python main.py --num-classes 6 --growth 24 --layers 4,4,4
```

Prints the layer table, total/trainable parameters, model size, MACs/FLOPs
per patch, batch-1 latency, batch-256 throughput, and a forward-pass sanity
check on random noise.

### 2. Synthetic training demo (no real data bundled)

```bash
python main.py --train-demo [--epochs 30] [--batch 128] [--lr 1e-3]
```

Learns on deterministic noise patches whose class is encoded by frequency,
proving the full forward/backward/optimiser path end to end. A learning-curve
figure is written to `outputs/learning_curve.png`. Results here are
informative only -- real evaluation requires the companion benchmark
repository.

### Architecture flags

| Flag | Default | Description |
|---|---|---|
| `--num-classes` | 6 | output classes |
| `--patch-size` | 32 | input patch size |
| `--stem-ch` | 48 | stem channels |
| `--growth` | 24 | dense stage growth rate |
| `--layers` | 4,4,4 | blocks per stage |
| `--expand` | 4.0 | inverted-residual expansion |
| `--reduction` | 0.5 | inter-stage compression |
| `--dropout` | 0.2 | head dropout |
| `--block` | se_mbconv | dense block type (`se_mbconv`/`bottleneck`) |
| `--use-multiscale` / `--no-multiscale` | on | multi-scale dilated stem |
| `--use-attn-pool` / `--no-attn-pool` | on | attention pooling head |
| `--seed`, `--device`, `--output-dir`, `--json` | | run controls |

Every flag mirrors an `ILDEXNET_*` environment variable (see
`ildexnet/config.py`), so the same run can be reproduced headlessly.

## Reference cohort

ILD-TexNet is trained and evaluated on an expert-annotated high-resolution CT
cohort of ILD texture patterns. The cohort, the patch-extraction pipeline, the
three-rung leakage ladder (random / ROI-disjoint / patient-disjoint), and the
full benchmark and ablation results are described in the **companion
repository**:

- **https://github.com/rsebany/TexNet**

That repository also documents the training schedule used for the published
results: AdamW (`lr 1e-3`, `weight_decay 1e-4`), cosine annealing over
`epochs x steps`, batch size 128, cross-entropy with label smoothing 0.05,
tempered inverse-frequency class weights, and horizontal/vertical flip plus
90-degree-rotation augmentation. No ImageNet or other pretraining is used.

## Project layout

```
ILD-TexNet/
├── main.py                  # one-command entry point
├── pyproject.toml           # packaging / CLI / tool config
├── requirements.txt
├── README.md
├── ildexnet/
│   ├── __init__.py          # version + public API
│   ├── config.py            # architecture + demo settings, ILDEXNET_* env vars
│   ├── cli.py               # argparse front end
│   ├── complexity.py        # MACs/FLOPs, latency, throughput, layer table
│   ├── train.py             # synthetic demo trainer (+ figure export)
│   └── models/
│       ├── __init__.py
│       ├── components.py    # MultiScaleStem, DenseSEMBConv, DenseLayer,
│       │                    # SqueezeExcite, AttentionPool2d, _init_weights
│       └── ildexnet.py      # ILDTexNet
└── tests/
    └── test_ildexnet.py     # forward/backward, ablations, complexity, demo
```

## Tests

```bash
pytest
```

Runs on CPU with random data; no dataset download required.

## Citation

If you use ILD-TexNet in your work, please cite the benchmark paper
(compiled in the companion repository at
`https://github.com/rsebany/TexNet`).

## License

MIT. See [LICENSE](LICENSE).