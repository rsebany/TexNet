"""Command-line entry point for ILD-TexNet.

Builds the model and prints a layer table, parameter count, model size,
MACs/FLOPs, latency, throughput, plus a forward-pass sanity check on noise.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ildexnet import __version__, config
from ildexnet.complexity import inspect_model, print_layer_summary
from ildexnet.models import ILDTexNet

DESCRIPTION = (
    "ILD-TexNet: build and inspect the proposed compact texture network "
    "(https://github.com/rsebany/TexNet)."
)


def _add_arch_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("architecture")
    g.add_argument("--num-classes", type=int, dest="num_classes", default=None,
                   help=f"output classes (default {config.NUM_CLASSES})")
    g.add_argument("--patch-size", type=int, default=config.PATCH_SIZE,
                   help=f"input patch size (default {config.PATCH_SIZE})")
    g.add_argument("--stem-ch", type=int, dest="stem_ch", default=None,
                   help=f"stem channels (default {config.STEM_CH})")
    g.add_argument("--growth", type=int, default=None,
                   help=f"dense stage growth rate (default {config.GROWTH})")
    g.add_argument("--layers", default=None,
                   help="blocks per stage as comma list, e.g. 4,4,4")
    g.add_argument("--expand", type=float, default=None,
                   help=f"inverted-residual expansion (default {config.EXPAND})")
    g.add_argument("--reduction", type=float, default=None,
                   help=f"inter-stage compression (default {config.REDUCTION})")
    g.add_argument("--dropout", type=float, default=None,
                   help=f"head dropout (default {config.DROPOUT})")
    g.add_argument("--block", choices=("se_mbconv", "bottleneck"),
                   default=config.BLOCK,
                   help="dense block type (default se_mbconv)")
    g.add_argument("--use-multiscale", dest="use_multiscale",
                   action="store_true", default=None,
                   help="multi-scale dilated stem (default)")
    g.add_argument("--no-multiscale", dest="use_multiscale",
                   action="store_false",
                   help="plain 3x3 stem (ablation control)")
    g.add_argument("--use-attn-pool", dest="use_attn_pool",
                   action="store_true", default=None,
                   help="attention pooling head (default)")
    g.add_argument("--no-attn-pool", dest="use_attn_pool",
                   action="store_false",
                   help="adaptive average pooling (ablation control)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ildexnet", description=DESCRIPTION)
    parser.add_argument("--version", action="version",
                        version=f"ILD-TexNet {__version__}")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help=f"random seed (default {config.SEED})")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        default="auto", help="compute device (default auto)")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="also write the inspect report to this JSON file")
    _add_arch_args(parser)
    return parser


def _cmdline_kwargs(args) -> dict:
    layers = None
    if args.layers:
        layers = tuple(int(v) for v in args.layers.split(",") if v.strip())
    return {
        "num_classes": args.num_classes,
        "stem_ch": args.stem_ch,
        "growth": args.growth,
        "layers": layers,
        "expand": args.expand,
        "reduction": args.reduction,
        "dropout": args.dropout,
        "block": args.block,
        "use_multiscale": (config.USE_MULTISCALE
                           if args.use_multiscale is None
                           else args.use_multiscale),
        "use_attn_pool": (config.USE_ATTN_POOL
                          if args.use_attn_pool is None
                          else args.use_attn_pool),
    }


def _forward_sanity(model, device, patch, num_classes, seed):
    """Random-input forward with a seeded distribution report."""
    torch.manual_seed(seed)
    model.eval()
    x = torch.randn(64, 1, patch, patch, device=device)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1).float()
    counts = torch.histc(pred, bins=num_classes, min=0,
                         max=num_classes - 1).int().tolist()
    return {"logits_shape": list(logits.shape),
            "mean_probability_mass": float(logits.softmax(-1).sum(-1).mean()),
            "argmax_counts_64": counts}


def _run_inspect(args, device) -> dict:
    print("=" * 78)
    print("ILD-TexNet - model inspection")
    print("=" * 78)
    kwargs = config.build_kwargs(_cmdline_kwargs(args))
    model = ILDTexNet(**kwargs).to(device)
    torch.manual_seed(args.seed)

    keys = ["num_classes", "in_ch", "stem_ch", "growth", "layers", "expand",
            "reduction", "dropout", "block", "use_multiscale", "use_attn_pool"]
    print("\narchitecture:")
    for k in keys:
        print(f"  {k:<16} {kwargs[k]}")
    print(f"  {'seed':<16} {args.seed}")
    print(f"  {'device':<16} {device}")

    info = inspect_model(model, device, (kwargs["in_ch"], args.patch_size,
                                         args.patch_size))
    sanity = _forward_sanity(model, device, args.patch_size,
                             kwargs["num_classes"], args.seed)

    print("\nlayer summary:")
    print_layer_summary(info["layers"])
    print()
    print("complexity:")
    print(f"  {'params total':<30} {info['params_total']:,}")
    print(f"  {'params trainable':<30} {info['params_trainable']:,}")
    print(f"  {'buffers':<30} {info['buffers']:,}")
    print(f"  {'model size':<30} {info['model_size_mb']:.3f} MB")
    print(f"  {'MACs (per patch)':<30} {info['macs']:,}")
    print("      conv/linear/attention: "
          f"{info['macs_conv']:,} / {info['macs_linear']:,} / "
          f"{info['macs_attention']:,}")
    print(f"  {'FLOPs (2 x MACs)':<30} {info['flops']:,}")
    print("timing:")
    print(f"  {'latency bs1':<30} {info['latency_ms_per_patch_bs1']:.3f} ms")
    print(f"  {'  p90 bs1':<30} {info['latency_ms_p90_bs1']:.3f} ms")
    print(f"  {'throughput bs256':<30} "
          f"{info['throughput_patches_per_s']:,.0f} patches/s")
    print("forward sanity (random noise):")
    print(f"  {'logits shape':<30} {sanity['logits_shape']}")
    print(f"  {'argmax histogram':<30} {sanity['argmax_counts_64']}")
    print(f"  {'mean probability mass':<30} "
          f"{sanity['mean_probability_mass']:.4f} (expect ~1.0)")

    if args.json_path:
        report = {"kwargs": {k: kwargs[k] for k in keys},
                  "seed": args.seed, "device": str(device),
                  "patch_size": args.patch_size,
                  "complexity": {k: v for k, v in info.items()
                                 if k != "layers"},
                  "layers": info["layers"], "forward_sanity": sanity}
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)),
                    exist_ok=True)
        with open(args.json_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\ndetail report -> {args.json_path}")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    t_all = time.time()
    device = config.device(args.device)
    _run_inspect(args, device)
    print(f"\nwall time: {time.time() - t_all:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())