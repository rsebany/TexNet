"""Model complexity and timing instrumentation.

Multiply-accumulate counts are computed from forward hooks on ``nn.Conv2d``
and ``nn.Linear`` plus an explicit term for the attention matmuls inside
``nn.MultiheadAttention``, which carry no parameters and would otherwise be
invisible. FLOPs are taken as ``2 x MACs``, the usual convention. Latency and
throughput are measured after a warmup with ``torch.cuda.synchronize()``
around the timed region: without the synchronize, CUDA's asynchronous
dispatch makes these numbers meaningless.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn


def count_macs(model, input_shape=(1, 32, 32), device="cpu"):
    """MACs for one forward pass of a single sample."""
    totals = {"conv": 0, "linear": 0, "attention": 0}
    handles = []

    def conv_hook(mod, _inp, out):
        out_elems = out.shape[2] * out.shape[3]
        kernel = mod.kernel_size[0] * mod.kernel_size[1]
        totals["conv"] += int(out_elems * mod.out_channels
                              * (mod.in_channels // mod.groups) * kernel)

    def linear_hook(mod, _inp, out):
        shapes = out.shape[1:-1]
        repeats = int(np.prod(shapes)) if shapes else 1
        totals["linear"] += int(repeats * mod.in_features * mod.out_features)

    def mha_hook(mod, inp, _out):
        # out_proj is an nn.Linear child and is counted by linear_hook; the
        # in-projections are bare Parameters, and the two batched matmuls have
        # no parameters at all.
        q = inp[0]
        n_tok, dim = int(q.shape[1]), int(q.shape[2])
        totals["attention"] += int(n_tok * dim * dim * 3)
        totals["attention"] += int(2 * n_tok * n_tok * dim)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.MultiheadAttention):
            handles.append(m.register_forward_hook(mha_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape, device=device))
    for h in handles:
        h.remove()
    model.train(was_training)
    total = sum(totals.values())
    return {"macs": total, "flops": 2 * total,
            "macs_conv": totals["conv"], "macs_linear": totals["linear"],
            "macs_attention": totals["attention"]}


def model_complexity(model, input_shape=(1, 32, 32), device="cpu"):
    """Parameter, buffer and model-size summary plus MAC counts."""
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buf = sum(b.numel() for b in model.buffers())
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_ += sum(b.numel() * b.element_size() for b in model.buffers())
    out = {"params_total": n_total, "params_trainable": n_train,
           "buffers": n_buf, "model_size_mb": bytes_ / (1024 ** 2)}
    out.update(count_macs(model, input_shape, device))
    return out


@torch.no_grad()
def measure_latency(model, device, input_shape=(1, 32, 32), batch=1,
                    warmup=20, iters=100):
    """Latency statistics for repeated forward passes at a fixed batch size."""
    model.eval()
    x = torch.zeros(batch, *input_shape, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples)
    return {"mean_s": float(arr.mean()), "std_s": float(arr.std()),
            "p50_s": float(np.percentile(arr, 50)),
            "p90_s": float(np.percentile(arr, 90))}


def measure_timing(model, device, input_shape=(1, 32, 32)):
    """Batch-1 latency and batch-256 throughput, after warmup."""
    gpu = device.type == "cuda"
    single = measure_latency(model, device, input_shape, batch=1,
                             warmup=20 if gpu else 3, iters=100 if gpu else 20)
    bulk = measure_latency(model, device, input_shape, batch=256,
                           warmup=5 if gpu else 1, iters=30 if gpu else 3)
    return {
        "latency_ms_per_patch_bs1": single["mean_s"] * 1e3,
        "latency_ms_std_bs1": single["std_s"] * 1e3,
        "latency_ms_p90_bs1": single["p90_s"] * 1e3,
        "throughput_patches_per_s": 256.0 / bulk["mean_s"],
        "latency_ms_per_batch256": bulk["mean_s"] * 1e3,
    }


def peak_memory_mb(device):
    """Peak CUDA memory used so far; NaN on CPU."""
    if device.type != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def _out_shape(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    return tuple(int(s) for s in out.shape) if torch.is_tensor(out) else None


def layer_summary(model, input_shape=(1, 32, 32), device="cpu"):
    """Per-module type / output-shape / parameter table via forward hooks.

    Module names follow PyTorch's ``named_modules`` structure. Each row reports
    the parameters owned *directly* by that module (containers report 0 and
    their children carry the counts), together with the activation shape that
    module produced on the reference forward pass.
    """
    shapes = {}
    handles = []
    for _name, mod in model.named_modules():
        handles.append(mod.register_forward_hook(
            lambda _mod, _inp, _out, _s=shapes, _m=mod:
            _s.__setitem__(id(_m), _out_shape(_out))))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape, device=device))
    model.train(was_training)
    for h in handles:
        h.remove()

    rows = []
    for name, mod in model.named_modules():
        params = sum(p.numel() for p in mod.parameters(recurse=False))
        out = shapes.get(id(mod), None)
        rows.append({"name": name or "(root)", "kind": type(mod).__name__,
                     "params": params,
                     "output": "x".join(map(str, out)) if out else ""})
    rows.sort(key=lambda r: (r["name"].count("."), r["name"]))
    return rows


def print_layer_summary(rows):
    """Render :func:`layer_summary` output as a fixed-width table."""
    width = max(len(r["name"]) for r in rows)
    kind_w = max(len(r["kind"]) for r in rows)
    head = (f"{'Module':<{width}} {'Type':<{kind_w}} "
            f"{'Params':>10} {'Output shape':>18}")
    print(head)
    print("-" * (width + kind_w + 34))
    for r in rows:
        params = f"{r['params']:,}" if r["params"] else "-"
        print(f"{r['name']:<{width}} {r['kind']:<{kind_w}} "
              f"{params:>10} {r['output']:>18}")


def inspect_model(model, device, input_shape=(1, 32, 32)):
    """One-stop report: layer table, complexity and timing for a model."""
    model = model.to(device)
    info = model_complexity(model, input_shape, device)
    info.update(measure_timing(model, device, input_shape))
    info["layers"] = layer_summary(model, input_shape, device)
    return info