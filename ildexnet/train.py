"""Synthetic 'pipeline-proof' training demo.

No real CT data is bundled with this package. :func:`train_demo` learns on
noise patches whose class is encoded by a low-frequency sinusoid purely to
prove the full forward/backward/optimiser path of ILD-TexNet works end to
end. It is a sanity check, **not** a meaningful accuracy result -- the real
training schedule and dataset live in the companion benchmark repository.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ildexnet import config


def augment_batch(batch, rng):
    """Random h/v flip + random 90-degree rotation.

    ``(N, H, W)`` inputs flip on axes 1/2; ``(N, K, H, W)`` 2.5D stacks flip
    on the spatial axes while leaving the channel axis untouched.
    """
    y_ax, x_ax = (-2, -1) if batch.ndim == 4 else (1, 2)
    if rng.rand() < 0.5:
        batch = np.flip(batch, axis=y_ax)
    if rng.rand() < 0.5:
        batch = np.flip(batch, axis=x_ax)
    k = rng.randint(4)
    if k:
        batch = np.rot90(batch, k, axes=(y_ax, x_ax))
    return np.ascontiguousarray(batch)


def batch_to_tensor(batch):
    """``(N, H, W)`` -> ``(N, 1, H, W)``; ``(N, K, H, W)`` passes through."""
    batch = batch.astype(np.float32)
    if batch.ndim == 3:
        return torch.from_numpy(batch)[:, None]
    return torch.from_numpy(batch)


def class_weights_for(y_train, num_classes, power=0.5, device="cpu"):
    """Tempered inverse-frequency class weights, mean-normalised over the
    classes actually present (classes absent from this split get weight 0)."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    present = counts > 0
    w = np.zeros_like(counts)
    if present.any():
        w[present] = (counts[present].sum()
                      / (present.sum() * counts[present])) ** power
        w[present] /= w[present].mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def classification_loss(logits, targets, class_w, cfg):
    """Cross-entropy with optional label smoothing and focal modulation."""
    ls = float(cfg.get("label_smoothing", 0.05))
    gamma = float(cfg.get("focal_gamma", 0.0))
    if gamma <= 0:
        return F.cross_entropy(logits, targets, weight=class_w,
                               label_smoothing=ls)
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    targets_oh = F.one_hot(targets, num_classes=logits.shape[-1]).float()
    if ls > 0:
        targets_oh = targets_oh * (1 - ls) + ls / logits.shape[-1]
    ce = -(targets_oh * logp).sum(dim=-1)
    pt = (p * targets_oh).sum(dim=-1).clamp(min=1e-8)
    focal = ((1 - pt) ** gamma) * ce
    if class_w is not None:
        focal = focal * class_w[targets]
    return focal.mean()


def synthetic_dataset(n_samples, num_classes, patch=32, train_frac=0.8,
                      seed=42):
    """Deterministic noise patches whose class is carried by brightness.

    Each patch is ``noise + class_bias + faint class-frequency texture``, so
    the classes are trivially separable and the demo can show loss converging
    and accuracy clearly above chance within a few epochs -- without any real
    imaging data.
    """
    rng = np.random.RandomState(seed)
    y = (np.arange(n_samples) % num_classes).astype(np.int64)
    rng.shuffle(y)
    noisy = rng.rand(n_samples, patch, patch).astype(np.float32)
    bias = 0.9 * (y + 1) / num_classes  # class brightness offset
    grid_x, _ = np.meshgrid(np.arange(patch), np.arange(patch))
    freq = 0.5 * (y + 1)  # faint per-class frequency, just for flavour
    texture = 0.25 * np.sin(
        freq[:, None, None] * grid_x[None, :, :]).astype(np.float32)
    X = noisy + bias[:, None, None] + texture
    split = int(train_frac * n_samples)
    return (X[:split], y[:split]), (X[split:], y[split:])


def train_demo(model, device, epochs=None, batch_size=None, lr=None,
               weight_decay=None, n_samples=None, seed=None,
               label_smoothing=None, output_dir=None):
    """Run the synthetic training demo and return its history.

    Returns ``{"history": [...], "final": {...}}``. When ``output_dir`` is
    given a learning-curve figure is written to ``learning_curve.png``.
    """
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE
    lr = lr if lr is not None else config.LR
    weight_decay = weight_decay if weight_decay is not None else config.WEIGHT_DECAY
    n_samples = n_samples or config.DEMO_SAMPLES
    seed = seed if seed is not None else config.SEED
    ls = label_smoothing if label_smoothing is not None else config.LABEL_SMOOTHING

    num_classes = model.head[-1].out_features
    patch = config.PATCH_SIZE
    (Xtr, ytr), (Xva, yva) = synthetic_dataset(
        n_samples, num_classes, patch=patch, seed=seed)

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    steps = max(1, len(Xtr) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs * steps, 1))
    class_w = class_weights_for(ytr, num_classes,
                                power=config.CLASS_WEIGHT_POWER, device=device)
    cfg = {"label_smoothing": ls, "focal_gamma": 0.0}

    history = []
    t_start = time.perf_counter()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(Xtr))
        run_loss, seen, n_correct = 0.0, 0, 0
        for s in range(steps):
            sel = order[s * batch_size:(s + 1) * batch_size]
            if len(sel) == 0:
                continue
            xb = batch_to_tensor(augment_batch(Xtr[sel], rng)).to(device)
            yb = torch.from_numpy(ytr[sel]).to(device)
            loss = classification_loss(model(xb), yb, class_w, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += float(loss.detach()) * len(sel)
            seen += len(sel)
            n_correct += int((model(xb).argmax(1) == yb).sum())

        model.eval()
        va_preds, va_loss, n_val = [], 0.0, 0
        with torch.no_grad():
            for s in range(0, len(Xva), 256):
                xb = batch_to_tensor(Xva[s:s + 256]).to(device)
                yb = torch.from_numpy(yva[s:s + 256]).to(device)
                logits = model(xb)
                va_loss += float(classification_loss(logits, yb, class_w, cfg)) * len(yb)
                n_val += len(yb)
                va_preds.append(logits.argmax(1).cpu().numpy())
        va_preds = np.concatenate(va_preds)
        va_acc = float((va_preds == yva).mean())
        history.append({
            "epoch": ep + 1,
            "train_loss": run_loss / max(seen, 1),
            "train_accuracy": n_correct / max(seen, 1),
            "val_loss": va_loss / max(n_val, 1),
            "val_accuracy": va_acc,
            "lr": float(sched.get_last_lr()[0]),
        })
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            h = history[-1]
            print(f"  ep{h['epoch']:>3}/{epochs}: "
                  f"train_loss={h['train_loss']:.4f} "
                  f"val_loss={h['val_loss']:.4f} "
                  f"val_acc={h['val_accuracy']:.3f}")

    model.eval()
    va_preds = []
    with torch.no_grad():
        for s in range(0, len(Xva), 512):
            xb = batch_to_tensor(Xva[s:s + 512]).to(device)
            va_preds.append(model(xb).argmax(1).cpu().numpy())
    va_preds = np.concatenate(va_preds)

    final = history[-1]
    final["val_accuracy"] = float((va_preds == yva).mean())
    final["val_balanced_accuracy"] = balanced_accuracy_step(
        yva, va_preds, num_classes)
    final["train_seconds"] = time.perf_counter() - t_start
    final["n_samples"] = n_samples
    final["note"] = ("synthetic demo data only -- not a benchmark result; "
                     "run the companion repository for real evaluation")

    if output_dir:
        _plot_history(history, output_dir)
    return {"history": history, "final": final}


def _plot_history(history, output_dir):
    """Save a compact loss/accuracy figure for the demo."""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(eps, [h["train_loss"] for h in history], "o-", ms=4, label="train")
    axes[0].plot(eps, [h["val_loss"] for h in history], "s--", ms=4, label="val")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(eps, [h["train_accuracy"] for h in history], "o-", ms=4, label="train")
    axes[1].plot(eps, [h["val_accuracy"] for h in history], "s--", ms=4, label="val")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle("ILD-TexNet synthetic training demo (no real data)", fontsize=10)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "learning_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nlearning curve -> {path}")


def balanced_accuracy_step(y_true, y_pred, num_classes):
    """Per-class mean recall (hand-rolled, so sklearn is not required)."""
    eps = 1e-8
    recalls = []
    for c in range(num_classes):
        mask = y_true == c
        if not mask.any():
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")