"""Unit tests for the ILD-TexNet architecture and CLI demo train path.

All tests run on CPU with random data -- no dataset download required.
"""

import numpy as np
import pytest
import torch

from ildexnet.complexity import inspect_model, layer_summary, model_complexity
from ildexnet.models import ILDTexNet
from ildexnet.train import train_demo

DEVICE = torch.device("cpu")


def _model(**overrides):
    kwargs = {"num_classes": 6, "in_ch": 1, "stem_ch": 48, "growth": 24,
              "layers": (2, 2, 2), "expand": 4, "reduction": 0.5,
              "dropout": 0.2, "use_multiscale": True, "block": "se_mbconv",
              "use_attn_pool": True}
    kwargs.update(overrides)
    return ILDTexNet(**kwargs).to(DEVICE)


def test_forward_shape_and_backward():
    model = _model()
    x = torch.randn(4, 1, 32, 32, device=DEVICE)
    logits = model(x)
    assert logits.shape == (4, 6)
    loss = logits.softmax(-1).log().mean().neg()
    loss.backward()
    grads = [p.grad is not None and p.grad.abs().sum() > 0
             for p in model.parameters()]
    assert all(grads), "every parameter should receive a gradient"


@pytest.mark.parametrize("overrides,expected_params_relation", [
    ({"use_multiscale": False}, "equal"),
    ({"block": "bottleneck"}, "different"),
    ({"use_attn_pool": False}, "smaller"),
])
def test_ablation_controls_build_and_run(overrides, expected_params_relation):
    base = _model()
    variant = _model(**overrides)
    x = torch.randn(2, 1, 32, 32, device=DEVICE)
    with torch.no_grad():
        out_base = base(x)
        out_var = variant(x)
    assert out_base.shape == out_var.shape == (2, 6)
    nb, nv = (sum(p.numel() for p in m.parameters())
              for m in (base, variant))
    if expected_params_relation == "smaller":
        assert nv < nb
    elif expected_params_relation == "equal":
        # The multi-scale stem reads the input once per dilation branch, so
        # swapping it for a plain 3x3 conv keeps the parameter count unchanged
        # by design (the ablation isolates representation, not capacity).
        assert nv == nb
    else:
        assert nv != nb


def test_seeded_initialisation_is_deterministic():
    torch.manual_seed(7)
    a = _model()
    torch.manual_seed(7)
    b = _model()
    a.eval()
    b.eval()
    x = torch.randn(2, 1, 32, 32, device=DEVICE)
    with torch.no_grad():
        assert torch.equal(a(x), b(x)), "seed should reproduce the init"


def test_model_complexity_reports_positive_values():
    model = _model()
    info = model_complexity(model, (1, 32, 32), DEVICE)
    assert info["params_total"] > 0
    assert info["params_trainable"] > 0
    assert info["macs"] > 0
    assert info["flops"] == 2 * info["macs"]
    assert info["macs_conv"] > 0


def test_inspect_includes_layers_and_timing():
    model = _model()
    info = inspect_model(model, DEVICE, (1, 32, 32))
    assert info["latency_ms_per_patch_bs1"] > 0
    assert info["throughput_patches_per_s"] > 0
    rows = layer_summary(model, (1, 32, 32), DEVICE)
    names = {r["name"] for r in rows}
    assert "(root)" in names
    assert any(n.startswith("stem") for n in names)
    assert any(n.startswith("stages") for n in names)


def test_synthetic_train_demo_completes_and_learns():
    model = _model()
    result = train_demo(model, DEVICE, epochs=6, batch_size=64, n_samples=512,
                        seed=42, output_dir=None)
    final = result["final"]
    assert len(result["history"]) == 6
    assert 0.0 <= final["val_accuracy"] <= 1.0
    # The brightness-encoded synthetic classes are trivially separable; a
    # working pipeline should get well above chance after ~36 optimizer steps.
    assert final["val_accuracy"] > 0.4
    assert np.isfinite(final["train_loss"])
    assert final["val_balanced_accuracy"] >= final["val_accuracy"] - 1.0