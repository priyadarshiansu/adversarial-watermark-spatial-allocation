import warnings

import pytest
import torch
from torch import nn

from awsa.models import Normalized, freeze
from awsa.saliency import degenerate_saliency, gradcam


class _Backbone(nn.Module):
    """Tiny CNN with a `layer4` producing a 7x7 map at 224x224 input (like ResNet-50)."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 8, 3, stride=4, padding=1), nn.ReLU(),
                                  nn.AvgPool2d(8))  # 224 -> 56 -> 7
        self.layer4 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        return self.fc(self.layer4(self.stem(x)).mean(dim=(2, 3)))


@pytest.fixture
def cam_model():
    torch.manual_seed(0)
    backbone = _Backbone()
    with torch.no_grad():
        # Positive class weights over non-negative activations: CAMs are never all-zero.
        backbone.fc.weight.abs_()
    return freeze(Normalized(backbone))


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    return torch.rand(3, 3, 224, 224, generator=g), torch.tensor([0, 3, 7])


def test_shape_range_dtype(cam_model, batch):
    x, y = batch
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a healthy model must not trigger the warning
        cam = gradcam(cam_model, x, y)
    assert not degenerate_saliency(cam).any()
    assert cam.shape == (3, 1, 28, 28) and cam.dtype == torch.float32
    assert cam.min() >= 0 and cam.max() <= 1
    assert torch.allclose(cam.amax(dim=(1, 2, 3)), torch.ones(3))


def test_batch_matches_per_sample(cam_model, batch):
    x, y = batch
    full = gradcam(cam_model, x, y)
    single = torch.cat([gradcam(cam_model, x[i:i + 1], y[i:i + 1]) for i in range(3)])
    assert torch.allclose(full, single, atol=1e-5)


def test_no_side_effects(cam_model, batch):
    x, y = batch
    # Unfreeze parameters to check that no .grad is written to them.
    for p in cam_model.parameters():
        p.requires_grad_(True)
    x = x.clone().requires_grad_(True)
    gradcam(cam_model, x, y)
    assert len(cam_model.model.layer4._forward_hooks) == 0
    assert all(p.grad is None for p in cam_model.parameters())
    assert x.grad is None
    assert not cam_model.training


def test_works_under_no_grad(cam_model, batch):
    x, y = batch
    ref = gradcam(cam_model, x, y)
    with torch.no_grad():
        cam = gradcam(cam_model, x, y)
    assert torch.allclose(cam, ref)
    assert len(cam_model.model.layer4._forward_hooks) == 0


def test_degenerate_saliency_flags_constant_map():
    cam = torch.rand(3, 1, 28, 28, generator=torch.Generator().manual_seed(0))
    cam[1] = 0.5
    assert degenerate_saliency(cam).tolist() == [False, True, False]


def test_degenerate_cam_warns_and_is_zero(cam_model, batch):
    x, y = batch
    # Zero the classifier weights: score gradients vanish, so every CAM is constant.
    with torch.no_grad():
        cam_model.model.fc.weight.zero_()
    with pytest.warns(RuntimeWarning, match="degenerate"):
        cam = gradcam(cam_model, x, y)
    assert degenerate_saliency(cam).all()
    assert (cam == 0).all()
