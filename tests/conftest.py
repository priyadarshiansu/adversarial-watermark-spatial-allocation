import pytest
import torch
from torch import nn


@pytest.fixture
def tiny_model():
    """A small random CNN standing in for ResNet-50 (no download, fast on CPU)."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(),
        nn.Linear(8 * 16, 10),
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture
def images():
    g = torch.Generator().manual_seed(0)
    return torch.rand(2, 3, 224, 224, generator=g)
