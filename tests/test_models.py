import torch
from torch import nn

from awsa.models import IMAGENET_MEAN, IMAGENET_STD, Normalized, freeze


class _Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.seen = None

    def forward(self, x):
        self.seen = x.clone()
        return x


def test_normalized_applies_imagenet_normalization(images):
    inner = _Recorder()
    out = Normalized(inner)(images)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    expected = (images - mean) / std
    assert torch.allclose(inner.seen, expected)
    assert torch.allclose(out, expected)


def test_freeze_sets_eval_and_disables_grads():
    model = Normalized(_Recorder())
    model.train()
    assert freeze(model) is model
    assert not model.training and not model.model.training
    assert all(not p.requires_grad for p in model.parameters())
