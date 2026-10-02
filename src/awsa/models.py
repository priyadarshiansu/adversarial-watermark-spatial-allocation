"""Classifier wrappers. Inputs are [0, 1] pixel tensors; normalization happens inside."""

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalized(nn.Module):
    """Wrap a model so it accepts [0, 1] pixels and applies ImageNet normalization itself.

    This keeps every attack, watermark and metric working in true pixel space,
    so epsilon = 2/255 really means 2/255 of the pixel range.
    """

    def __init__(self, model: nn.Module, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def freeze(model: nn.Module) -> nn.Module:
    """Eval mode and no parameter gradients (we only need gradients w.r.t. the input)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_resnet50(device: torch.device | str = "cpu") -> nn.Module:
    """Pretrained torchvision ResNet-50 (ImageNet-1k), wrapped to take [0, 1] pixels."""
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    return freeze(Normalized(model)).to(device)


def imagenet_categories() -> list[str]:
    """The 1000 ImageNet class names in torchvision index order (0 = tench)."""
    from torchvision.models import ResNet50_Weights

    return list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
