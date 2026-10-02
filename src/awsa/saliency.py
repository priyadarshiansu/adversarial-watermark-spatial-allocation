"""Grad-CAM saliency (Module 3).

Target layer for ResNet-50: model.model.layer4 (7x7 feature map at 224x224 input).
"""

import torch
from torch import nn


def gradcam(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Grad-CAM heatmap for class y, upsampled and averaged to the block grid.

    Returns (B, 1, 28, 28) float, min-max normalized per image to [0, 1].
    """
    raise NotImplementedError("Module 3")
