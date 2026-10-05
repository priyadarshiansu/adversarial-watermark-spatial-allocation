"""Grad-CAM saliency (Module 3).

Target layer for ResNet-50: model.model.layer4 (7x7 feature map at 224x224 input).
"""

import torch
import torch.nn.functional as F
from torch import nn


def gradcam(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Grad-CAM heatmap for class y, mapped to the 28x28 block grid.

    Returns:
        Tensor of shape (B, 1, 28, 28), normalized per image to [0, 1].
    """
    x = x.detach().requires_grad_(True)

    activations = {}

    def save_activation(module, inputs, output):
        activations["value"] = output
        output.retain_grad()

    target_layer = model.model.layer4
    handle = target_layer.register_forward_hook(save_activation)

    try:
        logits = model(x)

        # Score corresponding to each image's target class.
        scores = logits.gather(1, y.view(-1, 1)).sum()

        model.zero_grad(set_to_none=True)
        scores.backward()

        acts = activations["value"]
        grads = acts.grad

        # Grad-CAM channel weights.
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps.
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # ResNet layer4 gives 7x7; convert to our 28x28 block grid.
        cam = F.interpolate(
            cam,
            size=(28, 28),
            mode="bilinear",
            align_corners=False,
        )

        # Min-max normalize separately for every image.
        b = cam.shape[0]
        flat = cam.view(b, -1)

        cam_min = flat.min(dim=1).values.view(b, 1, 1, 1)
        cam_max = flat.max(dim=1).values.view(b, 1, 1, 1)

        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach()

    finally:
        handle.remove()