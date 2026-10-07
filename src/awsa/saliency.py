"""Grad-CAM saliency (Module 3).

Target layer for ResNet-50: model.model.layer4 (7x7 feature map at 224x224 input).
"""

import warnings

import torch
import torch.nn.functional as F
from torch import nn

from . import GRID


def degenerate_saliency(cam: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B,) bool: True where a saliency map is (near-)constant, i.e. max - min < eps.

    This happens when the ReLU in Grad-CAM zeroes the whole map. Such images carry no
    spatial ranking, so saliency-based masks for them are arbitrary (index order);
    runners should exclude them.
    """
    flat = cam.detach().flatten(1)
    return (flat.amax(dim=1) - flat.amin(dim=1)) < eps


def gradcam(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Grad-CAM heatmap for class y, mapped to the 28x28 block grid.

    Works under ``torch.no_grad()`` and never writes ``.grad`` to the model parameters
    or to ``x`` (gradients are taken with ``torch.autograd.grad``). The model is left in
    eval mode.

    Images whose map is degenerate (see :func:`degenerate_saliency`) get an all-zero map
    and a warning is emitted; call ``degenerate_saliency`` on the output to exclude them.

    Returns:
        Tensor of shape (B, 1, 28, 28), float32, normalized per image to [0, 1].
    """
    model.eval()
    activations = {}

    def save_activation(module, inputs, output):
        activations["value"] = output

    handle = model.model.layer4.register_forward_hook(save_activation)
    try:
        with torch.enable_grad():
            # Fresh leaf so the graph reaches layer4 even if all parameters are frozen.
            x_in = x.detach().requires_grad_(True)
            logits = model(x_in)
            score_sum = logits.gather(1, y.view(-1, 1)).sum()
            acts = activations["value"]
            grads = torch.autograd.grad(score_sum, acts)[0]

        acts = acts.detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(GRID, GRID), mode="bilinear", align_corners=False)

        flat = cam.flatten(1)
        cam_min = flat.amin(dim=1).view(-1, 1, 1, 1)
        cam_max = flat.amax(dim=1).view(-1, 1, 1, 1)
        span = cam_max - cam_min
        degenerate = degenerate_saliency(cam)
        cam = torch.where(span > 0, (cam - cam_min) / span.clamp_min(1e-12),
                          torch.zeros_like(cam))
        cam[degenerate] = 0.0
        if degenerate.any():
            warnings.warn(
                f"gradcam: {int(degenerate.sum())} of {len(degenerate)} image(s) have a "
                "degenerate (constant) saliency map; returned as all zeros. Exclude them "
                "with awsa.saliency.degenerate_saliency.",
                RuntimeWarning, stacklevel=2,
            )
        return cam.float().clamp(0, 1)
    finally:
        handle.remove()
