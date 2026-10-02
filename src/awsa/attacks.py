"""Adversarial attacks in pixel space (Module 2).

All attacks take a model that accepts [0, 1] pixels (see models.Normalized),
images x in [0, 1] of shape (B, 3, H, W), and integer labels y of shape (B,).

`mask` restricts where the perturbation may live: a float/bool tensor broadcastable
to x (e.g. (B, 1, H, W) pixel mask from masks.blocks_to_pixels). 1 = allowed.

`transform` is applied to the adversarial image before the model inside the loss.
Use it for the JPEG-aware attack (e.g. a differentiable JPEG, or a random
JPEG approximation with eot_samples > 1 for Expectation over Transformation).
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


def _apply_mask(delta: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return delta if mask is None else delta * mask.to(delta.dtype)


def fgsm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-step untargeted FGSM in pixel space."""
    x = x.detach()
    x_adv = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    (grad,) = torch.autograd.grad(loss, x_adv)
    delta = _apply_mask(eps * grad.sign(), mask)
    return (x + delta).clamp(0, 1).detach()


def pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    mask: torch.Tensor | None = None,
    random_start: bool = False,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    eot_samples: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Untargeted L-infinity PGD (Madry et al., 2018) in pixel space, optionally masked.

    Returns x_adv with ||x_adv - x||_inf <= eps, x_adv in [0, 1], and
    x_adv == x wherever mask == 0.
    """
    x = x.detach()
    if random_start:
        noise = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        delta = _apply_mask((noise * 2 - 1) * eps, mask)
    else:
        delta = torch.zeros_like(x)
    x_adv = (x + delta).clamp(0, 1)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = 0.0
        for _ in range(eot_samples):
            inp = transform(x_adv) if transform is not None else x_adv
            loss = loss + F.cross_entropy(model(inp), y)
        (grad,) = torch.autograd.grad(loss / eot_samples, x_adv)

        step = _apply_mask(alpha * grad.sign(), mask)
        delta = torch.clamp(x_adv.detach() + step - x, -eps, eps)
        x_adv = (x + _apply_mask(delta, mask)).clamp(0, 1)

    return x_adv.detach()
