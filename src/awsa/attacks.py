"""Adversarial attacks in pixel space (Module 2).

All attacks take a model that accepts [0, 1] pixels (see models.Normalized),
images x in [0, 1] of shape (B, 3, H, W), and integer labels y of shape (B,).

`mask` restricts where the perturbation may live: a float/bool tensor broadcastable
to x (e.g. (B, 1, H, W) pixel mask from masks.blocks_to_pixels). 1 = allowed.

`transform` is applied to the adversarial image before the model inside the loss.
Use it for the JPEG-aware attack (e.g. a differentiable JPEG, or a random
JPEG approximation with eot_samples > 1 for Expectation over Transformation).
`DifferentiableJPEG` is that transform. Results must still be validated through the real
pipeline, `distortions.roundtrip(x_adv, jpeg_quality)`, never through the approximation.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


class DifferentiableJPEG(nn.Module):
    """Differentiable stand-in for `distortions.roundtrip(x, quality)` inside the attack loss.

    uint8 quantization with a straight-through gradient, then kornia's differentiable
    JPEG codec (Reich et al., 2024; standard IJG tables, 4:2:0 chroma, like PIL).
    `quality` is an int, or a (low, high) range: each call then draws one quality per image
    uniformly from it, which with `pgd(..., eot_samples > 1)` is EOT over JPEG quality.
    H and W must be multiples of 16 (224 is).
    """

    def __init__(self, quality: int | tuple[int, int], generator: torch.Generator | None = None):
        super().__init__()
        self.quality = quality
        self.generator = generator

    def _qualities(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.quality, int):
            return torch.full((x.shape[0],), float(self.quality), device=x.device, dtype=x.dtype)
        low, high = self.quality
        q = torch.randint(low, high + 1, (x.shape[0],), generator=self.generator)
        return q.to(x.device, x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from kornia.enhance import jpeg_codec_differentiable

        x = x + (torch.round(x.clamp(0, 1) * 255) / 255 - x).detach()
        return jpeg_codec_differentiable(x, self._qualities(x)).clamp(0, 1)


def _apply_mask(delta: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return delta if mask is None else delta * mask.to(device=delta.device, dtype=delta.dtype)


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

    With `eot_samples > 1` the gradient is the mean over samples of the transformed loss
    (Expectation over Transformation). Each sample is backpropagated on its own and the
    gradients are accumulated, so peak memory is that of a single forward/backward pass.
    `generator` drives the random start; it may live on the CPU while x is on the GPU.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if eot_samples < 1:
        raise ValueError(f"eot_samples must be >= 1, got {eot_samples}")

    x = x.detach()
    if random_start:
        # Draw on the generator's own device (CPU when none is given), then move:
        # a CPU generator cannot drive torch.rand on CUDA and vice versa.
        gen_device = generator.device if generator is not None else torch.device("cpu")
        noise = torch.rand(x.shape, generator=generator, dtype=x.dtype,
                           device=gen_device).to(x.device)
        delta = _apply_mask((noise * 2 - 1) * eps, mask)
    else:
        delta = torch.zeros_like(x)
    x_adv = (x + delta).clamp(0, 1)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        grad = torch.zeros_like(x)
        for _ in range(eot_samples):
            inp = transform(x_adv) if transform is not None else x_adv
            loss_k = F.cross_entropy(model(inp), y)
            grad += torch.autograd.grad(loss_k / eot_samples, x_adv)[0].detach()

        step = _apply_mask(alpha * grad.sign(), mask)
        delta = torch.clamp(x_adv.detach() + step - x, -eps, eps)
        x_adv = (x + _apply_mask(delta, mask)).clamp(0, 1)

    return x_adv.detach()
