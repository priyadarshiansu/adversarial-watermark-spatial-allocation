"""Evaluation metrics (Module 1). All inputs are [0, 1] tensors (B, 3, H, W); outputs are (B,)."""

import torch


def attack_success(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1.0 where the prediction differs from the true label (untargeted success)."""
    return (logits.argmax(dim=1) != y).float()


def bit_error_rate(bits_true: torch.Tensor, bits_pred: torch.Tensor) -> torch.Tensor:
    """Fraction of payload bits decoded incorrectly, per image."""
    return (bits_true != bits_pred).float().mean(dim=1)


def psnr(x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Module 1")


def ssim(x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Module 1")


def lpips_distance(x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Module 1")
