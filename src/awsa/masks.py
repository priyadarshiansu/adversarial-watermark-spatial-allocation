"""Block-level region masks (Module 3).

Masks are bool (B, 1, 28, 28), one cell per 8x8 pixel block; convert to pixels only with
`blocks_to_pixels`.
"""

import torch
import torch.nn.functional as F

from . import BLOCK


def _check_finite(saliency_blocks: torch.Tensor) -> None:
    if not torch.isfinite(saliency_blocks).all():
        raise ValueError("saliency_blocks contains NaN or inf; cannot rank blocks")


def top_fraction_mask(saliency_blocks: torch.Tensor, area: float) -> torch.Tensor:
    """Select the top-saliency fraction of 28x28 blocks.

    Args:
        saliency_blocks: (B, 1, H, W), usually (B, 1, 28, 28), values in [0, 1]
        area: fraction of blocks to select, e.g. 0.25 or 0.50

    Returns:
        Boolean mask of shape (B, 1, H, W)
    """
    if not (0.0 <= area <= 1.0):
        raise ValueError("area must be between 0 and 1")

    if saliency_blocks.ndim != 4 or saliency_blocks.shape[1] != 1:
        raise ValueError("saliency_blocks must have shape (B, 1, H, W)")
    _check_finite(saliency_blocks)

    b, _, h, w = saliency_blocks.shape
    n = h * w
    k = round(area * n)

    flat = saliency_blocks.reshape(b, n)
    mask = torch.zeros_like(flat, dtype=torch.bool)

    if k == 0:
        return mask.reshape(b, 1, h, w)

    order = torch.argsort(flat, dim=1, descending=True, stable=True)
    top_idx = order[:, :k]
    mask.scatter_(1, top_idx, True)

    return mask.reshape(b, 1, h, w)


def bottom_fraction_mask(
    saliency_blocks: torch.Tensor,
    area: float,
    exclude: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select the lowest-saliency fraction of blocks.

    Blocks marked True in `exclude` can never be selected.
    This guarantees that high- and low-saliency masks can be disjoint.
    """
    if not (0.0 <= area <= 1.0):
        raise ValueError("area must be between 0 and 1")

    if saliency_blocks.ndim != 4 or saliency_blocks.shape[1] != 1:
        raise ValueError("saliency_blocks must have shape (B, 1, H, W)")
    _check_finite(saliency_blocks)

    b, _, h, w = saliency_blocks.shape
    n = h * w
    k = round(area * n)

    flat = saliency_blocks.reshape(b, n)

    if exclude is not None:
        excluded = exclude.reshape(b, n)
        available = (~excluded).sum(dim=1)

        if torch.any(available < k):
            raise ValueError("not enough unexcluded blocks for requested area")

        flat = flat.clone()
        flat[excluded] = float("inf")

    mask = torch.zeros((b, n), dtype=torch.bool, device=saliency_blocks.device)

    if k == 0:
        return mask.reshape(b, 1, h, w)

    order = torch.argsort(flat, dim=1, descending=False, stable=True)
    low_idx = order[:, :k]
    mask.scatter_(1, low_idx, True)

    return mask.reshape(b, 1, h, w)


def random_disjoint_masks(shape, area: float, generator=None, device=None):
    """Create two disjoint random masks of equal area.

    Args:
        shape: tuple like (B, 1, H, W)
        area: fraction of blocks in each mask
        generator: optional CPU torch.Generator for reproducibility
        device: device for the returned masks (sampling always happens on the CPU,
            so results are identical across devices for a given seed)

    Returns:
        (mask_a, mask_b), each boolean of shape (B, 1, H, W)
    """
    if not (0.0 <= area <= 1.0):
        raise ValueError("area must be between 0 and 1")

    if len(shape) != 4 or shape[1] != 1:
        raise ValueError("shape must be (B, 1, H, W)")

    b, _, h, w = shape
    n = h * w
    k = round(area * n)

    if 2 * k > n:
        raise ValueError("area too large: cannot create two disjoint masks")

    mask_a = torch.zeros((b, n), dtype=torch.bool)
    mask_b = torch.zeros((b, n), dtype=torch.bool)

    for i in range(b):
        perm = torch.randperm(n, generator=generator)
        idx_a = perm[:k]
        idx_b = perm[k:2 * k]

        mask_a[i, idx_a] = True
        mask_b[i, idx_b] = True

    return (
        mask_a.reshape(b, 1, h, w).to(device),
        mask_b.reshape(b, 1, h, w).to(device),
    )


def blocks_to_pixels(
    block_mask: torch.Tensor,
    block: int = BLOCK,
) -> torch.Tensor:
    """Expand a block mask (B, 1, H, W) to a float32 pixel mask (B, 1, H*block, W*block).

    Block (i, j) maps to pixels [block*i : block*i + block, block*j : block*j + block],
    i.e. [8i:8i+8, 8j:8j+8] for the default 8x8 blocks (row i, column j).
    """
    if block_mask.ndim != 4 or block_mask.shape[1] != 1:
        raise ValueError("block_mask must have shape (B, 1, H, W)")

    pixel_mask = block_mask.to(torch.float32)

    pixel_mask = (
        pixel_mask
        .repeat_interleave(block, dim=2)
        .repeat_interleave(block, dim=3)
    )

    return pixel_mask


def block_variance(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Per-block variance of luma, shape (B, 1, H/block, W/block), float32.

    Luma Y = 0.299 R + 0.587 G + 0.114 B; variance = E[Y^2] - E[Y]^2 over each block,
    clamped at 0. Used for the smoothness-confound check (proposal section 7).
    """
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("x must have shape (B, 3, H, W)")
    x = x.float()
    y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    mean = F.avg_pool2d(y, block)
    mean_sq = F.avg_pool2d(y * y, block)
    return (mean_sq - mean * mean).clamp_min(0)
