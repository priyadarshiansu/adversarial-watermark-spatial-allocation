import torch
from . import BLOCK

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

    b, _, h, w = saliency_blocks.shape
    n = h * w
    k = round(area * n)

    flat = saliency_blocks.view(b, n)
    mask = torch.zeros_like(flat, dtype=torch.bool)

    if k == 0:
        return mask.view(b, 1, h, w)

    order = torch.argsort(flat, dim=1, descending=True, stable=True)
    top_idx = order[:, :k]
    mask.scatter_(1, top_idx, True)

    return mask.view(b, 1, h, w)


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

    b, _, h, w = saliency_blocks.shape
    n = h * w
    k = round(area * n)

    flat = saliency_blocks.view(b, n)

    if exclude is not None:
        excluded = exclude.view(b, n)
        available = (~excluded).sum(dim=1)

        if torch.any(available < k):
            raise ValueError("not enough unexcluded blocks for requested area")

        flat = flat.clone()
        flat[excluded] = float("inf")

    mask = torch.zeros((b, n), dtype=torch.bool, device=saliency_blocks.device)

    if k == 0:
        return mask.view(b, 1, h, w)

    order = torch.argsort(flat, dim=1, descending=False, stable=True)
    low_idx = order[:, :k]
    mask.scatter_(1, low_idx, True)

    return mask.view(b, 1, h, w)


def random_disjoint_masks(shape, area: float, generator=None):
    """Create two disjoint random masks of equal area.

    Args:
        shape: tuple like (B, 1, H, W)
        area: fraction of blocks in each mask
        generator: optional torch.Generator for reproducibility

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
        mask_a.view(b, 1, h, w),
        mask_b.view(b, 1, h, w),
    )


def blocks_to_pixels(
    block_mask: torch.Tensor,
    block: int = BLOCK,
) -> torch.Tensor:
    if block_mask.ndim != 4 or block_mask.shape[1] != 1:
        raise ValueError("block_mask must have shape (B, 1, H, W)")

    pixel_mask = block_mask.to(torch.float32)

    pixel_mask = (
        pixel_mask
        .repeat_interleave(block, dim=2)
        .repeat_interleave(block, dim=3)
    )

    return pixel_mask