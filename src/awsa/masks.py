"""Block-level spatial masks (Module 3).

Masks live on the 8x8 block grid: bool tensors of shape (B, 1, 28, 28) for 224x224
images, so every region boundary lines up with the JPEG / block-DCT grid.
"""

import torch
import torch.nn.functional as F

from . import BLOCK


def blocks_to_pixels(block_mask: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """(B, 1, 28, 28) block mask -> (B, 1, 224, 224) float pixel mask (nearest upsampling)."""
    return F.interpolate(block_mask.float(), scale_factor=block, mode="nearest")


def top_fraction_mask(saliency_blocks: torch.Tensor, area: float) -> torch.Tensor:
    """Select the `area` fraction of blocks with the highest saliency (ties broken by index).

    saliency_blocks: (B, 1, 28, 28) float. Returns bool (B, 1, 28, 28) with exactly
    round(area * 784) True cells per image, so all masked arms have equal area.
    """
    raise NotImplementedError("Module 3")


def random_disjoint_masks(
    shape: tuple[int, int, int, int], area: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two non-overlapping random block masks, each covering `area` of the grid (RQ3 control)."""
    raise NotImplementedError("Module 3")
