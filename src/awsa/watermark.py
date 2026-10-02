"""Block-DCT invisible watermark (Module 4).

Embeds in the luma (Y) channel so JPEG chroma subsampling does not destroy it.
The region mask is treated as a secret key shared with the decoder (non-blind
decoding with side information); see docs/interfaces.md.
"""

import torch

PAYLOAD_BITS = 32


def embed(
    x: torch.Tensor, bits: torch.Tensor, mask: torch.Tensor, strength: float, key: int
) -> torch.Tensor:
    """Embed `bits` (B, 32) into the 8x8 blocks selected by `mask` (B, 1, 28, 28).

    Returns the watermarked image in [0, 1], same shape as x.
    """
    raise NotImplementedError("Module 4")


def extract(x: torch.Tensor, mask: torch.Tensor, key: int) -> torch.Tensor:
    """Recover the (B, 32) bit tensor from the blocks selected by `mask`."""
    raise NotImplementedError("Module 4")


def payload_coefficients() -> list[tuple[int, int]]:
    """The (u, v) DCT coefficient positions that carry the payload (for the mechanism analysis)."""
    raise NotImplementedError("Module 4")
