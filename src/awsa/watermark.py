"""Block-DCT invisible watermark (Module 4).

Each payload bit is written into the sign of the difference between two mid-frequency
DCT coefficients, c[2,3] - c[3,2], of an 8x8 block of the luma (Y) channel. Luma only,
so JPEG 4:2:0 chroma subsampling does not destroy it. The 32 payload bits are spread
over the masked blocks in a keyed pseudo-random order (bit j % 32 goes to the j-th block
of the permutation), so every bit gets several copies and is decoded by majority vote.

The decoder is blind: it never sees the original image. The block mask and the integer
key are secrets shared between the embedder and the decoder (side information), because
Grad-CAM cannot be recomputed after attack + JPEG; see docs/interfaces.md.

Public API: `embed`, `extract`, `payload_coefficients`, plus `block_bit_index` and
`extract_soft` for the per-block mechanism analysis (proposal §5).
"""
import math

import torch

PAYLOAD_BITS = 32

BLOCK_SIZE = 8

# The coefficient pair whose difference carries the bit (see payload_coefficients()).
_C_POS = (2, 3)
_C_NEG = (3, 2)


def _dct_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create the orthonormal 8x8 DCT-II transform matrix."""
    n = torch.arange(BLOCK_SIZE, device=device, dtype=dtype)
    k = torch.arange(BLOCK_SIZE, device=device, dtype=dtype).unsqueeze(1)

    matrix = torch.cos(math.pi / BLOCK_SIZE * (n + 0.5) * k)

    matrix[0] *= math.sqrt(1.0 / BLOCK_SIZE)
    matrix[1:] *= math.sqrt(2.0 / BLOCK_SIZE)

    return matrix


def _dct2(blocks: torch.Tensor) -> torch.Tensor:
    """Apply a 2D DCT to one or more 8x8 blocks (any leading dims)."""
    c = _dct_matrix(blocks.device, blocks.dtype)
    return c @ blocks @ c.T


def _idct2(coeffs: torch.Tensor) -> torch.Tensor:
    """Invert the 2D DCT."""
    c = _dct_matrix(coeffs.device, coeffs.dtype)
    return c.T @ coeffs @ c


def _embed_bit(
    block: torch.Tensor,
    bit: int,
    strength: float,
) -> torch.Tensor:
    """
    Embed one bit into one 8x8 block with the minimum change needed.

    bit = 1 -> DCT[2,3] - DCT[3,2] >= strength
    bit = 0 -> DCT[3,2] - DCT[2,3] >= strength
    """
    coeffs = _dct2(block.clone())

    c1 = coeffs[_C_POS]
    c2 = coeffs[_C_NEG]

    difference = c1 - c2

    if bit == 1:
        if difference < strength:
            adjustment = (strength - difference) / 2
            coeffs[_C_POS] += adjustment
            coeffs[_C_NEG] -= adjustment

    else:
        if difference > -strength:
            adjustment = (difference + strength) / 2
            coeffs[_C_POS] -= adjustment
            coeffs[_C_NEG] += adjustment

    return _idct2(coeffs)


def _block_margins(blocks: torch.Tensor) -> torch.Tensor:
    """Per-block margin c[2,3] - c[3,2] for blocks of shape (..., 8, 8) -> (...)."""
    coeffs = _dct2(blocks)
    return coeffs[..., _C_POS[0], _C_POS[1]] - coeffs[..., _C_NEG[0], _C_NEG[1]]


def _extract_bit(block: torch.Tensor) -> int:
    """Recover one embedded bit from one 8x8 block (1 iff the margin is positive)."""
    return int(_block_margins(block) > 0)


def _split_blocks(channel: torch.Tensor) -> torch.Tensor:
    """
    Split a single-channel image into non-overlapping 8x8 blocks.

    Input:
        (B, 1, 224, 224)

    Output:
        (B, 1, 28, 28, 8, 8)
    """
    return channel.unfold(2, BLOCK_SIZE, BLOCK_SIZE).unfold(3, BLOCK_SIZE, BLOCK_SIZE)


def _merge_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """
    Reassemble 8x8 blocks into the original image channel.
    """
    b, c, rows, cols, h, w = blocks.shape

    return blocks.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, rows * h, cols * w)


def _check_capacity(ordered_blocks: list[torch.Tensor]) -> None:
    for b, selected in enumerate(ordered_blocks):
        if len(selected) < PAYLOAD_BITS:
            raise ValueError(
                f"Need at least {PAYLOAD_BITS} selected blocks, "
                f"but image {b} has only {len(selected)}."
            )


def embed(
    x: torch.Tensor,
    bits: torch.Tensor,
    mask: torch.Tensor,
    strength: float,
    key: int,
) -> torch.Tensor:
    """
    Embed a 32-bit payload into the selected 8x8 blocks.

    x:
        (B, 3, 224, 224) RGB image in [0, 1]

    bits:
        (B, 32) payload bits (int or bool)

    mask:
        (B, 1, 28, 28) boolean block mask; needs at least 32 selected blocks per image,
        otherwise ValueError.

    strength:
        minimum margin |c[2,3] - c[3,2]| enforced in each selected block, in
        orthonormal-DCT units of the Y channel with Y in [0, 1]. It is a
        coefficient-margin parameter, NOT a QIM step: a block whose coefficient
        difference already has the right sign and size is left untouched, and
        otherwise the change is (strength -/+ existing difference) / 2 per
        coefficient. The distortion therefore depends on each block's own content.

    key:
        seeds the reproducible pseudo-random ordering of the selected blocks; the
        ordering depends only on (mask, key), never on the batch position.
    """
    if bits.shape[1] != PAYLOAD_BITS:
        raise ValueError(f"Expected {PAYLOAD_BITS} payload bits.")

    y, _, _ = _rgb_to_ycbcr(x)

    blocks = _split_blocks(y).clone()
    cols = mask.shape[-1]

    ordered_blocks = _ordered_selected_blocks(mask, key)
    _check_capacity(ordered_blocks)

    for b in range(x.shape[0]):
        for j, flat_index in enumerate(ordered_blocks[b].tolist()):
            row, col = divmod(flat_index, cols)
            bit = int(bits[b, j % PAYLOAD_BITS].item())

            blocks[b, 0, row, col] = _embed_bit(
                blocks[b, 0, row, col],
                bit=bit,
                strength=strength,
            )

    y_watermarked = _merge_blocks(blocks)

    # Changing luminance Y while keeping chroma fixed is equivalent to adding the
    # same luminance change to R, G and B. Pixels outside selected blocks stay unchanged.
    delta_y = y_watermarked - y

    return (x + delta_y.repeat(1, 3, 1, 1)).clamp(0, 1)


def block_bit_index(mask: torch.Tensor, key: int) -> torch.Tensor:
    """Which payload bit each masked block carries.

    Returns (B, 1, H, W) int64 (H, W = the mask's block grid, normally 28x28) with values
    in 0..31 inside the mask and -1 outside. Uses the same keyed ordering as embed/extract.
    """
    b, _, h, w = mask.shape
    index = torch.full((b, h * w), -1, dtype=torch.int64, device=mask.device)

    for i, selected in enumerate(_ordered_selected_blocks(mask, key)):
        index[i, selected] = torch.arange(len(selected), device=mask.device) % PAYLOAD_BITS

    return index.view(b, 1, h, w)


def extract_soft(x: torch.Tensor, mask: torch.Tensor, key: int) -> torch.Tensor:
    """Per-block soft decision: the margin c[2,3] - c[3,2] of the Y-channel DCT.

    Returns (B, 1, H, W) float32, 0 outside the mask. Positive means the block votes 1.
    Per-block bit error: `(soft > 0) != bits[idx]` with `idx = block_bit_index(mask, key)`.
    `key` is unused here (margins do not depend on the ordering); it is kept so the
    signature mirrors `extract`.
    """
    del key
    y, _, _ = _rgb_to_ycbcr(x)
    margins = _block_margins(_split_blocks(y)).float()  # (B, 1, H, W)
    return torch.where(mask.to(margins.device).bool(), margins, torch.zeros_like(margins))


def extract(
    x: torch.Tensor,
    mask: torch.Tensor,
    key: int,
) -> torch.Tensor:
    """
    Recover the 32-bit payload from the selected 8x8 blocks (blind: no original needed).

    Each payload bit is embedded in several blocks. Every copy votes
    (1 iff c[2,3] > c[3,2]) and the bit is decided by majority vote. A tied vote
    (possible with an even number of copies) is broken by the sign of the summed
    margin c[2,3] - c[3,2] over that bit's copies; an exactly zero sum decodes as 0.

    Returns (B, 32) int64. Raises ValueError if an image has fewer than 32 masked blocks.
    """
    _check_capacity(_ordered_selected_blocks(mask, key))

    soft = extract_soft(x, mask, key).flatten(1)  # (B, H*W)
    index = block_bit_index(mask, key).to(soft.device).flatten(1)
    inside = index >= 0
    scatter_index = index.clamp(min=0)

    def per_bit_sum(values: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((soft.shape[0], PAYLOAD_BITS), dtype=values.dtype, device=soft.device)
        return out.scatter_add_(1, scatter_index, torch.where(inside, values, 0))

    ones = per_bit_sum((soft > 0).to(torch.int64))
    copies = per_bit_sum(inside.to(torch.int64))
    zeros = copies - ones
    margin_sum = per_bit_sum(soft)

    recovered = torch.where(ones == zeros, margin_sum > 0, ones > zeros)
    return recovered.to(torch.int64).to(x.device)


def payload_coefficients() -> list[tuple[int, int]]:
    """Return the DCT coefficient positions used to encode watermark bits."""
    return [_C_POS, _C_NEG]


def _rgb_to_ycbcr(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert RGB image tensor in [0, 1] to Y, Cb, Cr channels.
    Input shape:  (B, 3, H, W)
    Output shapes: each (B, 1, H, W)
    """
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

    return y, cb, cr


def _ycbcr_to_rgb(
    y: torch.Tensor,
    cb: torch.Tensor,
    cr: torch.Tensor,
) -> torch.Tensor:
    """
    Convert Y, Cb, Cr channels back to RGB.

    Test-only: not used by embed/extract (they add the luma change to R, G and B
    directly). Kept because tests/test_watermark.py checks the colour transform
    round-trips with it.
    """
    cb_shifted = cb - 0.5
    cr_shifted = cr - 0.5

    r = y + 1.402 * cr_shifted
    g = y - 0.344136 * cb_shifted - 0.714136 * cr_shifted
    b = y + 1.772 * cb_shifted

    return torch.cat([r, g, b], dim=1).clamp(0, 1)


def _ordered_selected_blocks(
    mask: torch.Tensor,
    key: int,
) -> list[torch.Tensor]:
    """
    Return the selected flat block indices for each image, shuffled deterministically
    with `key` alone (not the batch position, so an image decodes the same alone or
    inside a batch).

    mask shape: (B, 1, H, W), normally (B, 1, 28, 28); indices are in 0..H*W-1.
    """
    ordered = []

    for b in range(mask.shape[0]):
        selected = torch.nonzero(mask[b, 0].reshape(-1).cpu(), as_tuple=False).squeeze(1)

        generator = torch.Generator()
        generator.manual_seed(key)

        permutation = torch.randperm(len(selected), generator=generator)

        ordered.append(selected[permutation].to(mask.device))

    return ordered
