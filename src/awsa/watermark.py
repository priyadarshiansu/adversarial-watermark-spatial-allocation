"""Block-DCT invisible watermark (Module 4).

Embeds in the luma (Y) channel so JPEG chroma subsampling does not destroy it.
The region mask is treated as a secret key shared with the decoder (non-blind
decoding with side information); see docs/interfaces.md.
"""
import math

import torch

PAYLOAD_BITS = 32

BLOCK_SIZE = 8


def _dct_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create the orthonormal 8x8 DCT-II transform matrix."""
    n = torch.arange(BLOCK_SIZE, device=device, dtype=dtype)
    k = torch.arange(BLOCK_SIZE, device=device, dtype=dtype).unsqueeze(1)

    matrix = torch.cos(
        math.pi / BLOCK_SIZE * (n + 0.5) * k
    )

    matrix[0] *= math.sqrt(1.0 / BLOCK_SIZE)
    matrix[1:] *= math.sqrt(2.0 / BLOCK_SIZE)

    return matrix


def _dct2(blocks: torch.Tensor) -> torch.Tensor:
    """Apply a 2D DCT to one or more 8x8 blocks."""
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

    c1 = coeffs[2, 3]
    c2 = coeffs[3, 2]

    difference = c1 - c2

    if bit == 1:
        if difference < strength:
            adjustment = (strength - difference) / 2
            coeffs[2, 3] += adjustment
            coeffs[3, 2] -= adjustment

    else:
        if difference > -strength:
            adjustment = (difference + strength) / 2
            coeffs[2, 3] -= adjustment
            coeffs[3, 2] += adjustment

    return _idct2(coeffs)

def _extract_bit(block: torch.Tensor) -> int:
    """Recover one embedded bit from one 8x8 block."""
    coeffs = _dct2(block)

    return int(coeffs[2, 3] > coeffs[3, 2])

def _split_blocks(channel: torch.Tensor) -> torch.Tensor:
    """
    Split a single-channel image into non-overlapping 8x8 blocks.

    Input:
        (B, 1, 224, 224)

    Output:
        (B, 1, 28, 28, 8, 8)
    """
    return channel.unfold(2, BLOCK_SIZE, BLOCK_SIZE).unfold(
        3, BLOCK_SIZE, BLOCK_SIZE
    )


def _merge_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """
    Reassemble 8x8 blocks into the original image channel.
    """
    b, c, rows, cols, h, w = blocks.shape

    return (
        blocks.permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(b, c, rows * h, cols * w)
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
        (B, 32) payload bits

    mask:
        (B, 1, 28, 28) boolean block mask

    strength:
        separation enforced between the two DCT coefficients

    key:
        determines the reproducible ordering of selected blocks
    """
    if bits.shape[1] != PAYLOAD_BITS:
        raise ValueError(f"Expected {PAYLOAD_BITS} payload bits.")

    y, _, _ = _rgb_to_ycbcr(x)

    blocks = _split_blocks(y).clone()

    ordered_blocks = _ordered_selected_blocks(mask, key)

    for b in range(x.shape[0]):
        selected = ordered_blocks[b]

        if len(selected) < PAYLOAD_BITS:
            raise ValueError(
                f"Need at least {PAYLOAD_BITS} selected blocks, "
                f"but image {b} has only {len(selected)}."
            )

        for j, flat_index in enumerate(selected):
            bit_index = j % PAYLOAD_BITS

            row = int(flat_index) // 28
            col = int(flat_index) % 28

            bit = int(bits[b, bit_index].item())

            blocks[b, 0, row, col] = _embed_bit(
                blocks[b, 0, row, col],
                bit=bit,
                strength=strength,
            )

    y_watermarked = _merge_blocks(blocks)
    
    #Changing Luminance Y while keeping chroma fixed
    #is equivalent to adding same luminance change to R, G, and B
    #Also guarantees that pixels outside selected blocks stay unchanged

    delta_y = y_watermarked - y

    return (x + delta_y.repeat(1,3,1,1)).clamp(0,1)

def extract(
    x: torch.Tensor,
    mask: torch.Tensor,
    key: int,
) -> torch.Tensor:
    """
    Recover the 32-bit payload from the selected 8x8 blocks.

    Each payload bit may have been embedded in several blocks.
    We decode every copy and use majority vote.
    """
    y, _, _ = _rgb_to_ycbcr(x)
    blocks = _split_blocks(y)

    ordered_blocks = _ordered_selected_blocks(mask, key)

    recovered = torch.zeros(
        (x.shape[0], PAYLOAD_BITS),
        dtype=torch.int64,
        device=x.device,
    )

    for b in range(x.shape[0]):
        selected = ordered_blocks[b]

        if len(selected) < PAYLOAD_BITS:
            raise ValueError(
                f"Need at least {PAYLOAD_BITS} selected blocks, "
                f"but image {b} has only {len(selected)}."
            )

        votes = [[] for _ in range(PAYLOAD_BITS)]

        for j, flat_index in enumerate(selected):
            bit_index = j % PAYLOAD_BITS

            row = int(flat_index) // 28
            col = int(flat_index) % 28

            vote = _extract_bit(
                blocks[b, 0, row, col]
            )

            votes[bit_index].append(vote)

        for bit_index in range(PAYLOAD_BITS):
            ones = sum(votes[bit_index])
            zeros = len(votes[bit_index]) - ones

            recovered[b, bit_index] = int(ones > zeros)

    return recovered


def payload_coefficients() -> list[tuple[int, int]]:
    """Return the DCT coefficient positions used to encode watermark bits."""
    return [(2, 3), (3, 2)]

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
    Return the selected block indices for each image,
    shuffled deterministically using `key`.

    mask shape: (B, 1, 28, 28)

    Each returned tensor contains flattened block indices
    in the range 0..783.
    """
    ordered = []

    for b in range(mask.shape[0]):
        selected = torch.nonzero(
            mask[b, 0].reshape(-1),
            as_tuple=False,
        ).squeeze(1)

        generator = torch.Generator()
        generator.manual_seed(key + b)

        permutation = torch.randperm(
            len(selected),
            generator=generator,
        )

        ordered.append(selected[permutation].to(mask.device))

    return ordered