import pytest
import torch
import torch.nn.functional as F

from awsa.distortions import quantize_uint8, roundtrip
from awsa.masks import blocks_to_pixels
from awsa.metrics import bit_error_rate
from awsa.watermark import (
    PAYLOAD_BITS,
    _dct2,
    _embed_bit,
    _extract_bit,
    _idct2,
    _merge_blocks,
    _ordered_selected_blocks,
    _rgb_to_ycbcr,
    _split_blocks,
    _ycbcr_to_rgb,
    block_bit_index,
    embed,
    extract,
    extract_soft,
    payload_coefficients,
)


def test_dct_roundtrip():
    x = torch.rand(4, 8, 8)

    coeffs = _dct2(x)
    reconstructed = _idct2(coeffs)

    error = (x - reconstructed).abs().max()

    assert error < 1e-5



def test_embed_and_extract_one():
    block = torch.rand(8, 8)

    watermarked = _embed_bit(block, bit=1, strength=4.0)

    recovered = _extract_bit(watermarked)

    assert recovered == 1


def test_embed_and_extract_zero():
    block = torch.rand(8, 8)

    watermarked = _embed_bit(block, bit=0, strength=4.0)

    recovered = _extract_bit(watermarked)

    assert recovered == 0

def test_split_and_merge_blocks():
    x = torch.rand(2, 1, 224, 224)

    blocks = _split_blocks(x)

    assert blocks.shape == (2, 1, 28, 28, 8, 8)

    reconstructed = _merge_blocks(blocks)

    assert torch.allclose(x, reconstructed)

def test_rgb_ycbcr_roundtrip():
    x = torch.rand(2, 3, 224, 224)

    y, cb, cr = _rgb_to_ycbcr(x)
    reconstructed = _ycbcr_to_rgb(y, cb, cr)

    assert reconstructed.shape == x.shape
    assert torch.allclose(x, reconstructed, atol=1e-4)

def test_selected_blocks_are_deterministic():
    mask = torch.ones(1, 1, 28, 28, dtype=torch.bool)

    first = _ordered_selected_blocks(mask, key=123)[0]
    second = _ordered_selected_blocks(mask, key=123)[0]

    assert torch.equal(first, second)
    assert len(first) == 784

def test_embed_returns_valid_image():
    x = torch.rand(1, 3, 224, 224)

    bits = torch.randint(
        0,
        2,
        (1, 32),
        dtype=torch.int64,
    )

    mask = torch.ones(
        1,
        1,
        28,
        28,
        dtype=torch.bool,
    )

    watermarked = embed(
        x,
        bits,
        mask,
        strength=0.1,
        key=123,
    )

    assert watermarked.shape == x.shape
    assert watermarked.min() >= 0
    assert watermarked.max() <= 1

    # The watermark should actually modify the image.
    assert not torch.equal(x, watermarked)

def test_embed_extract_full_payload():
    # Keep pixels away from 0 and 1 so clipping does not interfere
    # with this first clean round-trip test.
    x = torch.rand(1, 3, 224, 224) * 0.6 + 0.2

    bits = torch.randint(
        0,
        2,
        (1, 32),
        dtype=torch.int64,
    )

    mask = torch.ones(
        1,
        1,
        28,
        28,
        dtype=torch.bool,
    )

    watermarked = embed(
        x,
        bits,
        mask,
        strength=0.1,
        key=123,
    )

    recovered = extract(
        watermarked,
        mask,
        key=123,
    )

    assert recovered.shape == bits.shape
    assert torch.equal(recovered, bits)

def test_payload_coefficients():
    assert payload_coefficients() == [(2, 3), (3, 2)]

def test_watermark_survives_uint8_quantization():
    x = torch.rand(1, 3, 224, 224) * 0.6 + 0.2

    bits = torch.randint(
        0,
        2,
        (1, 32),
        dtype=torch.int64,
    )

    mask = torch.ones(
        1,
        1,
        28,
        28,
        dtype=torch.bool,
    )

    watermarked = embed(
        x,
        bits,
        mask,
        strength=0.1,
        key=123,
    )

    processed = quantize_uint8(watermarked)

    recovered = extract(
        processed,
        mask,
        key=123,
    )

    assert torch.equal(recovered, bits)

def test_watermark_changes_only_selected_blocks():
    x = torch.rand(1, 3, 224, 224) * 0.6 + 0.2

    bits = torch.randint(0, 2, (1, 32), dtype=torch.int64)

    mask = torch.zeros(1, 1, 28, 28, dtype=torch.bool)

    # Select exactly 32 blocks.
    mask.view(-1)[:32] = True

    watermarked = embed(
        x,
        bits,
        mask,
        strength=0.06,
        key=123,
    )

    pixel_mask = blocks_to_pixels(mask).bool().expand_as(x)

    # Absolutely nothing outside the selected region should change.
    assert torch.equal(
        watermarked[~pixel_mask],
        x[~pixel_mask],
    )

    # Something inside the selected region should change.
    assert not torch.equal(
        watermarked[pixel_mask],
        x[pixel_mask],
    )


# --- helpers for the tests below -------------------------------------------------------


def _smooth_images(n: int, seed: int = 0) -> torch.Tensor:
    """Low-pass random images (upsampled 28x28 noise) in [0.2, 0.8], so JPEG is benign."""
    g = torch.Generator().manual_seed(seed)
    coarse = torch.rand(n, 3, 28, 28, generator=g)
    x = F.interpolate(coarse, size=(224, 224), mode="bilinear", align_corners=False)
    return x * 0.6 + 0.2


def _random_bits(n: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n, PAYLOAD_BITS), generator=g, dtype=torch.int64)


def _random_masks(n: int, area: float, seed: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    k = round(area * 784)
    mask = torch.zeros(n, 784, dtype=torch.bool)
    for i in range(n):
        mask[i, torch.randperm(784, generator=g)[:k]] = True
    return mask.view(n, 1, 28, 28)


# --- batch independence, keys, robustness ------------------------------------------------


def test_extract_is_batch_independent():
    x = _smooth_images(4)
    bits = _random_bits(4)
    mask = _random_masks(4, 0.5)

    x_wm = roundtrip(embed(x, bits, mask, strength=0.08, key=7))

    batched = extract(x_wm, mask, key=7)
    single = torch.cat([extract(x_wm[i:i + 1], mask[i:i + 1], key=7) for i in range(4)])

    assert torch.equal(batched, single)
    assert torch.equal(batched, bits)


def test_embed_is_batch_independent():
    x = _smooth_images(3)
    bits = _random_bits(3)
    mask = _random_masks(3, 0.25)

    batched = embed(x, bits, mask, strength=0.06, key=5)
    single = torch.cat(
        [embed(x[i:i + 1], bits[i:i + 1], mask[i:i + 1], strength=0.06, key=5) for i in range(3)]
    )

    assert torch.allclose(batched, single, atol=1e-6)


def test_wrong_key_gives_chance_ber():
    x = _smooth_images(4)
    bits = _random_bits(4)
    mask = torch.ones(4, 1, 28, 28, dtype=torch.bool)

    x_wm = roundtrip(embed(x, bits, mask, strength=0.08, key=123))

    assert bit_error_rate(bits, extract(x_wm, mask, key=123)).max() == 0
    assert bit_error_rate(bits, extract(x_wm, mask, key=124)).mean() > 0.3


def test_survives_jpeg_q75():
    x = _smooth_images(2, seed=3)
    bits = _random_bits(2, seed=4)
    mask = _random_masks(2, 0.5, seed=5)

    x_wm = embed(x, bits, mask, strength=0.08, key=123)
    recovered = extract(roundtrip(x_wm, 75), mask, key=123)

    assert bit_error_rate(bits, recovered).max() == 0


# --- tie-breaking -------------------------------------------------------------------------


def _add_margin(x: torch.Tensor, flat_index: int, margin: float) -> torch.Tensor:
    """Add `margin` to c[2,3] - c[3,2] of one block's luma by editing pixels."""
    coeffs = torch.zeros(8, 8)
    coeffs[2, 3] = margin / 2
    coeffs[3, 2] = -margin / 2
    delta = _idct2(coeffs)
    row, col = divmod(flat_index, 28)
    x = x.clone()
    x[:, :, row * 8:(row + 1) * 8, col * 8:(col + 1) * 8] += delta
    return x


@pytest.mark.parametrize("margins, expected", [((0.2, -0.1), 1), ((0.1, -0.2), 0)])
def test_tied_vote_is_broken_by_margin_sum(margins, expected):
    # 64 selected blocks -> every bit has exactly 2 copies.
    mask = torch.zeros(1, 1, 28, 28, dtype=torch.bool)
    mask.view(-1)[100:164] = True

    index = block_bit_index(mask, key=9).view(-1)
    copies = torch.nonzero(index == 0).squeeze(1).tolist()
    assert len(copies) == 2

    x = torch.full((1, 3, 224, 224), 0.5)
    for flat_index, margin in zip(copies, margins):
        x = _add_margin(x, flat_index, margin)

    soft = extract_soft(x, mask, key=9).view(-1)
    assert (soft[copies[0]] > 0) != (soft[copies[1]] > 0)  # one vote each: a tie

    assert extract(x, mask, key=9)[0, 0].item() == expected


# --- mechanism-analysis helpers -----------------------------------------------------------


def test_block_bit_index_layout():
    mask = _random_masks(2, 0.35)
    index = block_bit_index(mask, key=11)

    assert index.shape == (2, 1, 28, 28)
    assert index.dtype == torch.int64
    assert torch.all(index[~mask] == -1)
    assert torch.all(index[mask] >= 0)

    for i in range(2):
        counts = torch.bincount(index[i][mask[i]], minlength=PAYLOAD_BITS)
        assert len(counts) == PAYLOAD_BITS
        assert counts.min() >= 1
        assert counts.max() - counts.min() <= 1


def test_extract_soft_matches_hard_decisions():
    x = _smooth_images(2)
    bits = _random_bits(2)
    mask = _random_masks(2, 0.25)

    x_wm = roundtrip(embed(x, bits, mask, strength=0.08, key=3))
    soft = extract_soft(x_wm, mask, key=3)
    index = block_bit_index(mask, key=3)

    assert soft.shape == (2, 1, 28, 28)
    assert soft.dtype == torch.float32
    assert torch.all(soft[~mask] == 0)

    # Every copy decodes correctly after uint8, so sign(soft) equals the payload bit per block.
    per_block_bits = torch.gather(bits, 1, index.flatten(1).clamp(min=0)).view_as(index)
    assert torch.equal((soft > 0)[mask], per_block_bits.bool()[mask])
    assert torch.equal(extract(x_wm, mask, key=3), bits)


# --- input handling -----------------------------------------------------------------------


def test_bool_payload():
    x = _smooth_images(1)
    bits = _random_bits(1).bool()
    mask = _random_masks(1, 0.5)

    recovered = extract(roundtrip(embed(x, bits, mask, strength=0.08, key=1)), mask, key=1)

    assert torch.equal(recovered, bits.long())


def test_too_few_blocks_raises():
    x = _smooth_images(1)
    bits = _random_bits(1)
    mask = torch.zeros(1, 1, 28, 28, dtype=torch.bool)
    mask.view(-1)[:31] = True

    with pytest.raises(ValueError):
        embed(x, bits, mask, strength=0.08, key=1)
    with pytest.raises(ValueError):
        extract(x, mask, key=1)
