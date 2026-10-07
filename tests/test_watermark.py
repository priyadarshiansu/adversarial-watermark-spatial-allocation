import torch

from awsa.distortions import quantize_uint8
from awsa.masks import blocks_to_pixels
from awsa.watermark import (
    _dct2,
    _embed_bit,
    _extract_bit,
    _idct2,
    _merge_blocks,
    _ordered_selected_blocks,
    _rgb_to_ycbcr,
    _split_blocks,
    _ycbcr_to_rgb,
    embed,
    extract,
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