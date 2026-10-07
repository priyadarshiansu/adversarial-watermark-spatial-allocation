import pytest
import torch

from awsa.masks import (
    block_variance,
    blocks_to_pixels,
    bottom_fraction_mask,
    random_disjoint_masks,
    top_fraction_mask,
)


def test_blocks_to_pixels_shape_and_alignment():
    m = torch.zeros(1, 1, 28, 28, dtype=torch.bool)
    m[0, 0, 0, 1] = True
    px = blocks_to_pixels(m)
    assert px.shape == (1, 1, 224, 224)
    assert px.sum() == 64
    assert px[0, 0, 0:8, 8:16].all()

def test_top_fraction_mask_exact_area():
    saliency = torch.arange(784, dtype=torch.float32).view(1, 1, 28, 28)

    mask = top_fraction_mask(saliency, area=0.25)

    assert mask.dtype == torch.bool
    assert mask.shape == (1, 1, 28, 28)
    assert mask.sum().item() == 196


def test_bottom_fraction_mask_is_disjoint():
    saliency = torch.arange(784, dtype=torch.float32).view(1, 1, 28, 28)

    high = top_fraction_mask(saliency, area=0.25)
    low = bottom_fraction_mask(saliency, area=0.25, exclude=high)

    assert low.sum().item() == 196
    assert (high & low).sum().item() == 0


def test_random_disjoint_masks():
    g = torch.Generator().manual_seed(0)

    mask_a, mask_b = random_disjoint_masks(
        (2, 1, 28, 28),
        area=0.25,
        generator=g,
    )

    assert mask_a.shape == (2, 1, 28, 28)
    assert mask_b.shape == (2, 1, 28, 28)

    assert torch.all(mask_a.sum(dim=(1, 2, 3)) == 196)
    assert torch.all(mask_b.sum(dim=(1, 2, 3)) == 196)

    assert not (mask_a & mask_b).any()

def test_top_fraction_mask_ties_break_by_index():
    saliency = torch.zeros(1, 1, 28, 28)

    mask = top_fraction_mask(saliency, area=2 / 784)

    flat = mask.view(-1)

    assert flat.sum().item() == 2
    assert flat[0]
    assert flat[1]


def test_blocks_to_pixels_alignment_interior_block():
    m = torch.zeros(1, 1, 28, 28, dtype=torch.bool)
    m[0, 0, 5, 17] = True
    px = blocks_to_pixels(m)
    assert px[0, 0, 40:48, 136:144].all()
    assert px.sum() == 64


def test_random_disjoint_masks_reproducible():
    shape = (3, 1, 28, 28)
    a1, b1 = random_disjoint_masks(shape, 0.25, generator=torch.Generator().manual_seed(1))
    a2, b2 = random_disjoint_masks(shape, 0.25, generator=torch.Generator().manual_seed(1))
    a3, _ = random_disjoint_masks(shape, 0.25, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
    assert not torch.equal(a1, a3)
    assert not torch.equal(a1[0], a1[1])  # each image in a batch gets its own draw


def test_random_disjoint_masks_device_and_area_limit():
    a, b = random_disjoint_masks((1, 1, 28, 28), 0.5, device="cpu")
    assert a.device.type == "cpu" and (a | b).all()
    with pytest.raises(ValueError):
        random_disjoint_masks((1, 1, 28, 28), 0.51)


def test_bottom_fraction_mask_without_exclude():
    saliency = torch.arange(784, dtype=torch.float32).view(1, 1, 28, 28)
    low = bottom_fraction_mask(saliency, area=0.25)
    assert low.sum().item() == 196
    assert low.view(-1)[:196].all()


def test_bottom_fraction_mask_not_enough_blocks():
    saliency = torch.rand(1, 1, 28, 28)
    high = top_fraction_mask(saliency, area=0.75)
    with pytest.raises(ValueError, match="not enough"):
        bottom_fraction_mask(saliency, area=0.5, exclude=high)


@pytest.mark.parametrize("area,expected", [(0.0, 0), (1.0, 784)])
def test_fraction_mask_extremes(area, expected):
    saliency = torch.rand(2, 1, 28, 28)
    assert (top_fraction_mask(saliency, area).sum(dim=(1, 2, 3)) == expected).all()
    assert (bottom_fraction_mask(saliency, area).sum(dim=(1, 2, 3)) == expected).all()


def test_fraction_masks_reject_non_finite():
    saliency = torch.rand(1, 1, 28, 28)
    saliency[0, 0, 3, 3] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        top_fraction_mask(saliency, 0.25)
    with pytest.raises(ValueError, match="NaN"):
        bottom_fraction_mask(saliency, 0.25)


def test_block_variance():
    x = torch.full((2, 3, 224, 224), 0.3)
    v = block_variance(x)
    assert v.shape == (2, 1, 28, 28) and v.dtype == torch.float32
    assert torch.allclose(v, torch.zeros_like(v), atol=1e-7)

    g = torch.Generator().manual_seed(0)
    x[0, :, 8:16, 16:24] = torch.rand(3, 8, 8, generator=g)  # textured block (1, 2)
    v = block_variance(x)
    assert v[0, 0, 1, 2] > 1e-3
    assert v[0, 0, 1, 2] > v[0, 0, 0, 0]
