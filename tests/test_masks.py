import torch

from awsa.masks import (
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
