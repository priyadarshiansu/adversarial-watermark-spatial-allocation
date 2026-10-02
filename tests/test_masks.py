import torch

from awsa.masks import blocks_to_pixels


def test_blocks_to_pixels_shape_and_alignment():
    m = torch.zeros(1, 1, 28, 28, dtype=torch.bool)
    m[0, 0, 0, 1] = True
    px = blocks_to_pixels(m)
    assert px.shape == (1, 1, 224, 224)
    assert px.sum() == 64
    assert px[0, 0, 0:8, 8:16].all()
