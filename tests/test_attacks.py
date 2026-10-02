import torch

from awsa.attacks import fgsm, pgd
from awsa.masks import blocks_to_pixels

EPS = 2 / 255


def test_pgd_respects_eps_and_range(tiny_model, images):
    y = torch.tensor([1, 2])
    x_adv = pgd(tiny_model, images, y, eps=EPS, alpha=0.5 / 255, steps=5, random_start=True)
    assert (x_adv - images).abs().max() <= EPS + 1e-6
    assert x_adv.min() >= 0 and x_adv.max() <= 1


def test_pgd_mask_leaves_outside_untouched(tiny_model, images):
    y = torch.tensor([1, 2])
    block_mask = torch.zeros(2, 1, 28, 28, dtype=torch.bool)
    block_mask[:, :, :14] = True  # top half only
    mask = blocks_to_pixels(block_mask)
    x_adv = pgd(tiny_model, images, y, eps=EPS, alpha=0.5 / 255, steps=5, mask=mask,
                random_start=True)
    diff = (x_adv - images).abs()
    assert diff[..., 112:, :].max() == 0
    assert diff[..., :112, :].max() > 0


def test_fgsm_respects_eps(tiny_model, images):
    y = torch.tensor([1, 2])
    x_adv = fgsm(tiny_model, images, y, eps=EPS)
    assert (x_adv - images).abs().max() <= EPS + 1e-6
