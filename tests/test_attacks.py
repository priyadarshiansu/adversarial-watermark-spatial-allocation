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


def test_differentiable_jpeg_tracks_real_jpeg():
    """The kornia approximation must be much closer to PIL JPEG than JPEG is to the input."""
    from awsa.attacks import DifferentiableJPEG
    from awsa.distortions import roundtrip

    g = torch.Generator().manual_seed(0)
    smooth = torch.nn.functional.interpolate(torch.rand(2, 3, 28, 28, generator=g), size=224,
                                             mode="bilinear").clamp(0, 1)
    for q in (90, 75, 60):
        real = roundtrip(smooth, jpeg_quality=q)
        approx = DifferentiableJPEG(q)(smooth)
        assert approx.shape == smooth.shape
        assert (approx - real).abs().mean() < 0.6 * (real - smooth).abs().mean()


def test_differentiable_jpeg_passes_gradients(images):
    from awsa.attacks import DifferentiableJPEG

    x = images.clone().requires_grad_(True)
    DifferentiableJPEG((50, 90), generator=torch.Generator().manual_seed(0))(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_jpeg_aware_pgd_keeps_guarantees(tiny_model, images):
    from awsa.attacks import DifferentiableJPEG

    y = torch.tensor([1, 2])
    block_mask = torch.zeros(2, 1, 28, 28, dtype=torch.bool)
    block_mask[:, :, :14] = True
    mask = blocks_to_pixels(block_mask)
    x_adv = pgd(tiny_model, images, y, eps=EPS, alpha=0.5 / 255, steps=3, mask=mask,
                transform=DifferentiableJPEG(75), eot_samples=2)
    diff = (x_adv - images).abs()
    assert diff.max() <= EPS + 1e-6
    assert diff[..., 112:, :].max() == 0
    assert x_adv.min() >= 0 and x_adv.max() <= 1
