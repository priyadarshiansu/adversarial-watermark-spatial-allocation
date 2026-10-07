import pytest
import torch
import torch.nn.functional as F

from awsa.attacks import DifferentiableJPEG, fgsm, pgd
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
    from awsa.distortions import roundtrip

    g = torch.Generator().manual_seed(0)
    smooth = torch.nn.functional.interpolate(torch.rand(2, 3, 28, 28, generator=g), size=224,
                                             mode="bilinear").clamp(0, 1)
    for q in (90, 75, 60):
        real = roundtrip(smooth, jpeg_quality=q)
        approx = DifferentiableJPEG(q)(smooth)
        assert approx.shape == smooth.shape
        assert (approx - real).abs().mean() < 0.45 * (real - smooth).abs().mean()


def test_differentiable_jpeg_passes_gradients(images):
    x = images.clone().requires_grad_(True)
    DifferentiableJPEG((50, 90), generator=torch.Generator().manual_seed(0))(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_jpeg_aware_pgd_keeps_guarantees(tiny_model, images):
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


def _top_half_mask(b: int) -> torch.Tensor:
    block_mask = torch.zeros(b, 1, 28, 28, dtype=torch.bool)
    block_mask[:, :, :14] = True
    return blocks_to_pixels(block_mask)


def _assert_guarantees(x_adv, x, eps, mask=None):
    diff = (x_adv - x).abs()
    assert diff.max() <= eps + 1e-6
    assert x_adv.min() >= 0 and x_adv.max() <= 1
    if mask is not None:
        assert diff[(mask == 0).expand_as(diff)].max() == 0


def test_pgd_increases_loss(tiny_model, images):
    y = torch.tensor([1, 2])
    eps = 8 / 255
    x_fgsm = fgsm(tiny_model, images, y, eps=eps)
    x_adv = pgd(tiny_model, images, y, eps=eps, alpha=2 / 255, steps=10)
    with torch.no_grad():
        clean = F.cross_entropy(tiny_model(images), y)
        fgsm_loss = F.cross_entropy(tiny_model(x_fgsm), y)
        adv = F.cross_entropy(tiny_model(x_adv), y)
    assert adv > clean
    assert adv >= fgsm_loss - 1e-4


def test_differentiable_jpeg_quality_range():
    jpeg = DifferentiableJPEG((50, 95), generator=torch.Generator().manual_seed(0))
    q = jpeg._qualities(torch.zeros(500, 3, 16, 16))
    assert q.min() >= 50 and q.max() <= 95
    assert q.min() < 60 and q.max() > 85  # actually spans the range


def test_eot_pgd_seeded_is_reproducible(tiny_model, images):
    y = torch.tensor([1, 2])

    def run():
        g = torch.Generator().manual_seed(123)
        return pgd(tiny_model, images, y, eps=EPS, alpha=0.5 / 255, steps=2, random_start=True,
                   transform=DifferentiableJPEG((50, 95), generator=g), eot_samples=2,
                   generator=g)

    assert torch.equal(run(), run())


def test_random_start_reproducible_and_respects_mask(tiny_model, images):
    y = torch.tensor([1, 2])
    mask = _top_half_mask(2)

    def run(seed):
        return pgd(tiny_model, images, y, eps=EPS, alpha=0.5 / 255, steps=1, mask=mask,
                   random_start=True, generator=torch.Generator().manual_seed(seed))

    a, b, c = run(0), run(0), run(1)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    _assert_guarantees(a, images, EPS, mask)


def test_random_start_respects_mask_and_bounds_before_steps(images):
    """With a zero-gradient model only the random start moves x: check it alone."""

    class Zero(torch.nn.Module):
        def forward(self, x):
            return (x * 0).flatten(1)[:, :10]

    y = torch.tensor([1, 2])
    mask = _top_half_mask(2)
    x_adv = pgd(Zero(), images, y, eps=EPS, alpha=0.0, steps=1, mask=mask, random_start=True,
                generator=torch.Generator().manual_seed(0))
    _assert_guarantees(x_adv, images, EPS, mask)
    assert (x_adv - images).abs()[..., :112, :].max() > 0


def test_pgd_batch_size_one(tiny_model, images):
    x = images[:1]
    y = torch.tensor([3])
    x_adv = pgd(tiny_model, x, y, eps=EPS, alpha=0.5 / 255, steps=3, random_start=True,
                transform=DifferentiableJPEG((50, 95), generator=torch.Generator().manual_seed(0)),
                eot_samples=2, generator=torch.Generator().manual_seed(0))
    assert x_adv.shape == x.shape
    _assert_guarantees(x_adv, x, EPS)


def test_straight_through_rounding_is_identity():
    x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(0), requires_grad=True)
    rounded = x + (torch.round(x.clamp(0, 1) * 255) / 255 - x).detach()
    g = torch.randn(x.shape, generator=torch.Generator().manual_seed(1))
    (rounded * g).sum().backward()
    assert torch.equal(x.grad, g)
    assert torch.equal(rounded.detach(), torch.round(x.detach() * 255) / 255)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_guarantees_on_constant_images(tiny_model, value):
    x = torch.full((2, 3, 224, 224), value)
    y = torch.tensor([1, 2])
    mask = _top_half_mask(2)
    for kwargs in ({}, {"mask": mask}, {"transform": DifferentiableJPEG(75)}):
        x_adv = pgd(tiny_model, x, y, eps=EPS, alpha=0.5 / 255, steps=3, random_start=True,
                    generator=torch.Generator().manual_seed(0), **kwargs)
        _assert_guarantees(x_adv, x, EPS, kwargs.get("mask"))


@pytest.mark.parametrize("kwargs", [{"eot_samples": 0}, {"steps": 0}])
def test_pgd_rejects_bad_counts(tiny_model, images, kwargs):
    y = torch.tensor([1, 2])
    args = {"eps": EPS, "alpha": 0.5 / 255, "steps": 1} | kwargs
    with pytest.raises(ValueError):
        pgd(tiny_model, images, y, **args)


def test_per_sample_eot_matches_summed_loss(tiny_model, images):
    """Accumulating per-sample gradients equals the gradient of the averaged loss."""
    y = torch.tensor([1, 2])
    alpha, n = 0.5 / 255, 2

    x_adv = pgd(tiny_model, images, y, eps=EPS, alpha=alpha, steps=1,
                transform=DifferentiableJPEG((50, 95), generator=torch.Generator().manual_seed(7)),
                eot_samples=n)

    # Old formulation: sum all sample losses into one graph, one autograd.grad.
    transform = DifferentiableJPEG((50, 95), generator=torch.Generator().manual_seed(7))
    x_req = images.clone().requires_grad_(True)
    loss = sum(F.cross_entropy(tiny_model(transform(x_req)), y) for _ in range(n))
    (grad,) = torch.autograd.grad(loss / n, x_req)
    expected = (images + (alpha * grad.sign()).clamp(-EPS, EPS)).clamp(0, 1)

    torch.testing.assert_close(x_adv, expected)
