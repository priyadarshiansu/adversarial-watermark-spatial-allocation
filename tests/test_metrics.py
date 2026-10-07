import math

import torch

from awsa import metrics
from awsa.metrics import attack_success, bit_error_rate, psnr, ssim


def test_psnr_known_value(images):
    noisy = (images + 0.01).clamp(0, 1)
    mse = ((noisy - images) ** 2).flatten(1).mean(1)
    assert torch.allclose(psnr(noisy, images), 10 * torch.log10(1 / mse))
    assert psnr(noisy, images).shape == (2,)


def test_psnr_identical_is_inf(images):
    assert math.isinf(psnr(images, images)[0].item())


def test_ssim_range_and_order(images):
    g = torch.Generator().manual_seed(1)
    small = (images + 0.01 * torch.randn(images.shape, generator=g)).clamp(0, 1)
    large = (images + 0.10 * torch.randn(images.shape, generator=g)).clamp(0, 1)
    s_id, s_small, s_large = ssim(images, images), ssim(small, images), ssim(large, images)
    assert s_id.shape == (2,)
    assert torch.allclose(s_id, torch.ones(2))
    assert (s_small > s_large).all() and (s_large > -1).all()


def test_lpips_rescales_to_minus_one_one(images, monkeypatch):
    """Checks the wrapper without downloading AlexNet: fake net records its inputs."""
    seen = {}

    def fake_net(a, b, normalize):
        seen.update(a=a, b=b, normalize=normalize)
        return torch.zeros(a.shape[0], 1, 1, 1)

    monkeypatch.setitem(metrics._LPIPS_NETS, str(images.device), fake_net)
    out = metrics.lpips_distance(images, images)
    assert out.shape == (2,)
    assert seen["normalize"] is True  # lpips maps [0, 1] -> [-1, 1] when normalize=True


def test_attack_success_and_ber():
    logits = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    assert torch.equal(attack_success(logits, torch.tensor([0, 0])), torch.tensor([0.0, 1.0]))
    bits = torch.zeros(2, 32, dtype=torch.int64)
    flipped = bits.clone()
    flipped[0, :8] = 1
    assert torch.equal(bit_error_rate(bits, flipped), torch.tensor([0.25, 0.0]))


def test_confidence_drop():
    from awsa.metrics import confidence_drop

    clean = torch.tensor([[10.0, 0.0], [0.0, 0.0]])
    adv = torch.tensor([[0.0, 10.0], [0.0, 0.0]])
    drop = confidence_drop(clean, adv, torch.tensor([0, 0]))
    assert drop[0] > 0.99 and drop[1] == 0
