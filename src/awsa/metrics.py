"""Evaluation metrics (Module 1). All inputs are [0, 1] tensors (B, 3, H, W); outputs are (B,)."""

import torch


def attack_success(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1.0 where the prediction differs from the true label (untargeted success)."""
    return (logits.argmax(dim=1) != y).float()


def confidence_drop(logits_clean: torch.Tensor, logits_adv: torch.Tensor,
                    y: torch.Tensor) -> torch.Tensor:
    """Drop in softmax probability of the true class, clean minus adversarial."""
    p_clean = logits_clean.softmax(1).gather(1, y[:, None]).squeeze(1)
    p_adv = logits_adv.softmax(1).gather(1, y[:, None]).squeeze(1)
    return p_clean - p_adv


def bit_error_rate(bits_true: torch.Tensor, bits_pred: torch.Tensor) -> torch.Tensor:
    """Fraction of payload bits decoded incorrectly, per image."""
    return (bits_true != bits_pred).float().mean(dim=1)


def psnr(x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB with peak 1.0. inf where the images are identical."""
    mse = ((x.float() - x_ref.float()) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse)


def ssim(x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Mean SSIM (Wang et al., 2004) over RGB, via scikit-image (7x7 window, data_range=1)."""
    from skimage.metrics import structural_similarity

    a = x.detach().float().cpu().numpy()
    b = x_ref.detach().float().cpu().numpy()
    vals = [structural_similarity(ai, bi, channel_axis=0, data_range=1.0) for ai, bi in zip(a, b)]
    return torch.tensor(vals, dtype=torch.float32, device=x.device)


_LPIPS_NETS: dict = {}


def _lpips_net(device: torch.device) -> torch.nn.Module:
    """LPIPS-AlexNet (Zhang et al., 2018), built once per device. Downloads AlexNet on first use."""
    key = str(device)
    if key not in _LPIPS_NETS:
        import lpips

        _LPIPS_NETS[key] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return _LPIPS_NETS[key]


@torch.no_grad()
def lpips_distance(x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """LPIPS-AlexNet distance (lower = more similar). Inputs in [0, 1]; rescaled to [-1, 1]."""
    net = _lpips_net(x.device)
    return net(x.float(), x_ref.float(), normalize=True).flatten()
