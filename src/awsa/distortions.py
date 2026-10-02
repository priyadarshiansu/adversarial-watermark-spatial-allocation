"""Post-processing used in RQ2, plus the shared uint8 / JPEG round trip.

Every evaluation (attack success AND watermark decoding) must go through the
same functions here, so all arms see identical processing.
"""

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur as _tv_gaussian_blur


def quantize_uint8(x: torch.Tensor) -> torch.Tensor:
    """Round to the 8-bit grid (what saving a PNG does). Perturbations < 0.5/255 vanish here."""
    return torch.round(x.clamp(0, 1) * 255) / 255


def _to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def _from_pil(img: Image.Image, like: torch.Tensor) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(like.device, like.dtype)


def jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Real (non-differentiable) JPEG round trip with PIL, 4:2:0 chroma subsampling."""
    out = []
    for img in x:
        buf = io.BytesIO()
        _to_pil(img).save(buf, format="JPEG", quality=quality, subsampling=2)
        buf.seek(0)
        out.append(_from_pil(Image.open(buf), x))
    return torch.stack(out)


def resize_roundtrip(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Downscale by `scale`, then upscale back to the original size (keeps the 8x8 grid valid)."""
    h, w = x.shape[-2:]
    small = F.interpolate(
        x, size=(round(h * scale), round(w * scale)), mode="bilinear", antialias=True,
        align_corners=False,
    )
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = 2 * int(np.ceil(3 * sigma)) + 1
    return _tv_gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


def roundtrip(x: torch.Tensor, jpeg_quality: int | None = None) -> torch.Tensor:
    """The canonical path every result goes through: uint8 quantization, then optional JPEG."""
    x = quantize_uint8(x)
    return jpeg(x, jpeg_quality) if jpeg_quality is not None else x
