import io

import numpy as np
import pytest
import torch
from PIL import Image, JpegImagePlugin

from awsa.distortions import gaussian_blur, jpeg, quantize_uint8, resize_roundtrip, roundtrip


def test_quantize_erases_sub_half_step_perturbation():
    x = torch.full((1, 3, 8, 8), 100 / 255)
    assert torch.equal(quantize_uint8(x + 0.4 / 255), quantize_uint8(x))


def test_processing_keeps_shape_and_range(images):
    for out in (jpeg(images, 75), resize_roundtrip(images, 0.5), gaussian_blur(images, 1.0),
                roundtrip(images, 90)):
        assert out.shape == images.shape
        assert out.min() >= 0 and out.max() <= 1


def test_high_quality_jpeg_is_close():
    x = torch.full((1, 3, 224, 224), 0.5)
    assert (jpeg(x, 95) - x).abs().max() < 4 / 255


def _on_uint8_grid(x):
    return torch.equal(x, quantize_uint8(x))


def test_roundtrip_matches_definition(images):
    assert torch.equal(roundtrip(images), quantize_uint8(images))
    assert torch.equal(roundtrip(images, 75), jpeg(quantize_uint8(images), 75))


def test_quantize_keeps_perturbation_above_half_step():
    x = torch.full((1, 3, 8, 8), 100 / 255)
    assert not torch.equal(quantize_uint8(x + 0.6 / 255), quantize_uint8(x))


def test_jpeg_preserves_device_dtype_and_accepts_requires_grad(images):
    x = images.clone().requires_grad_(True)
    out = jpeg(x, 75)
    assert out.device == images.device and out.dtype == images.dtype
    assert _on_uint8_grid(out)


def test_jpeg_uses_420_subsampling(images):
    out = jpeg(images[:1], 75)
    arr = (quantize_uint8(images[0]) * 255).round().byte().permute(1, 2, 0).numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=75, subsampling=2)
    buf.seek(0)
    ref = Image.open(buf)
    assert JpegImagePlugin.get_sampling(ref) == 2  # 4:2:0
    expected = torch.from_numpy(np.asarray(ref.convert("RGB"), dtype=np.float32) / 255)
    assert torch.equal(out[0], expected.permute(2, 0, 1))


@pytest.mark.parametrize("scale", [0.5, 0.75])
def test_resize_roundtrip_size_and_grid(images, scale):
    out = resize_roundtrip(images, scale)
    assert out.shape == (2, 3, 224, 224)
    assert _on_uint8_grid(out)


def test_gaussian_blur_grid_and_validation(images):
    assert _on_uint8_grid(gaussian_blur(images, 1.0))
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            gaussian_blur(images, bad)
