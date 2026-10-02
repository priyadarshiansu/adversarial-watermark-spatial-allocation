import torch

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
