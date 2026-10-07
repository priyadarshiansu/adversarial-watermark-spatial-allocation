import torch

from awsa.data import load_images, load_metadata
from awsa.distortions import jpeg, quantize_uint8
from awsa.masks import bottom_fraction_mask
from awsa.metrics import bit_error_rate
from awsa.models import load_resnet50
from awsa.saliency import gradcam
from awsa.utils import get_device
from awsa.watermark import embed, extract

DEVICE = get_device()
STRENGTH = 0.1
MASK_AREA = 0.50
KEY = 123


def main():
    meta = load_metadata()

    image_id = meta.iloc[0].ImageId
    label = int(meta.iloc[0].label)

    x = load_images([image_id]).to(DEVICE)
    y = torch.tensor([label], device=DEVICE)

    print("device:", DEVICE)
    print("image:", image_id)

    # Module 3: create a low-saliency watermark region.
    model = load_resnet50(DEVICE)
    saliency = gradcam(model, x, y)

    # Keep the mask on CPU for the current watermark implementation.
    mask = bottom_fraction_mask(
        saliency.cpu(),
        area=MASK_AREA,
    )

    # Fixed 32-bit payload for reproducibility.
    generator = torch.Generator().manual_seed(42)
    bits = torch.randint(
        0,
        2,
        (1, 32),
        generator=generator,
        dtype=torch.int64,
    ).to(DEVICE)

    watermarked = embed(
        x,
        bits,
        mask,
        strength=STRENGTH,
        key=KEY,
    )

    print("\nOriginal bits:")
    print(bits[0].tolist())

    # No processing
    recovered = extract(watermarked, mask, key=KEY)
    ber = bit_error_rate(bits, recovered)

    print(f"\nNo processing BER: {ber.item():.4f}")

    # Normal uint8 image round-trip
    processed = quantize_uint8(watermarked)
    recovered = extract(processed, mask, key=KEY)

    print(
        f"Uint8 BER: "
        f"{bit_error_rate(bits, recovered).item():.4f}"
    )

    # Real JPEG
    for quality in [95, 90, 80, 75, 60]:
        processed = jpeg(
            quantize_uint8(watermarked),
            quality,
        )

        recovered = extract(
            processed,
            mask,
            key=KEY,
        )

        ber = bit_error_rate(bits, recovered)

        print(
            f"JPEG q{quality} BER: "
            f"{ber.item():.4f}"
        )

    max_delta = (watermarked - x).abs().max().item()

    print(
        f"\nMaximum pixel change: "
        f"{max_delta * 255:.2f}/255"
    )


if __name__ == "__main__":
    main()