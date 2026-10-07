import math

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

CANDIDATES = [
    (0.25, 0.08),
    (0.35, 0.06),
    (0.35, 0.08),
    (0.50, 0.06),
]

JPEG_QUALITIES = [90, 75, 60]

N_IMAGES = 100
KEY = 123


def main():
    meta = load_metadata().head(N_IMAGES)
    model = load_resnet50(DEVICE)

    results = {}

    for area, strength in CANDIDATES:
        results[(area, strength)] = {
            "clean": [],
            "q90": [],
            "q75": [],
            "q60": [],
            "psnr": [],
        }

    for i, row in meta.iterrows():
        image_id = row.ImageId
        label = int(row.label)

        print(f"Image {i + 1}/{N_IMAGES}: {image_id}")

        x = load_images([image_id]).to(DEVICE)

        y = torch.tensor([label], device=DEVICE)

        saliency = gradcam(model, x, y)

        # Same payload for every candidate on this image,
        # so the comparison is paired.
        generator = torch.Generator().manual_seed(42 + i)

        bits = torch.randint(
            0,
            2,
            (1, 32),
            generator=generator,
            dtype=torch.int64,
        ).to(DEVICE)

        for area, strength in CANDIDATES:
            mask = bottom_fraction_mask(
                saliency.cpu(),
                area=area,
            )

            watermarked = embed(
                x,
                bits,
                mask,
                strength=strength,
                key=KEY,
            )

            recovered = extract(
                watermarked,
                mask,
                key=KEY,
            )

            results[(area, strength)]["clean"].append(
                bit_error_rate(bits, recovered).item()
            )

            for quality in JPEG_QUALITIES:
                processed = jpeg(
                    quantize_uint8(watermarked),
                    quality,
                )

                recovered = extract(
                    processed,
                    mask,
                    key=KEY,
                )

                results[(area, strength)][f"q{quality}"].append(
                    bit_error_rate(bits, recovered).item()
                )

            mse = ((watermarked - x) ** 2).mean().item()

            psnr = (
                float("inf")
                if mse == 0
                else 10 * math.log10(1.0 / mse)
            )

            results[(area, strength)]["psnr"].append(psnr)

    print("\nRESULTS")
    print(
        "area | strength | blocks | clean BER | "
        "q90 BER | q75 BER | q60 BER | PSNR"
    )
    print("-" * 82)

    for area, strength in CANDIDATES:
        blocks = round(area * 784)

        r = results[(area, strength)]

        mean = lambda values: sum(values) / len(values)

        print(
            f"{area:4.2f} | "
            f"{strength:8.2f} | "
            f"{blocks:6d} | "
            f"{mean(r['clean']):9.4f} | "
            f"{mean(r['q90']):7.4f} | "
            f"{mean(r['q75']):7.4f} | "
            f"{mean(r['q60']):7.4f} | "
            f"{mean(r['psnr']):5.2f}"
        )


if __name__ == "__main__":
    main()