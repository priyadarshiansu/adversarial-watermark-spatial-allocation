import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import (
    InterpolationMode,
    pil_to_tensor,
    resize,
)

from awsa.data import DATA_DIR, load_metadata
from awsa.distortions import jpeg, quantize_uint8
from awsa.masks import bottom_fraction_mask
from awsa.metrics import bit_error_rate
from awsa.models import load_resnet50
from awsa.saliency import gradcam
from awsa.watermark import embed, extract


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STRENGTHS = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
JPEG_QUALITIES = [90, 75, 60]

MASK_AREA = 0.50
KEY = 123
N_IMAGES = 10


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = pil_to_tensor(img).float() / 255.0

    x = resize(
        x,
        [224, 224],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    return x.unsqueeze(0)


def main():
    meta = load_metadata().head(N_IMAGES)
    model = load_resnet50(DEVICE)

    results = {
        strength: {
            "clean_ber": [],
            "q90": [],
            "q75": [],
            "q60": [],
            "max_delta": [],
            "psnr": [],
        }
        for strength in STRENGTHS
    }

    generator = torch.Generator().manual_seed(42)

    for i, row in meta.iterrows():
        image_id = row.ImageId
        label = int(row.label)

        print(f"Image {i + 1}/{N_IMAGES}: {image_id}")

        path = DATA_DIR / "raw" / f"{image_id}.png"

        x = load_image(path).to(DEVICE)
        y = torch.tensor([label], device=DEVICE)

        saliency = gradcam(model, x, y)

        mask = bottom_fraction_mask(
            saliency.cpu(),
            area=MASK_AREA,
        )

        bits = torch.randint(
            0,
            2,
            (1, 32),
            generator=generator,
            dtype=torch.int64,
        ).to(DEVICE)

        for strength in STRENGTHS:
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

            results[strength]["clean_ber"].append(
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

                results[strength][f"q{quality}"].append(
                    bit_error_rate(bits, recovered).item()
                )

            delta = watermarked - x

            results[strength]["max_delta"].append(
                delta.abs().max().item() * 255
            )

            mse = (delta ** 2).mean().item()

            psnr = (
                float("inf")
                if mse == 0
                else 10 * math.log10(1.0 / mse)
            )

            results[strength]["psnr"].append(psnr)

    print("\nRESULTS")
    print(
        "strength | clean BER | q90 BER | q75 BER | "
        "q60 BER | max Δ/255 | PSNR"
    )
    print("-" * 79)

    for strength in STRENGTHS:
        r = results[strength]

        mean = lambda values: sum(values) / len(values)

        print(
            f"{strength:8.2f} | "
            f"{mean(r['clean_ber']):9.4f} | "
            f"{mean(r['q90']):7.4f} | "
            f"{mean(r['q75']):7.4f} | "
            f"{mean(r['q60']):7.4f} | "
            f"{mean(r['max_delta']):9.2f} | "
            f"{mean(r['psnr']):5.2f}"
        )


if __name__ == "__main__":
    main()