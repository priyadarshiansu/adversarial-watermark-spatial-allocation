"""Module 4: single-image watermark pilot (sanity check before the calibration sweep).

    uv run python scripts/watermark_pilot.py --area 0.5 --strength 0.08

Embeds one payload in the lowest-saliency blocks of the first selected image and prints the
BER after the uint8 roundtrip and several JPEG qualities, plus PSNR and max pixel change.
"""

import argparse

import torch
from watermark_area_strength_sweep import DEFAULT_CONFIG, load_config, payload

from awsa.data import load_images, load_selected_metadata
from awsa.distortions import roundtrip
from awsa.masks import bottom_fraction_mask
from awsa.metrics import bit_error_rate, psnr
from awsa.models import load_resnet50
from awsa.saliency import gradcam
from awsa.utils import get_device, set_seed
from awsa.watermark import embed, extract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--area", type=float, default=0.50)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--jpeg", type=int, nargs="+", default=[95, 90, 80, 75, 60])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"])
    device = get_device(args.device)
    row = load_selected_metadata().iloc[0]
    x = load_images([row.ImageId]).to(device)
    y = torch.tensor([int(row.label)], device=device)
    print(f"device={device}  image={row.ImageId}")

    saliency = gradcam(load_resnet50(device), x, y)
    mask = bottom_fraction_mask(saliency, area=args.area)
    bits = payload(cfg, 0, device)
    x_wm = embed(x, bits, mask, strength=args.strength, key=cfg["key"])

    print("payload:", bits[0].tolist())
    for q in [None, *args.jpeg]:
        recovered = extract(roundtrip(x_wm, q), mask, key=cfg["key"])
        label = "uint8" if q is None else f"JPEG q{q}"
        print(f"{label:>9} BER: {bit_error_rate(bits, recovered).item():.4f}")

    x_saved = roundtrip(x_wm)
    print(f"\nPSNR (uint8): {psnr(x_saved, x).item():.2f} dB")
    print(f"max pixel change: {(x_saved - x).abs().max().item() * 255:.0f}/255")


if __name__ == "__main__":
    main()
