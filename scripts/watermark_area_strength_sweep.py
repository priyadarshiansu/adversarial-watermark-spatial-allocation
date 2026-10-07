"""Module 4: watermark area x strength calibration on low-saliency masks.

    uv run python scripts/watermark_area_strength_sweep.py --config configs/watermark_calibration.yaml

For every selected image and every (area, strength) candidate, embeds a 32-bit payload in the
lowest-saliency Grad-CAM blocks and decodes it after the shared processing: uint8 roundtrip
("clean"), JPEG at each quality, resize down/up and Gaussian blur. The payload for an image is
the same for every candidate, so candidates are compared pairwise. Writes per-image rows to
<out_dir>/per_image.csv and one summary row per candidate to <summary_csv>.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from awsa.data import load_images, load_selected_metadata
from awsa.distortions import gaussian_blur, resize_roundtrip, roundtrip
from awsa.masks import bottom_fraction_mask
from awsa.metrics import bit_error_rate, psnr
from awsa.models import load_resnet50
from awsa.saliency import degenerate_saliency, gradcam
from awsa.utils import get_device, set_seed
from awsa.watermark import PAYLOAD_BITS, embed, extract

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "watermark_calibration.yaml"


def processings(cfg: dict) -> dict:
    """name -> function applied to the watermarked float image before decoding."""
    procs = {"clean": lambda x: roundtrip(x)}
    for q in cfg["jpeg_qualities"]:
        procs[f"jpeg{q}"] = lambda x, q=q: roundtrip(x, q)
    procs[f"resize{cfg['resize_scale']}"] = lambda x: resize_roundtrip(x, cfg["resize_scale"])
    procs[f"blur{cfg['blur_sigma']}"] = lambda x: gaussian_blur(x, cfg["blur_sigma"])
    return procs


def payload(cfg: dict, i: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator().manual_seed(cfg["payload_seed"] + i)
    bits = torch.randint(0, 2, (1, PAYLOAD_BITS), generator=generator, dtype=torch.int64)
    return bits.to(device)


def run_sweep(cfg: dict, n_images: int, device: torch.device) -> pd.DataFrame:
    """One row per (image, candidate) with blocks, PSNR and BER after every processing."""
    model = load_resnet50(device)
    meta = load_selected_metadata().head(n_images)
    procs = processings(cfg)
    key = cfg["key"]
    print(f"device={device}  images={len(meta)}  candidates={len(cfg['candidates'])}")

    rows = []
    dropped = []
    for i, row in meta.iterrows():
        x = load_images([row.ImageId]).to(device)
        y = torch.tensor([int(row.label)], device=device)
        saliency = gradcam(model, x, y)
        if degenerate_saliency(saliency).item():
            dropped.append(row.ImageId)  # all-zero CAM: the low-saliency mask would be arbitrary
            continue
        bits = payload(cfg, i, device)  # same payload for every candidate: paired comparison

        for cand in cfg["candidates"]:
            area, strength = cand["area"], cand["strength"]
            mask = bottom_fraction_mask(saliency, area=area)
            x_wm = embed(x, bits, mask, strength=strength, key=key)
            record = {
                "ImageId": row.ImageId, "area": area, "strength": strength,
                "blocks": int(mask.sum()),
                "psnr": psnr(roundtrip(x_wm), x).item(),
            }
            for name, proc in procs.items():
                recovered = extract(proc(x_wm), mask, key=key)
                record[f"ber_{name}"] = bit_error_rate(bits, recovered).item()
            rows.append(record)
        print(f"  {i + 1}/{len(meta)} images")
    if dropped:
        print(f"dropped {len(dropped)} image(s) with a degenerate Grad-CAM: {dropped}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate: mean blocks, n, mean/min PSNR, mean BER per processing."""
    ber_cols = [c for c in df.columns if c.startswith("ber_")]
    grouped = df.groupby(["area", "strength"], sort=False)
    summary = grouped[["blocks"]].mean()
    summary["n_images"] = grouped.size()
    summary["psnr_mean"] = grouped["psnr"].mean()
    summary["psnr_min"] = grouped["psnr"].min()
    return summary.join(grouped[ber_cols].mean().add_suffix("_mean"))


def print_summary(summary: pd.DataFrame) -> None:
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print("\nMean BER per processing, PSNR of the uint8 watermarked image:")
        print(summary.round(4))


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--n-images", type=int, default=None, help="override n_images")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    n_images = args.n_images if args.n_images is not None else cfg["n_images"]

    set_seed(cfg["seed"])
    df = run_sweep(cfg, n_images, get_device(args.device))

    out_dir = REPO / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_image.csv", index=False)

    summary = summarize(df)
    summary_path = REPO / cfg["summary_csv"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, float_format="%.4f")
    print_summary(summary)
    print(f"\nwrote {out_dir / 'per_image.csv'} and {summary_path}")


if __name__ == "__main__":
    main()
