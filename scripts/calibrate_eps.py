"""Module 2: epsilon calibration for plain and JPEG-aware PGD.

    uv run python scripts/calibrate_eps.py --config configs/eps_calibration.yaml

For every attack x region x eps, attacks the selected images (scripts/prepare_data.py) and
validates through the real pipeline, distortions.roundtrip (uint8, then PIL JPEG), at each
evaluation quality. Writes one CSV row per (attack, region, eps, eval quality) and prints the
attack-success table used to pick 3-4 eps values on the steep part of the curve.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from awsa.attacks import DifferentiableJPEG, pgd
from awsa.data import load_images, load_selected_metadata
from awsa.distortions import roundtrip
from awsa.masks import blocks_to_pixels, top_fraction_mask
from awsa.metrics import attack_success, confidence_drop, lpips_distance, psnr, ssim
from awsa.models import load_resnet50
from awsa.saliency import gradcam
from awsa.utils import get_device, set_seed

REPO = Path(__file__).resolve().parents[1]


def make_transform(spec: dict, generator: torch.Generator):
    q = spec.get("jpeg_quality")
    if q is None:
        return None
    return DifferentiableJPEG(tuple(q) if isinstance(q, list) else int(q), generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs" / "eps_calibration.yaml"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    set_seed(cfg["seed"])
    device = get_device(args.device)
    model = load_resnet50(device)
    meta = load_selected_metadata().head(cfg["n_images"])
    print(f"device={device}  images={len(meta)}")

    rows = []
    bs = cfg["batch_size"]
    for start in range(0, len(meta), bs):
        chunk = meta.iloc[start:start + bs]
        x = load_images(chunk.ImageId).to(device)
        y = torch.tensor(chunk.label.values, device=device)
        with torch.no_grad():
            logits_clean = model(x)
        masks = {"global": None}
        if "salient" in cfg["regions"]:
            blocks = top_fraction_mask(gradcam(model, x, y), cfg["mask_area"])
            masks["salient"] = blocks_to_pixels(blocks).to(device)

        for spec in cfg["attacks"]:
            generator = torch.Generator().manual_seed(cfg["seed"] + start)
            transform = make_transform(spec, generator)
            for region in cfg["regions"]:
                for eps_255 in cfg["eps_255"]:
                    eps = eps_255 / 255
                    x_adv = pgd(model, x, y, eps=eps, alpha=cfg["alpha_ratio"] * eps,
                                steps=cfg["steps"], mask=masks[region], transform=transform,
                                eot_samples=spec.get("eot_samples", 1))
                    x_saved = roundtrip(x_adv)  # what the image owner publishes (uint8)
                    fidelity = {"psnr": psnr(x_saved, x), "ssim": ssim(x_saved, x),
                                "lpips": lpips_distance(x_saved, x)}
                    for q in cfg["eval_jpeg_qualities"]:
                        with torch.no_grad():
                            logits = model(roundtrip(x_adv, jpeg_quality=q))
                        succ = attack_success(logits, y)
                        drop = confidence_drop(logits_clean, logits, y)
                        for i, image_id in enumerate(chunk.ImageId):
                            rows.append({
                                "ImageId": image_id, "attack": spec["name"], "region": region,
                                "eps_255": eps_255, "eval_jpeg": q if q is not None else "none",
                                "success": succ[i].item(), "confidence_drop": drop[i].item(),
                                **{k: v[i].item() for k, v in fidelity.items()},
                            })
        print(f"  {min(start + bs, len(meta))}/{len(meta)} images")

    out_dir = REPO / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_image.csv", index=False)

    summary = (df.groupby(["attack", "region", "eps_255", "eval_jpeg"], sort=False)
                 [["success", "confidence_drop", "psnr", "ssim", "lpips"]].mean())
    summary.to_csv(out_dir / "summary.csv")
    table = summary["success"].unstack("eval_jpeg").map("{:.1%}".format)
    with pd.option_context("display.width", 120):
        print("\nAttack success after the real roundtrip (columns = eval JPEG quality):")
        print(table)
    print(f"\nwrote {out_dir / 'per_image.csv'} and summary.csv")


if __name__ == "__main__":
    main()
