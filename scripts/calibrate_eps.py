"""Module 2: epsilon calibration for plain and JPEG-aware PGD.

    uv run python scripts/calibrate_eps.py --config configs/eps_calibration.yaml
    uv run python scripts/calibrate_eps.py --quick --device cpu      # smoke run

For every attack x region x mask area x eps, attacks the selected images
(scripts/prepare_data.py) and validates through the real pipeline, distortions.roundtrip
(uint8, then PIL JPEG), at each evaluation quality. Each eval quality has its own clean
baseline, model(roundtrip(x, q)): `clean_correct` says whether the processed clean image is
still classified correctly, and success restricted to those images is the headline number.

Writes one row per (image, attack, region, area, eps, eval quality) to out_dir/per_image.csv
(git-ignored) and one row per cell to summary_csv (small, committable), and prints the
attack-success table used to pick 3-4 eps values on the steep part of the curve.
"""

import argparse
import math
import zlib
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
import yaml

from awsa.attacks import DifferentiableJPEG, pgd
from awsa.data import load_images, load_selected_metadata
from awsa.distortions import roundtrip
from awsa.masks import blocks_to_pixels, bottom_fraction_mask, top_fraction_mask
from awsa.metrics import attack_success, confidence_drop, lpips_distance, psnr, ssim
from awsa.models import load_resnet50
from awsa.saliency import degenerate_saliency, gradcam
from awsa.utils import get_device, set_seed

REPO = Path(__file__).resolve().parents[1]
GLOBAL = "global"
MASK_FNS = {"salient": top_fraction_mask, "low": bottom_fraction_mask}
SUMMARY_KEYS = ["attack", "region", "area", "eps_255", "eval_jpeg"]


def cell_generator(seed: int, batch_start: int, region: str, area: float | None,
                   eps_255: float) -> torch.Generator:
    """A fresh CPU generator per (batch, region, area, eps) cell.

    Deterministic across processes (crc32, not Python's salted hash). The attack name is
    deliberately left out, so every attack on the same cell shares its random start.
    """
    key = f"{seed}|{batch_start}|{region}|{area}|{eps_255}".encode()
    return torch.Generator().manual_seed(zlib.crc32(key))


def make_transform(spec: dict, generator: torch.Generator):
    q = spec.get("jpeg_quality")
    if q is None:
        return None
    return DifferentiableJPEG(tuple(q) if isinstance(q, list) else int(q), generator=generator)


def default_fidelity(x_saved: torch.Tensor, x: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"psnr": psnr(x_saved, x), "ssim": ssim(x_saved, x),
            "lpips": lpips_distance(x_saved, x)}


def build_masks(saliency: torch.Tensor | None, regions: list[str],
                areas: list[float]) -> dict[tuple[str, float | None], torch.Tensor | None]:
    """{(region, area): pixel mask or None}. Global ignores area (key area None)."""
    masks: dict = {}
    for region in regions:
        if region == GLOBAL:
            masks[(GLOBAL, None)] = None
            continue
        if region not in MASK_FNS:
            raise ValueError(f"unknown region {region!r}; expected global, {', '.join(MASK_FNS)}")
        for area in areas:
            masks[(region, area)] = blocks_to_pixels(MASK_FNS[region](saliency, area))
    return masks


def evaluate_batch(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, image_ids,
                   masks: dict, cfg: dict, batch_start: int,
                   fidelity_fn: Callable = default_fidelity) -> list[dict]:
    """Attack one batch for every (attack, region, area, eps) and evaluate at every quality.

    `masks` comes from build_masks. Returns per-image rows. fidelity_fn is measured on
    roundtrip(x_adv) (what the owner publishes) against x.
    """
    qualities = cfg["eval_jpeg_qualities"]
    with torch.no_grad():
        logits_clean = {q: model(roundtrip(x, jpeg_quality=q)) for q in qualities}
    clean_correct = {q: logits_clean[q].argmax(1) == y for q in qualities}

    rows = []
    for (region, area), mask in masks.items():
        mask = None if mask is None else mask.to(x.device)
        for eps_255 in cfg["eps_255"]:
            eps = eps_255 / 255
            for spec in cfg["attacks"]:
                # Same seed for every attack on this cell: paired random start / JPEG draws.
                generator = cell_generator(cfg["seed"], batch_start, region, area, eps_255)
                x_adv = pgd(model, x, y, eps=eps, alpha=cfg["alpha_ratio"] * eps,
                            steps=cfg["steps"], mask=mask,
                            random_start=cfg.get("random_start", False),
                            transform=make_transform(spec, generator),
                            eot_samples=spec.get("eot_samples", 1), generator=generator)
                fidelity = fidelity_fn(roundtrip(x_adv), x)
                for q in qualities:
                    with torch.no_grad():
                        logits = model(roundtrip(x_adv, jpeg_quality=q))
                    succ = attack_success(logits, y)
                    drop = confidence_drop(logits_clean[q], logits, y)
                    for i, image_id in enumerate(image_ids):
                        rows.append({
                            "ImageId": image_id, "attack": spec["name"], "region": region,
                            "area": math.nan if area is None else area, "eps_255": eps_255,
                            "eval_jpeg": q if q is not None else "none",
                            "clean_correct": bool(clean_correct[q][i]),
                            "success": succ[i].item(), "confidence_drop": drop[i].item(),
                            **{k: v[i].item() for k, v in fidelity.items()},
                        })
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (attack, region, area, eps_255, eval_jpeg)."""
    def cell(g: pd.DataFrame) -> pd.Series:
        cc = g[g.clean_correct]
        return pd.Series({
            "n": len(g), "n_clean_correct": len(cc),
            "success_clean_correct": cc.success.mean() if len(cc) else math.nan,
            "success_all": g.success.mean(),
            "confidence_drop": g.confidence_drop.mean(),
            "psnr": g.psnr.mean(), "ssim": g.ssim.mean(), "lpips": g.lpips.mean(),
        })

    grouped = df.groupby(SUMMARY_KEYS, sort=False, dropna=False)
    out = grouped[df.columns.drop(SUMMARY_KEYS)].apply(cell).reset_index()
    return out.astype({"n": int, "n_clean_correct": int})


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(REPO / "configs" / "eps_calibration.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-images", type=int, default=None, help="override n_images")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch_size")
    parser.add_argument("--quick", action="store_true",
                        help="CPU smoke run: 8 images, 2 eps, first attack, first mask area; "
                             "summary goes to out_dir, not the committed summary_csv")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.quick:
        cfg["n_images"] = 8
        cfg["batch_size"] = 4
        cfg["eps_255"] = [cfg["eps_255"][0], cfg["eps_255"][-1]]
        cfg["attacks"] = cfg["attacks"][:1]
        cfg["mask_areas"] = cfg["mask_areas"][:1]
        cfg["summary_csv"] = str(Path(cfg["out_dir"]) / "summary_quick.csv")
    if args.n_images is not None:
        cfg["n_images"] = args.n_images
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    return cfg


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args)

    set_seed(cfg["seed"])
    device = get_device(args.device)
    model = load_resnet50(device)
    meta = load_selected_metadata().head(cfg["n_images"])
    print(f"device={device}  images={len(meta)}")

    masked = [r for r in cfg["regions"] if r != GLOBAL]
    rows = []
    dropped = []
    bs = cfg["batch_size"]
    for start in range(0, len(meta), bs):
        chunk = meta.iloc[start:start + bs]
        x = load_images(chunk.ImageId).to(device)
        y = torch.tensor(chunk.label.values, device=device)
        saliency = gradcam(model, x, y) if masked else None
        if saliency is not None:
            # A degenerate CAM (all zeros) would silently pick the top-left blocks; drop it.
            bad = degenerate_saliency(saliency)
            if bad.any():
                dropped += [i for i, b in zip(chunk.ImageId, bad.tolist()) if b]
                keep = ~bad
                x, y, saliency, chunk = x[keep], y[keep], saliency[keep], chunk[keep.cpu().numpy()]
                if len(chunk) == 0:
                    continue
        masks = build_masks(saliency, cfg["regions"], cfg["mask_areas"])
        rows += evaluate_batch(model, x, y, list(chunk.ImageId), masks, cfg, start)
        print(f"  {min(start + bs, len(meta))}/{len(meta)} images")

    out_dir = REPO / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_image.csv", index=False)

    if df.empty:
        raise SystemExit("no rows to summarize: every image was dropped (degenerate Grad-CAM?)")
    summary = summarize(df)
    summary_path = REPO / cfg["summary_csv"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    cols = ["none" if q is None else q for q in cfg["eval_jpeg_qualities"]]
    table = (summary.set_index(SUMMARY_KEYS)["success_clean_correct"]
             .unstack("eval_jpeg")[cols].map("{:.1%}".format))
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print("\nAttack success after the real roundtrip, on images still correct when clean"
              " at that quality (columns = eval JPEG quality):")
        print(table)
    if dropped:
        print(f"\ndropped {len(dropped)} image(s) with a degenerate Grad-CAM: {dropped}")
    print(f"\nwrote {out_dir / 'per_image.csv'} and {summary_path}")


if __name__ == "__main__":
    main()
