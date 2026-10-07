"""End-to-end smoke test: real ResNet-50 + real NeurIPS 2017 images + package PGD.

Run from the repo root (needs data/processed/ and data/nips2017/selected.csv):
    uv run python scripts/smoke_test.py            # 16 images
    uv run python scripts/smoke_test.py --n 64

Checks that the pieces fit together on real data; it is NOT an experiment.
Expected: high clean accuracy, PGD (eps=2/255) flips most images, the flips survive
uint8 rounding, and most are undone by JPEG q=75 (why the JPEG-aware attack matters).
"""

import argparse
import time

import torch

from awsa.attacks import pgd
from awsa.data import load_images, load_selected_metadata
from awsa.distortions import gaussian_blur, resize_roundtrip, roundtrip
from awsa.metrics import attack_success, lpips_distance, psnr, ssim
from awsa.models import imagenet_categories, load_resnet50
from awsa.utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16, help="number of images")
    parser.add_argument("--eps", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(0)
    device = get_device(args.device)
    try:
        meta = load_selected_metadata().head(args.n)
        x = load_images(meta.ImageId).to(device)
    except FileNotFoundError as e:
        raise SystemExit(f"{e} See data/README.md.") from e
    y = torch.tensor(meta.label.values, device=device)
    print(f"device={device}  images={tuple(x.shape)}")

    model = load_resnet50(device)
    names = imagenet_categories()

    with torch.no_grad():
        clean_pred = model(x).argmax(1)
    correct = clean_pred == y
    print(f"clean accuracy: {correct.float().mean():.1%} ({int(correct.sum())}/{len(y)})")
    for i in range(min(3, len(y))):
        print(f"  {meta.ImageId.iloc[i]}: true={names[y[i]]!r} pred={names[clean_pred[i]]!r}")

    # Attack only images the model gets right (as in the real protocol).
    xc, yc = x[correct], y[correct]
    if len(yc) == 0:
        raise SystemExit("No correctly classified images; check the label offset.")
    t0 = time.time()
    x_adv = pgd(model, xc, yc, eps=args.eps, alpha=args.eps / 4, steps=args.steps)
    print(f"PGD eps={args.eps * 255:.1f}/255, {args.steps} steps: {time.time() - t0:.1f}s")

    linf = (x_adv - xc).abs().amax().item() * 255
    with torch.no_grad():
        asr_float = attack_success(model(x_adv), yc).mean()
        asr_uint8 = attack_success(model(roundtrip(x_adv)), yc).mean()
        asr_jpeg = attack_success(model(roundtrip(x_adv, 75)), yc).mean()
        asr_resize = attack_success(model(resize_roundtrip(x_adv, 0.5)), yc).mean()
        asr_blur = attack_success(model(gaussian_blur(x_adv, 1.0)), yc).mean()
        clean_jpeg_acc = 1 - attack_success(model(roundtrip(xc, 75)), yc).mean()
    x_adv_q = roundtrip(x_adv)
    print(f"max |delta| = {linf:.2f}/255 (should be <= {args.eps * 255:.2f})")
    print(f"attack success  float: {asr_float:.1%}  uint8: {asr_uint8:.1%}  "
          f"after JPEG q75: {asr_jpeg:.1%}")
    print(f"attack success  after resize x0.5: {asr_resize:.1%}  "
          f"after blur sigma=1: {asr_blur:.1%}")
    print(f"(clean images still correct after JPEG q75: {clean_jpeg_acc:.1%})")
    print(f"x_adv (uint8) vs x: PSNR {psnr(x_adv_q, xc).mean():.2f} dB  "
          f"SSIM {ssim(x_adv_q, xc).mean():.4f}  LPIPS {lpips_distance(x_adv_q, xc).mean():.4f}")


if __name__ == "__main__":
    main()
