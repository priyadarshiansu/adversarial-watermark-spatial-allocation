"""End-to-end smoke test: real ResNet-50 + real NeurIPS 2017 images + package PGD.

Run from the repo root:
    uv run python scripts/smoke_test.py            # 16 images
    uv run python scripts/smoke_test.py --n 64

Checks that the pieces fit together on real data; it is NOT an experiment.
Expected: high clean accuracy, PGD (eps=2/255) flips most images, the flips survive
uint8 rounding, and most are undone by JPEG q=75 (why the JPEG-aware attack matters).
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, resize

from awsa import IMAGE_SIZE
from awsa.attacks import pgd
from awsa.data import DATA_DIR, load_metadata
from awsa.distortions import jpeg, quantize_uint8
from awsa.metrics import attack_success
from awsa.models import imagenet_categories, load_resnet50
from awsa.utils import get_device, set_seed


def load_image(path: Path) -> torch.Tensor:
    img = pil_to_tensor(Image.open(path).convert("RGB")).float() / 255
    return resize(img, [IMAGE_SIZE, IMAGE_SIZE], antialias=True).clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16, help="number of images")
    parser.add_argument("--eps", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(0)
    device = get_device(args.device)
    raw = DATA_DIR / "raw"
    meta = load_metadata().head(args.n)
    missing = [i for i in meta.ImageId if not (raw / f"{i}.png").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing from {raw}. See data/README.md.")

    x = torch.stack([load_image(raw / f"{i}.png") for i in meta.ImageId]).to(device)
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
        asr_uint8 = attack_success(model(quantize_uint8(x_adv)), yc).mean()
        asr_jpeg = attack_success(model(jpeg(quantize_uint8(x_adv), 75)), yc).mean()
        clean_jpeg_acc = 1 - attack_success(model(jpeg(xc, 75)), yc).mean()
    print(f"max |delta| = {linf:.2f}/255 (should be <= {args.eps * 255:.2f})")
    print(f"attack success  float: {asr_float:.1%}  uint8: {asr_uint8:.1%}  "
          f"after JPEG q75: {asr_jpeg:.1%}")
    print(f"(clean images still correct after JPEG q75: {clean_jpeg_acc:.1%})")


if __name__ == "__main__":
    main()
