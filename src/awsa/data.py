"""NeurIPS 2017 adversarial dev set loading (Module 1). See data/README.md.

Preprocessing is done once (`preprocess_dataset`, run by scripts/prepare_data.py):
299x299 raw PNGs are resized directly to 224x224 (bicubic, antialiased, no crop) and saved
as PNGs in data/processed/. Everything else loads those files with `load_images`, so every
script and notebook sees the same uint8 pixels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.transforms.functional import InterpolationMode, resize

from awsa import IMAGE_SIZE

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
NIPS_DIR = DATA_DIR / "nips2017"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SELECTED_CSV = NIPS_DIR / "selected.csv"


def load_metadata(path: Path = NIPS_DIR / "images.csv") -> pd.DataFrame:
    """images.csv with an added `label` column in torchvision index order.

    The CSV's TrueLabel is 1-indexed (1 = tench); torchvision's ResNet-50 is 0-indexed.
    """
    df = pd.read_csv(path)
    df["label"] = df["TrueLabel"] - 1
    return df


def _read_png(path: Path) -> torch.Tensor:
    """PNG -> float32 (3, H, W) in [0, 1]."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def preprocess_image(x: torch.Tensor) -> torch.Tensor:
    """(3, H, W) in [0, 1] -> (3, 224, 224) on the uint8 grid. Bicubic, antialiased, no crop."""
    x = resize(x, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BICUBIC,
               antialias=True)
    return torch.round(x.clamp(0, 1) * 255) / 255


def preprocess_dataset(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR,
                       overwrite: bool = False) -> int:
    """Resize every raw PNG listed in images.csv into out_dir. Returns the number written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for image_id in load_metadata().ImageId:
        src, dst = raw_dir / f"{image_id}.png", out_dir / f"{image_id}.png"
        if dst.exists() and not overwrite:
            continue
        if not src.exists():
            raise FileNotFoundError(f"{src} missing. See data/README.md.")
        x = preprocess_image(_read_png(src))
        arr = (x * 255).round().byte().permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(dst)
        written += 1
    return written


def load_images(image_ids, processed_dir: Path = PROCESSED_DIR) -> torch.Tensor:
    """Load preprocessed images as a (B, 3, 224, 224) float32 batch in [0, 1]."""
    paths = [processed_dir / f"{i}.png" for i in image_ids]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} processed images missing (e.g. {missing[0]}). "
            "Run `uv run python scripts/prepare_data.py` first."
        )
    return torch.stack([_read_png(p) for p in paths])


@torch.no_grad()
def correctly_classified(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                         batch_size: int = 64) -> torch.Tensor:
    """(B,) bool: True where the model's clean prediction equals the label."""
    preds = [model(x[i:i + batch_size]).argmax(1) for i in range(0, len(x), batch_size)]
    return torch.cat(preds) == y


def load_selected_metadata(path: Path = SELECTED_CSV) -> pd.DataFrame:
    """Metadata restricted to the images ResNet-50 classifies correctly when clean.

    The experimental image set (proposal §5, Statistics). selected.csv is written by
    scripts/prepare_data.py.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run `uv run python scripts/prepare_data.py`.")
    ids = set(pd.read_csv(path).ImageId)
    df = load_metadata()
    return df[df.ImageId.isin(ids)].reset_index(drop=True)
