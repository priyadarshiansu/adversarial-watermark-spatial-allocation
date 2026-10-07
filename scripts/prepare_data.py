"""Module 1 data preparation: preprocess once, then select the experimental image set.

    uv run python scripts/prepare_data.py

1. Resizes data/raw/*.png (299x299) to data/processed/*.png (224x224, bicubic, antialiased).
2. Runs clean ResNet-50 on the processed images and writes data/nips2017/selected.csv:
   the images it classifies correctly (proposal §5, Statistics). Commit selected.csv so the
   whole team uses the same image set.
"""

import argparse

import pandas as pd
import torch

from awsa.data import (
    SELECTED_CSV,
    correctly_classified,
    load_images,
    load_metadata,
    preprocess_dataset,
)
from awsa.models import load_resnet50
from awsa.utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true", help="re-resize existing files")
    args = parser.parse_args()

    set_seed(0)
    device = get_device(args.device)

    written = preprocess_dataset(overwrite=args.overwrite)
    print(f"preprocessed {written} new images into data/processed/")

    meta = load_metadata()
    model = load_resnet50(device)
    ok = []
    for start in range(0, len(meta), 100):
        chunk = meta.iloc[start:start + 100]
        x = load_images(chunk.ImageId).to(device)
        y = torch.tensor(chunk.label.values, device=device)
        ok.append(correctly_classified(model, x, y).cpu())
    ok = torch.cat(ok).numpy()

    pd.DataFrame({"ImageId": meta.ImageId[ok]}).to_csv(SELECTED_CSV, index=False)
    print(f"clean accuracy {ok.mean():.1%}: {ok.sum()}/{len(meta)} images -> {SELECTED_CSV}")


if __name__ == "__main__":
    main()
