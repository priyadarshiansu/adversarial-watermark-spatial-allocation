"""NeurIPS 2017 adversarial dev set loading (Module 1). See data/README.md."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
NIPS_DIR = DATA_DIR / "nips2017"


def load_metadata(path: Path = NIPS_DIR / "images.csv") -> pd.DataFrame:
    """images.csv with an added `label` column in torchvision index order.

    The CSV's TrueLabel is 1-indexed (1 = tench); torchvision's ResNet-50 is 0-indexed.
    """
    df = pd.read_csv(path)
    df["label"] = df["TrueLabel"] - 1
    return df
