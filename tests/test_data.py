from awsa.data import NIPS_DIR, load_metadata


def test_labels_are_zero_indexed():
    df = load_metadata()
    assert len(df) == 1000
    assert df["label"].between(0, 999).all()


def test_category_offset():
    import pandas as pd

    cats = pd.read_csv(NIPS_DIR / "categories.csv")
    assert cats.loc[cats.CategoryId == 1, "CategoryName"].item().startswith("tench")


def test_preprocess_image_is_224_on_uint8_grid():
    import torch

    from awsa.data import preprocess_image

    x = torch.rand(3, 299, 299, generator=torch.Generator().manual_seed(0))
    out = preprocess_image(x)
    assert out.shape == (3, 224, 224)
    assert out.min() >= 0 and out.max() <= 1
    assert torch.equal(out, torch.round(out * 255) / 255)


def test_preprocess_and_load_roundtrip(tmp_path, monkeypatch):
    """preprocess_dataset writes PNGs that load_images reads back bit-exactly."""
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from awsa import data

    ids = ["a", "b"]
    monkeypatch.setattr(data, "load_metadata", lambda: pd.DataFrame({"ImageId": ids}))
    raw, out = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    rng = np.random.default_rng(0)
    for i in ids:
        Image.fromarray(rng.integers(0, 256, (299, 299, 3), dtype=np.uint8)).save(raw / f"{i}.png")

    assert data.preprocess_dataset(raw, out) == 2
    assert data.preprocess_dataset(raw, out) == 0  # skips existing files
    x = data.load_images(ids, out)
    assert x.shape == (2, 3, 224, 224) and x.dtype == torch.float32
    expected = data.preprocess_image(data._read_png(raw / "a.png"))
    assert torch.equal(x[0], expected)


def test_correctly_classified(tiny_model, images):

    from awsa.data import correctly_classified

    pred = tiny_model(images).argmax(1)
    ok = correctly_classified(tiny_model, images, pred, batch_size=1)
    assert ok.all()
    assert not correctly_classified(tiny_model, images, (pred + 1) % 10).any()


def test_selected_metadata_is_ordered_unique_subset():
    from awsa.data import load_selected_metadata

    full = load_metadata()
    sel = load_selected_metadata()
    assert len(sel) > 0
    assert sel.ImageId.is_unique
    assert set(sel.ImageId) <= set(full.ImageId)
    order = {iid: i for i, iid in enumerate(full.ImageId)}
    positions = [order[i] for i in sel.ImageId]
    assert positions == sorted(positions)
