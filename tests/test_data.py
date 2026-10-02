from awsa.data import NIPS_DIR, load_metadata


def test_labels_are_zero_indexed():
    df = load_metadata()
    assert len(df) == 1000
    assert df["label"].between(0, 999).all()


def test_category_offset():
    import pandas as pd

    cats = pd.read_csv(NIPS_DIR / "categories.csv")
    assert cats.loc[cats.CategoryId == 1, "CategoryName"].item().startswith("tench")
