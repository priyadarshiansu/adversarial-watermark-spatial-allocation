# Data

## NeurIPS 2017 adversarial competition dev set

1,000 ImageNet-compatible photos (299×299 RGB PNG) with ImageNet labels, from the
NIPS 2017 *Adversarial Attacks and Defences* competition (Kurakin et al., arXiv:1804.00097).

**Download:** [Kaggle: google-brain/nips-2017-adversarial-learning-development-set](https://www.kaggle.com/datasets/google-brain/nips-2017-adversarial-learning-development-set)

### Layout

```
data/
├── nips2017/
│   ├── images.csv       # committed: ImageId, bounding box, TrueLabel, TargetClass, license, author
│   ├── categories.csv   # committed: CategoryId (1-1000) -> name
│   └── selected.csv     # committed: correctly classified ImageIds (scripts/prepare_data.py)
├── raw/                 # NOT committed: put the 1,000 PNGs here (data/raw/<ImageId>.png)
└── processed/           # NOT committed: 224×224 PNGs produced by the preprocessing script
```

After downloading, copy the contents of the Kaggle `images/` folder into `data/raw/`, then run

```bash
uv run python scripts/prepare_data.py
```

This resizes every image once (299 → 224, bicubic, antialiased, no crop) into
`data/processed/` and writes `data/nips2017/selected.csv`, the images ResNet-50 classifies
correctly when clean. Load images in code with `awsa.data.load_images(ids)`; never resize
them yourself, so every script sees identical pixels.

### Gotchas

- **Labels are 1-indexed** (1 = tench). torchvision's ResNet-50 is 0-indexed, so
  `label = TrueLabel - 1`. `awsa.data.load_metadata()` adds this `label` column.
- `TargetClass` is for targeted attacks; this project uses untargeted attacks and ignores it.
- `x1, y1, x2, y2` are the main object's bounding box (fractions of width/height). Useful
  for sanity-checking Grad-CAM masks.
- **Licensing:** each image is Creative Commons licensed (`License`, `Author` columns).
  Do not commit the images. Any image shown in the report or slides needs an author credit.
- These images are ImageNet-*compatible*, not from the ImageNet validation set. Say
  "NeurIPS 2017 dev set" in the report.
