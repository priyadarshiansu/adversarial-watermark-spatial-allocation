# Spatial Allocation of Adversarial and Watermark Signals

CAP5610 Final Project — Introduction to Machine Learning, FIU

## Project overview

An image owner wants two invisible protections at once:

1. an **adversarial perturbation** that stops automated classifiers from recognizing the image, and
2. an **invisible watermark** that carries a recoverable ownership payload.

Both signals go into the same image. This project asks **where** they should go: on the same
pixels (co-located) or in separate, non-overlapping regions, with each signal's strength held
fixed (matched budget), and whether the answer survives JPEG compression, resizing and blur.

Full proposal: [`docs/proposal_v2.md`](docs/proposal_v2.md).

## Research questions

- **RQ1:** At matched perceptual distortion, does placing the perturbation and watermark in
  separate regions give a better joint result than placing them together?
- **RQ2:** Does any advantage of separation survive JPEG compression, resizing and blur?
- **RQ3:** Does the advantage come from separation itself or from saliency-aware placement?

Baseline checks (validate the setup, not contributions): **B1** perturbation in salient regions
beats low-saliency regions; **B2** co-located signals interfere measurably.

## Experimental setup

| Component | Choice |
|---|---|
| Classifier | ResNet-50 (torchvision, ImageNet-1k); ViT-B/16 optional |
| Attack | PGD (L∞), JPEG-aware variant |
| Saliency | Grad-CAM → equal-area 8×8 block masks |
| Watermark | Block-DCT, 32-bit payload |
| Dataset | NeurIPS 2017 adversarial dev set, 1,000 images → 224×224 ([`data/README.md`](data/README.md)) |
| Processing | JPEG, resize, Gaussian blur |
| Metrics | Attack success, bit error rate, PSNR, SSIM, LPIPS |

Conditions: adversarial only · watermark only · global/global · co-located ·
saliency-aware separated · random separated.

## Setup (uv)

Install [uv](https://docs.astral.sh/uv/) once:

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the repo root:

```bash
uv sync                    # creates .venv with Python 3.12 and the exact locked versions
uv run nbstripout --install   # once per clone: strips notebook outputs on commit
uv run pytest              # run the tests
uv run python scripts/prepare_data.py  # once: resize data/raw -> data/processed, select images
uv run python scripts/smoke_test.py   # end-to-end check (needs data, see data/README.md)
uv run jupyter lab         # notebooks
```

- Add a dependency: `uv add <package>` (or `uv add --dev <package>` for tools). Commit
  `pyproject.toml` **and** `uv.lock` together.
- `requirements.txt` is generated for anyone not using uv; don't edit it by hand:
  `uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`
- **GPU (NVIDIA):** on Windows and Linux, uv installs PyTorch built for **CUDA 13.2** from
  PyTorch's own index (see `[tool.uv.sources]` in `pyproject.toml`). You need an NVIDIA driver
  that supports CUDA ≥ 13.2: run `nvidia-smi` and check "CUDA Version" in the top-right corner.
- **No NVIDIA GPU, or an older driver:** everything still works on CPU; the code falls back
  automatically (`awsa.utils.get_device`). macOS uses the regular PyPI build (CPU).
  Windows-on-ARM is not supported by the CUDA index.
- **Pascal GPUs (e.g. the lab Quadro P1000):** CUDA 13 dropped Pascal (sm_61), so the `cu132`
  wheels run CPU-only there. On that machine install the CUDA 12.6 build instead and leave
  `pyproject.toml` alone (whether the whole team standardizes on `cu126` is still open):
  ```bash
  uv sync
  uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
  ```
  Re-run the second line after any `uv sync`, which restores the locked `cu132` wheels.
- **`requirements.txt` without uv:** it pins `torch==2.14.1+cu132` for Windows/Linux, so plain
  pip needs the PyTorch index too:
  `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132`.
- Check which device you got:
  ```bash
  uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  uv run python scripts/smoke_test.py   # end-to-end check on 16 real images; prints device=cuda/cpu
  ```

## Repository structure

```
src/awsa/            # reusable code (import as `awsa`)
  data.py            #   Module 1 · metadata, preprocessing, loading, image selection
  models.py          #   Module 1 · ResNet-50 wrapper that takes [0,1] pixels
  metrics.py         #   Module 1 · ASR, confidence drop, BER, PSNR, SSIM, LPIPS
  attacks.py         #   Module 2 · FGSM, masked PGD, differentiable-JPEG / EOT transform
  saliency.py        #   Module 3 · Grad-CAM
  masks.py           #   Module 3 · block-level masks
  watermark.py       #   Module 4 · block-DCT watermark
  distortions.py     #   shared · uint8/JPEG round trip, resize, blur
tests/               # pytest (fast, no downloads)
  utils.py           #   shared · set_seed, get_device
scripts/smoke_test.py  # real ResNet-50 + real images + PGD + processing sanity check
scripts/prepare_data.py   # Module 1 · preprocess once + select correctly classified images
scripts/calibrate_eps.py  # Module 2 · eps × mask-area sweep for plain / JPEG-aware PGD (configs/eps_calibration.yaml)
scripts/watermark_area_strength_sweep.py  # Module 4 · area × strength calibration (configs/watermark_calibration.yaml)
scripts/watermark_pilot.py, watermark_strength_sweep.py  # Module 4 · smaller pilots
notebooks/           # exploration and figures; import from awsa, outputs stripped
scripts/  configs/   # experiment runners and their YAML configs
data/                # CSV metadata committed; images git-ignored
results/  figures/   # small summary CSVs and plots are committed; results/runs/ (per-image dumps) is git-ignored
docs/                # proposal, interfaces & decisions, progress updates
papers/              # reading notes (no PDFs: the repo is public)
```

Conventions every module follows (tensor shapes, masks, labels) and the decisions log:
[`docs/interfaces.md`](docs/interfaces.md).

## Team workflow

- `main` is protected: work on a branch (`module1-data-pipeline`, `module2-attack`, …) and
  merge through a pull request.
- Run `uv run pytest` before opening a PR.
- Notebooks are for exploration; anything reused goes into `src/awsa/` with a test.

## Status

Modules 1–4 are implemented, tested and hardened (data pipeline and selected image set,
masked / JPEG-aware PGD, Grad-CAM block masks, luma block-DCT watermark, canonical uint8 /
JPEG / resize / blur processing). Next, on the lab GPU:

1. `uv run python scripts/calibrate_eps.py` and
   `uv run python scripts/watermark_area_strength_sweep.py` to choose the mask area, the ε set
   and the watermark strengths (candidates and the decision rule are in
   [`docs/interfaces.md`](docs/interfaces.md#pending-calibration-frozen-after-scriptscalibrate_epspy-and-scriptswatermark_area_strength_sweeppy-are-run-on-the-lab-pc)).
2. Freeze those in the decisions log.
3. Build the single experiment runner for the six conditions and both application orders
   (week-4 Decision Gate).
