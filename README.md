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
uv run python scripts/smoke_test.py   # end-to-end check (needs data/raw, see data/README.md)
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
- Check which device you got:
  ```bash
  uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  uv run python scripts/smoke_test.py   # end-to-end check on 16 real images; prints device=cuda/cpu
  ```

## Repository structure

```
src/awsa/            # reusable code (import as `awsa`)
  data.py            #   Module 1 · dataset metadata and loading
  models.py          #   Module 1 · ResNet-50 wrapper that takes [0,1] pixels
  metrics.py         #   Module 1 · ASR, BER, PSNR, SSIM, LPIPS
  attacks.py         #   Module 2 · FGSM, masked PGD, JPEG-aware hook
  saliency.py        #   Module 3 · Grad-CAM
  masks.py           #   Module 3 · block-level masks
  watermark.py       #   Module 4 · block-DCT watermark
  distortions.py     #   shared · uint8/JPEG round trip, resize, blur
tests/               # pytest (fast, no downloads)
scripts/smoke_test.py  # real ResNet-50 + real images + PGD sanity check
notebooks/           # exploration and figures; import from awsa, outputs stripped
scripts/  configs/   # experiment runners and their YAML configs
data/                # CSV metadata committed; images git-ignored
results/  figures/   # experiment outputs and plots
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

Environment, ResNet-50 baseline and single-image FGSM/PGD done
([`docs/progress_update_01.md`](docs/progress_update_01.md)). Repo reorganized into the
`awsa` package with uv. Next: Module 1 data pipeline (preprocessing + clean baseline on
the NeurIPS 2017 set).
