# CLAUDE.md

Guidance for Claude (and other AI coding assistants) working in this repo.
Humans: the same rules apply to you; see README.md and docs/interfaces.md.

## What this project is

CAP5610 research project. Question: at a matched per-signal budget, should an adversarial
perturbation (PGD vs ResNet-50) and an invisible block-DCT watermark share image regions or
use separate regions, and does the answer survive JPEG / resize / blur?
Full design: `docs/proposal_v2.md`. Shared conventions + decisions log: `docs/interfaces.md`.

## Commands (uv only, never pip)

```bash
uv sync                               # install / update the environment
uv run pytest                         # unit tests (fast, no downloads); must pass before a PR
uv run python scripts/smoke_test.py   # end-to-end check on real images (needs data/raw/)
uv run ruff check src tests scripts   # lint
uv add <pkg> / uv add --dev <pkg>     # add a dependency; commit pyproject.toml AND uv.lock
uv export --no-hashes --no-dev --no-emit-project -o requirements.txt   # after any dep change
```

## Layout

- `src/awsa/`: the package. Reusable code goes here, with a test in `tests/`.
- `notebooks/`: exploration and figures only; import from `awsa`, never copy its code.
- `scripts/` + `configs/`: experiment runners and their YAML configs.
- `data/nips2017/*.csv` committed; images in `data/raw/` and `data/processed/` are git-ignored.

## Conventions that must not be broken

- Images are `float32` in `[0, 1]`, shape `(B, 3, 224, 224)`, RGB, **pixel space**.
  ImageNet normalization happens only inside `models.Normalized`. Never normalize elsewhere.
- Block masks are `bool (B, 1, 28, 28)` (one cell per 8×8 block); convert with
  `masks.blocks_to_pixels`. Never build masks at pixel level.
- Labels are torchvision 0-indexed: `label = TrueLabel - 1` (`data.load_metadata` does it).
- Every reported metric goes through `distortions.roundtrip` / `distortions.*` so all
  experimental arms see identical uint8 + JPEG processing.
- Perturbation budgets are in pixel units (e.g. `eps = 2/255`).
- Each `src/awsa` module is owned by a team member (see `docs/interfaces.md`). Keep the public
  signatures; changing an interface means updating `docs/interfaces.md` first.
- Mask area, the ε set and the watermark strengths are *pending calibration*: parameterise
  them in `configs/`, never hard-code a final value, and never tune them per arm.
- Per-image result dumps go to `results/runs/` (git-ignored); commit only small summary CSVs.

## Rules

- Never commit to `main`; work on a branch and open a PR.
- Never commit dataset images, model weights (`*.pth`) or large result dumps.
- Notebooks are committed without outputs (nbstripout; run `uv run nbstripout --install` once).
- Don't change experimental decisions (ε, mask area, colour space, budget units) in code
  without recording them in the decisions log in `docs/interfaces.md`.
- Seed everything (`awsa.utils.set_seed`) and use `get_device()`; don't hard-code `cuda`.

## Environment notes

- Windows/Linux get CUDA 13.2 PyTorch via `[tool.uv.sources]`; CPU fallback is automatic.
- The team develops on Windows; `.gitattributes` normalizes line endings to LF in the repo.
