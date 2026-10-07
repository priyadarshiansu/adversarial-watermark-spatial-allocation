# Interfaces and Decisions

Shared contract between the four modules. Modules are built in parallel (proposal §9), so
everything below must hold for every function in `src/awsa/`. If you need to change a
convention, change it here first and tell the team.

## Tensor conventions

| Thing | Type / shape | Notes |
|---|---|---|
| Image | `float32`, `(B, 3, 224, 224)`, RGB, values in `[0, 1]` | Pixel space everywhere. ε = 2/255 means 2/255 of the pixel range. |
| Normalization | Only inside `models.Normalized` | Never normalize before calling an attack, watermark or metric. |
| Label | `int64`, `(B,)`, torchvision index (0 = tench) | NeurIPS CSV is 1-indexed: `label = TrueLabel - 1` (`data.load_metadata` does this). |
| Block mask | `bool`, `(B, 1, 28, 28)` | One cell per 8×8 block; 224 / 8 = 28. `True` = region in use. |
| Pixel mask | `float32`, `(B, 1, 224, 224)` | Only via `masks.blocks_to_pixels`. Never threshold at pixel level. |
| Payload | `int64` or `bool`, `(B, 32)` | 32 bits, no error correction initially. |

## Module APIs

### Module 1: data, models, metrics (`data.py`, `models.py`, `metrics.py`)
```python
load_metadata() -> DataFrame              # images.csv + 0-indexed `label`
load_selected_metadata() -> DataFrame     # only images ResNet-50 gets right clean (selected.csv)
load_images(image_ids) -> (B, 3, 224, 224)  # from data/processed/; the ONLY way to load images
preprocess_dataset() / preprocess_image(x)  # 299 -> 224 bicubic, once (scripts/prepare_data.py)
correctly_classified(model, x, y) -> (B,) bool
load_resnet50(device) -> nn.Module        # frozen, takes [0,1] pixels
attack_success(logits, y) -> (B,)         # untargeted: argmax != y
confidence_drop(logits_clean, logits_adv, y) -> (B,)  # true-class softmax prob, clean - adv
bit_error_rate(bits_true, bits_pred) -> (B,)
psnr(x, x_ref) / ssim(x, x_ref) / lpips_distance(x, x_ref) -> (B,)
```
PSNR uses peak 1.0; SSIM is scikit-image (7×7 window, mean over RGB); LPIPS is AlexNet v0.1.
Never resize images yourself: scripts and notebooks call `load_images`.

### Module 2: attacks (`attacks.py`)
```python
pgd(model, x, y, eps, alpha, steps, mask=None, random_start=False,
    transform=None, eot_samples=1, generator=None) -> x_adv
```
Guarantees: `|x_adv - x|_inf <= eps`, `x_adv` in `[0, 1]`, `x_adv == x` where `mask == 0`.
The JPEG-aware attack passes `DifferentiableJPEG(quality)` as `transform`: straight-through
uint8 rounding, then kornia's differentiable JPEG (standard tables, 4:2:0, like PIL).
`DifferentiableJPEG((low, high))` draws a random quality per image; with `eot_samples > 1`
that is EOT over JPEG. Report results only after `distortions.roundtrip(x_adv, q)`.

### Module 3: saliency and masks (`saliency.py`, `masks.py`)
```python
gradcam(model, x, y) -> (B, 1, 28, 28) float in [0, 1]      # target layer: layer4; works under no_grad
degenerate_saliency(cam) -> (B,) bool     # True where ReLU zeroed the whole map; exclude those images
top_fraction_mask(saliency_blocks, area) -> (B, 1, 28, 28) bool            # highest-saliency blocks
bottom_fraction_mask(saliency_blocks, area, exclude=None) -> (B, 1, 28, 28) bool  # lowest-saliency blocks, disjoint from `exclude`
random_disjoint_masks(shape, area, generator, device=None) -> (mask_a, mask_b)
blocks_to_pixels(block_mask) -> (B, 1, 224, 224) float       # block (i, j) -> pixels [8i:8i+8, 8j:8j+8]
block_variance(x) -> (B, 1, 28, 28) float   # per-block luma variance (smoothness-confound check, proposal §7)
```
Every masked arm covers exactly `round(area * 784)` blocks. The separated arm uses
`top_fraction_mask` for the attack and `bottom_fraction_mask(..., exclude=top)` for the watermark;
at area 0.5 the low mask is the exact complement of the high one, below 0.5 it is the genuine bottom
fraction. A degenerate CAM (all zeros) would otherwise pick top-left blocks silently: runners must
drop images flagged by `degenerate_saliency` and report how many.

### Module 4: watermark (`watermark.py`)
```python
embed(x, bits, mask, strength, key) -> x_wm
extract(x, mask, key) -> bits            # blind: needs only mask + key, never the original
payload_coefficients() -> list[(u, v)]   # [(2, 3), (3, 2)], for the mechanism analysis
block_bit_index(mask, key) -> (B, 1, 28, 28) int64   # payload bit carried by each block, -1 outside mask
extract_soft(x, mask, key) -> (B, 1, 28, 28) float   # per-block margin DCT[2,3] - DCT[3,2]; sign = bit
```
Scheme: luma-only, orthonormal 8×8 DCT aligned to the JPEG grid, coefficient-pair ordering
(`DCT[2,3] - DCT[3,2] >= +strength` for 1, `<= -strength` for 0) with the minimum change per
block; bit `j % 32` goes to the `j`-th block of a key-seeded permutation of the masked blocks;
decoding is a majority vote per bit with ties broken by the summed margin. `strength` is a
coefficient-margin parameter in orthonormal-DCT units of Y ∈ [0, 1]. It is **not** a QIM step:
the pixel change a block receives depends on that block's own coefficient difference, so the same
`strength` can produce different distortion across blocks because their existing DCT coefficients differ; salient and low-saliency regions are therefore measured rather than assumed equivalent. See the decisions
log ("Matched distortion") for how the arms are compared despite this.

### Shared processing (`distortions.py`)
```python
quantize_uint8(x)                 # round to the 8-bit grid (round-to-nearest)
roundtrip(x, jpeg_quality=None)   # uint8 quantization, then optional real PIL JPEG (4:2:0, standard tables)
jpeg(x, quality) / resize_roundtrip(x, scale) / gaussian_blur(x, sigma)
```
Canonical processing rule: every processed image is one that exists on the 8-bit grid.
`none` = uint8; `jpeg` = uint8 → JPEG; `resize` = uint8 → down/up-scale back to 224 → uint8;
`blur` = uint8 → Gaussian blur → uint8. **Every** reported number (attack success *and* bit
error rate) goes through `roundtrip` or one of these functions, so all arms see identical
processing (proposal §5, "Attack"). Attack success after processing is judged against the
*processed clean image* (`model(roundtrip(x, q))`), and images whose processed clean version is
already misclassified are excluded from the headline rate, so JPEG's own damage is not counted as
attack success.

## Decisions log

### Decided

| Date | Decision | Why |
|---|---|---|
| 2026-10-02 | Dataset: NeurIPS 2017 adversarial dev set (1,000 ImageNet-compatible images, 299×299). Images are not committed; see `data/README.md`. | Standard attack benchmark, ImageNet labels, free. CIFAR/MNIST break Grad-CAM, 8×8 blocks, JPEG and LPIPS at 32×32. |
| 2026-10-02 | Preprocess once: resize 299 → 224 directly (bicubic, antialiased), no crop; save PNG to `data/processed/`. | Images are already square, so cropping would only discard content. |
| 2026-10-02 | Labels: `label = TrueLabel - 1`. | CSV is 1-indexed, torchvision is 0-indexed. |
| 2026-10-02 | Dependencies managed with uv (`pyproject.toml` + `uv.lock`). | Faster, reproducible lockfile. |
| 2026-10-07 | Image set: images ResNet-50 classifies correctly when clean, listed in `data/nips2017/selected.csv` (written by `scripts/prepare_data.py`, committed). | Proposal §5 (Statistics): attack success is only meaningful on correctly classified images. |
| 2026-10-02 | PyTorch from the CUDA 13.2 index (`cu132`) on Windows/Linux; PyPI on macOS. CPU fallback is automatic. | PyPI's Windows wheel is CPU-only; the team has local NVIDIA GPUs (RTX 4070). Pascal cards (Quadro P1000) need `cu126`, see README; whether to standardize on it is open. |
| 2026-10-07 | Classifier weights: torchvision `ResNet50_Weights.IMAGENET1K_V2`, fed our direct 224 resize (no 232/centre-crop). Clean accuracy on the dev set: 963/1000. | V2 is the frozen checkpoint for this project, used with the direct 299→224 preprocessing above. Don't switch weights or preprocessing mid-project: `selected.csv` depends on them. |
| 2026-10-07 | Watermark colour space: luma (Y) only; the same ΔY is added to R, G, B so chroma is untouched. | JPEG subsamples chroma 4:2:0. The mechanism analysis measures perturbation energy in the Y-channel DCT. |
| 2026-10-07 | Decoder knowledge: blind decoding. The region mask and the key are shared secrets between owner and decoder; the original image is never needed. Grad-CAM is not recomputed at decode time. | Grad-CAM cannot be recomputed reliably after attack + JPEG. State this in the threat model. |
| 2026-10-07 | Watermark scheme: coefficient-pair ordering on DCT(2,3)/(3,2) with minimum change (see Module 4 above), not QIM. `strength` is a coefficient margin. | Implemented and calibrated this way; changing to QIM would invalidate the calibration. The content-dependent distortion is handled by matching measured distortion (next row). |
| 2026-10-07 | Matched distortion: reporting PSNR/SSIM/LPIPS per arm is necessary but not sufficient. The co-located vs separated comparison is made at matched *measured* distortion: either strengths calibrated per arm to equal mean LPIPS/PSNR, or comparable points read off the Pareto curves. | Proposal §6: equal mask area ≠ equal perceptual distortion, and the watermark's distortion depends on block content. |
| 2026-10-07 | Resize processing: downscale by `scale` then upscale back to 224 (`resize_roundtrip`), uint8 before and after. Blur likewise uint8 before and after. | Keeps the 8×8 grid and the mask valid; every processed image is a real 8-bit image. |
| 2026-10-07 | JPEG-aware attack: kornia differentiable JPEG (straight-through uint8 rounding first) as the PGD `transform`; a quality range + `eot_samples > 1` gives EOT over JPEG. Reported numbers always come from the real PIL JPEG (`roundtrip`). | Both are implemented; kornia's approximation differs from PIL by ~30–40% of JPEG's own error, which is fine for gradients but not for reporting. |
| 2026-10-07 | ε must be a whole multiple of 1/255. Candidates for calibration: {1, 2, 3, 4, 6, 8}/255. | Clean images sit on the 8-bit grid, so ±0.5/255 rounds to ±1/255 for most pixels and the stated bound is false after saving. |
| 2026-10-07 | PGD: 10 steps, α = 0.25·ε, seeded random start. | Standard Madry setting; random start and JPEG-quality draws are seeded per (region, area, ε, batch) and shared across the attack variants (plain / JPEG-aware / EOT), so those are paired. |

### Pending calibration (frozen after `scripts/calibrate_eps.py` and `scripts/watermark_area_strength_sweep.py` are run on the lab PC)

These are parameterised in `configs/eps_calibration.yaml` and `configs/watermark_calibration.yaml`;
nothing in the code hard-codes a final choice. Once chosen they move to the Decided table and
are **not** tuned differently for the co-located and separated arms.

| Topic | Candidates | How it is decided |
|---|---|---|
| Mask area (same for every masked arm) | 0.25, 0.35, 0.50 of the 784 blocks | Smallest area at which the watermark recovers cleanly *and* salient-region PGD has meaningful success, leaving unused blocks so "separated" is a real spatial manipulation. 0.50 only if smaller areas make the attack unusably weak. Global/global covers more area; label it, don't hide it. |
| ε set (3–4 values) | {1, 2, 3, 4, 6, 8}/255 | The values on the steep part of the attack-success curve after the real JPEG round trip, avoiding saturation (proposal §5). |
| Watermark strengths (2–3 values) | Finalists (area / strength): 0.25/0.08, 0.35/0.06, 0.35/0.08, 0.50/0.06 | 100-image rerun on the selected set with the corrected decoder and canonical processing. |
| JPEG-aware variant | fixed q75 vs EOT over [50, 95] | Whichever gives non-trivial post-JPEG success at the chosen ε. |
| Processing levels | JPEG q ∈ {90, 75, 60}; resize scale 0.5; blur σ = 1.0 | Fixed unless calibration shows a level is uninformative. |
| Joint score | success = misclassified AND BER ≤ threshold, per image | So paired tests work. Pareto curves remain the headline. |
