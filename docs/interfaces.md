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
load_resnet50(device) -> nn.Module        # frozen, takes [0,1] pixels
attack_success(logits, y) -> (B,)         # untargeted: argmax != y
bit_error_rate(bits_true, bits_pred) -> (B,)
psnr(x, x_ref) / ssim(x, x_ref) / lpips_distance(x, x_ref) -> (B,)
```

### Module 2: attacks (`attacks.py`)
```python
pgd(model, x, y, eps, alpha, steps, mask=None, random_start=False,
    transform=None, eot_samples=1, generator=None) -> x_adv
```
Guarantees: `|x_adv - x|_inf <= eps`, `x_adv` in `[0, 1]`, `x_adv == x` where `mask == 0`.
The JPEG-aware attack passes a differentiable JPEG (e.g. `kornia`) as `transform`.

### Module 3: saliency and masks (`saliency.py`, `masks.py`)
```python
gradcam(model, x, y) -> (B, 1, 28, 28) float in [0, 1]      # target layer: layer4
top_fraction_mask(saliency_blocks, area) -> (B, 1, 28, 28) bool
random_disjoint_masks(shape, area, generator) -> (mask_a, mask_b)
blocks_to_pixels(block_mask) -> (B, 1, 224, 224) float
```
Every masked arm covers exactly `round(area * 784)` blocks.

### Module 4: watermark (`watermark.py`)
```python
embed(x, bits, mask, strength, key) -> x_wm
extract(x, mask, key) -> bits
payload_coefficients() -> list[(u, v)]   # for the mechanism analysis
```

### Shared processing (`distortions.py`)
```python
roundtrip(x, jpeg_quality=None)   # uint8 quantization, then optional real JPEG
jpeg(x, quality) / resize_roundtrip(x, scale) / gaussian_blur(x, sigma)
```
**Every** reported number (attack success *and* bit error rate) goes through `roundtrip`
or one of these functions, so all arms see identical processing (proposal §5, "Attack").

## Decisions log

### Decided

| Date | Decision | Why |
|---|---|---|
| 2026-10-02 | Dataset: NeurIPS 2017 adversarial dev set (1,000 ImageNet-compatible images, 299×299). Images are not committed; see `data/README.md`. | Standard attack benchmark, ImageNet labels, free. CIFAR/MNIST break Grad-CAM, 8×8 blocks, JPEG and LPIPS at 32×32. |
| 2026-10-02 | Preprocess once: resize 299 → 224 directly (bicubic, antialiased), no crop; save PNG to `data/processed/`. | Images are already square, so cropping would only discard content. |
| 2026-10-02 | Labels: `label = TrueLabel - 1`. | CSV is 1-indexed, torchvision is 0-indexed. |
| 2026-10-02 | Dependencies managed with uv (`pyproject.toml` + `uv.lock`). | Faster, reproducible lockfile. |
| 2026-10-02 | PyTorch from the CUDA 13.2 index (`cu132`) on Windows/Linux; PyPI on macOS. CPU fallback is automatic. | PyPI's Windows wheel is CPU-only; the team has local NVIDIA GPUs (RTX 4070). |

### Proposed (team to confirm)

| Topic | Proposal |
|---|---|
| Decoder knowledge | The region mask (or random seed) is a secret key shared with the owner/decoder: non-blind decoding with side information. Grad-CAM cannot be recomputed after attack + JPEG. State this in the threat model. |
| Watermark colour space | Embed in luma (Y) only. JPEG subsamples chroma 4:2:0. Mechanism analysis measures perturbation energy in the Y-channel DCT coefficients. |
| Resize processing | Downscale then upscale back to 224 (`resize_roundtrip`), so the 8×8 grid and the mask stay valid. |
| Budget units | Attack: L∞ ε. Watermark: embedding step size (e.g. QIM step on a mid-frequency coefficient pair). Same mask area for every masked arm. Global/global covers more area; label it, don't hide it. |
| Mask area | 50% of blocks (392 of 784) for every masked arm, so ~12 blocks per payload bit. |
| ε calibration | Choose ε on the adversarial-only arm at the target JPEG quality, freeze it, then run the comparisons. |
| Joint score | Per image: success = misclassified AND BER ≤ threshold, so paired tests work. Pareto curves remain the headline. |
| JPEG-aware attack | Try `kornia`'s differentiable JPEG first; EOT over real JPEG is the fallback. |
