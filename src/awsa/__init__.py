"""awsa: Adversarial-Watermark Spatial Allocation.

Shared conventions (see docs/interfaces.md):
  * Images are float32 tensors in [0, 1], shape (B, 3, 224, 224), RGB.
  * ImageNet normalization happens ONLY inside the model wrapper (models.py).
  * Region masks are block-level bool tensors, shape (B, 1, 28, 28), one cell per 8x8 block.
"""

IMAGE_SIZE = 224
BLOCK = 8
GRID = IMAGE_SIZE // BLOCK  # 28

__version__ = "0.1.0"
