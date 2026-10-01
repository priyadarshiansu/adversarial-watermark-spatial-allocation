# Project Progress Update 01

## Project
**Spatial Allocation of Adversarial and Watermark Signals**

CAP5610 Final Project — Intro to Machine Learning

## What we have completed so far

### 1. Project repository and environment
- Created the public GitHub repository.
- Set up the project folder structure.
- Created a Python virtual environment.
- Installed the core ML packages.
- Confirmed the environment is running:
  - Python 3.12.10
  - PyTorch 2.14.1
  - Torchvision 0.29.1
- Current local development is CPU-only. GPU resources can be used later for larger experiments.

### 2. ResNet-50 baseline
We loaded a pretrained ResNet-50 model and tested it on a clean image.

For the test dog image, the clean model predicted:

**Samoyed — confidence 0.4366**

This establishes the clean baseline before adversarial perturbations are introduced.

### 3. FGSM sanity check
We implemented FGSM as the first simple adversarial attack.

We tested several perturbation budgets.

FGSM substantially reduced the model's confidence in the correct class, but did not reliably change the top-1 prediction for this particular image.

This was treated only as a sanity check rather than the final attack used in the project.

### 4. Pixel-space perturbation correction
We made sure that epsilon is measured in true image-pixel space rather than normalized ResNet input space.

This is important because values such as:

`epsilon = 2/255`

should correspond to the actual visual perturbation applied to the image.

### 5. PGD attack
We then implemented PGD, which will be the main adversarial attack for the project.

Using:

- epsilon = 2/255
- step size = 0.5/255
- 10 PGD steps

the prediction changed from:

**Samoyed — 0.4366**

to:

**West Highland white terrier — 0.5338**

The second-highest prediction was:

**Scotch terrier — 0.2609**

The adversarial image still appears almost identical to the original image to the human eye.

This confirms that the adversarial attack pipeline is working.

### 6. Perturbation visualization
We visualized:

1. the original image,
2. the PGD adversarial image,
3. the adversarial perturbation magnified 20×.

The perturbation is clearly visible when magnified, while the actual adversarial image remains visually very similar to the original.

## Current milestone

We have successfully demonstrated:

**Clean image → ResNet-50 → correct classification**

and

**Clean image → small PGD perturbation → incorrect classification**

This completes the first baseline component of the project.

## Next step

The next task is to implement **Grad-CAM**.

Grad-CAM will allow us to identify which spatial regions of the image ResNet considers most important.

We will then divide the image into high-saliency and low-saliency regions so that later experiments can control where:

- the adversarial perturbation is placed, and
- the invisible watermark is placed.

After the saliency pipeline works, we will build the block-DCT watermarking component.