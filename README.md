# adversarial-watermark-spatial-allocation
CAP5610 final project on spatial allocation of adversarial perturbations and invisible watermarks.
# Spatial Allocation of Adversarial and Watermark Signals

CAP5610 Final Project — Intro to Machine Learning

## Project Overview

This project studies how two invisible image modifications interact:

1. An adversarial perturbation designed to fool an image classifier.
2. An invisible watermark designed to carry recoverable ownership information.

The main question is whether these two signals work better when they are placed in the same image regions or in separate regions.

We will compare co-located, saliency-aware separated, and randomly separated placements while keeping perceptual distortion controlled.

We will also test whether the results remain stable after common image-processing operations such as:

- JPEG compression
- Image resizing
- Gaussian blur

## Main Research Questions

- Does spatial separation reduce interference between adversarial perturbations and invisible watermarks?
- Does any advantage of separation survive image processing?
- Does saliency-aware placement perform differently from random separation?

## Planned Experimental Setup

- Primary classifier: ResNet-50
- Adversarial attack: PGD
- Saliency method: Grad-CAM
- Watermark: block-DCT invisible watermark
- Dataset: ImageNet validation subset
- Evaluation:
  - Attack success rate
  - Watermark bit error rate
  - PSNR
  - SSIM
  - LPIPS

## Repository Structure

- `data/` — dataset files or download instructions
- `notebooks/` — experimental notebooks
- `src/` — reusable Python code
- `results/` — numerical experiment outputs
- `figures/` — generated plots and visualizations
- `papers/` — literature notes and reference material
- `docs/` — project documentation and reports

## Status

Project setup and literature review in progress.
