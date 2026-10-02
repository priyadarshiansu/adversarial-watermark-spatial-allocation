# Project Brief (v2): Spatial Allocation of Adversarial and Watermark Signals

*Prepared for the project team · CAP5610 final project*

> **Repo note:** this is the proposal as submitted to the professor. Later decisions that refine it (e.g. dataset: NeurIPS 2017 adversarial dev set instead of an ImageNet-val subset) are recorded in [`interfaces.md`](interfaces.md#decisions-log).

## 1. Problem statement

Existing dual-protection methods show that an adversarial perturbation and an invisible watermark interfere when combined, but none holds each signal's perceptual budget fixed while varying where each is placed, so it is unknown whether the two should share image regions or occupy separate ones. We test co-located, saliency-aware separated, and randomly separated placements at matched distortion, asking whether any benefit of separation comes from separation itself or from saliency-aware placement, and whether it survives JPEG compression, resizing, and blur, which spread each signal across region boundaries.

**Motivation (threat model).** An image owner wants two protections at once. The first is a Level-1 adversarial perturbation that prevents automated scrapers from classifying or indexing the image. The second is a Level-2 invisible watermark that carries a recoverable ownership payload if the image is taken anyway. Both must survive the re-encoding that images undergo on social platforms, which is dominated by JPEG.

## 2. State of the art

Joint adversarial-plus-watermark generation is well occupied, and existing methods fall into three families:

- Fused-signal methods, where the watermark itself acts as the perturbation: Adv-watermark (Jia et al., 2020), BHI (2023), ISWP (2025), and IRAW (2026). In IRAW, the invisible watermark is designed to function as the adversarial perturbation, optimized with basin hopping inside an evolutionary loop.
- Jointly trained methods: IAW (ACM TOMM 2024), CCIW (Entropy 2025), traceable adversarial examples (IEEE TCSVT 2024), and AdvEWM (2024).
- Saliency-guided placement of the combined signal: the 2026 Imaging Science Journal DCT method, in which modifications are targeted using saliency maps while tightly limiting how many coefficients change.

Interference is already documented. IAW's stated motivation is that simply superimposing watermarks and adversarial perturbations can cause mutual interference and hurt overall performance. IAP (arXiv 2303.11625) measured naive MBRS + I-FGSM combination in both orders at matched PSNR (about 36) and found that direct combination failed to recover the watermark correctly and slightly lowered attack success.

Placement matters for each signal separately. For attacks, FAPA showed that concentrating perturbation in foreground regions raises attack success and transferability. For the pair of signals together, Zhu et al. (CVPR 2024) weight the perturbation loss more heavily in the watermark region so the perturbation there stays small. That is a spatial trade-off chosen inside one method, not a controlled comparison.

Robustness to JPEG, resizing, and blur is extensively characterized on both sides (for example, MBRS on the watermark side).

## 3. Research gap and contribution

Prior work either fuses the two signals into one, trains them jointly end-to-end, or places the combined signal by saliency. Naive superposition is known to interfere. No work holds per-signal perceptual budgets fixed while varying the spatial relationship between two separable signals, so none can say whether co-location or separation is preferable, whether the answer holds after real-world processing, or why.

Our contribution is a matched-budget, region-controlled comparison of co-located and spatially separated two-signal designs under realistic processing, with a block-level measurement that explains the result. This is a characterization contribution, not a new method. A well-controlled null result ("placement does not matter after JPEG") is a legitimate and reportable outcome.

**Scope note on the trivial case.** On unprocessed images, disjoint placement prevents pixel-level overlap by construction, so separation is expected to win there. That result is a check that the setup works, not a finding. The research question lives in what happens after processing.

## 4. Research questions and hypotheses

### Core research questions

- RQ1: At matched perceptual distortion, does placing the perturbation and watermark in separate regions give a better joint result (attack success × watermark accuracy) than placing them together?
- RQ2: Does any advantage of separation survive JPEG compression, resizing, and blur?
- RQ3: Does the advantage come from separation itself or from saliency-aware placement?

### Baseline checks (these validate the setup; they are not contributions)

- B1: Perturbation in high-saliency regions beats equal-budget perturbation in low-saliency regions (confirms FAPA).
- B2: Co-located signals interfere measurably (confirms IAP and IAW).
- Optional: RQ-X asks whether the RQ1 result generalizes from ResNet-50 to ViT-B/16. Run it only if the core result is stable by Week 7.

### Hypotheses

- H1 (RQ1): Separation improves the joint trade-off over co-location at matched distortion.
- H2 (RQ2): The advantage shrinks after processing, most sharply under JPEG, because both signals occupy DCT coefficients that JPEG quantizes, and because blur and resampling carry energy across region boundaries.
- H3 (RQ3): Saliency-aware separation outperforms random separation, meaning placement matters beyond separation alone.

## 5. Experimental design

**Conditions.** The image set, classifier, attack parameters, payload length, and per-signal budgets are held constant across all conditions:

- Adversarial only
- Watermark only
- Global/global (naive combined baseline)
- Co-located (both signals in the same salient region)
- Saliency-aware separated (perturbation in the salient region, watermark in a disjoint low-saliency region)
- Random separated (equal-area, non-overlapping regions placed at random)

**Sweeps.**

- Perturbation budget ε: 3–4 values chosen on the steep part of the attack-success curve, avoiding saturation.
- Watermark strength: 2–3 values.
- Application order: watermark then attack, and attack then watermark.
- Processing: none, JPEG at several quality levels, resize, and Gaussian blur.

**Attack.** PGD with a JPEG-aware loss, using either a differentiable JPEG approximation or EOT over JPEG inside the loop, so that post-compression results are not trivially zero. Every attack is validated through the identical quantize-to-uint8, then JPEG, round trip that the watermark decoder sees.

**Metrics.**

- Attack success rate and confidence drop.
- Watermark bit error rate / bit accuracy.
- Perceptual fidelity: PSNR, SSIM, and LPIPS. No conclusion may rest on LPIPS alone.

**Mechanism analysis (the "why").** For each 8×8 block, measure the fraction of perturbation energy that falls in the watermark's embedding coefficients, and correlate it with per-block bit errors. This turns the Pareto result into an explanation.

**Headline result.** A Pareto characterization of attack success against watermark accuracy, with every point annotated with its measured LPIPS. It is reported per processing condition and per application order.

**Statistics.** Every image appears in every arm, so arms are compared with paired tests or bootstrap confidence intervals across images. Target 500–1,000 ImageNet validation images that are correctly classified when clean.

## 6. Non-negotiable design controls

- Matched perceptual budget. Equal mask area does not mean equal perceptual distortion. Hold per-signal budgets fixed and annotate every Pareto point with measured LPIPS, PSNR, and SSIM, so that no effect can be attributed to one arm altering the image more.
- The random-separated control. It is what answers RQ3. Without it, a separation advantage cannot be interpreted.
- A JPEG-robust attack. Without it, the answer to RQ2 collapses to zeros in every arm.

## 7. Risk register and contingencies

- Novelty depends on one unread paper. The closest competitor is now the Imaging Science Journal DCT method, which uses saliency-guided placement. It shares an author with IRAW, a group that is active in this area. In Week 1, one member reads its full text and checks whether it contains a separated-regions ablation. IRAW is lower risk because it uses a single fused signal and so has no allocation to vary. If the gap collapses, reframe around RQ2 and the mechanism analysis. Operation order is not a viable fallback, because IAP already covers it at matched PSNR.
- The effect may be small or null. At Decision Gate (end of Week 4), if co-location and separation show no measurable difference after processing, report the null, lead with the mechanism analysis, and state the practical implication: stacking the signals carries no penalty.
- Order effects may dominate allocation effects. Report both orders throughout. Do not average across them.
- Confound: low-saliency regions tend to be smooth. Watermarks are more perceptible in flat regions, so the separated arm may pay a hidden imperceptibility cost. Measure per-block local variance to confirm that low-saliency masks are not systematically smoother.
- The JPEG-robust attack may slip. This is the highest schedule risk. If it isn't working by the end of Week 4, fall back to reporting processing results only at high JPEG quality (90–95) and state this as a limitation.
- Integration bugs. Float-to-uint8 quantization can silently erase perturbations smaller than 1/255, and JPEG's 8×8 grid must align with the mask grid. These consume schedule, not compute.

## 8. Scope boundaries

- No training of a watermark network or classifier from scratch.
- PGD only, with an optional MI-FGSM extension if transferability becomes a question.
- ResNet-50 is the primary model; ViT-B/16 is late validation only.
- Three processing families: JPEG, resize, and blur.
- Findings apply to a handcrafted block-DCT watermark and gradient-based attacks. Transfer to jointly trained systems is stated as a limitation, not claimed.

## 9. Work allocation and timeline (8 weeks)

**Modules (these run in parallel from Week 1):**

- Module 1: Clean-classification pipeline, image selection, and metrics code.
- Module 2: Adversarial attack. Start with an FGSM sanity check, then PGD, then JPEG-aware PGD. Calibrate ε below saturation.
- Module 3: Spatial masks. Generate Grad-CAM maps, then equal-area 8×8 block-level saliency masks, then random-disjoint masks.
- Module 4: Block-DCT watermark with a 32-bit payload and no error correction initially. Record which coefficients carry the payload, for use in the mechanism analysis.

### Timeline

| Week | Work |
|---|---|
| 1–2 | Modules built and tested independently. Full-text reads of the Imaging Science Journal paper and IRAW. |
| 3 | JPEG-aware PGD. Round-trip validation. |
| 4 | Integrate all six conditions. Decision Gate. |
| 5–6 | Full sweeps: ε, watermark strength, both orders, all processing conditions. Mechanism analysis. |
| 7 | Statistics, Pareto plots, smoothness-confound check. ViT-B/16 only if on schedule. |
| 8 | Write-up. No new experiments. |

If the schedule slips, cut in this order: ViT validation first, then resize and blur, then the ε sweep (reduce to 2–3 points). Never cut the random-separated control or the JPEG-aware attack.

## Pre-reading (annotated)

### Tier 1: Foundations (everyone, before Week 1)

- Goodfellow, Shlens & Szegedy (2015), Explaining and Harnessing Adversarial Examples, ICLR (arXiv:1412.6572). FGSM and the adversarial-example framing.
- Madry et al. (2018), Towards Deep Learning Models Resistant to Adversarial Attacks, ICLR (arXiv:1706.06083). PGD, our attack.
- Selvaraju et al. (2017), Grad-CAM, ICCV (arXiv:1610.02391). Our saliency mechanism.

### Tier 2: Watermarking and robustness (Modules 2–4)

- Zhu, Kaplan, Johnson & Fei-Fei (2018), HiDDeN, ECCV (arXiv:1807.09937). The encoder–noise–decoder pattern.
- Jia, Fang, Zhang et al. (2021), MBRS, ACM MM (arXiv:2108.08211). JPEG robustness via mixed real and simulated compression.

### Tier 3: Competing and closest work (read closely; novelty depends on these)

- Read first: JPEG compression-resistant adversarial attack with invisible watermark embedding (2026), The Imaging Science Journal. This is the closest competitor. Check one thing: does its saliency-guided placement include any comparison of co-located against separated signals?
- IAP: Zhu et al. (2023), Information-containing Adversarial Perturbation for Combating Facial Manipulation Systems (arXiv:2303.11625). Measured naive combination in both orders at matched PSNR. This pre-empts B2 and the operation-order fallback.
- IAW: Wang et al. (2024), Invisible Adversarial Watermarking: A Novel Security Mechanism for Enhancing Copyright Protection, ACM TOMM. States interference as its motivation.
- IRAW (2026), Neural Networks (S0893608026004636). A single fused signal optimized by basin hopping. Differentiation: IRAW fuses the signals, while we keep them separable and vary their placement.
- CCIW: Li, Wang, Li & Ren (2025), Cover-Concealed Image Watermarking for Dual Protection of Privacy and Copyright, Entropy. Verified; replaces the unverified entry in v1. Uses channel attention, not spatial allocation.
- Li et al. (2024), Dual Protection for Image Privacy and Copyright via Traceable Adversarial Examples, IEEE TCSVT.
- Zhu, Takahashi & Kataoka (2024), Watermark-embedded Adversarial Examples for Copyright Protection against Diffusion Models, CVPR (arXiv:2404.09401). Its region-weighted perturbation loss is the nearest precedent for placement between the two signals.
- Yang et al. (2022), FAPA, Security and Communication Networks. The reason B1 is confirmatory.
- AdvEWM (2024), Journal of Information Security and Applications. The watermark-as-attack combination.
- Lineage (skim): Adv-watermark (Jia et al., ACM MM 2020), BHI (2023), and ISWP (Applied Intelligence, 2025). The line of work IRAW extends.
- Survey (skim its dual-protection section first): Adversarial Attacks for Good: A Survey of Proactive Protection across the Visual Content Lifecycle (arXiv:2608.04314, 2026). The fastest way to check for anything missed.

### Tier 4: Evaluation methodology (Module 1 owner and whoever writes the metrics code)

- Wang, Bovik, Sheikh & Simoncelli (2004), SSIM, IEEE TIP.
- Zhang et al. (2018), LPIPS, CVPR (arXiv:1801.03924).

### Optional

- Dong et al. (2018), MI-FGSM, CVPR (arXiv:1710.06081). Only if transferability becomes an extension.
- EOT: Athalye et al. (2018), Synthesizing Robust Adversarial Examples, ICML (arXiv:1707.07397). For the Module 2 owner implementing the JPEG-aware attack.

### Changes from v1

- New problem statement with a threat model.
- Research questions reduced to three core questions (together or apart, survives processing, separation or placement), with FAPA and the interference result moved to baseline checks.
- JPEG-aware PGD added as a non-negotiable control.
- Block-level mechanism analysis added to deliver the "why."
- ε and watermark-strength sweeps and paired statistics added.
- Novelty risk moved from IRAW to the Imaging Science Journal paper.
- Operation-order fallback removed.
- IAP, IAW, and the traceable-AE paper added; CCIW verified.
- Explicit 8-week timeline with a cut order.
