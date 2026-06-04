# COVID-19 Chest X-Ray Classification & Explainability

A complete pipeline for training deep learning models on chest X-ray images (COVID-19, Normal, Viral Pneumonia), interpreting their predictions with saliency and attention methods, and rigorously evaluating those explanations with quantitative metrics.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Structure](#dataset-structure)
- [Environment Setup](#environment-setup)
- [File Structure](#file-structure)
- [Part 1 — Model Training](#part-1--model-training)
  - [Models](#models)
  - [Training Configuration](#training-configuration)
  - [Running Training](#running-training)
  - [Training Outputs](#training-outputs)
- [Part 2 — Explainability](#part-2--explainability)
  - [CNN Methods](#cnn-methods)
  - [ViT Methods](#vit-methods)
  - [Quantitative Metrics](#quantitative-metrics)
  - [Running Explainability](#running-explainability)
  - [Explainability Outputs](#explainability-outputs)
- [Bonus Task — Novel Evaluation Proposal](#bonus-task--novel-evaluation-proposal)
- [Results Summary](#results-summary)
- [GPU / CPU Behaviour](#gpu--cpu-behaviour)
- [Ouputs](#outputs)


---

## Project Overview

| Task | Description |
|---|---|
| Classification | 3-class chest X-ray classification: **COVID-19**, **Normal**, **Viral Pneumonia** |
| Models | ResNet-style CNN + Vision Transformer (ViT) |
| Explainability | GradCAM, Integrated Gradients, Gradient×Input (CNN) · Attention Rollout, Raw Attention, Integrated Gradients (ViT) |
| Evaluation | Insertion AUC, Deletion AUC, Shannon Entropy, AOPC |
| Bonus | Novel saliency evaluation proposal + experimental validation |

---

## Dataset Structure

Place your dataset at the project root with this exact layout:

```
COVID_19_dataset/
├── train/
│   ├── COVID/
│   ├── Normal/
│   └── Viral Pneumonia/
├── val/
│   ├── COVID/
│   ├── Normal/
│   └── Viral Pneumonia/
└── test/
    ├── COVID/
    ├── Normal/
    └── Viral Pneumonia/
```

---

## Environment Setup

```bash
# Python 3.9+ recommended (preferably 3.12)
pip install torch torchvision          # core deep learning
pip install timm                       # ViT utilities
pip install captum                     # attribution methods (IntGrad etc.)
pip install scikit-learn               # metrics
pip install matplotlib seaborn         # plotting
pip install Pillow numpy               # image handling
```

All dependencies are standard; no proprietary packages required. The code auto-selects CUDA or CPU — no manual changes needed.

---

## File Structure

```
.
├── main2.py                 # Part 1 — training pipeline (uses gpu)
├── main.py                  (Same results as main2- using cpu for training)
├── effectiveness.py         # Part 2 — XAI pipeline
├── README.md
├── .gitignore
├── COVID_19_dataset/        # dataset
└── outputs/                 # auto-created; all artefacts saved here
    ├── ResNet_CNN_weights.pth
    ├── ViT_weights.pth
    ├── training_curves.png
    ├── confusion_matrices.png
    ├── metrics_comparison.png
    ├── per_class_f1.png
    ├── results_summary.json
    └── xai/
        ├── cnn_qualitative.png
        ├── vit_qualitative.png
        ├── insertion_deletion_curves.png
        ├── aopc_curves.png
        ├── entropy_comparison.png
        ├── summary_bar.png
        ├── summary_radar.png
        ├── xai_summary.json
        └── bonus/
           ├── all_metrics_heatmap.png
           ├── cwsc_ssi_comparison.png
           ├── cwsc_stability_curves.png
           ├── ggaf_qualitative.png
           ├── metric_correlation.png
           ├── statistical_significance.png
           ├── statistical_validation.json
           └── bonus_summary.json

```

---

## Part 1 — Model Training

### Models

**ResNet-style CNN** (`ResNetCNN`)
- 4 residual stages (32 → 64 → 128 → 256 → 512 channels)
- Each stage: 2 residual blocks with skip connections
- Global average pooling + dropout-regularised MLP head
- ~700K trainable parameters

**Vision Transformer** (`ViT`)
- Patch embedding (8×8 patches on 64×64 images → 64 tokens)
- Learnable CLS token + positional embeddings
- 4 Transformer encoder blocks (multi-head self-attention + MLP)
- Classification head on CLS token output
- ~560K trainable parameters

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 224×224 (64×64 demo) |
| Batch size | 64 (GPU) / 32 (CPU) |
| Epochs | 20 |
| Optimiser | AdamW |
| Learning rate | 3e-4 with Cosine Annealing |
| Weight decay | 1e-4 |
| Loss | CrossEntropyLoss + label smoothing 0.1 |
| Augmentation | RandomFlip, Rotation ±10°, ColorJitter |

### Running Training

```bash
python main2.py
```

The script will:
1. Detect GPU/CPU automatically and print the active configuration
2. Load the dataset
3. Train both models, printing epoch-level metrics
4. Save best-val-accuracy weights to `outputs/`
5. Generate training curves, confusion matrices, and metric comparison plots

### Training Outputs

| File | Description |
|---|---|
| `ResNet_CNN_weights.pth` | Best ResNet-CNN checkpoint |
| `ViT_weights.pth` | Best ViT checkpoint |
| `training_curves.png` | Train/val loss and accuracy over epochs |
| `confusion_matrices.png` | Per-class normalised confusion matrices |
| `metrics_comparison.png` | Accuracy, Precision, Recall, F1 bar chart |
| `per_class_f1.png` | Per-class F1 for each model |
| `results_summary.json` | All metrics in machine-readable form |

---

## Part 2 — Explainability

Explainability runs **after** training — it loads the saved `.pth` weights automatically.

### CNN Methods

| Method | How it works |
|---|---|
| **GradCAM** | Hooks into the last conv layer; weights feature maps by their gradient w.r.t. the target class score, then upsamples to image resolution |
| **Integrated Gradients** | Approximates the Aumann–Shapley attribution by integrating gradients along a straight path from a zero baseline to the input (50 steps, via Captum) |
| **Gradient × Input** | Element-wise product of input activations and input gradients — a fast, lightweight saliency baseline |

### ViT Methods

| Method | How it works |
|---|---|
| **Attention Rollout** | Multiplies attention matrices across all transformer layers, adding the identity at each step to model the residual stream; extracts CLS-row spatial importance |
| **Raw Attention (last layer)** | Reads the CLS-token attention directly from the final transformer block only — faster, but ignores information flow across layers |
| **Integrated Gradients** | Same Captum implementation as CNN; model-agnostic, works on ViT's input pixel space |

### Quantitative Metrics

**Insertion AUC**
Pixels are *revealed* from most to least important (MoRF order) on a mean-baseline image. Model confidence is recorded at each step. AUC of this curve measures how quickly confidence recovers — **higher is better**.

**Deletion AUC**
Pixels are *removed* from most to least important on the original image. Model confidence is recorded at each step. AUC measures how much confidence remains when key pixels are gone — **lower is better** (good saliency removes what matters most).

**Shannon Entropy**
$$H = -\sum_i p_i \log p_i \quad \text{where } p_i = \frac{s_i}{\sum_j s_j}$$
Measures concentration of the saliency distribution. A focused, peaked map has **lower entropy**; a diffuse, noisy map has higher entropy.

**AOPC (Area Over Perturbation Curve)**
$$\text{AOPC} = \frac{1}{K} \sum_{k=1}^{K} \left[ f(x) - f(x_{\backslash \text{top-}k}) \right]$$
Averages the drop in model confidence as the top-$k$ most important pixels are progressively removed. **Higher AOPC = saliency correctly identifies decisive pixels**.

### Running Explainability

```bash
python effectiveness.py
```

Requires `outputs/ResNet_CNN_weights.pth` and `outputs/ViT_weights.pth` to exist (run training first). Falls back to random-init models if weights are missing.

### Explainability Outputs

| File | Description |
|---|---|
| `cnn_qualitative.png` | Grid: Original · GradCAM · IntGrad · Grad×Input for each test sample |
| `vit_qualitative.png` | Grid: Original · Rollout · Raw Attention · IntGrad for each test sample |
| `insertion_deletion_curves.png` | Average insertion & deletion curves per method per model |
| `aopc_curves.png` | Running-mean AOPC perturbation curves |
| `entropy_comparison.png` | Box-and-strip plot of entropy distribution per method |
| `summary_bar.png` | Grouped bar chart: all 4 metrics × all methods × both models |
| `summary_radar.png` | Radar chart of normalised scores across all 4 axes |
| `xai_summary.json` | All quantitative results in machine-readable form |

---

## Bonus Task 

### Identified Limitations of Current Methods

Current saliency evaluation metrics each carry a specific blind spot:

| Limitation | Affected Metrics |
|---|---|
| **Baseline sensitivity** — results change significantly depending on the choice of uninformative baseline (zero, blurred, mean) | Insertion, Deletion, IntGrad |
| **Pixel independence assumption** — perturbing individual pixels ignores spatial correlations; models are never trained on such inputs (out-of-distribution perturbations) | Insertion, Deletion, AOPC |
| **No clinical correspondence** — no existing metric checks whether highlighted regions match medically meaningful anatomy (e.g. lung fields, consolidations) | All standard metrics |
| **Entropy is marginalisation-blind** — a map with the correct spatial distribution but wrong magnitude still gets the same entropy as a well-calibrated map | Entropy |

### Proposed Novel Metric: Anatomy-Constrained Faithfulness Score (ACFS)

**Core idea:** Rather than perturbing arbitrary pixels or measuring focus in isolation, ACFS evaluates whether the saliency map's *peak regions* spatially align with medically plausible anatomy — combined with a faithfulness score that uses clinically meaningful, structure-preserving perturbations instead of pixel-wise masking.

**Algorithm:**

1. Obtain a lung segmentation mask $M$ (U-Net or threshold-based) for each X-ray
2. Compute the saliency-inside-lung ratio:
   $$\text{Anatomy Score} = \frac{\sum_{i \in M} s_i}{\sum_i s_i}$$
3. Perturb the image by blurring *entire anatomical regions* (lobes, mediastinum) rather than pixels, preserving natural image statistics
4. Measure confidence drop per region, weighted by that region's saliency mass
5. ACFS = Anatomy Score × Region-weighted Faithfulness

**Why this is better:**
- Perturbations are in-distribution (blurred regions look like real X-rays)
- Clinical alignment is directly measured
- Aggregates spatial structure rather than individual pixels

**Experimental validation plan (on COVID-19 dataset):**
- Apply threshold-based lung segmentation on the test set
- Compute ACFS for all 6 methods (GradCAM, IntGrad, Grad×Input, Rollout, Raw Attention, ViT-IntGrad)
- Compare ranking with standard Insertion/Deletion/AOPC rankings
- Report Spearman correlation between ACFS ranking and radiologist preference (if annotations available)

> The full implementation of ACFS validation is structured in `effectiveness.py` and can be activated by setting `RUN_BONUS = True` at the top of the file once lung segmentation masks are available.

---

## Results Summary

Results below are from the dataset tested on locally, and ouput images are in drive link under Ouputs header. 

### Classification (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| ResNet-CNN | 0.9655 | 0.9657 | 0.9655 | 0.9654 |
| ViT | 0.8160 | 0.8232 | 0.8160 | 0.8094 |


### Explainability (Qualitative Interpretation)

| Method | Model | Insertion AUC ↑ | Deletion AUC ↓ | AOPC ↑ | Entropy ↓ |
|---|---|---|---|---|---|
| GradCAM | ResNet-CNN | 0.6706 | 0.6039 | 0.3068 | 10.1621 |
| IntGrad | ResNet-CNN | 0.5114 | 0.3979 | 0.5127 | 10.2912 |
| Grad×Input | ResNet-CNN | 0.4995 | 0.4736 | 0.4370 | 10.2327 |
| Rollout | ViT | 0.5896 | 0.4612 | 0.2529 | 10.511 |
| Raw Attention | ViT | 0.5334 | 0.6397 | 0.0742 | 10.6061 |
| IntGrad | ViT | 0.7115 | 0.4614 | 0.2527 | 10.1869 |


---

## GPU / CPU Behaviour

The code auto-detects hardware at startup and prints a configuration summary:

```
[Device] GPU detected: NVIDIA A100  (1 GPU(s))
         Mixed-precision AMP : ENABLED
         DataParallel        : NO (single GPU)
         num_workers         : 4
         pin_memory          : True
```

| Feature | GPU (CUDA) | CPU |
|---|---|---|
| Mixed precision (AMP) | ✅ Enabled | ❌ Disabled (no-op) |
| `pin_memory` | ✅ Yes | ❌ No |
| `non_blocking` transfers | ✅ Yes | ✅ Yes (harmless) |
| `cudnn.benchmark` | ✅ Yes | ❌ Skipped |
| DataParallel | ✅ If N_GPU > 1 | ❌ N/A |
| Batch size | 64 | 32 |

No code changes needed when switching between environments.

---

## Outputs

drive link: https://drive.google.com/drive/folders/1zM0tqPmL-ynHb5modMNqHGDck4csSxnu?usp=sharing

Outputs in drive:
| `training_curves.png` 
| `confusion_matrices.pngion result
| `metrics_comparison.png at a glance 
| `per_class_f1.png` 
| `cnn_qualitative.png`
| `vit_qualitative.png`
| `insertion_deletion_curitative XAI result 
| `aopc_curves.png` 
| `entropy_comparison.pngc 
| `summary_bar.png`
| `summary_radar.png` 
| `xai_summary.json` 
| `ResNet_CNN_weights.pth`
| `ViT_weights.pth` 


---