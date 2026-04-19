# RETFound — Generalization Evaluation for Diabetic Retinopathy

> Cross-dataset evaluation of the ophthalmic foundation model **[RETFound](https://github.com/rmaphoh/RETFound_MAE)** across three diabetic retinopathy (DR) datasets: APTOS 2019, IDRiD, and MESSIDOR-2.

---

## Table of Contents

- [Overview](#overview)
- [DR Grading](#dr-grading)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Downloading Assets](#downloading-assets)
- [Usage](#usage)
- [Outputs](#outputs)
- [Technical Notes](#technical-notes)

---

## Overview

This project automates the evaluation of **cross-dataset generalization** of RETFound on the diabetic retinopathy classification task. Each checkpoint, fine-tuned on a source dataset, is evaluated against the two other target datasets — producing a 3×3 performance matrix.

The three datasets used:

| Dataset | Description | Link |
|---|---|---|
| **APTOS 2019** | Kaggle Blindness Detection Challenge 2019 | [kaggle.com](https://www.kaggle.com/c/aptos2019-blindness-detection) |
| **IDRiD** | Indian Diabetic Retinopathy Image Dataset | [ieee-dataport.org](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) |
| **MESSIDOR-2** | Extension of the MESSIDOR dataset (Télécom Bretagne) | [adcis.net](https://www.adcis.net/en/third-party/messidor2/) |

---

## DR Grading

All datasets are remapped to a shared 5-class grading convention:

| Grade | Clinical Meaning |
|:---:|---|
| 0 | No diabetic retinopathy |
| 1 | Mild diabetic retinopathy |
| 2 | Moderate diabetic retinopathy |
| 3 | Severe diabetic retinopathy |
| 4 | Proliferative diabetic retinopathy |

---

## Project Structure

```
.
├── dataset/
│   ├── APTOS2019/
│   ├── IDRiD_data/
│   └── MESSIDOR2/
├── check/
│   ├── checkpoint-best-APTOS2019.pth
│   ├── checkpoint-best-IDRID.pth
│   └── checkpoint-best-MESSIDOR2.pth
├── results/
│   └── dr_matrix/
│       └── summary_metrics.csv
├── scripts/
│   ├── download_assets.py
│   ├── inspect_dr_datasets.py
│   ├── evaluate_pair.py
│   ├── evaluate_matrix.py
│   ├── summarize_results.py
│   └── make_markdown_report.py
└── requirements.txt
```

> The `dataset/`, `check/`, and `results/` directories are tracked in Git via `.gitkeep` files; their contents are ignored.

---

## Installation

Install the required Python dependencies into your environment:

```bash
python3 -m pip install -r requirements.txt
```

---

## Downloading Assets

### Download everything (datasets + checkpoints)

```bash
python3 scripts/download_assets.py
```

### Datasets only

```bash
python3 scripts/download_assets.py --datasets
```

### Checkpoints only

```bash
python3 scripts/download_assets.py --checkpoints
```

ZIP archives are kept in `dataset/_downloads/` before extraction. Checkpoints are downloaded directly under their expected names.

### Download Links

| Type | File | Link |
|---|---|---|
| Dataset | `APTOS19.zip` | [Google Drive](https://drive.google.com/file/d/162YPf4OhMVxj9TrQH0GnJv0n7z7gJWpj/view) |
| Dataset | `MESSIDOR2.zip` | [Google Drive](https://drive.google.com/file/d/1vOLBUK9xdzNV8eVkRjVdNrRwhPfaOmda/view) |
| Dataset | `IDRiD_data.zip` | [Google Drive](https://drive.google.com/file/d/1c6zexA705z-ANEBNXJOBsk6uCvRnzmr3/view) |
| Checkpoint | `checkpoint-best-APTOS2019.pth` | [Google Drive](https://drive.google.com/file/d/1Ujzb6Xd1naWC0NngHah-DbHSgqxOiyJX/view) |
| Checkpoint | `checkpoint-best-MESSIDOR2.pth` | [Google Drive](https://drive.google.com/file/d/1EKJ1TraOkBpP6-XDNZUT5vNnABBEYgcx/view) |
| Checkpoint | `checkpoint-best-IDRID.pth` | [Google Drive](https://drive.google.com/file/d/1b0grTwARX1cXnYnMB3ZJZES26aMkgkvZ/view) |

---

## Usage

### Inspect datasets

```bash
python3 scripts/inspect_dr_datasets.py
```

### Evaluate a source → target pair

Example: checkpoint fine-tuned on MESSIDOR-2, evaluated on IDRiD:

```bash
python3 scripts/evaluate_pair.py \
  --train-dataset MESSIDOR2 \
  --eval-dataset IDRiD_data \
  --batch-size 16
```

### Run the full matrix (3×3)

Includes diagonal entries (in-distribution) as a reference baseline:

```bash
python3 scripts/evaluate_matrix.py --batch-size 16
```

External validations only (off-diagonal):

```bash
python3 scripts/evaluate_matrix.py --external-only --batch-size 16
```

### Summarize existing results

```bash
python3 scripts/summarize_results.py --results-dir results/dr_matrix
```

### Generate a Markdown report

```bash
python3 scripts/make_markdown_report.py \
  --summary results/dr_matrix/summary_metrics.csv \
  --output results/dr_matrix/report.md
```

---

## Outputs

For each evaluated pair, the following files are created under `results/train-{SOURCE}__eval-{TARGET}/`:

| File | Content |
|---|---|
| `predictions.csv` | Per-image predictions |
| `metrics.csv` | Global metrics |
| `metrics.json` | Metrics, per-class report, and confusion matrix |
| `confusion_matrix.png` | Confusion matrix visualization |
| `roc_curves.png` | Per-class ROC curves |

The global matrix summary is available at `results/dr_matrix/summary_metrics.csv`.

---

## Technical Notes

| Parameter | Value |
|---|---|
| **Model** | [ViT-L/16](https://arxiv.org/abs/2010.11929) with `global_pool=True`, RETFound-compatible |
| **Preprocessing** | Resize 224×224 → `ToTensor` → [ImageNet](https://image-net.org/) normalization |
| **Weight loading** | `model` key from `.pth` checkpoint files |
| **Device** | `--device auto`: [MPS](https://developer.apple.com/metal/pytorch/) (Apple Silicon) > [CUDA](https://developer.nvidia.com/cuda-toolkit) > CPU |
| **Splits** | `train` + `val` + `test` merged by default; restrict with `--splits test` |

---

## Reference

If you use RETFound in your work, please cite:

> Zhou, Y., et al. *A foundation model for generalizable disease detection from retinal images.* **Nature**, 2023. [doi:10.1038/s41586-023-06555-x](https://doi.org/10.1038/s41586-023-06555-x)
