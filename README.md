# RETFound — Évaluation de généralisation sur la rétinopathie diabétique

> Évaluation croisée du modèle de fondation ophtalmologique **[RETFound](https://github.com/rmaphoh/RETFound_MAE)** sur trois datasets de rétinopathie diabétique (DR) : APTOS 2019, IDRiD et MESSIDOR-2.

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Grades DR](#grades-dr)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Téléchargement des données](#téléchargement-des-données)
- [Utilisation](#utilisation)
- [Sorties](#sorties)
- [Notes techniques](#notes-techniques)

---

## Vue d'ensemble

Ce projet automatise l'évaluation de la **généralisation cross-dataset** de RETFound sur la tâche de classification de la rétinopathie diabétique. Chaque checkpoint, fine-tuné sur un dataset source, est évalué sur les deux autres datasets cibles — produisant une matrice 3×3 de performances.

Les trois datasets utilisés :

| Dataset | Description | Lien |
|---|---|---|
| **APTOS 2019** | Kaggle Blindness Detection Challenge 2019 | [kaggle.com](https://www.kaggle.com/c/aptos2019-blindness-detection) |
| **IDRiD** | Indian Diabetic Retinopathy Image Dataset | [ieee-dataport.org](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) |
| **MESSIDOR-2** | Extension du dataset MESSIDOR (Télécom Bretagne) | [adcis.net](https://www.adcis.net/en/third-party/messidor2/) |

---

## Grades DR

Tous les datasets sont remappés vers une convention commune à 5 classes :

| Grade | Signification clinique |
|:---:|---|
| 0 | Pas de rétinopathie diabétique |
| 1 | Rétinopathie diabétique légère |
| 2 | Rétinopathie diabétique modérée |
| 3 | Rétinopathie diabétique sévère |
| 4 | Rétinopathie diabétique proliférante |

---

## Structure du projet

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

> Les dossiers `dataset/`, `check/` et `results/` sont versionnés avec des fichiers `.gitkeep` ; leur contenu est ignoré par Git.

---

## Installation

Installez les dépendances Python dans votre environnement :

```bash
python3 -m pip install -r requirements.txt
```

---

## Téléchargement des données

### Tout télécharger (datasets + checkpoints)

```bash
python3 scripts/download_assets.py
```

### Datasets uniquement

```bash
python3 scripts/download_assets.py --datasets
```

### Checkpoints uniquement

```bash
python3 scripts/download_assets.py --checkpoints
```

Les archives ZIP sont conservées dans `dataset/_downloads/` avant extraction. Les checkpoints sont téléchargés directement sous leurs noms attendus.

### Liens de téléchargement

| Type | Fichier | Lien |
|---|---|---|
| Dataset | `APTOS19.zip` | [Google Drive](https://drive.google.com/file/d/162YPf4OhMVxj9TrQH0GnJv0n7z7gJWpj/view) |
| Dataset | `MESSIDOR2.zip` | [Google Drive](https://drive.google.com/file/d/1vOLBUK9xdzNV8eVkRjVdNrRwhPfaOmda/view) |
| Dataset | `IDRiD_data.zip` | [Google Drive](https://drive.google.com/file/d/1c6zexA705z-ANEBNXJOBsk6uCvRnzmr3/view) |
| Checkpoint | `checkpoint-best-APTOS2019.pth` | [Google Drive](https://drive.google.com/file/d/1Ujzb6Xd1naWC0NngHah-DbHSgqxOiyJX/view) |
| Checkpoint | `checkpoint-best-MESSIDOR2.pth` | [Google Drive](https://drive.google.com/file/d/1EKJ1TraOkBpP6-XDNZUT5vNnABBEYgcx/view) |
| Checkpoint | `checkpoint-best-IDRID.pth` | [Google Drive](https://drive.google.com/file/d/1b0grTwARX1cXnYnMB3ZJZES26aMkgkvZ/view) |

---

## Utilisation

### Inspecter les datasets

```bash
python3 scripts/inspect_dr_datasets.py
```

### Évaluer une paire source → cible

Exemple : checkpoint fine-tuné sur MESSIDOR-2, évalué sur IDRiD :

```bash
python3 scripts/evaluate_pair.py \
  --train-dataset MESSIDOR2 \
  --eval-dataset IDRiD_data \
  --batch-size 16
```

### Lancer toute la matrice (3×3)

Inclut les diagonales (validation interne) comme référence :

```bash
python3 scripts/evaluate_matrix.py --batch-size 16
```

Validations externes uniquement (hors diagonale) :

```bash
python3 scripts/evaluate_matrix.py --external-only --batch-size 16
```

### Résumer des résultats existants

```bash
python3 scripts/summarize_results.py --results-dir results/dr_matrix
```

### Générer un rapport Markdown

```bash
python3 scripts/make_markdown_report.py \
  --summary results/dr_matrix/summary_metrics.csv \
  --output results/dr_matrix/report.md
```

---

## Sorties

Pour chaque paire évaluée, les fichiers suivants sont créés dans `results/train-{SOURCE}__eval-{TARGET}/` :

| Fichier | Contenu |
|---|---|
| `predictions.csv` | Prédictions image par image |
| `metrics.csv` | Métriques globales |
| `metrics.json` | Métriques, rapport par classe, matrice de confusion |
| `confusion_matrix.png` | Visualisation de la matrice de confusion |
| `roc_curves.png` | Courbes ROC par classe |

Le résumé global de la matrice est disponible dans `results/dr_matrix/summary_metrics.csv`.

---

## Notes techniques

| Paramètre | Valeur |
|---|---|
| **Modèle** | [ViT-L/16](https://arxiv.org/abs/2010.11929) avec `global_pool=True`, compatible RETFound |
| **Prétraitement** | Resize 224×224 → `ToTensor` → normalisation [ImageNet](https://image-net.org/) |
| **Chargement des poids** | Clé `model` dans les fichiers `.pth` |
| **Device** | `--device auto` : [MPS](https://developer.apple.com/metal/pytorch/) (Apple Silicon) > [CUDA](https://developer.nvidia.com/cuda-toolkit) > CPU |
| **Splits** | `train` + `val` + `test` fusionnés par défaut ; limitable via `--splits test` |

---

## Référence

Si vous utilisez RETFound dans vos travaux, merci de citer :

> Zhou, Y., et al. *A foundation model for generalizable disease detection from retinal images.* **Nature**, 2023. [doi:10.1038/s41586-023-06555-x](https://doi.org/10.1038/s41586-023-06555-x)
