# RETFound_Comparaison_For_Diabetic_Retinopathy_Evaluation
# Projet d'evaluation RETFound DR

Ce mini-projet automatise l'evaluation de generalisation de RETFound entre les trois datasets de retinopathie diabetique:

- `APTOS2019`
- `IDRiD_data`
- `MESSIDOR2`

Chaque dataset est mappe vers la meme convention de grades DR:

| Grade | Sens clinique |
|---:|---|
| 0 | Pas de retinopathie diabetique |
| 1 | Retinopathie diabetique legere |
| 2 | Retinopathie diabetique moderee |
| 3 | Retinopathie diabetique severe |
| 4 | Retinopathie diabetique proliferante |

Les datasets attendus sont dans `dataset/`:

- `dataset/APTOS2019`
- `dataset/IDRiD_data`
- `dataset/MESSIDOR2`

Les checkpoints attendus sont dans `check/`:

- `checkpoint-best-APTOS2019.pth`
- `checkpoint-best-IDRID.pth`
- `checkpoint-best-MESSIDOR2.pth`

## Installation

Installation des dépendances nécessaire dans votre environnement python:

```bash
python3 -m pip install -r requirements.txt
```

## Telecharger les datasets et checkpoints

Les dossiers `dataset/`, `check/` et `results/` sont gardes dans GitHub avec des fichiers `.gitkeep`, mais leur contenu local est ignore par Git.

Pour telecharger les trois datasets et les trois checkpoints:

```bash
python3 scripts/download_assets.py
```

Pour telecharger uniquement les datasets:

```bash
python3 scripts/download_assets.py --datasets
```

Pour telecharger uniquement les checkpoints:

```bash
python3 scripts/download_assets.py --checkpoints
```

Par defaut, les archives zip sont conservees dans `dataset/_downloads/` puis extraites dans `dataset/`. Les checkpoints sont telecharges directement avec les noms attendus par les scripts:

```text
check/checkpoint-best-APTOS2019.pth
check/checkpoint-best-IDRID.pth
check/checkpoint-best-MESSIDOR2.pth
```

Liens sources:

| Type | Fichier | Lien |
|---|---|---|
| Dataset | `APTOS19.zip` | https://drive.google.com/file/d/162YPf4OhMVxj9TrQH0GnJv0n7z7gJWpj/view |
| Dataset | `MESSIDOR2.zip` | https://drive.google.com/file/d/1vOLBUK9xdzNV8eVkRjVdNrRwhPfaOmda/view |
| Dataset | `IDRiD_data.zip` | https://drive.google.com/file/d/1c6zexA705z-ANEBNXJOBsk6uCvRnzmr3/view |
| Checkpoint | `checkpoint-best-APTOS2019.pth` | https://drive.google.com/file/d/1Ujzb6Xd1naWC0NngHah-DbHSgqxOiyJX/view |
| Checkpoint | `checkpoint-best-MESSIDOR2.pth` | https://drive.google.com/file/d/1EKJ1TraOkBpP6-XDNZUT5vNnABBEYgcx/view |
| Checkpoint | `checkpoint-best-IDRID.pth` | https://drive.google.com/file/d/1b0grTwARX1cXnYnMB3ZJZES26aMkgkvZ/view |

## Inspecter les datasets

```bash
python3 scripts/inspect_dr_datasets.py
```

## Evaluer une paire

Exemple: checkpoint fine-tune sur MESSIDOR2, evaluation externe sur IDRiD:

```bash
python3 scripts/evaluate_pair.py \
  --train-dataset MESSIDOR2 \
  --eval-dataset IDRiD_data \
  --batch-size 16
```

Les sorties sont creees dans:

```text
results/train-MESSIDOR2__eval-IDRiD_data/
```

avec:

- `predictions.csv`: prediction image par image
- `metrics.csv`: metriques globales
- `metrics.json`: metriques, rapport par classe, matrice de confusion
- `confusion_matrix.png`
- `roc_curves.png`

## Lancer toute la matrice

Inclut les diagonales, utiles comme reference interne:

```bash
python3 scripts/evaluate_matrix.py --batch-size 16
```

Seulement les validations externes:

```bash
python3 scripts/evaluate_matrix.py --external-only --batch-size 16
```

Resume global:

```text
results/dr_matrix/summary_metrics.csv
```

## Resumer des resultats deja calcules

```bash
python3 scripts/summarize_results.py --results-dir results/dr_matrix
```

## Generer un rapport Markdown

```bash
python3 scripts/make_markdown_report.py \
  --summary results/dr_matrix/summary_metrics.csv \
  --output results/dr_matrix/report.md
```

## Notes

- Le preprocessing prend en paramètre`: resize `224x224`, `ToTensor`, normalisation ImageNet.
- Le modele est un ViT-L/16 compatible RETFound avec `global_pool=True`.
- Les scripts chargent les poids depuis la cle `model` des checkpoints.
- Par defaut, `--device auto` choisit MPS sur Apple Silicon, sinon CUDA, sinon CPU.
- Les splits `train`, `val`, `test` sont fusionnes pour reproduire l'evaluation globale du notebook. On peut limiter avec `--splits test`.
