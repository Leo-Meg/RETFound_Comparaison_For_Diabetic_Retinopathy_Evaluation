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

Les checkpoints attendus sont dans `check/`:

- `checkpoint-best-APTOS2019.pth`
- `checkpoint-best-IDRID.pth`
- `checkpoint-best-MESSIDOR2.pth`

## Installation

L'environnement local actuel ne contient pas encore les dependances deep learning. Dans un environnement Python adapte:

```bash
pip install -r requirements.txt
```

## Inspecter les datasets

```bash
python scripts/inspect_dr_datasets.py
```

## Evaluer une paire

Exemple: checkpoint fine-tune sur MESSIDOR2, evaluation externe sur IDRiD:

```bash
python scripts/evaluate_pair.py \
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
python scripts/evaluate_matrix.py --batch-size 16
```

Seulement les validations externes:

```bash
python scripts/evaluate_matrix.py --external-only --batch-size 16
```

Resume global:

```text
results/dr_matrix/summary_metrics.csv
```

## Resumer des resultats deja calcules

```bash
python scripts/summarize_results.py --results-dir results/dr_matrix
```

## Generer un rapport Markdown

```bash
python scripts/make_markdown_report.py \
  --summary results/dr_matrix/summary_metrics.csv \
  --output results/dr_matrix/report.md
```

## Notes

- Le preprocessing reprend le notebook `manip.ipynb`: resize `224x224`, `ToTensor`, normalisation ImageNet.
- Le modele est un ViT-L/16 compatible RETFound avec `global_pool=True`.
- Les scripts chargent les poids depuis la cle `model` des checkpoints, comme dans le notebook.
- Par defaut, `--device auto` choisit MPS sur Apple Silicon, sinon CUDA, sinon CPU.
- Les splits `train`, `val`, `test` sont fusionnes pour reproduire l'evaluation globale du notebook. On peut limiter avec `--splits test`.
