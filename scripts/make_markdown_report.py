#!/usr/bin/env python3
"""Create a compact Markdown report from summary_metrics.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "accuracy",
    "auroc_macro_ovr",
    "f1_macro",
    "cohen_kappa",
    "composite_f1_auc_kappa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genere un rapport Markdown.")
    parser.add_argument("--summary", default="results/dr_matrix/summary_metrics.csv")
    parser.add_argument("--output", default="results/dr_matrix/report.md")
    return parser.parse_args()


def fmt(value) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    output_path = Path(args.output)
    df = pd.read_csv(summary_path)

    lines = [
        "# Rapport d'evaluation RETFound DR",
        "",
        "Chaque ligne correspond a un checkpoint fine-tune sur le dataset source, puis evalue sur le dataset cible.",
        "",
        "| Source checkpoint | Dataset cible | N images | Accuracy | AUROC | F1 macro | Kappa | Composite |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in df.sort_values(["train_dataset", "eval_dataset"]).iterrows():
        lines.append(
            "| "
            f"{row['train_dataset']} | "
            f"{row['eval_dataset']} | "
            f"{int(row['n_images'])} | "
            f"{fmt(row['accuracy'])} | "
            f"{fmt(row['auroc_macro_ovr'])} | "
            f"{fmt(row['f1_macro'])} | "
            f"{fmt(row['cohen_kappa'])} | "
            f"{fmt(row['composite_f1_auc_kappa'])} |"
        )

    external = df[df["train_dataset"] != df["eval_dataset"]].copy()
    if not external.empty:
        best = external.sort_values("composite_f1_auc_kappa", ascending=False).iloc[0]
        lines.extend(
            [
                "",
                "## Meilleure validation externe",
                "",
                (
                    f"- Source: `{best['train_dataset']}`; cible: `{best['eval_dataset']}`; "
                    f"composite: `{fmt(best['composite_f1_auc_kappa'])}`; "
                    f"AUROC: `{fmt(best['auroc_macro_ovr'])}`."
                ),
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Rapport ecrit: {output_path}")


if __name__ == "__main__":
    main()
