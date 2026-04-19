"""Plot helpers for evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from .config import NUM_CLASSES


SHORT_LABELS = (
    "Grade 0\nNo DR",
    "Grade 1\nMild",
    "Grade 2\nModerate",
    "Grade 3\nSevere",
    "Grade 4\nProlif.",
)


def save_confusion_matrix(cm: np.ndarray, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )
    annot = np.array(
        [
            [f"{cm[i, j]}\n{cm_norm[i, j] * 100:.1f}%" for j in range(NUM_CLASSES)]
            for i in range(NUM_CLASSES)
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=SHORT_LABELS,
        yticklabels=SHORT_LABELS,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Proportion par vraie classe"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Grade predit")
    ax.set_ylabel("Grade reel")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    colors = ["#27ae60", "#f39c12", "#e67e22", "#e74c3c", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(8, 7))
    for class_idx in range(NUM_CLASSES):
        binary_true = (y_true == class_idx).astype(int)
        if binary_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary_true, y_prob[:, class_idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=colors[class_idx],
            lw=2,
            label=f"Grade {class_idx} (AUC={roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Hasard")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
