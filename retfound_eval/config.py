"""Configuration for the DR datasets used in external validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


NUM_CLASSES = 5
INPUT_SIZE = 224
SPLITS = ("train", "val", "test")

CLASS_NAMES = (
    "0 - Pas de DR",
    "1 - DR legere",
    "2 - DR moderee",
    "3 - DR severe",
    "4 - DR proliferante",
)

GRADE_DESCRIPTIONS = (
    "Pas de retinopathie diabetique",
    "Retinopathie diabetique legere",
    "Retinopathie diabetique moderee",
    "Retinopathie diabetique severe",
    "Retinopathie diabetique proliferante",
)


@dataclass(frozen=True)
class DatasetConfig:
    """Local dataset metadata."""

    name: str
    path: str
    folder_to_label: dict[str, int]
    checkpoint_name: str

    @property
    def root(self) -> Path:
        return Path(self.path)


DR_DATASETS: dict[str, DatasetConfig] = {
    "APTOS2019": DatasetConfig(
        name="APTOS2019",
        path="dataset/APTOS2019",
        checkpoint_name="checkpoint-best-APTOS2019.pth",
        folder_to_label={
            "anodr": 0,
            "bmilddr": 1,
            "cmoderatedr": 2,
            "dseveredr": 3,
            "eproliferativedr": 4,
        },
    ),
    "IDRiD_data": DatasetConfig(
        name="IDRiD_data",
        path="dataset/IDRiD_data",
        checkpoint_name="checkpoint-best-IDRID.pth",
        folder_to_label={
            "anoDR": 0,
            "bmildDR": 1,
            "cmoderateDR": 2,
            "dsevereDR": 3,
            "eproDR": 4,
        },
    ),
    "MESSIDOR2": DatasetConfig(
        name="MESSIDOR2",
        path="dataset/MESSIDOR2",
        checkpoint_name="checkpoint-best-MESSIDOR2.pth",
        folder_to_label={
            "anodr": 0,
            "bmilddr": 1,
            "cmoderatedr": 2,
            "dseveredr": 3,
            "eproliferativedr": 4,
        },
    ),
}


def dataset_names() -> list[str]:
    return list(DR_DATASETS)


def get_dataset_config(name: str) -> DatasetConfig:
    try:
        return DR_DATASETS[name]
    except KeyError as exc:
        valid = ", ".join(dataset_names())
        raise KeyError(f"Dataset inconnu: {name}. Choix valides: {valid}") from exc


def checkpoint_path(check_dir: str | Path, dataset_name: str) -> Path:
    cfg = get_dataset_config(dataset_name)
    return Path(check_dir) / cfg.checkpoint_name
