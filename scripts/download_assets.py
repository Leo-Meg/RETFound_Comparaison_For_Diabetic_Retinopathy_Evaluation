"""Download the DR datasets and RETFound checkpoints used by this project."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DriveAsset:
    name: str
    file_id: str
    output: Path
    extract_to: Path | None = None

    @property
    def url(self) -> str:
        return f"https://drive.google.com/uc?id={self.file_id}"


DATASETS = (
    DriveAsset(
        name="APTOS2019",
        file_id="162YPf4OhMVxj9TrQH0GnJv0n7z7gJWpj",
        output=Path("dataset/_downloads/APTOS19.zip"),
        extract_to=Path("dataset"),
    ),
    DriveAsset(
        name="MESSIDOR2",
        file_id="1vOLBUK9xdzNV8eVkRjVdNrRwhPfaOmda",
        output=Path("dataset/_downloads/MESSIDOR2.zip"),
        extract_to=Path("dataset"),
    ),
    DriveAsset(
        name="IDRiD_data",
        file_id="1c6zexA705z-ANEBNXJOBsk6uCvRnzmr3",
        output=Path("dataset/_downloads/IDRiD_data.zip"),
        extract_to=Path("dataset"),
    ),
)

CHECKPOINTS = (
    DriveAsset(
        name="APTOS2019",
        file_id="1Ujzb6Xd1naWC0NngHah-DbHSgqxOiyJX",
        output=Path("check/checkpoint-best-APTOS2019.pth"),
    ),
    DriveAsset(
        name="MESSIDOR2",
        file_id="1EKJ1TraOkBpP6-XDNZUT5vNnABBEYgcx",
        output=Path("check/checkpoint-best-MESSIDOR2.pth"),
    ),
    DriveAsset(
        name="IDRID",
        file_id="1b0grTwARX1cXnYnMB3ZJZES26aMkgkvZ",
        output=Path("check/checkpoint-best-IDRID.pth"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Telecharge les datasets DR et les checkpoints RETFound."
    )
    parser.add_argument(
        "--datasets",
        action="store_true",
        help="Telecharger uniquement les datasets.",
    )
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="Telecharger uniquement les checkpoints.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Conserver les fichiers zip sans les extraire.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retelercharger les fichiers deja presents.",
    )
    return parser.parse_args()


def download_asset(asset: DriveAsset, force: bool) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "Dependance manquante: gdown. Installez les dependances avec "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc

    asset.output.parent.mkdir(parents=True, exist_ok=True)
    if asset.output.exists() and not force:
        print(f"[skip] {asset.output} existe deja")
        return

    print(f"[download] {asset.name} -> {asset.output}")
    downloaded = gdown.download(asset.url, str(asset.output), quiet=False)
    if downloaded is None:
        raise RuntimeError(f"Echec du telechargement: {asset.name}")


def extract_dataset(asset: DriveAsset, force: bool) -> None:
    if asset.extract_to is None:
        return
    if not asset.output.exists():
        raise FileNotFoundError(f"Archive introuvable: {asset.output}")

    asset.extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {asset.output} -> {asset.extract_to}")
    mode = "w" if force else "x"
    root = asset.extract_to.resolve()
    with zipfile.ZipFile(asset.output) as archive:
        for member in archive.infolist():
            destination = asset.extract_to / member.filename
            if not destination.resolve().is_relative_to(root):
                raise RuntimeError(f"Chemin dangereux dans l'archive: {member.filename}")
            if destination.exists() and not force:
                continue
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open(mode + "b") as target:
                shutil.copyfileobj(source, target)


def main() -> None:
    args = parse_args()
    selected_datasets = args.datasets or not args.checkpoints
    selected_checkpoints = args.checkpoints or not args.datasets

    if selected_datasets:
        for asset in DATASETS:
            download_asset(asset, force=args.force)
            if not args.no_extract:
                extract_dataset(asset, force=args.force)

    if selected_checkpoints:
        for asset in CHECKPOINTS:
            download_asset(asset, force=args.force)

    print("[done] Fichiers prets pour l'evaluation.")


if __name__ == "__main__":
    main()
