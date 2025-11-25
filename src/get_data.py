"""
Dataset downloader and preparer

Downloads the dataset from Kaggle (via `kagglehub`) and arranges files into
the expected `data/` layout with `images/` and `jsons/`. If the target
directories already exist, the script will skip the download.

Example (complete call):
    python src/get_data.py --path original1/scalebar-dataset --data_dir data

This script is intended to be run once to prepare the dataset prior to
conversion/training.
"""

import argparse
import logging as log
import os
from pathlib import Path
import shutil

import kagglehub


def download_from_path(
    path: str, data_dir: Path, images_dir: Path, jsons_dir: Path
) -> None:
    """Download a Kaggle dataset via `kagglehub` and arrange data directories.

    Args:
        path (str): Kaggle dataset path or identifier to download.
        data_dir (Path): Destination dataset root directory.
        images_dir (Path): Destination directory for image files.
        jsons_dir (Path): Destination directory for JSON annotation files.

    Returns:
        None: Downloads and moves files into the provided directories.
    """
    # Get the dataset files if needed
    os.makedirs(data_dir, exist_ok=True)
    # Download the dataset
    path = kagglehub.dataset_download(path)

    # Move images & jsons to data folder
    shutil.move(Path(path) / "data_publish" / "figures", images_dir)
    shutil.move(Path(path) / "data_publish" / "jsons", jsons_dir)

    # Remove the original dataset folder
    shutil.rmtree(Path(path).parent.parent.parent, ignore_errors=True)


def main() -> None:
    """CLI entry point to download and prepare dataset directories."""
    parser = argparse.ArgumentParser(
        description="Download dataset and prepare folders."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="original1/scalebar-dataset",
        help="Kaggle dataset path to download from.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory to store the dataset."
    )

    args = parser.parse_args()

    # Define directory paths
    DATA_DIR = Path(args.data_dir)
    IMAGES_DIR = DATA_DIR / "images"
    JSONS_DIR = DATA_DIR / "jsons"

    # Check if data directories exist, if not download the dataset
    if not (
        os.path.exists(DATA_DIR)
        and os.path.exists(IMAGES_DIR)
        and os.path.exists(JSONS_DIR)
    ):
        download_from_path(args.path, DATA_DIR, IMAGES_DIR, JSONS_DIR)
    else:
        log.info("Data directories already exist. Skipping download.")


if __name__ == "__main__":
    main()
