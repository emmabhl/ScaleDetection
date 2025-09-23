"""Script to download dataset from Kaggle and prepare data directories.
If the data directories already exist, it skips the download.

The script handles:
- Downloading the dataset from Kaggle using kagglehub.
- Moving images and JSON files to the appropriate data directories.
- Creating necessary directories for labels, models, and outputs.

Usage:
    python src/get_data.py --path original1/scalebar-dataset --data_dir data
"""

import os
from pathlib import Path
import argparse
import shutil
import kagglehub


def download_from_path(
        path: str ,
        data_dir: Path,
        images_dir: Path,
        jsons_dir: Path
    ) -> None:
    """
    Downloads a kaggle dataset from the given path and moves it to the data directory.

    Args:
        path (str): The file path.
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
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Download dataset and prepare folders.')
    parser.add_argument('--path', type=str, default="original1/scalebar-dataset",
                        help='Kaggle dataset path to download from.')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to store the dataset.')
    
    args = parser.parse_args()

    # Define directory paths
    DATA_DIR = Path(args.data_dir)
    IMAGES_DIR = DATA_DIR / "images"
    JSONS_DIR = DATA_DIR / "jsons"

    # Check if data directories exist, if not download the dataset
    if not (
        os.path.exists(DATA_DIR) and 
        os.path.exists(IMAGES_DIR) and 
        os.path.exists(JSONS_DIR)
    ):
        download_from_path(args.path, DATA_DIR, IMAGES_DIR, JSONS_DIR)
    else:
        print("Data directories already exist. Skipping download.")

if __name__ == "__main__":
    main()