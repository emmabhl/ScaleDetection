"""
Convert JSON annotations to YOLO format (module intro)

This script converts polygon-style JSON annotations produced for the
scale-bar dataset into YOLOv8 text files and builds a `data.yaml` for
training. It outputs label `.txt` files and organizes `images/` and
`labels/` into `train` / `val` splits.

Example (complete call):
    python src/convert_jsons_to_yolo.py --data_dir data --train_split 0.8 --validate --sample_size 100

See the functions below for the exact conversion logic.
"""

import argparse
import json
import logging as log
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml


def points_to_yolo_bbox(
    points: List[List[float]], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """Convert polygon corner points to YOLO-format normalized bbox.

    Args:
        points (List[List[float]]): List of [x,y] point coordinates (at least two corners).
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.

    Returns:
        yolo_box (Tuple[float,float,float,float]): (x_center, y_center, width, height) normalized to [0,1].
    """
    if len(points) < 2:
        raise ValueError("At least 2 points required for bounding box")

    # Convert to numpy array for easier manipulation
    points_arr = np.array(points, dtype=np.float32)

    # Get bounding box coordinates
    x_min = np.min(points_arr[:, 0])
    y_min = np.min(points_arr[:, 1])
    x_max = np.max(points_arr[:, 0])
    y_max = np.max(points_arr[:, 1])

    # Clip to image boundaries (shouldn't be necessary but just in case)
    x_min = max(0, min(x_min, img_width))
    y_min = max(0, min(y_min, img_height))
    x_max = max(0, min(x_max, img_width))
    y_max = max(0, min(y_max, img_height))

    # Convert to YOLO format (center, width, height) and normalize
    # (https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return float(x_center), float(y_center), float(width), float(height)


def convert_json_to_yolo(
    json_path: str, label_dir: Path, class_mapping: Dict[str, int]
) -> bool:
    """Convert a single JSON annotation file into a YOLO `.txt` label file.

    Args:
        json_path (str): Path to the JSON annotation file.
        label_dir (Path): Directory where YOLO `.txt` files will be created.
        class_mapping (Dict[str,int]): Mapping from logical class names to integer class ids.

    Returns:
        success (bool): True on successful conversion, False otherwise.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract image dimensions
        img_width = data.get("width", 0)
        img_height = data.get("height", 0)

        if img_width <= 0 or img_height <= 0:
            log.error(f"Invalid image dimensions in {json_path}")
            return False

        # Prepare YOLO annotations
        yolo_annotations = []

        # Process scale bars (class 0)
        for bar in data.get("bars", []):
            if "points" in bar and len(bar["points"]) >= 2:
                x_center, y_center, width, height = points_to_yolo_bbox(
                    bar["points"], img_width, img_height
                )
                # Skip if bbox is too small
                if width > 0.001 and height > 0.001:
                    yolo_annotations.append(
                        f"{class_mapping['scalebar']} \
                                {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )

        # Process scale text (class 1)
        for label in data.get("labels", []):
            if "points" in label and len(label["points"]) >= 2:
                x_center, y_center, width, height = points_to_yolo_bbox(
                    label["points"], img_width, img_height
                )
                # Skip if bbox is too small
                if width > 0.001 and height > 0.001:
                    yolo_annotations.append(
                        f"{class_mapping['scalelabel']} \
                            {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )

        # Write YOLO annotation file
        json_filename = os.path.basename(json_path)
        txt_filename = json_filename.replace(".json", ".txt")
        txt_path = os.path.join(label_dir, txt_filename)

        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_annotations))

        return True

    except Exception as e:
        log.error(f"Error processing {json_path}: {e}")
        return False


def create_dataset_yaml(
    data_dir: Path, class_mapping: Dict[str, int], train_split: float = 0.8
) -> str:
    """Create a `data.yaml` file describing the YOLO dataset layout.

    Args:
        data_dir (Path): Root dataset directory containing `images/` and `labels/`.
        class_mapping (Dict[str,int]): Mapping of class names to ids.
        train_split (float, optional): Fraction of dataset to use for training. Defaults to 0.8.

    Returns:
        yaml_path (str): Path to the written YAML configuration file.
    """
    yaml_path = os.path.join(data_dir, "data.yaml")

    # Get list of all annotation files
    annotation_files = [
        f for f in os.listdir(data_dir / "labels") if f.endswith(".txt")
    ]
    annotation_files.sort()

    # Split into train/val
    n_train = int(len(annotation_files) * train_split)
    train_files = annotation_files[:n_train]
    val_files = annotation_files[n_train:]

    # Create dataset structure
    img_train_dir = data_dir / "images" / "train"
    img_val_dir = data_dir / "images" / "val"
    lab_train_dir = data_dir / "labels" / "train"
    lab_val_dir = data_dir / "labels" / "val"
    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(lab_train_dir, exist_ok=True)
    os.makedirs(lab_val_dir, exist_ok=True)

    dataset_config = {
        "path": os.path.relpath(data_dir),
        "train": os.path.relpath(img_train_dir, start=data_dir),
        "val": os.path.relpath(img_val_dir, start=data_dir),
        "test": None,
        "nc": len(class_mapping),
        "names": {v: k for k, v in class_mapping.items()},
    }

    # Move files
    for txt_file in train_files:
        img_file = txt_file.replace(".txt", ".jpg")
        os.rename(
            os.path.join(data_dir / "images", img_file),
            os.path.join(img_train_dir, img_file),
        )
        os.rename(
            os.path.join(data_dir / "labels", txt_file),
            os.path.join(lab_train_dir, txt_file),
        )
    for txt_file in val_files:
        img_file = txt_file.replace(".txt", ".jpg")
        os.rename(
            os.path.join(data_dir / "images", img_file),
            os.path.join(img_val_dir, img_file),
        )
        os.rename(
            os.path.join(data_dir / "labels", txt_file),
            os.path.join(lab_val_dir, txt_file),
        )

    # Write YAML config
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    return yaml_path


def convert_dataset(data_dir: Path, train_split: float = 0.8) -> str:
    """Convert all JSON annotations in a dataset to YOLO format and build splits.

    Args:
        data_dir (Path): Dataset root containing `images/` and `jsons/`.
        train_split (float, optional): Fraction for train split. Defaults to 0.8.

    Returns:
        yaml_path (str): Path to the generated `data.yaml` file.
    """
    # Configure and create label directory
    json_dir = data_dir / "jsons"
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"

    os.makedirs(label_dir, exist_ok=True)

    # Class mapping
    class_mapping = {"scalebar": 0, "scalelabel": 1}

    # Get all JSON files
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    json_files.sort()

    # Convert each JSON file
    successful_conversions = 0
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        if convert_json_to_yolo(json_path, label_dir, class_mapping):
            successful_conversions += 1

    # Create dataset YAML
    yaml_path = create_dataset_yaml(data_dir, class_mapping, train_split)

    return yaml_path


def validate_conversion(
    label_dir: Path, sample_size: int = -1
) -> Dict[str, Union[float, int]]:
    """Validate a set of YOLO annotation files and return simple statistics.

    Args:
        label_dir (Path): Directory containing YOLO `.txt` annotation files.
        sample_size (int, optional): How many files to sample (-1 = all). Defaults to -1.

    Returns:
        stats (Dict[str, Union[float,int]]): Summary statistics about annotations.
    """
    txt_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    sample_files = txt_files[:sample_size] if sample_size != -1 else txt_files

    stats = {
        "total_files": len(txt_files),
        "sample_size": len(sample_files),
        "files_with_annotations": 0,
        "total_scalebars": 0,
        "total_scalelabels": 0,
        "avg_annotations_per_file": 0.0,
    }

    for txt_file in sample_files:
        txt_path = os.path.join(label_dir, txt_file)

        with open(txt_path, "r") as f:
            lines = f.readlines()

        if lines:
            stats["files_with_annotations"] += 1
            stats["total_scalebars"] += sum(
                1 for line in lines if line.startswith("0 ")
            )
            stats["total_scalelabels"] += sum(
                1 for line in lines if line.startswith("1 ")
            )

    stats["avg_annotations_per_file"] = (
        stats["total_scalebars"] + stats["total_scalelabels"]
    ) / max(stats["files_with_annotations"], 1)

    return stats


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert JSON annotations to YOLO format"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the data (images and JSON annotation files)",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run validation after conversion"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=-1,
        help="Number of files to sample for validation (-1 for all)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Define directory paths
    DATA_DIR = Path(args.data_dir)
    IMAGES_DIR = DATA_DIR / "images"
    JSONS_DIR = DATA_DIR / "jsons"
    LABELS_DIR = DATA_DIR / "labels"

    # Check if already converted
    if os.path.exists(LABELS_DIR) and os.listdir(LABELS_DIR):
        log.info(f"{LABELS_DIR} already exists and is not empty. Skipping conversion")
        yaml_path = DATA_DIR / "data.yaml"
    else:

        os.makedirs(LABELS_DIR, exist_ok=True)

        # Count files
        image_files = list(Path(IMAGES_DIR).glob("*.jpg"))
        json_files = list(Path(JSONS_DIR).glob("*.json"))

        log.info(f"Found {len(image_files)} image files")
        log.info(f"Found {len(json_files)} JSON annotation files")

        # Convert dataset
        yaml_path = convert_dataset(DATA_DIR, args.train_split)

        # Run validation if requested
        if args.verbose:
            stats = validate_conversion(
                LABELS_DIR / "train", sample_size=args.sample_size
            )
            log.info(f"Validation results:")
            for key, value in stats.items():
                log.info(f"  {key}: {value}")

    log.info(f"Dataset ready for YOLO training. Configuration file: {yaml_path}")


if __name__ == "__main__":
    main()
