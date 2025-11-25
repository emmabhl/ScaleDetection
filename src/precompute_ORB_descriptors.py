"""
Precompute ORB descriptors for template images

This utility computes ORB keypoints and descriptors for template images
organized by type (e.g. `root/type/*.jpg`) and saves per-type `.pkl` files
containing serializable keypoint dictionaries and descriptors.

Example (complete call):
    python src/precompute_ORB_descriptors.py --template_root emmabhl/atypical-scalebar --output_root .precomputed_ORB_descriptors --nfeatures 1000

The produced `.pkl` files are consumed by `classifier.py` for template matching.
"""

import argparse
import logging as log
import os
import pickle
import shutil
from typing import Any, Dict, Generator, List, Optional, Tuple, cast

import cv2
import numpy as np
from huggingface_hub import snapshot_download


def compute_orb(
    orb: cv2.ORB, img_gray: np.ndarray
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[np.ndarray]]:
    """Compute ORB keypoints and descriptors and serialize keypoints.

    Args:
        orb (cv2.ORB): ORB detector instance.
        img_gray (np.ndarray): Grayscale image to process.

    Returns:
        (kp_dicts, descriptors): Serialized keypoint dicts and descriptor array,
            or (None, None) if no keypoints found.
    """
    keypoints, descriptors = orb.detectAndCompute(
        img_gray, cast(cv2.typing.MatLike, None)
    )
    if descriptors is None or len(keypoints) == 0:
        return None, None

    # Convert kp to serializable dicts
    kp_dicts = [
        {
            "pt": kp.pt,
            "size": kp.size,
            "angle": kp.angle,
            "response": kp.response,
            "octave": kp.octave,
            "class_id": kp.class_id,
        }
        for kp in keypoints
    ]

    return kp_dicts, descriptors


def process_template_file(orb: cv2.ORB, file_path: str) -> Optional[Dict[str, Any]]:
    """Load an image file and compute ORB keypoints/descriptors.

    Args:
        orb (cv2.ORB): ORB detector instance.
        file_path (str): Path to the image file.

    Returns:
        info (Optional[Dict]): Dict with 'filename','image_shape','keypoints','descriptors', or None on error.
    """
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.error(f"Could not load file: {file_path}")
        return None

    keypoints, descriptors = compute_orb(orb, img)
    if keypoints is None:
        log.error(f"No keypoints in {os.path.basename(file_path)}")
        return None

    return {
        "filename": os.path.basename(file_path),
        "image_shape": img.shape,
        "keypoints": keypoints,
        "descriptors": descriptors,
    }


def iterate_template_files(
    template_root: str,
) -> Generator[Tuple[str, str], None, None]:
    """Yield (scale_type, image_path) pairs for template images under a root.

    Args:
        template_root (str): Root directory or HF dataset id containing subfolders per scale type.

    Yields:
        (scale_type, file_path): Tuple with type name and absolute image path.
    """
    if not os.path.exists(template_root):
        template_root = snapshot_download(template_root, repo_type="dataset")

    for scale_type in os.listdir(template_root):
        scale_dir = os.path.join(template_root, scale_type)
        if not os.path.isdir(scale_dir):
            continue

        for fname in os.listdir(scale_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                yield scale_type, os.path.join(scale_dir, fname)


def save_templates(output_path: str, templates: List[Dict[str, Any]]) -> None:
    """Persist a list of template dictionaries to a pickle file.

    Args:
        output_path (str): Destination pickle file path.
        templates (List[Dict[str,Any]]): Templates to serialize.

    Returns:
        None: Writes `output_path`.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(templates, f)


def main(
    template_root: str = "emmabhl/atypical-scalebar",
    output_root: str = ".precomputed_ORB_descriptors",
    nfeatures: int = 1000,
) -> None:
    """Compute ORB descriptors for all templates and save per-type pickles.

    Args:
        template_root (str): Root folder or HF dataset id with template images.
        output_root (str): Output directory to store `.pkl` files.
        nfeatures (int): Number of ORB features to compute per image.

    Returns:
        None: Produces `.pkl` files under `output_root`.
    """
    os.makedirs(output_root, exist_ok=True)

    # Create ORB exactly once
    orb = cv2.ORB_create(
        nfeatures=nfeatures
    )  # pyright: ignore[reportAttributeAccessIssue]

    # Group files by type
    type_to_files = {}
    for ttype, file_path in iterate_template_files(template_root):
        type_to_files.setdefault(ttype, []).append(file_path)

    # Process each scale-bar type
    for ttype, files in type_to_files.items():
        results = []
        for file_path in files:
            info = process_template_file(orb, file_path)
            if info is not None:
                results.append(info)

        # Save
        output_file = os.path.join(output_root, f"{ttype}.pkl")
        save_templates(output_file, results)

        log.info(f"Saved {len(results)} templates for '{ttype}' → {output_file}")


# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template_root", type=str, default="emmabhl/atypical-scalebar"
    )
    parser.add_argument(
        "--output_root", type=str, default=".precomputed_ORB_descriptors"
    )
    parser.add_argument("--nfeatures", type=int, default=1000)
    args = parser.parse_args()

    main(args.template_root, args.output_root, args.nfeatures)
