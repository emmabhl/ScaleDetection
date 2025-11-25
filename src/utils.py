"""
Utility helpers

Small helpers used across the project. Currently includes:
- `ensure_model_available(local_path, repo_id, filename)`: ensure a model file exists locally and
    download it from Hugging Face if needed.

Example usage:
    from utils import ensure_model_available
    local = ensure_model_available('models/train/weights/best.pt', 'emmabhl/yolov8m-ScalebarDetection', 'best.pt')
"""

import logging as log
import os

from huggingface_hub import hf_hub_download


def ensure_model_available(local_path: str, repo_id: str, filename: str) -> str:
    """
    Ensures that a model file exists locally.
    If not, downloads it from a private Hugging Face repo.

    Args:
        local_path (str): Final local path (e.g. 'models/train/weights/best.pt')
        repo_id (str): Hugging Face repo ID (e.g. 'yourname/my-private-yolov8m')
        filename (str): File inside the HF repo (e.g. 'best.pt')
    """

    # If already present, nothing to do
    if os.path.exists(local_path):
        return local_path

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Download to HF cache
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )

    # Copy from cache to your desired folder
    import shutil

    shutil.copy(downloaded_path, local_path)

    log.info(
        f"   Model downloaded from Hugging Face ({repo_id}/{filename}) and saved to {local_path}"
    )
    return local_path
