"""
YOLOv8 Training Script for Scale Detection

Train a YOLOv8 model to detect scale bars and scale labels. This module
wraps the common training flow (creating config overrides and calling the
Ultralytics `YOLO.train()` API) and also provides export and ClearML
integration helpers.

Example (complete call):
    python src/train_yolo.py --data_yaml data/data.yaml --model_name yolov8m.pt --epochs 100

Use programmatically via `train_model(...)` or run as a script to train
from the command line.
"""

import argparse
import logging as log
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from ultralytics import YOLO


def train_model(
    data_yaml: str,
    model_name: str = "yolov8m.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 16,
    lr0: float = 0.01,
    weight_decay: float = 0.0005,
    warmup_epochs: int = 3,
    patience: int = 100,
    save_period: int = 2,
    device: str = "auto",
    workers: int = 4,
    model_dir: Path = Path("models"),
    name: str = "train",
    resume: bool = False,
) -> Tuple[YOLO, Any]:
    """Train a YOLOv8 model with the provided training configuration.

    Args:
        data_yaml (str): Path to dataset YAML.
        model_name (str): Pretrained model name or path.
        epochs (int): Number of epochs.
        imgsz (int): Training image size.
        batch (int): Batch size.
        lr0 (float): Initial learning rate.
        weight_decay (float): Weight decay.
        warmup_epochs (int): Warmup epochs.
        patience (int): Early stopping patience.
        save_period (int): Checkpoint save frequency.
        device (str): Device identifier for training.
        workers (int): Data loader worker count.
        model_dir (Path): Directory to save training outputs.
        name (str): Experiment name.
        resume (bool): Whether to resume from last checkpoint.

    Returns:
        (model, results): Trained YOLO model object and training results.
    """

    if resume:
        path = str(model_dir) + "/" + name + "/weights/last.pt"
        model = YOLO(path)
    else:
        model = YOLO(model_name)

    config = create_training_config(
        data_yaml=data_yaml,
        model_name=model_name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        patience=patience,
        save_period=save_period,
        device=device,
        workers=workers,
        project=model_dir,
        resume=resume,
    )
    results = model.train(**config)

    return model, results


def create_training_config(
    data_yaml: str,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    lr0: float,
    weight_decay: float,
    warmup_epochs: int,
    patience: int,
    save_period: int,
    device: str,
    workers: int,
    project: Path,
    resume: bool = False,
) -> Dict[str, Any]:
    """Build a set of overrides for model.train()."""
    """Create a config dict of training overrides for `YOLO.train()`.

    Args:
        data_yaml (str): Path to dataset YAML.
        model_name (str): Pretrained model name or path.
        epochs (int): Number of training epochs.
        imgsz (int): Image size.
        batch (int): Batch size.
        lr0 (float): Initial learning rate.
        weight_decay (float): Weight decay.
        warmup_epochs (int): Warmup epochs.
        patience (int): Early stopping patience.
        save_period (int): Checkpoint save frequency.
        device (str): Device string for training.
        workers (int): Number of data loader workers.
        project (Path): Output project directory.
        resume (bool): Resume flag.

    Returns:
        config (Dict[str,Any]): Dictionary of keyword arguments to pass to `YOLO.train()`.
    """

    config = {
        # Training
        "data": data_yaml,  # Path to dataset YAML file
        "model": model_name,  # Pretrained model name
        "epochs": epochs,  # Number of training epochs
        "imgsz": imgsz,  # Input image size
        "batch": batch,  # Batch size
        #'lr0': lr0,                     # Initial learning rate
        #'weight_decay': weight_decay,   # Weight decay
        #'warmup_epochs': warmup_epochs, # Warmup epochs
        #'patience': patience,           # Early stopping patience
        "save_period": save_period,  # Save checkpoint every N epochs
        "device": device,  # Device to use for training
        "workers": workers,  # Number of data loader workers
        "project": str(project),  # Project name
        "resume": resume,  # Resume from last checkpoint if True
        # Validation
        "val": True,  # Activate validation during training
        "split": "val",  # Use 'val' split for validation
        #'conf': 0.001,                  # Confidence threshold for validation
        #'iou': 0.6,                     # IoU threshold for NMS
        #'max_det': 25,                  # Maximum detections per image
        # Logging and visualization
        "verbose": False,  # Display training progress
        "save": True,  # Save checkpoints
        "plots": True,  # Save plots during validation
        "save_txt": True,  # Save labels as txt
        "save_conf": True,  # Save confidences in txt labels
        "save_crop": False,  # Do not save cropped images
        "show_labels": True,  # Show labels on images
        "show_conf": True,  # Show confidences on images
        "visualize": False,  # Do not visualize predictions
    }
    return config


def export_model(
    model: YOLO,
    export_path: str,
    format: str = "onnx",
    imgsz: int = 1280,
    opset: int = 12,
    half: bool = True,
    dynamic: bool = False,
    simplify: bool = True,
    workspace: int = 4,
) -> str:
    """Export a trained YOLO model to the requested format.

    Args:
        model (YOLO): Trained model object.
        export_path (str): Destination file path for the exported model.
        format (str): Export format (e.g., 'onnx').
        imgsz (int): Inference image size.
        opset (int): ONNX opset version.
        half (bool): Use FP16 where supported.
        dynamic (bool): Use dynamic axes.
        simplify (bool): Apply model simplification.
        workspace (int): Workspace size for export.

    Returns:
        exported_path (str): Path to the exported model file.
    """
    export_kwargs = {
        "format": format,
        "imgsz": imgsz,
        "opset": opset,
        "half": half,
        "dynamic": dynamic,
        "simplify": simplify,
        "workspace": workspace,
    }

    # Export model
    exported_path = model.export(**export_kwargs)

    # Move to desired location if different
    if export_path != exported_path:
        import shutil

        shutil.move(exported_path, export_path)
        exported_path = export_path

    log.info(f"Model exported to: {exported_path}")
    return exported_path


def main():
    """CLI entry for training a YOLOv8 model using the provided options."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for scale detection"
    )

    # Training options
    parser.add_argument(
        "--data_yaml",
        type=str,
        default="data/data.yaml",
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--model_name", type=str, default="yolov8m.pt", help="Pretrained model name"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="Weight decay"
    )
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument(
        "--patience", type=int, default=100, help="Early stopping patience"
    )
    parser.add_argument(
        "--save_period", type=int, default=25, help="Save checkpoint every N epochs"
    )

    # Hardware and directories
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use for training"
    )
    parser.add_argument(
        "--model_dir", type=str, default="models", help="Directory to save models"
    )
    parser.add_argument(
        "--name", type=str, default="yolov8m_train", help="Experiment name"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )

    # Export options
    parser.add_argument(
        "--export", action="store_true", help="Export model after training"
    )
    parser.add_argument(
        "--export_name",
        type=str,
        default="scale_detection_model.onnx",
        help="Exported model filename",
    )
    parser.add_argument(
        "--export_format", type=str, default="onnx", help="Export format"
    )

    # Logging options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging output"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Initialize ClearML Task
    try:
        from clearml import Task

        task = Task.init(
            project_name="scale_detection",
        )
    except Exception as e:
        log.warning(f"ClearML not initialized ({e}). Proceeding without tracking.")
        task = None

    # Create model directory if it doesn't exist
    MODEL_DIR = Path(args.model_dir)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Train model
    log.info("Starting YOLOv8 training...")
    model, results = train_model(
        data_yaml=args.data_yaml,
        model_name=args.model_name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        save_period=args.save_period,
        device=args.device,
        model_dir=MODEL_DIR,
        name=args.name,
        resume=args.resume,
    )

    log.info("Training completed!")

    # Determine run directory from trainer if available
    run_dir = getattr(model.trainer, "save_dir", os.path.join(MODEL_DIR, args.name))

    # Export model if requested
    if args.export:
        exported_path = export_model(
            model, os.path.join(MODEL_DIR, args.export_name), format=args.export_format
        )
    else:
        exported_path = None

    # Log artifacts to ClearML
    if task is not None:
        try:
            # Connect basic hyperparameters
            overrides = create_training_config(
                data_yaml=args.data_yaml,
                model_name=args.model_name,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                lr0=args.lr0,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                patience=args.patience,
                save_period=args.save_period,
                device=args.device,
                workers=args.workers,
                project=MODEL_DIR,
                resume=args.resume,
            )
            task.connect(overrides)

            # Common artifacts
            results_csv = os.path.join(run_dir, "results.csv")
            best_weights = os.path.join(run_dir, "weights", "best.pt")
            last_weights = os.path.join(run_dir, "weights", "last.pt")

            if os.path.exists(results_csv):
                task.upload_artifact("results.csv", artifact_object=results_csv)
            if os.path.exists(best_weights):
                task.upload_artifact("best.pt", artifact_object=best_weights)
            if os.path.exists(last_weights):
                task.upload_artifact("last.pt", artifact_object=last_weights)
            if exported_path and os.path.exists(exported_path):
                task.upload_artifact(
                    Path(exported_path).name, artifact_object=exported_path
                )
        except Exception as e:
            log.warning(f"Failed to upload ClearML artifacts: {e}")


if __name__ == "__main__":
    main()
