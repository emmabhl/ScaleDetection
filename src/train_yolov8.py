"""
YOLOv8 Training Script for Scale Detection

Minimal training wrapper for Ultralytics YOLOv8, focused on essential knobs only.
Keeps: data, model, epochs, imgsz, batch, lr0, weight_decay, device, workers,
project/name. Exposes a simple CLI and optional ONNX export.
"""

import os
import argparse
from typing import Dict, Any, Optional, Tuple
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

def train_model(
    data_yaml: str,
    model_name: str = "yolov8m.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 8,
    device: str = "auto",
    workers: int = 8,
    project: str = "scale_detection",
    lr0: float = 0.01,
    weight_decay: float = 0.0005,
    warmup_epochs: int = 3,
    patience: int = 50,
    save_period: int = -1
) -> Tuple[YOLO, Any]:
    """Train YOLOv8 model with a minimal, current-friendly config."""

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
        project=project
    )
    results = model.train(**config)

    return model, results


def create_training_config(
    data_yaml: str,
    model_name: str = "yolov8m.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 8,
    lr0: float = 0.01,
    weight_decay: float = 0.0005,
    warmup_epochs: int = 3,
    cos_lr: bool = True,
    lrf: float = 0.01,
    patience: int = 100,
    save_period: int = 10,
    device: str = "auto",
    workers: int = 8,
    project: str = "scale_detection"
) -> Dict[str, Any]:
    
    """Build a set of overrides for model.train()."""

    config = {
        'data': data_yaml,      # Path to dataset YAML file
        'model': model_name,    # Pretrained model name
        'epochs': epochs,       # Number of training epochs
        'imgsz': imgsz,         # Input image size
        'batch': batch,         # Batch size
        'lr0': lr0,             # Initial learning rate
        'weight_decay': weight_decay, # Weight decay
        'warmup_epochs': warmup_epochs, # Warmup epochs
        'cos_lr': cos_lr,       # Use cosine LR schedule
        'lrf': lrf,             # Final learning rate factor
        'patience': patience,   # Early stopping patience
        'save_period': save_period, # Save checkpoint every N epochs
        'device': device,       # Device to use for training
        'workers': workers,     # Number of worker threads
        'project': project,     # Project name

        # Data augmentation
        # Color augmentation (important for different microscopy techniques)
        'hsv_h': 0.015,         # Slight hue variation
        'hsv_s': 0.7,           # Higher saturation variation for different staining
        'hsv_v': 0.4,           # Value variation for different lighting

        # Geometric augmentation (conservative for microscopy)
        'degrees': 0.0,         # No rotation (scale bars should be horizontal)
        'translate': 0.1,       # Small translation
        'scale': 0.2,           # Moderate scale variation
        'shear': 0.0,           # No shear
        'perspective': 0.0,     # No perspective distortion

        # Flip augmentation
        'flipud': 0.0,          # Vertical flip
        'fliplr': 0.0,          # Horizontal flip

        # Advanced augmentation
        'mosaic': 0.0,          # No mosaic augmentation
        'mixup': 0.0,           # No mixup (preserves scale relationships)
        'copy_paste': 0.0,      # No copy-paste (preserves scale relationships)

        # Optimization
        'optimizer': 'auto',    # SGD optimizer as used in paper
        'seed': 42,             # Random seed for reproducibility
        'close_mosaic': 0,      # Disable mosaic augmentation for last N epochs

        # Validation
        'val': True,            # Activate validation during training
        'split': 'val',
        'save_json': True,
        'save_hybrid': False,
        'conf': 0.001,  # Confidence threshold for validation
        'iou': 0.6,     # IoU threshold for NMS
        'max_det': 300, # Maximum detections per image
        'half': False,  # Use FP16 inference
        'dnn': False,   # Use OpenCV DNN for ONNX inference

        # Logging and visualization
        'verbose': True,        # Print training progress
        'save': True,           # Save checkpoints
        'save_txt': True,       # Save labels as txt
        'save_conf': True,      # Save confidences in txt labels
        'save_crop': False,     # Do not save cropped images
        'show_labels': True,    # Show labels on images
        'show_conf': True,      # Show confidences on images
        'visualize': False,     # Do not visualize predictions
        'augment': True,        # Use augmentation
        'agnostic_nms': False,  # Do not use class-agnostic NMS
        'retina_masks': False  # Do not use retina masks
    }
    return config


def export_model(model: YOLO, export_path: str, format: str = "onnx", imgsz: int = 1280, 
                opset: int = 12, half: bool = True, dynamic: bool = False, 
                simplify: bool = True, workspace: int = 4) -> str:
    """
    Export trained model to various formats.
    
    Args:
        model: Trained YOLO model
        export_path: Path to save exported model
        format: Export format (onnx, torchscript, etc.)
        imgsz: Input image size
        opset: ONNX opset version
        half: Whether to use FP16
        dynamic: Whether to use dynamic axes
        simplify: Whether to simplify model
        workspace: ONNX workspace size
        
    Returns:
        Path to exported model
    """
    export_kwargs = {
        'format': format,
        'imgsz': imgsz,
        'opset': opset,
        'half': half,
        'dynamic': dynamic,
        'simplify': simplify,
        'workspace': workspace
    }
    
    # Export model
    exported_path = model.export(**export_kwargs)
    
    # Move to desired location if different
    if export_path != exported_path:
        import shutil
        shutil.move(exported_path, export_path)
        exported_path = export_path
    
    print(f"Model exported to: {exported_path}")
    return exported_path


def plot_training_results(results_path: str, save_path: Optional[str] = None) -> None:
    """
    Plot training results and metrics.
    
    Args:
        results_path: Path to training results
        save_path: Path to save plots (optional)
    """
    # Load results
    results_csv = os.path.join(results_path, "results.csv")
    if not os.path.exists(results_csv):
        print(f"Results CSV not found at {results_csv}")
        return
    
    import pandas as pd
    df = pd.read_csv(results_csv)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='blue')
    axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red')
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', color='blue')
    axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', color='red')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # mAP curves
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
    axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange')
    axes[1, 0].set_title('mAP Metrics')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate
    axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='LR', color='purple')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to: {save_path}")
    
    plt.show()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for scale detection')
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--output_dir', type=str, default='runs/detect', help='Output directory for training results')
    parser.add_argument('--model_name', type=str, default='yolov8m.pt', help='Pretrained model name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=1280, help='Input image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for training')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads')
    parser.add_argument('--project', type=str, default='scale_detection', help='Project name')
    parser.add_argument('--name', type=str, default='yolov8m_scale_detection', help='Experiment name')
    parser.add_argument('--export', action='store_true', help='Export model after training')
    parser.add_argument('--export_path', type=str, default='scale_detection_model.onnx', help='Path to save exported model')
    parser.add_argument('--export_format', type=str, default='onnx', help='Export format')
    parser.add_argument('--plot', action='store_true', help='Plot training results')
    # ClearML flags
    parser.add_argument('--clearml', action='store_true', help='Enable ClearML experiment tracking')
    parser.add_argument('--clearml_project', type=str, help='ClearML project name (defaults to --project)')
    parser.add_argument('--clearml_task', type=str, help='ClearML task name (defaults to --name)')
    parser.add_argument('--clearml_output_uri', type=str, help='ClearML output URI for artifacts (e.g., s3://bucket/folder)')
    
    args = parser.parse_args()
    
    # Optional: initialize ClearML Task
    task = None
    if args.clearml:
        try:
            from clearml import Task
            task = Task.init(
                project_name=args.clearml_project or args.project,
                task_name=args.clearml_task or args.name,
                output_uri=args.clearml_output_uri if args.clearml_output_uri else None
            )
        except Exception as e:
            print(f"Warning: ClearML not initialized ({e}). Continue without tracking.")
            task = None

    # Train model
    print("Starting YOLOv8 training...")
    model, results = train_model(
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name
    )
    
    print("Training completed!")

    # Determine run directory from trainer if available
    run_dir = None
    try:
        run_dir = getattr(model.trainer, 'save_dir', None)
    except Exception:
        run_dir = None
    if not run_dir:
        # Fallback to expected structure
        run_dir = os.path.join('runs', 'detect', args.project, args.name)
    run_dir = str(run_dir)
    
    # Export model if requested
    if args.export:
        print("Exporting model...")
        exported_path = export_model(model, args.export_path, format=args.export_format)
    else:
        exported_path = None
    
    # Plot results if requested
    if args.plot:
        print("Plotting training results...")
        plot_path = os.path.join(run_dir, "training_plots.png")
        plot_training_results(run_dir, plot_path)
    else:
        plot_path = None

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
                lr0=0.01,
                weight_decay=0.0005,
                device=args.device,
                workers=args.workers,
                project=args.project,
                name=args.name
            )
            task.connect(overrides)

            # Common artifacts
            results_csv = os.path.join(run_dir, 'results.csv')
            best_weights = os.path.join(run_dir, 'weights', 'best.pt')
            last_weights = os.path.join(run_dir, 'weights', 'last.pt')

            if os.path.exists(results_csv):
                task.upload_artifact('results.csv', artifact_object=results_csv)
            if os.path.exists(best_weights):
                task.upload_artifact('best.pt', artifact_object=best_weights)
            if os.path.exists(last_weights):
                task.upload_artifact('last.pt', artifact_object=last_weights)
            if exported_path and os.path.exists(exported_path):
                task.upload_artifact(Path(exported_path).name, artifact_object=exported_path)
            if plot_path and os.path.exists(plot_path):
                task.upload_artifact('training_plots.png', artifact_object=plot_path)
        except Exception as e:
            print(f"Warning: failed to upload ClearML artifacts: {e}")


if __name__ == "__main__":
    main()