"""
JSON to YOLO Format Conversion Script

This script converts the provided JSON annotations to YOLOv8 format for training
a joint detection model for scale bars and scale text regions.

Classes:
- 0: scale_bar
- 1: scale_text

The script handles:
- Converting polygon points to bounding boxes
- Normalizing coordinates to [0,1] range
- Creating YOLO dataset structure
- Handling edge cases and validation
"""

import json
import os
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import argparse


def points_to_bbox(
        points: List[List[float]], 
        img_width: int, 
        img_height: int
    ) -> Tuple[float, float, float, float]:
    """
    Convert polygon points to YOLO format bounding box.
    
    Args:
        points: List of [x, y] coordinate pairs
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0,1]
    """
    if len(points) < 2:
        raise ValueError("At least 2 points required for bounding box")
    
    # Convert to numpy array for easier manipulation
    points = np.array(points, dtype=np.float32)
    
    # Get bounding box coordinates
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    
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
    
    return x_center, y_center, width, height


def convert_json_to_yolo(json_path: str, label_dir: str, class_mapping: Dict[str, int]) -> bool:
    """
    Convert a single JSON annotation file to YOLO format.
    
    Args:
        json_path: Path to the JSON annotation file
        label_dir: Directory to save YOLO annotation files
        class_mapping: Mapping from class names to class indices
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract image dimensions
        img_width = data.get('width', 0)
        img_height = data.get('height', 0)
        
        if img_width <= 0 or img_height <= 0:
            print(f"Warning: Invalid image dimensions in {json_path}")
            return False
        
        # Prepare YOLO annotations
        yolo_annotations = []
        
        # Process scale bars (class 0)
        for bar in data.get('bars', []):
            if 'points' in bar and len(bar['points']) >= 2:
                try:
                    x_center, y_center, width, height = points_to_bbox(
                        bar['points'], img_width, img_height
                    )
                    # Skip if bbox is too small
                    if width > 0.001 and height > 0.001:
                        yolo_annotations.append(f"{class_mapping['scale_bar']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                except Exception as e:
                    print(f"Warning: Failed to process bar in {json_path}: {e}")
                    continue
        
        # Process scale text (class 1)
        for label in data.get('labels', []):
            if 'points' in label and len(label['points']) >= 2:
                try:
                    x_center, y_center, width, height = points_to_bbox(
                        label['points'], img_width, img_height
                    )
                    # Skip if bbox is too small
                    if width > 0.001 and height > 0.001:
                        yolo_annotations.append(f"{class_mapping['scale_text']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                except Exception as e:
                    print(f"Warning: Failed to process label in {json_path}: {e}")
                    continue
        
        # Write YOLO annotation file
        json_filename = os.path.basename(json_path)
        txt_filename = json_filename.replace('.json', '.txt')
        txt_path = os.path.join(label_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        return True
        
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return False


def create_dataset_yaml(data_dir: str, class_names: List[str], train_split: float = 0.8) -> str:
    """
    Create YOLO dataset configuration file.
    
    Args:
        data_dir: Directory containing the dataset
        class_names: List of class names
        train_split: Fraction of data to use for training
        
    Returns:
        Path to the created YAML file
    """
    yaml_path = os.path.join(data_dir, 'data.yaml')
    
    # Get list of all annotation files
    annotation_files = [f for f in os.listdir(data_dir / 'labels') if f.endswith('.txt')]
    annotation_files.sort()
    
    # Split into train/val
    n_train = int(len(annotation_files) * train_split)
    train_files = annotation_files[:n_train]
    val_files = annotation_files[n_train:]
    
    # Create dataset structure
    image_train_dir = data_dir / 'images' / 'train'
    image_val_dir = data_dir / 'images' / 'val'
    label_train_dir = data_dir / 'labels' / 'train'
    label_val_dir = data_dir / 'labels' / 'val'
    os.makedirs(image_train_dir, exist_ok=True)
    os.makedirs(image_val_dir, exist_ok=True)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    dataset_config = {
        'path': os.path.abspath(data_dir),
        'train': os.path.abspath(image_train_dir),
        'val': os.path.abspath(image_val_dir),
        'test': None,
        'nc': len(class_names),
        'names': class_names
    }
    
    # Move files
    for txt_file in train_files:
        img_file = txt_file.replace('.txt', '.jpg')
        os.rename(os.path.join(data_dir / 'images', img_file), os.path.join(image_train_dir, img_file))
        os.rename(os.path.join(data_dir / 'labels', txt_file), os.path.join(label_train_dir, txt_file))
    for txt_file in val_files:
        img_file = txt_file.replace('.txt', '.jpg')
        os.rename(os.path.join(data_dir / 'images', img_file), os.path.join(image_val_dir, img_file))
        os.rename(os.path.join(data_dir / 'labels', txt_file), os.path.join(label_val_dir, txt_file))

    # Write YAML config
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    return yaml_path


def convert_dataset(data_dir: Path, train_split: float = 0.8) -> str:
    """
    Convert entire dataset from JSON to YOLO format.
    
    Args:
        data_dir: Directory containing 'images', 'jsons', and where 'labels' will be created
        train_split: Fraction of data to use for training
        
    Returns:
        Path to the created dataset YAML file
    """
    # Configure and create label directory
    json_dir = data_dir / 'jsons'
    image_dir = data_dir / 'images'
    label_dir = data_dir / 'labels'
    os.makedirs(label_dir, exist_ok=True)

    # Class mapping
    class_mapping = {
        'scale_bar': 0,
        'scale_text': 1
    }
    
    class_names = ['scale_bar', 'scale_text']
    
    # Get all JSON files
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    json_files.sort()
    
    print(f"Found {len(json_files)} JSON annotation files")
    
    # Convert each JSON file
    successful_conversions = 0
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        if convert_json_to_yolo(json_path, label_dir, class_mapping):
            successful_conversions += 1
    
    print(f"Successfully converted {successful_conversions}/{len(json_files)} files")
    
    # Create dataset YAML
    yaml_path = create_dataset_yaml(data_dir, class_names, train_split)
    print(f"Created dataset configuration: {yaml_path}")
    
    return yaml_path


def validate_conversion(label_dir: str, sample_size: int = -1) -> Dict[str, float]:
    """
    Validate the conversion by checking a sample of files.
    
    Args:
        label_dir: Directory containing YOLO annotations
        sample_size: Number of files to check (-1 for all)
        
    Returns:
        Dictionary with validation statistics
    """
    txt_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    sample_files = txt_files[:sample_size] if sample_size != -1 else txt_files
    
    stats = {
        'total_files': len(txt_files),
        'sample_size': len(sample_files),
        'files_with_annotations': 0,
        'total_scale_bars': 0,
        'total_scale_text': 0,
        'avg_annotations_per_file': 0
    }
    
    for txt_file in sample_files:
        txt_path = os.path.join(label_dir, txt_file)
        
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            if lines:
                stats['files_with_annotations'] += 1
                stats['total_scale_bars'] += sum(1 for line in lines if line.startswith('0 '))
                stats['total_scale_text'] += sum(1 for line in lines if line.startswith('1 '))
        
        except Exception as e:
            print(f"Error validating {txt_file}: {e}")
    
    stats['avg_annotations_per_file'] = (stats['total_scale_bars'] + stats['total_scale_text']) / max(stats['files_with_annotations'], 1)
    
    return stats


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Convert JSON annotations to YOLO format')
    parser.add_argument('--json_dir', type=str, required=True, help='Directory containing JSON annotation files')
    parser.add_argument('--label_dir', type=str, required=True, help='Label directory for YOLO format files')
    parser.add_argument('--train_split', type=float, default=0.8, help='Fraction of data to use for training')
    parser.add_argument('--validate', action='store_true', help='Run validation after conversion')
    
    args = parser.parse_args()
    
    # Convert dataset
    yaml_path = convert_dataset(args.json_dir, args.label_dir, args.train_split)
    
    # Run validation if requested
    if args.validate:
        print("\nRunning validation...")
        stats = validate_conversion(args.label_dir)
        print(f"Validation results:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Files with annotations: {stats['files_with_annotations']}")
        print(f"  Total scale bars: {stats['total_scale_bars']}")
        print(f"  Total scale text: {stats['total_scale_text']}")
        print(f"  Average annotations per file: {stats['avg_annotations_per_file']:.2f}")


if __name__ == "__main__":
    main()
