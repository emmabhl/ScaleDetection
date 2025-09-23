#!/usr/bin/env python3
"""
Test script for the updated YOLO + OCR pipeline.

This script demonstrates how to use the modified pipeline that:
1. Uses YOLO to detect both scale bars and text labels
2. Applies OCR specifically to text label bounding boxes
3. Matches detected text to nearby scale bars
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from src.ocr_and_match import ScaleDetectionPipeline
from ultralytics import YOLO

def test_yolo_ocr_pipeline():
    """Test the updated YOLO + OCR pipeline."""
    
    # Configuration
    model_path = 'models/train/weights/best.pt'
    test_image_path = 'data_small/images/train/2.jpg'  # Use small dataset for testing
    output_dir = 'output'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using the pipeline notebook.")
        return
    
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        return
    
    print("=" * 60)
    print("YOLO + OCR PIPELINE TEST")
    print("=" * 60)
    
    # Load YOLO model
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    # Load test image
    print(f"Loading test image from {test_image_path}...")
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image.shape}")
    
    # Initialize OCR pipeline
    print("Initializing OCR pipeline...")
    pipeline = ScaleDetectionPipeline(
        ocr_backend='paddle',
        confidence_threshold=0.15,
        max_distance_ratio=1.5
    )
    
    # Run YOLO detection
    print("Running YOLO detection...")
    yolo_results = model.predict(image, conf=0.25, verbose=False)
    
    # Print YOLO detection results
    boxes = yolo_results[0].boxes
    if boxes is not None:
        print(f"YOLO detected {len(boxes)} objects:")
        for i, (box, score, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
            x1, y1, x2, y2 = box.cpu().numpy()
            class_name = "scale_bar" if cls == 0 else "text_label"
            print(f"  {i+1}. {class_name}: confidence={score:.3f}, bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    else:
        print("No objects detected by YOLO")
        return
    
    # Process with updated pipeline
    print("\nProcessing with updated OCR pipeline...")
    results = pipeline.process_yolo_detections(image, yolo_results[0])
    
    # Print results
    print("\n" + "=" * 40)
    print("PIPELINE RESULTS")
    print("=" * 40)
    print(f"YOLO scale bars: {results.get('yolo_scale_bars', 0)}")
    print(f"YOLO text labels: {results.get('yolo_text_labels', 0)}")
    print(f"Text detections from OCR: {results['total_text_detections']}")
    print(f"Scale bar detections: {results['total_bar_detections']}")
    print(f"Successful matches: {results['successful_matches']}")
    
    # Show detailed matches
    if results['matches']:
        print(f"\nDetailed matches:")
        for i, match in enumerate(results['matches']):
            print(f"\nMatch {i+1}:")
            print(f"  Text: '{match.text.text}'")
            print(f"  Parsed: {match.text.parsed_value} {match.text.normalized_unit}")
            print(f"  Confidence: {match.text.confidence:.3f}")
            print(f"  Distance to bar: {match.distance:.2f} pixels")
            if match.um_per_pixel:
                print(f"  Scale: {match.um_per_pixel:.6f} um/pixel")
    else:
        print("\nNo successful matches found.")
    
    # Save results
    output_path = os.path.join(output_dir, 'yolo_ocr_test_results.json')
    pipeline.save_results(results, output_path)
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_yolo_ocr_pipeline()
