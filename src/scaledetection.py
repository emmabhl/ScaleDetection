"""
Scale detection + OCR pipeline

This script implements the end-to-end pipeline used to detect scale bars,
apply post-processing to localize endpoints, run OCR on nearby text labels,
and compute pixel-to-mm ratios. It integrates template classification for
atypical scale bars, a YOLO detector for standard cases, and the
post-processing/OCR modules.

Example (complete call):
    python src/scaledetection.py \
        --image data/images/val/9.jpg \
        --model models/yolov8m_train/weights/best.pt \
        --output_dir outputs --plot

This file exposes a `ScaleDetectionPipeline` class for programmatic use
and a `main()` entry that supports processing single images or directories.
"""
import argparse
import glob
import json
import logging as log
import math
import os
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForObjectDetection
from ultralytics import YOLO

from atypical_scalebars import (
    extract_black_vertical_lines,
    extract_white_horizontal_shape,
)
from classifier import ScaleBarClassifier
from ocr import OCRProcessor, LabelDetection
from postprocess_scalebar import ScalebarDetection, ScalebarProcessor
from utils import ensure_model_available


@dataclass
class Scale:
    """Data class for matched scale bar and label."""
    scale_bar_found: bool
    measured_scale_length: Optional[float] = None
    declared_scale_length: Optional[float] = None
    units: Optional[str] = None
    pixel_to_mm_ratio: Optional[float] = None
    orientation: Optional[str] = None
    scale_bar_confidence: Optional[float] = None
    #scale_length_uncertainty: Optional[float] = None
    scale_length_flag: Optional[bool] = False
    text_label_confidence: Optional[float] = None
    orientation_confidence: Optional[float] = None
    type_: str = "normal"  # normal, graduation_middleunit, ruler_photo


class ScaleDetectionPipeline:
    """Complete pipeline for scale detection and OCR matching."""

    def __init__(
            self, 
            max_distance_ratio: float = 1.5,
            debug_dir: Optional[str] = None,
        ):
        """Initialize the pipeline.

        Args:
            max_distance_ratio: Maximum distance ratio for text-bar matching
            debug_dir: Directory to save debug plots (if None, no plots are saved)
        """
        self.scalebar_processor = ScalebarProcessor()
        self.ocr_processor = OCRProcessor()
        self.max_distance_ratio = max_distance_ratio
        self.debug_dir = debug_dir
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)


    def process_yolo_detections(
        self, image: np.ndarray, detection_results: object, plot_path: str
    ) -> Scale:
        """Process YOLO detections: post-process scalebars, run OCR and compute scale.

        Args:
            image (np.ndarray): Full input image (H,W,C or H,W).
            detection_results (object): YOLO `Results` object for the image.
            plot_path (str): Path prefix to save per-image debug visualizations.

        Returns:
            result (Scale): Data class containing the matched scale information.
        """
        # Extract bounding boxes, scores, and classes from YOLO results
        boxes = detection_results.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
        classes = detection_results.boxes.cls.cpu().numpy()  # Class IDs
        scores = detection_results.boxes.conf.cpu().numpy()  # Confidence scores

        # Separate bar and label detections
        bar_box, label_box, bar_score, label_score = self.match_text_to_bars(boxes, classes, scores)

        if bar_score == 0.0 or label_score == 0.0:
            return Scale(
                scale_bar_found=False,
                scale_bar_confidence=float(bar_score),
                text_label_confidence=float(label_score),
                orientation=None,
                orientation_confidence=1.0
            )

        # ---------- SCALEBAR DETECTION & MEASUREMENT ----------
        bar_box = self.prepare_bbox(image, bar_box, padding_percent=0.1)
        scalebar_detection = self.scalebar_processor.localize_scalebar_endpoints(
            image, bar_box, plot_path=plot_path
        )

        # --------- TEXT LABEL DETECTION & RECOGNITION ---------
        label_box = self.prepare_bbox(image, label_box, padding_percent=0.1)
        label_detection = self.ocr_processor.extract_text_labels(
            image, label_box, reversed=False, plot_path=plot_path
        )

        # ----------- MATCHING & RESULTS PREPARATION -----------
        scale = self.get_scale(scalebar_detection, label_detection, bar_score, label_score)

        return scale


    def process_atypical_detections(
            self, image: np.ndarray, detection_results: Dict[str, Any], plot_path: str
        ) -> Scale:
        """Process atypical (template-classified) detections and return scale results.

        Args:
            image (np.ndarray): Full input image (H,W,C or H,W).
            detection_results (Dict[str,Any]): Template classification result dict.
            plot_path (str): Path prefix to save debug visualizations.

        Returns:
            result (Scale): Data class with the detection, measurement and metadata.
        """
        bbox = detection_results['bbox']
        cls_score = float(detection_results['score'])

        if detection_results['scale_type'] == 'graduation_middleunit':
            # ---------- SCALEBAR DETECTION & MEASUREMENT ----------
            scalebar_detection = extract_white_horizontal_shape(image, bbox, plot_path=plot_path)

            # --------- TEXT LABEL DETECTION & RECOGNITION ---------
            bbox_xywh = (bbox[0][0], bbox[0][1], bbox[2][0]-bbox[0][0], bbox[2][1]-bbox[0][1])
            label_detection = self.ocr_processor.extract_text_labels(
                image, bbox_xywh, reversed=True, plot_path=plot_path
            )

            # ----------- MATCHING & RESULTS PREPARATION -----------
            scale = self.get_scale(scalebar_detection, label_detection, cls_score, cls_score)

        elif detection_results['scale_type'] == 'ruler_photo':
            avg_distance = extract_black_vertical_lines(image, bbox, plot_path=plot_path)

            if avg_distance == 0.0:
                scale = Scale(
                    scale_bar_found=False, scale_bar_confidence=cls_score,
                    text_label_confidence=cls_score, orientation_confidence=1.0
                )
            else:                
                scale = Scale(
                    scale_bar_found=True,
                    measured_scale_length=float(10 * avg_distance),
                    declared_scale_length=1.0,
                    units='cm',
                    pixel_to_mm_ratio=1.0 / avg_distance,
                    scale_bar_confidence=cls_score,
                    text_label_confidence=cls_score,
                    orientation_confidence=1.0,
                )

        else:
            raise ValueError(f"Unknown atypical scale bar type: {detection_results['scale_type']}")

        scale.type_ = detection_results['scale_type']
        return scale


    def prepare_bbox(
        self, image: np.ndarray, box: np.ndarray, padding_percent: float
    ) -> Tuple[int, int, int, int]:
        """Convert an (x_min,y_min,x_max,y_max) box to a padded (x,y,w,h) ROI.

        Args:
            image (np.ndarray): Full input image used to clip padding to image bounds.
            box (np.ndarray): YOLO box in (x_min, y_min, x_max, y_max) format.
            padding_percent (float): Fractional padding to add around the detected box.

        Returns:
            roi (Tuple[int,int,int,int]): Padded ROI as (x, y, w, h).
        """
        # Convert box to (x, y, w, h) format
        x_min, y_min, x_max, y_max = box

        # Increase ROI to include some context and ensure full scale bar is captured
        h = y_max - y_min
        w = x_max - x_min
        hpad = padding_percent * h
        vpad = padding_percent * w
        x = int(max(0, x_min - vpad))
        y = int(max(0, y_min - hpad))
        w = int(min(image.shape[1] - x, w + 2 * vpad))
        h = int(min(image.shape[0] - y, h + 2 * hpad))

        return (x, y, w, h)


    def match_text_to_bars(
            self, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Match detected text boxes to detected scale bar boxes using spatial proximity.

        Args:
            boxes (np.ndarray): Bounding boxes in YOLO xyxy format (N,4).
            classes (np.ndarray): Class ids per box (N,).
            scores (np.ndarray): Confidence scores per box (N,).

        Returns:
            matched_bar (np.ndarray): Matched bar box in xyxy format or zeros if none.
            matched_label (np.ndarray): Matched label box in xyxy format or zeros if none.
            bar_score (float): Confidence of the matched bar.
            label_score (float): Confidence of the matched label.
        """
        # Helper functions
        def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
            """Calculate Euclidean distance between two points."""
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

        def calculate_bar_diagonal(bbox: np.ndarray) -> float:
            """Calculate diagonal length of bounding box."""
            x, y, w, h = bbox
            return math.sqrt(w**2 + h**2)

        # Separate bar and label detections
        bar_boxes = boxes[classes == 0]
        label_boxes = boxes[classes == 1]
        bar_scores = scores[classes == 0]
        label_scores = scores[classes == 1]


        if len(label_boxes) == 0 or len(bar_boxes) == 0:
            if len(label_boxes) == 0:
                log.info("No text labels detected.")
            if len(bar_boxes) == 0:
                log.info("No scale bars detected.")
            return np.zeros(4), np.zeros(4), 0.0, 0.0

        elif len(label_boxes) == 1 and len(bar_boxes) == 1:
            # Check if the single text label is close enough to the single scale bar
            label_box = label_boxes[0]
            label_center = ((label_box[0] + label_box[2]) / 2, (label_box[1] + label_box[3]) / 2)
            bar_box = bar_boxes[0]
            bar_center = ((bar_box[0] + bar_box[2]) / 2, (bar_box[1] + bar_box[3]) / 2)
            distance = calculate_distance(label_center, bar_center)
            bar_diagonal = calculate_bar_diagonal(bar_box)
            max_distance = bar_diagonal * self.max_distance_ratio

            if distance <= max_distance:
                return bar_box, label_box, bar_scores[0], label_scores[0]
            else:
                return np.zeros(4), np.zeros(4), 0.0, 0.0

        else:
            # Multiple bars and labels: check if the most confident ones are close enough
            sorted_bar_indices = np.argsort(-bar_scores)
            sorted_label_indices = np.argsort(-label_scores)
            for label_idx in sorted_label_indices:
                label_box = label_boxes[label_idx]
                label_center = ((label_box[0] + label_box[2])/ 2, (label_box[1] + label_box[3])/ 2)
                for bar_idx in sorted_bar_indices:
                    bar_box = bar_boxes[bar_idx]
                    bar_center = ((bar_box[0] + bar_box[2]) / 2, (bar_box[1] + bar_box[3]) / 2)
                    bar_diagonal = calculate_bar_diagonal(bar_box)
                    max_distance = bar_diagonal * self.max_distance_ratio
                    distance = calculate_distance(label_center, bar_center)

                    if distance <= max_distance:
                        bar_score = float(bar_scores[bar_idx])
                        label_score = float(label_scores[label_idx])
                        return bar_box, label_box, bar_score, label_score
            return np.zeros(4), np.zeros(4), 0.0, 0.0


    def get_scale(
        self,
        scalebar_detection: ScalebarDetection,
        label_detection: LabelDetection,
        bar_score: float,
        label_score: float
    ) -> Scale:
        """Compute pixel-to-mm conversion from a scalebar detection and parsed label.

        Args:
            scalebar_detection (ScalebarDetection): Result from endpoint localization.
            label_detection (LabelDetection): Parsed OCR label detection.
            bar_score (float): Confidence score for the scalebar detection.
            label_score (float): Confidence score for the text label detection.

        Returns:
            scale (Scale): Data class containing measured and declared scale, units and ratio.
        """
        px_len = scalebar_detection.pixel_length
        value = label_detection.parsed_value
        unit = label_detection.normalized_unit

        # --- Early exit if scalebar or label is invalid ---
        if not px_len or label_detection.confidence == 0.0:
            return Scale(
                scale_bar_found=False,
                measured_scale_length=float(px_len) if px_len else None,
                declared_scale_length=float(value) if value else None,
                units=unit,
                pixel_to_mm_ratio=0.0,
                scale_bar_confidence=float(bar_score),
                scale_length_flag=scalebar_detection.flag,
                text_label_confidence=float(label_score),
                orientation_confidence=1.0,
            )

        # --- Unit conversion dictionary (to mm) ---
        unit_to_mm = {
            "mm": 1,
            "cm": 10,
            "um": 1e-3,
            "nm": 1e-6,
        }

        # Ensure unit is a string (not None) and normalize to lowercase before lookup
        unit_key = (unit or "").lower()
        mm_factor = unit_to_mm.get(unit_key, 0.0)
        mm_per_pixel = (value * mm_factor) / px_len if mm_factor > 0 else 0.0

        return Scale(
            scale_bar_found=bool(mm_per_pixel > 0),
            measured_scale_length=float(px_len),
            declared_scale_length=float(value) if value else None,
            units=unit,
            pixel_to_mm_ratio=float(mm_per_pixel),
            scale_bar_confidence=float(bar_score),
            text_label_confidence=float(label_score),
            orientation_confidence=1.0,
        )


    def save_results(self, results: Scale, output_path: str) -> None:
        """Save a `Scale` dataclass instance to a JSON file.

        Args:
            results (Scale): Dataclass with the pipeline results.
            output_path (str): Destination JSON file path.

        Returns:
            None: Writes JSON to `output_path`.
        """
        try:
            if not is_dataclass(results):
                raise TypeError("Expected a dataclass instance of type Scale.")

            with open(output_path, 'w') as f:
                json.dump(asdict(results), f, indent=4)
            log.info(f"Saved results to {output_path}")

        except Exception as e:
            log.error(f"Failed to save results to {output_path}: {e}")


def main():
    """CLI entry for running the scale detection pipeline on one or many images."""
    parser = argparse.ArgumentParser(description='OCR and scale matching pipeline')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to input image (single)')
    group.add_argument('--image_dir', type=str, help='Path to folder containing images to process')

    parser.add_argument('--model', type=str, default='models/yolov8m_train/weights/best.pt',
                        help='Path to YOLO model for detection')
    parser.add_argument('--atypical_data', type=str, default=None,
                        help='Path to precomputed ORB descriptors for scale bar classification')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Path to save output and visualization')
    parser.add_argument('--max_distance', type=float, default=1.5,
                        help='Maximum distance ratio for text-bar matching')
    parser.add_argument('--classification_threshold', type=float, default=0.42,
                        help='Threshold for scale bar image classification')
    parser.add_argument('--yolo_conf', type=float, default=0.01,
                        help='YOLO confidence threshold for detections')
    parser.add_argument('--classifier_score_threshold', type=float, default=0.15,
                        help='Threshold for reporting scale bar classification results')
    parser.add_argument('--nfeatures_ORB', type=int, default=1000,
                        help='Number of ORB features for scale bar classification')
    parser.add_argument('--plot', action='store_true', 
                        help='Plot intermediate results and save debug images')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Enable verbose logging output')

    args = parser.parse_args()
    
    # Set up logging
    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Build list of images to process
    image_paths = []
    if args.image:
        image_paths = [args.image]
    else:
        # Accept common image extensions
        exts = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp')
        for e in exts:
            image_paths.extend(sorted(glob.glob(os.path.join(args.image_dir, e))))
        if len(image_paths) == 0:
            log.warning(f"No images found in directory {args.image_dir} with extensions: {exts}.")
            return

    # Initialize heavy objects once
    clf = ScaleBarClassifier(
        score_threshold=args.classifier_score_threshold,
        nfeatures=args.nfeatures_ORB,
        atypical_data_path=args.atypical_data,
    )

    # Load YOLO model once
    local_model_path = ensure_model_available(
        local_path=args.model,
        repo_id='emmabhl/yolov8m-ScalebarDetection',
        filename='best.pt'
    )
    model = YOLO(local_model_path)

    pipeline = ScaleDetectionPipeline(
        max_distance_ratio=args.max_distance,
        debug_dir=args.output_dir if args.plot else None,
    )

    # Process images sequentially
    for img_path in tqdm(image_paths, desc="Processing images"):
        image = cv2.imread(img_path)
        if image is None:
            log.warning(f"Could not read image {img_path}, skipping.")
            continue

        # Keep your previous conversion to RGB (YOLO/other modules may expect RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_name = os.path.splitext(os.path.basename(img_path))[0]
        # per-image base for plots and outputs
        plot_base = os.path.join(args.output_dir, image_name)
        os.makedirs(args.output_dir, exist_ok=True)  # ensure main output dir exists

        # 1) Classification (atypical scale bars)
        matches = clf.classify_scale_bar(image)

        # Optionally save classification results per-image
        if args.plot:
            clf.save_results(
                image,
                matches,
                os.path.join(args.output_dir, f'{image_name}_classification.png')
            )

        # 2) If no atypical matches -> run YOLO standard detection + pipeline
        if not matches:
            log.info("No atypical scale bar detected, proceeding with normal detection")
            try:
                yolo_results = model.predict(image, conf=args.yolo_conf, verbose=False)
            except Exception as e:
                log.error(f"Error running YOLO on {img_path}: {e}")
                continue

            # Save YOLO visualization if requested
            if args.plot:
                try:
                    # yolo_results[0].save expects a path; we'll use per-image base
                    yolo_results[0].save(plot_base + '_yolo.png')
                except Exception as e:
                    log.warning(f"Failed to save YOLO visualization: {e}")

            # Process YOLO detections
            try:
                results = pipeline.process_yolo_detections(image, yolo_results[0], plot_base)
            except Exception as e:
                log.error(f"Error processing YOLO detections for {img_path}: {e}")
                continue

        else:
            log.info("Atypical scale bar detected, proceeding with specialized processing")

            # Process the best detected atypical scale bar
            try:
                results = pipeline.process_atypical_detections(image, matches[0], plot_base)
            except Exception as e:
                log.error(f"Error processing atypical detection for {img_path}: {e}")
                continue

        # Save the results JSON
        pipeline.save_results(results, os.path.join(args.output_dir, f'{image_name}.json'))

    log.info("Processing complete.")

if __name__ == "__main__":
    main()
