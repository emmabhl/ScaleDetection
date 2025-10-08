"""
Pipeline for scale detection and OCR matching.
"""
import os
import json
import cv2
import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from ocr import OCRProcessor, LabelDetection
from postprocess_scalebar import ScalebarProcessor, ScalebarDetection


@dataclass
class Scale:
    """Data class for matched scale bar and label."""
    scalebar_length: float # in pixels
    text_label: str
    mm_per_pixel: Optional[float] = None
    confidence: Optional[float] = None


class ScaleDetectionPipeline:
    """Complete pipeline for scale detection and OCR matching."""
    
    def __init__(
            self, 
            max_distance_ratio: float = 1.5,
            ocr_version: str = 'PP-OCRv5',
            debug_dir: Optional[str] = None,
        ):
        """
        Initialize the pipeline.
        
        Args:
            ocr_backend: OCR backend to use
            confidence_threshold: Minimum confidence for text detection
            max_distance_ratio: Maximum distance ratio for text-bar matching
            debug_dir: Directory to save debug plots (if None, no plots are saved)
        """
        self.scalebar_processor = ScalebarProcessor()
        self.ocr_processor = OCRProcessor(ocr_version=ocr_version)
        self.max_distance_ratio = max_distance_ratio
        self.debug_dir = debug_dir
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)


    def process_yolo_detections(
            self, image: np.ndarray, yolo_results, image_name: str
        ) -> Dict[str, Any]:
        """
        Process YOLO detection results to extract scale bars and text labels, then apply 
        post-processing and OCR to get the final scale.
        
        Args:
            image: Input image
            yolo_results: YOLO detection results (ultralytics Results object)
            image_name: Name of the image (for saving debug plots)
            
        Returns:
            Dictionary containing results
        """
        plot_path = os.path.join(self.debug_dir, f'{image_name}') if self.debug_dir else None

        # Extract bounding boxes, scores, and classes from YOLO results
        boxes = yolo_results.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
        classes = yolo_results.boxes.cls.cpu().numpy()  # Class IDs
        scores = yolo_results.boxes.conf.cpu().numpy()  # Confidence scores

        # Separate bar and label detections
        bar_box, label_box = self.match_text_to_bars(boxes, classes, scores)

        # ---------- SCALEBAR DETECTION & MEASUREMENT ----------
        bar_box = self.prepare_bbox(image, bar_box, padding_percent=0.1)
        scalebar_detection = self.scalebar_processor.localize_scalebar_endpoints(
            image, bar_box, plot_path=plot_path
        )

        # --------- TEXT LABEL DETECTION & RECOGNITION ---------
        label_box = self.prepare_bbox(image, label_box, padding_percent=0.1)
        label_detection = self.ocr_processor.extract_text_labels(
            image, label_box, plot_path=plot_path
        )

        # ----------- MATCHING & RESULTS PREPARATION -----------
        if scalebar_detection.confidence == 0.0 or label_detection.confidence == 0.0:
            if scalebar_detection.confidence == 0.0:
                print("No valid scale bar after post-processing.")
            if label_detection.confidence == 0.0:
                print("No valid text label after OCR.")
            return {}
        
        scale = self.get_scale(scalebar_detection, label_detection)

        results = {
            'scalebar_detection': scalebar_detection,
            'label_detection': label_detection,
            'scale': scale
        }
        return results

    def prepare_bbox(
            self, image: np.ndarray, box: np.ndarray, padding_percent: float
        ) -> Tuple[int, int, int, int]:
        """
        Prepare bounding boxes for scale bar and text label from YOLO outputs.
        Args:
            image: Input image
            box: Bounding box in (x_min, y_min, x_max, y_max) format detected by YOLO
            padding_percent: Percentage of padding to add around the box
        Returns:
            Bounding box in (x, y, w, h) format with padding
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
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match text detections to scale bar detections.

        Args:
            boxes: Array of bounding boxes in xyxy format
            classes: Array of class IDs
            scores: Array of confidence scores

        Returns:
            Tuple of matched scale bar box and text label boxs in (x, y, w, h) format
        """
        # Helper functions
        def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
            """Calculate Euclidean distance between two points."""
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

        def calculate_bar_diagonal(bbox: Tuple[int, int, int, int]) -> float:
            """Calculate diagonal length of bounding box."""
            x, y, w, h = bbox
            return math.sqrt(w**2 + h**2)
        
        # Separate bar and label detections
        bar_boxes = boxes[classes == 0]
        label_boxes = boxes[classes == 1]
        

        if len(label_boxes) == 0 or len(bar_boxes) == 0:
            if len(label_boxes) == 0:
                print("No text labels detected.")
            if len(bar_boxes) == 0:
                print("No scale bars detected.")
            return [0, 0, 0, 0], [0, 0, 0, 0]
        
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
                return bar_boxes[0], label_boxes[0]
            else:
                return [0, 0, 0, 0], [0, 0, 0, 0]

        elif len(label_boxes) == 1:
            # Match the single text label to the closest scale bar
            label_box = label_boxes[0]
            label_center = ((label_box[0] + label_box[2]) / 2, (label_box[1] + label_box[3]) / 2)

            best_match = None
            best_distance = float('inf')
            for bar in bar_boxes:
                bar_center = ((bar[0] + bar[2]) / 2, (bar[1] + bar[3]) / 2)
                distance = calculate_distance(label_center, bar_center)
                bar_diagonal = calculate_bar_diagonal(bar)
                max_distance = bar_diagonal * self.max_distance_ratio
                
                if distance < best_distance and distance <= max_distance:
                    best_match = bar
                    best_distance = distance

            return best_match, label_box

        else:
            # Multiple labels and scale bars, take the most confident bar and match to closest label
            bar_box = bar_boxes[np.argmax(scores[classes == 0])]
            bar_center = ((bar_box[0] + bar_box[2]) / 2, (bar_box[1] + bar_box[3]) / 2)
            bar_diagonal = calculate_bar_diagonal(bar_box)
            max_distance = bar_diagonal * self.max_distance_ratio

            best_match = None
            best_distance = float('inf')
            for label in label_boxes:
                label_center = ((label[0] + label[2]) / 2, (label[1] + label[3]) / 2)
                distance = calculate_distance(label_center, bar_center)

                if distance < best_distance and distance <= max_distance:
                    best_match = label
                    best_distance = distance

            return bar_box, best_match


    def get_scale(
        self, scalebar_detection: ScalebarDetection, label_detection: LabelDetection,
    ) -> Scale:
        """
        Get the scale from the detected scale bar and recognized label.
        """
        if label_detection.normalized_unit == 'mm':
            mm_per_pixel = label_detection.parsed_value / scalebar_detection.pixel_length
        elif label_detection.normalized_unit == 'cm':
            mm_per_pixel = (label_detection.parsed_value * 10) / scalebar_detection.pixel_length
        elif label_detection.normalized_unit == 'um':
            mm_per_pixel = (label_detection.parsed_value / 1e3) / scalebar_detection.pixel_length
        elif label_detection.normalized_unit == 'nm':
            mm_per_pixel = (label_detection.parsed_value / 1e6) / scalebar_detection.pixel_length
        else:
            mm_per_pixel = None
    
        return Scale(
            scalebar_length=scalebar_detection.pixel_length,
            text_label=label_detection.text,
            mm_per_pixel=mm_per_pixel,
            confidence=min(scalebar_detection.confidence, label_detection.confidence)
        )

    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary
            output_path: Path to save JSON file
        """
        # Convert results to JSON-serializable format
        def serialize(obj):
            if isinstance(obj, (ScalebarDetection, LabelDetection, Scale)):
                return obj.__dict__
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError("Type not serializable")
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, default=serialize, indent=4)


def main():
    """Main function for command-line usage."""
    import argparse
    from ultralytics import YOLO
    
    parser = argparse.ArgumentParser(description='OCR and scale matching pipeline')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/train/weights/best.pt', 
                        help='Path to YOLO model for detection')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Path to save output and visualization')
    parser.add_argument('--confidence', type=float, default=0.15, 
                       help='Minimum confidence for text detection')
    parser.add_argument('--max_distance', type=float, default=1.5,
                       help='Maximum distance ratio for text-bar matching')
    parser.add_argument('--yolo_conf', type=float, default=0.15,
                       help='YOLO confidence threshold for detections')
    parser.add_argument('--ocr_version', type=str, default='PP-OCRv5',
                       help='Version of OCR model for text recognition')
    parser.add_argument('--plot_debug', action='store_true', help='Plot intermediate results')
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare output directory
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    os.makedirs(args.output_dir, exist_ok=True)
        
    # Initialize pipeline
    pipeline = ScaleDetectionPipeline(
        max_distance_ratio=args.max_distance,
        ocr_version=args.ocr_version,
        debug_dir=args.output_dir if args.plot_debug else None,
    )

    # Use YOLO model for detection
    model = YOLO(args.model)
    # yolo_results = model.predict(source=image, imgsz=(image.shape[1], image.shape[0]), conf=0.4, device='mps', half=True, save_conf=True)
    yolo_results = model.predict(image, conf=args.yolo_conf, verbose=False)

    # Save YOLO detection visualization
    if args.plot_debug:
        yolo_results[0].save(args.output_dir + f'/{image_name}_yolo.jpg')

    # Process YOLO detections
    results = pipeline.process_yolo_detections(image, yolo_results[0], image_name)
    
    # Save results
    pipeline.save_results(results, args.output_dir + f'/{image_name}.json')
    
    # Print summary
    print(f"Processed image: {args.image}")
    print(f"Label detections: {results.get('label_detection')}")
    print(f"Scalebar detections: {results.get('scalebar_detection')}")
    print(f"Final scale: {results.get('scale')}")

if __name__ == "__main__":
    main()