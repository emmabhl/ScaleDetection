"""
OCR and Text-to-Bar Matching System

This module implements OCR using PaddleOCR for scale text recognition and
matches detected text regions to their corresponding scale bars.

Features:
- PaddleOCR integration with scientific symbol support
- Robust text parsing with regex patterns
- Unit normalization (μm/um/µm → um, nm → nm, mm → mm)
- Spatial matching between text and scale bars
- Confidence-based filtering
- Fallback handling for ambiguous cases
"""

import cv2
import numpy as np
import re
import json
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import math
from dataclasses import dataclass
import logging
from paddleocr import PaddleOCR

@dataclass
class TextDetection:
    """Data class for text detection results."""
    text: str
    confidence: float
    bbox: List[List[int]]  # 4 points of bounding box
    center: Tuple[float, float]
    parsed_value: Optional[float] = None
    parsed_unit: Optional[str] = None
    normalized_unit: Optional[str] = None


@dataclass
class ScaleBarDetection:
    """Data class for scale bar detection results."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]
    confidence: float
    pixel_length: Optional[float] = None
    endpoints: Optional[List[Tuple[float, float]]] = None


@dataclass
class MatchedScale:
    """Data class for matched scale bar and text."""
    scale_bar: ScaleBarDetection
    text: TextDetection
    distance: float
    um_per_pixel: Optional[float] = None


class TextParser:
    """Text parsing utilities for scale values and units."""
    
    # Regex patterns for different unit formats
    UNIT_PATTERNS = {
        'micrometer': r'([0-9]+(?:[.,][0-9]+)?)\s*(μm|um|µm|μ|u|micrometer|micrometre)',
        'nanometer': r'([0-9]+(?:[.,][0-9]+)?)\s*(nm|nanometer|nanometre)',
        'millimeter': r'([0-9]+(?:[.,][0-9]+)?)\s*(mm|millimeter|millimetre)',
        'meter': r'([0-9]+(?:[.,][0-9]+)?)\s*(m|meter|metre)',
        'generic': r'([0-9]+(?:[.,][0-9]+)?)\s*([a-zA-Zμµ]+)'
    }
    
    # Unit normalization mapping
    UNIT_NORMALIZATION = {
        'μm': 'um', 'µm': 'um', 'μ': 'um', 'u': 'um',
        'micrometer': 'um', 'micrometre': 'um',
        'nm': 'nm', 'nanometer': 'nm', 'nanometre': 'nm',
        'mm': 'mm', 'millimeter': 'mm', 'millimetre': 'mm',
        'm': 'mm', 'meter': 'mm', 'metre': 'mm'  # Convert meters to mm
    }
    
    @classmethod
    def parse_text(cls, text: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """
        Parse text to extract numeric value and unit.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (value, original_unit, normalized_unit)
        """
        text = text.strip()
        
        # Try each pattern in order of specificity
        for unit_type, pattern in cls.UNIT_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = match.group(1)
                unit = match.group(2)
                
                # Convert value to float (handle both comma and dot as decimal separator)
                try:
                    value = float(value_str.replace(',', '.'))
                except ValueError:
                    continue
                
                # Normalize unit
                normalized_unit = cls.UNIT_NORMALIZATION.get(unit.lower(), unit.lower())
                
                return value, unit, normalized_unit
        
        return None, None, None
    
    @classmethod
    def is_plausible_scale_value(cls, value: float, unit: str) -> bool:
        """
        Check if a parsed value is plausible for microscopy scale.
        
        Args:
            value: Numeric value
            unit: Unit string
            
        Returns:
            True if plausible, False otherwise
        """
        if unit == 'um':
            return 0.1 <= value <= 1000  # 0.1um to 1000um
        elif unit == 'nm':
            return 10 <= value <= 1000000  # 10nm to 1000um
        elif unit == 'mm':
            return 0.001 <= value <= 10  # 0.001mm to 10mm
        else:
            return False


class OCRProcessor:
    """OCR processor with multiple backend support."""
    
    def __init__(self, backend: str = 'paddle', lang: str = 'en', 
                 confidence_threshold: float = 0.15):
        """
        Initialize OCR processor.
        
        Args:
            backend: OCR backend ('paddle', 'easyocr', 'tesseract')
            lang: Language code
            confidence_threshold: Minimum confidence for detections
        """
        self.backend = backend
        self.lang = lang
        self.confidence_threshold = confidence_threshold
        self.ocr_engine = None
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the OCR engine based on backend."""
        if self.backend == 'paddle':
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True, 
                lang=self.lang,
                use_gpu=True,
                show_log=False
            )
        else:
            raise ValueError(f"Backend {self.backend} not available or not installed")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Resize if too small (minimum height of 32 pixels)
        if gray.shape[0] < 32:
            scale_factor = 32 / gray.shape[0]
            new_width = int(gray.shape[1] * scale_factor)
            gray = cv2.resize(gray, (new_width, 32), interpolation=cv2.INTER_CUBIC)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Denoise
        gray = cv2.medianBlur(gray, 3)
        
        return gray
    
    def detect_text(self, image: np.ndarray) -> List[TextDetection]:
        """
        Detect text in image using the configured OCR engine.
        
        Args:
            image: Input image
            
        Returns:
            List of TextDetection objects
        """
        preprocessed = self.preprocess_image(image)
        detections = []
        
        try:
            if self.backend == 'paddle':
                results = self.ocr_engine.ocr(preprocessed, cls=True)
                if results and results[0]:
                    for line in results[0]:
                        bbox = line[0]
                        text_info = line[1]
                        text = text_info[0]
                        confidence = text_info[1]
                        
                        if confidence >= self.confidence_threshold:
                            # Calculate center point
                            center_x = np.mean([point[0] for point in bbox])
                            center_y = np.mean([point[1] for point in bbox])
                            
                            detections.append(TextDetection(
                                text=text,
                                confidence=confidence,
                                bbox=bbox,
                                center=(center_x, center_y)
                            ))        
        except Exception as e:
            logging.warning(f"OCR detection failed: {e}")
        
        return detections


class ScaleMatcher:
    """Matches detected text regions to scale bars."""
    
    def __init__(self, max_distance_ratio: float = 1.5):
        """
        Initialize scale matcher.
        
        Args:
            max_distance_ratio: Maximum distance as ratio of bar diagonal
        """
        self.max_distance_ratio = max_distance_ratio
    
    def calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_bar_diagonal(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate diagonal length of bounding box."""
        x, y, w, h = bbox
        return math.sqrt(w**2 + h**2)
    
    def match_text_to_bars(self, text_detections: List[TextDetection], 
                          bar_detections: List[ScaleBarDetection]) -> List[MatchedScale]:
        """
        Match text detections to scale bar detections.
        
        Args:
            text_detections: List of detected text regions
            bar_detections: List of detected scale bars
            
        Returns:
            List of matched scales
        """
        matches = []
        
        for text in text_detections:
            # Parse text to get value and unit
            value, unit, normalized_unit = TextParser.parse_text(text.text)
            
            if value is None or unit is None:
                continue
            
            # Check if value is plausible
            if not TextParser.is_plausible_scale_value(value, normalized_unit):
                continue
            
            # Update text detection with parsed values
            text.parsed_value = value
            text.parsed_unit = unit
            text.normalized_unit = normalized_unit
            
            # Find closest scale bar
            best_match = None
            best_distance = float('inf')
            
            for bar in bar_detections:
                distance = self.calculate_distance(text.center, bar.center)
                bar_diagonal = self.calculate_bar_diagonal(bar.bbox)
                max_distance = bar_diagonal * self.max_distance_ratio
                
                if distance <= max_distance and distance < best_distance:
                    best_match = bar
                    best_distance = distance
            
            if best_match:
                # Calculate um_per_pixel if bar has pixel length
                um_per_pixel = None
                if best_match.pixel_length and best_match.pixel_length > 0:
                    if normalized_unit == 'um':
                        um_per_pixel = value / best_match.pixel_length
                    elif normalized_unit == 'nm':
                        um_per_pixel = (value / 1000) / best_match.pixel_length  # Convert nm to um
                    elif normalized_unit == 'mm':
                        um_per_pixel = (value * 1000) / best_match.pixel_length  # Convert mm to um
                
                matches.append(MatchedScale(
                    scale_bar=best_match,
                    text=text,
                    distance=best_distance,
                    um_per_pixel=um_per_pixel
                ))
        
        return matches


class ScaleDetectionPipeline:
    """Complete pipeline for scale detection and OCR matching."""
    
    def __init__(self, ocr_backend: str = 'paddle', confidence_threshold: float = 0.15,
                 max_distance_ratio: float = 1.5):
        """
        Initialize the pipeline.
        
        Args:
            ocr_backend: OCR backend to use
            confidence_threshold: Minimum confidence for text detection
            max_distance_ratio: Maximum distance ratio for text-bar matching
        """
        self.ocr_processor = OCRProcessor(ocr_backend, confidence_threshold=confidence_threshold)
        self.scale_matcher = ScaleMatcher(max_distance_ratio)
        self.text_parser = TextParser()
    
    def process_image(self, image: np.ndarray, bar_detections: List[ScaleBarDetection]) -> Dict[str, Any]:
        """
        Process image to detect text and match with scale bars.
        
        Args:
            image: Input image
            bar_detections: List of detected scale bars
            
        Returns:
            Dictionary containing results
        """
        # Detect text in the image
        text_detections = self.ocr_processor.detect_text(image)
        
        # Match text to scale bars
        matches = self.scale_matcher.match_text_to_bars(text_detections, bar_detections)
        
        # Prepare results
        results = {
            'text_detections': text_detections,
            'bar_detections': bar_detections,
            'matches': matches,
            'successful_matches': len(matches),
            'total_text_detections': len(text_detections),
            'total_bar_detections': len(bar_detections)
        }
        
        return results
    
    def process_crops(self, image: np.ndarray, bar_detections: List[ScaleBarDetection]) -> Dict[str, Any]:
        """
        Process individual crops around detected scale bars for better OCR.
        
        Args:
            image: Input image
            bar_detections: List of detected scale bars
            
        Returns:
            Dictionary containing results
        """
        all_text_detections = []
        
        for bar in bar_detections:
            x, y, w, h = bar.bbox
            
            # Expand crop slightly for context
            margin = 10
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(image.shape[1], x + w + margin)
            y_end = min(image.shape[0], y + h + margin)
            
            crop = image[y_start:y_end, x_start:x_end]
            
            if crop.size == 0:
                continue
            
            # Detect text in crop
            crop_text_detections = self.ocr_processor.detect_text(crop)
            
            # Adjust coordinates back to original image
            for text in crop_text_detections:
                # Adjust bbox coordinates
                adjusted_bbox = []
                for point in text.bbox:
                    adjusted_bbox.append([point[0] + x_start, point[1] + y_start])
                text.bbox = adjusted_bbox
                
                # Adjust center coordinates
                text.center = (text.center[0] + x_start, text.center[1] + y_start)
            
            all_text_detections.extend(crop_text_detections)
        
        # Match text to scale bars
        matches = self.scale_matcher.match_text_to_bars(all_text_detections, bar_detections)
        
        # Prepare results
        results = {
            'text_detections': all_text_detections,
            'bar_detections': bar_detections,
            'matches': matches,
            'successful_matches': len(matches),
            'total_text_detections': len(all_text_detections),
            'total_bar_detections': len(bar_detections)
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary
            output_path: Path to save JSON file
        """
        # Convert results to JSON-serializable format
        json_results = {
            'successful_matches': results['successful_matches'],
            'total_text_detections': results['total_text_detections'],
            'total_bar_detections': results['total_bar_detections'],
            'matches': []
        }
        
        for match in results['matches']:
            match_data = {
                'scale_bar': {
                    'bbox': match.scale_bar.bbox,
                    'center': match.scale_bar.center,
                    'confidence': match.scale_bar.confidence,
                    'pixel_length': match.scale_bar.pixel_length,
                    'endpoints': match.scale_bar.endpoints
                },
                'text': {
                    'text': match.text.text,
                    'confidence': match.text.confidence,
                    'bbox': match.text.bbox,
                    'center': match.text.center,
                    'parsed_value': match.text.parsed_value,
                    'parsed_unit': match.text.parsed_unit,
                    'normalized_unit': match.text.normalized_unit
                },
                'distance': match.distance,
                'um_per_pixel': match.um_per_pixel
            }
            json_results['matches'].append(match_data)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR and scale matching pipeline')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--bars_json', type=str, required=True, help='Path to scale bar detections JSON')
    parser.add_argument('--output', type=str, required=True, help='Path to save results JSON')
    parser.add_argument('--ocr_backend', type=str, default='paddle', 
                       choices=['paddle', 'easyocr', 'tesseract'],
                       help='OCR backend to use')
    parser.add_argument('--confidence', type=float, default=0.15, 
                       help='Minimum confidence for text detection')
    parser.add_argument('--max_distance', type=float, default=1.5,
                       help='Maximum distance ratio for text-bar matching')
    parser.add_argument('--use_crops', action='store_true',
                       help='Process crops around scale bars instead of full image')
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load scale bar detections
    with open(args.bars_json, 'r') as f:
        bars_data = json.load(f)
    
    # Convert to ScaleBarDetection objects
    bar_detections = []
    for bar_data in bars_data.get('bars', []):
        bbox = bar_data['bbox']
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        bar_detections.append(ScaleBarDetection(
            bbox=tuple(bbox),
            center=center,
            confidence=bar_data.get('confidence', 1.0),
            pixel_length=bar_data.get('pixel_length'),
            endpoints=bar_data.get('endpoints')
        ))
    
    # Initialize pipeline
    pipeline = ScaleDetectionPipeline(
        ocr_backend=args.ocr_backend,
        confidence_threshold=args.confidence,
        max_distance_ratio=args.max_distance
    )
    
    # Process image
    if args.use_crops:
        results = pipeline.process_crops(image, bar_detections)
    else:
        results = pipeline.process_image(image, bar_detections)
    
    # Save results
    pipeline.save_results(results, args.output)
    
    # Print summary
    print(f"Processed image: {args.image}")
    print(f"Text detections: {results['total_text_detections']}")
    print(f"Scale bar detections: {results['total_bar_detections']}")
    print(f"Successful matches: {results['successful_matches']}")
    
    for i, match in enumerate(results['matches']):
        print(f"Match {i+1}:")
        print(f"  Text: '{match.text.text}' -> {match.text.parsed_value} {match.text.normalized_unit}")
        print(f"  Distance: {match.distance:.2f}")
        if match.um_per_pixel:
            print(f"  Scale: {match.um_per_pixel:.6f} um/pixel")


if __name__ == "__main__":
    main()
