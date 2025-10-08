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
import matplotlib.pyplot as plt
import numpy as np
import re
import json
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging
from paddleocr import PaddleOCR


@dataclass
class LabelDetection:
    """Data class for label detection results."""
    text: str
    confidence: float
    bbox: List[List[int]]  # 4 points of bounding box
    parsed_value: Optional[float] = None
    parsed_unit: Optional[str] = None
    normalized_unit: Optional[str] = None



class TextParser:
    """Text parsing utilities for scale values and units."""
    
    # Regex patterns for different unit formats
    UNIT_PATTERNS = {
        'micrometer': r'([0-9]+(?:[.,][0-9]+)?)\s*(μm|um|µm|μ|u|micrometer|micrometre)',
        'nanometer': r'([0-9]+(?:[.,][0-9]+)?)\s*(nm|nanometer|nanometre)',
        'millimeter': r'([0-9]+(?:[.,][0-9]+)?)\s*(mm|millimeter|millimetre)',
        'centimeter': r'([0-9]+(?:[.,][0-9]+)?)\s*(cm|centimeter|centimetre)',
        'meter': r'([0-9]+(?:[.,][0-9]+)?)\s*(m|meter|metre)',
        'generic': r'([0-9]+(?:[.,][0-9]+)?)\s*([a-zA-Zμµ]+)'
    }
    
    # Unit normalization mapping
    UNIT_NORMALIZATION = {
        'um': 'um', 'μm': 'um', 'µm': 'um', 'μ': 'um', 'u': 'um', 
        'micrometer': 'um', 'micrometre': 'um',
        'nm': 'nm', 'nanometer': 'nm', 'nanometre': 'nm',
        'mm': 'mm', 'millimeter': 'mm', 'millimetre': 'mm',
        'cm': 'cm', 'centimeter': 'cm', 'centimetre': 'cm',
        'm': 'mm', 'meter': 'mm', 'metre': 'mm'  # /!\ Convert meters to mm
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
    def is_plausible_value(cls, value: float, unit: str) -> bool:
        """
        Check if a parsed value is plausible for microscopy scale.
        
        Args:
            value: Numeric value
            unit: Unit string
            
        Returns:
            True if plausible, False otherwise
        """
        if unit == 'cm':
            return 0.001 <= value <= 10  # 0.001cm to 10cm
        elif unit == 'mm':
            return 0.001 <= value <= 100  # 0.001mm to 100mm
        elif unit == 'um':
            return 0.1 <= value <= 1000  # 0.1um to 1000um
        elif unit == 'nm':
            return 10 <= value <= 1000  # 10nm to 1000nm
        else:
            return False


class OCRProcessor:
    """OCR processor with multiple backend support."""

    def __init__(self, ocr_version: str):
        """
        Initialize OCR processor.
        
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.model = PaddleOCR(
            ocr_version=ocr_version, 
            lang='en', 
            use_doc_orientation_classify=False, 
            use_doc_unwarping=False, 
            use_textline_orientation=False
        )


    def extract_text_labels(
            self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
            plot_path: Optional[str] = None
        ) -> LabelDetection:
        """
        Extract text labels from the image within the given bounding box.
        """
        # Apply OCR to text label regions only
        x, y, w, h = bbox

        roi = image[y:y+h, x:x+w]

        if roi.size == 0:
            return LabelDetection(text='', confidence=0.0, bbox=[])

        try:
            # OCR prediction
            output = self.model.predict(
                roi, 
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )

            text = " ".join(output[0]['rec_texts'])
            conf = np.min(output[0]['rec_scores'])

            print(f"Detected text: {text} with confidence {conf}")
            if plot_path is not None:
                output[0].save_to_img(plot_path + '_ocr.png')

            # Parse text to get value and unit
            value, unit, norm_unit = TextParser.parse_text(text)

            if value is None or unit is None or not TextParser.is_plausible_value(value, norm_unit):
                return LabelDetection(text=text, confidence=0.0, bbox=bbox)

            return LabelDetection(
                text=text,
                confidence=conf,
                bbox=bbox,
                parsed_value=value,
                parsed_unit=unit,
                normalized_unit=norm_unit
            )
        
        except Exception as e:
            return LabelDetection(text='', confidence=0.0, bbox=[])

    
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




