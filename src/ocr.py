"""
OCR and Text-to-Bar Matching System

This module implements OCR using PaddleOCR for extracting scale text
present in the images and provides utilities to parse and normalize
detected text (values + units). It also contains logic to match recognized
text regions to nearby scale bars.

Public API:
- `OCRProcessor`: wraps the OCR model and provides `extract_text_labels`
- `TextParser`: helpers to parse numeric values and normalize units

No CLI entry point; used by `scaledetection.py`.
"""

import json
import logging as log
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import PaddleOCR

MODEL = PaddleOCR(
    ocr_version="PP-OCRv5",
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


@dataclass
class LabelDetection:
    """Data class for label detection results."""

    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    parsed_value: Optional[float] = None
    parsed_unit: Optional[str] = None
    normalized_unit: Optional[str] = None


class TextParser:
    """Text parsing utilities for scale values and units."""

    # Regex patterns for different unit formats
    UNIT_PATTERNS = {
        "micrometer": r"([0-9]+(?:[.,][0-9]+)?)\s*(μm|um|µm|μ|u|micrometer|micrometre)",
        "nanometer": r"([0-9]+(?:[.,][0-9]+)?)\s*(nm|nanometer|nanometre)",
        "millimeter": r"([0-9]+(?:[.,][0-9]+)?)\s*(mm|millimeter|millimetre)",
        "centimeter": r"([0-9]+(?:[.,][0-9]+)?)\s*(cm|centimeter|centimetre)",
        "meter": r"([0-9]+(?:[.,][0-9]+)?)\s*(m|meter|metre)",
        "generic": r"([0-9]+(?:[.,][0-9]+)?)\s*([a-zA-Zμµ]+)",
    }

    REVERSE_UNIT_PATTERNS = {
        "micrometer": r"(μm|um|µm|μ|u|micrometer|micrometre)\s*([0-9]+(?:[.,][0-9]+)?)",
        "nanometer": r"(nm|nanometer|nanometre)\s*([0-9]+(?:[.,][0-9]+)?)",
        "millimeter": r"(mm|millimeter|millimetre)\s*([0-9]+(?:[.,][0-9]+)?)",
        "centimeter": r"(cm|centimeter|centimetre)\s*([0-9]+(?:[.,][0-9]+)?)",
        "meter": r"(m|meter|metre)\s*([0-9]+(?:[.,][0-9]+)?)",
        "generic": r"([a-zA-Zμµ]+)\s*([0-9]+(?:[.,][0-9]+)?)",
    }

    # Unit normalization mapping
    UNIT_NORMALIZATION = {
        "um": "um",
        "μm": "um",
        "µm": "um",
        "μ": "um",
        "u": "um",
        "micrometer": "um",
        "micrometre": "um",
        "nm": "nm",
        "nanometer": "nm",
        "nanometre": "nm",
        "mm": "mm",
        "millimeter": "mm",
        "millimetre": "mm",
        "cm": "cm",
        "centimeter": "cm",
        "centimetre": "cm",
        "m": "mm",
        "meter": "mm",
        "metre": "mm",  # /!\ Convert meters to mm
    }

    @classmethod
    def parse_text(
        cls, text: str, reversed: bool
    ) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """Parse a string to extract numeric value and unit.

        Args:
            text (str): OCR'd text string to parse.
            reversed (bool): If True, apply patterns where the unit precedes the value.

        Returns:
            result (Tuple[Optional[float], Optional[str], Optional[str]]):
                (value, original_unit, normalized_unit) or (None, None, None) on failure.
        """
        text = text.strip()

        patterns = cls.REVERSE_UNIT_PATTERNS if reversed else cls.UNIT_PATTERNS

        # Check each pattern in order of specificity
        for unit_type, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = match.group(1) if not reversed else match.group(2)
                unit = match.group(2) if not reversed else match.group(1)

                # Convert value to float (handle both comma and dot as decimal separator)
                try:
                    value = float(value_str.replace(",", "."))
                except ValueError:
                    continue

                # Normalize unit
                normalized_unit = cls.UNIT_NORMALIZATION.get(unit.lower(), unit.lower())

                return value, unit, normalized_unit

        return None, None, None

    @classmethod
    def is_plausible_value(cls, value: float, unit: Optional[str]) -> bool:
        """Validate whether a parsed numeric value is plausible for microscopy.

        Args:
            value (float): Numeric value parsed from text.
            unit (Optional[str]): Normalized unit string (e.g., 'um', 'mm', 'nm').

        Returns:
            plausible (bool): True if the value/unit pair is within expected ranges.
        """
        if unit == "cm":
            return 0.001 <= value <= 10  # 0.001cm to 10cm
        elif unit == "mm":
            return 0.001 <= value <= 100  # 0.001mm to 100mm
        elif unit == "um":
            return 0.1 <= value <= 1000  # 0.1um to 1000um
        elif unit == "nm":
            return 10 <= value <= 1000  # 10nm to 1000nm
        else:
            return False


class OCRProcessor:
    """OCR processor with multiple backend support."""

    def __init__(self):
        """
        Initialize OCR processor.

        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.model = MODEL

    def extract_text_labels(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        reversed: bool = False,
        plot_path: Optional[str] = None,
    ) -> LabelDetection:
        """Run OCR on a cropped label ROI and parse numeric value + unit.

        Args:
            image (np.ndarray): Full input image (H,W,C or H,W).
            bbox (Tuple[int,int,int,int]): ROI as (x, y, w, h) to apply OCR on.
            reversed (bool, optional): If True, use reversed parsing patterns. Defaults to False.
            plot_path (Optional[str], optional): Path prefix to save OCR debug image. Defaults to None.

        Returns:
            detection (LabelDetection): Parsed OCR result with confidence and optional parsed value/unit.
        """
        # Apply OCR to text label regions only
        x, y, w, h = bbox

        roi = image[y : y + h, x : x + w]

        if roi.size == 0:
            return LabelDetection(text="", confidence=0.0, bbox=None)

        try:
            # OCR prediction
            output = self.model.predict(roi)

            text = "".join(output[0]["rec_texts"])
            conf = np.min(output[0]["rec_scores"])

            log.info(f"Detected text: {text} with confidence {conf}")
            if plot_path is not None:
                output[0].save_to_img(plot_path + "_ocr.png")

            # Parse text to get value and unit
            value, unit, norm_unit = TextParser.parse_text(text, reversed=reversed)

            if (
                value is None
                or unit is None
                or not TextParser.is_plausible_value(value, norm_unit)
            ):
                return LabelDetection(text=text, confidence=0.0, bbox=bbox)

            return LabelDetection(
                text=text,
                confidence=conf,
                bbox=bbox,
                parsed_value=value,
                parsed_unit=unit,
                normalized_unit=norm_unit,
            )

        except Exception as e:
            log.error(f"OCR processing failed: {e}")
            return LabelDetection(text="", confidence=0.0, bbox=None)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess an image to improve OCR performance.

        Args:
            image (np.ndarray): Input image (H,W,C or H,W).

        Returns:
            preprocessed (np.ndarray): Grayscale, contrast-enhanced and denoised image.
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
