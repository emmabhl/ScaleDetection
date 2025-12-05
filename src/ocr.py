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
from inflect import unit
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
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
    bbox: Optional[np.ndarray]  # (x_min, y_min, x_max, y_max)
    parsed_value: Optional[float] = None
    normalized_unit: Optional[str] = None


class TextParser:
    """Text parsing utilities for scale values and units."""

    # Regex patterns for different unit formats
    UNIT_PATTERNS = {
        "um": r"(?i)^(?=.*\d)(?=.*(?:\b|\d)(μm|um|µm|micrometer|micrometre)(?:\b|\d)).*$",
        "nm": r"(?i)^(?=.*\d)(?=.*(?:\b|\d)(nm|nanometer|nanometre)(?:\b|\d)).*$",
        "mm": r"(?i)^(?=.*\d)(?=.*(?:\b|\d)(mm|millimeter|millimetre|m)(?:\b|\d)).*$",
        "cm": r"(?i)^(?=.*\d)(?=.*(?:\b|\d)(cm|centimeter|centimetre)(?:\b|\d)).*$",
    }

    @classmethod
    def parse_text(
        cls,
        text: str,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Parse a string to extract numeric value and unit.

        Args:
            text (str): OCR'd text string to parse.
            reversed (bool): If True, apply patterns where the unit precedes the value.

        Returns:
            result (Tuple[Optional[float], Optional[str]]):
                (value, normalized_unit) or (None, None) on failure.
        """
        patterns = cls.UNIT_PATTERNS

        # Check each pattern in order of specificity
        for unit_type, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value_str = re.findall(r"\d+[.,]?\d*", text)
                unit_str = unit_type

                # Convert value to float (handle both comma and dot as decimal separator)
                # convert found numeric substrings to floats (handle comma as decimal sep)
                def _to_float(s):
                    try:
                        return float(s.replace(",", "."))
                    except (ValueError, AttributeError):
                        return None

                max_val = max(
                    (n for n in (_to_float(s) for s in value_str) if n is not None),
                    default=0.0,
                )

                return max_val, unit_str

        return None, None

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

    def __init__(self, confidence_threshold: float = 0.25):
        """
        Initialize OCR processor.

        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.model = MODEL
        self.conf_thr = confidence_threshold

    def extract_text_labels(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        plot_path: Optional[str] = None,
    ) -> LabelDetection:
        """Run OCR on a cropped label ROI and parse numeric value + unit.

        Args:
            image (np.ndarray): Full input image (H,W,C or H,W).
            bbox (np.ndarray): ROI as (x_min, y_min, x_max, y_max) to apply OCR on.
            reversed (bool, optional): If True, use reversed parsing patterns. Defaults to False.
            plot_path (Optional[str], optional): Path prefix to save OCR debug image. Defaults to None.

        Returns:
            detection (LabelDetection): Parsed OCR result with confidence and optional parsed value/unit.
        """
        try:
            # Apply OCR to text label regions only
            x_min, y_min, x_max, y_max = bbox

            # Extract ROI with some padding
            roi = image[
                int(max(y_min - 20, 0)) : int(min(y_max + 20, image.shape[0])),
                int(max(x_min - 20, 0)) : int(min(x_max + 20, image.shape[1])),
            ]
            if roi.size == 0:
                return LabelDetection(text="", confidence=0.0, bbox=None)

            roi = self.preprocess_image(roi)

            # OCR prediction
            output = self.model.predict(roi)[0]

            if plot_path is not None:
                visualize_output(output, plot_path + "_ocr.png")

            # Parse text to get value and unit
            text = [
                text
                for text, score in zip(output["rec_texts"], output["rec_scores"])
                if score >= self.conf_thr and text not in {"0", "O"}
            ]
            text = " ".join(text).strip()
            value, unit = TextParser.parse_text(text)

            if (
                value is None
                or unit is None
                or not TextParser.is_plausible_value(value, unit)
            ):
                return LabelDetection(text=text, confidence=0.0, bbox=bbox)

            return LabelDetection(
                text=text,
                confidence=np.min(
                    [s for s in output["rec_scores"] if s >= self.conf_thr]
                ),
                bbox=bbox,
                parsed_value=value,
                normalized_unit=unit,
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
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize if too small (minimum height of 32 pixels)
        if gray.shape[0] < 32:
            scale_factor = 32 / gray.shape[0]
            new_width = int(gray.shape[1] * scale_factor)
            gray = cv2.resize(gray, (new_width, 32), interpolation=cv2.INTER_CUBIC)

        # Enhance contrast
        p2, p98 = np.percentile(gray, (2, 98))
        gray = np.clip((gray - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)

        return gray


def visualize_output(res: Dict[str, Any], plot_path: str) -> None:
    """
    Create a 3-panel PNG:
        [input image] [image with bounding boxes] [legend with recognized text (small list)]
    Save to plot_path (string).
    """
    # --- get image from common keys ---
    dp = res.get("doc_preprocessor_res", {}) if isinstance(res, dict) else {}
    img = None
    for key in ("output_img", "input_img", "rot_img"):
        cand = dp.get(key)
        if isinstance(cand, np.ndarray) and getattr(cand, "ndim", 0) >= 2:
            img = cand
            break
    if img is None:
        for key in ("img", "res_img", "input_img", "ori_img", "image"):
            cand = res.get(key)
            if isinstance(cand, np.ndarray) and getattr(cand, "ndim", 0) >= 2:
                img = cand
                break
    if img is None:
        log.warning("No valid image found in OCR result.")
        return

    # Normalize dtype/shape
    if not np.issubdtype(img.dtype, np.integer):
        img = (
            (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            if img.dtype == float
            else img.astype(np.uint8)
        )
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    h, w = img.shape[:2]

    # --- get recognized texts and boxes/polys ---
    rec_texts = res.get("rec_texts") or []
    rec_scores = res.get("rec_scores") or []
    rec_boxes = res.get("rec_boxes")
    boxes = None

    # If rec_boxes is a numpy array, use it (expected shape Nx4: x1,y1,x2,y2)
    if (
        isinstance(rec_boxes, np.ndarray)
        and rec_boxes.ndim == 2
        and rec_boxes.shape[1] >= 4
    ):
        boxes = rec_boxes.astype(float)
    else:
        # fallback: build boxes from rec_polys by taking bounding rectangle
        rec_polys = res.get("rec_polys") or res.get("dt_polys") or []
        bxs = []
        for poly in rec_polys:
            try:
                arr = np.asarray(poly, dtype=float)
                if arr.ndim == 2 and arr.size:
                    x_min, y_min = arr[:, 0].min(), arr[:, 1].min()
                    x_max, y_max = arr[:, 0].max(), arr[:, 1].max()
                    bxs.append([x_min, y_min, x_max, y_max])
            except Exception:
                continue
        if bxs:
            boxes = np.array(bxs, dtype=float)

    # --- plotting ---
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, max(4, h * 14 / max(w, 1))),
        gridspec_kw={"width_ratios": [1.1, 1.1, 0.6]},
    )
    ax_in, ax_boxes, ax_legend = axes

    # left: input image
    ax_in.imshow(img)
    ax_in.set_title("Input")
    ax_in.axis("off")

    # middle: image with bounding boxes
    ax_boxes.imshow(img)
    ax_boxes.set_title("Recognized boxes")
    ax_boxes.axis("off")
    if boxes is not None and isinstance(boxes, np.ndarray) and boxes.size:
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b[:4]
            # clip to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                rect = Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1.2,
                    edgecolor="yellow",
                    facecolor="none",
                )
                ax_boxes.add_patch(rect)
                # small index label
                ax_boxes.text(
                    x1 + 2,
                    y1 + 2,
                    str(i + 1),
                    fontsize=7,
                    color="black",
                    bbox=dict(facecolor="yellow", edgecolor="none", pad=0.2, alpha=0.8),
                )

    # right: legend with recognized texts (compact, centered, enlarged)
    ax_legend.axis("off")
    ax_legend.set_title("Recognized text")

    lines = []
    n = max(len(rec_texts), 0)
    for i in range(n):
        txt = rec_texts[i] if i < len(rec_texts) else ""
        score = rec_scores[i] if i < len(rec_scores) else None
        label = f"{txt}"
        if score is not None:
            label += f" ({score:.2f})"
        lines.append(label)

    if not lines:
        ax_legend.text(
            0.5, 0.5, "No recognized\ntext", ha="center", va="center", fontsize=36
        )
    else:
        # vertical stacking
        line_height = 0.1  # adjust for huge font
        y = 0.9
        for line in lines:
            ax_legend.text(
                0.5,
                y,
                line,
                transform=ax_legend.transAxes,
                fontsize=36,
                ha="center",
                va="center",
                wrap=True,
            )
            y -= line_height
            if y < 0.05:
                break

    # tighten and save
    plt.tight_layout()
    try:
        fig.savefig(str(plot_path), bbox_inches="tight", pad_inches=0.1)
    except Exception as e:
        log.error("Failed to save subplot visualization: %s", e)
    finally:
        plt.close(fig)
