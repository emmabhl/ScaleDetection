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
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from paddleocr import PaddleOCR

# ROI size cap: capping the crop fed to OCR at 480×96 px keeps inputs small
# enough for reliable recognition of short scale-bar labels ("1 mm", "200 µm").
# This also avoids MKL kernel selection issues on large ROIs from high-res images.
_OCR_MAX_WIDTH = 480
_OCR_MAX_HEIGHT = 96

MODEL = PaddleOCR(
    ocr_version="PP-OCRv5",
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=True,
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
        "um": r"(?i)^(?=.*\d)(?=.*(μm|um|µm|micrometer|micrometre)).*$",
        "nm": r"(?i)^(?=.*\d)(?=.*(nm|nanometer|nanometre)).*$",
        "mm": r"(?i)^(?=.*\d)(?=.*(mm|millimeter|millimetre|(?:\b|\d)m)).*$",
        "cm": r"(?i)^(?=.*\d)(?=.*(cm|centimeter|centimetre)).*$",
    }

    @classmethod
    def parse_text(
        cls,
        text: str,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Parse a string to extract numeric value and unit.

        Args:
            text (str): OCR'd text string to parse.

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

                def _to_float(s: str) -> Optional[float]:
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
    """OCR processor backed by PaddleOCR."""

    def __init__(self, confidence_threshold: float = 0.01):
        """Initialise the OCR processor.

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
            plot_path (Optional[str]): Path prefix to save OCR debug image.

        Returns:
            detection (LabelDetection): Parsed OCR result with confidence and
                optional parsed value/unit.
        """
        try:
            x_min, y_min, x_max, y_max = bbox

            # Pad generously on the left so a leading digit ("2 mm", "1 mm") that
            # sits outside the detected label bbox is still included in the OCR crop.
            # Right and vertical padding remain modest to avoid pulling in the scale
            # bar stroke, which previously lowered unit-token confidence.
            label_w = x_max - x_min
            x_pad_left = label_w
            x_pad_right = label_w // 2
            y_pad = (y_max - y_min) // 2
            roi = image[
                int(max(y_min - y_pad, 0)) : int(min(y_max + y_pad, image.shape[0])),
                int(max(x_min - x_pad_left, 0)) : int(min(x_max + x_pad_right, image.shape[1])),
            ]
            if roi.size == 0:
                return LabelDetection(text="", confidence=0.0, bbox=None)

            roi = self.preprocess_image(roi)

            # OCR prediction
            output = self.model.predict(roi)[0]

            if plot_path is not None:
                visualize_output(output, plot_path + "_ocr.png")

            # Parse text to get value and unit
            _DIGIT_MISREADS = {"l": "1", "I": "1", "i": "1", "|": "1", ")": "2", "?": "2"}
            text = [
                _DIGIT_MISREADS.get(t, t)
                for t, score in zip(output["rec_texts"], output["rec_scores"])
                if score >= self.conf_thr and t not in {"0", "O"}
            ]
            text = " ".join(text).strip()
            value, unit = TextParser.parse_text(text)

            if (
                value is None
                or unit is None
                or not TextParser.is_plausible_value(value, unit)
            ):
                return LabelDetection(text=text, confidence=0.0, bbox=bbox)

            passing_scores = [s for s in output["rec_scores"] if s >= self.conf_thr]
            return LabelDetection(
                text=text,
                confidence=float(np.min(passing_scores)) if passing_scores else 0.0,
                bbox=bbox,
                parsed_value=value,
                normalized_unit=unit,
            )

        except Exception as e:
            log.error("OCR processing failed: %s", e)
            return LabelDetection(text="", confidence=0.0, bbox=None)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Pre-process an image to improve OCR performance.

        Args:
            image (np.ndarray): Input image (H,W,C or H,W).

        Returns:
            preprocessed (np.ndarray): Contrast-enhanced and denoised image.
        """
        img = image.copy()

        # Scale up if too small (PaddleOCR text detector needs ≥32 px height)
        if img.shape[0] < 32:
            scale_factor = 32 / img.shape[0]
            new_width = int(img.shape[1] * scale_factor)
            img = cv2.resize(img, (new_width, 32), interpolation=cv2.INTER_CUBIC)

        # Cap width to _OCR_MAX_WIDTH.
        # PaddlePaddle 3.3.x + PP-OCRv5 (PIR models) + MKL-DNN: the new
        # executor selects an oneDNN kernel by input shape.  For large inputs
        # (≳4k-pixel source images the padded ROI can exceed 1000 px) the
        # selected kernel hits a fatal C-level MKL library load error that
        # kills the process.  Scale-bar labels are short ASCII strings; 960 px
        # width is more than sufficient for reliable recognition.
        # Cap both dimensions.  The oneDNN crash is shape-dependent: large
        # feature-map heights (from tall label ROIs on high-res images) trigger
        # the same MKL kernel selection bug as wide inputs.
        if img.shape[1] > _OCR_MAX_WIDTH or img.shape[0] > _OCR_MAX_HEIGHT:
            w_scale = _OCR_MAX_WIDTH / img.shape[1] if img.shape[1] > _OCR_MAX_WIDTH else 1.0
            h_scale = _OCR_MAX_HEIGHT / img.shape[0] if img.shape[0] > _OCR_MAX_HEIGHT else 1.0
            scale_factor = min(w_scale, h_scale)
            new_w = max(1, int(img.shape[1] * scale_factor))
            new_h = max(32, int(img.shape[0] * scale_factor))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Enhance contrast
        p1, p99 = np.percentile(img, (1, 99))
        img = np.clip((img - p1) * (255.0 / (p99 - p1)), 0, 255).astype(np.uint8)

        return img


def visualize_output(res: Dict[str, Any], plot_path: str) -> None:
    """Create a 3-panel PNG: input image | boxes | recognised text legend.

    Saves to ``plot_path``.

    Args:
        res (Dict[str, Any]): OCR result dictionary returned by PaddleOCR.
        plot_path (str): Destination file path for the visualisation PNG.
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

    # Normalise dtype/shape
    if not np.issubdtype(img.dtype, np.integer):
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) if img.dtype == float else img.astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    h, w = img.shape[:2]

    # --- get recognised texts and boxes/polys ---
    rec_texts = res.get("rec_texts") or []
    rec_scores = res.get("rec_scores") or []
    rec_boxes = res.get("rec_boxes")
    boxes = None

    if isinstance(rec_boxes, np.ndarray) and rec_boxes.ndim == 2 and rec_boxes.shape[1] >= 4:
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
    ax_boxes.set_title("Recognised boxes")
    ax_boxes.axis("off")
    if boxes is not None and isinstance(boxes, np.ndarray) and boxes.size:
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b[:4]
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
                ax_boxes.text(
                    x1 + 2,
                    y1 + 2,
                    str(i + 1),
                    fontsize=7,
                    color="black",
                    bbox=dict(facecolor="yellow", edgecolor="none", pad=0.2, alpha=0.8),
                )

    # right: legend with recognised texts
    ax_legend.axis("off")
    ax_legend.set_title("Recognised text")

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
        ax_legend.text(0.5, 0.5, "No recognised\ntext", ha="center", va="center", fontsize=36)
    else:
        line_height = 0.1
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

    plt.tight_layout()
    try:
        fig.savefig(str(plot_path), bbox_inches="tight", pad_inches=0.1)
    except Exception as e:
        log.error("Failed to save subplot visualisation: %s", e)
    finally:
        plt.close(fig)
