"""
Handlers for atypical scale bars (graduated bars, ruler photos)

This module provides functions to detect and extract characteristics of
non-standard scale bars such as graduated bars with units in the middle or
ruler-like photos with vertical graduations. The functions are used by the
main pipeline for handling "atypical" cases that do not fit the simple
scale-bar+label pattern.

Public functions:
- `extract_white_horizontal_shape(...)` : extract a graduated horizontal scalebar
- `extract_black_vertical_lines(...)` : analyze ruler-like vertical graduations
- visualization helpers for debugging

These utilities are intended to be imported and called by the main pipeline
(`scaledetection.py`) or unit tests. No CLI entry point is provided.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from postprocess_scalebar import ScalebarDetection


# ------------------------------ GRADUATED BAR WITH UNIT IN THE MIDDLE -----------------------------


def extract_white_horizontal_shape(
    image: np.ndarray,
    bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    plot_path=None,
) -> ScalebarDetection:
    """Detect a white horizontal graduated scalebar inside a polygonal bbox.

    Args:
        image (np.ndarray): Input image (H,W,C or H,W).
        bbox (Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]):
            Polygonal bounding box as four (x,y) corners.
        plot_path (Optional[str], optional): Path prefix to save debug visualizations. Defaults to None.

    Returns:
        detection (ScalebarDetection): Detected scalebar information (bbox, pixel_length, endpoints).
    """
    # 1) Crop and convert to grayscale
    ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = bbox
    image = image[y1:y3, x1:x2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2) Apply Gaussian blur and thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    T = np.quantile(blur, 0.95)
    _, thresh = cv2.threshold(
        blur, thresh=float(T), maxval=255.0, type=cv2.THRESH_BINARY
    )

    # 3) Morphological operations to clean noise
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ((image.shape[1] // 4) | 1, 1))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 4) Find contours of the shapes
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5) Filter contours
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / (h + 1e-5)
        if w >= gray.shape[1] * 0.3 and aspect_ratio >= 5.0:
            candidates.append((x, y, w, h))

    if plot_path is not None:
        visualize_endpoint_detection(
            image, gray, thresh, cleaned, candidates, plot_path + "_scalebar.png"
        )

    # 6) Get length
    length = np.max([w for x, y, w, h in candidates]) if candidates else None

    # 7) Get endpoints (middle of the longest shape)
    if candidates:
        x, y, w, h = max(candidates, key=lambda item: item[2])
        start = (y + h // 2, x)
        end = (y + h // 2, x + w)

    return ScalebarDetection(
        bbox=(x, y, w, h) if candidates else (0, 0, 0, 0),
        pixel_length=length,
        endpoints=(
            [(start[1] + x1, start[0] + y1), (end[1] + x1, end[0] + y1)]
            if candidates
            else None
        ),
        # uncertainty=0.0,
        flag=True if length is not None and length < 0.5 * gray.shape[1] else False,
    )


# --------------------------------- PICTURE OF A RULER IN THE IMAGE --------------------------------


def extract_black_vertical_lines(
    image: np.ndarray,
    bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    plot_path: Optional[str] = None,
) -> float:
    """Estimate average distance between vertical graduations in a ruler-like ROI.

    Args:
        image (np.ndarray): Input image (H,W,C or H,W).
        bbox (Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]):
            Polygonal bounding box as four (x,y) corners.
        plot_path (Optional[str], optional): Path prefix to save debug visualizations. Defaults to None.

    Returns:
        avg_distance (float): Average distance in pixels between graduations, 0.0 if none detected.
    """
    # 1) Convert to grayscale, increase contrast, invert colors and pad
    ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = bbox
    roi = image[y1:y3, x1:x2]
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    p2, p98 = np.percentile(roi, (2, 98))
    roi = cv2.normalize(
        roi, dst=np.zeros_like(roi), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    roi = np.clip((roi - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    inv = cv2.bitwise_not(roi)
    inv = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # 2) Morphological operations to remove all but long vertical shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, (roi.shape[0] // 15) | 1))
    cleaned = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, (roi.shape[0] // 15) | 1))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # 3) Thresholding with gaussian blur
    blur = cv2.GaussianBlur(cleaned, (3, 3), 0)
    T = float(np.quantile(blur, 0.95))
    _, thresh = cv2.threshold(blur, T, 255.0, cv2.THRESH_BINARY)

    # 4) Skeletonization to keep only center lines
    skeleton = cv2.ximgproc.thinning(thresh)

    # 5) Find band containing graduations on horizontal projection + estimate number of peaks
    horizontal_projection = np.sum(skeleton / 255, axis=1)
    max_peak = np.argmax(horizontal_projection)
    band_OI = signal.peak_widths(horizontal_projection, [max_peak], rel_height=0.33)
    peaks_num = np.round(band_OI[1][0] * 1.469)

    # 6) Estimate distance between graduations on vertical projection
    vertical_projection = np.sum(
        skeleton[int(band_OI[2][0]) : int(band_OI[3][0]), :], axis=0
    )
    peaks_range = np.where(vertical_projection > 0)[0]
    if len(peaks_range) == 0:
        return 0.0  # No peaks found
    peaks_range = peaks_range[  # Remove outliers assuming uniform distribution
        (peaks_range >= np.median(peaks_range) - 2 * np.std(peaks_range))
        & (peaks_range <= np.median(peaks_range) + 2 * np.std(peaks_range))
    ]
    first_peak, last_peak = peaks_range[0], peaks_range[-1]
    estimated_distance = (
        (last_peak - first_peak) / (peaks_num - 1) if peaks_num > 1 else 0
    )

    # 7) Find peaks in vertical projection
    vertical_projection = vertical_projection[first_peak : last_peak + 1]
    peaks, properties = signal.find_peaks(
        vertical_projection, distance=estimated_distance * 0.5
    )
    peak_diffs = np.diff(peaks)
    avg_distance = float(np.mean(peak_diffs[peak_diffs < estimated_distance * 1.5]))
    if np.isnan(avg_distance):
        return 0.0  # No valid peaks found

    if plot_path is not None:
        visualize_ruler_detection(
            roi,
            cleaned,
            thresh,
            skeleton,
            band_OI,
            first_peak,
            avg_distance,
            plot_path + "_ruler.png",
        )

    return avg_distance


def visualize_endpoint_detection(
    image: np.ndarray,
    gray: np.ndarray,
    thresh: np.ndarray,
    cleaned: np.ndarray,
    candidates: List[Tuple[int, int, int, int]],
    debug_path: str,
) -> None:
    """Save a debug visualization showing intermediate endpoint detection steps.

    Args:
        image (np.ndarray): Original image (RGB or grayscale).
        gray (np.ndarray): Grayscale ROI image used for processing.
        thresh (np.ndarray): Thresholded binary image.
        cleaned (np.ndarray): Morphologically cleaned image.
        candidates (List[Tuple[int, int, int, int]]): Candidate bounding boxes (x,y,w,h).
        debug_path (str): File path where the visualization image will be saved.

    Returns:
        None: Writes an image file to `debug_path`.
    """
    debug_img = image.copy()
    for x, y, w, h in candidates:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.figure(figsize=(15, 9))
    plt.subplot(1, 4, 1)
    plt.title("Grayscale")
    plt.imshow(gray, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 4, 2)
    plt.title("Thresholded")
    plt.imshow(thresh, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 4, 3)
    plt.title("Cleaned")
    plt.imshow(cleaned, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 4, 4)
    plt.title("Detected Shapes")
    plt.imshow(debug_img)
    plt.axis("off")
    plt.savefig(debug_path)
    plt.close()


def visualize_ruler_detection(
    img: np.ndarray,
    cleaned: np.ndarray,
    thresh: np.ndarray,
    skeleton: np.ndarray,
    band_OI: np.ndarray,
    first_peak: int,
    avg_distance: float,
    debug_path: str,
) -> None:
    """Save a debug visualization for ruler-graduation detection stages.

    Args:
        img (np.ndarray): Preprocessed grayscale ROI image.
        cleaned (np.ndarray): Morphologically cleaned image.
        thresh (np.ndarray): Thresholded binary image.
        skeleton (np.ndarray): Skeletonized representation of graduations.
        band_OI (np.ndarray): Band-of-interest metrics from peak analysis.
        first_peak (int): Index of the first detected peak.
        avg_distance (float): Average distance between detected peaks in pixels.
        debug_path (str): File path to save the visualization image.

    Returns:
        None: Writes an image file to `debug_path`.
    """

    plt.figure(figsize=(15, 9))
    plt.subplot(1, 4, 1)
    plt.title("Preprocessed")
    plt.imshow(img, cmap="gray")
    # Add lines for graduation band
    plt.axhline(y=int(band_OI[2][0]), color="blue", linestyle="--")
    plt.axhline(y=int(band_OI[3][0]), color="blue", linestyle="--")
    # Add a line from first_peak and 10*avg_distance long
    plt.axhline(
        y=int(band_OI[3][0] + 1.5 * band_OI[0][0]),
        xmin=first_peak / img.shape[1],
        xmax=(first_peak + 10 * avg_distance) / img.shape[1],
        color="red",
        linestyle="-",
        linewidth=1,
    )
    plt.axis("off")
    plt.subplot(1, 4, 2)
    plt.title("Cleaned")
    plt.imshow(cleaned, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 4, 3)
    plt.title("Thresholded")
    plt.imshow(thresh, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 4, 4)
    plt.title("Skeleton")
    plt.imshow(skeleton, cmap="gray")
    plt.axis("off")
    plt.savefig(debug_path)
    plt.close()
