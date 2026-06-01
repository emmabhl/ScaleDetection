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
import logging as logger
import warnings

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
    try:
        flag = False
        # 1) Crop to bbox and convert to grayscale
        ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = bbox
        bbox_xyxy = [min(x1, x2, x3, x4), min(y1, y2, y3, y4), max(x1, x2, x3, x4), max(y1, y2, y3, y4)]
        image = image[bbox_xyxy[1]:image.shape[0], bbox_xyxy[0]:image.shape[1]]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 2) Edge detection (kept for visualization)
        edges = cv2.Canny(gray, 50, 150)

        # 3) Locate the black rectangle by thresholding dark pixels and finding its
        #    bounding box. This is more robust than contour-based re-cropping, which
        #    is sensitive to noise, partial occlusion, and bbox rotation.
        _, dark_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dark_coords = cv2.findNonZero(dark_mask)
        crop = gray  # fallback: use full bbox crop
        crop_offset_x, crop_offset_y = 0, 0
        if dark_coords is not None:
            rx, ry, rw, rh = cv2.boundingRect(dark_coords)
            bbox_w = bbox_xyxy[2] - bbox_xyxy[0]
            bbox_h = bbox_xyxy[3] - bbox_xyxy[1]
            # Only accept if the detected dark region is large enough to be the rectangle
            # and has a landscape aspect ratio (wider than tall)
            if rw >= 0.4 * bbox_w and rh >= 0.3 * bbox_h and rw >= rh:
                crop = gray[ry : ry + rh, rx : rx + rw]
                crop_offset_x, crop_offset_y = rx, ry

        # 4) Threshold based on high quantile after Gaussian blur (bar is white)
        blur = cv2.GaussianBlur(crop, (5, 5), 0)
        T = np.quantile(blur, 0.95)
        _, thresh = cv2.threshold(
            blur, thresh=float(T), maxval=255.0, type=cv2.THRESH_BINARY
        )

        # 5) Morphological operations to clean noise (keep long horizontal shapes & thicken them).
        #    Use crop.shape[1] // 8 for the open kernel so it's not so wide that it erases
        #    the thin bar strip at the very bottom of a tightly-cropped rectangle.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ((crop.shape[1] // 8) | 1, 1))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_DILATE, kernel)

        # 6) Find contours of the white long shapes
        cleaned = cv2.copyMakeBorder(cleaned, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 7) Filter contours based on width and aspect ratio.
        #    Relax the bottom-position threshold from 0.9 to 0.75 since a tight
        #    dark-region crop places the bar strip closer to the centre of the crop.
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / (h + 1e-5)
            if (
                w >= crop.shape[1] * 0.67  # sufficiently long
                and aspect_ratio >= 5.0  # long horizontal shape
                and y > crop.shape[0] * 0.75  # near bottom of crop
            ):
                candidates.append((x, y, w, h))

        # 7) Get length
        if candidates:
            x, y, w, h = max(candidates, key=lambda item: item[2])
            length = float(w)

            # 8) Get endpoints (middle of the longest shape)
            start = (y + h // 2, x)
            end = (y + h // 2, x + w)
            flag = False
        else:
            logger.warning("No suitable horizontal shape found inside bbox.")
            flag = True
            length = (
                np.max([cv2.boundingRect(cnt)[2] for cnt in contours])
                if contours
                else 0.0
            )
            start = (crop.shape[0] // 2, 0)
            end = (crop.shape[0] // 2, crop.shape[1])
            if length <= 0.5 * crop.shape[1]:
                logger.warning(
                    "No shape found with sufficient length. Using the full crop width."
                )
                length = float(crop.shape[1])

        if plot_path is not None:
            vis_res = {
                "length": length,
                "gray": gray,
                "edges": edges,
                "crop": crop,
                "thresh": thresh,
                "cleaned": cleaned,
                "candidates": candidates,
            }
            visualize_endpoint_detection(
                vis_res, debug_path=plot_path + "_scalebar.png"
            )

        return ScalebarDetection(
            bbox=np.array(bbox_xyxy) if candidates else np.zeros((4,)),
            pixel_length=length,
            endpoints=[
                (start[1] + crop_offset_x + bbox_xyxy[0], start[0] + crop_offset_y + bbox_xyxy[1]),
                (end[1] + crop_offset_x + bbox_xyxy[0], end[0] + crop_offset_y + bbox_xyxy[1]),
            ],
            flag=flag,
        )

    except Exception as e:
        logger.error(f"Error in extracting white graduated scale bar: {e}")

        if plot_path is not None:
            vis_res = {
                "length": length if "length" in locals() else None,
                "candidates": candidates if "candidates" in locals() else None,
                "gray": gray if "gray" in locals() else None,
                "blur": blur if "blur" in locals() else None,
                "threshold_value": T if "T" in locals() else None,
                "thresh": thresh if "thresh" in locals() else None,
                "cleaned": cleaned if "cleaned" in locals() else None,
            }
            visualize_endpoint_detection(
                vis_res, debug_path=plot_path + "_scalebar.png"
            )

        return ScalebarDetection(
            bbox=np.array(bbox_xyxy) if bbox is not None else np.zeros((4,)),
            pixel_length=0.0,
            endpoints=None,
            flag=flag,
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
    try:
        # 1) Convert to grayscale, increase contrast, invert colors and pad
        ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) = bbox
        bbox_xyxy = [min(x1, x2, x3, x4), min(y1, y2, y3, y4), max(x1, x2, x3, x4), max(y1, y2, y3, y4)]
        roi = image[bbox_xyxy[1]:image.shape[0], bbox_xyxy[0]:image.shape[1]]
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        p2, p98 = np.percentile(roi, (2, 98))
        roi = cv2.normalize(
            roi, dst=np.zeros_like(roi), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        roi = np.clip((roi - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
        inv = cv2.bitwise_not(roi)
        inv = cv2.copyMakeBorder(inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        # 2) Morphological operations to remove all but long vertical shapes
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, (roi.shape[0] // 15) | 1)
        )
        cleaned = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, (roi.shape[0] // 15) | 1)
        )
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
        peaks_num = np.round(
            band_OI[1][0] * 1.469
        )  # empirical factor to estimate number of peaks

        # 6) Estimate distance between graduations on vertical projection
        vertical_projection = np.sum(
            skeleton[int(band_OI[2][0]) : int(band_OI[3][0]), :], axis=0
        )
        peaks_range = np.where(vertical_projection > 0)[0]
        if len(peaks_range) == 0:
            avg_distance = 0.0  # No peaks found

        else:
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
            if len(peaks) < 2:
                avg_distance = 0.0
            else:
                peak_diffs = np.diff(peaks)
                filtered_diffs = peak_diffs[peak_diffs < estimated_distance * 1.5]
                avg_distance = float(np.mean(filtered_diffs)) if len(filtered_diffs) > 0 else 0.0

        if plot_path is not None:
            vis_res = {
                "cleaned": cleaned,
                "thresh": thresh,
                "skeleton": skeleton,
                "band_OI": band_OI,
                "first_peak": first_peak,
                "avg_distance": avg_distance,
            }
            visualize_ruler_detection(roi, vis_res, plot_path + "_ruler.png")

        return avg_distance if avg_distance is not None else 0.0
    except Exception as e:
        logger.error(f"Error in extracting black vertical lines: {e}")

        if plot_path is not None:
            vis_res = {
                "cleaned": cleaned if "cleaned" in locals() else None,
                "thresh": thresh if "thresh" in locals() else None,
                "skeleton": skeleton if "skeleton" in locals() else None,
                "band_OI": band_OI if "band_OI" in locals() else None,
                "first_peak": first_peak if "first_peak" in locals() else None,
                "avg_distance": avg_distance if "avg_distance" in locals() else None,
            }
            visualize_ruler_detection(roi, vis_res, plot_path + "_ruler.png")

        return 0.0


def visualize_endpoint_detection(
    results: Dict[str, Any],
    debug_path: str,
) -> None:
    """Save a debug visualization showing intermediate endpoint detection steps.

    Args:
        image (np.ndarray): Original image (RGB or grayscale).
        results (Dict[str, Any]): Dictionary with intermediate results including:
            - length: Detected length of the shape.
            - gray: Grayscale image.
            - edges: Edges detected in the image.
            - crop: Cropped image around the shape.
            - thresh: Thresholded binary image.
            - cleaned: Morphologically cleaned image.
            - candidates: List of detected shape bounding boxes.
        debug_path (str): File path where the visualization image will be saved.

    Returns:
        None: Writes an image file to `debug_path`.
    """
    if results.get("gray") is not None:
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 3, 1)
        plt.title("Grayscale")
        plt.imshow(results["gray"], cmap="gray")
        plt.axis("off")
    if results.get("edges") is not None:
        plt.subplot(2, 3, 2)
        plt.title("Edges")
        plt.imshow(results["edges"], cmap="gray")
        plt.axis("off")
    if results.get("crop") is not None:
        plt.subplot(2, 3, 3)
        plt.title("Cropped")
        plt.imshow(results["crop"], cmap="gray")
        plt.axis("off")
    if results.get("thresh") is not None:
        plt.subplot(2, 3, 4)
        plt.title("Thresholded")
        plt.imshow(results["thresh"], cmap="gray")
        plt.axis("off")
    if results.get("cleaned") is not None:
        plt.subplot(2, 3, 5)
        plt.title("Cleaned")
        plt.imshow(results["cleaned"], cmap="gray")
        plt.axis("off")
    debug_img = cv2.cvtColor(results["crop"].copy(), cv2.COLOR_GRAY2RGB)
    if results.get("candidates") is not None:
        for x, y, w, h in results["candidates"]:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plt.subplot(2, 3, 6)
    plt.title(f"Detected Shapes, length={results.get('length', 'N/A'):.1f}px")
    plt.imshow(debug_img, cmap="gray")
    plt.tight_layout()
    plt.savefig(debug_path)
    plt.close()


def visualize_ruler_detection(
    img: np.ndarray,
    results: Dict[str, Any],
    debug_path: str,
) -> None:
    """Save a debug visualization for ruler-graduation detection stages.

    Args:
        img (np.ndarray): Preprocessed grayscale ROI image.
        results (Dict[str, Any]): Dictionary with intermediate results including:
            - "cleaned": Morphologically cleaned image.
            - "thresh": Thresholded binary image.
            - "skeleton": Skeletonized image.
            - "band_OI": Tuple with band of interest for graduations.
            - "first_peak": X-coordinate of the first detected peak.
            - "avg_distance": Average distance between graduations.
        debug_path (str): File path to save the visualization image.

    Returns:
        None: Writes an image file to `debug_path`.
    """
    plt.figure(figsize=(15, 9))
    plt.subplot(2, 2, 1)
    plt.title("Preprocessed")
    plt.imshow(img, cmap="gray")
    # Add lines for graduation band
    if results.get("band_OI") is not None:
        plt.axhline(y=int(results["band_OI"][2][0]), color="blue", linestyle="--")
        plt.axhline(y=int(results["band_OI"][3][0]), color="blue", linestyle="--")
    # Add a line from first_peak and 10*avg_distance long
    if results.get("avg_distance", 0) > 0:
        plt.axhline(
            y=int(results["band_OI"][3][0] + 1.5 * results["band_OI"][0][0]),
            xmin=results["first_peak"] / img.shape[1],
            xmax=(results["first_peak"] + 10 * results["avg_distance"]) / img.shape[1],
            color="red",
            linestyle="-",
            linewidth=1,
        )
        plt.axis("off")
    if results.get("cleaned") is not None:
        plt.subplot(2, 2, 2)
        plt.title("Cleaned")
        plt.imshow(results["cleaned"], cmap="gray")
        plt.axis("off")
        plt.subplot(2, 2, 3)
    if results.get("thresh") is not None:
        plt.title("Thresholded")
        plt.imshow(results["thresh"], cmap="gray")
        plt.axis("off")
    if results.get("skeleton") is not None:
        plt.subplot(2, 2, 4)
        plt.title("Skeleton")
        plt.imshow(results["skeleton"], cmap="gray")
        plt.axis("off")
    plt.savefig(debug_path)
    plt.close()
