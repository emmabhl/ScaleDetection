from calendar import c
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any, Sequence
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import logging
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)


@dataclass
class ScalebarDetection:
    """Data class for scale bar detection results."""

    bbox: np.ndarray  # (x_min, y_min, x_max, y_max)
    pixel_length: Optional[float] = None
    endpoints: Optional[List[Tuple[float, float]]] = None
    flag: Optional[bool] = False


class ScalebarProcessor:
    """Class for post-processing scale bar detection results."""

    def __init__(self, extend_ratio=0.25):
        self.extend_ratio = extend_ratio

    def localize_scalebar_endpoints(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        plot_path: Optional[str] = None,
    ) -> ScalebarDetection:
        """Localize scale bar endpoints within a bounding box.
        bbox is expected as (x_min, y_min, x_max, y_max).

        Args:
            image: Input image.
            bbox: Bounding box around the scale bar.
            plot_path: Optional path to save debug visualization.

        Returns:
            ScalebarDetection object with results.
        """
        bbox = bbox.astype(int)
        x_min, y_min, x_max, y_max = bbox

        # basic bbox sanity
        if not (
            0 <= x_min < x_max <= image.shape[1]
            and 0 <= y_min < y_max <= image.shape[0]
        ):
            return ScalebarDetection(
                bbox=bbox, pixel_length=0.0, endpoints=None, flag=False
            )
        try:
            # Extend box
            ext_bbox = self._extend_bbox(bbox, image.shape)

            # Crop ROI and convert to grayscale + contrast enhancement
            roi = self._extract_roi(image, ext_bbox)

            # Compute gradient magnitude with Scharr operator
            grad = self._compute_gradient(roi)

            # Thresholding to get edges (Otsu)
            binary = self._thresholding(grad)

            # Extract connected components and keep largest thin one
            filled_contour, flag = self._get_contours(binary)

            # Estimate length using PCA
            pca_result = self._pca_length_estimation(filled_contour)

            if pca_result[0] is None or pca_result[1] is None:
                raise ValueError("PCA length estimation failed.")

            # Compute final length and endpoints in global image coordinates
            p1, p2 = self._compute_endpoints(pca_result[0], pca_result[1], ext_bbox)

            # Compute flat distance as pixel length
            pixel_length = self._compute_flat_distance(p1, p2)

            # Optionally save debug visualization
            if plot_path is not None:
                viz_dict = {
                    "pixel_length": pixel_length,
                    "bbox_ext": ext_bbox,
                    "gradient_mag": grad,
                    "binary_image": binary,
                    "filled_contour": filled_contour,
                    "pca_projection": pca_result[2],
                    "endpoints": (p1, p2),
                }

                self.visualize_endpoint_detection(
                    image,
                    (x_min, y_min, x_max, y_max),
                    viz_dict,
                    save_path=plot_path + "_scalebar.png",
                )

            return ScalebarDetection(
                bbox=bbox,
                pixel_length=pixel_length,
                endpoints=[(p1[0], p1[1]), (p2[0], p2[1])],
                flag=(pixel_length < 0.66 * max(x_max - x_min, y_max - y_min)) or flag,
            )

        except Exception as e:
            logger.error(f"Error in localizing scalebar endpoints: {e}")

            if plot_path is not None:
                self.visualize_endpoint_detection(
                    image,
                    (x_min, y_min, x_max, y_max),
                    {
                        "pixel_length": 0.0,
                        "bbox_ext": ext_bbox if "ext_bbox" in locals() else None,
                        "gradient_mag": grad if "grad" in locals() else None,
                        "binary_image": binary if "binary" in locals() else None,
                        "filled_contour": (
                            filled_contour if "filled_contour" in locals() else None
                        ),
                        "pca_projection": (
                            pca_result[2]
                            if ("pca_result" in locals() and pca_result[2] is not None)
                            else None
                        ),
                        "endpoints": None,
                    },
                    save_path=plot_path + "_scalebar.png",
                )

            return ScalebarDetection(
                bbox=bbox, pixel_length=0.0, endpoints=None, flag=flag
            )

    def _extend_bbox(
        self, bbox: np.ndarray, img_shape: Sequence[int]
    ) -> Tuple[int, int, int, int]:
        """Extend the bbox by self.extend_ratio in all directions.

        Args:
            bbox (np.ndarray): Original bounding box (x1, y1, x2, y2).
            img_shape (Sequence[int]): Shape of the image (height, width[, channels]).

        Returns:
            ext_bbox (Tuple[int, int, int, int]): Extended bounding box (x1, y1, x2, y2).
        """
        # Support images with 2 or 3 dimensions by slicing the first two elements
        h, w = img_shape[:2]
        x1, y1, x2, y2 = bbox

        bw, bh = x2 - x1, y2 - y1
        dx, dy = int(bw * self.extend_ratio), int(bh * self.extend_ratio)

        x1n = max(x1 - dx, 0)
        y1n = max(y1 - dy, 0)
        x2n = min(x2 + dx, w)
        y2n = min(y2 + dy, h)

        return (x1n, y1n, x2n, y2n)

    def _extract_roi(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Crop the region of interest and return it in grayscale.

        Args:
            image (np.ndarray): Original image.
            bbox (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2).

        Returns:
            roi (np.ndarray): Grayscale cropped region.
        """
        x1, y1, x2, y2 = bbox
        roi = image[int(y1) : int(y2), int(x1) : int(x2)]

        if len(roi.shape) > 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Increase contrast to help edge detection
        p5, p95 = np.percentile(roi, (5, 95))
        roi = np.clip((roi - p5) * (255.0 / (p95 - p5 + 1e-8)), 0, 255).astype(np.uint8)

        return roi

    def _compute_gradient(self, image: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude using Scharr operator.

        Args:
            image (np.ndarray): Grayscale image.

        Returns:
            grad_norm (np.ndarray): Normalized gradient magnitude image.
        """
        # Pad image to avoid border artifacts
        image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        gx = cv2.Scharr(image, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(image, cv2.CV_32F, 0, 1)
        # Absolute gradient magnitude
        grad = cv2.magnitude(gx, gy)
        # Normalize to 8-bit before thresholding
        dst = np.empty_like(grad, dtype=np.float32)
        grad_norm = cv2.normalize(grad, dst, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return grad_norm

    def _thresholding(self, image: np.ndarray) -> np.ndarray:
        """Simple Otsu thresholding on normalized gradient magnitude.

        Args:
            image (np.ndarray): Gradient magnitude image.

        Returns:
            edges (np.ndarray): Binary edge image.
        """
        _, edges = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return edges

    def _connected_components(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Run connected components and keep only the largest component after filtering by area and AR.
        (Not used in the current pipeline because we use edge contours instead.)

        Args:
            image (np.ndarray): Binary image.

        Returns:
            clean_cc (np.ndarray): Cleaned binary image with only the selected components.
            flag (bool): Whether no component passed the aspect ratio filter.
        """
        # Compute connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            image, connectivity=8
        )

        # Filter 1 — remove very small components
        area_thresh = 0.01 * image.size
        areas = stats[:, cv2.CC_STAT_AREA]
        candidates = [i for i in range(1, num_labels) if areas[i] > area_thresh]

        # Filter 2 — long thin shapes (aspect ratio)
        final_keep = []
        for i in candidates:
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(w / h, h / w)
            if aspect > 3:
                final_keep.append(i)

        # Fallback: if nothing passed aspect filter, keep only largest component
        flag = False
        if len(final_keep) == 0:
            largest = np.argmax(areas[1:]) + 1
            final_keep = [largest]
            flag = True

        # If largest component dominates by ×3, keep only it
        if len(final_keep) > 1:
            largest = max(final_keep, key=lambda i: areas[i])
            second = sorted(final_keep, key=lambda i: areas[i])[-2]
            if areas[largest] / areas[second] > 3:
                final_keep = [largest]

        clean_cc = np.isin(labels, final_keep).astype(np.uint8) * 255
        return clean_cc, flag

    def _get_contours(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Find contours in the binary image.

        Args:
            image (np.ndarray): Binary image.

        Returns:
            mask (np.ndarray): Binary mask with only the selected contours.
            flag (bool): Whether no contour passed the aspect ratio filter.
        """
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Filter 1 — remove very small contours
        area_thresh = 0.01 * image.size
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_thresh]

        # Filter 2 — long thin shapes (aspect ratio)
        filtered_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w / h, h / w)
            if aspect > 3:
                filtered_contours.append(cnt)

        # Fallback: if nothing passed aspect filter, keep only largest contour
        flag = False
        if len(filtered_contours) == 0 and len(contours) > 0:
            filtered_contours = [contours[0]]
            flag = True

        # If largest contour dominates by ×3, keep only it
        if len(filtered_contours) > 1:
            largest = max(filtered_contours, key=cv2.contourArea)
            second = sorted(filtered_contours, key=cv2.contourArea)[-2]
            if cv2.contourArea(largest) / cv2.contourArea(second) > 3:
                filtered_contours = [largest]

        # Create mask from filtered contours
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(mask, filtered_contours, -1, color=255, thickness=cv2.FILLED)

        # Keep only the core by eroding slightly (correct the dilation effect of previous steps)
        kernel_shape = (3, 1) if mask.shape[1] >= mask.shape[0] else (1, 3)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, kernel_shape
        )  # erode only on longer side
        mask = cv2.erode(mask, kernel, iterations=1)

        return mask, flag

    def _pca_length_estimation(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate length using PCA on the component pixels.

        Args:
            image (np.ndarray): Binary mask of the scale bar component.

        Returns:
            p1 (np.ndarray): First endpoint in local ROI coordinates (y, x).
            p2 (np.ndarray): Second endpoint in local ROI coordinates (y, x).
            proj (np.ndarray): Projections of points onto principal axis.
        """
        # Get coordinates of non-zero pixels
        pts = np.column_stack(np.where(image > 0))

        if len(pts) < 2:
            logger.warning("Not enough masked pixels for PCA length estimation.")
            return None, None, None

        # PCA
        mean = pts.mean(axis=0)
        pts_centered = pts - mean
        _, _, vh = np.linalg.svd(pts_centered, full_matrices=False)
        direction = vh[0]

        # Project points onto the principal axis and find extremums
        proj = pts_centered @ direction

        # Remove outliers if any (assume uniform distribution along the bar)
        proj_filtered = proj[
            (proj >= np.median(proj) - 2 * np.std(proj))
            & (proj <= np.median(proj) + 2 * np.std(proj))
        ]
        t_min, t_max = proj_filtered.min(), proj_filtered.max()

        p1 = mean + t_min * direction
        p2 = mean + t_max * direction

        return p1, p2, proj

    def _compute_endpoints(
        self, p1: np.ndarray, p2: np.ndarray, ext_bbox: Tuple[int, int, int, int]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute global image coordinates of the endpoints.

        Args:
            p1 (np.ndarray): First endpoint in local ROI coordinates (y, x).
            p2 (np.ndarray): Second endpoint in local ROI coordinates (y, x).
            ext_bbox (Tuple[int, int, int, int]): Extended bounding box (x1, y1, x2, y2).

        Returns:
            p1_global (Tuple[float, float]): First endpoint in global image coordinates (x, y).
            p2_global (Tuple[float, float]): Second endpoint in global image coordinates (x, y).
        """
        x1, y1, _, _ = ext_bbox

        p1_global = (p1[1] + x1, p1[0] + y1)  # (x, y)
        p2_global = (p2[1] + x1, p2[0] + y1)  # (x, y)

        return p1_global, p2_global

    def _compute_flat_distance(
        self, start: Tuple[float, float], end: Tuple[float, float]
    ) -> float:
        """
        Compute the flat (purely horizontal/vertical) distance between two points.

        Args:
            start (Tuple[float, float]): Starting point (x, y).
            end (Tuple[float, float]): Ending point (x, y).

        Returns:
            distance (float): Flat distance between the two points.
        """
        h_dist = abs(end[0] - start[0])
        v_dist = abs(end[1] - start[1])
        return float(max(h_dist, v_dist))

    def visualize_endpoint_detection(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize the endpoint detection process and results.

        Args:
            image (np.ndarray): Original image.
            bbox (Tuple[int, int, int, int]): Bounding box around the scale bar.
            results (Dict[str, Any]): Dictionary containing intermediate results, including:
                - "pixel_length": Estimated pixel length of the scale bar.
                - "bbox_ext": Extended bounding box.
                - "gradient_mag": Gradient magnitude image.
                - "binary_image": Binary edge image.
                - "filled_contour": Binary mask of the largest component.
                - "pca_projection": PCA projections of component pixels.
                - "endpoints": Detected endpoints.
            save_path (Optional[str]): Path to save the visualization. If None, display it.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Full image
        axes[0, 0].imshow(image, cmap="gray")

        # Draw bounding box
        rect = Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=1,
            edgecolor="blue",
            facecolor="none",
        )
        axes[0, 0].add_patch(rect)
        axes[0, 0].set_title("Full Image Detection")
        axes[0, 0].axis("off")

        # Gradient magnitude
        if results.get("gradient_mag") is not None:
            axes[0, 1].imshow(results["gradient_mag"], cmap="gray")
            axes[0, 1].set_title("Gradient Magnitude")
            axes[0, 1].axis("off")

        # Binary image
        if results.get("binary_image") is not None:
            axes[0, 2].imshow(results["binary_image"], cmap="gray")
            axes[0, 2].set_title("Binary Image")
            axes[0, 2].axis("off")

        # Largest component
        if results.get("filled_contour") is not None:
            axes[1, 0].imshow(results["filled_contour"], cmap="gray")
            axes[1, 0].set_title("Largest Component")
            axes[1, 0].axis("off")

        # Cleaned image
        if results.get("pca_projection") is not None:
            # Plot histogram of PCA projections (normalize so that max is 1)
            axes[1, 1].hist(results["pca_projection"], bins=30, color="gray")
            axes[1, 1].axvline(
                np.median(results["pca_projection"])
                - 2 * np.std(results["pca_projection"]),
                np.median(results["pca_projection"])
                + 2 * np.std(results["pca_projection"]),
                color="red",
                linestyle="dashed",
                label="Outlier Thresholds",
            )
            axes[1, 1].set_title("PCA Projection Histogram")
            axes[1, 1].set_xlabel("Projection Value")
            axes[1, 1].set_ylabel("Frequency")
            # Replace the tick labels with normalized values
            max_count = axes[1, 1].get_ylim()[1]
            axes[1, 1].set_yticks(np.linspace(0, max_count, num=5))
            axes[1, 1].set_yticklabels(np.linspace(0, 1, num=5).round(2))
            axes[1, 1].legend()

        # Result visualization
        x_min, y_min, x_max, y_max = results["bbox_ext"]
        roi = image[int(y_min) : int(y_max), int(x_min) : int(x_max)]

        axes[1, 2].imshow(roi, cmap="gray")
        if results.get("endpoints") is not None:
            start_pt, end_pt = results["endpoints"]
            # Convert back to ROI coordinates
            rel_start = (start_pt[0] - x_min, start_pt[1] - y_min)
            rel_end = (end_pt[0] - x_min, end_pt[1] - y_min)

            axes[1, 2].plot(
                [rel_start[0], rel_end[0]],
                [rel_start[1], rel_end[1]],
                "r-",
                linewidth=2,
                label="Detected scale bar",
            )
            axes[1, 2].plot(
                rel_start[0], rel_start[1], "go", markersize=3, label="Start"
            )
            axes[1, 2].plot(rel_end[0], rel_end[1], "ro", markersize=3, label="End")
            axes[1, 2].legend()
        axes[1, 2].set_title(f'Result (Length: {results["pixel_length"]:.1f}px)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
