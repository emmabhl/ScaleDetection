"""
Template-based Scale Bar Classifier

This module implements `ScaleBarClassifier`, a template-matching classifier
based on ORB descriptors and BFMatcher. It loads precomputed ORB template
descriptors (from `.precomputed_ORB_descriptors`) and attempts to match
an input image to known scale-bar types.

Typical usage (imported from code):
    from classifier import ScaleBarClassifier
    clf = ScaleBarClassifier(atypical_data_path='path/to/templates')
    matches = clf.classify_scale_bar(image)

There is no CLI entry point; the class is used by the main pipeline
(`scaledetection.py`).
"""

import logging as log
import os
import pickle
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from precompute_ORB_descriptors import main as precompute_ORB_descriptors

class ScaleBarClassifier:
    """
    Encapsulates the scale-bar template matching pipeline from the provided script.

    Args
        - score_threshold (float): Minimum score threshold to report a match (default 0.15).
        - nfeatures (int): Number of ORB features to create (used when computing descriptors for a 
            target image) (default 1000).
        - ratio_thresh (float): Lowe's ratio test threshold (default 0.75).
        - min_match_count (int): Minimum number of good matches required to attempt homography 
            (default 10).
        - atypical_data_path (Optional[str]): If precomputed_dir does not exist, this directory 
            is used to compute and save templates.
    """

    def __init__(
        self,
        score_threshold: float = 0.15,
        nfeatures: int = 1000,
        ratio_thresh: float = 0.75,
        min_match_count: int = 10,
        atypical_data_path: Optional[str] = None,
    ):
        self.precomputed_dir = ".precomputed_ORB_descriptors"
        self.score_threshold = score_threshold
        self.nfeatures = nfeatures
        self.ratio_thresh = ratio_thresh
        self.min_match_count = min_match_count

        # ORB + BFMatcher (knn + ratio test)
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures) # pyright: ignore[reportAttributeAccessIssue]
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Loaded templates (dict: scale_type -> list of template_info dicts)
        self.templates_per_type: Dict[str, List[Dict[str, Any]]] = {}
        if os.path.isdir(self.precomputed_dir):
            self.load_templates()
        else:
            if atypical_data_path is None:
                raise ValueError(
                    "precomputed_dir does not exist and no atypical_data_path path provided."
                )
            
            precompute_ORB_descriptors(
                atypical_data_path, 
                self.precomputed_dir, 
                nfeatures=self.nfeatures
            )
            self.load_templates()

    def load_templates(self) -> None:
        """Load precomputed ORB template pickles into memory.

        Returns:
            None: Populates `self.templates_per_type` with loaded template lists.
        """
        if not os.path.isdir(self.precomputed_dir):
            raise FileNotFoundError(f"Precomputed directory not found: {self.precomputed_dir}")

        for fname in os.listdir(self.precomputed_dir):
            if fname.endswith(".pkl"):
                type_name = fname.replace(".pkl", "")
                path = os.path.join(self.precomputed_dir, fname)
                with open(path, "rb") as f:
                    self.templates_per_type[type_name] = pickle.load(f)
        log.info("Templates loaded: %s", list(self.templates_per_type.keys()))

    @staticmethod
    def reconstruct_keypoints(kp_dicts: List[Dict[str, Any]]) -> List[cv2.KeyPoint]:
        """Reconstruct cv2.KeyPoint objects from serialized dictionaries.

        Args:
            kp_dicts (List[Dict[str, Any]]): List of keypoint attribute dictionaries.

        Returns:
            keypoints (List[cv2.KeyPoint]): Reconstructed KeyPoint objects.
        """
        kps = [
            cv2.KeyPoint(
                float(kp["pt"][0]),
                float(kp["pt"][1]),
                float(kp["size"]),
                float(kp["angle"]),
                float(kp["response"]),
                int(kp["octave"]),
                int(kp["class_id"]),
            )
            for kp in kp_dicts
        ]
        return kps

    def classify_scale_bar(self, target_image: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        """Classify an image by matching ORB descriptors against templates.

        Args:
            target_image (np.ndarray): Image to classify (H,W,C or H,W).

        Returns:
            results (Optional[List[Dict[str,Any]]]): Sorted list of match dicts
                containing 'scale_type','template','score','bbox', or None if no descriptors.
        """
        # Convert to grayscale if needed
        if len(target_image.shape) == 3:
            target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target_image

        kp2, des2 = self.orb.detectAndCompute(target_gray, None)
        if des2 is None or len(kp2) == 0:
            return None

        results: List[Dict[str, Any]] = []

        for scale_type, templates in self.templates_per_type.items():
            best_score = 0.0
            best_bbox = None
            best_template_fname = None

            for template_info in templates:
                des1 = template_info["descriptors"]
                kp1_dicts = template_info["keypoints"]
                kp1 = self.reconstruct_keypoints(kp1_dicts)

                # Match descriptors using KNN
                matches = self.bf.knnMatch(des1, des2, k=2)

                # Apply Lowe's ratio test
                good_matches = []
                ratio_thresh = self.ratio_thresh
                for pair in matches:
                    # original script assumes pairs (m, n)
                    if len(pair) < 2:
                        continue
                    m, n = pair[0], pair[1]
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)

                if len(good_matches) >= self.min_match_count:
                    # Prepare points for homography
                    src_pts = np.array(
                        [kp1[m.queryIdx].pt for m in good_matches]
                    ).astype(np.float32).reshape(-1, 1, 2)
                    dst_pts = np.array(
                        [kp2[m.trainIdx].pt for m in good_matches]
                    ).astype(np.float32).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
                    if M is not None and mask is not None:
                        mask = mask.ravel()
                        inliers = int(np.sum(mask))

                        # Compute a normalized score (as in original script)
                        score = inliers / len(kp1) if len(kp1) > 0 else 0.0

                        if score > best_score:
                            best_score = score
                            h, w = template_info["image_shape"]
                            pts = np.array(
                                [[0, 0], [w, 0], [w, h], [0, h]]
                            ).astype(np.float32).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            best_bbox = dst
                            best_template_fname = template_info["filename"]
                # Fallback if not enough matches or homography fails (as in original script)
                elif len(good_matches) >= 5:
                    dst_pts = np.array(
                        [kp2[m.trainIdx].pt for m in good_matches]
                    ).astype(np.float32)
                    x_min, y_min = float(dst_pts[:, 0].min()), float(dst_pts[:, 1].min())
                    x_max, y_max = float(dst_pts[:, 0].max()), float(dst_pts[:, 1].max())
                    score = len(good_matches) / len(kp1) if len(kp1) > 0 else 0.0
                    if score > best_score:
                        best_score = score
                        best_bbox = np.array(
                            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                            dtype=np.float32,
                        ).reshape(-1, 1, 2)
                        best_template_fname = template_info["filename"]

            if best_score > self.score_threshold:
                best_bbox = best_bbox.reshape(-1, 2).astype(int) if best_bbox is not None else None

                results.append(
                    {
                        "scale_type": scale_type,
                        "template": best_template_fname,
                        "score": best_score,
                        "bbox": best_bbox,
                    }
                )

        # Sort results by descending score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


    def save_results(
        self, 
        target_img: np.ndarray, 
        matches: Optional[List[Dict[str, Any]]], 
        save_path: Optional[str] = None
    ) -> None:
        """Visualize and optionally save classification overlay on the image.

        Args:
            target_img (np.ndarray): Image on which to draw detections (BGR/RGB).
            matches (Optional[List[Dict[str,Any]]]): Match results as returned by `classify_scale_bar`.
            save_path (Optional[str], optional): Path to save the visualization image. Defaults to None.

        Returns:
            None: Displays or writes an image depending on `save_path`.
        """
        if matches is None or len(matches) == 0:
                return
        try:
            cm = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            for i, m in enumerate(matches):
                log.info(f"{m['template']}, Type: {m['scale_type']}, Score: {m['score']:.2f}")
                bbox = m['bbox']
                if bbox is not None:
                    # Draw bounding box on image for visualization
                    bbox_pts = bbox
                    cv2.polylines(
                        target_img, [bbox_pts.reshape(-1,2)], 
                        isClosed=True, color=cm[i % len(cm)], thickness=2
                    )
                    legend = f"{m['scale_type']} ({m['score']:.2f})"
                    cv2.putText(target_img, legend, (bbox_pts[0][0], bbox_pts[0][1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cm[i % len(cm)], 2)

            if save_path:
                plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
                plt.title("Classification Results")
                plt.axis("off")
                plt.savefig(save_path)
                plt.close()
            else:
                plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
                plt.title("Classification Results")
                plt.axis("off")
                plt.show()
        except Exception as e:
            log.error(f"Error in save_results: {e}")