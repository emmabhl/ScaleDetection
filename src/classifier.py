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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from precompute_feature_descriptors import main as precompute_features_descriptors


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
        score_threshold: float = 0.02,
        nfeatures: int = 5000,
        ratio_thresh: float = 0.75,
        min_match_count: int = 6,
        atypical_data_path: Optional[str] = None,
    ):
        self.precomputed_dir = ".precomputed_feature_descriptors"
        self.score_threshold = score_threshold
        self.nfeatures = nfeatures
        self.ratio_thresh = ratio_thresh
        self.min_match_count = min_match_count

        # ORB + AKAZE + SIFT detectors
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures) # pyright: ignore[reportAttributeAccessIssue]
        self.akaze = cv2.AKAZE_create() # pyright: ignore[reportAttributeAccessIssue]
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures) # pyright: ignore[reportAttributeAccessIssue]
        
        # Matchers for different descriptor types
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # ORB + AKAZE
        self.bf_l2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)            # SIFT

        # Loaded templates (dict: scale_type -> list of template_info dicts)
        self.templates_per_type: Dict[str, List[Dict[str, Any]]] = {}
        if os.path.isdir(self.precomputed_dir):
            self.load_templates()
        else:
            if atypical_data_path is None:
                # FIXME: Load from emmabhl/atypical-scalebar dataset if no local path provided
                
                raise ValueError(
                    "precomputed_dir does not exist and no atypical_data_path path provided."
                )

            precompute_features_descriptors(
                atypical_data_path, self.precomputed_dir, nfeatures=self.nfeatures
            )
            self.load_templates()

    def load_templates(self) -> None:
        """Load precomputed ORB template pickles into memory.

        Returns:
            None: Populates `self.templates_per_type` with loaded template lists.
        """
        if not os.path.isdir(self.precomputed_dir):
            raise FileNotFoundError(
                f"Precomputed directory not found: {self.precomputed_dir}"
            )

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
        if not kp_dicts:
            return []

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
    
    
    def _match_with_method(
        self, 
        method_name: str, 
        kp1: List[cv2.KeyPoint], 
        des1: np.ndarray, 
        kp2: List[cv2.KeyPoint], 
        des2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int, Optional[List[cv2.DMatch]]]:
        """Match descriptors using specified method and compute homography.
        
        Args:
            method_name (str): One of 'orb', 'akaze', 'sift'.
            kp1 (List[cv2.KeyPoint]): Keypoints from template image.
            des1 (np.ndarray): Descriptors from template image.
            kp2 (List[cv2.KeyPoint]): Keypoints from target image.
            des2 (np.ndarray): Descriptors from target image.
            
        Returns:
            M (Optional[np.ndarray]): Homography matrix if found, else None.
            inlier_count (int): Number of inliers from homography.
            good_matches (Optional[List[cv2.DMatch]]): List of good matches, or None.
        """
        if des1 is None or des2 is None:
            return None, 0, None

        if method_name in ("orb", "akaze"):
            matcher = self.bf_hamming
        else:
            matcher = self.bf_l2

        # knn match
        matches = matcher.knnMatch(des1, des2, k=2)

        # ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < self.ratio_thresh * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.min_match_count:
            return None, len(good_matches), None

        # homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if M is None or mask is None:
            return None, len(good_matches), None

        return M, int(np.sum(mask)), good_matches


    def classify_scale_bar(
        self, target_image: np.ndarray
    ) -> Optional[List[Dict[str, Any]]]:
        """Classify an image by matching ORB descriptors against templates.

        Args:
            target_image (np.ndarray): Image to classify (H,W,C or H,W).

        Returns:
            results (Optional[List[Dict[str,Any]]]): Sorted list of match dicts
                containing 'scale_type','template','score','bbox', or None if no descriptors.
        """
        # Convert to grayscale if needed
        if len(target_image.shape) == 3:
            target_gray = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY).copy()
        else:
            target_gray = target_image.copy()

        kp_orb, des_orb     = self.orb.detectAndCompute(target_gray, None)
        kp_akaze, des_akaze = self.akaze.detectAndCompute(target_gray, None)
        kp_sift, des_sift   = self.sift.detectAndCompute(target_gray, None)
        
        # Plot keypoints in red for ORB, green for AKAZE, blue for SIFT for debugging
        img_kp = target_gray.copy()
        if len(img_kp.shape) == 2:
            img_kp = cv2.cvtColor(img_kp, cv2.COLOR_GRAY2BGR)

        img_kp = cv2.drawKeypoints(img_kp, kp_orb, img_kp, color=(0, 0, 255), flags=0)
        img_kp = cv2.drawKeypoints(img_kp, kp_akaze, img_kp, color=(0, 255, 0), flags=0)
        img_kp = cv2.drawKeypoints(img_kp, kp_sift, img_kp, color=(255, 0, 0), flags=0)

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        plt.title("Keypoints: ORB (Red), AKAZE (Green), SIFT (Blue)")
        plt.axis("off")
        plt.show()
        
        if (
            all(des is None for des in [des_orb, des_akaze, des_sift]) or 
            all(len(kp) == 0 for kp in [kp_orb, kp_akaze, kp_sift])
        ):
            return None

        methods = [
            ("orb",   "keypoints_orb",   "descriptors_orb",   kp_orb,   des_orb),
            ("akaze", "keypoints_akaze", "descriptors_akaze", kp_akaze, des_akaze),
            ("sift",  "keypoints_sift",  "descriptors_sift",  kp_sift,  des_sift),
        ]
        results: List[Dict[str, Any]] = []

        for scale_type, templates in self.templates_per_type.items():
            best_score = 0.0
            best_bbox = None
            best_template_fname = None

            for template_info in templates:
                for method, kp_key, des_key, kp2, des2 in methods:
                    # Load template features
                    kp1 = self.reconstruct_keypoints(template_info[kp_key])
                    des1 = template_info[des_key]

                    M, inliers, gm = self._match_with_method(method, kp1, des1, kp2, des2)

                    if M is not None:
                        score = inliers / len(kp1) if len(kp1) else 0
                        if score > best_score:
                            best_score = score
                            best_template_fname = template_info["filename"]
                            h, w = template_info["image_shape"]
                            pts = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
                            dst = cv2.perspectiveTransform(pts, M)
                            best_bbox = dst
                            best_template_fname = template_info["filename"]

            if best_score > self.score_threshold:
                best_bbox = (
                    best_bbox.reshape(-1, 2).astype(int)
                    if best_bbox is not None
                    else None
                )

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
        image: np.ndarray,
        matches: Optional[List[Dict[str, Any]]],
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize and optionally save classification overlay on the image.

        Args:
            target_img (np.ndarray): Image on which to draw detections (BGR/RGB).
            matches (Optional[List[Dict[str,Any]]]): Match results as returned by `classify_scale_bar`.
            save_path (Optional[str], optional): Path to save the visualization image. Defaults to None.

        Returns:
            None: Displays or writes an image depending on `save_path`.
        """
        img = image.copy()
        if matches is None or len(matches) == 0:
            return
        try:
            cm = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            for i, m in enumerate(matches):
                log.info(
                    f"{m['template']}, Type: {m['scale_type']}, Score: {m['score']:.2f}"
                )
                bbox = m["bbox"]
                if bbox is not None:
                    # Draw bounding box on image for visualization
                    bbox_pts = bbox
                    cv2.polylines(
                        img,
                        [bbox_pts.reshape(-1, 2)],
                        isClosed=True,
                        color=cm[i % len(cm)],
                        thickness=2,
                    )
                    legend = f"{m['scale_type']} ({m['score']:.2f})"
                    cv2.putText(
                        img,
                        legend,
                        (bbox_pts[0][0], bbox_pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cm[i % len(cm)],
                        2,
                    )

            plt.imshow(img)
            plt.title("Classification Results")
            plt.axis("off")
            if save_path is not None:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

        except Exception as e:
            log.error(f"Error in save_results: {e}")
