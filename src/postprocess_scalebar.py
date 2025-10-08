"""
Scale Bar Endpoint Localization

This module implements fine-grained endpoint localization for scale bars following
the Uni-AIMS paper methodology. It refines detected scale bar bounding boxes to
find precise endpoints and compute accurate pixel lengths.

Algorithm:
1. Channel selection (choose channel with strongest edges)
2. Local thresholding for high-contrast binary image
3. Morphological operations to clean up noise
4. Connected component analysis to isolate scale bar
5. Edge projection to find horizontal/vertical orientation
6. Peak detection to find endpoints
7. Subpixel refinement using center of mass
8. Compute pixel length and return endpoints

Usage:
    python postprocess_scalebar.py --image data/images/val/9.jpg --model models/train/weights/best.pt --output_dir outputs --visualize
"""

import os
import cv2
import numpy as np
import json
from typing import Tuple, List, Optional, Dict, Any
from scipy.signal import find_peaks
from skimage.filters import threshold_local
import matplotlib.pyplot as plt
from ultralytics import YOLO
from dataclasses import dataclass


@dataclass
class ScalebarDetection:
    """Data class for scale bar detection results."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    pixel_length: Optional[float] = None
    endpoints: Optional[List[Tuple[float, float]]] = None


class ScalebarProcessor:
    """Class for post-processing scale bar detection results."""
    def localize_scalebar_endpoints(
            self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
            plot_path: Optional[str] = None
        ) -> ScalebarDetection:
        """
        Main function to localize scale bar endpoints within a bounding box.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
        """
        x, y, w, h = bbox
        
        # Extract ROI
        roi = image[y:y+h, x:x+w]

        if roi.size == 0:
            return ScalebarDetection(
                bbox=bbox, confidence=0.0, pixel_length=0.0, endpoints=None
            )
        
        try:
            # Step 1: Channel selection
            best_channel = self.select_best_channel(roi)
            
            # Step 2: Local thresholding
            radius = 2*h
            binary = self.apply_local_thresholding(best_channel, radius=radius)

            # Step 3: Morphological cleanup (opening and closing)
            kernel_size = int(h/4) | 1  # Ensure odd kernel size
            cleaned = self.morphological_cleanup(binary, kernel_size=kernel_size)

            # Step 4: Find largest component (/!\ Only works if scale bar is in one piece)
            largest_component = self.find_largest_component(cleaned)

            # Step 5: Compute edge projection
            v_proj, h_proj = self.compute_edge_projection(largest_component)

            # Step 6: Find endpoints
            min_length = int(0.5 * np.max((w, h)))
            x_start, x_end = self.find_scalebar_endpoints(v_proj, min_length) # For horizontal scalebar
            y_start, y_end = self.find_scalebar_endpoints(h_proj, min_length) # For vertical scalebar

            # Step 7: Direction determination
            if (x_end - x_start) >= (y_end - y_start):
                start, end = x_start, x_end
                center = h // 2
                orientation = 'horizontal'
            else:
                start, end = y_start, y_end
                center = w // 2
                orientation = 'vertical'

            # Step 8: Subpixel refinement
            refined_start, refined_end = self.subpixel_refinement(
                best_channel, start, end, center, window_size=center, orientation=orientation
            )

            # Step 9: Compute pixel length
            pixel_length = abs(refined_end - refined_start)

            # Step 10: Compute absolute coordinates for verbose output
            if orientation == 'horizontal':
                abs_y = int(y + center)
                endpoints = [(int(x + refined_start), abs_y), (int(x + refined_end), abs_y)]
            else:
                abs_x = int(x + center)
                endpoints = [(abs_x, int(y + refined_start)), (abs_x, int(y + refined_end))]

            if plot_path is not None:
                self.visualize_endpoint_detection(
                    image, 
                    (x, y, w, h), 
                    {
                        'pixel_length': pixel_length, 'endpoints': endpoints,
                        'best_channel': best_channel, 'binary_image': binary,
                        'cleaned_image': cleaned, 'largest_component': largest_component,
                        'v_projection': v_proj, 'h_projection': h_proj,
                        'x_peaks1': x_start, 'x_peaks2': x_end,
                        'y_peaks1': y_start, 'y_peaks2': y_end
                    },
                    save_path=plot_path + '_scalebar.png'
                )
                    
            return ScalebarDetection(
                bbox=bbox,
                confidence=1.0,
                pixel_length=pixel_length,
                endpoints=endpoints
            )

        except Exception as e:
            return ScalebarDetection(
                bbox=bbox, confidence=0.0, pixel_length=0.0, endpoints=None
            )


    def select_best_channel(self, image: np.ndarray) -> np.ndarray:
        """
        Select the channel with the strongest edges for scale bar detection.
        
        Args:
            image: Input image (H, W, C) or (H, W) for grayscale
            
        Returns:
            Single channel image with strongest edges
        """
        # If image is grayscale, return as is
        if len(image.shape) == 2:
            return image
            
        # Compute Sobel magnitude for each channel
        sobel_magnitudes = []
        for i in range(image.shape[2]):
            channel = image[:, :, i]
            sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_magnitudes.append(np.mean(magnitude))
        
        # Select channel with highest edge strength
        best_channel_idx = np.argmax(sobel_magnitudes)
        return image[:, :, best_channel_idx].astype(np.uint8)


    def apply_local_thresholding(self, image, radius=15):
        """
        Apply local thresholding to generate a high-contrast binary image.

        Args:
            image: Grayscale input image (values in [0, 255] or [0, 1]).
            radius: Radius of the local neighborhood used to compute local thresholds.

        Returns:
            binary: Binarized image with True for foreground and False for background.
        """
        # Compute local threshold using skimage's threshold_local
        block_size = 2 * radius + 1
        local_thresh = threshold_local(image, block_size, method='gaussian')

        # Apply threshold
        binary = image > local_thresh

        return binary.astype(np.uint8) * 255


    def morphological_cleanup(self, binary: np.ndarray, kernel_size) -> np.ndarray:
        """
        Apply morphological operations to clean up the binary image.
        
        Args:
            binary: Binary image
            kernel_size: Size of morphological kernel
            
        Returns:
            Cleaned binary image
        """
        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Remove small white noise
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Fill small black noise
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned


    def find_largest_component(self, binary: np.ndarray) -> np.ndarray:
        """
        Find the largest connected component in the binary image.
        
        Args:
            binary: Binary image
            
        Returns:
            Binary image with only the largest component
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        if num_labels <= 1:
            return binary
        
        # Find largest component (excluding background)
        largest_comp = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if num_labels > 2:
            _2nd_largest_comp = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[-2]
            if stats[largest_comp, cv2.CC_STAT_AREA] < \
                1.5 * stats[_2nd_largest_comp, cv2.CC_STAT_AREA]:
                return binary  # Return original if no dominant component

        # Create mask for largest component
        mask = (labels == largest_comp).astype(np.uint8) * 255

        return mask


    def compute_edge_projection(self, binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute vertical edge projection to find horizontal scale bar.
        
        Args:
            binary: Binary image
            
        Returns:
            Vertical projection profile
        """    
        # Compute vertical gradients
        gradx = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=1)
        grady = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=1)
        gradmag = cv2.magnitude(gradx, grady)

        # Sum gradients vertically and horizontally
        vertical_projection = np.mean(gradmag, axis=0)
        horizontal_projection = np.mean(gradmag, axis=1)

        # Normalize projections
        vertical_projection = vertical_projection / np.max(vertical_projection) \
            if np.max(vertical_projection) > 0 else vertical_projection
        horizontal_projection = horizontal_projection / np.max(horizontal_projection) \
            if np.max(horizontal_projection) > 0 else horizontal_projection
        return vertical_projection, horizontal_projection


    def find_scalebar_endpoints(self, projection: np.ndarray, min_length: int) -> Tuple[int, int]:
        """
        Find scale bar endpoints using peak detection on the projection profile.
        
        Args:
            projection: Vertical projection profile
            min_length: Minimum length for scale bar
            peak_prominence: Minimum prominence for peaks
            
        Returns:
            Tuple of (start_x, end_x) coordinates
        """
        # Find peaks
        peaks, properties = find_peaks(projection, distance=min_length)

        # Keep the two most prominent peaks
        if len(peaks) > 2:
            # Select two most prominent peaks
            prominences = properties['prominences']
            top_two_indices = np.argsort(prominences)[-2:]
            peaks = peaks[top_two_indices]
            peaks = np.sort(peaks)

        if len(peaks) < 2:
            return (0, 0)
        
        start = peaks[0]
        end = peaks[-1]

        return (start, end)


    def subpixel_refinement(self, image: np.ndarray, start: int, end: int, center: int, 
            window_size: int, orientation: str = 'horizontal',
        ) -> Tuple[float, float]:
        """
        Refine endpoint coordinates using subpixel interpolation.
        
        Args:
            image: Original grayscale image
            start: Approximate start coordinate
            end: Approximate end coordinate
            center: Center coordinate (y for horizontal, x for vertical)
            orientation: Orientation of the scale bar ('horizontal' or 'vertical')
            window_size: Window size for refinement
            
        Returns:
            Refined (start, end) coordinates
        """
        if orientation == 'vertical':
            # Transpose for vertical processing
            image_ = image.T.copy()
        else:
            image_ = image.copy()
        

        def refine_single_point(point_to_adjust, ref_point):
            # Extract window around the point
            ref_start = max(0, ref_point - window_size // 2)
            ref_end = min(image_.shape[0], ref_point + window_size // 2 + 1)
            refine_window_start = max(0, point_to_adjust - window_size // 2)
            refine_window_end = min(image_.shape[1], point_to_adjust + window_size // 2 + 1)

            window = image_[ref_start:ref_end, refine_window_start:refine_window_end]

            if window.size == 0:
                return point_to_adjust

            # Compute center of mass
            moments = cv2.moments(window)
            if moments['m00'] > 0:
                center_of_mass = moments['m10'] / moments['m00']
                adjusted_point = refine_window_start + center_of_mass
                return adjusted_point

            return point_to_adjust

        # Refine both endpoints
        refined_start = refine_single_point(start, center)
        refined_end = refine_single_point(end, center)

        return refined_start, refined_end


    def visualize_endpoint_detection(
            self,
            image: np.ndarray, 
            xywh: Tuple[int, int, int, int],
            results: Dict[str, Any],
            save_path: Optional[str] = None
        ) -> None:
        """
        Visualize the endpoint detection process and results.
        
        Args:
            image: Original image
            xywh: padded bounding box (x, y, width, height)
            results: Results from localize_scalebar_endpoints
            save_path: Path to save visualization (optional)
        """
        x, y, w, h = xywh
        roi = image[y:y+h, x:x+w]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Original ROI
        axes[0, 0].imshow(roi, cmap='gray')
        axes[0, 0].set_title('Original ROI')
        axes[0, 0].axis('off')
        
        # Best channel
        if 'best_channel' in results and results['best_channel'] is not None:
            axes[0, 1].imshow(results['best_channel'], cmap='gray')
            axes[0, 1].set_title('Best Channel')
            axes[0, 1].axis('off')

        # Binary image
        if 'binary_image' in results and results['binary_image'] is not None:
            axes[0, 2].imshow(results['binary_image'], cmap='gray')
            axes[0, 2].set_title('Binary Image')
            axes[0, 2].axis('off')

        # Cleaned image
        if 'cleaned_image' in results and results['cleaned_image'] is not None:
            axes[0, 3].imshow(results['cleaned_image'], cmap='gray')
            axes[0, 3].set_title('Cleaned Image')
            axes[0, 3].axis('off')

        # Largest component
        if 'largest_component' in results and results['largest_component'] is not None:
            axes[1, 0].imshow(results['largest_component'], cmap='gray')
            axes[1, 0].set_title('Largest Component')
            axes[1, 0].axis('off')

        # Projection profile
        if 'v_projection' in results and results['v_projection'] is not None:
            axes[1, 1].plot(results['v_projection'])
            if results.get('x_peaks1') and results.get('x_peaks2'):
                axes[1, 1].axvline(results['x_peaks1'], color='r', linestyle='--')
                axes[1, 1].axvline(results['x_peaks2'], color='r', linestyle='--')
        if 'h_projection' in results and results['h_projection'] is not None:
            axes[1, 1].plot(results['h_projection'])
            if results.get('x_peaks1') and results.get('x_peaks2'):
                axes[1, 1].axvline(results['x_peaks1'], color='g', linestyle='--')
                axes[1, 1].axvline(results['x_peaks2'], color='g', linestyle='--')
        axes[1, 1].set_xlabel('Pixel position')
        axes[1, 1].set_ylabel('Projection strength')
        axes[1, 1].grid(True)
        axes[1, 1].set_title('Edge Projection Profiles')

        # Result visualization
        axes[1, 2].imshow(roi, cmap='gray')
        if results['endpoints']:
            start_pt, end_pt = results['endpoints']
            # Convert back to ROI coordinates
            rel_start = (start_pt[0] - x, start_pt[1] - y)
            rel_end = (end_pt[0] - x, end_pt[1] - y)
            
            axes[1, 2].plot([rel_start[0], rel_end[0]], [rel_start[1], rel_end[1]], 
                        'r-', linewidth=3, label='Detected scale bar')
            axes[1, 2].plot(rel_start[0], rel_start[1], 'go', markersize=8, label='Start')
            axes[1, 2].plot(rel_end[0], rel_end[1], 'ro', markersize=8, label='End')
            axes[1, 2].legend()

        axes[1, 2].set_title(f'Result (Length: {results["pixel_length"]:.1f}px)')
        axes[1, 2].axis('off')
        
        # Full image with detection
        axes[1, 3].imshow(image, cmap='gray')
        if results['endpoints']:
            start_pt, end_pt = results['endpoints']
            axes[1, 3].plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                        'r-', linewidth=3, label='Detected scale bar')
            axes[1, 3].plot(start_pt[0], start_pt[1], 'go', markersize=8, label='Start')
            axes[1, 3].plot(end_pt[0], end_pt[1], 'ro', markersize=8, label='End')
            axes[1, 3].legend()
        
        # Draw bounding box
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
        axes[1, 3].add_patch(rect)
        axes[1, 3].set_title('Full Image Detection')
        axes[1, 3].axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")