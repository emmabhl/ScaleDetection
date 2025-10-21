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

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from skimage.filters import threshold_local
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.signal


@dataclass
class ScalebarDetection:
    """Data class for scale bar detection results."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    pixel_length: Optional[float] = None
    endpoints: Optional[List[Tuple[float, float]]] = None
    flag: Optional[bool] = False
    #uncertainty: Optional[float] = None


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
                bbox=bbox, pixel_length=0.0, endpoints=[None, None]#, uncertainty=0.0
            )

        try:
            # Step 1: Channel selection
            best_channel = self.select_best_channel(roi)
            
            # Step 2: Local thresholding
            binary = self.apply_local_thresholding(best_channel, h=h)

            # Step 3: Morphological cleanup (opening and closing)
            kernel_size = int(self.bar_width_estimation(binary) / 2) + 1
            cleaned = self.morphological_cleanup(binary, kernel_size=kernel_size)

            # Step 4: Find largest component (/!\ Only works if scale bar is in one piece)
            largest_component = self.find_largest_component(cleaned)
            
            # Step 5: Skeletonization to thin the scale bar
            skeleton = self.skeletonize(largest_component)

            # Step 6: Find endpoints
            start, end = self.find_endpoints(skeleton) # For horizontal scalebar

            # Step 7: Compute pixel length
            pixel_length = self.compute_flat_distance(start, end)

            if plot_path is not None:
                self.visualize_endpoint_detection(
                    image, 
                    (x, y, w, h), 
                    {
                        'pixel_length': pixel_length, 'best_channel': best_channel, 
                        'binary_image': binary, 'cleaned_image': cleaned, 
                        'largest_component': largest_component, 'skeleton': skeleton,
                        'endpoints': [(start[1]+x, start[0]+y), (end[1]+x, end[0]+y)]
                    },
                    save_path=plot_path + '_scalebar.png'
                )
                    
            return ScalebarDetection(
                bbox=bbox,
                pixel_length=pixel_length,
                endpoints=[(start[1]+x, start[0]+y), (end[1]+x, end[0]+y)],
                #uncertainty=0.0 # Placeholder for uncertainty estimation,
                flag=True if pixel_length < 0.75 * max(w, h) else False
            )

        except Exception as e:
            print(f"Error occurred while localizing scale bar endpoints: {e}")
            
            if plot_path is not None:
                self.visualize_endpoint_detection(
                    image, 
                    (x, y, w, h), 
                    {
                        'pixel_length': 0.0, 
                        'best_channel': best_channel if 'best_channel' in locals() else None,
                        'binary_image': binary if 'binary' in locals() else None,
                        'cleaned_image': cleaned if 'cleaned' in locals() else None,
                        'largest_component': largest_component if 'largest_component' in locals() else None,
                        'skeleton': skeleton if 'skeleton' in locals() else None,
                        'endpoints': None
                    },
                    save_path=plot_path + '_scalebar_error.png'
                )
            return ScalebarDetection(
                bbox=bbox, pixel_length=0.0, endpoints=[None, None]#, uncertainty=0.0
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


    def apply_local_thresholding(self, image: np.ndarray, h: int):
        """
        Apply local thresholding to generate a high-contrast binary image.

        Args:
            image: Grayscale input image (values in [0, 255] or [0, 1]).
            radius: Radius of the local neighborhood used to compute local thresholds.

        Returns:
            binary: Binarized image with True for foreground and False for background.
        """
        # Compute local threshold using skimage's threshold_local
        block_size = 4 * h + 1
        local_thresh = threshold_local(image, block_size, method='mean')
        
        # Apply threshold
        binary = image > local_thresh
        
        # Determine if background is white or black by checking border pixels
        border_pixels = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
        middle_pixels = binary[
            binary.shape[0]//4:-binary.shape[0]//4, 
            binary.shape[1]//4:-binary.shape[1]//4
        ].flatten()
        is_background_white = np.mean(border_pixels) > np.mean(middle_pixels)

        # If border is mostly white, assume scale bar is black → invert
        if is_background_white:
            binary = ~binary
        
        return binary.astype(np.uint8) * 255
    
    
    def bar_width_estimation(self, binary: np.ndarray) -> int:
        """
        Estimate the scale bar width from the binary image.
        
        Args:
            binary: Binary image
        Returns:
            Estimated bar width in pixels
        """
        # Get smaller dimension
        h, w = binary.shape
        if h < w:
            proj = np.sum(binary, axis=1)
        else:
            proj = np.sum(binary, axis=0)

        # Count the number of lines with more than 50% pixels on
        num_lines = np.sum(proj > 0.5 * np.max(proj))
        return num_lines

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
        
        # If no components or only background, return original
        if num_labels <= 1:
            return binary
                
        # Find largest component (excluding background)
        largest_comp = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if num_labels > 2:
            _2nd_largest = 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[-2]
            if stats[largest_comp, cv2.CC_STAT_AREA] < 1.5 * stats[_2nd_largest, cv2.CC_STAT_AREA]:
                return binary  # Return original if no dominant component

        # Create mask for largest component
        mask = (labels == largest_comp).astype(np.uint8) * 255

        return mask

    def skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """
        Skeletonize the binary image using thinning algorithm.
        
        Args:
            binary: Binary image
            
        Returns:
            Skeletonized binary image
        """
        # Make sure the border is background
        binary[0, :] = 0
        binary[-1, :] = 0
        binary[:, 0] = 0
        binary[:, -1] = 0

        # Apply thinning algorithm
        skeleton = cv2.ximgproc.thinning(binary)
        return skeleton
    

    def find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find scale bar endpoints in the skeletonized image
        
        Args:
            skeleton: Skeletonized binary image
            min_length: Minimum length for scale bar
            
        Returns:
            Two endpoints (x1, y1), (x2, y2)
        """
        # Kernel to sum the neighbours
        kernel = [[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]]
        # 2D convolution (cast image to int32 to avoid overflow)
        img_conv = scipy.signal.convolve2d(skeleton.astype(np.int32), kernel, mode='same')
        # Pick points where pixel is 255 and neighbours sum 255
        endpoints = np.stack(np.where((skeleton == 255) & (img_conv == 255)), axis=1)
        
        if len(endpoints) < 2:
            raise ValueError("Less than 2 endpoints detected in skeletonized scale bar.")
        elif len(endpoints) > 2:
            # If more than 2 endpoints, pick the two farthest apart
            dists = np.linalg.norm(endpoints[:, None] - endpoints[None, :], axis=2)
            np.fill_diagonal(dists, 0)  # Ignore self-distances
            max_dist_idx = np.unravel_index(np.argmax(dists), dists.shape)
            endpoints = endpoints[list(max_dist_idx)]
        else:
            endpoints = endpoints.tolist()
                        
        return endpoints

    def compute_flat_distance(self, start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """
        Compute the flat (purely horizontal/vertical) distance between two points.

        Args:
            start: Starting point (x1, y1)
            end: Ending point (x2, y2)

        Returns:
            Distance in pixels
        """
        h_dist = abs(end[0] - start[0])
        v_dist = abs(end[1] - start[1])
        return max(h_dist, v_dist)
        

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

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # Full image
        axes[0, 0].imshow(image, cmap='gray')
        
        # Draw bounding box
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='blue', facecolor='none')
        axes[0, 0].add_patch(rect)
        axes[0, 0].set_title('Full Image Detection')
        axes[0, 0].axis('off')

        # Original ROI
        axes[0, 1].imshow(roi, cmap='gray')
        axes[0, 1].set_title('Original ROI')
        axes[0, 1].axis('off')

        # Best channel
        if 'best_channel' in results and results['best_channel'] is not None:
            # Histogram of best channel
            axes[0, 2].hist(results['best_channel'].ravel(), bins=256, color='gray', alpha=0.7)
            axes[0, 2].set_title('Best Channel Distribution')
            axes[0, 2].set_xlabel('Pixel Intensity')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].grid(True)
            axes[0, 2].set_xlim(0, 255)
            
            # Show best channel image
            axes[1, 0].imshow(results['best_channel'], cmap='gray')
            axes[1, 0].set_title('Best Channel')
            axes[1, 0].axis('off')
            
        # Binary image
        if 'binary_image' in results and results['binary_image'] is not None:
            axes[1, 1].imshow(results['binary_image'], cmap='gray')
            axes[1, 1].set_title('Binary Image')
            axes[1, 1].axis('off')

        # Cleaned image
        if 'cleaned_image' in results and results['cleaned_image'] is not None:
            axes[1, 2].imshow(results['cleaned_image'], cmap='gray')
            axes[1, 2].set_title('Cleaned Image')
            axes[1, 2].axis('off')

        # Largest component
        if 'largest_component' in results and results['largest_component'] is not None:
            axes[2, 0].imshow(results['largest_component'], cmap='gray')
            axes[2, 0].set_title('Largest Component')
            axes[2, 0].axis('off')
            
        # Skeleton
        if 'skeleton' in results and results['skeleton'] is not None:
            axes[2, 1].imshow(results['skeleton'], cmap='gray')
            axes[2, 1].set_title('Skeleton')
            axes[2, 1].axis('off')


        # Result visualization
        axes[2, 2].imshow(roi, cmap='gray')
        if results['endpoints']:
            start_pt, end_pt = results['endpoints']
            # Convert back to ROI coordinates
            rel_start = (start_pt[0] - x, start_pt[1] - y)
            rel_end = (end_pt[0] - x, end_pt[1] - y)

            axes[2, 2].plot([rel_start[0], rel_end[0]], [rel_start[1], rel_end[1]],
                            'r-', linewidth=3, label='Detected scale bar')
            axes[2, 2].plot(rel_start[0], rel_start[1], 'go', markersize=8, label='Start')
            axes[2, 2].plot(rel_end[0], rel_end[1], 'ro', markersize=8, label='End')
            axes[2, 2].legend()

        axes[2, 2].set_title(f'Result (Length: {results["pixel_length"]:.1f}px)')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
            
        plt.close(fig)