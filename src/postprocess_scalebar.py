"""
Scale Bar Endpoint Localization

This module implements fine-grained endpoint localization for scale bars following
the Uni-AIMS paper methodology. It refines detected scale bar bounding boxes to
find precise endpoints and compute accurate pixel lengths.

Algorithm:
1. Channel selection (choose channel with strongest edges)
2. Local adaptive thresholding (Sauvola or Otsu)
3. Morphological operations to clean up noise
4. Vertical edge projection and peak detection
5. Subpixel refinement for accurate endpoint detection
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.filters import threshold_sauvola, threshold_otsu
import matplotlib.pyplot as plt


def select_best_channel(image: np.ndarray) -> np.ndarray:
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


def apply_adaptive_threshold(image: np.ndarray, method: str = 'sauvola', 
                           window_size: int = 15, k: float = 0.2) -> np.ndarray:
    """
    Apply adaptive thresholding to binarize the image.
    
    Args:
        image: Input grayscale image
        method: Thresholding method ('sauvola' or 'otsu')
        window_size: Window size for Sauvola thresholding
        k: Parameter for Sauvola thresholding
        
    Returns:
        Binary image
    """
    if method == 'sauvola':
        threshold = threshold_sauvola(image, window_size=window_size, k=k)
    elif method == 'otsu':
        threshold = threshold_otsu(image)
    else:
        raise ValueError(f"Unknown thresholding method: {method}")
    
    binary = image > threshold
    return binary.astype(np.uint8) * 255


def morphological_cleanup(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
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
    
    # Remove small noise
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Fill small holes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def find_largest_component(binary: np.ndarray) -> np.ndarray:
    """
    Find the largest connected component in the binary image.
    
    Args:
        binary: Binary image
        
    Returns:
        Binary image with only the largest component
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels <= 1:
        return binary
    
    # Find largest component (excluding background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_component = np.argmax(areas) + 1
    
    # Create mask for largest component
    mask = (labels == largest_component).astype(np.uint8) * 255
    
    return mask


def compute_vertical_edge_projection(binary: np.ndarray) -> np.ndarray:
    """
    Compute vertical edge projection to find horizontal scale bar.
    
    Args:
        binary: Binary image
        
    Returns:
        Vertical projection profile
    """
    # Compute vertical gradients
    sobel_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    
    # Sum along vertical axis to get horizontal profile
    projection = np.sum(np.abs(sobel_y), axis=0)
    
    return projection


def find_scale_bar_endpoints(projection: np.ndarray, min_length: int = 10, 
                           peak_prominence: float = 0.1) -> Tuple[int, int]:
    """
    Find scale bar endpoints using peak detection on the projection profile.
    
    Args:
        projection: Vertical projection profile
        min_length: Minimum length for scale bar
        peak_prominence: Minimum prominence for peaks
        
    Returns:
        Tuple of (start_x, end_x) coordinates
    """
    # Normalize projection
    projection_norm = projection / np.max(projection) if np.max(projection) > 0 else projection
    
    # Find peaks
    peaks, properties = find_peaks(projection_norm, prominence=peak_prominence, 
                                 distance=min_length)
    
    if len(peaks) < 2:
        # Fallback: use threshold-based approach
        threshold = np.mean(projection_norm) + np.std(projection_norm)
        active_regions = projection_norm > threshold
        
        if not np.any(active_regions):
            return 0, len(projection_norm) - 1
        
        # Find start and end of active region
        start_idx = np.where(active_regions)[0][0]
        end_idx = np.where(active_regions)[0][-1]
        
        return start_idx, end_idx
    
    # Use first and last significant peaks
    start_x = peaks[0]
    end_x = peaks[-1]
    
    return start_x, end_x


def subpixel_refinement(image: np.ndarray, start_x: int, end_x: int, 
                       y_center: int, window_size: int = 5) -> Tuple[float, float]:
    """
    Refine endpoint coordinates using subpixel interpolation.
    
    Args:
        image: Original grayscale image
        start_x: Approximate start x coordinate
        end_x: Approximate end x coordinate
        y_center: Y coordinate of the scale bar
        window_size: Window size for refinement
        
    Returns:
        Refined (start_x, end_x) coordinates
    """
    def refine_single_point(x, y):
        # Extract window around the point
        y_start = max(0, y - window_size // 2)
        y_end = min(image.shape[0], y + window_size // 2 + 1)
        x_start = max(0, x - window_size // 2)
        x_end = min(image.shape[1], x + window_size // 2 + 1)
        
        window = image[y_start:y_end, x_start:x_end]
        
        if window.size == 0:
            return x
        
        # Compute center of mass
        moments = cv2.moments(window)
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            refined_x = x_start + cx
            return refined_x
        
        return x
    
    # Refine both endpoints
    refined_start_x = refine_single_point(start_x, y_center)
    refined_end_x = refine_single_point(end_x, y_center)
    
    return refined_start_x, refined_end_x


def localize_scale_bar_endpoints(
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int],
        method: str = 'sauvola', 
        window_size: int = 15,
        k: float = 0.2, 
        min_length: int = 10,
        peak_prominence: float = 0.1
    ) -> Dict[str, Any]:
    """
    Main function to localize scale bar endpoints within a bounding box.
    
    Args:
        image: Input image
        bbox: Bounding box (x, y, width, height)
        method: Thresholding method ('sauvola' or 'otsu')
        window_size: Window size for Sauvola thresholding
        k: Parameter for Sauvola thresholding
        min_length: Minimum length for scale bar
        peak_prominence: Minimum prominence for peaks
        
    Returns:
        Dictionary containing endpoint coordinates and metadata
    """
    x, y, w, h = bbox
    
    # Extract ROI
    roi = image[y:y+h, x:x+w]
    
    if roi.size == 0:
        return {
            'success': False,
            'error': 'Empty ROI',
            'endpoints': None,
            'pixel_length': 0,
            'confidence': 0.0
        }
    
    try:
        # Step 1: Channel selection
        best_channel = select_best_channel(roi)
        
        # Step 2: Adaptive thresholding
        binary = apply_adaptive_threshold(best_channel, method, window_size, k)
        
        # Step 3: Morphological cleanup
        cleaned = morphological_cleanup(binary)
        
        # Step 4: Find largest component
        largest_component = find_largest_component(cleaned)
        
        # Step 5: Compute vertical edge projection
        projection = compute_vertical_edge_projection(largest_component)
        
        # Step 6: Find endpoints
        start_x, end_x = find_scale_bar_endpoints(projection, min_length, peak_prominence)
        
        # Step 7: Subpixel refinement
        y_center = h // 2
        refined_start_x, refined_end_x = subpixel_refinement(
            best_channel, start_x, end_x, y_center
        )
        
        # Convert back to original image coordinates
        abs_start_x = x + refined_start_x
        abs_end_x = x + refined_end_x
        
        # Compute pixel length
        pixel_length = abs(abs_end_x - abs_start_x)
        
        # Compute confidence based on projection strength
        projection_strength = np.max(projection) / np.mean(projection) if np.mean(projection) > 0 else 0
        confidence = min(1.0, projection_strength / 10.0)
        
        return {
            'success': True,
            'error': None,
            'endpoints': [(abs_start_x, y + y_center), (abs_end_x, y + y_center)],
            'pixel_length': pixel_length,
            'confidence': confidence,
            'projection': projection,
            'binary_image': largest_component
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'endpoints': None,
            'pixel_length': 0,
            'confidence': 0.0
        }


def visualize_endpoint_detection(image: np.ndarray, bbox: Tuple[int, int, int, int],
                               result: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Visualize the endpoint detection process and results.
    
    Args:
        image: Original image
        bbox: Bounding box (x, y, width, height)
        result: Result from localize_scale_bar_endpoints
        save_path: Path to save visualization (optional)
    """
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original ROI
    axes[0, 0].imshow(roi, cmap='gray')
    axes[0, 0].set_title('Original ROI')
    axes[0, 0].axis('off')
    
    # Best channel
    if 'binary_image' in result and result['binary_image'] is not None:
        axes[0, 1].imshow(result['binary_image'], cmap='gray')
        axes[0, 1].set_title('Binary Image')
        axes[0, 1].axis('off')
    
    # Projection profile
    if 'projection' in result and result['projection'] is not None:
        axes[0, 2].plot(result['projection'])
        axes[0, 2].set_title('Vertical Projection')
        axes[0, 2].set_xlabel('X coordinate')
        axes[0, 2].set_ylabel('Projection strength')
        axes[0, 2].grid(True)
    
    # Result visualization
    axes[1, 0].imshow(roi, cmap='gray')
    if result['success'] and result['endpoints']:
        start_pt, end_pt = result['endpoints']
        # Convert back to ROI coordinates
        rel_start = (start_pt[0] - x, start_pt[1] - y)
        rel_end = (end_pt[0] - x, end_pt[1] - y)
        
        axes[1, 0].plot([rel_start[0], rel_end[0]], [rel_start[1], rel_end[1]], 
                       'r-', linewidth=3, label='Detected scale bar')
        axes[1, 0].plot(rel_start[0], rel_start[1], 'go', markersize=8, label='Start')
        axes[1, 0].plot(rel_end[0], rel_end[1], 'ro', markersize=8, label='End')
        axes[1, 0].legend()
    
    axes[1, 0].set_title(f'Result (Length: {result["pixel_length"]:.1f}px)')
    axes[1, 0].axis('off')
    
    # Full image with detection
    axes[1, 1].imshow(image, cmap='gray')
    if result['success'] and result['endpoints']:
        start_pt, end_pt = result['endpoints']
        axes[1, 1].plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 
                       'r-', linewidth=3, label='Detected scale bar')
        axes[1, 1].plot(start_pt[0], start_pt[1], 'go', markersize=8, label='Start')
        axes[1, 1].plot(end_pt[0], end_pt[1], 'ro', markersize=8, label='End')
        axes[1, 1].legend()
    
    # Draw bounding box
    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
    axes[1, 1].add_patch(rect)
    axes[1, 1].set_title('Full Image Detection')
    axes[1, 1].axis('off')
    
    # Hide unused subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def batch_localize_endpoints(images: List[np.ndarray], bboxes: List[Tuple[int, int, int, int]],
                           **kwargs) -> List[Dict[str, Any]]:
    """
    Localize endpoints for multiple images and bounding boxes.
    
    Args:
        images: List of input images
        bboxes: List of bounding boxes for each image
        **kwargs: Additional arguments for localize_scale_bar_endpoints
        
    Returns:
        List of results for each image
    """
    results = []
    
    for i, (image, bbox) in enumerate(zip(images, bboxes)):
        print(f"Processing image {i+1}/{len(images)}")
        result = localize_scale_bar_endpoints(image, bbox, **kwargs)
        results.append(result)
    
    return results


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scale bar endpoint localization')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--bbox', type=str, required=True, help='Bounding box as "x,y,w,h"')
    parser.add_argument('--method', type=str, default='sauvola', choices=['sauvola', 'otsu'],
                       help='Thresholding method')
    parser.add_argument('--window_size', type=int, default=15, help='Window size for Sauvola')
    parser.add_argument('--k', type=float, default=0.2, help='Parameter for Sauvola')
    parser.add_argument('--min_length', type=int, default=10, help='Minimum scale bar length')
    parser.add_argument('--peak_prominence', type=float, default=0.1, help='Peak prominence threshold')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--save_path', type=str, help='Path to save visualization')
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Parse bounding box
    bbox = tuple(map(int, args.bbox.split(',')))
    
    # Localize endpoints
    result = localize_scale_bar_endpoints(
        image, bbox, 
        method=args.method,
        window_size=args.window_size,
        k=args.k,
        min_length=args.min_length,
        peak_prominence=args.peak_prominence
    )
    
    # Print results
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Endpoints: {result['endpoints']}")
        print(f"Pixel length: {result['pixel_length']:.2f}")
        print(f"Confidence: {result['confidence']:.3f}")
    else:
        print(f"Error: {result['error']}")
    
    # Visualize if requested
    if args.visualize:
        visualize_endpoint_detection(image, bbox, result, args.save_path)


if __name__ == "__main__":
    main()
