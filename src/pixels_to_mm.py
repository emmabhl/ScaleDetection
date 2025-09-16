"""
Pixel to Physical Unit Conversion Utilities

This module provides utilities for converting pixel coordinates and measurements
to physical units (mm, μm, nm) based on detected scale bars.

Features:
- Convert pixel coordinates to physical coordinates
- Convert pixel areas to physical areas
- Convert pixel distances to physical distances
- Support for different physical units
- Batch conversion utilities
- Validation and error handling
"""

import numpy as np
from typing import Union, List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import math


@dataclass
class ScaleInfo:
    """Data class for scale information."""
    um_per_pixel: float
    pixel_length: float
    physical_length: float
    unit: str
    confidence: float = 1.0


class PixelToPhysicalConverter:
    """Main class for pixel to physical unit conversions."""
    
    def __init__(self, scale_info: ScaleInfo):
        """
        Initialize converter with scale information.
        
        Args:
            scale_info: Scale information containing conversion factor
        """
        self.scale_info = scale_info
        self.um_per_pixel = scale_info.um_per_pixel
        
        # Validate scale info
        if self.um_per_pixel <= 0:
            raise ValueError("um_per_pixel must be positive")
    
    def pixels_to_um(self, pixels: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert pixels to micrometers.
        
        Args:
            pixels: Pixel value(s) to convert
            
        Returns:
            Converted value(s) in micrometers
        """
        return pixels * self.um_per_pixel
    
    def pixels_to_mm(self, pixels: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert pixels to millimeters.
        
        Args:
            pixels: Pixel value(s) to convert
            
        Returns:
            Converted value(s) in millimeters
        """
        um = self.pixels_to_um(pixels)
        return um / 1000.0
    
    def pixels_to_nm(self, pixels: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert pixels to nanometers.
        
        Args:
            pixels: Pixel value(s) to convert
            
        Returns:
            Converted value(s) in nanometers
        """
        um = self.pixels_to_um(pixels)
        return um * 1000.0
    
    def convert_distance(self, pixel_distance: float, target_unit: str = 'mm') -> float:
        """
        Convert pixel distance to target physical unit.
        
        Args:
            pixel_distance: Distance in pixels
            target_unit: Target unit ('mm', 'um', 'nm')
            
        Returns:
            Distance in target unit
        """
        if target_unit == 'mm':
            return self.pixels_to_mm(pixel_distance)
        elif target_unit == 'um':
            return self.pixels_to_um(pixel_distance)
        elif target_unit == 'nm':
            return self.pixels_to_nm(pixel_distance)
        else:
            raise ValueError(f"Unsupported unit: {target_unit}")
    
    def convert_coordinates(self, pixel_coords: Union[List[Tuple[float, float]], 
                                                    np.ndarray], 
                          target_unit: str = 'mm') -> Union[List[Tuple[float, float]], 
                                                          np.ndarray]:
        """
        Convert pixel coordinates to physical coordinates.
        
        Args:
            pixel_coords: List of (x, y) tuples or numpy array of shape (N, 2)
            target_unit: Target unit ('mm', 'um', 'nm')
            
        Returns:
            Converted coordinates in target unit
        """
        if isinstance(pixel_coords, list):
            # Convert list of tuples
            converted = []
            for x, y in pixel_coords:
                conv_x = self.convert_distance(x, target_unit)
                conv_y = self.convert_distance(y, target_unit)
                converted.append((conv_x, conv_y))
            return converted
        elif isinstance(pixel_coords, np.ndarray):
            # Convert numpy array
            if pixel_coords.shape[1] != 2:
                raise ValueError("Coordinate array must have shape (N, 2)")
            
            converted = np.zeros_like(pixel_coords)
            converted[:, 0] = self.convert_distance(pixel_coords[:, 0], target_unit)
            converted[:, 1] = self.convert_distance(pixel_coords[:, 1], target_unit)
            return converted
        else:
            raise TypeError("pixel_coords must be list of tuples or numpy array")
    
    def convert_area(self, pixel_area: float, target_unit: str = 'mm') -> float:
        """
        Convert pixel area to physical area.
        
        Args:
            pixel_area: Area in square pixels
            target_unit: Target unit ('mm', 'um', 'nm')
            
        Returns:
            Area in square target unit
        """
        # Convert linear dimension first
        linear_conversion = self.convert_distance(1.0, target_unit)
        
        # Area conversion is square of linear conversion
        return pixel_area * (linear_conversion ** 2)
    
    def convert_bbox(self, bbox: Tuple[float, float, float, float], 
                    target_unit: str = 'mm') -> Tuple[float, float, float, float]:
        """
        Convert bounding box from pixels to physical units.
        
        Args:
            bbox: Bounding box as (x, y, width, height) in pixels
            target_unit: Target unit ('mm', 'um', 'nm')
            
        Returns:
            Bounding box in target unit
        """
        x, y, w, h = bbox
        
        # Convert position and dimensions
        conv_x = self.convert_distance(x, target_unit)
        conv_y = self.convert_distance(y, target_unit)
        conv_w = self.convert_distance(w, target_unit)
        conv_h = self.convert_distance(h, target_unit)
        
        return (conv_x, conv_y, conv_w, conv_h)
    
    def convert_mask_coordinates(self, mask: np.ndarray, 
                               target_unit: str = 'mm') -> np.ndarray:
        """
        Convert mask coordinates to physical units.
        
        Args:
            mask: Binary mask array
            target_unit: Target unit ('mm', 'um', 'nm')
            
        Returns:
            Mask with physical unit coordinates
        """
        # Get coordinates of non-zero pixels
        coords = np.column_stack(np.where(mask > 0))
        
        if len(coords) == 0:
            return coords
        
        # Convert coordinates (note: mask coordinates are (y, x))
        converted_coords = np.zeros_like(coords, dtype=float)
        converted_coords[:, 1] = self.convert_distance(coords[:, 1], target_unit)  # x
        converted_coords[:, 0] = self.convert_distance(coords[:, 0], target_unit)  # y
        
        return converted_coords
    
    def calculate_physical_area(self, mask: np.ndarray, target_unit: str = 'mm') -> float:
        """
        Calculate physical area of a binary mask.
        
        Args:
            mask: Binary mask array
            target_unit: Target unit ('mm', 'um', 'nm')
            
        Returns:
            Area in square target unit
        """
        pixel_area = np.sum(mask > 0)
        return self.convert_area(pixel_area, target_unit)
    
    def calculate_physical_perimeter(self, mask: np.ndarray, target_unit: str = 'mm') -> float:
        """
        Calculate physical perimeter of a binary mask.
        
        Args:
            mask: Binary mask array
            target_unit: Target unit ('mm', 'um', 'nm')
            
        Returns:
            Perimeter in target unit
        """
        from scipy import ndimage
        
        # Find perimeter using morphological operations
        eroded = ndimage.binary_erosion(mask)
        perimeter_mask = mask & ~eroded
        
        pixel_perimeter = np.sum(perimeter_mask > 0)
        return self.convert_distance(pixel_perimeter, target_unit)
    
    def calculate_physical_centroid(self, mask: np.ndarray, target_unit: str = 'mm') -> Tuple[float, float]:
        """
        Calculate physical centroid of a binary mask.
        
        Args:
            mask: Binary mask array
            target_unit: Target unit ('mm', 'um', 'nm')
            
        Returns:
            Centroid coordinates in target unit
        """
        from scipy import ndimage
        
        # Calculate centroid in pixels
        centroid = ndimage.center_of_mass(mask)
        y_pixel, x_pixel = centroid
        
        # Convert to physical units
        x_physical = self.convert_distance(x_pixel, target_unit)
        y_physical = self.convert_distance(y_pixel, target_unit)
        
        return (x_physical, y_physical)
    
    def get_scale_info(self) -> Dict[str, Any]:
        """
        Get scale information as dictionary.
        
        Returns:
            Dictionary containing scale information
        """
        return {
            'um_per_pixel': self.um_per_pixel,
            'pixel_length': self.scale_info.pixel_length,
            'physical_length': self.scale_info.physical_length,
            'unit': self.scale_info.unit,
            'confidence': self.scale_info.confidence,
            'mm_per_pixel': self.um_per_pixel / 1000.0,
            'nm_per_pixel': self.um_per_pixel * 1000.0
        }


class BatchConverter:
    """Batch converter for multiple images with different scales."""
    
    def __init__(self):
        """Initialize batch converter."""
        self.converters = {}
    
    def add_converter(self, image_id: str, scale_info: ScaleInfo) -> None:
        """
        Add converter for specific image.
        
        Args:
            image_id: Unique identifier for image
            scale_info: Scale information for the image
        """
        self.converters[image_id] = PixelToPhysicalConverter(scale_info)
    
    def convert_coordinates_batch(self, image_id: str, pixel_coords: Union[List[Tuple[float, float]], 
                                                                          np.ndarray],
                                target_unit: str = 'mm') -> Union[List[Tuple[float, float]], 
                                                                np.ndarray]:
        """
        Convert coordinates for specific image.
        
        Args:
            image_id: Image identifier
            pixel_coords: Pixel coordinates to convert
            target_unit: Target unit
            
        Returns:
            Converted coordinates
        """
        if image_id not in self.converters:
            raise ValueError(f"No converter found for image {image_id}")
        
        return self.converters[image_id].convert_coordinates(pixel_coords, target_unit)
    
    def convert_areas_batch(self, image_id: str, pixel_areas: List[float], 
                          target_unit: str = 'mm') -> List[float]:
        """
        Convert areas for specific image.
        
        Args:
            image_id: Image identifier
            pixel_areas: List of pixel areas
            target_unit: Target unit
            
        Returns:
            List of converted areas
        """
        if image_id not in self.converters:
            raise ValueError(f"No converter found for image {image_id}")
        
        converter = self.converters[image_id]
        return [converter.convert_area(area, target_unit) for area in pixel_areas]
    
    def get_all_scale_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get scale information for all images.
        
        Returns:
            Dictionary mapping image IDs to scale information
        """
        return {img_id: converter.get_scale_info() for img_id, converter in self.converters.items()}


def create_converter_from_matches(matches: List[Any]) -> Optional[PixelToPhysicalConverter]:
    """
    Create converter from OCR matching results.
    
    Args:
        matches: List of matched scale bar and text results
        
    Returns:
        PixelToPhysicalConverter or None if no valid matches
    """
    if not matches:
        return None
    
    # Use the match with highest confidence
    best_match = max(matches, key=lambda m: m.scale_bar.confidence * m.text.confidence)
    
    if best_match.um_per_pixel is None or best_match.um_per_pixel <= 0:
        return None
    
    # Create scale info
    scale_info = ScaleInfo(
        um_per_pixel=best_match.um_per_pixel,
        pixel_length=best_match.scale_bar.pixel_length or 0,
        physical_length=best_match.text.parsed_value or 0,
        unit=best_match.text.normalized_unit or 'um',
        confidence=best_match.scale_bar.confidence * best_match.text.confidence
    )
    
    return PixelToPhysicalConverter(scale_info)


def validate_conversion(converter: PixelToPhysicalConverter, 
                       test_pixel_length: float, 
                       expected_physical_length: float, 
                       tolerance: float = 0.1) -> bool:
    """
    Validate conversion accuracy using known measurements.
    
    Args:
        converter: PixelToPhysicalConverter instance
        test_pixel_length: Known pixel length
        expected_physical_length: Expected physical length
        tolerance: Acceptable error tolerance (fraction)
        
    Returns:
        True if conversion is within tolerance
    """
    converted_length = converter.pixels_to_um(test_pixel_length)
    error = abs(converted_length - expected_physical_length) / expected_physical_length
    
    return error <= tolerance


def create_conversion_report(converter: PixelToPhysicalConverter, 
                           test_measurements: List[Tuple[float, float, str]] = None) -> Dict[str, Any]:
    """
    Create a comprehensive conversion report.
    
    Args:
        converter: PixelToPhysicalConverter instance
        test_measurements: List of (pixel_length, expected_physical_length, unit) tuples
        
    Returns:
        Dictionary containing conversion report
    """
    scale_info = converter.get_scale_info()
    
    report = {
        'scale_info': scale_info,
        'conversion_factors': {
            'pixels_to_mm': scale_info['mm_per_pixel'],
            'pixels_to_um': scale_info['um_per_pixel'],
            'pixels_to_nm': scale_info['nm_per_pixel']
        },
        'validation_results': []
    }
    
    if test_measurements:
        for pixel_len, expected_phys, unit in test_measurements:
            if unit == 'mm':
                expected_um = expected_phys * 1000
            elif unit == 'um':
                expected_um = expected_phys
            elif unit == 'nm':
                expected_um = expected_phys / 1000
            else:
                continue
            
            converted_um = converter.pixels_to_um(pixel_len)
            error = abs(converted_um - expected_um) / expected_um if expected_um > 0 else float('inf')
            
            report['validation_results'].append({
                'pixel_length': pixel_len,
                'expected_physical_length': expected_phys,
                'expected_unit': unit,
                'converted_um': converted_um,
                'error_fraction': error,
                'within_tolerance': error <= 0.1
            })
    
    return report


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pixel to physical unit conversion')
    parser.add_argument('--um_per_pixel', type=float, required=True, 
                       help='Micrometers per pixel conversion factor')
    parser.add_argument('--pixel_length', type=float, required=True,
                       help='Pixel length of reference scale bar')
    parser.add_argument('--physical_length', type=float, required=True,
                       help='Physical length of reference scale bar')
    parser.add_argument('--unit', type=str, default='um', choices=['um', 'nm', 'mm'],
                       help='Unit of physical length')
    parser.add_argument('--confidence', type=float, default=1.0,
                       help='Confidence of scale detection')
    parser.add_argument('--test_pixel', type=float, help='Test pixel measurement')
    parser.add_argument('--test_expected', type=float, help='Expected physical measurement')
    parser.add_argument('--test_unit', type=str, default='um', choices=['um', 'nm', 'mm'],
                       help='Unit of expected measurement')
    
    args = parser.parse_args()
    
    # Convert physical length to micrometers
    if args.unit == 'mm':
        physical_length_um = args.physical_length * 1000
    elif args.unit == 'nm':
        physical_length_um = args.physical_length / 1000
    else:  # um
        physical_length_um = args.physical_length
    
    # Create scale info
    scale_info = ScaleInfo(
        um_per_pixel=args.um_per_pixel,
        pixel_length=args.pixel_length,
        physical_length=physical_length_um,
        unit='um',
        confidence=args.confidence
    )
    
    # Create converter
    converter = PixelToPhysicalConverter(scale_info)
    
    # Print scale information
    print("Scale Information:")
    scale_info_dict = converter.get_scale_info()
    for key, value in scale_info_dict.items():
        print(f"  {key}: {value}")
    
    # Test conversion if provided
    if args.test_pixel is not None and args.test_expected is not None:
        print(f"\nTest Conversion:")
        print(f"  Pixel measurement: {args.test_pixel}")
        print(f"  Expected physical: {args.test_expected} {args.test_unit}")
        
        # Convert expected to micrometers
        if args.test_unit == 'mm':
            expected_um = args.test_expected * 1000
        elif args.test_unit == 'nm':
            expected_um = args.test_expected / 1000
        else:  # um
            expected_um = args.test_expected
        
        converted_um = converter.pixels_to_um(args.test_pixel)
        error = abs(converted_um - expected_um) / expected_um if expected_um > 0 else float('inf')
        
        print(f"  Converted: {converted_um:.6f} um")
        print(f"  Error: {error:.2%}")
        print(f"  Within tolerance: {error <= 0.1}")


if __name__ == "__main__":
    main()
