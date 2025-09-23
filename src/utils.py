import numpy as np
from typing import List, Tuple

def points_to_xywh(points: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Convert polygon points to bounding box (x, y, width, height).

    Args:
        points (List[Tuple[int, int]]): List of points defining the polygon 
        (x_min, y_min), (x_max, y_max).

    Returns:
        Tuple[int, int, int, int]: Bounding box in (x_min, y_min, width, height) format.
    """
    points = np.array(points)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    return bbox