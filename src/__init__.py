"""
Cooling Tower Detection and Segmentation Package
"""

from .detection import CoolingTowerDetector
from .segmentation import SAM2Segmentor
from .utils import load_config, setup_directories, setup_logging
from .visualization import (
    draw_boxes_on_image,
    visualize_mask_overlay,
    create_comparison_grid,
    plot_detection_statistics
)

__version__ = "1.0.0"
__author__ = "Aman Jain"

__all__ = [
    'CoolingTowerDetector',
    'SAM2Segmentor',
    'load_config',
    'setup_directories',
    'setup_logging',
    'draw_boxes_on_image',
    'visualize_mask_overlay',
    'create_comparison_grid',
    'plot_detection_statistics',
]
