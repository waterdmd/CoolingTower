"""
Utility functions for cooling tower detection and segmentation.
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(base_dir: str, subdirs: list = None) -> Dict[str, str]:
    """
    Create output directories if they don't exist.
    
    Args:
        base_dir: Base output directory
        subdirs: List of subdirectory names to create
        
    Returns:
        Dictionary mapping subdirectory names to their paths
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    dir_paths = {'base': str(base_path)}
    
    if subdirs:
        for subdir in subdirs:
            subdir_path = base_path / subdir
            subdir_path.mkdir(parents=True, exist_ok=True)
            dir_paths[subdir] = str(subdir_path)
    
    return dir_paths


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to save logs to file
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def get_image_files(directory: str, extensions: tuple = ('.png', '.jpg', '.jpeg')) -> list:
    """
    Get list of image files from directory.
    
    Args:
        directory: Directory to search
        extensions: Tuple of valid file extensions
        
    Returns:
        List of image file paths
    """
    image_files = []
    for ext in extensions:
        image_files.extend(Path(directory).glob(f"*{ext}"))
    return sorted([str(f) for f in image_files])


def expand_bounding_box(x1: int, y1: int, x2: int, y2: int, 
                        image_shape: tuple, 
                        padding_percentage: float = 0.25,
                        fixed_padding: int = 50) -> tuple:
    """
    Expand a bounding box by a given percentage and fixed padding.
    
    Args:
        x1, y1, x2, y2: Original bounding box coordinates
        image_shape: Shape of the image (height, width, channels)
        padding_percentage: Percentage to expand the box
        fixed_padding: Fixed pixel padding to add
        
    Returns:
        Tuple of expanded coordinates (x1, y1, x2, y2)
    """
    height, width = image_shape[:2]
    
    box_width = x2 - x1
    box_height = y2 - y1
    
    # Calculate expansion
    expand_w = int(box_width * padding_percentage) + fixed_padding
    expand_h = int(box_height * padding_percentage) + fixed_padding
    
    # Apply expansion with bounds checking
    x1 = max(0, x1 - expand_w)
    y1 = max(0, y1 - expand_h)
    x2 = min(width, x2 + expand_w)
    y2 = min(height, y2 + expand_h)
    
    return x1, y1, x2, y2


def validate_bounding_box(x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Validate that bounding box coordinates are valid.
    
    Args:
        x1, y1, x2, y2: Bounding box coordinates
        
    Returns:
        True if valid, False otherwise
    """
    return x2 > x1 and y2 > y1


def merge_configs(base_config: Dict[str, Any], 
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
