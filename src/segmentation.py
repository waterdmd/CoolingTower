"""
Cooling tower segmentation using SAM2.
"""
import os
import sys
import site
import logging
import shutil
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add user site packages to path for SAM2
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    logger.error("SAM2 not found. Please install SAM2 from https://github.com/facebookresearch/segment-anything-2")
    raise

from .utils import expand_bounding_box, validate_bounding_box

logger = logging.getLogger(__name__)


class SAM2Segmentor:
    """
    Segmentation using fine-tuned SAM2 model.
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 config_dir: str,
                 config_name: str = "sam2.1_hiera_b+.yaml",
                 device: str = "cuda",
                 box_expansion: float = 0.25,
                 fixed_padding: int = 50):
        """
        Initialize SAM2 segmentor.
        
        Args:
            checkpoint_path: Path to SAM2 checkpoint file
            config_dir: Directory containing SAM2 config files
            config_name: Name of config file
            device: Device to run on ('cuda' or 'cpu')
            box_expansion: Percentage to expand bounding boxes
            fixed_padding: Fixed pixel padding for boxes
        """
        self.checkpoint_path = checkpoint_path
        self.config_dir = config_dir
        self.config_name = config_name
        self.device = device
        self.box_expansion = box_expansion
        self.fixed_padding = fixed_padding
        
        # Load SAM2 model
        self._load_sam2_model()
        
    def _load_sam2_model(self):
        """Load and initialize SAM2 model."""
        logger.info(f"Loading SAM2 from {self.checkpoint_path}")
        
        # Validate paths
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not os.path.exists(self.config_dir):
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        
        config_path = os.path.join(self.config_dir, self.config_name)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            # Clear any existing Hydra sessions
            GlobalHydra.instance().clear()
            
            # Initialize Hydra with config directory
            with initialize_config_dir(config_dir=self.config_dir, version_base=None):
                # Load configuration
                cfg = compose(config_name=self.config_name)
                
                # Build model architecture
                logger.info("Building SAM2 model architecture...")
                sam2_model = instantiate(cfg.model, _recursive_=True)
                
                # Load weights
                logger.info("Loading model weights...")
                state_dict = torch.load(self.checkpoint_path, map_location="cpu")
                
                # Unwrap weights if nested under 'model' key
                if "model" in state_dict:
                    logger.info("Unwrapping 'model' key from checkpoint")
                    state_dict = state_dict["model"]
                
                sam2_model.load_state_dict(state_dict, strict=True)
                
                # Move to device and set to eval mode
                sam2_model = sam2_model.to(self.device)
                sam2_model.eval()
                
                # Create predictor
                self.predictor = SAM2ImagePredictor(sam2_model)
                
                logger.info("SAM2 model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading SAM2 model: {e}")
            raise
    
    def segment_from_detections(self,
                                detections: List[Dict],
                                output_dir: str,
                                mask_size: int = 768,
                                clear_cache: bool = True) -> Dict[str, str]:
        """
        Generate segmentation masks from detection results.
        
        Args:
            detections: List of detection dictionaries
            output_dir: Directory to save masks
            mask_size: Output mask size
            clear_cache: Whether to clear CUDA cache between images
            
        Returns:
            Dictionary mapping image paths to mask paths
        """
        os.makedirs(output_dir, exist_ok=True)
        mask_paths = {}
        
        logger.info(f"Segmenting {len(detections)} images")
        
        for detection in tqdm(detections, desc="Generating masks"):
            image_path = detection['image_path']
            boxes = detection['boxes']
            
            # Generate output path
            image_name = os.path.basename(image_path)
            mask_path = os.path.join(output_dir, image_name)
            
            if len(boxes) == 0:
                # No detections - save empty mask
                empty_mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
                Image.fromarray(empty_mask).save(mask_path)
                mask_paths[image_path] = mask_path
                continue
            
            # Load image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create full mask
            full_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
            
            # Process each detection
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Expand box
                x1, y1, x2, y2 = expand_bounding_box(
                    x1, y1, x2, y2, 
                    image_rgb.shape,
                    self.box_expansion,
                    self.fixed_padding
                )
                
                if not validate_bounding_box(x1, y1, x2, y2):
                    logger.warning(f"Invalid box in {image_name}: ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                # Extract region
                region = image_rgb[y1:y2, x1:x2]
                
                try:
                    # Run SAM2 on region
                    self.predictor.set_image(region)
                    input_box = np.array([[0, 0, region.shape[1], region.shape[0]]])
                    masks, _, _ = self.predictor.predict(box=input_box, multimask_output=True)
                    
                    # Merge masks
                    for mask in masks:
                        resized = cv2.resize(
                            mask.astype(np.uint8), 
                            (x2 - x1, y2 - y1),
                            interpolation=cv2.INTER_NEAREST
                        )
                        full_mask[y1:y2, x1:x2] |= resized
                    
                except RuntimeError as e:
                    logger.error(f"SAM2 error for {image_name}: {e}")
                    continue
            
            # Resize and save mask
            if full_mask.shape != (mask_size, mask_size):
                full_mask = cv2.resize(full_mask, (mask_size, mask_size), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # Convert to binary and save
            binary_mask = (full_mask > 0).astype(np.uint8) * 255
            Image.fromarray(binary_mask).save(mask_path)
            mask_paths[image_path] = mask_path
            
            # Clear CUDA cache if requested
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        
        logger.info(f"Generated {len(mask_paths)} masks in {output_dir}")
        return mask_paths
    
    def segment_single(self, 
                       image_path: str,
                       boxes: np.ndarray,
                       output_path: str,
                       mask_size: int = 768) -> str:
        """
        Segment a single image given bounding boxes.
        
        Args:
            image_path: Path to image
            boxes: Array of bounding boxes
            output_path: Path to save mask
            mask_size: Output mask size
            
        Returns:
            Path to saved mask
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create full mask
        full_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
        
        # Process each box
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Expand box
            x1, y1, x2, y2 = expand_bounding_box(
                x1, y1, x2, y2,
                image_rgb.shape,
                self.box_expansion,
                self.fixed_padding
            )
            
            if not validate_bounding_box(x1, y1, x2, y2):
                continue
            
            # Extract and segment region
            region = image_rgb[y1:y2, x1:x2]
            self.predictor.set_image(region)
            input_box = np.array([[0, 0, region.shape[1], region.shape[0]]])
            masks, _, _ = self.predictor.predict(box=input_box, multimask_output=True)
            
            # Merge masks
            for mask in masks:
                resized = cv2.resize(mask.astype(np.uint8), (x2 - x1, y2 - y1),
                                   interpolation=cv2.INTER_NEAREST)
                full_mask[y1:y2, x1:x2] |= resized
        
        # Resize and save
        if full_mask.shape != (mask_size, mask_size):
            full_mask = cv2.resize(full_mask, (mask_size, mask_size),
                                  interpolation=cv2.INTER_NEAREST)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        binary_mask = (full_mask > 0).astype(np.uint8) * 255
        Image.fromarray(binary_mask).save(output_path)
        
        return output_path
