"""
Cooling tower detection using YOLO.
"""
import os
import logging
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

from .utils import get_image_files, validate_bounding_box

logger = logging.getLogger(__name__)


class CoolingTowerDetector:
    """
    Cooling tower detector using YOLO.
    """
    
    def __init__(self, 
                 model_path: str,
                 conf_threshold: float = 0.4,
                 img_size: int = 768,
                 augment: bool = True,
                 device: str = "cuda",
                 verbose: bool = False):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            img_size: Input image size for YOLO
            augment: Whether to use test-time augmentation
            device: Device to run inference on ('cuda' or 'cpu')
            verbose: Whether to print verbose YOLO output
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.augment = augment
        self.device = device
        self.verbose = verbose
        
        # Load model
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
    
    def detect_single(self, image_path: str) -> Dict:
        """
        Run detection on a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing detection results
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            imgsz=self.img_size,
            augment=self.augment,
            verbose=self.verbose
        )
        
        # Extract boxes and confidences
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
        confidences = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
        
        return {
            'image_path': image_path,
            'boxes': boxes,
            'confidences': confidences,
            'num_detections': len(boxes)
        }
    
    def detect_batch(self, 
                     image_dir: str,
                     num_workers: int = 4,
                     save_results: bool = True,
                     output_file: Optional[str] = None) -> List[Dict]:
        """
        Run detection on a batch of images with multi-threading.
        
        Args:
            image_dir: Directory containing images
            num_workers: Number of worker threads
            save_results: Whether to save results to file
            output_file: Path to save results (if save_results=True)
            
        Returns:
            List of detection result dictionaries
        """
        # Get all image files
        image_files = get_image_files(image_dir)
        logger.info(f"Found {len(image_files)} images in {image_dir}")
        
        if len(image_files) == 0:
            logger.warning("No images found!")
            return []
        
        # Process images with threading
        all_results = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results_iter = tqdm(
                executor.map(self._process_single_threaded, image_files),
                total=len(image_files),
                desc="Detecting cooling towers"
            )
            
            for result in results_iter:
                if result is not None:
                    all_results.append(result)
        
        # Log statistics
        total_detections = sum(r['num_detections'] for r in all_results)
        logger.info(f"Detected {total_detections} cooling towers across {len(all_results)} images")
        
        # Save results if requested
        if save_results:
            if output_file is None:
                output_file = os.path.join(image_dir, "detections.pkl")
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'wb') as f:
                pickle.dump(all_results, f)
            logger.info(f"Results saved to {output_file}")
        
        return all_results
    
    def _process_single_threaded(self, image_path: str) -> Optional[Dict]:
        """
        Thread-safe single image processing.
        Each thread loads its own YOLO model to avoid conflicts.
        
        Args:
            image_path: Path to image
            
        Returns:
            Detection results or None if processing failed
        """
        try:
            # Load model in thread (YOLO is not thread-safe when sharing instances)
            thread_model = YOLO(self.model_path)
            
            results = thread_model.predict(
                source=image_path,
                conf=self.conf_threshold,
                imgsz=self.img_size,
                augment=self.augment,
                verbose=False
            )
            
            boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
            confidences = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else np.array([])
            
            return {
                'image_path': image_path,
                'boxes': boxes,
                'confidences': confidences,
                'num_detections': len(boxes)
            }
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def visualize_detections(self, 
                            image_path: str,
                            boxes: np.ndarray,
                            confidences: np.ndarray,
                            output_path: str,
                            box_color: Tuple[int, int, int] = (0, 255, 0),
                            thickness: int = 2):
        """
        Visualize detection results on an image.
        
        Args:
            image_path: Path to original image
            boxes: Array of bounding boxes
            confidences: Array of confidence scores
            output_path: Path to save visualization
            box_color: BGR color for boxes
            thickness: Line thickness
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            if validate_bounding_box(x1, y1, x2, y2):
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), box_color, thickness)
                
                # Add confidence label
                label = f'{conf:.2f}'
                cv2.putText(image_rgb, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, thickness)
        
        # Save visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
        
    @staticmethod
    def load_detections(detection_file: str) -> List[Dict]:
        """
        Load detection results from pickle file.
        
        Args:
            detection_file: Path to pickle file
            
        Returns:
            List of detection dictionaries
        """
        with open(detection_file, 'rb') as f:
            detections = pickle.load(f)
        return detections
