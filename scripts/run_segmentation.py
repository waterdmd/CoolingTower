#!/usr/bin/env python
"""
Run SAM2 segmentation on detected cooling towers.

Usage:
    python scripts/run_segmentation.py \
        --input_dir data/sample_images \
        --detection_file outputs/detections/detections.pkl \
        --output_dir outputs/masks \
        --sam2_checkpoint models/sam2/checkpoints/sam2.1_hiera_base_plus.pt \
        --sam2_config models/sam2/configs/sam2.1
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import CoolingTowerDetector
from src.segmentation import SAM2Segmentor
from src.utils import setup_logging, setup_directories


def parse_args():
    parser = argparse.ArgumentParser(description='Run SAM2 segmentation on detections')
    
    # Required arguments
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--detection_file', type=str, required=True,
                       help='Path to detection results pickle file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save segmentation masks')
    
    # SAM2 model arguments
    parser.add_argument('--sam2_checkpoint', type=str,
                       default='models/sam2/checkpoints/sam2.1_hiera_base_plus.pt',
                       help='Path to SAM2 checkpoint')
    parser.add_argument('--sam2_config', type=str,
                       default='models/sam2/configs/sam2.1',
                       help='Path to SAM2 config directory')
    parser.add_argument('--sam2_config_name', type=str,
                       default='sam2.1_hiera_b+.yaml',
                       help='SAM2 config file name')
    
    # Segmentation arguments
    parser.add_argument('--box_expansion', type=float, default=0.25,
                       help='Percentage to expand bounding boxes')
    parser.add_argument('--fixed_padding', type=int, default=50,
                       help='Fixed pixel padding for boxes')
    parser.add_argument('--mask_size', type=int, default=768,
                       help='Output mask size')
    
    # Processing arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--clear_cache', action='store_true',
                       help='Clear CUDA cache between images')
    
    # Logging arguments
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("COOLING TOWER SEGMENTATION")
    logger.info("=" * 80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Detection file: {args.detection_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"SAM2 checkpoint: {args.sam2_checkpoint}")
    logger.info("=" * 80)
    
    # Load detections
    logger.info("Loading detection results...")
    detections = CoolingTowerDetector.load_detections(args.detection_file)
    logger.info(f"Loaded {len(detections)} detection results")
    
    # Filter detections with boxes
    detections_with_boxes = [d for d in detections if len(d['boxes']) > 0]
    logger.info(f"Found {len(detections_with_boxes)} images with detections")
    
    # Initialize segmentor
    logger.info("Initializing SAM2 segmentor...")
    segmentor = SAM2Segmentor(
        checkpoint_path=args.sam2_checkpoint,
        config_dir=args.sam2_config,
        config_name=args.sam2_config_name,
        device=args.device,
        box_expansion=args.box_expansion,
        fixed_padding=args.fixed_padding
    )
    
    # Run segmentation
    logger.info("Running segmentation...")
    mask_paths = segmentor.segment_from_detections(
        detections=detections,
        output_dir=args.output_dir,
        mask_size=args.mask_size,
        clear_cache=args.clear_cache
    )
    
    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total masks generated: {len(mask_paths)}")
    logger.info(f"Images with detections: {len(detections_with_boxes)}")
    logger.info(f"Masks saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
