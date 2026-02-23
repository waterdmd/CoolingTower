#!/usr/bin/env python
"""
Run cooling tower detection on a directory of images.

Usage:
    python scripts/run_detection.py \
        --input_dir data/sample_images \
        --output_dir outputs/detections \
        --model_path models/yolo/best.pt \
        --conf_threshold 0.4 \
        --num_workers 4
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import CoolingTowerDetector
from src.utils import setup_logging, setup_directories
from src.visualization import draw_boxes_on_image, plot_detection_statistics


def parse_args():
    parser = argparse.ArgumentParser(description='Run cooling tower detection')
    
    # Required arguments
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save detection results')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default='models/yolo/best.pt',
                       help='Path to YOLO model weights')
    parser.add_argument('--conf_threshold', type=float, default=0.4,
                       help='Confidence threshold for detections')
    parser.add_argument('--img_size', type=int, default=768,
                       help='Input image size for YOLO')
    parser.add_argument('--augment', action='store_true',
                       help='Use test-time augmentation')
    
    # Processing arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker threads')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    
    # Output arguments
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization images')
    parser.add_argument('--plot_stats', action='store_true',
                       help='Plot detection statistics')
    
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
    logger.info("COOLING TOWER DETECTION")
    logger.info("=" * 80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Confidence threshold: {args.conf_threshold}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info("=" * 80)
    
    # Setup output directories
    dirs = setup_directories(args.output_dir, subdirs=['visualizations', 'stats'])
    
    # Initialize detector
    detector = CoolingTowerDetector(
        model_path=args.model_path,
        conf_threshold=args.conf_threshold,
        img_size=args.img_size,
        augment=args.augment,
        device=args.device,
        verbose=False
    )
    
    # Run detection
    output_file = os.path.join(args.output_dir, 'detections.pkl')
    detections = detector.detect_batch(
        image_dir=args.input_dir,
        num_workers=args.num_workers,
        save_results=True,
        output_file=output_file
    )
    
    # Visualize results
    if args.visualize and len(detections) > 0:
        logger.info("Creating visualizations...")
        vis_dir = dirs['visualizations']
        
        for detection in detections[:10]:  # Visualize first 10
            image_path = detection['image_path']
            boxes = detection['boxes']
            confs = detection['confidences']
            
            output_path = os.path.join(
                vis_dir,
                os.path.basename(image_path)
            )
            
            draw_boxes_on_image(
                image_path, boxes, confs, output_path,
                conf_threshold=args.conf_threshold
            )
        
        logger.info(f"Visualizations saved to {vis_dir}")
    
    # Plot statistics
    if args.plot_stats and len(detections) > 0:
        logger.info("Plotting statistics...")
        stats_path = os.path.join(dirs['stats'], 'detection_stats.png')
        plot_detection_statistics(detections, stats_path)
        logger.info(f"Statistics plot saved to {stats_path}")
    
    # Summary
    total_detections = sum(d['num_detections'] for d in detections)
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total images processed: {len(detections)}")
    logger.info(f"Total cooling towers detected: {total_detections}")
    logger.info(f"Average detections per image: {total_detections/len(detections):.2f}")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
