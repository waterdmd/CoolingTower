#!/usr/bin/env python3
"""
Helper script to set up SAM2 configuration and checkpoints.

This script helps you download and set up SAM2 properly.
"""

import os
import sys
import shutil
from pathlib import Path
import urllib.request


def download_file(url: str, destination: str):
    """Download a file with progress indication."""
    print(f"Downloading {Path(destination).name}...")
    
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, destination, reporthook)
    print("\n  ✓ Download complete")


def setup_sam2():
    """Set up SAM2 model and configs."""
    print("\n" + "="*60)
    print("SAM2 SETUP")
    print("="*60 + "\n")
    
    # Create directories
    models_dir = Path("models/sam2")
    configs_dir = models_dir / "configs" / "sam2.1"
    checkpoints_dir = models_dir / "checkpoints"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    print("✓ Created directory structure\n")
    
    # Check if SAM2 is installed
    try:
        import sam2
        print("✓ SAM2 package is installed")
        
        # Try to copy config files from package
        sam2_path = Path(sam2.__file__).parent
        sam2_configs = sam2_path / "configs"
        
        if sam2_configs.exists():
            print(f"  Found SAM2 configs at: {sam2_configs}")
            
            # Copy config files
            for yaml_file in sam2_configs.glob("**/*.yaml"):
                rel_path = yaml_file.relative_to(sam2_configs)
                dest = configs_dir.parent / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(yaml_file, dest)
                print(f"  Copied: {rel_path}")
            
            print("\n✓ Config files copied successfully")
        else:
            print("⚠️  Could not find SAM2 config files in package")
            print("  You may need to manually copy them from the SAM2 repository")
    
    except ImportError:
        print("❌ SAM2 is not installed")
        print("\nPlease install SAM2 first:")
        print("  pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        return False
    
    # Checkpoint download info
    print("\n" + "-"*60)
    print("CHECKPOINT DOWNLOAD")
    print("-"*60 + "\n")
    
    checkpoint_urls = {
        "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    }
    
    print("Available checkpoints:")
    for i, name in enumerate(checkpoint_urls.keys(), 1):
        print(f"  {i}. {name}")
    
    print("\nNote: You can download checkpoints manually from:")
    print("  https://github.com/facebookresearch/segment-anything-2/releases")
    print(f"\nPlace checkpoint files in: {checkpoints_dir}")
    
    # Ask user if they want to download
    response = input("\nDownload sam2.1_hiera_base_plus.pt now? (y/n): ")
    
    if response.lower() == 'y':
        checkpoint_path = checkpoints_dir / "sam2.1_hiera_base_plus.pt"
        
        if checkpoint_path.exists():
            print(f"\n✓ Checkpoint already exists: {checkpoint_path}")
        else:
            try:
                download_file(
                    checkpoint_urls["sam2.1_hiera_base_plus.pt"],
                    str(checkpoint_path)
                )
                print(f"✓ Checkpoint saved to: {checkpoint_path}")
            except Exception as e:
                print(f"\n❌ Error downloading checkpoint: {e}")
                print("Please download manually from the link above")
                return False
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60 + "\n")
    
    print("Directory structure:")
    print(f"  {models_dir}/")
    print(f"    ├── checkpoints/")
    print(f"    └── configs/")
    print(f"          └── sam2.1/")
    
    print("\nNext steps:")
    print("  1. Ensure you have a checkpoint file in models/sam2/checkpoints/")
    print("  2. Update config.yaml with the correct paths")
    print("  3. Run segmentation: python scripts/run_segmentation.py")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = setup_sam2()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
