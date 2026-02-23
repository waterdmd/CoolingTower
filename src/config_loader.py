"""Configuration loader for the cooling tower detection pipeline."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the detection pipeline."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default.
        """
        if config_path is None:
            # Default to config/config.yaml relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """Validate that required configuration keys exist."""
        required_sections = ['detection', 'segmentation', 'processing', 'paths']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Key in dot notation (e.g., 'detection.confidence_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def create_directories(self):
        """Create all output directories specified in config."""
        paths = self.config.get('paths', {})
        
        for key, path in paths.items():
            if key.endswith('_dir'):
                Path(path).mkdir(parents=True, exist_ok=True)
                print(f"✓ Created directory: {path}")
    
    @property
    def detection_config(self) -> Dict[str, Any]:
        """Get detection configuration."""
        return self.config.get('detection', {})
    
    @property
    def segmentation_config(self) -> Dict[str, Any]:
        """Get segmentation configuration."""
        return self.config.get('segmentation', {})
    
    @property
    def processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return self.config.get('processing', {})
    
    @property
    def paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self.config.get('paths', {})
    
    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = Config()
        print("✓ Configuration loaded successfully")
        print(f"\nDetection confidence threshold: {config.get('detection.confidence_threshold')}")
        print(f"Device: {config.get('processing.device')}")
        print(f"Input directory: {config.get('paths.input_dir')}")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
