"""
Dataset Merger for Smart Crop Health Monitoring System
Combines multiple YOLO datasets into a single unified dataset
"""

import os
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Set
import hashlib

DATASET_DIR = Path(__file__).parent.parent / "dataset" / "dataset"
DATA_DIR = Path(__file__).parent.parent / "data"
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetMerger:
    """Merges multiple YOLO datasets into a single unified dataset"""
    
    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = {}  # Maps original class names to unified class IDs
        self.unified_classes = []  # List of all unique class names
        self.dataset_stats = defaultdict(int)
        
    def scan_datasets(self) -> List[Path]:
        """Scan for all dataset directories containing data.yaml files"""
        datasets = []
        for item in self.source_dir.iterdir():
            if item.is_dir():
                yaml_file = item / "data.yaml"
                if yaml_file.exists():
                    datasets.append(item)
                    logger.info(f"Found dataset: {item.name}")
        return datasets
    
    def load_dataset_config(self, dataset_path: Path) -> Dict:
        """Load and parse dataset configuration from data.yaml"""
        yaml_file = dataset_path / "data.yaml"
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading {yaml_file}: {e}")
            return {}
    
    def build_unified_class_mapping(self, datasets: List[Path]) -> None:
        """Build unified class mapping across all datasets"""
        all_classes = set()
        dataset_classes = {}
        
        # Collect all unique class names
        for dataset_path in datasets:
            config = self.load_dataset_config(dataset_path)
            if 'names' in config:
                classes = config['names']
                dataset_classes[dataset_path.name] = classes
                
                # Normalize class names (lowercase, remove extra spaces)
                normalized_classes = [self._normalize_class_name(cls) for cls in classes]
                all_classes.update(normalized_classes)
                
                logger.info(f"Dataset {dataset_path.name}: {len(classes)} classes")
        
        # Create unified class list (sorted for consistency)
        self.unified_classes = sorted(list(all_classes))
        logger.info(f"Total unified classes: {len(self.unified_classes)}")
        
        # Create mapping from original class names to unified IDs
        for dataset_name, classes in dataset_classes.items():
            self.class_mapping[dataset_name] = {}
            for original_id, class_name in enumerate(classes):
                normalized_name = self._normalize_class_name(class_name)
                unified_id = self.unified_classes.index(normalized_name)
                self.class_mapping[dataset_name][original_id] = unified_id
    
    def _normalize_class_name(self, class_name: str) -> str:
        """Normalize class names for consistency"""
        # Convert to lowercase and clean up
        normalized = class_name.lower().strip()
        
        # Handle common variations
        replacements = {
            'bacterial leaf spot': 'bacterial spot',
            'bacterial_spot': 'bacterial spot',
            'late_blight': 'late blight',
            'early_blight': 'early blight',
            'spider_mite': 'spider mite',
            'target_spot': 'target spot',
            'septoria_leaf_spot': 'septoria leaf spot',
            'curl_virus': 'curl virus',
            'leaf_mold': 'leaf mold',
        }
        
        for old, new in replacements.items():
            if normalized == old:
                normalized = new
                break
        
        return normalized
    
    def copy_images_and_labels(self, datasets: List[Path]) -> None:
        """Copy and process images and labels from all datasets"""
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            # Create output directories
            images_dir = self.output_dir / split / 'images'
            labels_dir = self.output_dir / split / 'labels'
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            split_count = 0
            
            for dataset_path in datasets:
                dataset_name = dataset_path.name
                source_images = dataset_path / split / 'images'
                source_labels = dataset_path / split / 'labels'
                
                if not source_images.exists():
                    logger.warning(f"No {split} images found in {dataset_name}")
                    continue
                
                # Process images
                for img_file in source_images.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        # Create unique filename to avoid conflicts - consistent naming
                        new_stem = f"{dataset_name}_{img_file.stem}"
                        new_name = f"{new_stem}{img_file.suffix}"
                        dest_img = images_dir / new_name
                        
                        # Copy image
                        shutil.copy2(img_file, dest_img)
                        
                        # Process corresponding label with matching filename
                        label_file = source_labels / f"{img_file.stem}.txt"
                        if label_file.exists():
                            # Use the same stem as the image file
                            dest_label = labels_dir / f"{new_stem}.txt"
                            self._process_label_file(label_file, dest_label, dataset_name)
                        else:
                            logger.warning(f"No label file found for {img_file.name} in {dataset_name}")
                        
                        split_count += 1
            
            self.dataset_stats[f"{split}_images"] = split_count
            logger.info(f"Processed {split_count} images for {split} split")
    
    def _process_label_file(self, source_label: Path, dest_label: Path, dataset_name: str) -> None:
        """Process label file and update class IDs according to unified mapping"""
        try:
            with open(source_label, 'r') as f:
                lines = f.readlines()
            
            processed_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:  # class_id x y w h
                        original_class_id = int(parts[0])
                        
                        # Map to unified class ID
                        if dataset_name in self.class_mapping:
                            if original_class_id in self.class_mapping[dataset_name]:
                                unified_class_id = self.class_mapping[dataset_name][original_class_id]
                                parts[0] = str(unified_class_id)
                                processed_lines.append(' '.join(parts))
                            else:
                                logger.warning(f"Unknown class ID {original_class_id} in {dataset_name}")
                        else:
                            logger.warning(f"No class mapping for dataset {dataset_name}")
            
            # Write processed labels
            with open(dest_label, 'w') as f:
                f.write('\n'.join(processed_lines))
                if processed_lines:  # Add final newline if there are lines
                    f.write('\n')
                    
        except Exception as e:
            logger.error(f"Error processing label file {source_label}: {e}")
    
    def create_unified_yaml(self) -> None:
        """Create unified data.yaml configuration file"""
        # Use absolute paths to avoid path resolution issues
        config = {
            'train': str(self.output_dir / 'train' / 'images'),
            'val': str(self.output_dir / 'valid' / 'images'),
            'test': str(self.output_dir / 'test' / 'images'),
            'nc': len(self.unified_classes),
            'names': self.unified_classes
        }
        
        yaml_file = self.output_dir / 'data.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created unified data.yaml with {len(self.unified_classes)} classes")
    
    def save_class_mapping(self) -> None:
        """Save detailed class mapping information"""
        mapping_file = self.output_dir / 'class_mapping.yaml'
        mapping_data = {
            'unified_classes': self.unified_classes,
            'dataset_mappings': self.class_mapping,
            'statistics': dict(self.dataset_stats)
        }
        
        with open(mapping_file, 'w') as f:
            yaml.dump(mapping_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved class mapping to {mapping_file}")
    
    def merge_datasets(self) -> bool:
        """Main method to merge all datasets"""
        try:
            logger.info("Starting dataset merge process...")
            
            # Clean output directory
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Scan for datasets
            datasets = self.scan_datasets()
            if not datasets:
                logger.error("No datasets found!")
                return False
            
            # Build unified class mapping
            self.build_unified_class_mapping(datasets)
            
            # Copy and process images/labels
            self.copy_images_and_labels(datasets)
            
            # Create configuration files
            self.create_unified_yaml()
            self.save_class_mapping()
            
            # Print summary
            logger.info("\n=== Dataset Merge Summary ===")
            logger.info(f"Source datasets: {len(datasets)}")
            logger.info(f"Unified classes: {len(self.unified_classes)}")
            for key, value in self.dataset_stats.items():
                logger.info(f"{key}: {value}")
            
            logger.info("Dataset merge completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during dataset merge: {e}")
            return False


def main():
    """Main function to run dataset merger"""
    merger = DatasetMerger(DATASET_DIR, DATA_DIR)
    success = merger.merge_datasets()
    
    if success:
        print("\n✅ Dataset merge completed!")
        print(f"📁 Unified dataset location: {DATA_DIR}")
        print(f"📊 Class mapping saved to: {DATA_DIR / 'class_mapping.yaml'}")
        print(f"⚙️  Training config: {DATA_DIR / 'data.yaml'}")
    else:
        print("\n❌ Dataset merge failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
