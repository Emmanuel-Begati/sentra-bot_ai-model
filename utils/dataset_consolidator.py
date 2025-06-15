"""
Dataset Consolidation Script for Plant Disease Detection
Combines similar diseases and prepares data for YOLOv8 object detection
"""

import yaml
import json
import shutil
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import re
from difflib import SequenceMatcher
from datetime import datetime

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import DATA_YAML_PATH, DATASET_DIR
except ImportError:
    # If config import fails, use relative paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_YAML_PATH = PROJECT_ROOT / "data" / "data.yaml"
    DATASET_DIR = PROJECT_ROOT / "data"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiseaseConsolidator:
    """Consolidates similar diseases and prepares YOLOv8 dataset"""
    
    def __init__(self):
        self.dataset_dir = DATASET_DIR
        self.original_data_yaml = DATA_YAML_PATH
        self.class_mapping_file = DATASET_DIR / "class_mapping.yaml"
        
        # Load existing data
        self.original_classes = self._load_original_classes()
        self.class_mappings = self._load_class_mappings()
        
        # Enhanced disease consolidation rules based on your actual 178 classes
        self.consolidation_rules = {
            # Blight diseases - combining all blight types
            'blight': [
                'blight', 'early_blight', 'earlyblight', 'late_blight', 'lateblight',
                'bacterial blight', 'bacterial-leaf-blight', 'leaf_blight',
                'brownblight', 'celery early blight', 'potato early blight',
                'potato late blight', 'tomato early blight', 'tomato late blight',
                'carrot alternaria leaf blight', 'garlic leaf blight',
                'corn northern leaf blight', 'bean halo blight', 'cassava bacterial blight',
                'soybean bacterial blight', 'botrytis_blight', 'blueberry botrytis blight',
                'eggplant phytophthora blight'
            ],
            
            # All spot diseases
            'leaf_spot': [
                'spot', 'leaf_spot', 'brown_spot', 'brown-spot', 'brown sport', 'black_spot',
                'target_spot', 'target spot', 'angular leaf spot', 'cercospora_spot',
                'septoria_spot', 'narrow-brown-spot', 'cherry leaf spot',
                'grape leaf spot', 'raspberry leaf spot', 'septoria leaf spot',
                'tomato septoria leaf spot', 'bell pepper bacterial spot',
                'tomato bacterial leaf spot', 'plum bacterial spot',
                'broccoli alternaria leaf spot', 'cabbage alternaria leaf spot',
                'cauliflower alternaria leaf spot', 'eggplant cercospora leaf spot',
                'corn gray leaf spot', 'grey_leaf_spot', 'cucumber angular leaf spot',
                'ginger leaf spot', 'maple tar spot', 'soybean frog eye leaf spot',
                'tobacco frogeye leaf spot', 'frogeye leaf spot',
                'bell pepper frogeye leaf spot', 'coffee brown eye spot',
                'tobacco brown spot', 'soybean brown spot', 'broccoli ring spot',
                'carrot cavity spot'
            ],
            
            # All rust diseases
            'rust': [
                'rust', 'brown rust', 'yellow rust', 'stem rust', 'corn rust', 'corn_rust',
                'apple rust', 'bean rust', 'blueberry rust', 'garlic rust',
                'peach rust', 'plum rust', 'raspberry yellow rust', 'soybean rust',
                'sorghum rust', 'wheat leaf rust', 'wheat stem rust', 'wheat stripe rust',
                'coffee leaf rust'
            ],
            
            # All mildew diseases
            'mildew': [
                'mildew', 'powdery_mildew', 'powdery mildew', 'downy_mildew',
                'downy mildew', 'basil downy mildew', 'broccoli downy mildew',
                'cabbage downy mildew', 'cherry powdery mildew', 'cucumber powdery mildew',
                'grape downy mildew', 'lettuce downy mildew', 'bell pepper powdery mildew',
                'soybean downy mildew', 'squash powdery mildew', 'wheat powdery mildew',
                'zucchini downy mildew', 'zucchini powdery mildew', 'tobacco blue mold'
            ],
            
            # All virus and mosaic diseases
            'virus_disease': [
                'virus', 'mosaic', 'curl virus', 'leaf_curl', 'apple mosaic virus',
                'bean mosaic virus', 'lettuce mosaic virus', 'tobacco mosaic virus',
                'tobacco mosaic virus -tmv-', 'tomato mosaic virus', 'soybean mosaic',
                'tobacco rattle virus', 'plum pox virus', 'tomato yellow leaf curl virus',
                'tobacco leaf curl disease -tlcd-', 'zucchini yellow mosaic virus',
                'grapevine leafroll disease', 'tungro-virus', 'cassava mosaic disease',
                'peach leaf curl', 'citrus greening disease'
            ],
            
            # All anthracnose diseases
            'anthracnose': [
                'anthracnose', 'anthracnose_disease', 'anthrancose', 'banana anthracnose',
                'blueberry anthracnose', 'celery anthracnose', 'peach anthracnose',
                'strawberry anthracnose'
            ],
            
            # All rot diseases
            'rot_disease': [
                'rot', 'black_rot', 'brown_rot', 'apple black rot', 'cabbage black rot',
                'coffee black rot', 'grape black rot', 'peach brown rot', 'plum brown rot',
                'eggplant phomopsis fruit rot', 'banana cigar end rot',
                'cauliflower bacterial soft rot'
            ],
            
            # All scab diseases
            'scab': [
                'scab', 'apple scab', 'peach scab', 'wheat head scab'
            ],
            
            # All smut diseases
            'smut': [
                'smut', 'corn smut', 'corn_smut', 'sorghum loose smut', 'wheat loose smut'
            ],
            
            # Rice-specific diseases
            'rice_disease': [
                'rice blast', 'rice-blast', 'rice sheath blight', 'sheath-blight'
            ],
            
            # Banana-specific diseases
            'banana_disease': [
                'banana black leaf streak', 'banana bunchy top', 'banana cordana leaf spot', 
                'banana panama disease', 'sigatoka'
            ],
            
            # Healthy plants
            'healthy': [
                'healthy', 'healthy leaf'
            ],
            
            # Wilt diseases
            'wilt_disease': [
                'wilt', 'cucumber bacterial wilt', 'zucchini bacterial wilt'
            ],
            
            # Mold and gray mold
            'mold_disease': [
                'mold', 'gray_mold', 'raspberry gray mold', 'sorghum grain mold',
                'tomato leaf mold'
            ],
            
            # Fire blight
            'fire_blight': [
                'fire blight', 'raspberry fire blight'
            ],
            
            # Canker diseases
            'canker': [
                'canker', 'citrus canker'
            ],
            
            # Pest damage
            'pest_damage': [
                'spider mite', 'leaf-miner', 'cassava green mite', 'leaf-3ihq'
            ],
            
            # Bacterial diseases (general bacterial that don't fit other categories)
            'bacterial_disease': [
                'bacterialspot'
            ],
            
            # Septoria diseases
            'septoria': [
                'septoria', 'wheat septoria blotch'
            ],
            
            # Other specific diseases
            'blossom_end_rot': ['bell pepper blossom end rot'],
            'leaf_scorch': ['blueberry scorch', 'strawberry leaf scorch'],
            'mummy_berry': ['blueberry mummy berry'],
            'black_shank': ['black shank'],
            'cordana': ['cordana'],
            'dead': ['dead'],
            'ergot': ['sorghum ergot'],
            'ginger_sheath_blight': ['ginger sheath blight'],
            'mycosphaerella': ['mycosphaerella_leaf_blotch'],
            'pestalotiopsis': ['pestalotiopsis'],
            'pocket_disease': ['plum pocket disease'],
            'coffee_berry_blotch': ['coffee berry blotch'],
            'carrot_cercospora': ['carrot cercospora leaf blight'],
            'cassava_brown_streak': ['cassava brown streak disease'],
            'xanthomonas': ['xanthomonas'],
            'wheat_bacterial_streak': ['wheat bacterial leaf streak -black chaff-']
        }
        
        # Initialize consolidated classes
        self.consolidated_classes = []
        self.class_consolidation_map = {}
    
    def _load_original_classes(self) -> List[str]:
        """Load original classes from data.yaml"""
        try:
            with open(self.original_data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            classes = data.get('names', [])
            logger.info(f"Loaded {len(classes)} original classes")
            return classes
        except Exception as e:
            logger.error(f"Error loading data.yaml: {e}")
            return []
    
    def _load_class_mappings(self) -> Dict:
        """Load class mappings from class_mapping.yaml"""
        try:
            with open(self.class_mapping_file, 'r') as f:
                mappings = yaml.safe_load(f)
            logger.info("Loaded existing class mappings")
            return mappings
        except Exception as e:
            logger.warning(f"Error loading class mappings: {e}")
            return {}
    
    def consolidate_classes(self) -> Dict[str, List[str]]:
        """Consolidate disease classes using predefined rules"""
        logger.info("Starting disease class consolidation...")
        
        # Track which classes have been assigned
        assigned_classes = set()
        consolidation_groups = {}
        
        # Apply consolidation rules
        for consolidated_name, keywords in self.consolidation_rules.items():
            matching_classes = []
            
            for original_class in self.original_classes:
                if original_class in assigned_classes:
                    continue
                
                # Normalize class name for comparison
                original_lower = original_class.lower().replace('_', ' ').replace('-', ' ')
                
                for keyword in keywords:
                    keyword_lower = keyword.lower().replace('_', ' ').replace('-', ' ')
                    
                    # Multiple matching strategies
                    if (original_lower == keyword_lower or  # Exact match
                        keyword_lower in original_lower or   # Keyword in original
                        original_lower in keyword_lower or   # Original in keyword
                        self._fuzzy_match(original_lower, keyword_lower, threshold=0.8)):  # Fuzzy match
                        
                        matching_classes.append(original_class)
                        assigned_classes.add(original_class)
                        break
            
            if matching_classes:
                consolidation_groups[consolidated_name] = matching_classes
        
        # Handle unassigned classes
        unassigned = [cls for cls in self.original_classes if cls not in assigned_classes]
        logger.info(f"Found {len(unassigned)} unassigned classes: {unassigned[:10]}{'...' if len(unassigned) > 10 else ''}")
        
        # Group unassigned classes
        for unassigned_class in unassigned:
            # Create individual groups for unassigned classes
            clean_name = self._clean_class_name(unassigned_class)
            consolidation_groups[clean_name] = [unassigned_class]
        
        # Create final mapping
        self.consolidated_classes = list(consolidation_groups.keys())
        self.class_consolidation_map = {}
        
        for new_class_id, (consolidated_name, original_classes) in enumerate(consolidation_groups.items()):
            for original_class in original_classes:
                original_class_id = self.original_classes.index(original_class)
                self.class_consolidation_map[original_class_id] = new_class_id
        
        logger.info(f"Consolidated {len(self.original_classes)} classes to {len(self.consolidated_classes)} classes")
        logger.info(f"Reduction: {len(self.original_classes) - len(self.consolidated_classes)} classes")
        
        return consolidation_groups
    
    def _get_consolidation_groups(self) -> Dict[str, List[str]]:
        """Get consolidation groups for reporting"""
        groups = defaultdict(list)
        
        for old_id, new_id in self.class_consolidation_map.items():
            old_class = self.original_classes[old_id]
            new_class = self.consolidated_classes[new_id]
            groups[new_class].append(old_class)
        
        return dict(groups)
    
    def _fuzzy_match(self, str1: str, str2: str, threshold: float = 0.8) -> bool:
        """Check if two strings are similar enough"""
        return SequenceMatcher(None, str1, str2).ratio() >= threshold
    
    def _group_unassigned_classes(self, unassigned_classes: List[str]) -> Dict[str, List[str]]:
        """Group remaining unassigned classes"""
        groups = {}
        used = set()
        
        for class_name in unassigned_classes:
            if class_name in used:
                continue
            
            # Try to find plant-specific grouping
            plant_name = self._extract_plant_name(class_name)
            if plant_name:
                group_name = f"{plant_name}_disease"
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append(class_name)
                used.add(class_name)
            else:
                # Create individual group
                clean_name = self._clean_class_name(class_name)
                groups[clean_name] = [class_name]
                used.add(class_name)
        
        return groups
    
    def _extract_plant_name(self, class_name: str) -> str:
        """Extract plant name from class name"""
        plants = [
            'tomato', 'potato', 'corn', 'wheat', 'rice', 'soybean', 'bean',
            'apple', 'grape', 'banana', 'citrus', 'coffee', 'tobacco',
            'cassava', 'bell pepper', 'cucumber', 'carrot', 'cabbage',
            'broccoli', 'cauliflower', 'celery', 'eggplant', 'garlic',
            'ginger', 'lettuce', 'raspberry', 'strawberry', 'peach',
            'plum', 'cherry', 'blueberry', 'squash', 'zucchini', 'sorghum'
        ]
        
        class_lower = class_name.lower()
        for plant in plants:
            if plant in class_lower:
                return plant
        
        return ""
    
    def _clean_class_name(self, class_name: str) -> str:
        """Clean class name for use as consolidated name"""
        clean = class_name.lower()
        clean = re.sub(r'[^a-z0-9\s]', '_', clean)
        clean = re.sub(r'\s+', '_', clean)
        clean = re.sub(r'_+', '_', clean).strip('_')
        return clean
    
    def create_consolidated_dataset(self, output_dir: Path = None):
        """Create consolidated dataset with new class mappings and YOLO format"""
        if output_dir is None:
            output_dir = self.dataset_dir / "consolidated"
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create new data.yaml for YOLOv8
        new_data_yaml = {
            'train': str(output_dir / 'train' / 'images'),
            'val': str(output_dir / 'valid' / 'images'),
            'test': str(output_dir / 'test' / 'images'),
            'nc': len(self.consolidated_classes),
            'names': self.consolidated_classes
        }
        
        # Save new data.yaml
        with open(output_dir / "data.yaml", 'w') as f:
            yaml.dump(new_data_yaml, f, default_flow_style=False)
        
        logger.info(f"Created consolidated data.yaml with {len(self.consolidated_classes)} classes")
        
        # Process each split (train, valid, test)
        splits = ['train', 'valid', 'test']
        stats = {'total_images': 0, 'total_labels': 0, 'split_stats': {}}
        
        for split in splits:
            split_stats = self._process_split(split, output_dir)
            stats['split_stats'][split] = split_stats
            stats['total_images'] += split_stats['images_processed']
            stats['total_labels'] += split_stats['labels_processed']
        
        # Save consolidation mapping and statistics
        consolidation_info = {
            'original_classes': self.original_classes,
            'consolidated_classes': self.consolidated_classes,
            'class_mapping': self.class_consolidation_map,
            'consolidation_groups': self._get_consolidation_groups(),
            'statistics': stats,
            'created_at': datetime.now().isoformat()
        }
        
        with open(output_dir / "consolidation_mapping.json", 'w') as f:
            json.dump(consolidation_info, f, indent=2)
        
        logger.info("Dataset consolidation completed!")
        logger.info(f"Total images processed: {stats['total_images']}")
        logger.info(f"Total labels processed: {stats['total_labels']}")
        
        return output_dir
    
    def _process_split(self, split: str, output_dir: Path) -> Dict:
        """Process a dataset split (train/valid/test)"""
        input_images_dir = self.dataset_dir / split / "images"
        input_labels_dir = self.dataset_dir / split / "labels"
        
        output_images_dir = output_dir / split / "images"
        output_labels_dir = output_dir / split / "labels"
        
        output_images_dir.mkdir(exist_ok=True, parents=True)
        output_labels_dir.mkdir(exist_ok=True, parents=True)
        
        stats = {'images_processed': 0, 'labels_processed': 0, 'labels_converted': 0, 'labels_skipped': 0}
        
        if not input_images_dir.exists():
            logger.warning(f"Input directory not found: {input_images_dir}")
            return stats
        
        # Process each image and its corresponding label
        for image_file in input_images_dir.glob("*"):
            if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            # Copy image
            shutil.copy2(image_file, output_images_dir / image_file.name)
            stats['images_processed'] += 1
            
            # Process label file
            label_file = input_labels_dir / f"{image_file.stem}.txt"
            output_label_file = output_labels_dir / f"{image_file.stem}.txt"
            
            if label_file.exists():
                label_stats = self._process_label_file(label_file, output_label_file)
                stats['labels_processed'] += 1
                stats['labels_converted'] += label_stats['converted']
                stats['labels_skipped'] += label_stats['skipped']
            else:
                # Create empty label file
                output_label_file.touch()
        
        logger.info(f"Processed {split}: {stats['images_processed']} images, {stats['labels_processed']} labels")
        return stats
    
    def _process_label_file(self, input_label: Path, output_label: Path) -> Dict:
        """Process a single label file, converting classes and removing segmentation"""
        stats = {'converted': 0, 'skipped': 0}
        
        try:
            with open(input_label, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    stats['skipped'] += 1
                    continue  # Skip invalid lines
                
                old_class_id = int(parts[0])
                
                # Map to new class ID
                if old_class_id not in self.class_consolidation_map:
                    stats['skipped'] += 1
                    continue  # Skip unmapped classes
                
                new_class_id = self.class_consolidation_map[old_class_id]
                
                # Convert to bounding box format (remove segmentation if present)
                if len(parts) == 5:
                    # Already in bbox format: class x_center y_center width height
                    new_lines.append(f"{new_class_id} {' '.join(parts[1:5])}\n")
                    stats['converted'] += 1
                elif len(parts) > 5:
                    # Segmentation format: convert polygon to bbox
                    polygon_points = [float(x) for x in parts[1:]]
                    
                    if len(polygon_points) % 2 != 0:
                        stats['skipped'] += 1
                        continue  # Skip invalid polygons
                    
                    # Convert polygon to bounding box
                    x_coords = polygon_points[0::2]
                    y_coords = polygon_points[1::2]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Convert to YOLO format (center_x, center_y, width, height)
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Ensure values are within [0, 1] range
                    center_x = max(0, min(1, center_x))
                    center_y = max(0, min(1, center_y))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    new_lines.append(f"{new_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                    stats['converted'] += 1
            
            # Write new label file
            with open(output_label, 'w') as f:
                f.writelines(new_lines)
            
        except Exception as e:
            logger.error(f"Error processing {input_label}: {e}")
        
        return stats
    
    def generate_report(self, output_dir: Path):
        """Generate comprehensive consolidation report"""
        consolidation_groups = self._get_consolidation_groups()
        
        # Generate text report
        report_lines = [
            "=" * 80,
            "PLANT DISEASE DATASET CONSOLIDATION REPORT",
            "=" * 80,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY:",
            f"  Original classes: {len(self.original_classes)}",
            f"  Consolidated classes: {len(self.consolidated_classes)}",
            f"  Classes reduced by: {len(self.original_classes) - len(self.consolidated_classes)}",
            f"  Reduction percentage: {((len(self.original_classes) - len(self.consolidated_classes)) / len(self.original_classes) * 100):.1f}%",
            "",
            "MAJOR CONSOLIDATION GROUPS:",
            "-" * 50
        ]
        
        # Sort groups by number of original classes (largest first)
        sorted_groups = sorted(consolidation_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        for i, (new_class, original_classes) in enumerate(sorted_groups[:20]):  # Top 20 groups
            report_lines.append(f"\n{i+1:2d}. {new_class} ({len(original_classes)} classes)")
            if len(original_classes) <= 8:
                for orig_class in sorted(original_classes):
                    report_lines.append(f"    - {orig_class}")
            else:
                for orig_class in sorted(original_classes)[:5]:
                    report_lines.append(f"    - {orig_class}")
                report_lines.append(f"    ... and {len(original_classes) - 5} more classes")
        
        if len(sorted_groups) > 20:
            report_lines.append(f"\n... and {len(sorted_groups) - 20} more consolidated groups")
        
        # Add individual classes that weren't consolidated
        individual_classes = [name for name, classes in consolidation_groups.items() if len(classes) == 1]
        if individual_classes:
            report_lines.extend([
                f"\n\nINDIVIDUAL CLASSES (not consolidated): {len(individual_classes)}",
                "-" * 50
            ])
            for cls in sorted(individual_classes)[:10]:  # Show first 10
                report_lines.append(f"  - {cls}")
            if len(individual_classes) > 10:
                report_lines.append(f"  ... and {len(individual_classes) - 10} more")
        
        # Save text report
        report_file = output_dir / "consolidation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Print summary to console
        print("\n" + "="*80)
        print("DATASET CONSOLIDATION COMPLETED!")
        print("="*80)
        print(f"Original classes: {len(self.original_classes)}")
        print(f"Consolidated classes: {len(self.consolidated_classes)}")
        print(f"Reduction: {len(self.original_classes) - len(self.consolidated_classes)} classes ({((len(self.original_classes) - len(self.consolidated_classes)) / len(self.original_classes) * 100):.1f}% reduction)")
        print(f"Output directory: {output_dir}")
        print(f"Report saved to: {report_file}")
    
def main():
    """Main function to run the consolidation"""
    
    logger.info("Starting Disease Dataset Consolidation...")
    print(f"Starting consolidation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    consolidator = DiseaseConsolidator()
    
    if not consolidator.original_classes:
        logger.error("No classes found in data.yaml. Please check the file path.")
        print("❌ Error: No disease classes found!")
        print(f"Expected data.yaml at: {DATA_YAML_PATH}")
        print("Please ensure the data.yaml file exists and contains disease class names.")
        return
    
    print(f"✅ Loaded {len(consolidator.original_classes)} original disease classes")
    
    # Step 1: Consolidate classes
    print("\n📋 Step 1: Consolidating similar disease classes...")
    consolidation_groups = consolidator.consolidate_classes()
    
    # Step 2: Create consolidated dataset
    print("\n📁 Step 2: Creating consolidated dataset...")
    output_dir = consolidator.create_consolidated_dataset()
    
    # Step 3: Generate report
    print("\n📊 Step 3: Generating consolidation report...")
    consolidator.generate_report(output_dir)
    
    print(f"\n✅ Consolidation completed successfully!")
    print(f"📁 Output saved to: {output_dir}")
    
    # Show top consolidation groups
    print(f"\n🔝 Top consolidation groups:")
    sorted_groups = sorted(consolidation_groups.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (group_name, original_classes) in enumerate(sorted_groups[:15]):
        print(f"  {i+1:2d}. {group_name:25s} ({len(original_classes):2d} classes)")
    
    print(f"\n📈 Statistics:")
    print(f"  • Reduction from {len(consolidator.original_classes)} to {len(consolidator.consolidated_classes)} classes")
    print(f"  • {((len(consolidator.original_classes) - len(consolidator.consolidated_classes)) / len(consolidator.original_classes) * 100):.1f}% reduction in complexity")
    print(f"  • Ready for YOLOv8 training!")


if __name__ == "__main__":
    main()