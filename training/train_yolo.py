"""
YOLOv8 Training Script for Crop Health Monitoring
Trains YOLOv8 model on merged plant disease/pest dataset
"""

import os
import torch
import logging
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
import glob
torch.backends.cudnn.benchmark = True

from config import (
    CONSOLIDATED_DATASET_DIR, DATASET_DIR, MODELS_DIR, YOLO_MODEL_SIZE, 
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, DEVICE, TRAINING_CONFIG
)

# Enhanced GPU detection and configuration
def detect_device():
    """Detect and configure the best available device"""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU detected: {gpu_name}")
        print(f"🎮 Available GPUs: {gpu_count}")
        print(f"🎮 CUDA version: {torch.version.cuda}")
        
        # Set CUDA device
        torch.cuda.set_device(0)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("🍎 Apple Metal Performance Shaders (MPS) detected")
        return device
    else:
        device = 'cpu'
        print("⚠️  No GPU detected, using CPU")
        return device

# Override device configuration
DETECTED_DEVICE = detect_device()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def on_train_epoch_end(trainer):
    """Callback function for training epoch end events"""
    try:
        # Update progress bar if it exists
        if hasattr(trainer, 'epoch_pbar') and trainer.epoch_pbar is not None:
            current_epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            
            # Update progress bar
            trainer.epoch_pbar.set_postfix({
                'epoch': f"{current_epoch}/{total_epochs}",
                'loss': f"{trainer.loss:.4f}" if hasattr(trainer, 'loss') else "N/A"
            })
            trainer.epoch_pbar.update(1)
            
        # Log epoch completion
        logger.info(f"Completed epoch {trainer.epoch + 1}/{trainer.epochs}")
        
    except Exception as e:
        logger.warning(f"Epoch callback error: {e}")


class YOLOTrainer:
    """YOLOv8 model trainer optimized for laptop"""
    
    def __init__(self, data_yaml_path: str = None, model_size: str = None, resume_from: str = None):
        # Auto-detect consolidated dataset
        if data_yaml_path is None:
            if (CONSOLIDATED_DATASET_DIR / "data.yaml").exists():
                data_yaml_path = str(CONSOLIDATED_DATASET_DIR / "data.yaml")
                logger.info("Using consolidated dataset")
            else:
                data_yaml_path = str(DATASET_DIR / "data.yaml")
                logger.info("Using original dataset")
        
        self.data_yaml_path = Path(data_yaml_path)
        self.model_size = model_size or YOLO_MODEL_SIZE
        self.models_dir = Path(MODELS_DIR)
        self.models_dir.mkdir(exist_ok=True)
        
        # Use detected device
        self.device = DETECTED_DEVICE
        
        # Resume configuration
        self.resume_from = resume_from
        self.last_checkpoint = self._find_last_checkpoint() if resume_from else None
        
        # Training configuration optimized for laptop
        self.config = {
            'model_size': self.model_size,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'image_size': IMAGE_SIZE,
            'device': self.device,
            'data_yaml': str(self.data_yaml_path),
            'timestamp': datetime.now().isoformat(),
            'resume_from': resume_from,
            'environment': 'laptop',
            'dataset_type': 'consolidated' if 'consolidated' in str(self.data_yaml_path) else 'original'
        }
        
        # Validate data yaml exists
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"Data YAML not found: {self.data_yaml_path}")
        
        # Load and display dataset info
        self._display_dataset_info()

    def _find_last_checkpoint(self) -> Path:
        """Find the most recent checkpoint to resume from"""
        try:
            if self.resume_from:
                # Resume from specific path
                if self.resume_from == "auto":
                    # Auto-detect last experiment
                    experiment_dirs = [d for d in self.models_dir.iterdir() if d.is_dir() and d.name.startswith('crop_health_')]
                    if not experiment_dirs:
                        logger.warning("No previous experiments found for auto-resume")
                        return None
                    
                    # Get most recent experiment
                    latest_experiment = max(experiment_dirs, key=os.path.getctime)
                    logger.info(f"Auto-detected experiment: {latest_experiment}")
                    
                    # Look for last.pt checkpoint
                    checkpoint_path = latest_experiment / "weights" / "last.pt"
                    if checkpoint_path.exists():
                        logger.info(f"Found checkpoint: {checkpoint_path}")
                        return checkpoint_path
                    else:
                        logger.warning(f"No checkpoint found in {latest_experiment}")
                        return None
                
                else:
                    # Resume from specific file path
                    checkpoint_path = Path(self.resume_from)
                    if checkpoint_path.exists():
                        logger.info(f"Resuming from: {checkpoint_path}")
                        return checkpoint_path
                    else:
                        logger.error(f"Checkpoint not found: {checkpoint_path}")
                        return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding checkpoint: {e}")
            return None

    def setup_model(self) -> YOLO:
        """Initialize YOLOv8 model with optional resume capability"""
        try:
            with tqdm(desc="🔧 Setting up model", unit="step") as pbar:
                pbar.set_postfix({"status": "initializing"})
                
                if self.last_checkpoint:
                    logger.info(f"Resuming from checkpoint: {self.last_checkpoint}")
                    model = YOLO(str(self.last_checkpoint))
                    pbar.set_postfix({"status": "loaded checkpoint"})
                else:
                    logger.info(f"Initializing new YOLOv8 model: {self.model_size}")
                    model = YOLO(self.model_size)
                    pbar.set_postfix({"status": "new model"})
                
                pbar.update(1)
                
                # Log model information
                pbar.set_postfix({"status": "analyzing"})
                param_count = sum(p.numel() for p in model.model.parameters())
                logger.info(f"Model parameters: {param_count:,}")
                logger.info(f"Training device: {self.device}")
                
                # GPU memory info if using CUDA
                if self.device == 'cuda':
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"GPU memory: {gpu_allocated:.2f}GB / {gpu_memory:.2f}GB")
                
                pbar.update(1)
                
                pbar.set_postfix({"status": "complete"})
                return model
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise

    def _display_dataset_info(self):
        """Display information about the dataset being used"""
        try:
            with open(self.data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            num_classes = data_config.get('nc', 0)
            class_names = data_config.get('names', [])
            
            print(f"📊 Dataset Information:")
            print(f"   - Classes: {num_classes}")
            print(f"   - Dataset type: {self.config['dataset_type']}")
            print(f"   - Model: {self.model_size}")
            print(f"   - Sample classes: {class_names[:10]}{'...' if len(class_names) > 10 else ''}")
            
            # Show consolidation info if available
            if self.config['dataset_type'] == 'consolidated':
                mapping_file = self.data_yaml_path.parent / "consolidation_mapping.json"
                if mapping_file.exists():
                    with open(mapping_file, 'r') as f:
                        mapping_data = json.load(f)
                    original_classes = len(mapping_data.get('original_classes', []))
                    print(f"   - Original classes: {original_classes} → {num_classes} (consolidated)")
                    
        except Exception as e:
            logger.warning(f"Could not display dataset info: {e}")

    def train(self, 
              resume: bool = False,
              patience: int = None,
              save_period: int = None,
              val_split: float = 0.2,
              augment: bool = True) -> Path:
        """Train YOLOv8 model optimized for laptop"""
        
        # Use config defaults if not specified
        patience = patience or TRAINING_CONFIG.get('patience', 25)
        save_period = save_period or TRAINING_CONFIG.get('save_period', 5)
        
        # Get training parameters from config
        training_epochs = self.config.get('epochs', EPOCHS)
        training_batch_size = self.config.get('batch_size', BATCH_SIZE)
        
        try:
            model = self.setup_model()
            
            # Determine if we're resuming
            is_resuming = self.last_checkpoint is not None
            
            # Create experiment directory
            with tqdm(desc="📁 Setting up experiment", unit="step") as pbar:
                if is_resuming:
                    experiment_name = self.last_checkpoint.parent.parent.name
                    experiment_dir = self.last_checkpoint.parent.parent
                    logger.info(f"Resuming experiment: {experiment_name}")
                else:
                    experiment_name = f"crop_health_{self.model_size.replace('.pt', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    experiment_dir = self.models_dir / experiment_name
                    experiment_dir.mkdir(exist_ok=True)
                    logger.info(f"Starting new experiment: {experiment_name}")
                
                pbar.update(1)
                
                # Save training configuration
                config_file = experiment_dir / "training_config.json"
                with open(config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
                pbar.update(1)
            
            # Training arguments optimized for laptop
            train_args = {
                'data': str(self.data_yaml_path),
                'epochs': training_epochs,
                'batch': training_batch_size,
                'imgsz': IMAGE_SIZE,
                'device': self.device,
                'project': str(self.models_dir),
                'name': experiment_name,
                'exist_ok': True,
                'patience': patience,
                'save_period': save_period,
                'val': True,
                'plots': True,
                'verbose': True,
                'amp': TRAINING_CONFIG.get('amp', True),
                'cache': TRAINING_CONFIG.get('cache', False),
                'rect': TRAINING_CONFIG.get('rect', True),
                'resume': is_resuming,
                'optimizer': TRAINING_CONFIG.get('optimizer', 'SGD'),
                'lr0': TRAINING_CONFIG.get('lr0', 0.01),
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': TRAINING_CONFIG.get('warmup_epochs', 2),
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'workers': 4,  # Laptop-optimized worker count
                # Laptop-optimized hyperparameters
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
            }
            
            print(f"\n🚀 Starting LAPTOP training with {self.model_size}...")
            print(f"📊 Dataset: {self.config['dataset_type']} ({self.data_yaml_path})")
            print(f"🔧 Epochs: {training_epochs}, Batch: {training_batch_size}, Device: {self.device.upper()}")
            print(f"💾 Memory optimized: cache={train_args['cache']}, rect={train_args['rect']}")
            
            # Add callback
            model.add_callback('on_train_epoch_end', on_train_epoch_end)
            
            # Initialize progress tracking
            self.epoch_pbar = tqdm(total=training_epochs, desc="🏃 Training Progress", unit="epoch")
            
            logger.info("Beginning laptop-optimized model training...")
            results = model.train(**train_args)
            
            # Close progress bar
            self.epoch_pbar.close()
            
            # Process results
            with tqdm(desc="💾 Saving results", unit="step") as pbar:
                best_model_path = Path(results.save_dir) / "weights" / "best.pt"
                pbar.update(1)
                
                # Copy best model with descriptive name
                final_model_path = self.models_dir / f"{experiment_name}_best.pt"
                if best_model_path.exists():
                    import shutil
                    shutil.copy2(best_model_path, final_model_path)
                    logger.info(f"Best model saved to: {final_model_path}")
                pbar.update(1)
                
                # Save training summary
                self._save_training_summary(experiment_dir, results)
                pbar.update(1)
            
            logger.info("Laptop training completed successfully!")
            return final_model_path
            
        except Exception as e:
            if hasattr(self, 'epoch_pbar'):
                self.epoch_pbar.close()
            logger.error(f"Training failed: {e}")
            raise
    
    def validate_model(self, model_path: Path) -> dict:
        """Validate trained model with progress tracking"""
        try:
            with tqdm(desc="🔍 Validating model", unit="step") as pbar:
                logger.info(f"Validating model: {model_path}")
                pbar.set_postfix({"status": "loading model"})
                
                model = YOLO(str(model_path))
                pbar.update(1)
                
                pbar.set_postfix({"status": "running validation"})
                results = model.val(
                    data=str(self.data_yaml_path),
                    device=self.device,  # Use detected device
                    plots=True,
                    save_txt=True,
                    save_conf=True
                )
                pbar.update(1)
                
                pbar.set_postfix({"status": "extracting metrics"})
                # Extract key metrics
                metrics = {
                    'mAP50': float(results.box.map50),
                    'mAP50-95': float(results.box.map),
                    'precision': float(results.box.mp),
                    'recall': float(results.box.mr),
                    'f1': float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr)),
                }
                pbar.update(1)
                
                logger.info("Validation metrics:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")
                
                pbar.set_postfix({"status": "complete"})
                return metrics
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {}
    
    def comprehensive_evaluation(self, model_path: Path) -> dict:
        """Comprehensive model evaluation with detailed metrics and visualizations"""
        try:
            eval_results = {}
            
            with tqdm(desc="📊 Comprehensive Evaluation", unit="step", total=6) as pbar:
                # Load model
                pbar.set_postfix({"status": "loading model"})
                model = YOLO(str(model_path))
                pbar.update(1)
                
                # Basic validation metrics
                pbar.set_postfix({"status": "basic validation"})
                basic_metrics = self.validate_model(model_path)
                eval_results['basic_metrics'] = basic_metrics
                pbar.update(1)
                
                # Per-class metrics
                pbar.set_postfix({"status": "per-class analysis"})
                per_class_metrics = self._get_per_class_metrics(model)
                eval_results['per_class_metrics'] = per_class_metrics
                pbar.update(1)
                
                # Confusion matrix
                pbar.set_postfix({"status": "confusion matrix"})
                confusion_matrix = self._generate_confusion_matrix(model)
                eval_results['confusion_matrix'] = confusion_matrix
                pbar.update(1)
                
                # Speed benchmarks
                pbar.set_postfix({"status": "speed benchmark"})
                speed_metrics = self._benchmark_speed(model)
                eval_results['speed_metrics'] = speed_metrics
                pbar.update(1)
                
                # Model size analysis
                pbar.set_postfix({"status": "model analysis"})
                model_analysis = self._analyze_model_size(model_path)
                eval_results['model_analysis'] = model_analysis
                pbar.update(1)
                
                pbar.set_postfix({"status": "complete"})
            
            # Save comprehensive evaluation report
            self._save_evaluation_report(model_path.parent, eval_results)
            
            return eval_results
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            return {}
    
    def _get_per_class_metrics(self, model) -> dict:
        """Get detailed per-class metrics"""
        try:
            results = model.val(data=str(self.data_yaml_path), device=self.device, verbose=False)
            
            # Load class names from data.yaml
            with open(self.data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            class_names = data_config.get('names', {})
            
            per_class = {}
            if hasattr(results, 'box') and hasattr(results.box, 'ap_class_index'):
                for i, class_idx in enumerate(results.box.ap_class_index):
                    class_name = class_names.get(int(class_idx), f"class_{class_idx}")
                    per_class[class_name] = {
                        'AP50': float(results.box.ap50[i]) if i < len(results.box.ap50) else 0,
                        'AP50-95': float(results.box.ap[i]) if i < len(results.box.ap) else 0,
                        'precision': float(results.box.p[i]) if i < len(results.box.p) else 0,
                        'recall': float(results.box.r[i]) if i < len(results.box.r) else 0,
                    }
            
            return per_class
            
        except Exception as e:
            logger.warning(f"Failed to get per-class metrics: {e}")
            return {}
    
    def _generate_confusion_matrix(self, model) -> dict:
        """Generate and save confusion matrix"""
        try:
            results = model.val(data=str(self.data_yaml_path), device=self.device, verbose=False)
            
            if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
                cm = results.confusion_matrix.matrix
                return {
                    'matrix': cm.tolist() if hasattr(cm, 'tolist') else str(cm),
                    'shape': cm.shape if hasattr(cm, 'shape') else None
                }
            
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to generate confusion matrix: {e}")
            return {}
    
    def _benchmark_speed(self, model) -> dict:
        """Benchmark model inference speed"""
        try:
            # Create dummy input on the correct device
            dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(self.device)
            
            # Warmup
            for _ in range(10):
                model.predict(dummy_input, verbose=False)
            
            # Benchmark
            times = []
            for _ in range(100):
                start_time = time.time()
                model.predict(dummy_input, verbose=False)
                times.append(time.time() - start_time)
            
            return {
                'avg_inference_time': np.mean(times),
                'std_inference_time': np.std(times),
                'fps': 1 / np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
            
        except Exception as e:
            logger.warning(f"Speed benchmark failed: {e}")
            return {}
    
    def _analyze_model_size(self, model_path: Path) -> dict:
        """Analyze model size and parameters"""
        try:
            model = YOLO(str(model_path))
            
            # File size
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # Parameter count
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            
            return {
                'file_size_mb': file_size_mb,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size': self.model_size
            }
            
        except Exception as e:
            logger.warning(f"Model analysis failed: {e}")
            return {}
    
    def _save_evaluation_report(self, output_dir: Path, eval_results: dict) -> None:
        """Save comprehensive evaluation report"""
        try:
            report_file = output_dir / "comprehensive_evaluation.json"
            with open(report_file, 'w') as f:
                json.dump(eval_results, f, indent=2, default=str)
            
            logger.info(f"Comprehensive evaluation report saved: {report_file}")
            
            # Print summary
            print("\n📊 COMPREHENSIVE EVALUATION SUMMARY")
            print("=" * 50)
            
            if 'basic_metrics' in eval_results:
                metrics = eval_results['basic_metrics']
                print(f"🎯 mAP50: {metrics.get('mAP50', 0):.4f}")
                print(f"🎯 mAP50-95: {metrics.get('mAP50-95', 0):.4f}")
                print(f"🎯 Precision: {metrics.get('precision', 0):.4f}")
                print(f"🎯 Recall: {metrics.get('recall', 0):.4f}")
                print(f"🎯 F1-Score: {metrics.get('f1', 0):.4f}")
            
            if 'speed_metrics' in eval_results:
                speed = eval_results['speed_metrics']
                print(f"⚡ Average FPS: {speed.get('fps', 0):.2f}")
                print(f"⚡ Inference Time: {speed.get('avg_inference_time', 0)*1000:.2f}ms")
            
            if 'model_analysis' in eval_results:
                analysis = eval_results['model_analysis']
                print(f"📦 Model Size: {analysis.get('file_size_mb', 0):.2f} MB")
                print(f"🔢 Parameters: {analysis.get('total_parameters', 0):,}")
            
            print("=" * 50)
            
        except Exception as e:
            logger.warning(f"Failed to save evaluation report: {e}")
    
    def export_model(self, model_path: Path, formats: list = None) -> dict:
        """Export model to different formats with progress tracking"""
        if formats is None:
            formats = ['onnx', 'torchscript']
        
        try:
            exported_paths = {}
            
            with tqdm(desc="📤 Exporting model", total=len(formats), unit="format") as pbar:
                logger.info(f"Exporting model to formats: {formats}")
                model = YOLO(str(model_path))
                
                for format_type in formats:
                    try:
                        pbar.set_postfix({"current": format_type})
                        export_path = model.export(format=format_type, device=self.device)
                        exported_paths[format_type] = export_path
                        logger.info(f"Exported {format_type}: {export_path}")
                        pbar.update(1)
                    except Exception as e:
                        logger.warning(f"Failed to export {format_type}: {e}")
                        pbar.update(1)
            
            return exported_paths
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {}
    
    def _save_training_summary(self, experiment_dir: Path, results) -> None:
        """Save training summary and metrics"""
        try:
            summary = {
                'experiment_config': self.config,
                'final_metrics': {
                    'mAP50': float(results.box.map50) if hasattr(results, 'box') else 0,
                    'mAP50-95': float(results.box.map) if hasattr(results, 'box') else 0,
                    'precision': float(results.box.mp) if hasattr(results, 'box') else 0,
                    'recall': float(results.box.mr) if hasattr(results, 'box') else 0,
                },
                'training_completed': datetime.now().isoformat(),
                'model_path': str(results.save_dir) if hasattr(results, 'save_dir') else None
            }
            
            summary_file = experiment_dir / "training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Training summary saved: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save training summary: {e}")
    
    def load_best_model(self) -> YOLO:
        """Load the best available trained model"""
        try:
            # Look for best model files
            model_files = list(self.models_dir.glob("*_best.pt"))
            
            if not model_files:
                logger.warning("No trained models found, using pretrained model")
                return YOLO(self.model_size)
            
            # Get the most recent model
            latest_model = max(model_files, key=os.path.getctime)
            logger.info(f"Loading best model: {latest_model}")
            
            return YOLO(str(latest_model))
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return YOLO(self.model_size)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOv8 Training for Crop Health Monitoring')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Resume training from checkpoint. Use "auto" to auto-detect last experiment or specify path to checkpoint file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--model-size', type=str, default=None,
                       help='Model size: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt')
    return parser.parse_args()


def main():
    """Main training function optimized for laptop"""
    try:
        args = parse_arguments()
        
        print("🌱 CROP HEALTH MONITORING - LAPTOP TRAINING PIPELINE")
        print("=" * 60)
        
        # Get current configuration values before override
        current_epochs = EPOCHS
        current_batch_size = BATCH_SIZE
        current_model_size = YOLO_MODEL_SIZE
        
        print(f"🔧 Optimized for: Laptop training with {current_model_size}")
        
        # Override config with command line arguments
        if args.epochs:
            current_epochs = args.epochs
            print(f"🔧 Epochs overridden: {current_epochs}")
        if args.batch_size:
            current_batch_size = args.batch_size
            print(f"🔧 Batch size overridden: {current_batch_size}")
        if args.model_size:
            current_model_size = args.model_size
            print(f"🔧 Model size overridden: {current_model_size}")
        
        print(f"🎮 Training Device: {DETECTED_DEVICE.upper()}")
        if DETECTED_DEVICE == 'cuda':
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        if args.resume:
            print(f"🔄 Resume mode: {args.resume}")
        
        print("=" * 60)
        
        # Check for consolidated dataset
        with tqdm(desc="🔍 Checking dataset", unit="step") as pbar:
            consolidated_yaml = CONSOLIDATED_DATASET_DIR / "data.yaml"
            original_yaml = DATASET_DIR / "data.yaml"
            
            if consolidated_yaml.exists():
                data_yaml = consolidated_yaml
                print("✅ Using consolidated dataset")
            elif original_yaml.exists():
                data_yaml = original_yaml
                print("⚠️ Using original dataset (run consolidation for better results)")
            else:
                logger.error("No dataset found. Please run dataset consolidation first.")
                return 1
            pbar.update(1)
        
        # Initialize trainer with current values
        trainer = YOLOTrainer(str(data_yaml), current_model_size, resume_from=args.resume)
        
        # Update trainer config with override values
        trainer.config['epochs'] = current_epochs
        trainer.config['batch_size'] = current_batch_size
        trainer.config['model_size'] = current_model_size
        
        # Start training
        logger.info("Starting laptop-optimized YOLOv8 training...")
        best_model_path = trainer.train()
        
        # Validate the trained model
        logger.info("Validating trained model...")
        metrics = trainer.validate_model(best_model_path)
        
        # Quick evaluation (laptop-optimized)
        logger.info("Running evaluation...")
        eval_results = trainer.comprehensive_evaluation(best_model_path)
        
        # Export model
        logger.info("Exporting model...")
        exported = trainer.export_model(best_model_path, formats=['onnx'])
        
        # Print final summary
        print("\n🎉 LAPTOP TRAINING COMPLETED!")
        print("=" * 60)
        print(f"📁 Best model: {best_model_path}")
        print(f"📊 mAP50: {metrics.get('mAP50', 'N/A'):.4f}")
        print(f"📊 mAP50-95: {metrics.get('mAP50-95', 'N/A'):.4f}")
        print(f"🔄 Exported formats: {list(exported.keys())}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
