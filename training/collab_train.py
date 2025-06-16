"""
Google Colab Training Script for Smart Crop Health Monitoring System
Optimized for A100 GPU with YOLOv8m
"""

# Install required packages
import subprocess
import sys

def install_packages():
    """Install required packages for training"""
    packages = [
        'ultralytics',
        'roboflow',
        'wandb',
        'tensorboard',
        'tqdm',
        'psutil',
        'matplotlib',
        'seaborn',
        'plotly',
        'ipywidgets',
        'rarfile'  # Added for RAR file support
    ]
    
    print("📦 Installing required packages...")
    for package in packages:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
    
    # Install unrar tool for rarfile
    print("🔧 Installing unrar tool...")
    try:
        subprocess.check_call(['apt-get', 'update', '-qq'])
        subprocess.check_call(['apt-get', 'install', '-y', '-qq', 'unrar'])
        print("✅ unrar tool installed successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not install unrar tool: {e}")
        print("💡 RAR extraction might not work properly")
    
    print("✅ All packages installed successfully!")

# Install packages
install_packages()

# Import libraries
import os
import zipfile
import shutil
import time
import threading
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import torch
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from google.colab import drive, files
from tqdm.notebook import tqdm
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import rarfile

# Configuration (optimized for Colab)
class Config:
    # Model configuration optimized for A100
    YOLO_MODEL_SIZE = "yolov8m.pt"  # Medium model for Colab
    IMAGE_SIZE = 640
    BATCH_SIZE = 32  # Good for A100 GPU with yolov8m
    EPOCHS = 125
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths for Colab environment
    PROJECT_ROOT = Path("/content/pest_model")
    DATASET_DIR = PROJECT_ROOT / "dataset"
    MODELS_DIR = PROJECT_ROOT / "models" 
    TRAINING_DIR = PROJECT_ROOT / "training"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Training optimization for Colab
    PATIENCE = 50
    SAVE_PERIOD = 10
    WORKERS = 8
    AMP = True
    
    # Progress tracking
    UPDATE_INTERVAL = 5
    PLOT_UPDATE_INTERVAL = 30

# Create directories
config = Config()
for dir_path in [config.DATASET_DIR, config.MODELS_DIR, config.TRAINING_DIR, config.LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class ProgressTracker:
    """Comprehensive progress tracking for training"""
    
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_times = deque(maxlen=10)  # Keep last 10 epoch times
        self.metrics_history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'map50': [],
            'map50_95': [],
            'precision': [],
            'recall': [],
            'lr': [],
            'gpu_memory': [],
            'cpu_usage': [],
            'epoch_time': []
        }
        
        # Progress bars
        self.main_pbar = None
        self.epoch_pbar = None
        self.batch_pbar = None
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        
        # Live plots
        self.live_plots = True
        self.plot_fig = None
        
        # Widgets for live display
        self.setup_widgets()
    
    def setup_widgets(self):
        """Setup interactive widgets for progress display"""
        self.progress_output = widgets.Output()
        self.metrics_output = widgets.Output()
        self.plots_output = widgets.Output()
        
        # Create tabs
        self.tab = widgets.Tab()
        self.tab.children = [self.progress_output, self.metrics_output, self.plots_output]
        self.tab.set_title(0, 'Progress')
        self.tab.set_title(1, 'Metrics')
        self.tab.set_title(2, 'Live Plots')
    
    def start_training(self):
        """Initialize training progress tracking"""
        self.start_time = time.time()
        
        # Initialize main progress bar
        self.main_pbar = tqdm(
            total=self.total_epochs,
            desc="🚀 Training Progress",
            unit="epoch",
            colour="green",
            position=0,
            leave=True
        )
        
        # Start monitoring
        self.start_monitoring()
        
        # Display widgets
        display(self.tab)
        
        print("🌟 Training started with comprehensive progress tracking!")
        print(f"📊 Total epochs: {self.total_epochs}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def start_monitoring(self):
        """Start system monitoring thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_system(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # GPU memory if available
                gpu_memory = 0
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                
                # Update current metrics (for display)
                self.current_cpu = cpu_percent
                self.current_gpu_memory = gpu_memory
                
                time.sleep(config.UPDATE_INTERVAL)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)
    
    def update_epoch(self, epoch, metrics):
        """Update progress for completed epoch"""
        epoch_time = time.time() - self.epoch_start if hasattr(self, 'epoch_start') else 0
        self.epoch_times.append(epoch_time)
        
        # Store metrics
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['epoch_time'].append(epoch_time)
        if hasattr(self, 'current_cpu'):
            self.metrics_history['cpu_usage'].append(self.current_cpu)
        if hasattr(self, 'current_gpu_memory'):
            self.metrics_history['gpu_memory'].append(self.current_gpu_memory)
        
        # Calculate ETA
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else epoch_time
        remaining_epochs = self.total_epochs - epoch
        eta_seconds = remaining_epochs * avg_epoch_time
        eta = str(timedelta(seconds=int(eta_seconds)))
        
        # Update main progress bar
        self.main_pbar.set_postfix({
            'Loss': f"{metrics.get('loss', 0):.4f}",
            'mAP50': f"{metrics.get('map50', 0):.3f}",
            'ETA': eta,
            'GPU': f"{self.current_gpu_memory:.1f}%" if hasattr(self, 'current_gpu_memory') else 'N/A'
        })
        self.main_pbar.update(1)
        
        # Update widgets
        self._update_progress_widget(epoch, eta, metrics)
        self._update_metrics_widget()
        
        # Update plots every few epochs
        if epoch % (config.PLOT_UPDATE_INTERVAL // avg_epoch_time) == 0 or epoch == self.total_epochs:
            self._update_plots_widget()
    
    def start_epoch(self, epoch):
        """Mark start of new epoch"""
        self.epoch_start = time.time()
        self.current_epoch = epoch
        
        with self.progress_output:
            clear_output(wait=True)
            print(f"🔄 Epoch {epoch}/{self.total_epochs} started")
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    
    def _update_progress_widget(self, epoch, eta, metrics):
        """Update progress widget with current status"""
        with self.progress_output:
            clear_output(wait=True)
            
            # Training progress
            progress_percent = (epoch / self.total_epochs) * 100
            print(f"🚀 Training Progress: {progress_percent:.1f}% ({epoch}/{self.total_epochs})")
            print(f"⏱️  ETA: {eta}")
            print(f"📊 Current Loss: {metrics.get('loss', 0):.4f}")
            print(f"🎯 Current mAP50: {metrics.get('map50', 0):.3f}")
            
            # System info
            if hasattr(self, 'current_cpu'):
                print(f"💻 CPU Usage: {self.current_cpu:.1f}%")
            if hasattr(self, 'current_gpu_memory'):
                print(f"🎮 GPU Memory: {self.current_gpu_memory:.1f}%")
            
            # Best metrics so far
            if self.metrics_history['map50']:
                best_map50 = max(self.metrics_history['map50'])
                best_epoch = self.metrics_history['epoch'][np.argmax(self.metrics_history['map50'])]
                print(f"🏆 Best mAP50: {best_map50:.3f} (Epoch {best_epoch})")
    
    def _update_metrics_widget(self):
        """Update metrics widget with detailed statistics"""
        with self.metrics_output:
            clear_output(wait=True)
            
            if not self.metrics_history['epoch']:
                print("📊 No metrics available yet...")
                return
            
            print("📈 TRAINING METRICS SUMMARY")
            print("=" * 40)
            
            # Current metrics
            latest_idx = -1
            current_metrics = {
                'Loss': self.metrics_history['loss'][latest_idx] if self.metrics_history['loss'] else 0,
                'Val Loss': self.metrics_history['val_loss'][latest_idx] if self.metrics_history['val_loss'] else 0,
                'mAP50': self.metrics_history['map50'][latest_idx] if self.metrics_history['map50'] else 0,
                'mAP50-95': self.metrics_history['map50_95'][latest_idx] if self.metrics_history['map50_95'] else 0,
                'Precision': self.metrics_history['precision'][latest_idx] if self.metrics_history['precision'] else 0,
                'Recall': self.metrics_history['recall'][latest_idx] if self.metrics_history['recall'] else 0,
            }
            
            print("📊 Current Metrics:")
            for metric, value in current_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Best metrics
            print("\n🏆 Best Metrics:")
            if self.metrics_history['map50']:
                best_map50 = max(self.metrics_history['map50'])
                print(f"  Best mAP50: {best_map50:.4f}")
            if self.metrics_history['loss']:
                best_loss = min(self.metrics_history['loss'])
                print(f"  Best Loss: {best_loss:.4f}")
            
            # Training speed
            if self.epoch_times:
                avg_time = np.mean(self.epoch_times)
                print(f"\n⚡ Average Epoch Time: {avg_time:.2f}s")
                print(f"📈 Training Speed: {len(self.metrics_history['epoch']) / (time.time() - self.start_time):.2f} epochs/hour")
    
    def _update_plots_widget(self):
        """Update live plots widget"""
        with self.plots_output:
            clear_output(wait=True)
            
            if len(self.metrics_history['epoch']) < 2:
                print("📊 Generating plots... (need more data)")
                return
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Live Training Metrics', fontsize=16, fontweight='bold')
            
            epochs = self.metrics_history['epoch']
            
            # Loss plot
            if self.metrics_history['loss']:
                axes[0, 0].plot(epochs, self.metrics_history['loss'], 'b-', label='Training Loss', linewidth=2)
                if self.metrics_history['val_loss']:
                    axes[0, 0].plot(epochs, self.metrics_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
                axes[0, 0].set_title('Training Loss', fontweight='bold')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # mAP plot
            if self.metrics_history['map50']:
                axes[0, 1].plot(epochs, self.metrics_history['map50'], 'g-', label='mAP50', linewidth=2)
                if self.metrics_history['map50_95']:
                    axes[0, 1].plot(epochs, self.metrics_history['map50_95'], 'orange', label='mAP50-95', linewidth=2)
                axes[0, 1].set_title('Mean Average Precision', fontweight='bold')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('mAP')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Precision/Recall plot
            if self.metrics_history['precision'] and self.metrics_history['recall']:
                axes[1, 0].plot(epochs, self.metrics_history['precision'], 'purple', label='Precision', linewidth=2)
                axes[1, 0].plot(epochs, self.metrics_history['recall'], 'brown', label='Recall', linewidth=2)
                axes[1, 0].set_title('Precision & Recall', fontweight='bold')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # System resources plot
            if self.metrics_history['gpu_memory']:
                ax2 = axes[1, 1]
                ax2.plot(epochs, self.metrics_history['gpu_memory'], 'red', label='GPU Memory %', linewidth=2)
                ax2.set_ylabel('GPU Memory %', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                if self.metrics_history['cpu_usage']:
                    ax3 = ax2.twinx()
                    ax3.plot(epochs, self.metrics_history['cpu_usage'], 'blue', label='CPU Usage %', linewidth=2)
                    ax3.set_ylabel('CPU Usage %', color='blue')
                    ax3.tick_params(axis='y', labelcolor='blue')
                
                axes[1, 1].set_title('System Resources', fontweight='bold')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def save_training_log(self, output_dir):
        """Save comprehensive training log"""
        log_file = output_dir / "training_progress_log.json"
        
        training_log = {
            'config': {
                'total_epochs': self.total_epochs,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': time.time() - self.start_time
            },
            'metrics_history': self.metrics_history,
            'summary': {
                'best_map50': max(self.metrics_history['map50']) if self.metrics_history['map50'] else 0,
                'best_loss': min(self.metrics_history['loss']) if self.metrics_history['loss'] else 0,
                'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
                'total_epochs_completed': len(self.metrics_history['epoch'])
            }
        }
        
        with open(log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        print(f"📊 Training log saved: {log_file}")
    
    def finish_training(self):
        """Clean up progress tracking"""
        self.monitoring = False
        if self.main_pbar:
            self.main_pbar.close()
        if self.epoch_pbar:
            self.epoch_pbar.close()
        if self.batch_pbar:
            self.batch_pbar.close()
        
        # Final summary
        total_time = time.time() - self.start_time
        print(f"\n🎉 Training completed in {str(timedelta(seconds=int(total_time)))}")

def check_gpu():
    """Check GPU availability and type with progress tracking"""
    with tqdm(desc="🔍 Checking GPU", total=3, colour="blue") as pbar:
        pbar.set_postfix({"status": "detecting"})
        
        if torch.cuda.is_available():
            pbar.update(1)
            pbar.set_postfix({"status": "found GPU"})
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Available: {gpu_name}")
            print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
            pbar.update(1)
            
            # Optimize batch size based on GPU memory
            if "A100" in gpu_name:
                config.BATCH_SIZE = 32  # Good for A100 with yolov8m
                print("🚀 A100 detected - using optimized settings for YOLOv8m")
            elif gpu_memory > 15:
                config.BATCH_SIZE = 24  # High-end GPU
            elif gpu_memory > 10:
                config.BATCH_SIZE = 16  # Mid-range GPU
            else:
                config.BATCH_SIZE = 8   # Lower-end GPU
            
            pbar.update(1)
            pbar.set_postfix({"status": "optimized"})
            return True
        else:
            pbar.update(3)
            print("❌ No GPU available")
            config.DEVICE = "cpu"
            config.BATCH_SIZE = 4
            return False

def check_existing_dataset():
    """Check if dataset already exists locally"""
    print("🔍 CHECKING FOR EXISTING DATASET")
    print("=" * 40)
    
    # Look for data.yaml file
    yaml_files = list(config.PROJECT_ROOT.rglob("data.yaml"))
    
    if yaml_files:
        # Use the first data.yaml found
        yaml_file = yaml_files[0]
        dataset_path = yaml_file.parent
        print(f"✅ Found existing dataset config: {yaml_file}")
        print(f"📁 Dataset root: {dataset_path}")
        
        # Verify dataset structure
        expected_dirs = ['train', 'valid', 'test']
        found_dirs = [item.name for item in dataset_path.iterdir() if item.is_dir()]
        
        valid_structure = True
        for expected_dir in expected_dirs[:2]:  # Check at least train and valid
            if expected_dir in found_dirs:
                subdir = dataset_path / expected_dir
                images_dir = subdir / 'images'
                labels_dir = subdir / 'labels'
                if images_dir.exists() and labels_dir.exists():
                    img_count = len(list(images_dir.glob('*')))
                    label_count = len(list(labels_dir.glob('*.txt')))
                    print(f"📊 {expected_dir}: {img_count} images, {label_count} labels")
                    if img_count == 0:
                        valid_structure = False
                else:
                    valid_structure = False
                    break
        
        if valid_structure:
            print("✅ Dataset structure verified - using existing dataset")
            return dataset_path
        else:
            print("⚠️ Dataset structure incomplete - will re-extract")
    
    # Fallback: look for standard dataset structure
    extracted_dirs = [d for d in config.PROJECT_ROOT.iterdir() if d.is_dir() and d.name not in ['logs', 'models', 'training']]
    
    for dir_path in extracted_dirs:
        try:
            subdirs = [d.name for d in dir_path.iterdir() if d.is_dir()]
            
            if any(subdir in subdirs for subdir in ['train', 'valid', 'test']):
                # Verify it has actual data
                train_dir = dir_path / 'train' / 'images'
                valid_dir = dir_path / 'valid' / 'images'
                
                if train_dir.exists() and valid_dir.exists():
                    train_count = len(list(train_dir.glob('*')))
                    valid_count = len(list(valid_dir.glob('*')))
                    
                    if train_count > 0 and valid_count > 0:
                        print(f"✅ Found valid dataset structure in: {dir_path}")
                        print(f"📊 Training images: {train_count}, Validation images: {valid_count}")
                        return dir_path
        except Exception as e:
            print(f"⚠️ Could not check {dir_path}: {e}")
            continue
    
    print("❌ No existing dataset found - will extract from Google Drive")
    return None

def mount_drive_and_extract_dataset():
    """Mount Google Drive and extract dataset with proper authentication"""
    # First check if dataset already exists
    existing_dataset = check_existing_dataset()
    if existing_dataset:
        print("🎉 Using existing dataset - skipping Google Drive extraction")
        return existing_dataset
    
    print("\n🔗 MOUNTING GOOGLE DRIVE")
    print("=" * 40)
    print("📋 You will be prompted to authenticate with Google.")
    print("📋 Please follow the authentication link and enter the code.")
    print("=" * 40)
    
    try:
        # Mount Google Drive - this will prompt for authentication
        drive.mount('/content/drive', force_remount=True)
        print("✅ Google Drive mounted successfully!")
        
        # Verify mount by listing drive contents
        drive_path = Path('/content/drive')
        if drive_path.exists():
            print(f"📁 Drive contents: {list(drive_path.iterdir())}")
        
        # Check MyDrive specifically
        mydrive_path = Path('/content/drive/MyDrive')
        if mydrive_path.exists():
            print("✅ MyDrive folder found")
        else:
            print("❌ MyDrive folder not found")
            raise FileNotFoundError("MyDrive folder not accessible")
            
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        print("🔄 Retrying with force remount...")
        try:
            drive.mount('/content/drive', force_remount=True)
            print("✅ Google Drive mounted successfully on retry!")
        except Exception as e2:
            print(f"❌ Failed to mount Google Drive on retry: {e2}")
            raise e2
    
    # Look for data.rar in MyDrive
    dataset_rar_path = "/content/drive/MyDrive/data.rar"
    print(f"\n📦 LOOKING FOR DATASET")
    print("=" * 40)
    print(f"🔍 Checking for dataset at: {dataset_rar_path}")
    
    if not os.path.exists(dataset_rar_path):
        print(f"❌ Dataset not found at: {dataset_rar_path}")
        print("💡 Please ensure your merged dataset RAR file is named 'data.rar' and placed in your Google Drive root (MyDrive)")
        print("💡 Use the dataset_merger.py script to create a unified dataset first")
        
        # Try to help by showing available files in MyDrive
        try:
            mydrive_files = [f.name for f in Path('/content/drive/MyDrive').iterdir() if f.is_file()]
            archive_files = [f for f in mydrive_files if f.lower().endswith(('.zip', '.rar', '.tar', '.gz'))]
            if archive_files:
                print(f"📂 Found these archive files in MyDrive: {archive_files}")
                print("💡 Please rename one of these to 'data.rar' or upload your merged dataset as 'data.rar'")
            else:
                print("📂 No archive files found in MyDrive")
                print("💡 Please upload your merged dataset as 'data.rar' to your Google Drive root")
        except Exception as e:
            print(f"⚠️ Could not list MyDrive contents: {e}")
        
        raise FileNotFoundError(f"Dataset not found at {dataset_rar_path}")
    
    # Check file size
    file_size = os.path.getsize(dataset_rar_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"✅ Dataset found! Size: {file_size_mb:.2f} MB")
    
    # Extract dataset with progress tracking
    print(f"\n📦 EXTRACTING RAR DATASET")
    print("=" * 40)
    print(f"📂 Source: {dataset_rar_path}")
    print(f"📂 Destination: {config.PROJECT_ROOT}")
    
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="📦 Extracting", colour="green") as pbar:
        try:
            with rarfile.RarFile(dataset_rar_path, 'r') as rar_ref:
                # Get list of files to extract
                file_list = rar_ref.infolist()
                print(f"📋 Found {len(file_list)} files in archive")
                
                # Extract files one by one with progress
                for file_info in file_list:
                    try:
                        rar_ref.extract(file_info, config.PROJECT_ROOT)
                        pbar.update(file_info.file_size)
                    except Exception as e:
                        print(f"⚠️ Warning: Could not extract {file_info.filename}: {e}")
                        continue
                        
        except rarfile.BadRarFile:
            print("❌ Error: Invalid or corrupted RAR file")
            raise
        except rarfile.RarCannotExec:
            print("❌ Error: unrar tool not found or not working")
            print("💡 Trying alternative extraction method...")
            
            # Fallback to system unrar command
            try:
                import subprocess
                result = subprocess.run([
                    'unrar', 'x', '-y', dataset_rar_path, str(config.PROJECT_ROOT)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Extraction completed using system unrar!")
                    pbar.update(file_size)  # Update progress bar to completion
                else:
                    print(f"❌ System unrar failed: {result.stderr}")
                    raise Exception("RAR extraction failed")
                    
            except FileNotFoundError:
                print("❌ Error: unrar tool not available")
                print("💡 Please convert your RAR file to ZIP format and rename it to 'data.zip'")
                raise Exception("RAR extraction not supported - please use ZIP format")
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            raise
    
    print("✅ Extraction completed!")
    
    # Find the extracted dataset directory (should contain merged dataset structure)
    print("\n🔍 LOCATING MERGED DATASET")
    print("=" * 40)
    
    # Look for data.yaml file (created by dataset_merger.py)
    dataset_path = None
    yaml_files = list(config.PROJECT_ROOT.rglob("data.yaml"))
    
    if yaml_files:
        # Use the first data.yaml found (should be the merged dataset)
        yaml_file = yaml_files[0]
        dataset_path = yaml_file.parent
        print(f"✅ Found merged dataset config: {yaml_file}")
        print(f"📁 Dataset root: {dataset_path}")
        
        # Verify it's a merged dataset by checking for class_mapping.yaml
        mapping_file = dataset_path / "class_mapping.yaml"
        if mapping_file.exists():
            print("✅ Confirmed: This is a merged dataset (class_mapping.yaml found)")
            
            # Show some info about the merged dataset
            try:
                with open(mapping_file, 'r') as f:
                    mapping_data = yaml.safe_load(f)
                unified_classes = mapping_data.get('unified_classes', [])
                stats = mapping_data.get('statistics', {})
                print(f"📊 Unified classes ({len(unified_classes)}): {unified_classes[:5]}..." if len(unified_classes) > 5 else f"📊 Unified classes: {unified_classes}")
                print(f"📈 Dataset statistics: {stats}")
            except Exception as e:
                print(f"⚠️ Could not read mapping file: {e}")
        else:
            print("⚠️ No class_mapping.yaml found - this might not be a merged dataset")
    else:
        # Fallback: look for standard dataset structure
        extracted_dirs = [d for d in config.PROJECT_ROOT.iterdir() if d.is_dir() and d.name not in ['logs', 'models', 'training']]
        print(f"📂 Found directories: {[d.name for d in extracted_dirs]}")
        
        # Look for common dataset structure
        for dir_path in extracted_dirs:
            try:
                subdirs = [d.name for d in dir_path.iterdir() if d.is_dir()]
                print(f"📁 {dir_path.name} contains: {subdirs}")
                
                if any(subdir in subdirs for subdir in ['train', 'valid', 'test']):
                    dataset_path = dir_path
                    print(f"✅ Dataset structure found in: {dataset_path}")
                    break
            except Exception as e:
                print(f"⚠️ Could not check {dir_path}: {e}")
                continue
        
        if dataset_path is None:
            if extracted_dirs:
                dataset_path = extracted_dirs[0]
                print(f"⚠️ No standard structure detected. Using: {dataset_path}")
            else:
                dataset_path = config.PROJECT_ROOT
                print(f"⚠️ No extracted directories found. Using: {dataset_path}")
    
    print(f"📁 Final dataset path: {dataset_path}")
    
    # Verify dataset structure
    try:
        contents = list(dataset_path.iterdir())
        print(f"📋 Dataset contents: {[item.name for item in contents[:10]]}")  # Show first 10 items
        
        # Check for expected merged dataset structure
        expected_dirs = ['train', 'valid', 'test']
        found_dirs = [item.name for item in contents if item.is_dir()]
        
        for expected_dir in expected_dirs:
            if expected_dir in found_dirs:
                subdir = dataset_path / expected_dir
                images_dir = subdir / 'images'
                labels_dir = subdir / 'labels'
                if images_dir.exists() and labels_dir.exists():
                    img_count = len(list(images_dir.glob('*')))
                    label_count = len(list(labels_dir.glob('*.txt')))
                    print(f"📊 {expected_dir}: {img_count} images, {label_count} labels")
    except Exception as e:
        print(f"⚠️ Could not list dataset contents: {e}")
    
    return dataset_path

def create_dataset_yaml(dataset_path):
    """Create or verify dataset.yaml file"""
    yaml_path = dataset_path / "data.yaml"
    
    # Check if data.yaml exists
    if yaml_path.exists():
        print("✅ Found existing data.yaml")
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            print(f"📊 Dataset:")
            print(f"   - Classes: {config_data.get('nc', 'unknown')}")
            
            return yaml_path
        except Exception as e:
            print(f"⚠️ Error reading data.yaml: {e}")
    
    # If no dataset config found, create basic one
    print("🔄 Creating basic dataset configuration...")
    
    # Look for train/val directories
    train_dir = None
    val_dir = None
    test_dir = None
    
    possible_structures = [
        (dataset_path / "train" / "images", dataset_path / "valid" / "images", dataset_path / "test" / "images"),
        (dataset_path / "train", dataset_path / "valid", dataset_path / "test"),
        (dataset_path / "images" / "train", dataset_path / "images" / "valid", dataset_path / "images" / "test"),
        (dataset_path / "train" / "images", dataset_path / "val" / "images", dataset_path / "test" / "images"),
    ]
    
    for train_path, val_path, test_path in possible_structures:
        if train_path.exists() and val_path.exists():
            train_dir = str(train_path)
            val_dir = str(val_path)
            if test_path.exists():
                test_dir = str(test_path)
            break
    
    if train_dir is None:
        # More detailed search
        print("🔍 Searching for dataset directories...")
        all_dirs = [d for d in dataset_path.rglob("*") if d.is_dir()]
        train_candidates = [d for d in all_dirs if 'train' in d.name.lower()]
        val_candidates = [d for d in all_dirs if any(name in d.name.lower() for name in ['valid', 'val'])]
        
        print(f"📂 Found train candidates: {[str(d) for d in train_candidates]}")
        print(f"📂 Found validation candidates: {[str(d) for d in val_candidates]}")
        
        if train_candidates and val_candidates:
            # Use the first viable candidates
            train_dir = str(train_candidates[0])
            val_dir = str(val_candidates[0])
        else:
            raise ValueError("Could not find train/valid directories in dataset")
    
    # Try to detect classes from dataset
    classes = []
    
    # Look for class_mapping.yaml first (from merged dataset)
    mapping_file = dataset_path / "class_mapping.yaml"
    if mapping_file.exists():
        try:
            with open(mapping_file, 'r') as f:
                mapping_data = yaml.safe_load(f)
            classes = mapping_data.get('unified_classes', [])
            print(f"✅ Loaded {len(classes)} classes from class_mapping.yaml")
        except Exception as e:
            print(f"⚠️ Error reading class_mapping.yaml: {e}")
    
    # Fallback: look for other class files
    if not classes:
        for class_file in ["classes.txt", "names.txt", "data.names"]:
            class_path = dataset_path / class_file
            if class_path.exists():
                with open(class_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines() if line.strip()]
                break
    
    # Try to extract classes from label files
    if not classes:
        print("🔍 Attempting to extract classes from label files...")
        label_dirs = []
        
        # Look for labels directories
        train_path = Path(train_dir)
        if (train_path.parent / "labels").exists():
            label_dirs.append(train_path.parent / "labels")
        elif (train_path / "labels").exists():
            label_dirs.append(train_path / "labels")
        
        # Extract unique class IDs from label files
        class_ids = set()
        for label_dir in label_dirs:
            for label_file in label_dir.glob("*.txt"):
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_ids.add(int(parts[0]))
                except:
                    continue
        
        if class_ids:
            # Create generic class names
            max_class_id = max(class_ids)
            classes = [f"class_{i}" for i in range(max_class_id + 1)]
            print(f"✅ Extracted {len(classes)} classes from label files")
    
    # Last resort: use a reasonable default for crop health monitoring
    if not classes:
        classes = [
            "healthy", "disease", "pest", "nutrient_deficiency", "fungal_infection",
            "bacterial_infection", "viral_infection", "insect_damage", "drought_stress",
            "leaf_spot", "rust", "blight", "powdery_mildew", "mosaic_virus"
        ]
        print(f"⚠️ Using default crop health classes: {len(classes)} classes")
    
    # Create dataset.yaml
    dataset_yaml = {
        'path': str(dataset_path),
        'train': train_dir,
        'val': val_dir,
        'nc': len(classes),
        'names': classes
    }
    
    # Add test path if it exists
    if test_dir:
        dataset_yaml['test'] = test_dir
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"✅ Created dataset.yaml with {len(classes)} classes")
    print(f"📋 Classes: {classes[:10]}..." if len(classes) > 10 else f"📋 Classes: {classes}")
    print(f"📁 Paths:")
    print(f"   - Train: {train_dir}")
    print(f"   - Val: {val_dir}")
    if test_dir:
        print(f"   - Test: {test_dir}")
    
    return yaml_path

def find_checkpoint():
    """Find the latest checkpoint to resume training"""
    checkpoint_path = None
    checkpoint_epoch = 0
    
    # Look for existing training runs
    runs_dir = config.TRAINING_DIR / "crop_health_model"
    if runs_dir.exists():
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if run_dirs:
            # Get the most recent run
            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
            weights_dir = latest_run / "weights"
            
            # Look for the latest checkpoint
            if weights_dir.exists():
                checkpoint_files = list(weights_dir.glob("epoch*.pt"))
                if checkpoint_files:
                    # Get the latest epoch checkpoint
                    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                    checkpoint_path = latest_checkpoint
                    
                    # Extract epoch number from filename
                    try:
                        epoch_str = latest_checkpoint.stem.replace("epoch", "")
                        checkpoint_epoch = int(epoch_str)
                    except:
                        checkpoint_epoch = 0
                elif (weights_dir / "last.pt").exists():
                    checkpoint_path = weights_dir / "last.pt"
                    # Try to determine epoch from results file
                    results_file = latest_run / "results.csv"
                    if results_file.exists():
                        try:
                            import pandas as pd
                            df = pd.read_csv(results_file)
                            checkpoint_epoch = len(df)
                        except:
                            checkpoint_epoch = 0
    
    return checkpoint_path, checkpoint_epoch

def save_checkpoint(model, epoch, metrics, output_dir):
    """Save training checkpoint"""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    
    # Save model state
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.model.state_dict() if hasattr(model, 'model') else model.state_dict(),
            'metrics': metrics,
            'config': {
                'batch_size': config.BATCH_SIZE,
                'image_size': config.IMAGE_SIZE,
                'model_size': config.YOLO_MODEL_SIZE
            }
        }, checkpoint_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Keep only last 3 checkpoints to save space
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()
                print(f"🗑️ Removed old checkpoint: {old_checkpoint.name}")
                
    except Exception as e:
        print(f"⚠️ Failed to save checkpoint: {e}")

def train_model_with_tracking(dataset_yaml_path, tracker):
    """Train YOLO model with YOLOv8m and comprehensive tracking"""
    print("🚀 Starting Colab training with YOLOv8m...")
    
    # Display dataset information
    try:
        with open(dataset_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        num_classes = data_config.get('nc', 0)
        print(f"🎯 Training with {num_classes} classes using YOLOv8m")
            
    except Exception as e:
        print(f"⚠️ Could not read dataset config: {e}")
    
    # Find checkpoint
    checkpoint_path, start_epoch = find_checkpoint()
    
    if checkpoint_path and start_epoch > 0:
        print(f"🔄 Found checkpoint: {checkpoint_path}")
        print(f"📊 Resuming from epoch: {start_epoch}")
        response = input("Do you want to resume from this checkpoint? (y/n): ").strip().lower()
        if response != 'y':
            checkpoint_path = None
            start_epoch = 0
            print("🆕 Starting fresh training")
    else:
        print("🆕 No checkpoint found, starting fresh training")
    
    # Load YOLO model
    with tqdm(desc="🔧 Loading model", total=2) as pbar:
        model = YOLO(config.YOLO_MODEL_SIZE)
        pbar.update(1)
        
        # Add custom callback
        def on_train_epoch_end(trainer):
            """Custom callback for epoch completion"""
            epoch = trainer.epoch + 1
            
            # Extract metrics safely
            metrics = {}
            try:
                if hasattr(trainer, 'loss') and trainer.loss is not None:
                    metrics['loss'] = float(trainer.loss.item())
                if hasattr(trainer, 'validator') and trainer.validator:
                    if hasattr(trainer.validator, 'metrics'):
                        val_metrics = trainer.validator.metrics
                        metrics['map50'] = float(val_metrics.get('mAP50', 0))
                        metrics['map50_95'] = float(val_metrics.get('mAP50-95', 0))
                        metrics['precision'] = float(val_metrics.get('precision', 0))
                        metrics['recall'] = float(val_metrics.get('recall', 0))
                if hasattr(trainer, 'optimizer') and trainer.optimizer:
                    metrics['lr'] = float(trainer.optimizer.param_groups[0]['lr'])
            except Exception as e:
                print(f"⚠️ Error extracting metrics: {e}")
                metrics = {
                    'loss': 0.0, 'map50': 0.0, 'map50_95': 0.0,
                    'precision': 0.0, 'recall': 0.0, 'lr': 0.0
                }
            
            # Update tracker
            tracker.update_epoch(epoch, metrics)
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                save_dir = Path(trainer.save_dir) if hasattr(trainer, 'save_dir') else config.TRAINING_DIR / "crop_health_model"
                save_checkpoint(model, epoch, metrics, save_dir)
        
        model.add_callback('on_train_epoch_end', on_train_epoch_end)
        pbar.update(1)
    
    # Training arguments optimized for Colab
    train_args = {
        'data': str(dataset_yaml_path),
        'epochs': config.EPOCHS,
        'imgsz': config.IMAGE_SIZE,
        'batch': config.BATCH_SIZE,
        'device': config.DEVICE,
        'workers': config.WORKERS,
        'project': str(config.TRAINING_DIR),
        'name': f'crop_health_model',
        'exist_ok': True,
        'patience': config.PATIENCE,
        'save_period': config.SAVE_PERIOD,
        'amp': config.AMP,
        'verbose': False,
        'plots': True,
        'save': True,
        'cache': True,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        # Colab-optimized hyperparameters
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }
    
    # Add resume functionality
    if checkpoint_path:
        train_args['resume'] = str(checkpoint_path)
        print(f"🔄 Resuming training from: {checkpoint_path}")
    else:
        train_args['resume'] = False
    
    print(f"📊 Colab Training Configuration:")
    print(f"   - Model: {config.YOLO_MODEL_SIZE} (YOLOv8m)")
    print(f"   - Epochs: {config.EPOCHS}")
    print(f"   - Batch Size: {config.BATCH_SIZE}")
    print(f"   - Image Size: {config.IMAGE_SIZE}")
    print(f"   - Device: {config.DEVICE}")
    print(f"   - Classes: {num_classes}")
    
    # Start training
    tracker.start_training()
    
    try:
        results = model.train(**train_args)
        print("✅ Colab training completed!")
        
        # Save final checkpoint
        final_checkpoint_dir = config.TRAINING_DIR / "crop_health_model"
        if final_checkpoint_dir.exists():
            latest_run = max([d for d in final_checkpoint_dir.iterdir() if d.is_dir()], 
                           key=lambda x: x.stat().st_mtime, default=final_checkpoint_dir)
            save_checkpoint(model, config.EPOCHS, {}, latest_run)
        
        return model, results
    finally:
        tracker.finish_training()

def download_best_model():
    """Download the best trained model"""
    print("📥 Preparing model download...")
    
    # Find the best model
    runs_dir = config.TRAINING_DIR / "crop_health_model"
    best_model_path = None
    
    # Look for the most recent training run
    if runs_dir.exists():
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if run_dirs:
            # Get the most recent run
            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
            weights_dir = latest_run / "weights"
            
            if (weights_dir / "best.pt").exists():
                best_model_path = weights_dir / "best.pt"
            elif (weights_dir / "last.pt").exists():
                best_model_path = weights_dir / "last.pt"
    
    if best_model_path and best_model_path.exists():
        print(f"📦 Best model found: {best_model_path}")
        
        # Create a descriptive filename
        model_filename = f"crop_health_model_best_{config.EPOCHS}epochs.pt"
        
        # Copy to a download location
        download_path = config.PROJECT_ROOT / model_filename
        shutil.copy2(best_model_path, download_path)
        
        print(f"⬇️ Downloading model as: {model_filename}")
        files.download(str(download_path))
        
        # Also download training results if available
        results_dir = best_model_path.parent.parent
        if (results_dir / "results.png").exists():
            results_filename = f"training_results_{config.EPOCHS}epochs.png"
            results_download_path = config.PROJECT_ROOT / results_filename
            shutil.copy2(results_dir / "results.png", results_download_path)
            files.download(str(results_download_path))
            print(f"📊 Downloaded training results: {results_filename}")
        
        print("✅ Model download completed!")
    else:
        print("❌ Could not find trained model")

def main():
    """Main training pipeline optimized for Colab with YOLOv8m"""
    print("🌾 SMART CROP HEALTH MONITORING - GOOGLE COLAB TRAINING")
    print("=" * 60)
    print("🚀 Optimized for A100 GPU with YOLOv8m")
    print("=" * 60)
    
    # Check GPU
    gpu_available = check_gpu()
    if not gpu_available:
        print("⚠️ Warning: Training without GPU will be very slow")
        response = input("Do you want to continue? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Training cancelled")
            return
    
    try:
        # Mount drive and extract dataset
        dataset_path = mount_drive_and_extract_dataset()
        
        # Create dataset configuration
        dataset_yaml_path = create_dataset_yaml(dataset_path)
        
        # Initialize progress tracker
        tracker = ProgressTracker(config.EPOCHS)
        
        # Train model with comprehensive tracking
        model, results = train_model_with_tracking(dataset_yaml_path, tracker)
        
        # Save training log
        experiment_dir = config.TRAINING_DIR / "crop_health_model"
        if experiment_dir.exists():
            latest_run = max([d for d in experiment_dir.iterdir() if d.is_dir()], 
                           key=lambda x: x.stat().st_mtime, default=experiment_dir)
            tracker.save_training_log(latest_run)
        
        # Download best model
        download_best_model()
        
        print("\n🎉 COLAB TRAINING PIPELINE COMPLETED!")
        print("=" * 60)
        print("📝 Training completed with YOLOv8m")
        print("📄 Model optimized for production deployment")
        print("📊 Check training logs for detailed metrics")
        print("🚀 Ready for inference and deployment!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ ERROR DURING TRAINING: {str(e)}")
        print("🔍 Please check the error message and try again")
        raise

if __name__ == "__main__":
    main()
