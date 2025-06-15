"""
Simplified YOLOv8 Inference Engine
"""

import logging
from pathlib import Path
from typing import List, Dict, Union
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from config import MODELS_DIR, DEVICE, IMAGE_SIZE, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class YOLOInference:
    """Simplified YOLO inference engine"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []
        
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 not available. Install with: pip install ultralytics")
        
        # Load model
        model_path = self._find_best_model(model_path)
        self._load_model(model_path)
    
    def _find_best_model(self, model_path: str = None) -> str:
        """Find the best available model"""
        if model_path and Path(model_path).exists():
            return model_path
        
        # Look for trained models
        if MODELS_DIR.exists():
            model_files = list(MODELS_DIR.glob("*.pt"))
            # Filter out pretrained models
            trained_models = [f for f in model_files if not f.name.startswith('yolov8')]
            
            if trained_models:
                # Return most recent
                latest_model = max(trained_models, key=lambda x: x.stat().st_mtime)
                logger.info(f"Found trained model: {latest_model}")
                return str(latest_model)
        
        # Fallback to pretrained
        logger.warning("No trained model found, using pretrained YOLOv8s")
        return "yolov8s.pt"
    
    def _load_model(self, model_path: str):
        """Load YOLO model"""
        try:
            logger.info(f"Loading model: {model_path}")
            self.model = YOLO(model_path)
            
            # Extract class names
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                self.class_names = list(self.model.model.names.values())
            
            logger.info(f"Model loaded with {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> List[Dict]:
        """Run inference on image"""
        try:
            if self.model is None:
                return []
            
            # Run inference
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                device=DEVICE,
                imgsz=IMAGE_SIZE,
                verbose=False
            )
            
            # Process results
            return self._process_results(results[0])
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []
    
    def _process_results(self, result) -> List[Dict]:
        """Process YOLO results"""
        detections = []
        
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            return detections
        
        try:
            # Get image dimensions
            img_height, img_width = result.orig_shape
            
            # Extract detection data
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                confidence = float(confidences[i])  # Ensure Python float
                class_id = int(class_ids[i])  # Ensure Python int
                
                # Get class name
                if 0 <= class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                # Calculate normalized bbox and area - ensure Python floats
                center_x = float((x1 + x2) / 2 / img_width)
                center_y = float((y1 + y2) / 2 / img_height)
                width = float((x2 - x1) / img_width)
                height = float((y2 - y1) / img_height)
                area = float(width * height)
                
                detection = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': (center_x, center_y, width, height),
                    'area': area
                }
                
                detections.append(detection)
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error processing results: {e}")
        
        return detections
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'status': 'loaded' if self.model else 'not_loaded',
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'confidence_threshold': self.confidence_threshold,
            'device': DEVICE
        }
    
    def predict_with_plant_context(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict:
        """Run inference optimized for plant health analysis"""
        detections = self.predict(image)
        
        # Categorize detections
        health_issues = []
        healthy_detections = []
        
        for detection in detections:
            if detection['class_name'].lower() == 'healthy':
                healthy_detections.append(detection)
            else:
                health_issues.append(detection)
        
        return {
            'all_detections': detections,
            'health_issues': health_issues,
            'healthy_regions': healthy_detections,
            'has_issues': len(health_issues) > 0,
            'total_detections': len(detections)
        }
