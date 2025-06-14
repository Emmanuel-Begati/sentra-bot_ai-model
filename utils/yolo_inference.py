"""
YOLOv8 Inference Engine for Crop Disease/Pest Detection
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Union
import numpy as np
from PIL import Image
import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics not available. Install with: pip install ultralytics")

from config import MODELS_DIR, DEVICE, IMAGE_SIZE

logger = logging.getLogger(__name__)


class YOLOInference:
    """YOLOv8 inference engine for disease and pest detection"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []
        self.device = DEVICE
        
        if not YOLO_AVAILABLE:
            raise ImportError("YOLOv8 not available. Install ultralytics package.")
        
        # Load model
        self.model_path = self._find_best_model(model_path)
        if self.model_path:
            self._load_model()
        else:
            logger.warning("No trained model found. Using pretrained YOLOv8 model.")
            self._load_pretrained_model()
    
    def _find_best_model(self, model_path: str = None) -> str:
        """Find the best available trained model"""
        if model_path and Path(model_path).exists():
            return model_path
        
        models_dir = Path(MODELS_DIR)
        if not models_dir.exists():
            return None
        
        # Look for best model files - check multiple patterns
        model_patterns = ["*_best.pt", "best.pt", "*.pt"]
        model_files = []
        
        for pattern in model_patterns:
            found_files = list(models_dir.glob(pattern))
            model_files.extend(found_files)
        
        # Remove duplicates and filter out pretrained models
        model_files = list(set(model_files))
        model_files = [f for f in model_files if not f.name.startswith('yolov8')]
        
        if not model_files:
            logger.warning("No trained models found in models directory")
            return None
        
        # Return the most recent model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found best model: {latest_model}")
        return str(latest_model)
    
    def _load_model(self):
        """Load trained YOLOv8 model"""
        try:
            logger.info(f"Loading YOLOv8 model: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Extract class names from model
            if hasattr(self.model.model, 'names'):
                self.class_names = list(self.model.model.names.values())
            else:
                logger.warning("Could not extract class names from model")
            
            logger.info(f"Model loaded successfully with {len(self.class_names)} classes")
            logger.info(f"Classes: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """Load pretrained YOLOv8 model as fallback"""
        try:
            logger.info("Loading pretrained YOLOv8s model")
            self.model = YOLO('yolov8s.pt')
            self.class_names = list(self.model.model.names.values())
            logger.warning("Using pretrained model - disease detection may not be accurate")
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise
    
    def predict(self, 
                image: Union[str, Path, Image.Image, np.ndarray],
                confidence: float = None,
                iou_threshold: float = 0.45,
                max_detections: int = 300) -> List[Dict]:
        """
        Run inference on image and return detections
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            confidence: Confidence threshold (uses default if None)
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections
            
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            # Use provided confidence or default
            conf_threshold = confidence if confidence is not None else self.confidence_threshold
            
            # Run inference
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                max_det=max_detections,
                imgsz=IMAGE_SIZE,
                verbose=False
            )
            
            # Process results
            detections = self._process_results(results[0])
            
            logger.info(f"Detected {len(detections)} objects with confidence >= {conf_threshold}")
            return detections
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return []
    
    def _process_results(self, result) -> List[Dict]:
        """Process YOLOv8 results into structured format"""
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Get image dimensions for normalization
        img_height, img_width = result.orig_shape
        
        # Extract detection data
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            confidence = float(confidences[i])  # Convert to Python float
            class_id = int(class_ids[i])  # Convert to Python int
            
            # Get class name
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            # Convert to normalized center coordinates and dimensions (all as Python floats)
            center_x = float((x1 + x2) / 2 / img_width)
            center_y = float((y1 + y2) / 2 / img_height)
            width = float((x2 - x1) / img_width)
            height = float((y2 - y1) / img_height)
            
            detection = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': (center_x, center_y, width, height),  # normalized
                'bbox_pixels': (int(x1), int(y1), int(x2), int(y2)),  # pixel coordinates
                'area': float(width * height),  # normalized area
                'image_shape': (int(img_width), int(img_height)),
                'severity': self._assess_severity(class_name, confidence),
                'disease_type': self._categorize_disease(class_name)
            }
            
            detections.append(detection)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def _assess_severity(self, class_name: str, confidence: float) -> str:
        """Assess severity based on disease type and confidence"""
        # High severity diseases/pests
        high_severity = ['late blight', 'early blight', 'fusarium', 'rust', 'bacterial blight', 
                        'black rot', 'anthracnose', 'armyworm', 'spider mite']
        
        # Medium severity diseases/pests  
        medium_severity = ['leaf spot', 'powdery mildew', 'downy mildew', 'mosaic virus',
                          'leaf curl', 'bacterial spot', 'scab']
        
        # Check if disease name contains high severity keywords
        for disease in high_severity:
            if disease.lower() in class_name.lower():
                return 'high' if confidence > 0.7 else 'medium'
        
        # Check if disease name contains medium severity keywords
        for disease in medium_severity:
            if disease.lower() in class_name.lower():
                return 'medium' if confidence > 0.5 else 'low'
        
        # Default based on confidence
        if confidence > 0.8:
            return 'high'
        elif confidence > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_disease(self, class_name: str) -> str:
        """Categorize disease type"""
        class_lower = class_name.lower()
        
        if any(term in class_lower for term in ['virus', 'mosaic', 'curl', 'yellow']):
            return 'viral'
        elif any(term in class_lower for term in ['bacterial', 'blight', 'rot', 'spot']):
            return 'bacterial'
        elif any(term in class_lower for term in ['rust', 'mildew', 'mold', 'anthracnose', 'scab']):
            return 'fungal'
        elif any(term in class_lower for term in ['mite', 'worm', 'insect']):
            return 'pest'
        elif 'deficiency' in class_lower:
            return 'nutritional'
        else:
            return 'unknown'
    
    def predict_batch(self, 
                     images: List[Union[str, Path, Image.Image, np.ndarray]],
                     confidence: float = None,
                     iou_threshold: float = 0.45) -> List[List[Dict]]:
        """
        Run batch inference on multiple images
        
        Args:
            images: List of input images
            confidence: Confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detection lists (one per image)
        """
        batch_results = []
        
        for image in images:
            detections = self.predict(
                image=image,
                confidence=confidence,
                iou_threshold=iou_threshold
            )
            batch_results.append(detections)
        
        return batch_results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {'status': 'not_loaded'}
        
        info = {
            'status': 'loaded',
            'model_path': self.model_path,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'device': str(self.device),
            'confidence_threshold': self.confidence_threshold
        }
        
        # Add model-specific info if available
        if hasattr(self.model, 'model'):
            try:
                # Count parameters
                total_params = sum(p.numel() for p in self.model.model.parameters())
                info['parameters'] = total_params
                info['model_size'] = f"{total_params / 1e6:.1f}M"
            except:
                pass
        
        return info
    
    def visualize_predictions(self, 
                            image: Union[str, Path, Image.Image, np.ndarray],
                            detections: List[Dict] = None,
                            save_path: str = None,
                            show_confidence: bool = True,
                            conf_threshold: float = None) -> Image.Image:
        """
        Visualize predictions on image
        
        Args:
            image: Input image
            detections: Detections to visualize (if None, runs inference)
            save_path: Path to save visualization
            show_confidence: Whether to show confidence scores
            conf_threshold: Confidence threshold for visualization
            
        Returns:
            PIL Image with visualizations
        """
        if detections is None:
            detections = self.predict(image, confidence=conf_threshold)
        
        # Convert image to PIL if needed
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.copy()
        
        # Use YOLOv8's built-in visualization if available
        if self.model and hasattr(self.model, 'predict'):
            try:
                results = self.model.predict(
                    source=image,
                    conf=conf_threshold or self.confidence_threshold,
                    device=self.device,
                    save=False,
                    show=False
                )
                
                if results and len(results) > 0:
                    # Get annotated image
                    annotated_img = results[0].plot()
                    pil_image = Image.fromarray(annotated_img[..., ::-1])  # BGR to RGB
                    
                    if save_path:
                        pil_image.save(save_path)
                    
                    return pil_image
            except Exception as e:
                logger.warning(f"Built-in visualization failed: {e}")
        
        # Fallback: return original image
        if save_path:
            pil_image.save(save_path)
        
        return pil_image


def test_inference():
    """Test function for YOLO inference"""
    try:
        inference = YOLOInference()
        info = inference.get_model_info()
        
        print("🔍 YOLO Inference Test")
        print(f"Model status: {info['status']}")
        print(f"Number of classes: {info.get('num_classes', 'Unknown')}")
        print(f"Device: {info.get('device', 'Unknown')}")
        
        if info['status'] == 'loaded':
            print("✅ YOLO inference engine ready")
            return True
        else:
            print("❌ YOLO inference engine not ready")
            return False
            
    except Exception as e:
        print(f"❌ YOLO inference test failed: {e}")
        return False


if __name__ == "__main__":
    test_inference()
