"""
FastAPI Backend for Smart Crop Health Monitoring System
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from PIL import Image
import io
import numpy as np

from utils.plantnet_api import PlantNetAPI
from utils.health_scorer import HealthScorer, SensorData, DetectionResult, HealthAssessment
from utils.yolo_inference import YOLOInference
from config import API_HOST, API_PORT, MODELS_DIR, LOG_FILE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create logs directory
LOG_FILE.parent.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Crop Health Monitoring API",
    description="AI-powered crop health analysis with disease detection and farming advice",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
plantnet_api = PlantNetAPI()
health_scorer = HealthScorer()
yolo_inference = None  # Will be initialized when needed

# In-memory storage for predictions (in production, use a database)
predictions_store = {}


# Pydantic models
class SensorDataModel(BaseModel):
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    soil_moisture: Optional[float] = None
    ph: Optional[float] = None
    light_intensity: Optional[float] = None
    timestamp: Optional[str] = None


class PredictionRequest(BaseModel):
    sensor_data: Optional[SensorDataModel] = None
    location: Optional[str] = None
    crop_type: Optional[str] = None


class PlantIdentificationResponse(BaseModel):
    success: bool
    plant_id: Optional[str] = None
    scientific_name: Optional[str] = None
    common_name: Optional[str] = None
    confidence: Optional[float] = None
    all_results: Optional[List[Dict]] = None


class HealthAnalysisResponse(BaseModel):
    prediction_id: str
    overall_health: float
    risk_level: str
    confidence: float
    detected_issues: List[Dict]
    disease_classification: Dict  # New field for disease classification
    environmental_analysis: Dict
    recommendations: List[str]
    timestamp: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global yolo_inference
    
    try:
        logger.info("Initializing YOLO inference engine...")
        from utils.yolo_inference import YOLOInference
        yolo_inference = YOLOInference()
        logger.info("YOLO inference engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize YOLO inference: {e}")
        logger.warning("Disease detection will not be available")


# Health check endpoint
@app.get("/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "yolo_inference": yolo_inference is not None,
            "plantnet_api": bool(plantnet_api.api_key),
            "health_scorer": True
        }
    }


# Plant identification endpoint
@app.post("/identify-plant", response_model=PlantIdentificationResponse)
async def identify_plant(image: UploadFile = File(...)):
    """Identify plant species using PlantNet API"""
    try:
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await image.read()
        
        # Call PlantNet API
        logger.info(f"Identifying plant from image: {image.filename}")
        result = plantnet_api.identify_plant(image_data)
        
        if not result or not result.get('success'):
            return PlantIdentificationResponse(
                success=False,
                plant_id=None,
                scientific_name=None,
                common_name=None,
                confidence=None
            )
        
        best_match = result.get('best_match')
        if not best_match:
            return PlantIdentificationResponse(success=False)
        
        # Generate plant ID
        plant_id = str(uuid.uuid4())
        
        return PlantIdentificationResponse(
            success=True,
            plant_id=plant_id,
            scientific_name=best_match.get('scientific_name'),
            common_name=best_match.get('primary_common_name'),
            confidence=best_match.get('score'),
            all_results=result.get('all_results', [])
        )
        
    except Exception as e:
        logger.error(f"Plant identification error: {e}")
        raise HTTPException(status_code=500, detail=f"Plant identification failed: {str(e)}")


# Main prediction endpoint
@app.post("/predict", response_model=HealthAnalysisResponse)
async def predict_crop_health(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    sensor_data: Optional[str] = None,
    location: Optional[str] = None,
    crop_type: Optional[str] = None
):
    """
    Analyze crop health from image and sensor data
    
    Args:
        image: Crop image file
        sensor_data: JSON string of sensor readings
        location: Geographic location (optional)
        crop_type: Known crop type (optional)
    """
    try:
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Validate and read image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Parse sensor data
        sensor_obj = None
        if sensor_data:
            try:
                sensor_dict = json.loads(sensor_data)
                sensor_obj = SensorData(**sensor_dict)
            except Exception as e:
                logger.warning(f"Invalid sensor data: {e}")
        
        # Step 1: Identify plant species (if not provided)
        plant_species = crop_type
        if not plant_species and plantnet_api.api_key:
            logger.info("Attempting plant identification...")
            plant_result = plantnet_api.identify_plant(image_data)
            if plant_result and plant_result.get('success'):
                best_match = plant_result.get('best_match')
                if best_match and best_match.get('score', 0) > 0.3:
                    plant_species = best_match.get('primary_common_name')
                    logger.info(f"Identified plant: {plant_species}")
        
        # Step 2: Run disease/pest detection
        detections = []
        disease_classification = {
            'total_detections': 0,
            'disease_types': {},
            'severity_levels': {},
            'highest_confidence': 0.0,
            'primary_disease': None
        }
        
        if yolo_inference:
            logger.info("Running YOLO inference for disease/pest detection...")
            detection_results = yolo_inference.predict(pil_image)
            
            # Convert to DetectionResult objects and analyze classifications
            disease_types = {}
            severity_levels = {}
            highest_conf = 0.0
            primary_disease = None
            
            for detection in detection_results:
                # Convert numpy types to Python types for serialization
                detection_obj = DetectionResult(
                    class_name=detection.get('class_name', ''),
                    confidence=float(detection.get('confidence', 0.0)),
                    bbox=tuple(float(x) for x in detection.get('bbox', (0, 0, 0, 0)))
                )
                detections.append(detection_obj)
                
                # Track disease classification
                disease_type = detection.get('disease_type', 'unknown')
                severity = detection.get('severity', 'unknown')
                conf = float(detection.get('confidence', 0.0))
                
                disease_types[disease_type] = disease_types.get(disease_type, 0) + 1
                severity_levels[severity] = severity_levels.get(severity, 0) + 1
                
                if conf > highest_conf:
                    highest_conf = conf
                    primary_disease = detection.get('class_name', '')
            
            disease_classification = {
                'total_detections': len(detection_results),
                'disease_types': disease_types,
                'severity_levels': severity_levels,
                'highest_confidence': float(highest_conf),
                'primary_disease': primary_disease
            }
            
        else:
            logger.warning("YOLO inference not available")
        
        # Step 3: Calculate health score
        logger.info("Calculating crop health score...")
        health_assessment = health_scorer.calculate_health_score(
            detections=detections,
            sensor_data=sensor_obj,
            plant_species=plant_species
        )
        
        # Step 4: Store prediction for later retrieval
        prediction_data = {
            'prediction_id': prediction_id,
            'timestamp': datetime.now().isoformat(),
            'plant_species': plant_species,
            'location': location,
            'health_assessment': health_assessment,
            'disease_classification': disease_classification,
            'image_filename': image.filename,
            'sensor_data': sensor_obj.__dict__ if sensor_obj else None
        }
        
        predictions_store[prediction_id] = prediction_data
        
        # Step 5: Log prediction (background task)
        background_tasks.add_task(log_prediction, prediction_data)
        
        # Prepare response
        response = HealthAnalysisResponse(
            prediction_id=prediction_id,
            overall_health=float(health_assessment.overall_health),
            risk_level=health_assessment.risk_level,
            confidence=float(health_assessment.confidence),
            detected_issues=[
                {
                    'class_name': d.class_name,
                    'confidence': float(d.confidence),
                    'bbox': tuple(float(x) for x in d.bbox)
                } for d in health_assessment.detected_issues
            ],
            disease_classification=disease_classification,
            environmental_analysis=health_assessment.sensor_analysis,
            recommendations=health_assessment.recommendations,
            timestamp=prediction_data['timestamp']
        )
        
        logger.info(f"Prediction completed: {prediction_id} - Health: {health_assessment.overall_health}%")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Get detailed health analysis
@app.get("/health/{prediction_id}")
async def get_health_analysis(prediction_id: str):
    """Get detailed health analysis for a prediction"""
    if prediction_id not in predictions_store:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    prediction_data = predictions_store[prediction_id]
    health_assessment = prediction_data['health_assessment']
    
    return {
        'prediction_id': prediction_id,
        'timestamp': prediction_data['timestamp'],
        'plant_species': prediction_data.get('plant_species'),
        'location': prediction_data.get('location'),
        'overall_health': float(health_assessment.overall_health),
        'detailed_scores': {
            'disease_score': float(health_assessment.disease_score),
            'pest_score': float(health_assessment.pest_score),
            'environmental_score': float(health_assessment.environmental_score)
        },
        'risk_level': health_assessment.risk_level,
        'confidence': float(health_assessment.confidence),
        'detected_issues': [
            {
                'class_name': d.class_name,
                'confidence': float(d.confidence),
                'bbox': tuple(float(x) for x in d.bbox),
                'severity': getattr(d, 'severity', 'unknown')
            } for d in health_assessment.detected_issues
        ],
        'disease_classification': prediction_data.get('disease_classification', {}),
        'environmental_analysis': health_assessment.sensor_analysis,
        'recommendations': health_assessment.recommendations,
        'sensor_data': prediction_data.get('sensor_data')
    }


# Get farming advice
@app.get("/advice/{prediction_id}")
async def get_farming_advice(prediction_id: str):
    """Get detailed farming advice for a prediction"""
    if prediction_id not in predictions_store:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    prediction_data = predictions_store[prediction_id]
    health_assessment = prediction_data['health_assessment']
    
    # Generate comprehensive advice (this could be enhanced with RAG/LLM)
    advice = {
        'prediction_id': prediction_id,
        'plant_species': prediction_data.get('plant_species'),
        'health_summary': {
            'overall_health': health_assessment.overall_health,
            'risk_level': health_assessment.risk_level,
            'primary_concerns': [d.class_name for d in health_assessment.detected_issues[:3]]
        },
        'immediate_actions': health_assessment.recommendations[:5],
        'long_term_care': [
            "Monitor plant health regularly",
            "Maintain optimal environmental conditions",
            "Follow integrated pest management practices",
            "Ensure proper nutrition and water management"
        ],
        'prevention_tips': [
            "Inspect plants weekly for early signs of problems",
            "Maintain good air circulation",
            "Avoid overhead watering when possible",
            "Practice crop rotation if applicable",
            "Keep growing area clean and free of debris"
        ],
        'next_inspection': "Recommended in 3-7 days" if health_assessment.risk_level in ['high', 'critical'] else "Recommended in 1-2 weeks"
    }
    
    return advice


# List recent predictions
@app.get("/predictions")
async def list_predictions(limit: int = 20):
    """List recent predictions"""
    recent_predictions = sorted(
        predictions_store.values(),
        key=lambda x: x['timestamp'],
        reverse=True
    )[:limit]
    
    return {
        'predictions': [
            {
                'prediction_id': p['prediction_id'],
                'timestamp': p['timestamp'],
                'plant_species': p.get('plant_species'),
                'overall_health': p['health_assessment'].overall_health,
                'risk_level': p['health_assessment'].risk_level,
                'location': p.get('location')
            } for p in recent_predictions
        ],
        'total': len(predictions_store)
    }


# Background task for logging
async def log_prediction(prediction_data: Dict[str, Any]):
    """Log prediction data for analysis and record keeping"""
    try:
        log_entry = {
            'timestamp': prediction_data['timestamp'],
            'prediction_id': prediction_data['prediction_id'],
            'plant_species': prediction_data.get('plant_species'),
            'location': prediction_data.get('location'),
            'health_score': prediction_data['health_assessment'].overall_health,
            'risk_level': prediction_data['health_assessment'].risk_level,
            'detected_issues_count': len(prediction_data['health_assessment'].detected_issues),
            'sensor_data_available': prediction_data.get('sensor_data') is not None
        }
        
        # In production, this would write to a database
        log_file = LOG_FILE.parent / "predictions.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


# Development server runner
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
