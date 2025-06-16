"""
Simplified FastAPI Backend for Crop Health Monitoring
"""

import logging
from datetime import datetime
from typing import List, Optional
import json
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from PIL import Image
import io

from utils.yolo_inference import YOLOInference
from utils.health_scorer import HealthScorer, DetectionResult, SensorData
from utils.rag_agent import SimpleRAGAgent
from utils.plantnet_api import PlantNetAPI
from config import API_HOST, API_PORT, DATASET_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Crop Health Monitoring API",
    description="AI-powered crop health analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
yolo_inference = None
health_scorer = HealthScorer()
rag_agent = SimpleRAGAgent()
plantnet_api = PlantNetAPI()

# In-memory storage
predictions_store = {}


# Pydantic models
class SensorDataModel(BaseModel):
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    soil_moisture: Optional[float] = None
    ph: Optional[float] = None


class PlantIdentificationModel(BaseModel):
    scientific_name: str
    common_names: List[str]
    primary_common_name: str
    family: str
    genus: str
    score: float
    confidence: str


class HealthResponse(BaseModel):
    prediction_id: str
    overall_health: float
    disease_score: float
    environmental_score: float
    risk_level: str
    confidence: float
    detected_diseases: List[dict]
    recommendations: List[str]
    plant_identification: Optional[PlantIdentificationModel] = None
    timestamp: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services"""
    global yolo_inference
    
    try:
        yolo_inference = YOLOInference()
        logger.info("YOLO inference initialized")
        
        # Log dataset information
        data_yaml = DATASET_DIR / "data.yaml"
        
        if data_yaml.exists():
            logger.info("✅ Dataset available")
            try:
                import yaml
                with open(data_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                logger.info(f"📊 Model trained on {data.get('nc', 'unknown')} classes")
            except Exception as e:
                logger.warning(f"Could not read dataset info: {e}")
        else:
            logger.warning("❌ No dataset configuration found")
            
    except Exception as e:
        logger.error(f"Failed to initialize YOLO: {e}")
        yolo_inference = None
    
    # Test PlantNet API
    if plantnet_api.api_key:
        logger.info("PlantNet API initialized")
    else:
        logger.warning("PlantNet API key not configured")


# Health check
@app.get("/health")
async def health_check():
    """API health check"""
    # Check dataset status
    dataset_info = {
        "available": (DATASET_DIR / "data.yaml").exists(),
        "type": "standard"
    }
    
    if dataset_info["available"]:
        try:
            import yaml
            with open(DATASET_DIR / "data.yaml", 'r') as f:
                data = yaml.safe_load(f)
            dataset_info["classes"] = data.get('nc', 0)
        except:
            pass
    
    return {
        "status": "healthy",
        "services": {
            "yolo_inference": yolo_inference is not None,
            "health_scorer": True,
            "rag_agent": True,
            "plantnet_api": plantnet_api.api_key is not None
        },
        "dataset": dataset_info,
        "timestamp": datetime.now().isoformat()
    }


# Main prediction endpoint
@app.post("/predict", response_model=HealthResponse)
async def predict_crop_health(
    image: UploadFile = File(...),
    sensor_data: Optional[str] = None,
    include_plant_id: bool = True
):
    """Analyze crop health from image and sensor data"""
    try:
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
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
        
        # Run disease detection
        detections = []
        if yolo_inference:
            detection_results = yolo_inference.predict(pil_image)
            
            for detection in detection_results:
                det_obj = DetectionResult(
                    class_name=detection['class_name'],
                    confidence=detection['confidence'],
                    bbox=detection['bbox'],
                    area=detection['area']
                )
                detections.append(det_obj)
        
        # Run plant identification
        plant_identification = None
        if include_plant_id and plantnet_api.api_key:
            try:
                logger.info("Starting plant identification...")
                plant_result = plantnet_api.identify_plant(image_data, organs=["auto"])  # Changed from ["leaf"] to ["auto"]
                
                if plant_result and plant_result.get('success', False) and plant_result.get('best_match'):
                    best_match = plant_result['best_match']
                    plant_identification = PlantIdentificationModel(
                        scientific_name=best_match['scientific_name'],
                        common_names=best_match['common_names'],
                        primary_common_name=best_match['primary_common_name'],
                        family=best_match['family'],
                        genus=best_match['genus'],
                        score=best_match['score'],
                        confidence=best_match['confidence']
                    )
                    logger.info(f"Plant identified: {best_match['primary_common_name']} (confidence: {best_match['confidence']})")
                else:
                    logger.warning("Plant identification returned no valid results")
                    
            except Exception as e:
                logger.warning(f"Plant identification failed: {e}")

        
        # Calculate health score
        health_assessment = health_scorer.calculate_health_score(
            detections=detections,
            sensor_data=sensor_obj
        )
        
        # Store prediction
        prediction_data = {
            'prediction_id': prediction_id,
            'timestamp': datetime.now().isoformat(),
            'health_assessment': health_assessment,
            'plant_identification': plant_identification,
            'image_filename': image.filename
        }
        predictions_store[prediction_id] = prediction_data
        
        # Prepare response
        response = HealthResponse(
            prediction_id=prediction_id,
            overall_health=health_assessment.overall_health,
            disease_score=health_assessment.disease_score,
            environmental_score=health_assessment.environmental_score,
            risk_level=health_assessment.risk_level,
            confidence=health_assessment.confidence,
            detected_diseases=health_assessment.detected_issues,
            recommendations=health_assessment.recommendations,
            plant_identification=plant_identification,
            timestamp=prediction_data['timestamp']
        )
        
        logger.info(f"Prediction completed: {prediction_id} - Health: {health_assessment.overall_health}%")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# New endpoint for plant identification only
@app.post("/identify-plant")
async def identify_plant_species(
    image: UploadFile = File(...),
    organs: Optional[str] = "auto",  # Changed from "leaf" to "auto"
    nb_results: int = 5
):
    """Identify plant species from image"""
    try:
        if not plantnet_api.api_key:
            raise HTTPException(status_code=503, detail="PlantNet API not configured")
        
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_data = await image.read()
        
        # Parse organs - handle 'auto' as single organ
        if organs == "auto":
            organ_list = ["auto"]
        else:
            organ_list = [organ.strip() for organ in organs.split(",")]
        
        logger.info(f"Identifying plant with organs: {organ_list}")
        
        # Identify plant
        result = plantnet_api.identify_plant(
            image_data, 
            organs=organ_list,
            nb_results=nb_results
        )
        
        if not result:
            # Return a structured error response instead of raising exception
            return {
                'success': False,
                'error': 'Plant identification service unavailable',
                'species_count': 0,
                'best_match': None,
                'all_results': [],
                'query_info': {
                    'error': 'API service unavailable'
                }
            }
        
        return {
            'success': result.get('success', False),
            'species_count': result['species_count'],
            'best_match': result['best_match'],
            'all_results': result['all_results'][:nb_results],
            'query_info': result['query_info']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plant identification error: {e}")
        return {
            'success': False,
            'error': f"Identification failed: {str(e)}",
            'species_count': 0,
            'best_match': None,
            'all_results': [],
            'query_info': {
                'error': str(e)
            }
        }


# Get detailed advice
@app.get("/advice/{prediction_id}")
async def get_farming_advice(prediction_id: str):
    """Get detailed farming advice for a prediction"""
    if prediction_id not in predictions_store:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    prediction_data = predictions_store[prediction_id]
    health_assessment = prediction_data['health_assessment']
    plant_identification = prediction_data.get('plant_identification')
    
    advice = {
        'prediction_id': prediction_id,
        'overall_health': health_assessment.overall_health,
        'risk_level': health_assessment.risk_level,
        'immediate_actions': health_assessment.recommendations,
        'general_care': [
            "Monitor plants regularly",
            "Maintain proper watering schedule",
            "Ensure good air circulation",
            "Practice crop rotation when possible"
        ]
    }
    
    # Add plant-specific advice
    if plant_identification:
        advice['plant_species'] = {
            'scientific_name': plant_identification.scientific_name,
            'common_name': plant_identification.primary_common_name,
            'family': plant_identification.family,
            'confidence': plant_identification.confidence
        }
    
    # Get enhanced advice for primary disease
    if health_assessment.detected_issues:
        primary_disease = health_assessment.detected_issues[0]['class_name']
        enhanced_advice = rag_agent.get_disease_advice(primary_disease)
        
        advice['enhanced_advice'] = {
            'disease': enhanced_advice['disease_name'],
            'detailed_summary': enhanced_advice['summary'],
            'confidence': enhanced_advice['confidence']
        }
    
    return advice


# Disease-specific advice
@app.get("/disease-advice/{disease_name}")
async def get_disease_advice(disease_name: str):
    """Get advice for specific disease"""
    try:
        advice = rag_agent.get_disease_advice(disease_name)
        return advice
    except Exception as e:
        logger.error(f"Disease advice error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get advice")


# List recent predictions
@app.get("/predictions")
async def list_predictions(limit: int = 20):
    """List recent predictions"""
    recent = sorted(
        predictions_store.values(),
        key=lambda x: x['timestamp'],
        reverse=True
    )[:limit]
    
    return {
        'predictions': [
            {
                'prediction_id': p['prediction_id'],
                'timestamp': p['timestamp'],
                'overall_health': p['health_assessment'].overall_health,
                'risk_level': p['health_assessment'].risk_level,
                'plant_species': p.get('plant_identification').primary_common_name if p.get('plant_identification') else None
            } for p in recent
        ]
    }


# Test PlantNet API configuration
@app.get("/test-plantnet")
async def test_plantnet_api():
    """Test PlantNet API configuration"""
    from utils.plantnet_api import test_plantnet_integration
    
    if not plantnet_api.api_key:
        return {
            "status": "error",
            "message": "PlantNet API key not configured",
            "api_key_configured": False
        }
    
    return {
        "status": "testing",
        "message": "Check server logs for detailed test results",
        "api_key_configured": True,
        "api_key_preview": plantnet_api.api_key[:10] + "...",
        "project": plantnet_api.project,
        "base_url": plantnet_api.base_url
    }


# Add dataset information endpoint
@app.get("/dataset-info")
async def get_dataset_info():
    """Get information about the current dataset"""
    try:
        data_yaml = DATASET_DIR / "data.yaml"
        
        info = {
            "available": data_yaml.exists(),
            "active_dataset": "standard" if data_yaml.exists() else "none"
        }
        
        # Get information about dataset
        if data_yaml.exists():
            try:
                import yaml
                with open(data_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                
                info["dataset"] = {
                    "classes": data.get('nc', 0),
                    "class_names": data.get('names', [])[:20],  # First 20 classes
                    "total_class_names": len(data.get('names', []))
                }
            except Exception as e:
                logger.error(f"Error reading dataset: {e}")
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dataset information")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
