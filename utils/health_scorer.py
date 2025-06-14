"""
Crop Health Scoring System
Calculates comprehensive health percentage based on disease detection and environmental factors
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from config import HEALTH_WEIGHTS, OPTIMAL_RANGES

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Structure for disease/pest detection results"""
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, w, h (normalized)
    severity: str = "unknown"  # low, medium, high


@dataclass
class SensorData:
    """Structure for environmental sensor data"""
    temperature: Optional[float] = None  # Celsius
    humidity: Optional[float] = None     # Percentage
    soil_moisture: Optional[float] = None  # Percentage
    ph: Optional[float] = None           # pH scale
    light_intensity: Optional[float] = None  # Lux
    timestamp: Optional[str] = None


@dataclass
class HealthAssessment:
    """Complete health assessment result"""
    overall_health: float  # 0-100%
    disease_score: float   # 0-100%
    pest_score: float      # 0-100%
    environmental_score: float  # 0-100%
    confidence: float      # 0-100%
    risk_level: str        # low, medium, high, critical
    recommendations: List[str]
    detected_issues: List[DetectionResult]
    sensor_analysis: Dict


class HealthScorer:
    """Calculates crop health scores based on multiple factors"""
    
    def __init__(self):
        self.disease_classes = self._load_disease_classes()
        self.pest_classes = self._load_pest_classes()
    
    def calculate_health_score(self, 
                             detections: List[DetectionResult],
                             sensor_data: SensorData,
                             plant_species: str = None) -> HealthAssessment:
        """
        Calculate comprehensive health score
        
        Args:
            detections: List of detected diseases/pests
            sensor_data: Environmental sensor readings
            plant_species: Identified plant species (for species-specific scoring)
            
        Returns:
            Complete health assessment
        """
        try:
            # Separate diseases and pests
            diseases = [d for d in detections if self._is_disease(d.class_name)]
            pests = [d for d in detections if self._is_pest(d.class_name)]
            
            # Calculate individual scores
            disease_score = self._calculate_disease_score(diseases)
            pest_score = self._calculate_pest_score(pests)
            env_score = self._calculate_environmental_score(sensor_data, plant_species)
            
            # Calculate weighted overall health
            weights = HEALTH_WEIGHTS
            overall_health = (
                disease_score * weights['disease_score'] +
                pest_score * weights['pest_score'] +
                env_score * weights['environmental']
            )
            
            # Ensure score is between 0-100
            overall_health = max(0, min(100, overall_health))
            
            # Determine risk level and confidence
            risk_level = self._determine_risk_level(overall_health, detections)
            confidence = self._calculate_confidence(detections, sensor_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                diseases, pests, sensor_data, overall_health, plant_species
            )
            
            # Analyze sensor data
            sensor_analysis = self._analyze_sensor_data(sensor_data)
            
            return HealthAssessment(
                overall_health=round(overall_health, 1),
                disease_score=round(disease_score, 1),
                pest_score=round(pest_score, 1),
                environmental_score=round(env_score, 1),
                confidence=round(confidence, 1),
                risk_level=risk_level,
                recommendations=recommendations,
                detected_issues=detections,
                sensor_analysis=sensor_analysis
            )
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return self._create_error_assessment()
    
    def _calculate_disease_score(self, diseases: List[DetectionResult]) -> float:
        """Calculate disease impact score (100 = healthy, 0 = severely diseased)"""
        if not diseases:
            return 100.0
        
        total_impact = 0.0
        max_impact = 0.0
        
        for disease in diseases:
            # Base impact based on confidence
            impact = disease.confidence * 100
            
            # Adjust for disease severity
            severity_multiplier = self._get_severity_multiplier(disease.class_name)
            impact *= severity_multiplier
            
            # Adjust for detection area (larger detections = more impact)
            area = disease.bbox[2] * disease.bbox[3]  # width * height
            area_multiplier = min(2.0, 1.0 + area)  # Cap at 2x multiplier
            impact *= area_multiplier
            
            total_impact += impact
            max_impact = max(max_impact, impact)
        
        # Use combination of average and maximum impact
        avg_impact = total_impact / len(diseases)
        combined_impact = (avg_impact + max_impact) / 2
        
        # Convert to health score (inverse)
        health_score = max(0, 100 - min(100, combined_impact))
        
        return health_score
    
    def _calculate_pest_score(self, pests: List[DetectionResult]) -> float:
        """Calculate pest impact score (100 = no pests, 0 = severe infestation)"""
        if not pests:
            return 100.0
        
        total_impact = 0.0
        
        for pest in pests:
            # Base impact
            impact = pest.confidence * 100
            
            # Pest-specific severity
            severity_multiplier = self._get_pest_severity(pest.class_name)
            impact *= severity_multiplier
            
            # Area impact
            area = pest.bbox[2] * pest.bbox[3]
            area_multiplier = min(1.5, 1.0 + area * 0.5)
            impact *= area_multiplier
            
            total_impact += impact
        
        # Average impact
        avg_impact = total_impact / len(pests)
        
        # Convert to health score
        health_score = max(0, 100 - min(100, avg_impact))
        
        return health_score
    
    def _calculate_environmental_score(self, 
                                     sensor_data: SensorData, 
                                     plant_species: str = None) -> float:
        """Calculate environmental conditions score"""
        if not sensor_data:
            return 50.0  # Neutral score if no data
        
        scores = []
        
        # Temperature score
        if sensor_data.temperature is not None:
            temp_score = self._score_parameter(
                sensor_data.temperature, 
                OPTIMAL_RANGES['temperature']
            )
            scores.append(temp_score)
        
        # Humidity score
        if sensor_data.humidity is not None:
            humidity_score = self._score_parameter(
                sensor_data.humidity,
                OPTIMAL_RANGES['humidity']
            )
            scores.append(humidity_score)
        
        # Soil moisture score
        if sensor_data.soil_moisture is not None:
            moisture_score = self._score_parameter(
                sensor_data.soil_moisture,
                OPTIMAL_RANGES['soil_moisture']
            )
            scores.append(moisture_score)
        
        # pH score
        if sensor_data.ph is not None:
            ph_score = self._score_parameter(
                sensor_data.ph,
                OPTIMAL_RANGES['ph']
            )
            scores.append(ph_score)
        
        # Return average of available scores
        return np.mean(scores) if scores else 50.0
    
    def _score_parameter(self, value: float, optimal_range: Tuple[float, float]) -> float:
        """Score a parameter based on optimal range"""
        min_val, max_val = optimal_range
        
        if min_val <= value <= max_val:
            return 100.0  # Perfect score
        elif value < min_val:
            # Below optimal - score decreases linearly
            distance = min_val - value
            max_distance = min_val * 0.5  # 50% below minimum = 0 score
            score = max(0, 100 - (distance / max_distance) * 100)
        else:
            # Above optimal - score decreases linearly
            distance = value - max_val
            max_distance = max_val * 0.5  # 50% above maximum = 0 score
            score = max(0, 100 - (distance / max_distance) * 100)
        
        return score
    
    def _determine_risk_level(self, health_score: float, detections: List[DetectionResult]) -> str:
        """Determine risk level based on health score and detections"""
        if health_score >= 80:
            return "low"
        elif health_score >= 60:
            return "medium" 
        elif health_score >= 30:
            return "high"
        else:
            return "critical"
    
    def _calculate_confidence(self, detections: List[DetectionResult], sensor_data: SensorData) -> float:
        """Calculate confidence in the health assessment"""
        confidence_factors = []
        
        # Detection confidence
        if detections:
            avg_detection_confidence = np.mean([d.confidence for d in detections]) * 100
            confidence_factors.append(avg_detection_confidence)
        else:
            confidence_factors.append(90.0)  # High confidence if no detections
        
        # Sensor data availability
        sensor_availability = 0
        total_sensors = 4  # temp, humidity, soil_moisture, ph
        
        if sensor_data:
            if sensor_data.temperature is not None: sensor_availability += 1
            if sensor_data.humidity is not None: sensor_availability += 1  
            if sensor_data.soil_moisture is not None: sensor_availability += 1
            if sensor_data.ph is not None: sensor_availability += 1
        
        sensor_confidence = (sensor_availability / total_sensors) * 100
        confidence_factors.append(sensor_confidence)
        
        return np.mean(confidence_factors)
    
    def _generate_recommendations(self, 
                                diseases: List[DetectionResult],
                                pests: List[DetectionResult],
                                sensor_data: SensorData,
                                health_score: float,
                                plant_species: str = None) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Disease-specific recommendations
        if diseases:
            disease_names = [d.class_name for d in diseases]
            if any('blight' in name.lower() for name in disease_names):
                recommendations.append("Apply fungicide treatment for blight control")
                recommendations.append("Improve air circulation around plants")
            if any('spot' in name.lower() for name in disease_names):
                recommendations.append("Remove affected leaves and destroy them")
                recommendations.append("Avoid overhead watering")
            if any('rust' in name.lower() for name in disease_names):
                recommendations.append("Apply copper-based fungicide")
                recommendations.append("Ensure good drainage")
        
        # Pest-specific recommendations
        if pests:
            pest_names = [p.class_name for p in pests]
            if any('mite' in name.lower() for name in pest_names):
                recommendations.append("Increase humidity around plants")
                recommendations.append("Apply miticide or insecticidal soap")
            if any('aphid' in name.lower() for name in pest_names):
                recommendations.append("Use beneficial insects like ladybugs")
                recommendations.append("Apply neem oil treatment")
        
        # Environmental recommendations
        if sensor_data:
            if sensor_data.temperature is not None:
                temp_range = OPTIMAL_RANGES['temperature']
                if sensor_data.temperature < temp_range[0]:
                    recommendations.append("Increase temperature - consider protective covering")
                elif sensor_data.temperature > temp_range[1]:
                    recommendations.append("Provide shade or cooling - temperature too high")
            
            if sensor_data.humidity is not None:
                humidity_range = OPTIMAL_RANGES['humidity']
                if sensor_data.humidity < humidity_range[0]:
                    recommendations.append("Increase humidity around plants")
                elif sensor_data.humidity > humidity_range[1]:
                    recommendations.append("Improve ventilation - humidity too high")
            
            if sensor_data.soil_moisture is not None:
                moisture_range = OPTIMAL_RANGES['soil_moisture']
                if sensor_data.soil_moisture < moisture_range[0]:
                    recommendations.append("Increase watering frequency")
                elif sensor_data.soil_moisture > moisture_range[1]:
                    recommendations.append("Reduce watering - soil too wet")
        
        # General health recommendations
        if health_score < 50:
            recommendations.append("Schedule immediate inspection by agricultural expert")
            recommendations.append("Consider soil testing for nutrient deficiencies")
        elif health_score < 70:
            recommendations.append("Monitor closely and take preventive measures")
            recommendations.append("Ensure proper nutrition and water management")
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    def _analyze_sensor_data(self, sensor_data: SensorData) -> Dict:
        """Analyze sensor data and provide detailed breakdown"""
        analysis = {
            'temperature': {'value': None, 'status': 'unknown', 'score': None},
            'humidity': {'value': None, 'status': 'unknown', 'score': None},
            'soil_moisture': {'value': None, 'status': 'unknown', 'score': None},
            'ph': {'value': None, 'status': 'unknown', 'score': None}
        }
        
        if not sensor_data:
            return analysis
        
        # Analyze each parameter
        for param, optimal_range in OPTIMAL_RANGES.items():
            value = getattr(sensor_data, param, None)
            if value is not None:
                score = self._score_parameter(value, optimal_range)
                
                if score >= 80:
                    status = 'optimal'
                elif score >= 60:
                    status = 'acceptable'
                elif score >= 40:
                    status = 'suboptimal'
                else:
                    status = 'poor'
                
                analysis[param] = {
                    'value': value,
                    'status': status,
                    'score': round(score, 1),
                    'optimal_range': optimal_range
                }
        
        return analysis
    
    def _load_disease_classes(self) -> List[str]:
        """Load known disease class names"""
        return [
            'bacterial spot', 'bacterial blight', 'bacterial wilt',
            'early blight', 'late blight', 'leaf blight',
            'black rot', 'brown rot', 'soft rot',
            'rust', 'leaf rust', 'stem rust',
            'powdery mildew', 'downy mildew',
            'anthracnose', 'scab', 'canker',
            'mosaic virus', 'curl virus', 'yellow virus',
            'leaf spot', 'target spot', 'septoria leaf spot',
            'leaf mold', 'gray mold', 'white mold'
        ]
    
    def _load_pest_classes(self) -> List[str]:
        """Load known pest class names"""
        return [
            'spider mite', 'aphid', 'whitefly',
            'thrips', 'scale', 'mealybug',
            'caterpillar', 'cutworm', 'army worm',
            'leaf miner', 'borer', 'weevil'
        ]
    
    def _is_disease(self, class_name: str) -> bool:
        """Check if detection is a disease"""
        class_lower = class_name.lower()
        return any(disease in class_lower for disease in self.disease_classes)
    
    def _is_pest(self, class_name: str) -> bool:
        """Check if detection is a pest"""
        class_lower = class_name.lower()
        return any(pest in class_lower for pest in self.pest_classes)
    
    def _get_severity_multiplier(self, disease_name: str) -> float:
        """Get severity multiplier for specific diseases"""
        disease_lower = disease_name.lower()
        
        # High severity diseases
        if any(term in disease_lower for term in ['blight', 'wilt', 'rot', 'canker']):
            return 1.5
        # Medium severity
        elif any(term in disease_lower for term in ['rust', 'mildew', 'virus']):
            return 1.2
        # Lower severity
        else:
            return 1.0
    
    def _get_pest_severity(self, pest_name: str) -> float:
        """Get severity multiplier for specific pests"""
        pest_lower = pest_name.lower()
        
        # High damage pests
        if any(term in pest_lower for term in ['borer', 'cutworm', 'army worm']):
            return 1.4
        # Medium damage
        elif any(term in pest_lower for term in ['aphid', 'thrips', 'whitefly']):
            return 1.2
        # Lower damage
        else:
            return 1.0
    
    def _create_error_assessment(self) -> HealthAssessment:
        """Create a default assessment for error cases"""
        return HealthAssessment(
            overall_health=50.0,
            disease_score=50.0,
            pest_score=50.0,
            environmental_score=50.0,
            confidence=0.0,
            risk_level="unknown",
            recommendations=["Unable to assess health - please check inputs"],
            detected_issues=[],
            sensor_analysis={}
        )
