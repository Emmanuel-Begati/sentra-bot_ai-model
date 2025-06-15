"""
Simplified Crop Health Scoring System
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from config import OPTIMAL_RANGES, DISEASE_IMPACT_WEIGHTS
from .disease_classifier import DiseaseClassifier

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    class_name: str
    confidence: float
    bbox: tuple
    area: float = 0.0


@dataclass
class SensorData:
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    soil_moisture: Optional[float] = None
    ph: Optional[float] = None


@dataclass
class HealthAssessment:
    overall_health: float
    disease_score: float
    environmental_score: float
    risk_level: str
    confidence: float
    recommendations: List[str]
    detected_issues: List[Dict]
    sensor_analysis: Dict


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


class HealthScorer:
    """Simplified health scorer using disease classifier"""
    
    def __init__(self):
        self.disease_classifier = DiseaseClassifier()
    
    def calculate_health_score(self, 
                             detections: List[DetectionResult],
                             sensor_data: Optional[SensorData] = None) -> HealthAssessment:
        """Calculate comprehensive health score"""
        
        # Analyze detections
        disease_analysis = self._analyze_detections(detections)
        
        # Calculate scores
        disease_score = self._calculate_disease_score(disease_analysis)
        env_score = self._calculate_environmental_score(sensor_data)
        
        # Calculate overall health (weighted average)
        disease_weight = 0.7
        env_weight = 0.3
        
        overall_health = (disease_score * disease_weight + env_score * env_weight)
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_health, disease_analysis)
        
        # Calculate confidence
        confidence = self._calculate_confidence(detections, sensor_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(disease_analysis, sensor_data)
        
        # Analyze sensor data
        sensor_analysis = self._analyze_sensor_data(sensor_data)
        
        # Convert all numpy types to Python types
        classified_detections = convert_numpy_types(disease_analysis['classified_detections'])
        sensor_analysis = convert_numpy_types(sensor_analysis)
        
        return HealthAssessment(
            overall_health=float(overall_health),
            disease_score=float(disease_score),
            environmental_score=float(env_score),
            risk_level=risk_level,
            confidence=float(confidence),
            recommendations=recommendations,
            detected_issues=classified_detections,
            sensor_analysis=sensor_analysis
        )
    
    def _analyze_detections(self, detections: List[DetectionResult]) -> Dict:
        """Analyze detections using disease classifier"""
        classified_detections = []
        total_impact = 0.0
        max_priority = 0
        total_area = 0.0
        
        for detection in detections:
            classification = self.disease_classifier.classify_detection(
                detection.class_name, 
                detection.confidence
            )
            
            classification['area'] = float(detection.area)
            classified_detections.append(classification)
            
            total_impact += classification['weighted_impact']
            max_priority = max(max_priority, classification['treatment_priority'])
            total_area += detection.area
        
        return {
            'classified_detections': classified_detections,
            'total_impact': float(total_impact),
            'max_priority': int(max_priority),
            'affected_area': float(total_area),
            'detection_count': len(detections)
        }
    
    def _calculate_disease_score(self, disease_analysis: Dict) -> float:
        """Calculate disease score from analysis"""
        if disease_analysis['detection_count'] == 0:
            return 100.0
        
        # Base impact from diseases
        base_impact = disease_analysis['total_impact'] / disease_analysis['detection_count']
        
        # Area penalty
        area_penalty = min(30, disease_analysis['affected_area'] * 100)
        
        # Priority penalty
        priority_penalty = disease_analysis['max_priority'] * 5
        
        total_penalty = base_impact + area_penalty + priority_penalty
        disease_score = max(0, 100 - total_penalty)
        
        return float(disease_score)
    
    def _calculate_environmental_score(self, sensor_data: Optional[SensorData]) -> float:
        """Calculate environmental score"""
        if not sensor_data:
            return 75.0  # Default score when no sensor data
        
        scores = []
        
        for param, optimal_range in OPTIMAL_RANGES.items():
            value = getattr(sensor_data, param, None)
            if value is not None:
                score = self._score_parameter(value, optimal_range)
                scores.append(score)
        
        return float(np.mean(scores)) if scores else 75.0
    
    def _score_parameter(self, value: float, optimal_range: tuple) -> float:
        """Score individual parameter"""
        min_val, max_val = optimal_range
        range_size = max_val - min_val
        
        if min_val <= value <= max_val:
            return 100.0
        elif value < min_val:
            deviation = (min_val - value) / range_size
            return max(0, 100 - deviation * 100)
        else:
            deviation = (value - max_val) / range_size
            return max(0, 100 - deviation * 100)
    
    def _determine_risk_level(self, health_score: float, disease_analysis: Dict) -> str:
        """Determine risk level"""
        if health_score >= 80:
            risk = "low"
        elif health_score >= 60:
            risk = "medium"
        elif health_score >= 30:
            risk = "high"
        else:
            risk = "critical"
        
        # Upgrade risk for urgent conditions
        if disease_analysis['max_priority'] >= 4:
            if risk == "low":
                risk = "medium"
            elif risk == "medium":
                risk = "high"
        
        return risk
    
    def _calculate_confidence(self, detections: List[DetectionResult], 
                            sensor_data: Optional[SensorData]) -> float:
        """Calculate confidence in assessment"""
        confidence_factors = []
        
        # Detection confidence
        if detections:
            avg_confidence = np.mean([d.confidence for d in detections]) * 100
            confidence_factors.append(avg_confidence)
        else:
            confidence_factors.append(90.0)  # High confidence if no issues detected
        
        # Sensor data availability
        if sensor_data:
            available_sensors = sum(1 for param in ['temperature', 'humidity', 'soil_moisture', 'ph']
                                  if getattr(sensor_data, param, None) is not None)
            sensor_confidence = (available_sensors / 4) * 100
            confidence_factors.append(sensor_confidence)
        else:
            confidence_factors.append(50.0)  # Moderate confidence without sensors
        
        return float(np.mean(confidence_factors))
    
    def _generate_recommendations(self, disease_analysis: Dict, 
                                sensor_data: Optional[SensorData]) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        # Disease-specific recommendations
        for detection in disease_analysis['classified_detections'][:3]:  # Top 3 issues
            disease_recs = self.disease_classifier.get_treatment_recommendations(
                detection['class_name']
            )
            recommendations.extend(disease_recs[:2])  # Top 2 per disease
        
        # Environmental recommendations
        if sensor_data:
            env_recs = self._get_environmental_recommendations(sensor_data)
            recommendations.extend(env_recs)
        
        # General recommendations
        if disease_analysis['max_priority'] >= 3:
            recommendations.append("Schedule follow-up inspection within 3-5 days")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)
        
        return unique_recs[:8]  # Limit to 8 recommendations
    
    def _get_environmental_recommendations(self, sensor_data: SensorData) -> List[str]:
        """Get environmental recommendations"""
        recommendations = []
        
        if sensor_data.temperature is not None:
            temp_range = OPTIMAL_RANGES['temperature']
            if sensor_data.temperature < temp_range[0]:
                recommendations.append("Provide protection from cold temperatures")
            elif sensor_data.temperature > temp_range[1]:
                recommendations.append("Provide shade or cooling measures")
        
        if sensor_data.humidity is not None:
            humidity_range = OPTIMAL_RANGES['humidity']
            if sensor_data.humidity < humidity_range[0]:
                recommendations.append("Increase humidity around plants")
            elif sensor_data.humidity > humidity_range[1]:
                recommendations.append("Improve ventilation to reduce humidity")
        
        if sensor_data.soil_moisture is not None:
            moisture_range = OPTIMAL_RANGES['soil_moisture']
            if sensor_data.soil_moisture < moisture_range[0]:
                recommendations.append("Increase watering frequency")
            elif sensor_data.soil_moisture > moisture_range[1]:
                recommendations.append("Reduce watering to prevent root rot")
        
        return recommendations
    
    def _analyze_sensor_data(self, sensor_data: Optional[SensorData]) -> Dict:
        """Analyze sensor data for response"""
        analysis = {}
        
        if not sensor_data:
            return analysis
        
        for param, optimal_range in OPTIMAL_RANGES.items():
            value = getattr(sensor_data, param, None)
            if value is not None:
                score = self._score_parameter(value, optimal_range)
                
                if score >= 80:
                    status = 'optimal'
                elif score >= 60:
                    status = 'acceptable'
                else:
                    status = 'poor'
                
                analysis[param] = {
                    'value': float(value),
                    'status': status,
                    'score': float(score),
                    'optimal_range': optimal_range
                }
        
        return analysis
