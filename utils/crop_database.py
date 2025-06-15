"""
Comprehensive Crop Database
Maps diseases, pests, and crop-specific information for robust health assessment
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import yaml
import logging

logger = logging.getLogger(__name__)


class DiseaseType(Enum):
    FUNGAL = "fungal"
    BACTERIAL = "bacterial"
    VIRAL = "viral"
    PEST = "pest"
    PHYSIOLOGICAL = "physiological"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DiseaseInfo:
    """Comprehensive disease information"""
    name: str
    type: DiseaseType
    severity: SeverityLevel
    impact_score: float  # 0-100, higher = more damaging
    affected_crops: List[str]
    symptoms: List[str]
    treatment_priority: int  # 1-5, 1 = immediate treatment needed
    spreading_rate: str  # slow, medium, fast
    environmental_factors: List[str]


@dataclass
class CropInfo:
    """Crop-specific information"""
    name: str
    scientific_name: str
    common_diseases: List[str]
    growth_stage_sensitive: List[str]  # stages most vulnerable
    optimal_temp_range: Tuple[float, float]
    optimal_humidity_range: Tuple[float, float]
    optimal_ph_range: Tuple[float, float]
    optimal_moisture_range: Tuple[float, float]
    resistance_level: Dict[str, float]  # disease -> resistance (0-1)


class CropDatabase:
    """Comprehensive database of crops, diseases, and their relationships"""
    
    def __init__(self):
        self.diseases = self._initialize_disease_database()
        self.crops = self._initialize_crop_database()
        self.class_mappings = self._load_class_mappings()
    
    def _initialize_disease_database(self) -> Dict[str, DiseaseInfo]:
        """Initialize comprehensive disease database"""
        diseases = {}
        
        # Fungal diseases
        fungal_diseases = [
            ("anthracnose", ["tomato", "bean", "pepper", "fruit trees"], 85, 4, "fast"),
            ("blight", ["potato", "tomato", "rice", "wheat"], 90, 5, "fast"),
            ("rust", ["wheat", "corn", "coffee", "beans"], 75, 3, "medium"),
            ("powdery mildew", ["grape", "cucumber", "pepper"], 60, 2, "medium"),
            ("downy mildew", ["grape", "lettuce", "cabbage"], 70, 3, "fast"),
            ("black rot", ["apple", "grape", "cabbage"], 80, 4, "medium"),
            ("gray mold", ["strawberry", "grape", "raspberry"], 65, 3, "fast"),
            ("leaf spot", ["corn", "tomato", "rice"], 55, 2, "slow"),
            ("scab", ["apple", "peach", "potato"], 70, 3, "medium"),
            ("smut", ["corn", "wheat", "sorghum"], 85, 4, "medium")
        ]
        
        for disease, crops, impact, priority, spread in fungal_diseases:
            diseases[disease] = DiseaseInfo(
                name=disease,
                type=DiseaseType.FUNGAL,
                severity=self._get_severity_from_impact(impact),
                impact_score=impact,
                affected_crops=crops,
                symptoms=[f"{disease} symptoms", "leaf discoloration", "reduced yield"],
                treatment_priority=priority,
                spreading_rate=spread,
                environmental_factors=["high humidity", "moderate temperature"]
            )
        
        # Bacterial diseases
        bacterial_diseases = [
            ("bacterial spot", ["tomato", "pepper", "eggplant"], 80, 4, "fast"),
            ("bacterial blight", ["rice", "cassava", "bean"], 85, 5, "fast"),
            ("bacterial wilt", ["cucumber", "tomato", "potato"], 95, 5, "fast"),
            ("canker", ["citrus", "stone fruits"], 90, 5, "medium"),
            ("soft rot", ["potato", "carrot", "cauliflower"], 75, 4, "fast")
        ]
        
        for disease, crops, impact, priority, spread in bacterial_diseases:
            diseases[disease] = DiseaseInfo(
                name=disease,
                type=DiseaseType.BACTERIAL,
                severity=self._get_severity_from_impact(impact),
                impact_score=impact,
                affected_crops=crops,
                symptoms=[f"{disease} symptoms", "water-soaked lesions", "wilting"],
                treatment_priority=priority,
                spreading_rate=spread,
                environmental_factors=["high humidity", "warm temperature", "poor sanitation"]
            )
        
        # Viral diseases
        viral_diseases = [
            ("mosaic virus", ["tomato", "tobacco", "cucumber"], 70, 3, "medium"),
            ("curl virus", ["tomato", "cotton", "okra"], 75, 4, "medium"),
            ("yellow virus", ["sugar beet", "potato"], 80, 4, "fast"),
            ("streak disease", ["cassava", "banana"], 85, 4, "fast")
        ]
        
        for disease, crops, impact, priority, spread in viral_diseases:
            diseases[disease] = DiseaseInfo(
                name=disease,
                type=DiseaseType.VIRAL,
                severity=self._get_severity_from_impact(impact),
                impact_score=impact,
                affected_crops=crops,
                symptoms=[f"{disease} symptoms", "stunted growth", "yield reduction"],
                treatment_priority=priority,
                spreading_rate=spread,
                environmental_factors=["vector presence", "warm weather"]
            )
        
        # Pests
        pests = [
            ("spider mite", ["various crops"], 60, 2, "fast"),
            ("aphid", ["various crops"], 50, 2, "fast"),
            ("leaf miner", ["leafy vegetables"], 40, 1, "medium"),
            ("whitefly", ["tomato", "pepper", "eggplant"], 55, 2, "fast")
        ]
        
        for pest, crops, impact, priority, spread in pests:
            diseases[pest] = DiseaseInfo(
                name=pest,
                type=DiseaseType.PEST,
                severity=self._get_severity_from_impact(impact),
                impact_score=impact,
                affected_crops=crops,
                symptoms=[f"{pest} damage", "feeding damage", "reduced vigor"],
                treatment_priority=priority,
                spreading_rate=spread,
                environmental_factors=["warm weather", "dry conditions"]
            )
        
        return diseases
    
    def _initialize_crop_database(self) -> Dict[str, CropInfo]:
        """Initialize crop-specific database"""
        crops = {
            "tomato": CropInfo(
                name="tomato",
                scientific_name="Solanum lycopersicum",
                common_diseases=["blight", "bacterial spot", "mosaic virus", "leaf mold"],
                growth_stage_sensitive=["flowering", "fruit development"],
                optimal_temp_range=(18, 25),
                optimal_humidity_range=(50, 70),
                optimal_ph_range=(6.0, 6.8),
                optimal_moisture_range=(60, 80),
                resistance_level={"blight": 0.3, "bacterial spot": 0.4}
            ),
            "potato": CropInfo(
                name="potato",
                scientific_name="Solanum tuberosum",
                common_diseases=["late blight", "early blight", "bacterial wilt", "scab"],
                growth_stage_sensitive=["tuber formation", "bulking"],
                optimal_temp_range=(15, 20),
                optimal_humidity_range=(60, 80),
                optimal_ph_range=(5.0, 6.5),
                optimal_moisture_range=(70, 85),
                resistance_level={"late blight": 0.2, "early blight": 0.5}
            ),
            "rice": CropInfo(
                name="rice",
                scientific_name="Oryza sativa",
                common_diseases=["blast", "bacterial blight", "sheath blight"],
                growth_stage_sensitive=["tillering", "heading"],
                optimal_temp_range=(25, 35),
                optimal_humidity_range=(70, 90),
                optimal_ph_range=(5.5, 6.5),
                optimal_moisture_range=(80, 100),
                resistance_level={"blast": 0.4, "bacterial blight": 0.6}
            ),
            "corn": CropInfo(
                name="corn",
                scientific_name="Zea mays",
                common_diseases=["rust", "smut", "leaf blight", "gray leaf spot"],
                growth_stage_sensitive=["silking", "grain filling"],
                optimal_temp_range=(20, 30),
                optimal_humidity_range=(50, 70),
                optimal_ph_range=(6.0, 7.0),
                optimal_moisture_range=(65, 80),
                resistance_level={"rust": 0.5, "smut": 0.3}
            ),
            "wheat": CropInfo(
                name="wheat",
                scientific_name="Triticum aestivum",
                common_diseases=["rust", "powdery mildew", "head scab", "septoria blotch"],
                growth_stage_sensitive=["heading", "grain filling"],
                optimal_temp_range=(15, 25),
                optimal_humidity_range=(40, 60),
                optimal_ph_range=(6.0, 7.5),
                optimal_moisture_range=(50, 70),
                resistance_level={"rust": 0.6, "powdery mildew": 0.4}
            )
        }
        
        return crops
    
    def _load_class_mappings(self) -> Dict[str, str]:
        """Load class name mappings from data.yaml"""
        try:
            # This would load your actual class mappings
            # For now, return a simplified mapping
            return {
                "tomato bacterial leaf spot": "bacterial spot",
                "tomato early blight": "early blight",
                "tomato late blight": "late blight",
                "potato early blight": "early blight",
                "potato late blight": "late blight",
                "corn rust": "rust",
                "wheat leaf rust": "rust",
                # Add more mappings based on your data.yaml
            }
        except Exception as e:
            logger.warning(f"Could not load class mappings: {e}")
            return {}
    
    def get_disease_info(self, class_name: str) -> Optional[DiseaseInfo]:
        """Get disease information from class name"""
        # Direct lookup
        if class_name in self.diseases:
            return self.diseases[class_name]
        
        # Try mapping
        mapped_name = self.class_mappings.get(class_name)
        if mapped_name and mapped_name in self.diseases:
            return self.diseases[mapped_name]
        
        # Fuzzy matching
        class_lower = class_name.lower()
        for disease_name, disease_info in self.diseases.items():
            if disease_name in class_lower or any(word in class_lower for word in disease_name.split()):
                return disease_info
        
        return None
    
    def get_crop_info(self, crop_name: str) -> Optional[CropInfo]:
        """Get crop information"""
        if not crop_name:
            return None
        
        crop_lower = crop_name.lower()
        
        # Direct lookup
        if crop_lower in self.crops:
            return self.crops[crop_lower]
        
        # Partial matching
        for crop, info in self.crops.items():
            if crop in crop_lower or crop_lower in crop:
                return info
        
        return None
    
    def get_crop_specific_ranges(self, crop_name: str) -> Dict[str, Tuple[float, float]]:
        """Get crop-specific optimal ranges"""
        crop_info = self.get_crop_info(crop_name)
        if crop_info:
            return {
                "temperature": crop_info.optimal_temp_range,
                "humidity": crop_info.optimal_humidity_range,
                "ph": crop_info.optimal_ph_range,
                "soil_moisture": crop_info.optimal_moisture_range
            }
        
        # Default ranges if crop not found
        return {
            "temperature": (18, 25),
            "humidity": (50, 70),
            "ph": (6.0, 7.0),
            "soil_moisture": (60, 80)
        }
    
    def _get_severity_from_impact(self, impact_score: float) -> SeverityLevel:
        """Convert impact score to severity level"""
        if impact_score >= 90:
            return SeverityLevel.CRITICAL
        elif impact_score >= 75:
            return SeverityLevel.HIGH
        elif impact_score >= 50:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def get_disease_recommendations(self, disease_info: DiseaseInfo, crop_name: str = None) -> List[str]:
        """Get specific recommendations for a disease"""
        recommendations = []
        
        # Treatment priority based recommendations
        if disease_info.treatment_priority >= 4:
            recommendations.append("URGENT: Immediate treatment required")
            
        # Disease type specific recommendations
        if disease_info.type == DiseaseType.FUNGAL:
            recommendations.extend([
                "Apply appropriate fungicide treatment",
                "Improve air circulation around plants",
                "Reduce humidity levels if possible"
            ])
        elif disease_info.type == DiseaseType.BACTERIAL:
            recommendations.extend([
                "Apply copper-based bactericide",
                "Remove and destroy infected plant material",
                "Improve sanitation practices"
            ])
        elif disease_info.type == DiseaseType.VIRAL:
            recommendations.extend([
                "Remove infected plants immediately",
                "Control vector insects",
                "Use virus-free planting material"
            ])
        elif disease_info.type == DiseaseType.PEST:
            recommendations.extend([
                "Apply appropriate insecticide or biological control",
                "Monitor pest population levels",
                "Implement integrated pest management"
            ])
        
        # Spreading rate considerations
        if disease_info.spreading_rate == "fast":
            recommendations.append("Monitor surrounding plants closely - rapid spread possible")
        
        return recommendations
