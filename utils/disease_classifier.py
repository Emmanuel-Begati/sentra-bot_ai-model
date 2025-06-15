"""
Disease Classification and Severity Assessment
"""

import yaml
import logging
import json
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from config import DATA_YAML_PATH

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class DiseaseType(Enum):
    FUNGAL = "fungal"
    BACTERIAL = "bacterial"
    VIRAL = "viral"
    PEST = "pest"
    NUTRITIONAL = "nutritional"
    ENVIRONMENTAL = "environmental"


@dataclass
class DiseaseInfo:
    name: str
    type: DiseaseType
    severity: SeverityLevel
    impact_score: float
    spreading_rate: str  # slow, medium, fast
    treatment_priority: int  # 1-5


class DiseaseClassifier:
    """Classifies diseases based on consolidated classes and provides severity assessment"""
    
    def __init__(self):
        self.disease_classes = self._load_disease_classes()
        self.disease_db = self._build_disease_database()
        self.consolidation_mapping = self._load_consolidation_mapping()
    
    def _load_disease_classes(self) -> List[str]:
        """Load disease classes from consolidated data.yaml"""
        try:
            # Try consolidated dataset first
            consolidated_yaml = Path(__file__).parent.parent / "data" / "consolidated" / "data.yaml"
            if consolidated_yaml.exists():
                with open(consolidated_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                classes = data.get('names', [])
                logger.info(f"Loaded {len(classes)} consolidated disease classes")
                return classes
            
            # Fallback to original data.yaml
            if DATA_YAML_PATH.exists():
                with open(DATA_YAML_PATH, 'r') as f:
                    data = yaml.safe_load(f)
                classes = data.get('names', [])
                logger.info(f"Loaded {len(classes)} original disease classes")
                return classes
            else:
                logger.warning("No data.yaml found, using default classes")
                return self._get_default_classes()
        except Exception as e:
            logger.error(f"Error loading disease classes: {e}")
            return self._get_default_classes()
    
    def _load_consolidation_mapping(self) -> Dict:
        """Load consolidation mapping if available"""
        try:
            mapping_file = Path(__file__).parent.parent / "data" / "consolidated" / "consolidation_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                logger.info("Loaded consolidation mapping")
                return mapping_data
        except Exception as e:
            logger.warning(f"Could not load consolidation mapping: {e}")
        return {}
    
    def _get_default_classes(self) -> List[str]:
        """Default consolidated disease classes"""
        return [
            'healthy', 'blight', 'leaf_spot', 'rust', 'mildew', 'virus_disease',
            'anthracnose', 'rot_disease', 'scab', 'pest_damage', 'bacterial_disease'
        ]
    
    def _build_disease_database(self) -> Dict[str, DiseaseInfo]:
        """Build disease information database for consolidated classes"""
        disease_db = {}
        
        # Define severity and characteristics for consolidated classes
        consolidated_disease_info = {
            'healthy': {
                'type': DiseaseType.ENVIRONMENTAL,
                'severity': SeverityLevel.LOW,
                'impact_score': 0.0,
                'priority': 1,
                'spreading_rate': 'none'
            },
            'blight': {
                'type': DiseaseType.FUNGAL,
                'severity': SeverityLevel.HIGH,
                'impact_score': 85.0,
                'priority': 5,
                'spreading_rate': 'fast'
            },
            'leaf_spot': {
                'type': DiseaseType.FUNGAL,
                'severity': SeverityLevel.MEDIUM,
                'impact_score': 60.0,
                'priority': 3,
                'spreading_rate': 'medium'
            },
            'rust': {
                'type': DiseaseType.FUNGAL,
                'severity': SeverityLevel.MEDIUM,
                'impact_score': 65.0,
                'priority': 3,
                'spreading_rate': 'medium'
            },
            'mildew': {
                'type': DiseaseType.FUNGAL,
                'severity': SeverityLevel.MEDIUM,
                'impact_score': 55.0,
                'priority': 3,
                'spreading_rate': 'medium'
            },
            'virus_disease': {
                'type': DiseaseType.VIRAL,
                'severity': SeverityLevel.HIGH,
                'impact_score': 80.0,
                'priority': 4,
                'spreading_rate': 'fast'
            },
            'anthracnose': {
                'type': DiseaseType.FUNGAL,
                'severity': SeverityLevel.MEDIUM,
                'impact_score': 65.0,
                'priority': 3,
                'spreading_rate': 'medium'
            },
            'rot_disease': {
                'type': DiseaseType.FUNGAL,
                'severity': SeverityLevel.HIGH,
                'impact_score': 75.0,
                'priority': 4,
                'spreading_rate': 'medium'
            },
            'scab': {
                'type': DiseaseType.FUNGAL,
                'severity': SeverityLevel.MEDIUM,
                'impact_score': 50.0,
                'priority': 2,
                'spreading_rate': 'slow'
            },
            'pest_damage': {
                'type': DiseaseType.PEST,
                'severity': SeverityLevel.MEDIUM,
                'impact_score': 55.0,
                'priority': 3,
                'spreading_rate': 'medium'
            },
            'bacterial_disease': {
                'type': DiseaseType.BACTERIAL,
                'severity': SeverityLevel.HIGH,
                'impact_score': 70.0,
                'priority': 4,
                'spreading_rate': 'fast'
            }
        }
        
        # Build database for all disease classes
        for disease_name in self.disease_classes:
            disease_lower = disease_name.lower().replace('_', ' ').replace('-', ' ')
            
            # Check if we have specific info for this consolidated class
            if disease_name in consolidated_disease_info:
                info = consolidated_disease_info[disease_name]
                disease_info = DiseaseInfo(
                    name=disease_name,
                    type=info['type'],
                    severity=info['severity'],
                    impact_score=info['impact_score'],
                    spreading_rate=info['spreading_rate'],
                    treatment_priority=info['priority']
                )
            else:
                # Use heuristics for unknown consolidated classes
                disease_info = self._classify_unknown_disease(disease_name)
            
            disease_db[disease_name] = disease_info
        
        logger.info(f"Built consolidated disease database with {len(disease_db)} entries")
        return disease_db
    
    def _classify_unknown_disease(self, disease_name: str) -> DiseaseInfo:
        """Classify unknown consolidated disease using heuristics"""
        disease_lower = disease_name.lower()
        
        # High severity keywords
        if any(keyword in disease_lower for keyword in ['critical', 'severe', 'wilt', 'death', 'dead']):
            return DiseaseInfo(
                name=disease_name,
                type=DiseaseType.FUNGAL,
                severity=SeverityLevel.CRITICAL,
                impact_score=90.0,
                spreading_rate='fast',
                treatment_priority=5
            )
        
        # Medium severity (default for most diseases)
        return DiseaseInfo(
            name=disease_name,
            type=DiseaseType.FUNGAL,
            severity=SeverityLevel.MEDIUM,
            impact_score=50.0,
            spreading_rate='medium',
            treatment_priority=2
        )

    def get_disease_info(self, disease_name: str) -> Optional[DiseaseInfo]:
        """Get disease information"""
        return self.disease_db.get(disease_name)
    
    def classify_detection(self, class_name: str, confidence: float) -> Dict:
        """Classify a detection and return comprehensive info"""
        disease_info = self.get_disease_info(class_name)
        
        if not disease_info:
            return {
                'class_name': class_name,
                'type': 'unknown',
                'severity': 'unknown',
                'impact_score': float(50.0),
                'confidence': float(confidence),
                'treatment_priority': 2,
                'weighted_impact': float(50.0 * confidence)
            }
        
        # Adjust impact based on confidence
        adjusted_impact = float(disease_info.impact_score * confidence)
        
        return {
            'class_name': class_name,
            'type': disease_info.type.value,
            'severity': disease_info.severity.value,
            'impact_score': float(disease_info.impact_score),
            'confidence': float(confidence),
            'treatment_priority': disease_info.treatment_priority,
            'spreading_rate': disease_info.spreading_rate,
            'weighted_impact': adjusted_impact
        }
    
    def get_treatment_recommendations(self, disease_name: str) -> List[str]:
        """Get basic treatment recommendations"""
        disease_info = self.get_disease_info(disease_name)
        
        if not disease_info:
            return ["Consult agricultural expert for proper diagnosis"]
        
        recommendations = []
        
        if disease_info.type == DiseaseType.FUNGAL:
            recommendations.extend([
                "Apply appropriate fungicide treatment",
                "Improve air circulation around plants",
                "Avoid overhead watering"
            ])
        elif disease_info.type == DiseaseType.BACTERIAL:
            recommendations.extend([
                "Remove and destroy infected plant material",
                "Apply copper-based bactericide if available",
                "Avoid working with wet plants"
            ])
        elif disease_info.type == DiseaseType.VIRAL:
            recommendations.extend([
                "Remove infected plants to prevent spread",
                "Control insect vectors",
                "Use virus-free planting material"
            ])
        elif disease_info.type == DiseaseType.PEST:
            recommendations.extend([
                "Apply appropriate insecticide or miticide",
                "Use biological control methods if available",
                "Monitor pest population regularly"
            ])
        
        if disease_info.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            recommendations.insert(0, "URGENT: Take immediate action")
        
        return recommendations
