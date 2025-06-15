"""
Simplified RAG Agent for Agricultural Advice
"""

import logging
import requests
import os
from typing import Dict, List
import re

from config import SERPAPI_KEY, OPENAI_API_KEY, HUGGINGFACE_API_KEY

logger = logging.getLogger(__name__)


class SimpleRAGAgent:
    """Simplified RAG agent for disease advice"""
    
    def __init__(self):
        self.serpapi_key = SERPAPI_KEY
        self.openai_key = OPENAI_API_KEY
        self.huggingface_key = HUGGINGFACE_API_KEY
        
        # Knowledge base for common diseases
        self.knowledge_base = self._build_knowledge_base()
    
    def _build_knowledge_base(self) -> Dict[str, str]:
        """Build basic knowledge base for common diseases"""
        return {
            'early_blight': """
            Early blight is a common fungal disease affecting tomatoes and potatoes.
            Symptoms: Dark spots with concentric rings on leaves, yellowing.
            Treatment: Apply fungicide, improve air circulation, avoid overhead watering.
            Prevention: Use resistant varieties, crop rotation, proper spacing.
            """,
            
            'late_blight': """
            Late blight is a serious disease that can destroy crops quickly.
            Symptoms: Water-soaked spots on leaves, white mold on undersides.
            Treatment: URGENT - Apply copper or mancozeb fungicide immediately.
            Prevention: Avoid wet conditions, destroy infected plants, use resistant varieties.
            """,
            
            'bacterial_spot': """
            Bacterial spot affects tomatoes and peppers.
            Symptoms: Small dark spots on leaves and fruits.
            Treatment: Apply copper-based bactericide, remove infected material.
            Prevention: Use disease-free seeds, avoid overhead watering, good sanitation.
            """,
            
            'powdery_mildew': """
            Powdery mildew appears as white powdery coating on leaves.
            Symptoms: White powder on leaf surfaces, stunted growth.
            Treatment: Apply sulfur or potassium bicarbonate spray.
            Prevention: Good air circulation, avoid overcrowding, resistant varieties.
            """,
            
            'spider_mite': """
            Spider mites are tiny pests that cause stippling on leaves.
            Symptoms: Fine webbing, yellow stippling, leaf bronzing.
            Treatment: Apply miticide, increase humidity, release beneficial insects.
            Prevention: Avoid water stress, maintain humidity, regular monitoring.
            """
        }
    
    def get_disease_advice(self, disease_name: str) -> Dict:
        """Get advice for a specific disease"""
        try:
            # Check local knowledge base first
            simplified_name = self._simplify_disease_name(disease_name)
            
            if simplified_name in self.knowledge_base:
                advice = self.knowledge_base[simplified_name]
                return {
                    'disease_name': disease_name,
                    'summary': self._format_advice(advice),
                    'confidence': 'high',
                    'source': 'local_knowledge'
                }
            
            # Try online sources if API keys available
            if self.huggingface_key:
                return self._get_llm_advice(disease_name)
            
            # Fallback to generic advice
            return self._get_generic_advice(disease_name)
            
        except Exception as e:
            logger.error(f"Error getting disease advice: {e}")
            return self._get_generic_advice(disease_name)
    
    def _simplify_disease_name(self, disease_name: str) -> str:
        """Simplify disease name for knowledge base lookup"""
        # Remove common suffixes and normalize
        simplified = disease_name.lower().replace('_', ' ')
        simplified = re.sub(r'\s+', '_', simplified.strip())
        
        # Map variations
        if 'early blight' in simplified or 'earlyblight' in simplified:
            return 'early_blight'
        elif 'late blight' in simplified or 'lateblight' in simplified:
            return 'late_blight'
        elif 'bacterial spot' in simplified or 'bacterialspot' in simplified:
            return 'bacterial_spot'
        elif 'powdery mildew' in simplified:
            return 'powdery_mildew'
        elif 'spider mite' in simplified:
            return 'spider_mite'
        
        return simplified
    
    def _format_advice(self, raw_advice: str) -> str:
        """Format advice text for better presentation"""
        lines = [line.strip() for line in raw_advice.strip().split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _get_llm_advice(self, disease_name: str) -> Dict:
        """Get advice using LLM"""
        try:
            prompt = f"""Provide practical farming advice for {disease_name}. Include:
1. Symptoms to identify
2. Treatment options
3. Prevention methods
Keep it simple and actionable for farmers."""

            # Try HuggingFace Mistral
            response = self._call_huggingface_api(prompt)
            if response:
                return {
                    'disease_name': disease_name,
                    'summary': response,
                    'confidence': 'medium',
                    'source': 'llm_generated'
                }
            
        except Exception as e:
            logger.error(f"LLM advice generation failed: {e}")
        
        return self._get_generic_advice(disease_name)
    
    def _call_huggingface_api(self, prompt: str) -> str:
        """Call HuggingFace API"""
        try:
            api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
            headers = {"Authorization": f"Bearer {self.huggingface_key}"}
            
            payload = {
                "inputs": f"[INST] {prompt} [/INST]",
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.3,
                    "return_full_text": False
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
            
        except Exception as e:
            logger.error(f"HuggingFace API call failed: {e}")
        
        return ""
    
    def _get_generic_advice(self, disease_name: str) -> Dict:
        """Provide generic advice for unknown diseases"""
        generic_advice = f"""
        General recommendations for {disease_name}:
        
        1. IDENTIFICATION: Monitor plants regularly for symptoms
        2. IMMEDIATE ACTION: Remove and destroy affected plant parts
        3. TREATMENT: Consult local agricultural extension for specific treatments
        4. PREVENTION: 
           - Improve air circulation around plants
           - Avoid overhead watering when possible
           - Practice crop rotation
           - Use disease-resistant varieties
           - Maintain proper plant spacing
        5. MONITORING: Check plants weekly for disease progression
        6. SANITATION: Keep growing area clean and free of plant debris
        
        For specific treatment recommendations, contact your local agricultural advisor.
        """
        
        return {
            'disease_name': disease_name,
            'summary': generic_advice.strip(),
            'confidence': 'low',
            'source': 'generic_guidelines'
        }
