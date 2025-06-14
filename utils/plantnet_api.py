"""
PlantNet API Integration for Plant Species Identification
"""

import requests
import base64
from io import BytesIO
from PIL import Image
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

from config import PLANTNET_API_KEY, PLANTNET_BASE_URL, PLANTNET_PROJECT, PLANTNET_ORGANS

logger = logging.getLogger(__name__)


class PlantNetAPI:
    """Interface for PlantNet plant identification API"""
    
    def __init__(self, api_key: str = None, project: str = None):
        self.api_key = api_key or PLANTNET_API_KEY
        self.project = project or PLANTNET_PROJECT
        self.base_url = PLANTNET_BASE_URL
        
        if not self.api_key:
            logger.warning("PlantNet API key not provided. Plant identification will not work.")
    
    def identify_plant(self, 
                      image: Union[str, Path, Image.Image, bytes], 
                      organs: List[str] = None,
                      include_related: bool = True,
                      no_reject: bool = False,
                      nb_results: int = 5,
                      lang: str = "en") -> Optional[Dict]:
        """
        Identify plant species from image using PlantNet API
        
        Args:
            image: Image file path, PIL Image, or image bytes
            organs: List of plant organs (leaf, flower, fruit, bark)
            include_related: Include related images in results
            no_reject: Don't reject uncertain identifications
            nb_results: Number of results to return
            lang: Language for common names
            
        Returns:
            Dictionary with identification results or None if failed
        """
        if not self.api_key:
            logger.error("PlantNet API key not configured")
            return None
        
        try:
            # Prepare image data
            image_data = self._prepare_image(image)
            if not image_data:
                return None
            
            # Prepare request
            url = f"{self.base_url}/identify/{self.project}"
            
            params = {
                "api-key": self.api_key,
                "include-related-images": include_related,
                "no-reject": no_reject,
                "nb-results": nb_results,
                "lang": lang
            }
            
            # Default organs if not specified
            if organs is None:
                organs = PLANTNET_ORGANS[:1]  # Use only 'leaf' by default
            
            files = []
            data = {}
            
            # Add images for each organ
            for i, organ in enumerate(organs):
                files.append(('images', (f'image_{i}.jpg', image_data, 'image/jpeg')))
                data[f'organs[{i}]'] = organ
            
            # Make API request
            response = requests.post(url, params=params, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"PlantNet identification successful: {len(result.get('results', []))} results")
                return self._process_results(result)
            else:
                logger.error(f"PlantNet API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling PlantNet API: {e}")
            return None
    
    def _prepare_image(self, image: Union[str, Path, Image.Image, bytes]) -> Optional[bytes]:
        """Convert various image inputs to bytes"""
        try:
            if isinstance(image, (str, Path)):
                # File path
                with open(image, 'rb') as f:
                    return f.read()
            elif isinstance(image, Image.Image):
                # PIL Image
                buffer = BytesIO()
                image.save(buffer, format='JPEG', quality=85)
                return buffer.getvalue()
            elif isinstance(image, bytes):
                # Already bytes
                return image
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return None
        except Exception as e:
            logger.error(f"Error preparing image: {e}")
            return None
    
    def _process_results(self, api_response: Dict) -> Dict:
        """Process and structure API response"""
        results = []
        
        for species in api_response.get('results', []):
            species_info = species.get('species', {})
            
            result = {
                'scientific_name': species_info.get('scientificNameWithoutAuthor', ''),
                'common_names': species_info.get('commonNames', []),
                'family': species_info.get('family', {}).get('scientificNameWithoutAuthor', ''),
                'genus': species_info.get('genus', {}).get('scientificNameWithoutAuthor', ''),
                'score': species.get('score', 0.0),
                'confidence': self._calculate_confidence(species.get('score', 0.0))
            }
            
            # Get the best common name
            if result['common_names']:
                result['primary_common_name'] = result['common_names'][0]
            else:
                result['primary_common_name'] = result['scientific_name']
            
            results.append(result)
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        processed_response = {
            'success': True,
            'species_count': len(results),
            'best_match': results[0] if results else None,
            'all_results': results,
            'query_info': {
                'project': api_response.get('project', ''),
                'language': api_response.get('lang', ''),
                'remaining_identification_requests': api_response.get('remainingIdentificationRequests', 0)
            }
        }
        
        return processed_response
    
    def _calculate_confidence(self, score: float) -> str:
        """Convert numeric score to confidence level"""
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.1:
            return "low"
        else:
            return "very_low"
    
    def get_plant_info(self, scientific_name: str) -> Dict:
        """Get detailed information about a plant species"""
        # This could be extended to fetch additional plant information
        # from botanical databases or agricultural resources
        
        # For now, return basic structure
        return {
            'scientific_name': scientific_name,
            'care_info': {
                'optimal_temperature': None,
                'optimal_humidity': None,
                'common_diseases': [],
                'common_pests': [],
                'growing_season': None
            }
        }


def test_plantnet_integration():
    """Test function for PlantNet API integration"""
    api = PlantNetAPI()
    
    if not api.api_key:
        print("❌ PlantNet API key not configured")
        return False
    
    # Test with a sample image (you'd need to provide a real image)
    print("🔍 Testing PlantNet API integration...")
    print("⚠️  Add a test image to test the identification")
    
    return True


if __name__ == "__main__":
    test_plantnet_integration()
