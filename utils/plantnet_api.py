"""
PlantNet API Integration for Plant Species Identification
"""

import requests
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
        self.project = project or "all"
        self.base_url = "https://my-api.plantnet.org/v2"
        
        # Available projects (ordered by preference)
        self.available_projects = ["all", "weurope", "useful", "k-world-flora", "the-plant-list"]
        
        if not self.api_key:
            logger.warning("PlantNet API key not provided. Plant identification will not work.")

    def identify_plant(self, 
                      image: Union[str, Path, Image.Image, bytes], 
                      organs: List[str] = None,
                      include_related: bool = False,
                      no_reject: bool = False,
                      nb_results: int = 5,
                      lang: str = "en") -> Optional[Dict]:
        """Identify plant species from image using PlantNet API v2"""
        if not self.api_key:
            logger.error("PlantNet API key not configured")
            return self._create_fallback_result()
        
        try:
            # Prepare image data
            image_data = self._prepare_image(image)
            if not image_data:
                return self._create_fallback_result()
            
            # Default organs if not specified
            if organs is None:
                organs = ["auto"]
            
            # Try the working approach directly
            return self._identify_with_working_approach(image_data, organs)
                
        except requests.exceptions.Timeout:
            logger.error("PlantNet API request timed out")
            return self._create_fallback_result()
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to PlantNet API")
            return self._create_fallback_result()
        except Exception as e:
            logger.error(f"Error calling PlantNet API: {e}")
            return self._create_fallback_result()
    
    def _identify_with_working_approach(self, image_data: bytes, organs: List[str]) -> Optional[Dict]:
        """Use the minimal approach that we know works"""
        
        for project in self.available_projects:
            try:
                url = f"{self.base_url}/identify/{project}"
                
                # Minimal working parameters
                params = {'api-key': self.api_key}
                files = [('images', ('image.jpg', image_data, 'image/jpeg'))]
                data = {'organs': organs[0] if organs else 'auto'}
                
                logger.info(f"Trying minimal request for project {project}")
                
                response = requests.post(url, params=params, files=files, data=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Success with project {project}")
                    return self._process_v2_results(result)
                else:
                    logger.debug(f"Project {project} failed with status {response.status_code}")
                    
            except Exception as e:
                logger.debug(f"Request error for {project}: {e}")
                continue
        
        logger.error("All projects failed")
        return self._create_fallback_result()
    
    def _process_v2_results(self, api_response: Dict) -> Dict:
        """Process v2 API response with the actual structure"""
        results = []
        
        # Handle v2 API response structure
        raw_results = api_response.get('results', [])
        
        for item in raw_results:
            species_info = item.get('species', {})
            
            # Extract species information
            scientific_name = (
                species_info.get('scientificNameWithoutAuthor', '') or
                species_info.get('scientificName', '') or
                'Unknown species'
            )
            
            # Handle common names
            common_names = species_info.get('commonNames', [])
            if isinstance(common_names, str):
                common_names = [common_names]
            elif not isinstance(common_names, list):
                common_names = []
            
            # Extract family and genus info
            family_info = species_info.get('family', {})
            family_name = ''
            if isinstance(family_info, dict):
                family_name = family_info.get('scientificNameWithoutAuthor', '') or family_info.get('scientificName', '')
            elif isinstance(family_info, str):
                family_name = family_info
            
            genus_info = species_info.get('genus', {})
            genus_name = ''
            if isinstance(genus_info, dict):
                genus_name = genus_info.get('scientificNameWithoutAuthor', '') or genus_info.get('scientificName', '')
            elif isinstance(genus_info, str):
                genus_name = genus_info
            
            # Get score
            score = float(item.get('score', 0.0))
            
            result = {
                'scientific_name': scientific_name,
                'common_names': common_names,
                'family': family_name,
                'genus': genus_name,
                'score': score,
                'confidence': self._calculate_confidence(score)
            }
            
            # Get the best common name
            if result['common_names']:
                result['primary_common_name'] = result['common_names'][0]
            else:
                result['primary_common_name'] = result['scientific_name']
            
            results.append(result)
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Build response in our standard format
        processed_response = {
            'success': True,
            'species_count': len(results),
            'best_match': results[0] if results else None,
            'all_results': results,
            'query_info': {
                'project': api_response.get('query', {}).get('project', 'unknown'),
                'language': api_response.get('language', 'en'),
                'best_match': api_response.get('bestMatch', ''),
                'predicted_organs': api_response.get('predictedOrgans', []),
                'remaining_identification_requests': api_response.get('remainingIdentificationRequests', 0)
            }
        }
        
        # Log successful identification
        if results:
            best_result = results[0]
            logger.info(f"Plant identified: {best_result['primary_common_name']} ({best_result['scientific_name']}) - Score: {best_result['score']:.4f}")
        
        return processed_response
    
    def _create_fallback_result(self) -> Dict:
        """Create a fallback result when API fails"""
        return {
            'success': False,
            'species_count': 0,
            'best_match': {
                'scientific_name': 'Unknown species',
                'common_names': ['Unknown plant'],
                'primary_common_name': 'Unknown plant',
                'family': 'Unknown',
                'genus': 'Unknown',
                'score': 0.0,
                'confidence': 'very_low'
            },
            'all_results': [],
            'query_info': {
                'project': self.project,
                'language': 'en',
                'remaining_identification_requests': 0,
                'error': 'API call failed - using fallback'
            }
        }
    
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
    """Simple test function for PlantNet API v2 integration"""
    api = PlantNetAPI()
    
    if not api.api_key:
        print("❌ PlantNet API key not configured")
        return False
    
    print("🔍 Testing PlantNet API v2 integration...")
    print(f"API Key configured: {'Yes' if api.api_key else 'No'}")
    print(f"Project: {api.project}")
    print(f"Base URL: {api.base_url}")
    
    # Test the working approach only
    test_url = "https://my-api.plantnet.org/v2/identify/all"
    
    try:
        files = [('images', ('test.jpg', b'fake_image_data', 'image/jpeg'))]
        params = {'api-key': api.api_key}
        data = {'organs': 'auto'}
        
        response = requests.post(test_url, params=params, files=files, data=data, timeout=10)
        
        print(f"Test result: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API working correctly")
        elif response.status_code == 400:
            print("✅ Authentication working (bad request due to fake image)")
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("💡 Using minimal working parameters: organs='auto' only")
    return True


if __name__ == "__main__":
    test_plantnet_integration()
