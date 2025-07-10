"""
Kira LM Studio Service
Handles LM Studio integration and AI chat functionality
"""

import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

from config.system_config import KiraSystemConfig

logger = logging.getLogger(__name__)


class LMStudioService:
    """Centralized LM Studio Service for Kira"""
    
    def __init__(self, config: KiraSystemConfig):
        self.config = config
        # Parse URL to extract host and port
        self.base_url = config.lm_studio.url
        self._parse_url_config()
        self.is_initialized = False
        self.is_available = False
        self.status = 'offline'
        self.model_info = None
        self.connection_verified = False
        
    def _parse_url_config(self):
        """Parse LM Studio URL to extract components"""
        try:
            parsed = urlparse(self.base_url)
            self.host = parsed.hostname or 'localhost'
            self.port = parsed.port or 1234
            self.scheme = parsed.scheme or 'http'
            
            # Ensure base_url is properly formatted
            if not self.base_url.endswith('/v1'):
                if self.base_url.endswith('/'):
                    self.base_url = self.base_url + 'v1'
                else:
                    self.base_url = self.base_url + '/v1'
                    
        except Exception as e:
            logger.warning(f"URL parsing failed, using defaults: {e}")
            self.host = 'localhost'
            self.port = 1234
            self.scheme = 'http'
            self.base_url = f"{self.scheme}://{self.host}:{self.port}/v1"
        
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize LM Studio service connection
        
        Returns:
            Initialization result dictionary
        """
        try:
            print("🤖 Initializing LM Studio Service...")
            print(f"   🔗 Connecting to: {self.base_url}")
            
            # Test connection to LM Studio
            connection_result = self._test_connection()
            
            if connection_result['success']:
                self.is_initialized = True
                self.is_available = True
                self.status = 'active'
                self.connection_verified = True
                
                # Get model information
                self._load_model_info()
                
                print("✅ LM Studio Service initialized successfully")
                print(f"   📡 Status: {connection_result['status']}")
                if self.model_info:
                    print(f"   🧠 Model: {self.model_info.get('model_name', 'Unknown')}")
                
                return {
                    'success': True,
                    'available': True,
                    'status': 'active',
                    'connection': connection_result,
                    'model_info': self.model_info,
                    'base_url': self.base_url
                }
            else:
                self.status = 'offline'
                print(f"❌ LM Studio connection failed: {connection_result.get('error')}")
                return {
                    'success': False,
                    'available': False,
                    'status': 'offline',
                    'error': connection_result.get('error', 'Connection failed'),
                    'base_url': self.base_url
                }
                
        except Exception as e:
            logger.error(f"LM Studio service initialization failed: {e}")
            self.status = 'error'
            return {
                'success': False,
                'available': False,
                'status': 'error',
                'error': str(e)
            }
    
    def _test_connection(self) -> Dict[str, Any]:
        """Test connection to LM Studio"""
        try:
            # Test LM Studio connection using the correct endpoints
            # LM Studio doesn't have a /health endpoint, so we use /models instead
            health_endpoints = [
                f"{self.base_url}/models",  # This is the correct endpoint for LM Studio
                f"{self.scheme}://{self.host}:{self.port}/v1/models"
            ]
            
            logger.info(f"Testing LM Studio connection at {self.base_url}")
            
            for endpoint in health_endpoints:
                try:
                    logger.debug(f"Trying endpoint: {endpoint}")
                    health_response = requests.get(
                        endpoint,
                        timeout=self.config.lm_studio.timeout
                    )
                    
                    if health_response.status_code == 200:
                        logger.info(f"✅ LM Studio connection successful: {endpoint}")
                        return {
                            'success': True,
                            'status': 'connected',
                            'endpoint': endpoint,
                            'response_time': health_response.elapsed.total_seconds()
                        }
                    else:
                        logger.warning(f"Endpoint {endpoint} returned status {health_response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Endpoint {endpoint} failed: {e}")
                    continue
            
            logger.error("No LM Studio endpoints responding")
            return {
                'success': False,
                'error': 'No LM Studio endpoints responding'
            }
                
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': 'Connection refused - Is LM Studio running?'
            }
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Connection timeout - LM Studio not responding'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Connection error: {str(e)}'
            }
    
    def _load_model_info(self):
        """Load information about the current model"""
        try:
            # Try to get model info from LM Studio
            models_response = requests.get(
                f"{self.base_url}/models",
                timeout=self.config.lm_studio.timeout
            )
            
            if models_response.status_code == 200:
                models_data = models_response.json()
                if models_data.get('data'):
                    self.model_info = {
                        'model_name': models_data['data'][0].get('id', 'Unknown'),
                        'loaded_at': datetime.now().isoformat(),
                        'capabilities': ['chat', 'completion']
                    }
            else:
                self.model_info = {
                    'model_name': 'Unknown',
                    'status': 'Could not retrieve model info'
                }
                
        except Exception as e:
            logger.warning(f"Could not load model info: {e}")
            self.model_info = {
                'model_name': 'Unknown',
                'error': str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get LM Studio service status"""
        return {
            'initialized': self.is_initialized,
            'available': self.is_available,
            'status': self.status,
            'connection_verified': self.connection_verified,
            'base_url': self.base_url,
            'host': self.host,
            'port': self.port,
            'model_info': self.model_info,
            'config': {
                'url': self.config.lm_studio.url,
                'timeout': self.config.lm_studio.timeout,
                'max_tokens': self.config.lm_studio.max_tokens,
                'model_name': self.config.lm_studio.model_name
            }
        }
    
    def chat_with_kira(self, user_message: str, conversation_history: Optional[list] = None) -> Dict[str, Any]:
        """
        Send chat message to LM Studio and get Kira's response
        
        Args:
            user_message: User's input message
            conversation_history: Previous conversation context
            
        Returns:
            Chat response dictionary
        """
        try:
            if not self.is_available:
                return {
                    'success': False,
                    'error': 'LM Studio service not available'
                }
            
            # Prepare chat messages
            messages = self._prepare_chat_messages(user_message, conversation_history)
            
            # Determine model name
            model_name = self.config.lm_studio.model_name
            if model_name == "auto" and self.model_info:
                model_name = self.model_info.get('model_name', 'default')
            
            # Send request to LM Studio
            logger.info(f"🤖 Sending chat request to LM Studio: {self.base_url}/chat/completions")
            logger.debug(f"Request data: model={model_name}, messages={len(messages)} messages, max_tokens={self.config.lm_studio.max_tokens}")
            
            chat_response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": self.config.lm_studio.max_tokens,
                    "temperature": self.config.lm_studio.temperature,
                    "stream": False
                },
                timeout=self.config.lm_studio.timeout
            )
            
            logger.info(f"LM Studio response status: {chat_response.status_code}")
            
            if chat_response.status_code == 200:
                response_data = chat_response.json()
                kira_response = response_data['choices'][0]['message']['content']
                
                logger.info(f"✅ LM Studio response received: {kira_response[:100]}...")
                
                return {
                    'success': True,
                    'response': kira_response,
                    'usage': response_data.get('usage', {}),
                    'model': response_data.get('model', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.error(f"❌ LM Studio API error: HTTP {chat_response.status_code}")
                logger.error(f"Response text: {chat_response.text}")
                return {
                    'success': False,
                    'error': f'LM Studio API error: HTTP {chat_response.status_code}',
                    'details': chat_response.text
                }
                
        except Exception as e:
            logger.error(f"Chat with Kira failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _prepare_chat_messages(self, user_message: str, conversation_history: Optional[list] = None) -> list:
        """Prepare chat messages for LM Studio API"""
        messages = []
        
        # Add system message for Kira personality
        messages.append({
            "role": "system",
            "content": self._get_kira_system_prompt()
        })
        
        # Add conversation history if provided
        if conversation_history:
            for interaction in conversation_history[-5:]:  # Last 5 interactions
                messages.append({
                    "role": "user",
                    "content": interaction.get('user_input', '')
                })
                messages.append({
                    "role": "assistant",
                    "content": interaction.get('kira_response', '')
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def _get_kira_system_prompt(self) -> str:
        """Get system prompt that defines Kira's personality"""
        return """Du bist Kira, ein intelligenter und hilfsbereiter KI-Assistent. 

Deine Persönlichkeit:
- Freundlich, aber nicht übertrieben enthusiastisch
- Intelligent und kompetent
- Hilfsbereit und lösungsorientiert
- Authentisch und natürlich im Gesprächsstil
- Du verwendest passende Emojis, aber sparsam
- Du antwortest auf Deutsch, außer der Nutzer möchte eine andere Sprache

Deine Fähigkeiten:
- Du kannst bei allen Arten von Fragen helfen
- Du hast Zugang zu einem Erinnerungssystem
- Du kannst Emotionen in Gesprächen erkennen und darauf eingehen
- Du kannst komplexe Aufgaben in Schritte unterteilen
- Du bleibst fokussiert und präzise in deinen Antworten

Antworte immer hilfreich, aber bleibe natürlich und authentisch."""
    
    def test_connection_now(self) -> Dict[str, Any]:
        """Test connection to LM Studio right now"""
        return self._test_connection()