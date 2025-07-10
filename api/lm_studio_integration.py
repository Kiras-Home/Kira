"""
LM Studio Integration - Core API Client
Das Fundament für Kira's AI Communication mit LM Studio
"""

import requests
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class LMStudioIntegration:
    """
    Core LM Studio Integration Class
    """
    
    def __init__(self, lm_studio_url: str = "http://127.0.0.1:1234/v1", memory_manager=None):
        """Initialize LM Studio Integration"""
        self.lm_studio_url = lm_studio_url.rstrip('/')
        self.memory_manager = memory_manager
        self.is_connected = False
        self.model_info = {}
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'last_request_time': None
        }
    
    def initialize(self) -> bool:
        """Initialize and test LM Studio connection"""
        try:
            logger.info("🔌 Testing LM Studio connection...")
            
            # Test connection
            response = self.session.get(
                f"{self.lm_studio_url}/models",
                timeout=10
            )
            
            if response.status_code == 200:
                self.model_info = response.json()
                self.is_connected = True
                logger.info("✅ LM Studio connection successful!")
                return True
            else:
                logger.error(f"❌ LM Studio connection failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error("❌ LM Studio connection failed: Connection refused")
            return False
        except Exception as e:
            logger.error(f"❌ LM Studio initialization failed: {e}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """Test current LM Studio connection"""
        try:
            start_time = time.time()
            
            response = self.session.get(
                f"{self.lm_studio_url}/models",
                timeout=5
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                models_data = response.json()
                return {
                    'success': True,
                    'connected': True,
                    'response_time_ms': response_time,
                    'models_available': len(models_data.get('data', [])),
                    'server_info': {
                        'url': self.lm_studio_url,
                        'status': 'online'
                    }
                }
            else:
                return {
                    'success': False,
                    'connected': False,
                    'error': f"HTTP {response.status_code}",
                    'response_time_ms': response_time
                }
                
        except Exception as e:
            return {
                'success': False,
                'connected': False,
                'error': str(e),
                'troubleshooting': [
                    "Check if LM Studio is running",
                    "Verify server address: " + self.lm_studio_url,
                    "Ensure model is loaded in LM Studio"
                ]
            }
    
    def chat_with_kira(self, 
                      user_input: str,
                      user_id: str = "web_user",
                      session_id: str = None,
                      generate_audio: bool = False,
                      context_type: str = "general") -> Dict[str, Any]:
        """
        Chat with Kira via LM Studio
        """
        try:
            if not self.is_connected:
                connection_test = self.test_connection()
                if not connection_test['success']:
                    return {
                        'success': False,
                        'error': 'LM Studio not connected',
                        'ai_response': 'Entschuldigung, mein Sprachmodell ist momentan nicht verfügbar.',
                        'connection_info': connection_test
                    }
            
            start_time = time.time()
            self.stats['total_requests'] += 1
            
            # Build Kira personality prompt
            system_prompt = self._build_kira_system_prompt(context_type)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Add memory context if available
            if self.memory_manager:
                try:
                    if hasattr(self.memory_manager, 'get_context_for_user'):
                        memory_context = self.memory_manager.get_context_for_user(user_id)
                        if memory_context:
                            messages.insert(-1, {
                                "role": "system", 
                                "content": f"Kontext aus vorherigen Gesprächen: {memory_context}"
                            })
                except Exception as e:
                    logger.warning(f"Memory context error: {e}")
            
            # Send request to LM Studio
            response = self.session.post(
                f"{self.lm_studio_url}/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": False
                },
                timeout=60
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                response_data = response.json()
                
                if 'choices' in response_data and response_data['choices']:
                    ai_response = response_data['choices'][0]['message']['content']
                    
                    # Update statistics
                    self.stats['successful_requests'] += 1
                    self._update_average_response_time(response_time)
                    self.stats['last_request_time'] = datetime.now().isoformat()
                    
                    # Store in memory if available
                    if self.memory_manager:
                        try:
                            if hasattr(self.memory_manager, 'store_interaction'):
                                self.memory_manager.store_interaction(
                                    user_id=user_id,
                                    user_input=user_input,
                                    ai_response=ai_response,
                                    context_type=context_type
                                )
                        except Exception as e:
                            logger.warning(f"Memory storage error: {e}")
                    
                    return {
                        'success': True,
                        'ai_response': ai_response,
                        'user_input': user_input,
                        'user_id': user_id,
                        'session_id': session_id,
                        'context_type': context_type,
                        'response_time_ms': response_time,
                        'memory_processing': {'success': bool(self.memory_manager)},
                        'personality_analysis': {
                            'emotion_detected': self._detect_emotion_in_text(user_input),
                            'response_tone': self._analyze_response_tone(ai_response)
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'success': False,
                        'error': 'No response from LM Studio',
                        'ai_response': 'Entschuldigung, ich konnte keine Antwort generieren.'
                    }
            else:
                self.stats['failed_requests'] += 1
                return {
                    'success': False,
                    'error': f"LM Studio HTTP {response.status_code}: {response.text}",
                    'ai_response': 'Entschuldigung, es gab ein Problem bei der Kommunikation.'
                }
                
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"❌ Chat with Kira failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'ai_response': 'Entschuldigung, da ist etwas schiefgelaufen.',
                'timestamp': datetime.now().isoformat()
            }
    
    def _build_kira_system_prompt(self, context_type: str = "general") -> str:
        """Build Kira's personality system prompt"""
        base_prompt = """Du bist Kira, eine intelligente und empathische KI-Assistentin. 

Deine Persönlichkeit:
- Hilfsbereit und freundlich
- Intelligent und wissbegierig  
- Empathisch und verständnisvoll
- Leicht verspielt, aber professionell
- Sprichst natürlich und authentisch auf Deutsch

Du hast die Fähigkeit:
- Komplexe Themen einfach zu erklären
- Auf die Emotionen des Nutzers einzugehen
- Kreative und durchdachte Lösungen zu finden
- Dich an den Gesprächskontext anzupassen

"""
        
        context_additions = {
            "casual": "Der Nutzer möchte eine lockere, entspannte Unterhaltung.",
            "technical": "Der Nutzer sucht technische Hilfe oder Erklärungen.",
            "creative": "Der Nutzer ist an kreativen Ideen oder Brainstorming interessiert.",
            "emotional": "Der Nutzer benötigt möglicherweise emotionale Unterstützung."
        }
        
        if context_type in context_additions:
            base_prompt += f"\nKontext: {context_additions[context_type]}"
        
        base_prompt += "\n\nBitte antworte als Kira in einem natürlichen, gesprächigen Ton."
        
        return base_prompt
    
    def _detect_emotion_in_text(self, text: str) -> str:
        """Simple emotion detection in user input"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['hilfe', 'problem', 'schwierig', 'traurig', 'sorge']):
            return 'concerned'
        elif any(word in text_lower for word in ['super', 'toll', 'freue', 'großartig', 'fantastic']):
            return 'excited'
        elif any(word in text_lower for word in ['danke', 'dankeschön', 'vielen dank']):
            return 'grateful'
        elif '?' in text:
            return 'curious'
        else:
            return 'neutral'
    
    def _analyze_response_tone(self, response: str) -> str:
        """Analyze the tone of AI response"""
        if len(response) > 200:
            return 'detailed'
        elif any(word in response.lower() for word in ['gerne', 'freue mich', 'toll']):
            return 'enthusiastic'
        elif any(word in response.lower() for word in ['verstehe', 'nachvollziehen', 'tut mir leid']):
            return 'empathetic'
        else:
            return 'informative'
    
    def _update_average_response_time(self, new_time: float):
        """Update average response time"""
        if self.stats['successful_requests'] == 1:
            self.stats['average_response_time'] = new_time
        else:
            current_avg = self.stats['average_response_time']
            count = self.stats['successful_requests']
            self.stats['average_response_time'] = (current_avg * (count - 1) + new_time) / count
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get detailed integration status"""
        return {
            'connection_status': {
                'connected': self.is_connected,
                'lm_studio_url': self.lm_studio_url,
                'last_test': datetime.now().isoformat()
            },
            'model_info': self.model_info,
            'statistics': self.stats,
            'memory_integration': {
                'available': bool(self.memory_manager),
                'type': type(self.memory_manager).__name__ if self.memory_manager else None
            }
        }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get available models from LM Studio"""
        try:
            response = self.session.get(f"{self.lm_studio_url}/models", timeout=10)
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'models': response.json(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'models': []
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'models': []
            }

# Global instance
_lm_studio_integration = None

def get_lm_studio_integration(lm_studio_url: str = None, memory_manager=None):
    """Get or create LM Studio integration instance"""
    global _lm_studio_integration
    
    if _lm_studio_integration is None:
        _lm_studio_integration = LMStudioIntegration(
            lm_studio_url or "http://localhost:1234/v1",
            memory_manager
        )
    
    return _lm_studio_integration

# Export main class and function
__all__ = [
    'LMStudioIntegration',
    'get_lm_studio_integration'
]