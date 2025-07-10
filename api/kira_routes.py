"""
Kira-specific API Routes
Handles Kira personality, brain activity and AI-specific endpoints
"""

import logging
from flask import Blueprint, jsonify, request
from datetime import datetime
import random
from typing import Dict, Any

logger = logging.getLogger(__name__)


def create_kira_routes(system_state: Dict[str, Any], services: Dict[str, Any]) -> Blueprint:
    """
    Create Kira-specific API routes blueprint
    
    Args:
        system_state: Current system state
        services: Available services
        
    Returns:
        Blueprint with Kira routes
    """
    
    kira_bp = Blueprint('kira_api', __name__, url_prefix='/api/kira')
    
    @kira_bp.route('/status', methods=['GET'])
    def get_kira_status():
        """Get Kira's current status and capabilities"""
        try:
            kira_status = {
                'name': 'Kira',
                'version': '2.0',
                'status': 'active' if system_state.get('kira_ready') else 'initializing',
                'capabilities': _get_kira_capabilities(services),
                'personality': _get_kira_personality(),
                'current_mood': _get_current_mood(),
                'last_interaction': _get_last_interaction_info(services),
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify({
                'success': True,
                'kira': kira_status
            })
            
        except Exception as e:
            logger.error(f"Kira status error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @kira_bp.route('/brain-activity', methods=['GET'])
    def get_brain_activity():
        """Get simulated Kira brain activity data"""
        try:
            # Generate realistic brain activity simulation
            brain_activity = _generate_brain_activity(services)
            
            return jsonify({
                'success': True,
                'brain_activity': brain_activity,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Brain activity error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @kira_bp.route('/personality', methods=['GET'])
    def get_personality():
        """Get Kira's personality profile"""
        try:
            personality = {
                'traits': {
                    'friendliness': 85,
                    'intelligence': 95,
                    'creativity': 80,
                    'empathy': 90,
                    'curiosity': 88,
                    'helpfulness': 95
                },
                'communication_style': {
                    'formality': 'casual_professional',
                    'emoji_usage': 'moderate',
                    'response_length': 'adaptive',
                    'language_preference': 'german_primary'
                },
                'specialties': [
                    'Problemlösung',
                    'Technische Hilfe',
                    'Kreative Unterstützung',
                    'Emotionale Intelligenz',
                    'Wissensmanagement'
                ],
                'memory_retention': {
                    'conversation_history': system_state.get('systems_status', {}).get('memory_system', {}).get('available', False),
                    'emotional_context': services.get('memory') is not None,
                    'user_preferences': True
                }
            }
            
            return jsonify({
                'success': True,
                'personality': personality
            })
            
        except Exception as e:
            logger.error(f"Personality error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @kira_bp.route('/memory-stats', methods=['GET'])
    def get_memory_stats():
        """Get Kira's memory statistics"""
        try:
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            # Get memory statistics
            memory_stats = {
                'total_conversations': 0,
                'total_interactions': 0,
                'memory_categories': {},
                'recent_activity': [],
                'emotional_data': {}
            }
            
            # Try to get actual stats from memory service
            if hasattr(memory_service, 'get_conversation_history'):
                try:
                    recent_conversations = memory_service.get_conversation_history(limit=50)
                    memory_stats['total_conversations'] = len(recent_conversations)
                    memory_stats['recent_activity'] = recent_conversations[-5:] if recent_conversations else []
                except Exception as e:
                    logger.warning(f"Could not get conversation history: {e}")
            
            return jsonify({
                'success': True,
                'memory_stats': memory_stats,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Memory stats error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @kira_bp.route('/voice-settings', methods=['GET'])
    def get_voice_settings():
        """Get Kira's voice configuration"""
        try:
            voice_service = services.get('voice')
            
            if not voice_service:
                return jsonify({
                    'success': False,
                    'error': 'Voice service not available'
                }), 503
            
            voice_settings = voice_service.get_status() if hasattr(voice_service, 'get_status') else {}
            
            return jsonify({
                'success': True,
                'voice_settings': voice_settings
            })
            
        except Exception as e:
            logger.error(f"Voice settings error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @kira_bp.route('/mood', methods=['POST'])
    def set_mood():
        """Set Kira's current mood"""
        try:
            data = request.get_json()
            new_mood = data.get('mood', 'neutral')
            
            # Validate mood
            valid_moods = ['happy', 'neutral', 'excited', 'focused', 'creative', 'analytical']
            if new_mood not in valid_moods:
                return jsonify({
                    'success': False,
                    'error': f'Invalid mood. Valid moods: {valid_moods}'
                }), 400
            
            # Here you would update Kira's mood in the system
            # For now, just return success
            
            return jsonify({
                'success': True,
                'message': f'Kira\'s mood set to {new_mood}',
                'new_mood': new_mood,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Set mood error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @kira_bp.route('/test-response', methods=['POST'])
    def test_response():
        """Test Kira's response system"""
        try:
            data = request.get_json()
            test_message = data.get('message', 'Hello Kira!')
            
            # Get LM Studio service
            lm_service = services.get('lm_studio')
            
            if not lm_service or not lm_service.is_available:
                # Fallback response
                response = "Hi! Ich bin Kira, aber mein LM Studio ist gerade nicht verfügbar. Trotzdem freue ich mich, von dir zu hören! 😊"
                return jsonify({
                    'success': True,
                    'response': response,
                    'source': 'fallback',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Get response from LM Studio
            chat_result = lm_service.chat_with_kira(test_message)
            
            if chat_result['success']:
                return jsonify({
                    'success': True,
                    'response': chat_result['response'],
                    'source': 'lm_studio',
                    'usage': chat_result.get('usage', {}),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'success': False,
                    'error': chat_result.get('error', 'Unknown error')
                }), 500
            
        except Exception as e:
            logger.error(f"Test response error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def _get_kira_capabilities(services: Dict) -> Dict:
        """Get Kira's current capabilities based on available services"""
        capabilities = {
            'chat': services.get('lm_studio') is not None and getattr(services.get('lm_studio'), 'is_available', False),
            'voice_synthesis': services.get('voice') is not None and getattr(services.get('voice'), 'is_initialized', False),
            'voice_recognition': False,  # Would check voice service capabilities
            'memory_retention': services.get('memory') is not None,
            'emotion_analysis': services.get('memory') is not None,  # Assuming emotion is part of memory
            'learning': True,  # Kira can always learn from conversations
            'multilingual': True
        }
        
        return capabilities
    
    def _get_kira_personality() -> Dict:
        """Get Kira's personality description"""
        return {
            'description': 'Intelligent, helpful, and friendly AI assistant',
            'primary_traits': ['helpful', 'intelligent', 'empathetic', 'creative'],
            'communication_style': 'Natural and authentic',
            'language_preference': 'German (with multilingual support)',
            'interaction_approach': 'Solution-oriented and supportive'
        }
    
    def _get_current_mood() -> str:
        """Get Kira's current mood (simulated)"""
        moods = ['focused', 'creative', 'helpful', 'analytical', 'cheerful']
        return random.choice(moods)
    
    def _get_last_interaction_info(services: Dict) -> Dict:
        """Get information about last interaction"""
        memory_service = services.get('memory')
        
        if memory_service and hasattr(memory_service, 'get_conversation_history'):
            try:
                recent = memory_service.get_conversation_history(limit=1)
                if recent:
                    return {
                        'timestamp': recent[0].get('timestamp', 'unknown'),
                        'type': 'conversation',
                        'success': True
                    }
            except:
                pass
        
        return {
            'timestamp': None,
            'type': 'none',
            'success': False
        }
    
    def _generate_brain_activity(services: Dict) -> Dict:
        """Generate simulated brain activity data"""
        # Simulate different brain regions and their activity
        base_activity = {
            'language_processing': random.randint(60, 95),
            'memory_access': random.randint(40, 80),
            'logical_reasoning': random.randint(70, 90),
            'creativity': random.randint(50, 85),
            'emotional_analysis': random.randint(30, 70),
            'pattern_recognition': random.randint(65, 90)
        }
        
        # Adjust based on active services
        if services.get('lm_studio') and getattr(services.get('lm_studio'), 'is_available', False):
            base_activity['language_processing'] = min(95, base_activity['language_processing'] + 15)
            base_activity['logical_reasoning'] = min(95, base_activity['logical_reasoning'] + 10)
        
        if services.get('memory'):
            base_activity['memory_access'] = min(90, base_activity['memory_access'] + 20)
        
        if services.get('voice'):
            base_activity['language_processing'] = min(95, base_activity['language_processing'] + 10)
        
        return {
            'regions': base_activity,
            'overall_activity': sum(base_activity.values()) / len(base_activity),
            'active_processes': [k for k, v in base_activity.items() if v > 70],
            'timestamp': datetime.now().isoformat()
        }
    
    return kira_bp