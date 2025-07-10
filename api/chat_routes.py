"""
Kira Chat API Routes
Handles chat conversations, message processing and voice integration
"""

import logging
from flask import Blueprint, jsonify, request
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def create_chat_routes(system_state: Dict[str, Any], services: Dict[str, Any]) -> Blueprint:
    """
    Create chat API routes blueprint
    
    Args:
        system_state: Current system state
        services: Available services
        
    Returns:
        Blueprint with chat routes
    """
    
    chat_bp = Blueprint('chat_api', __name__, url_prefix='/api/chat')
    
    @chat_bp.route('/kira', methods=['POST'])
    def chat_with_kira():
        """Main chat endpoint for conversations with Kira"""
        try:
            data = request.get_json()
            
            if not data or 'message' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Message is required'
                }), 400
            
            user_message = data['message'].strip()
            if not user_message:
                return jsonify({
                    'success': False,
                    'error': 'Message cannot be empty'
                }), 400
            
            # Additional parameters
            include_voice = data.get('include_voice', False)
            conversation_id = data.get('conversation_id')
            user_emotion = data.get('emotion')
            
            # Process the chat message
            chat_result = _process_chat_message(
                user_message, 
                services, 
                conversation_id=conversation_id,
                user_emotion=user_emotion
            )
            
            # Add voice synthesis if requested
            if include_voice and chat_result['success']:
                voice_result = _add_voice_to_response(chat_result['response'], services, user_emotion)
                if voice_result['success']:
                    chat_result['voice'] = voice_result
            
            # Save conversation to memory
            if chat_result['success']:
                _save_conversation_to_memory(user_message, chat_result['response'], services, {
                    'conversation_id': conversation_id,
                    'user_emotion': user_emotion,
                    'timestamp': datetime.now().isoformat()
                })
            
            return jsonify(chat_result)
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @chat_bp.route('/voice', methods=['POST'])
    def chat_with_voice():
        """Chat endpoint that always includes voice response"""
        try:
            data = request.get_json()
            
            if not data or 'message' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Message is required'
                }), 400
            
            user_message = data['message'].strip()
            emotion = data.get('emotion', 'neutral')
            
            # Get text response
            chat_result = _process_chat_message(user_message, services)
            
            if not chat_result['success']:
                return jsonify(chat_result), 500
            
            # Generate voice response
            voice_result = _add_voice_to_response(chat_result['response'], services, emotion)
            
            return jsonify({
                'success': True,
                'text_response': chat_result['response'],
                'voice_response': voice_result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Voice chat error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @chat_bp.route('/history', methods=['GET'])
    def get_conversation_history():
        """Get conversation history"""
        try:
            # Get query parameters
            limit = request.args.get('limit', 20, type=int)
            conversation_id = request.args.get('conversation_id')
            
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            # Get conversation history
            if hasattr(memory_service, 'get_conversation_history'):
                try:
                    history = memory_service.get_conversation_history(limit=limit)
                    
                    # Filter by conversation_id if provided
                    if conversation_id:
                        history = [h for h in history if h.get('conversation_id') == conversation_id]
                    
                    return jsonify({
                        'success': True,
                        'history': history,
                        'total_count': len(history)
                    })
                    
                except Exception as e:
                    logger.error(f"Error getting conversation history: {e}")
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 500
            else:
                return jsonify({
                    'success': False,
                    'error': 'Conversation history not available'
                }), 503
            
        except Exception as e:
            logger.error(f"History error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @chat_bp.route('/emotion', methods=['POST'])
    def analyze_emotion():
        """Analyze emotion in text"""
        try:
            data = request.get_json()
            
            if not data or 'text' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Text is required'
                }), 400
            
            text = data['text']
            
            # Simple emotion analysis (would be replaced with actual emotion service)
            emotion_result = _analyze_text_emotion(text)
            
            return jsonify({
                'success': True,
                'emotion_analysis': emotion_result,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @chat_bp.route('/typing', methods=['POST'])
    def typing_indicator():
        """Handle typing indicator for real-time chat"""
        try:
            data = request.get_json()
            conversation_id = data.get('conversation_id')
            is_typing = data.get('is_typing', False)
            
            # This would normally update a real-time typing status
            # For now, just return success
            
            return jsonify({
                'success': True,
                'conversation_id': conversation_id,
                'typing_status': is_typing,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Typing indicator error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def _process_chat_message(user_message: str, services: Dict, conversation_id: Optional[str] = None, user_emotion: Optional[str] = None) -> Dict[str, Any]:
        """Process a chat message and get Kira's response"""
        try:
            lm_service = services.get('lm_studio')
            memory_service = services.get('memory')
            
            # Check if LM Studio is available
            if not lm_service or not getattr(lm_service, 'is_available', False):
                # Fallback response when LM Studio is not available
                fallback_responses = [
                    "Entschuldige, mein Hauptsystem ist gerade nicht verfügbar, aber ich bin trotzdem hier für dich! 😊",
                    "Hi! Ich arbeite gerade im Fallback-Modus. Wie kann ich dir helfen?",
                    "Mein LM Studio ist offline, aber ich kann dir trotzdem mit einfachen Fragen helfen!",
                ]
                
                import random
                response = random.choice(fallback_responses)
                
                return {
                    'success': True,
                    'response': response,
                    'source': 'fallback',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get conversation history for context
            conversation_history = []
            if memory_service and hasattr(memory_service, 'get_conversation_history'):
                try:
                    conversation_history = memory_service.get_conversation_history(limit=5)
                except Exception as e:
                    logger.warning(f"Could not get conversation history: {e}")
            
            # Get response from LM Studio
            chat_result = lm_service.chat_with_kira(user_message, conversation_history)
            
            if chat_result['success']:
                return {
                    'success': True,
                    'response': chat_result['response'],
                    'source': 'lm_studio',
                    'model': chat_result.get('model'),
                    'usage': chat_result.get('usage', {}),
                    'conversation_id': conversation_id,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': chat_result.get('error', 'Unknown LM Studio error'),
                    'source': 'lm_studio'
                }
                
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _add_voice_to_response(text_response: str, services: Dict, emotion: Optional[str] = None) -> Dict[str, Any]:
        """Add voice synthesis to text response"""
        try:
            voice_service = services.get('voice')
            
            if not voice_service or not getattr(voice_service, 'is_initialized', False):
                return {
                    'success': False,
                    'error': 'Voice service not available'
                }
            
            # Synthesize speech
            if hasattr(voice_service, 'synthesize_speech'):
                voice_result = voice_service.synthesize_speech(text_response, emotion)
                return voice_result
            else:
                return {
                    'success': False,
                    'error': 'Voice synthesis method not available'
                }
                
        except Exception as e:
            logger.error(f"Voice synthesis error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_conversation_to_memory(user_input: str, kira_response: str, services: Dict, metadata: Optional[Dict] = None):
        """Save conversation to memory service"""
        try:
            memory_service = services.get('memory')
            
            if memory_service and hasattr(memory_service, 'add_conversation'):
                memory_service.add_conversation(user_input, kira_response, metadata)
                
        except Exception as e:
            logger.warning(f"Could not save conversation to memory: {e}")
    
    def _analyze_text_emotion(text: str) -> Dict[str, Any]:
        """Simple emotion analysis (placeholder implementation)"""
        # This would be replaced with actual emotion analysis
        emotion_keywords = {
            'happy': ['freude', 'glücklich', 'toll', 'super', 'fantastisch', '😊', '😄', '🎉'],
            'sad': ['traurig', 'schlecht', 'müde', 'down', '😢', '😞'],
            'angry': ['ärgerlich', 'wütend', 'nervig', 'frustriert', '😠', '😡'],
            'excited': ['aufgeregt', 'begeistert', 'wow', 'amazing', '🤩', '✨'],
            'neutral': []
        }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score
        
        # Find dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        return {
            'dominant_emotion': dominant_emotion[0] if dominant_emotion[1] > 0 else 'neutral',
            'emotion_scores': emotion_scores,
            'confidence': min(dominant_emotion[1] * 0.3, 1.0),  # Simple confidence calculation
            'analysis_method': 'keyword_based'
        }
    
    return chat_bp