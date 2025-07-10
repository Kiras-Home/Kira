"""
Enhanced Chat Routes - Service-based architecture for app factory compatibility
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def create_chat_routes(system_state: Dict[str, Any], services: Dict[str, Any]) -> Blueprint:
    """
    Create enhanced chat routes with memory integration
    
    Args:
        system_state: Current system state
        services: Available services
        
    Returns:
        Blueprint with chat routes
    """
    
    # Create blueprint
    chat_bp = Blueprint('chat', __name__, url_prefix='/api/chat')
    
    @chat_bp.route('/message', methods=['POST'])
    def chat_message():
        """Enhanced Chat endpoint for messages"""
        try:
            data = request.get_json()
            user_message = data.get('message', '').strip()
            conversation_id = data.get('conversation_id', f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            user_name = data.get('user_name', 'User')
            
            if not user_message:
                return jsonify({'success': False, 'error': 'No message provided'}), 400

            logger.info(f"💬 Processing message from {user_name}: {user_message[:50]}...")

            # ✅ GENERATE KIRA RESPONSE
            kira_response = generate_kira_response(user_message, services)
            
            # ✅ ENHANCED RESPONSE
            response_data = {
                'success': True,
                'response': kira_response,
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'user_name': user_name,
                'system_info': {
                    'memory_available': bool(services.get('memory')),
                    'lm_studio_connected': bool(services.get('lm_studio')),
                    'voice_system_available': bool(services.get('voice')),
                    'learning_active': bool(services.get('memory'))
                }
            }
            
            # ✅ UPDATE SYSTEM STATE
            update_system_interaction(system_state, user_message, kira_response, conversation_id)
            
            logger.info(f"✅ Chat response generated successfully")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"❌ Chat message error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    @chat_bp.route('/status', methods=['GET'])
    def chat_status():
        """Get chat system status"""
        try:
            return jsonify({
                'success': True,
                'system_status': {
                    'memory_service': bool(services.get('memory')),
                    'lm_studio_service': bool(services.get('lm_studio')),
                    'voice_service': bool(services.get('voice')),
                    'last_interaction': system_state.get('last_interaction', {}),
                    'total_interactions': system_state.get('interaction_count', 0)
                },
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"❌ Chat status error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return chat_bp


def generate_kira_response(user_message: str, services: Dict[str, Any]) -> str:
    """
    Generate Kira's response to user message
    
    Args:
        user_message: User's input message
        services: Available services
        
    Returns:
        Kira's response text
    """
    try:
        # Check if LM Studio service is available
        lm_studio_service = services.get('lm_studio')
        
        if lm_studio_service:
            # Try to get response from LM Studio
            try:
                if hasattr(lm_studio_service, 'generate_response'):
                    response = lm_studio_service.generate_response(user_message)
                    if response:
                        return response
                elif hasattr(lm_studio_service, 'get_response'):
                    response = lm_studio_service.get_response(user_message)
                    if response:
                        return response
            except Exception as e:
                logger.warning(f"LM Studio service error: {e}")
        
        # Fallback: Enhanced context-aware responses
        return generate_fallback_response(user_message)
        
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return f"I encountered an error while processing your message. Please try again."


def generate_fallback_response(user_message: str) -> str:
    """
    Generate fallback response when LM Studio is not available
    
    Args:
        user_message: User's input message
        
    Returns:
        Fallback response
    """
    user_lower = user_message.lower()
    
    # Greeting patterns
    if any(greeting in user_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        return "Hello! I'm Kira, your AI assistant. How can I help you today?"
    
    # Help patterns
    if any(help_word in user_lower for help_word in ['help', 'assist', 'support']):
        return "I'm here to help! You can ask me questions, have conversations, or request assistance with various tasks. What would you like to know?"
    
    # Question patterns
    if user_message.strip().endswith('?'):
        return f"That's an interesting question about '{user_message}'. I'd be happy to help you explore that topic further."
    
    # System status
    if any(status_word in user_lower for status_word in ['status', 'how are you', 'working']):
        return "I'm functioning well and ready to assist you. My systems are online and I'm here to help with whatever you need."
    
    # Default response
    return f"I understand you're saying: '{user_message}'. I'm currently running in fallback mode. For full AI capabilities, please ensure LM Studio is connected and running."


def update_system_interaction(system_state: Dict[str, Any], user_message: str, kira_response: str, conversation_id: str):
    """
    Update system state with interaction information
    
    Args:
        system_state: System state dictionary
        user_message: User's message
        kira_response: Kira's response
        conversation_id: Conversation ID
    """
    try:
        # Update last interaction
        if 'last_interaction' not in system_state:
            system_state['last_interaction'] = {}
        
        system_state['last_interaction'].update({
            'user_message': user_message,
            'kira_response': kira_response,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update interaction count
        if 'interaction_count' not in system_state:
            system_state['interaction_count'] = 0
        system_state['interaction_count'] += 1
        
        logger.debug(f"System interaction updated: {system_state['interaction_count']} total interactions")
        
    except Exception as e:
        logger.error(f"Error updating system interaction: {e}")
