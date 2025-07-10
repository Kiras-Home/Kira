"""
Enhanced Chat Routes - Service-based architecture for app factory compatibility
"""

from flask import Blueprint, jsonify, request, render_template
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
    
    # Add template routes
    @chat_bp.route('/page')
    def chat_page():
        """Modern chat page"""
        return render_template('modern-chat.html')
    
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
            
            # ✅ MEMORY INTEGRATION - Store conversation
            memory_result = store_conversation_in_memory(user_message, kira_response, conversation_id, services)
            
            # ✅ GENERATE AUDIO RESPONSE
            audio_data = generate_audio_response(kira_response, services)
            
            # ✅ ENHANCED RESPONSE
            response_data = {
                'success': True,
                'response': kira_response,
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'user_name': user_name,
                'audio': audio_data,  # Include audio data
                'memory_processing': memory_result,  # Include memory processing result
                'system_info': {
                    'memory_available': bool(services.get('memory')),
                    'memory_stored': memory_result.get('success', False),
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


def store_conversation_in_memory(user_message: str, kira_response: str, conversation_id: str, services: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store conversation in memory system
    
    Args:
        user_message: User's input message
        kira_response: Kira's response
        conversation_id: Conversation ID
        services: Available services
        
    Returns:
        Memory storage result
    """
    try:
        memory_service = services.get('memory')
        
        if not memory_service:
            logger.warning("Memory service not available for conversation storage")
            return {
                'success': False,
                'error': 'Memory service not available'
            }
        
        # Check if memory service has add_conversation method
        if hasattr(memory_service, 'add_conversation'):
            try:
                logger.info(f"💾 Storing conversation in memory: {user_message[:30]}...")
                
                metadata = {
                    'conversation_id': conversation_id,
                    'timestamp': datetime.now().isoformat(),
                    'user_message_length': len(user_message),
                    'kira_response_length': len(kira_response),
                    'message_type': 'chat_interaction'
                }
                
                memory_service.add_conversation(user_message, kira_response, metadata)
                
                logger.info("✅ Conversation stored in memory successfully")
                return {
                    'success': True,
                    'stored': True,
                    'conversation_id': conversation_id
                }
                
            except Exception as e:
                logger.error(f"❌ Error storing conversation: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
        
        # Try alternative memory methods
        elif hasattr(memory_service, 'conversation_memory'):
            try:
                conversation_memory = memory_service.conversation_memory
                
                if hasattr(conversation_memory, 'add_interaction'):
                    result = conversation_memory.add_interaction(
                        user_input=user_message,
                        assistant_response=kira_response,
                        metadata={
                            'conversation_id': conversation_id,
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                    
                    logger.info("✅ Conversation stored via conversation_memory")
                    return {
                        'success': True,
                        'stored': True,
                        'result': result
                    }
                    
            except Exception as e:
                logger.error(f"❌ Error with conversation_memory: {e}")
        
        # Try memory manager if available
        elif hasattr(memory_service, 'memory_manager'):
            try:
                memory_manager = memory_service.memory_manager
                
                if hasattr(memory_manager, 'store_conversation'):
                    result = memory_manager.store_conversation(
                        user_message, kira_response, conversation_id
                    )
                    
                    logger.info("✅ Conversation stored via memory_manager")
                    return {
                        'success': True,
                        'stored': True,
                        'result': result
                    }
                    
            except Exception as e:
                logger.error(f"❌ Error with memory_manager: {e}")
        
        logger.warning("No suitable memory storage method found")
        return {
            'success': False,
            'error': 'No suitable memory storage method available'
        }
        
    except Exception as e:
        logger.error(f"❌ Memory storage error: {e}")
        return {
            'success': False,
            'error': str(e)
        }


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
                if hasattr(lm_studio_service, 'chat_with_kira'):
                    logger.info(f"🤖 Sending message to LM Studio: {user_message[:50]}...")
                    response_data = lm_studio_service.chat_with_kira(user_message)
                    
                    if response_data.get('success') and response_data.get('response'):
                        logger.info("✅ LM Studio response received successfully")
                        return response_data['response']
                    else:
                        logger.warning(f"LM Studio response failed: {response_data.get('error', 'Unknown error')}")
                        
                elif hasattr(lm_studio_service, 'generate_response'):
                    response = lm_studio_service.generate_response(user_message)
                    if response:
                        return response
                elif hasattr(lm_studio_service, 'get_response'):
                    response = lm_studio_service.get_response(user_message)
                    if response:
                        return response
                else:
                    logger.warning("LM Studio service has no recognized response method")
                    
            except Exception as e:
                logger.warning(f"LM Studio service error: {e}")
        else:
            logger.warning("LM Studio service not available in services dict")
        
        # Fallback: Enhanced context-aware responses
        logger.info("Using fallback response generation")
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
        
        # Update conversations tracking
        if 'conversations' not in system_state:
            system_state['conversations'] = {}
        
        if conversation_id not in system_state['conversations']:
            system_state['conversations'][conversation_id] = {
                'created_at': datetime.now().isoformat(),
                'message_count': 0,
                'last_activity': datetime.now().isoformat()
            }
        
        # Update conversation data
        system_state['conversations'][conversation_id]['message_count'] += 1
        system_state['conversations'][conversation_id]['last_activity'] = datetime.now().isoformat()
        
        # Set system uptime if not set
        if 'uptime' not in system_state:
            system_state['uptime'] = datetime.now().isoformat()
        
        logger.debug(f"System interaction updated: {system_state['interaction_count']} total interactions, {len(system_state['conversations'])} conversations")
        
    except Exception as e:
        logger.error(f"Error updating system interaction: {e}")


def generate_audio_response(text: str, services: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate audio response using Enterprise Voice service
    
    Args:
        text: Text to convert to speech
        services: Available services
        
    Returns:
        Audio response data
    """
    try:
        voice_service = services.get('voice')
        
        if not voice_service:
            logger.warning("Voice service not available")
            return {
                'success': False,
                'error': 'Voice service not available'
            }
        
        # Try to use the 'speak' method directly if available
        if hasattr(voice_service, 'speak'):
            try:
                logger.info(f"🔊 Attempting voice service speak: '{text[:50]}...'")
                audio_result = voice_service.speak(text, emotion="calm")  # Remove auto_play argument
                
                if audio_result and audio_result.get('success'):
                    logger.info("✅ Voice service speak successful")
                    return {
                        'success': True,
                        'audio_url': audio_result.get('audio_url'),
                        'filename': audio_result.get('filename'),
                        'duration_estimate': audio_result.get('duration_estimate', len(text) * 0.1)
                    }
                else:
                    logger.warning(f"❌ Voice service speak failed: {audio_result}")
            except Exception as e:
                logger.warning(f"❌ Voice service speak error: {e}")
                
        # Try to synthesize speech directly through voice service
        if hasattr(voice_service, 'synthesize_speech'):
            try:
                logger.info(f"🔊 Attempting direct voice synthesis: '{text[:50]}...'")
                audio_result = voice_service.synthesize_speech(text, emotion="calm")
                
                if audio_result and audio_result.get('success'):
                    logger.info("✅ Direct voice synthesis successful")
                    return {
                        'success': True,
                        'audio_url': audio_result.get('audio_url'),
                        'filename': audio_result.get('filename'),
                        'duration_estimate': audio_result.get('duration_estimate', len(text) * 0.1)
                    }
                else:
                    logger.warning(f"❌ Direct voice synthesis failed: {audio_result}")
            except Exception as e:
                logger.warning(f"❌ Direct voice synthesis error: {e}")
        
        # Check Enterprise Voice Manager
        if hasattr(voice_service, 'voice_manager'):
            voice_manager = voice_service.voice_manager
            logger.info("🔍 Checking Enterprise Voice Manager")
            
            # Try synthesize method
            if hasattr(voice_manager, 'synthesize'):
                try:
                    logger.info("🎤 Attempting Enterprise Voice Manager synthesis")
                    audio_result = voice_manager.synthesize(text, emotion="calm")
                    
                    if audio_result and audio_result.get('success'):
                        audio_path = audio_result.get('audio_path')
                        if audio_path:
                            audio_filename = audio_path.split('/')[-1]
                            logger.info(f"✅ Enterprise synthesis successful: {audio_filename}")
                            return {
                                'success': True,
                                'audio_url': f"/api/audio/{audio_filename}",
                                'filename': audio_filename,
                                'duration_estimate': len(text) * 0.1
                            }
                except Exception as e:
                    logger.warning(f"❌ Enterprise Voice Manager synthesis error: {e}")
            
            # Check for Bark engine access
            if hasattr(voice_manager, 'bark_engine') and voice_manager.bark_engine:
                bark_engine = voice_manager.bark_engine
                logger.info("🗣️ Found Bark engine in voice manager")
                
                # Try various bark synthesis methods
                for method_name in ['synthesize', 'generate_speech', 'speak', 'generate']:
                    if hasattr(bark_engine, method_name):
                        try:
                            logger.info(f"🎯 Trying Bark method: {method_name}")
                            method = getattr(bark_engine, method_name)
                            
                            if method_name in ['synthesize', 'generate_speech']:
                                audio_result = method(text, emotion="calm")
                            elif method_name == 'speak':
                                audio_result = method(text, emotion="calm", auto_play=False)
                            else:
                                audio_result = method(text)
                            
                            if audio_result:
                                if isinstance(audio_result, dict) and audio_result.get('success'):
                                    audio_path = audio_result.get('audio_path')
                                    if audio_path:
                                        audio_filename = audio_path.split('/')[-1]
                                        logger.info(f"✅ Bark {method_name} successful: {audio_filename}")
                                        return {
                                            'success': True,
                                            'audio_url': f"/api/audio/{audio_filename}",
                                            'filename': audio_filename,
                                            'duration_estimate': len(text) * 0.1
                                        }
                                elif hasattr(audio_result, 'name'):  # Path object
                                    logger.info(f"✅ Bark {method_name} returned path: {audio_result.name}")
                                    return {
                                        'success': True,
                                        'audio_url': f"/api/audio/{audio_result.name}",
                                        'filename': audio_result.name,
                                        'duration_estimate': len(text) * 0.1
                                    }
                                elif isinstance(audio_result, str):  # String path
                                    audio_filename = audio_result.split('/')[-1]
                                    logger.info(f"✅ Bark {method_name} returned string: {audio_filename}")
                                    return {
                                        'success': True,
                                        'audio_url': f"/api/audio/{audio_filename}",
                                        'filename': audio_filename,
                                        'duration_estimate': len(text) * 0.1
                                    }
                                    
                        except Exception as e:
                            logger.warning(f"❌ Bark {method_name} failed: {e}")
                            continue
        
        # Log available methods for debugging
        if voice_service:
            available_methods = [method for method in dir(voice_service) if not method.startswith('_')]
            logger.info(f"🔍 Available voice service methods: {available_methods[:10]}...")  # First 10 methods
            
            # Try the speak method that we know exists
            if 'speak' in available_methods:
                try:
                    logger.info(f"🎤 Trying voice service speak method: '{text[:50]}...'")
                    speak_method = getattr(voice_service, 'speak')
                    audio_result = speak_method(text)
                    
                    if audio_result:
                        logger.info(f"✅ Voice service speak returned: {type(audio_result)}")
                        # Handle different return types
                        if isinstance(audio_result, dict) and audio_result.get('success'):
                            return {
                                'success': True,
                                'audio_url': audio_result.get('audio_url'),
                                'filename': audio_result.get('filename'),
                                'duration_estimate': audio_result.get('duration_estimate', len(text) * 0.1)
                            }
                        elif isinstance(audio_result, str):  # Path returned
                            audio_filename = audio_result.split('/')[-1]
                            return {
                                'success': True,
                                'audio_url': f"/api/audio/{audio_filename}",
                                'filename': audio_filename,
                                'duration_estimate': len(text) * 0.1
                            }
                        else:
                            logger.info(f"✅ Voice speak result: {audio_result}")
                    else:
                        logger.warning("❌ Voice service speak returned None/False")
                        
                except Exception as e:
                    logger.warning(f"❌ Voice service speak error: {e}")
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        logger.info("❌ No compatible audio generation method found")
        return {
            'success': False,
            'error': 'No compatible audio generation method found'
        }
        
    except Exception as e:
        logger.error(f"❌ Audio response generation error: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e)
        }
