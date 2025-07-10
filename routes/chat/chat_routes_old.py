"""
Enhanced Chat Routes mit integriertem Conversation Memory System
Service-based architecture compatible with app factory
"""

from flask import Blueprint, jsonify, request, render_template
from datetime import datetime, timedelta
import asyncio
import logging
import json
from typing import Dict, Any, Optional

# ✅ MEMORY SYSTEM IMPORTS
from memory.core.conversation_memory import ConversationMemorySystem
from memory.core.short_term_memory import HumanLikeShortTermMemory
from memory.core.long_term_memory import HumanLikeLongTermMemory

logger = logging.getLogger(__name__)

# ✅ GLOBALES CONVERSATION MEMORY SYSTEM
conversation_memory: Optional[ConversationMemorySystem] = None

# Service containers - will be populated by create_chat_routes function
_memory_service = None
_lm_studio_service = None
_voice_service = None
_system_state = {}

def create_chat_routes(system_state: Dict[str, Any], services: Dict[str, Any]) -> Blueprint:
    """
    Create enhanced chat routes with memory integration
    
    Args:
        system_state: Current system state
        services: Available services
        
    Returns:
        Blueprint with chat routes
    """
    
    # Initialize service containers
    global _memory_service, _lm_studio_service, _voice_service, _system_state
    _memory_service = services.get('memory')
    _lm_studio_service = services.get('lm_studio')
    _voice_service = services.get('voice')
    _system_state = system_state
    
    # Initialize conversation memory
    initialize_conversation_memory()
    
    # Create blueprint
    chat_bp = Blueprint('chat', __name__, url_prefix='/api/chat')
    
    @chat_bp.route('/message', methods=['POST'])
    def chat_message():
        """Enhanced Chat mit Memory Integration"""
        try:
            data = request.get_json()
            user_message = data.get('message', '').strip()
            conversation_id = data.get('conversation_id', f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            user_name = data.get('user_name', 'User')
            session_data = data.get('session', {})
            
            if not user_message:
                return jsonify({'success': False, 'error': 'No message provided'}), 400

            logger.info(f"💬 Processing message from {user_name}: {user_message[:50]}...")

            # ✅ GENERATE KIRA RESPONSE (Enhanced)
            kira_response = generate_kira_response(user_message, conversation_id, session_data)
            
            # ✅ MEMORY PROCESSING
            memory_result = {}
            if conversation_memory:
                try:
                    # Erstelle Conversation Context
                    conversation_context = {
                        'conversation_id': conversation_id,
                        'user_name': user_name,
                        'session_duration_minutes': session_data.get('duration_minutes', 0),
                        'user_initiated': True,
                        'timestamp': datetime.now().isoformat(),
                        'follow_up_count': session_data.get('message_count', 0),
                        'conversation_depth': len(session_data.get('message_history', [])),
                        'user_engagement': calculate_user_engagement(session_data)
                    }
                    
                    # Asynchrone Memory Processing
                    memory_result = asyncio.run(conversation_memory.process_conversation(
                        user_input=user_message,
                        kira_response=kira_response,
                        context=conversation_context
                ))
                
                logger.info(f"🧠 Memory processed: importance={memory_result.get('importance_score', 0):.2f}, "
                          f"storage={memory_result.get('storage_location', 'none')}")
                
            except Exception as e:
                logger.error(f"⚠️ Memory processing failed: {e}")
                memory_result = {
                    'error': str(e),
                    'importance_score': 0.0,
                    'storage_location': 'failed'
                }
        else:
            memory_result = {
                'error': 'Memory system not available',
                'importance_score': 0.0,
                'storage_location': 'none'
            }

        # ✅ UPDATE SYSTEM STATE
        update_system_interaction_state(user_message, kira_response, memory_result)

        # ✅ ENHANCED RESPONSE
        response_data = {
            'success': True,
            'response': kira_response,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'memory_processing': {
                'processed': memory_result.get('error') is None,
                'importance_score': memory_result.get('importance_score', 0.0),
                'storage_location': memory_result.get('storage_location', 'none'),
                'emotional_impact': memory_result.get('emotional_impact', 0.0),
                'memory_ids': memory_result.get('memory_ids', [])
            },
            'ai_status': {
                'confidence': calculate_ai_confidence(kira_response),
                'response_time': '1.2s',  # Could be measured
                'learning_active': bool(_memory_system and conversation_memory)
            },
            'conversation_meta': {
                'topic_detected': memory_result.get('conversation_summary', {}).get('topic_category', 'general'),
                'conversation_type': memory_result.get('conversation_summary', {}).get('conversation_type', 'casual'),
                'follow_up_suggested': should_suggest_followup(user_message, kira_response)
            }
        }

        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Chat message error: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def generate_kira_response(user_message: str, conversation_id: str, session_data: Dict[str, Any]) -> str:
    """Generiert Kira Response (Enhanced mit Memory Context)"""
    
    # ✅ MEMORY CONTEXT INTEGRATION
    memory_context = ""
    if conversation_memory:
        try:
            # Hole relevante Memories für Context
            relevant_memories = asyncio.run(
                conversation_memory.search_conversations(user_message, limit=3)
            )
            
            if relevant_memories:
                memory_context = " [Memory Context: " + "; ".join([
                    f"{mem.content[:50]}..." for mem in relevant_memories
                ]) + "]"
                
        except Exception as e:
            logger.warning(f"Memory context retrieval failed: {e}")
    
    # ✅ LM STUDIO INTEGRATION (wenn verfügbar)
    if _lm_studio_integration:
        try:
            enhanced_prompt = f"""
            User Message: {user_message}
            Conversation ID: {conversation_id}
            Previous Context: {memory_context}
            
            Please respond as Kira, a helpful AI assistant. Be conversational and remember our context.
            """
            
            # Hier würde der echte LM Studio Call stehen
            # kira_response = _lm_studio_integration.generate_response(enhanced_prompt)
            
            # Fallback für Demo
            kira_response = generate_contextual_response(user_message, memory_context, session_data)
            
        except Exception as e:
            logger.warning(f"LM Studio response failed: {e}")
            kira_response = generate_contextual_response(user_message, memory_context, session_data)
    else:
        kira_response = generate_contextual_response(user_message, memory_context, session_data)
    
    return kira_response

def generate_contextual_response(user_message: str, memory_context: str, session_data: Dict[str, Any]) -> str:
    """Generiert kontextuelle Antwort basierend auf Message und Memory"""
    
    user_lower = user_message.lower()
    
    # ✅ MEMORY-AWARE RESPONSES
    if memory_context:
        memory_indicator = "I remember our previous conversations about this. "
    else:
        memory_indicator = ""
    
    # ✅ CONTEXTUAL RESPONSES
    if any(word in user_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        return f"{memory_indicator}Hello! It's great to see you again. How can I help you today?"
        
    elif any(word in user_lower for word in ['how are you', 'how do you feel']):
        return f"{memory_indicator}I'm doing well, thank you for asking! I'm here and ready to help. How are you feeling today?"
        
    elif any(word in user_lower for word in ['memory', 'remember', 'recall', 'forget']):
        memory_info = ""
        if conversation_memory:
            stats = conversation_memory.get_memory_stats()
            memory_info = f" I currently have {stats.get('stm_capacity', 0)} items in my short-term memory and {stats.get('ltm_total_memories', 0)} consolidated memories."
        
        return f"{memory_indicator}Yes, I have a sophisticated memory system that helps me remember our conversations.{memory_info} What would you like to know about my memory capabilities?"
        
    elif any(word in user_lower for word in ['help', 'assist', 'support']):
        return f"{memory_indicator}I'm here to help! I can assist with questions, provide information, help with problems, or just have a conversation. What do you need help with?"
        
    elif any(word in user_lower for word in ['learn', 'teach', 'explain', 'understand']):
        return f"{memory_indicator}I love learning and teaching! I'm constantly learning from our interactions and I'm happy to explain things. What would you like to learn about or what can I help you understand?"
        
    elif any(word in user_lower for word in ['feel', 'emotion', 'sad', 'happy', 'excited', 'worried']):
        return f"{memory_indicator}I can sense that you're sharing something emotional with me. I'm here to listen and support you. Would you like to talk more about how you're feeling?"
        
    elif any(word in user_lower for word in ['plan', 'future', 'goal', 'project']):
        return f"{memory_indicator}Planning and goal-setting are so important! I can help you think through your plans and break them down into manageable steps. What are you working on?"
        
    elif '?' in user_message:
        return f"{memory_indicator}That's a great question! Let me think about that... {generate_question_response(user_message)}"
        
    else:
        # Default response
        return f"{memory_indicator}I understand what you're saying. That's interesting! Can you tell me more about that, or is there something specific I can help you with?"

def generate_question_response(question: str) -> str:
    """Generiert spezifische Antworten auf Fragen"""
    
    q_lower = question.lower()
    
    if any(word in q_lower for word in ['what', 'define', 'meaning']):
        return "I can help explain that concept. Could you be more specific about what aspect you'd like me to define or explain?"
        
    elif any(word in q_lower for word in ['how', 'way', 'method']):
        return "There are usually several approaches to consider. Let me help you think through the best way to handle this."
        
    elif any(word in q_lower for word in ['why', 'reason', 'because']):
        return "That's a thoughtful question about the reasoning behind something. Let me help you understand the 'why' behind this."
        
    elif any(word in q_lower for word in ['when', 'time', 'schedule']):
        return "Timing is important! Let me help you think about the best timing for this."
        
    elif any(word in q_lower for word in ['where', 'location', 'place']):
        return "Location and context matter. Let me help you figure out the best approach for your situation."
        
    else:
        return "I'm processing your question and want to give you a thoughtful response."

def calculate_user_engagement(session_data: Dict[str, Any]) -> float:
    """Berechnet User Engagement Score"""
    engagement = 0.5  # Baseline
    
    message_count = session_data.get('message_count', 0)
    if message_count > 1:
        engagement += min(0.3, message_count * 0.05)
    
    duration = session_data.get('duration_minutes', 0)
    if duration > 5:
        engagement += min(0.2, duration * 0.02)
    
    return min(1.0, engagement)

def calculate_ai_confidence(response: str) -> float:
    """Berechnet AI Confidence basierend auf Response"""
    confidence = 0.8  # Baseline
    
    # Länge der Antwort
    if len(response) > 50:
        confidence += 0.1
    if len(response) > 100:
        confidence += 0.05
        
    # Definiteness indicators
    if any(phrase in response.lower() for phrase in ['i can', 'i will', 'certainly', 'definitely']):
        confidence += 0.05
        
    return min(1.0, confidence)

def should_suggest_followup(user_message: str, kira_response: str) -> bool:
    """Bestimmt ob Follow-up vorgeschlagen werden soll"""
    
    # Suggest follow-up for questions
    if '?' in user_message:
        return True
        
    # Suggest follow-up for learning topics
    if any(word in user_message.lower() for word in ['learn', 'understand', 'explain']):
        return True
        
    # Suggest follow-up for emotional content
    if any(word in user_message.lower() for word in ['feel', 'emotion', 'problem', 'help']):
        return True
        
    return False

def update_system_interaction_state(user_message: str, kira_response: str, memory_result: Dict[str, Any]):
    """Aktualisiert System State mit Interaction Info"""
    try:
        if 'last_interaction' not in SYSTEM_STATE:
            SYSTEM_STATE['last_interaction'] = {}
            
        SYSTEM_STATE['last_interaction'].update({
            'timestamp': datetime.now().isoformat(),
            'user_message_length': len(user_message),
            'kira_response_length': len(kira_response),
            'memory_importance': memory_result.get('importance_score', 0.0),
            'memory_storage': memory_result.get('storage_location', 'none')
        })
        
        # Update interaction counter
        if 'interaction_count' not in SYSTEM_STATE:
            SYSTEM_STATE['interaction_count'] = 0
        SYSTEM_STATE['interaction_count'] += 1
        
    except Exception as e:
        logger.warning(f"System state update failed: {e}")

# ✅ MEMORY-SPECIFIC ROUTES

@chat_bp.route('/memory/stats', methods=['GET'])
def get_memory_stats():
    """Gibt Memory System Statistiken zurück"""
    try:
        if conversation_memory:
            stats = conversation_memory.get_memory_stats()
            
            # Erweitere mit System-spezifischen Stats
            enhanced_stats = {
                **stats,
                'system_integration': {
                    'lm_studio_connected': bool(_lm_studio_integration),
                    'voice_system_available': bool(_voice_system),
                    'main_memory_system_connected': bool(_memory_system)
                },
                'performance': {
                    'total_interactions': SYSTEM_STATE.get('interaction_count', 0),
                    'last_interaction': SYSTEM_STATE.get('last_interaction', {}).get('timestamp'),
                    'average_importance': calculate_average_importance()
                }
            }
            
            return jsonify({'success': True, 'memory_stats': enhanced_stats})
        else:
            return jsonify({
                'success': False, 
                'error': 'Memory system not available',
                'fallback_stats': get_fallback_memory_stats()
            }), 503
                
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@chat_bp.route('/memory/search', methods=['POST'])
def search_memories():
    """Durchsucht Conversation Memories"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        limit = data.get('limit', 10)
        
        if not query:
            return jsonify({'success': False, 'error': 'No search query provided'}), 400
            
        if conversation_memory:
            results = asyncio.run(conversation_memory.search_conversations(query, limit))
            
            # Konvertiere Memory-Objekte zu JSON-serializable format
            serialized_results = []
            for memory in results:
                serialized_results.append({
                    'memory_id': memory.memory_id,
                    'content': memory.content,
                    'importance': memory.importance,
                    'timestamp': memory.timestamp.isoformat(),
                    'memory_type': memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type),
                    'tags': memory.tags,
                    'emotional_intensity': memory.emotional_intensity,
                    'context': {
                        'conversation_id': memory.context.get('conversation_id'),
                        'speaker': memory.context.get('speaker'),
                        'topic_category': memory.context.get('topic_category')
                    }
                })
            
            return jsonify({
                'success': True,
                'results': serialized_results,
                'query': query,
                'total_found': len(serialized_results),
                'search_timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False, 
                'error': 'Memory system not available',
                'results': []
            }), 503
                
    except Exception as e:
        logger.error(f"Memory search error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@chat_bp.route('/conversation/summary/<conversation_id>', methods=['GET'])
def get_conversation_summary(conversation_id):
    """Gibt Conversation Summary zurück"""
    try:
        if conversation_memory:
            summary = asyncio.run(conversation_memory.get_conversation_summary(conversation_id))
            return jsonify({
                'success': True, 
                'summary': summary,
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False, 
                'error': 'Memory system not available',
                'summary': get_fallback_conversation_summary(conversation_id)
            }), 503
                
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@chat_bp.route('/memory/consolidate', methods=['POST'])
def trigger_memory_consolidation():
    """Triggert manuelle Memory Konsolidierung"""
    try:
        if conversation_memory:
            # Trigger STM to LTM consolidation
            consolidation_result = asyncio.run(conversation_memory._trigger_ltm_consolidation())
            
            return jsonify({
                'success': True,
                'message': 'Memory consolidation triggered',
                'result': consolidation_result,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Memory system not available'
            }), 503
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ✅ HELPER FUNCTIONS

def calculate_average_importance() -> float:
    """Berechnet durchschnittliche Importance der letzten Interactions"""
    try:
        if conversation_memory and hasattr(conversation_memory, 'conversation_buffer'):
            recent_conversations = conversation_memory.conversation_buffer[-10:]
            if recent_conversations:
                total_importance = sum(conv.get('importance_score', 0) for conv in recent_conversations)
                return round(total_importance / len(recent_conversations), 2)
        return 0.0
    except:
        return 0.0

def get_fallback_memory_stats() -> Dict[str, Any]:
    """Fallback Memory Stats wenn System nicht verfügbar"""
    return {
        'stm_capacity': 0,
        'stm_max_capacity': 7,
        'ltm_total_memories': 0,
        'conversation_buffer_size': 0,
        'status': 'unavailable'
    }

def get_fallback_conversation_summary(conversation_id: str) -> Dict[str, Any]:
    """Fallback Conversation Summary"""
    return {
        'conversation_id': conversation_id,
        'summary': 'Memory system not available',
        'total_exchanges': 0,
        'memory_count': 0,
        'status': 'unavailable'
    }

# ✅ INTEGRATION CHECK ROUTE
@chat_bp.route('/system/status', methods=['GET'])
def get_chat_system_status():
    """Gibt detaillierten Chat System Status zurück"""
    try:
        status = {
            'conversation_memory': {
                'available': bool(conversation_memory),
                'initialized': conversation_memory is not None,
                'stm_active': bool(conversation_memory and hasattr(conversation_memory, 'stm')),
                'ltm_active': bool(conversation_memory and hasattr(conversation_memory, 'ltm'))
            },
            'integrations': {
                'main_memory_system': bool(_memory_system),
                'lm_studio': bool(_lm_studio_integration),
                'voice_system': bool(_voice_system)
            },
            'performance': {
                'total_interactions': SYSTEM_STATE.get('interaction_count', 0),
                'last_interaction': SYSTEM_STATE.get('last_interaction', {}).get('timestamp'),
                'memory_efficiency': calculate_memory_efficiency()
            },
            'capabilities': {
                'memory_storage': bool(conversation_memory),
                'context_awareness': bool(conversation_memory),
                'learning_active': bool(conversation_memory and _memory_system),
                'emotional_processing': bool(conversation_memory)
            }
        }
        
        return jsonify({
            'success': True,
            'status': status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def calculate_memory_efficiency() -> float:
    """Berechnet Memory System Efficiency"""
    try:
        if conversation_memory:
            stats = conversation_memory.get_memory_stats()
            
            # Efficiency basierend auf STM Nutzung und LTM Konsolidierung
            stm_efficiency = 1.0 - (stats.get('stm_capacity', 0) / stats.get('stm_max_capacity', 7))
            ltm_growth = min(1.0, stats.get('ltm_total_memories', 0) / 100)  # Normalized to 100 memories
            
            return round((stm_efficiency * 0.3 + ltm_growth * 0.7) * 100, 1)
        return 0.0
    except:
        return 0.0