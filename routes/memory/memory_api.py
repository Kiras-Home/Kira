"""
Memory API Routes - Access to Kira's memory system
"""

from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def create_memory_routes(system_state: Dict[str, Any], services: Dict[str, Any]) -> Blueprint:
    """
    Create memory API routes
    
    Args:
        system_state: Current system state
        services: Available services
        
    Returns:
        Blueprint with memory routes
    """
    
    memory_bp = Blueprint('memory_api', __name__, url_prefix='/api/memory')
    
    @memory_bp.route('/status', methods=['GET'])
    def memory_status():
        """Get memory system status"""
        try:
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            # Get memory status
            if hasattr(memory_service, 'get_status'):
                status = memory_service.get_status()
            else:
                status = {
                    'initialized': bool(memory_service),
                    'status': 'active' if memory_service else 'offline'
                }
            
            return jsonify({
                'success': True,
                'memory_status': status,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Memory status error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @memory_bp.route('/recent', methods=['GET'])
    def get_recent_memories():
        """Get recent memories"""
        try:
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            limit = request.args.get('limit', 10, type=int)
            
            memories = []
            
            # Try to get recent memories from conversation memory
            if hasattr(memory_service, 'conversation_memory'):
                conversation_memory = memory_service.conversation_memory
                
                if hasattr(conversation_memory, 'get_recent_memories'):
                    memories = conversation_memory.get_recent_memories(limit=limit)
                elif hasattr(conversation_memory, 'conversation_history'):
                    # Get from conversation history
                    history = conversation_memory.conversation_history
                    memories = list(history.values())[-limit:] if history else []
            
            # Try short-term memory if available
            elif hasattr(memory_service, 'short_term_memory'):
                stm = memory_service.short_term_memory
                if hasattr(stm, 'get_all_memories'):
                    memories = stm.get_all_memories()[-limit:]
            
            # Format memories for response
            formatted_memories = []
            for memory in memories:
                if hasattr(memory, '__dict__'):
                    formatted_memories.append({
                        'id': getattr(memory, 'memory_id', 'unknown'),
                        'content': getattr(memory, 'content', str(memory)),
                        'type': getattr(memory, 'memory_type', 'conversation'),
                        'importance': getattr(memory, 'importance', 'normal'),
                        'timestamp': getattr(memory, 'timestamp', datetime.now()).isoformat() if hasattr(getattr(memory, 'timestamp', None), 'isoformat') else str(getattr(memory, 'timestamp', 'unknown')),
                        'emotional_state': getattr(memory, 'emotional_state', 'neutral')
                    })
                else:
                    formatted_memories.append({
                        'id': 'unknown',
                        'content': str(memory),
                        'type': 'unknown',
                        'timestamp': datetime.now().isoformat()
                    })
            
            return jsonify({
                'success': True,
                'memories': formatted_memories,
                'count': len(formatted_memories),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Get recent memories error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @memory_bp.route('/search', methods=['POST'])
    def search_memories():
        """Search memories by content or metadata"""
        try:
            data = request.get_json()
            query = data.get('query', '').strip()
            
            if not query:
                return jsonify({
                    'success': False,
                    'error': 'Search query is required'
                }), 400
            
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            search_results = []
            
            # Try search engine if available
            if hasattr(memory_service, 'search_engine'):
                search_engine = memory_service.search_engine
                if hasattr(search_engine, 'search'):
                    search_results = search_engine.search(query)
            
            # Fallback: search conversation memory
            elif hasattr(memory_service, 'conversation_memory'):
                conversation_memory = memory_service.conversation_memory
                if hasattr(conversation_memory, 'search_memories'):
                    search_results = conversation_memory.search_memories(query)
            
            # Format search results
            formatted_results = []
            for result in search_results:
                if hasattr(result, '__dict__'):
                    formatted_results.append({
                        'id': getattr(result, 'memory_id', 'unknown'),
                        'content': getattr(result, 'content', str(result)),
                        'relevance': getattr(result, 'relevance_score', 1.0),
                        'type': getattr(result, 'memory_type', 'conversation'),
                        'timestamp': getattr(result, 'timestamp', datetime.now()).isoformat() if hasattr(getattr(result, 'timestamp', None), 'isoformat') else str(getattr(result, 'timestamp', 'unknown'))
                    })
                else:
                    formatted_results.append({
                        'id': 'unknown',
                        'content': str(result),
                        'relevance': 1.0,
                        'type': 'unknown',
                        'timestamp': datetime.now().isoformat()
                    })
            
            return jsonify({
                'success': True,
                'query': query,
                'results': formatted_results,
                'count': len(formatted_results),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Memory search error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @memory_bp.route('/importance', methods=['GET'])
    def get_important_memories():
        """Get memories marked as important"""
        try:
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            important_memories = []
            
            # Try to get important memories from long-term memory
            if hasattr(memory_service, 'long_term_memory'):
                ltm = memory_service.long_term_memory
                if hasattr(ltm, 'get_important_memories'):
                    important_memories = ltm.get_important_memories()
                elif hasattr(ltm, 'consolidated_memories'):
                    # Filter by importance
                    for memory in ltm.consolidated_memories.values():
                        if hasattr(memory, 'importance') and memory.importance >= 6:
                            important_memories.append(memory)
            
            # Format important memories
            formatted_memories = []
            for memory in important_memories:
                if hasattr(memory, '__dict__'):
                    formatted_memories.append({
                        'id': getattr(memory, 'memory_id', 'unknown'),
                        'content': getattr(memory, 'content', str(memory)),
                        'importance': getattr(memory, 'importance', 'high'),
                        'type': getattr(memory, 'memory_type', 'conversation'),
                        'timestamp': getattr(memory, 'timestamp', datetime.now()).isoformat() if hasattr(getattr(memory, 'timestamp', None), 'isoformat') else str(getattr(memory, 'timestamp', 'unknown')),
                        'category': getattr(memory, 'category', 'general')
                    })
            
            return jsonify({
                'success': True,
                'important_memories': formatted_memories,
                'count': len(formatted_memories),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Get important memories error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @memory_bp.route('/stats', methods=['GET'])
    def get_memory_stats():
        """Get memory system statistics"""
        try:
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            stats = {
                'short_term_memory': {},
                'long_term_memory': {},
                'conversation_memory': {},
                'total_interactions': system_state.get('interaction_count', 0)
            }
            
            # Get STM stats
            if hasattr(memory_service, 'short_term_memory'):
                stm = memory_service.short_term_memory
                if hasattr(stm, 'stats'):
                    stats['short_term_memory'] = stm.stats
                elif hasattr(stm, 'working_memory'):
                    stats['short_term_memory'] = {
                        'current_count': len(stm.working_memory),
                        'capacity': getattr(stm, 'capacity', 7)
                    }
            
            # Get LTM stats
            if hasattr(memory_service, 'long_term_memory'):
                ltm = memory_service.long_term_memory
                if hasattr(ltm, 'stats'):
                    stats['long_term_memory'] = ltm.stats
                elif hasattr(ltm, 'consolidated_memories'):
                    stats['long_term_memory'] = {
                        'total_memories': len(ltm.consolidated_memories),
                        'max_capacity': getattr(ltm, 'max_memories', 10000)
                    }
            
            # Get conversation memory stats
            if hasattr(memory_service, 'conversation_memory'):
                conv_mem = memory_service.conversation_memory
                if hasattr(conv_mem, 'stats'):
                    stats['conversation_memory'] = conv_mem.stats
                elif hasattr(conv_mem, 'conversation_history'):
                    stats['conversation_memory'] = {
                        'total_conversations': len(conv_mem.conversation_history)
                    }
            
            return jsonify({
                'success': True,
                'memory_stats': stats,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Get memory stats error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @memory_bp.route('/analytics', methods=['GET'])
    def get_memory_analytics():
        """Get live chat analytics and memory metrics"""
        try:
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            # Get current time for analytics
            now = datetime.now()
            today = now.date()
            
            # Initialize analytics data
            analytics = {
                'live_chat_activity': {
                    'messages_today': 0,
                    'messages_last_hour': 0,
                    'messages_last_10_minutes': 0,
                    'active_conversations': 0,
                    'average_response_time': 0,
                    'last_activity': None
                },
                'memory_metrics': {
                    'memories_stored_today': 0,
                    'important_memories_created': 0,
                    'consolidation_events': 0,
                    'memory_utilization': 0
                },
                'user_engagement': {
                    'unique_conversations': 0,
                    'conversation_duration_avg': 0,
                    'most_active_hour': None,
                    'interaction_patterns': {}
                }
            }
            
            # Try to get analytics from conversation memory
            if hasattr(memory_service, 'conversation_memory'):
                conversation_memory = memory_service.conversation_memory
                
                if hasattr(conversation_memory, 'conversation_history'):
                    history = conversation_memory.conversation_history
                    
                    # Count messages and conversations
                    messages_today = 0
                    messages_last_hour = 0
                    messages_last_10_minutes = 0
                    active_conversations = set()
                    
                    for conv_id, interactions in history.items():
                        if isinstance(interactions, list):
                            for interaction in interactions:
                                try:
                                    if hasattr(interaction, 'timestamp'):
                                        timestamp = interaction.timestamp
                                        if hasattr(timestamp, 'date'):
                                            if timestamp.date() == today:
                                                messages_today += 1
                                            if (now - timestamp).total_seconds() < 3600:  # Last hour
                                                messages_last_hour += 1
                                                active_conversations.add(conv_id)
                                            if (now - timestamp).total_seconds() < 600:  # Last 10 minutes
                                                messages_last_10_minutes += 1
                                except:
                                    continue
                    
                    analytics['live_chat_activity']['messages_today'] = messages_today
                    analytics['live_chat_activity']['messages_last_hour'] = messages_last_hour
                    analytics['live_chat_activity']['messages_last_10_minutes'] = messages_last_10_minutes
                    analytics['live_chat_activity']['active_conversations'] = len(active_conversations)
            
            # Get system state for additional analytics
            total_interactions = system_state.get('interaction_count', 0)
            last_interaction = system_state.get('last_interaction', {})
            
            if last_interaction:
                analytics['live_chat_activity']['last_activity'] = last_interaction.get('timestamp')
            
            # Memory utilization
            if hasattr(memory_service, 'short_term_memory'):
                stm = memory_service.short_term_memory
                if hasattr(stm, 'working_memory') and hasattr(stm, 'capacity'):
                    current_count = len(stm.working_memory)
                    capacity = stm.capacity
                    analytics['memory_metrics']['memory_utilization'] = (current_count / capacity) * 100 if capacity > 0 else 0
            
            # User engagement metrics
            analytics['user_engagement']['unique_conversations'] = len(system_state.get('conversations', {}))
            analytics['user_engagement']['total_interactions'] = total_interactions
            
            return jsonify({
                'success': True,
                'analytics': analytics,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Get memory analytics error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @memory_bp.route('/conversations/active', methods=['GET'])
    def get_active_conversations():
        """Get currently active conversations"""
        try:
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            active_conversations = []
            
            # Get active conversations from memory
            if hasattr(memory_service, 'conversation_memory'):
                conversation_memory = memory_service.conversation_memory
                
                if hasattr(conversation_memory, 'conversation_history'):
                    history = conversation_memory.conversation_history
                    now = datetime.now()
                    
                    for conv_id, interactions in history.items():
                        if isinstance(interactions, list) and interactions:
                            # Check if conversation has recent activity (last 30 minutes)
                            try:
                                last_interaction = interactions[-1]
                                if hasattr(last_interaction, 'timestamp'):
                                    time_diff = (now - last_interaction.timestamp).total_seconds()
                                    if time_diff < 1800:  # 30 minutes
                                        active_conversations.append({
                                            'conversation_id': conv_id,
                                            'last_activity': last_interaction.timestamp.isoformat(),
                                            'message_count': len(interactions),
                                            'duration_minutes': int(time_diff / 60),
                                            'status': 'active' if time_diff < 300 else 'recent'  # 5 minutes for active
                                        })
                            except:
                                continue
            
            # Add system state conversations
            system_conversations = system_state.get('conversations', {})
            for conv_id, conv_data in system_conversations.items():
                if conv_id not in [c['conversation_id'] for c in active_conversations]:
                    active_conversations.append({
                        'conversation_id': conv_id,
                        'last_activity': conv_data.get('last_activity'),
                        'message_count': conv_data.get('message_count', 0),
                        'status': 'system'
                    })
            
            return jsonify({
                'success': True,
                'active_conversations': active_conversations,
                'count': len(active_conversations),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Get active conversations error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @memory_bp.route('/live-stats', methods=['GET'])
    def get_live_stats():
        """Get live memory and chat statistics"""
        try:
            memory_service = services.get('memory')
            
            stats = {
                'memory_system': {
                    'status': 'active' if memory_service else 'offline',
                    'initialized': bool(memory_service)
                },
                'chat_activity': {
                    'total_messages': system_state.get('interaction_count', 0),
                    'active_conversations': len(system_state.get('conversations', {})),
                    'last_message_time': system_state.get('last_interaction', {}).get('timestamp'),
                    'system_uptime': system_state.get('uptime', 'unknown')
                },
                'memory_performance': {
                    'storage_backend': 'postgresql',
                    'memory_consolidation': 'active',
                    'search_available': bool(hasattr(memory_service, 'search_engine') if memory_service else False)
                }
            }
            
            # Add memory service specific stats
            if memory_service:
                if hasattr(memory_service, 'short_term_memory'):
                    stm = memory_service.short_term_memory
                    stats['memory_performance']['stm_utilization'] = f"{len(getattr(stm, 'working_memory', []))}/{getattr(stm, 'capacity', 7)}"
                
                if hasattr(memory_service, 'long_term_memory'):
                    ltm = memory_service.long_term_memory
                    stats['memory_performance']['ltm_memories'] = len(getattr(ltm, 'consolidated_memories', {}))
            
            return jsonify({
                'success': True,
                'live_stats': stats,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Get live stats error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @memory_bp.route('/brain/search', methods=['POST'])
    def search_brain_memories():
        """Search memories using Brain-Like Memory System"""
        try:
            data = request.get_json()
            query = data.get('query', '').strip()
            context_cues = data.get('context_cues', {})
            limit = data.get('limit', 10)
            
            if not query:
                return jsonify({
                    'success': False,
                    'error': 'Search query is required'
                }), 400
            
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            # Use brain memory search
            brain_results = memory_service.search_brain_memories(
                query=query,
                context_cues=context_cues,
                limit=limit
            )
            
            return jsonify({
                'success': True,
                'query': query,
                'brain_memories': brain_results,
                'count': len(brain_results),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Brain memory search error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @memory_bp.route('/brain/consolidate', methods=['POST'])
    def consolidate_brain_memories():
        """Consolidate brain memories (like during sleep)"""
        try:
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            consolidated_count = memory_service.consolidate_brain_memories()
            
            return jsonify({
                'success': True,
                'consolidated_memories': consolidated_count,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Brain memory consolidation error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @memory_bp.route('/brain/forget', methods=['POST'])
    def forget_weak_memories():
        """Forget weak memories (natural forgetting process)"""
        try:
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            forgotten_count = memory_service.forget_weak_memories()
            
            return jsonify({
                'success': True,
                'forgotten_memories': forgotten_count,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Memory forgetting error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @memory_bp.route('/brain/stats', methods=['GET'])
    def get_brain_memory_stats():
        """Get brain memory statistics"""
        try:
            memory_service = services.get('memory')
            
            if not memory_service:
                return jsonify({
                    'success': False,
                    'error': 'Memory service not available'
                }), 503
            
            brain_stats = memory_service.get_brain_memory_stats()
            
            return jsonify({
                'success': True,
                'brain_memory_stats': brain_stats,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Brain memory stats error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return memory_bp
