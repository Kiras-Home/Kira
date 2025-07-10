# ✅ NEUE DATEI: routes/emotion_api.py

"""
🎭 EMOTION & PERSONALITY API ROUTES
RESTful API für Emotion Engine und Personality Profiling
"""

import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from typing import Dict, Any

logger = logging.getLogger(__name__)

def create_emotion_api_blueprint(memory_system=None) -> Blueprint:
    """
    Erstellt Emotion API Blueprint
    
    Args:
        memory_system: Memory system mit emotion engine
        
    Returns:
        Flask Blueprint
    """
    
    emotion_bp = Blueprint('emotion_api', __name__, url_prefix='/emotion')
    
    @emotion_bp.route('/analyze', methods=['POST'])
    def analyze_emotion():
        """
        ✅ EMOTION ANALYSIS API
        
        POST /api/emotion/analyze
        {
            "text": "I'm feeling great today!",
            "user_id": "user123",
            "context": {"conversation_type": "casual"}
        }
        """
        try:
            if not memory_system or not hasattr(memory_system, 'emotion_engine') or not memory_system.emotion_engine:
                return jsonify({
                    'success': False,
                    'error': 'Emotion engine not available'
                }), 503
            
            data = request.get_json()
            if not data or 'text' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Text parameter required'
                }), 400
            
            text = data['text']
            user_id = data.get('user_id')
            context = data.get('context', {})
            
            # Analyze emotion
            emotion_analysis = memory_system.emotion_engine.analyze_emotion(
                text=text,
                user_id=user_id,
                context=context
            )
            
            return jsonify({
                'success': True,
                'emotion_analysis': {
                    'primary_emotion': emotion_analysis.primary_emotion.value,
                    'emotion_intensity': emotion_analysis.emotion_intensity,
                    'emotion_confidence': emotion_analysis.emotion_confidence,
                    'secondary_emotions': [
                        {'emotion': e.value, 'intensity': s} 
                        for e, s in emotion_analysis.secondary_emotions
                    ],
                    'emotion_triggers': emotion_analysis.emotion_triggers,
                    'context_factors': emotion_analysis.context_factors,
                    'timestamp': emotion_analysis.timestamp.isoformat()
                },
                'text_analyzed': text,
                'user_id': user_id
            })
            
        except Exception as e:
            logger.error(f"Emotion analysis API error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @emotion_bp.route('/personality/<user_id>', methods=['GET'])
    def get_personality_profile(user_id: str):
        """
        ✅ PERSONALITY PROFILE API
        
        GET /api/emotion/personality/user123
        """
        try:
            if not memory_system or not hasattr(memory_system, 'emotion_engine') or not memory_system.emotion_engine:
                return jsonify({
                    'success': False,
                    'error': 'Emotion engine not available'
                }), 503
            
            emotion_engine = memory_system.emotion_engine
            
            if user_id not in emotion_engine.user_profiles:
                return jsonify({
                    'success': False,
                    'error': f'No personality profile found for user {user_id}',
                    'user_id': user_id,
                    'profile_exists': False
                }), 404
            
            profile = emotion_engine.user_profiles[user_id]
            
            return jsonify({
                'success': True,
                'user_id': user_id,
                'personality_profile': {
                    'traits': {trait.value: score for trait, score in profile.traits.items()},
                    'preferences': profile.preferences,
                    'communication_style': profile.communication_style,
                    'confidence_level': profile.confidence_level,
                    'last_updated': profile.last_updated.isoformat(),
                    'learned_patterns_count': len(profile.learned_patterns)
                },
                'dominant_traits': emotion_engine._get_dominant_traits(profile),
                'recent_emotional_state': emotion_engine._get_recent_emotional_state(user_id),
                'profile_exists': True
            })
            
        except Exception as e:
            logger.error(f"Personality profile API error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'user_id': user_id
            }), 500
    
    @emotion_bp.route('/guidance/<user_id>', methods=['GET'])
    def get_response_guidance(user_id: str):
        """
        ✅ RESPONSE GUIDANCE API
        
        GET /api/emotion/guidance/user123
        """
        try:
            if not memory_system or not hasattr(memory_system, 'emotion_engine') or not memory_system.emotion_engine:
                return jsonify({
                    'success': False,
                    'error': 'Emotion engine not available'
                }), 503
            
            # Get personalized response guidance
            guidance = memory_system.emotion_engine.get_personalized_response_style(user_id)
            
            return jsonify({
                'success': True,
                'user_id': user_id,
                'response_guidance': guidance['response_style'],
                'personality_context': guidance['personality_context'],
                'recommendations': guidance['recommendations'],
                'guidance_available': user_id in memory_system.emotion_engine.user_profiles
            })
            
        except Exception as e:
            logger.error(f"Response guidance API error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'user_id': user_id
            }), 500
    
    @emotion_bp.route('/statistics', methods=['GET'])
    def emotion_statistics():
        """
        ✅ EMOTION ENGINE STATISTICS API
        
        GET /api/emotion/statistics
        """
        try:
            if not memory_system or not hasattr(memory_system, 'emotion_engine') or not memory_system.emotion_engine:
                return jsonify({
                    'success': False,
                    'error': 'Emotion engine not available'
                }), 503
            
            stats = memory_system.emotion_engine.get_engine_statistics()
            
            return jsonify({
                'success': True,
                'statistics': stats,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Emotion statistics API error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @emotion_bp.route('/users', methods=['GET'])
    def list_users_with_profiles():
        """
        ✅ LIST USERS WITH PERSONALITY PROFILES API
        
        GET /api/emotion/users
        """
        try:
            if not memory_system or not hasattr(memory_system, 'emotion_engine') or not memory_system.emotion_engine:
                return jsonify({
                    'success': False,
                    'error': 'Emotion engine not available'
                }), 503
            
            emotion_engine = memory_system.emotion_engine
            
            users_data = []
            for user_id, profile in emotion_engine.user_profiles.items():
                users_data.append({
                    'user_id': user_id,
                    'confidence_level': profile.confidence_level,
                    'last_updated': profile.last_updated.isoformat(),
                    'dominant_traits': emotion_engine._get_dominant_traits(profile),
                    'recent_emotional_state': emotion_engine._get_recent_emotional_state(user_id),
                    'emotion_history_count': len(emotion_engine.emotion_history.get(user_id, []))
                })
            
            return jsonify({
                'success': True,
                'users_with_profiles': users_data,
                'total_users': len(users_data),
                'active_emotion_tracking': len(emotion_engine.emotion_history)
            })
            
        except Exception as e:
            logger.error(f"Users list API error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return emotion_bp

# Export
__all__ = ['create_emotion_api_blueprint']