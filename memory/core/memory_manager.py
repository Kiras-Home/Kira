"""
🔄 LEGACY MEMORY MANAGER
Backward compatibility wrapper für den neuen UnifiedMemorySystem
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from ..integration import UnifiedMemorySystem
from .memory_types import Memory, MemoryType
from core.emotion_engine import EmotionEngine, EmotionType, PersonalityTrait

logger = logging.getLogger(__name__)

class MemoryManagerState(Enum):
    """Memory Manager State Enumeration"""
    INITIALIZING = "initializing"
    ACTIVE = "active" 
    PAUSED = "paused"
    ERROR = "error"
    OFFLINE = "offline"

@dataclass
class MemorySession:
    """Memory Session Data Structure"""
    session_id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    context: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.metadata is None:
            self.metadata = {}
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def add_context(self, key: str, value: Any):
        """Add context data"""
        self.context[key] = value
    
    def get_context(self, key: str, default=None):
        """Get context data"""
        return self.context.get(key, default)

class HumanLikeMemoryManager:
    """
    🔄 LEGACY MEMORY MANAGER
    Wrapper um UnifiedMemorySystem für Backward Compatibility
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Legacy Memory Manager"""
        logger.warning("⚠️ HumanLikeMemoryManager is deprecated. Use UnifiedMemorySystem instead.")
        
        # Create unified system
        self.unified_system = UnifiedMemorySystem(config)
        
        # Legacy attributes for compatibility
        self.short_term_memory = None
        self.long_term_memory = None
        self.working_memory = None
        
        # Initialize
        if self.unified_system.initialize():
            self.short_term_memory = self.unified_system.stm
            self.long_term_memory = self.unified_system.ltm
            self.working_memory = self.unified_system.stm.working_memory if self.unified_system.stm else []
            logger.info("✅ Legacy Memory Manager initialized")
        else:
            logger.error("❌ Legacy Memory Manager initialization failed")

        try:
            self.emotion_engine = EmotionEngine()
            logger.info("✅ Emotion Engine initialized")
        except Exception as e:
            logger.error(f"❌ Emotion Engine initialization failed: {e}")
            self.emotion_engine = None
    
    # ✅ LEGACY API METHODS
    
    def store_memory(self, content: str, importance: int = 5, **kwargs) -> Optional[Memory]:
        """Legacy memory storage method"""
        return self.unified_system.store_memory(
            content=content,
            importance=importance,
            **kwargs
        )
    
    def search_memories(self, query: str, limit: int = 10, **kwargs) -> List[Memory]:
        """Legacy memory search method"""
        return self.unified_system.search_memories(
            query=query,
            limit=limit,
            **kwargs
        )
    
    def get_memories(self) -> List[Memory]:
        """Legacy get all memories method"""
        try:
            all_memories = []
            
            if self.short_term_memory:
                all_memories.extend(self.short_term_memory.get_all_memories())
            
            if self.long_term_memory:
                all_memories.extend(self.long_term_memory.get_all_memories())
            
            return all_memories
        except Exception as e:
            logger.error(f"Get memories failed: {e}")
            return []
    
    def consolidate_memories(self, **kwargs) -> Dict[str, Any]:
        """Legacy consolidation method"""
        return self.unified_system.consolidate_memories(**kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Legacy stats method"""
        return self.unified_system.get_system_stats()
    
    def health_check(self) -> Dict[str, Any]:
        """Legacy health check method"""
        return self.unified_system.health_check()
    
async def analyze_conversation_emotion(self, 
                                     user_input: str,
                                     kira_response: str,
                                     user_id: str = None,
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    ✅ NEUE: Analyze emotions in conversation exchange
    
    Args:
        user_input: User's message
        kira_response: Kira's response
        user_id: User identifier
        context: Additional context
        
    Returns:
        Emotion analysis for both messages
    """
    try:
        if not self.emotion_engine:
            return {
                'success': False,
                'error': 'Emotion engine not available',
                'user_emotion': None,
                'kira_emotion': None
            }
        
        context = context or {}
        
        # Analyze user emotion
        user_emotion = self.emotion_engine.analyze_emotion(
            text=user_input,
            user_id=user_id,
            context={**context, 'role': 'user'}
        )
        
        # Analyze Kira's emotional response
        kira_emotion = self.emotion_engine.analyze_emotion(
            text=kira_response,
            user_id='kira',
            context={**context, 'role': 'assistant', 'responding_to': user_emotion.primary_emotion.value}
        )
        
        # Update personality profile if user_id provided
        personality_update = None
        if user_id:
            try:
                personality_update = self.emotion_engine.analyze_personality(
                    user_id=user_id,
                    conversation_history=[user_input],
                    emotion_history=[user_emotion]
                )
            except Exception as e:
                logger.warning(f"Personality update failed: {e}")
        
        return {
            'success': True,
            'user_emotion': {
                'primary_emotion': user_emotion.primary_emotion.value,
                'intensity': user_emotion.emotion_intensity,
                'confidence': user_emotion.emotion_confidence,
                'secondary_emotions': [(e.value, s) for e, s in user_emotion.secondary_emotions],
                'triggers': user_emotion.emotion_triggers,
                'context_factors': user_emotion.context_factors
            },
            'kira_emotion': {
                'primary_emotion': kira_emotion.primary_emotion.value,
                'intensity': kira_emotion.emotion_intensity,
                'confidence': kira_emotion.emotion_confidence,
                'secondary_emotions': [(e.value, s) for e, s in kira_emotion.secondary_emotions],
                'triggers': kira_emotion.emotion_triggers,
                'context_factors': kira_emotion.context_factors
            },
            'personality_update': {
                'updated': personality_update is not None,
                'confidence_level': personality_update.confidence_level if personality_update else 0.0,
                'dominant_traits': personality_update and self.emotion_engine._get_dominant_traits(personality_update) or []
            },
            'emotional_compatibility': self._calculate_emotional_compatibility(user_emotion, kira_emotion),
            'conversation_emotional_summary': self._generate_emotional_summary(user_emotion, kira_emotion)
        }
        
    except Exception as e:
        logger.error(f"Conversation emotion analysis failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'user_emotion': None,
            'kira_emotion': None
        }

def get_personalized_response_guidance(self, user_id: str) -> Dict[str, Any]:
    """
    ✅ NEUE: Get personalized response guidance for user
    
    Args:
        user_id: User identifier
        
    Returns:
        Personalized response guidance
    """
    try:
        if not self.emotion_engine:
            return {
                'success': False,
                'error': 'Emotion engine not available',
                'guidance': self._get_default_guidance()
            }
        
        # Get personalized style from emotion engine
        style_guidance = self.emotion_engine.get_personalized_response_style(user_id)
        
        # Add memory context
        memory_context = {}
        try:
            if self.stm and user_id:
                recent_memories = self.stm.search_memories(f"user:{user_id}", limit=5)
                memory_context['recent_interactions'] = len(recent_memories)
                memory_context['recent_topics'] = list(set([
                    m.context.get('topic_category', 'general') 
                    for m in recent_memories 
                    if m.context
                ]))
        except Exception as e:
            logger.warning(f"Memory context retrieval failed: {e}")
        
        return {
            'success': True,
            'response_guidance': style_guidance['response_style'],
            'personality_context': style_guidance['personality_context'],
            'recommendations': style_guidance['recommendations'],
            'memory_context': memory_context,
            'user_profile_available': user_id in self.emotion_engine.user_profiles,
            'guidance_confidence': style_guidance['personality_context']['confidence_level']
        }
        
    except Exception as e:
        logger.error(f"Personalized response guidance failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'guidance': self._get_default_guidance()
        }

def get_emotion_statistics(self) -> Dict[str, Any]:
    """
    ✅ NEUE: Get emotion engine statistics
    
    Returns:
        Comprehensive emotion statistics
    """
    try:
        if not self.emotion_engine:
            return {
                'success': False,
                'error': 'Emotion engine not available',
                'statistics': {}
            }
        
        engine_stats = self.emotion_engine.get_engine_statistics()
        
        # Add memory integration stats
        memory_stats = {}
        if hasattr(self, 'stm') and self.stm:
            # Count memories with emotional data
            all_memories = self.stm.get_all_memories()
            emotional_memories = [
                m for m in all_memories 
                if m.context and m.context.get('emotion_analysis')
            ]
            memory_stats['memories_with_emotion'] = len(emotional_memories)
            memory_stats['total_memories'] = len(all_memories)
            memory_stats['emotion_coverage'] = len(emotional_memories) / max(1, len(all_memories)) * 100
        
        return {
            'success': True,
            'engine_statistics': engine_stats,
            'memory_integration': memory_stats,
            'system_health': {
                'emotion_engine_active': True,
                'personality_profiling_active': len(self.emotion_engine.user_profiles) > 0,
                'emotion_history_active': len(self.emotion_engine.emotion_history) > 0
            }
        }
        
    except Exception as e:
        logger.error(f"Emotion statistics failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'statistics': {}
        }

# ✅ HELPER METHODS

def _calculate_emotional_compatibility(self, user_emotion, kira_emotion) -> float:
    """Calculate emotional compatibility between user and Kira"""
    try:
        # Simple compatibility scoring
        user_valence = self._get_emotion_valence(user_emotion.primary_emotion)
        kira_valence = self._get_emotion_valence(kira_emotion.primary_emotion)
        
        # Perfect empathy: matching emotional intensity
        intensity_match = 1.0 - abs(user_emotion.emotion_intensity - kira_emotion.emotion_intensity)
        
        # Emotional appropriateness
        appropriateness = self._calculate_response_appropriateness(
            user_emotion.primary_emotion, 
            kira_emotion.primary_emotion
        )
        
        # Combined compatibility score
        compatibility = (intensity_match * 0.4) + (appropriateness * 0.6)
        
        return min(max(compatibility, 0.0), 1.0)
        
    except Exception as e:
        logger.error(f"Emotional compatibility calculation failed: {e}")
        return 0.5

def _get_emotion_valence(self, emotion: EmotionType) -> float:
    """Get emotional valence (-1.0 to 1.0)"""
    valence_map = {
        EmotionType.JOY: 0.8,
        EmotionType.LOVE: 0.9,
        EmotionType.EXCITEMENT: 0.7,
        EmotionType.SATISFACTION: 0.6,
        EmotionType.SURPRISE: 0.2,
        EmotionType.NEUTRAL: 0.0,
        EmotionType.CONFUSION: -0.2,
        EmotionType.FEAR: -0.6,
        EmotionType.SADNESS: -0.7,
        EmotionType.ANGER: -0.8,
        EmotionType.FRUSTRATION: -0.7,
        EmotionType.DISGUST: -0.8
    }
    return valence_map.get(emotion, 0.0)

def _calculate_response_appropriateness(self, user_emotion: EmotionType, kira_emotion: EmotionType) -> float:
    """Calculate how appropriate Kira's emotional response is"""
    # Define appropriate responses for each user emotion
    appropriate_responses = {
        EmotionType.SADNESS: [EmotionType.CALM, EmotionType.LOVE],
        EmotionType.ANGER: [EmotionType.CALM, EmotionType.NEUTRAL],
        EmotionType.FEAR: [EmotionType.CALM, EmotionType.LOVE],
        EmotionType.JOY: [EmotionType.JOY, EmotionType.EXCITEMENT],
        EmotionType.EXCITEMENT: [EmotionType.JOY, EmotionType.EXCITEMENT],
        EmotionType.CONFUSION: [EmotionType.CALM, EmotionType.NEUTRAL],
        EmotionType.FRUSTRATION: [EmotionType.CALM, EmotionType.SATISFACTION]
    }
    
    if user_emotion in appropriate_responses:
        if kira_emotion in appropriate_responses[user_emotion]:
            return 1.0
        else:
            return 0.3
    
    return 0.6  # Neutral appropriateness

def _generate_emotional_summary(self, user_emotion, kira_emotion) -> str:
    """Generate emotional summary of conversation"""
    try:
        user_desc = f"{user_emotion.primary_emotion.value} (intensity: {user_emotion.emotion_intensity:.1f})"
        kira_desc = f"{kira_emotion.primary_emotion.value} (intensity: {kira_emotion.emotion_intensity:.1f})"
        
        compatibility = self._calculate_emotional_compatibility(user_emotion, kira_emotion)
        
        if compatibility > 0.8:
            tone = "harmoniously"
        elif compatibility > 0.6:
            tone = "appropriately"
        elif compatibility > 0.4:
            tone = "adequately"
        else:
            tone = "with some discord"
        
        return f"User expressed {user_desc}, Kira responded {tone} with {kira_desc}"
        
    except Exception as e:
        logger.error(f"Emotional summary generation failed: {e}")
        return "Emotional analysis completed"

def _get_default_guidance(self) -> Dict[str, Any]:
    """Default response guidance"""
    return {
        'formality_level': 0.5,
        'empathy_level': 0.7,
        'detail_level': 0.6,
        'emotional_tone': 'friendly',
        'recommendations': ['Use balanced, empathetic tone']
    }

# Export for backward compatibility
__all__ = ['HumanLikeMemoryManager']