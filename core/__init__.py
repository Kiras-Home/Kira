"""
Core Module - Compatibility bridge and component access
"""

import sys
import logging

logger = logging.getLogger(__name__)

__version__ = "2.0.0"
__all__ = [
    'memory_analysis',
    'personality_engine', 
    'emotion_system',
    'learning_core'
]

# Lazy Loading für Submodule
def get_memory_analysis():
    """Lazy import für Memory Analysis"""
    try:
        from . import memory_analysis
        return memory_analysis
    except ImportError as e:
        logger.debug(f"Memory Analysis nicht verfügbar: {e}")
        return None

def get_personality_engine():
    """Lazy import für Personality Engine"""
    try:
        from . import personality_engine
        return personality_engine
    except ImportError as e:
        logger.debug(f"Personality Engine nicht verfügbar: {e}")
        return None

def get_emotion_system():
    """Lazy import für Emotion System"""
    try:
        from . import emotion_system
        return emotion_system
    except ImportError as e:
        logger.debug(f"Emotion System nicht verfügbar: {e}")
        return None

def get_learning_core():
    """Lazy import für Learning Core"""
    try:
        from . import learning_core
        return learning_core
    except ImportError as e:
        logger.debug(f"Learning Core nicht verfügbar: {e}")
        return None

# Memory Core Compatibility Bridge
def _setup_memory_compatibility():
    """Setup memory system compatibility"""
    try:
        # Try to import from memory.core
        from memory.core import emotion_engine
        
        # Create core.emotion_engine module alias
        sys.modules['core.emotion_engine'] = emotion_engine
        
        # Also make individual components available
        if hasattr(emotion_engine, 'EmotionEngine'):
            globals()['EmotionEngine'] = emotion_engine.EmotionEngine
        if hasattr(emotion_engine, 'EmotionType'):
            globals()['EmotionType'] = emotion_engine.EmotionType
        if hasattr(emotion_engine, 'PersonalityTrait'):
            globals()['PersonalityTrait'] = emotion_engine.PersonalityTrait
        
        logger.debug("✅ Memory core compatibility bridge established")
        return True
        
    except ImportError as e:
        logger.warning(f"Memory core compatibility failed: {e}")
        
        # Create fallback emotion_engine module
        class FallbackEmotionEngine:
            def __init__(self):
                self.initialized = True
            
            def analyze_emotion(self, text, user_id=None, context=None):
                return {
                    'primary_emotion': 'neutral',
                    'emotion_intensity': 0.5,
                    'emotion_confidence': 0.5,
                    'fallback': True
                }
        
        class FallbackEmotionType:
            NEUTRAL = 'neutral'
            JOY = 'joy'
            SADNESS = 'sadness'
            ANGER = 'anger'
            FEAR = 'fear'
            SURPRISE = 'surprise'
        
        # Create fallback module
        import types
        fallback_module = types.ModuleType('core.emotion_engine')
        fallback_module.EmotionEngine = FallbackEmotionEngine
        fallback_module.EmotionType = FallbackEmotionType
        
        sys.modules['core.emotion_engine'] = fallback_module
        
        # Also add to globals
        globals()['EmotionEngine'] = FallbackEmotionEngine
        globals()['EmotionType'] = FallbackEmotionType
        
        logger.info("✅ Fallback emotion engine created")
        return False

# Initialize compatibility bridge
_setup_memory_compatibility()

logger.info("🔗 Core module initialized with memory compatibility")