"""
Memory Core Module - Sichere Initialisierung mit korrekten Klassennamen
"""

import logging
import sys

logger = logging.getLogger(__name__)

# Ensure core.emotion_engine compatibility before importing anything
def _ensure_emotion_engine_compatibility():
    """Ensure core.emotion_engine is available before any imports"""
    if 'core.emotion_engine' not in sys.modules:
        try:
            from . import emotion_engine
            sys.modules['core.emotion_engine'] = emotion_engine
            logger.debug("✅ core.emotion_engine compatibility established")
        except ImportError:
            # Create minimal fallback
            import types
            fallback = types.ModuleType('core.emotion_engine')
            class EmotionEngine:
                def __init__(self): pass
            fallback.EmotionEngine = EmotionEngine
            sys.modules['core.emotion_engine'] = fallback
            logger.warning("⚠️ Using fallback core.emotion_engine")

_ensure_emotion_engine_compatibility()

# Safe imports with CORRECT class names
available_components = {}

# Memory Manager - Use correct class names
try:
    from .memory_manager import HumanLikeMemoryManager, MemoryManagerState, MemorySession
    available_components['memory_manager'] = True
    logger.debug("✅ HumanLikeMemoryManager imported")
    
    # Create aliases for backward compatibility
    MemoryManager = HumanLikeMemoryManager
    globals()['MemoryManager'] = MemoryManager
    globals()['HumanLikeMemoryManager'] = HumanLikeMemoryManager
    globals()['MemoryManagerState'] = MemoryManagerState
    globals()['MemorySession'] = MemorySession
    
except ImportError as e:
    available_components['memory_manager'] = False
    logger.warning(f"⚠️ Memory Manager import failed: {e}")

# Conversation Memory - Use correct class names
try:
    from .conversation_memory import ConversationMemorySystem
    available_components['conversation_memory'] = True
    logger.debug("✅ ConversationMemorySystem imported")
    
    # Create alias for backward compatibility
    ConversationMemory = ConversationMemorySystem
    globals()['ConversationMemory'] = ConversationMemory
    globals()['ConversationMemorySystem'] = ConversationMemorySystem
    
except ImportError as e:
    available_components['conversation_memory'] = False
    logger.warning(f"⚠️ Conversation Memory import failed: {e}")

# Short-term Memory - Use correct class names
try:
    from .short_term_memory import HumanLikeShortTermMemory
    available_components['short_term_memory'] = True
    logger.debug("✅ HumanLikeShortTermMemory imported")
    
    # Create alias for backward compatibility
    ShortTermMemory = HumanLikeShortTermMemory
    globals()['ShortTermMemory'] = ShortTermMemory
    globals()['HumanLikeShortTermMemory'] = HumanLikeShortTermMemory
    
except ImportError as e:
    available_components['short_term_memory'] = False
    logger.warning(f"⚠️ Short-term Memory import failed: {e}")

# Long-term Memory - Use correct class names
try:
    from .long_term_memory import HumanLikeLongTermMemory
    available_components['long_term_memory'] = True
    logger.debug("✅ HumanLikeLongTermMemory imported")
    
    # Create alias for backward compatibility
    LongTermMemory = HumanLikeLongTermMemory
    globals()['LongTermMemory'] = LongTermMemory
    globals()['HumanLikeLongTermMemory'] = HumanLikeLongTermMemory
    
except ImportError as e:
    available_components['long_term_memory'] = False
    logger.warning(f"⚠️ Long-term Memory import failed: {e}")

# Memory Types
try:
    from .memory_types import Memory, MemoryType, create_memory
    available_components['memory_types'] = True
    logger.debug("✅ Memory Types imported")
    
    globals()['Memory'] = Memory
    globals()['MemoryType'] = MemoryType
    globals()['create_memory'] = create_memory
    
except ImportError as e:
    available_components['memory_types'] = False
    logger.warning(f"⚠️ Memory Types import failed: {e}")

# Memory Models - Import from storage/memory_models.py where MemoryModel is actually defined
try:
    from ..storage.memory_models import MemoryModel, HumanLikeMemoryManager as StorageMemoryManager
    available_components['memory_models'] = True
    logger.debug("✅ Memory Models imported from storage")
    
    globals()['MemoryModel'] = MemoryModel
    
    # Also try core memory_models if it exists
    try:
        from .memory_models import EmotionType, ImportanceLevel, MemoryType as CoreMemoryType
        globals()['EmotionType'] = EmotionType
        globals()['ImportanceLevel'] = ImportanceLevel
        globals()['CoreMemoryType'] = CoreMemoryType
    except ImportError:
        logger.debug("Core memory_models not available, using storage version")
    
except ImportError as e:
    available_components['memory_models'] = False
    logger.warning(f"⚠️ Memory Models import failed: {e}")

# Emotion Engine
try:
    from .emotion_engine import EmotionEngine
    available_components['emotion_engine'] = True
    logger.debug("✅ EmotionEngine imported")
    
    globals()['EmotionEngine'] = EmotionEngine
    
    # Try to get additional emotion classes
    try:
        from .emotion_engine import EmotionType as EEEmotionType, PersonalityTrait, EmotionAnalysis
        globals()['EEEmotionType'] = EEEmotionType
        globals()['PersonalityTrait'] = PersonalityTrait
        globals()['EmotionAnalysis'] = EmotionAnalysis
    except ImportError:
        logger.debug("Additional emotion classes not available")
        
except ImportError as e:
    available_components['emotion_engine'] = False
    logger.warning(f"⚠️ Emotion Engine import failed: {e}")

# Search Engine
try:
    from .search_engine import SearchEngine
    available_components['search_engine'] = True
    logger.debug("✅ SearchEngine imported")
    
    globals()['SearchEngine'] = SearchEngine
    
except ImportError as e:
    # Try to find search functionality elsewhere
    try:
        from search_engine import MemorySearchEngine
        available_components['search_engine'] = True
        logger.debug("✅ SearchEngine imported from search module")
        globals()['SearchEngine'] = MemorySearchEngine
    except ImportError:
        available_components['search_engine'] = False
        logger.warning(f"⚠️ Search Engine import failed: {e}")

# Memory Database Interface
try:
    from ..storage.memory_database import MemoryDatabase
    available_components['memory_database'] = True
    logger.debug("✅ MemoryDatabase imported")
    
    globals()['MemoryDatabase'] = MemoryDatabase
    
except ImportError as e:
    available_components['memory_database'] = False
    logger.warning(f"⚠️ MemoryDatabase import failed: {e}")

logger.info(f"✅ Memory core loaded - {sum(available_components.values())}/{len(available_components)} components available")

# Create __all__ dynamically based on what's available
__all__ = []

# Add available components to __all__
for component_name, is_available in available_components.items():
    if is_available:
        __all__.append(component_name)

# Add aliases to __all__
available_classes = [
    'MemoryManager', 'HumanLikeMemoryManager', 'MemoryManagerState', 'MemorySession',
    'ConversationMemory', 'ConversationMemorySystem',
    'ShortTermMemory', 'HumanLikeShortTermMemory',
    'LongTermMemory', 'HumanLikeLongTermMemory',
    'Memory', 'MemoryType', 'create_memory',
    'MemoryModel', 'EmotionEngine', 'SearchEngine', 'MemoryDatabase'
]

for class_name in available_classes:
    if class_name in globals():
        if class_name not in __all__:
            __all__.append(class_name)

# Export component availability for other modules
AVAILABLE_COMPONENTS = available_components

logger.debug(f"📋 Available exports: {__all__}")