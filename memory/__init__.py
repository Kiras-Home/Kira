"""
🧠 KIRA MEMORY SYSTEM - Unified Architecture
Zentrales Memory System mit einheitlicher API
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version
__version__ = "2.0.0"

# ✅ CORE IMPORTS - Fixed Import Order
try:
    from .core.memory_types import Memory, MemoryType, MemoryImportance, create_memory
    from .core.short_term_memory import HumanLikeShortTermMemory
    from .core.long_term_memory import HumanLikeLongTermMemory
    from .core.conversation_memory import ConversationMemorySystem, ConversationContext
    from .core.memory_manager import HumanLikeMemoryManager
    CORE_AVAILABLE = True
    logger.info("✅ Core memory components imported successfully")
except ImportError as e:
    logger.error(f"❌ Core memory components not available: {e}")
    CORE_AVAILABLE = False
    # Define fallback classes
    class ConversationMemorySystem:
        pass
    class ConversationContext:
        pass
    class HumanLikeMemoryManager:
        pass

# ✅ STORAGE IMPORTS
try:
    from .storage.postgresql_storage import PostgreSQLMemoryStorage
    from .storage.memory_database import EnhancedMemoryDatabase
    STORAGE_AVAILABLE = True
    logger.info("✅ Storage components imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Storage components not available: {e}")
    STORAGE_AVAILABLE = False

# ✅ EMOTION ENGINE IMPORTS
try:
    from .core.emotion_engine import EmotionEngine, EmotionType, EmotionState
    EMOTION_ENGINE_AVAILABLE = True
    logger.info("✅ Emotion engine imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Emotion engine not available: {e}")
    EMOTION_ENGINE_AVAILABLE = False

# ✅ INTEGRATION IMPORTS
try:
    from .integration import MemorySystemIntegration
    INTEGRATION_AVAILABLE = True
    logger.info("✅ Memory integration imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Memory integration not available: {e}")
    INTEGRATION_AVAILABLE = False

# ✅ STORAGE INTERFACE IMPORTS
try:
    from .storage.memory_storage_interface import (
        MemoryStorageInterface,
        MemoryStorageFactory,
        MemoryStorageConfig
    )
    STORAGE_INTERFACE_AVAILABLE = True
    logger.info("✅ Storage interface imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Storage interface not available: {e}")
    STORAGE_INTERFACE_AVAILABLE = False

# ✅ SYSTEM STATUS
SYSTEM_STATUS = {
    'core_available': CORE_AVAILABLE,
    'storage_available': STORAGE_AVAILABLE,
    'emotion_engine_available': EMOTION_ENGINE_AVAILABLE,
    'integration_available': INTEGRATION_AVAILABLE,
    'storage_interface_available': STORAGE_INTERFACE_AVAILABLE
}

def get_memory_system_status() -> Dict[str, Any]:
    """
    Get current memory system status
    
    Returns:
        Dictionary with system status
    """
    return {
        'system_status': SYSTEM_STATUS,
        'components_available': sum(SYSTEM_STATUS.values()),
        'total_components': len(SYSTEM_STATUS),
        'health_score': sum(SYSTEM_STATUS.values()) / len(SYSTEM_STATUS),
        'version': __version__,
        'timestamp': datetime.now().isoformat()
    }

def create_memory_manager(
    data_dir: str = "data/kira_memory",
    enable_storage: bool = True,
    storage_type: str = "sqlite"
) -> Optional['HumanLikeMemoryManager']:
    """
    Create a complete memory manager instance
    
    Args:
        data_dir: Directory for memory data
        enable_storage: Whether to enable persistent storage
        storage_type: Type of storage backend
        
    Returns:
        HumanLikeMemoryManager instance or None
    """
    try:
        if not CORE_AVAILABLE:
            logger.error("❌ Core memory components not available")
            return None
        
        # Create data directory
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        # Create memory manager
        memory_manager = HumanLikeMemoryManager(data_dir=data_dir)
        
        # Configure storage if available and enabled
        if enable_storage and STORAGE_INTERFACE_AVAILABLE:
            try:
                config = MemoryStorageConfig(
                    storage_type=storage_type,
                    data_dir=data_dir,
                    enable_caching=True
                )
                
                storage = MemoryStorageFactory.create_storage(config)
                
                # Attach storage to memory manager if possible
                if hasattr(memory_manager, 'attach_storage'):
                    memory_manager.attach_storage(storage)
                    logger.info(f"✅ Storage attached: {storage_type}")
                
            except Exception as storage_e:
                logger.warning(f"⚠️ Storage setup failed: {storage_e}")
        
        # Initialize emotion engine if available
        if EMOTION_ENGINE_AVAILABLE and hasattr(memory_manager, 'emotion_engine'):
            try:
                if not memory_manager.emotion_engine:
                    memory_manager.emotion_engine = EmotionEngine()
                    logger.info("✅ Emotion engine initialized")
            except Exception as emotion_e:
                logger.warning(f"⚠️ Emotion engine setup failed: {emotion_e}")
        
        logger.info(f"✅ Memory manager created successfully")
        return memory_manager
        
    except Exception as e:
        logger.error(f"❌ Memory manager creation failed: {e}")
        return None

def create_conversation_memory(
    data_dir: str = "data/conversations",
    enable_storage: bool = True
) -> Optional['ConversationMemorySystem']:
    """
    Create a conversation memory system
    
    Args:
        data_dir: Directory for conversation data
        enable_storage: Whether to enable persistent storage
        
    Returns:
        ConversationMemorySystem instance or None
    """
    try:
        if not CORE_AVAILABLE:
            logger.error("❌ Core memory components not available")
            return None
        
        # Create data directory
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        # Create conversation memory system
        conversation_memory = ConversationMemorySystem(data_dir=data_dir)
        
        # Configure storage if available
        if enable_storage and STORAGE_INTERFACE_AVAILABLE:
            try:
                config = MemoryStorageConfig(
                    storage_type="sqlite",
                    data_dir=data_dir,
                    enable_caching=True
                )
                
                storage = MemoryStorageFactory.create_storage(config)
                
                if hasattr(conversation_memory, 'attach_storage'):
                    conversation_memory.attach_storage(storage)
                    logger.info("✅ Conversation storage attached")
                
            except Exception as storage_e:
                logger.warning(f"⚠️ Conversation storage setup failed: {storage_e}")
        
        logger.info("✅ Conversation memory system created")
        return conversation_memory
        
    except Exception as e:
        logger.error(f"❌ Conversation memory creation failed: {e}")
        return None

def test_memory_system() -> Dict[str, Any]:
    """
    Test memory system functionality
    
    Returns:
        Test results
    """
    test_results = {
        'core_test': False,
        'storage_test': False,
        'emotion_test': False,
        'integration_test': False,
        'overall_success': False
    }
    
    try:
        # Test core functionality
        if CORE_AVAILABLE:
            memory_manager = create_memory_manager(
                data_dir="data/test_memory",
                enable_storage=False
            )
            
            if memory_manager:
                test_results['core_test'] = True
                logger.info("✅ Core memory test passed")
        
        # Test storage interface
        if STORAGE_INTERFACE_AVAILABLE:
            try:
                config = MemoryStorageConfig(
                    storage_type="sqlite",
                    data_dir="data/test_storage"
                )
                
                storage = MemoryStorageFactory.create_storage(config)
                
                # Test basic operations
                test_id = storage.store_memory(
                    session_id="test",
                    user_id="test_user",
                    memory_type="test",
                    content="Test memory",
                    importance=5
                )
                
                if test_id:
                    test_results['storage_test'] = True
                    logger.info("✅ Storage test passed")
                
                storage.close()
                
            except Exception as storage_test_e:
                logger.warning(f"⚠️ Storage test failed: {storage_test_e}")
        
        # Test emotion engine
        if EMOTION_ENGINE_AVAILABLE:
            try:
                emotion_engine = EmotionEngine()
                
                test_emotion = emotion_engine.analyze_emotion(
                    text="I am happy today!",
                    user_id="test_user"
                )
                
                if test_emotion:
                    test_results['emotion_test'] = True
                    logger.info("✅ Emotion engine test passed")
                
            except Exception as emotion_test_e:
                logger.warning(f"⚠️ Emotion engine test failed: {emotion_test_e}")
        
        # Test integration
        if INTEGRATION_AVAILABLE:
            try:
                integration = MemorySystemIntegration()
                test_results['integration_test'] = True
                logger.info("✅ Integration test passed")
                
            except Exception as integration_test_e:
                logger.warning(f"⚠️ Integration test failed: {integration_test_e}")
        
        # Overall success
        test_results['overall_success'] = any([
            test_results['core_test'],
            test_results['storage_test']
        ])
        
        logger.info(f"🧠 Memory system test completed: {test_results['overall_success']}")
        
    except Exception as e:
        logger.error(f"❌ Memory system test failed: {e}")
        test_results['error'] = str(e)
    
    return test_results

# ✅ EXPORT IMPORTANT COMPONENTS
__all__ = [
    # Core components
    'Memory',
    'MemoryType', 
    'MemoryImportance',
    'create_memory',
    'HumanLikeShortTermMemory',
    'HumanLikeLongTermMemory',
    'ConversationMemorySystem',
    'ConversationContext',
    'HumanLikeMemoryManager',
    
    # Storage components
    'MemoryStorageInterface',
    'MemoryStorageFactory',
    'MemoryStorageConfig',
    
    # Emotion components
    'EmotionEngine',
    'EmotionType',
    'EmotionState',
    
    # Integration
    'MemorySystemIntegration',
    
    # Factory functions
    'create_memory_manager',
    'create_conversation_memory',
    
    # Utility functions
    'get_memory_system_status',
    'test_memory_system',
    
    # Status variables
    'CORE_AVAILABLE',
    'STORAGE_AVAILABLE',
    'EMOTION_ENGINE_AVAILABLE',
    'INTEGRATION_AVAILABLE',
    'STORAGE_INTERFACE_AVAILABLE',
    'SYSTEM_STATUS'
]

# ✅ STARTUP MESSAGE
if __name__ == "__main__":
    status = get_memory_system_status()
    print(f"🧠 Kira Memory System v{__version__}")
    print(f"   Components Available: {status['components_available']}/{status['total_components']}")
    print(f"   Health Score: {status['health_score']:.1%}")
    
    if status['health_score'] >= 0.8:
        print("   Status: ✅ Excellent")
    elif status['health_score'] >= 0.6:
        print("   Status: ⚠️ Good")
    else:
        print("   Status: ❌ Needs Attention")