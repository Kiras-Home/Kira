"""
Kira Memory Service
Handles memory system initialization and management
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from config.system_config import KiraSystemConfig

logger = logging.getLogger(__name__)


class MemoryService:
    """Centralized Memory Service for Kira"""
    
    def __init__(self, config: KiraSystemConfig):
        self.config = config
        self.memory_manager = None
        self.conversation_memory = None
        self.emotion_engine = None
        self.emotion_memory = None
        self.short_term_memory = None
        self.long_term_memory = None
        self.search_engine = None
        self.brain_memory_system = None
        self.is_initialized = False
        self.status = 'offline'
        
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize memory service with all components
        
        Returns:
            Initialization result dictionary
        """
        try:
            print("🧠 Initializing Memory Service...")
            
            # Check if memory is enabled
            if not self.config.memory.enable_memory:
                return {
                    'success': False,
                    'available': False,
                    'status': 'disabled',
                    'message': 'Memory system is disabled in configuration'
                }
            
            # Initialize memory components with correct paths
            memory_init_result = self._initialize_memory_components()
            
            if memory_init_result['success']:
                self.is_initialized = True
                self.status = 'active'
                print("✅ Memory Service initialized successfully")
                return {
                    'success': True,
                    'available': True,
                    'status': 'active',
                    'components': memory_init_result['components'],
                    'memory_manager': self.memory_manager
                }
            else:
                self.status = 'partial'  # Some components may work
                print("⚠️ Memory Service partially initialized")
                return {
                    'success': True,  # Continue even with partial success
                    'available': True,
                    'status': 'partial',
                    'components': memory_init_result['components'],
                    'warnings': memory_init_result.get('warnings', [])
                }
                
        except Exception as e:
            logger.error(f"Memory service initialization failed: {e}")
            self.status = 'error'
            return {
                'success': False,
                'available': False,
                'status': 'error',
                'error': str(e)
            }
    
    def _initialize_memory_components(self) -> Dict[str, Any]:
        """Initialize memory system components with correct import paths based on actual structure"""
        components = {}
        warnings = []
        
        # 1. Memory Manager (memory.core.memory_manager)
        try:
            from memory.core.memory_manager import MemoryManager
            self.memory_manager = MemoryManager()
            components['memory_manager'] = True
            print("  ✅ Core Memory Manager initialized")
        except Exception as e:
            logger.warning(f"Core Memory Manager failed: {e}")
            self.memory_manager = FallbackMemoryManager()
            components['memory_manager'] = True
            warnings.append(f"Using fallback Memory Manager: {e}")
            print("  ✅ Fallback Memory Manager initialized")
        
        # 2. Conversation Memory (memory.core.conversation_memory)
        try:
            from memory.core.conversation_memory import ConversationMemorySystem
            # Initialize without requiring database parameter
            self.conversation_memory = ConversationMemorySystem()
            components['conversation_memory'] = True
            print("  ✅ Core Conversation Memory initialized")
        except Exception as e:
            logger.warning(f"Core Conversation Memory failed: {e}")
            try:
                # Try alternative initialization
                from memory.core.conversation_memory import ConversationMemory
                self.conversation_memory = ConversationMemory()
                components['conversation_memory'] = True
                print("  ✅ Alternative Conversation Memory initialized")
            except Exception as e2:
                logger.warning(f"Alternative Conversation Memory failed: {e2}")
                self.conversation_memory = FallbackConversationMemory()
                components['conversation_memory'] = True
                warnings.append(f"Using fallback Conversation Memory: {e}")
                print("  ✅ Fallback Conversation Memory initialized")
        
        # 3. Emotion Engine (memory.core.emotion_engine) - This exists!
        try:
            from memory.core.emotion_engine import EmotionEngine
            self.emotion_engine = EmotionEngine()
            components['emotion_engine'] = True
            print("  ✅ Emotion Engine initialized")
        except Exception as e:
            logger.warning(f"Emotion Engine failed: {e}")
            self.emotion_engine = FallbackEmotionEngine()
            components['emotion_engine'] = True
            warnings.append(f"Using fallback Emotion Engine: {e}")
            print("  ✅ Fallback Emotion Engine initialized")
        
        # 4. Emotion Memory (memory.emotion.emotion_memory) - This needs database parameter
        try:
            # Try to get database instance first
            database = self._get_database_instance()
            if database:
                from memory.emotion.emotion_memory import EmotionMemory
                self.emotion_memory = EmotionMemory(database)
                components['emotion_memory'] = True
                print("  ✅ Emotion Memory with database initialized")
            else:
                raise Exception("Database instance not available")
        except Exception as e:
            logger.warning(f"Emotion Memory failed: {e}")
            # Try emotion_memory.emotion alternative
            try:
                from memory.emotion_memory.emotion import EmotionMemory
                self.emotion_memory = EmotionMemory()
                components['emotion_memory'] = True
                print("  ✅ Alternative Emotion Memory initialized")
            except Exception as e2:
                logger.warning(f"Alternative Emotion Memory failed: {e2}")
                self.emotion_memory = FallbackEmotionMemory()
                components['emotion_memory'] = True
                warnings.append(f"Using fallback Emotion Memory: {e}")
                print("  ✅ Fallback Emotion Memory initialized")
        
        # 5. Short-term Memory (memory.core.short_term_memory)
        try:
            from memory.core.short_term_memory import ShortTermMemory
            self.short_term_memory = ShortTermMemory()
            components['short_term_memory'] = True
            print("  ✅ Short-term Memory initialized")
        except Exception as e:
            logger.warning(f"Short-term Memory failed: {e}")
            components['short_term_memory'] = False
            warnings.append(f"Short-term Memory not available: {e}")
        
        # 6. Long-term Memory (memory.core.long_term_memory)
        try:
            from memory.core.long_term_memory import LongTermMemory
            self.long_term_memory = LongTermMemory()
            components['long_term_memory'] = True
            print("  ✅ Long-term Memory initialized")
        except Exception as e:
            logger.warning(f"Long-term Memory failed: {e}")
            components['long_term_memory'] = False
            warnings.append(f"Long-term Memory not available: {e}")
        
        # 7. Search Engine (memory.core.search_engine)
        try:
            from memory.core.search_engine import SearchEngine
            self.search_engine = SearchEngine()
            components['search_engine'] = True
            print("  ✅ Search Engine initialized")
        except Exception as e:
            logger.warning(f"Search Engine failed: {e}")
            components['search_engine'] = False
            warnings.append(f"Search Engine not available: {e}")
        
        # 8. Brain-Like Memory System (NEW)
        try:
            from memory.brain_memory_system import BrainLikeMemorySystem
            
            # Benötigt storage_backend
            if hasattr(self.conversation_memory, 'storage'):
                storage_backend = self.conversation_memory.storage
            else:
                # Fallback: Verwende PostgreSQL-Storage direkt
                from memory.storage.postgresql_storage import PostgreSQLMemoryStorage
                storage_backend = PostgreSQLMemoryStorage()
                storage_backend.initialize()
            
            self.brain_memory_system = BrainLikeMemorySystem(storage_backend)
            components['brain_memory_system'] = True
            print("  ✅ Brain-Like Memory System initialized")
            
        except Exception as e:
            logger.warning(f"Brain-Like Memory System failed: {e}")
            self.brain_memory_system = None
            components['brain_memory_system'] = False
            warnings.append(f"Brain-Like Memory System not available: {e}")
            print("  ⚠️ Brain-Like Memory System not available")
        
        # Determine overall success
        core_components = ['memory_manager', 'conversation_memory']
        core_success = all(components.get(comp, False) for comp in core_components)
        
        return {
            'success': core_success,
            'components': components,
            'warnings': warnings
        }
    
    def _get_database_instance(self):
        """Try to get database instance for components that need it"""
        try:
            # Try to initialize database if needed
            from memory.storage.memory_database import MemoryDatabase
            return MemoryDatabase()
        except Exception as e:
            logger.warning(f"Could not get database instance: {e}")
            try:
                # Try alternative database
                from memory.storage.postgresql_storage import PostgreSQLStorage
                return PostgreSQLStorage()
            except Exception as e2:
                logger.warning(f"Could not get PostgreSQL storage: {e2}")
                return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get memory service status"""
        return {
            'initialized': self.is_initialized,
            'status': self.status,
            'components': {
                'memory_manager': self.memory_manager is not None,
                'conversation_memory': self.conversation_memory is not None,
                'emotion_engine': self.emotion_engine is not None,
                'emotion_memory': self.emotion_memory is not None,
                'short_term_memory': self.short_term_memory is not None,
                'long_term_memory': self.long_term_memory is not None,
                'search_engine': self.search_engine is not None,
                'brain_memory_system': self.brain_memory_system is not None
            },
            'config': {
                'enabled': self.config.memory.enable_memory,
                'storage_enabled': self.config.memory.enable_storage,
                'data_dir': self.config.memory.data_dir
            }
        }
    
    def add_conversation(self, user_input: str, kira_response: str, metadata: Optional[Dict] = None):
        """Add conversation to memory with Brain-Like Memory integration"""
        try:
            # Basic conversation memory
            if self.conversation_memory:
                if hasattr(self.conversation_memory, 'add_interaction'):
                    self.conversation_memory.add_interaction(user_input, kira_response, metadata)
                elif hasattr(self.conversation_memory, 'save_conversation'):
                    self.conversation_memory.save_conversation(user_input, kira_response, metadata)
                elif hasattr(self.conversation_memory, 'store_conversation'):
                    self.conversation_memory.store_conversation(user_input, kira_response, metadata)
            
            # Brain-Like Memory System integration
            if hasattr(self, 'brain_memory_system') and self.brain_memory_system:
                try:
                    # Speichere User-Message gehirnähnlich
                    user_memory_id = self.brain_memory_system.store_message(
                        content=user_input,
                        user_id=metadata.get('user_id', 'default') if metadata else 'default',
                        conversation_id=metadata.get('conversation_id', 'default') if metadata else 'default',
                        message_type="user",
                        emotional_context=metadata.get('emotional_context') if metadata else None
                    )
                    
                    # Speichere Kira-Response gehirnähnlich
                    kira_memory_id = self.brain_memory_system.store_message(
                        content=kira_response,
                        user_id=metadata.get('user_id', 'default') if metadata else 'default',
                        conversation_id=metadata.get('conversation_id', 'default') if metadata else 'default',
                        message_type="assistant",
                        emotional_context=metadata.get('emotional_context') if metadata else None
                    )
                    
                    logger.info(f"🧠 Brain-like storage: User={user_memory_id}, Kira={kira_memory_id}")
                    
                except Exception as brain_error:
                    logger.warning(f"Brain-like storage error: {brain_error}")
            
            # Memory Manager fallback
            if self.memory_manager and hasattr(self.memory_manager, 'save_conversation'):
                self.memory_manager.save_conversation(user_input, kira_response, metadata)
                    
        except Exception as e:
            logger.error(f"Failed to add conversation to memory: {e}")
    
    def get_conversation_history(self, limit: int = 10) -> list:
        """Get conversation history"""
        try:
            # Try conversation memory first
            if self.conversation_memory:
                if hasattr(self.conversation_memory, 'get_recent_conversations'):
                    return self.conversation_memory.get_recent_conversations(limit)
                elif hasattr(self.conversation_memory, 'get_conversations'):
                    return self.conversation_memory.get_conversations(limit)
                elif hasattr(self.conversation_memory, 'get_history'):
                    return self.conversation_memory.get_history(limit)
                elif hasattr(self.conversation_memory, 'retrieve_conversations'):
                    return self.conversation_memory.retrieve_conversations(limit)
            
            # Fallback to memory manager
            if self.memory_manager and hasattr(self.memory_manager, 'get_conversation_history'):
                return self.memory_manager.get_conversation_history(limit)
                    
            return []
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def analyze_emotion(self, text: str, user_id: str = None, context: Dict = None) -> Dict[str, Any]:
        """Analyze emotion using emotion engine"""
        try:
            if self.emotion_engine and hasattr(self.emotion_engine, 'analyze_emotion'):
                result = self.emotion_engine.analyze_emotion(text, user_id, context)
                
                # Store emotion data if emotion memory is available
                if self.emotion_memory:
                    emotion_data = {
                        'user_id': user_id,
                        'text': text,
                        'emotion_result': result,
                        'context': context
                    }
                    self.add_emotion_data(emotion_data)
                
                return {
                    'success': True,
                    'emotion_analysis': result
                }
            else:
                return {
                    'success': False,
                    'error': 'Emotion engine not available'
                }
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def add_emotion_data(self, emotion_data: Dict):
        """Add emotion data to memory"""
        try:
            if self.emotion_memory:
                if hasattr(self.emotion_memory, 'save_emotion_data'):
                    self.emotion_memory.save_emotion_data(emotion_data)
                elif hasattr(self.emotion_memory, 'add_emotion'):
                    self.emotion_memory.add_emotion(emotion_data)
                elif hasattr(self.emotion_memory, 'store_emotion'):
                    self.emotion_memory.store_emotion(emotion_data)
                elif hasattr(self.emotion_memory, 'save_emotion'):
                    self.emotion_memory.save_emotion(emotion_data)
        except Exception as e:
            logger.error(f"Failed to add emotion data: {e}")
    
    def search_memory(self, query: str, limit: int = 5) -> list:
        """Search through memory"""
        try:
            # Try search engine first
            if self.search_engine and hasattr(self.search_engine, 'search'):
                return self.search_engine.search(query, limit)
            
            # Fallback: simple text search in conversation history
            history = self.get_conversation_history(50)
            results = []
            
            query_lower = query.lower()
            for conversation in history:
                user_input = conversation.get('user_input', '').lower()
                kira_response = conversation.get('kira_response', '').lower()
                
                if query_lower in user_input or query_lower in kira_response:
                    results.append(conversation)
                    if len(results) >= limit:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
        
    def search_brain_memories(self, query: str, context_cues: Optional[Dict] = None, limit: int = 10) -> list:
        """
        Search using Brain-Like Memory System
        
        Args:
            query: Search query
            context_cues: Optional context hints
            limit: Maximum results
            
        Returns:
            List of brain-like memories
        """
        try:
            if hasattr(self, 'brain_memory_system') and self.brain_memory_system:
                return self.brain_memory_system.retrieve_memories(
                    query=query,
                    context_cues=context_cues,
                    limit=limit
                )
            return []
            
        except Exception as e:
            logger.error(f"Brain memory search failed: {e}")
            return []
    
    def consolidate_brain_memories(self) -> int:
        """
        Consolidate brain memories (like during sleep)
        
        Returns:
            Number of consolidated memories
        """
        try:
            if hasattr(self, 'brain_memory_system') and self.brain_memory_system:
                return self.brain_memory_system.consolidate_memories()
            return 0
            
        except Exception as e:
            logger.error(f"Brain memory consolidation failed: {e}")
            return 0
    
    def forget_weak_memories(self) -> int:
        """
        Forget weak memories (natural forgetting process)
        
        Returns:
            Number of forgotten memories
        """
        try:
            if hasattr(self, 'brain_memory_system') and self.brain_memory_system:
                return self.brain_memory_system.forget_weak_memories()
            return 0
            
        except Exception as e:
            logger.error(f"Memory forgetting failed: {e}")
            return 0
    
    def get_brain_memory_stats(self) -> Dict[str, Any]:
        """
        Get brain memory statistics
        
        Returns:
            Brain memory statistics dictionary
        """
        try:
            if hasattr(self, 'brain_memory_system') and self.brain_memory_system:
                return self.brain_memory_system.get_memory_statistics()
            return {}
            
        except Exception as e:
            logger.error(f"Brain memory stats failed: {e}")
            return {}


# Enhanced Fallback Classes with better compatibility

class FallbackMemoryManager:
    """Enhanced fallback memory manager that mimics the real one"""
    
    def __init__(self):
        self.conversations = []
        self.metadata = {}
        self.data_dir = Path('memory/data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_conversation(self, user_input: str, kira_response: str, metadata: Optional[Dict] = None):
        """Save conversation with enhanced metadata"""
        from datetime import datetime
        conversation = {
            'user_input': user_input,
            'kira_response': kira_response,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'id': len(self.conversations)
        }
        self.conversations.append(conversation)
        
        if len(self.conversations) > 500:
            self.conversations = self.conversations[-500:]
        
        self._persist_conversation(conversation)
    
    def get_conversation_history(self, limit: int = 10) -> list:
        """Get recent conversations"""
        return self.conversations[-limit:] if self.conversations else []
    
    def _persist_conversation(self, conversation: Dict):
        """Persist conversation to file"""
        try:
            import json
            conversations_file = self.data_dir / 'fallback_conversations.json'
            
            existing = []
            if conversations_file.exists():
                with open(conversations_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            
            existing.append(conversation)
            
            if len(existing) > 1000:
                existing = existing[-1000:]
            
            with open(conversations_file, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not persist conversation: {e}")


class FallbackConversationMemory:
    """Enhanced fallback conversation memory"""
    
    def __init__(self):
        self.interactions = []
        
    def add_interaction(self, user_input: str, kira_response: str, metadata: Optional[Dict] = None):
        """Add interaction"""
        from datetime import datetime
        interaction = {
            'user_input': user_input,
            'kira_response': kira_response,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.interactions.append(interaction)
        
        if len(self.interactions) > 200:
            self.interactions = self.interactions[-200:]
    
    def get_recent_conversations(self, limit: int = 10) -> list:
        """Get recent conversations"""
        return self.interactions[-limit:] if self.interactions else []
    
    def get_conversations(self, limit: int = 10) -> list:
        """Alternative method name"""
        return self.get_recent_conversations(limit)
    
    def store_conversation(self, user_input: str, kira_response: str, metadata: Optional[Dict] = None):
        """Alternative method name"""
        self.add_interaction(user_input, kira_response, metadata)


class FallbackEmotionEngine:
    """Fallback emotion engine with basic functionality"""
    
    def __init__(self):
        self.emotion_history = {}
    
    def analyze_emotion(self, text: str, user_id: str = None, context: Dict = None):
        """Basic emotion analysis"""
        from datetime import datetime
        
        # Simple emotion detection based on keywords
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['toll', 'super', 'great', 'fantastic', 'love']):
            emotion = 'joy'
            intensity = 0.7
        elif any(word in text_lower for word in ['traurig', 'sad', 'schlecht', 'terrible']):
            emotion = 'sadness'
            intensity = 0.6
        elif any(word in text_lower for word in ['ärgerlich', 'angry', 'wütend', 'mad']):
            emotion = 'anger'
            intensity = 0.8
        elif '?' in text:
            emotion = 'curiosity'
            intensity = 0.5
        else:
            emotion = 'neutral'
            intensity = 0.3
        
        return {
            'primary_emotion': emotion,
            'emotion_intensity': intensity,
            'emotion_confidence': 0.6,
            'timestamp': datetime.now().isoformat(),
            'fallback': True
        }


class FallbackEmotionMemory:
    """Fallback emotion memory"""
    
    def __init__(self):
        self.emotions = []
    
    def save_emotion_data(self, emotion_data: Dict):
        """Save emotion data"""
        from datetime import datetime
        emotion_entry = {
            'emotion_data': emotion_data,
            'timestamp': datetime.now().isoformat()
        }
        self.emotions.append(emotion_entry)
        
        if len(self.emotions) > 300:
            self.emotions = self.emotions[-300:]
    
    def add_emotion(self, emotion_data: Dict):
        """Alternative method name"""
        self.save_emotion_data(emotion_data)
    
    def store_emotion(self, emotion_data: Dict):
        """Another alternative method name"""
        self.save_emotion_data(emotion_data)
    
    def save_emotion(self, emotion_data: Dict):
        """Yet another alternative method name"""
        self.save_emotion_data(emotion_data)