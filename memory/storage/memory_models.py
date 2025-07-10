
"""
Human-Like Memory Manager - VOLLSTÄNDIG ÜBERARBEITET
Entfernt alle simulierten Funktionen und verwendet echte Storage/Memory Systeme
"""

import logging
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum

# 🔧 ECHTE IMPORTS - Keine Simulationen mehr
from ..core.memory_types import MemoryType
from ..storage.memory_storage_interface import (
    MemoryStorageInterface, 
    MemoryStorageFactory,
    MemoryStorageConfig,
    MemorySearchFilter
)
from ..storage.postgresql_storage import PostgreSQLMemoryStorage, DatabaseStats

logger = logging.getLogger(__name__)

try:
    from memory.core.memory_types import Memory, MemoryType, MemoryImportance
    MEMORY_TYPES_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Memory types imported successfully")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Memory types import failed: {e}")
    MEMORY_TYPES_AVAILABLE = False
    
    # ✅ FALLBACK MEMORY CLASS DEFINITION
    @dataclass
    class Memory:
        """Fallback Memory class"""
        memory_id: str = field(default_factory=lambda: f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        content: str = ""
        memory_type: str = "general"
        importance: int = 5
        emotional_intensity: float = 0.0
        context: Dict[str, Any] = field(default_factory=dict)
        tags: List[str] = field(default_factory=list)
        created_at: datetime = field(default_factory=datetime.now)
        last_accessed: Optional[datetime] = None
        access_count: int = 0
    
    class MemoryType:
        """Fallback MemoryType enum"""
        CONVERSATION = "conversation"
        EXPERIENCE = "experience"
        LEARNING = "learning"
        EMOTION = "emotion"
        SYSTEM = "system"
    
    class MemoryImportance:
        """Fallback MemoryImportance enum"""
        LOW = 1
        MEDIUM = 5
        HIGH = 8
        CRITICAL = 10

class MemoryManagerState(Enum):
    """Zustand des Memory-Managers"""
    INITIALIZING = "initializing"
    ACTIVE = "active" 
    CONSOLIDATING = "consolidating"
    SLEEPING = "sleeping"
    MAINTENANCE = "maintenance"
    ERROR = "error"

@dataclass
class MemorySession:
    """Memory Session Tracking"""
    session_id: str
    user_id: str
    started_at: datetime
    last_activity: datetime
    interaction_count: int = 0
    memory_formations: int = 0
    emotional_peaks: List[Dict] = None
    attention_focuses: List[str] = None
    
    def __post_init__(self):
        if self.emotional_peaks is None:
            self.emotional_peaks = []
        if self.attention_focuses is None:
            self.attention_focuses = []

class ChatMemory(Memory):
    """Spezialisierte Memory-Klasse für Chat-Nachrichten"""
    def __init__(self, 
                 content: str,
                 user_id: str,
                 conversation_id: str,
                 message_type: str = "user",  # "user" oder "assistant" 
                 **kwargs):
        super().__init__(
            content=content,
            memory_type=MemoryType.CONVERSATION,
            user_id=user_id,
            **kwargs
        )
        self.conversation_id = conversation_id
        self.message_type = message_type

class HumanLikeMemoryManager:
    """
    🚀 VOLLSTÄNDIG ÜBERARBEITETER MEMORY MANAGER
    - Verwendet echte Storage-Systeme
    - Keine simulierten Funktionen mehr
    - Vollständige Integration mit STM/LTM
    """
    
    def __init__(self, data_dir: str = "data", connection_string: Optional[str] = None, config: Optional[Dict] = None):
        """Initialisierung mit echten Storage-Systemen"""
        logger.info("🧠 Initialisiere überarbeiteten Memory Manager...")
        
        # Core Setup
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.state = MemoryManagerState.INITIALIZING
        self._lock = threading.Lock()
        
        # Sessions und Stats
        self.active_sessions: Dict[str, MemorySession] = {}
        self.processing_stats = {
            'total_interactions': 0,
            'memories_created': 0,
            'error_count': 0,
            'stm_operations': 0,
            'ltm_operations': 0,
            'search_operations': 0,
            'consolidation_operations': 0,
            'last_reset': datetime.now()
        }
        
        try:
            # 1. ECHTES STORAGE BACKEND
            self.storage_backend = self._initialize_storage_backend(connection_string, config)
            
            # 2. ECHTE MEMORY SUBSYSTEME
            self._initialize_memory_subsystems()
            
            # 3. KONFIGURATION
            self._initialize_configurations()
            
            # Setup Complete
            self.state = MemoryManagerState.ACTIVE
            logger.info("✅ Überarbeiteter Memory Manager erfolgreich initialisiert")
            
        except Exception as e:
            logger.error(f"❌ Memory Manager Initialisierung fehlgeschlagen: {e}")
            self.state = MemoryManagerState.ERROR
            raise

    def _initialize_storage_backend(self, connection_string: Optional[str] = None, config: Optional[Dict] = None) -> MemoryStorageInterface:
        """🔧 ECHTES Storage Backend Initialization"""
        logger.info("🔧 Initialisiere echtes Storage Backend...")
        
        try:
            # Konfiguration erstellen
            storage_config = MemoryStorageConfig(
                storage_type="postgresql",
                connection_string=connection_string or self._build_default_connection_string(config),
                enable_stm_ltm_integration=True,
                enable_full_text_search=True,
                enable_auto_maintenance=True,
                data_dir=str(self.data_dir)
            )
            
            # Factory verwenden für echtes Storage
            storage = MemoryStorageFactory.create_storage(storage_config)
            
            # Initialisieren
            if storage.initialize():
                logger.info("✅ Echtes Storage Backend initialisiert")
                return storage
            else:
                raise Exception("Storage Initialisierung fehlgeschlagen")
                
        except Exception as e:
            logger.error(f"❌ Storage Backend Error: {e}")
            raise

    def _build_default_connection_string(self, config: Optional[Dict] = None) -> str:
        """Default PostgreSQL Connection String"""
        if config and 'connection_string' in config:
            return config['connection_string']
        
        if config:
            host = config.get('host', 'localhost')
            port = config.get('port', 5432)
            database = config.get('database', 'kira_memory')
            user = config.get('user', 'kira')
            password = config.get('password', 'kira_password')
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        return "postgresql://kira:kira_password@localhost:5432/kira_memory"

    def _initialize_memory_subsystems(self):
        """🔧 ECHTE Memory Subsysteme laden"""
        logger.info("🔧 Initialisiere echte Memory Subsysteme...")
        
        try:
            # Memory Database Interface
            from ..storage.memory_database import MemoryDatabase
            self.memory_database = MemoryDatabase(storage_backend=self.storage_backend)
            logger.info("✅ MemoryDatabase Interface erstellt")
            
            # Short Term Memory
            from ..core.short_term_memory import HumanLikeShortTermMemory
            self.short_term_memory = HumanLikeShortTermMemory(
                memory_database=self.memory_database,
                capacity=7
            )
            logger.info("✅ STM (HumanLikeShortTermMemory) geladen")
            
            # Long Term Memory
            from ..core.long_term_memory import HumanLikeLongTermMemory
            self.long_term_memory = HumanLikeLongTermMemory(
                memory_database=self.memory_database
            )
            logger.info("✅ LTM (HumanLikeLongTermMemory) geladen")
            
            # Aliases für Kompatibilität
            self.short_term = self.short_term_memory
            self.long_term = self.long_term_memory
            
        except ImportError as ie:
            logger.error(f"❌ Import Error: {ie}")
            raise Exception(f"Memory Subsysteme nicht verfügbar: {ie}")
        except Exception as e:
            logger.error(f"❌ Memory Subsystem Error: {e}")
            raise

    def _initialize_configurations(self):
        """🔧 Konfigurationssystem"""
        logger.info("🔧 Initialisiere Konfigurationen...")
        
        self.attention_config = {
            'focus_threshold': 0.6,
            'attention_decay_rate': 0.1,
            'focus_window_size': 5,
            'context_window': 10
        }
        
        self.memory_formation_config = {
            'formation_threshold': 0.5,
            'consolidation_delay_hours': 24,
            'importance_boost_factor': 1.5,
            'stm_to_ltm_threshold': 0.7
        }
        
        self.processing_config = {
            'parallel_processing_enabled': True,
            'background_consolidation': True,
            'real_time_analysis': True,
            'batch_size': 100
        }
        
        # Combined config
        self.config = {
            'attention': self.attention_config,
            'memory_formation': self.memory_formation_config,
            'processing': self.processing_config
        }
        
        logger.info("✅ Konfigurationen initialisiert")

    # 🚀 HAUPTFUNKTIONEN - Vollständig überarbeitet

    def store_memory(self, **kwargs) -> Optional[int]:
        """
        🔧 ÜBERARBEITET: Speichert Memory mit echten Subsystemen
        """
        try:
            # Parameter validierung
            content = kwargs.get('content', '')
            if not content:
                logger.error("❌ store_memory: content ist erforderlich")
                return None
            
            user_id = kwargs.get('user_id', 'default')
            session_id = kwargs.get('session_id', 'main')
            memory_type = kwargs.get('memory_type', MemoryType.CONVERSATION.value)
            
            # Importance/Significance handling
            importance = kwargs.get('importance', 5)
            significance = kwargs.get('significance', importance / 10.0)
            
            logger.debug(f"📝 store_memory: {memory_type} | importance={importance}")
            
            # 🔧 ECHTE STORAGE STRATEGIE
            memory_id = None
            
            # High importance → LTM
            if importance >= 7 or significance >= 0.7:
                try:
                    ltm_params = {
                        'content': content,
                        'user_id': user_id,
                        'session_id': session_id,
                        'emotional_weight': kwargs.get('emotional_weight', importance / 10.0),
                        'personal_relevance': kwargs.get('personal_relevance', importance / 10.0),
                        'context': kwargs.get('context', {}),
                        'detected_patterns': kwargs.get('detected_patterns', []),
                        'significance_level': significance
                    }
                    
                    memory_id = self.long_term_memory.store_significant_memory(**ltm_params)
                    if memory_id:
                        logger.info(f"✅ LTM Storage: {memory_id}")
                        self._update_stats('ltm_operations')
                        self._update_stats('memories_created')
                        return memory_id
                        
                except Exception as ltm_e:
                    logger.warning(f"⚠️ LTM Storage Error: {ltm_e}")
            
            # Medium importance → STM
            if significance >= 0.4:
                try:
                    stm_params = {
                        'content': content,
                        'user_id': user_id,
                        'significance': significance,
                        'context': kwargs.get('context', 'general'),
                        'session_id': session_id
                    }
                    
                    memory_id = self.short_term_memory.add_experience(**stm_params)
                    if memory_id is not None:
                        logger.info(f"✅ STM Storage: {memory_id}")
                        self._update_stats('stm_operations')
                        self._update_stats('memories_created')
                        return memory_id
                        
                except Exception as stm_e:
                    logger.warning(f"⚠️ STM Storage Error: {stm_e}")
            
            # Fallback → Direct Storage Backend
            try:
                # Prepare clean parameters for storage backend
                storage_params = {
                    'session_id': session_id,
                    'user_id': user_id,
                    'memory_type': memory_type,
                    'content': content,
                    'importance': importance,
                    'user_context': str(kwargs.get('context', '')),
                    'emotion_type': kwargs.get('emotion_type', 'neutral'),
                    'emotion_intensity': kwargs.get('emotion_intensity', 0.0),
                    'emotion_valence': kwargs.get('emotion_valence', 0.0),
                    'device_context': kwargs.get('device_context', 'unknown'),
                    'conversation_context': kwargs.get('conversation_context', ''),
                    'metadata': kwargs.get('metadata', {})
                }
                
                memory_id = self.storage_backend.store_enhanced_memory(**storage_params)
                if memory_id:
                    logger.info(f"✅ Direct Storage: {memory_id}")
                    self._update_stats('memories_created')
                    return memory_id
                    
            except Exception as direct_e:
                logger.error(f"❌ Direct Storage Error: {direct_e}")
            
            logger.error("❌ Alle Storage-Strategien fehlgeschlagen")
            self._update_stats('error_count')
            return None
            
        except Exception as e:
            logger.error(f"❌ store_memory Error: {e}")
            self._update_stats('error_count')
            return None

    def search_memories(self, **kwargs) -> List[Dict[str, Any]]:
        """🔧 ÜBERARBEITET: Suche mit echten Subsystemen"""
        try:
            query = kwargs.get('query', '')
            user_id = kwargs.get('user_id', 'default')
            limit = kwargs.get('limit', 10)
            
            self._update_stats('search_operations')
            results = []
            
            # 1. STM Search
            try:
                stm_memories = self.short_term_memory.get_recent_experiences(limit=5)
                for memory in stm_memories:
                    content = memory.get('content', '')
                    if query.lower() in content.lower():
                        results.append({
                            'id': memory.get('id', 'stm_unknown'),
                            'content': content,
                            'source': 'short_term',
                            'timestamp': memory.get('timestamp', ''),
                            'relevance': 0.9,
                            'importance': memory.get('significance', 0.5) * 10
                        })
            except Exception as stm_e:
                logger.warning(f"⚠️ STM Search Error: {stm_e}")
            
            # 2. LTM Search
            try:
                # LTM hat recall_memories Methode
                if hasattr(self.long_term_memory, 'recall_memories'):
                    ltm_results = self.long_term_memory.recall_memories(query, limit=limit)
                    for result in ltm_results:
                        results.append({
                            'id': result.get('id', 'ltm_unknown'),
                            'content': result.get('content', ''),
                            'source': 'long_term',
                            'timestamp': result.get('created_at', ''),
                            'relevance': result.get('importance', 5) / 10.0,
                            'importance': result.get('importance', 5)
                        })
            except Exception as ltm_e:
                logger.warning(f"⚠️ LTM Search Error: {ltm_e}")
            
            # 3. Direct Storage Search
            try:
                search_filter = MemorySearchFilter(
                    query=query,
                    user_id=user_id,
                    limit=limit,
                    enable_full_text_search=True
                )
                
                storage_results = self.storage_backend.search_memories(search_filter)
                for result in storage_results:
                    results.append({
                        'id': result.get('id', 'storage_unknown'),
                        'content': result.get('content', ''),
                        'source': 'storage',
                        'timestamp': result.get('created_at', ''),
                        'relevance': result.get('importance', 5) / 10.0,
                        'importance': result.get('importance', 5)
                    })
                    
            except Exception as storage_e:
                logger.warning(f"⚠️ Storage Search Error: {storage_e}")
            
            # Deduplizierung und Ranking
            unique_results = self._deduplicate_and_rank_results(results, limit)
            return unique_results
            
        except Exception as e:
            logger.error(f"❌ search_memories Error: {e}")
            return []

    def process_user_interaction(
        self,
        user_input: str,
        ai_response: str,
        user_id: str = "default",
        session_id: str = "main",
        context: Optional[Dict] = None,
        emotional_indicators: Optional[Dict] = None,
        device_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """🔧 ÜBERARBEITET: Vollständige Interaction Processing"""
        
        result = {
            'success': False,
            'memory_id': None,
            'stm_processing': {'success': False, 'memory_id': None},
            'ltm_processing': {'success': False, 'memory_id': None},
            'consolidation': {'triggered': False, 'count': 0},
            'errors': []
        }
        
        try:
            self._update_stats('total_interactions')
            
            # Session Management
            session = self._get_or_create_session(session_id, user_id)
            session.interaction_count += 1
            session.last_activity = datetime.now()
            
            # Emotional Analysis
            emotion_type = 'neutral'
            emotion_intensity = 0.0
            if emotional_indicators:
                emotion_type = emotional_indicators.get('type', 'neutral')
                emotion_intensity = emotional_indicators.get('intensity', 0.0)
            
            # Content Analysis
            content = f"User: {user_input}\nKira: {ai_response}"
            importance = self._calculate_interaction_importance(
                user_input, ai_response, emotion_intensity, context
            )
            
            # Store Memory
            memory_params = {
                'content': content,
                'session_id': session_id,
                'user_id': user_id,
                'memory_type': MemoryType.CONVERSATION.value,
                'importance': importance,
                'emotion_type': emotion_type,
                'emotion_intensity': emotion_intensity,
                'device_context': device_context or 'unknown',
                'conversation_context': ai_response,
                'context': context or {},
                'metadata': {
                    'user_input_length': len(user_input),
                    'ai_response_length': len(ai_response),
                    'processing_timestamp': datetime.now().isoformat()
                }
            }
            
            memory_id = self.store_memory(**memory_params)
            result['memory_id'] = memory_id
            
            # Update session stats
            if memory_id:
                session.memory_formations += 1
                result['success'] = True
                
                # Trigger consolidation if needed
                if self._should_trigger_consolidation(session):
                    try:
                        consolidation_result = self._trigger_memory_consolidation(user_id)
                        result['consolidation'] = consolidation_result
                    except Exception as cons_e:
                        logger.warning(f"⚠️ Consolidation Error: {cons_e}")
                        result['errors'].append(f"consolidation: {cons_e}")
            else:
                self._update_stats('error_count')
                result['errors'].append("memory_storage_failed")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Process User Interaction Error: {e}")
            self._update_stats('error_count')
            result['errors'].append(f"critical: {e}")
            return result

    # 🔧 HELPER METHODS - Überarbeitet

    def _calculate_interaction_importance(
        self, 
        user_input: str, 
        ai_response: str, 
        emotion_intensity: float,
        context: Optional[Dict]
    ) -> int:
        """Berechnet Wichtigkeit einer Interaction"""
        base_importance = 5
        
        # Länge der Antworten
        if len(user_input) > 100 or len(ai_response) > 200:
            base_importance += 1
        
        # Emotionale Intensität
        if emotion_intensity > 0.7:
            base_importance += 2
        elif emotion_intensity > 0.4:
            base_importance += 1
        
        # Context Richness
        if context and len(context) > 3:
            base_importance += 1
        
        # Spezielle Keywords
        important_keywords = ['problem', 'help', 'important', 'urgent', 'remember']
        if any(keyword in user_input.lower() for keyword in important_keywords):
            base_importance += 2
        
        return min(10, max(1, base_importance))

    def _should_trigger_consolidation(self, session: MemorySession) -> bool:
        """Prüft ob Consolidation ausgelöst werden soll"""
        # Trigger basierend auf Interaction Count
        if session.interaction_count % 20 == 0:
            return True
        
        # Trigger basierend auf Memory Formations
        if session.memory_formations >= 10:
            return True
        
        # Trigger basierend auf Zeit
        if datetime.now() - session.started_at > timedelta(hours=2):
            return True
        
        return False

    def _trigger_memory_consolidation(self, user_id: str) -> Dict[str, Any]:
        """Triggert Memory Consolidation"""
        logger.info(f"🔄 Triggering consolidation for user: {user_id}")
        
        try:
            self._update_stats('consolidation_operations')
            
            # Use storage backend consolidation if available
            if hasattr(self.storage_backend, 'consolidate_memories'):
                result = self.storage_backend.consolidate_memories(user_id=user_id)
                return {
                    'triggered': True,
                    'count': result.get('consolidated_count', 0),
                    'method': 'storage_backend'
                }
            
            # Fallback: Basic consolidation
            return {
                'triggered': True,
                'count': 0,
                'method': 'basic_fallback'
            }
            
        except Exception as e:
            logger.error(f"❌ Consolidation Error: {e}")
            return {
                'triggered': False,
                'count': 0,
                'error': str(e)
            }

    def _deduplicate_and_rank_results(self, results: List[Dict], limit: int) -> List[Dict]:
        """Dedupliziert und rankt Suchergebnisse"""
        unique_results = []
        seen_content = set()
        
        # Sortiere zunächst nach Relevanz
        results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        
        for result in results:
            content_key = result['content'][:100].lower().strip()
            if content_key not in seen_content and content_key:
                seen_content.add(content_key)
                unique_results.append(result)
                
                if len(unique_results) >= limit:
                    break
        
        return unique_results

    def _get_or_create_session(self, session_id: str, user_id: str) -> MemorySession:
        """Session Management"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = MemorySession(
                session_id=session_id,
                user_id=user_id,
                started_at=datetime.now(),
                last_activity=datetime.now()
            )
        return self.active_sessions[session_id]

    def _update_stats(self, key: str):
        """Stats Update"""
        with self._lock:
            if key in self.processing_stats:
                self.processing_stats[key] += 1

    # 🚀 CONVENIENCE METHODS - Überarbeitet

    def store_conversation(self, user_input: str, ai_response: str, **kwargs) -> Optional[int]:
        """Speichert Conversation"""
        content = f"User: {user_input}\nKira: {ai_response}"
        return self.store_memory(
            content=content,
            memory_type=MemoryType.CONVERSATION.value,
            conversation_context=ai_response,
            user_context=user_input,
            **kwargs
        )

    def store_experience(self, content: str, significance: float = 0.5, **kwargs) -> Optional[int]:
        """Speichert Experience"""
        importance = max(1, int(significance * 10))
        return self.store_memory(
            content=content,
            importance=importance,
            memory_type=MemoryType.SHORT_TERM_EXPERIENCE.value,
            **kwargs
        )

    def store_significant_memory(self, content: str, importance: int = 7, **kwargs) -> Optional[int]:
        """Speichert Significant Memory"""
        return self.store_memory(
            content=content,
            importance=importance,
            memory_type=MemoryType.SIGNIFICANT_MEMORY.value,
            **kwargs
        )

    def get_relevant_memories(self, query: str, user_id: str = "default", limit: int = 5) -> List[Dict[str, Any]]:
        """Alias für search_memories"""
        return self.search_memories(query=query, user_id=user_id, limit=limit)

    def recall_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Alias für search_memories"""
        return self.search_memories(query=query, limit=limit)

    # 🚀 STATUS & HEALTH METHODS - Überarbeitet

    def get_database_stats(self) -> Dict[str, Any]:
        """Database Statistics"""
        try:
            if self.storage_backend and hasattr(self.storage_backend, 'get_enhanced_stats'):
                stats = self.storage_backend.get_enhanced_stats()
                return {
                    'database_stats': {
                        'total_memories': getattr(stats, 'total_memories', 0),
                        'unique_users': getattr(stats, 'unique_users', 0),
                        'memory_types': getattr(stats, 'memory_types', {}),
                        'recent_activity': getattr(stats, 'recent_activity', 0)
                    },
                    'processing_stats': self.processing_stats.copy(),
                    'subsystem_stats': {
                        'short_term_active': hasattr(self, 'short_term_memory'),
                        'long_term_active': hasattr(self, 'long_term_memory'),
                        'storage_backend_type': type(self.storage_backend).__name__
                    }
                }
            else:
                return {
                    'database_stats': {'error': 'No enhanced stats available'},
                    'processing_stats': self.processing_stats,
                    'subsystem_stats': {'limited_functionality': True}
                }
        except Exception as e:
            logger.error(f"❌ Database Stats Error: {e}")
            return {
                'database_stats': {'error': str(e)},
                'processing_stats': self.processing_stats,
                'subsystem_stats': {'error': True}
            }

    def get_system_health(self) -> Dict[str, Any]:
        """System Health Check"""
        health = {
            'overall_health': 0.0,
            'components': {
                'storage_backend': self.storage_backend is not None,
                'short_term_memory': hasattr(self, 'short_term_memory'),
                'long_term_memory': hasattr(self, 'long_term_memory'),
                'memory_database': hasattr(self, 'memory_database')
            },
            'state': self.state.value,
            'processing_stats': self.processing_stats.copy(),
            'active_sessions': len(self.active_sessions),
            'configuration': {
                'data_dir_exists': self.data_dir.exists(),
                'storage_initialized': getattr(self.storage_backend, '_initialized', False) if self.storage_backend else False
            }
        }
        
        # Calculate health score
        healthy_components = sum(1 for status in health['components'].values() if status)
        total_components = len(health['components'])
        health['overall_health'] = healthy_components / total_components if total_components > 0 else 0.0
        
        # Adjust for state
        if self.state == MemoryManagerState.ACTIVE:
            health['overall_health'] = min(1.0, health['overall_health'] + 0.1)
        elif self.state == MemoryManagerState.ERROR:
            health['overall_health'] = max(0.0, health['overall_health'] - 0.5)
        
        return health

    def get_memory_overview(self, user_id: str = "default") -> Dict[str, Any]:
        """Memory Overview"""
        overview = {
            'memory_regions': {
                'short_term_memory': {
                    'active': hasattr(self, 'short_term_memory'),
                    'capacity': getattr(self.short_term_memory, 'capacity', 7) if hasattr(self, 'short_term_memory') else 0,
                    'current_items': 0  # Will be updated below
                },
                'long_term_memory': {
                    'active': hasattr(self, 'long_term_memory'),
                    'consolidated_memories': 0  # Will be updated below
                },
                'storage_backend': {
                    'active': self.storage_backend is not None,
                    'type': type(self.storage_backend).__name__ if self.storage_backend else 'None'
                }
            },
            'processing_efficiency': self._calculate_processing_efficiency(),
            'session_summary': {
                'active_sessions': len(self.active_sessions),
                'total_interactions': self.processing_stats['total_interactions'],
                'memories_created': self.processing_stats['memories_created']
            }
        }
        
        # Try to get detailed info from subsystems
        try:
            if hasattr(self, 'short_term_memory'):
                stm_info = getattr(self.short_term_memory, 'get_status', lambda: {})()
                overview['memory_regions']['short_term_memory']['current_items'] = stm_info.get('current_items', 0)
        except:
            pass
        
        try:
            if hasattr(self, 'long_term_memory'):
                ltm_info = getattr(self.long_term_memory, 'get_enhanced_stats', lambda: {})()
                overview['memory_regions']['long_term_memory']['consolidated_memories'] = ltm_info.get('total_memories', 0)
        except:
            pass
        
        return overview

    def _calculate_processing_efficiency(self) -> float:
        """Processing Efficiency Calculation"""
        total_ops = self.processing_stats['total_interactions']
        if total_ops > 0:
            success_rate = self.processing_stats['memories_created'] / total_ops
            error_rate = self.processing_stats['error_count'] / total_ops
            return max(0.0, min(1.0, success_rate - error_rate))
        return 0.0

    # 🔧 CLEANUP METHODS

    def close(self):
        """Memory Manager cleanup"""
        logger.info("🔄 Closing Memory Manager...")
        
        try:
            # Close subsystems
            if hasattr(self, 'storage_backend') and self.storage_backend:
                self.storage_backend.close()
            
            # Clear sessions
            self.active_sessions.clear()
            
            # Update state
            self.state = MemoryManagerState.SLEEPING
            
            logger.info("✅ Memory Manager closed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error closing Memory Manager: {e}")

@dataclass
class MemoryModel:
    """Base Memory Model for Storage"""
    id: Optional[str] = None
    content: str = ""
    memory_type: str = "general"
    importance: int = 5
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def update(self):
        """Update the last modified timestamp"""
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'content': self.content,
            'memory_type': self.memory_type,
            'importance': self.importance,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryModel':
        """Create from dictionary"""
        return cls(
            id=data.get('id'),
            content=data.get('content', ''),
            memory_type=data.get('memory_type', 'general'),
            importance=data.get('importance', 5),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
            metadata=data.get('metadata', {})
        )
    
@dataclass
class ConversationModel:
    """Conversation Model for Storage compatibility"""
    id: Optional[str] = None
    session_id: str = "default"
    user_id: str = "default"
    user_input: str = ""
    ai_response: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    importance: int = 5
    emotion_type: str = "neutral"
    emotion_intensity: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'user_input': self.user_input,
            'ai_response': self.ai_response,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata,
            'importance': self.importance,
            'emotion_type': self.emotion_type,
            'emotion_intensity': self.emotion_intensity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationModel':
        """Create from dictionary"""
        return cls(
            id=data.get('id'),
            session_id=data.get('session_id', 'default'),
            user_id=data.get('user_id', 'default'),
            user_input=data.get('user_input', ''),
            ai_response=data.get('ai_response', ''),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
            metadata=data.get('metadata', {}),
            importance=data.get('importance', 5),
            emotion_type=data.get('emotion_type', 'neutral'),
            emotion_intensity=data.get('emotion_intensity', 0.0)
        )

# 🚀 FACTORY COMPATIBILITY
MemoryManager = HumanLikeMemoryManager