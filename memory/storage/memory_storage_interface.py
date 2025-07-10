"""
Memory Storage Interface - FINALE IMPLEMENTATION
🚀 Echte PostgreSQL Integration ohne Simulationen
🗄️ Abstrakte Basis mit konkreten Default-Implementierungen
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json
import hashlib
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# 🔧 ENHANCED DATA CLASSES
# ============================================================================

@dataclass
class MemorySearchFilter:
    """
    🚀 ERWEITERTE Memory Search Filter mit vollständiger PostgreSQL Integration
    """
    # Basic Search
    query: Optional[str] = None
    user_id: str = "default"
    memory_type: Optional[str] = None
    session_id: Optional[str] = None

    # Content Filters
    importance_min: int = 1
    importance_max: int = 10
    content_min_length: int = 0
    content_max_length: int = 100000
    has_tags: bool = False
    tags: Optional[List[str]] = None

    # Time Filters
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    created_within_hours: Optional[int] = None
    accessed_within_hours: Optional[int] = None

    # Emotional Filters
    emotion_type: Optional[str] = None
    emotion_intensity_min: float = 0.0
    emotion_intensity_max: float = 1.0
    emotion_valence_min: float = -1.0
    emotion_valence_max: float = 1.0

    # STM/LTM Filters
    stm_activation_min: float = 0.0
    stm_activation_max: float = 1.0
    ltm_significance_min: float = 0.0
    ltm_significance_max: float = 1.0
    consolidation_score_min: float = 0.0
    memory_strength_min: float = 0.0
    reinforcement_count_min: int = 0

    # Advanced Filters
    has_metadata: bool = False
    metadata_keys: Optional[List[str]] = None
    content_hash: Optional[str] = None
    expires_after: Optional[datetime] = None
    access_count_min: int = 0
    related_memory_ids: Optional[List[int]] = None

    # Search Options
    enable_fuzzy_search: bool = True
    enable_full_text_search: bool = True
    similarity_threshold: float = 0.5
    case_sensitive: bool = False

    # Paging & Ordering
    limit: int = 50
    offset: int = 0
    order_by: str = "created_at"
    order_direction: str = "DESC"

    # Performance Options
    use_index_hint: bool = True
    return_count: bool = False


@dataclass
class StorageStats:
    """
    🚀 ERWEITERTE Storage Statistics mit echten Database-Metriken
    """
    # Basic Metrics
    total_memories: int = 0
    total_users: int = 0
    total_sessions: int = 0
    avg_memories_per_user: float = 0.0
    avg_importance: float = 0.0

    # Memory Type Distribution
    memory_types: Dict[str, int] = field(default_factory=dict)
    memory_type_percentages: Dict[str, float] = field(default_factory=dict)

    # Recent Activity
    recent_memories_24h: int = 0
    recent_memories_7d: int = 0
    recent_memories_30d: int = 0
    most_active_users: List[Dict[str, Any]] = field(default_factory=list)

    # STM/LTM Metrics
    stm_memories: int = 0
    ltm_memories: int = 0
    avg_stm_activation: float = 0.0
    avg_ltm_significance: float = 0.0
    avg_consolidation_score: float = 0.0
    avg_memory_strength: float = 0.0
    consolidation_candidates: int = 0

    # Emotional Memory Metrics
    emotional_memories: int = 0
    emotion_distribution: Dict[str, int] = field(default_factory=dict)
    avg_emotion_intensity: float = 0.0
    avg_emotion_valence: float = 0.0
    emotional_diversity_index: float = 0.0

    # Performance Metrics
    avg_access_count: float = 0.0
    total_access_count: int = 0
    most_accessed_memories: List[Dict[str, Any]] = field(default_factory=list)
    least_accessed_memories: int = 0

    # Database Metrics
    database_size_mb: float = 0.0
    index_size_mb: float = 0.0
    table_size_mb: float = 0.0
    total_storage_mb: float = 0.0

    # Maintenance Metrics
    last_vacuum: Optional[str] = None
    last_analyze: Optional[str] = None
    last_maintenance: Optional[str] = None
    fragmentation_ratio: float = 0.0

    # Schema Information
    schema_version: str = "3.0"
    table_count: int = 0
    index_count: int = 0
    constraint_count: int = 0

    # Timestamps
    stats_generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    database_uptime_hours: Optional[float] = None

    def __post_init__(self):
        """Calculate derived metrics"""
        if self.total_users > 0:
            self.avg_memories_per_user = self.total_memories / self.total_users

        # Calculate memory type percentages
        if self.memory_types and self.total_memories > 0:
            for mem_type, count in self.memory_types.items():
                self.memory_type_percentages[mem_type] = (count / self.total_memories) * 100

        # Calculate emotional diversity index (Shannon diversity)
        if self.emotion_distribution:
            total_emotional = sum(self.emotion_distribution.values())
            if total_emotional > 0:
                diversity = 0
                for count in self.emotion_distribution.values():
                    if count > 0:
                        p = count / total_emotional
                        diversity -= p * (p.bit_length() - 1)  # Approximate log2
                self.emotional_diversity_index = diversity

        # Calculate total storage
        self.total_storage_mb = self.database_size_mb + self.index_size_mb


@dataclass
class MemoryStorageConfig:
    """
    🚀 ERWEITERTE Konfiguration für Memory Storage mit PostgreSQL-Optimierungen
    """
    # Connection Settings
    storage_type: str = "postgresql"
    connection_string: Optional[str] = None
    host: str = "localhost"
    port: int = 5432
    database: str = "kira_memory"
    username: str = "kira"
    password: str = "kira_password"

    # Connection Pool Settings
    max_connections: int = 10
    min_connections: int = 2
    connection_timeout: int = 30
    idle_timeout: int = 300
    max_overflow: int = 5

    # Performance Settings
    enable_connection_pooling: bool = True
    enable_query_caching: bool = True
    enable_prepared_statements: bool = True
    query_timeout: int = 60
    bulk_insert_size: int = 1000

    # Search & Indexing
    enable_full_text_search: bool = True
    enable_vector_search: bool = False
    enable_fuzzy_search: bool = True
    search_language: str = "german"

    # STM/LTM Configuration
    enable_stm_ltm_integration: bool = True
    stm_capacity: int = 7
    ltm_significance_threshold: float = 0.7
    consolidation_threshold: float = 0.6
    auto_consolidation: bool = True
    consolidation_interval_hours: int = 1

    # Maintenance Settings
    enable_auto_maintenance: bool = True
    maintenance_interval_hours: int = 24
    auto_vacuum: bool = True
    auto_analyze: bool = True
    retention_days: Optional[int] = None

    # Security & Backup
    enable_ssl: bool = False
    ssl_cert_path: Optional[str] = None
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30

    # Logging & Monitoring
    enable_query_logging: bool = False
    enable_performance_monitoring: bool = True
    log_slow_queries: bool = True
    slow_query_threshold_ms: int = 1000

    # Data Directory
    data_dir: str = "data"
    backup_dir: str = "backups"
    log_dir: str = "logs"

    # Advanced Settings
    schema_validation: bool = True
    migration_enabled: bool = True
    custom_extensions: List[str] = field(default_factory=list)
    additional_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and enhance configuration"""
        if not self.connection_string:
            self.connection_string = (
                f"host={self.host} port={self.port} "
                f"dbname={self.database} user={self.username} password={self.password}"
            )

        # Ensure directories exist
        for dir_path in [self.data_dir, self.backup_dir, self.log_dir]:
            Path(dir_path).mkdir(exist_ok=True)

        # Validate critical settings
        if self.max_connections < self.min_connections:
            self.max_connections = self.min_connections + 1

        if self.consolidation_interval_hours < 1:
            self.consolidation_interval_hours = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'storage_type': self.storage_type,
            'connection_string': self.connection_string,
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'username': self.username,
            'password': '***',  # Redacted for security
            'max_connections': self.max_connections,
            'min_connections': self.min_connections,
            'enable_stm_ltm_integration': self.enable_stm_ltm_integration,
            'enable_full_text_search': self.enable_full_text_search,
            'auto_consolidation': self.auto_consolidation,
            'data_dir': self.data_dir,
            **self.additional_config
        }

    def get_connection_string(self, hide_password: bool = False) -> str:
        """Get connection string with optional password redaction"""
        if hide_password and self.password:
            return self.connection_string.replace(f"password={self.password}", "password=***")
        return self.connection_string


# ============================================================================
# 🚀 ENHANCED MEMORY STORAGE INTERFACE
# ============================================================================

class MemoryStorageInterface(ABC):
    """
    🚀 FINALE Memory Storage Interface - Abstrakte Basis ohne Simulationen
    Alle Default-Implementierungen basieren auf echten PostgreSQL-Operationen
    """

    def __init__(self, connection_string: Optional[str] = None, config: Optional[MemoryStorageConfig] = None):
        """
        Initialize storage interface

        Args:
            connection_string: Database connection string
            config: Enhanced storage configuration
        """
        self.connection_string = connection_string
        self.config = config or MemoryStorageConfig(connection_string=connection_string)
        self._initialized = False
        self._stats_cache = {}
        self._last_stats_update = None

        # Performance counters
        self.operation_counters = {
            'store_operations': 0,
            'search_operations': 0,
            'get_operations': 0,
            'update_operations': 0,
            'maintenance_operations': 0
        }

    # ========================================================================
    # 🚀 ABSTRACT METHODS - Must be implemented by concrete classes
    # ========================================================================

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the storage backend

        Returns:
            bool: True if initialization successful
        """
        pass

    @abstractmethod
    def get_connection(self):
        """
        Get database connection with context manager support

        Returns:
            Context manager for database connection
        """
        pass

    @abstractmethod
    def store_enhanced_memory(
            self,
            content: str,
            user_id: str = "default",
            memory_type: str = "general",
            session_id: str = "main",
            metadata: Optional[Dict] = None,
            importance: int = 5,

            # Emotional Context
            emotion_type: str = "neutral",
            emotion_intensity: float = 0.0,
            emotion_valence: float = 0.0,

            # STM/LTM Specific
            stm_activation_level: float = 0.0,
            ltm_significance_score: float = 0.0,
            consolidation_score: float = 0.0,
            memory_strength: float = 1.0,

            # Advanced Options
            attention_weight: float = 0.5,
            learning_weight: float = 1.0,
            expires_at: Optional[datetime] = None,
            tags: Optional[List[str]] = None,

            **kwargs
    ) -> Optional[int]:
        """
        Store enhanced memory with full STM/LTM integration

        Returns:
            Optional[int]: Memory ID if successful, None otherwise
        """
        pass

    @abstractmethod
    def search_memories(
            self,
            search_filter: Union[MemorySearchFilter, Dict, str],
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search memories with advanced filtering

        Args:
            search_filter: Enhanced search filter or query string

        Returns:
            List[Dict[str, Any]]: List of matching memories
        """
        pass

    @abstractmethod
    def get_enhanced_stats(self, user_id: Optional[str] = None) -> StorageStats:
        """
        Get comprehensive storage statistics

        Args:
            user_id: Optional user filter for stats

        Returns:
            StorageStats: Comprehensive statistics object
        """
        pass

    @abstractmethod
    def perform_maintenance(self) -> Dict[str, Any]:
        """
        Perform database maintenance operations

        Returns:
            Dict[str, Any]: Maintenance operation results
        """
        pass

    @abstractmethod
    def close(self):
        """Close storage connections and cleanup"""
        pass

    # ========================================================================
    # 🚀 CONCRETE IMPLEMENTATIONS - Real PostgreSQL-based defaults
    # ========================================================================

    def store_memory(
            self,
            session_id: str,
            user_id: str,
            memory_type: str,
            content: str,
            metadata: Optional[Dict] = None,
            importance: int = 5,
            tags: Optional[List[str]] = None,
            emotion_intensity: float = 0.5,
            **kwargs
    ) -> Optional[int]:
        """
        🚀 ECHTE Memory Storage - delegates to enhanced method
        """
        return self.store_enhanced_memory(
            content=content,
            user_id=user_id,
            memory_type=memory_type,
            session_id=session_id,
            metadata=metadata,
            importance=importance,
            tags=tags,
            emotion_intensity=emotion_intensity,
            **kwargs
        )

    def get_memories(
            self,
            user_id: str = "default",
            session_id: Optional[str] = None,
            memory_type: Optional[str] = None,
            limit: int = 50,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        🚀 ECHTE Memory Retrieval - uses enhanced search
        """
        search_filter = MemorySearchFilter(
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            limit=limit,
            order_by=kwargs.get('order_by', 'created_at'),
            order_direction=kwargs.get('order_direction', 'DESC')
        )

        self.operation_counters['get_operations'] += 1
        return self.search_memories(search_filter)

    def semantic_search(
            self,
            query: str,
            user_id: str = "default",
            limit: int = 10,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        🚀 ECHTE Semantic Search - PostgreSQL Full-Text Search statt Mock
        """
        if not query or not query.strip():
            return []

        try:
            # Use enhanced search with full-text capabilities
            search_filter = MemorySearchFilter(
                query=query.strip(),
                user_id=user_id,
                limit=limit,
                enable_full_text_search=True,
                enable_fuzzy_search=True,
                similarity_threshold=kwargs.get('similarity_threshold', 0.3),
                order_by='importance',
                order_direction='DESC'
            )

            results = self.search_memories(search_filter)

            # Enhanced semantic scoring
            for result in results:
                # Calculate semantic relevance score
                content = result.get('content', '').lower()
                query_lower = query.lower()

                # Basic similarity scoring
                semantic_score = 0.0

                # Exact match bonus
                if query_lower in content:
                    semantic_score += 0.8

                # Word overlap scoring
                query_words = set(query_lower.split())
                content_words = set(content.split())
                if query_words and content_words:
                    overlap = len(query_words.intersection(content_words))
                    semantic_score += (overlap / len(query_words)) * 0.6

                # Importance weighting
                importance = result.get('importance', 5)
                semantic_score += (importance / 10.0) * 0.2

                # Add semantic score to result
                result['semantic_score'] = min(1.0, semantic_score)
                result['search_query'] = query

            # Sort by semantic score
            results.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)

            logger.debug(f"🔍 Semantic search for '{query}': {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"❌ Semantic search error: {e}")
            return []

    def update_memory_access(self, memory_id: int, user_id: str = "default") -> bool:
        """
        🚀 ECHTE Memory Access Update - PostgreSQL implementation
        """
        if not self._initialized:
            return False

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Update access count and timestamp
                    cursor.execute('''
                                   UPDATE memory_entries
                                   SET access_count         = access_count + 1,
                                       last_accessed        = NOW(),
                                       stm_activation_level = LEAST(1.0, stm_activation_level + 0.05),
                                       memory_strength      = LEAST(1.0, memory_strength + 0.02)
                                   WHERE id = %s
                                     AND user_id = %s
                                   ''', [memory_id, user_id])

                    success = cursor.rowcount > 0
                    if success:
                        self.operation_counters['update_operations'] += 1
                        logger.debug(f"✅ Memory access updated: ID={memory_id}")
                    else:
                        logger.warning(f"⚠️ Memory not found for access update: ID={memory_id}")

                    return success

        except Exception as e:
            logger.error(f"❌ Memory access update error: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """
        🚀 ECHTE Database Statistics - delegates to enhanced stats
        """
        try:
            enhanced_stats = self.get_enhanced_stats()

            # Convert StorageStats to dictionary for compatibility
            return {
                'total_memories': enhanced_stats.total_memories,
                'unique_users': enhanced_stats.total_users,
                'memory_types': enhanced_stats.memory_types,
                'recent_activity': enhanced_stats.recent_memories_24h,
                'avg_importance': enhanced_stats.avg_importance,
                'database_size_mb': enhanced_stats.database_size_mb,
                'stm_memories': enhanced_stats.stm_memories,
                'ltm_memories': enhanced_stats.ltm_memories,
                'avg_stm_activation': enhanced_stats.avg_stm_activation,
                'avg_ltm_significance': enhanced_stats.avg_ltm_significance,
                'emotional_memories': enhanced_stats.emotional_memories,
                'last_maintenance': enhanced_stats.last_maintenance,
                'schema_version': enhanced_stats.schema_version,
                'enhanced_features': [
                    'PostgreSQL_Native',
                    'STM_LTM_Integration',
                    'EmotionalMemory',
                    'FullTextSearch',
                    'ConnectionPooling'
                ]
            }

        except Exception as e:
            logger.error(f"❌ Database stats error: {e}")
            return {
                'total_memories': 0,
                'unique_users': 0,
                'error': str(e)
            }

    # ========================================================================
    # 🚀 STM/LTM INTEGRATION METHODS - Real implementations
    # ========================================================================

    def store_stm_experience(
            self,
            content: str,
            context: Dict,
            user_id: str = "default",
            activation_level: float = 1.0,
            **kwargs
    ) -> Optional[int]:
        """
        🚀 ECHTE STM Experience Storage
        """
        return self.store_enhanced_memory(
            content=content,
            user_id=user_id,
            memory_type='short_term_experience',
            session_id=kwargs.get('session_id', 'stm_session'),
            metadata=context,
            importance=min(10, max(1, int(activation_level * 10))),
            stm_activation_level=activation_level,
            attention_weight=kwargs.get('attention_weight', 0.8),
            memory_strength=activation_level,
            **kwargs
        )

    def store_ltm_memory(
            self,
            content: str,
            significance: float,
            user_id: str = "default",
            **kwargs
    ) -> Optional[int]:
        """
        🚀 ECHTE LTM Memory Storage
        """
        return self.store_enhanced_memory(
            content=content,
            user_id=user_id,
            memory_type='long_term_memory',
            session_id=kwargs.get('session_id', 'ltm_session'),
            metadata=kwargs.get('metadata', {}),
            importance=min(10, max(1, int(significance * 10))),
            ltm_significance_score=significance,
            consolidation_score=significance * 0.8,
            memory_strength=min(1.0, significance + 0.2),
            **kwargs
        )

    def get_stm_memories(
            self,
            user_id: str = "default",
            active_only: bool = True,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        🚀 ECHTE STM Memories Retrieval
        """
        search_filter = MemorySearchFilter(
            user_id=user_id,
            memory_type='short_term_experience',
            stm_activation_min=0.3 if active_only else 0.0,
            limit=kwargs.get('limit', 20),
            order_by='stm_activation_level',
            order_direction='DESC'
        )

        return self.search_memories(search_filter)

    def get_ltm_memories(
            self,
            user_id: str = "default",
            min_significance: float = 0.5,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        🚀 ECHTE LTM Memories Retrieval
        """
        search_filter = MemorySearchFilter(
            user_id=user_id,
            memory_type='long_term_memory',
            ltm_significance_min=min_significance,
            limit=kwargs.get('limit', 100),
            order_by='ltm_significance_score',
            order_direction='DESC'
        )

        return self.search_memories(search_filter)

    def consolidate_memories(
            self,
            user_id: str = "default",
            consolidation_threshold: float = 0.6,
            max_consolidations: int = 100
    ) -> Dict[str, Any]:
        """
        🚀 ECHTE Memory Consolidation Implementation
        """
        if not self._initialized:
            return {'status': 'error', 'error': 'Storage not initialized'}

        try:
            consolidation_results = {
                'started_at': datetime.now().isoformat(),
                'user_id': user_id,
                'threshold': consolidation_threshold,
                'processed_memories': 0,
                'consolidated_count': 0,
                'strengthened_count': 0,
                'status': 'success'
            }

            with self.get_connection() as conn:
                with conn.cursor() as cursor:

                    # Find consolidation candidates
                    cursor.execute('''
                                   SELECT id,
                                          stm_activation_level,
                                          ltm_significance_score,
                                          consolidation_score,
                                          memory_strength,
                                          reinforcement_count
                                   FROM memory_entries
                                   WHERE user_id = %s
                                     AND consolidation_score >= %s
                                     AND (consolidation_timestamp IS NULL
                                       OR consolidation_timestamp < NOW() - INTERVAL '24 hours')
                                   ORDER BY consolidation_score DESC
                                       LIMIT %s
                                   ''', [user_id, consolidation_threshold, max_consolidations])

                    candidates = cursor.fetchall()
                    consolidation_results['processed_memories'] = len(candidates)

                    for candidate in candidates:
                        memory_id, stm_level, ltm_score, cons_score, strength, reinforcements = candidate

                        # Calculate consolidation improvements
                        new_strength = min(1.0, (strength or 1.0) + 0.1)
                        new_ltm_score = min(1.0, (ltm_score or 0.0) + 0.05)
                        new_reinforcements = (reinforcements or 0) + 1

                        # Update memory with consolidation
                        cursor.execute('''
                                       UPDATE memory_entries
                                       SET memory_strength         = %s,
                                           ltm_significance_score  = %s,
                                           reinforcement_count     = %s,
                                           consolidation_timestamp = NOW(),
                                           last_accessed           = NOW()
                                       WHERE id = %s
                                       ''', [new_strength, new_ltm_score, new_reinforcements, memory_id])

                        if cursor.rowcount > 0:
                            consolidation_results['consolidated_count'] += 1

                    # Apply decay to non-consolidated memories
                    cursor.execute('''
                                   UPDATE memory_entries
                                   SET stm_activation_level = GREATEST(0.0, stm_activation_level * 0.95),
                                       memory_strength      = GREATEST(0.1, memory_strength * 0.98)
                                   WHERE user_id = %s
                                     AND consolidation_score < %s
                                     AND last_accessed < NOW() - INTERVAL '7 days'
                                   ''', [user_id, consolidation_threshold])

                    consolidation_results['strengthened_count'] = cursor.rowcount

            consolidation_results['completed_at'] = datetime.now().isoformat()
            logger.info(f"✅ Memory consolidation completed: {consolidation_results['consolidated_count']} memories")
            return consolidation_results

        except Exception as e:
            logger.error(f"❌ Memory consolidation error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'consolidated_count': 0
            }

    # ========================================================================
    # 🚀 UTILITY METHODS
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get storage status information"""
        return {
            'initialized': self._initialized,
            'storage_type': self.config.storage_type if self.config else 'unknown',
            'connection_string': self.config.get_connection_string(hide_password=True) if self.config else None,
            'operation_counters': self.operation_counters.copy(),
            'config': self.config.to_dict() if self.config else {},
            'status_timestamp': datetime.now().isoformat()
        }

    def reset_counters(self):
        """Reset operation counters"""
        for key in self.operation_counters:
            self.operation_counters[key] = 0
        logger.info("🔄 Operation counters reset")

    def validate_memory_data(self, content: str, user_id: str, memory_type: str) -> Tuple[bool, str]:
        """
        Validate memory data before storage

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not content or not content.strip():
            return False, "Content cannot be empty"

        if not user_id or not user_id.strip():
            return False, "User ID cannot be empty"

        if not memory_type or not memory_type.strip():
            return False, "Memory type cannot be empty"

        if len(content) > 100000:  # 100KB limit
            return False, "Content too large (max 100KB)"

        if len(user_id) > 255:
            return False, "User ID too long (max 255 chars)"

        return True, ""


# ============================================================================
# 🚀 ENHANCED MEMORY STORAGE FACTORY
# ============================================================================

class MemoryStorageFactory:
    """
    🚀 ERWEITERTE Factory für Memory Storage Backends
    Echte PostgreSQL Integration ohne Simulationen
    """

    _instance = None
    _storage_classes = {}
    _default_configs = {}
    _storage_classes = {}
    _initialized = False

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls.instance = cls()
        return cls._instance

    def __new__(cls):
        """Ensure Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(MemoryStorageFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance


    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.storages = {}

    @classmethod
    def create_storage(cls, config: Union[MemoryStorageConfig, Dict, str]) -> MemoryStorageInterface:
        """
        Create or get cached storage instance
        
        Args:
            config: Storage configuration
        Returns:
            MemoryStorageInterface: Storage instance
        """
        instance = cls.get_instance()

        # Normalize configuration
        if isinstance(config, str):
            config = MemoryStorageConfig(storage_type=config)
        elif isinstance(config, dict):
            config = MemoryStorageConfig(**config)
        
        storage_type = config.storage_type.lower()
        connection_string = config.connection_string

        # Check cache first
        cache_key = f"{storage_type}:{connection_string}"
        if cache_key in instance._storages:
            return instance._storages[cache_key]

        # Create new storage instance
        try:
            if storage_type == "postgresql":
                from .postgresql_storage import PostgreSQLMemoryStorage
                storage = PostgreSQLMemoryStorage(
                    connection_string=connection_string,
                    config=config
                )
                if storage.initialize():
                    instance._storages[cache_key] = storage
                    logger.info(f"✅ Created new {storage_type} storage")
                    return storage
                else:
                    raise RuntimeError(f"Failed to initialize {storage_type} storage")
            else:
                raise ValueError(f"Unknown storage type: {storage_type}")

        except Exception as e:
            logger.error(f"❌ Storage creation failed: {e}")
            raise

    @classmethod
    def register_storage(cls, storage_type: str, storage_class, default_config: Optional[Dict] = None):
        """Register new storage type"""
        cls._storage_classes[storage_type] = storage_class
        if default_config:
            cls._default_configs[storage_type] = default_config
        logger.info(f"📦 Registered storage type: {storage_type}")

    

    @classmethod
    def get_available_storages(cls) -> List[str]:
        """Get list of available storage types"""
        return list(cls._storage_classes.keys())

    @classmethod
    def create_default_postgresql_storage(
            cls,
            connection_string: Optional[str] = None,
            **kwargs
    ) -> MemoryStorageInterface:
        """
        Create default PostgreSQL storage with optimal settings
        """
        config = MemoryStorageConfig(
            storage_type="postgresql",
            connection_string=connection_string or
                              "host=localhost port=5432 dbname=kira_memory user=kira password=kira_password",
            enable_stm_ltm_integration=True,
            enable_full_text_search=True,
            auto_consolidation=True,
            enable_auto_maintenance=True,
            **kwargs
        )
        return cls.create_storage(config)

    @classmethod
    def create_stm_ltm_optimized_storage(
            cls,
            connection_string: Optional[str] = None,
            **kwargs
    ) -> MemoryStorageInterface:
        """
        Create storage optimized for STM/LTM operations
        """
        config = MemoryStorageConfig(
            storage_type="postgresql",
            connection_string=connection_string or
                              "host=localhost port=5432 dbname=kira_memory user=kira password=kira_password",
            enable_stm_ltm_integration=True,
            stm_capacity=7,
            ltm_significance_threshold=0.7,
            consolidation_threshold=0.6,
            auto_consolidation=True,
            consolidation_interval_hours=1,
            enable_auto_maintenance=True,
            maintenance_interval_hours=6,
            **kwargs
        )
        return cls.create_storage(config)


# ============================================================================
# 🚀 UTILITY FUNCTIONS - Real implementations
# ============================================================================

def create_stm_ltm_compatible_storage(connection_string: Optional[str] = None) -> MemoryStorageInterface:
    """Factory function for STM/LTM-compatible storage"""
    return MemoryStorageFactory.create_stm_ltm_optimized_storage(connection_string)


def create_default_storage(connection_string: Optional[str] = None) -> MemoryStorageInterface:
    """Factory function for default PostgreSQL storage"""
    return MemoryStorageFactory.create_default_postgresql_storage(connection_string)


def serialize_memory_type(memory_type) -> str:
    """Universal MemoryType serialization"""
    if hasattr(memory_type, 'value'):
        return memory_type.value
    elif hasattr(memory_type, 'name'):
        return memory_type.name
    else:
        return str(memory_type)


def deserialize_memory_type(memory_type_str: str):
    """Universal MemoryType deserialization"""
    try:
        from ..core.memory_types import MemoryType
        for mem_type in MemoryType:
            if mem_type.value == memory_type_str:
                return mem_type
        return MemoryType[memory_type_str]
    except (ImportError, KeyError, AttributeError):
        return memory_type_str


def result_to_memory_object(result: Dict[str, Any]):
    """
    🚀 ECHTE Conversion: Database Result to Memory Object
    """
    try:
        from ..core.memory_types import Memory
        import json

        # Memory Type konvertieren
        memory_type = deserialize_memory_type(result.get('memory_type', 'conversation'))

        # Timestamp handling
        timestamp_str = result.get('created_at', datetime.now().isoformat())
        if isinstance(timestamp_str, str):
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = timestamp_str or datetime.now()

        # Metadata parsing
        metadata = result.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        # Extract context
        context = metadata.get('context', {})
        if not context and isinstance(metadata, dict):
            # Use entire metadata as context if no specific context field
            context = metadata

        # Tags processing
        tags = result.get('tags', [])
        if isinstance(tags, str):
            if tags.strip():
                try:
                    tags = json.loads(tags) if tags.startswith('[') else [t.strip() for t in tags.split(',') if
                                                                          t.strip()]
                except json.JSONDecodeError:
                    tags = [t.strip() for t in tags.split(',') if t.strip()]
            else:
                tags = []

        # Related memory IDs
        related_ids = result.get('related_memory_ids', [])
        if isinstance(related_ids, str):
            try:
                related_ids = json.loads(related_ids) if related_ids else []
            except json.JSONDecodeError:
                related_ids = []

        return Memory(
            content=result.get('content', ''),
            memory_type=memory_type,
            importance=result.get('importance', 5),
            timestamp=timestamp,
            user_id=result.get('user_id', 'default'),
            session_id=result.get('session_id', 'main'),
            emotional_intensity=result.get('emotion_intensity', 0.0),
            context=context,
            tags=tags,
            memory_id=result.get('id'),

            # Enhanced fields
            emotion_type=result.get('emotion_type', 'neutral'),
            emotion_valence=result.get('emotion_valence', 0.0),
            stm_activation_level=result.get('stm_activation_level', 0.0),
            ltm_significance_score=result.get('ltm_significance_score', 0.0),
            consolidation_score=result.get('consolidation_score', 0.0),
            memory_strength=result.get('memory_strength', 1.0),
            access_count=result.get('access_count', 0),
            last_accessed=result.get('last_accessed'),
            related_memory_ids=related_ids
        )

    except Exception as e:
        logger.error(f"❌ Result to Memory conversion failed: {e}")
        logger.debug(f"   Result data: {result}")
        return None


def validate_connection_string(connection_string: str) -> Tuple[bool, str]:
    """
    Validate PostgreSQL connection string

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not connection_string:
        return False, "Connection string cannot be empty"

    required_params = ['host', 'dbname', 'user']
    missing_params = []

    for param in required_params:
        if param not in connection_string:
            missing_params.append(param)

    if missing_params:
        return False, f"Missing required parameters: {', '.join(missing_params)}"

    return True, ""


def create_memory_hash(content: str, user_id: str, memory_type: str) -> str:
    """Create unique hash for memory content"""
    hash_input = f"{content}_{user_id}_{memory_type}_{datetime.now().date()}"
    return hashlib.md5(hash_input.encode()).hexdigest()


# ============================================================================
# 🚀 MODULE EXPORTS
# ============================================================================

__all__ = [
    # Main Interface
    'MemoryStorageInterface',

    # Data Classes
    'MemorySearchFilter',
    'StorageStats',
    'MemoryStorageConfig',

    # Factory
    'MemoryStorageFactory',

    # Utility Functions
    'create_stm_ltm_compatible_storage',
    'create_default_storage',
    'serialize_memory_type',
    'deserialize_memory_type',
    'result_to_memory_object',
    'validate_connection_string',
    'create_memory_hash'
]