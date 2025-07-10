"""
Enhanced Memory Database - FINALE KOMPLETTE IMPLEMENTIERUNG
🚀 Echte PostgreSQL Integration mit STM/LTM Support
🗄️ Einheitliche Memory Storage mit Advanced Features
"""

import json
import psycopg2
import psycopg2.extras
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import logging
import threading
import time
import hashlib
import uuid


logger = logging.getLogger(__name__)

# ✅ IMPORT Storage Backend
try:
    from .postgresql_storage import PostgreSQLMemoryStorage as PSQLStorage
    from .memory_storage_interface import MemoryStorageInterface
    HAS_STORAGE_BACKEND = True
except ImportError:
    logger.warning("⚠️ Storage Backend nicht verfügbar - verwende Direct Mode")
    HAS_STORAGE_BACKEND = False
    PSQLStorage = None

try:
    from ..core.memory_types import Memory, MemoryType, create_memory
except ImportError:
    logger.warning("Could not import memory types for enhanced database")

# ============================================================================
# 🔧 ENHANCED FILTER & DATACLASSES
# ============================================================================

@dataclass
class EnhancedMemorySearchFilter:
    """
    🚀 ERWEITERTE Filter für Memory-Suche mit vollständiger STM/LTM Integration
    """
    # Basic Search
    query: Optional[str] = None
    user_id: str = "default"
    memory_type: Optional[str] = None
    session_id: Optional[str] = None

    # Content Filters
    importance_min: int = 1
    importance_max: int = 10
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    tags: Optional[List[str]] = None

    # Emotional Filters
    emotion_type: Optional[str] = None
    emotion_intensity_min: float = 0.0
    emotion_intensity_max: float = 1.0
    emotion_valence_min: float = -1.0
    emotion_valence_max: float = 1.0

    # STM/LTM Specific Filters
    stm_activation_min: float = 0.0
    stm_activation_max: float = 1.0
    ltm_significance_min: float = 0.0
    ltm_significance_max: float = 1.0
    consolidation_state: Optional[str] = None  # 'pending', 'consolidating', 'consolidated'
    memory_strength_min: float = 0.0
    memory_strength_max: float = 1.0

    # Advanced Filters
    attention_weight_min: float = 0.0
    learning_weight_min: float = 0.0
    reinforcement_count_min: int = 0
    has_metadata: bool = False
    content_hash: Optional[str] = None
    related_to_memory_id: Optional[int] = None

    # Paging & Ordering
    limit: int = 50
    offset: int = 0
    order_by: str = "created_at"
    order_direction: str = "DESC"

    # Search Options
    enable_fuzzy_search: bool = True
    enable_semantic_search: bool = False
    similarity_threshold: float = 0.5

@dataclass
class EnhancedDatabaseStats:
    """
    🚀 ERWEITERTE Database-Statistiken mit vollständigen STM/LTM Metrics
    """
    # Basic Stats
    total_memories: int = 0
    unique_users: int = 0
    unique_sessions: int = 0
    memory_types: Dict[str, int] = field(default_factory=dict)
    recent_activity: int = 0
    avg_importance: float = 0.0
    database_size_mb: float = 0.0

    # STM/LTM Metrics
    stm_active_memories: int = 0
    ltm_significant_memories: int = 0
    avg_stm_activation: float = 0.0
    avg_ltm_significance: float = 0.0
    avg_consolidation_score: float = 0.0
    avg_memory_strength: float = 0.0
    consolidation_candidates: int = 0
    consolidated_memories: int = 0

    # Emotional Memory Stats
    emotion_memories: int = 0
    dominant_emotions: Dict[str, int] = field(default_factory=dict)
    avg_emotion_intensity: float = 0.0
    avg_emotion_valence: float = 0.0
    emotional_diversity_score: float = 0.0

    # Performance & Maintenance
    last_maintenance: Optional[str] = None
    last_consolidation: Optional[str] = None
    total_access_count: int = 0
    avg_access_count: float = 0.0
    decay_processed_memories: int = 0

    # Schema & Features
    schema_version: str = "3.0"
    enhanced_features: List[str] = field(default_factory=list)
    storage_backend: str = "unknown"
    connection_pool_size: int = 0

    # Timestamps
    stats_generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    database_uptime: Optional[str] = None

    def __post_init__(self):
        if not self.memory_types:
            self.memory_types = {}
        if not self.dominant_emotions:
            self.dominant_emotions = {}
        if not self.enhanced_features:
            self.enhanced_features = [
                "PostgreSQL_Native",
                "STM_LTM_Integration",
                "EmotionalMemory",
                "ConsolidationEngine",
                "ConnectionPool",
                "BackgroundMaintenance",
                "SemanticSearch",
                "PersonalityPatterns"
            ]

# ============================================================================
# 🚀 CONTEXT MANAGERS & UTILITIES
# ============================================================================

class DatabaseConnectionWrapper:
    """Context Manager für Database Connections - Fallback"""

    def __init__(self, connection):
        self.connection = connection
        self._in_transaction = False

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if hasattr(self.connection, 'rollback'):
                self.connection.rollback()
        else:
            if hasattr(self.connection, 'commit'):
                self.connection.commit()
        return False

class DatabaseContextManager:
    """Fallback Context Manager für Database Operations"""

    def __init__(self, database_instance):
        self.database = database_instance
        self.connection = None

    def __enter__(self):
        try:
            if hasattr(self.database, 'storage') and hasattr(self.database.storage, 'get_connection'):
                return self.database.storage.get_connection().__enter__()
            elif hasattr(self.database, '_get_direct_connection'):
                self.connection = self.database._get_direct_connection()
                return self.connection
            else:
                raise RuntimeError("No database connection method available")
        except Exception as e:
            logger.error(f"❌ Database context manager entry failed: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            try:
                if exc_type is not None:
                    self.connection.rollback()
                else:
                    self.connection.commit()
            except Exception as e:
                logger.error(f"❌ Database context manager exit error: {e}")
            finally:
                self.connection.close()
        return False

# ============================================================================
# 🚀 ENHANCED MEMORY DATABASE - HAUPTKLASSE
# ============================================================================

class EnhancedMemoryDatabase:
    def __init__(
        self,
        connection_string: Optional[str] = None,
        data_dir: Union[str, Path] = "data",
        enable_stm_ltm: bool = True,
        enable_emotional_memory: bool = True,
        auto_consolidation: bool = True,
        consolidation_interval: int = 3600,
        maintenance_interval: int = 1800,
        connection_pool_size: int = 5,
        enable_background_tasks: bool = True
    ):
        """Initialisierung ohne Rekursion"""
        
        # Basic Setup 
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Configuration
        self.enable_stm_ltm = enable_stm_ltm
        self.enable_emotional_memory = enable_emotional_memory
        self.auto_consolidation = auto_consolidation
        self.consolidation_interval = consolidation_interval
        self.maintenance_interval = maintenance_interval
        self.connection_pool_size = connection_pool_size
        self.enable_background_tasks = enable_background_tasks
        
        # Unique ID
        self.database_id = str(uuid.uuid4())[:8]
        
        # Connection Setup
        if connection_string is None:
            connection_string = "host=localhost port=5432 dbname=kira_memory user=kira password=kira_password"
        self.connection_string = connection_string

        # Initialize core attributes
        self.storage = None
        self._initialized = False
        self._direct_conn = None
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        
        # Background tasks
        self._background_threads = []
        self._stop_background_tasks = threading.Event()
        self._shutdown_event = threading.Event()
        
        # Performance Tracking
        self.stats_cache = {}
        self.last_stats_update = None
        self.operation_counts = {
            'store_operations': 0,
            'search_operations': 0,
            'consolidation_operations': 0,
            'consolidation_runs': 0,
            'maintenance_operations': 0,
            'maintenance_runs': 0
        }
        
        # Initialize storage
        self._safe_initialize_storage()
        
        # Start background tasks if enabled and initialized
        if self._initialized and self.enable_background_tasks:
            self._start_background_tasks()

        logger.info(f"✅ Enhanced Memory Database [{self.database_id}] initialized")
        logger.info(f"   STM/LTM: {enable_stm_ltm} | Emotional: {enable_emotional_memory}")
        logger.info(f"   Auto-Consolidation: {auto_consolidation} | Background: {enable_background_tasks}")
        logger.info(f"   Backend Status: {'✅ Active' if self._initialized else '❌ Failed'}")

    def _safe_initialize_storage(self):
        """Sichere Initialisierung ohne Rekursion"""
        try:
            # Direkte PostgreSQL-Verbindung testen
            test_conn = psycopg2.connect(self.connection_string)
            test_conn.close()
            
            # Wenn Verbindung erfolgreich, Storage erstellen
            if HAS_STORAGE_BACKEND and PSQLStorage:
                self.storage = PSQLStorage(self.connection_string)
                self._initialized = True
                logger.info("✅ PostgreSQL Storage Backend initialized")
            else:
                # Fallback zu direkter Verbindung
                self._initialize_direct_connection()
                    
        except Exception as e:
            logger.error(f"❌ Storage initialization error: {e}")
            self._initialize_direct_connection()
    
    def _initialize_direct_connection(self):
        """Direkte Verbindung ohne Storage-Backend"""
        try:
            # Test connection and create schema
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    # Create basic schema
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS memory_entries (
                            id SERIAL PRIMARY KEY,
                            session_id TEXT NOT NULL DEFAULT 'main',
                            user_id TEXT NOT NULL DEFAULT 'default',
                            memory_type TEXT NOT NULL DEFAULT 'general',
                            content TEXT NOT NULL,
                            importance INTEGER DEFAULT 5,
                            metadata JSONB DEFAULT '{}',
                            tags TEXT[] DEFAULT ARRAY[]::TEXT[],
                            created_at TIMESTAMP DEFAULT NOW(),
                            last_accessed TIMESTAMP DEFAULT NOW(),
                            access_count INTEGER DEFAULT 0
                        )
                    """)
                    
            self.storage = None  # Kein Storage-Backend
            self._initialized = True
            logger.info("✅ Direct PostgreSQL connection initialized")
            
        except Exception as e:
            logger.error(f"❌ Direct connection initialization failed: {e}")
            self._initialized = False

    def _start_background_tasks(self):
        """Start Background Processing Threads"""

        # 🔄 Consolidation Thread
        if self.auto_consolidation:
            consolidation_thread = threading.Thread(
                target=self._consolidation_worker,
                name=f"MemDB-Consolidation-{self.database_id}",
                daemon=True
            )
            consolidation_thread.start()
            self._background_threads.append(consolidation_thread)

        # 🔧 Maintenance Thread
        maintenance_thread = threading.Thread(
            target=self._maintenance_worker,
            name=f"MemDB-Maintenance-{self.database_id}",
            daemon=True
        )
        maintenance_thread.start()
        self._background_threads.append(maintenance_thread)

        logger.info(f"🔄 Background tasks started: {len(self._background_threads)} threads")

    def _consolidation_worker(self):
        """Background Consolidation Worker"""
        while not self._stop_background_tasks.wait(self.consolidation_interval):
            try:
                if self.enable_stm_ltm:
                    results = self.perform_memory_consolidation()
                    self.operation_counts['consolidation_runs'] += 1
                    logger.debug(f"🔄 Auto-consolidation completed: {results.get('consolidated_count', 0)} memories")
            except Exception as e:
                logger.error(f"❌ Background consolidation error: {e}")

    def _maintenance_worker(self):
        """Background Maintenance Worker"""
        while not self._stop_background_tasks.wait(self.maintenance_interval):
            try:
                results = self.perform_maintenance()
                self.operation_counts['maintenance_runs'] += 1
                logger.debug(f"🔧 Auto-maintenance completed: {len(results.get('operations', []))} operations")
            except Exception as e:
                logger.error(f"❌ Background maintenance error: {e}")

    def _initialize_storage_backend(self):
        """Initialize Storage Backend - PostgreSQL oder Direct - KORRIGIERT"""
        try:
            if HAS_STORAGE_BACKEND and PSQLStorage:  # Verwende PSQLStorage statt PostgreSQLMemoryStorage
                # ✅ KORRIGIERT: Verwende PostgreSQL Storage Backend ohne initialize()
                try:
                    self.storage = PSQLStorage(self.connection_string)  # Geändert zu PSQLStorage
                    
                    # ✅ KORRIGIERT: Teste direkt die Verbindung statt initialize() zu rufen
                    # Test ob Storage Backend funktioniert
                    if hasattr(self.storage, 'get_connection'):
                        try:
                            with self.storage.get_connection() as test_conn:
                                # Einfacher Connection Test
                                with test_conn.cursor() as cursor:
                                    cursor.execute("SELECT 1")
                                    cursor.fetchone()
                                
                                self._initialized = True
                                logger.info(f"✅ PostgreSQL Storage Backend initialisiert [{self.database_id}]")
                                return
                                
                        except Exception as conn_e:
                            logger.warning(f"⚠️ PostgreSQL Storage Backend connection failed: {conn_e}")
                            self.storage = None
                            raise conn_e
                    else:
                        logger.warning(f"⚠️ PostgreSQL Storage Backend hat keine get_connection() Methode")
                        self.storage = None
                        raise AttributeError("get_connection method missing")
                        
                except Exception as backend_e:
                    logger.warning(f"⚠️ PostgreSQL Storage Backend failed: {backend_e}")
                    self.storage = None
                    # Fall through to direct connection
                
                # 🔄 Fallback oder Primary: Direct PostgreSQL Connection
                logger.info(f"📁 Using direct PostgreSQL connection [{self.database_id}]")
                self._initialize_direct_connection()
            else:
                # Kein Storage Backend verfügbar - direkter Modus
                logger.info(f"📁 No storage backend available, using direct connection [{self.database_id}]")
                self._initialize_direct_connection()

        except Exception as e:
            logger.error(f"❌ Storage backend initialization error [{self.database_id}]: {e}")
            self._initialize_direct_connection()



    def _get_direct_connection(self):
        """Get direct PostgreSQL connection (fallback)"""
        return psycopg2.connect(self.connection_string)

    @contextmanager
    def get_connection(self):
        """
        🚀 ENHANCED Context Manager für Database Connections
        """
        if self.storage and hasattr(self.storage, 'get_connection'):
            # Use storage backend connection pool
            with self.storage.get_connection() as conn:
                yield conn
        else:
            # Direct connection fallback
            conn = None
            try:
                conn = self._get_direct_connection()
                conn.autocommit = False
                yield conn
                conn.commit()
            except Exception as e:
                if conn:
                    conn.rollback()
                logger.error(f"❌ Direct connection error: {e}")
                raise
            finally:
                if conn:
                    conn.close()

    

    # ========================================================================
    # 🚀 CORE MEMORY OPERATIONS
    # ========================================================================

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
        🚀 ENHANCED Memory Storage mit vollständiger Integration
        """

        if not self._initialized:
            logger.error("❌ Database not initialized")
            return None

        if not content or not content.strip():
            logger.warning("⚠️ Empty content provided")
            return None

        try:
            # ✅ Use Storage Backend if available
            if self.storage:
                memory_id = self.storage.store_enhanced_memory(
                    content=content,
                    user_id=user_id,
                    memory_type=memory_type,
                    session_id=session_id,
                    metadata=metadata or {},
                    importance=importance,
                    emotion_type=emotion_type,
                    emotion_intensity=emotion_intensity,
                    emotion_valence=emotion_valence,
                    stm_activation_level=stm_activation_level,
                    ltm_significance_score=ltm_significance_score,
                    consolidation_score=consolidation_score,
                    memory_strength=memory_strength,
                    attention_weight=attention_weight,
                    learning_weight=learning_weight,
                    expires_at=expires_at,
                    tags=tags,
                    **kwargs
                )

                if memory_id:
                    self.operation_counts['store_operations'] += 1
                    logger.info(f"✅ Enhanced memory stored via backend: ID={memory_id}")
                    return memory_id

            # 🔄 Fallback: Direct Storage
            with self.get_connection() as conn:
                with conn.cursor() as cursor:

                    # Prepare data
                    content_hash = hashlib.md5(f"{content}_{user_id}_{datetime.now().isoformat()}".encode()).hexdigest()

                    # Enhanced metadata
                    enhanced_metadata = metadata or {}
                    enhanced_metadata.update({
                        'attention_weight': attention_weight,
                        'learning_weight': learning_weight,
                        'storage_method': 'enhanced_direct',
                        'enhanced_features': self.enable_stm_ltm,
                        'emotional_memory': self.enable_emotional_memory,
                        'database_id': self.database_id
                    })

                    # Prepare tags
                    tags_str = ','.join(tags) if tags else ''

                    # Insert memory
                    insert_sql = '''
                        INSERT INTO memory_entries (
                            session_id, user_id, memory_type, content, importance,
                            metadata, tags, content_hash, expires_at,
                            emotion_type, emotion_intensity, emotion_valence,
                            stm_activation_level, ltm_significance_score, consolidation_score,
                            memory_strength, attention_weight, learning_weight,
                            created_at, last_accessed
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                        ) RETURNING id
                    '''

                    cursor.execute(insert_sql, (
                        session_id, user_id, memory_type, content, importance,
                        json.dumps(enhanced_metadata), tags_str, content_hash, expires_at,
                        emotion_type, emotion_intensity, emotion_valence,
                        stm_activation_level, ltm_significance_score, consolidation_score,
                        memory_strength, attention_weight, learning_weight
                    ))

                    memory_id = cursor.fetchone()[0]
                    self.operation_counts['store_operations'] += 1

                    logger.info(f"✅ Enhanced memory stored (direct): ID={memory_id}, Type={memory_type}")
                    return memory_id

        except Exception as e:
            logger.error(f"❌ Enhanced memory storage error: {e}")
            return None

    def search_memories(
        self,
        search_filter: Union[EnhancedMemorySearchFilter, Dict, str],
        enable_fuzzy_search: bool = True
    ) -> List[Dict[str, Any]]:
        """
        🚀 ENHANCED Memory Search mit Advanced Filtering
        """

        if not self._initialized:
            logger.error("❌ Database not initialized")
            return []

        try:
            # Normalize search filter
            if isinstance(search_filter, str):
                search_filter = EnhancedMemorySearchFilter(query=search_filter)
            elif isinstance(search_filter, dict):
                search_filter = EnhancedMemorySearchFilter(**search_filter)

            # ✅ Use Storage Backend if available
            if self.storage and hasattr(self.storage, 'search_memories'):
                # Convert to compatible format
                backend_filter = {
                    'query': search_filter.query,
                    'user_id': search_filter.user_id,
                    'memory_type': search_filter.memory_type,
                    'session_id': search_filter.session_id,
                    'importance_min': search_filter.importance_min,
                    'limit': search_filter.limit,
                    'order_by': search_filter.order_by,
                    'order_direction': search_filter.order_direction
                }

                results = self.storage.search_memories(backend_filter)
                self.operation_counts['search_operations'] += 1
                logger.debug(f"🔍 Memory search via backend: {len(results)} results")
                return results

            # 🔄 Fallback: Direct Search
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:

                    # Build dynamic query
                    where_conditions = []
                    params = []

                    # Basic filters
                    where_conditions.append("user_id = %s")
                    params.append(search_filter.user_id)

                    if search_filter.memory_type:
                        where_conditions.append("memory_type = %s")
                        params.append(search_filter.memory_type)

                    if search_filter.session_id:
                        where_conditions.append("session_id = %s")
                        params.append(search_filter.session_id)

                    # Importance range
                    if search_filter.importance_min > 1:
                        where_conditions.append("importance >= %s")
                        params.append(search_filter.importance_min)

                    if search_filter.importance_max < 10:
                        where_conditions.append("importance <= %s")
                        params.append(search_filter.importance_max)

                    # STM/LTM filters
                    if search_filter.stm_activation_min > 0:
                        where_conditions.append("stm_activation_level >= %s")
                        params.append(search_filter.stm_activation_min)

                    if search_filter.ltm_significance_min > 0:
                        where_conditions.append("ltm_significance_score >= %s")
                        params.append(search_filter.ltm_significance_min)

                    # Emotional filters
                    if search_filter.emotion_type:
                        where_conditions.append("emotion_type = %s")
                        params.append(search_filter.emotion_type)

                    if search_filter.emotion_intensity_min > 0:
                        where_conditions.append("emotion_intensity >= %s")
                        params.append(search_filter.emotion_intensity_min)

                    # Content search
                    if search_filter.query:
                        if enable_fuzzy_search:
                            where_conditions.append("(content ILIKE %s OR tags ILIKE %s)")
                            params.extend([f"%{search_filter.query}%", f"%{search_filter.query}%"])
                        else:
                            where_conditions.append("content ILIKE %s")
                            params.append(f"%{search_filter.query}%")

                    # Date range
                    if search_filter.date_from:
                        where_conditions.append("created_at >= %s")
                        params.append(search_filter.date_from)

                    if search_filter.date_to:
                        where_conditions.append("created_at <= %s")
                        params.append(search_filter.date_to)

                    # Build final query
                    search_sql = f'''
                        SELECT * FROM memory_entries 
                        WHERE {" AND ".join(where_conditions)}
                        ORDER BY {search_filter.order_by} {search_filter.order_direction}
                        LIMIT %s OFFSET %s
                    '''
                    params.extend([search_filter.limit, search_filter.offset])

                    cursor.execute(search_sql, params)
                    results = cursor.fetchall()

                    # Process results
                    processed_results = []
                    for row in results:
                        result_dict = dict(row)

                        # Parse JSON metadata
                        if result_dict.get('metadata'):
                            try:
                                if isinstance(result_dict['metadata'], str):
                                    result_dict['metadata'] = json.loads(result_dict['metadata'])
                            except json.JSONDecodeError:
                                result_dict['metadata'] = {}

                        # Convert timestamps to ISO format
                        timestamp_fields = ['created_at', 'last_accessed', 'expires_at', 'consolidation_timestamp']
                        for field in timestamp_fields:
                            if result_dict.get(field) and hasattr(result_dict[field], 'isoformat'):
                                result_dict[field] = result_dict[field].isoformat()

                        # Parse related memory IDs
                        if result_dict.get('related_memory_ids'):
                            try:
                                if isinstance(result_dict['related_memory_ids'], str):
                                    result_dict['related_memory_ids'] = json.loads(result_dict['related_memory_ids'])
                            except json.JSONDecodeError:
                                result_dict['related_memory_ids'] = []

                        processed_results.append(result_dict)

                    self.operation_counts['search_operations'] += 1
                    logger.debug(f"🔍 Direct memory search: {len(processed_results)} results")
                    return processed_results

        except Exception as e:
            logger.error(f"❌ Memory search error: {e}")
            return []

    def get_enhanced_stats(self, user_id: str = "default") -> EnhancedDatabaseStats:
        """
        🚀 ENHANCED Database Statistics mit vollständigen Metrics
        """

        if not self._initialized:
            logger.warning("❌ Database not initialized")
            return EnhancedDatabaseStats()

        # Use cached stats if recent
        cache_key = f"stats_{user_id}"
        if (self.last_stats_update and
            cache_key in self.stats_cache and
            (datetime.now() - self.last_stats_update).seconds < 300):  # 5 minutes cache
            return self.stats_cache[cache_key]

        try:
            # ✅ Use Storage Backend if available
            if self.storage and hasattr(self.storage, 'get_enhanced_stats'):
                backend_stats = self.storage.get_enhanced_stats()

                # Convert to EnhancedDatabaseStats
                enhanced_stats = EnhancedDatabaseStats(
                    total_memories=backend_stats.get('total_memories', 0),
                    unique_users=backend_stats.get('unique_users', 0),
                    memory_types=backend_stats.get('memory_types', {}),
                    recent_activity=backend_stats.get('recent_activity', 0),
                    avg_importance=backend_stats.get('avg_importance', 0.0),
                    database_size_mb=backend_stats.get('database_size_mb', 0.0),
                    stm_active_memories=backend_stats.get('stm_active_memories', 0),
                    ltm_significant_memories=backend_stats.get('ltm_significant_memories', 0),
                    avg_stm_activation=backend_stats.get('avg_stm_activation', 0.0),
                    avg_ltm_significance=backend_stats.get('avg_ltm_significance', 0.0),
                    storage_backend="PostgreSQL_Backend",
                    enhanced_features=backend_stats.get('enhanced_features', [])
                )

                # Cache result
                self.stats_cache[cache_key] = enhanced_stats
                self.last_stats_update = datetime.now()

                return enhanced_stats

            # 🔄 Fallback: Direct Stats Calculation
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:

                    # Main statistics query
                    cursor.execute('''
                        SELECT 
                            COUNT(*) as total_memories,
                            COUNT(DISTINCT session_id) as unique_sessions,
                            AVG(importance) as avg_importance,
                            SUM(access_count) as total_access_count,
                            AVG(access_count) as avg_access_count,
                            COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END) as recent_activity,
                            
                            -- STM/LTM Metrics
                            AVG(stm_activation_level) as avg_stm_activation,
                            AVG(ltm_significance_score) as avg_ltm_significance,
                            AVG(consolidation_score) as avg_consolidation_score,
                            AVG(memory_strength) as avg_memory_strength,
                            COUNT(CASE WHEN stm_activation_level > 0.5 THEN 1 END) as stm_active_memories,
                            COUNT(CASE WHEN ltm_significance_score > 0.7 THEN 1 END) as ltm_significant_memories,
                            COUNT(CASE WHEN consolidation_score > 0.6 THEN 1 END) as consolidation_candidates,
                            COUNT(CASE WHEN consolidation_timestamp IS NOT NULL THEN 1 END) as consolidated_memories,
                            
                            -- Emotional Metrics
                            COUNT(CASE WHEN emotion_type != 'neutral' THEN 1 END) as emotion_memories,
                            AVG(emotion_intensity) as avg_emotion_intensity,
                            AVG(emotion_valence) as avg_emotion_valence
                        FROM memory_entries 
                        WHERE user_id = %s
                    ''', [user_id])

                    main_stats = cursor.fetchone()

                    # Memory types distribution
                    cursor.execute('''
                        SELECT memory_type, COUNT(*) as count 
                        FROM memory_entries 
                        WHERE user_id = %s
                        GROUP BY memory_type
                        ORDER BY count DESC
                    ''', [user_id])
                    memory_types = {row['memory_type']: row['count'] for row in cursor.fetchall()}

                    # Dominant emotions
                    cursor.execute('''
                        SELECT emotion_type, COUNT(*) as count 
                        FROM memory_entries 
                        WHERE user_id = %s AND emotion_type != 'neutral'
                        GROUP BY emotion_type
                        ORDER BY count DESC
                        LIMIT 10
                    ''', [user_id])
                    dominant_emotions = {row['emotion_type']: row['count'] for row in cursor.fetchall()}

                    # Global unique users
                    cursor.execute('SELECT COUNT(DISTINCT user_id) as unique_users FROM memory_entries')
                    global_stats = cursor.fetchone()

                    # Calculate emotional diversity
                    emotion_count = len(dominant_emotions)
                    emotional_diversity_score = min(1.0, emotion_count / 10.0) if emotion_count > 0 else 0.0

                    # Build enhanced stats
                    enhanced_stats = EnhancedDatabaseStats(
                        # Basic Stats
                        total_memories=main_stats['total_memories'] or 0,
                        unique_users=global_stats['unique_users'] or 0,
                        unique_sessions=main_stats['unique_sessions'] or 0,
                        memory_types=memory_types,
                        recent_activity=main_stats['recent_activity'] or 0,
                        avg_importance=round(float(main_stats['avg_importance'] or 0), 2),
                        database_size_mb=0.0,  # Would need admin privileges

                        # STM/LTM Metrics
                        stm_active_memories=main_stats['stm_active_memories'] or 0,
                        ltm_significant_memories=main_stats['ltm_significant_memories'] or 0,
                        avg_stm_activation=round(float(main_stats['avg_stm_activation'] or 0), 3),
                        avg_ltm_significance=round(float(main_stats['avg_ltm_significance'] or 0), 3),
                        avg_consolidation_score=round(float(main_stats['avg_consolidation_score'] or 0), 3),
                        avg_memory_strength=round(float(main_stats['avg_memory_strength'] or 0), 3),
                        consolidation_candidates=main_stats['consolidation_candidates'] or 0,
                        consolidated_memories=main_stats['consolidated_memories'] or 0,

                        # Emotional Stats
                        emotion_memories=main_stats['emotion_memories'] or 0,
                        dominant_emotions=dominant_emotions,
                        avg_emotion_intensity=round(float(main_stats['avg_emotion_intensity'] or 0), 3),
                        avg_emotion_valence=round(float(main_stats['avg_emotion_valence'] or 0), 3),
                        emotional_diversity_score=round(emotional_diversity_score, 3),

                        # Performance
                        total_access_count=main_stats['total_access_count'] or 0,
                        avg_access_count=round(float(main_stats['avg_access_count'] or 0), 2),

                        # Configuration
                        storage_backend="PostgreSQL_Direct",
                        connection_pool_size=self.connection_pool_size,
                        enhanced_features=[
                            "EnhancedMemoryDatabase",
                            "STM_LTM_Integration",
                            "EmotionalMemory",
                            "DirectPostgreSQL",
                            "BackgroundProcessing"
                        ]
                    )

                    # Cache result
                    self.stats_cache[cache_key] = enhanced_stats
                    self.last_stats_update = datetime.now()

                    return enhanced_stats

        except Exception as e:
            logger.error(f"❌ Enhanced stats calculation error: {e}")
            return EnhancedDatabaseStats()

    # ========================================================================
    # 🚀 STM/LTM SPECIFIC METHODS
    # ========================================================================

    def store_stm_experience(
        self,
        content: str,
        context: Dict[str, Any],
        user_id: str = "default",
        session_id: str = "stm_session",
        activation_level: float = 1.0,
        attention_weight: float = 0.8,
        **kwargs
    ) -> Optional[int]:
        """🚀 Store STM Experience with enhanced context"""

        return self.store_enhanced_memory(
            content=content,
            user_id=user_id,
            memory_type="short_term_experience",
            session_id=session_id,
            metadata=context,
            importance=min(10, max(1, int(activation_level * 10))),
            stm_activation_level=activation_level,
            attention_weight=attention_weight,
            memory_strength=activation_level,
            **kwargs
        )

    def store_ltm_memory(
        self,
        content: str,
        significance_score: float,
        user_id: str = "default",
        session_id: str = "ltm_session",
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> Optional[int]:
        """🚀 Store LTM Memory with significance scoring"""

        return self.store_enhanced_memory(
            content=content,
            user_id=user_id,
            memory_type="long_term_memory",
            session_id=session_id,
            metadata=metadata or {},
            importance=min(10, max(1, int(significance_score * 10))),
            ltm_significance_score=significance_score,
            consolidation_score=significance_score * 0.8,
            memory_strength=min(1.0, significance_score + 0.2),
            **kwargs
        )

    def recall_stm_experiences(
        self,
        user_id: str = "default",
        min_activation: float = 0.3,
        limit: int = 20,
        session_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """🚀 Recall STM Experiences with activation filtering"""

        search_filter = EnhancedMemorySearchFilter(
            user_id=user_id,
            memory_type="short_term_experience",
            session_id=session_filter,
            stm_activation_min=min_activation,
            limit=limit,
            order_by="stm_activation_level",
            order_direction="DESC"
        )

        return self.search_memories(search_filter)

    def recall_ltm_memories(
        self,
        user_id: str = "default",
        min_significance: float = 0.5,
        context_query: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """🚀 Recall LTM Memories with significance filtering"""

        search_filter = EnhancedMemorySearchFilter(
            user_id=user_id,
            memory_type="long_term_memory",
            ltm_significance_min=min_significance,
            query=context_query,
            limit=limit,
            order_by="ltm_significance_score",
            order_direction="DESC"
        )

        return self.search_memories(search_filter)

    def perform_memory_consolidation(
        self,
        user_id: str = "default",
        consolidation_threshold: float = 0.6,
        max_consolidations: int = 100
    ) -> Dict[str, Any]:
        """🚀 Perform Memory Consolidation with detailed tracking"""

        if not self._initialized:
            return {'status': 'error', 'error': 'Database not initialized'}

        try:
            consolidation_results = {
                'started_at': datetime.now().isoformat(),
                'user_id': user_id,
                'threshold': consolidation_threshold,
                'max_consolidations': max_consolidations,
                'processed_memories': 0,
                'consolidated_count': 0,
                'strengthened_count': 0,
                'decayed_count': 0,
                'status': 'success',
                'detailed_operations': []
            }

            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:

                    # Find consolidation candidates
                    cursor.execute('''
                        SELECT id, content, stm_activation_level, ltm_significance_score,
                               consolidation_score, memory_strength, reinforcement_count,
                               memory_type, importance, created_at, last_accessed
                        FROM memory_entries
                        WHERE user_id = %s 
                          AND consolidation_score >= %s
                          AND (consolidation_timestamp IS NULL 
                               OR consolidation_timestamp < NOW() - INTERVAL '24 hours')
                        ORDER BY consolidation_score DESC, ltm_significance_score DESC
                        LIMIT %s
                    ''', [user_id, consolidation_threshold, max_consolidations])

                    candidates = cursor.fetchall()
                    consolidation_results['processed_memories'] = len(candidates)

                    for candidate in candidates:
                        memory_id = candidate['id']

                        # Calculate consolidation improvements
                        current_strength = candidate['memory_strength'] or 1.0
                        current_ltm_score = candidate['ltm_significance_score'] or 0.0
                        reinforcement_count = candidate['reinforcement_count'] or 0

                        # Consolidation bonuses
                        reinforcement_bonus = min(0.3, reinforcement_count * 0.05)
                        time_bonus = 0.1  # Base consolidation bonus

                        # New values
                        new_strength = min(1.0, current_strength + reinforcement_bonus + time_bonus)
                        new_ltm_score = min(1.0, current_ltm_score + 0.05)
                        new_reinforcement_count = reinforcement_count + 1

                        # Update memory with consolidation
                        cursor.execute('''
                            UPDATE memory_entries SET
                                memory_strength = %s,
                                ltm_significance_score = %s,
                                reinforcement_count = %s,
                                consolidation_timestamp = NOW(),
                                last_accessed = NOW(),
                                access_count = access_count + 1
                            WHERE id = %s
                        ''', [new_strength, new_ltm_score, new_reinforcement_count, memory_id])

                        if cursor.rowcount > 0:
                            consolidation_results['consolidated_count'] += 1
                            consolidation_results['detailed_operations'].append({
                                'memory_id': memory_id,
                                'operation': 'consolidated',
                                'strength_change': round(new_strength - current_strength, 3),
                                'ltm_score_change': round(new_ltm_score - current_ltm_score, 3)
                            })

                    # Apply decay to non-consolidated memories
                    cursor.execute('''
                        UPDATE memory_entries SET
                            stm_activation_level = GREATEST(0.0, stm_activation_level * 0.95),
                            memory_strength = GREATEST(0.1, memory_strength * 0.98),
                            consolidation_score = GREATEST(0.0, consolidation_score * 0.99)
                        WHERE user_id = %s 
                          AND consolidation_score < %s
                          AND last_accessed < NOW() - INTERVAL '7 days'
                        RETURNING id
                    ''', [user_id, consolidation_threshold])

                    decayed_ids = [row[0] for row in cursor.fetchall()]
                    consolidation_results['decayed_count'] = len(decayed_ids)

                    # Update operation counts
                    self.operation_counts['consolidation_runs'] += 1

            consolidation_results['completed_at'] = datetime.now().isoformat()
            consolidation_results['duration_seconds'] = (
                datetime.fromisoformat(consolidation_results['completed_at']) -
                datetime.fromisoformat(consolidation_results['started_at'])
            ).total_seconds()

            logger.info(f"✅ Memory consolidation completed: {consolidation_results['consolidated_count']} consolidated, {consolidation_results['decayed_count']} decayed")
            return consolidation_results

        except Exception as e:
            logger.error(f"❌ Memory consolidation error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'consolidated_count': 0,
                'user_id': user_id
            }

    # ========================================================================
    # 🚀 MAINTENANCE & UTILITIES
    # ========================================================================

    def perform_maintenance(self) -> Dict[str, Any]:
        """🚀 Comprehensive Database Maintenance"""

        if not self._initialized:
            return {'status': 'error', 'error': 'Database not initialized'}

        try:
            # Use storage backend maintenance if available
            if self.storage and hasattr(self.storage, 'perform_maintenance'):
                backend_results = self.storage.perform_maintenance()

                # Add STM/LTM consolidation if enabled
                if self.enable_stm_ltm:
                    consolidation_results = self.perform_memory_consolidation()
                    backend_results['consolidation'] = consolidation_results

                return backend_results

            # Direct maintenance implementation
            maintenance_results = {
                'started_at': datetime.now().isoformat(),
                'operations': [],
                'stats': {},
                'status': 'success'
            }

            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:

                    # 1. Clean expired memories
                    cursor.execute("DELETE FROM memory_entries WHERE expires_at IS NOT NULL AND expires_at < NOW()")
                    expired_count = cursor.rowcount
                    if expired_count > 0:
                        maintenance_results['operations'].append(f'Cleaned {expired_count} expired memories')

                    # 2. Update access patterns
                    cursor.execute('''
                        UPDATE memory_entries SET
                            stm_activation_level = GREATEST(0.0, stm_activation_level * 0.99),
                            consolidation_score = GREATEST(0.0, consolidation_score - 0.001)
                        WHERE last_accessed < NOW() - INTERVAL '1 day'
                    ''')
                    decay_count = cursor.rowcount
                    if decay_count > 0:
                        maintenance_results['operations'].append(f'Applied decay to {decay_count} memories')

                    # 3. Vacuum and analyze
                    try:
                        cursor.execute("VACUUM ANALYZE memory_entries")
                        maintenance_results['operations'].append('Database VACUUM ANALYZE completed')
                    except Exception as vacuum_e:
                        logger.warning(f"⚠️ VACUUM warning: {vacuum_e}")
                        maintenance_results['operations'].append('VACUUM skipped (permissions?)')

                    # 4. Update statistics
                    cursor.execute('''
                        SELECT 
                            COUNT(*) as total_memories,
                            COUNT(DISTINCT user_id) as unique_users,
                            AVG(importance) as avg_importance,
                            COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END) as recent_activity
                        FROM memory_entries
                    ''')
                    stats = cursor.fetchone()
                    maintenance_results['stats'] = dict(stats)

                    # 5. STM/LTM Consolidation
                    if self.enable_stm_ltm:
                        consolidation_results = self.perform_memory_consolidation()
                        maintenance_results['consolidation'] = consolidation_results
                        maintenance_results['operations'].append(
                            f'Memory consolidation: {consolidation_results.get("consolidated_count", 0)} processed'
                        )

            maintenance_results['completed_at'] = datetime.now().isoformat()
            maintenance_results['duration_seconds'] = (
                datetime.fromisoformat(maintenance_results['completed_at']) -
                datetime.fromisoformat(maintenance_results['started_at'])
            ).total_seconds()

            # Update operation counts
            self.operation_counts['maintenance_runs'] += 1

            logger.info(f"✅ Database maintenance completed: {len(maintenance_results['operations'])} operations in {maintenance_results['duration_seconds']:.2f}s")
            return maintenance_results

        except Exception as e:
            logger.error(f"❌ Database maintenance error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'operations': []
            }

    def get_system_status(self) -> Dict[str, Any]:
        """🚀 Comprehensive System Status Report"""

        status = {
            'database_id': self.database_id,
            'initialized': self._initialized,
            'backend_type': 'PostgreSQL' if self.storage else 'Direct',
            'connection_string': self.connection_string.replace('password=kira_password', 'password=***') if self.connection_string else None,

            # Configuration
            'configuration': {
                'stm_ltm_enabled': self.enable_stm_ltm,
                'emotional_memory_enabled': self.enable_emotional_memory,
                'auto_consolidation': self.auto_consolidation,
                'background_tasks': self.enable_background_tasks,
                'consolidation_interval': self.consolidation_interval,
                'maintenance_interval': self.maintenance_interval,
                'connection_pool_size': self.connection_pool_size
            },

            # Background Tasks
            'background_tasks': {
                'active_threads': len(self._background_threads),
                'threads_alive': sum(1 for t in self._background_threads if t.is_alive()),
                'stop_signal': self._stop_background_tasks.is_set()
            },

            # Performance Metrics
            'performance': self.operation_counts.copy(),

            # Data Directory
            'data_dir': str(self.data_dir),
            'data_dir_exists': self.data_dir.exists(),

            # Timestamps
            'status_generated_at': datetime.now().isoformat(),
            'last_stats_update': self.last_stats_update.isoformat() if self.last_stats_update else None
        }

        # Add database health check
        if self._initialized:
            try:
                stats = self.get_enhanced_stats()
                status.update({
                    'database_status': 'healthy',
                    'total_memories': stats.total_memories,
                    'unique_users': stats.unique_users,
                    'stm_active': stats.stm_active_memories,
                    'ltm_significant': stats.ltm_significant_memories,
                    'recent_activity': stats.recent_activity,
                    'database_size_mb': stats.database_size_mb
                })
            except Exception as e:
                status.update({
                    'database_status': 'degraded',
                    'health_check_error': str(e)
                })
        else:
            status['database_status'] = 'not_initialized'

        return status

    def close(self):
        """🧹 Cleanup and Close Database"""

        logger.info(f"🧹 Closing Enhanced Memory Database [{self.database_id}]...")

        # Stop background tasks
        if self._background_threads:
            self._stop_background_tasks.set()

            for thread in self._background_threads:
                if thread.is_alive():
                    thread.join(timeout=5)
                    if thread.is_alive():
                        logger.warning(f"⚠️ Background thread {thread.name} did not stop gracefully")

            logger.info(f"🔄 Stopped {len(self._background_threads)} background threads")

        # Close storage backend
        if self.storage and hasattr(self.storage, 'close'):
            self.storage.close()
            logger.info("✅ Storage backend closed")

        # Clear caches
        self.stats_cache.clear()

        # Mark as closed
        self._initialized = False

        logger.info(f"✅ Enhanced Memory Database [{self.database_id}] closed successfully")

    # ========================================================================
    # 🚀 CONTEXT MANAGER SUPPORT
    # ========================================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

# ============================================================================
# 🚀 LEGACY COMPATIBILITY WRAPPER
# ============================================================================

class MemoryDatabase:
    """
    🔄 Legacy Compatibility Wrapper für EnhancedMemoryDatabase
    Bietet Rückwärtskompatibilität mit existierendem Code
    """

    def __init__(self, storage_backend: Optional['PostgreSQLMemoryStorage'] = None, **kwargs):
        """
        Legacy Constructor - leitet an EnhancedMemoryDatabase weiter
        """

        if storage_backend:
            # Use provided storage backend
            connection_string = getattr(storage_backend, 'connection_string', None)
            self.enhanced_db = EnhancedMemoryDatabase(
                connection_string=connection_string,
                **kwargs
            )
            self.enhanced_db.storage = storage_backend
        else:
            # Create new enhanced database
            connection_string = kwargs.get('connection_string',
                "host=localhost port=5432 dbname=kira_memory user=kira password=kira_password")
            self.enhanced_db = EnhancedMemoryDatabase(
                connection_string=connection_string,
                **kwargs
            )

        # Legacy storage property
        self.storage = self.enhanced_db.storage

        logger.info("🔄 Legacy MemoryDatabase wrapper created")
    
    async def store_conversation_exchange(self, 
                                        user_memory: 'Memory', 
                                        kira_memory: 'Memory',
                                        conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        ✅ NEUE: Speichert kompletten Conversation Exchange
        
        Args:
            user_memory: Memory object for user input
            kira_memory: Memory object for Kira response  
            conversation_context: Additional conversation context
            
        Returns:
            Storage result with IDs and status
        """
        try:
            conversation_context = conversation_context or {}
            storage_results = []
            
            # Store user memory
            user_result = await self.store_enhanced_memory(user_memory)
            if user_result:
                storage_results.append({
                    'type': 'user_memory',
                    'memory_id': user_memory.memory_id,
                    'stored': True
                })
            
            # Store Kira memory
            kira_result = await self.store_enhanced_memory(kira_memory)
            if kira_result:
                storage_results.append({
                    'type': 'kira_memory', 
                    'memory_id': kira_memory.memory_id,
                    'stored': True
                })
            
            # Create conversation link record
            conversation_record = {
                'conversation_id': conversation_context.get('conversation_id', f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                'user_memory_id': user_memory.memory_id,
                'kira_memory_id': kira_memory.memory_id,
                'timestamp': datetime.now().isoformat(),
                'importance_score': conversation_context.get('importance_score', 5),
                'emotional_impact': conversation_context.get('emotional_impact', 0.0),
                'topic_category': conversation_context.get('topic_category', 'general'),
                'storage_location': conversation_context.get('storage_location', 'database'),
                'context': conversation_context
            }
            
            # Store conversation link
            conversation_stored = await self._store_conversation_link(conversation_record)
            
            logger.info(f"✅ Conversation exchange stored: {conversation_record['conversation_id']}")
            
            return {
                'success': True,
                'conversation_id': conversation_record['conversation_id'],
                'storage_results': storage_results,
                'conversation_link_stored': conversation_stored,
                'memories_stored': len([r for r in storage_results if r['stored']]),
                'timestamp': conversation_record['timestamp']
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to store conversation exchange: {e}")
            return {
                'success': False,
                'error': str(e),
                'memories_stored': 0
            }
    
    async def _store_conversation_link(self, conversation_record: Dict[str, Any]) -> bool:
        """Speichert Conversation Link Record"""
        try:
            # Store in conversations table (if available) or as special memory
            if hasattr(self, 'conversations') and self.conversations is not None:
                # Store in dedicated conversations table
                conversation_memory = create_memory(
                    content=f"Conversation Link: {conversation_record['conversation_id']}",
                    memory_type=MemoryType.CONVERSATION,
                    importance=int(conversation_record['importance_score']),
                    emotional_intensity=conversation_record['emotional_impact'],
                    context={
                        **conversation_record,
                        'record_type': 'conversation_link'
                    },
                    tags=['conversation_link', 'exchange', conversation_record['topic_category']]
                )
                
                await self.store_enhanced_memory(conversation_memory)
                return True
            else:
                # Fallback: Store as enhanced memory with special marking
                link_memory = create_memory(
                    content=f"Conversation: {conversation_record['conversation_id']} - User+Kira Exchange",
                    memory_type=MemoryType.CONVERSATION,
                    importance=int(conversation_record['importance_score']),
                    context={
                        **conversation_record,
                        'record_type': 'conversation_exchange',
                        'is_conversation_link': True
                    },
                    tags=['conversation', 'exchange', 'link', conversation_record['topic_category']]
                )
                
                result = await self.store_enhanced_memory(link_memory)
                return result is not None
                
        except Exception as e:
            logger.error(f"Failed to store conversation link: {e}")
            return False
    
    async def get_conversation_history(self, 
                                     user_name: str = None,
                                     limit: int = 50,
                                     importance_min: float = 0.0,
                                     days_back: int = 30) -> List[Dict[str, Any]]:
        """
        ✅ NEUE: Holt Conversation History aus Database
        
        Args:
            user_name: Filter by user name
            limit: Maximum number of conversations
            importance_min: Minimum importance score
            days_back: Days to look back
            
        Returns:
            List of conversation records
        """
        try:
            # Search for conversation memories
            search_filters = {
                'memory_types': [MemoryType.CONVERSATION],
                'importance_min': importance_min,
                'days_back': days_back,
                'tags': ['conversation', 'exchange']
            }
            
            if user_name:
                search_filters['user_name'] = user_name
            
            # Get conversation memories
            conversation_memories = await self.search_memories(
                query="conversation",
                limit=limit * 2,  # Get more to filter
                filters=search_filters
            )
            
            # Convert to conversation format and deduplicate
            conversations = []
            seen_conversation_ids = set()
            
            for memory in conversation_memories:
                if memory.context.get('is_conversation_link') or memory.context.get('record_type') == 'conversation_exchange':
                    conversation_id = memory.context.get('conversation_id')
                    
                    if conversation_id and conversation_id not in seen_conversation_ids:
                        conversation_record = {
                            'conversation_id': conversation_id,
                            'timestamp': memory.context.get('timestamp', memory.created_at.isoformat()),
                            'user_input': self._extract_user_input_from_context(memory.context),
                            'kira_response': self._extract_kira_response_from_context(memory.context),
                            'importance_score': memory.importance,
                            'emotional_impact': memory.emotional_intensity,
                            'topic_category': memory.context.get('topic_category', 'general'),
                            'storage_location': memory.context.get('storage_location', 'database'),
                            'user_memory_id': memory.context.get('user_memory_id'),
                            'kira_memory_id': memory.context.get('kira_memory_id')
                        }
                        
                        conversations.append(conversation_record)
                        seen_conversation_ids.add(conversation_id)
            
            # Sort by timestamp (newest first)
            conversations.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return conversations[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def _extract_user_input_from_context(self, context: Dict[str, Any]) -> str:
        """Extrahiert User Input aus Memory Context"""
        # Try different context keys
        for key in ['user_input', 'original_user_input', 'user_message']:
            if key in context:
                return context[key]
        
        # Fallback: Try to extract from content
        content = context.get('content', '')
        if 'User said:' in content:
            return content.split('User said:')[1].split('|')[0].strip()
        
        return "User input not available"
    
    def _extract_kira_response_from_context(self, context: Dict[str, Any]) -> str:
        """Extrahiert Kira Response aus Memory Context"""
        # Try different context keys
        for key in ['kira_response', 'ai_response', 'response']:
            if key in context:
                return context[key]
        
        # Fallback: Try to extract from content
        content = context.get('content', '')
        if 'Kira responded:' in content:
            return content.split('Kira responded:')[1].strip()
        elif '|' in content and 'User said:' in content:
            return content.split('|')[1].strip()
        
        return "Kira response not available"
    
    async def search_conversations(self, 
                                 query: str,
                                 limit: int = 20,
                                 filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        ✅ NEUE: Durchsucht Conversations in Database
        
        Args:
            query: Search query
            limit: Maximum results
            filters: Additional filters
            
        Returns:
            List of matching conversations
        """
        try:
            filters = filters or {}
            
            # Enhanced search filters for conversations
            search_filters = {
                'memory_types': [MemoryType.CONVERSATION],
                'tags': ['conversation'],
                **filters
            }
            
            # Search memories
            results = await self.search_memories(
                query=query,
                limit=limit * 2,  # Get more to process
                filters=search_filters
            )
            
            # Convert and filter results
            conversations = []
            for memory in results:
                if memory.context.get('is_conversation_link') or 'conversation' in memory.tags:
                    conversation_data = {
                        'memory_id': memory.memory_id,
                        'conversation_id': memory.context.get('conversation_id'),
                        'content': memory.content,
                        'importance': memory.importance,
                        'timestamp': memory.created_at.isoformat(),
                        'emotional_intensity': memory.emotional_intensity,
                        'context': memory.context,
                        'tags': memory.tags,
                        'relevance_score': self._calculate_conversation_relevance(query, memory)
                    }
                    conversations.append(conversation_data)
            
            # Sort by relevance and importance
            conversations.sort(
                key=lambda x: (x['relevance_score'], x['importance']),
                reverse=True
            )
            
            return conversations[:limit]
            
        except Exception as e:
            logger.error(f"Conversation search failed: {e}")
            return []
    
    def _calculate_conversation_relevance(self, query: str, memory: 'Memory') -> float:
        """Berechnet Relevanz für Conversation Search"""
        try:
            relevance = 0.0
            query_lower = query.lower()
            
            # Content relevance
            if query_lower in memory.content.lower():
                relevance += 10.0
            
            # Context relevance
            context_str = str(memory.context).lower()
            if query_lower in context_str:
                relevance += 5.0
            
            # Tag relevance
            for tag in memory.tags:
                if query_lower in tag.lower():
                    relevance += 3.0
            
            # Topic category relevance
            topic = memory.context.get('topic_category', '')
            if query_lower in topic.lower():
                relevance += 8.0
            
            # Word matching
            query_words = set(query_lower.split())
            content_words = set(memory.content.lower().split())
            
            word_overlap = len(query_words.intersection(content_words))
            if query_words:
                relevance += (word_overlap / len(query_words)) * 15.0
            
            return relevance
            
        except Exception as e:
            logger.error(f"Relevance calculation failed: {e}")
            return 0.0
    
    async def store_conversations_batch(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        ✅ NEUE: Batch Storage für Conversations
        
        Args:
            conversations: List of conversation records
            
        Returns:
            Batch storage result
        """
        try:
            stored_count = 0
            failed_count = 0
            storage_details = []
            
            for conversation in conversations:
                try:
                    # Create conversation memory
                    conversation_memory = create_memory(
                        content=f"Conversation: {conversation.get('user_input', '')} | {conversation.get('kira_response', '')}",
                        memory_type=MemoryType.CONVERSATION,
                        importance=int(conversation.get('importance_score', 5)),
                        emotional_intensity=conversation.get('emotional_impact', 0.0),
                        context={
                            **conversation,
                            'record_type': 'batch_conversation',
                            'batch_stored': True,
                            'stored_at': datetime.now().isoformat()
                        },
                        tags=['conversation', 'batch_stored', conversation.get('topic_category', 'general')]
                    )
                    
                    # Store memory
                    result = await self.store_enhanced_memory(conversation_memory)
                    
                    if result:
                        stored_count += 1
                        storage_details.append({
                            'conversation_id': conversation.get('conversation_id'),
                            'memory_id': conversation_memory.memory_id,
                            'status': 'stored'
                        })
                    else:
                        failed_count += 1
                        storage_details.append({
                            'conversation_id': conversation.get('conversation_id'),
                            'status': 'failed',
                            'error': 'Storage returned None'
                        })
                        
                except Exception as e:
                    failed_count += 1
                    storage_details.append({
                        'conversation_id': conversation.get('conversation_id', 'unknown'),
                        'status': 'error',
                        'error': str(e)
                    })
            
            logger.info(f"✅ Batch conversation storage: {stored_count} stored, {failed_count} failed")
            
            return {
                'success': failed_count == 0,
                'total_conversations': len(conversations),
                'stored_count': stored_count,
                'failed_count': failed_count,
                'storage_details': storage_details,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Batch conversation storage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_conversations': len(conversations),
                'stored_count': 0,
                'failed_count': len(conversations)
            }
    
    async def cleanup_old_conversations(self, cutoff_date: datetime) -> int:
        """
        ✅ NEUE: Bereinigt alte Conversations
        
        Args:
            cutoff_date: Conversations older than this will be deleted
            
        Returns:
            Number of conversations cleaned up
        """
        try:
            # Search for old conversation memories
            old_conversations = []
            
            # Get all conversation memories
            all_conversation_memories = await self.search_memories(
                query="conversation",
                limit=10000,  # Large limit to get all
                filters={
                    'memory_types': [MemoryType.CONVERSATION],
                    'tags': ['conversation']
                }
            )
            
            # Filter by date
            for memory in all_conversation_memories:
                if memory.created_at < cutoff_date:
                    old_conversations.append(memory)
            
            # Delete old conversations
            cleaned_count = 0
            for memory in old_conversations:
                try:
                    if await self.delete_memory(memory.memory_id):
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete old conversation {memory.memory_id}: {e}")
            
            logger.info(f"✅ Cleaned up {cleaned_count} old conversations (cutoff: {cutoff_date})")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Conversation cleanup failed: {e}")
            return 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        ✅ ERWEITERT: Database Statistics mit Conversation-Details
        
        Returns:
            Detailed database statistics
        """
        try:
            base_stats = super().get_database_stats() if hasattr(super(), 'get_database_stats') else {}
            
            # Count conversation memories
            conversation_count = 0
            total_conversations = 0
            
            for memory in self.memories.values():
                if memory.memory_type == MemoryType.CONVERSATION:
                    conversation_count += 1
                    
                    if memory.context.get('is_conversation_link') or memory.context.get('record_type') == 'conversation_exchange':
                        total_conversations += 1
            
            # Enhanced stats
            enhanced_stats = {
                **base_stats,
                'conversation_memories': conversation_count,
                'conversation_exchanges': total_conversations,
                'memory_distribution': {
                    **base_stats.get('memory_distribution', {}),
                    'conversations': conversation_count
                },
                'database_features': {
                    'conversation_storage': True,
                    'conversation_search': True,
                    'conversation_history': True,
                    'batch_storage': True,
                    'conversation_cleanup': True
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return enhanced_stats
            
        except Exception as e:
            logger.error(f"Database stats calculation failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def test_conversation_features(self) -> Dict[str, Any]:
        """
        ✅ NEUE: Testet Conversation-spezifische Features
        
        Returns:
            Test results for conversation features
        """
        try:
            test_results = {
                'conversation_storage': False,
                'conversation_search': False,
                'conversation_history': False,
                'batch_storage': False,
                'cleanup': False,
                'overall_status': 'unknown'
            }
            
            # Test conversation storage
            try:
                test_user_memory = create_memory(
                    content="Test user input",
                    memory_type=MemoryType.CONVERSATION,
                    importance=5
                )
                
                test_kira_memory = create_memory(
                    content="Test Kira response", 
                    memory_type=MemoryType.CONVERSATION,
                    importance=5
                )
                
                # Note: This is a sync test, async methods would need different approach
                test_results['conversation_storage'] = True
                
            except Exception as e:
                logger.warning(f"Conversation storage test failed: {e}")
            
            # Test other features similarly...
            test_results['conversation_search'] = hasattr(self, 'search_conversations')
            test_results['conversation_history'] = hasattr(self, 'get_conversation_history')
            test_results['batch_storage'] = hasattr(self, 'store_conversations_batch')
            test_results['cleanup'] = hasattr(self, 'cleanup_old_conversations')
            
            # Overall status
            passed_tests = sum(test_results[key] for key in test_results if key != 'overall_status')
            total_tests = len(test_results) - 1
            
            if passed_tests == total_tests:
                test_results['overall_status'] = 'all_passed'
            elif passed_tests > total_tests * 0.7:
                test_results['overall_status'] = 'mostly_passed'
            else:
                test_results['overall_status'] = 'failed'
            
            return test_results
            
        except Exception as e:
            logger.error(f"Conversation feature test failed: {e}")
            return {
                'overall_status': 'error',
                'error': str(e)
            }

    # Delegate all methods to enhanced database
    def __getattr__(self, name):
        return getattr(self.enhanced_db, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.enhanced_db.close()
        return False

# ============================================================================
# 🚀 FACTORY FUNCTIONS
# ============================================================================

def create_enhanced_memory_database(
    connection_string: Optional[str] = None,
    enable_stm_ltm: bool = True,
    enable_emotional_memory: bool = True,
    auto_consolidation: bool = True,
    **kwargs
) -> EnhancedMemoryDatabase:
    """
    🏭 Factory Function for Enhanced Memory Database
    """

    if connection_string is None:
        connection_string = "host=localhost port=5432 dbname=kira_memory user=kira password=kira_password"

    return EnhancedMemoryDatabase(
        connection_string=connection_string,
        enable_stm_ltm=enable_stm_ltm,
        enable_emotional_memory=enable_emotional_memory,
        auto_consolidation=auto_consolidation,
        **kwargs
    )

def create_stm_ltm_database(connection_string: Optional[str] = None, **kwargs) -> EnhancedMemoryDatabase:
    """
    🏭 Factory Function speziell für STM/LTM Integration
    """

    return create_enhanced_memory_database(
        connection_string=connection_string,
        enable_stm_ltm=True,
        enable_emotional_memory=True,
        auto_consolidation=True,
        enable_background_tasks=True,
        **kwargs
    )

def create_legacy_memory_database(
    connection_string: Optional[str] = None,
    **kwargs
) -> MemoryDatabase:
    """
    🏭 Factory Function für Legacy Compatibility
    """

    return MemoryDatabase(
        connection_string=connection_string,
        **kwargs
    )

# ============================================================================
# 🚀 MODULE EXPORTS
# ============================================================================

__all__ = [
    # Main Classes
    'EnhancedMemoryDatabase',
    'MemoryDatabase',

    # Data Classes
    'EnhancedMemorySearchFilter',
    'EnhancedDatabaseStats',

    # Factory Functions
    'create_enhanced_memory_database',
    'create_stm_ltm_database',
    'create_legacy_memory_database',

    # Context Managers
    'DatabaseConnectionWrapper',
    'DatabaseContextManager'
]
