"""
PostgreSQL Memory Storage - SAUBERE Neuimplementierung mit Conversation Memory Integration
"""

import json
import psycopg2
import psycopg2.extras
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Import base interface
try:
    from .memory_storage_interface import MemoryStorageInterface, MemoryStorageConfig, MemorySearchFilter, StorageStats
    HAS_INTERFACE = True
except ImportError:
    logger.warning("⚠️ Memory Storage Interface nicht verfügbar - verwende Legacy Mode")
    HAS_INTERFACE = False
    
    class MemoryStorageInterface:
        def initialize(self) -> bool:
            return True
        def close(self):
            pass
        def perform_maintenance(self):
            pass
        def semantic_search(self, query: str, **kwargs):
            return []
        def update_memory_access(self, memory_id: int):
            pass

@dataclass
class MemorySearchFilter:
    """Filter für Memory-Suche"""
    query: Optional[str] = None
    user_id: str = "default"
    memory_type: Optional[str] = None
    session_id: Optional[str] = None
    importance_min: Optional[int] = None
    limit: int = 10
    order_by: str = "created_at"
    order_direction: str = "DESC"

@dataclass
class DatabaseStats:
    """Database-Statistiken"""
    total_memories: int = 0
    unique_users: int = 0
    memory_types: Dict[str, int] = None
    recent_activity: int = 0
    avg_importance: float = 0.0
    database_size_mb: float = 0.0
    enhanced_features: List[str] = None
    
    def __post_init__(self):
        if self.memory_types is None:
            self.memory_types = {}
        if self.enhanced_features is None:
            self.enhanced_features = ["PostgreSQL", "psycopg2", "JSONB", "ConversationMemory"]

class PostgreSQLMemoryStorage(MemoryStorageInterface):
    """
    🚀 SAUBERE PostgreSQL Memory Storage Implementation mit Conversation Memory Integration
    Fokus auf Stabilität und Einfachheit
    """
    
    def __init__(self, connection_string: Optional[str] = None, database_config: Optional[Dict] = None):
        """
        🚀 VEREINFACHTER Constructor mit Conversation Memory Support
        
        Args:
            connection_string: PostgreSQL connection string
            database_config: Optional config dict für Conversation Memory
        """
        
        # Connection String Setup
        if connection_string:
            self.connection_string = connection_string
        elif database_config:
            # Build from config
            host = database_config.get('host', 'localhost')
            port = database_config.get('port', 5432)
            dbname = database_config.get('dbname', 'kira_memory')
            user = database_config.get('user', 'kira')
            password = database_config.get('password', 'kira_password')
            self.connection_string = f"host={host} port={port} dbname={dbname} user={user} password={password}"
        else:
            # Default connection
            self.connection_string = "host=localhost port=5432 dbname=kira_memory user=kira password=kira_password"
        
        # Extract database info for auto-creation
        self.db_config = self._parse_connection_string(self.connection_string)
        
        # Simple state tracking
        self._initialized = False
        self._connection_pool = []
        self._max_pool_size = 3
        
        logger.info(f"🔧 PostgreSQL Storage mit Conversation Memory initialisiert: {self.connection_string[:50]}...")
    
    def initialize(self) -> bool:
        """🚀 VEREINFACHTE Initialisierung mit automatischer Datenbank-Erstellung"""
        try:
            # Schritt 1: Erstelle Datenbank falls nicht vorhanden
            if not self._create_database_if_not_exists():
                logger.error("❌ Automatische Datenbank-Erstellung fehlgeschlagen")
                return False
            
            # Schritt 2: Erstelle Benutzer falls nicht vorhanden
            if not self._create_user_if_not_exists():
                logger.warning("⚠️ Benutzer-Erstellung fehlgeschlagen (nicht kritisch)")
            
            # Schritt 3: Test connection zur eigentlichen Datenbank
            if not self._test_connection():
                logger.error("❌ PostgreSQL Connection Test fehlgeschlagen")
                return False
            
            # Schritt 4: Create schema including conversation tables
            self._create_schema()
            
            self._initialized = True
            logger.info("✅ PostgreSQL Memory Storage mit Conversation Memory erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL Initialization Error: {e}")
            return False
    
    def _test_connection(self) -> bool:
        """Testet PostgreSQL Verbindung mit verbesserter Fehlerbehandlung"""
        try:
            logger.info(f"🔍 Teste PostgreSQL Verbindung zu {self.db_config.get('dbname', 'kira_memory')}...")
            conn = psycopg2.connect(self.connection_string)
            
            # Teste auch eine einfache Query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if result and result[0] == 1:
                logger.info("✅ PostgreSQL Connection Test erfolgreich")
                return True
            else:
                logger.error("❌ PostgreSQL Connection Test: Unerwartetes Ergebnis")
                return False
                
        except psycopg2.OperationalError as e:
            if "database" in str(e) and "does not exist" in str(e):
                logger.error(f"❌ Datenbank existiert nicht: {e}")
            else:
                logger.error(f"❌ PostgreSQL Connection Test fehlgeschlagen: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ PostgreSQL Connection Test fehlgeschlagen: {e}")
            return False
    
    @contextmanager
    def get_connection(self):
        """🚀 KORRIGIERT: Context Manager mit Standard Cursor"""
        conn = None
        try:
            logger.debug(f"🔗 Versuche PostgreSQL Connection: {self.connection_string[:30]}...")
            conn = psycopg2.connect(self.connection_string)
            conn.autocommit = False
            logger.debug("✅ PostgreSQL Connection erfolgreich")
            yield conn
            conn.commit()
            logger.debug("✅ PostgreSQL Transaction committed")
        except psycopg2.Error as pg_e:
            if conn:
                conn.rollback()
                logger.warning("⚠️ PostgreSQL Transaction rolled back")
            logger.error(f"❌ PostgreSQL Error: {pg_e}")
            logger.error(f"❌ PostgreSQL Error Code: {getattr(pg_e, 'pgcode', 'UNKNOWN')}")
            logger.error(f"❌ PostgreSQL Error Details: {getattr(pg_e, 'pgerror', 'NO_DETAILS')}")
            raise
        except Exception as e:
            if conn:
                conn.rollback()
                logger.warning("⚠️ Generic Transaction rolled back")
            logger.error(f"❌ Generic Connection Error: {type(e).__name__}: {e}")
            raise
        finally:
            if conn:
                conn.close()
                logger.debug("🔒 PostgreSQL Connection closed")
    
    def _create_schema(self):
        """🚀 ERWEITERT: Schema Creation mit Conversation Memory Tables"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    # 🔧 HAUPTTABELLE: memory_entries (wie STM/LTM erwartet)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS memory_entries (
                            id SERIAL PRIMARY KEY,
                            session_id TEXT NOT NULL DEFAULT 'main',
                            user_id TEXT NOT NULL DEFAULT 'default',
                            memory_type TEXT NOT NULL DEFAULT 'general',
                            content TEXT NOT NULL,
                            importance INTEGER DEFAULT 5,
                            
                            -- Basis-Metadaten
                            metadata JSONB DEFAULT '{}',
                            tags TEXT DEFAULT '',
                            
                            -- Enhanced Fields für STM/LTM Kompatibilität
                            user_context TEXT DEFAULT '',
                            emotion_type TEXT DEFAULT 'neutral',
                            emotion_intensity REAL DEFAULT 0.0,
                            emotion_valence REAL DEFAULT 0.0,
                            device_context TEXT DEFAULT 'unknown',
                            conversation_context TEXT DEFAULT '',
                            
                            -- STM/LTM Integration Fields
                            stm_activation_level REAL DEFAULT 0.0,
                            ltm_significance_score REAL DEFAULT 0.0,
                            consolidation_score REAL DEFAULT 0.0,
                            memory_strength REAL DEFAULT 1.0,
                            decay_rate REAL DEFAULT 0.1,
                            reinforcement_count INTEGER DEFAULT 0,
                            
                            -- Timestamps
                            created_at TIMESTAMP DEFAULT NOW(),
                            last_accessed TIMESTAMP DEFAULT NOW(),
                            access_count INTEGER DEFAULT 0,
                            expires_at TIMESTAMP,
                            content_hash TEXT,
                            
                            -- Zusätzliche STM/LTM Fields
                            consolidation_timestamp TIMESTAMP,
                            attention_weight REAL DEFAULT 0.5,
                            learning_weight REAL DEFAULT 1.0,
                            memory_category TEXT DEFAULT '',
                            pattern_tags TEXT DEFAULT '',
                            source_memory_id INTEGER,
                            related_memory_ids TEXT DEFAULT '[]'
                        )
                    ''')
                    
                    # ✅ CONVERSATION MEMORY TABELLE - NEU!
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS conversations (
                            id SERIAL PRIMARY KEY,
                            conversation_id VARCHAR(255) UNIQUE NOT NULL,
                            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                            user_input TEXT NOT NULL,
                            kira_response TEXT NOT NULL,
                            importance_score FLOAT DEFAULT 0,
                            emotional_impact FLOAT DEFAULT 0,
                            storage_location VARCHAR(50) DEFAULT 'stm',
                            storage_reason TEXT,
                            topic_category VARCHAR(100),
                            conversation_type VARCHAR(100),
                            key_indicators JSONB DEFAULT '[]',
                            user_memory_id VARCHAR(255),
                            kira_memory_id VARCHAR(255),
                            personal_relevance FLOAT DEFAULT 0,
                            learning_value FLOAT DEFAULT 0,
                            user_name VARCHAR(255) DEFAULT 'User',
                            session_context JSONB DEFAULT '{}',
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        )
                    ''')
                    
                    # ✅ CONVERSATION ANALYTICS TABELLE - NEU!
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS conversation_analytics (
                            id SERIAL PRIMARY KEY,
                            conversation_id VARCHAR(255) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                            user_id VARCHAR(255) DEFAULT 'default',
                            session_id VARCHAR(255) DEFAULT 'main',
                            
                            -- Analytics Data
                            response_time_ms INTEGER DEFAULT 0,
                            user_satisfaction FLOAT DEFAULT 0.0,
                            conversation_quality FLOAT DEFAULT 0.0,
                            topic_coherence FLOAT DEFAULT 0.0,
                            emotional_flow JSONB DEFAULT '{}',
                            learning_indicators JSONB DEFAULT '{}',
                            
                            -- Context Analysis
                            follow_up_questions INTEGER DEFAULT 0,
                            clarification_requests INTEGER DEFAULT 0,
                            user_engagement_score FLOAT DEFAULT 0.0,
                            conversation_depth INTEGER DEFAULT 1,
                            
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        )
                    ''')
                    
                    # ✅ CONVERSATION PATTERNS TABELLE - NEU!
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS conversation_patterns (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(255) NOT NULL,
                            pattern_type VARCHAR(100) NOT NULL,
                            pattern_data JSONB NOT NULL,
                            confidence_score FLOAT DEFAULT 0.0,
                            frequency_count INTEGER DEFAULT 1,
                            last_observed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            
                            -- Pattern Metadata
                            context_tags TEXT[] DEFAULT '{}',
                            emotional_context VARCHAR(50) DEFAULT 'neutral',
                            time_of_day_pattern JSONB DEFAULT '{}',
                            
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        )
                    ''')
                    
                    # 🔧 LEGACY enhanced_memories → memory_entries Migration
                    cursor.execute('''
                        DO $$
                        BEGIN
                            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'enhanced_memories') THEN
                                INSERT INTO memory_entries (
                                    session_id, user_id, memory_type, content, importance,
                                    metadata, user_context, emotion_type, emotion_intensity, emotion_valence,
                                    device_context, conversation_context, stm_activation_level,
                                    ltm_significance_score, consolidation_score, memory_strength,
                                    reinforcement_count, created_at, last_accessed
                                )
                                SELECT 
                                    session_id, user_id, memory_type, content, importance,
                                    personality_data, user_context, emotion_type, emotion_intensity, emotion_valence,
                                    device_context, conversation_context, stm_activation_level,
                                    ltm_significance_score, consolidation_score, memory_strength,
                                    reinforcement_count, created_at, last_accessed
                                FROM enhanced_memories
                                WHERE NOT EXISTS (
                                    SELECT 1 FROM memory_entries 
                                    WHERE memory_entries.content = enhanced_memories.content 
                                    AND memory_entries.user_id = enhanced_memories.user_id
                                );
                                
                                RAISE NOTICE 'Enhanced memories migrated to memory_entries';
                            END IF;
                        END $$;
                    ''')
                    
                    # ✅ PERFORMANCE INDEXES für alle Tabellen
                    indexes = [
                        # Memory Entries Indexes
                        'CREATE INDEX IF NOT EXISTS idx_memory_user_type ON memory_entries(user_id, memory_type)',
                        'CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_entries(session_id)',
                        'CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_entries(created_at)',
                        'CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory_entries(importance)',
                        'CREATE INDEX IF NOT EXISTS idx_memory_emotion ON memory_entries(emotion_type)',
                        'CREATE INDEX IF NOT EXISTS idx_memory_metadata ON memory_entries USING GIN(metadata)',
                        
                        # Conversations Indexes
                        'CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)',
                        'CREATE INDEX IF NOT EXISTS idx_conversations_importance ON conversations(importance_score)',
                        'CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_name)',
                        'CREATE INDEX IF NOT EXISTS idx_conversations_topic ON conversations(topic_category)',
                        'CREATE INDEX IF NOT EXISTS idx_conversations_search ON conversations USING GIN(to_tsvector(\'english\', user_input || \' \' || kira_response))',
                        'CREATE INDEX IF NOT EXISTS idx_conversations_storage ON conversations(storage_location)',
                        'CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_context)',
                        
                        # Analytics Indexes
                        'CREATE INDEX IF NOT EXISTS idx_analytics_conversation ON conversation_analytics(conversation_id)',
                        'CREATE INDEX IF NOT EXISTS idx_analytics_user ON conversation_analytics(user_id)',
                        'CREATE INDEX IF NOT EXISTS idx_analytics_quality ON conversation_analytics(conversation_quality)',
                        
                        # Patterns Indexes
                        'CREATE INDEX IF NOT EXISTS idx_patterns_user ON conversation_patterns(user_id)',
                        'CREATE INDEX IF NOT EXISTS idx_patterns_type ON conversation_patterns(pattern_type)',
                        'CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON conversation_patterns(confidence_score)',
                        'CREATE INDEX IF NOT EXISTS idx_patterns_frequency ON conversation_patterns(frequency_count)'
                    ]
                    
                    for index_sql in indexes:
                        try:
                            cursor.execute(index_sql)
                        except Exception as idx_e:
                            logger.warning(f"⚠️ Index creation warning: {idx_e}")
                    
                    # 🔧 ZUSATZTABELLEN für Enhanced Features
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS emotion_memories (
                            id SERIAL PRIMARY KEY,
                            memory_id INTEGER REFERENCES memory_entries(id) ON DELETE CASCADE,
                            user_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            emotion_type TEXT NOT NULL,
                            emotion_intensity REAL NOT NULL,
                            emotion_valence REAL DEFAULT 0.0,
                            emotional_memory_strength REAL DEFAULT 1.0,
                            created_at TIMESTAMP DEFAULT NOW()
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS personality_patterns (
                            id SERIAL PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            pattern_type TEXT NOT NULL,
                            pattern_name TEXT NOT NULL,
                            confidence_score REAL NOT NULL,
                            reinforcement_count INTEGER DEFAULT 1,
                            created_at TIMESTAMP DEFAULT NOW(),
                            UNIQUE(user_id, pattern_type, pattern_name)
                        )
                    ''')
                    
                finally:
                    cursor.close()
                
                logger.info("✅ PostgreSQL Schema (memory_entries + conversations + analytics) erfolgreich erstellt")
                
        except Exception as e:
            logger.error(f"❌ Schema Creation Error: {e}")
            raise
    
    # ✅ CONVERSATION MEMORY METHODS - NEU!
    
    async def store_conversation(self, conversation_record: Dict[str, Any]) -> bool:
        """✅ Speichert eine vollständige Conversation"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    query = """
                    INSERT INTO conversations (
                        conversation_id, timestamp, user_input, kira_response,
                        importance_score, emotional_impact, storage_location, storage_reason,
                        topic_category, conversation_type, key_indicators,
                        user_memory_id, kira_memory_id, personal_relevance, learning_value,
                        user_name, session_context
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (conversation_id) DO UPDATE SET
                        importance_score = EXCLUDED.importance_score,
                        emotional_impact = EXCLUDED.emotional_impact,
                        storage_location = EXCLUDED.storage_location,
                        updated_at = NOW()
                    """
                    
                    cursor.execute(
                        query,
                        (
                            conversation_record['conversation_id'],
                            conversation_record['timestamp'],
                            conversation_record['user_input'],
                            conversation_record['kira_response'],
                            conversation_record['importance_score'],
                            conversation_record['emotional_impact'],
                            conversation_record['storage_location'],
                            conversation_record['storage_reason'],
                            conversation_record['topic_category'],
                            conversation_record['conversation_type'],
                            json.dumps(conversation_record['key_indicators']),
                            conversation_record['user_memory_id'],
                            conversation_record['kira_memory_id'],
                            conversation_record['personal_relevance'],
                            conversation_record['learning_value'],
                            conversation_record['user_name'],
                            json.dumps(conversation_record['session_context'])
                        )
                    )
                    
                    logger.info(f"✅ Conversation {conversation_record['conversation_id']} gespeichert")
                    return True
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Failed to store conversation: {e}")
            return False

    async def get_conversation_history(self,
                                     user_name: str = None,
                                     limit: int = 50,
                                     importance_min: float = 0.0) -> List[Dict[str, Any]]:
        """✅ Holt Conversation History"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                try:
                    query = """
                    SELECT * FROM conversations 
                    WHERE importance_score >= %s
                    """
                    params = [importance_min]
                    
                    if user_name:
                        query += " AND user_name = %s"
                        params.append(user_name)
                        
                    query += " ORDER BY timestamp DESC LIMIT %s"
                    params.append(limit)
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    result = []
                    for row in rows:
                        row_dict = dict(row)
                        
                        # Parse JSON fields
                        if row_dict.get('key_indicators'):
                            try:
                                row_dict['key_indicators'] = json.loads(row_dict['key_indicators'])
                            except:
                                row_dict['key_indicators'] = []
                                
                        if row_dict.get('session_context'):
                            try:
                                row_dict['session_context'] = json.loads(row_dict['session_context'])
                            except:
                                row_dict['session_context'] = {}
                        
                        # Convert timestamps
                        if row_dict.get('timestamp') and hasattr(row_dict['timestamp'], 'isoformat'):
                            row_dict['timestamp'] = row_dict['timestamp'].isoformat()
                        if row_dict.get('created_at') and hasattr(row_dict['created_at'], 'isoformat'):
                            row_dict['created_at'] = row_dict['created_at'].isoformat()
                            
                        result.append(row_dict)
                    
                    logger.info(f"✅ Retrieved {len(result)} conversations from history")
                    return result
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Failed to get conversation history: {e}")
            return []

    async def search_conversations(self,
                                 query: str,
                                 limit: int = 20,
                                 filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """✅ Durchsucht Conversations"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                try:
                    search_query = """
                    SELECT *, ts_rank(to_tsvector('english', user_input || ' ' || kira_response), 
                                     plainto_tsquery('english', %s)) as relevance_score
                    FROM conversations 
                    WHERE (user_input ILIKE %s OR kira_response ILIKE %s)
                    """
                    params = [query, f"%{query}%", f"%{query}%"]
                    
                    if filters:
                        if 'importance_min' in filters:
                            search_query += " AND importance_score >= %s"
                            params.append(filters['importance_min'])
                            
                        if 'topic_category' in filters:
                            search_query += " AND topic_category = %s"
                            params.append(filters['topic_category'])
                            
                        if 'date_from' in filters:
                            search_query += " AND timestamp >= %s"
                            params.append(filters['date_from'])
                            
                        if 'storage_location' in filters:
                            search_query += " AND storage_location = %s"
                            params.append(filters['storage_location'])
                            
                    search_query += " ORDER BY relevance_score DESC, importance_score DESC, timestamp DESC LIMIT %s"
                    params.append(limit)
                    
                    cursor.execute(search_query, params)
                    rows = cursor.fetchall()
                    
                    result = []
                    for row in rows:
                        row_dict = dict(row)
                        
                        # Parse JSON fields
                        if row_dict.get('key_indicators'):
                            try:
                                row_dict['key_indicators'] = json.loads(row_dict['key_indicators'])
                            except:
                                row_dict['key_indicators'] = []
                                
                        if row_dict.get('session_context'):
                            try:
                                row_dict['session_context'] = json.loads(row_dict['session_context'])
                            except:
                                row_dict['session_context'] = {}
                        
                        # Convert timestamps
                        if row_dict.get('timestamp') and hasattr(row_dict['timestamp'], 'isoformat'):
                            row_dict['timestamp'] = row_dict['timestamp'].isoformat()
                            
                        # Add memory_id and other fields for compatibility
                        row_dict['memory_id'] = row_dict['conversation_id']
                        row_dict['content'] = f"User: {row_dict['user_input']} | Kira: {row_dict['kira_response']}"
                        row_dict['importance'] = row_dict['importance_score']
                        row_dict['emotional_intensity'] = row_dict['emotional_impact']
                        row_dict['memory_type'] = 'conversation'
                        row_dict['context'] = {
                            'conversation_id': row_dict['conversation_id'],
                            'speaker': 'user',
                            'topic_category': row_dict['topic_category'],
                            'storage_decision': row_dict['storage_location']
                        }
                        row_dict['tags'] = ['conversation', row_dict.get('topic_category', 'general')]
                        
                        result.append(row_dict)
                    
                    logger.info(f"✅ Found {len(result)} conversations matching '{query}'")
                    return result
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Failed to search conversations: {e}")
            return []

    async def store_conversations_batch(self, conversation_records: List[Dict[str, Any]]) -> bool:
        """✅ Batch Storage für Conversations"""
        try:
            if not conversation_records:
                return True
                
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    query = """
                    INSERT INTO conversations (
                        conversation_id, timestamp, user_input, kira_response,
                        importance_score, emotional_impact, storage_location
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (conversation_id) DO NOTHING
                    """
                    
                    batch_data = [
                        (
                            record['conversation_id'],
                            record['timestamp'],
                            record['user_input'],
                            record['kira_response'],
                            record['importance_score'],
                            record['emotional_impact'],
                            record['storage_location']
                        )
                        for record in conversation_records
                    ]
                    
                    cursor.executemany(query, batch_data)
                    logger.info(f"✅ Batch stored {len(batch_data)} conversations")
                    return True
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Batch conversation storage failed: {e}")
            return False

    async def cleanup_old_conversations(self, cutoff_date: datetime) -> int:
        """✅ Bereinigt alte Conversations"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    query = "DELETE FROM conversations WHERE timestamp < %s"
                    cursor.execute(query, (cutoff_date,))
                    
                    cleaned_count = cursor.rowcount
                    logger.info(f"✅ Cleaned up {cleaned_count} old conversations")
                    return cleaned_count
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Conversation cleanup failed: {e}")
            return 0

    # ✅ CONVERSATION ANALYTICS METHODS - NEU!
    
    async def store_conversation_analytics(self, analytics_data: Dict[str, Any]) -> bool:
        """✅ Speichert Conversation Analytics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    query = """
                    INSERT INTO conversation_analytics (
                        conversation_id, user_id, session_id,
                        response_time_ms, user_satisfaction, conversation_quality,
                        topic_coherence, emotional_flow, learning_indicators,
                        follow_up_questions, clarification_requests,
                        user_engagement_score, conversation_depth
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    cursor.execute(
                        query,
                        (
                            analytics_data['conversation_id'],
                            analytics_data.get('user_id', 'default'),
                            analytics_data.get('session_id', 'main'),
                            analytics_data.get('response_time_ms', 0),
                            analytics_data.get('user_satisfaction', 0.0),
                            analytics_data.get('conversation_quality', 0.0),
                            analytics_data.get('topic_coherence', 0.0),
                            json.dumps(analytics_data.get('emotional_flow', {})),
                            json.dumps(analytics_data.get('learning_indicators', {})),
                            analytics_data.get('follow_up_questions', 0),
                            analytics_data.get('clarification_requests', 0),
                            analytics_data.get('user_engagement_score', 0.0),
                            analytics_data.get('conversation_depth', 1)
                        )
                    )
                    
                    return True
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Failed to store conversation analytics: {e}")
            return False

    async def get_conversation_analytics(self, 
                                       conversation_id: str = None,
                                       user_id: str = None,
                                       limit: int = 100) -> List[Dict[str, Any]]:
        """✅ Holt Conversation Analytics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                try:
                    query = "SELECT * FROM conversation_analytics WHERE 1=1"
                    params = []
                    
                    if conversation_id:
                        query += " AND conversation_id = %s"
                        params.append(conversation_id)
                        
                    if user_id:
                        query += " AND user_id = %s"
                        params.append(user_id)
                    
                    query += " ORDER BY created_at DESC LIMIT %s"
                    params.append(limit)
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    result = []
                    for row in rows:
                        row_dict = dict(row)
                        
                        # Parse JSON fields
                        if row_dict.get('emotional_flow'):
                            try:
                                row_dict['emotional_flow'] = json.loads(row_dict['emotional_flow'])
                            except:
                                row_dict['emotional_flow'] = {}
                                
                        if row_dict.get('learning_indicators'):
                            try:
                                row_dict['learning_indicators'] = json.loads(row_dict['learning_indicators'])
                            except:
                                row_dict['learning_indicators'] = {}
                        
                        result.append(row_dict)
                    
                    return result
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Failed to get conversation analytics: {e}")
            return []

    # 🚀 HAUPTMETHODEN - Erweitert für Conversation Memory Support
    
    def store_enhanced_memory(self, **kwargs) -> Optional[int]:
        """🚀 KORRIGIERT: Store mit korrekter psycopg2 Cursor Usage"""
        try:
            # Extract parameters
            session_id = kwargs.get('session_id', 'main')
            user_id = kwargs.get('user_id', 'default')
            memory_type = kwargs.get('memory_type', 'general')
            content = kwargs.get('content', '')
            importance = kwargs.get('importance', 5)
            
            logger.debug(f"📝 Storing memory: user={user_id}, type={memory_type}, content_len={len(content)}")
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    # INSERT in memory_entries Tabelle
                    insert_sql = '''
                        INSERT INTO memory_entries (
                            session_id, user_id, memory_type, content, importance,
                            user_context, emotion_type, emotion_intensity, emotion_valence,
                            device_context, conversation_context, metadata,
                            stm_activation_level, ltm_significance_score, consolidation_score,
                            memory_strength, reinforcement_count, created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        ) RETURNING id
                    '''
                    
                    insert_params = (
                        session_id,
                        user_id,
                        memory_type,
                        content,
                        importance,
                        kwargs.get('user_context', ''),
                        kwargs.get('emotion_type', 'neutral'),
                        kwargs.get('emotion_intensity', 0.0),
                        kwargs.get('emotion_valence', 0.0),
                        kwargs.get('device_context', 'unknown'),
                        kwargs.get('conversation_context', ''),
                        json.dumps(kwargs.get('metadata', {})),
                        kwargs.get('stm_activation_level', 0.0),
                        kwargs.get('ltm_significance_score', 0.0),
                        kwargs.get('consolidation_score', 0.0),
                        kwargs.get('memory_strength', 1.0),
                        kwargs.get('reinforcement_count', 0),
                        datetime.now()
                    )
                    
                    logger.debug(f"🚀 Executing INSERT with {len(insert_params)} parameters")
                    cursor.execute(insert_sql, insert_params)
                    
                    result = cursor.fetchone()
                    memory_id = result[0] if result else None
                    
                    if memory_id:
                        logger.info(f"✅ Enhanced Memory {memory_id} erfolgreich gespeichert")
                        return memory_id
                    else:
                        logger.warning("⚠️ INSERT erfolgreich aber kein memory_id zurückgegeben")
                        return None
                        
                finally:
                    cursor.close()
                    
        except psycopg2.Error as pg_e:
            logger.error(f"❌ PostgreSQL Store Error: {pg_e}")
            logger.error(f"❌ PostgreSQL Error Code: {getattr(pg_e, 'pgcode', 'UNKNOWN')}")
            logger.error(f"❌ PostgreSQL Error Details: {getattr(pg_e, 'pgerror', 'NO_DETAILS')}")
            return None
        except Exception as e:
            logger.error(f"❌ Generic Store Enhanced Memory Error: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"❌ Store Error Traceback: {traceback.format_exc()}")
            return None
    
    def search_memories(self, 
                       search_filter: Union[MemorySearchFilter, Dict, str],
                       **kwargs) -> List[Dict[str, Any]]:
        """🚀 KORRIGIERT: Search in memory_entries UND conversations"""
        try:
            if isinstance(search_filter, MemorySearchFilter):
                filter_obj = search_filter
            elif isinstance(search_filter, dict):
                filter_obj = MemorySearchFilter(**search_filter)
            else:
                # String query
                filter_obj = MemorySearchFilter(
                    query=search_filter,
                    user_id=kwargs.get('user_id', 'default'),
                    limit=kwargs.get('limit', 10),
                    memory_type=kwargs.get('memory_type'),
                    session_id=kwargs.get('session_id')
                )
            
            logger.debug(f"🔍 Searching memory_entries + conversations: user={filter_obj.user_id}")
            
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                try:
                    results = []
                    
                    # 1. Search in memory_entries
                    query_sql = "SELECT *, 'memory_entry' as source_table FROM memory_entries WHERE 1=1"
                    params = []
                    
                    if search_filter.user_id:
                        query_sql += " AND user_id = %s"
                        params.append(search_filter.user_id)
                    
                    if search_filter.session_id:
                        query_sql += " AND session_id = %s"
                        params.append(search_filter.session_id)
                    
                    if search_filter.memory_type and search_filter.memory_type != 'conversation':
                        query_sql += " AND memory_type = %s"
                        params.append(search_filter.memory_type)
                    
                    if search_filter.query:
                        query_sql += " AND content ILIKE %s"
                        params.append(f"%{search_filter.query}%")
                    
                    if search_filter.importance_min:
                        query_sql += " AND importance >= %s"
                        params.append(search_filter.importance_min)
                    
                    query_sql += f" ORDER BY {search_filter.order_by} {search_filter.order_direction}"
                    query_sql += " LIMIT %s"
                    params.append(search_filter.limit // 2)  # Half for memory_entries
                    
                    cursor.execute(query_sql, params)
                    memory_rows = cursor.fetchall()
                    
                    # 2. Search in conversations (if not filtering for specific non-conversation type)
                    if not search_filter.memory_type or search_filter.memory_type == 'conversation':
                        conv_query = """
                        SELECT 
                            conversation_id as id,
                            user_name as user_id,
                            'main' as session_id,
                            'conversation' as memory_type,
                            (user_input || ' | ' || kira_response) as content,
                            CAST(importance_score as INTEGER) as importance,
                            '{}' as metadata,
                            '' as tags,
                            user_input as user_context,
                            'neutral' as emotion_type,
                            emotional_impact as emotion_intensity,
                            0.0 as emotion_valence,
                            'conversation' as device_context,
                            topic_category as conversation_context,
                            0.0 as stm_activation_level,
                            importance_score as ltm_significance_score,
                            0.0 as consolidation_score,
                            1.0 as memory_strength,
                            0.1 as decay_rate,
                            0 as reinforcement_count,
                            timestamp as created_at,
                            timestamp as last_accessed,
                            0 as access_count,
                            NULL as expires_at,
                            NULL as content_hash,
                            NULL as consolidation_timestamp,
                            0.5 as attention_weight,
                            1.0 as learning_weight,
                            topic_category as memory_category,
                            '' as pattern_tags,
                            NULL as source_memory_id,
                            '[]' as related_memory_ids,
                            'conversation' as source_table
                        FROM conversations 
                        WHERE 1=1
                        """
                        conv_params = []
                        
                        if search_filter.user_id and search_filter.user_id != "default":
                            conv_query += " AND user_name = %s"
                            conv_params.append(search_filter.user_id)
                        
                        if search_filter.query:
                            conv_query += " AND (user_input ILIKE %s OR kira_response ILIKE %s)"
                            conv_params.append(f"%{search_filter.query}%")
                            conv_params.append(f"%{search_filter.query}%")
                        
                        if search_filter.importance_min:
                            conv_query += " AND importance_score >= %s"
                            conv_params.append(search_filter.importance_min)
                        
                        conv_query += " ORDER BY timestamp DESC LIMIT %s"
                        conv_params.append(search_filter.limit // 2)  # Other half for conversations
                        
                        cursor.execute(conv_query, conv_params)
                        conv_rows = cursor.fetchall()
                        
                        # Combine results
                        all_rows = list(memory_rows) + list(conv_rows)
                    else:
                        all_rows = memory_rows
                    
                    # Process results
                    for row in all_rows:
                        try:
                            result_dict = dict(row)
                            
                            # Parse JSON metadata
                            if result_dict.get('metadata'):
                                try:
                                    if isinstance(result_dict['metadata'], str):
                                        result_dict['metadata'] = json.loads(result_dict['metadata'])
                                except:
                                    result_dict['metadata'] = {}
                            
                            # Convert timestamps
                            for ts_field in ['created_at', 'last_accessed', 'expires_at', 'consolidation_timestamp']:
                                if result_dict.get(ts_field) and hasattr(result_dict[ts_field], 'isoformat'):
                                    result_dict[ts_field] = result_dict[ts_field].isoformat()
                            
                            results.append(result_dict)
                            
                        except Exception as row_e:
                            logger.warning(f"⚠️ Row processing error: {row_e}")
                            continue
                    
                    # Sort combined results by creation date
                    results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                    results = results[:search_filter.limit]
                    
                    logger.info(f"✅ Found {len(results)} memories (memory_entries + conversations)")
                    return results
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Search Memories Error: {e}")
            return []
    
    def get_memories(self, user_id: str = "default", session_id: Optional[str] = None,
                    memory_type: Optional[str] = None, limit: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """🚀 KORRIGIERT: Get Memories aus memory_entries + conversations"""
        search_filter = MemorySearchFilter(
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            limit=limit
        )
        return self.search_memories(search_filter)
    
    def get_enhanced_stats(self, user_id: Optional[str] = None) -> StorageStats:
        """🚀 ERWEITERT: Stats mit Conversation Memory Support"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                try:
                    # Base query parts
                    user_filter = ""
                    params = []
                    if user_id:
                        user_filter = " WHERE user_id = %s"
                        params.append(user_id)
                    
                    # Memory entries stats
                    cursor.execute(f'''
                        SELECT 
                            COUNT(*) as total_memories,
                            COUNT(DISTINCT user_id) as unique_users,
                            COUNT(DISTINCT session_id) as unique_sessions,
                            AVG(importance) as avg_importance,
                            COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 day' THEN 1 END) as recent_24h,
                            COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END) as recent_7d,
                            COUNT(CASE WHEN created_at > NOW() - INTERVAL '30 days' THEN 1 END) as recent_30d
                        FROM memory_entries{user_filter}
                    ''', params)
                    
                    memory_stats = cursor.fetchone()
                    
                    # Memory types distribution
                    cursor.execute(f'''
                        SELECT memory_type, COUNT(*) as count 
                        FROM memory_entries{user_filter}
                        GROUP BY memory_type
                    ''', params)
                    
                    memory_types = {row['memory_type']: row['count'] for row in cursor.fetchall()}
                    
                    # Calculate total for percentages
                    total_memories = memory_stats['total_memories'] or 0
                    memory_type_percentages = {}
                    if total_memories > 0:
                        memory_type_percentages = {
                            mem_type: (count / total_memories) * 100 
                            for mem_type, count in memory_types.items()
                        }
                    
                    # Most active users (if no specific user_id)
                    most_active_users = []
                    if not user_id:
                        cursor.execute('''
                            SELECT user_id, COUNT(*) as memory_count
                            FROM memory_entries
                            GROUP BY user_id
                            ORDER BY memory_count DESC
                            LIMIT 10
                        ''')
                        most_active_users = [
                            {"user_id": row['user_id'], "memory_count": row['memory_count']}
                            for row in cursor.fetchall()
                        ]
                    
                    # Create and return StorageStats
                    stats = StorageStats()
                    stats.total_memories = total_memories
                    stats.total_users = memory_stats['unique_users'] or 0
                    stats.total_sessions = memory_stats['unique_sessions'] or 0
                    stats.avg_memories_per_user = total_memories / max(stats.total_users, 1)
                    stats.avg_importance = float(memory_stats['avg_importance'] or 0)
                    stats.memory_types = memory_types
                    stats.memory_type_percentages = memory_type_percentages
                    stats.recent_memories_24h = memory_stats['recent_24h'] or 0
                    stats.recent_memories_7d = memory_stats['recent_7d'] or 0
                    stats.recent_memories_30d = memory_stats['recent_30d'] or 0
                    stats.most_active_users = most_active_users
                    
                    return stats
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Enhanced Stats Error: {e}")
            return StorageStats()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """🚀 ERWEITERT: Stats mit Conversation Memory Integration"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                try:
                    # Memory entries stats
                    cursor.execute('''
                        SELECT 
                            COUNT(*) as total_memories,
                            COUNT(DISTINCT user_id) as unique_users,
                            AVG(importance) as avg_importance,
                            COUNT(CASE WHEN created_at > NOW() - INTERVAL '7 days' THEN 1 END) as recent_activity
                        FROM memory_entries
                    ''')
                    
                    memory_stats = cursor.fetchone()
                    
                    # Conversation stats
                    cursor.execute('''
                        SELECT 
                            COUNT(*) as total_conversations,
                            COUNT(DISTINCT user_name) as unique_conversation_users,
                            AVG(importance_score) as avg_conversation_importance,
                            COUNT(CASE WHEN timestamp > NOW() - INTERVAL '7 days' THEN 1 END) as recent_conversations,
                            COUNT(CASE WHEN storage_location = 'ltm' THEN 1 END) as ltm_conversations,
                            COUNT(CASE WHEN storage_location = 'stm' THEN 1 END) as stm_conversations
                        FROM conversations
                    ''')
                    
                    conv_stats = cursor.fetchone()
                    
                    # Memory types
                    cursor.execute('''
                        SELECT memory_type, COUNT(*) as count 
                        FROM memory_entries 
                        GROUP BY memory_type
                    ''')
                    
                    memory_types = {row['memory_type']: row['count'] for row in cursor.fetchall()}
                    memory_types['conversation'] = conv_stats['total_conversations'] or 0
                    
                    # Conversation categories
                    cursor.execute('''
                        SELECT topic_category, COUNT(*) as count 
                        FROM conversations 
                        WHERE topic_category IS NOT NULL
                        GROUP BY topic_category
                    ''')
                    
                    conversation_topics = {row['topic_category']: row['count'] for row in cursor.fetchall()}
                    
                    return {
                        'total_memories': (memory_stats['total_memories'] or 0) + (conv_stats['total_conversations'] or 0),
                        'unique_users': max(memory_stats['unique_users'] or 0, conv_stats['unique_conversation_users'] or 0),
                        'memory_types': memory_types,
                        'recent_activity': (memory_stats['recent_activity'] or 0) + (conv_stats['recent_conversations'] or 0),
                        'avg_importance': round(float((memory_stats['avg_importance'] or 0) + (conv_stats['avg_conversation_importance'] or 0)) / 2, 2),
                        
                        # ✅ CONVERSATION MEMORY SPECIFIC STATS
                        'conversation_stats': {
                            'total_conversations': conv_stats['total_conversations'] or 0,
                            'unique_conversation_users': conv_stats['unique_conversation_users'] or 0,
                            'avg_conversation_importance': round(float(conv_stats['avg_conversation_importance'] or 0), 2),
                            'recent_conversations': conv_stats['recent_conversations'] or 0,
                            'ltm_conversations': conv_stats['ltm_conversations'] or 0,
                            'stm_conversations': conv_stats['stm_conversations'] or 0,
                            'conversation_topics': conversation_topics
                        },
                        
                        'table_name': 'memory_entries + conversations',
                        'enhanced_features': ["PostgreSQL", "memory_entries", "conversations", "analytics", "patterns", "JSONB", "STM_LTM_Compatible", "ConversationMemory"]
                    }
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"❌ Database Stats Error: {e}")
            return {
                'total_memories': 0,
                'unique_users': 0,
                'memory_types': {},
                'conversation_stats': {},
                'error': str(e)
            }
    
    # 🚀 COMPATIBILITY METHODS - Erweitert
    
    def store_memory(self, session_id: str, user_id: str, memory_type: str, 
                    content: str, metadata: Optional[Dict] = None, 
                    importance: int = 5, tags: Optional[List[str]] = None,
                    expires_at: Optional[str] = None, **kwargs) -> Optional[int]:
        """🚀 KORRIGIERT: Store Memory in memory_entries Tabelle"""
        try:
            logger.debug(f"📝 Storing memory: user={user_id}, type={memory_type}")
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Content Hash für Deduplizierung
                    import hashlib
                    content_hash = hashlib.md5(f"{content}{user_id}{memory_type}".encode()).hexdigest()
                    
                    # INSERT in memory_entries
                    insert_sql = '''
                        INSERT INTO memory_entries (
                            session_id, user_id, memory_type, content, importance,
                            metadata, tags, content_hash, expires_at,
                            user_context, emotion_type, emotion_intensity, emotion_valence,
                            device_context, conversation_context,
                            stm_activation_level, ltm_significance_score, consolidation_score,
                            memory_strength, decay_rate, reinforcement_count,
                            attention_weight, learning_weight, memory_category,
                            created_at, last_accessed
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        ) RETURNING id
                    '''
                    
                    tags_str = ','.join(tags) if tags else ''
                    
                    insert_params = (
                        session_id,
                        user_id,
                        memory_type,
                        content,
                        importance,
                        json.dumps(metadata) if metadata else '{}',
                        tags_str,
                        content_hash,
                        expires_at,
                        kwargs.get('user_context', ''),
                        kwargs.get('emotion_type', 'neutral'),
                        kwargs.get('emotion_intensity', 0.0),
                        kwargs.get('emotion_valence', 0.0),
                        kwargs.get('device_context', 'unknown'),
                        kwargs.get('conversation_context', ''),
                        kwargs.get('stm_activation_level', 0.0),
                        kwargs.get('ltm_significance_score', 0.0),
                        kwargs.get('consolidation_score', 0.0),
                        kwargs.get('memory_strength', 1.0),
                        kwargs.get('decay_rate', 0.1),
                        kwargs.get('reinforcement_count', 0),
                        kwargs.get('attention_weight', 0.5),
                        kwargs.get('learning_weight', 1.0),
                        kwargs.get('memory_category', ''),
                        datetime.now(),
                        datetime.now()
                    )
                    
                    cursor.execute(insert_sql, insert_params)
                    result = cursor.fetchone()
                    memory_id = result[0] if result else None
                    
                    if memory_id:
                        logger.info(f"✅ Memory {memory_id} erfolgreich in memory_entries gespeichert")
                        return memory_id
                    else:
                        logger.warning("⚠️ INSERT erfolgreich aber kein memory_id zurückgegeben")
                        return None
                        
                finally:
                    cursor.close()
                    
        except psycopg2.Error as pg_e:
            logger.error(f"❌ PostgreSQL Store Error: {pg_e}")
            logger.error(f"❌ Error Code: {getattr(pg_e, 'pgcode', 'UNKNOWN')}")
            return None
        except Exception as e:
            logger.error(f"❌ Store Memory Error: {e}")
            return None
    
    # 🚀 INTERFACE METHODS
    
    def close(self):
        """Cleanup"""
        self._initialized = False
        logger.info("✅ PostgreSQL Storage mit Conversation Memory geschlossen")
    
    def perform_maintenance(self) -> Dict[str, Any]:
        """Basic maintenance auf memory_entries + conversations"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("ANALYZE memory_entries")
                    cursor.execute("ANALYZE conversations")
                    cursor.execute("ANALYZE conversation_analytics")
                    return {
                        'status': 'success', 
                        'action': 'analyze_completed',
                        'tables': ['memory_entries', 'conversations', 'conversation_analytics']
                    }
                finally:
                    cursor.close()
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def semantic_search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Semantic search fallback - includes conversations"""
        search_filter = MemorySearchFilter(
            query=query,
            user_id=kwargs.get('user_id', 'default'),
            limit=kwargs.get('limit', 10)
        )
        return self.search_memories(search_filter)
    
    def update_memory_access(self, memory_id: int, user_id: str = "default") -> bool:
        """🚀 KORRIGIERT: Update Access in memory_entries"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute('''
                        UPDATE memory_entries 
                        SET last_accessed = NOW(), access_count = access_count + 1
                        WHERE id = %s
                    ''', (memory_id,))
                    
                    if cursor.rowcount > 0:
                        logger.debug(f"📊 Memory {memory_id} access updated")
                        return True
                    else:
                        logger.warning(f"⚠️ Memory {memory_id} nicht gefunden für Access Update")
                        return False
                        
                finally:
                    cursor.close()
        except Exception as e:
            logger.warning(f"⚠️ Update Access Error: {e}")
            return False
        
    def _parse_connection_string(self, conn_str: str) -> Dict[str, str]:
        """Parst die Connection String in ihre Komponenten"""
        config = {}
        parts = conn_str.split()
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                config[key] = value
        
        return config
    
    def _create_database_if_not_exists(self) -> bool:
        """Erstellt die Datenbank automatisch, falls sie nicht existiert"""
        try:
            # Verbindung ohne Datenbank-Name für Database-Creation
            temp_config = self.db_config.copy()
            original_dbname = temp_config.get('dbname', 'kira_memory')
            temp_config['dbname'] = 'postgres'  # Standard-DB für Admin-Operationen
            
            # Baue temporäre Connection String
            temp_conn_str = ' '.join([f"{k}={v}" for k, v in temp_config.items()])
            
            logger.info(f"🔍 Prüfe ob Datenbank '{original_dbname}' existiert...")
            
            # Verbinde zur postgres DB
            with psycopg2.connect(temp_conn_str) as conn:
                conn.autocommit = True
                cursor = conn.cursor()
                
                # Prüfe ob Datenbank existiert
                cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (original_dbname,)
                )
                
                if cursor.fetchone():
                    logger.info(f"✅ Datenbank '{original_dbname}' existiert bereits")
                    return True
                
                # Erstelle Datenbank
                logger.info(f"🏗️ Erstelle Datenbank '{original_dbname}'...")
                cursor.execute(f'CREATE DATABASE "{original_dbname}"')
                
                logger.info(f"✅ Datenbank '{original_dbname}' erfolgreich erstellt")
                return True
                
        except psycopg2.Error as e:
            logger.error(f"❌ Fehler beim Erstellen der Datenbank: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unerwarteter Fehler bei Datenbank-Erstellung: {e}")
            return False
    
    def _create_user_if_not_exists(self) -> bool:
        """Erstellt den Benutzer, falls er nicht existiert"""
        try:
            # Verbindung ohne Datenbank-Name für User-Creation
            temp_config = self.db_config.copy()
            temp_config['dbname'] = 'postgres'
            temp_config['user'] = 'postgres'  # Admin user
            
            # Baue temporäre Connection String
            temp_conn_str = ' '.join([f"{k}={v}" for k, v in temp_config.items()])
            
            original_user = self.db_config.get('user', 'kira')
            original_password = self.db_config.get('password', 'kira_password')
            
            logger.info(f"🔍 Prüfe ob Benutzer '{original_user}' existiert...")
            
            try:
                with psycopg2.connect(temp_conn_str) as conn:
                    conn.autocommit = True
                    cursor = conn.cursor()
                    
                    # Prüfe ob Benutzer existiert
                    cursor.execute(
                        "SELECT 1 FROM pg_user WHERE usename = %s",
                        (original_user,)
                    )
                    
                    if cursor.fetchone():
                        logger.info(f"✅ Benutzer '{original_user}' existiert bereits")
                        return True
                    
                    # Erstelle Benutzer
                    logger.info(f"👤 Erstelle Benutzer '{original_user}'...")
                    cursor.execute(
                        f"CREATE USER {original_user} WITH PASSWORD %s",
                        (original_password,)
                    )
                    
                    # Gebe Berechtigungen
                    cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {self.db_config.get('dbname', 'kira_memory')} TO {original_user}")
                    
                    logger.info(f"✅ Benutzer '{original_user}' erfolgreich erstellt")
                    return True
                    
            except psycopg2.Error as e:
                if "already exists" in str(e):
                    logger.info(f"✅ Benutzer '{original_user}' existiert bereits")
                    return True
                logger.warning(f"⚠️ Fehler beim Erstellen des Benutzers (möglicherweise bereits vorhanden): {e}")
                return True  # Continue anyway
                
        except Exception as e:
            logger.warning(f"⚠️ Benutzer-Erstellung fehlgeschlagen (nicht kritisch): {e}")
            return True  # Continue anyway
        

# 🚀 FACTORY REGISTRATION
try:
    from .memory_storage_interface import MemoryStorageFactory
    MemoryStorageFactory.register_storage('postgresql', PostgreSQLMemoryStorage)
    logger.info("✅ PostgreSQL Memory Storage mit Conversation Memory registered")
except:
    pass

# Aliases
EnhancedMemoryDatabase = PostgreSQLMemoryStorage