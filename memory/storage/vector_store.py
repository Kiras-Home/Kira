"""
Vector Store für Kira's Memory System
Nutzt pgvector für semantische Memory-Suche
"""

import logging
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
from sentence_transformers import SentenceTransformer
from config.system_config import KiraSystemConfig

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector Storage System für semantische Memory-Suche
    """
    
    def __init__(self, config: KiraSystemConfig = None):
        self.config = config or KiraSystemConfig()
        self.connection = None
        self.embedding_model = None
        self.embedding_dimension = 384  # Standard für all-MiniLM-L6-v2
        self.is_initialized = False
        
    def initialize(self) -> Dict[str, Any]:
        """
        Initialisiert Vector Store mit pgvector
        """
        try:
            # 1. Database Connection
            self._connect_database()
            
            # 2. Embedding Model laden
            self._load_embedding_model()
            
            # 3. Tables erstellen
            self._create_vector_tables()
            
            # 4. Indizes erstellen
            self._create_vector_indexes()
            
            self.is_initialized = True
            logger.info("✅ VectorStore initialized successfully")
            
            return {
                'success': True,
                'message': 'VectorStore initialized',
                'embedding_dimension': self.embedding_dimension,
                'model': str(self.embedding_model)
            }
            
        except Exception as e:
            logger.error(f"❌ VectorStore initialization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _connect_database(self):
        """Verbindung zur PostgreSQL Datenbank"""
        try:
            # Fix: Prüfe DatabaseConfig Struktur
            db_config = self.config.database
            
            # Debug: Schaue was DatabaseConfig wirklich hat
            logger.debug(f"DatabaseConfig attributes: {dir(db_config)}")
            logger.debug(f"DatabaseConfig type: {type(db_config)}")
            
            # Flexible Verbindung basierend auf verfügbaren Attributen
            if hasattr(db_config, 'connection_string') and db_config.connection_string:
                # Option 1: Connection String verwenden
                self.connection = psycopg2.connect(db_config.connection_string)
                logger.info("✅ Database connected via connection string")
                
            elif hasattr(db_config, 'host'):
                # Option 2: Einzelne Parameter
                self.connection = psycopg2.connect(
                    host=db_config.host,
                    port=getattr(db_config, 'port', 5432),
                    database=getattr(db_config, 'database', 'kira_memory'),
                    user=getattr(db_config, 'user', 'postgres'),
                    password=getattr(db_config, 'password', '')
                )
                logger.info("✅ Database connected via individual parameters")
                
            else:
                # Option 3: Fallback zu Standard-Werten
                logger.warning("⚠️ Using fallback database configuration")
                self.connection = psycopg2.connect(
                    host=getattr(db_config, 'host', 'localhost'),
                    port=getattr(db_config, 'port', 5432),
                    database=getattr(db_config, 'name', 'kira_memory'),  # Manchmal 'name' statt 'database'
                    user=getattr(db_config, 'username', getattr(db_config, 'user', 'postgres')),
                    password=getattr(db_config, 'password', '')
                )
                logger.info("✅ Database connected with fallback config")
            
            self.connection.autocommit = True
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            logger.error(f"   Config object: {self.config.database}")
            logger.error(f"   Available attributes: {[attr for attr in dir(self.config.database) if not attr.startswith('_')]}")
            raise
    
    def _load_embedding_model(self):
        """Lädt das Embedding Model"""
        try:
            # Nutze deutsches Sentence-Transformer Model
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            logger.info(f"✅ Embedding model loaded: {model_name}")
            logger.info(f"   Dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    def _create_vector_tables(self):
        """Erstellt pgvector Tables"""
        try:
            cursor = self.connection.cursor()
            
            # pgvector Extension aktivieren
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Memory Vectors Table
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS memory_vectors (
                id SERIAL PRIMARY KEY,
                memory_id VARCHAR(255) UNIQUE NOT NULL,
                content TEXT NOT NULL,
                embedding vector({self.embedding_dimension}),
                memory_type VARCHAR(100),
                user_id VARCHAR(255),
                conversation_id VARCHAR(255),
                emotional_context JSONB,
                strength FLOAT DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                metadata JSONB DEFAULT '{{}}'::jsonb
            );
            """
            cursor.execute(create_table_sql)
            
            # Memory Connections Table (für Neural Networks)
            create_connections_sql = """
            CREATE TABLE IF NOT EXISTS memory_connections (
                id SERIAL PRIMARY KEY,
                source_memory_id VARCHAR(255) NOT NULL,
                target_memory_id VARCHAR(255) NOT NULL,
                connection_strength FLOAT DEFAULT 0.5,
                connection_type VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_memory_id, target_memory_id)
            );
            """
            cursor.execute(create_connections_sql)
            
            cursor.close()
            logger.info("✅ Vector tables created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create vector tables: {e}")
            raise
    
    def _create_vector_indexes(self):
        """Erstellt Indizes für performante Suche"""
        try:
            cursor = self.connection.cursor()
            
            # Vector Index für semantische Suche
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS memory_vectors_embedding_idx 
                ON memory_vectors USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """)
            
            # Standard Indizes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_vectors_user_id ON memory_vectors(user_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_vectors_type ON memory_vectors(memory_type);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_vectors_created ON memory_vectors(created_at);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_connections_source ON memory_connections(source_memory_id);")
            
            cursor.close()
            logger.info("✅ Vector indexes created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {e}")
            raise
    
    def store_memory(self, 
                    content: str,
                    memory_id: str,
                    memory_type: str = "conversation",
                    user_id: str = None,
                    conversation_id: str = None,
                    emotional_context: Dict = None,
                    strength: float = 1.0,
                    metadata: Dict = None) -> Dict[str, Any]:
        """
        Speichert Memory mit Embedding
        """
        try:
            if not self.is_initialized:
                raise Exception("VectorStore not initialized")
            
            # 1. Embedding generieren
            embedding = self._generate_embedding(content)
            
            # 2. In Database speichern
            cursor = self.connection.cursor()
            
            insert_sql = """
                INSERT INTO memory_vectors 
                (memory_id, content, embedding, memory_type, user_id, conversation_id, 
                 emotional_context, strength, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (memory_id) 
                DO UPDATE SET 
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    accessed_at = CURRENT_TIMESTAMP,
                    access_count = memory_vectors.access_count + 1
                RETURNING id;
            """
            
            cursor.execute(insert_sql, (
                memory_id,
                content,
                embedding.tolist(),  # Convert numpy array to list
                memory_type,
                user_id,
                conversation_id,
                json.dumps(emotional_context or {}),
                strength,
                json.dumps(metadata or {})
            ))
            
            result = cursor.fetchone()
            cursor.close()
            
            logger.info(f"✅ Memory stored: {memory_id}")
            
            return {
                'success': True,
                'memory_id': memory_id,
                'database_id': result[0] if result else None
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to store memory: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_similar_memories(self, 
                              query: str = None,
                              embedding: np.ndarray = None,
                              limit: int = 10,
                              similarity_threshold: float = 0.7,
                              user_id: str = None,
                              memory_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Sucht ähnliche Memories basierend auf semantischer Ähnlichkeit
        """
        try:
            if not self.is_initialized:
                raise Exception("VectorStore not initialized")
            
            # Embedding generieren wenn Query gegeben
            if query and embedding is None:
                embedding = self._generate_embedding(query)
            elif embedding is None:
                raise ValueError("Either query or embedding must be provided")
            
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # SQL Query für semantische Suche
            where_conditions = []
            params = [embedding.tolist(), limit]
            
            if user_id:
                where_conditions.append("user_id = %s")
                params.append(user_id)
            
            if memory_types:
                where_conditions.append("memory_type = ANY(%s)")
                params.append(memory_types)
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            search_sql = f"""
                SELECT 
                    memory_id,
                    content,
                    memory_type,
                    user_id,
                    conversation_id,
                    emotional_context,
                    strength,
                    created_at,
                    accessed_at,
                    access_count,
                    metadata,
                    (1 - (embedding <=> %s)) as similarity
                FROM memory_vectors
                {where_clause}
                ORDER BY embedding <=> %s
                LIMIT %s;
            """
            
            # Parameter anpassen für doppelte embedding usage
            search_params = [embedding.tolist()] + params + [embedding.tolist()]
            
            cursor.execute(search_sql, search_params)
            results = cursor.fetchall()
            cursor.close()
            
            # Filter by similarity threshold
            filtered_results = [
                dict(row) for row in results 
                if row['similarity'] >= similarity_threshold
            ]
            
            # Update access count für gefundene Memories
            self._update_access_stats([r['memory_id'] for r in filtered_results])
            
            logger.info(f"✅ Found {len(filtered_results)} similar memories")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Memory search failed: {e}")
            return []
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generiert Embedding für Text"""
        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise
    
    def _update_access_stats(self, memory_ids: List[str]):
        """Aktualisiert Zugriffs-Statistiken"""
        try:
            if not memory_ids:
                return
                
            cursor = self.connection.cursor()
            
            update_sql = """
                UPDATE memory_vectors 
                SET accessed_at = CURRENT_TIMESTAMP,
                    access_count = access_count + 1
                WHERE memory_id = ANY(%s);
            """
            
            cursor.execute(update_sql, (memory_ids,))
            cursor.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to update access stats: {e}")
    
    def create_memory_connection(self, 
                               source_memory_id: str,
                               target_memory_id: str,
                               connection_strength: float = 0.5,
                               connection_type: str = "semantic") -> bool:
        """
        Erstellt Verbindung zwischen zwei Memories (Neural Network)
        """
        try:
            cursor = self.connection.cursor()
            
            insert_sql = """
                INSERT INTO memory_connections 
                (source_memory_id, target_memory_id, connection_strength, connection_type)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_memory_id, target_memory_id)
                DO UPDATE SET 
                    connection_strength = EXCLUDED.connection_strength,
                    connection_type = EXCLUDED.connection_type;
            """
            
            cursor.execute(insert_sql, (
                source_memory_id, target_memory_id, connection_strength, connection_type
            ))
            cursor.close()
            
            logger.info(f"✅ Memory connection created: {source_memory_id} -> {target_memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create memory connection: {e}")
            return False
    
    def get_memory_connections(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Holt alle Verbindungen für eine Memory
        """
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # Bidirectional connections
            query_sql = """
                (SELECT target_memory_id as connected_memory_id, connection_strength, connection_type, 'outgoing' as direction
                 FROM memory_connections WHERE source_memory_id = %s)
                UNION
                (SELECT source_memory_id as connected_memory_id, connection_strength, connection_type, 'incoming' as direction
                 FROM memory_connections WHERE target_memory_id = %s)
                ORDER BY connection_strength DESC;
            """
            
            cursor.execute(query_sql, (memory_id, memory_id))
            connections = cursor.fetchall()
            cursor.close()
            
            return [dict(conn) for conn in connections]
            
        except Exception as e:
            logger.error(f"❌ Failed to get memory connections: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Liefert VectorStore Statistiken
        """
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # Grundstatistiken
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_memories,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT memory_type) as memory_types,
                    AVG(strength) as avg_strength,
                    MAX(access_count) as max_access_count,
                    AVG(access_count) as avg_access_count
                FROM memory_vectors;
            """)
            
            stats = dict(cursor.fetchone())
            
            # Memory Types Verteilung
            cursor.execute("""
                SELECT memory_type, COUNT(*) as count
                FROM memory_vectors
                GROUP BY memory_type
                ORDER BY count DESC;
            """)
            
            memory_types = cursor.fetchall()
            stats['memory_type_distribution'] = {row['memory_type']: row['count'] for row in memory_types}
            
            # Connection Statistics
            cursor.execute("SELECT COUNT(*) as total_connections FROM memory_connections;")
            stats['total_connections'] = cursor.fetchone()['total_connections']
            
            cursor.close()
            
            return {
                'success': True,
                'statistics': stats,
                'embedding_dimension': self.embedding_dimension,
                'is_initialized': self.is_initialized
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {'success': False, 'error': str(e)}
    
    def close(self):
        """Schließt VectorStore Verbindungen"""
        try:
            if self.connection:
                self.connection.close()
                logger.info("✅ VectorStore connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing VectorStore: {e}")

# Singleton Pattern
_vector_store_instance = None

def get_vector_store() -> VectorStore:
    """Gibt VectorStore Singleton zurück"""
    global _vector_store_instance
    
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
        
    return _vector_store_instance