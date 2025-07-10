#!/usr/bin/env python3
"""
SQLite Memory Storage - Einfache Fallback-Implementierung
"""

import sqlite3
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class SQLiteStorage:
    """
    SQLite-basierter Memory Storage als Fallback für PostgreSQL
    
    Bietet die gleiche Interface wie PostgreSQL Storage aber mit SQLite
    """
    
    def __init__(self, db_path: str = "memory/data/kira_memory.db"):
        self.db_path = db_path
        self._initialized = False
        
        # Erstelle absoluten Pfad
        if not os.path.isabs(db_path):
            # Relativer Pfad zum Projekt-Root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.db_path = os.path.join(project_root, db_path)
        
        # Erstelle Verzeichnis falls nötig
        db_dir = os.path.dirname(self.db_path)
        if db_dir:  # Nur wenn Verzeichnis angegeben
            os.makedirs(db_dir, exist_ok=True)
        
        logger.info(f"🔧 SQLite Storage initialisiert: {self.db_path}")
    
    def initialize(self) -> bool:
        """Initialisiere SQLite Datenbank"""
        try:
            # Teste Verbindung
            if not self._test_connection():
                return False
            
            # Erstelle Schema
            self._create_schema()
            
            self._initialized = True
            logger.info("✅ SQLite Memory Storage erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ SQLite Initialization Error: {e}")
            return False
    
    def _test_connection(self) -> bool:
        """Teste SQLite Verbindung"""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                
                if result and result[0] == 1:
                    logger.info("✅ SQLite Connection Test erfolgreich")
                    return True
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"❌ SQLite Connection Test fehlgeschlagen: {e}")
            return False
    
    @contextmanager
    def get_connection(self):
        """Context Manager für SQLite Verbindung"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row  # Für dict-ähnliche Zugriffe
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"❌ SQLite Error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _create_schema(self):
        """Erstelle SQLite Schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Memory entries Tabelle
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT 'main',
                    user_id TEXT NOT NULL DEFAULT 'default',
                    memory_type TEXT NOT NULL DEFAULT 'general',
                    content TEXT NOT NULL,
                    importance INTEGER DEFAULT 5,
                    
                    -- Metadata
                    metadata TEXT DEFAULT '{}',
                    tags TEXT DEFAULT '',
                    
                    -- Enhanced Fields
                    user_context TEXT DEFAULT '',
                    emotion_type TEXT DEFAULT 'neutral',
                    emotion_intensity REAL DEFAULT 0.0,
                    emotion_valence REAL DEFAULT 0.0,
                    device_context TEXT DEFAULT 'unknown',
                    conversation_context TEXT DEFAULT '',
                    
                    -- STM/LTM Fields
                    stm_activation_level REAL DEFAULT 0.0,
                    ltm_significance_score REAL DEFAULT 0.0,
                    consolidation_score REAL DEFAULT 0.0,
                    memory_strength REAL DEFAULT 1.0,
                    decay_rate REAL DEFAULT 0.1,
                    reinforcement_count INTEGER DEFAULT 0,
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    expires_at TIMESTAMP,
                    content_hash TEXT,
                    
                    -- Additional Fields
                    consolidation_timestamp TIMESTAMP,
                    attention_weight REAL DEFAULT 0.5,
                    learning_weight REAL DEFAULT 1.0,
                    memory_category TEXT DEFAULT '',
                    pattern_tags TEXT DEFAULT '',
                    source_memory_id INTEGER,
                    related_memory_ids TEXT DEFAULT '[]'
                )
            ''')
            
            # Conversations Tabelle
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    conversation_type TEXT DEFAULT 'chat',
                    title TEXT,
                    
                    -- Conversation metadata
                    metadata TEXT DEFAULT '{}',
                    context TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Stats
                    message_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    importance_score REAL DEFAULT 0.0
                )
            ''')
            
            # Conversation Messages Tabelle
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    
                    -- Message metadata
                    metadata TEXT DEFAULT '{}',
                    tokens INTEGER DEFAULT 0,
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            ''')
            
            # Indizes erstellen
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_entries_session ON memory_entries(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_entries_user ON memory_entries(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_entries_type ON memory_entries(memory_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_entries_created ON memory_entries(created_at)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conv_messages_conv ON conversation_messages(conversation_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conv_messages_role ON conversation_messages(role)')
            
            logger.info("✅ SQLite Schema erfolgreich erstellt")
    
    def store_memory(self, content: str, user_id: str = "default", 
                    session_id: str = "main", memory_type: str = "general",
                    metadata: Dict[str, Any] = None, **kwargs) -> int:
        """Speichere Memory Entry"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO memory_entries (
                        session_id, user_id, memory_type, content, metadata,
                        importance, tags, user_context, emotion_type, 
                        emotion_intensity, emotion_valence, device_context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id, user_id, memory_type, content,
                    json.dumps(metadata or {}),
                    kwargs.get('importance', 5),
                    kwargs.get('tags', ''),
                    kwargs.get('user_context', ''),
                    kwargs.get('emotion_type', 'neutral'),
                    kwargs.get('emotion_intensity', 0.0),
                    kwargs.get('emotion_valence', 0.0),
                    kwargs.get('device_context', 'unknown')
                ))
                
                memory_id = cursor.lastrowid
                logger.debug(f"✅ Memory gespeichert: ID {memory_id}")
                return memory_id
                
        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern: {e}")
            return -1
    
    def retrieve_memories(self, user_id: str = "default", 
                         session_id: str = "main", limit: int = 10) -> List[Dict]:
        """Lade Memory Entries"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM memory_entries 
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (user_id, session_id, limit))
                
                results = []
                for row in cursor.fetchall():
                    memory = dict(row)
                    # JSON-Felder parsen
                    if memory.get('metadata'):
                        try:
                            memory['metadata'] = json.loads(memory['metadata'])
                        except:
                            memory['metadata'] = {}
                    results.append(memory)
                
                logger.debug(f"✅ {len(results)} Memories geladen")
                return results
                
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden: {e}")
            return []
    
    def search_memories(self, query: str, user_id: str = "default", 
                       limit: int = 10) -> List[Dict]:
        """Einfache Textsuche in Memories"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM memory_entries 
                    WHERE user_id = ? AND (
                        content LIKE ? OR 
                        tags LIKE ? OR 
                        user_context LIKE ?
                    )
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (user_id, f'%{query}%', f'%{query}%', f'%{query}%', limit))
                
                results = []
                for row in cursor.fetchall():
                    memory = dict(row)
                    if memory.get('metadata'):
                        try:
                            memory['metadata'] = json.loads(memory['metadata'])
                        except:
                            memory['metadata'] = {}
                    results.append(memory)
                
                logger.debug(f"✅ {len(results)} Memories gefunden für Query: {query}")
                return results
                
        except Exception as e:
            logger.error(f"❌ Fehler bei der Suche: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Datenbank-Statistiken"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM memory_entries")
                memory_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM conversations")
                conv_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM conversation_messages")
                msg_count = cursor.fetchone()[0]
                
                return {
                    'memory_entries': memory_count,
                    'conversations': conv_count,
                    'messages': msg_count,
                    'backend': 'sqlite',
                    'db_path': self.db_path,
                    'db_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                }
                
        except Exception as e:
            logger.error(f"❌ Fehler bei Statistiken: {e}")
            return {'error': str(e)}
    
    def close(self):
        """Schließe Storage (für SQLite nicht nötig)"""
        self._initialized = False
        logger.info("✅ SQLite Storage geschlossen")

if __name__ == "__main__":
    # Test der SQLite Storage
    print("🧪 SQLite Storage Test")
    print("=" * 40)
    
    # Test-Pfad
    test_db = "test_sqlite_storage.db"
    
    try:
        # Erstelle Storage
        storage = SQLiteStorage(test_db)
        
        # Initialisiere
        if storage.initialize():
            print("✅ SQLite Storage initialisiert")
            
            # Speichere Test-Memory
            memory_id = storage.store_memory(
                content="Test Memory für SQLite",
                user_id="test_user",
                memory_type="test",
                metadata={"test": True}
            )
            
            if memory_id > 0:
                print(f"✅ Memory gespeichert: ID {memory_id}")
            
            # Lade Memories
            memories = storage.retrieve_memories(user_id="test_user")
            print(f"✅ {len(memories)} Memories geladen")
            
            # Suche
            results = storage.search_memories("Test", user_id="test_user")
            print(f"✅ {len(results)} Memories gefunden")
            
            # Statistiken
            stats = storage.get_stats()
            print(f"✅ Statistiken: {stats}")
            
        else:
            print("❌ SQLite Storage Initialisierung fehlgeschlagen")
            
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")
    
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.remove(test_db)
            print("🧹 Test-Datenbank gelöscht")
    
    print("🎯 SQLite Storage Test abgeschlossen")
