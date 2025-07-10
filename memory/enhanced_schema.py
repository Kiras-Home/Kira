"""
Enhanced Memory Database Schema - Vollständige Erweiterung
Fügt neue Spalten, Indizes und Tabellen für erweiterte Funktionalität hinzu
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def upgrade_memory_database(db_path: str = None):
    """Erweitert Memory Database Schema mit allen neuen Features"""
    
    if db_path is None:
        db_path = Path("data/kira_memory.db")
    else:
        db_path = Path(db_path)
    
    if not db_path.exists():
        print("❌ Memory database nicht gefunden")
        print(f"   Erstelle neue Datenbank: {db_path}")
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        conn = sqlite3.connect(str(db_path))
        
        print("🔧 Starte Enhanced Memory Database Schema Update...")
        print("=" * 60)
        
        # ✅ 1. ERWEITERTE MEMORY_ENTRIES SPALTEN
        print("\n1️⃣ Erweitere memory_entries Tabelle:")
        
        new_columns = [
            ("user_context", "TEXT", "Benutzer-spezifischer Kontext"),
            ("conversation_context", "TEXT", "Unterhaltungskontext"),
            ("emotion_type", "TEXT", "Emotionstyp der Memory"),
            ("emotion_intensity", "REAL DEFAULT 0.5", "Emotionsintensität (0-1)"),
            ("learning_weight", "REAL DEFAULT 1.0", "Lerngewichtung für KI"),
            ("device_context", "TEXT", "Geräte-Kontext (Smart Home)"),
            ("intent_detected", "TEXT", "Erkannte Benutzerabsicht"),
            ("confidence_score", "REAL DEFAULT 1.0", "Vertrauensscores für Memory"),
            ("voice_context", "TEXT", "Voice-spezifischer Kontext"),
            ("personality_aspect", "TEXT", "Persönlichkeitsaspekt"),
            ("consolidation_score", "REAL DEFAULT 0.0", "Score für Langzeit-Konsolidierung"),
            ("source_system", "TEXT DEFAULT 'general'", "Quellsystem der Memory"),
            ("memory_category", "TEXT", "Kategorisierung der Memory"),
            ("semantic_vector", "TEXT", "Semantischer Vektor (JSON)"),
            ("related_memories", "TEXT", "Verknüpfte Memory IDs (JSON)")
        ]
        
        for column_name, column_type, description in new_columns:
            try:
                conn.execute(f"ALTER TABLE memory_entries ADD COLUMN {column_name} {column_type}")
                print(f"  ✅ {column_name} - {description}")
            except sqlite3.OperationalError:
                print(f"  ℹ️ {column_name} bereits vorhanden")
        
        # ✅ 2. NEUE EMOTION_MEMORIES TABELLE
        print("\n2️⃣ Erstelle emotion_memories Tabelle:")
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS emotion_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                user_id TEXT NOT NULL,
                emotion_type TEXT NOT NULL,
                emotion_intensity REAL DEFAULT 0.5,
                emotion_context TEXT,
                trigger_words TEXT,
                response_emotion TEXT,
                learning_feedback REAL DEFAULT 0.0,
                session_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memory_entries (id)
            )
        ''')
        print("  ✅ emotion_memories Tabelle erstellt")
        
        # ✅ 3. NEUE CONVERSATION_FLOW TABELLE
        print("\n3️⃣ Erstelle conversation_flow Tabelle:")
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS conversation_flow (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                conversation_turn INTEGER NOT NULL,
                user_input TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                intent_detected TEXT,
                entities_extracted TEXT,
                emotion_flow TEXT,
                context_carried TEXT,
                response_quality_score REAL DEFAULT 0.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("  ✅ conversation_flow Tabelle erstellt")
        
        # ✅ 4. NEUE SMART_HOME_CONTEXT TABELLE
        print("\n4️⃣ Erstelle smart_home_context Tabelle:")
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS smart_home_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                device_id TEXT NOT NULL,
                device_type TEXT NOT NULL,
                action_performed TEXT,
                context_data TEXT,
                success_rate REAL DEFAULT 1.0,
                user_satisfaction REAL DEFAULT 0.5,
                learning_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memory_entries (id)
            )
        ''')
        print("  ✅ smart_home_context Tabelle erstellt")
        
        # ✅ 5. NEUE LEARNING_FEEDBACK TABELLE
        print("\n5️⃣ Erstelle learning_feedback Tabelle:")
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS learning_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                user_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                feedback_value REAL NOT NULL,
                feedback_text TEXT,
                improvement_suggestion TEXT,
                applied BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                applied_at DATETIME,
                FOREIGN KEY (memory_id) REFERENCES memory_entries (id)
            )
        ''')
        print("  ✅ learning_feedback Tabelle erstellt")
        
        # ✅ 6. NEUE MEMORY_ANALYTICS TABELLE
        print("\n6️⃣ Erstelle memory_analytics Tabelle:")
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memory_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_analyzed DATE NOT NULL,
                total_memories INTEGER DEFAULT 0,
                emotion_distribution TEXT,
                most_active_users TEXT,
                top_categories TEXT,
                learning_progress REAL DEFAULT 0.0,
                system_performance TEXT,
                recommendations TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date_analyzed)
            )
        ''')
        print("  ✅ memory_analytics Tabelle erstellt")
        
        # ✅ 7. FTS5 FULL-TEXT SEARCH
        print("\n7️⃣ Erstelle Full-Text Search (FTS5):")
        
        try:
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_search USING fts5(
                    content, tags, metadata, user_context, conversation_context,
                    content='memory_entries',
                    content_rowid='id'
                )
            ''')
            print("  ✅ FTS5 Volltextsuche aktiviert")
            
            # Trigger für automatische FTS-Updates
            conn.execute('''
                CREATE TRIGGER IF NOT EXISTS memory_entries_ai AFTER INSERT ON memory_entries BEGIN
                    INSERT INTO memory_search(rowid, content, tags, metadata, user_context, conversation_context)
                    VALUES (new.id, new.content, new.tags, new.metadata, new.user_context, new.conversation_context);
                END
            ''')
            
            conn.execute('''
                CREATE TRIGGER IF NOT EXISTS memory_entries_ad AFTER DELETE ON memory_entries BEGIN
                    INSERT INTO memory_search(memory_search, rowid, content, tags, metadata, user_context, conversation_context)
                    VALUES ('delete', old.id, old.content, old.tags, old.metadata, old.user_context, old.conversation_context);
                END
            ''')
            
            conn.execute('''
                CREATE TRIGGER IF NOT EXISTS memory_entries_au AFTER UPDATE ON memory_entries BEGIN
                    INSERT INTO memory_search(memory_search, rowid, content, tags, metadata, user_context, conversation_context)
                    VALUES ('delete', old.id, old.content, old.tags, old.metadata, old.user_context, old.conversation_context);
                    INSERT INTO memory_search(rowid, content, tags, metadata, user_context, conversation_context)
                    VALUES (new.id, new.content, new.tags, new.metadata, new.user_context, new.conversation_context);
                END
            ''')
            
            print("  ✅ FTS5 Auto-Update Trigger erstellt")
            
        except Exception as e:
            print(f"  ⚠️ FTS5 nicht verfügbar: {e}")
        
        # ✅ 8. ERWEITERTE INDIZES
        print("\n8️⃣ Erstelle erweiterte Indizes:")
        
        enhanced_indexes = [
            ("idx_memory_emotion", "memory_entries(emotion_type, emotion_intensity)"),
            ("idx_memory_learning", "memory_entries(learning_weight, access_count)"),
            ("idx_memory_device", "memory_entries(device_context, source_system)"),
            ("idx_memory_conversation", "memory_entries(session_id, created_at, conversation_context)"),
            ("idx_memory_consolidation", "memory_entries(consolidation_score, importance)"),
            ("idx_memory_category", "memory_entries(memory_category, user_id)"),
            ("idx_emotion_user_type", "emotion_memories(user_id, emotion_type)"),
            ("idx_emotion_intensity", "emotion_memories(emotion_intensity, created_at)"),
            ("idx_conversation_session", "conversation_flow(session_id, conversation_turn)"),
            ("idx_conversation_intent", "conversation_flow(intent_detected, user_id)"),
            ("idx_smarthome_device", "smart_home_context(device_id, device_type)"),
            ("idx_smarthome_success", "smart_home_context(success_rate, created_at)"),
            ("idx_feedback_type", "learning_feedback(feedback_type, applied)"),
            ("idx_feedback_memory", "learning_feedback(memory_id, feedback_value)"),
            ("idx_analytics_date", "memory_analytics(date_analyzed)")
        ]
        
        for index_name, index_definition in enhanced_indexes:
            try:
                conn.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON {index_definition}')
                print(f"  ✅ {index_name}")
            except Exception as e:
                print(f"  ⚠️ {index_name} Fehler: {e}")
        
        # ✅ 9. VIEWS FÜR KOMPLEXE ABFRAGEN
        print("\n9️⃣ Erstelle Views für Analytics:")
        
        # User Emotion Profile View
        conn.execute('''
            CREATE VIEW IF NOT EXISTS user_emotion_profile AS
            SELECT 
                user_id,
                emotion_type,
                AVG(emotion_intensity) as avg_intensity,
                COUNT(*) as emotion_count,
                MAX(created_at) as last_emotion
            FROM emotion_memories
            GROUP BY user_id, emotion_type
        ''')
        print("  ✅ user_emotion_profile View")
        
        # Memory Quality View
        conn.execute('''
            CREATE VIEW IF NOT EXISTS memory_quality AS
            SELECT 
                me.id,
                me.user_id,
                me.memory_type,
                me.importance,
                me.access_count,
                me.learning_weight,
                COALESCE(AVG(lf.feedback_value), 0) as avg_feedback,
                COUNT(lf.id) as feedback_count
            FROM memory_entries me
            LEFT JOIN learning_feedback lf ON me.id = lf.memory_id
            GROUP BY me.id
        ''')
        print("  ✅ memory_quality View")
        
        # Conversation Analytics View
        conn.execute('''
            CREATE VIEW IF NOT EXISTS conversation_analytics AS
            SELECT 
                session_id,
                user_id,
                COUNT(*) as total_turns,
                AVG(response_quality_score) as avg_quality,
                STRING_AGG(DISTINCT intent_detected, ', ') as intents_detected,
                MIN(created_at) as session_start,
                MAX(created_at) as session_end
            FROM conversation_flow
            GROUP BY session_id, user_id
        ''')
        print("  ✅ conversation_analytics View")
        
        # ✅ 10. INITIAL DATA MIGRATION
        print("\n🔄 Migriere bestehende Daten:")
        
        # Update existing memories with default values
        conn.execute('''
            UPDATE memory_entries 
            SET 
                emotion_intensity = 0.5,
                learning_weight = CASE 
                    WHEN importance > 7 THEN 1.5
                    WHEN importance > 5 THEN 1.0
                    ELSE 0.8
                END,
                source_system = 'legacy',
                memory_category = memory_type
            WHERE emotion_intensity IS NULL
        ''')
        
        updated_count = conn.total_changes
        print(f"  ✅ {updated_count} bestehende Memories aktualisiert")
        
        # ✅ 11. SCHEMA VERSION TRACKING
        print("\n📋 Schema Version Tracking:")
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS schema_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_number TEXT NOT NULL,
                description TEXT NOT NULL,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                applied_by TEXT DEFAULT 'system'
            )
        ''')
        
        # Record this schema update
        schema_version = "2.0.0"
        schema_description = "Enhanced Memory System mit Emotion, Conversation Flow, Smart Home Context"
        
        conn.execute('''
            INSERT OR REPLACE INTO schema_versions (version_number, description)
            VALUES (?, ?)
        ''', (schema_version, schema_description))
        
        print(f"  ✅ Schema Version {schema_version} registriert")
        
        # ✅ 12. DATENBANKSTATISTIKEN
        print("\n📊 Database Statistics nach Update:")
        
        # Table counts
        tables = [
            'memory_entries', 'knowledge_entities', 'knowledge_relationships',
            'emotion_memories', 'conversation_flow', 'smart_home_context',
            'learning_feedback', 'memory_analytics'
        ]
        
        for table in tables:
            try:
                count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                print(f"  📋 {table}: {count} Einträge")
            except:
                print(f"  ⚠️ {table}: Tabelle nicht verfügbar")
        
        # Database size
        db_size = db_path.stat().st_size / 1024 / 1024
        print(f"  💾 Datenbankgröße: {db_size:.2f} MB")
        
        conn.commit()
        conn.close()
        
        print("\n" + "=" * 60)
        print("✅ Enhanced Memory Database Schema Update ERFOLGREICH!")
        print(f"🎯 Schema Version: {schema_version}")
        print(f"📁 Database Path: {db_path}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Schema Update Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_schema_version(db_path: str = None) -> Dict[str, Any]:
    """Prüft aktuelle Schema Version"""
    
    if db_path is None:
        db_path = Path("data/kira_memory.db")
    else:
        db_path = Path(db_path)
    
    if not db_path.exists():
        return {"error": "Database nicht gefunden"}
    
    try:
        conn = sqlite3.connect(str(db_path))
        
        # Check if schema_versions table exists
        cursor = conn.execute('''
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='schema_versions'
        ''')
        
        if cursor.fetchone():
            # Get latest version
            version_info = conn.execute('''
                SELECT version_number, description, applied_at 
                FROM schema_versions 
                ORDER BY applied_at DESC 
                LIMIT 1
            ''').fetchone()
            
            if version_info:
                return {
                    "current_version": version_info[0],
                    "description": version_info[1],
                    "applied_at": version_info[2],
                    "schema_available": True
                }
        
        return {
            "current_version": "1.0.0",
            "description": "Legacy Schema",
            "schema_available": False
        }
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

def create_test_data(db_path: str = None):
    """Erstellt Test-Daten für Enhanced Schema"""
    
    if db_path is None:
        db_path = Path("data/kira_memory.db")
    else:
        db_path = Path(db_path)
    
    try:
        conn = sqlite3.connect(str(db_path))
        
        print("🧪 Erstelle Test-Daten für Enhanced Memory Schema...")
        
        # Test Memory Entries
        test_memories = [
            {
                "session_id": "test_session_001",
                "user_id": "test_user",
                "memory_type": "conversation",
                "content": "Der Benutzer möchte das Wohnzimmer-Licht einschalten",
                "emotion_type": "neutral",
                "emotion_intensity": 0.5,
                "device_context": "smart_light_living_room",
                "intent_detected": "device_control",
                "memory_category": "smart_home"
            },
            {
                "session_id": "test_session_001",
                "user_id": "test_user",
                "memory_type": "emotion",
                "content": "Benutzer war frustriert über langsame Antwort",
                "emotion_type": "frustration",
                "emotion_intensity": 0.7,
                "personality_aspect": "impatience",
                "learning_weight": 1.2,
                "memory_category": "personality"
            },
            {
                "session_id": "test_session_002",
                "user_id": "test_user",
                "memory_type": "fact",
                "content": "Benutzer arbeitet als Software-Entwickler",
                "emotion_type": "neutral",
                "emotion_intensity": 0.5,
                "user_context": "professional_info",
                "memory_category": "personal_info"
            }
        ]
        
        for memory in test_memories:
            conn.execute('''
                INSERT INTO memory_entries 
                (session_id, user_id, memory_type, content, emotion_type, 
                 emotion_intensity, device_context, intent_detected, 
                 user_context, personality_aspect, learning_weight, 
                 memory_category, importance, created_at, last_accessed, 
                 content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory["session_id"],
                memory["user_id"],
                memory["memory_type"],
                memory["content"],
                memory.get("emotion_type"),
                memory.get("emotion_intensity", 0.5),
                memory.get("device_context"),
                memory.get("intent_detected"),
                memory.get("user_context"),
                memory.get("personality_aspect"),
                memory.get("learning_weight", 1.0),
                memory.get("memory_category"),
                5,  # importance
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                f"test_hash_{memory['session_id']}_{memory['memory_type']}"
            ))
        
        # Test Emotion Memories
        conn.execute('''
            INSERT INTO emotion_memories 
            (user_id, emotion_type, emotion_intensity, emotion_context, session_id)
            VALUES (?, ?, ?, ?, ?)
        ''', ("test_user", "happiness", 0.8, "successful_smart_home_control", "test_session_001"))
        
        # Test Conversation Flow
        conn.execute('''
            INSERT INTO conversation_flow 
            (session_id, user_id, conversation_turn, user_input, ai_response, 
             intent_detected, response_quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            "test_session_001", 
            "test_user", 
            1, 
            "Schalte das Licht im Wohnzimmer ein",
            "Gerne! Ich schalte das Wohnzimmer-Licht für dich ein.",
            "device_control",
            0.9
        ))
        
        # Test Smart Home Context
        conn.execute('''
            INSERT INTO smart_home_context 
            (device_id, device_type, action_performed, success_rate, user_satisfaction)
            VALUES (?, ?, ?, ?, ?)
        ''', ("smart_light_living_room", "smart_light", "turn_on", 1.0, 0.9))
        
        conn.commit()
        conn.close()
        
        print("✅ Test-Daten erfolgreich erstellt!")
        
    except Exception as e:
        print(f"❌ Test-Daten Fehler: {e}")

if __name__ == "__main__":
    print("🚀 Enhanced Memory Database Schema Upgrade")
    print("Führe Schema-Update durch...")
    
    # Run upgrade
    success = upgrade_memory_database()
    
    if success:
        print("\n🧪 Erstelle Test-Daten...")
        create_test_data()
        
        print("\n📊 Prüfe Schema Version...")
        version_info = check_schema_version()
        print(f"Version Info: {version_info}")
    
    print("\n🎉 Schema Update abgeschlossen!")