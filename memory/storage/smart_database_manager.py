#!/usr/bin/env python3
"""
Smart Database Auto-Selection
Automatische Auswahl zwischen PostgreSQL und SQLite basierend auf Verfügbarkeit
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SmartDatabaseManager:
    """
    Intelligente Datenbank-Auswahl und -Erstellung
    
    Funktionsweise:
    1. Versuche PostgreSQL (mit Auto-Creation)
    2. Fallback zu SQLite falls PostgreSQL nicht verfügbar
    3. Biete einheitliche Interface
    """
    
    def __init__(self, prefer_postgresql: bool = True):
        self.prefer_postgresql = prefer_postgresql
        self.active_backend = None
        self.storage_instance = None
        
    def initialize(self, database_config: Optional[Dict] = None, 
                   connection_string: Optional[str] = None) -> bool:
        """
        Initialisiere die beste verfügbare Datenbank
        
        Args:
            database_config: PostgreSQL config dict
            connection_string: PostgreSQL connection string
            
        Returns:
            bool: True wenn erfolgreich initialisiert
        """
        
        if self.prefer_postgresql:
            # Versuche PostgreSQL zuerst
            if self._try_postgresql(database_config, connection_string):
                return True
            
            # Fallback zu SQLite
            return self._try_sqlite()
        else:
            # Versuche SQLite zuerst
            if self._try_sqlite():
                return True
            
            # Fallback zu PostgreSQL
            return self._try_postgresql(database_config, connection_string)
    
    def _try_postgresql(self, database_config: Optional[Dict] = None, 
                        connection_string: Optional[str] = None) -> bool:
        """Versuche PostgreSQL mit Auto-Creation"""
        try:
            logger.info("🔍 Versuche PostgreSQL mit Auto-Database Creation...")
            
            # Prüfe ob PostgreSQL verfügbar ist
            import psycopg2
            
            # Teste Server-Verbindung
            try:
                test_conn = psycopg2.connect(
                    host='localhost',
                    port=5432,
                    dbname='postgres',
                    user='postgres',
                    password='postgres',
                    connect_timeout=5
                )
                test_conn.close()
                logger.info("✅ PostgreSQL Server ist erreichbar")
            except Exception as e:
                logger.warning(f"⚠️ PostgreSQL Server nicht erreichbar: {e}")
                return False
            
            # Initialisiere PostgreSQL Storage
            from .postgresql_storage import PostgreSQLStorage
            
            self.storage_instance = PostgreSQLStorage(
                connection_string=connection_string,
                database_config=database_config
            )
            
            if self.storage_instance.initialize():
                self.active_backend = 'postgresql'
                logger.info("✅ PostgreSQL Storage erfolgreich initialisiert")
                return True
            else:
                logger.warning("⚠️ PostgreSQL Storage Initialisierung fehlgeschlagen")
                return False
                
        except ImportError:
            logger.warning("⚠️ psycopg2 nicht verfügbar - PostgreSQL nicht möglich")
            return False
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL Setup fehlgeschlagen: {e}")
            return False
    
    def _try_sqlite(self) -> bool:
        """Versuche SQLite Fallback"""
        try:
            logger.info("🔍 Versuche SQLite Fallback...")
            
            # Erstelle SQLite Storage
            from .sqlite_storage import SQLiteStorage
            
            # SQLite Datenbank im memory/data/ Verzeichnis
            db_path = os.path.join('memory', 'data', 'kira_memory.db')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            self.storage_instance = SQLiteStorage(db_path=db_path)
            
            if self.storage_instance.initialize():
                self.active_backend = 'sqlite'
                logger.info(f"✅ SQLite Storage erfolgreich initialisiert: {db_path}")
                return True
            else:
                logger.error("❌ SQLite Storage Initialisierung fehlgeschlagen")
                return False
                
        except ImportError:
            logger.error("❌ SQLite Storage nicht verfügbar")
            return False
        except Exception as e:
            logger.error(f"❌ SQLite Setup fehlgeschlagen: {e}")
            return False
    
    def get_storage(self):
        """Gibt die aktive Storage-Instanz zurück"""
        return self.storage_instance
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Gibt Informationen über das aktive Backend zurück"""
        return {
            'backend': self.active_backend,
            'available': self.storage_instance is not None,
            'initialized': self.storage_instance._initialized if self.storage_instance else False,
            'connection_info': getattr(self.storage_instance, 'connection_string', None) if self.active_backend == 'postgresql' else getattr(self.storage_instance, 'db_path', None)
        }

# Convenience-Funktionen
def create_smart_storage(prefer_postgresql: bool = True, 
                        database_config: Optional[Dict] = None,
                        connection_string: Optional[str] = None):
    """
    Erstellt automatisch die beste verfügbare Datenbank
    
    Args:
        prefer_postgresql: Bevorzuge PostgreSQL falls verfügbar
        database_config: PostgreSQL Konfiguration
        connection_string: PostgreSQL Connection String
        
    Returns:
        Tuple[storage_instance, backend_info]
    """
    manager = SmartDatabaseManager(prefer_postgresql=prefer_postgresql)
    
    if manager.initialize(database_config, connection_string):
        return manager.get_storage(), manager.get_backend_info()
    else:
        return None, {'backend': None, 'available': False, 'error': 'Keine Datenbank verfügbar'}

def get_recommended_storage():
    """Gibt die empfohlene Storage-Konfiguration zurück"""
    return create_smart_storage(prefer_postgresql=True)

if __name__ == "__main__":
    # Test der Smart Database Selection
    print("🧪 Smart Database Manager Test")
    print("=" * 50)
    
    # Test verschiedene Szenarien
    scenarios = [
        {
            'name': 'Standard (PostgreSQL bevorzugt)',
            'prefer_postgresql': True,
            'config': None,
            'connection_string': None
        },
        {
            'name': 'SQLite bevorzugt',
            'prefer_postgresql': False,
            'config': None,
            'connection_string': None
        },
        {
            'name': 'Custom PostgreSQL Config',
            'prefer_postgresql': True,
            'config': {
                'host': 'localhost',
                'port': 5432,
                'dbname': 'kira_smart_test',
                'user': 'kira',
                'password': 'kira_password'
            },
            'connection_string': None
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🔍 Test: {scenario['name']}")
        print("-" * 30)
        
        storage, info = create_smart_storage(
            prefer_postgresql=scenario['prefer_postgresql'],
            database_config=scenario['config'],
            connection_string=scenario['connection_string']
        )
        
        if storage:
            print(f"✅ Storage erstellt")
            print(f"   Backend: {info['backend']}")
            print(f"   Verfügbar: {info['available']}")
            print(f"   Initialisiert: {info['initialized']}")
            if info.get('connection_info'):
                print(f"   Connection: {str(info['connection_info'])[:50]}...")
        else:
            print(f"❌ Storage-Erstellung fehlgeschlagen")
            print(f"   Error: {info.get('error', 'Unbekannt')}")
    
    print("\n" + "=" * 50)
    print("🎯 Smart Database Manager Test abgeschlossen")
