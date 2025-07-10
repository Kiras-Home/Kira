"""
💾 MEMORY STORAGE SYSTEM - Unified Storage Factory
Zentrales Storage System mit intelligenter Backend-Auswahl
"""

import logging
from typing import Dict, Any, Optional, List, Type, Union
from datetime import datetime
from pathlib import Path

# Storage Interface
from .memory_storage_interface import MemoryStorageInterface

# Available Storage Backends
STORAGE_BACKENDS = {}
STORAGE_AVAILABLE = {}

logger = logging.getLogger(__name__)

# ✅ DYNAMIC STORAGE BACKEND REGISTRATION
def register_storage_backend():
    """Registriert verfügbare Storage Backends"""
    global STORAGE_BACKENDS, STORAGE_AVAILABLE
    
    # PostgreSQL Storage
    try:
        from .postgresql_storage import PostgreSQLMemoryStorage
        STORAGE_BACKENDS['postgresql'] = PostgreSQLMemoryStorage
        STORAGE_AVAILABLE['postgresql'] = True
        logger.info("✅ PostgreSQL storage backend registered")
    except ImportError as e:
        STORAGE_AVAILABLE['postgresql'] = False
        logger.warning(f"⚠️ PostgreSQL storage not available: {e}")
    
    # Enhanced Memory Database
    try:
        from .memory_database import EnhancedMemoryDatabase
        STORAGE_BACKENDS['enhanced_memory'] = EnhancedMemoryDatabase
        STORAGE_AVAILABLE['enhanced_memory'] = True
        logger.info("✅ Enhanced memory database registered")
    except ImportError as e:
        STORAGE_AVAILABLE['enhanced_memory'] = False
        logger.warning(f"⚠️ Enhanced memory database not available: {e}")
    
    # Memory Models
    try:
        from .memory_models import MemoryModel, ConversationModel
        STORAGE_BACKENDS['memory_models'] = (MemoryModel, ConversationModel)
        STORAGE_AVAILABLE['memory_models'] = True
        logger.info("✅ Memory models registered")
    except ImportError as e:
        STORAGE_AVAILABLE['memory_models'] = False
        logger.warning(f"⚠️ Memory models not available: {e}")

# Initialize backends
register_storage_backend()

# ✅ UNIFIED STORAGE FACTORY
class UnifiedStorageFactory:
    """
    🏭 UNIFIED STORAGE FACTORY
    Zentrale Factory für alle Storage-Backends
    """
    
    @staticmethod
    def create_storage(
        storage_type: str = 'auto',
        config: Optional[Dict[str, Any]] = None,
        fallback_enabled: bool = True
    ) -> Optional[MemoryStorageInterface]:
        """
        Erstellt Storage Backend basierend auf Typ und Verfügbarkeit
        
        Args:
            storage_type: 'postgresql', 'enhanced_memory', 'auto'
            config: Storage-spezifische Konfiguration
            fallback_enabled: Enable fallback to available storage
            
        Returns:
            Storage instance or None
        """
        try:
            config = config or {}
            
            # Auto-detection
            if storage_type == 'auto':
                storage_type = UnifiedStorageFactory._detect_best_storage()
            
            # Erstelle spezifischen Storage
            if storage_type in STORAGE_BACKENDS and STORAGE_AVAILABLE.get(storage_type, False):
                storage_class = STORAGE_BACKENDS[storage_type]
                
                try:
                    # Erstelle Storage mit Config
                    if storage_type == 'postgresql':
                        storage = storage_class(
                            host=config.get('host', 'localhost'),
                            port=config.get('port', 5432),
                            database=config.get('database', 'kira_memory'),
                            user=config.get('user', 'kira_user'),
                            password=config.get('password', ''),
                            **{k: v for k, v in config.items() if k not in ['host', 'port', 'database', 'user', 'password']}
                        )
                    elif storage_type == 'enhanced_memory':
                        storage = storage_class(**config)
                    else:
                        storage = storage_class(**config)
                    
                    # Test storage connection
                    if hasattr(storage, 'test_connection'):
                        if storage.test_connection():
                            logger.info(f"✅ Storage created: {storage_type}")
                            return storage
                        else:
                            logger.warning(f"⚠️ Storage connection test failed: {storage_type}")
                            if fallback_enabled:
                                return UnifiedStorageFactory._create_fallback_storage(config)
                    else:
                        logger.info(f"✅ Storage created (no connection test): {storage_type}")
                        return storage
                        
                except Exception as e:
                    logger.error(f"❌ Failed to create {storage_type} storage: {e}")
                    if fallback_enabled:
                        return UnifiedStorageFactory._create_fallback_storage(config)
            
            else:
                logger.warning(f"⚠️ Storage type not available: {storage_type}")
                if fallback_enabled:
                    return UnifiedStorageFactory._create_fallback_storage(config)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Storage creation failed: {e}")
            return None
    
    @staticmethod
    def _detect_best_storage() -> str:
        """Erkennt das beste verfügbare Storage System"""
        
        # Priority: PostgreSQL > Enhanced Memory > Fallback
        if STORAGE_AVAILABLE.get('postgresql', False):
            return 'postgresql'
        elif STORAGE_AVAILABLE.get('enhanced_memory', False):
            return 'enhanced_memory'
        else:
            logger.warning("⚠️ No database storage available, using memory-only")
            return 'memory_only'
    
    @staticmethod
    def _create_fallback_storage(config: Dict[str, Any]) -> Optional[MemoryStorageInterface]:
        """Erstellt Fallback Storage"""
        try:
            # Try enhanced memory database first
            if STORAGE_AVAILABLE.get('enhanced_memory', False):
                storage_class = STORAGE_BACKENDS['enhanced_memory']
                return storage_class(**config)
            
            # Ultimate fallback - in-memory storage
            logger.warning("⚠️ Using in-memory storage only (no persistence)")
            return None
            
        except Exception as e:
            logger.error(f"❌ Fallback storage creation failed: {e}")
            return None
    
    @staticmethod
    def get_available_storage_types() -> List[str]:
        """Gibt verfügbare Storage Types zurück"""
        return [
            storage_type for storage_type, available 
            in STORAGE_AVAILABLE.items() 
            if available
        ]
    
    @staticmethod
    def check_storage_health() -> Dict[str, Any]:
        """Überprüft die Gesundheit aller Storage Backends"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'storage_backends': {},
            'overall_status': 'unknown',
            'recommendations': []
        }
        
        healthy_backends = 0
        total_backends = len(STORAGE_BACKENDS)
        
        for backend_name, backend_class in STORAGE_BACKENDS.items():
            try:
                # Basic availability check
                is_available = STORAGE_AVAILABLE.get(backend_name, False)
                
                backend_health = {
                    'available': is_available,
                    'backend_class': backend_class.__name__ if hasattr(backend_class, '__name__') else str(backend_class),
                    'status': 'healthy' if is_available else 'unavailable'
                }
                
                # Connection test if possible
                if is_available and hasattr(backend_class, '__call__'):
                    try:
                        test_instance = backend_class()
                        if hasattr(test_instance, 'test_connection'):
                            connection_ok = test_instance.test_connection()
                            backend_health['connection_test'] = connection_ok
                            if not connection_ok:
                                backend_health['status'] = 'connection_failed'
                        else:
                            backend_health['connection_test'] = 'not_supported'
                    except Exception as e:
                        backend_health['connection_test'] = False
                        backend_health['status'] = 'creation_failed'
                        backend_health['error'] = str(e)
                
                if backend_health['status'] == 'healthy':
                    healthy_backends += 1
                
                health_report['storage_backends'][backend_name] = backend_health
                
            except Exception as e:
                health_report['storage_backends'][backend_name] = {
                    'available': False,
                    'status': 'error',
                    'error': str(e)
                }
        
        # Overall status
        if healthy_backends == total_backends:
            health_report['overall_status'] = 'excellent'
        elif healthy_backends > 0:
            health_report['overall_status'] = 'good'
            health_report['recommendations'].append(f"Some storage backends unavailable ({total_backends - healthy_backends}/{total_backends})")
        else:
            health_report['overall_status'] = 'critical'
            health_report['recommendations'].append("No storage backends available - data will not persist")
        
        return health_report

# ✅ STORAGE CONFIGURATION MANAGEMENT
class StorageConfig:
    """Configuration Manager für Storage Systems"""
    
    DEFAULT_POSTGRESQL_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'kira_memory',
        'user': 'kira_user',
        'password': '',
        'pool_size': 5,
        'max_overflow': 10,
        'pool_timeout': 30,
        'pool_recycle': 3600
    }
    
    DEFAULT_ENHANCED_MEMORY_CONFIG = {
        'persistence_file': 'memory/data/enhanced_memory.db',
        'backup_enabled': True,
        'backup_interval_hours': 6,
        'max_memory_size': 1000000,  # 1M memories
        'compression_enabled': True
    }
    
    @staticmethod
    def get_default_config(storage_type: str) -> Dict[str, Any]:
        """Holt Default-Konfiguration für Storage Type"""
        if storage_type == 'postgresql':
            return StorageConfig.DEFAULT_POSTGRESQL_CONFIG.copy()
        elif storage_type == 'enhanced_memory':
            return StorageConfig.DEFAULT_ENHANCED_MEMORY_CONFIG.copy()
        else:
            return {}
    
    @staticmethod
    def load_config_from_file(config_path: str) -> Dict[str, Any]:
        """Lädt Storage-Konfiguration aus Datei"""
        try:
            import json
            from pathlib import Path
            
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {config_path}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

# ✅ CONVENIENCE FUNCTIONS
def create_storage(storage_type: str = 'auto', **kwargs) -> Optional[MemoryStorageInterface]:
    """Convenience function für Storage-Erstellung"""
    return UnifiedStorageFactory.create_storage(storage_type=storage_type, config=kwargs)

def create_conversation_storage(**kwargs) -> Optional[MemoryStorageInterface]:
    """Spezialisierte Funktion für Conversation Storage"""
    # Priorität auf PostgreSQL für Conversations
    if STORAGE_AVAILABLE.get('postgresql', False):
        return create_storage('postgresql', **kwargs)
    else:
        return create_storage('auto', **kwargs)

def create_memory_storage(**kwargs) -> Optional[MemoryStorageInterface]:
    """Spezialisierte Funktion für Memory Storage"""
    return create_storage('auto', **kwargs)

def get_storage_status() -> Dict[str, Any]:
    """Holt aktuellen Storage Status"""
    return {
        'available_backends': UnifiedStorageFactory.get_available_storage_types(),
        'backend_availability': STORAGE_AVAILABLE.copy(),
        'recommended_backend': UnifiedStorageFactory._detect_best_storage(),
        'health_check': UnifiedStorageFactory.check_storage_health()
    }

# ✅ STORAGE INTERFACE VALIDATION
def validate_storage_interface(storage: Any) -> bool:
    """Validiert ob Storage das Interface implementiert"""
    required_methods = [
        'store_memory',
        'get_memory',
        'search_memories',
        'delete_memory'
    ]
    
    for method in required_methods:
        if not hasattr(storage, method):
            logger.error(f"Storage missing required method: {method}")
            return False
    
    return True

# ✅ PUBLIC EXPORTS
__all__ = [
    # Factory
    'UnifiedStorageFactory',
    'StorageConfig',
    
    # Convenience functions
    'create_storage',
    'create_conversation_storage', 
    'create_memory_storage',
    'get_storage_status',
    'validate_storage_interface',
    
    # Status
    'STORAGE_BACKENDS',
    'STORAGE_AVAILABLE',
    
    # Interface
    'MemoryStorageInterface'
]

# ✅ INITIALIZATION MESSAGE
available_count = sum(STORAGE_AVAILABLE.values())
total_count = len(STORAGE_AVAILABLE)

logger.info(f"💾 Storage System initialized")
logger.info(f"   Available backends: {available_count}/{total_count}")
logger.info(f"   Recommended: {UnifiedStorageFactory._detect_best_storage()}")
for backend, available in STORAGE_AVAILABLE.items():
    status = "✅" if available else "❌"
    logger.info(f"   {status} {backend}")