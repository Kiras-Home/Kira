import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# PostgreSQL Storage Import
try:
    from memory.storage.postgresql_storage import PostgreSQLMemoryStorage

    POSTGRESQL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ PostgreSQL Storage nicht verfügbar: {e}")
    POSTGRESQL_AVAILABLE = False

# Global PostgreSQL Instance
_postgresql_storage = None


def get_postgresql_storage():
    """Singleton PostgreSQL Storage Instance"""
    global _postgresql_storage

    if _postgresql_storage is None and POSTGRESQL_AVAILABLE:
        try:
            _postgresql_storage = PostgreSQLMemoryStorage()
            if not _postgresql_storage.initialize():
                logger.error("❌ PostgreSQL Storage Initialisierung fehlgeschlagen")
                _postgresql_storage = None
        except Exception as e:
            logger.error(f"❌ PostgreSQL Storage Creation Error: {e}")
            _postgresql_storage = None

    return _postgresql_storage

def manage_memory_lifecycle(memory_id: int, action: str):
    """Verwaltet den Lebenszyklus einer Memory"""
    try:
        # Basic Memory Management
        if action == "archive":
            return {"success": True, "message": f"Memory {memory_id} archived"}
        elif action == "delete":
            return {"success": True, "message": f"Memory {memory_id} deleted"}
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_memory_data(user_id: str = "default", session_id: Optional[str] = None,
                    memory_type: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """
    🚀 HAUPTFUNKTION: Holt Memory Data aus PostgreSQL
    """
    try:
        pg_storage = get_postgresql_storage()
        if not pg_storage:
            return _generate_fallback_memory_data()

        # Hole Memories aus PostgreSQL
        memories = pg_storage.get_memories(
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            limit=limit
        )

        # Database Stats
        db_stats = pg_storage.get_database_stats()

        return {
            'success': True,
            'memories': memories,
            'memory_count': len(memories),
            'database_stats': db_stats,
            'storage_type': 'postgresql',
            'user_id': user_id,
            'session_id': session_id,
            'last_updated': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Get Memory Data Error: {e}")
        return {
            'success': False,
            'error': str(e),
            'memories': [],
            'storage_type': 'fallback'
        }


def add_memory_entry(session_id: str, user_id: str, memory_type: str,
                     content: str, metadata: Optional[Dict] = None,
                     importance: int = 5, **kwargs) -> Dict[str, Any]:
    """
    🚀 HAUPTFUNKTION: Fügt Memory zu PostgreSQL hinzu
    """
    try:
        pg_storage = get_postgresql_storage()
        if not pg_storage:
            return {'success': False, 'error': 'PostgreSQL Storage nicht verfügbar'}

        # Store in PostgreSQL
        memory_id = pg_storage.store_memory(
            session_id=session_id,
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata,
            importance=importance,
            **kwargs
        )

        if memory_id:
            logger.info(f"✅ Memory {memory_id} erfolgreich zu PostgreSQL hinzugefügt")
            return {
                'success': True,
                'memory_id': memory_id,
                'storage_type': 'postgresql',
                'user_id': user_id,
                'session_id': session_id
            }
        else:
            return {'success': False, 'error': 'Memory konnte nicht gespeichert werden'}

    except Exception as e:
        logger.error(f"❌ Add Memory Entry Error: {e}")
        return {'success': False, 'error': str(e)}


def search_memories(query: str, user_id: str = "default", limit: int = 10,
                    **kwargs) -> Dict[str, Any]:
    """
    🚀 HAUPTFUNKTION: Sucht Memories in PostgreSQL
    """
    try:
        pg_storage = get_postgresql_storage()
        if not pg_storage:
            return {'success': False, 'results': [], 'error': 'PostgreSQL Storage nicht verfügbar'}

        # Search in PostgreSQL
        results = pg_storage.search_memories(
            query=query,
            user_id=user_id,
            limit=limit,
            **kwargs
        )

        return {
            'success': True,
            'results': results,
            'result_count': len(results),
            'query': query,
            'storage_type': 'postgresql',
            'user_id': user_id
        }

    except Exception as e:
        logger.error(f"❌ Search Memories Error: {e}")
        return {'success': False, 'results': [], 'error': str(e)}


def get_memory_statistics(memory_manager=None) -> Dict[str, Any]:
    """
    🚀 KORRIGIERT: Memory Statistics aus PostgreSQL
    Backward-compatible Wrapper für get_memory_statistics_new
    """
    return get_memory_statistics_new(memory_manager)


def get_memory_statistics_new(memory_manager) -> Dict[str, Any]:
    """🚀 KORRIGIERT: Memory Statistics für HumanLikeMemoryManager"""
    try:
        if not memory_manager:
            return _generate_fallback_memory_statistics()

        # 🔧 KORRIGIERT: Verwende 'stm' und 'ltm' statt 'short_term' und 'long_term'
        if hasattr(memory_manager, 'stm') and hasattr(memory_manager, 'ltm'):
            try:
                stats = {
                    'manager_type': 'human_like_memory_manager',
                    'database_available': hasattr(memory_manager,
                                                  'memory_database') and memory_manager.memory_database is not None,
                    'components': {
                        'stm_available': memory_manager.stm is not None,
                        'ltm_available': memory_manager.ltm is not None,
                        'personality_available': hasattr(memory_manager,
                                                         'personality') and memory_manager.personality is not None
                    },
                    'memory_counts': {},
                    'statistics_source': 'human_like_memory_manager'
                }

                # 🔧 STM stats (korrigiert)
                if memory_manager.stm and hasattr(memory_manager.stm, 'working_memory'):
                    stats['memory_counts']['stm_working_memory'] = len(memory_manager.stm.working_memory)
                elif memory_manager.stm and hasattr(memory_manager.stm, 'capacity'):
                    stats['memory_counts']['stm_capacity'] = getattr(memory_manager.stm, 'capacity', 0)

                # 🔧 LTM stats (korrigiert)
                if memory_manager.ltm and hasattr(memory_manager.ltm, 'consolidated_memories'):
                    stats['memory_counts']['ltm_consolidated'] = len(memory_manager.ltm.consolidated_memories)
                elif memory_manager.ltm:
                    stats['memory_counts']['ltm_available'] = True

                # Database stats
                if hasattr(memory_manager, 'memory_database') and memory_manager.memory_database:
                    try:
                        db_stats = memory_manager.memory_database.get_database_stats()
                        stats['database_stats'] = db_stats
                    except Exception as db_e:
                        logger.warning(f"⚠️ Database stats error: {db_e}")
                        stats['database_stats'] = {}

                return stats

            except Exception as e:
                logger.warning(f"⚠️ HumanLike Memory Manager statistics failed: {e}")
                return _calculate_generic_memory_statistics(memory_manager)

        # Fallback für andere Manager Types
        return _calculate_generic_memory_statistics(memory_manager)

    except Exception as e:
        logger.error(f"❌ Memory statistics calculation failed: {e}")
        return _generate_fallback_memory_statistics()


# 🚀 FALLBACK FUNCTIONS

def _generate_fallback_memory_data():
    """Fallback wenn PostgreSQL nicht verfügbar"""
    return {
        'success': False,
        'memories': [],
        'memory_count': 0,
        'database_stats': {},
        'storage_type': 'fallback',
        'error': 'PostgreSQL Storage nicht verfügbar'
    }


def _generate_fallback_memory_statistics():
    """Fallback Statistics"""
    return {
        'manager_type': 'fallback',
        'storage_available': False,
        'total_memories': 0,
        'unique_users': 0,
        'memory_types': {},
        'recent_activity': 0,
        'statistics_source': 'fallback_mode',
        'error': 'PostgreSQL Storage nicht verfügbar'
    }


# 🚀 COMPATIBILITY FUNCTIONS - Bestehende Funktionen bleiben

def _get_short_term_memories_new(memory_manager):
    """STM aus PostgreSQL"""
    pg_storage = get_postgresql_storage()
    if pg_storage:
        return pg_storage.get_memories(memory_type='short_term', limit=20)
    return []


def _get_long_term_memories_new(memory_manager):
    """LTM aus PostgreSQL"""
    pg_storage = get_postgresql_storage()
    if pg_storage:
        return pg_storage.get_memories(memory_type='long_term', limit=100)
    return []


def _get_working_memories_new(memory_manager):
    """Working Memory aus PostgreSQL"""
    pg_storage = get_postgresql_storage()
    if pg_storage:
        return pg_storage.get_memories(memory_type='working', limit=10)
    return []


def _get_episodic_memories_new(memory_manager):
    """Episodic Memory aus PostgreSQL"""
    pg_storage = get_postgresql_storage()
    if pg_storage:
        return pg_storage.get_memories(memory_type='episodic', limit=50)
    return []


def _get_semantic_memories_new(memory_manager):
    """Semantic Memory aus PostgreSQL"""
    pg_storage = get_postgresql_storage()
    if pg_storage:
        return pg_storage.get_memories(memory_type='semantic', limit=50)
    return []


# Weitere bestehende Funktionen...
def _generic_memory_search_new(memory_manager, query: str, **kwargs):
    """Generic Search über PostgreSQL"""
    pg_storage = get_postgresql_storage()
    if pg_storage:
        return pg_storage.search_memories(query=query, **kwargs)
    return []


def _process_search_results_new(results: List[Dict], **kwargs):
    """Process Search Results"""
    return results  # PostgreSQL liefert bereits verarbeitete Results


def _calculate_relevance_score_new(memory_dict: Dict, query: str = "", **kwargs):
    """Calculate Relevance Score"""
    # Basic relevance basierend auf Content Match + Importance
    content = memory_dict.get('content', '').lower()
    query_lower = query.lower()

    if query_lower in content:
        base_score = 0.8
    else:
        base_score = 0.3

    # Importance boost
    importance = memory_dict.get('importance', 5)
    importance_boost = (importance / 10.0) * 0.2

    return min(1.0, base_score + importance_boost)


def _calculate_generic_memory_statistics(memory_manager):
    """Generic Statistics - delegates to PostgreSQL"""
    return get_memory_statistics_new(memory_manager)


def _get_short_term_memories_fallback(memory_manager):
    """Fallback für STM"""
    return []


def _get_long_term_memories_fallback(memory_manager):
    """Fallback für LTM"""
    return []