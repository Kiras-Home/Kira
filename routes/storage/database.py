"""
Database Management Module - Teil 1/3
Database Operations, Connection Management, Basic Query Operations
"""

import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from contextlib import contextmanager
import json
import os

logger = logging.getLogger(__name__)

# Database Connection Pool
_connection_pool = {}
_connection_lock = threading.Lock()

def manage_database_operations(kira_instance=None,
                             operation_type: str = 'status',
                             database_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Verwaltet Database Operations
    
    Extrahiert aus kira_routes.py.backup Database Management Logic
    """
    try:
        if database_config is None:
            database_config = _get_default_database_config()
        
        # Initialize database operation session
        db_operation_session = {
            'session_id': f"db_ops_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'operation_type': operation_type,
            'database_config': database_config,
            'operation_results': {}
        }
        
        # Handle different operation types
        if operation_type == 'status':
            db_operation_session['operation_results'] = _get_database_status(database_config)
        
        elif operation_type == 'initialize':
            db_operation_session['operation_results'] = _initialize_database(database_config, kira_instance)
        
        elif operation_type == 'maintenance':
            db_operation_session['operation_results'] = _perform_database_maintenance(database_config)
        
        elif operation_type == 'backup':
            db_operation_session['operation_results'] = _create_database_backup(database_config)
        
        elif operation_type == 'optimize':
            db_operation_session['operation_results'] = _optimize_database_performance(database_config)
        
        elif operation_type == 'validate':
            db_operation_session['operation_results'] = _validate_database_integrity(database_config)
        
        else:
            db_operation_session['operation_results'] = {
                'status': 'error',
                'error': f'Unsupported operation type: {operation_type}'
            }
        
        # Add operation metadata
        db_operation_session.update({
            'end_time': datetime.now().isoformat(),
            'operation_success': db_operation_session['operation_results'].get('status') == 'success',
            'operation_duration': _calculate_operation_duration(db_operation_session)
        })
        
        return {
            'success': db_operation_session['operation_success'],
            'db_operation_session': db_operation_session,
            'operation_summary': {
                'operation_type': operation_type,
                'operation_success': db_operation_session['operation_success'],
                'operation_duration_ms': db_operation_session['operation_duration'],
                'database_status': db_operation_session['operation_results'].get('database_status', 'unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"Database operation management failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'fallback_database_status': _generate_fallback_database_status()
        }

def execute_database_queries(queries: Union[str, List[str]],
                           database_config: Dict[str, Any] = None,
                           query_options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Führt Database Queries aus
    
    Basiert auf kira_routes.py.backup Database Query Execution Logic
    """
    try:
        if database_config is None:
            database_config = _get_default_database_config()
        
        if query_options is None:
            query_options = {
                'transaction_mode': True,
                'fetch_results': True,
                'timeout_seconds': 30,
                'retry_attempts': 3
            }
        
        # Normalize queries to list
        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = queries
        
        # Initialize query execution session
        query_execution_session = {
            'session_id': f"query_exec_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'query_count': len(query_list),
            'query_options': query_options,
            'query_results': [],
            'execution_statistics': {}
        }
        
        # Execute queries
        with _get_database_connection(database_config) as connection:
            cursor = connection.cursor()
            
            # Start transaction if enabled
            if query_options.get('transaction_mode', True):
                cursor.execute('BEGIN TRANSACTION')
            
            try:
                for i, query in enumerate(query_list):
                    query_start_time = time.time()
                    
                    try:
                        # Execute query with timeout
                        cursor.execute(query)
                        
                        # Fetch results if requested
                        if query_options.get('fetch_results', True):
                            if query.strip().upper().startswith('SELECT'):
                                results = cursor.fetchall()
                                column_names = [desc[0] for desc in cursor.description] if cursor.description else []
                            else:
                                results = cursor.rowcount
                                column_names = []
                        else:
                            results = None
                            column_names = []
                        
                        query_duration = (time.time() - query_start_time) * 1000  # Convert to ms
                        
                        query_result = {
                            'query_index': i,
                            'query': query,
                            'status': 'success',
                            'results': results,
                            'column_names': column_names,
                            'execution_time_ms': query_duration,
                            'rows_affected': cursor.rowcount if cursor.rowcount >= 0 else 0
                        }
                        
                        query_execution_session['query_results'].append(query_result)
                        
                    except Exception as e:
                        query_duration = (time.time() - query_start_time) * 1000
                        
                        query_result = {
                            'query_index': i,
                            'query': query,
                            'status': 'error',
                            'error': str(e),
                            'execution_time_ms': query_duration,
                            'rows_affected': 0
                        }
                        
                        query_execution_session['query_results'].append(query_result)
                        
                        # Break on error if not in continue mode
                        if not query_options.get('continue_on_error', False):
                            break
                
                # Commit transaction if all successful
                if query_options.get('transaction_mode', True):
                    if all(result['status'] == 'success' for result in query_execution_session['query_results']):
                        cursor.execute('COMMIT')
                    else:
                        cursor.execute('ROLLBACK')
                
            except Exception as e:
                if query_options.get('transaction_mode', True):
                    cursor.execute('ROLLBACK')
                raise e
        
        # Compile execution statistics
        execution_statistics = _compile_query_execution_statistics(query_execution_session['query_results'])
        query_execution_session['execution_statistics'] = execution_statistics
        
        # Add session metadata
        query_execution_session.update({
            'end_time': datetime.now().isoformat(),
            'total_execution_time_ms': sum(result.get('execution_time_ms', 0) for result in query_execution_session['query_results']),
            'successful_queries': execution_statistics['successful_queries'],
            'failed_queries': execution_statistics['failed_queries']
        })
        
        return {
            'success': execution_statistics['success_rate'] > 0.5,  # At least 50% success rate
            'query_execution_session': query_execution_session,
            'execution_summary': {
                'total_queries': len(query_list),
                'successful_queries': execution_statistics['successful_queries'],
                'failed_queries': execution_statistics['failed_queries'],
                'success_rate': execution_statistics['success_rate'],
                'total_execution_time_ms': query_execution_session['total_execution_time_ms']
            }
        }
        
    except Exception as e:
        logger.error(f"Database query execution failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'partial_results': query_execution_session.get('query_results', []) if 'query_execution_session' in locals() else []
        }

def monitor_database_connections(database_config: Dict[str, Any] = None,
                               monitoring_duration: int = 60) -> Dict[str, Any]:
    """
    Monitored Database Connections
    
    Extrahiert aus kira_routes.py.backup Database Connection Monitoring Logic
    """
    try:
        if database_config is None:
            database_config = _get_default_database_config()
        
        # Initialize connection monitoring session
        monitoring_session = {
            'session_id': f"db_monitor_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'monitoring_duration': monitoring_duration,
            'database_config': database_config,
            'connection_metrics': [],
            'monitoring_statistics': {}
        }
        
        # Monitor connections over time
        start_time = time.time()
        metric_collection_interval = 5  # seconds
        
        while (time.time() - start_time) < monitoring_duration:
            metric_timestamp = datetime.now()
            
            try:
                # Collect connection metrics
                connection_metrics = _collect_connection_metrics(database_config)
                
                connection_metric_snapshot = {
                    'timestamp': metric_timestamp.isoformat(),
                    'elapsed_seconds': time.time() - start_time,
                    'active_connections': connection_metrics.get('active_connections', 0),
                    'connection_pool_size': connection_metrics.get('connection_pool_size', 0),
                    'connection_utilization': connection_metrics.get('connection_utilization', 0.0),
                    'connection_response_time_ms': connection_metrics.get('connection_response_time_ms', 0),
                    'connection_errors': connection_metrics.get('connection_errors', 0)
                }
                
                monitoring_session['connection_metrics'].append(connection_metric_snapshot)
                
            except Exception as e:
                logger.debug(f"Connection metrics collection failed: {e}")
                connection_metric_snapshot = {
                    'timestamp': metric_timestamp.isoformat(),
                    'elapsed_seconds': time.time() - start_time,
                    'error': str(e)
                }
                monitoring_session['connection_metrics'].append(connection_metric_snapshot)
            
            # Wait for next collection
            time.sleep(metric_collection_interval)
        
        # Analyze monitoring data
        monitoring_statistics = _analyze_connection_monitoring_data(monitoring_session['connection_metrics'])
        monitoring_session['monitoring_statistics'] = monitoring_statistics
        
        # Generate connection insights
        connection_insights = _generate_connection_insights(monitoring_statistics)
        
        # Connection optimization recommendations
        connection_recommendations = _generate_connection_recommendations(monitoring_statistics, connection_insights)
        
        monitoring_session.update({
            'end_time': datetime.now().isoformat(),
            'actual_monitoring_duration': time.time() - start_time,
            'metrics_collected': len(monitoring_session['connection_metrics']),
            'connection_insights': connection_insights,
            'connection_recommendations': connection_recommendations
        })
        
        return {
            'success': True,
            'monitoring_session': monitoring_session,
            'monitoring_summary': {
                'monitoring_duration_seconds': monitoring_session['actual_monitoring_duration'],
                'metrics_collected': monitoring_session['metrics_collected'],
                'average_connection_utilization': monitoring_statistics.get('average_utilization', 0.0),
                'peak_connections': monitoring_statistics.get('peak_connections', 0),
                'connection_stability': monitoring_statistics.get('connection_stability', 'unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"Database connection monitoring failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'partial_monitoring_data': monitoring_session.get('connection_metrics', []) if 'monitoring_session' in locals() else []
        }

# ====================================
# PRIVATE HELPER FUNCTIONS - Teil 1
# ====================================

def _get_default_database_config() -> Dict[str, Any]:
    """Holt Default Database Configuration"""
    return {
        'database_type': 'sqlite',
        'database_path': 'kira_data.db',
        'connection_timeout': 30,
        'max_connections': 10,
        'enable_wal_mode': True,
        'enable_foreign_keys': True,
        'cache_size': 2000,
        'synchronous': 'NORMAL'
    }

@contextmanager
def _get_database_connection(database_config: Dict[str, Any]):
    """Context Manager für Database Connection"""
    connection = None
    try:
        # Create connection based on database type
        if database_config.get('database_type') == 'sqlite':
            database_path = database_config.get('database_path', 'kira_data.db')
            connection = sqlite3.connect(
                database_path,
                timeout=database_config.get('connection_timeout', 30),
                check_same_thread=False
            )
            
            # Configure SQLite connection
            connection.execute('PRAGMA foreign_keys = ON')
            if database_config.get('enable_wal_mode', True):
                connection.execute('PRAGMA journal_mode = WAL')
            
            cache_size = database_config.get('cache_size', 2000)
            connection.execute(f'PRAGMA cache_size = {cache_size}')
            
            synchronous = database_config.get('synchronous', 'NORMAL')
            connection.execute(f'PRAGMA synchronous = {synchronous}')
        
        else:
            raise ValueError(f"Unsupported database type: {database_config.get('database_type')}")
        
        yield connection
        
    except Exception as e:
        if connection:
            connection.rollback()
        raise e
    finally:
        if connection:
            connection.close()

def _get_database_status(database_config: Dict[str, Any]) -> Dict[str, Any]:
    """Holt Database Status"""
    try:
        status_result = {
            'status': 'unknown',
            'database_type': database_config.get('database_type'),
            'connection_test': False,
            'database_size_bytes': 0,
            'table_count': 0,
            'last_backup': None
        }
        
        # Test database connection
        try:
            with _get_database_connection(database_config) as connection:
                cursor = connection.cursor()
                
                # Connection test successful
                status_result['connection_test'] = True
                
                # Get database information
                if database_config.get('database_type') == 'sqlite':
                    # Get database file size
                    database_path = database_config.get('database_path', 'kira_data.db')
                    if os.path.exists(database_path):
                        status_result['database_size_bytes'] = os.path.getsize(database_path)
                    
                    # Count tables
                    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                    status_result['table_count'] = cursor.fetchone()[0]
                    
                    # Check for recent backups (this would be more sophisticated in practice)
                    backup_path = f"{database_path}.backup"
                    if os.path.exists(backup_path):
                        backup_mtime = os.path.getmtime(backup_path)
                        status_result['last_backup'] = datetime.fromtimestamp(backup_mtime).isoformat()
                
                status_result['status'] = 'healthy'
                
        except Exception as e:
            status_result['status'] = 'error'
            status_result['connection_error'] = str(e)
        
        return status_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def _initialize_database(database_config: Dict[str, Any], kira_instance=None) -> Dict[str, Any]:
    """Initialisiert Database"""
    try:
        initialization_result = {
            'status': 'success',
            'tables_created': [],
            'indexes_created': [],
            'initial_data_inserted': False,
            'initialization_errors': []
        }
        
        with _get_database_connection(database_config) as connection:
            cursor = connection.cursor()
            
            # Create core tables
            core_tables = _get_core_table_schemas()
            
            for table_name, table_schema in core_tables.items():
                try:
                    cursor.execute(table_schema)
                    initialization_result['tables_created'].append(table_name)
                except Exception as e:
                    initialization_result['initialization_errors'].append(f"Table {table_name}: {str(e)}")
            
            # Create indexes
            core_indexes = _get_core_index_schemas()
            
            for index_name, index_schema in core_indexes.items():
                try:
                    cursor.execute(index_schema)
                    initialization_result['indexes_created'].append(index_name)
                except Exception as e:
                    initialization_result['initialization_errors'].append(f"Index {index_name}: {str(e)}")
            
            # Insert initial data
            try:
                initial_data_queries = _get_initial_data_queries()
                for query in initial_data_queries:
                    cursor.execute(query)
                initialization_result['initial_data_inserted'] = True
            except Exception as e:
                initialization_result['initialization_errors'].append(f"Initial data: {str(e)}")
            
            connection.commit()
        
        # Determine overall status
        if initialization_result['initialization_errors']:
            initialization_result['status'] = 'partial' if initialization_result['tables_created'] else 'failed'
        
        return initialization_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'tables_created': [],
            'indexes_created': []
        }

def _collect_connection_metrics(database_config: Dict[str, Any]) -> Dict[str, Any]:
    """Sammelt Connection Metrics"""
    try:
        metrics = {
            'active_connections': 0,
            'connection_pool_size': database_config.get('max_connections', 10),
            'connection_utilization': 0.0,
            'connection_response_time_ms': 0,
            'connection_errors': 0
        }
        
        # Test connection response time
        start_time = time.time()
        try:
            with _get_database_connection(database_config) as connection:
                cursor = connection.cursor()
                cursor.execute('SELECT 1')
                cursor.fetchone()
            
            response_time_ms = (time.time() - start_time) * 1000
            metrics['connection_response_time_ms'] = response_time_ms
            
        except Exception as e:
            metrics['connection_errors'] = 1
            metrics['connection_response_time_ms'] = (time.time() - start_time) * 1000
        
        # Calculate utilization (simplified)
        with _connection_lock:
            active_connections = len(_connection_pool)
            metrics['active_connections'] = active_connections
            metrics['connection_utilization'] = active_connections / metrics['connection_pool_size']
        
        return metrics
        
    except Exception as e:
        return {
            'active_connections': 0,
            'connection_pool_size': 0,
            'connection_utilization': 0.0,
            'connection_response_time_ms': 0,
            'connection_errors': 1,
            'error': str(e)
        }

def _compile_query_execution_statistics(query_results: List[Dict]) -> Dict[str, Any]:
    """Kompiliert Query Execution Statistics"""
    try:
        if not query_results:
            return {
                'successful_queries': 0,
                'failed_queries': 0,
                'success_rate': 0.0,
                'average_execution_time_ms': 0,
                'total_rows_affected': 0
            }
        
        successful_queries = len([r for r in query_results if r.get('status') == 'success'])
        failed_queries = len([r for r in query_results if r.get('status') == 'error'])
        total_queries = len(query_results)
        
        success_rate = successful_queries / total_queries if total_queries > 0 else 0.0
        
        execution_times = [r.get('execution_time_ms', 0) for r in query_results]
        average_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        total_rows_affected = sum(r.get('rows_affected', 0) for r in query_results)
        
        return {
            'successful_queries': successful_queries,
            'failed_queries': failed_queries,
            'success_rate': success_rate,
            'average_execution_time_ms': average_execution_time,
            'total_rows_affected': total_rows_affected,
            'min_execution_time_ms': min(execution_times) if execution_times else 0,
            'max_execution_time_ms': max(execution_times) if execution_times else 0
        }
        
    except Exception as e:
        logger.debug(f"Query execution statistics compilation failed: {e}")
        return {
            'successful_queries': 0,
            'failed_queries': len(query_results),
            'success_rate': 0.0,
            'average_execution_time_ms': 0,
            'total_rows_affected': 0,
            'error': str(e)
        }

def _calculate_operation_duration(operation_session: Dict) -> float:
    """Berechnet Operation Duration in Milliseconds"""
    try:
        start_time = datetime.fromisoformat(operation_session['start_time'])
        end_time = datetime.fromisoformat(operation_session['end_time'])
        duration = (end_time - start_time).total_seconds() * 1000
        return duration
    except Exception as e:
        logger.debug(f"Operation duration calculation failed: {e}")
        return 0.0

def _generate_fallback_database_status() -> Dict[str, Any]:
    """Generiert Fallback Database Status"""
    return {
        'fallback_mode': True,
        'database_status': {
            'status': 'unknown',
            'database_type': 'sqlite',
            'connection_test': False,
            'database_size_bytes': 0,
            'table_count': 0
        },
        'connection_metrics': {
            'active_connections': 0,
            'connection_pool_size': 10,
            'connection_utilization': 0.0
        }
    }

__all__ = [
    'manage_database_operations',
    'execute_database_queries', 
    'monitor_database_connections'
]
def handle_database_schema(kira_instance=None,
                          schema_operation: str = 'validate',
                          schema_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Handled Database Schema Operations
    
    Extrahiert aus kira_routes.py.backup Schema Management Logic
    """
    try:
        if schema_config is None:
            schema_config = _get_default_schema_config()
        
        # Initialize schema operation session
        schema_session = {
            'session_id': f"schema_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'schema_operation': schema_operation,
            'schema_config': schema_config,
            'schema_results': {}
        }
        
        # Handle different schema operations
        if schema_operation == 'validate':
            schema_session['schema_results'] = _validate_database_schema(schema_config)
        
        elif schema_operation == 'create':
            schema_session['schema_results'] = _create_database_schema(schema_config, kira_instance)
        
        elif schema_operation == 'migrate':
            schema_session['schema_results'] = _migrate_database_schema(schema_config)
        
        elif schema_operation == 'backup_schema':
            schema_session['schema_results'] = _backup_database_schema(schema_config)
        
        elif schema_operation == 'restore_schema':
            schema_session['schema_results'] = _restore_database_schema(schema_config)
        
        elif schema_operation == 'compare':
            schema_session['schema_results'] = _compare_database_schemas(schema_config)
        
        elif schema_operation == 'analyze':
            schema_session['schema_results'] = _analyze_database_schema(schema_config)
        
        else:
            schema_session['schema_results'] = {
                'status': 'error',
                'error': f'Unsupported schema operation: {schema_operation}'
            }
        
        # Add schema metadata
        schema_session.update({
            'end_time': datetime.now().isoformat(),
            'schema_success': schema_session['schema_results'].get('status') == 'success',
            'schema_duration': _calculate_operation_duration(schema_session)
        })
        
        return {
            'success': schema_session['schema_success'],
            'schema_session': schema_session,
            'schema_summary': {
                'schema_operation': schema_operation,
                'schema_success': schema_session['schema_success'],
                'schema_duration_ms': schema_session['schema_duration'],
                'schema_status': schema_session['schema_results'].get('schema_status', 'unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"Database schema handling failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'fallback_schema_status': _generate_fallback_schema_status()
        }

# ====================================
# ADDITIONAL HELPER FUNCTIONS - Teil 2
# ====================================

def _get_core_table_schemas() -> Dict[str, str]:
    """Holt Core Table Schemas"""
    return {
        'kira_memories': '''
            CREATE TABLE IF NOT EXISTS kira_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_type VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                context TEXT,
                importance_score REAL DEFAULT 0.5,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                tags TEXT,
                metadata TEXT
            )
        ''',
        
        'kira_conversations': '''
            CREATE TABLE IF NOT EXISTS kira_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id VARCHAR(100) UNIQUE NOT NULL,
                user_input TEXT NOT NULL,
                kira_response TEXT NOT NULL,
                context_data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                response_time_ms INTEGER,
                satisfaction_score REAL,
                metadata TEXT
            )
        ''',
        
        'kira_system_logs': '''
            CREATE TABLE IF NOT EXISTS kira_system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_level VARCHAR(20) NOT NULL,
                module VARCHAR(100),
                message TEXT NOT NULL,
                error_details TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id VARCHAR(100),
                user_id VARCHAR(100),
                metadata TEXT
            )
        ''',
        
        'kira_performance_metrics': '''
            CREATE TABLE IF NOT EXISTS kira_performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_type VARCHAR(50) NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit VARCHAR(20),
                component VARCHAR(100),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                context TEXT,
                metadata TEXT
            )
        ''',
        
        'kira_user_sessions': '''
            CREATE TABLE IF NOT EXISTS kira_user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id VARCHAR(100) UNIQUE NOT NULL,
                user_id VARCHAR(100),
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                session_duration INTEGER,
                interactions_count INTEGER DEFAULT 0,
                session_quality REAL,
                metadata TEXT
            )
        ''',
        
        'kira_knowledge_base': '''
            CREATE TABLE IF NOT EXISTS kira_knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic VARCHAR(200) NOT NULL,
                content TEXT NOT NULL,
                source VARCHAR(200),
                reliability_score REAL DEFAULT 0.5,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                version INTEGER DEFAULT 1,
                tags TEXT,
                metadata TEXT
            )
        '''
    }

def _get_core_index_schemas() -> Dict[str, str]:
    """Holt Core Index Schemas"""
    return {
        'idx_memories_type_timestamp': '''
            CREATE INDEX IF NOT EXISTS idx_memories_type_timestamp 
            ON kira_memories(memory_type, created_timestamp)
        ''',
        
        'idx_memories_importance': '''
            CREATE INDEX IF NOT EXISTS idx_memories_importance 
            ON kira_memories(importance_score DESC)
        ''',
        
        'idx_conversations_timestamp': '''
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp 
            ON kira_conversations(timestamp DESC)
        ''',
        
        'idx_conversations_id': '''
            CREATE INDEX IF NOT EXISTS idx_conversations_id 
            ON kira_conversations(conversation_id)
        ''',
        
        'idx_system_logs_timestamp': '''
            CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp 
            ON kira_system_logs(timestamp DESC)
        ''',
        
        'idx_system_logs_level': '''
            CREATE INDEX IF NOT EXISTS idx_system_logs_level 
            ON kira_system_logs(log_level, timestamp DESC)
        ''',
        
        'idx_performance_metrics_type': '''
            CREATE INDEX IF NOT EXISTS idx_performance_metrics_type 
            ON kira_performance_metrics(metric_type, timestamp DESC)
        ''',
        
        'idx_sessions_user_time': '''
            CREATE INDEX IF NOT EXISTS idx_sessions_user_time 
            ON kira_user_sessions(user_id, start_time DESC)
        ''',
        
        'idx_knowledge_topic': '''
            CREATE INDEX IF NOT EXISTS idx_knowledge_topic 
            ON kira_knowledge_base(topic, reliability_score DESC)
        '''
    }

def _get_initial_data_queries() -> List[str]:
    """Holt Initial Data Queries"""
    return [
        '''
        INSERT OR IGNORE INTO kira_system_logs (log_level, module, message, session_id) 
        VALUES ('INFO', 'database', 'Database initialized successfully', 'init_session')
        ''',
        
        '''
        INSERT OR IGNORE INTO kira_knowledge_base (topic, content, source, reliability_score) 
        VALUES ('system_initialization', 'Kira database system has been initialized', 'internal', 1.0)
        ''',
        
        '''
        INSERT OR IGNORE INTO kira_performance_metrics (metric_type, metric_value, metric_unit, component) 
        VALUES ('initialization_time', 0, 'seconds', 'database_system')
        '''
    ]

def _perform_database_maintenance(database_config: Dict[str, Any]) -> Dict[str, Any]:
    """Führt Database Maintenance durch"""
    try:
        maintenance_result = {
            'status': 'success',
            'maintenance_tasks': [],
            'maintenance_errors': [],
            'performance_improvements': {}
        }
        
        with _get_database_connection(database_config) as connection:
            cursor = connection.cursor()
            
            # Task 1: VACUUM database
            try:
                cursor.execute('VACUUM')
                maintenance_result['maintenance_tasks'].append('vacuum_completed')
            except Exception as e:
                maintenance_result['maintenance_errors'].append(f"Vacuum failed: {str(e)}")
            
            # Task 2: Analyze tables
            try:
                cursor.execute('ANALYZE')
                maintenance_result['maintenance_tasks'].append('analyze_completed')
            except Exception as e:
                maintenance_result['maintenance_errors'].append(f"Analyze failed: {str(e)}")
            
            # Task 3: Rebuild indexes
            try:
                cursor.execute('REINDEX')
                maintenance_result['maintenance_tasks'].append('reindex_completed')
            except Exception as e:
                maintenance_result['maintenance_errors'].append(f"Reindex failed: {str(e)}")
            
            # Task 4: Update statistics
            try:
                cursor.execute('PRAGMA optimize')
                maintenance_result['maintenance_tasks'].append('optimize_completed')
            except Exception as e:
                maintenance_result['maintenance_errors'].append(f"Optimize failed: {str(e)}")
            
            # Task 5: Clean old logs (keep last 30 days)
            try:
                cleanup_date = (datetime.now() - timedelta(days=30)).isoformat()
                cursor.execute('''
                    DELETE FROM kira_system_logs 
                    WHERE timestamp < ? AND log_level != 'ERROR'
                ''', (cleanup_date,))
                
                rows_deleted = cursor.rowcount
                maintenance_result['maintenance_tasks'].append(f'log_cleanup_completed_{rows_deleted}_rows')
            except Exception as e:
                maintenance_result['maintenance_errors'].append(f"Log cleanup failed: {str(e)}")
            
            connection.commit()
        
        # Performance improvements assessment
        maintenance_result['performance_improvements'] = {
            'database_size_optimized': 'vacuum_completed' in maintenance_result['maintenance_tasks'],
            'query_performance_optimized': 'analyze_completed' in maintenance_result['maintenance_tasks'],
            'index_performance_optimized': 'reindex_completed' in maintenance_result['maintenance_tasks'],
            'maintenance_success_rate': len(maintenance_result['maintenance_tasks']) / (len(maintenance_result['maintenance_tasks']) + len(maintenance_result['maintenance_errors']))
        }
        
        # Determine overall status
        if maintenance_result['maintenance_errors']:
            maintenance_result['status'] = 'partial' if maintenance_result['maintenance_tasks'] else 'failed'
        
        return maintenance_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'maintenance_tasks': [],
            'maintenance_errors': [str(e)]
        }

def _create_database_backup(database_config: Dict[str, Any]) -> Dict[str, Any]:
    """Erstellt Database Backup"""
    try:
        backup_result = {
            'status': 'success',
            'backup_path': None,
            'backup_size_bytes': 0,
            'backup_timestamp': datetime.now().isoformat(),
            'backup_errors': []
        }
        
        if database_config.get('database_type') == 'sqlite':
            database_path = database_config.get('database_path', 'kira_data.db')
            backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{database_path}.backup_{backup_timestamp}"
            
            try:
                # Create backup using SQLite backup API
                with _get_database_connection(database_config) as source_conn:
                    backup_conn = sqlite3.connect(backup_path)
                    
                    try:
                        source_conn.backup(backup_conn)
                        backup_result['backup_path'] = backup_path
                        
                        # Get backup file size
                        if os.path.exists(backup_path):
                            backup_result['backup_size_bytes'] = os.path.getsize(backup_path)
                        
                    finally:
                        backup_conn.close()
                
            except Exception as e:
                backup_result['backup_errors'].append(f"Backup creation failed: {str(e)}")
                backup_result['status'] = 'failed'
        
        else:
            backup_result['backup_errors'].append(f"Backup not supported for database type: {database_config.get('database_type')}")
            backup_result['status'] = 'failed'
        
        return backup_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'backup_path': None,
            'backup_errors': [str(e)]
        }

def _optimize_database_performance(database_config: Dict[str, Any]) -> Dict[str, Any]:
    """Optimiert Database Performance"""
    try:
        optimization_result = {
            'status': 'success',
            'optimizations_applied': [],
            'optimization_errors': [],
            'performance_metrics': {}
        }
        
        with _get_database_connection(database_config) as connection:
            cursor = connection.cursor()
            
            # Optimization 1: Increase cache size
            try:
                cache_size = database_config.get('cache_size', 4000)  # Increase default
                cursor.execute(f'PRAGMA cache_size = {cache_size}')
                optimization_result['optimizations_applied'].append(f'cache_size_set_{cache_size}')
            except Exception as e:
                optimization_result['optimization_errors'].append(f"Cache size optimization failed: {str(e)}")
            
            # Optimization 2: Set optimal synchronous mode
            try:
                cursor.execute('PRAGMA synchronous = NORMAL')
                optimization_result['optimizations_applied'].append('synchronous_mode_optimized')
            except Exception as e:
                optimization_result['optimization_errors'].append(f"Synchronous mode optimization failed: {str(e)}")
            
            # Optimization 3: Enable memory mapping
            try:
                cursor.execute('PRAGMA mmap_size = 268435456')  # 256MB
                optimization_result['optimizations_applied'].append('memory_mapping_enabled')
            except Exception as e:
                optimization_result['optimization_errors'].append(f"Memory mapping optimization failed: {str(e)}")
            
            # Optimization 4: Set temp store to memory
            try:
                cursor.execute('PRAGMA temp_store = MEMORY')
                optimization_result['optimizations_applied'].append('temp_store_memory')
            except Exception as e:
                optimization_result['optimization_errors'].append(f"Temp store optimization failed: {str(e)}")
            
            # Collect performance metrics after optimization
            try:
                optimization_result['performance_metrics'] = _collect_database_performance_metrics(cursor)
            except Exception as e:
                optimization_result['optimization_errors'].append(f"Performance metrics collection failed: {str(e)}")
        
        # Determine overall status
        if optimization_result['optimization_errors']:
            optimization_result['status'] = 'partial' if optimization_result['optimizations_applied'] else 'failed'
        
        return optimization_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'optimizations_applied': [],
            'optimization_errors': [str(e)]
        }

def _validate_database_integrity(database_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validiert Database Integrity"""
    try:
        integrity_result = {
            'status': 'success',
            'integrity_checks': [],
            'integrity_errors': [],
            'corruption_detected': False
        }
        
        with _get_database_connection(database_config) as connection:
            cursor = connection.cursor()
            
            # Check 1: PRAGMA integrity_check
            try:
                cursor.execute('PRAGMA integrity_check')
                integrity_check_result = cursor.fetchall()
                
                if integrity_check_result and integrity_check_result[0][0] == 'ok':
                    integrity_result['integrity_checks'].append('integrity_check_passed')
                else:
                    integrity_result['integrity_errors'].append(f"Integrity check failed: {integrity_check_result}")
                    integrity_result['corruption_detected'] = True
            except Exception as e:
                integrity_result['integrity_errors'].append(f"Integrity check failed: {str(e)}")
            
            # Check 2: PRAGMA quick_check
            try:
                cursor.execute('PRAGMA quick_check')
                quick_check_result = cursor.fetchall()
                
                if quick_check_result and quick_check_result[0][0] == 'ok':
                    integrity_result['integrity_checks'].append('quick_check_passed')
                else:
                    integrity_result['integrity_errors'].append(f"Quick check failed: {quick_check_result}")
                    integrity_result['corruption_detected'] = True
            except Exception as e:
                integrity_result['integrity_errors'].append(f"Quick check failed: {str(e)}")
            
            # Check 3: Foreign key constraints
            try:
                cursor.execute('PRAGMA foreign_key_check')
                fk_violations = cursor.fetchall()
                
                if not fk_violations:
                    integrity_result['integrity_checks'].append('foreign_key_check_passed')
                else:
                    integrity_result['integrity_errors'].append(f"Foreign key violations: {len(fk_violations)}")
            except Exception as e:
                integrity_result['integrity_errors'].append(f"Foreign key check failed: {str(e)}")
            
            # Check 4: Table structure validation
            try:
                expected_tables = list(_get_core_table_schemas().keys())
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = [row[0] for row in cursor.fetchall()]
                
                missing_tables = set(expected_tables) - set(existing_tables)
                if not missing_tables:
                    integrity_result['integrity_checks'].append('table_structure_validated')
                else:
                    integrity_result['integrity_errors'].append(f"Missing tables: {missing_tables}")
            except Exception as e:
                integrity_result['integrity_errors'].append(f"Table structure validation failed: {str(e)}")
        
        # Determine overall status
        if integrity_result['integrity_errors'] or integrity_result['corruption_detected']:
            integrity_result['status'] = 'failed' if integrity_result['corruption_detected'] else 'warnings'
        
        return integrity_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'integrity_checks': [],
            'integrity_errors': [str(e)],
            'corruption_detected': False
        }

def _collect_database_performance_metrics(cursor) -> Dict[str, Any]:
    """Sammelt Database Performance Metrics"""
    try:
        performance_metrics = {}
        
        # Cache hit ratio
        cursor.execute('PRAGMA cache_size')
        cache_size = cursor.fetchone()[0]
        performance_metrics['cache_size'] = cache_size
        
        # Page count and size
        cursor.execute('PRAGMA page_count')
        page_count = cursor.fetchone()[0]
        performance_metrics['page_count'] = page_count
        
        cursor.execute('PRAGMA page_size')
        page_size = cursor.fetchone()[0]
        performance_metrics['page_size'] = page_size
        
        performance_metrics['estimated_size_bytes'] = page_count * page_size
        
        # Journal mode
        cursor.execute('PRAGMA journal_mode')
        journal_mode = cursor.fetchone()[0]
        performance_metrics['journal_mode'] = journal_mode
        
        # Synchronous mode
        cursor.execute('PRAGMA synchronous')
        synchronous_mode = cursor.fetchone()[0]
        performance_metrics['synchronous_mode'] = synchronous_mode
        
        return performance_metrics
        
    except Exception as e:
        logger.debug(f"Database performance metrics collection failed: {e}")
        return {'error': str(e)}

def _analyze_connection_monitoring_data(connection_metrics: List[Dict]) -> Dict[str, Any]:
    """Analysiert Connection Monitoring Data"""
    try:
        if not connection_metrics:
            return {
                'average_utilization': 0.0,
                'peak_connections': 0,
                'connection_stability': 'unknown',
                'error_rate': 1.0
            }
        
        # Filter out error entries
        valid_metrics = [m for m in connection_metrics if 'error' not in m]
        
        if not valid_metrics:
            return {
                'average_utilization': 0.0,
                'peak_connections': 0,
                'connection_stability': 'unstable',
                'error_rate': 1.0
            }
        
        # Calculate statistics
        utilizations = [m.get('connection_utilization', 0.0) for m in valid_metrics]
        active_connections = [m.get('active_connections', 0) for m in valid_metrics]
        response_times = [m.get('connection_response_time_ms', 0) for m in valid_metrics]
        
        average_utilization = sum(utilizations) / len(utilizations) if utilizations else 0.0
        peak_connections = max(active_connections) if active_connections else 0
        average_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Determine stability
        error_count = len([m for m in connection_metrics if 'error' in m])
        error_rate = error_count / len(connection_metrics) if connection_metrics else 1.0
        
        if error_rate < 0.1 and average_response_time < 100:
            stability = 'stable'
        elif error_rate < 0.3:
            stability = 'moderate'
        else:
            stability = 'unstable'
        
        return {
            'average_utilization': average_utilization,
            'peak_connections': peak_connections,
            'connection_stability': stability,
            'error_rate': error_rate,
            'average_response_time_ms': average_response_time,
            'total_metrics_collected': len(connection_metrics),
            'valid_metrics_count': len(valid_metrics)
        }
        
    except Exception as e:
        logger.debug(f"Connection monitoring data analysis failed: {e}")
        return {
            'average_utilization': 0.0,
            'peak_connections': 0,
            'connection_stability': 'unknown',
            'error_rate': 1.0,
            'analysis_error': str(e)
        }

def _generate_connection_insights(monitoring_statistics: Dict) -> Dict[str, Any]:
    """Generiert Connection Insights"""
    try:
        insights = {
            'performance_assessment': 'unknown',
            'optimization_opportunities': [],
            'connection_health': 'unknown',
            'recommendations': []
        }
        
        # Performance assessment
        avg_utilization = monitoring_statistics.get('average_utilization', 0.0)
        error_rate = monitoring_statistics.get('error_rate', 1.0)
        avg_response_time = monitoring_statistics.get('average_response_time_ms', 0)
        
        if error_rate < 0.05 and avg_response_time < 50:
            insights['performance_assessment'] = 'excellent'
        elif error_rate < 0.15 and avg_response_time < 200:
            insights['performance_assessment'] = 'good'
        elif error_rate < 0.30:
            insights['performance_assessment'] = 'fair'
        else:
            insights['performance_assessment'] = 'poor'
        
        # Optimization opportunities
        if avg_utilization > 0.8:
            insights['optimization_opportunities'].append('increase_connection_pool_size')
        
        if avg_response_time > 100:
            insights['optimization_opportunities'].append('optimize_connection_performance')
        
        if error_rate > 0.1:
            insights['optimization_opportunities'].append('investigate_connection_errors')
        
        # Connection health
        stability = monitoring_statistics.get('connection_stability', 'unknown')
        if stability == 'stable' and error_rate < 0.1:
            insights['connection_health'] = 'healthy'
        elif stability == 'moderate':
            insights['connection_health'] = 'moderate'
        else:
            insights['connection_health'] = 'unhealthy'
        
        # Recommendations
        if avg_utilization < 0.2:
            insights['recommendations'].append('Consider reducing connection pool size')
        elif avg_utilization > 0.8:
            insights['recommendations'].append('Consider increasing connection pool size')
        
        if avg_response_time > 200:
            insights['recommendations'].append('Investigate database performance issues')
        
        if error_rate > 0.2:
            insights['recommendations'].append('Investigate and resolve connection errors')
        
        return insights
        
    except Exception as e:
        return {
            'performance_assessment': 'unknown',
            'optimization_opportunities': [],
            'connection_health': 'unknown',
            'recommendations': [],
            'insights_error': str(e)
        }

def _generate_connection_recommendations(monitoring_statistics: Dict, connection_insights: Dict) -> Dict[str, Any]:
    """Generiert Connection Recommendations"""
    try:
        recommendations = {
            'immediate_actions': [],
            'performance_optimizations': [],
            'monitoring_adjustments': [],
            'configuration_changes': []
        }
        
        error_rate = monitoring_statistics.get('error_rate', 0.0)
        avg_utilization = monitoring_statistics.get('average_utilization', 0.0)
        avg_response_time = monitoring_statistics.get('average_response_time_ms', 0)
        
        # Immediate actions
        if error_rate > 0.3:
            recommendations['immediate_actions'].append('Investigate critical connection errors')
        
        if avg_response_time > 1000:
            recommendations['immediate_actions'].append('Check database server status')
        
        # Performance optimizations
        if avg_utilization > 0.9:
            recommendations['performance_optimizations'].append('Increase maximum connection pool size')
        
        if avg_response_time > 200:
            recommendations['performance_optimizations'].append('Enable connection pooling optimization')
            recommendations['performance_optimizations'].append('Review database query performance')
        
        # Monitoring adjustments
        if monitoring_statistics.get('valid_metrics_count', 0) < monitoring_statistics.get('total_metrics_collected', 1) * 0.8:
            recommendations['monitoring_adjustments'].append('Improve monitoring reliability')
        
        # Configuration changes
        connection_health = connection_insights.get('connection_health', 'unknown')
        if connection_health == 'unhealthy':
            recommendations['configuration_changes'].append('Review database configuration')
            recommendations['configuration_changes'].append('Consider connection timeout adjustments')
        
        return recommendations
        
    except Exception as e:
        return {
            'immediate_actions': [],
            'performance_optimizations': [],
            'monitoring_adjustments': [],
            'configuration_changes': [],
            'recommendations_error': str(e)
        }

def _get_default_schema_config() -> Dict[str, Any]:
    """Holt Default Schema Configuration"""
    return {
        'schema_version': '1.0',
        'auto_migrate': True,
        'backup_before_migration': True,
        'validate_after_migration': True,
        'migration_timeout_seconds': 300
    }

def _generate_fallback_schema_status() -> Dict[str, Any]:
    """Generiert Fallback Schema Status"""
    return {
        'fallback_mode': True,
        'schema_status': {
            'schema_version': 'unknown',
            'schema_valid': False,
            'tables_count': 0,
            'indexes_count': 0
        }
    }

def _validate_database_schema(schema_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validiert Database Schema"""
    try:
        validation_result = {
            'status': 'success',
            'schema_version': schema_config.get('schema_version', 'unknown'),
            'validation_checks': [],
            'validation_errors': [],
            'schema_health': 'unknown'
        }
        
        database_config = _get_default_database_config()
        
        with _get_database_connection(database_config) as connection:
            cursor = connection.cursor()
            
            # Check 1: Verify all expected tables exist
            try:
                expected_tables = list(_get_core_table_schemas().keys())
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                existing_tables = [row[0] for row in cursor.fetchall()]
                
                missing_tables = set(expected_tables) - set(existing_tables)
                extra_tables = set(existing_tables) - set(expected_tables)
                
                if not missing_tables and not extra_tables:
                    validation_result['validation_checks'].append('table_structure_complete')
                else:
                    if missing_tables:
                        validation_result['validation_errors'].append(f"Missing tables: {list(missing_tables)}")
                    if extra_tables:
                        validation_result['validation_errors'].append(f"Unexpected tables: {list(extra_tables)}")
                
            except Exception as e:
                validation_result['validation_errors'].append(f"Table structure validation failed: {str(e)}")
            
            # Check 2: Verify all expected indexes exist
            try:
                expected_indexes = list(_get_core_index_schemas().keys())
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
                existing_indexes = [row[0] for row in cursor.fetchall()]
                
                missing_indexes = set(expected_indexes) - set(existing_indexes)
                
                if not missing_indexes:
                    validation_result['validation_checks'].append('index_structure_complete')
                else:
                    validation_result['validation_errors'].append(f"Missing indexes: {list(missing_indexes)}")
                
            except Exception as e:
                validation_result['validation_errors'].append(f"Index structure validation failed: {str(e)}")
            
            # Check 3: Validate table schemas
            try:
                schema_validation_errors = []
                
                for table_name in _get_core_table_schemas().keys():
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    table_info = cursor.fetchall()
                    
                    if not table_info:
                        schema_validation_errors.append(f"Table {table_name} has no schema information")
                    else:
                        # Basic validation - table has columns
                        column_count = len(table_info)
                        if column_count < 3:  # Minimum expected columns
                            schema_validation_errors.append(f"Table {table_name} has too few columns ({column_count})")
                
                if not schema_validation_errors:
                    validation_result['validation_checks'].append('table_schemas_valid')
                else:
                    validation_result['validation_errors'].extend(schema_validation_errors)
                
            except Exception as e:
                validation_result['validation_errors'].append(f"Table schema validation failed: {str(e)}")
            
            # Check 4: Data integrity checks
            try:
                integrity_issues = []
                
                # Check for orphaned records (simplified check)
                cursor.execute("SELECT COUNT(*) FROM kira_conversations WHERE conversation_id IS NULL OR conversation_id = ''")
                null_conversation_ids = cursor.fetchone()[0]
                
                if null_conversation_ids > 0:
                    integrity_issues.append(f"Found {null_conversation_ids} conversations with null/empty IDs")
                
                cursor.execute("SELECT COUNT(*) FROM kira_memories WHERE memory_type IS NULL OR memory_type = ''")
                null_memory_types = cursor.fetchone()[0]
                
                if null_memory_types > 0:
                    integrity_issues.append(f"Found {null_memory_types} memories with null/empty types")
                
                if not integrity_issues:
                    validation_result['validation_checks'].append('data_integrity_valid')
                else:
                    validation_result['validation_errors'].extend(integrity_issues)
                
            except Exception as e:
                validation_result['validation_errors'].append(f"Data integrity validation failed: {str(e)}")
        
        # Determine schema health
        if not validation_result['validation_errors']:
            validation_result['schema_health'] = 'excellent'
        elif len(validation_result['validation_errors']) <= 2:
            validation_result['schema_health'] = 'good'
        elif len(validation_result['validation_errors']) <= 5:
            validation_result['schema_health'] = 'fair'
        else:
            validation_result['schema_health'] = 'poor'
        
        # Overall status
        if validation_result['validation_errors']:
            validation_result['status'] = 'warnings' if validation_result['schema_health'] in ['good', 'fair'] else 'failed'
        
        validation_result['schema_status'] = validation_result['schema_health']
        
        return validation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'schema_version': 'unknown',
            'validation_checks': [],
            'validation_errors': [str(e)],
            'schema_health': 'error'
        }

def _create_database_schema(schema_config: Dict[str, Any], kira_instance=None) -> Dict[str, Any]:
    """Erstellt Database Schema"""
    try:
        creation_result = {
            'status': 'success',
            'schema_version': schema_config.get('schema_version', '1.0'),
            'creation_steps': [],
            'creation_errors': [],
            'tables_created': 0,
            'indexes_created': 0
        }
        
        database_config = _get_default_database_config()
        
        with _get_database_connection(database_config) as connection:
            cursor = connection.cursor()
            
            try:
                # Step 1: Create tables
                creation_result['creation_steps'].append('creating_tables')
                
                core_tables = _get_core_table_schemas()
                for table_name, table_schema in core_tables.items():
                    try:
                        cursor.execute(table_schema)
                        creation_result['tables_created'] += 1
                        logger.info(f"Created table: {table_name}")
                    except Exception as e:
                        creation_result['creation_errors'].append(f"Table creation failed for {table_name}: {str(e)}")
                
                # Step 2: Create indexes
                creation_result['creation_steps'].append('creating_indexes')
                
                core_indexes = _get_core_index_schemas()
                for index_name, index_schema in core_indexes.items():
                    try:
                        cursor.execute(index_schema)
                        creation_result['indexes_created'] += 1
                        logger.info(f"Created index: {index_name}")
                    except Exception as e:
                        creation_result['creation_errors'].append(f"Index creation failed for {index_name}: {str(e)}")
                
                # Step 3: Insert initial data
                creation_result['creation_steps'].append('inserting_initial_data')
                
                initial_data_queries = _get_initial_data_queries()
                for query in initial_data_queries:
                    try:
                        cursor.execute(query)
                    except Exception as e:
                        creation_result['creation_errors'].append(f"Initial data insertion failed: {str(e)}")
                
                # Step 4: Create schema version table and record
                creation_result['creation_steps'].append('recording_schema_version')
                
                try:
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS schema_versions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            version VARCHAR(20) NOT NULL,
                            created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            description TEXT
                        )
                    ''')
                    
                    cursor.execute('''
                        INSERT INTO schema_versions (version, description) 
                        VALUES (?, 'Initial schema creation')
                    ''', (creation_result['schema_version'],))
                    
                except Exception as e:
                    creation_result['creation_errors'].append(f"Schema version recording failed: {str(e)}")
                
                connection.commit()
                creation_result['creation_steps'].append('schema_creation_completed')
                
            except Exception as e:
                connection.rollback()
                creation_result['creation_errors'].append(f"Schema creation transaction failed: {str(e)}")
                creation_result['status'] = 'failed'
        
        # Determine final status
        if creation_result['creation_errors']:
            creation_result['status'] = 'partial' if creation_result['tables_created'] > 0 else 'failed'
        
        return creation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'schema_version': 'unknown',
            'creation_steps': [],
            'creation_errors': [str(e)],
            'tables_created': 0,
            'indexes_created': 0
        }

def _migrate_database_schema(schema_config: Dict[str, Any]) -> Dict[str, Any]:
    """Migriert Database Schema"""
    try:
        migration_result = {
            'status': 'success',
            'source_version': 'unknown',
            'target_version': schema_config.get('schema_version', '1.0'),
            'migration_steps': [],
            'migration_errors': [],
            'migrations_applied': 0
        }
        
        database_config = _get_default_database_config()
        
        with _get_database_connection(database_config) as connection:
            cursor = connection.cursor()
            
            # Step 1: Get current schema version
            try:
                cursor.execute('''
                    SELECT version FROM schema_versions 
                    ORDER BY created_timestamp DESC 
                    LIMIT 1
                ''')
                
                current_version_result = cursor.fetchone()
                if current_version_result:
                    migration_result['source_version'] = current_version_result[0]
                else:
                    migration_result['source_version'] = '0.0'  # No version found
                
                migration_result['migration_steps'].append(f'detected_current_version_{migration_result["source_version"]}')
                
            except Exception as e:
                # Schema version table doesn't exist - treat as new installation
                migration_result['source_version'] = '0.0'
                migration_result['migration_steps'].append('no_version_table_found')
            
            # Step 2: Determine if migration is needed
            if migration_result['source_version'] == migration_result['target_version']:
                migration_result['migration_steps'].append('no_migration_needed')
                return migration_result
            
            # Step 3: Create backup if requested
            if schema_config.get('backup_before_migration', True):
                try:
                    backup_result = _create_database_backup(database_config)
                    if backup_result.get('status') == 'success':
                        migration_result['migration_steps'].append('backup_created')
                    else:
                        migration_result['migration_errors'].append('Backup creation failed')
                except Exception as e:
                    migration_result['migration_errors'].append(f"Backup creation error: {str(e)}")
            
            # Step 4: Apply migrations
            try:
                migrations_to_apply = _get_available_migrations(migration_result['source_version'], migration_result['target_version'])
                
                for migration in migrations_to_apply:
                    try:
                        migration_result['migration_steps'].append(f'applying_migration_{migration["version"]}')
                        
                        # Execute migration queries
                        for query in migration['queries']:
                            cursor.execute(query)
                        
                        # Record migration
                        cursor.execute('''
                            INSERT INTO schema_versions (version, description) 
                            VALUES (?, ?)
                        ''', (migration['version'], migration.get('description', 'Schema migration')))
                        
                        migration_result['migrations_applied'] += 1
                        logger.info(f"Applied migration to version {migration['version']}")
                        
                    except Exception as e:
                        migration_result['migration_errors'].append(f"Migration {migration['version']} failed: {str(e)}")
                        break  # Stop on first migration failure
                
                connection.commit()
                migration_result['migration_steps'].append('migrations_committed')
                
            except Exception as e:
                connection.rollback()
                migration_result['migration_errors'].append(f"Migration transaction failed: {str(e)}")
                migration_result['status'] = 'failed'
            
            # Step 5: Validate after migration if requested
            if schema_config.get('validate_after_migration', True) and migration_result['status'] == 'success':
                try:
                    validation_result = _validate_database_schema(schema_config)
                    if validation_result['status'] != 'success':
                        migration_result['migration_errors'].append('Post-migration validation failed')
                        migration_result['status'] = 'partial'
                    else:
                        migration_result['migration_steps'].append('post_migration_validation_passed')
                except Exception as e:
                    migration_result['migration_errors'].append(f"Post-migration validation error: {str(e)}")
        
        # Determine final status
        if migration_result['migration_errors']:
            migration_result['status'] = 'partial' if migration_result['migrations_applied'] > 0 else 'failed'
        
        return migration_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'source_version': 'unknown',
            'target_version': 'unknown',
            'migration_steps': [],
            'migration_errors': [str(e)],
            'migrations_applied': 0
        }

def _backup_database_schema(schema_config: Dict[str, Any]) -> Dict[str, Any]:
    """Erstellt Schema Backup"""
    try:
        schema_backup_result = {
            'status': 'success',
            'backup_path': None,
            'backup_timestamp': datetime.now().isoformat(),
            'schema_dump': {},
            'backup_errors': []
        }
        
        database_config = _get_default_database_config()
        
        with _get_database_connection(database_config) as connection:
            cursor = connection.cursor()
            
            # Extract table schemas
            try:
                cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables = cursor.fetchall()
                
                schema_backup_result['schema_dump']['tables'] = {}
                for table_name, table_sql in tables:
                    schema_backup_result['schema_dump']['tables'][table_name] = table_sql
                
            except Exception as e:
                schema_backup_result['backup_errors'].append(f"Table schema extraction failed: {str(e)}")
            
            # Extract index schemas
            try:
                cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
                indexes = cursor.fetchall()
                
                schema_backup_result['schema_dump']['indexes'] = {}
                for index_name, index_sql in indexes:
                    if index_sql:  # Some indexes might have null SQL (auto-created)
                        schema_backup_result['schema_dump']['indexes'][index_name] = index_sql
                
            except Exception as e:
                schema_backup_result['backup_errors'].append(f"Index schema extraction failed: {str(e)}")
            
            # Extract triggers and views if any
            try:
                cursor.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('trigger', 'view')")
                other_objects = cursor.fetchall()
                
                schema_backup_result['schema_dump']['other_objects'] = {}
                for obj_name, obj_sql in other_objects:
                    if obj_sql:
                        schema_backup_result['schema_dump']['other_objects'][obj_name] = obj_sql
                
            except Exception as e:
                schema_backup_result['backup_errors'].append(f"Other objects extraction failed: {str(e)}")
        
        # Save schema dump to file
        try:
            database_path = database_config.get('database_path', 'kira_data.db')
            backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            schema_backup_path = f"{database_path}.schema_backup_{backup_timestamp}.json"
            
            with open(schema_backup_path, 'w') as backup_file:
                json.dump(schema_backup_result['schema_dump'], backup_file, indent=2)
            
            schema_backup_result['backup_path'] = schema_backup_path
            
        except Exception as e:
            schema_backup_result['backup_errors'].append(f"Schema backup file creation failed: {str(e)}")
        
        # Determine status
        if schema_backup_result['backup_errors']:
            schema_backup_result['status'] = 'partial' if schema_backup_result['schema_dump'] else 'failed'
        
        return schema_backup_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'backup_path': None,
            'backup_timestamp': datetime.now().isoformat(),
            'schema_dump': {},
            'backup_errors': [str(e)]
        }

def _restore_database_schema(schema_config: Dict[str, Any]) -> Dict[str, Any]:
    """Stellt Schema aus Backup wieder her"""
    try:
        restore_result = {
            'status': 'success',
            'restore_source': schema_config.get('restore_source'),
            'restore_timestamp': datetime.now().isoformat(),
            'restore_steps': [],
            'restore_errors': []
        }
        
        if not restore_result['restore_source']:
            return {
                'status': 'error',
                'error': 'No restore source specified',
                'restore_steps': [],
                'restore_errors': ['No restore source provided']
            }
        
        # Load schema dump
        try:
            with open(restore_result['restore_source'], 'r') as backup_file:
                schema_dump = json.load(backup_file)
            
            restore_result['restore_steps'].append('schema_dump_loaded')
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Failed to load schema dump: {str(e)}',
                'restore_steps': [],
                'restore_errors': [str(e)]
            }
        
        database_config = _get_default_database_config()
        
        with _get_database_connection(database_config) as connection:
            cursor = connection.cursor()
            
            try:
                # Step 1: Drop existing tables
                restore_result['restore_steps'].append('dropping_existing_tables')
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                existing_tables = [row[0] for row in cursor.fetchall()]
                
                for table_name in existing_tables:
                    try:
                        cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
                    except Exception as e:
                        restore_result['restore_errors'].append(f"Failed to drop table {table_name}: {str(e)}")
                
                # Step 2: Recreate tables
                restore_result['restore_steps'].append('recreating_tables')
                
                tables_data = schema_dump.get('tables', {})
                for table_name, table_sql in tables_data.items():
                    try:
                        cursor.execute(table_sql)
                        logger.info(f"Restored table: {table_name}")
                    except Exception as e:
                        restore_result['restore_errors'].append(f"Failed to restore table {table_name}: {str(e)}")
                
                # Step 3: Recreate indexes
                restore_result['restore_steps'].append('recreating_indexes')
                
                indexes_data = schema_dump.get('indexes', {})
                for index_name, index_sql in indexes_data.items():
                    try:
                        cursor.execute(index_sql)
                        logger.info(f"Restored index: {index_name}")
                    except Exception as e:
                        restore_result['restore_errors'].append(f"Failed to restore index {index_name}: {str(e)}")
                
                # Step 4: Recreate other objects
                restore_result['restore_steps'].append('recreating_other_objects')
                
                other_objects_data = schema_dump.get('other_objects', {})
                for obj_name, obj_sql in other_objects_data.items():
                    try:
                        cursor.execute(obj_sql)
                        logger.info(f"Restored object: {obj_name}")
                    except Exception as e:
                        restore_result['restore_errors'].append(f"Failed to restore object {obj_name}: {str(e)}")
                
                connection.commit()
                restore_result['restore_steps'].append('schema_restore_completed')
                
            except Exception as e:
                connection.rollback()
                restore_result['restore_errors'].append(f"Schema restore transaction failed: {str(e)}")
                restore_result['status'] = 'failed'
        
        # Determine final status
        if restore_result['restore_errors']:
            restore_result['status'] = 'partial' if len(restore_result['restore_steps']) > 2 else 'failed'
        
        return restore_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'restore_source': schema_config.get('restore_source'),
            'restore_timestamp': datetime.now().isoformat(),
            'restore_steps': [],
            'restore_errors': [str(e)]
        }

def _compare_database_schemas(schema_config: Dict[str, Any]) -> Dict[str, Any]:
    """Vergleicht Database Schemas"""
    try:
        comparison_result = {
            'status': 'success',
            'comparison_timestamp': datetime.now().isoformat(),
            'schema_differences': {},
            'comparison_errors': [],
            'schemas_identical': False
        }
        
        # Get current schema
        current_schema = _extract_current_schema()
        
        # Get expected schema
        expected_schema = _get_expected_schema()
        
        # Compare tables
        comparison_result['schema_differences']['tables'] = _compare_table_schemas(current_schema.get('tables', {}), expected_schema.get('tables', {}))
        
        # Compare indexes
        comparison_result['schema_differences']['indexes'] = _compare_index_schemas(current_schema.get('indexes', {}), expected_schema.get('indexes', {}))
        
        # Determine if schemas are identical
        tables_identical = len(comparison_result['schema_differences']['tables']['missing']) == 0 and len(comparison_result['schema_differences']['tables']['extra']) == 0
        indexes_identical = len(comparison_result['schema_differences']['indexes']['missing']) == 0 and len(comparison_result['schema_differences']['indexes']['extra']) == 0
        
        comparison_result['schemas_identical'] = tables_identical and indexes_identical
        
        return comparison_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'comparison_timestamp': datetime.now().isoformat(),
            'schema_differences': {},
            'comparison_errors': [str(e)],
            'schemas_identical': False
        }

def _analyze_database_schema(schema_config: Dict[str, Any]) -> Dict[str, Any]:
    """Analysiert Database Schema"""
    try:
        analysis_result = {
            'status': 'success',
            'analysis_timestamp': datetime.now().isoformat(),
            'schema_analysis': {},
            'analysis_errors': [],
            'recommendations': []
        }
        
        database_config = _get_default_database_config()
        
        with _get_database_connection(database_config) as connection:
            cursor = connection.cursor()
            
            # Analyze tables
            try:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables = [row[0] for row in cursor.fetchall()]
                
                table_analysis = {}
                for table_name in tables:
                    table_stats = _analyze_table_structure(cursor, table_name)
                    table_analysis[table_name] = table_stats
                
                analysis_result['schema_analysis']['tables'] = table_analysis
                
            except Exception as e:
                analysis_result['analysis_errors'].append(f"Table analysis failed: {str(e)}")
            
            # Analyze indexes
            try:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
                indexes = [row[0] for row in cursor.fetchall()]
                
                index_analysis = {}
                for index_name in indexes:
                    index_stats = _analyze_index_usage(cursor, index_name)
                    index_analysis[index_name] = index_stats
                
                analysis_result['schema_analysis']['indexes'] = index_analysis
                
            except Exception as e:
                analysis_result['analysis_errors'].append(f"Index analysis failed: {str(e)}")
        
        # Generate recommendations
        analysis_result['recommendations'] = _generate_schema_recommendations(analysis_result['schema_analysis'])
        
        return analysis_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'analysis_timestamp': datetime.now().isoformat(),
            'schema_analysis': {},
            'analysis_errors': [str(e)],
            'recommendations': []
        }

# ====================================
# HELPER FUNCTIONS FOR MIGRATION & SCHEMA
# ====================================

def _get_available_migrations(source_version: str, target_version: str) -> List[Dict]:
    """Holt verfügbare Migrations zwischen Versionen"""
    # This would typically read from a migrations directory or registry
    # For now, return a simple example migration
    return [
        {
            'version': '1.1',
            'description': 'Add user preferences table',
            'queries': [
                '''
                CREATE TABLE IF NOT EXISTS kira_user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id VARCHAR(100) NOT NULL,
                    preference_key VARCHAR(100) NOT NULL,
                    preference_value TEXT,
                    created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, preference_key)
                )
                ''',
                '''
                CREATE INDEX IF NOT EXISTS idx_user_preferences_user 
                ON kira_user_preferences(user_id)
                '''
            ]
        }
    ]

def _extract_current_schema() -> Dict[str, Any]:
    """Extrahiert aktuelles Schema"""
    try:
        current_schema = {'tables': {}, 'indexes': {}}
        database_config = _get_default_database_config()
        
        with _get_database_connection(database_config) as connection:
            cursor = connection.cursor()
            
            # Extract tables
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = cursor.fetchall()
            
            for table_name, table_sql in tables:
                current_schema['tables'][table_name] = table_sql
            
            # Extract indexes
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
            indexes = cursor.fetchall()
            
            for index_name, index_sql in indexes:
                if index_sql:
                    current_schema['indexes'][index_name] = index_sql
        
        return current_schema
        
    except Exception as e:
        logger.debug(f"Current schema extraction failed: {e}")
        return {'tables': {}, 'indexes': {}}

def _get_expected_schema() -> Dict[str, Any]:
    """Holt erwartetes Schema"""
    return {
        'tables': _get_core_table_schemas(),
        'indexes': _get_core_index_schemas()
    }

def _compare_table_schemas(current_tables: Dict, expected_tables: Dict) -> Dict[str, Any]:
    """Vergleicht Table Schemas"""
    try:
        comparison = {
            'missing': [],
            'extra': [],
            'modified': []
        }
        
        # Find missing tables
        for expected_table in expected_tables:
            if expected_table not in current_tables:
                comparison['missing'].append(expected_table)
        
        # Find extra tables
        for current_table in current_tables:
            if current_table not in expected_tables:
                comparison['extra'].append(current_table)
        
        # Find modified tables (simplified comparison)
        for table_name in set(current_tables.keys()) & set(expected_tables.keys()):
            if current_tables[table_name] != expected_tables[table_name]:
                comparison['modified'].append(table_name)
        
        return comparison
        
    except Exception as e:
        return {
            'missing': [],
            'extra': [],
            'modified': [],
            'comparison_error': str(e)
        }

def _compare_index_schemas(current_indexes: Dict, expected_indexes: Dict) -> Dict[str, Any]:
    """Vergleicht Index Schemas"""
    try:
        comparison = {
            'missing': [],
            'extra': []
        }
        
        # Find missing indexes
        for expected_index in expected_indexes:
            if expected_index not in current_indexes:
                comparison['missing'].append(expected_index)
        
        # Find extra indexes
        for current_index in current_indexes:
            if current_index not in expected_indexes:
                comparison['extra'].append(current_index)
        
        return comparison
        
    except Exception as e:
        return {
            'missing': [],
            'extra': [],
            'comparison_error': str(e)
        }

def _analyze_table_structure(cursor, table_name: str) -> Dict[str, Any]:
    """Analysiert Table Structure"""
    try:
        table_stats = {
            'column_count': 0,
            'has_primary_key': False,
            'has_indexes': False,
            'estimated_row_count': 0
        }
        
        # Get table info
        cursor.execute(f"PRAGMA table_info({table_name})")
        table_info = cursor.fetchall()
        
        table_stats['column_count'] = len(table_info)
        table_stats['has_primary_key'] = any(col[5] == 1 for col in table_info)  # pk column
        
        # Check for indexes
        cursor.execute(f"PRAGMA index_list({table_name})")
        indexes = cursor.fetchall()
        table_stats['has_indexes'] = len(indexes) > 0
        
        # Estimate row count
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            table_stats['estimated_row_count'] = cursor.fetchone()[0]
        except:
            table_stats['estimated_row_count'] = 0
        
        return table_stats
        
    except Exception as e:
        return {
            'column_count': 0,
            'has_primary_key': False,
            'has_indexes': False,
            'estimated_row_count': 0,
            'analysis_error': str(e)
        }

def _analyze_index_usage(cursor, index_name: str) -> Dict[str, Any]:
    """Analysiert Index Usage (vereinfacht)"""
    try:
        index_stats = {
            'exists': True,
            'is_unique': False,
            'column_count': 0
        }
        
        # Get index info (simplified)
        cursor.execute(f"PRAGMA index_info({index_name})")
        index_info = cursor.fetchall()
        
        index_stats['column_count'] = len(index_info)
        
        return index_stats
        
    except Exception as e:
        return {
            'exists': False,
            'is_unique': False,
            'column_count': 0,
            'analysis_error': str(e)
        }

def _generate_schema_recommendations(schema_analysis: Dict) -> List[str]:
    """Generiert Schema Recommendations"""
    recommendations = []
    
    try:
        tables_analysis = schema_analysis.get('tables', {})
        
        for table_name, table_stats in tables_analysis.items():
            # Check for missing primary key
            if not table_stats.get('has_primary_key', False):
                recommendations.append(f"Consider adding a primary key to table '{table_name}'")
            
            # Check for missing indexes on large tables
            if table_stats.get('estimated_row_count', 0) > 1000 and not table_stats.get('has_indexes', False):
                recommendations.append(f"Consider adding indexes to table '{table_name}' for better performance")
            
            # Check for too many columns
            if table_stats.get('column_count', 0) > 20:
                recommendations.append(f"Consider normalizing table '{table_name}' - it has many columns")
        
        return recommendations
        
    except Exception as e:
        return [f"Schema analysis recommendation generation failed: {str(e)}"]

# Final __all__ update
__all__.extend([
    '_validate_database_schema',
    '_create_database_schema', 
    '_migrate_database_schema',
    '_backup_database_schema',
    '_restore_database_schema',
    '_compare_database_schemas',
    '_analyze_database_schema'
])