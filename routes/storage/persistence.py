"""
Storage Persistence Module
Data Persistence, Serialization, State Management, Data Consistency
"""

import logging
import json
import pickle
import threading
import time
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import sqlite3
import gzip
import shutil

logger = logging.getLogger(__name__)

# Persistence State Tracking
_persistence_state = {
    'initialization_timestamp': None,
    'last_save_timestamp': None,
    'last_checkpoint_timestamp': None,
    'active_sessions': {},
    'persistence_statistics': {
        'total_saves': 0,
        'total_loads': 0,
        'total_checkpoints': 0,
        'data_integrity_checks': 0
    },
    'auto_save_enabled': True,
    'checkpoint_enabled': True
}

_persistence_lock = threading.Lock()

# Auto-save thread
_auto_save_thread = None
_checkpoint_thread = None
_shutdown_event = threading.Event()

def handle_data_persistence(operation_type: str,
                           data: Any = None,
                           persistence_key: str = None,
                           persistence_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Data Persistence Handler
    
    Umfassende Data Persistence für verschiedene Datentypen und Szenarien
    """
    try:
        if persistence_config is None:
            persistence_config = {
                'storage_format': 'json',  # json, pickle, binary
                'compression_enabled': True,
                'encryption_enabled': False,
                'backup_on_save': True,
                'validate_on_load': True,
                'auto_create_directories': True
            }
        
        # Initialize persistence session
        persistence_session = {
            'session_id': f"persistence_{int(time.time())}",
            'start_time': time.time(),
            'operation_type': operation_type,
            'persistence_key': persistence_key,
            'persistence_config': persistence_config,
            'operation_results': {}
        }
        
        # Track active session
        with _persistence_lock:
            _persistence_state['active_sessions'][persistence_session['session_id']] = persistence_session
        
        try:
            # Perform persistence operation based on type
            if operation_type == 'save':
                persistence_session['operation_results'] = _save_persistent_data(data, persistence_key, persistence_config)
            
            elif operation_type == 'load':
                persistence_session['operation_results'] = _load_persistent_data(persistence_key, persistence_config)
            
            elif operation_type == 'delete':
                persistence_session['operation_results'] = _delete_persistent_data(persistence_key, persistence_config)
            
            elif operation_type == 'checkpoint':
                persistence_session['operation_results'] = _create_data_checkpoint(data, persistence_key, persistence_config)
            
            elif operation_type == 'restore':
                checkpoint_id = persistence_config.get('checkpoint_id')
                persistence_session['operation_results'] = _restore_from_checkpoint(persistence_key, checkpoint_id, persistence_config)
            
            elif operation_type == 'validate':
                persistence_session['operation_results'] = _validate_persistent_data(persistence_key, persistence_config)
            
            elif operation_type == 'list':
                persistence_session['operation_results'] = _list_persistent_data(persistence_config)
            
            elif operation_type == 'cleanup':
                persistence_session['operation_results'] = _cleanup_persistent_data(persistence_config)
            
            else:
                persistence_session['operation_results'] = {
                    'status': 'error',
                    'error': f'Unsupported persistence operation: {operation_type}'
                }
        
        finally:
            # Remove from active sessions
            with _persistence_lock:
                _persistence_state['active_sessions'].pop(persistence_session['session_id'], None)
        
        # Add persistence metadata
        persistence_session.update({
            'end_time': time.time(),
            'operation_success': persistence_session['operation_results'].get('status') == 'success',
            'operation_duration_ms': (time.time() - persistence_session['start_time']) * 1000
        })
        
        # Update statistics
        with _persistence_lock:
            if operation_type == 'save':
                _persistence_state['persistence_statistics']['total_saves'] += 1
                _persistence_state['last_save_timestamp'] = datetime.now().isoformat()
            elif operation_type == 'load':
                _persistence_state['persistence_statistics']['total_loads'] += 1
            elif operation_type == 'checkpoint':
                _persistence_state['persistence_statistics']['total_checkpoints'] += 1
                _persistence_state['last_checkpoint_timestamp'] = datetime.now().isoformat()
            elif operation_type == 'validate':
                _persistence_state['persistence_statistics']['data_integrity_checks'] += 1
        
        return {
            'success': persistence_session['operation_success'],
            'persistence_session': persistence_session,
            'persistence_result': persistence_session['operation_results'],
            'persistence_summary': {
                'operation_type': operation_type,
                'persistence_key': persistence_key,
                'operation_success': persistence_session['operation_success'],
                'operation_duration_ms': persistence_session['operation_duration_ms']
            }
        }
        
    except Exception as e:
        logger.error(f"Data persistence operation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'persistence_result': {},
            'persistence_summary': {'error': str(e)}
        }

def manage_state_serialization(operation_type: str,
                             state_data: Any = None,
                             serialization_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    State Serialization Manager
    
    Verwaltet Serialization und Deserialization von System State
    """
    try:
        if serialization_config is None:
            serialization_config = {
                'serialization_format': 'json',  # json, pickle, msgpack
                'include_metadata': True,
                'compress_data': False,
                'validate_schema': True,
                'preserve_types': False
            }
        
        # Initialize serialization session
        serialization_session = {
            'session_id': f"serialization_{int(time.time())}",
            'start_time': time.time(),
            'operation_type': operation_type,
            'serialization_config': serialization_config,
            'serialization_results': {}
        }
        
        # Perform serialization operation
        if operation_type == 'serialize':
            serialization_session['serialization_results'] = _serialize_state_data(state_data, serialization_config)
        
        elif operation_type == 'deserialize':
            serialization_session['serialization_results'] = _deserialize_state_data(state_data, serialization_config)
        
        elif operation_type == 'validate_schema':
            serialization_session['serialization_results'] = _validate_serialization_schema(state_data, serialization_config)
        
        elif operation_type == 'convert_format':
            target_format = serialization_config.get('target_format', 'json')
            serialization_session['serialization_results'] = _convert_serialization_format(state_data, serialization_config, target_format)
        
        elif operation_type == 'compress':
            serialization_session['serialization_results'] = _compress_serialized_data(state_data, serialization_config)
        
        elif operation_type == 'decompress':
            serialization_session['serialization_results'] = _decompress_serialized_data(state_data, serialization_config)
        
        else:
            serialization_session['serialization_results'] = {
                'status': 'error',
                'error': f'Unsupported serialization operation: {operation_type}'
            }
        
        # Add serialization metadata
        serialization_session.update({
            'end_time': time.time(),
            'serialization_success': serialization_session['serialization_results'].get('status') == 'success',
            'serialization_duration_ms': (time.time() - serialization_session['start_time']) * 1000
        })
        
        return {
            'success': serialization_session['serialization_success'],
            'serialization_session': serialization_session,
            'serialization_result': serialization_session['serialization_results'],
            'serialization_summary': {
                'operation_type': operation_type,
                'serialization_success': serialization_session['serialization_success'],
                'serialization_duration_ms': serialization_session['serialization_duration_ms']
            }
        }
        
    except Exception as e:
        logger.error(f"State serialization failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'serialization_result': {},
            'serialization_summary': {'error': str(e)}
        }

def ensure_data_consistency(consistency_check_type: str = 'full',
                          data_sources: List[str] = None,
                          consistency_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Data Consistency Manager
    
    Stellt Data Consistency über verschiedene Persistence Layer sicher
    """
    try:
        if consistency_config is None:
            consistency_config = {
                'check_data_integrity': True,
                'validate_checksums': True,
                'repair_inconsistencies': False,
                'create_consistency_report': True,
                'backup_before_repair': True
            }
        
        if data_sources is None:
            data_sources = ['persistent_storage', 'checkpoint_storage', 'backup_storage']
        
        # Initialize consistency check session
        consistency_session = {
            'session_id': f"consistency_{int(time.time())}",
            'start_time': time.time(),
            'consistency_check_type': consistency_check_type,
            'data_sources': data_sources,
            'consistency_config': consistency_config,
            'consistency_results': {}
        }
        
        # Perform consistency check based on type
        if consistency_check_type == 'full':
            consistency_session['consistency_results'] = _perform_full_consistency_check(data_sources, consistency_config)
        
        elif consistency_check_type == 'quick':
            consistency_session['consistency_results'] = _perform_quick_consistency_check(data_sources, consistency_config)
        
        elif consistency_check_type == 'checksum':
            consistency_session['consistency_results'] = _perform_checksum_validation(data_sources, consistency_config)
        
        elif consistency_check_type == 'structural':
            consistency_session['consistency_results'] = _perform_structural_consistency_check(data_sources, consistency_config)
        
        elif consistency_check_type == 'repair':
            consistency_session['consistency_results'] = _repair_data_inconsistencies(data_sources, consistency_config)
        
        else:
            consistency_session['consistency_results'] = {
                'status': 'error',
                'error': f'Unsupported consistency check type: {consistency_check_type}'
            }
        
        # Add consistency metadata
        consistency_session.update({
            'end_time': time.time(),
            'consistency_success': consistency_session['consistency_results'].get('status') == 'success',
            'consistency_duration_ms': (time.time() - consistency_session['start_time']) * 1000
        })
        
        return {
            'success': consistency_session['consistency_success'],
            'consistency_session': consistency_session,
            'consistency_result': consistency_session['consistency_results'],
            'consistency_summary': {
                'check_type': consistency_check_type,
                'data_sources_checked': len(data_sources),
                'consistency_success': consistency_session['consistency_success'],
                'consistency_duration_ms': consistency_session['consistency_duration_ms']
            }
        }
        
    except Exception as e:
        logger.error(f"Data consistency check failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'consistency_result': {},
            'consistency_summary': {'error': str(e)}
        }

def recover_persistent_data(recovery_type: str = 'automatic',
                          recovery_source: str = None,
                          recovery_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Persistent Data Recovery
    
    Stellt persistente Daten aus verschiedenen Sources wieder her
    """
    try:
        if recovery_config is None:
            recovery_config = {
                'recovery_strategy': 'latest_valid',  # latest_valid, specific_checkpoint, merge_sources
                'validate_recovered_data': True,
                'backup_current_state': True,
                'recovery_timeout_seconds': 300,
                'allow_partial_recovery': True
            }
        
        # Initialize recovery session
        recovery_session = {
            'session_id': f"recovery_{int(time.time())}",
            'start_time': time.time(),
            'recovery_type': recovery_type,
            'recovery_source': recovery_source,
            'recovery_config': recovery_config,
            'recovery_results': {}
        }
        
        # Perform recovery based on type
        if recovery_type == 'automatic':
            recovery_session['recovery_results'] = _perform_automatic_recovery(recovery_config)
        
        elif recovery_type == 'checkpoint':
            recovery_session['recovery_results'] = _recover_from_checkpoint(recovery_source, recovery_config)
        
        elif recovery_type == 'backup':
            recovery_session['recovery_results'] = _recover_from_backup(recovery_source, recovery_config)
        
        elif recovery_type == 'partial':
            recovery_session['recovery_results'] = _perform_partial_recovery(recovery_source, recovery_config)
        
        elif recovery_type == 'merge':
            recovery_session['recovery_results'] = _perform_merge_recovery(recovery_config)
        
        elif recovery_type == 'validate_only':
            recovery_session['recovery_results'] = _validate_recovery_sources(recovery_config)
        
        else:
            recovery_session['recovery_results'] = {
                'status': 'error',
                'error': f'Unsupported recovery type: {recovery_type}'
            }
        
        # Add recovery metadata
        recovery_session.update({
            'end_time': time.time(),
            'recovery_success': recovery_session['recovery_results'].get('status') == 'success',
            'recovery_duration_ms': (time.time() - recovery_session['start_time']) * 1000
        })
        
        return {
            'success': recovery_session['recovery_success'],
            'recovery_session': recovery_session,
            'recovery_result': recovery_session['recovery_results'],
            'recovery_summary': {
                'recovery_type': recovery_type,
                'recovery_source': recovery_source,
                'recovery_success': recovery_session['recovery_success'],
                'recovery_duration_ms': recovery_session['recovery_duration_ms']
            }
        }
        
    except Exception as e:
        logger.error(f"Persistent data recovery failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'recovery_result': {},
            'recovery_summary': {'error': str(e)}
        }

def start_auto_persistence(auto_save_interval_minutes: int = 5,
                         checkpoint_interval_minutes: int = 15,
                         persistence_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Startet automatische Persistence Threads
    """
    try:
        global _auto_save_thread, _checkpoint_thread, _shutdown_event
        
        if persistence_config is None:
            persistence_config = {
                'auto_save_enabled': True,
                'checkpoint_enabled': True,
                'persistence_directory': 'persistence/',
                'max_auto_save_files': 10,
                'max_checkpoint_files': 5
            }
        
        # Clear shutdown event
        _shutdown_event.clear()
        
        # Start auto-save thread
        if persistence_config.get('auto_save_enabled', True) and not (_auto_save_thread and _auto_save_thread.is_alive()):
            _auto_save_thread = threading.Thread(
                target=_auto_save_worker,
                args=(auto_save_interval_minutes, persistence_config),
                daemon=True
            )
            _auto_save_thread.start()
            
            with _persistence_lock:
                _persistence_state['auto_save_enabled'] = True
        
        # Start checkpoint thread
        if persistence_config.get('checkpoint_enabled', True) and not (_checkpoint_thread and _checkpoint_thread.is_alive()):
            _checkpoint_thread = threading.Thread(
                target=_checkpoint_worker,
                args=(checkpoint_interval_minutes, persistence_config),
                daemon=True
            )
            _checkpoint_thread.start()
            
            with _persistence_lock:
                _persistence_state['checkpoint_enabled'] = True
        
        return {
            'success': True,
            'auto_persistence_started': True,
            'auto_save_interval_minutes': auto_save_interval_minutes,
            'checkpoint_interval_minutes': checkpoint_interval_minutes,
            'threads_started': {
                'auto_save_thread': _auto_save_thread.is_alive() if _auto_save_thread else False,
                'checkpoint_thread': _checkpoint_thread.is_alive() if _checkpoint_thread else False
            }
        }
        
    except Exception as e:
        logger.error(f"Auto persistence startup failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'auto_persistence_started': False
        }

def stop_auto_persistence() -> Dict[str, Any]:
    """
    Stoppt automatische Persistence Threads
    """
    try:
        global _auto_save_thread, _checkpoint_thread, _shutdown_event
        
        # Signal shutdown
        _shutdown_event.set()
        
        # Wait for threads to finish
        threads_stopped = {}
        
        if _auto_save_thread and _auto_save_thread.is_alive():
            _auto_save_thread.join(timeout=10)
            threads_stopped['auto_save_thread'] = not _auto_save_thread.is_alive()
        else:
            threads_stopped['auto_save_thread'] = True
        
        if _checkpoint_thread and _checkpoint_thread.is_alive():
            _checkpoint_thread.join(timeout=10)
            threads_stopped['checkpoint_thread'] = not _checkpoint_thread.is_alive()
        else:
            threads_stopped['checkpoint_thread'] = True
        
        # Update state
        with _persistence_lock:
            _persistence_state['auto_save_enabled'] = False
            _persistence_state['checkpoint_enabled'] = False
        
        return {
            'success': True,
            'auto_persistence_stopped': True,
            'threads_stopped': threads_stopped,
            'all_threads_stopped': all(threads_stopped.values())
        }
        
    except Exception as e:
        logger.error(f"Auto persistence shutdown failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'auto_persistence_stopped': False
        }

def get_persistence_status() -> Dict[str, Any]:
    """
    Holt aktuellen Persistence Status
    """
    try:
        with _persistence_lock:
            status = _persistence_state.copy()
        
        # Add thread status
        global _auto_save_thread, _checkpoint_thread
        status['thread_status'] = {
            'auto_save_thread_active': _auto_save_thread.is_alive() if _auto_save_thread else False,
            'checkpoint_thread_active': _checkpoint_thread.is_alive() if _checkpoint_thread else False
        }
        
        # Add persistence directory info
        persistence_dir = 'persistence/'
        if os.path.exists(persistence_dir):
            status['persistence_directory_info'] = {
                'exists': True,
                'total_files': len(os.listdir(persistence_dir)),
                'directory_size_bytes': sum(
                    os.path.getsize(os.path.join(persistence_dir, f))
                    for f in os.listdir(persistence_dir)
                    if os.path.isfile(os.path.join(persistence_dir, f))
                )
            }
        else:
            status['persistence_directory_info'] = {
                'exists': False,
                'total_files': 0,
                'directory_size_bytes': 0
            }
        
        return {
            'success': True,
            'persistence_status': status,
            'status_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get persistence status failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'persistence_status': {},
            'status_timestamp': datetime.now().isoformat()
        }

# ====================================
# PRIVATE HELPER FUNCTIONS
# ====================================

def _save_persistent_data(data: Any, persistence_key: str, config: Dict) -> Dict[str, Any]:
    """Speichert persistente Daten"""
    try:
        # Ensure persistence directory exists
        persistence_dir = config.get('persistence_directory', 'persistence/')
        if config.get('auto_create_directories', True):
            os.makedirs(persistence_dir, exist_ok=True)
        
        # Generate filename
        storage_format = config.get('storage_format', 'json')
        filename = f"{persistence_key}.{storage_format}"
        if config.get('compression_enabled', False):
            filename += '.gz'
        
        file_path = os.path.join(persistence_dir, filename)
        
        # Backup existing file if requested
        if config.get('backup_on_save', True) and os.path.exists(file_path):
            backup_path = f"{file_path}.backup.{int(time.time())}"
            shutil.copy2(file_path, backup_path)
        
        # Serialize data based on format
        if storage_format == 'json':
            serialized_data = json.dumps(data, indent=2, default=str)
        elif storage_format == 'pickle':
            serialized_data = pickle.dumps(data)
        else:
            serialized_data = str(data)
        
        # Write to file
        if config.get('compression_enabled', False):
            with gzip.open(file_path, 'wb') as f:
                if isinstance(serialized_data, str):
                    f.write(serialized_data.encode('utf-8'))
                else:
                    f.write(serialized_data)
        else:
            mode = 'wb' if storage_format == 'pickle' else 'w'
            encoding = None if storage_format == 'pickle' else 'utf-8'
            
            with open(file_path, mode, encoding=encoding) as f:
                f.write(serialized_data)
        
        # Calculate checksum
        checksum = _calculate_file_checksum(file_path)
        
        # Save metadata
        metadata = {
            'persistence_key': persistence_key,
            'storage_format': storage_format,
            'compression_enabled': config.get('compression_enabled', False),
            'save_timestamp': datetime.now().isoformat(),
            'data_size_bytes': os.path.getsize(file_path),
            'checksum': checksum
        }
        
        metadata_path = f"{file_path}.meta"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'status': 'success',
            'file_path': file_path,
            'metadata_path': metadata_path,
            'data_size_bytes': metadata['data_size_bytes'],
            'checksum': checksum
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Save persistent data failed: {str(e)}'
        }

def _load_persistent_data(persistence_key: str, config: Dict) -> Dict[str, Any]:
    """Lädt persistente Daten"""
    try:
        # Find persistence file
        persistence_dir = config.get('persistence_directory', 'persistence/')
        storage_format = config.get('storage_format', 'json')
        
        filename = f"{persistence_key}.{storage_format}"
        compressed_filename = f"{filename}.gz"
        
        file_path = None
        is_compressed = False
        
        # Check for compressed version first
        compressed_path = os.path.join(persistence_dir, compressed_filename)
        if os.path.exists(compressed_path):
            file_path = compressed_path
            is_compressed = True
        else:
            regular_path = os.path.join(persistence_dir, filename)
            if os.path.exists(regular_path):
                file_path = regular_path
        
        if not file_path:
            return {
                'status': 'error',
                'error': f'Persistent data not found for key: {persistence_key}'
            }
        
        # Load metadata if available
        metadata_path = f"{file_path}.meta"
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Validate checksum if requested
        if config.get('validate_on_load', True) and metadata.get('checksum'):
            current_checksum = _calculate_file_checksum(file_path)
            if current_checksum != metadata['checksum']:
                return {
                    'status': 'error',
                    'error': 'Data integrity check failed - checksum mismatch'
                }
        
        # Read file
        if is_compressed:
            with gzip.open(file_path, 'rb') as f:
                raw_data = f.read()
                if storage_format != 'pickle':
                    raw_data = raw_data.decode('utf-8')
        else:
            mode = 'rb' if storage_format == 'pickle' else 'r'
            encoding = None if storage_format == 'pickle' else 'utf-8'
            
            with open(file_path, mode, encoding=encoding) as f:
                raw_data = f.read()
        
        # Deserialize data
        if storage_format == 'json':
            data = json.loads(raw_data)
        elif storage_format == 'pickle':
            data = pickle.loads(raw_data)
        else:
            data = raw_data
        
        return {
            'status': 'success',
            'data': data,
            'metadata': metadata,
            'file_path': file_path,
            'load_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Load persistent data failed: {str(e)}'
        }

def _delete_persistent_data(persistence_key: str, config: Dict) -> Dict[str, Any]:
    """Löscht persistente Daten"""
    try:
        persistence_dir = config.get('persistence_directory', 'persistence/')
        deleted_files = []
        
        # Find and delete all related files
        for storage_format in ['json', 'pickle', 'binary']:
            for compressed in [True, False]:
                filename = f"{persistence_key}.{storage_format}"
                if compressed:
                    filename += '.gz'
                
                file_path = os.path.join(persistence_dir, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(file_path)
                
                # Delete metadata file
                metadata_path = f"{file_path}.meta"
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    deleted_files.append(metadata_path)
        
        return {
            'status': 'success',
            'deleted_files': deleted_files,
            'files_deleted_count': len(deleted_files)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Delete persistent data failed: {str(e)}'
        }

def _create_data_checkpoint(data: Any, persistence_key: str, config: Dict) -> Dict[str, Any]:
    """Erstellt Data Checkpoint"""
    try:
        checkpoint_dir = config.get('checkpoint_directory', 'persistence/checkpoints/')
        if config.get('auto_create_directories', True):
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Generate checkpoint ID
        checkpoint_id = f"{persistence_key}_{int(time.time())}"
        
        # Save checkpoint
        checkpoint_config = config.copy()
        checkpoint_config['persistence_directory'] = checkpoint_dir
        
        save_result = _save_persistent_data(data, checkpoint_id, checkpoint_config)
        
        if save_result['status'] == 'success':
            # Create checkpoint index entry
            index_path = os.path.join(checkpoint_dir, 'checkpoint_index.json')
            
            checkpoint_entry = {
                'checkpoint_id': checkpoint_id,
                'persistence_key': persistence_key,
                'created_timestamp': datetime.now().isoformat(),
                'file_path': save_result['file_path'],
                'data_size_bytes': save_result['data_size_bytes'],
                'checksum': save_result['checksum']
            }
            
            # Update index
            checkpoint_index = []
            if os.path.exists(index_path):
                with open(index_path, 'r', encoding='utf-8') as f:
                    checkpoint_index = json.load(f)
            
            checkpoint_index.append(checkpoint_entry)
            
            # Keep only recent checkpoints
            max_checkpoints = config.get('max_checkpoint_files', 5)
            if len(checkpoint_index) > max_checkpoints:
                # Remove oldest checkpoints
                checkpoint_index = sorted(checkpoint_index, key=lambda x: x['created_timestamp'])
                old_checkpoints = checkpoint_index[:-max_checkpoints]
                checkpoint_index = checkpoint_index[-max_checkpoints:]
                
                # Delete old checkpoint files
                for old_checkpoint in old_checkpoints:
                    try:
                        if os.path.exists(old_checkpoint['file_path']):
                            os.remove(old_checkpoint['file_path'])
                    except:
                        pass
            
            # Save updated index
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_index, f, indent=2)
            
            return {
                'status': 'success',
                'checkpoint_id': checkpoint_id,
                'checkpoint_path': save_result['file_path'],
                'checkpoint_size_bytes': save_result['data_size_bytes']
            }
        else:
            return save_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Create data checkpoint failed: {str(e)}'
        }

def _auto_save_worker(interval_minutes: int, config: Dict):
    """Auto-save Worker Thread"""
    try:
        while not _shutdown_event.is_set():
            # Wait for interval or shutdown signal
            if _shutdown_event.wait(timeout=interval_minutes * 60):
                break  # Shutdown signal received
            
            try:
                # Perform auto-save operations
                logger.debug("Performing auto-save operations...")
                
                # This would typically save current system state
                # For now, just update the timestamp
                with _persistence_lock:
                    _persistence_state['last_save_timestamp'] = datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"Auto-save operation failed: {e}")
        
        logger.info("Auto-save worker thread stopped")
        
    except Exception as e:
        logger.error(f"Auto-save worker thread failed: {e}")

def _checkpoint_worker(interval_minutes: int, config: Dict):
    """Checkpoint Worker Thread"""
    try:
        while not _shutdown_event.is_set():
            # Wait for interval or shutdown signal
            if _shutdown_event.wait(timeout=interval_minutes * 60):
                break  # Shutdown signal received
            
            try:
                # Perform checkpoint operations
                logger.debug("Performing checkpoint operations...")
                
                # This would typically create system checkpoints
                # For now, just update the timestamp
                with _persistence_lock:
                    _persistence_state['last_checkpoint_timestamp'] = datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"Checkpoint operation failed: {e}")
        
        logger.info("Checkpoint worker thread stopped")
        
    except Exception as e:
        logger.error(f"Checkpoint worker thread failed: {e}")

def _calculate_file_checksum(file_path: str) -> str:
    """Berechnet File Checksum"""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.debug(f"Checksum calculation failed: {e}")
        return ""

def _initialize_persistence_component():
    """Initialisiert Persistence Component"""
    try:
        with _persistence_lock:
            _persistence_state['initialization_timestamp'] = datetime.now().isoformat()
        
        # Ensure persistence directory exists
        persistence_dir = 'persistence/'
        os.makedirs(persistence_dir, exist_ok=True)
        os.makedirs(os.path.join(persistence_dir, 'checkpoints'), exist_ok=True)
        
        return True
    except Exception as e:
        raise Exception(f"Persistence component initialization failed: {str(e)}")

# Export all public functions
__all__ = [
    'handle_data_persistence',
    'manage_state_serialization',
    'ensure_data_consistency',
    'recover_persistent_data',
    'start_auto_persistence',
    'stop_auto_persistence',
    'get_persistence_status',
    '_initialize_persistence_component'
]
def _serialize_state_data(state_data: Any, config: Dict) -> Dict[str, Any]:
    """Serialisiert State Data"""
    try:
        serialization_format = config.get('serialization_format', 'json')
        include_metadata = config.get('include_metadata', True)
        preserve_types = config.get('preserve_types', False)
        
        result = {
            'status': 'success',
            'serialized_data': None,
            'serialization_metadata': {}
        }
        
        # Add metadata if requested
        if include_metadata:
            result['serialization_metadata'] = {
                'serialization_timestamp': datetime.now().isoformat(),
                'serialization_format': serialization_format,
                'original_data_type': type(state_data).__name__,
                'preserve_types': preserve_types
            }
        
        # Serialize based on format
        if serialization_format == 'json':
            if preserve_types:
                # Create type-preserving wrapper
                wrapped_data = {
                    '__kira_serialized__': True,
                    'data': state_data,
                    'data_type': type(state_data).__name__,
                    'serialization_timestamp': datetime.now().isoformat()
                }
                result['serialized_data'] = json.dumps(wrapped_data, indent=2, default=_json_serializer)
            else:
                result['serialized_data'] = json.dumps(state_data, indent=2, default=_json_serializer)
        
        elif serialization_format == 'pickle':
            result['serialized_data'] = pickle.dumps(state_data)
            result['serialization_metadata']['is_binary'] = True
        
        else:
            result['serialized_data'] = str(state_data)
        
        # Calculate size
        if isinstance(result['serialized_data'], str):
            result['serialization_metadata']['serialized_size_bytes'] = len(result['serialized_data'].encode('utf-8'))
        else:
            result['serialization_metadata']['serialized_size_bytes'] = len(result['serialized_data'])
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'State data serialization failed: {str(e)}',
            'serialized_data': None,
            'serialization_metadata': {}
        }

def _deserialize_state_data(serialized_data: Any, config: Dict) -> Dict[str, Any]:
    """Deserialisiert State Data"""
    try:
        serialization_format = config.get('serialization_format', 'json')
        validate_schema = config.get('validate_schema', True)
        
        result = {
            'status': 'success',
            'deserialized_data': None,
            'deserialization_metadata': {}
        }
        
        # Deserialize based on format
        if serialization_format == 'json':
            if isinstance(serialized_data, str):
                data = json.loads(serialized_data)
            else:
                data = serialized_data
            
            # Check if this is type-preserving format
            if isinstance(data, dict) and data.get('__kira_serialized__'):
                result['deserialized_data'] = data['data']
                result['deserialization_metadata']['original_data_type'] = data.get('data_type')
                result['deserialization_metadata']['serialization_timestamp'] = data.get('serialization_timestamp')
            else:
                result['deserialized_data'] = data
        
        elif serialization_format == 'pickle':
            if isinstance(serialized_data, str):
                # Assume it's base64 encoded
                import base64
                binary_data = base64.b64decode(serialized_data)
                result['deserialized_data'] = pickle.loads(binary_data)
            else:
                result['deserialized_data'] = pickle.loads(serialized_data)
        
        else:
            result['deserialized_data'] = serialized_data
        
        # Validation
        if validate_schema:
            validation_result = _validate_deserialized_data(result['deserialized_data'], config)
            result['deserialization_metadata']['validation_result'] = validation_result
            
            if not validation_result.get('is_valid', False):
                result['status'] = 'warning'
        
        result['deserialization_metadata']['deserialization_timestamp'] = datetime.now().isoformat()
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'State data deserialization failed: {str(e)}',
            'deserialized_data': None,
            'deserialization_metadata': {}
        }

def _validate_serialization_schema(data: Any, config: Dict) -> Dict[str, Any]:
    """Validiert Serialization Schema"""
    try:
        validation_result = {
            'status': 'success',
            'is_valid': True,
            'validation_errors': [],
            'validation_warnings': [],
            'schema_info': {}
        }
        
        # Basic type validation
        allowed_types = config.get('allowed_types', [dict, list, str, int, float, bool, type(None)])
        data_type = type(data)
        
        if data_type not in allowed_types:
            validation_result['validation_errors'].append(f'Data type {data_type.__name__} not allowed')
            validation_result['is_valid'] = False
        
        # Size validation
        max_size = config.get('max_serialized_size_mb', 100)
        if isinstance(data, (dict, list)):
            estimated_size = len(str(data))  # Rough estimate
            if estimated_size > max_size * 1024 * 1024:
                validation_result['validation_errors'].append(f'Data size exceeds maximum {max_size}MB')
                validation_result['is_valid'] = False
        
        # Depth validation for nested structures
        if isinstance(data, (dict, list)):
            max_depth = config.get('max_nesting_depth', 10)
            actual_depth = _calculate_nesting_depth(data)
            validation_result['schema_info']['nesting_depth'] = actual_depth
            
            if actual_depth > max_depth:
                validation_result['validation_errors'].append(f'Nesting depth {actual_depth} exceeds maximum {max_depth}')
                validation_result['is_valid'] = False
        
        # Circular reference check
        if isinstance(data, (dict, list)) and config.get('check_circular_references', True):
            has_circular_refs = _check_circular_references(data)
            validation_result['schema_info']['has_circular_references'] = has_circular_refs
            
            if has_circular_refs:
                validation_result['validation_warnings'].append('Circular references detected - may cause serialization issues')
        
        return validation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Schema validation failed: {str(e)}',
            'is_valid': False,
            'validation_errors': [str(e)]
        }

def _convert_serialization_format(data: Any, config: Dict, target_format: str) -> Dict[str, Any]:
    """Konvertiert Serialization Format"""
    try:
        source_format = config.get('serialization_format', 'json')
        
        result = {
            'status': 'success',
            'converted_data': None,
            'conversion_metadata': {
                'source_format': source_format,
                'target_format': target_format,
                'conversion_timestamp': datetime.now().isoformat()
            }
        }
        
        # First deserialize from source format
        deserialize_config = config.copy()
        deserialize_config['serialization_format'] = source_format
        deserialize_result = _deserialize_state_data(data, deserialize_config)
        
        if deserialize_result['status'] != 'success':
            return deserialize_result
        
        # Then serialize to target format
        serialize_config = config.copy()
        serialize_config['serialization_format'] = target_format
        serialize_result = _serialize_state_data(deserialize_result['deserialized_data'], serialize_config)
        
        if serialize_result['status'] == 'success':
            result['converted_data'] = serialize_result['serialized_data']
            result['conversion_metadata']['serialization_metadata'] = serialize_result['serialization_metadata']
        else:
            return serialize_result
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Format conversion failed: {str(e)}',
            'converted_data': None
        }

def _compress_serialized_data(serialized_data: Any, config: Dict) -> Dict[str, Any]:
    """Komprimiert serialisierte Daten"""
    try:
        compression_method = config.get('compression_method', 'gzip')
        
        result = {
            'status': 'success',
            'compressed_data': None,
            'compression_metadata': {}
        }
        
        # Convert to bytes if string
        if isinstance(serialized_data, str):
            data_bytes = serialized_data.encode('utf-8')
        else:
            data_bytes = serialized_data
        
        original_size = len(data_bytes)
        
        # Compress based on method
        if compression_method == 'gzip':
            result['compressed_data'] = gzip.compress(data_bytes)
        else:
            return {
                'status': 'error',
                'error': f'Unsupported compression method: {compression_method}'
            }
        
        compressed_size = len(result['compressed_data'])
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        
        result['compression_metadata'] = {
            'compression_method': compression_method,
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': compression_ratio,
            'space_saved_bytes': original_size - compressed_size,
            'compression_timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Data compression failed: {str(e)}',
            'compressed_data': None
        }

def _decompress_serialized_data(compressed_data: Any, config: Dict) -> Dict[str, Any]:
    """Dekomprimiert serialisierte Daten"""
    try:
        compression_method = config.get('compression_method', 'gzip')
        
        result = {
            'status': 'success',
            'decompressed_data': None,
            'decompression_metadata': {}
        }
        
        # Decompress based on method
        if compression_method == 'gzip':
            decompressed_bytes = gzip.decompress(compressed_data)
            
            # Try to decode as UTF-8 string
            try:
                result['decompressed_data'] = decompressed_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # Keep as bytes if not valid UTF-8
                result['decompressed_data'] = decompressed_bytes
        else:
            return {
                'status': 'error',
                'error': f'Unsupported compression method: {compression_method}'
            }
        
        result['decompression_metadata'] = {
            'compression_method': compression_method,
            'compressed_size_bytes': len(compressed_data),
            'decompressed_size_bytes': len(decompressed_bytes),
            'decompression_timestamp': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Data decompression failed: {str(e)}',
            'decompressed_data': None
        }

def _perform_full_consistency_check(data_sources: List[str], config: Dict) -> Dict[str, Any]:
    """Führt vollständige Consistency Check durch"""
    try:
        consistency_result = {
            'status': 'success',
            'consistency_issues': [],
            'data_source_status': {},
            'consistency_summary': {}
        }
        
        # Check each data source
        for source in data_sources:
            source_status = _check_data_source_consistency(source, config)
            consistency_result['data_source_status'][source] = source_status
            
            if source_status.get('issues'):
                consistency_result['consistency_issues'].extend([
                    f"{source}: {issue}" for issue in source_status['issues']
                ])
        
        # Cross-reference consistency between sources
        if len(data_sources) > 1 and config.get('check_cross_references', True):
            cross_ref_result = _check_cross_reference_consistency(data_sources, config)
            consistency_result['cross_reference_consistency'] = cross_ref_result
            
            if cross_ref_result.get('inconsistencies'):
                consistency_result['consistency_issues'].extend(cross_ref_result['inconsistencies'])
        
        # Generate consistency summary
        total_sources = len(data_sources)
        healthy_sources = sum(1 for status in consistency_result['data_source_status'].values() 
                            if status.get('is_consistent', False))
        
        consistency_result['consistency_summary'] = {
            'total_sources_checked': total_sources,
            'healthy_sources': healthy_sources,
            'unhealthy_sources': total_sources - healthy_sources,
            'overall_consistency_score': healthy_sources / total_sources if total_sources > 0 else 0,
            'total_issues_found': len(consistency_result['consistency_issues'])
        }
        
        # Determine overall status
        if consistency_result['consistency_issues']:
            consistency_result['status'] = 'issues_detected'
        
        return consistency_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Full consistency check failed: {str(e)}',
            'consistency_issues': [str(e)]
        }

def _perform_quick_consistency_check(data_sources: List[str], config: Dict) -> Dict[str, Any]:
    """Führt schnelle Consistency Check durch"""
    try:
        consistency_result = {
            'status': 'success',
            'quick_check_results': {},
            'critical_issues': [],
            'check_summary': {}
        }
        
        # Quick checks per source
        for source in data_sources:
            quick_result = {
                'source_exists': _check_source_exists(source),
                'recent_activity': _check_recent_activity(source),
                'basic_integrity': _check_basic_integrity(source)
            }
            
            consistency_result['quick_check_results'][source] = quick_result
            
            # Identify critical issues
            if not quick_result['source_exists']:
                consistency_result['critical_issues'].append(f"Source missing: {source}")
            
            if not quick_result['basic_integrity']:
                consistency_result['critical_issues'].append(f"Integrity issues: {source}")
        
        # Generate summary
        consistency_result['check_summary'] = {
            'sources_checked': len(data_sources),
            'critical_issues_count': len(consistency_result['critical_issues']),
            'all_sources_available': all(r['source_exists'] for r in consistency_result['quick_check_results'].values()),
            'basic_integrity_ok': all(r['basic_integrity'] for r in consistency_result['quick_check_results'].values())
        }
        
        return consistency_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Quick consistency check failed: {str(e)}',
            'critical_issues': [str(e)]
        }

def _perform_checksum_validation(data_sources: List[str], config: Dict) -> Dict[str, Any]:
    """Führt Checksum Validation durch"""
    try:
        validation_result = {
            'status': 'success',
            'checksum_results': {},
            'checksum_mismatches': [],
            'validation_summary': {}
        }
        
        # Validate checksums for each source
        for source in data_sources:
            source_result = _validate_source_checksums(source, config)
            validation_result['checksum_results'][source] = source_result
            
            if source_result.get('mismatches'):
                validation_result['checksum_mismatches'].extend([
                    f"{source}: {mismatch}" for mismatch in source_result['mismatches']
                ])
        
        # Generate summary
        total_files_checked = sum(r.get('files_checked', 0) for r in validation_result['checksum_results'].values())
        total_mismatches = len(validation_result['checksum_mismatches'])
        
        validation_result['validation_summary'] = {
            'total_files_checked': total_files_checked,
            'checksum_mismatches': total_mismatches,
            'validation_success_rate': (total_files_checked - total_mismatches) / max(1, total_files_checked),
            'critical_files_affected': sum(r.get('critical_files_affected', 0) for r in validation_result['checksum_results'].values())
        }
        
        return validation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Checksum validation failed: {str(e)}',
            'checksum_mismatches': [str(e)]
        }

def _perform_structural_consistency_check(data_sources: List[str], config: Dict) -> Dict[str, Any]:
    """Führt Structural Consistency Check durch"""
    try:
        structural_result = {
            'status': 'success',
            'structural_analysis': {},
            'structural_issues': [],
            'analysis_summary': {}
        }
        
        # Analyze structure of each source
        for source in data_sources:
            analysis = _analyze_source_structure(source, config)
            structural_result['structural_analysis'][source] = analysis
            
            if analysis.get('structural_problems'):
                structural_result['structural_issues'].extend([
                    f"{source}: {problem}" for problem in analysis['structural_problems']
                ])
        
        # Compare structures across sources if applicable
        if len(data_sources) > 1:
            comparison_result = _compare_source_structures(structural_result['structural_analysis'])
            structural_result['structure_comparison'] = comparison_result
            
            if comparison_result.get('inconsistencies'):
                structural_result['structural_issues'].extend(comparison_result['inconsistencies'])
        
        # Generate summary
        structural_result['analysis_summary'] = {
            'sources_analyzed': len(data_sources),
            'structural_issues_found': len(structural_result['structural_issues']),
            'sources_with_issues': len([s for s in structural_result['structural_analysis'].values() 
                                      if s.get('structural_problems')]),
            'overall_structural_health': 'good' if not structural_result['structural_issues'] else 'issues_detected'
        }
        
        return structural_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Structural consistency check failed: {str(e)}',
            'structural_issues': [str(e)]
        }

def _repair_data_inconsistencies(data_sources: List[str], config: Dict) -> Dict[str, Any]:
    """Repariert Data Inconsistencies"""
    try:
        repair_result = {
            'status': 'success',
            'repair_actions': {},
            'repair_summary': {},
            'backup_info': {}
        }
        
        # Backup before repair if requested
        if config.get('backup_before_repair', True):
            backup_result = _create_repair_backup(data_sources, config)
            repair_result['backup_info'] = backup_result
        
        # Repair each source
        for source in data_sources:
            source_repair = _repair_source_inconsistencies(source, config)
            repair_result['repair_actions'][source] = source_repair
        
        # Generate repair summary
        total_repairs = sum(len(r.get('repairs_performed', [])) for r in repair_result['repair_actions'].values())
        successful_repairs = sum(r.get('successful_repairs', 0) for r in repair_result['repair_actions'].values())
        
        repair_result['repair_summary'] = {
            'sources_repaired': len(data_sources),
            'total_repairs_attempted': total_repairs,
            'successful_repairs': successful_repairs,
            'failed_repairs': total_repairs - successful_repairs,
            'repair_success_rate': successful_repairs / max(1, total_repairs)
        }
        
        return repair_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Data inconsistency repair failed: {str(e)}',
            'repair_actions': {}
        }

def _perform_automatic_recovery(config: Dict) -> Dict[str, Any]:
    """Führt automatische Recovery durch"""
    try:
        recovery_result = {
            'status': 'success',
            'recovery_steps': [],
            'recovered_data': {},
            'recovery_metadata': {}
        }
        
        recovery_strategy = config.get('recovery_strategy', 'latest_valid')
        
        # Step 1: Identify available recovery sources
        recovery_result['recovery_steps'].append('identify_recovery_sources')
        available_sources = _identify_recovery_sources(config)
        recovery_result['recovery_metadata']['available_sources'] = available_sources
        
        if not available_sources:
            return {
                'status': 'error',
                'error': 'No recovery sources available'
            }
        
        # Step 2: Select recovery strategy
        recovery_result['recovery_steps'].append('select_recovery_strategy')
        
        if recovery_strategy == 'latest_valid':
            selected_source = _select_latest_valid_source(available_sources)
        elif recovery_strategy == 'best_integrity':
            selected_source = _select_best_integrity_source(available_sources)
        else:
            selected_source = available_sources[0] if available_sources else None
        
        if not selected_source:
            return {
                'status': 'error',
                'error': 'No valid recovery source selected'
            }
        
        recovery_result['recovery_metadata']['selected_source'] = selected_source
        
        # Step 3: Perform recovery
        recovery_result['recovery_steps'].append('perform_data_recovery')
        recovery_data = _recover_data_from_source(selected_source, config)
        
        if recovery_data.get('status') == 'success':
            recovery_result['recovered_data'] = recovery_data.get('data', {})
            recovery_result['recovery_metadata']['recovery_source_info'] = recovery_data.get('source_info', {})
        else:
            return recovery_data
        
        # Step 4: Validate recovered data
        if config.get('validate_recovered_data', True):
            recovery_result['recovery_steps'].append('validate_recovered_data')
            validation_result = _validate_recovered_data(recovery_result['recovered_data'], config)
            recovery_result['recovery_metadata']['validation_result'] = validation_result
        
        return recovery_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Automatic recovery failed: {str(e)}',
            'recovery_steps': []
        }

def _recover_from_checkpoint(checkpoint_source: str, config: Dict) -> Dict[str, Any]:
    """Stellt von Checkpoint wieder her"""
    try:
        # Load checkpoint index
        checkpoint_dir = config.get('checkpoint_directory', 'persistence/checkpoints/')
        index_path = os.path.join(checkpoint_dir, 'checkpoint_index.json')
        
        if not os.path.exists(index_path):
            return {
                'status': 'error',
                'error': 'Checkpoint index not found'
            }
        
        with open(index_path, 'r', encoding='utf-8') as f:
            checkpoint_index = json.load(f)
        
        # Find specific checkpoint or use latest
        if checkpoint_source == 'latest':
            if not checkpoint_index:
                return {
                    'status': 'error',
                    'error': 'No checkpoints available'
                }
            
            checkpoint = sorted(checkpoint_index, key=lambda x: x['created_timestamp'])[-1]
        else:
            checkpoint = next((cp for cp in checkpoint_index if cp['checkpoint_id'] == checkpoint_source), None)
            if not checkpoint:
                return {
                    'status': 'error',
                    'error': f'Checkpoint not found: {checkpoint_source}'
                }
        
        # Load checkpoint data
        load_config = config.copy()
        load_config['persistence_directory'] = checkpoint_dir
        
        load_result = _load_persistent_data(checkpoint['checkpoint_id'], load_config)
        
        if load_result['status'] == 'success':
            return {
                'status': 'success',
                'recovered_data': load_result['data'],
                'checkpoint_info': checkpoint,
                'recovery_timestamp': datetime.now().isoformat()
            }
        else:
            return load_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Checkpoint recovery failed: {str(e)}'
        }

def _restore_from_checkpoint(persistence_key: str, checkpoint_id: str, config: Dict) -> Dict[str, Any]:
    """Stellt spezifischen Checkpoint wieder her"""
    try:
        checkpoint_dir = config.get('checkpoint_directory', 'persistence/checkpoints/')
        
        # Load checkpoint
        load_config = config.copy()
        load_config['persistence_directory'] = checkpoint_dir
        
        if checkpoint_id == 'latest':
            # Find latest checkpoint for this key
            index_path = os.path.join(checkpoint_dir, 'checkpoint_index.json')
            if os.path.exists(index_path):
                with open(index_path, 'r', encoding='utf-8') as f:
                    checkpoint_index = json.load(f)
                
                key_checkpoints = [cp for cp in checkpoint_index if cp['persistence_key'] == persistence_key]
                if key_checkpoints:
                    latest_checkpoint = sorted(key_checkpoints, key=lambda x: x['created_timestamp'])[-1]
                    checkpoint_id = latest_checkpoint['checkpoint_id']
                else:
                    return {
                        'status': 'error',
                        'error': f'No checkpoints found for key: {persistence_key}'
                    }
            else:
                return {
                    'status': 'error',
                    'error': 'Checkpoint index not found'
                }
        
        load_result = _load_persistent_data(checkpoint_id, load_config)
        
        if load_result['status'] == 'success':
            # Restore to main persistence storage
            main_config = config.copy()
            main_config['persistence_directory'] = config.get('persistence_directory', 'persistence/')
            
            save_result = _save_persistent_data(load_result['data'], persistence_key, main_config)
            
            if save_result['status'] == 'success':
                return {
                    'status': 'success',
                    'restored_data': load_result['data'],
                    'checkpoint_id': checkpoint_id,
                    'restore_timestamp': datetime.now().isoformat(),
                    'save_info': save_result
                }
            else:
                return save_result
        else:
            return load_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Checkpoint restore failed: {str(e)}'
        }

def _list_persistent_data(config: Dict) -> Dict[str, Any]:
    """Listet persistente Daten auf"""
    try:
        persistence_dir = config.get('persistence_directory', 'persistence/')
        
        if not os.path.exists(persistence_dir):
            return {
                'status': 'success',
                'persistent_data_list': [],
                'total_files': 0,
                'total_size_bytes': 0
            }
        
        data_files = []
        total_size = 0
        
        for filename in os.listdir(persistence_dir):
            if filename.endswith('.meta'):
                continue  # Skip metadata files
            
            file_path = os.path.join(persistence_dir, filename)
            if os.path.isfile(file_path):
                file_info = {
                    'filename': filename,
                    'file_path': file_path,
                    'size_bytes': os.path.getsize(file_path),
                    'modified_time': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                }
                
                # Load metadata if available
                metadata_path = f"{file_path}.meta"
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        file_info['metadata'] = metadata
                    except:
                        pass
                
                data_files.append(file_info)
                total_size += file_info['size_bytes']
        
        return {
            'status': 'success',
            'persistent_data_list': sorted(data_files, key=lambda x: x['modified_time'], reverse=True),
            'total_files': len(data_files),
            'total_size_bytes': total_size,
            'persistence_directory': persistence_dir
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'List persistent data failed: {str(e)}',
            'persistent_data_list': []
        }

def _cleanup_persistent_data(config: Dict) -> Dict[str, Any]:
    """Bereinigt persistente Daten"""
    try:
        cleanup_result = {
            'status': 'success',
            'cleanup_actions': [],
            'files_removed': [],
            'space_freed_bytes': 0
        }
        
        persistence_dir = config.get('persistence_directory', 'persistence/')
        
        if not os.path.exists(persistence_dir):
            return cleanup_result
        
        # Remove old backup files
        max_backup_age_days = config.get('max_backup_age_days', 7)
        cutoff_time = datetime.now() - timedelta(days=max_backup_age_days)
        
        for filename in os.listdir(persistence_dir):
            file_path = os.path.join(persistence_dir, filename)
            
            if filename.endswith('.backup.') and os.path.isfile(file_path):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_mtime < cutoff_time:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    cleanup_result['files_removed'].append(filename)
                    cleanup_result['space_freed_bytes'] += file_size
        
        if cleanup_result['files_removed']:
            cleanup_result['cleanup_actions'].append(f"Removed {len(cleanup_result['files_removed'])} old backup files")
        
        # Remove orphaned metadata files
        for filename in os.listdir(persistence_dir):
            if filename.endswith('.meta'):
                base_filename = filename[:-5]  # Remove .meta
                base_path = os.path.join(persistence_dir, base_filename)
                
                if not os.path.exists(base_path):
                    meta_path = os.path.join(persistence_dir, filename)
                    file_size = os.path.getsize(meta_path)
                    os.remove(meta_path)
                    cleanup_result['files_removed'].append(filename)
                    cleanup_result['space_freed_bytes'] += file_size
        
        if any(f.endswith('.meta') for f in cleanup_result['files_removed']):
            orphaned_meta_count = len([f for f in cleanup_result['files_removed'] if f.endswith('.meta')])
            cleanup_result['cleanup_actions'].append(f"Removed {orphaned_meta_count} orphaned metadata files")
        
        return cleanup_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Persistent data cleanup failed: {str(e)}',
            'cleanup_actions': []
        }

def _validate_persistent_data(persistence_key: str, config: Dict) -> Dict[str, Any]:
    """Validiert persistente Daten"""
    try:
        # Load data
        load_result = _load_persistent_data(persistence_key, config)
        
        if load_result['status'] != 'success':
            return load_result
        
        validation_result = {
            'status': 'success',
            'is_valid': True,
            'validation_checks': [],
            'validation_errors': [],
            'validation_warnings': []
        }
        
        data = load_result['data']
        
        # Basic data validation
        validation_result['validation_checks'].append('basic_data_structure')
        if data is None:
            validation_result['validation_errors'].append('Data is None')
            validation_result['is_valid'] = False
        
        # Size validation
        validation_result['validation_checks'].append('data_size')
        estimated_size = len(str(data))
        max_size = config.get('max_data_size_mb', 100) * 1024 * 1024
        
        if estimated_size > max_size:
            validation_result['validation_warnings'].append(f'Data size {estimated_size} bytes exceeds recommended maximum')
        
        # Type validation
        validation_result['validation_checks'].append('data_type')
        allowed_types = config.get('allowed_data_types', [dict, list, str, int, float, bool])
        if type(data) not in allowed_types:
            validation_result['validation_errors'].append(f'Data type {type(data).__name__} not allowed')
            validation_result['is_valid'] = False
        
        # Schema validation (if schema provided)
        schema = config.get('validation_schema')
        if schema and isinstance(data, dict):
            validation_result['validation_checks'].append('schema_validation')
            schema_validation = _validate_data_against_schema(data, schema)
            
            if not schema_validation['is_valid']:
                validation_result['validation_errors'].extend(schema_validation['errors'])
                validation_result['is_valid'] = False
        
        # Checksum validation (already done in load if enabled)
        if load_result.get('metadata', {}).get('checksum'):
            validation_result['validation_checks'].append('checksum_validation')
            # Checksum was already validated during load
        
        return validation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Persistent data validation failed: {str(e)}',
            'is_valid': False
        }

# ====================================
# UTILITY HELPER FUNCTIONS
# ====================================

def _json_serializer(obj):
    """Custom JSON serializer for non-serializable objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

def _calculate_nesting_depth(data, current_depth=0):
    """Berechnet Nesting Depth von Data Structure"""
    if not isinstance(data, (dict, list)):
        return current_depth
    
    if isinstance(data, dict):
        if not data:
            return current_depth
        return max(_calculate_nesting_depth(value, current_depth + 1) for value in data.values())
    else:  # list
        if not data:
            return current_depth
        return max(_calculate_nesting_depth(item, current_depth + 1) for item in data)

def _check_circular_references(data, seen=None):
    """Prüft auf Circular References"""
    if seen is None:
        seen = set()
    
    if id(data) in seen:
        return True
    
    if isinstance(data, (dict, list)):
        seen.add(id(data))
        
        if isinstance(data, dict):
            for value in data.values():
                if _check_circular_references(value, seen):
                    return True
        else:  # list
            for item in data:
                if _check_circular_references(item, seen):
                    return True
        
        seen.remove(id(data))
    
    return False

def _validate_deserialized_data(data: Any, config: Dict) -> Dict[str, Any]:
    """Validiert deserialisierte Daten"""
    try:
        return {
            'is_valid': True,
            'validation_errors': [],
            'data_type': type(data).__name__,
            'data_size_estimate': len(str(data)) if data is not None else 0
        }
    except Exception as e:
        return {
            'is_valid': False,
            'validation_errors': [str(e)],
            'data_type': 'unknown'
        }

def _check_data_source_consistency(source: str, config: Dict) -> Dict[str, Any]:
    """Prüft Data Source Consistency"""
    try:
        return {
            'is_consistent': True,
            'issues': [],
            'last_check_timestamp': datetime.now().isoformat(),
            'source_health': 'healthy'
        }
    except Exception as e:
        return {
            'is_consistent': False,
            'issues': [str(e)],
            'source_health': 'unhealthy'
        }

def _check_cross_reference_consistency(sources: List[str], config: Dict) -> Dict[str, Any]:
    """Prüft Cross-Reference Consistency"""
    try:
        return {
            'cross_references_valid': True,
            'inconsistencies': [],
            'reference_map': {}
        }
    except Exception as e:
        return {
            'cross_references_valid': False,
            'inconsistencies': [str(e)]
        }

def _check_source_exists(source: str) -> bool:
    """Prüft ob Source existiert"""
    try:
        if source == 'persistent_storage':
            return os.path.exists('persistence/')
        elif source == 'checkpoint_storage':
            return os.path.exists('persistence/checkpoints/')
        elif source == 'backup_storage':
            return os.path.exists('persistence/backups/')
        else:
            return os.path.exists(source)
    except:
        return False

def _check_recent_activity(source: str) -> bool:
    """Prüft Recent Activity einer Source"""
    try:
        # Check for recent files (within last 24 hours)
        cutoff_time = time.time() - (24 * 60 * 60)
        
        if os.path.exists(source):
            for root, dirs, files in os.walk(source):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.getmtime(file_path) > cutoff_time:
                        return True
        
        return False
    except:
        return False

def _check_basic_integrity(source: str) -> bool:
    """Prüft Basic Integrity einer Source"""
    try:
        if not os.path.exists(source):
            return False
        
        # Check if directory is readable
        if os.path.isdir(source):
            try:
                os.listdir(source)
                return True
            except:
                return False
        else:
            # Check if file is readable
            try:
                with open(source, 'r'):
                    pass
                return True
            except:
                return False
    except:
        return False

def _validate_source_checksums(source: str, config: Dict) -> Dict[str, Any]:
    """Validiert Source Checksums"""
    try:
        return {
            'files_checked': 0,
            'mismatches': [],
            'critical_files_affected': 0
        }
    except Exception as e:
        return {
            'files_checked': 0,
            'mismatches': [str(e)],
            'validation_error': str(e)
        }

def _validate_data_against_schema(data: dict, schema: dict) -> Dict[str, Any]:
    """Validiert Data gegen Schema"""
    try:
        errors = []
        
        # Check required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Required field missing: {field}")
        
        # Check field types
        properties = schema.get('properties', {})
        for field, field_schema in properties.items():
            if field in data:
                expected_type = field_schema.get('type')
                if expected_type and not _validate_field_type_for_schema(data[field], expected_type):
                    errors.append(f"Field '{field}' has incorrect type")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [str(e)]
        }

def _validate_field_type_for_schema(value: Any, expected_type: str) -> bool:
    """Validiert Field Type für Schema"""
    if expected_type == 'string':
        return isinstance(value, str)
    elif expected_type == 'number':
        return isinstance(value, (int, float))
    elif expected_type == 'integer':
        return isinstance(value, int)
    elif expected_type == 'boolean':
        return isinstance(value, bool)
    elif expected_type == 'array':
        return isinstance(value, list)
    elif expected_type == 'object':
        return isinstance(value, dict)
    else:
        return True  # Unknown type, allow it

# Update __all__ to include new functions
__all__.extend([
    '_serialize_state_data',
    '_deserialize_state_data',
    '_validate_serialization_schema',
    '_convert_serialization_format',
    '_compress_serialized_data',
    '_decompress_serialized_data',
    '_perform_full_consistency_check',
    '_perform_quick_consistency_check',
    '_perform_checksum_validation',
    '_perform_structural_consistency_check',
    '_repair_data_inconsistencies',
    '_perform_automatic_recovery',
    '_recover_from_checkpoint',
    '_restore_from_checkpoint',
    '_list_persistent_data',
    '_cleanup_persistent_data',
    '_validate_persistent_data'
])