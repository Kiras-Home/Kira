"""
Storage Files Module
File Management, Upload/Download, File Operations, Storage Organization
"""

import logging
import os
import shutil
import mimetypes
import hashlib
import threading
import time
import json
import zipfile
import tarfile
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from pathlib import Path
import magic  # For file type detection
from PIL import Image, ImageOps  # For image processing
import subprocess

logger = logging.getLogger(__name__)

# File Management State Tracking
_files_state = {
    'initialization_timestamp': None,
    'last_cleanup_timestamp': None,
    'active_uploads': {},
    'active_downloads': {},
    'file_statistics': {
        'total_files_managed': 0,
        'total_uploads': 0,
        'total_downloads': 0,
        'total_storage_bytes': 0,
        'files_processed_today': 0
    },
    'storage_directories': {
        'uploads': 'storage/uploads/',
        'downloads': 'storage/downloads/',
        'temp': 'storage/temp/',
        'archives': 'storage/archives/',
        'thumbnails': 'storage/thumbnails/'
    },
    'file_type_stats': {},
    'cleanup_enabled': True
}

_files_lock = threading.Lock()

# File cleanup thread
_cleanup_thread = None
_shutdown_event = threading.Event()

# File type configurations
FILE_TYPE_CONFIG = {
    'images': {
        'extensions': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff'],
        'max_size_mb': 50,
        'generate_thumbnails': True,
        'thumbnail_sizes': [(150, 150), (300, 300), (600, 600)]
    },
    'documents': {
        'extensions': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.pages'],
        'max_size_mb': 100,
        'extract_text': True,
        'generate_preview': True
    },
    'videos': {
        'extensions': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'],
        'max_size_mb': 500,
        'generate_thumbnails': True,
        'extract_metadata': True
    },
    'audio': {
        'extensions': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'],
        'max_size_mb': 100,
        'extract_metadata': True
    },
    'archives': {
        'extensions': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
        'max_size_mb': 200,
        'auto_extract': False,
        'scan_contents': True
    },
    'code': {
        'extensions': ['.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml'],
        'max_size_mb': 10,
        'syntax_highlight': True,
        'detect_language': True
    }
}

def handle_file_upload(file_data: Any,
                      upload_config: Dict[str, Any] = None,
                      file_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    File Upload Handler
    
    Umfassendes File Upload Management mit Validation, Processing und Storage
    """
    try:
        if upload_config is None:
            upload_config = {
                'allowed_file_types': ['images', 'documents', 'archives'],
                'max_file_size_mb': 100,
                'auto_organize': True,
                'generate_thumbnails': True,
                'scan_for_viruses': False,
                'extract_metadata': True,
                'create_backup': False
            }
        
        if file_metadata is None:
            file_metadata = {}
        
        # Initialize upload session
        upload_session = {
            'session_id': f"upload_{int(time.time())}",
            'start_time': time.time(),
            'file_metadata': file_metadata,
            'upload_config': upload_config,
            'upload_results': {}
        }
        
        # Track active upload
        with _files_lock:
            _files_state['active_uploads'][upload_session['session_id']] = upload_session
        
        try:
            # Step 1: Validate file upload
            upload_session['upload_results']['validation'] = _validate_file_upload(file_data, upload_config, file_metadata)
            
            if upload_session['upload_results']['validation']['status'] != 'success':
                return _finalize_upload_session(upload_session, success=False)
            
            # Step 2: Process and store file
            upload_session['upload_results']['storage'] = _process_and_store_file(file_data, upload_config, file_metadata)
            
            if upload_session['upload_results']['storage']['status'] != 'success':
                return _finalize_upload_session(upload_session, success=False)
            
            # Step 3: Post-processing (thumbnails, metadata extraction, etc.)
            upload_session['upload_results']['post_processing'] = _post_process_uploaded_file(
                upload_session['upload_results']['storage'], upload_config
            )
            
            # Step 4: Update file index
            upload_session['upload_results']['indexing'] = _update_file_index(upload_session['upload_results']['storage'])
            
        finally:
            # Remove from active uploads
            with _files_lock:
                _files_state['active_uploads'].pop(upload_session['session_id'], None)
        
        return _finalize_upload_session(upload_session, success=True)
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'upload_result': {},
            'upload_summary': {'error': str(e)}
        }

def handle_file_download(file_identifier: str,
                        download_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    File Download Handler
    
    Verwaltet File Downloads mit Access Control und Logging
    """
    try:
        if download_config is None:
            download_config = {
                'check_permissions': True,
                'log_downloads': True,
                'generate_download_link': False,
                'download_timeout_seconds': 300,
                'include_metadata': True
            }
        
        # Initialize download session
        download_session = {
            'session_id': f"download_{int(time.time())}",
            'start_time': time.time(),
            'file_identifier': file_identifier,
            'download_config': download_config,
            'download_results': {}
        }
        
        # Track active download
        with _files_lock:
            _files_state['active_downloads'][download_session['session_id']] = download_session
        
        try:
            # Step 1: Locate file
            download_session['download_results']['file_location'] = _locate_file(file_identifier)
            
            if download_session['download_results']['file_location']['status'] != 'success':
                return _finalize_download_session(download_session, success=False)
            
            # Step 2: Check permissions
            if download_config.get('check_permissions', True):
                download_session['download_results']['permission_check'] = _check_download_permissions(file_identifier)
                
                if not download_session['download_results']['permission_check'].get('allowed', False):
                    return _finalize_download_session(download_session, success=False, error="Download not permitted")
            
            # Step 3: Prepare download
            download_session['download_results']['download_preparation'] = _prepare_file_download(
                download_session['download_results']['file_location'], download_config
            )
            
            if download_session['download_results']['download_preparation']['status'] != 'success':
                return _finalize_download_session(download_session, success=False)
            
            # Step 4: Log download
            if download_config.get('log_downloads', True):
                _log_file_download(file_identifier, download_session)
        
        finally:
            # Remove from active downloads
            with _files_lock:
                _files_state['active_downloads'].pop(download_session['session_id'], None)
        
        return _finalize_download_session(download_session, success=True)
        
    except Exception as e:
        logger.error(f"File download failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'download_result': {},
            'download_summary': {'error': str(e)}
        }

def manage_file_operations(operation_type: str,
                          file_targets: Union[str, List[str]],
                          operation_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    File Operations Manager
    
    Verwaltet verschiedene File Operations (copy, move, delete, rename, etc.)
    """
    try:
        if operation_config is None:
            operation_config = {
                'create_backups': True,
                'validate_operations': True,
                'update_index': True,
                'log_operations': True,
                'handle_conflicts': 'skip'  # skip, overwrite, rename
            }
        
        # Normalize file targets
        if isinstance(file_targets, str):
            file_targets = [file_targets]
        
        # Initialize operation session
        operation_session = {
            'session_id': f"operation_{int(time.time())}",
            'start_time': time.time(),
            'operation_type': operation_type,
            'file_targets': file_targets,
            'operation_config': operation_config,
            'operation_results': {}
        }
        
        # Perform operation based on type
        if operation_type == 'copy':
            operation_session['operation_results'] = _copy_files_operation(file_targets, operation_config)
        
        elif operation_type == 'move':
            operation_session['operation_results'] = _move_files_operation(file_targets, operation_config)
        
        elif operation_type == 'delete':
            operation_session['operation_results'] = _delete_files_operation(file_targets, operation_config)
        
        elif operation_type == 'rename':
            operation_session['operation_results'] = _rename_files_operation(file_targets, operation_config)
        
        elif operation_type == 'duplicate':
            operation_session['operation_results'] = _duplicate_files_operation(file_targets, operation_config)
        
        elif operation_type == 'organize':
            operation_session['operation_results'] = _organize_files_operation(file_targets, operation_config)
        
        elif operation_type == 'compress':
            operation_session['operation_results'] = _compress_files_operation(file_targets, operation_config)
        
        elif operation_type == 'extract':
            operation_session['operation_results'] = _extract_files_operation(file_targets, operation_config)
        
        else:
            operation_session['operation_results'] = {
                'status': 'error',
                'error': f'Unsupported operation type: {operation_type}'
            }
        
        # Add operation metadata
        operation_session.update({
            'end_time': time.time(),
            'operation_success': operation_session['operation_results'].get('status') == 'success',
            'operation_duration_ms': (time.time() - operation_session['start_time']) * 1000
        })
        
        return {
            'success': operation_session['operation_success'],
            'operation_session': operation_session,
            'operation_result': operation_session['operation_results'],
            'operation_summary': {
                'operation_type': operation_type,
                'files_processed': len(file_targets),
                'operation_success': operation_session['operation_success'],
                'operation_duration_ms': operation_session['operation_duration_ms']
            }
        }
        
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'operation_result': {},
            'operation_summary': {'error': str(e)}
        }

def organize_storage_structure(organization_type: str = 'by_type',
                             organization_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Storage Structure Organizer
    
    Organisiert Storage Structure nach verschiedenen Kriterien
    """
    try:
        if organization_config is None:
            organization_config = {
                'source_directory': 'storage/',
                'create_subdirectories': True,
                'move_files': True,
                'update_index': True,
                'backup_before_organize': True,
                'organization_rules': {}
            }
        
        # Initialize organization session
        organization_session = {
            'session_id': f"organize_{int(time.time())}",
            'start_time': time.time(),
            'organization_type': organization_type,
            'organization_config': organization_config,
            'organization_results': {}
        }
        
        # Perform organization based on type
        if organization_type == 'by_type':
            organization_session['organization_results'] = _organize_by_file_type(organization_config)
        
        elif organization_type == 'by_date':
            organization_session['organization_results'] = _organize_by_date(organization_config)
        
        elif organization_type == 'by_size':
            organization_session['organization_results'] = _organize_by_size(organization_config)
        
        elif organization_type == 'by_usage':
            organization_session['organization_results'] = _organize_by_usage(organization_config)
        
        elif organization_type == 'custom':
            organization_session['organization_results'] = _organize_by_custom_rules(organization_config)
        
        elif organization_type == 'cleanup':
            organization_session['organization_results'] = _cleanup_storage_structure(organization_config)
        
        else:
            organization_session['organization_results'] = {
                'status': 'error',
                'error': f'Unsupported organization type: {organization_type}'
            }
        
        # Add organization metadata
        organization_session.update({
            'end_time': time.time(),
            'organization_success': organization_session['organization_results'].get('status') == 'success',
            'organization_duration_ms': (time.time() - organization_session['start_time']) * 1000
        })
        
        return {
            'success': organization_session['organization_success'],
            'organization_session': organization_session,
            'organization_result': organization_session['organization_results'],
            'organization_summary': {
                'organization_type': organization_type,
                'organization_success': organization_session['organization_success'],
                'organization_duration_ms': organization_session['organization_duration_ms']
            }
        }
        
    except Exception as e:
        logger.error(f"Storage organization failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'organization_result': {},
            'organization_summary': {'error': str(e)}
        }

def generate_file_analytics(analytics_type: str = 'comprehensive',
                          analytics_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    File Analytics Generator
    
    Generiert umfassende Analytics über File Storage und Usage
    """
    try:
        if analytics_config is None:
            analytics_config = {
                'include_file_types': True,
                'include_size_distribution': True,
                'include_usage_stats': True,
                'include_storage_efficiency': True,
                'include_growth_trends': True,
                'time_period_days': 30
            }
        
        # Initialize analytics session
        analytics_session = {
            'session_id': f"analytics_{int(time.time())}",
            'start_time': time.time(),
            'analytics_type': analytics_type,
            'analytics_config': analytics_config,
            'analytics_results': {}
        }
        
        # Generate analytics based on type
        if analytics_type == 'comprehensive':
            analytics_session['analytics_results'] = _generate_comprehensive_analytics(analytics_config)
        
        elif analytics_type == 'storage_usage':
            analytics_session['analytics_results'] = _generate_storage_usage_analytics(analytics_config)
        
        elif analytics_type == 'file_types':
            analytics_session['analytics_results'] = _generate_file_type_analytics(analytics_config)
        
        elif analytics_type == 'access_patterns':
            analytics_session['analytics_results'] = _generate_access_pattern_analytics(analytics_config)
        
        elif analytics_type == 'efficiency':
            analytics_session['analytics_results'] = _generate_storage_efficiency_analytics(analytics_config)
        
        elif analytics_type == 'trends':
            analytics_session['analytics_results'] = _generate_storage_trend_analytics(analytics_config)
        
        else:
            analytics_session['analytics_results'] = {
                'status': 'error',
                'error': f'Unsupported analytics type: {analytics_type}'
            }
        
        # Add analytics metadata
        analytics_session.update({
            'end_time': time.time(),
            'analytics_success': analytics_session['analytics_results'].get('status') == 'success',
            'analytics_duration_ms': (time.time() - analytics_session['start_time']) * 1000
        })
        
        return {
            'success': analytics_session['analytics_success'],
            'analytics_session': analytics_session,
            'analytics_result': analytics_session['analytics_results'],
            'analytics_summary': {
                'analytics_type': analytics_type,
                'analytics_success': analytics_session['analytics_success'],
                'analytics_duration_ms': analytics_session['analytics_duration_ms']
            }
        }
        
    except Exception as e:
        logger.error(f"File analytics generation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'analytics_result': {},
            'analytics_summary': {'error': str(e)}
        }

def start_auto_cleanup(cleanup_interval_hours: int = 24,
                      cleanup_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Startet automatische File Cleanup
    """
    try:
        global _cleanup_thread, _shutdown_event
        
        if cleanup_config is None:
            cleanup_config = {
                'remove_temp_files': True,
                'remove_old_thumbnails': True,
                'remove_incomplete_uploads': True,
                'max_temp_file_age_hours': 24,
                'max_thumbnail_age_days': 30,
                'compact_logs': True
            }
        
        # Clear shutdown event
        _shutdown_event.clear()
        
        # Start cleanup thread
        if not (_cleanup_thread and _cleanup_thread.is_alive()):
            _cleanup_thread = threading.Thread(
                target=_cleanup_worker,
                args=(cleanup_interval_hours, cleanup_config),
                daemon=True
            )
            _cleanup_thread.start()
            
            with _files_lock:
                _files_state['cleanup_enabled'] = True
        
        return {
            'success': True,
            'auto_cleanup_started': True,
            'cleanup_interval_hours': cleanup_interval_hours,
            'cleanup_thread_active': _cleanup_thread.is_alive() if _cleanup_thread else False
        }
        
    except Exception as e:
        logger.error(f"Auto cleanup startup failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'auto_cleanup_started': False
        }

def stop_auto_cleanup() -> Dict[str, Any]:
    """
    Stoppt automatische File Cleanup
    """
    try:
        global _cleanup_thread, _shutdown_event
        
        # Signal shutdown
        _shutdown_event.set()
        
        # Wait for thread to finish
        thread_stopped = True
        if _cleanup_thread and _cleanup_thread.is_alive():
            _cleanup_thread.join(timeout=10)
            thread_stopped = not _cleanup_thread.is_alive()
        
        # Update state
        with _files_lock:
            _files_state['cleanup_enabled'] = False
        
        return {
            'success': True,
            'auto_cleanup_stopped': True,
            'cleanup_thread_stopped': thread_stopped
        }
        
    except Exception as e:
        logger.error(f"Auto cleanup shutdown failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'auto_cleanup_stopped': False
        }

def get_files_status() -> Dict[str, Any]:
    """
    Holt aktuellen Files Status
    """
    try:
        with _files_lock:
            status = _files_state.copy()
        
        # Add thread status
        global _cleanup_thread
        status['thread_status'] = {
            'cleanup_thread_active': _cleanup_thread.is_alive() if _cleanup_thread else False
        }
        
        # Add storage directory info
        storage_info = {}
        for dir_name, dir_path in status['storage_directories'].items():
            if os.path.exists(dir_path):
                storage_info[dir_name] = {
                    'exists': True,
                    'total_files': len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]),
                    'total_size_bytes': sum(
                        os.path.getsize(os.path.join(dir_path, f))
                        for f in os.listdir(dir_path)
                        if os.path.isfile(os.path.join(dir_path, f))
                    )
                }
            else:
                storage_info[dir_name] = {
                    'exists': False,
                    'total_files': 0,
                    'total_size_bytes': 0
                }
        
        status['storage_directory_info'] = storage_info
        
        return {
            'success': True,
            'files_status': status,
            'status_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get files status failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'files_status': {},
            'status_timestamp': datetime.now().isoformat()
        }

# ====================================
# PRIVATE HELPER FUNCTIONS - Teil 1
# ====================================

def _validate_file_upload(file_data: Any, config: Dict, metadata: Dict) -> Dict[str, Any]:
    """Validiert File Upload"""
    try:
        validation_result = {
            'status': 'success',
            'validation_errors': [],
            'validation_warnings': [],
            'file_info': {}
        }
        
        # Basic file validation
        if not file_data:
            validation_result['validation_errors'].append('No file data provided')
            validation_result['status'] = 'error'
            return validation_result
        
        # File size validation
        file_size = len(file_data) if isinstance(file_data, bytes) else len(str(file_data))
        max_size_bytes = config.get('max_file_size_mb', 100) * 1024 * 1024
        
        if file_size > max_size_bytes:
            validation_result['validation_errors'].append(f'File size {file_size} exceeds maximum {max_size_bytes}')
            validation_result['status'] = 'error'
        
        validation_result['file_info']['size_bytes'] = file_size
        
        # File type validation
        filename = metadata.get('filename', 'unknown')
        file_extension = os.path.splitext(filename)[1].lower()
        
        allowed_types = config.get('allowed_file_types', ['images', 'documents'])
        type_allowed = False
        
        for file_type in allowed_types:
            if file_type in FILE_TYPE_CONFIG:
                if file_extension in FILE_TYPE_CONFIG[file_type]['extensions']:
                    type_allowed = True
                    validation_result['file_info']['detected_type'] = file_type
                    break
        
        if not type_allowed:
            validation_result['validation_errors'].append(f'File type {file_extension} not allowed')
            validation_result['status'] = 'error'
        
        validation_result['file_info']['filename'] = filename
        validation_result['file_info']['extension'] = file_extension
        
        return validation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File upload validation failed: {str(e)}',
            'validation_errors': [str(e)]
        }

def _process_and_store_file(file_data: Any, config: Dict, metadata: Dict) -> Dict[str, Any]:
    """Verarbeitet und speichert File"""
    try:
        # Ensure upload directory exists
        upload_dir = _files_state['storage_directories']['uploads']
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        filename = metadata.get('filename', 'upload')
        timestamp = int(time.time())
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{timestamp}{ext}"
        
        # Determine storage path
        if config.get('auto_organize', True):
            file_type = _detect_file_type(unique_filename)
            type_dir = os.path.join(upload_dir, file_type)
            os.makedirs(type_dir, exist_ok=True)
            file_path = os.path.join(type_dir, unique_filename)
        else:
            file_path = os.path.join(upload_dir, unique_filename)
        
        # Write file
        if isinstance(file_data, bytes):
            mode = 'wb'
        else:
            mode = 'w'
            if not isinstance(file_data, str):
                file_data = str(file_data)
        
        with open(file_path, mode) as f:
            f.write(file_data)
        
        # Calculate file hash
        file_hash = _calculate_file_hash(file_path)
        
        # Create file metadata
        file_metadata = {
            'original_filename': filename,
            'stored_filename': unique_filename,
            'file_path': file_path,
            'file_size_bytes': os.path.getsize(file_path),
            'file_hash': file_hash,
            'upload_timestamp': datetime.now().isoformat(),
            'file_type': _detect_file_type(unique_filename),
            'mime_type': mimetypes.guess_type(file_path)[0]
        }
        
        # Save metadata
        metadata_path = f"{file_path}.meta"
        with open(metadata_path, 'w') as f:
            json.dump(file_metadata, f, indent=2)
        
        return {
            'status': 'success',
            'file_path': file_path,
            'metadata_path': metadata_path,
            'file_metadata': file_metadata
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File processing and storage failed: {str(e)}'
        }

def _post_process_uploaded_file(storage_result: Dict, config: Dict) -> Dict[str, Any]:
    """Post-Processing für uploaded Files"""
    try:
        if storage_result['status'] != 'success':
            return storage_result
        
        file_path = storage_result['file_path']
        file_metadata = storage_result['file_metadata']
        file_type = file_metadata['file_type']
        
        post_processing_results = {
            'status': 'success',
            'processing_steps': [],
            'generated_files': []
        }
        
        # Generate thumbnails for images
        if file_type == 'images' and config.get('generate_thumbnails', True):
            post_processing_results['processing_steps'].append('generate_thumbnails')
            thumbnail_result = _generate_image_thumbnails(file_path, file_metadata)
            if thumbnail_result.get('thumbnails'):
                post_processing_results['generated_files'].extend(thumbnail_result['thumbnails'])
        
        # Extract metadata for various file types
        if config.get('extract_metadata', True):
            post_processing_results['processing_steps'].append('extract_metadata')
            metadata_result = _extract_file_metadata(file_path, file_type)
            post_processing_results['extracted_metadata'] = metadata_result
        
        # Generate preview for documents
        if file_type == 'documents' and config.get('generate_preview', False):
            post_processing_results['processing_steps'].append('generate_preview')
            preview_result = _generate_document_preview(file_path)
            if preview_result.get('preview_path'):
                post_processing_results['generated_files'].append(preview_result['preview_path'])
        
        # Scan archive contents
        if file_type == 'archives' and config.get('scan_contents', True):
            post_processing_results['processing_steps'].append('scan_archive_contents')
            scan_result = _scan_archive_contents(file_path)
            post_processing_results['archive_contents'] = scan_result
        
        return post_processing_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File post-processing failed: {str(e)}',
            'processing_steps': []
        }

def _finalize_upload_session(upload_session: Dict, success: bool, error: str = None) -> Dict[str, Any]:
    """Finalisiert Upload Session"""
    try:
        upload_session.update({
            'end_time': time.time(),
            'upload_success': success,
            'upload_duration_ms': (time.time() - upload_session['start_time']) * 1000
        })
        
        if error:
            upload_session['error'] = error
        
        # Update statistics
        with _files_lock:
            _files_state['file_statistics']['total_uploads'] += 1
            if success:
                _files_state['file_statistics']['total_files_managed'] += 1
        
        return {
            'success': success,
            'upload_session': upload_session,
            'upload_result': upload_session.get('upload_results', {}),
            'upload_summary': {
                'upload_success': success,
                'upload_duration_ms': upload_session['upload_duration_ms'],
                'error': error
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Upload session finalization failed: {str(e)}',
            'upload_result': {},
            'upload_summary': {'error': str(e)}
        }

def _finalize_download_session(download_session: Dict, success: bool, error: str = None) -> Dict[str, Any]:
    """Finalisiert Download Session"""
    try:
        download_session.update({
            'end_time': time.time(),
            'download_success': success,
            'download_duration_ms': (time.time() - download_session['start_time']) * 1000
        })
        
        if error:
            download_session['error'] = error
        
        # Update statistics
        with _files_lock:
            _files_state['file_statistics']['total_downloads'] += 1
        
        return {
            'success': success,
            'download_session': download_session,
            'download_result': download_session.get('download_results', {}),
            'download_summary': {
                'download_success': success,
                'download_duration_ms': download_session['download_duration_ms'],
                'error': error
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Download session finalization failed: {str(e)}',
            'download_result': {},
            'download_summary': {'error': str(e)}
        }

def _detect_file_type(filename: str) -> str:
    """Detektiert File Type basierend auf Extension"""
    try:
        file_extension = os.path.splitext(filename)[1].lower()
        
        for file_type, config in FILE_TYPE_CONFIG.items():
            if file_extension in config['extensions']:
                return file_type
        
        return 'other'
        
    except Exception as e:
        logger.debug(f"File type detection failed: {e}")
        return 'unknown'

def _calculate_file_hash(file_path: str) -> str:
    """Berechnet File Hash (SHA-256)"""
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.debug(f"File hash calculation failed: {e}")
        return ""

def _cleanup_worker(interval_hours: int, config: Dict):
    """Cleanup Worker Thread"""
    try:
        while not _shutdown_event.is_set():
            # Wait for interval or shutdown signal
            if _shutdown_event.wait(timeout=interval_hours * 3600):
                break  # Shutdown signal received
            
            try:
                # Perform cleanup operations
                logger.debug("Performing file cleanup operations...")
                
                cleanup_result = _perform_file_cleanup(config)
                
                with _files_lock:
                    _files_state['last_cleanup_timestamp'] = datetime.now().isoformat()
                
                logger.debug(f"File cleanup completed: {cleanup_result}")
                
            except Exception as e:
                logger.error(f"File cleanup operation failed: {e}")
        
        logger.info("File cleanup worker thread stopped")
        
    except Exception as e:
        logger.error(f"File cleanup worker thread failed: {e}")

def _initialize_files_component():
    """Initialisiert Files Component"""
    try:
        with _files_lock:
            _files_state['initialization_timestamp'] = datetime.now().isoformat()
        
        # Ensure storage directories exist
        for dir_name, dir_path in _files_state['storage_directories'].items():
            os.makedirs(dir_path, exist_ok=True)
        
        return True
    except Exception as e:
        raise Exception(f"Files component initialization failed: {str(e)}")

# Export all public functions
__all__ = [
    'handle_file_upload',
    'handle_file_download',
    'manage_file_operations',
    'organize_storage_structure',
    'generate_file_analytics',
    'start_auto_cleanup',
    'stop_auto_cleanup',
    'get_files_status',
    '_initialize_files_component'
]
def _locate_file(file_identifier: str) -> Dict[str, Any]:
    """Lokalisiert File basierend auf Identifier"""
    try:
        # Check different storage directories
        storage_dirs = _files_state['storage_directories']
        
        for dir_name, dir_path in storage_dirs.items():
            if not os.path.exists(dir_path):
                continue
            
            # Search by filename
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                
                if filename == file_identifier or filename.startswith(file_identifier):
                    # Load metadata if available
                    metadata_path = f"{file_path}.meta"
                    metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    
                    return {
                        'status': 'success',
                        'file_path': file_path,
                        'filename': filename,
                        'storage_directory': dir_name,
                        'file_size_bytes': os.path.getsize(file_path),
                        'metadata': metadata
                    }
        
        return {
            'status': 'error',
            'error': f'File not found: {file_identifier}'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File location failed: {str(e)}'
        }

def _check_download_permissions(file_identifier: str) -> Dict[str, Any]:
    """Prüft Download Permissions"""
    try:
        # Basic permission check (would be more sophisticated in real implementation)
        return {
            'allowed': True,
            'permission_level': 'full_access',
            'restrictions': [],
            'check_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'allowed': False,
            'error': str(e),
            'permission_level': 'no_access'
        }

def _prepare_file_download(file_location: Dict, config: Dict) -> Dict[str, Any]:
    """Bereitet File Download vor"""
    try:
        file_path = file_location['file_path']
        
        # Verify file exists and is readable
        if not os.path.exists(file_path):
            return {
                'status': 'error',
                'error': 'File no longer exists'
            }
        
        if not os.access(file_path, os.R_OK):
            return {
                'status': 'error',
                'error': 'File not readable'
            }
        
        # Generate download info
        download_info = {
            'status': 'success',
            'file_path': file_path,
            'file_size_bytes': os.path.getsize(file_path),
            'mime_type': mimetypes.guess_type(file_path)[0],
            'download_ready': True
        }
        
        # Generate download link if requested
        if config.get('generate_download_link', False):
            download_token = f"dl_{int(time.time())}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}"
            download_info['download_token'] = download_token
            download_info['download_url'] = f"/api/download/{download_token}"
        
        # Include metadata if requested
        if config.get('include_metadata', True):
            metadata_path = f"{file_path}.meta"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    download_info['file_metadata'] = json.load(f)
        
        return download_info
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Download preparation failed: {str(e)}'
        }

def _log_file_download(file_identifier: str, download_session: Dict):
    """Loggt File Download"""
    try:
        log_entry = {
            'event': 'file_download',
            'file_identifier': file_identifier,
            'session_id': download_session['session_id'],
            'timestamp': datetime.now().isoformat(),
            'file_info': download_session.get('download_results', {}).get('file_location', {})
        }
        
        # Write to download log
        log_dir = 'storage/logs/'
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"downloads_{datetime.now().strftime('%Y%m%d')}.log")
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
    except Exception as e:
        logger.debug(f"Download logging failed: {e}")

def _copy_files_operation(file_targets: List[str], config: Dict) -> Dict[str, Any]:
    """Copy Files Operation"""
    try:
        copy_results = {
            'status': 'success',
            'files_copied': [],
            'copy_errors': [],
            'total_copied': 0,
            'total_errors': 0
        }
        
        destination_dir = config.get('destination_directory', 'storage/copies/')
        os.makedirs(destination_dir, exist_ok=True)
        
        for file_target in file_targets:
            try:
                # Locate source file
                locate_result = _locate_file(file_target)
                if locate_result['status'] != 'success':
                    copy_results['copy_errors'].append({
                        'file_target': file_target,
                        'error': locate_result['error']
                    })
                    continue
                
                source_path = locate_result['file_path']
                filename = os.path.basename(source_path)
                
                # Handle naming conflicts
                dest_path = os.path.join(destination_dir, filename)
                if os.path.exists(dest_path):
                    if config.get('handle_conflicts', 'skip') == 'rename':
                        name, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(destination_dir, f"{name}_copy_{counter}{ext}")
                            counter += 1
                    elif config.get('handle_conflicts', 'skip') == 'skip':
                        copy_results['copy_errors'].append({
                            'file_target': file_target,
                            'error': 'File already exists and conflict handling is set to skip'
                        })
                        continue
                
                # Perform copy
                shutil.copy2(source_path, dest_path)
                
                # Copy metadata if exists
                source_meta = f"{source_path}.meta"
                if os.path.exists(source_meta):
                    dest_meta = f"{dest_path}.meta"
                    shutil.copy2(source_meta, dest_meta)
                
                copy_results['files_copied'].append({
                    'source_path': source_path,
                    'destination_path': dest_path,
                    'filename': filename
                })
                copy_results['total_copied'] += 1
                
            except Exception as e:
                copy_results['copy_errors'].append({
                    'file_target': file_target,
                    'error': str(e)
                })
        
        copy_results['total_errors'] = len(copy_results['copy_errors'])
        
        if copy_results['total_errors'] > 0 and copy_results['total_copied'] == 0:
            copy_results['status'] = 'error'
        elif copy_results['total_errors'] > 0:
            copy_results['status'] = 'partial_success'
        
        return copy_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Copy operation failed: {str(e)}',
            'files_copied': [],
            'copy_errors': []
        }

def _move_files_operation(file_targets: List[str], config: Dict) -> Dict[str, Any]:
    """Move Files Operation"""
    try:
        move_results = {
            'status': 'success',
            'files_moved': [],
            'move_errors': [],
            'total_moved': 0,
            'total_errors': 0
        }
        
        destination_dir = config.get('destination_directory', 'storage/moved/')
        os.makedirs(destination_dir, exist_ok=True)
        
        for file_target in file_targets:
            try:
                # Locate source file
                locate_result = _locate_file(file_target)
                if locate_result['status'] != 'success':
                    move_results['move_errors'].append({
                        'file_target': file_target,
                        'error': locate_result['error']
                    })
                    continue
                
                source_path = locate_result['file_path']
                filename = os.path.basename(source_path)
                
                # Handle naming conflicts
                dest_path = os.path.join(destination_dir, filename)
                if os.path.exists(dest_path):
                    if config.get('handle_conflicts', 'skip') == 'rename':
                        name, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(destination_dir, f"{name}_moved_{counter}{ext}")
                            counter += 1
                    elif config.get('handle_conflicts', 'skip') == 'skip':
                        move_results['move_errors'].append({
                            'file_target': file_target,
                            'error': 'File already exists and conflict handling is set to skip'
                        })
                        continue
                
                # Create backup if requested
                if config.get('create_backups', True):
                    _create_file_backup(source_path)
                
                # Perform move
                shutil.move(source_path, dest_path)
                
                # Move metadata if exists
                source_meta = f"{source_path}.meta"
                if os.path.exists(source_meta):
                    dest_meta = f"{dest_path}.meta"
                    shutil.move(source_meta, dest_meta)
                
                move_results['files_moved'].append({
                    'source_path': source_path,
                    'destination_path': dest_path,
                    'filename': filename
                })
                move_results['total_moved'] += 1
                
            except Exception as e:
                move_results['move_errors'].append({
                    'file_target': file_target,
                    'error': str(e)
                })
        
        move_results['total_errors'] = len(move_results['move_errors'])
        
        if move_results['total_errors'] > 0 and move_results['total_moved'] == 0:
            move_results['status'] = 'error'
        elif move_results['total_errors'] > 0:
            move_results['status'] = 'partial_success'
        
        return move_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Move operation failed: {str(e)}',
            'files_moved': [],
            'move_errors': []
        }

def _delete_files_operation(file_targets: List[str], config: Dict) -> Dict[str, Any]:
    """Delete Files Operation"""
    try:
        delete_results = {
            'status': 'success',
            'files_deleted': [],
            'delete_errors': [],
            'total_deleted': 0,
            'total_errors': 0,
            'backup_info': {}
        }
        
        for file_target in file_targets:
            try:
                # Locate source file
                locate_result = _locate_file(file_target)
                if locate_result['status'] != 'success':
                    delete_results['delete_errors'].append({
                        'file_target': file_target,
                        'error': locate_result['error']
                    })
                    continue
                
                source_path = locate_result['file_path']
                filename = os.path.basename(source_path)
                
                # Create backup if requested
                backup_path = None
                if config.get('create_backups', True):
                    backup_result = _create_file_backup(source_path)
                    if backup_result.get('success'):
                        backup_path = backup_result.get('backup_path')
                
                # Delete metadata first
                source_meta = f"{source_path}.meta"
                if os.path.exists(source_meta):
                    os.remove(source_meta)
                
                # Delete main file
                os.remove(source_path)
                
                delete_results['files_deleted'].append({
                    'source_path': source_path,
                    'filename': filename,
                    'backup_path': backup_path
                })
                delete_results['total_deleted'] += 1
                
            except Exception as e:
                delete_results['delete_errors'].append({
                    'file_target': file_target,
                    'error': str(e)
                })
        
        delete_results['total_errors'] = len(delete_results['delete_errors'])
        
        if delete_results['total_errors'] > 0 and delete_results['total_deleted'] == 0:
            delete_results['status'] = 'error'
        elif delete_results['total_errors'] > 0:
            delete_results['status'] = 'partial_success'
        
        return delete_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Delete operation failed: {str(e)}',
            'files_deleted': [],
            'delete_errors': []
        }

def _rename_files_operation(file_targets: List[str], config: Dict) -> Dict[str, Any]:
    """Rename Files Operation"""
    try:
        rename_results = {
            'status': 'success',
            'files_renamed': [],
            'rename_errors': [],
            'total_renamed': 0,
            'total_errors': 0
        }
        
        rename_pattern = config.get('rename_pattern', '{original_name}_{timestamp}')
        
        for file_target in file_targets:
            try:
                # Locate source file
                locate_result = _locate_file(file_target)
                if locate_result['status'] != 'success':
                    rename_results['rename_errors'].append({
                        'file_target': file_target,
                        'error': locate_result['error']
                    })
                    continue
                
                source_path = locate_result['file_path']
                source_dir = os.path.dirname(source_path)
                original_filename = os.path.basename(source_path)
                name, ext = os.path.splitext(original_filename)
                
                # Generate new filename
                new_name = rename_pattern.format(
                    original_name=name,
                    timestamp=int(time.time()),
                    extension=ext[1:] if ext else '',
                    index=rename_results['total_renamed'] + 1
                )
                
                if not new_name.endswith(ext):
                    new_name += ext
                
                new_path = os.path.join(source_dir, new_name)
                
                # Check if new name already exists
                if os.path.exists(new_path):
                    counter = 1
                    base_new_name = new_name
                    while os.path.exists(new_path):
                        name_part, ext_part = os.path.splitext(base_new_name)
                        new_path = os.path.join(source_dir, f"{name_part}_{counter}{ext_part}")
                        counter += 1
                
                # Create backup if requested
                if config.get('create_backups', True):
                    _create_file_backup(source_path)
                
                # Perform rename
                os.rename(source_path, new_path)
                
                # Rename metadata if exists
                source_meta = f"{source_path}.meta"
                if os.path.exists(source_meta):
                    new_meta = f"{new_path}.meta"
                    os.rename(source_meta, new_meta)
                
                rename_results['files_renamed'].append({
                    'original_path': source_path,
                    'new_path': new_path,
                    'original_filename': original_filename,
                    'new_filename': os.path.basename(new_path)
                })
                rename_results['total_renamed'] += 1
                
            except Exception as e:
                rename_results['rename_errors'].append({
                    'file_target': file_target,
                    'error': str(e)
                })
        
        rename_results['total_errors'] = len(rename_results['rename_errors'])
        
        if rename_results['total_errors'] > 0 and rename_results['total_renamed'] == 0:
            rename_results['status'] = 'error'
        elif rename_results['total_errors'] > 0:
            rename_results['status'] = 'partial_success'
        
        return rename_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Rename operation failed: {str(e)}',
            'files_renamed': [],
            'rename_errors': []
        }

def _duplicate_files_operation(file_targets: List[str], config: Dict) -> Dict[str, Any]:
    """Duplicate Files Operation"""
    try:
        duplicate_results = {
            'status': 'success',
            'files_duplicated': [],
            'duplicate_errors': [],
            'total_duplicated': 0,
            'total_errors': 0
        }
        
        duplicate_suffix = config.get('duplicate_suffix', '_copy')
        
        for file_target in file_targets:
            try:
                # Locate source file
                locate_result = _locate_file(file_target)
                if locate_result['status'] != 'success':
                    duplicate_results['duplicate_errors'].append({
                        'file_target': file_target,
                        'error': locate_result['error']
                    })
                    continue
                
                source_path = locate_result['file_path']
                source_dir = os.path.dirname(source_path)
                original_filename = os.path.basename(source_path)
                name, ext = os.path.splitext(original_filename)
                
                # Generate duplicate filename
                duplicate_filename = f"{name}{duplicate_suffix}{ext}"
                duplicate_path = os.path.join(source_dir, duplicate_filename)
                
                # Handle naming conflicts
                counter = 1
                base_duplicate_path = duplicate_path
                while os.path.exists(duplicate_path):
                    duplicate_filename = f"{name}{duplicate_suffix}_{counter}{ext}"
                    duplicate_path = os.path.join(source_dir, duplicate_filename)
                    counter += 1
                
                # Perform duplication (copy)
                shutil.copy2(source_path, duplicate_path)
                
                # Duplicate metadata if exists
                source_meta = f"{source_path}.meta"
                if os.path.exists(source_meta):
                    duplicate_meta = f"{duplicate_path}.meta"
                    shutil.copy2(source_meta, duplicate_meta)
                    
                    # Update metadata for duplicate
                    with open(duplicate_meta, 'r') as f:
                        meta_data = json.load(f)
                    
                    meta_data['duplicate_of'] = source_path
                    meta_data['duplicate_timestamp'] = datetime.now().isoformat()
                    meta_data['stored_filename'] = duplicate_filename
                    
                    with open(duplicate_meta, 'w') as f:
                        json.dump(meta_data, f, indent=2)
                
                duplicate_results['files_duplicated'].append({
                    'original_path': source_path,
                    'duplicate_path': duplicate_path,
                    'original_filename': original_filename,
                    'duplicate_filename': duplicate_filename
                })
                duplicate_results['total_duplicated'] += 1
                
            except Exception as e:
                duplicate_results['duplicate_errors'].append({
                    'file_target': file_target,
                    'error': str(e)
                })
        
        duplicate_results['total_errors'] = len(duplicate_results['duplicate_errors'])
        
        if duplicate_results['total_errors'] > 0 and duplicate_results['total_duplicated'] == 0:
            duplicate_results['status'] = 'error'
        elif duplicate_results['total_errors'] > 0:
            duplicate_results['status'] = 'partial_success'
        
        return duplicate_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Duplicate operation failed: {str(e)}',
            'files_duplicated': [],
            'duplicate_errors': []
        }

def _create_file_backup(file_path: str) -> Dict[str, Any]:
    """Erstellt File Backup"""
    try:
        backup_dir = _files_state['storage_directories'].get('temp', 'storage/temp/') + 'backups/'
        os.makedirs(backup_dir, exist_ok=True)
        
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        
        backup_filename = f"{name}_backup_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy file to backup location
        shutil.copy2(file_path, backup_path)
        
        # Copy metadata if exists
        source_meta = f"{file_path}.meta"
        if os.path.exists(source_meta):
            backup_meta = f"{backup_path}.meta"
            shutil.copy2(source_meta, backup_meta)
        
        return {
            'success': True,
            'backup_path': backup_path,
            'backup_filename': backup_filename,
            'backup_timestamp': timestamp
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'File backup failed: {str(e)}'
        }

def _update_file_index(storage_result: Dict) -> Dict[str, Any]:
    """Aktualisiert File Index"""
    try:
        if storage_result.get('status') != 'success':
            return storage_result
        
        # Load existing file index
        index_path = 'storage/file_index.json'
        file_index = {}
        
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                file_index = json.load(f)
        
        # Add new file to index
        file_metadata = storage_result.get('file_metadata', {})
        file_id = file_metadata.get('file_hash', f"file_{int(time.time())}")
        
        file_index[file_id] = {
            'file_metadata': file_metadata,
            'index_timestamp': datetime.now().isoformat(),
            'file_path': storage_result.get('file_path', ''),
            'metadata_path': storage_result.get('metadata_path', '')
        }
        
        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(file_index, f, indent=2)
        
        return {
            'status': 'success',
            'file_id': file_id,
            'index_updated': True,
            'total_indexed_files': len(file_index)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File index update failed: {str(e)}'
        }

# Update __all__ to include new helper functions
__all__.extend([
    '_locate_file',
    '_check_download_permissions',
    '_prepare_file_download',
    '_copy_files_operation',
    '_move_files_operation',
    '_delete_files_operation',
    '_rename_files_operation',
    '_duplicate_files_operation'
])

def _organize_files_operation(file_targets: List[str], config: Dict) -> Dict[str, Any]:
    """Organize Files Operation"""
    try:
        organize_results = {
            'status': 'success',
            'files_organized': [],
            'organize_errors': [],
            'total_organized': 0,
            'total_errors': 0,
            'organization_structure': {}
        }
        
        organization_strategy = config.get('organization_strategy', 'by_type')
        base_dir = config.get('organization_directory', 'storage/organized/')
        os.makedirs(base_dir, exist_ok=True)
        
        for file_target in file_targets:
            try:
                # Locate source file
                locate_result = _locate_file(file_target)
                if locate_result['status'] != 'success':
                    organize_results['organize_errors'].append({
                        'file_target': file_target,
                        'error': locate_result['error']
                    })
                    continue
                
                source_path = locate_result['file_path']
                filename = os.path.basename(source_path)
                
                # Determine organization directory
                if organization_strategy == 'by_type':
                    file_type = _detect_file_type(filename)
                    org_dir = os.path.join(base_dir, file_type)
                elif organization_strategy == 'by_date':
                    file_date = datetime.fromtimestamp(os.path.getmtime(source_path))
                    org_dir = os.path.join(base_dir, file_date.strftime('%Y/%m'))
                elif organization_strategy == 'by_size':
                    file_size = os.path.getsize(source_path)
                    if file_size < 1024 * 1024:  # < 1MB
                        org_dir = os.path.join(base_dir, 'small')
                    elif file_size < 10 * 1024 * 1024:  # < 10MB
                        org_dir = os.path.join(base_dir, 'medium')
                    else:
                        org_dir = os.path.join(base_dir, 'large')
                else:
                    org_dir = os.path.join(base_dir, 'other')
                
                os.makedirs(org_dir, exist_ok=True)
                
                # Move file to organized location
                dest_path = os.path.join(org_dir, filename)
                
                # Handle naming conflicts
                if os.path.exists(dest_path):
                    name, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(org_dir, f"{name}_{counter}{ext}")
                        counter += 1
                
                shutil.move(source_path, dest_path)
                
                # Move metadata
                source_meta = f"{source_path}.meta"
                if os.path.exists(source_meta):
                    dest_meta = f"{dest_path}.meta"
                    shutil.move(source_meta, dest_meta)
                
                # Track organization structure
                org_category = os.path.basename(org_dir)
                if org_category not in organize_results['organization_structure']:
                    organize_results['organization_structure'][org_category] = []
                organize_results['organization_structure'][org_category].append(filename)
                
                organize_results['files_organized'].append({
                    'source_path': source_path,
                    'destination_path': dest_path,
                    'organization_category': org_category,
                    'filename': filename
                })
                organize_results['total_organized'] += 1
                
            except Exception as e:
                organize_results['organize_errors'].append({
                    'file_target': file_target,
                    'error': str(e)
                })
        
        organize_results['total_errors'] = len(organize_results['organize_errors'])
        
        if organize_results['total_errors'] > 0 and organize_results['total_organized'] == 0:
            organize_results['status'] = 'error'
        elif organize_results['total_errors'] > 0:
            organize_results['status'] = 'partial_success'
        
        return organize_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Organize operation failed: {str(e)}',
            'files_organized': [],
            'organize_errors': []
        }

def _compress_files_operation(file_targets: List[str], config: Dict) -> Dict[str, Any]:
    """Compress Files Operation"""
    try:
        compress_results = {
            'status': 'success',
            'compressed_archives': [],
            'compress_errors': [],
            'total_compressed': 0,
            'total_errors': 0,
            'compression_stats': {}
        }
        
        compression_format = config.get('compression_format', 'zip')  # zip, tar, tar.gz
        archive_name = config.get('archive_name', f'archive_{int(time.time())}')
        archive_dir = config.get('archive_directory', _files_state['storage_directories']['archives'])
        os.makedirs(archive_dir, exist_ok=True)
        
        # Collect all source files
        source_files = []
        total_size_before = 0
        
        for file_target in file_targets:
            try:
                locate_result = _locate_file(file_target)
                if locate_result['status'] == 'success':
                    source_path = locate_result['file_path']
                    source_files.append(source_path)
                    total_size_before += os.path.getsize(source_path)
                else:
                    compress_results['compress_errors'].append({
                        'file_target': file_target,
                        'error': locate_result['error']
                    })
            except Exception as e:
                compress_results['compress_errors'].append({
                    'file_target': file_target,
                    'error': str(e)
                })
        
        if not source_files:
            return {
                'status': 'error',
                'error': 'No valid files found for compression',
                'compressed_archives': [],
                'compress_errors': compress_results['compress_errors']
            }
        
        # Create archive
        if compression_format == 'zip':
            archive_path = os.path.join(archive_dir, f"{archive_name}.zip")
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for source_file in source_files:
                    arcname = os.path.basename(source_file)
                    zipf.write(source_file, arcname)
        
        elif compression_format in ['tar', 'tar.gz']:
            extension = '.tar.gz' if compression_format == 'tar.gz' else '.tar'
            archive_path = os.path.join(archive_dir, f"{archive_name}{extension}")
            mode = 'w:gz' if compression_format == 'tar.gz' else 'w'
            
            with tarfile.open(archive_path, mode) as tarf:
                for source_file in source_files:
                    arcname = os.path.basename(source_file)
                    tarf.add(source_file, arcname=arcname)
        
        else:
            return {
                'status': 'error',
                'error': f'Unsupported compression format: {compression_format}',
                'compressed_archives': [],
                'compress_errors': []
            }
        
        # Calculate compression statistics
        total_size_after = os.path.getsize(archive_path)
        compression_ratio = (total_size_before - total_size_after) / total_size_before if total_size_before > 0 else 0
        
        compress_results['compression_stats'] = {
            'original_size_bytes': total_size_before,
            'compressed_size_bytes': total_size_after,
            'compression_ratio': compression_ratio,
            'space_saved_bytes': total_size_before - total_size_after,
            'files_in_archive': len(source_files)
        }
        
        # Create archive metadata
        archive_metadata = {
            'archive_name': archive_name,
            'compression_format': compression_format,
            'created_timestamp': datetime.now().isoformat(),
            'source_files': [os.path.basename(f) for f in source_files],
            'compression_stats': compress_results['compression_stats']
        }
        
        metadata_path = f"{archive_path}.meta"
        with open(metadata_path, 'w') as f:
            json.dump(archive_metadata, f, indent=2)
        
        compress_results['compressed_archives'].append({
            'archive_path': archive_path,
            'metadata_path': metadata_path,
            'archive_metadata': archive_metadata
        })
        compress_results['total_compressed'] = 1
        compress_results['total_errors'] = len(compress_results['compress_errors'])
        
        return compress_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Compress operation failed: {str(e)}',
            'compressed_archives': [],
            'compress_errors': []
        }

def _extract_files_operation(file_targets: List[str], config: Dict) -> Dict[str, Any]:
    """Extract Files Operation"""
    try:
        extract_results = {
            'status': 'success',
            'extracted_files': [],
            'extract_errors': [],
            'total_extracted': 0,
            'total_errors': 0,
            'extraction_info': {}
        }
        
        extract_dir = config.get('extract_directory', 'storage/extracted/')
        os.makedirs(extract_dir, exist_ok=True)
        
        for file_target in file_targets:
            try:
                # Locate archive file
                locate_result = _locate_file(file_target)
                if locate_result['status'] != 'success':
                    extract_results['extract_errors'].append({
                        'file_target': file_target,
                        'error': locate_result['error']
                    })
                    continue
                
                archive_path = locate_result['file_path']
                filename = os.path.basename(archive_path)
                name, ext = os.path.splitext(filename)
                
                # Create extraction subdirectory
                target_extract_dir = os.path.join(extract_dir, name)
                os.makedirs(target_extract_dir, exist_ok=True)
                
                extracted_file_list = []
                
                # Extract based on file type
                if ext.lower() == '.zip':
                    with zipfile.ZipFile(archive_path, 'r') as zipf:
                        zipf.extractall(target_extract_dir)
                        extracted_file_list = zipf.namelist()
                
                elif ext.lower() in ['.tar', '.gz'] or filename.endswith('.tar.gz'):
                    with tarfile.open(archive_path, 'r:*') as tarf:
                        tarf.extractall(target_extract_dir)
                        extracted_file_list = tarf.getnames()
                
                else:
                    extract_results['extract_errors'].append({
                        'file_target': file_target,
                        'error': f'Unsupported archive format: {ext}'
                    })
                    continue
                
                extract_results['extracted_files'].append({
                    'archive_path': archive_path,
                    'extract_directory': target_extract_dir,
                    'extracted_files': extracted_file_list,
                    'files_count': len(extracted_file_list)
                })
                extract_results['total_extracted'] += len(extracted_file_list)
                
            except Exception as e:
                extract_results['extract_errors'].append({
                    'file_target': file_target,
                    'error': str(e)
                })
        
        extract_results['total_errors'] = len(extract_results['extract_errors'])
        
        if extract_results['total_errors'] > 0 and extract_results['total_extracted'] == 0:
            extract_results['status'] = 'error'
        elif extract_results['total_errors'] > 0:
            extract_results['status'] = 'partial_success'
        
        return extract_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Extract operation failed: {str(e)}',
            'extracted_files': [],
            'extract_errors': []
        }

def _organize_by_file_type(config: Dict) -> Dict[str, Any]:
    """Organisiert Files nach Type"""
    try:
        source_dir = config.get('source_directory', 'storage/')
        organization_results = {
            'status': 'success',
            'organized_files': {},
            'organization_errors': [],
            'total_organized': 0
        }
        
        # Scan all files in source directory
        for root, dirs, files in os.walk(source_dir):
            for filename in files:
                if filename.endswith('.meta'):
                    continue  # Skip metadata files
                
                try:
                    file_path = os.path.join(root, filename)
                    file_type = _detect_file_type(filename)
                    
                    # Create type directory
                    type_dir = os.path.join(source_dir, 'organized', file_type)
                    os.makedirs(type_dir, exist_ok=True)
                    
                    # Move file
                    dest_path = os.path.join(type_dir, filename)
                    if file_path != dest_path and not os.path.exists(dest_path):
                        shutil.move(file_path, dest_path)
                        
                        # Move metadata if exists
                        meta_source = f"{file_path}.meta"
                        if os.path.exists(meta_source):
                            meta_dest = f"{dest_path}.meta"
                            shutil.move(meta_source, meta_dest)
                        
                        if file_type not in organization_results['organized_files']:
                            organization_results['organized_files'][file_type] = []
                        organization_results['organized_files'][file_type].append(filename)
                        organization_results['total_organized'] += 1
                
                except Exception as e:
                    organization_results['organization_errors'].append({
                        'filename': filename,
                        'error': str(e)
                    })
        
        return organization_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Organization by file type failed: {str(e)}',
            'organized_files': {},
            'organization_errors': []
        }

def _organize_by_date(config: Dict) -> Dict[str, Any]:
    """Organisiert Files nach Datum"""
    try:
        source_dir = config.get('source_directory', 'storage/')
        organization_results = {
            'status': 'success',
            'organized_files': {},
            'organization_errors': [],
            'total_organized': 0
        }
        
        for root, dirs, files in os.walk(source_dir):
            for filename in files:
                if filename.endswith('.meta'):
                    continue
                
                try:
                    file_path = os.path.join(root, filename)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    date_str = file_mtime.strftime('%Y-%m')
                    
                    # Create date directory
                    date_dir = os.path.join(source_dir, 'organized', date_str)
                    os.makedirs(date_dir, exist_ok=True)
                    
                    # Move file
                    dest_path = os.path.join(date_dir, filename)
                    if file_path != dest_path and not os.path.exists(dest_path):
                        shutil.move(file_path, dest_path)
                        
                        if date_str not in organization_results['organized_files']:
                            organization_results['organized_files'][date_str] = []
                        organization_results['organized_files'][date_str].append(filename)
                        organization_results['total_organized'] += 1
                
                except Exception as e:
                    organization_results['organization_errors'].append({
                        'filename': filename,
                        'error': str(e)
                    })
        
        return organization_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Organization by date failed: {str(e)}',
            'organized_files': {},
            'organization_errors': []
        }

def _organize_by_size(config: Dict) -> Dict[str, Any]:
    """Organisiert Files nach Größe"""
    try:
        source_dir = config.get('source_directory', 'storage/')
        organization_results = {
            'status': 'success',
            'organized_files': {},
            'organization_errors': [],
            'total_organized': 0
        }
        
        size_categories = {
            'tiny': (0, 1024),  # < 1KB
            'small': (1024, 1024*1024),  # 1KB - 1MB
            'medium': (1024*1024, 10*1024*1024),  # 1MB - 10MB
            'large': (10*1024*1024, 100*1024*1024),  # 10MB - 100MB
            'huge': (100*1024*1024, float('inf'))  # > 100MB
        }
        
        for root, dirs, files in os.walk(source_dir):
            for filename in files:
                if filename.endswith('.meta'):
                    continue
                
                try:
                    file_path = os.path.join(root, filename)
                    file_size = os.path.getsize(file_path)
                    
                    # Determine size category
                    size_category = 'unknown'
                    for category, (min_size, max_size) in size_categories.items():
                        if min_size <= file_size < max_size:
                            size_category = category
                            break
                    
                    # Create size directory
                    size_dir = os.path.join(source_dir, 'organized', size_category)
                    os.makedirs(size_dir, exist_ok=True)
                    
                    # Move file
                    dest_path = os.path.join(size_dir, filename)
                    if file_path != dest_path and not os.path.exists(dest_path):
                        shutil.move(file_path, dest_path)
                        
                        if size_category not in organization_results['organized_files']:
                            organization_results['organized_files'][size_category] = []
                        organization_results['organized_files'][size_category].append(filename)
                        organization_results['total_organized'] += 1
                
                except Exception as e:
                    organization_results['organization_errors'].append({
                        'filename': filename,
                        'error': str(e)
                    })
        
        return organization_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Organization by size failed: {str(e)}',
            'organized_files': {},
            'organization_errors': []
        }

def _organize_by_usage(config: Dict) -> Dict[str, Any]:
    """Organisiert Files nach Usage Pattern"""
    try:
        # Simplified usage-based organization
        return {
            'status': 'success',
            'organized_files': {'recent': [], 'old': []},
            'organization_errors': [],
            'total_organized': 0,
            'note': 'Usage-based organization requires access logs'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Organization by usage failed: {str(e)}',
            'organized_files': {},
            'organization_errors': []
        }

def _organize_by_custom_rules(config: Dict) -> Dict[str, Any]:
    """Organisiert Files nach Custom Rules"""
    try:
        custom_rules = config.get('organization_rules', {})
        if not custom_rules:
            return {
                'status': 'error',
                'error': 'No custom organization rules provided',
                'organized_files': {},
                'organization_errors': []
            }
        
        # Apply custom organization rules
        organization_results = {
            'status': 'success',
            'organized_files': {},
            'organization_errors': [],
            'total_organized': 0
        }
        
        # This would implement custom rule logic
        # For now, return placeholder
        return organization_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Custom organization failed: {str(e)}',
            'organized_files': {},
            'organization_errors': []
        }

def _cleanup_storage_structure(config: Dict) -> Dict[str, Any]:
    """Bereinigt Storage Structure"""
    try:
        cleanup_results = {
            'status': 'success',
            'cleanup_actions': [],
            'cleaned_items': [],
            'space_freed_bytes': 0,
            'cleanup_errors': []
        }
        
        source_dir = config.get('source_directory', 'storage/')
        
        # Remove empty directories
        for root, dirs, files in os.walk(source_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # Empty directory
                        os.rmdir(dir_path)
                        cleanup_results['cleaned_items'].append(f"Empty directory: {dir_path}")
                        cleanup_results['cleanup_actions'].append('remove_empty_directories')
                except Exception as e:
                    cleanup_results['cleanup_errors'].append({
                        'item': dir_path,
                        'error': str(e)
                    })
        
        # Remove duplicate files
        file_hashes = {}
        for root, dirs, files in os.walk(source_dir):
            for filename in files:
                if filename.endswith('.meta'):
                    continue
                
                try:
                    file_path = os.path.join(root, filename)
                    file_hash = _calculate_file_hash(file_path)
                    
                    if file_hash in file_hashes:
                        # Duplicate found
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        
                        # Remove metadata if exists
                        meta_path = f"{file_path}.meta"
                        if os.path.exists(meta_path):
                            os.remove(meta_path)
                        
                        cleanup_results['cleaned_items'].append(f"Duplicate file: {filename}")
                        cleanup_results['space_freed_bytes'] += file_size
                        
                        if 'remove_duplicates' not in cleanup_results['cleanup_actions']:
                            cleanup_results['cleanup_actions'].append('remove_duplicates')
                    else:
                        file_hashes[file_hash] = file_path
                
                except Exception as e:
                    cleanup_results['cleanup_errors'].append({
                        'item': filename,
                        'error': str(e)
                    })
        
        return cleanup_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Storage cleanup failed: {str(e)}',
            'cleanup_actions': [],
            'cleaned_items': []
        }

def _generate_comprehensive_analytics(config: Dict) -> Dict[str, Any]:
    """Generiert umfassende File Analytics"""
    try:
        analytics_results = {
            'status': 'success',
            'file_type_distribution': {},
            'size_distribution': {},
            'storage_usage': {},
            'access_patterns': {},
            'growth_trends': {},
            'efficiency_metrics': {}
        }
        
        # Scan all storage directories
        total_files = 0
        total_size = 0
        file_type_counts = {}
        
        storage_dirs = _files_state['storage_directories']
        
        for dir_name, dir_path in storage_dirs.items():
            if not os.path.exists(dir_path):
                continue
            
            dir_stats = {
                'file_count': 0,
                'total_size_bytes': 0,
                'file_types': {}
            }
            
            for root, dirs, files in os.walk(dir_path):
                for filename in files:
                    if filename.endswith('.meta'):
                        continue
                    
                    try:
                        file_path = os.path.join(root, filename)
                        file_size = os.path.getsize(file_path)
                        file_type = _detect_file_type(filename)
                        
                        # Update totals
                        total_files += 1
                        total_size += file_size
                        
                        # Update directory stats
                        dir_stats['file_count'] += 1
                        dir_stats['total_size_bytes'] += file_size
                        
                        if file_type not in dir_stats['file_types']:
                            dir_stats['file_types'][file_type] = 0
                        dir_stats['file_types'][file_type] += 1
                        
                        # Update global file type counts
                        if file_type not in file_type_counts:
                            file_type_counts[file_type] = 0
                        file_type_counts[file_type] += 1
                    
                    except Exception as e:
                        logger.debug(f"Analytics file processing error: {e}")
            
            analytics_results['storage_usage'][dir_name] = dir_stats
        
        # File type distribution
        analytics_results['file_type_distribution'] = {
            'total_files': total_files,
            'type_counts': file_type_counts,
            'type_percentages': {
                file_type: (count / total_files * 100) if total_files > 0 else 0
                for file_type, count in file_type_counts.items()
            }
        }
        
        # Size distribution
        size_categories = {
            'tiny': 0, 'small': 0, 'medium': 0, 'large': 0, 'huge': 0
        }
        
        analytics_results['size_distribution'] = {
            'total_size_bytes': total_size,
            'average_file_size_bytes': total_size / total_files if total_files > 0 else 0,
            'size_categories': size_categories
        }
        
        # Efficiency metrics
        analytics_results['efficiency_metrics'] = {
            'storage_efficiency_score': 0.85,  # Placeholder
            'deduplication_potential': 0.15,   # Placeholder
            'compression_potential': 0.30      # Placeholder
        }
        
        return analytics_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Comprehensive analytics generation failed: {str(e)}',
            'file_type_distribution': {},
            'size_distribution': {}
        }

def _generate_image_thumbnails(file_path: str, file_metadata: Dict) -> Dict[str, Any]:
    """Generiert Image Thumbnails"""
    try:
        thumbnail_results = {
            'status': 'success',
            'thumbnails': [],
            'thumbnail_errors': []
        }
        
        file_type = file_metadata.get('file_type', '')
        if file_type != 'images':
            return {
                'status': 'skip',
                'reason': 'Not an image file'
            }
        
        thumbnail_dir = _files_state['storage_directories']['thumbnails']
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        
        # Generate thumbnails for different sizes
        thumbnail_sizes = FILE_TYPE_CONFIG['images']['thumbnail_sizes']
        
        try:
            with Image.open(file_path) as img:
                # Auto-orient based on EXIF data
                img = ImageOps.exif_transpose(img)
                
                for size in thumbnail_sizes:
                    try:
                        thumbnail_name = f"{name}_thumb_{size[0]}x{size[1]}.jpg"
                        thumbnail_path = os.path.join(thumbnail_dir, thumbnail_name)
                        
                        # Create thumbnail
                        thumb_img = img.copy()
                        thumb_img.thumbnail(size, Image.Resampling.LANCZOS)
                        
                        # Convert to RGB if necessary (for JPEG)
                        if thumb_img.mode in ('RGBA', 'P'):
                            thumb_img = thumb_img.convert('RGB')
                        
                        thumb_img.save(thumbnail_path, 'JPEG', quality=85)
                        
                        thumbnail_results['thumbnails'].append({
                            'size': size,
                            'thumbnail_path': thumbnail_path,
                            'thumbnail_name': thumbnail_name
                        })
                    
                    except Exception as e:
                        thumbnail_results['thumbnail_errors'].append({
                            'size': size,
                            'error': str(e)
                        })
        
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Image processing failed: {str(e)}',
                'thumbnails': []
            }
        
        return thumbnail_results
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Thumbnail generation failed: {str(e)}',
            'thumbnails': []
        }

def _extract_file_metadata(file_path: str, file_type: str) -> Dict[str, Any]:
    """Extrahiert File Metadata"""
    try:
        metadata_result = {
            'status': 'success',
            'extracted_metadata': {},
            'metadata_type': file_type
        }
        
        # Basic file information
        stat_info = os.stat(file_path)
        metadata_result['extracted_metadata'] = {
            'file_size_bytes': stat_info.st_size,
            'created_timestamp': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            'modified_timestamp': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            'accessed_timestamp': datetime.fromtimestamp(stat_info.st_atime).isoformat()
        }
        
        # Type-specific metadata extraction
        if file_type == 'images':
            try:
                with Image.open(file_path) as img:
                    metadata_result['extracted_metadata'].update({
                        'image_format': img.format,
                        'image_mode': img.mode,
                        'image_size': img.size,
                        'image_has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                    })
                    
                    # EXIF data if available
                    if hasattr(img, '_getexif') and img._getexif():
                        metadata_result['extracted_metadata']['exif_data'] = dict(img._getexif())
            except Exception as e:
                metadata_result['extracted_metadata']['image_metadata_error'] = str(e)
        
        return metadata_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Metadata extraction failed: {str(e)}',
            'extracted_metadata': {}
        }

def _scan_archive_contents(file_path: str) -> Dict[str, Any]:
    """Scannt Archive Contents"""
    try:
        scan_result = {
            'status': 'success',
            'archive_type': '',
            'contents': [],
            'total_files': 0,
            'total_size_compressed': os.path.getsize(file_path)
        }
        
        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(filename)
        
        if ext.lower() == '.zip':
            scan_result['archive_type'] = 'zip'
            with zipfile.ZipFile(file_path, 'r') as zipf:
                for info in zipf.infolist():
                    scan_result['contents'].append({
                        'filename': info.filename,
                        'file_size': info.file_size,
                        'compressed_size': info.compress_size,
                        'is_directory': info.is_dir()
                    })
                scan_result['total_files'] = len(scan_result['contents'])
        
        elif ext.lower() in ['.tar', '.gz'] or filename.endswith('.tar.gz'):
            scan_result['archive_type'] = 'tar'
            with tarfile.open(file_path, 'r:*') as tarf:
                for member in tarf.getmembers():
                    scan_result['contents'].append({
                        'filename': member.name,
                        'file_size': member.size,
                        'is_directory': member.isdir(),
                        'is_file': member.isfile()
                    })
                scan_result['total_files'] = len(scan_result['contents'])
        
        else:
            return {
                'status': 'error',
                'error': f'Unsupported archive format: {ext}'
            }
        
        return scan_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Archive scanning failed: {str(e)}',
            'contents': []
        }

def _perform_file_cleanup(config: Dict) -> Dict[str, Any]:
    """Führt File Cleanup durch"""
    try:
        cleanup_result = {
            'status': 'success',
            'cleanup_actions': [],
            'files_removed': [],
            'space_freed_bytes': 0
        }
        
        # Remove old temporary files
        if config.get('remove_temp_files', True):
            temp_dir = _files_state['storage_directories']['temp']
            max_age_hours = config.get('max_temp_file_age_hours', 24)
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            if os.path.exists(temp_dir):
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        cleanup_result['files_removed'].append(filename)
                        cleanup_result['space_freed_bytes'] += file_size
                
                if cleanup_result['files_removed']:
                    cleanup_result['cleanup_actions'].append('remove_temp_files')
        
        # Remove old thumbnails
        if config.get('remove_old_thumbnails', True):
            thumbnail_dir = _files_state['storage_directories']['thumbnails']
            max_age_days = config.get('max_thumbnail_age_days', 30)
            cutoff_time = time.time() - (max_age_days * 24 * 3600)
            
            if os.path.exists(thumbnail_dir):
                for filename in os.listdir(thumbnail_dir):
                    if filename.startswith('thumb_'):
                        file_path = os.path.join(thumbnail_dir, filename)
                        if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleanup_result['files_removed'].append(filename)
                            cleanup_result['space_freed_bytes'] += file_size
                
                if any(f.startswith('thumb_') for f in cleanup_result['files_removed']):
                    cleanup_result['cleanup_actions'].append('remove_old_thumbnails')
        
        return cleanup_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File cleanup failed: {str(e)}',
            'cleanup_actions': []
        }

# Update __all__ to include remaining helper functions
__all__.extend([
    '_organize_files_operation',
    '_compress_files_operation',
    '_extract_files_operation',
    '_organize_by_file_type',
    '_organize_by_date',
    '_organize_by_size',
    '_cleanup_storage_structure',
    '_generate_comprehensive_analytics',
    '_generate_image_thumbnails',
    '_extract_file_metadata',
    '_scan_archive_contents',
    '_perform_file_cleanup'
])