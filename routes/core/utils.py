"""
Core Utils Module
Utility Functions, Helper Methods, Common Operations
"""

import logging
import json
import re
import hashlib
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import os
import sys

from .constants import (
    SYSTEM_CONFIG,
    RESPONSE_TEMPLATES,
    VALIDATION_RULES,
    ERROR_CODES,
    PERFORMANCE_THRESHOLDS
)

logger = logging.getLogger(__name__)

# Utils State Tracking
_utils_state = {
    'initialization_timestamp': None,
    'operations_count': 0,
    'performance_metrics': {},
    'cache': {},
    'active_sessions': {}
}

def format_response(success: bool = True,
                   data: Any = None,
                   message: str = None,
                   error_code: str = None,
                   error_details: Any = None,
                   execution_time_ms: float = None,
                   template_type: str = 'success') -> Dict[str, Any]:
    """
    Response Formatting Utility
    
    Extrahiert aus kira_routes.py.backup Response Formatting Logic
    """
    try:
        # Get appropriate template
        if template_type in RESPONSE_TEMPLATES:
            response = RESPONSE_TEMPLATES[template_type].copy()
        else:
            response = RESPONSE_TEMPLATES['success'].copy() if success else RESPONSE_TEMPLATES['error'].copy()
        
        # Update response with provided data
        response['success'] = success
        response['timestamp'] = datetime.now().isoformat()
        
        if execution_time_ms is not None:
            response['execution_time_ms'] = execution_time_ms
        
        if success:
            response['status'] = 'success'
            if data is not None:
                response['data'] = data
            if message:
                response['message'] = message
        else:
            response['status'] = 'error'
            if error_code:
                response['error_code'] = error_code
            if message:
                response['error_message'] = message
            if error_details:
                response['details'] = error_details
        
        # Add response metadata
        response['response_id'] = str(uuid.uuid4())[:8]
        response['response_version'] = SYSTEM_CONFIG.get('version', '1.0.0')
        
        return response
        
    except Exception as e:
        logger.error(f"Response formatting failed: {e}")
        return {
            'success': False,
            'status': 'error',
            'error_code': 'RESPONSE_FORMATTING_FAILED',
            'error_message': f'Response formatting error: {str(e)}',
            'timestamp': datetime.now().isoformat(),
            'response_id': str(uuid.uuid4())[:8]
        }

def validate_request_data(data: Dict[str, Any],
                         validation_schema: Dict[str, Any] = None,
                         strict_mode: bool = False) -> Dict[str, Any]:
    """
    Request Data Validation Utility
    
    Basiert auf kira_routes.py.backup Request Validation Logic
    """
    try:
        validation_result = {
            'is_valid': True,
            'validation_errors': [],
            'validation_warnings': [],
            'validated_data': {},
            'validation_summary': {}
        }
        
        if validation_schema is None:
            validation_schema = _get_default_validation_schema()
        
        # Initialize validation session
        validation_session = {
            'session_id': f"validation_{int(time.time())}",
            'start_time': time.time(),
            'validation_schema': validation_schema,
            'strict_mode': strict_mode,
            'fields_validated': 0,
            'fields_failed': 0
        }
        
        # Validate each field
        for field_name, field_rules in validation_schema.items():
            field_validation = _validate_field(data, field_name, field_rules, strict_mode)
            
            validation_result['validated_data'][field_name] = field_validation.get('validated_value')
            validation_session['fields_validated'] += 1
            
            if not field_validation['is_valid']:
                validation_result['validation_errors'].extend(field_validation['errors'])
                validation_session['fields_failed'] += 1
            
            if field_validation.get('warnings'):
                validation_result['validation_warnings'].extend(field_validation['warnings'])
        
        # Check for required fields
        required_fields = [name for name, rules in validation_schema.items() if rules.get('required', False)]
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        
        if missing_fields:
            validation_result['validation_errors'].extend([f"Required field missing: {field}" for field in missing_fields])
            validation_result['is_valid'] = False
        
        # Check for unexpected fields in strict mode
        if strict_mode:
            unexpected_fields = [field for field in data.keys() if field not in validation_schema]
            if unexpected_fields:
                validation_result['validation_warnings'].extend([f"Unexpected field: {field}" for field in unexpected_fields])
        
        # Final validation status
        validation_result['is_valid'] = len(validation_result['validation_errors']) == 0
        
        # Validation summary
        validation_session['end_time'] = time.time()
        validation_session['validation_duration_ms'] = (validation_session['end_time'] - validation_session['start_time']) * 1000
        
        validation_result['validation_summary'] = {
            'fields_validated': validation_session['fields_validated'],
            'fields_failed': validation_session['fields_failed'],
            'validation_duration_ms': validation_session['validation_duration_ms'],
            'validation_success_rate': (validation_session['fields_validated'] - validation_session['fields_failed']) / max(1, validation_session['fields_validated'])
        }
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Request data validation failed: {e}")
        return {
            'is_valid': False,
            'validation_errors': [f'Validation system error: {str(e)}'],
            'validation_warnings': [],
            'validated_data': {},
            'validation_summary': {'error': str(e)}
        }

def calculate_metrics(operation_data: Dict[str, Any],
                     metric_types: List[str] = None,
                     time_window_seconds: int = 300) -> Dict[str, Any]:
    """
    Metrics Calculation Utility
    
    Extrahiert aus kira_routes.py.backup Metrics Calculation Logic
    """
    try:
        if metric_types is None:
            metric_types = ['performance', 'usage', 'efficiency', 'quality']
        
        # Initialize metrics calculation session
        metrics_session = {
            'session_id': f"metrics_{int(time.time())}",
            'start_time': time.time(),
            'operation_data': operation_data,
            'metric_types': metric_types,
            'calculated_metrics': {},
            'metric_errors': []
        }
        
        # Calculate different metric types
        for metric_type in metric_types:
            try:
                if metric_type == 'performance':
                    metrics_session['calculated_metrics']['performance'] = _calculate_performance_metrics(operation_data, time_window_seconds)
                
                elif metric_type == 'usage':
                    metrics_session['calculated_metrics']['usage'] = _calculate_usage_metrics(operation_data, time_window_seconds)
                
                elif metric_type == 'efficiency':
                    metrics_session['calculated_metrics']['efficiency'] = _calculate_efficiency_metrics(operation_data, time_window_seconds)
                
                elif metric_type == 'quality':
                    metrics_session['calculated_metrics']['quality'] = _calculate_quality_metrics(operation_data, time_window_seconds)
                
                elif metric_type == 'reliability':
                    metrics_session['calculated_metrics']['reliability'] = _calculate_reliability_metrics(operation_data, time_window_seconds)
                
                else:
                    metrics_session['metric_errors'].append(f'Unknown metric type: {metric_type}')
                    
            except Exception as e:
                metrics_session['metric_errors'].append(f'Metric calculation failed for {metric_type}: {str(e)}')
        
        # Generate metric insights
        metric_insights = _generate_metric_insights(metrics_session['calculated_metrics'])
        
        # Metric recommendations
        metric_recommendations = _generate_metric_recommendations(metrics_session['calculated_metrics'], metric_insights)
        
        metrics_session.update({
            'end_time': time.time(),
            'calculation_duration_ms': (time.time() - metrics_session['start_time']) * 1000,
            'metrics_calculated': len(metrics_session['calculated_metrics']),
            'metric_insights': metric_insights,
            'metric_recommendations': metric_recommendations
        })
        
        return {
            'success': True,
            'metrics_session': metrics_session,
            'metrics': metrics_session['calculated_metrics'],
            'insights': metric_insights,
            'recommendations': metric_recommendations,
            'calculation_summary': {
                'metrics_calculated': metrics_session['metrics_calculated'],
                'calculation_duration_ms': metrics_session['calculation_duration_ms'],
                'errors_count': len(metrics_session['metric_errors'])
            }
        }
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'metrics': {},
            'insights': {},
            'recommendations': []
        }

def process_data_structures(data: Any,
                          processing_type: str = 'normalize',
                          processing_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Data Structure Processing Utility
    
    Basiert auf kira_routes.py.backup Data Processing Logic
    """
    try:
        if processing_config is None:
            processing_config = {
                'deep_processing': True,
                'preserve_types': True,
                'error_handling': 'skip',
                'max_depth': 10
            }
        
        # Initialize processing session
        processing_session = {
            'session_id': f"processing_{int(time.time())}",
            'start_time': time.time(),
            'processing_type': processing_type,
            'processing_config': processing_config,
            'original_data_type': type(data).__name__,
            'processing_results': {}
        }
        
        # Perform processing based on type
        if processing_type == 'normalize':
            processing_session['processing_results'] = _normalize_data_structure(data, processing_config)
        
        elif processing_type == 'sanitize':
            processing_session['processing_results'] = _sanitize_data_structure(data, processing_config)
        
        elif processing_type == 'transform':
            processing_session['processing_results'] = _transform_data_structure(data, processing_config)
        
        elif processing_type == 'validate':
            processing_session['processing_results'] = _validate_data_structure(data, processing_config)
        
        elif processing_type == 'compress':
            processing_session['processing_results'] = _compress_data_structure(data, processing_config)
        
        elif processing_type == 'analyze':
            processing_session['processing_results'] = _analyze_data_structure(data, processing_config)
        
        else:
            processing_session['processing_results'] = {
                'status': 'error',
                'error': f'Unsupported processing type: {processing_type}'
            }
        
        # Add processing metadata
        processing_session.update({
            'end_time': time.time(),
            'processing_success': processing_session['processing_results'].get('status') == 'success',
            'processing_duration_ms': (time.time() - processing_session['start_time']) * 1000
        })
        
        return {
            'success': processing_session['processing_success'],
            'processing_session': processing_session,
            'processed_data': processing_session['processing_results'].get('processed_data'),
            'processing_statistics': processing_session['processing_results'].get('processing_statistics', {}),
            'processing_summary': {
                'processing_type': processing_type,
                'processing_success': processing_session['processing_success'],
                'processing_duration_ms': processing_session['processing_duration_ms'],
                'original_data_type': processing_session['original_data_type']
            }
        }
        
    except Exception as e:
        logger.error(f"Data structure processing failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'processed_data': None,
            'processing_statistics': {},
            'processing_summary': {'error': str(e)}
        }

def handle_file_operations(operation_type: str,
                          file_path: str = None,
                          file_data: Any = None,
                          operation_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    File Operations Utility
    
    Extrahiert aus kira_routes.py.backup File Handling Logic
    """
    try:
        if operation_config is None:
            operation_config = {
                'encoding': 'utf-8',
                'backup_before_write': False,
                'create_directories': True,
                'file_permissions': 0o644,
                'max_file_size_mb': 50
            }
        
        # Initialize file operation session
        file_session = {
            'session_id': f"file_ops_{int(time.time())}",
            'start_time': time.time(),
            'operation_type': operation_type,
            'file_path': file_path,
            'operation_config': operation_config,
            'operation_results': {}
        }
        
        # Perform file operation based on type
        if operation_type == 'read':
            file_session['operation_results'] = _read_file_operation(file_path, operation_config)
        
        elif operation_type == 'write':
            file_session['operation_results'] = _write_file_operation(file_path, file_data, operation_config)
        
        elif operation_type == 'append':
            file_session['operation_results'] = _append_file_operation(file_path, file_data, operation_config)
        
        elif operation_type == 'delete':
            file_session['operation_results'] = _delete_file_operation(file_path, operation_config)
        
        elif operation_type == 'copy':
            destination_path = operation_config.get('destination_path')
            file_session['operation_results'] = _copy_file_operation(file_path, destination_path, operation_config)
        
        elif operation_type == 'move':
            destination_path = operation_config.get('destination_path')
            file_session['operation_results'] = _move_file_operation(file_path, destination_path, operation_config)
        
        elif operation_type == 'info':
            file_session['operation_results'] = _get_file_info_operation(file_path, operation_config)
        
        elif operation_type == 'validate':
            file_session['operation_results'] = _validate_file_operation(file_path, operation_config)
        
        else:
            file_session['operation_results'] = {
                'status': 'error',
                'error': f'Unsupported file operation: {operation_type}'
            }
        
        # Add file operation metadata
        file_session.update({
            'end_time': time.time(),
            'operation_success': file_session['operation_results'].get('status') == 'success',
            'operation_duration_ms': (time.time() - file_session['start_time']) * 1000
        })
        
        return {
            'success': file_session['operation_success'],
            'file_session': file_session,
            'file_operation_result': file_session['operation_results'],
            'file_summary': {
                'operation_type': operation_type,
                'file_path': file_path,
                'operation_success': file_session['operation_success'],
                'operation_duration_ms': file_session['operation_duration_ms']
            }
        }
        
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'file_operation_result': {},
            'file_summary': {'error': str(e)}
        }

def manage_timestamps(timestamp_operation: str = 'current',
                     timestamp_format: str = 'iso',
                     timezone: str = 'UTC',
                     timestamp_data: Any = None) -> Dict[str, Any]:
    """
    Timestamp Management Utility
    
    Basiert auf kira_routes.py.backup Timestamp Handling Logic
    """
    try:
        # Initialize timestamp session
        timestamp_session = {
            'session_id': f"timestamp_{int(time.time())}",
            'start_time': time.time(),
            'timestamp_operation': timestamp_operation,
            'timestamp_format': timestamp_format,
            'timezone': timezone,
            'timestamp_results': {}
        }
        
        # Perform timestamp operation
        if timestamp_operation == 'current':
            timestamp_session['timestamp_results'] = _get_current_timestamp(timestamp_format, timezone)
        
        elif timestamp_operation == 'parse':
            timestamp_session['timestamp_results'] = _parse_timestamp(timestamp_data, timestamp_format, timezone)
        
        elif timestamp_operation == 'format':
            timestamp_session['timestamp_results'] = _format_timestamp(timestamp_data, timestamp_format, timezone)
        
        elif timestamp_operation == 'calculate':
            timestamp_session['timestamp_results'] = _calculate_timestamp_difference(timestamp_data, timezone)
        
        elif timestamp_operation == 'validate':
            timestamp_session['timestamp_results'] = _validate_timestamp(timestamp_data, timestamp_format)
        
        elif timestamp_operation == 'convert':
            target_format = timestamp_data.get('target_format', 'iso') if isinstance(timestamp_data, dict) else timestamp_format
            timestamp_session['timestamp_results'] = _convert_timestamp(timestamp_data, timestamp_format, target_format, timezone)
        
        else:
            timestamp_session['timestamp_results'] = {
                'status': 'error',
                'error': f'Unsupported timestamp operation: {timestamp_operation}'
            }
        
        # Add timestamp metadata
        timestamp_session.update({
            'end_time': time.time(),
            'operation_success': timestamp_session['timestamp_results'].get('status') == 'success',
            'operation_duration_ms': (time.time() - timestamp_session['start_time']) * 1000
        })
        
        return {
            'success': timestamp_session['operation_success'],
            'timestamp_session': timestamp_session,
            'timestamp_result': timestamp_session['timestamp_results'].get('result'),
            'timestamp_summary': {
                'operation': timestamp_operation,
                'format': timestamp_format,
                'timezone': timezone,
                'success': timestamp_session['operation_success']
            }
        }
        
    except Exception as e:
        logger.error(f"Timestamp management failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp_result': None,
            'timestamp_summary': {'error': str(e)}
        }

# ====================================
# PRIVATE HELPER FUNCTIONS
# ====================================

def _get_default_validation_schema() -> Dict[str, Any]:
    """Holt Standard Validation Schema"""
    return {
        'message': {
            'required': True,
            'type': 'string',
            'min_length': 1,
            'max_length': VALIDATION_RULES['string_fields']['max_length'],
            'sanitize': True
        },
        'user_id': {
            'required': False,
            'type': 'string',
            'max_length': 100,
            'pattern': r'^[a-zA-Z0-9_-]+$'
        },
        'session_id': {
            'required': False,
            'type': 'string',
            'max_length': 100
        },
        'timestamp': {
            'required': False,
            'type': 'datetime',
            'format': 'iso'
        },
        'priority': {
            'required': False,
            'type': 'number',
            'min_value': 0,
            'max_value': 10,
            'default': 5
        }
    }

def _validate_field(data: Dict, field_name: str, field_rules: Dict, strict_mode: bool) -> Dict[str, Any]:
    """Validiert einzelnes Feld"""
    try:
        field_validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'validated_value': None
        }
        
        field_value = data.get(field_name)
        
        # Check if field is required
        if field_rules.get('required', False) and (field_value is None or field_value == ''):
            field_validation['errors'].append(f"Field '{field_name}' is required")
            field_validation['is_valid'] = False
            return field_validation
        
        # Skip validation if field is not present and not required
        if field_value is None:
            field_validation['validated_value'] = field_rules.get('default')
            return field_validation
        
        # Type validation
        expected_type = field_rules.get('type', 'string')
        if not _validate_field_type(field_value, expected_type):
            field_validation['errors'].append(f"Field '{field_name}' must be of type {expected_type}")
            field_validation['is_valid'] = False
        
        # String-specific validations
        if expected_type == 'string' and isinstance(field_value, str):
            # Length validation
            min_length = field_rules.get('min_length', 0)
            max_length = field_rules.get('max_length', float('inf'))
            
            if len(field_value) < min_length:
                field_validation['errors'].append(f"Field '{field_name}' must be at least {min_length} characters")
                field_validation['is_valid'] = False
            
            if len(field_value) > max_length:
                field_validation['errors'].append(f"Field '{field_name}' must be no more than {max_length} characters")
                field_validation['is_valid'] = False
            
            # Pattern validation
            pattern = field_rules.get('pattern')
            if pattern and not re.match(pattern, field_value):
                field_validation['errors'].append(f"Field '{field_name}' does not match required pattern")
                field_validation['is_valid'] = False
            
            # Sanitization
            if field_rules.get('sanitize', False):
                field_validation['validated_value'] = _sanitize_string(field_value)
            else:
                field_validation['validated_value'] = field_value
        
        # Number-specific validations
        elif expected_type == 'number' and isinstance(field_value, (int, float)):
            min_value = field_rules.get('min_value', float('-inf'))
            max_value = field_rules.get('max_value', float('inf'))
            
            if field_value < min_value:
                field_validation['errors'].append(f"Field '{field_name}' must be at least {min_value}")
                field_validation['is_valid'] = False
            
            if field_value > max_value:
                field_validation['errors'].append(f"Field '{field_name}' must be no more than {max_value}")
                field_validation['is_valid'] = False
            
            field_validation['validated_value'] = field_value
        
        # Datetime-specific validations
        elif expected_type == 'datetime':
            try:
                if isinstance(field_value, str):
                    parsed_datetime = datetime.fromisoformat(field_value.replace('Z', '+00:00'))
                    field_validation['validated_value'] = parsed_datetime.isoformat()
                else:
                    field_validation['validated_value'] = field_value
            except ValueError:
                field_validation['errors'].append(f"Field '{field_name}' is not a valid datetime")
                field_validation['is_valid'] = False
        
        else:
            field_validation['validated_value'] = field_value
        
        return field_validation
        
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Field validation error for '{field_name}': {str(e)}"],
            'warnings': [],
            'validated_value': None
        }

def _validate_field_type(value: Any, expected_type: str) -> bool:
    """Validiert Feld-Typ"""
    try:
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'number':
            return isinstance(value, (int, float))
        elif expected_type == 'integer':
            return isinstance(value, int)
        elif expected_type == 'float':
            return isinstance(value, float)
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'list':
            return isinstance(value, list)
        elif expected_type == 'dict':
            return isinstance(value, dict)
        elif expected_type == 'datetime':
            if isinstance(value, str):
                try:
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                    return True
                except ValueError:
                    return False
            return isinstance(value, datetime)
        else:
            return True  # Unknown type, allow it
    except Exception:
        return False

def _sanitize_string(value: str) -> str:
    """Sanitized String Input"""
    try:
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';]', '', value)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        # Limit length
        max_length = VALIDATION_RULES['string_fields']['max_length']
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    except Exception:
        return str(value)[:100]  # Fallback: convert to string and limit

def _calculate_performance_metrics(operation_data: Dict, time_window_seconds: int) -> Dict[str, Any]:
    """Berechnet Performance Metrics"""
    try:
        performance_metrics = {
            'response_times': [],
            'throughput': 0,
            'error_rate': 0,
            'availability': 1.0,
            'performance_score': 1.0
        }
        
        # Extract response times from operation data
        response_times = operation_data.get('response_times', [])
        if response_times:
            performance_metrics['response_times'] = response_times
            performance_metrics['average_response_time'] = sum(response_times) / len(response_times)
            performance_metrics['min_response_time'] = min(response_times)
            performance_metrics['max_response_time'] = max(response_times)
            
            # Calculate performance score based on response times
            avg_response_time = performance_metrics['average_response_time']
            if avg_response_time <= PERFORMANCE_THRESHOLDS['response_time']['excellent_ms']:
                performance_metrics['performance_score'] = 1.0
            elif avg_response_time <= PERFORMANCE_THRESHOLDS['response_time']['good_ms']:
                performance_metrics['performance_score'] = 0.8
            elif avg_response_time <= PERFORMANCE_THRESHOLDS['response_time']['acceptable_ms']:
                performance_metrics['performance_score'] = 0.6
            else:
                performance_metrics['performance_score'] = 0.3
        
        # Calculate throughput (operations per second)
        total_operations = operation_data.get('total_operations', 0)
        if time_window_seconds > 0:
            performance_metrics['throughput'] = total_operations / time_window_seconds
        
        # Calculate error rate
        total_errors = operation_data.get('total_errors', 0)
        if total_operations > 0:
            performance_metrics['error_rate'] = total_errors / total_operations
        
        # Calculate availability
        downtime_seconds = operation_data.get('downtime_seconds', 0)
        if time_window_seconds > 0:
            performance_metrics['availability'] = max(0.0, (time_window_seconds - downtime_seconds) / time_window_seconds)
        
        return performance_metrics
        
    except Exception as e:
        logger.debug(f"Performance metrics calculation failed: {e}")
        return {
            'response_times': [],
            'throughput': 0,
            'error_rate': 1.0,
            'availability': 0.0,
            'performance_score': 0.0,
            'calculation_error': str(e)
        }

def _calculate_usage_metrics(operation_data: Dict, time_window_seconds: int) -> Dict[str, Any]:
    """Berechnet Usage Metrics"""
    try:
        usage_metrics = {
            'total_requests': 0,
            'unique_users': 0,
            'peak_concurrent_users': 0,
            'usage_patterns': {},
            'resource_utilization': {}
        }
        
        # Basic usage statistics
        usage_metrics['total_requests'] = operation_data.get('total_requests', 0)
        usage_metrics['unique_users'] = len(set(operation_data.get('user_ids', [])))
        usage_metrics['peak_concurrent_users'] = operation_data.get('peak_concurrent_users', 0)
        
        # Usage patterns analysis
        request_timestamps = operation_data.get('request_timestamps', [])
        if request_timestamps:
            # Hourly distribution
            hourly_distribution = {}
            for timestamp in request_timestamps:
                try:
                    hour = datetime.fromisoformat(timestamp).hour
                    hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
                except:
                    continue
            
            usage_metrics['usage_patterns']['hourly_distribution'] = hourly_distribution
            
            # Peak hour
            if hourly_distribution:
                peak_hour = max(hourly_distribution, key=hourly_distribution.get)
                usage_metrics['usage_patterns']['peak_hour'] = peak_hour
                usage_metrics['usage_patterns']['peak_hour_requests'] = hourly_distribution[peak_hour]
        
        # Resource utilization
        cpu_usage_samples = operation_data.get('cpu_usage_samples', [])
        memory_usage_samples = operation_data.get('memory_usage_samples', [])
        
        if cpu_usage_samples:
            usage_metrics['resource_utilization']['average_cpu_usage'] = sum(cpu_usage_samples) / len(cpu_usage_samples)
            usage_metrics['resource_utilization']['peak_cpu_usage'] = max(cpu_usage_samples)
        
        if memory_usage_samples:
            usage_metrics['resource_utilization']['average_memory_usage'] = sum(memory_usage_samples) / len(memory_usage_samples)
            usage_metrics['resource_utilization']['peak_memory_usage'] = max(memory_usage_samples)
        
        return usage_metrics
        
    except Exception as e:
        logger.debug(f"Usage metrics calculation failed: {e}")
        return {
            'total_requests': 0,
            'unique_users': 0,
            'peak_concurrent_users': 0,
            'usage_patterns': {},
            'resource_utilization': {},
            'calculation_error': str(e)
        }

def _calculate_efficiency_metrics(operation_data: Dict, time_window_seconds: int) -> Dict[str, Any]:
    """Berechnet Efficiency Metrics"""
    try:
        efficiency_metrics = {
            'requests_per_second': 0,
            'cpu_efficiency': 0,
            'memory_efficiency': 0,
            'resource_efficiency_score': 0,
            'cost_efficiency': 0
        }
        
        # Calculate requests per second
        total_requests = operation_data.get('total_requests', 0)
        if time_window_seconds > 0:
            efficiency_metrics['requests_per_second'] = total_requests / time_window_seconds
        
        # CPU efficiency calculation
        cpu_usage_samples = operation_data.get('cpu_usage_samples', [])
        if cpu_usage_samples and total_requests > 0:
            avg_cpu_usage = sum(cpu_usage_samples) / len(cpu_usage_samples)
            # Efficiency = Requests processed per unit of CPU usage
            efficiency_metrics['cpu_efficiency'] = total_requests / max(1, avg_cpu_usage)
        
        # Memory efficiency calculation
        memory_usage_samples = operation_data.get('memory_usage_samples', [])
        if memory_usage_samples and total_requests > 0:
            avg_memory_usage = sum(memory_usage_samples) / len(memory_usage_samples)
            efficiency_metrics['memory_efficiency'] = total_requests / max(1, avg_memory_usage)
        
        # Overall resource efficiency score
        cpu_eff = efficiency_metrics['cpu_efficiency']
        memory_eff = efficiency_metrics['memory_efficiency']
        
        if cpu_eff > 0 and memory_eff > 0:
            # Normalize and combine efficiency scores
            normalized_cpu_eff = min(1.0, cpu_eff / 100)  # Assuming 100 requests per CPU% is excellent
            normalized_memory_eff = min(1.0, memory_eff / 50)  # Assuming 50 requests per memory% is excellent
            efficiency_metrics['resource_efficiency_score'] = (normalized_cpu_eff + normalized_memory_eff) / 2
        
        # Cost efficiency (simplified)
        processing_cost = operation_data.get('processing_cost', 0)
        if processing_cost > 0 and total_requests > 0:
            efficiency_metrics['cost_efficiency'] = total_requests / processing_cost
        
        return efficiency_metrics
        
    except Exception as e:
        logger.debug(f"Efficiency metrics calculation failed: {e}")
        return {
            'requests_per_second': 0,
            'cpu_efficiency': 0,
            'memory_efficiency': 0,
            'resource_efficiency_score': 0,
            'cost_efficiency': 0,
            'calculation_error': str(e)
        }

def _calculate_quality_metrics(operation_data: Dict, time_window_seconds: int) -> Dict[str, Any]:
    """Berechnet Quality Metrics"""
    try:
        quality_metrics = {
            'success_rate': 0,
            'data_quality_score': 0,
            'user_satisfaction_score': 0,
            'response_accuracy': 0,
            'overall_quality_score': 0
        }
        
        # Success rate calculation
        total_operations = operation_data.get('total_operations', 0)
        successful_operations = operation_data.get('successful_operations', 0)
        
        if total_operations > 0:
            quality_metrics['success_rate'] = successful_operations / total_operations
        
        # Data quality score
        data_validation_results = operation_data.get('data_validation_results', [])
        if data_validation_results:
            valid_data_count = sum(1 for result in data_validation_results if result.get('is_valid', False))
            quality_metrics['data_quality_score'] = valid_data_count / len(data_validation_results)
        
        # User satisfaction score
        satisfaction_ratings = operation_data.get('satisfaction_ratings', [])
        if satisfaction_ratings:
            quality_metrics['user_satisfaction_score'] = sum(satisfaction_ratings) / len(satisfaction_ratings) / 10  # Normalize to 0-1
        
        # Response accuracy
        accuracy_scores = operation_data.get('accuracy_scores', [])
        if accuracy_scores:
            quality_metrics['response_accuracy'] = sum(accuracy_scores) / len(accuracy_scores)
        
        # Overall quality score (weighted average)
        quality_components = [
            quality_metrics['success_rate'] * 0.3,
            quality_metrics['data_quality_score'] * 0.2,
            quality_metrics['user_satisfaction_score'] * 0.3,
            quality_metrics['response_accuracy'] * 0.2
        ]
        
        quality_metrics['overall_quality_score'] = sum(quality_components)
        
        return quality_metrics
        
    except Exception as e:
        logger.debug(f"Quality metrics calculation failed: {e}")
        return {
            'success_rate': 0,
            'data_quality_score': 0,
            'user_satisfaction_score': 0,
            'response_accuracy': 0,
            'overall_quality_score': 0,
            'calculation_error': str(e)
        }

def _calculate_reliability_metrics(operation_data: Dict, time_window_seconds: int) -> Dict[str, Any]:
    """Berechnet Reliability Metrics"""
    try:
        reliability_metrics = {
            'uptime_percentage': 0,
            'mean_time_between_failures': 0,
            'error_frequency': 0,
            'recovery_time': 0,
            'reliability_score': 0
        }
        
        # Uptime calculation
        downtime_seconds = operation_data.get('downtime_seconds', 0)
        if time_window_seconds > 0:
            reliability_metrics['uptime_percentage'] = ((time_window_seconds - downtime_seconds) / time_window_seconds) * 100
        
        # Mean time between failures
        failure_times = operation_data.get('failure_timestamps', [])
        if len(failure_times) > 1:
            time_diffs = []
            for i in range(1, len(failure_times)):
                try:
                    prev_time = datetime.fromisoformat(failure_times[i-1])
                    curr_time = datetime.fromisoformat(failure_times[i])
                    time_diffs.append((curr_time - prev_time).total_seconds())
                except:
                    continue
            
            if time_diffs:
                reliability_metrics['mean_time_between_failures'] = sum(time_diffs) / len(time_diffs)
        
        # Error frequency
        total_errors = operation_data.get('total_errors', 0)
        if time_window_seconds > 0:
            reliability_metrics['error_frequency'] = total_errors / (time_window_seconds / 3600)  # errors per hour
        
        # Recovery time
        recovery_times = operation_data.get('recovery_times', [])
        if recovery_times:
            reliability_metrics['recovery_time'] = sum(recovery_times) / len(recovery_times)
        
        # Overall reliability score
        uptime_score = reliability_metrics['uptime_percentage'] / 100
        error_score = max(0, 1 - (reliability_metrics['error_frequency'] / 10))  # Assuming 10 errors/hour is very bad
        recovery_score = max(0, 1 - (reliability_metrics['recovery_time'] / 300))  # Assuming 5 minutes recovery is acceptable
        
        reliability_metrics['reliability_score'] = (uptime_score * 0.5 + error_score * 0.3 + recovery_score * 0.2)
        
        return reliability_metrics
        
    except Exception as e:
        logger.debug(f"Reliability metrics calculation failed: {e}")
        return {
            'uptime_percentage': 0,
            'mean_time_between_failures': 0,
            'error_frequency': 0,
            'recovery_time': 0,
            'reliability_score': 0,
            'calculation_error': str(e)
        }

def _initialize_utils_component():
    """Initialisiert Utils Component"""
    try:
        global _utils_state
        _utils_state['initialization_timestamp'] = datetime.now().isoformat()
        return True
    except Exception as e:
        raise Exception(f"Utils component initialization failed: {str(e)}")

# Export all public functions
__all__ = [
    'format_response',
    'validate_request_data',
    'calculate_metrics',
    'process_data_structures',
    'handle_file_operations',
    'manage_timestamps',
    '_initialize_utils_component'
]

# ...existing code...

# ====================================
# MISSING HELPER FUNCTIONS - Teil 2
# ====================================

def _generate_metric_insights(calculated_metrics: Dict) -> Dict[str, Any]:
    """Generiert Metric Insights"""
    try:
        insights = {
            'performance_insights': [],
            'usage_insights': [],
            'efficiency_insights': [],
            'quality_insights': [],
            'reliability_insights': [],
            'overall_system_health': 'unknown'
        }
        
        # Performance insights
        if 'performance' in calculated_metrics:
            perf_data = calculated_metrics['performance']
            performance_score = perf_data.get('performance_score', 0)
            
            if performance_score >= 0.9:
                insights['performance_insights'].append('Excellent system performance detected')
            elif performance_score >= 0.7:
                insights['performance_insights'].append('Good system performance with room for optimization')
            elif performance_score >= 0.5:
                insights['performance_insights'].append('Moderate performance issues detected')
            else:
                insights['performance_insights'].append('Critical performance issues require immediate attention')
            
            avg_response_time = perf_data.get('average_response_time', 0)
            if avg_response_time > PERFORMANCE_THRESHOLDS['response_time']['critical_ms']:
                insights['performance_insights'].append(f'Response times are critically high: {avg_response_time:.2f}ms')
        
        # Usage insights
        if 'usage' in calculated_metrics:
            usage_data = calculated_metrics['usage']
            total_requests = usage_data.get('total_requests', 0)
            unique_users = usage_data.get('unique_users', 0)
            
            if total_requests > 1000:
                insights['usage_insights'].append('High system usage detected')
            elif total_requests > 100:
                insights['usage_insights'].append('Moderate system usage')
            else:
                insights['usage_insights'].append('Low system usage')
            
            if unique_users > 0:
                avg_requests_per_user = total_requests / unique_users
                if avg_requests_per_user > 50:
                    insights['usage_insights'].append('High user engagement detected')
        
        # Efficiency insights
        if 'efficiency' in calculated_metrics:
            eff_data = calculated_metrics['efficiency']
            resource_efficiency = eff_data.get('resource_efficiency_score', 0)
            
            if resource_efficiency >= 0.8:
                insights['efficiency_insights'].append('Excellent resource efficiency')
            elif resource_efficiency >= 0.6:
                insights['efficiency_insights'].append('Good resource utilization')
            else:
                insights['efficiency_insights'].append('Resource utilization needs optimization')
        
        # Quality insights
        if 'quality' in calculated_metrics:
            quality_data = calculated_metrics['quality']
            overall_quality = quality_data.get('overall_quality_score', 0)
            
            if overall_quality >= 0.9:
                insights['quality_insights'].append('Exceptional service quality')
            elif overall_quality >= 0.7:
                insights['quality_insights'].append('Good service quality')
            else:
                insights['quality_insights'].append('Service quality requires improvement')
        
        # Reliability insights
        if 'reliability' in calculated_metrics:
            rel_data = calculated_metrics['reliability']
            reliability_score = rel_data.get('reliability_score', 0)
            uptime = rel_data.get('uptime_percentage', 0)
            
            if uptime >= 99.9:
                insights['reliability_insights'].append('Excellent system reliability')
            elif uptime >= 99.0:
                insights['reliability_insights'].append('Good system reliability')
            else:
                insights['reliability_insights'].append('System reliability needs attention')
        
        # Overall system health assessment
        health_scores = []
        if 'performance' in calculated_metrics:
            health_scores.append(calculated_metrics['performance'].get('performance_score', 0))
        if 'quality' in calculated_metrics:
            health_scores.append(calculated_metrics['quality'].get('overall_quality_score', 0))
        if 'reliability' in calculated_metrics:
            health_scores.append(calculated_metrics['reliability'].get('reliability_score', 0))
        
        if health_scores:
            avg_health = sum(health_scores) / len(health_scores)
            if avg_health >= 0.9:
                insights['overall_system_health'] = 'excellent'
            elif avg_health >= 0.7:
                insights['overall_system_health'] = 'good'
            elif avg_health >= 0.5:
                insights['overall_system_health'] = 'fair'
            else:
                insights['overall_system_health'] = 'poor'
        
        return insights
        
    except Exception as e:
        logger.debug(f"Metric insights generation failed: {e}")
        return {
            'performance_insights': [],
            'usage_insights': [],
            'efficiency_insights': [],
            'quality_insights': [],
            'reliability_insights': [],
            'overall_system_health': 'unknown',
            'generation_error': str(e)
        }

def _generate_metric_recommendations(calculated_metrics: Dict, insights: Dict) -> List[Dict]:
    """Generiert Metric Recommendations"""
    try:
        recommendations = []
        
        # Performance recommendations
        if 'performance' in calculated_metrics:
            perf_data = calculated_metrics['performance']
            performance_score = perf_data.get('performance_score', 0)
            
            if performance_score < 0.7:
                recommendations.append({
                    'category': 'performance',
                    'priority': 'high',
                    'recommendation': 'Optimize response times by implementing caching mechanisms',
                    'expected_impact': 'Reduce average response time by 30-50%'
                })
            
            error_rate = perf_data.get('error_rate', 0)
            if error_rate > 0.05:  # 5% error rate
                recommendations.append({
                    'category': 'performance',
                    'priority': 'critical',
                    'recommendation': 'Investigate and fix high error rate issues',
                    'expected_impact': 'Improve system reliability and user experience'
                })
        
        # Resource efficiency recommendations
        if 'efficiency' in calculated_metrics:
            eff_data = calculated_metrics['efficiency']
            resource_efficiency = eff_data.get('resource_efficiency_score', 0)
            
            if resource_efficiency < 0.6:
                recommendations.append({
                    'category': 'efficiency',
                    'priority': 'medium',
                    'recommendation': 'Implement resource pooling and optimize memory usage',
                    'expected_impact': 'Improve resource utilization by 20-40%'
                })
        
        # Quality recommendations
        if 'quality' in calculated_metrics:
            quality_data = calculated_metrics['quality']
            data_quality = quality_data.get('data_quality_score', 0)
            
            if data_quality < 0.8:
                recommendations.append({
                    'category': 'quality',
                    'priority': 'medium',
                    'recommendation': 'Enhance data validation and sanitization processes',
                    'expected_impact': 'Improve data quality and reduce processing errors'
                })
        
        # Usage optimization recommendations
        if 'usage' in calculated_metrics:
            usage_data = calculated_metrics['usage']
            peak_cpu = usage_data.get('resource_utilization', {}).get('peak_cpu_usage', 0)
            
            if peak_cpu > 80:
                recommendations.append({
                    'category': 'usage',
                    'priority': 'high',
                    'recommendation': 'Implement load balancing to distribute CPU usage',
                    'expected_impact': 'Reduce peak CPU usage and improve system stability'
                })
        
        # Reliability recommendations
        if 'reliability' in calculated_metrics:
            rel_data = calculated_metrics['reliability']
            uptime = rel_data.get('uptime_percentage', 0)
            
            if uptime < 99.0:
                recommendations.append({
                    'category': 'reliability',
                    'priority': 'critical',
                    'recommendation': 'Implement automated failover and health monitoring',
                    'expected_impact': 'Increase uptime to 99.5%+ and reduce downtime incidents'
                })
        
        # Sort recommendations by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return recommendations
        
    except Exception as e:
        logger.debug(f"Metric recommendations generation failed: {e}")
        return [{
            'category': 'system',
            'priority': 'low',
            'recommendation': 'Review system metrics generation process',
            'expected_impact': 'Improve metric analysis capabilities',
            'generation_error': str(e)
        }]

def _normalize_data_structure(data: Any, config: Dict) -> Dict[str, Any]:
    """Normalisiert Data Structure"""
    try:
        result = {
            'status': 'success',
            'processed_data': None,
            'processing_statistics': {}
        }
        
        if isinstance(data, dict):
            normalized_dict = {}
            for key, value in data.items():
                # Normalize keys (lowercase, replace spaces with underscores)
                normalized_key = str(key).lower().replace(' ', '_').replace('-', '_')
                
                # Recursively normalize nested structures
                if config.get('deep_processing', True) and isinstance(value, (dict, list)):
                    nested_result = _normalize_data_structure(value, config)
                    normalized_dict[normalized_key] = nested_result.get('processed_data', value)
                else:
                    normalized_dict[normalized_key] = value
            
            result['processed_data'] = normalized_dict
            result['processing_statistics']['keys_processed'] = len(data)
        
        elif isinstance(data, list):
            normalized_list = []
            for item in data:
                if config.get('deep_processing', True) and isinstance(item, (dict, list)):
                    nested_result = _normalize_data_structure(item, config)
                    normalized_list.append(nested_result.get('processed_data', item))
                else:
                    normalized_list.append(item)
            
            result['processed_data'] = normalized_list
            result['processing_statistics']['items_processed'] = len(data)
        
        else:
            result['processed_data'] = data
            result['processing_statistics']['primitive_type'] = type(data).__name__
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'processed_data': data,
            'processing_statistics': {}
        }

def _sanitize_data_structure(data: Any, config: Dict) -> Dict[str, Any]:
    """Sanitisiert Data Structure"""
    try:
        result = {
            'status': 'success',
            'processed_data': None,
            'processing_statistics': {}
        }
        
        if isinstance(data, dict):
            sanitized_dict = {}
            for key, value in data.items():
                # Sanitize key
                sanitized_key = _sanitize_string(str(key)) if isinstance(key, str) else key
                
                # Recursively sanitize nested structures
                if config.get('deep_processing', True) and isinstance(value, (dict, list)):
                    nested_result = _sanitize_data_structure(value, config)
                    sanitized_dict[sanitized_key] = nested_result.get('processed_data', value)
                elif isinstance(value, str):
                    sanitized_dict[sanitized_key] = _sanitize_string(value)
                else:
                    sanitized_dict[sanitized_key] = value
            
            result['processed_data'] = sanitized_dict
            result['processing_statistics']['keys_sanitized'] = len(data)
        
        elif isinstance(data, list):
            sanitized_list = []
            for item in data:
                if config.get('deep_processing', True) and isinstance(item, (dict, list)):
                    nested_result = _sanitize_data_structure(item, config)
                    sanitized_list.append(nested_result.get('processed_data', item))
                elif isinstance(item, str):
                    sanitized_list.append(_sanitize_string(item))
                else:
                    sanitized_list.append(item)
            
            result['processed_data'] = sanitized_list
            result['processing_statistics']['items_sanitized'] = len(data)
        
        elif isinstance(data, str):
            result['processed_data'] = _sanitize_string(data)
            result['processing_statistics']['string_sanitized'] = True
        
        else:
            result['processed_data'] = data
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'processed_data': data,
            'processing_statistics': {}
        }

def _transform_data_structure(data: Any, config: Dict) -> Dict[str, Any]:
    """Transformiert Data Structure"""
    try:
        result = {
            'status': 'success',
            'processed_data': None,
            'processing_statistics': {}
        }
        
        transformation_type = config.get('transformation_type', 'flatten')
        
        if transformation_type == 'flatten' and isinstance(data, dict):
            flattened = {}
            _flatten_dict(data, flattened, '')
            result['processed_data'] = flattened
            result['processing_statistics']['original_keys'] = len(data)
            result['processing_statistics']['flattened_keys'] = len(flattened)
        
        elif transformation_type == 'compact' and isinstance(data, dict):
            # Remove None values and empty structures
            compacted = _compact_dict(data)
            result['processed_data'] = compacted
        
        elif transformation_type == 'convert' and isinstance(data, dict):
            # Convert string numbers to actual numbers
            converted = _convert_string_numbers(data)
            result['processed_data'] = converted
        
        else:
            result['processed_data'] = data
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'processed_data': data,
            'processing_statistics': {}
        }

def _validate_data_structure(data: Any, config: Dict) -> Dict[str, Any]:
    """Validiert Data Structure"""
    try:
        result = {
            'status': 'success',
            'processed_data': data,
            'processing_statistics': {},
            'validation_results': {
                'is_valid': True,
                'validation_errors': [],
                'validation_warnings': []
            }
        }
        
        # Check data type
        expected_type = config.get('expected_type', 'any')
        if expected_type != 'any' and not _validate_field_type(data, expected_type):
            result['validation_results']['validation_errors'].append(f'Expected type {expected_type}, got {type(data).__name__}')
            result['validation_results']['is_valid'] = False
        
        # Check size limits
        max_size = config.get('max_size', float('inf'))
        if isinstance(data, (dict, list)) and len(data) > max_size:
            result['validation_results']['validation_errors'].append(f'Data size {len(data)} exceeds maximum {max_size}')
            result['validation_results']['is_valid'] = False
        
        # Check for required fields (if dict)
        if isinstance(data, dict):
            required_fields = config.get('required_fields', [])
            for field in required_fields:
                if field not in data or data[field] is None:
                    result['validation_results']['validation_errors'].append(f'Required field missing: {field}')
                    result['validation_results']['is_valid'] = False
        
        # Update status based on validation
        if not result['validation_results']['is_valid']:
            result['status'] = 'validation_failed'
        
        result['processing_statistics']['validation_performed'] = True
        result['processing_statistics']['errors_found'] = len(result['validation_results']['validation_errors'])
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'processed_data': data,
            'processing_statistics': {},
            'validation_results': {
                'is_valid': False,
                'validation_errors': [str(e)],
                'validation_warnings': []
            }
        }

def _compress_data_structure(data: Any, config: Dict) -> Dict[str, Any]:
    """Komprimiert Data Structure"""
    try:
        result = {
            'status': 'success',
            'processed_data': None,
            'processing_statistics': {}
        }
        
        # Simple compression by removing redundant data
        if isinstance(data, dict):
            compressed = {}
            for key, value in data.items():
                # Skip empty values if configured
                if config.get('skip_empty', True) and value in [None, '', [], {}]:
                    continue
                
                # Recursively compress nested structures
                if config.get('deep_processing', True) and isinstance(value, (dict, list)):
                    nested_result = _compress_data_structure(value, config)
                    compressed_value = nested_result.get('processed_data', value)
                    if compressed_value not in [None, '', [], {}]:
                        compressed[key] = compressed_value
                else:
                    compressed[key] = value
            
            result['processed_data'] = compressed
            result['processing_statistics']['original_size'] = len(data)
            result['processing_statistics']['compressed_size'] = len(compressed)
            result['processing_statistics']['compression_ratio'] = len(compressed) / max(1, len(data))
        
        elif isinstance(data, list):
            compressed = []
            for item in data:
                if config.get('skip_empty', True) and item in [None, '', [], {}]:
                    continue
                
                if config.get('deep_processing', True) and isinstance(item, (dict, list)):
                    nested_result = _compress_data_structure(item, config)
                    compressed_item = nested_result.get('processed_data', item)
                    if compressed_item not in [None, '', [], {}]:
                        compressed.append(compressed_item)
                else:
                    compressed.append(item)
            
            result['processed_data'] = compressed
            result['processing_statistics']['original_size'] = len(data)
            result['processing_statistics']['compressed_size'] = len(compressed)
        
        else:
            result['processed_data'] = data
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'processed_data': data,
            'processing_statistics': {}
        }

def _analyze_data_structure(data: Any, config: Dict) -> Dict[str, Any]:
    """Analysiert Data Structure"""
    try:
        result = {
            'status': 'success',
            'processed_data': data,
            'processing_statistics': {
                'data_type': type(data).__name__,
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        if isinstance(data, dict):
            result['processing_statistics'].update({
                'total_keys': len(data),
                'key_types': list(set(type(k).__name__ for k in data.keys())),
                'value_types': list(set(type(v).__name__ for v in data.values())),
                'nested_structures': sum(1 for v in data.values() if isinstance(v, (dict, list))),
                'null_values': sum(1 for v in data.values() if v is None),
                'empty_values': sum(1 for v in data.values() if v in ['', [], {}])
            })
        
        elif isinstance(data, list):
            result['processing_statistics'].update({
                'total_items': len(data),
                'item_types': list(set(type(item).__name__ for item in data)),
                'nested_structures': sum(1 for item in data if isinstance(item, (dict, list))),
                'null_values': sum(1 for item in data if item is None),
                'unique_items': len(set(str(item) for item in data if isinstance(item, (str, int, float, bool))))
            })
        
        elif isinstance(data, str):
            result['processing_statistics'].update({
                'string_length': len(data),
                'word_count': len(data.split()),
                'character_types': {
                    'alphabetic': sum(1 for c in data if c.isalpha()),
                    'numeric': sum(1 for c in data if c.isdigit()),
                    'whitespace': sum(1 for c in data if c.isspace()),
                    'special': sum(1 for c in data if not c.isalnum() and not c.isspace())
                }
            })
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'processed_data': data,
            'processing_statistics': {'analysis_error': str(e)}
        }

def _flatten_dict(d: dict, result: dict, prefix: str):
    """Flattens nested dictionary"""
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_dict(value, result, new_key)
        else:
            result[new_key] = value

def _compact_dict(d: dict) -> dict:
    """Removes None and empty values from dictionary"""
    if not isinstance(d, dict):
        return d
    
    compacted = {}
    for key, value in d.items():
        if value is None:
            continue
        elif isinstance(value, dict):
            compacted_value = _compact_dict(value)
            if compacted_value:  # Only add if not empty
                compacted[key] = compacted_value
        elif isinstance(value, list):
            compacted_list = [_compact_dict(item) if isinstance(item, dict) else item for item in value if item is not None]
            if compacted_list:  # Only add if not empty
                compacted[key] = compacted_list
        elif value != '':  # Keep non-empty strings
            compacted[key] = value
    
    return compacted

def _convert_string_numbers(data: Any) -> Any:
    """Converts string numbers to actual numbers"""
    if isinstance(data, dict):
        return {key: _convert_string_numbers(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_convert_string_numbers(item) for item in data]
    elif isinstance(data, str):
        # Try to convert to number
        try:
            if '.' in data:
                return float(data)
            else:
                return int(data)
        except ValueError:
            return data
    else:
        return data

# ====================================
# FILE OPERATION HELPER FUNCTIONS
# ====================================

def _read_file_operation(file_path: str, config: Dict) -> Dict[str, Any]:
    """Führt File Read Operation durch"""
    try:
        if not file_path or not os.path.exists(file_path):
            return {
                'status': 'error',
                'error': f'File not found: {file_path}'
            }
        
        # Check file size
        file_size = os.path.getsize(file_path)
        max_size_bytes = config.get('max_file_size_mb', 50) * 1024 * 1024
        
        if file_size > max_size_bytes:
            return {
                'status': 'error',
                'error': f'File size {file_size} exceeds maximum {max_size_bytes} bytes'
            }
        
        # Read file
        encoding = config.get('encoding', 'utf-8')
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
        
        return {
            'status': 'success',
            'file_content': content,
            'file_size': file_size,
            'encoding_used': encoding
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File read failed: {str(e)}'
        }

def _write_file_operation(file_path: str, file_data: Any, config: Dict) -> Dict[str, Any]:
    """Führt File Write Operation durch"""
    try:
        if not file_path:
            return {
                'status': 'error',
                'error': 'File path is required'
            }
        
        # Create directories if needed
        if config.get('create_directories', True):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Backup existing file if requested
        if config.get('backup_before_write', False) and os.path.exists(file_path):
            backup_path = f"{file_path}.backup.{int(time.time())}"
            import shutil
            shutil.copy2(file_path, backup_path)
        
        # Write file
        encoding = config.get('encoding', 'utf-8')
        content = str(file_data) if not isinstance(file_data, str) else file_data
        
        with open(file_path, 'w', encoding=encoding) as file:
            file.write(content)
        
        # Set permissions if specified
        if config.get('file_permissions'):
            os.chmod(file_path, config['file_permissions'])
        
        return {
            'status': 'success',
            'bytes_written': len(content.encode(encoding)),
            'file_path': file_path
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File write failed: {str(e)}'
        }

def _append_file_operation(file_path: str, file_data: Any, config: Dict) -> Dict[str, Any]:
    """Führt File Append Operation durch"""
    try:
        if not file_path:
            return {
                'status': 'error',
                'error': 'File path is required'
            }
        
        # Create directories if needed
        if config.get('create_directories', True):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Append to file
        encoding = config.get('encoding', 'utf-8')
        content = str(file_data) if not isinstance(file_data, str) else file_data
        
        with open(file_path, 'a', encoding=encoding) as file:
            file.write(content)
        
        return {
            'status': 'success',
            'bytes_appended': len(content.encode(encoding)),
            'file_path': file_path
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File append failed: {str(e)}'
        }

def _delete_file_operation(file_path: str, config: Dict) -> Dict[str, Any]:
    """Führt File Delete Operation durch"""
    try:
        if not file_path or not os.path.exists(file_path):
            return {
                'status': 'error',
                'error': f'File not found: {file_path}'
            }
        
        # Get file info before deletion
        file_size = os.path.getsize(file_path)
        
        # Delete file
        os.remove(file_path)
        
        return {
            'status': 'success',
            'deleted_file': file_path,
            'deleted_size': file_size
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File delete failed: {str(e)}'
        }

def _copy_file_operation(source_path: str, destination_path: str, config: Dict) -> Dict[str, Any]:
    """Führt File Copy Operation durch"""
    try:
        if not source_path or not os.path.exists(source_path):
            return {
                'status': 'error',
                'error': f'Source file not found: {source_path}'
            }
        
        if not destination_path:
            return {
                'status': 'error',
                'error': 'Destination path is required'
            }
        
        # Create destination directory if needed
        if config.get('create_directories', True):
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Copy file
        import shutil
        shutil.copy2(source_path, destination_path)
        
        return {
            'status': 'success',
            'source_path': source_path,
            'destination_path': destination_path,
            'copied_size': os.path.getsize(destination_path)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File copy failed: {str(e)}'
        }

def _move_file_operation(source_path: str, destination_path: str, config: Dict) -> Dict[str, Any]:
    """Führt File Move Operation durch"""
    try:
        if not source_path or not os.path.exists(source_path):
            return {
                'status': 'error',
                'error': f'Source file not found: {source_path}'
            }
        
        if not destination_path:
            return {
                'status': 'error',
                'error': 'Destination path is required'
            }
        
        # Get file size before move
        file_size = os.path.getsize(source_path)
        
        # Create destination directory if needed
        if config.get('create_directories', True):
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Move file
        import shutil
        shutil.move(source_path, destination_path)
        
        return {
            'status': 'success',
            'source_path': source_path,
            'destination_path': destination_path,
            'moved_size': file_size
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File move failed: {str(e)}'
        }

def _get_file_info_operation(file_path: str, config: Dict) -> Dict[str, Any]:
    """Holt File Information"""
    try:
        if not file_path or not os.path.exists(file_path):
            return {
                'status': 'error',
                'error': f'File not found: {file_path}'
            }
        
        stat = os.stat(file_path)
        
        return {
            'status': 'success',
            'file_info': {
                'path': file_path,
                'size': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'permissions': oct(stat.st_mode)[-3:],
                'is_file': os.path.isfile(file_path),
                'is_directory': os.path.isdir(file_path)
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File info failed: {str(e)}'
        }

def _validate_file_operation(file_path: str, config: Dict) -> Dict[str, Any]:
    """Validiert File"""
    try:
        validation_result = {
            'status': 'success',
            'file_valid': True,
            'validation_errors': [],
            'file_info': {}
        }
        
        # Check if file exists
        if not file_path or not os.path.exists(file_path):
            validation_result['validation_errors'].append(f'File not found: {file_path}')
            validation_result['file_valid'] = False
            validation_result['status'] = 'validation_failed'
            return validation_result
        
        # Get file info
        stat = os.stat(file_path)
        validation_result['file_info'] = {
            'size': stat.st_size,
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
        
        # Check file size
        max_size_bytes = config.get('max_file_size_mb', 50) * 1024 * 1024
        if stat.st_size > max_size_bytes:
            validation_result['validation_errors'].append(f'File size {stat.st_size} exceeds maximum {max_size_bytes}')
            validation_result['file_valid'] = False
        
        # Check file extension
        allowed_extensions = config.get('allowed_extensions', VALIDATION_RULES['file_fields']['allowed_extensions'])
        if allowed_extensions:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in allowed_extensions:
                validation_result['validation_errors'].append(f'File extension {file_ext} not allowed')
                validation_result['file_valid'] = False
        
        if not validation_result['file_valid']:
            validation_result['status'] = 'validation_failed'
        
        return validation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'File validation failed: {str(e)}',
            'file_valid': False,
            'validation_errors': [str(e)]
        }

# ====================================
# TIMESTAMP HELPER FUNCTIONS
# ====================================

def _get_current_timestamp(format_type: str, timezone: str) -> Dict[str, Any]:
    """Holt aktuellen Timestamp"""
    try:
        current_time = datetime.now()
        
        if format_type == 'iso':
            result = current_time.isoformat()
        elif format_type == 'unix':
            result = current_time.timestamp()
        elif format_type == 'human':
            result = current_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            result = current_time.isoformat()
        
        return {
            'status': 'success',
            'result': result,
            'format_used': format_type,
            'timezone': timezone
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def _parse_timestamp(timestamp_data: Any, format_type: str, timezone: str) -> Dict[str, Any]:
    """Parsed Timestamp"""
    try:
        if isinstance(timestamp_data, str):
            if format_type == 'iso':
                parsed_time = datetime.fromisoformat(timestamp_data.replace('Z', '+00:00'))
            else:
                parsed_time = datetime.fromisoformat(timestamp_data)
        elif isinstance(timestamp_data, (int, float)):
            parsed_time = datetime.fromtimestamp(timestamp_data)
        else:
            return {
                'status': 'error',
                'error': f'Unsupported timestamp data type: {type(timestamp_data)}'
            }
        
        return {
            'status': 'success',
            'result': parsed_time.isoformat(),
            'parsed_datetime': parsed_time
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def _format_timestamp(timestamp_data: Any, format_type: str, timezone: str) -> Dict[str, Any]:
    """Formatiert Timestamp"""
    try:
        # Parse timestamp data first
        parse_result = _parse_timestamp(timestamp_data, 'iso', timezone)
        if parse_result['status'] != 'success':
            return parse_result
        
        parsed_time = parse_result['parsed_datetime']
        
        if format_type == 'iso':
            result = parsed_time.isoformat()
        elif format_type == 'unix':
            result = parsed_time.timestamp()
        elif format_type == 'human':
            result = parsed_time.strftime('%Y-%m-%d %H:%M:%S')
        elif format_type == 'date_only':
            result = parsed_time.strftime('%Y-%m-%d')
        elif format_type == 'time_only':
            result = parsed_time.strftime('%H:%M:%S')
        else:
            result = parsed_time.isoformat()
        
        return {
            'status': 'success',
            'result': result,
            'format_used': format_type
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def _calculate_timestamp_difference(timestamp_data: Any, timezone: str) -> Dict[str, Any]:
    """Berechnet Timestamp Differenz"""
    try:
        if not isinstance(timestamp_data, dict) or 'start_time' not in timestamp_data or 'end_time' not in timestamp_data:
            return {
                'status': 'error',
                'error': 'timestamp_data must contain start_time and end_time'
            }
        
        # Parse timestamps
        start_parse = _parse_timestamp(timestamp_data['start_time'], 'iso', timezone)
        end_parse = _parse_timestamp(timestamp_data['end_time'], 'iso', timezone)
        
        if start_parse['status'] != 'success' or end_parse['status'] != 'success':
            return {
                'status': 'error',
                'error': 'Failed to parse timestamps'
            }
        
        start_time = start_parse['parsed_datetime']
        end_time = end_parse['parsed_datetime']
        
        # Calculate difference
        time_diff = end_time - start_time
        
        result = {
            'status': 'success',
            'result': {
                'total_seconds': time_diff.total_seconds(),
                'days': time_diff.days,
                'hours': time_diff.seconds // 3600,
                'minutes': (time_diff.seconds % 3600) // 60,
                'seconds': time_diff.seconds % 60,
                'human_readable': str(time_diff)
            }
        }
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def _validate_timestamp(timestamp_data: Any, format_type: str) -> Dict[str, Any]:
    """Validiert Timestamp"""
    try:
        parse_result = _parse_timestamp(timestamp_data, format_type, 'UTC')
        
        if parse_result['status'] == 'success':
            return {
                'status': 'success',
                'result': {
                    'is_valid': True,
                    'parsed_timestamp': parse_result['result']
                }
            }
        else:
            return {
                'status': 'success',
                'result': {
                    'is_valid': False,
                    'validation_error': parse_result['error']
                }
            }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def _convert_timestamp(timestamp_data: Any, source_format: str, target_format: str, timezone: str) -> Dict[str, Any]:
    """Konvertiert Timestamp zwischen Formaten"""
    try:
        # Parse with source format
        parse_result = _parse_timestamp(timestamp_data, source_format, timezone)
        if parse_result['status'] != 'success':
            return parse_result
        
        # Format with target format
        format_result = _format_timestamp(parse_result['parsed_datetime'], target_format, timezone)
        
        return format_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

# Update __all__ to include all new functions
__all__.extend([
    '_generate_metric_insights',
    '_generate_metric_recommendations',
    '_normalize_data_structure',
    '_sanitize_data_structure',
    '_transform_data_structure',
    '_validate_data_structure',
    '_compress_data_structure',
    '_analyze_data_structure',
    '_read_file_operation',
    '_write_file_operation',
    '_append_file_operation',
    '_delete_file_operation',
    '_copy_file_operation',
    '_move_file_operation',
    '_get_file_info_operation',
    '_validate_file_operation',
    '_get_current_timestamp',
    '_parse_timestamp',
    '_format_timestamp',
    '_calculate_timestamp_difference',
    '_validate_timestamp',
    '_convert_timestamp'
])