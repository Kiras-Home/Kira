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