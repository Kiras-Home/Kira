"""
Core Constants Module
System Constants, Configuration Values, Static Definitions
"""

from datetime import timedelta
from typing import Dict, List, Any

# ====================================
# SYSTEM CONFIGURATION
# ====================================

SYSTEM_CONFIG = {
    'version': '2.0.0',
    'environment': 'development',
    'debug_mode': True,
    'max_memory_usage_mb': 512,
    'max_processing_time_seconds': 30,
    'default_timeout_seconds': 10,
    'max_concurrent_operations': 5,
    'enable_detailed_logging': True,
    'enable_performance_monitoring': True,
    'enable_automatic_cleanup': True,
    'cleanup_interval_minutes': 15,
    'health_check_interval_seconds': 60
}

# ====================================
# ERROR CODES
# ====================================

ERROR_CODES = {
    # System Errors (1000-1999)
    'SYSTEM_INITIALIZATION_FAILED': 1001,
    'SYSTEM_VALIDATION_FAILED': 1002,
    'SYSTEM_HEALTH_CHECK_FAILED': 1003,
    'SYSTEM_RESOURCES_EXHAUSTED': 1004,
    'SYSTEM_TIMEOUT': 1005,
    'SYSTEM_MAINTENANCE_REQUIRED': 1006,
    
    # Memory Errors (2000-2999)
    'MEMORY_ALLOCATION_FAILED': 2001,
    'MEMORY_CORRUPTION_DETECTED': 2002,
    'MEMORY_LIMIT_EXCEEDED': 2003,
    'MEMORY_LEAK_DETECTED': 2004,
    'MEMORY_SYSTEM_UNAVAILABLE': 2005,
    
    # AI Processing Errors (3000-3999)
    'AI_PROCESSING_FAILED': 3001,
    'AI_MODEL_UNAVAILABLE': 3002,
    'AI_CONTEXT_OVERFLOW': 3003,
    'AI_RESPONSE_GENERATION_FAILED': 3004,
    'AI_LEARNING_DISABLED': 3005,
    
    # Database Errors (4000-4999)
    'DATABASE_CONNECTION_FAILED': 4001,
    'DATABASE_QUERY_FAILED': 4002,
    'DATABASE_SCHEMA_INVALID': 4003,
    'DATABASE_CORRUPTION_DETECTED': 4004,
    'DATABASE_BACKUP_FAILED': 4005,
    
    # Network/API Errors (5000-5999)
    'NETWORK_CONNECTION_FAILED': 5001,
    'API_REQUEST_FAILED': 5002,
    'API_RATE_LIMIT_EXCEEDED': 5003,
    'API_AUTHENTICATION_FAILED': 5004,
    'API_TIMEOUT': 5005,
    
    # Validation Errors (6000-6999)
    'VALIDATION_FAILED': 6001,
    'INPUT_VALIDATION_FAILED': 6002,
    'DATA_FORMAT_INVALID': 6003,
    'PARAMETER_MISSING': 6004,
    'PARAMETER_INVALID': 6005,
    
    # File System Errors (7000-7999)
    'FILE_NOT_FOUND': 7001,
    'FILE_ACCESS_DENIED': 7002,
    'FILE_CORRUPTION_DETECTED': 7003,
    'DISK_SPACE_INSUFFICIENT': 7004,
    'FILE_SIZE_EXCEEDED': 7005,
    
    # Generic Errors (9000-9999)
    'UNKNOWN_ERROR': 9001,
    'OPERATION_NOT_SUPPORTED': 9002,
    'CONFIGURATION_ERROR': 9003,
    'DEPENDENCY_UNAVAILABLE': 9004,
    'RESOURCE_UNAVAILABLE': 9005
}

# ====================================
# RESPONSE TEMPLATES
# ====================================

RESPONSE_TEMPLATES = {
    'success': {
        'success': True,
        'status': 'success',
        'message': 'Operation completed successfully',
        'data': None,
        'timestamp': None,
        'execution_time_ms': None
    },
    
    'error': {
        'success': False,
        'status': 'error',
        'error_code': None,
        'error_message': None,
        'details': None,
        'timestamp': None,
        'suggestion': None
    },
    
    'validation_error': {
        'success': False,
        'status': 'validation_error',
        'error_code': 'VALIDATION_FAILED',
        'validation_errors': [],
        'timestamp': None
    },
    
    'processing': {
        'success': None,
        'status': 'processing',
        'message': 'Operation in progress',
        'progress_percentage': 0,
        'estimated_completion': None,
        'timestamp': None
    },
    
    'partial_success': {
        'success': True,
        'status': 'partial_success',
        'message': 'Operation partially completed',
        'successful_operations': [],
        'failed_operations': [],
        'timestamp': None
    }
}

# ====================================
# VALIDATION RULES
# ====================================

VALIDATION_RULES = {
    'string_fields': {
        'min_length': 1,
        'max_length': 1000,
        'allowed_characters': 'alphanumeric_extended',
        'trim_whitespace': True
    },
    
    'numeric_fields': {
        'min_value': -999999,
        'max_value': 999999,
        'allow_decimals': True,
        'decimal_places': 2
    },
    
    'email_fields': {
        'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'max_length': 254
    },
    
    'date_fields': {
        'format': 'ISO8601',
        'allow_future_dates': True,
        'min_year': 1900,
        'max_year': 2100
    },
    
    'file_fields': {
        'max_size_mb': 10,
        'allowed_extensions': ['.txt', '.json', '.yaml', '.csv', '.log'],
        'scan_for_malware': False
    },
    
    'memory_fields': {
        'max_content_length': 10000,
        'required_fields': ['content', 'timestamp'],
        'optional_fields': ['context', 'importance']
    }
}

# ====================================
# PERFORMANCE THRESHOLDS
# ====================================

PERFORMANCE_THRESHOLDS = {
    'response_time': {
        'excellent_ms': 100,
        'good_ms': 300,
        'acceptable_ms': 1000,
        'poor_ms': 3000,
        'critical_ms': 5000
    },
    
    'memory_usage': {
        'low_percentage': 30,
        'moderate_percentage': 60,
        'high_percentage': 80,
        'critical_percentage': 95
    },
    
    'cpu_usage': {
        'low_percentage': 25,
        'moderate_percentage': 50,
        'high_percentage': 75,
        'critical_percentage': 90
    },
    
    'database_performance': {
        'query_time_ms': {
            'fast': 50,
            'normal': 200,
            'slow': 1000,
            'very_slow': 5000
        },
        'connection_pool': {
            'optimal_utilization': 70,
            'high_utilization': 85,
            'critical_utilization': 95
        }
    },
    
    'system_health': {
        'excellent_score': 0.95,
        'good_score': 0.80,
        'fair_score': 0.60,
        'poor_score': 0.40,
        'critical_score': 0.20
    }
}

# ====================================
# LOGGING CONFIGURATION
# ====================================

LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'max_log_file_size_mb': 50,
    'max_log_files': 5,
    'log_rotation_enabled': True,
    'console_logging_enabled': True,
    'file_logging_enabled': True,
    'log_file_path': 'logs/kira_system.log',
    
    'log_levels': {
        'CRITICAL': 50,
        'ERROR': 40,
        'WARNING': 30,
        'INFO': 20,
        'DEBUG': 10,
        'NOTSET': 0
    },
    
    'component_log_levels': {
        'system': 'INFO',
        'memory': 'DEBUG',
        'ai': 'INFO',
        'database': 'WARNING',
        'api': 'INFO',
        'monitoring': 'DEBUG'
    }
}

# ====================================
# SYSTEM LIMITS
# ====================================

SYSTEM_LIMITS = {
    'max_concurrent_users': 100,
    'max_session_duration_hours': 24,
    'max_memory_entries': 10000,
    'max_conversation_length': 50,
    'max_file_upload_size_mb': 100,
    'max_api_requests_per_minute': 60,
    'max_database_connections': 20,
    'max_background_tasks': 10,
    'max_cache_entries': 1000,
    'max_log_entries_per_session': 1000
}

# ====================================
# FEATURE FLAGS
# ====================================

FEATURE_FLAGS = {
    'enable_ai_learning': True,
    'enable_memory_persistence': True,
    'enable_conversation_history': True,
    'enable_performance_monitoring': True,
    'enable_auto_backup': True,
    'enable_real_time_monitoring': True,
    'enable_advanced_analytics': False,
    'enable_machine_learning': False,
    'enable_distributed_processing': False,
    'enable_experimental_features': False
}

# ====================================
# DEFAULT CONFIGURATIONS
# ====================================

DEFAULT_MEMORY_CONFIG = {
    'short_term_capacity': 100,
    'working_memory_capacity': 20,
    'long_term_persistence': True,
    'importance_threshold': 0.5,
    'auto_cleanup_enabled': True,
    'compression_enabled': False
}

DEFAULT_AI_CONFIG = {
    'response_style': 'helpful',
    'creativity_level': 0.7,
    'verbosity': 'moderate',
    'learning_enabled': True,
    'context_awareness': True,
    'emotional_intelligence': True
}

DEFAULT_MONITORING_CONFIG = {
    'health_check_interval': 60,
    'metric_collection_interval': 30,
    'alert_threshold_critical': 0.9,
    'alert_threshold_warning': 0.7,
    'retention_days': 30,
    'enable_alerts': True
}

# ====================================
# COMPONENT INITIALIZATION
# ====================================

def _initialize_constants_component():
    """Initialisiert Constants Component"""
    try:
        # Validate all configurations
        _validate_system_config()
        _validate_performance_thresholds()
        _validate_logging_config()
        
        return True
    except Exception as e:
        raise Exception(f"Constants component initialization failed: {str(e)}")

def _validate_system_config():
    """Validiert System Configuration"""
    required_keys = ['version', 'environment', 'max_memory_usage_mb']
    for key in required_keys:
        if key not in SYSTEM_CONFIG:
            raise ValueError(f"Missing required system config key: {key}")

def _validate_performance_thresholds():
    """Validiert Performance Thresholds"""
    if not isinstance(PERFORMANCE_THRESHOLDS, dict):
        raise ValueError("Performance thresholds must be a dictionary")

def _validate_logging_config():
    """Validiert Logging Configuration"""
    required_keys = ['log_level', 'log_format']
    for key in required_keys:
        if key not in LOGGING_CONFIG:
            raise ValueError(f"Missing required logging config key: {key}")

# Export all constants
__all__ = [
    'SYSTEM_CONFIG',
    'ERROR_CODES',
    'RESPONSE_TEMPLATES',
    'VALIDATION_RULES',
    'PERFORMANCE_THRESHOLDS',
    'LOGGING_CONFIG',
    'SYSTEM_LIMITS',
    'FEATURE_FLAGS',
    'DEFAULT_MEMORY_CONFIG',
    'DEFAULT_AI_CONFIG',
    'DEFAULT_MONITORING_CONFIG',
    '_initialize_constants_component'
]