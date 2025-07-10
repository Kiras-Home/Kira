"""
Kira Core Module
System Validation, Utils, Constants, Error Handling und Core Functionality

Module:
- system.py: System Validation, Core System Operations, System Health Checks
- utils.py: Utility Functions, Helper Methods, Common Operations
- constants.py: System Constants, Configuration Values, Static Definitions
- error_handling.py: Error Management, Exception Handling, Logging Systems
"""

from .system import (
    validate_system_integrity,
    perform_system_diagnostics,
    monitor_system_health,
    execute_system_operations
)

from .utils import (
    format_response,
    validate_request_data,
    calculate_metrics,
    process_data_structures,
    handle_file_operations,
    manage_timestamps
)

from .constants import (
    SYSTEM_CONFIG,
    ERROR_CODES,
    RESPONSE_TEMPLATES,
    VALIDATION_RULES,
    PERFORMANCE_THRESHOLDS,
    LOGGING_CONFIG
)

__all__ = [
    # System Operations
    'validate_system_integrity',
    'perform_system_diagnostics',
    'monitor_system_health',
    'execute_system_operations',
    
    # Utility Functions
    'format_response',
    'validate_request_data',
    'calculate_metrics',
    'process_data_structures',
    'handle_file_operations',
    'manage_timestamps',
    
    # Constants
    'SYSTEM_CONFIG',
    'ERROR_CODES',
    'RESPONSE_TEMPLATES',
    'VALIDATION_RULES',
    'PERFORMANCE_THRESHOLDS',
    'LOGGING_CONFIG'
]

# Core Module Information
CORE_MODULE_INFO = {
    'version': '2.0.0',
    'description': 'Kira Core System Module',
    'components': ['system', 'utils', 'constants'],
    'initialization_timestamp': None,
    'module_status': 'ready'
}

def initialize_core_module() -> dict:
    """
    Initialisiert das Core Module
    
    Returns:
        dict: Initialization Status
    """
    try:
        from datetime import datetime
        
        # Initialize all core components
        initialization_results = {
            'system_init': False,
            'utils_init': False,
            'constants_init': False,
            'errors': []
        }
        
        # Initialize system component
        try:
            from .system import _initialize_system_component
            _initialize_system_component()
            initialization_results['system_init'] = True
        except Exception as e:
            initialization_results['errors'].append(f"System init failed: {str(e)}")
        
        # Initialize utils component
        try:
            from .utils import _initialize_utils_component
            _initialize_utils_component()
            initialization_results['utils_init'] = True
        except Exception as e:
            initialization_results['errors'].append(f"Utils init failed: {str(e)}")
        
        # Initialize constants component
        try:
            from .constants import _initialize_constants_component
            _initialize_constants_component()
            initialization_results['constants_init'] = True
        except Exception as e:
            initialization_results['errors'].append(f"Constants init failed: {str(e)}")
        
        # Update module status
        CORE_MODULE_INFO['initialization_timestamp'] = datetime.now().isoformat()
        
        success_count = sum([
            initialization_results['system_init'],
            initialization_results['utils_init'],
            initialization_results['constants_init']
        ])
        
        if success_count >= 2:  # At least 2 out of 3 components must succeed
            CORE_MODULE_INFO['module_status'] = 'initialized'
        else:
            CORE_MODULE_INFO['module_status'] = 'partial_failure'
        
        return {
            'success': success_count >= 2,
            'initialization_results': initialization_results,
            'components_initialized': success_count,
            'total_components': 3,
            'module_info': CORE_MODULE_INFO.copy()
        }
        
    except Exception as e:
        CORE_MODULE_INFO['module_status'] = 'initialization_failed'
        return {
            'success': False,
            'error': str(e),
            'module_info': CORE_MODULE_INFO.copy()
        }

def get_core_module_status() -> dict:
    """
    Holt Core Module Status
    
    Returns:
        dict: Module Status Information
    """
    try:
        return {
            'success': True,
            'module_info': CORE_MODULE_INFO.copy(),
            'components_available': [
                'system' if 'validate_system_integrity' in globals() else None,
                'utils' if 'format_response' in globals() else None,
                'constants' if 'SYSTEM_CONFIG' in globals() else None
            ]
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'module_info': CORE_MODULE_INFO.copy()
        }

# Export additional utilities
__all__.extend([
    'initialize_core_module',
    'get_core_module_status',
    'CORE_MODULE_INFO'
])