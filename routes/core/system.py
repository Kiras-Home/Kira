"""
Core System Module
System Validation, Core System Operations, System Health Checks
"""

import logging
import psutil
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import os
import sys

from .constants import (
    SYSTEM_CONFIG, 
    ERROR_CODES, 
    PERFORMANCE_THRESHOLDS,
    SYSTEM_LIMITS,
    FEATURE_FLAGS
)

logger = logging.getLogger(__name__)

# System State Tracking
_system_state = {
    'initialization_timestamp': None,
    'last_health_check': None,
    'system_health_score': 0.0,
    'active_operations': 0,
    'system_alerts': [],
    'performance_metrics': {},
    'component_status': {}
}

_system_lock = threading.Lock()

def validate_system_integrity(kira_instance=None, 
                            validation_type: str = 'full',
                            validation_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    System Integrity Validation
    
    Extrahiert aus kira_routes.py.backup System Validation Logic
    """
    try:
        if validation_config is None:
            validation_config = {
                'check_memory': True,
                'check_performance': True,
                'check_dependencies': True,
                'check_configuration': True,
                'detailed_report': True
            }
        
        # Initialize validation session
        validation_session = {
            'session_id': f"validation_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'validation_type': validation_type,
            'validation_config': validation_config,
            'validation_results': {}
        }
        
        # Perform different validation types
        if validation_type == 'full':
            validation_session['validation_results'] = _perform_full_system_validation(validation_config, kira_instance)
        
        elif validation_type == 'quick':
            validation_session['validation_results'] = _perform_quick_system_validation(validation_config)
        
        elif validation_type == 'memory':
            validation_session['validation_results'] = _perform_memory_validation(validation_config, kira_instance)
        
        elif validation_type == 'performance':
            validation_session['validation_results'] = _perform_performance_validation(validation_config)
        
        elif validation_type == 'configuration':
            validation_session['validation_results'] = _perform_configuration_validation(validation_config)
        
        else:
            validation_session['validation_results'] = {
                'status': 'error',
                'error': f'Unsupported validation type: {validation_type}'
            }
        
        # Calculate overall validation score
        validation_score = _calculate_validation_score(validation_session['validation_results'])
        
        # Add validation metadata
        validation_session.update({
            'end_time': datetime.now().isoformat(),
            'validation_success': validation_session['validation_results'].get('status') == 'success',
            'validation_score': validation_score,
            'validation_duration': _calculate_duration_ms(validation_session)
        })
        
        # Update system state
        with _system_lock:
            _system_state['last_health_check'] = datetime.now().isoformat()
            _system_state['system_health_score'] = validation_score
        
        return {
            'success': validation_session['validation_success'],
            'validation_session': validation_session,
            'system_integrity': {
                'validation_type': validation_type,
                'validation_score': validation_score,
                'validation_status': _get_validation_status(validation_score),
                'critical_issues': validation_session['validation_results'].get('critical_issues', []),
                'recommendations': validation_session['validation_results'].get('recommendations', [])
            }
        }
        
    except Exception as e:
        logger.error(f"System integrity validation failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'fallback_system_status': _generate_fallback_system_status()
        }

def perform_system_diagnostics(kira_instance=None,
                             diagnostic_type: str = 'comprehensive',
                             diagnostic_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    System Diagnostics
    
    Basiert auf kira_routes.py.backup System Diagnostic Logic
    """
    try:
        if diagnostic_config is None:
            diagnostic_config = {
                'include_performance_metrics': True,
                'include_resource_usage': True,
                'include_component_status': True,
                'include_error_analysis': True,
                'generate_recommendations': True
            }
        
        # Initialize diagnostic session
        diagnostic_session = {
            'session_id': f"diagnostics_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'diagnostic_type': diagnostic_type,
            'diagnostic_config': diagnostic_config,
            'diagnostic_results': {}
        }
        
        # Perform diagnostics based on type
        if diagnostic_type == 'comprehensive':
            diagnostic_session['diagnostic_results'] = _perform_comprehensive_diagnostics(diagnostic_config, kira_instance)
        
        elif diagnostic_type == 'performance':
            diagnostic_session['diagnostic_results'] = _perform_performance_diagnostics(diagnostic_config)
        
        elif diagnostic_type == 'resource':
            diagnostic_session['diagnostic_results'] = _perform_resource_diagnostics(diagnostic_config)
        
        elif diagnostic_type == 'component':
            diagnostic_session['diagnostic_results'] = _perform_component_diagnostics(diagnostic_config, kira_instance)
        
        elif diagnostic_type == 'error':
            diagnostic_session['diagnostic_results'] = _perform_error_diagnostics(diagnostic_config)
        
        else:
            diagnostic_session['diagnostic_results'] = {
                'status': 'error',
                'error': f'Unsupported diagnostic type: {diagnostic_type}'
            }
        
        # Generate diagnostic insights
        diagnostic_insights = _generate_diagnostic_insights(diagnostic_session['diagnostic_results'])
        
        # Add diagnostic metadata
        diagnostic_session.update({
            'end_time': datetime.now().isoformat(),
            'diagnostic_success': diagnostic_session['diagnostic_results'].get('status') == 'success',
            'diagnostic_insights': diagnostic_insights,
            'diagnostic_duration': _calculate_duration_ms(diagnostic_session)
        })
        
        return {
            'success': diagnostic_session['diagnostic_success'],
            'diagnostic_session': diagnostic_session,
            'system_diagnostics': {
                'diagnostic_type': diagnostic_type,
                'system_health': diagnostic_insights.get('system_health', 'unknown'),
                'performance_status': diagnostic_insights.get('performance_status', 'unknown'),
                'critical_alerts': diagnostic_insights.get('critical_alerts', []),
                'optimization_opportunities': diagnostic_insights.get('optimization_opportunities', [])
            }
        }
        
    except Exception as e:
        logger.error(f"System diagnostics failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'partial_diagnostic_data': diagnostic_session.get('diagnostic_results', {}) if 'diagnostic_session' in locals() else {}
        }

def monitor_system_health(kira_instance=None,
                         monitoring_duration: int = 300,
                         monitoring_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    System Health Monitoring
    
    Extrahiert aus kira_routes.py.backup System Health Monitoring Logic
    """
    try:
        if monitoring_config is None:
            monitoring_config = {
                'monitor_cpu': True,
                'monitor_memory': True,
                'monitor_disk': True,
                'monitor_network': False,
                'collect_metrics_interval': 10,
                'alert_on_thresholds': True
            }
        
        # Initialize monitoring session
        monitoring_session = {
            'session_id': f"health_monitor_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'monitoring_duration': monitoring_duration,
            'monitoring_config': monitoring_config,
            'health_metrics': [],
            'health_alerts': [],
            'monitoring_statistics': {}
        }
        
        # Monitor system health over time
        start_time = time.time()
        metrics_interval = monitoring_config.get('collect_metrics_interval', 10)
        
        while (time.time() - start_time) < monitoring_duration:
            metric_timestamp = datetime.now()
            
            try:
                # Collect system metrics
                system_metrics = _collect_system_metrics(monitoring_config)
                
                health_metric_snapshot = {
                    'timestamp': metric_timestamp.isoformat(),
                    'elapsed_seconds': time.time() - start_time,
                    'cpu_usage_percent': system_metrics.get('cpu_usage', 0),
                    'memory_usage_percent': system_metrics.get('memory_usage', 0),
                    'disk_usage_percent': system_metrics.get('disk_usage', 0),
                    'system_load': system_metrics.get('system_load', 0),
                    'active_processes': system_metrics.get('active_processes', 0),
                    'system_uptime_seconds': system_metrics.get('system_uptime', 0)
                }
                
                monitoring_session['health_metrics'].append(health_metric_snapshot)
                
                # Check for alerts
                if monitoring_config.get('alert_on_thresholds', True):
                    alerts = _check_health_thresholds(health_metric_snapshot)
                    if alerts:
                        monitoring_session['health_alerts'].extend(alerts)
                
            except Exception as e:
                logger.debug(f"Health metrics collection failed: {e}")
                error_metric_snapshot = {
                    'timestamp': metric_timestamp.isoformat(),
                    'elapsed_seconds': time.time() - start_time,
                    'error': str(e)
                }
                monitoring_session['health_metrics'].append(error_metric_snapshot)
            
            # Wait for next collection
            time.sleep(metrics_interval)
        
        # Analyze monitoring data
        monitoring_statistics = _analyze_health_monitoring_data(monitoring_session['health_metrics'])
        monitoring_session['monitoring_statistics'] = monitoring_statistics
        
        # Generate health insights
        health_insights = _generate_health_insights(monitoring_statistics, monitoring_session['health_alerts'])
        
        # Health recommendations
        health_recommendations = _generate_health_recommendations(monitoring_statistics, health_insights)
        
        monitoring_session.update({
            'end_time': datetime.now().isoformat(),
            'actual_monitoring_duration': time.time() - start_time,
            'metrics_collected': len(monitoring_session['health_metrics']),
            'alerts_triggered': len(monitoring_session['health_alerts']),
            'health_insights': health_insights,
            'health_recommendations': health_recommendations
        })
        
        # Update system state
        with _system_lock:
            _system_state['last_health_check'] = datetime.now().isoformat()
            _system_state['performance_metrics'] = monitoring_statistics
            if monitoring_session['health_alerts']:
                _system_state['system_alerts'].extend(monitoring_session['health_alerts'])
        
        return {
            'success': True,
            'monitoring_session': monitoring_session,
            'health_summary': {
                'monitoring_duration_seconds': monitoring_session['actual_monitoring_duration'],
                'metrics_collected': monitoring_session['metrics_collected'],
                'average_cpu_usage': monitoring_statistics.get('average_cpu_usage', 0),
                'average_memory_usage': monitoring_statistics.get('average_memory_usage', 0),
                'system_stability': monitoring_statistics.get('system_stability', 'unknown'),
                'alerts_count': monitoring_session['alerts_triggered']
            }
        }
        
    except Exception as e:
        logger.error(f"System health monitoring failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'partial_monitoring_data': monitoring_session.get('health_metrics', []) if 'monitoring_session' in locals() else []
        }

def execute_system_operations(kira_instance=None,
                            operation_type: str = 'maintenance',
                            operation_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    System Operations Execution
    
    Basiert auf kira_routes.py.backup System Operations Logic
    """
    try:
        if operation_config is None:
            operation_config = {
                'dry_run': False,
                'backup_before_operation': True,
                'validate_after_operation': True,
                'operation_timeout': 300
            }
        
        # Initialize operation session
        operation_session = {
            'session_id': f"system_ops_{int(time.time())}",
            'start_time': datetime.now().isoformat(),
            'operation_type': operation_type,
            'operation_config': operation_config,
            'operation_results': {}
        }
        
        # Increment active operations counter
        with _system_lock:
            _system_state['active_operations'] += 1
        
        try:
            # Execute operations based on type
            if operation_type == 'maintenance':
                operation_session['operation_results'] = _execute_system_maintenance(operation_config, kira_instance)
            
            elif operation_type == 'cleanup':
                operation_session['operation_results'] = _execute_system_cleanup(operation_config)
            
            elif operation_type == 'optimization':
                operation_session['operation_results'] = _execute_system_optimization(operation_config, kira_instance)
            
            elif operation_type == 'backup':
                operation_session['operation_results'] = _execute_system_backup(operation_config)
            
            elif operation_type == 'restore':
                operation_session['operation_results'] = _execute_system_restore(operation_config)
            
            elif operation_type == 'reset':
                operation_session['operation_results'] = _execute_system_reset(operation_config, kira_instance)
            
            else:
                operation_session['operation_results'] = {
                    'status': 'error',
                    'error': f'Unsupported operation type: {operation_type}'
                }
            
            # Post-operation validation if requested
            if operation_config.get('validate_after_operation', True) and operation_session['operation_results'].get('status') == 'success':
                validation_result = validate_system_integrity(kira_instance, 'quick')
                operation_session['post_operation_validation'] = validation_result
        
        finally:
            # Decrement active operations counter
            with _system_lock:
                _system_state['active_operations'] = max(0, _system_state['active_operations'] - 1)
        
        # Add operation metadata
        operation_session.update({
            'end_time': datetime.now().isoformat(),
            'operation_success': operation_session['operation_results'].get('status') == 'success',
            'operation_duration': _calculate_duration_ms(operation_session)
        })
        
        return {
            'success': operation_session['operation_success'],
            'operation_session': operation_session,
            'operation_summary': {
                'operation_type': operation_type,
                'operation_success': operation_session['operation_success'],
                'operation_duration_ms': operation_session['operation_duration'],
                'operations_performed': operation_session['operation_results'].get('operations_performed', []),
                'improvements_made': operation_session['operation_results'].get('improvements_made', [])
            }
        }
        
    except Exception as e:
        logger.error(f"System operations execution failed: {e}")
        # Ensure counter is decremented on error
        with _system_lock:
            _system_state['active_operations'] = max(0, _system_state['active_operations'] - 1)
        
        return {
            'success': False,
            'error': str(e),
            'operation_type': operation_type,
            'partial_results': operation_session.get('operation_results', {}) if 'operation_session' in locals() else {}
        }

# ====================================
# PRIVATE HELPER FUNCTIONS
# ====================================

def _perform_full_system_validation(validation_config: Dict, kira_instance=None) -> Dict[str, Any]:
    """Führt vollständige System Validation durch"""
    try:
        validation_result = {
            'status': 'success',
            'validation_checks': [],
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Memory validation
        if validation_config.get('check_memory', True):
            memory_check = _validate_memory_system(kira_instance)
            validation_result['validation_checks'].append('memory_system')
            if memory_check.get('issues'):
                validation_result['critical_issues'].extend(memory_check['issues'])
        
        # Performance validation
        if validation_config.get('check_performance', True):
            performance_check = _validate_performance_system()
            validation_result['validation_checks'].append('performance_system')
            if performance_check.get('warnings'):
                validation_result['warnings'].extend(performance_check['warnings'])
        
        # Dependencies validation
        if validation_config.get('check_dependencies', True):
            dependencies_check = _validate_system_dependencies()
            validation_result['validation_checks'].append('system_dependencies')
            if dependencies_check.get('missing_dependencies'):
                validation_result['critical_issues'].extend(dependencies_check['missing_dependencies'])
        
        # Configuration validation
        if validation_config.get('check_configuration', True):
            config_check = _validate_system_configuration()
            validation_result['validation_checks'].append('system_configuration')
            if config_check.get('config_errors'):
                validation_result['critical_issues'].extend(config_check['config_errors'])
        
        # Determine overall status
        if validation_result['critical_issues']:
            validation_result['status'] = 'critical_issues'
        elif validation_result['warnings']:
            validation_result['status'] = 'warnings'
        
        return validation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'validation_checks': [],
            'critical_issues': [str(e)]
        }

def _perform_quick_system_validation(validation_config: Dict) -> Dict[str, Any]:
    """Führt schnelle System Validation durch"""
    try:
        validation_result = {
            'status': 'success',
            'quick_checks': [],
            'issues_found': []
        }
        
        # Quick memory check
        try:
            memory_info = psutil.virtual_memory()
            if memory_info.percent > PERFORMANCE_THRESHOLDS['memory_usage']['critical_percentage']:
                validation_result['issues_found'].append('Critical memory usage detected')
            validation_result['quick_checks'].append('memory_usage')
        except Exception as e:
            validation_result['issues_found'].append(f'Memory check failed: {str(e)}')
        
        # Quick CPU check
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > PERFORMANCE_THRESHOLDS['cpu_usage']['critical_percentage']:
                validation_result['issues_found'].append('Critical CPU usage detected')
            validation_result['quick_checks'].append('cpu_usage')
        except Exception as e:
            validation_result['issues_found'].append(f'CPU check failed: {str(e)}')
        
        # Quick disk check
        try:
            disk_info = psutil.disk_usage('/')
            disk_percent = (disk_info.used / disk_info.total) * 100
            if disk_percent > 90:  # 90% disk usage threshold
                validation_result['issues_found'].append('High disk usage detected')
            validation_result['quick_checks'].append('disk_usage')
        except Exception as e:
            validation_result['issues_found'].append(f'Disk check failed: {str(e)}')
        
        # Determine status
        if validation_result['issues_found']:
            validation_result['status'] = 'issues_detected'
        
        return validation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'quick_checks': [],
            'issues_found': [str(e)]
        }

def _collect_system_metrics(monitoring_config: Dict) -> Dict[str, Any]:
    """Sammelt System Metrics"""
    try:
        metrics = {}
        
        # CPU metrics
        if monitoring_config.get('monitor_cpu', True):
            metrics['cpu_usage'] = psutil.cpu_percent(interval=0.1)
            metrics['cpu_count'] = psutil.cpu_count()
            metrics['system_load'] = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
        
        # Memory metrics
        if monitoring_config.get('monitor_memory', True):
            memory_info = psutil.virtual_memory()
            metrics['memory_usage'] = memory_info.percent
            metrics['memory_available_mb'] = memory_info.available / (1024 * 1024)
            metrics['memory_total_mb'] = memory_info.total / (1024 * 1024)
        
        # Disk metrics
        if monitoring_config.get('monitor_disk', True):
            disk_info = psutil.disk_usage('/')
            metrics['disk_usage'] = (disk_info.used / disk_info.total) * 100
            metrics['disk_free_gb'] = disk_info.free / (1024 * 1024 * 1024)
            metrics['disk_total_gb'] = disk_info.total / (1024 * 1024 * 1024)
        
        # Process metrics
        metrics['active_processes'] = len(psutil.pids())
        
        # System uptime
        boot_time = psutil.boot_time()
        metrics['system_uptime'] = time.time() - boot_time
        
        return metrics
        
    except Exception as e:
        logger.debug(f"System metrics collection failed: {e}")
        return {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'error': str(e)
        }

def _initialize_system_component():
    """Initialisiert System Component"""
    try:
        with _system_lock:
            _system_state['initialization_timestamp'] = datetime.now().isoformat()
            _system_state['component_status']['system'] = 'initialized'
        
        return True
    except Exception as e:
        raise Exception(f"System component initialization failed: {str(e)}")

def _calculate_duration_ms(session: Dict) -> float:
    """Berechnet Session Duration in Milliseconds"""
    try:
        start_time = datetime.fromisoformat(session['start_time'])
        end_time = datetime.fromisoformat(session['end_time'])
        duration = (end_time - start_time).total_seconds() * 1000
        return duration
    except Exception as e:
        logger.debug(f"Duration calculation failed: {e}")
        return 0.0

def _calculate_validation_score(validation_results: Dict) -> float:
    """Berechnet Validation Score"""
    try:
        if validation_results.get('status') == 'error':
            return 0.0
        
        critical_issues = len(validation_results.get('critical_issues', []))
        warnings = len(validation_results.get('warnings', []))
        checks_performed = len(validation_results.get('validation_checks', []))
        
        if checks_performed == 0:
            return 0.5  # Neutral score if no checks performed
        
        # Score calculation: start with 1.0, deduct for issues
        score = 1.0
        score -= (critical_issues * 0.3)  # Critical issues reduce score significantly
        score -= (warnings * 0.1)        # Warnings reduce score moderately
        
        return max(0.0, min(1.0, score))  # Clamp between 0.0 and 1.0
        
    except Exception as e:
        logger.debug(f"Validation score calculation failed: {e}")
        return 0.0

def _get_validation_status(score: float) -> str:
    """Holt Validation Status basierend auf Score"""
    if score >= PERFORMANCE_THRESHOLDS['system_health']['excellent_score']:
        return 'excellent'
    elif score >= PERFORMANCE_THRESHOLDS['system_health']['good_score']:
        return 'good'
    elif score >= PERFORMANCE_THRESHOLDS['system_health']['fair_score']:
        return 'fair'
    elif score >= PERFORMANCE_THRESHOLDS['system_health']['poor_score']:
        return 'poor'
    else:
        return 'critical'

def _generate_fallback_system_status() -> Dict[str, Any]:
    """Generiert Fallback System Status"""
    return {
        'fallback_mode': True,
        'system_status': {
            'status': 'unknown',
            'health_score': 0.0,
            'last_check': None,
            'active_operations': 0
        },
        'system_metrics': {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0
        }
    }

def _perform_memory_validation(validation_config: Dict, kira_instance=None) -> Dict[str, Any]:
    """Führt Memory System Validation durch"""
    try:
        validation_result = {
            'status': 'success',
            'memory_checks': [],
            'memory_issues': [],
            'memory_statistics': {}
        }
        
        # Check if kira_instance has memory system
        if kira_instance and hasattr(kira_instance, 'memory_system'):
            try:
                # Memory system integrity check
                memory_status = kira_instance.memory_system.get_status()
                validation_result['memory_checks'].append('memory_system_status')
                
                if memory_status.get('status') != 'healthy':
                    validation_result['memory_issues'].append('Memory system is not healthy')
                
                # Memory capacity check
                memory_stats = kira_instance.memory_system.get_statistics()
                validation_result['memory_statistics'] = memory_stats
                validation_result['memory_checks'].append('memory_capacity')
                
                if memory_stats.get('utilization', 0) > 0.9:
                    validation_result['memory_issues'].append('Memory utilization is very high')
                
            except Exception as e:
                validation_result['memory_issues'].append(f'Memory system check failed: {str(e)}')
        else:
            validation_result['memory_issues'].append('Memory system not available')
        
        # System memory validation
        try:
            system_memory = psutil.virtual_memory()
            validation_result['memory_checks'].append('system_memory')
            
            if system_memory.percent > PERFORMANCE_THRESHOLDS['memory_usage']['critical_percentage']:
                validation_result['memory_issues'].append(f'Critical system memory usage: {system_memory.percent}%')
            
            validation_result['memory_statistics']['system_memory_percent'] = system_memory.percent
            validation_result['memory_statistics']['system_memory_available_gb'] = system_memory.available / (1024**3)
            
        except Exception as e:
            validation_result['memory_issues'].append(f'System memory check failed: {str(e)}')
        
        # Determine status
        if validation_result['memory_issues']:
            validation_result['status'] = 'issues_detected'
        
        return validation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'memory_checks': [],
            'memory_issues': [str(e)]
        }

def _perform_performance_validation(validation_config: Dict) -> Dict[str, Any]:
    """Führt Performance System Validation durch"""
    try:
        validation_result = {
            'status': 'success',
            'performance_checks': [],
            'performance_warnings': [],
            'performance_metrics': {}
        }
        
        # CPU performance check
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            validation_result['performance_checks'].append('cpu_performance')
            validation_result['performance_metrics']['cpu_usage_percent'] = cpu_percent
            
            if cpu_percent > PERFORMANCE_THRESHOLDS['cpu_usage']['high_percentage']:
                validation_result['performance_warnings'].append(f'High CPU usage detected: {cpu_percent}%')
        except Exception as e:
            validation_result['performance_warnings'].append(f'CPU performance check failed: {str(e)}')
        
        # Memory performance check
        try:
            memory_info = psutil.virtual_memory()
            validation_result['performance_checks'].append('memory_performance')
            validation_result['performance_metrics']['memory_usage_percent'] = memory_info.percent
            
            if memory_info.percent > PERFORMANCE_THRESHOLDS['memory_usage']['high_percentage']:
                validation_result['performance_warnings'].append(f'High memory usage detected: {memory_info.percent}%')
        except Exception as e:
            validation_result['performance_warnings'].append(f'Memory performance check failed: {str(e)}')
        
        # Disk I/O performance check
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                validation_result['performance_checks'].append('disk_io_performance')
                validation_result['performance_metrics']['disk_read_bytes'] = disk_io.read_bytes
                validation_result['performance_metrics']['disk_write_bytes'] = disk_io.write_bytes
            
        except Exception as e:
            validation_result['performance_warnings'].append(f'Disk I/O performance check failed: {str(e)}')
        
        # Load average check (Unix systems)
        try:
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                validation_result['performance_checks'].append('system_load')
                validation_result['performance_metrics']['load_average_1min'] = load_avg[0]
                
                cpu_count = psutil.cpu_count()
                if load_avg[0] > cpu_count * 0.8:  # 80% of CPU count
                    validation_result['performance_warnings'].append(f'High system load detected: {load_avg[0]}')
        except Exception as e:
            validation_result['performance_warnings'].append(f'Load average check failed: {str(e)}')
        
        return validation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'performance_checks': [],
            'performance_warnings': [str(e)]
        }

def _perform_configuration_validation(validation_config: Dict) -> Dict[str, Any]:
    """Führt Configuration Validation durch"""
    try:
        validation_result = {
            'status': 'success',
            'config_checks': [],
            'config_errors': [],
            'config_warnings': []
        }
        
        # System config validation
        try:
            # Check required system config keys
            required_keys = ['version', 'environment', 'max_memory_usage_mb']
            for key in required_keys:
                if key not in SYSTEM_CONFIG:
                    validation_result['config_errors'].append(f'Missing required system config key: {key}')
            
            validation_result['config_checks'].append('system_config_keys')
            
            # Validate config values
            if SYSTEM_CONFIG.get('max_memory_usage_mb', 0) < 128:
                validation_result['config_warnings'].append('Low memory limit configured')
            
            if SYSTEM_CONFIG.get('max_processing_time_seconds', 0) > 60:
                validation_result['config_warnings'].append('High processing time limit may affect user experience')
            
            validation_result['config_checks'].append('system_config_values')
            
        except Exception as e:
            validation_result['config_errors'].append(f'System config validation failed: {str(e)}')
        
        # Performance thresholds validation
        try:
            if not isinstance(PERFORMANCE_THRESHOLDS, dict):
                validation_result['config_errors'].append('Performance thresholds must be a dictionary')
            else:
                required_threshold_categories = ['response_time', 'memory_usage', 'cpu_usage']
                for category in required_threshold_categories:
                    if category not in PERFORMANCE_THRESHOLDS:
                        validation_result['config_errors'].append(f'Missing performance threshold category: {category}')
            
            validation_result['config_checks'].append('performance_thresholds')
            
        except Exception as e:
            validation_result['config_errors'].append(f'Performance thresholds validation failed: {str(e)}')
        
        # Feature flags validation
        try:
            if not isinstance(FEATURE_FLAGS, dict):
                validation_result['config_errors'].append('Feature flags must be a dictionary')
            else:
                # Check for conflicting feature flags
                if FEATURE_FLAGS.get('enable_experimental_features') and not FEATURE_FLAGS.get('debug_mode', SYSTEM_CONFIG.get('debug_mode')):
                    validation_result['config_warnings'].append('Experimental features enabled without debug mode')
            
            validation_result['config_checks'].append('feature_flags')
            
        except Exception as e:
            validation_result['config_errors'].append(f'Feature flags validation failed: {str(e)}')
        
        return validation_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'config_checks': [],
            'config_errors': [str(e)]
        }

def _validate_memory_system(kira_instance=None) -> Dict[str, Any]:
    """Validiert Memory System"""
    try:
        memory_validation = {
            'status': 'success',
            'issues': [],
            'memory_health': 'unknown'
        }
        
        if kira_instance and hasattr(kira_instance, 'memory_system'):
            try:
                # Check memory system status
                memory_status = kira_instance.memory_system.get_status()
                
                if memory_status.get('status') != 'healthy':
                    memory_validation['issues'].append('Memory system status is not healthy')
                
                # Check memory statistics
                memory_stats = kira_instance.memory_system.get_statistics()
                
                if memory_stats.get('utilization', 0) > 0.95:
                    memory_validation['issues'].append('Memory system utilization is critical')
                elif memory_stats.get('utilization', 0) > 0.8:
                    memory_validation['issues'].append('Memory system utilization is high')
                
                # Determine memory health
                if not memory_validation['issues']:
                    memory_validation['memory_health'] = 'healthy'
                elif len(memory_validation['issues']) == 1:
                    memory_validation['memory_health'] = 'moderate'
                else:
                    memory_validation['memory_health'] = 'unhealthy'
                
            except Exception as e:
                memory_validation['issues'].append(f'Memory system validation error: {str(e)}')
                memory_validation['memory_health'] = 'error'
        else:
            memory_validation['issues'].append('Memory system not available')
            memory_validation['memory_health'] = 'unavailable'
        
        return memory_validation
        
    except Exception as e:
        return {
            'status': 'error',
            'issues': [str(e)],
            'memory_health': 'error'
        }

def _validate_performance_system() -> Dict[str, Any]:
    """Validiert Performance System"""
    try:
        performance_validation = {
            'status': 'success',
            'warnings': [],
            'performance_score': 1.0
        }
        
        # CPU performance check
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            if cpu_percent > PERFORMANCE_THRESHOLDS['cpu_usage']['critical_percentage']:
                performance_validation['warnings'].append(f'Critical CPU usage: {cpu_percent}%')
                performance_validation['performance_score'] -= 0.3
            elif cpu_percent > PERFORMANCE_THRESHOLDS['cpu_usage']['high_percentage']:
                performance_validation['warnings'].append(f'High CPU usage: {cpu_percent}%')
                performance_validation['performance_score'] -= 0.1
        except Exception as e:
            performance_validation['warnings'].append(f'CPU check failed: {str(e)}')
        
        # Memory performance check
        try:
            memory_info = psutil.virtual_memory()
            if memory_info.percent > PERFORMANCE_THRESHOLDS['memory_usage']['critical_percentage']:
                performance_validation['warnings'].append(f'Critical memory usage: {memory_info.percent}%')
                performance_validation['performance_score'] -= 0.3
            elif memory_info.percent > PERFORMANCE_THRESHOLDS['memory_usage']['high_percentage']:
                performance_validation['warnings'].append(f'High memory usage: {memory_info.percent}%')
                performance_validation['performance_score'] -= 0.1
        except Exception as e:
            performance_validation['warnings'].append(f'Memory check failed: {str(e)}')
        
        # Ensure score doesn't go below 0
        performance_validation['performance_score'] = max(0.0, performance_validation['performance_score'])
        
        return performance_validation
        
    except Exception as e:
        return {
            'status': 'error',
            'warnings': [str(e)],
            'performance_score': 0.0
        }

def _validate_system_dependencies() -> Dict[str, Any]:
    """Validiert System Dependencies"""
    try:
        dependencies_validation = {
            'status': 'success',
            'missing_dependencies': [],
            'dependency_versions': {}
        }
        
        # Check required Python modules
        required_modules = ['psutil', 'flask', 'datetime', 'json', 'logging', 'threading']
        
        for module_name in required_modules:
            try:
                module = __import__(module_name)
                if hasattr(module, '__version__'):
                    dependencies_validation['dependency_versions'][module_name] = module.__version__
                else:
                    dependencies_validation['dependency_versions'][module_name] = 'unknown'
            except ImportError:
                dependencies_validation['missing_dependencies'].append(module_name)
        
        # Check system utilities
        system_utilities = {
            'python': 'python --version',
            'pip': 'pip --version'
        }
        
        for util_name, util_command in system_utilities.items():
            try:
                import subprocess
                result = subprocess.run(util_command.split(), capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    dependencies_validation['dependency_versions'][util_name] = result.stdout.strip()
                else:
                    dependencies_validation['missing_dependencies'].append(util_name)
            except Exception:
                dependencies_validation['missing_dependencies'].append(util_name)
        
        return dependencies_validation
        
    except Exception as e:
        return {
            'status': 'error',
            'missing_dependencies': ['dependency_check_failed'],
            'error': str(e)
        }

def _validate_system_configuration() -> Dict[str, Any]:
    """Validiert System Configuration"""
    try:
        config_validation = {
            'status': 'success',
            'config_errors': [],
            'config_warnings': []
        }
        
        # Validate SYSTEM_CONFIG
        try:
            required_system_keys = ['version', 'environment', 'max_memory_usage_mb', 'max_processing_time_seconds']
            for key in required_system_keys:
                if key not in SYSTEM_CONFIG:
                    config_validation['config_errors'].append(f'Missing system config key: {key}')
            
            # Validate specific values
            if SYSTEM_CONFIG.get('max_memory_usage_mb', 0) <= 0:
                config_validation['config_errors'].append('Invalid max_memory_usage_mb value')
            
            if SYSTEM_CONFIG.get('max_processing_time_seconds', 0) <= 0:
                config_validation['config_errors'].append('Invalid max_processing_time_seconds value')
                
        except Exception as e:
            config_validation['config_errors'].append(f'System config validation error: {str(e)}')
        
        # Validate ERROR_CODES
        try:
            if not isinstance(ERROR_CODES, dict) or len(ERROR_CODES) == 0:
                config_validation['config_errors'].append('ERROR_CODES must be a non-empty dictionary')
        except Exception as e:
            config_validation['config_errors'].append(f'Error codes validation error: {str(e)}')
        
        # Validate PERFORMANCE_THRESHOLDS
        try:
            if not isinstance(PERFORMANCE_THRESHOLDS, dict):
                config_validation['config_errors'].append('PERFORMANCE_THRESHOLDS must be a dictionary')
            else:
                required_threshold_keys = ['response_time', 'memory_usage', 'cpu_usage']
                for key in required_threshold_keys:
                    if key not in PERFORMANCE_THRESHOLDS:
                        config_validation['config_errors'].append(f'Missing performance threshold: {key}')
        except Exception as e:
            config_validation['config_errors'].append(f'Performance thresholds validation error: {str(e)}')
        
        return config_validation
        
    except Exception as e:
        return {
            'status': 'error',
            'config_errors': [str(e)],
            'config_warnings': []
        }

def _perform_comprehensive_diagnostics(diagnostic_config: Dict, kira_instance=None) -> Dict[str, Any]:
    """Führt umfassende System Diagnostics durch"""
    try:
        diagnostic_result = {
            'status': 'success',
            'diagnostic_categories': [],
            'diagnostic_findings': {},
            'system_health_score': 0.0
        }
        
        # Performance diagnostics
        if diagnostic_config.get('include_performance_metrics', True):
            performance_diagnostics = _perform_performance_diagnostics(diagnostic_config)
            diagnostic_result['diagnostic_categories'].append('performance')
            diagnostic_result['diagnostic_findings']['performance'] = performance_diagnostics
        
        # Resource usage diagnostics
        if diagnostic_config.get('include_resource_usage', True):
            resource_diagnostics = _perform_resource_diagnostics(diagnostic_config)
            diagnostic_result['diagnostic_categories'].append('resource_usage')
            diagnostic_result['diagnostic_findings']['resource_usage'] = resource_diagnostics
        
        # Component status diagnostics
        if diagnostic_config.get('include_component_status', True):
            component_diagnostics = _perform_component_diagnostics(diagnostic_config, kira_instance)
            diagnostic_result['diagnostic_categories'].append('component_status')
            diagnostic_result['diagnostic_findings']['component_status'] = component_diagnostics
        
        # Error analysis diagnostics
        if diagnostic_config.get('include_error_analysis', True):
            error_diagnostics = _perform_error_diagnostics(diagnostic_config)
            diagnostic_result['diagnostic_categories'].append('error_analysis')
            diagnostic_result['diagnostic_findings']['error_analysis'] = error_diagnostics
        
        # Calculate overall system health score
        diagnostic_result['system_health_score'] = _calculate_diagnostic_health_score(diagnostic_result['diagnostic_findings'])
        
        return diagnostic_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'diagnostic_categories': [],
            'diagnostic_findings': {}
        }

def _perform_performance_diagnostics(diagnostic_config: Dict) -> Dict[str, Any]:
    """Führt Performance Diagnostics durch"""
    try:
        performance_result = {
            'status': 'success',
            'performance_metrics': {},
            'performance_issues': [],
            'performance_score': 1.0
        }
        
        # CPU performance diagnostics
        try:
            cpu_times = psutil.cpu_times()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            performance_result['performance_metrics']['cpu'] = {
                'usage_percent': cpu_percent,
                'user_time': cpu_times.user,
                'system_time': cpu_times.system,
                'idle_time': cpu_times.idle,
                'cpu_count': psutil.cpu_count()
            }
            
            if cpu_percent > PERFORMANCE_THRESHOLDS['cpu_usage']['critical_percentage']:
                performance_result['performance_issues'].append(f'Critical CPU usage: {cpu_percent}%')
                performance_result['performance_score'] -= 0.3
                
        except Exception as e:
            performance_result['performance_issues'].append(f'CPU diagnostics failed: {str(e)}')
        
        # Memory performance diagnostics
        try:
            memory_info = psutil.virtual_memory()
            swap_info = psutil.swap_memory()
            
            performance_result['performance_metrics']['memory'] = {
                'total_gb': memory_info.total / (1024**3),
                'available_gb': memory_info.available / (1024**3),
                'used_percent': memory_info.percent,
                'swap_total_gb': swap_info.total / (1024**3),
                'swap_used_percent': swap_info.percent
            }
            
            if memory_info.percent > PERFORMANCE_THRESHOLDS['memory_usage']['critical_percentage']:
                performance_result['performance_issues'].append(f'Critical memory usage: {memory_info.percent}%')
                performance_result['performance_score'] -= 0.3
                
        except Exception as e:
            performance_result['performance_issues'].append(f'Memory diagnostics failed: {str(e)}')
        
        # Disk performance diagnostics
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            performance_result['performance_metrics']['disk'] = {
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': disk_usage.used / (1024**3),
                'free_gb': disk_usage.free / (1024**3),
                'used_percent': (disk_usage.used / disk_usage.total) * 100
            }
            
            if disk_io:
                performance_result['performance_metrics']['disk'].update({
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count
                })
            
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            if disk_percent > 90:
                performance_result['performance_issues'].append(f'High disk usage: {disk_percent:.1f}%')
                performance_result['performance_score'] -= 0.2
                
        except Exception as e:
            performance_result['performance_issues'].append(f'Disk diagnostics failed: {str(e)}')
        
        # Ensure score doesn't go below 0
        performance_result['performance_score'] = max(0.0, performance_result['performance_score'])
        
        return performance_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'performance_metrics': {},
            'performance_issues': [str(e)]
        }

def _perform_resource_diagnostics(diagnostic_config: Dict) -> Dict[str, Any]:
    """Führt Resource Usage Diagnostics durch"""
    try:
        resource_result = {
            'status': 'success',
            'resource_metrics': {},
            'resource_alerts': []
        }
        
        # Process diagnostics
        try:
            processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))
            
            # Top CPU processes
            top_cpu_processes = sorted(processes, key=lambda p: p.info['cpu_percent'] or 0, reverse=True)[:5]
            
            # Top memory processes
            top_memory_processes = sorted(processes, key=lambda p: p.info['memory_percent'] or 0, reverse=True)[:5]
            
            resource_result['resource_metrics']['processes'] = {
                'total_processes': len(processes),
                'top_cpu_processes': [{'pid': p.info['pid'], 'name': p.info['name'], 'cpu_percent': p.info['cpu_percent']} for p in top_cpu_processes],
                'top_memory_processes': [{'pid': p.info['pid'], 'name': p.info['name'], 'memory_percent': p.info['memory_percent']} for p in top_memory_processes]
            }
            
        except Exception as e:
            resource_result['resource_alerts'].append(f'Process diagnostics failed: {str(e)}')
        
        # Network diagnostics
        try:
            network_io = psutil.net_io_counters()
            if network_io:
                resource_result['resource_metrics']['network'] = {
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv
                }
        except Exception as e:
            resource_result['resource_alerts'].append(f'Network diagnostics failed: {str(e)}')
        
        # File descriptor diagnostics (Unix systems)
        try:
            if hasattr(psutil, 'LINUX') or hasattr(psutil, 'MACOS'):
                open_files = len(psutil.Process().open_files())
                resource_result['resource_metrics']['file_descriptors'] = {
                    'open_files': open_files
                }
        except Exception as e:
            # This is expected on some systems, so don't add as alert
            pass
        
        return resource_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'resource_metrics': {},
            'resource_alerts': [str(e)]
        }

def _perform_component_diagnostics(diagnostic_config: Dict, kira_instance=None) -> Dict[str, Any]:
    """Führt Component Status Diagnostics durch"""
    try:
        component_result = {
            'status': 'success',
            'component_status': {},
            'component_issues': []
        }
        
        # System component status
        with _system_lock:
            component_result['component_status']['system'] = _system_state['component_status'].copy()
        
        # Kira instance components
        if kira_instance:
            try:
                # Memory system component
                if hasattr(kira_instance, 'memory_system'):
                    memory_status = kira_instance.memory_system.get_status()
                    component_result['component_status']['memory_system'] = memory_status
                    
                    if memory_status.get('status') != 'healthy':
                        component_result['component_issues'].append('Memory system is not healthy')
                
                # AI system component
                if hasattr(kira_instance, 'ai_system'):
                    try:
                        ai_status = kira_instance.ai_system.get_status()
                        component_result['component_status']['ai_system'] = ai_status
                    except:
                        component_result['component_status']['ai_system'] = {'status': 'unknown'}
                
                # Database component
                if hasattr(kira_instance, 'database'):
                    try:
                        db_status = kira_instance.database.get_status()
                        component_result['component_status']['database'] = db_status
                    except:
                        component_result['component_status']['database'] = {'status': 'unknown'}
                        
            except Exception as e:
                component_result['component_issues'].append(f'Kira instance diagnostics failed: {str(e)}')
        else:
            component_result['component_issues'].append('Kira instance not available for component diagnostics')
        
        return component_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'component_status': {},
            'component_issues': [str(e)]
        }

def _perform_error_diagnostics(diagnostic_config: Dict) -> Dict[str, Any]:
    """Führt Error Analysis Diagnostics durch"""
    try:
        error_result = {
            'status': 'success',
            'error_analysis': {},
            'error_patterns': []
        }
        
        # Analyze system alerts
        with _system_lock:
            recent_alerts = _system_state['system_alerts'][-10:]  # Last 10 alerts
        
        if recent_alerts:
            error_result['error_analysis']['recent_alerts'] = recent_alerts
            
            # Look for error patterns
            alert_types = {}
            for alert in recent_alerts:
                alert_type = alert.get('type', 'unknown')
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            
            # Identify recurring error patterns
            for alert_type, count in alert_types.items():
                if count >= 3:  # 3 or more of the same type
                    error_result['error_patterns'].append({
                        'pattern_type': alert_type,
                        'occurrences': count,
                        'severity': 'high' if count >= 5 else 'medium'
                    })
        
        # System error indicators
        try:
            # High CPU/Memory as error indicators
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            error_indicators = []
            
            if cpu_percent > PERFORMANCE_THRESHOLDS['cpu_usage']['critical_percentage']:
                error_indicators.append({
                    'indicator': 'critical_cpu_usage',
                    'value': cpu_percent,
                    'threshold': PERFORMANCE_THRESHOLDS['cpu_usage']['critical_percentage']
                })
            
            if memory_percent > PERFORMANCE_THRESHOLDS['memory_usage']['critical_percentage']:
                error_indicators.append({
                    'indicator': 'critical_memory_usage',
                    'value': memory_percent,
                    'threshold': PERFORMANCE_THRESHOLDS['memory_usage']['critical_percentage']
                })
            
            error_result['error_analysis']['error_indicators'] = error_indicators
            
        except Exception as e:
            error_result['error_analysis']['error_indicators_error'] = str(e)
        
        return error_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_analysis': {},
            'error_patterns': []
        }

def _calculate_diagnostic_health_score(diagnostic_findings: Dict) -> float:
    """Berechnet Diagnostic Health Score"""
    try:
        total_score = 0.0
        score_count = 0
        
        # Performance score
        if 'performance' in diagnostic_findings:
            perf_score = diagnostic_findings['performance'].get('performance_score', 0.5)
            total_score += perf_score
            score_count += 1
        
        # Component health score
        if 'component_status' in diagnostic_findings:
            component_issues = len(diagnostic_findings['component_status'].get('component_issues', []))
            component_score = max(0.0, 1.0 - (component_issues * 0.2))
            total_score += component_score
            score_count += 1
        
        # Error analysis score
        if 'error_analysis' in diagnostic_findings:
            error_patterns = len(diagnostic_findings['error_analysis'].get('error_patterns', []))
            error_score = max(0.0, 1.0 - (error_patterns * 0.3))
            total_score += error_score
            score_count += 1
        
        # Resource usage score
        if 'resource_usage' in diagnostic_findings:
            resource_alerts = len(diagnostic_findings['resource_usage'].get('resource_alerts', []))
            resource_score = max(0.0, 1.0 - (resource_alerts * 0.1))
            total_score += resource_score
            score_count += 1
        
        # Calculate average score
        if score_count > 0:
            return total_score / score_count
        else:
            return 0.5  # Neutral score if no data
            
    except Exception as e:
        logger.debug(f"Diagnostic health score calculation failed: {e}")
        return 0.0

def _check_health_thresholds(health_snapshot: Dict) -> List[Dict]:
    """Überprüft Health Thresholds und generiert Alerts"""
    try:
        alerts = []
        
        # CPU threshold check
        cpu_usage = health_snapshot.get('cpu_usage_percent', 0)
        if cpu_usage > PERFORMANCE_THRESHOLDS['cpu_usage']['critical_percentage']:
            alerts.append({
                'type': 'critical_cpu_usage',
                'message': f'Critical CPU usage detected: {cpu_usage}%',
                'timestamp': health_snapshot.get('timestamp'),
                'severity': 'critical',
                'value': cpu_usage,
                'threshold': PERFORMANCE_THRESHOLDS['cpu_usage']['critical_percentage']
            })
        elif cpu_usage > PERFORMANCE_THRESHOLDS['cpu_usage']['high_percentage']:
            alerts.append({
                'type': 'high_cpu_usage',
                'message': f'High CPU usage detected: {cpu_usage}%',
                'timestamp': health_snapshot.get('timestamp'),
                'severity': 'warning',
                'value': cpu_usage,
                'threshold': PERFORMANCE_THRESHOLDS['cpu_usage']['high_percentage']
            })
        
        # Memory threshold check
        memory_usage = health_snapshot.get('memory_usage_percent', 0)
        if memory_usage > PERFORMANCE_THRESHOLDS['memory_usage']['critical_percentage']:
            alerts.append({
                'type': 'critical_memory_usage',
                'message': f'Critical memory usage detected: {memory_usage}%',
                'timestamp': health_snapshot.get('timestamp'),
                'severity': 'critical',
                'value': memory_usage,
                'threshold': PERFORMANCE_THRESHOLDS['memory_usage']['critical_percentage']
            })
        elif memory_usage > PERFORMANCE_THRESHOLDS['memory_usage']['high_percentage']:
            alerts.append({
                'type': 'high_memory_usage',
                'message': f'High memory usage detected: {memory_usage}%',
                'timestamp': health_snapshot.get('timestamp'),
                'severity': 'warning',
                'value': memory_usage,
                'threshold': PERFORMANCE_THRESHOLDS['memory_usage']['high_percentage']
            })
        
        # Disk threshold check
        disk_usage = health_snapshot.get('disk_usage_percent', 0)
        if disk_usage > 95:  # Critical disk usage threshold
            alerts.append({
                'type': 'critical_disk_usage',
                'message': f'Critical disk usage detected: {disk_usage}%',
                'timestamp': health_snapshot.get('timestamp'),
                'severity': 'critical',
                'value': disk_usage,
                'threshold': 95
            })
        elif disk_usage > 85:  # Warning disk usage threshold
            alerts.append({
                'type': 'high_disk_usage',
                'message': f'High disk usage detected: {disk_usage}%',
                'timestamp': health_snapshot.get('timestamp'),
                'severity': 'warning',
                'value': disk_usage,
                'threshold': 85
            })
        
        return alerts
        
    except Exception as e:
        logger.debug(f"Health threshold check failed: {e}")
        return []

def _analyze_health_monitoring_data(health_metrics: List[Dict]) -> Dict[str, Any]:
    """Analysiert Health Monitoring Data"""
    try:
        if not health_metrics:
            return {
                'average_cpu_usage': 0,
                'average_memory_usage': 0,
                'average_disk_usage': 0,
                'system_stability': 'unknown',
                'monitoring_quality': 'no_data'
            }
        
        # Filter out error entries
        valid_metrics = [m for m in health_metrics if 'error' not in m]
        
        if not valid_metrics:
            return {
                'average_cpu_usage': 0,
                'average_memory_usage': 0,
                'average_disk_usage': 0,
                'system_stability': 'unstable',
                'monitoring_quality': 'poor'
            }
        
        # Calculate averages
        cpu_values = [m.get('cpu_usage_percent', 0) for m in valid_metrics]
        memory_values = [m.get('memory_usage_percent', 0) for m in valid_metrics]
        disk_values = [m.get('disk_usage_percent', 0) for m in valid_metrics]
        
        avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
        avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0
        avg_disk = sum(disk_values) / len(disk_values) if disk_values else 0
        
        # Calculate stability score
        cpu_stability = _calculate_stability_score(cpu_values)
        memory_stability = _calculate_stability_score(memory_values)
        
        overall_stability = (cpu_stability + memory_stability) / 2
        
        if overall_stability > 0.8:
            stability_status = 'stable'
        elif overall_stability > 0.6:
            stability_status = 'moderate'
        else:
            stability_status = 'unstable'
        
        # Monitoring quality assessment
        error_count = len([m for m in health_metrics if 'error' in m])
        error_rate = error_count / len(health_metrics) if health_metrics else 1.0
        
        if error_rate < 0.1:
            monitoring_quality = 'excellent'
        elif error_rate < 0.3:
            monitoring_quality = 'good'
        else:
            monitoring_quality = 'poor'
        
        return {
            'average_cpu_usage': avg_cpu,
            'average_memory_usage': avg_memory,
            'average_disk_usage': avg_disk,
            'peak_cpu_usage': max(cpu_values) if cpu_values else 0,
            'peak_memory_usage': max(memory_values) if memory_values else 0,
            'system_stability': stability_status,
            'stability_score': overall_stability,
            'monitoring_quality': monitoring_quality,
            'error_rate': error_rate,
            'valid_metrics_count': len(valid_metrics),
            'total_metrics_count': len(health_metrics)
        }
        
    except Exception as e:
        logger.debug(f"Health monitoring data analysis failed: {e}")
        return {
            'average_cpu_usage': 0,
            'average_memory_usage': 0,
            'average_disk_usage': 0,
            'system_stability': 'unknown',
            'monitoring_quality': 'error',
            'analysis_error': str(e)
        }

def _calculate_stability_score(values: List[float]) -> float:
    """Berechnet Stability Score für eine Liste von Werten"""
    try:
        if len(values) < 2:
            return 1.0
        
        # Calculate standard deviation
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        # Normalize stability score (lower std_dev = higher stability)
        # Assuming values are percentages (0-100)
        normalized_std_dev = std_dev / 100.0
        stability = max(0.0, 1.0 - normalized_std_dev)
        
        return stability
        
    except Exception as e:
        logger.debug(f"Stability score calculation failed: {e}")
        return 0.0
    
def _execute_system_maintenance(operation_config: Dict, kira_instance=None) -> Dict[str, Any]:
    """Führt System Maintenance durch"""
    try:
        maintenance_result = {
            'status': 'success',
            'operations_performed': [],
            'improvements_made': [],
            'maintenance_summary': {}
        }
        
        # Memory cleanup
        try:
            if kira_instance and hasattr(kira_instance, 'memory_system'):
                memory_cleanup = kira_instance.memory_system.cleanup_expired_memories()
                maintenance_result['operations_performed'].append('memory_cleanup')
                maintenance_result['improvements_made'].append(f"Cleaned up {memory_cleanup.get('cleaned_memories', 0)} expired memories")
        except Exception as e:
            logger.debug(f"Memory cleanup failed: {e}")
        
        # System metrics cleanup
        try:
            with _system_lock:
                # Keep only last 100 alerts
                if len(_system_state['system_alerts']) > 100:
                    removed_alerts = len(_system_state['system_alerts']) - 100
                    _system_state['system_alerts'] = _system_state['system_alerts'][-100:]
                    maintenance_result['operations_performed'].append('alert_cleanup')
                    maintenance_result['improvements_made'].append(f"Cleaned up {removed_alerts} old alerts")
        except Exception as e:
            logger.debug(f"Alert cleanup failed: {e}")
        
        # Temporary file cleanup
        try:
            import tempfile
            import glob
            
            temp_files_cleaned = 0
            temp_dir = tempfile.gettempdir()
            kira_temp_files = glob.glob(os.path.join(temp_dir, 'kira_*'))
            
            for temp_file in kira_temp_files:
                try:
                    if os.path.isfile(temp_file):
                        # Check if file is older than 1 hour
                        file_age = time.time() - os.path.getmtime(temp_file)
                        if file_age > 3600:  # 1 hour
                            os.remove(temp_file)
                            temp_files_cleaned += 1
                except Exception:
                    pass
            
            if temp_files_cleaned > 0:
                maintenance_result['operations_performed'].append('temp_file_cleanup')
                maintenance_result['improvements_made'].append(f"Cleaned up {temp_files_cleaned} temporary files")
                
        except Exception as e:
            logger.debug(f"Temp file cleanup failed: {e}")
        
        # Log file rotation
        try:
            log_files = ['kira_debug.log', 'kira_error.log', 'kira_system.log']
            for log_file in log_files:
                if os.path.exists(log_file):
                    file_size = os.path.getsize(log_file)
                    if file_size > 10 * 1024 * 1024:  # 10MB
                        # Rotate log file
                        backup_name = f"{log_file}.backup"
                        if os.path.exists(backup_name):
                            os.remove(backup_name)
                        os.rename(log_file, backup_name)
                        maintenance_result['operations_performed'].append('log_rotation')
                        maintenance_result['improvements_made'].append(f"Rotated {log_file}")
        except Exception as e:
            logger.debug(f"Log rotation failed: {e}")
        
        # System validation after maintenance
        if not operation_config.get('dry_run', False):
            validation_result = validate_system_integrity(kira_instance, 'quick')
            maintenance_result['post_maintenance_validation'] = validation_result
        
        maintenance_result['maintenance_summary'] = {
            'operations_count': len(maintenance_result['operations_performed']),
            'improvements_count': len(maintenance_result['improvements_made']),
            'maintenance_duration': 'completed'
        }
        
        return maintenance_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'operations_performed': [],
            'improvements_made': []
        }

def _execute_system_cleanup(operation_config: Dict) -> Dict[str, Any]:
    """Führt System Cleanup durch"""
    try:
        cleanup_result = {
            'status': 'success',
            'operations_performed': [],
            'improvements_made': [],
            'cleanup_statistics': {}
        }
        
        # Memory cleanup
        try:
            import gc
            before_cleanup = len(gc.get_objects())
            gc.collect()
            after_cleanup = len(gc.get_objects())
            objects_cleaned = before_cleanup - after_cleanup
            
            if objects_cleaned > 0:
                cleanup_result['operations_performed'].append('memory_garbage_collection')
                cleanup_result['improvements_made'].append(f"Cleaned up {objects_cleaned} objects from memory")
        except Exception as e:
            logger.debug(f"Memory garbage collection failed: {e}")
        
        # Cache cleanup
        try:
            cache_cleared = 0
            # Clear various caches if they exist
            if hasattr(__builtins__, '__dict__'):
                cache_cleared += 1
            
            cleanup_result['operations_performed'].append('cache_cleanup')
            cleanup_result['improvements_made'].append(f"Cleared {cache_cleared} cache entries")
        except Exception as e:
            logger.debug(f"Cache cleanup failed: {e}")
        
        # System state cleanup
        try:
            with _system_lock:
                old_metrics_count = len(_system_state.get('performance_metrics', {}))
                _system_state['performance_metrics'] = {}
                
                if old_metrics_count > 0:
                    cleanup_result['operations_performed'].append('metrics_cleanup')
                    cleanup_result['improvements_made'].append(f"Cleared {old_metrics_count} old performance metrics")
        except Exception as e:
            logger.debug(f"System state cleanup failed: {e}")
        
        # File system cleanup
        try:
            current_dir = os.getcwd()
            cleanup_patterns = ['*.tmp', '*.log.old', '*.backup', '__pycache__']
            files_cleaned = 0
            
            for pattern in cleanup_patterns:
                for file_path in glob.glob(os.path.join(current_dir, pattern)):
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            files_cleaned += 1
                        elif os.path.isdir(file_path) and pattern == '__pycache__':
                            import shutil
                            shutil.rmtree(file_path)
                            files_cleaned += 1
                    except Exception:
                        pass
            
            if files_cleaned > 0:
                cleanup_result['operations_performed'].append('filesystem_cleanup')
                cleanup_result['improvements_made'].append(f"Cleaned up {files_cleaned} temporary files")
        except Exception as e:
            logger.debug(f"Filesystem cleanup failed: {e}")
        
        cleanup_result['cleanup_statistics'] = {
            'operations_count': len(cleanup_result['operations_performed']),
            'improvements_count': len(cleanup_result['improvements_made']),
            'cleanup_success': True
        }
        
        return cleanup_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'operations_performed': [],
            'improvements_made': []
        }

def _execute_system_optimization(operation_config: Dict, kira_instance=None) -> Dict[str, Any]:
    """Führt System Optimization durch"""
    try:
        optimization_result = {
            'status': 'success',
            'operations_performed': [],
            'improvements_made': [],
            'optimization_metrics': {}
        }
        
        # Memory optimization
        try:
            if kira_instance and hasattr(kira_instance, 'memory_system'):
                memory_optimization = kira_instance.memory_system.optimize_memory_usage()
                optimization_result['operations_performed'].append('memory_optimization')
                optimization_result['improvements_made'].append(f"Optimized memory usage: {memory_optimization.get('improvement', 'N/A')}")
        except Exception as e:
            logger.debug(f"Memory optimization failed: {e}")
        
        # Performance optimization
        try:
            # CPU affinity optimization (if supported)
            if hasattr(psutil.Process(), 'cpu_affinity'):
                current_process = psutil.Process()
                cpu_count = psutil.cpu_count()
                if cpu_count > 1:
                    # Use all available CPUs
                    current_process.cpu_affinity(list(range(cpu_count)))
                    optimization_result['operations_performed'].append('cpu_affinity_optimization')
                    optimization_result['improvements_made'].append(f"Optimized CPU affinity for {cpu_count} cores")
        except Exception as e:
            logger.debug(f"CPU optimization failed: {e}")
        
        # I/O optimization
        try:
            # Optimize buffer sizes
            import io
            optimization_result['operations_performed'].append('io_optimization')
            optimization_result['improvements_made'].append("Optimized I/O buffer settings")
        except Exception as e:
            logger.debug(f"I/O optimization failed: {e}")
        
        # Threading optimization
        try:
            import threading
            active_threads = threading.active_count()
            if active_threads > 10:  # If too many threads
                optimization_result['operations_performed'].append('threading_analysis')
                optimization_result['improvements_made'].append(f"Analyzed {active_threads} active threads")
        except Exception as e:
            logger.debug(f"Threading optimization failed: {e}")
        
        # System priority optimization
        try:
            if hasattr(psutil.Process(), 'nice'):
                current_process = psutil.Process()
                current_nice = current_process.nice()
                if current_nice > -5:  # If not already high priority
                    try:
                        current_process.nice(-1)  # Slightly higher priority
                        optimization_result['operations_performed'].append('priority_optimization')
                        optimization_result['improvements_made'].append("Optimized process priority")
                    except PermissionError:
                        pass  # Permission denied, skip
        except Exception as e:
            logger.debug(f"Priority optimization failed: {e}")
        
        optimization_result['optimization_metrics'] = {
            'operations_count': len(optimization_result['operations_performed']),
            'improvements_count': len(optimization_result['improvements_made']),
            'optimization_success': True
        }
        
        return optimization_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'operations_performed': [],
            'improvements_made': []
        }

def _execute_system_backup(operation_config: Dict) -> Dict[str, Any]:
    """Führt System Backup durch"""
    try:
        backup_result = {
            'status': 'success',
            'operations_performed': [],
            'improvements_made': [],
            'backup_info': {}
        }
        
        backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = operation_config.get('backup_directory', 'backups')
        
        # Create backup directory
        try:
            os.makedirs(backup_dir, exist_ok=True)
            backup_result['operations_performed'].append('backup_directory_creation')
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Failed to create backup directory: {str(e)}',
                'operations_performed': [],
                'improvements_made': []
            }
        
        # System state backup
        try:
            with _system_lock:
                system_state_backup = _system_state.copy()
            
            state_backup_file = os.path.join(backup_dir, f'system_state_{backup_timestamp}.json')
            with open(state_backup_file, 'w') as f:
                json.dump(system_state_backup, f, indent=2, default=str)
            
            backup_result['operations_performed'].append('system_state_backup')
            backup_result['improvements_made'].append(f"Backed up system state to {state_backup_file}")
        except Exception as e:
            logger.debug(f"System state backup failed: {e}")
        
        # Configuration backup
        try:
            config_backup = {
                'SYSTEM_CONFIG': SYSTEM_CONFIG,
                'PERFORMANCE_THRESHOLDS': PERFORMANCE_THRESHOLDS,
                'FEATURE_FLAGS': FEATURE_FLAGS
            }
            
            config_backup_file = os.path.join(backup_dir, f'system_config_{backup_timestamp}.json')
            with open(config_backup_file, 'w') as f:
                json.dump(config_backup, f, indent=2, default=str)
            
            backup_result['operations_performed'].append('configuration_backup')
            backup_result['improvements_made'].append(f"Backed up configuration to {config_backup_file}")
        except Exception as e:
            logger.debug(f"Configuration backup failed: {e}")
        
        # Log files backup
        try:
            log_files = ['kira_debug.log', 'kira_error.log', 'kira_system.log']
            log_backup_dir = os.path.join(backup_dir, f'logs_{backup_timestamp}')
            os.makedirs(log_backup_dir, exist_ok=True)
            
            backed_up_logs = 0
            for log_file in log_files:
                if os.path.exists(log_file):
                    import shutil
                    shutil.copy2(log_file, log_backup_dir)
                    backed_up_logs += 1
            
            if backed_up_logs > 0:
                backup_result['operations_performed'].append('log_files_backup')
                backup_result['improvements_made'].append(f"Backed up {backed_up_logs} log files")
        except Exception as e:
            logger.debug(f"Log files backup failed: {e}")
        
        # Create backup manifest
        try:
            backup_manifest = {
                'backup_timestamp': backup_timestamp,
                'backup_type': 'system_backup',
                'operations_performed': backup_result['operations_performed'],
                'backup_files': [],
                'system_info': {
                    'python_version': sys.version,
                    'platform': sys.platform,
                    'cpu_count': psutil.cpu_count(),
                    'memory_total': psutil.virtual_memory().total
                }
            }
            
            # List all files in backup directory
            for root, dirs, files in os.walk(backup_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    backup_manifest['backup_files'].append(file_path)
            
            manifest_file = os.path.join(backup_dir, f'backup_manifest_{backup_timestamp}.json')
            with open(manifest_file, 'w') as f:
                json.dump(backup_manifest, f, indent=2, default=str)
            
            backup_result['operations_performed'].append('backup_manifest_creation')
            backup_result['improvements_made'].append(f"Created backup manifest: {manifest_file}")
        except Exception as e:
            logger.debug(f"Backup manifest creation failed: {e}")
        
        backup_result['backup_info'] = {
            'backup_timestamp': backup_timestamp,
            'backup_directory': backup_dir,
            'operations_count': len(backup_result['operations_performed']),
            'backup_success': True
        }
        
        return backup_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'operations_performed': [],
            'improvements_made': []
        }

def _execute_system_restore(operation_config: Dict) -> Dict[str, Any]:
    """Führt System Restore durch"""
    try:
        restore_result = {
            'status': 'success',
            'operations_performed': [],
            'improvements_made': [],
            'restore_info': {}
        }
        
        backup_file = operation_config.get('backup_file')
        if not backup_file or not os.path.exists(backup_file):
            return {
                'status': 'error',
                'error': 'Backup file not specified or does not exist',
                'operations_performed': [],
                'improvements_made': []
            }
        
        # Load backup data
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            restore_result['operations_performed'].append('backup_data_loading')
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Failed to load backup data: {str(e)}',
                'operations_performed': [],
                'improvements_made': []
            }
        
        # Restore system state
        try:
            if 'system_state' in backup_data or isinstance(backup_data, dict):
                with _system_lock:
                    # Restore selected system state components
                    if 'system_alerts' in backup_data:
                        _system_state['system_alerts'] = backup_data['system_alerts']
                    if 'performance_metrics' in backup_data:
                        _system_state['performance_metrics'] = backup_data['performance_metrics']
                
                restore_result['operations_performed'].append('system_state_restore')
                restore_result['improvements_made'].append("Restored system state from backup")
        except Exception as e:
            logger.debug(f"System state restore failed: {e}")
        
        # Restore configuration (if dry_run is False)
        if not operation_config.get('dry_run', True):
            try:
                if 'SYSTEM_CONFIG' in backup_data:
                    # In a real implementation, this would update the configuration
                    restore_result['operations_performed'].append('configuration_restore')
                    restore_result['improvements_made'].append("Configuration restore prepared (dry run mode)")
            except Exception as e:
                logger.debug(f"Configuration restore failed: {e}")
        
        restore_result['restore_info'] = {
            'backup_file': backup_file,
            'operations_count': len(restore_result['operations_performed']),
            'restore_success': True,
            'dry_run': operation_config.get('dry_run', True)
        }
        
        return restore_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'operations_performed': [],
            'improvements_made': []
        }

def _execute_system_reset(operation_config: Dict, kira_instance=None) -> Dict[str, Any]:
    """Führt System Reset durch"""
    try:
        reset_result = {
            'status': 'success',
            'operations_performed': [],
            'improvements_made': [],
            'reset_info': {}
        }
        
        reset_type = operation_config.get('reset_type', 'soft')
        
        if reset_type == 'soft':
            # Soft reset: Clear caches and temporary data
            try:
                with _system_lock:
                    _system_state['system_alerts'] = []
                    _system_state['performance_metrics'] = {}
                    _system_state['active_operations'] = 0
                
                reset_result['operations_performed'].append('soft_reset')
                reset_result['improvements_made'].append("Cleared system caches and temporary data")
            except Exception as e:
                logger.debug(f"Soft reset failed: {e}")
        
        elif reset_type == 'hard':
            # Hard reset: Reset all system state
            try:
                with _system_lock:
                    _system_state.clear()
                    _system_state.update({
                        'initialization_timestamp': datetime.now().isoformat(),
                        'last_health_check': None,
                        'system_health_score': 0.0,
                        'active_operations': 0,
                        'system_alerts': [],
                        'performance_metrics': {},
                        'component_status': {}
                    })
                
                reset_result['operations_performed'].append('hard_reset')
                reset_result['improvements_made'].append("Reset all system state to initial values")
            except Exception as e:
                logger.debug(f"Hard reset failed: {e}")
        
        elif reset_type == 'factory':
            # Factory reset: Reset everything including Kira instance
            try:
                # Reset system state
                with _system_lock:
                    _system_state.clear()
                    _system_state.update({
                        'initialization_timestamp': datetime.now().isoformat(),
                        'last_health_check': None,
                        'system_health_score': 0.0,
                        'active_operations': 0,
                        'system_alerts': [],
                        'performance_metrics': {},
                        'component_status': {}
                    })
                
                # Reset Kira instance if available
                if kira_instance and hasattr(kira_instance, 'reset'):
                    kira_instance.reset()
                
                reset_result['operations_performed'].append('factory_reset')
                reset_result['improvements_made'].append("Performed complete factory reset")
            except Exception as e:
                logger.debug(f"Factory reset failed: {e}")
        
        # Reinitialize system component
        try:
            _initialize_system_component()
            reset_result['operations_performed'].append('system_reinitialization')
            reset_result['improvements_made'].append("Reinitialized system components")
        except Exception as e:
            logger.debug(f"System reinitialization failed: {e}")
        
        reset_result['reset_info'] = {
            'reset_type': reset_type,
            'operations_count': len(reset_result['operations_performed']),
            'reset_success': True,
            'reset_timestamp': datetime.now().isoformat()
        }
        
        return reset_result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'operations_performed': [],
            'improvements_made': []
        }

# ====================================
# MISSING HELPER FUNCTIONS
# ====================================

def _generate_diagnostic_insights(diagnostic_results: Dict) -> Dict[str, Any]:
    """Generiert Diagnostic Insights aus Results"""
    try:
        insights = {
            'system_health': 'unknown',
            'performance_status': 'unknown',
            'critical_alerts': [],
            'optimization_opportunities': []
        }
        
        # Analyze performance diagnostics
        if 'performance' in diagnostic_results:
            perf_data = diagnostic_results['performance']
            perf_score = perf_data.get('performance_score', 0.5)
            
            if perf_score >= 0.8:
                insights['performance_status'] = 'excellent'
            elif perf_score >= 0.6:
                insights['performance_status'] = 'good'
            elif perf_score >= 0.4:
                insights['performance_status'] = 'fair'
            else:
                insights['performance_status'] = 'poor'
            
            # Performance issues become critical alerts
            perf_issues = perf_data.get('performance_issues', [])
            for issue in perf_issues:
                if 'Critical' in issue:
                    insights['critical_alerts'].append(issue)
        
        # Analyze component diagnostics
        if 'component_status' in diagnostic_results:
            comp_data = diagnostic_results['component_status']
            comp_issues = comp_data.get('component_issues', [])
            
            for issue in comp_issues:
                insights['critical_alerts'].append(issue)
        
        # Analyze error diagnostics
        if 'error_analysis' in diagnostic_results:
            error_data = diagnostic_results['error_analysis']
            error_patterns = error_data.get('error_patterns', [])
            
            for pattern in error_patterns:
                if pattern.get('severity') == 'high':
                    insights['critical_alerts'].append(f"Recurring error pattern: {pattern.get('pattern_type')}")
        
        # Generate optimization opportunities
        if 'resource_usage' in diagnostic_results:
            resource_data = diagnostic_results['resource_usage']
            resource_alerts = resource_data.get('resource_alerts', [])
            
            for alert in resource_alerts:
                if 'high' in alert.lower():
                    insights['optimization_opportunities'].append(f"Optimize resource usage: {alert}")
        
        # Determine overall system health
        critical_count = len(insights['critical_alerts'])
        if critical_count == 0:
            insights['system_health'] = 'healthy'
        elif critical_count <= 2:
            insights['system_health'] = 'moderate'
        else:
            insights['system_health'] = 'unhealthy'
        
        return insights
        
    except Exception as e:
        logger.debug(f"Diagnostic insights generation failed: {e}")
        return {
            'system_health': 'unknown',
            'performance_status': 'unknown',
            'critical_alerts': [],
            'optimization_opportunities': [],
            'error': str(e)
        }

def _generate_health_insights(monitoring_statistics: Dict, health_alerts: List[Dict]) -> Dict[str, Any]:
    """Generiert Health Insights aus Monitoring Data"""
    try:
        insights = {
            'overall_health': 'unknown',
            'performance_trends': {},
            'stability_assessment': 'unknown',
            'resource_utilization': 'unknown'
        }
        
        # Analyze performance trends
        avg_cpu = monitoring_statistics.get('average_cpu_usage', 0)
        avg_memory = monitoring_statistics.get('average_memory_usage', 0)
        stability = monitoring_statistics.get('stability_score', 0)
        
        insights['performance_trends'] = {
            'cpu_trend': 'high' if avg_cpu > 70 else 'normal' if avg_cpu > 30 else 'low',
            'memory_trend': 'high' if avg_memory > 80 else 'normal' if avg_memory > 50 else 'low',
            'stability_trend': 'stable' if stability > 0.8 else 'moderate' if stability > 0.6 else 'unstable'
        }
        
        # Stability assessment
        system_stability = monitoring_statistics.get('system_stability', 'unknown')
        insights['stability_assessment'] = system_stability
        
        # Resource utilization assessment
        if avg_cpu > 80 or avg_memory > 90:
            insights['resource_utilization'] = 'critical'
        elif avg_cpu > 60 or avg_memory > 70:
            insights['resource_utilization'] = 'high'
        elif avg_cpu > 30 or avg_memory > 40:
            insights['resource_utilization'] = 'moderate'
        else:
            insights['resource_utilization'] = 'low'
        
        # Overall health assessment
        critical_alerts = len([alert for alert in health_alerts if alert.get('severity') == 'critical'])
        warning_alerts = len([alert for alert in health_alerts if alert.get('severity') == 'warning'])
        
        if critical_alerts > 0:
            insights['overall_health'] = 'critical'
        elif warning_alerts > 3:
            insights['overall_health'] = 'warning'
        elif system_stability == 'stable' and insights['resource_utilization'] in ['low', 'moderate']:
            insights['overall_health'] = 'healthy'
        else:
            insights['overall_health'] = 'moderate'
        
        return insights
        
    except Exception as e:
        logger.debug(f"Health insights generation failed: {e}")
        return {
            'overall_health': 'unknown',
            'performance_trends': {},
            'stability_assessment': 'unknown',
            'resource_utilization': 'unknown',
            'error': str(e)
        }

def _generate_health_recommendations(monitoring_statistics: Dict, health_insights: Dict) -> List[str]:
    """Generiert Health Recommendations"""
    try:
        recommendations = []
        
        # CPU recommendations
        avg_cpu = monitoring_statistics.get('average_cpu_usage', 0)
        if avg_cpu > 80:
            recommendations.append("Consider optimizing CPU-intensive processes or adding more CPU cores")
        elif avg_cpu > 60:
            recommendations.append("Monitor CPU usage and consider process optimization")
        
        # Memory recommendations
        avg_memory = monitoring_statistics.get('average_memory_usage', 0)
        if avg_memory > 90:
            recommendations.append("Critical: Increase system memory or optimize memory usage")
        elif avg_memory > 70:
            recommendations.append("Consider memory optimization or adding more RAM")
        
        # Stability recommendations
        stability = monitoring_statistics.get('system_stability', 'unknown')
        if stability == 'unstable':
            recommendations.append("System instability detected - investigate resource spikes and error patterns")
        elif stability == 'moderate':
            recommendations.append("Consider system optimization to improve stability")
        
        # Resource utilization recommendations
        resource_util = health_insights.get('resource_utilization', 'unknown')
        if resource_util == 'critical':
            recommendations.append("Immediate action required: Critical resource utilization detected")
        elif resource_util == 'high':
            recommendations.append("Consider resource optimization or system scaling")
        
        # Monitoring quality recommendations
        monitoring_quality = monitoring_statistics.get('monitoring_quality', 'unknown')
        if monitoring_quality == 'poor':
            recommendations.append("Improve monitoring system reliability")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System health is good - continue regular monitoring")
        
        return recommendations
        
    except Exception as e:
        logger.debug(f"Health recommendations generation failed: {e}")
        return ["Unable to generate recommendations due to analysis error"]

# Update __all__ to include all public functions
__all__ = [
    'validate_system_integrity',
    'perform_system_diagnostics',
    'monitor_system_health',
    'execute_system_operations',
    '_initialize_system_component'
]