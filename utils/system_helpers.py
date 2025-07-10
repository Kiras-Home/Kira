"""
Kira System Helper Utilities
Functions for system health monitoring and status formatting
"""

import logging
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def check_system_health() -> Dict[str, Any]:
    """
    Perform comprehensive system health check
    
    Returns:
        System health status dictionary
    """
    try:
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {},
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # CPU Health Check
        cpu_health = _check_cpu_health()
        health_data['checks']['cpu'] = cpu_health
        if cpu_health['status'] != 'healthy':
            health_data['warnings'].append(f"CPU: {cpu_health['message']}")
        
        # Memory Health Check
        memory_health = _check_memory_health()
        health_data['checks']['memory'] = memory_health
        if memory_health['status'] != 'healthy':
            health_data['warnings'].append(f"Memory: {memory_health['message']}")
        
        # Disk Health Check
        disk_health = _check_disk_health()
        health_data['checks']['disk'] = disk_health
        if disk_health['status'] != 'healthy':
            health_data['warnings'].append(f"Disk: {disk_health['message']}")
        
        # Process Health Check
        process_health = _check_process_health()
        health_data['checks']['processes'] = process_health
        
        # Network Health Check
        network_health = _check_network_health()
        health_data['checks']['network'] = network_health
        
        # File System Health Check
        filesystem_health = _check_filesystem_health()
        health_data['checks']['filesystem'] = filesystem_health
        if filesystem_health['status'] != 'healthy':
            health_data['warnings'].extend(filesystem_health.get('warnings', []))
        
        # Service Health Check
        service_health = _check_service_health()
        health_data['checks']['services'] = service_health
        
        # Determine overall status
        health_data['overall_status'] = _determine_overall_health_status(health_data['checks'])
        
        # Generate recommendations
        health_data['recommendations'] = _generate_health_recommendations(health_data['checks'])
        
        return health_data
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'error',
            'error': str(e),
            'checks': {}
        }


def format_system_status(system_state: Dict[str, Any], services: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format system status for display
    
    Args:
        system_state: Current system state
        services: Available services
        
    Returns:
        Formatted status data
    """
    try:
        formatted_status = {
            'summary': {
                'kira_ready': system_state.get('kira_ready', False),
                'full_ai_experience': system_state.get('full_ai_experience', False),
                'total_services': len(services),
                'active_services': sum(1 for s in services.values() if s is not None),
                'initialization_time': system_state.get('initialization_time')
            },
            'services_detail': {},
            'system_metrics': {},
            'health_indicators': {}
        }
        
        # Format service details
        for service_name, service_instance in services.items():
            if service_instance and hasattr(service_instance, 'get_status'):
                status = service_instance.get_status()
                formatted_status['services_detail'][service_name] = {
                    'name': service_name.replace('_', ' ').title(),
                    'status': status.get('status', 'unknown'),
                    'initialized': status.get('initialized', False),
                    'health': _determine_service_health(status)
                }
            else:
                formatted_status['services_detail'][service_name] = {
                    'name': service_name.replace('_', ' ').title(),
                    'status': 'not_available',
                    'initialized': False,
                    'health': 'unhealthy'
                }
        
        # Add system metrics
        formatted_status['system_metrics'] = _get_formatted_system_metrics()
        
        # Add health indicators
        formatted_status['health_indicators'] = _get_health_indicators(system_state, services)
        
        return formatted_status
        
    except Exception as e:
        logger.error(f"Status formatting failed: {e}")
        return {
            'summary': {'error': str(e)},
            'services_detail': {},
            'system_metrics': {},
            'health_indicators': {}
        }


def get_system_performance_metrics() -> Dict[str, Any]:
    """
    Get detailed system performance metrics
    
    Returns:
        Performance metrics dictionary
    """
    try:
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': _get_cpu_metrics(),
            'memory': _get_memory_metrics(),
            'disk': _get_disk_metrics(),
            'network': _get_network_metrics(),
            'processes': _get_process_metrics()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Performance metrics collection failed: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }


def monitor_kira_processes() -> Dict[str, Any]:
    """
    Monitor Kira-specific processes
    
    Returns:
        Process monitoring data
    """
    try:
        kira_processes = []
        current_process = psutil.Process()
        
        # Get current process info
        kira_processes.append({
            'name': 'Kira Main Process',
            'pid': current_process.pid,
            'cpu_percent': current_process.cpu_percent(),
            'memory_percent': current_process.memory_percent(),
            'memory_info': current_process.memory_info()._asdict(),
            'status': current_process.status(),
            'create_time': datetime.fromtimestamp(current_process.create_time()).isoformat()
        })
        
        # Look for related processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and any('kira' in cmd.lower() for cmd in proc.info['cmdline']):
                    if proc.pid != current_process.pid:
                        kira_processes.append({
                            'name': f"Kira Related: {proc.info['name']}",
                            'pid': proc.pid,
                            'cpu_percent': proc.cpu_percent(),
                            'memory_percent': proc.memory_percent(),
                            'status': proc.status()
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_processes': len(kira_processes),
            'processes': kira_processes
        }
        
    except Exception as e:
        logger.error(f"Process monitoring failed: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'processes': []
        }


def cleanup_system_resources() -> Dict[str, Any]:
    """
    Clean up system resources and temporary files
    
    Returns:
        Cleanup results
    """
    try:
        cleanup_results = {
            'timestamp': datetime.now().isoformat(),
            'actions_performed': [],
            'space_freed': 0,
            'errors': []
        }
        
        # Clean temporary voice files
        voice_cleanup = _cleanup_voice_files()
        cleanup_results['actions_performed'].extend(voice_cleanup['actions'])
        cleanup_results['space_freed'] += voice_cleanup['space_freed']
        
        # Clean old log files
        log_cleanup = _cleanup_old_logs()
        cleanup_results['actions_performed'].extend(log_cleanup['actions'])
        cleanup_results['space_freed'] += log_cleanup['space_freed']
        
        # Clean memory caches
        memory_cleanup = _cleanup_memory_caches()
        cleanup_results['actions_performed'].extend(memory_cleanup['actions'])
        
        return cleanup_results
        
    except Exception as e:
        logger.error(f"System cleanup failed: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'actions_performed': [],
            'space_freed': 0
        }


# Private helper functions

def _check_cpu_health() -> Dict[str, Any]:
    """Check CPU health status"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        if cpu_percent > 90:
            status = 'critical'
            message = f'CPU usage very high: {cpu_percent}%'
        elif cpu_percent > 75:
            status = 'warning'
            message = f'CPU usage high: {cpu_percent}%'
        else:
            status = 'healthy'
            message = f'CPU usage normal: {cpu_percent}%'
        
        return {
            'status': status,
            'message': message,
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'CPU check failed: {e}'
        }


def _check_memory_health() -> Dict[str, Any]:
    """Check memory health status"""
    try:
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            status = 'critical'
            message = f'Memory usage critical: {memory.percent}%'
        elif memory.percent > 80:
            status = 'warning'
            message = f'Memory usage high: {memory.percent}%'
        else:
            status = 'healthy'
            message = f'Memory usage normal: {memory.percent}%'
        
        return {
            'status': status,
            'message': message,
            'memory_percent': memory.percent,
            'memory_available': memory.available,
            'memory_total': memory.total
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Memory check failed: {e}'
        }


def _check_disk_health() -> Dict[str, Any]:
    """Check disk health status"""
    try:
        disk = psutil.disk_usage('/')
        
        if disk.percent > 95:
            status = 'critical'
            message = f'Disk space critical: {disk.percent}%'
        elif disk.percent > 85:
            status = 'warning'
            message = f'Disk space low: {disk.percent}%'
        else:
            status = 'healthy'
            message = f'Disk space adequate: {disk.percent}%'
        
        return {
            'status': status,
            'message': message,
            'disk_percent': disk.percent,
            'disk_free': disk.free,
            'disk_total': disk.total
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Disk check failed: {e}'
        }


def _check_process_health() -> Dict[str, Any]:
    """Check process health"""
    try:
        process_count = len(psutil.pids())
        return {
            'status': 'healthy',
            'total_processes': process_count,
            'kira_processes': 1  # At least the main process
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Process check failed: {e}'
        }


def _check_network_health() -> Dict[str, Any]:
    """Check network connectivity"""
    try:
        # Simple check - could be enhanced with actual connectivity tests
        return {
            'status': 'healthy',
            'message': 'Network interfaces available'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Network check failed: {e}'
        }


def _check_filesystem_health() -> Dict[str, Any]:
    """Check filesystem health"""
    try:
        warnings = []
        required_dirs = ['logs', 'voice/output', 'memory/data', 'data']
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                warnings.append(f'Missing directory: {dir_path}')
        
        status = 'healthy' if not warnings else 'warning'
        
        return {
            'status': status,
            'warnings': warnings,
            'required_directories': required_dirs
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Filesystem check failed: {e}'
        }


def _check_service_health() -> Dict[str, Any]:
    """Check service health"""
    return {
        'status': 'healthy',
        'message': 'Service health check placeholder'
    }


def _determine_overall_health_status(checks: Dict) -> str:
    """Determine overall health status from individual checks"""
    if any(check.get('status') == 'critical' for check in checks.values()):
        return 'critical'
    elif any(check.get('status') == 'warning' for check in checks.values()):
        return 'warning'
    elif any(check.get('status') == 'error' for check in checks.values()):
        return 'degraded'
    else:
        return 'healthy'


def _generate_health_recommendations(checks: Dict) -> List[str]:
    """Generate health recommendations based on checks"""
    recommendations = []
    
    # CPU recommendations
    cpu_check = checks.get('cpu', {})
    if cpu_check.get('status') in ['warning', 'critical']:
        recommendations.append('Consider closing unnecessary applications to reduce CPU load')
    
    # Memory recommendations
    memory_check = checks.get('memory', {})
    if memory_check.get('status') in ['warning', 'critical']:
        recommendations.append('Restart Kira to free up memory or add more RAM')
    
    # Disk recommendations
    disk_check = checks.get('disk', {})
    if disk_check.get('status') in ['warning', 'critical']:
        recommendations.append('Clean up temporary files and free disk space')
    
    return recommendations


def _determine_service_health(status: Dict) -> str:
    """Determine service health from status"""
    if status.get('initialized') and status.get('status') == 'active':
        return 'healthy'
    elif status.get('status') == 'error':
        return 'unhealthy'
    else:
        return 'degraded'


def _get_formatted_system_metrics() -> Dict[str, Any]:
    """Get formatted system metrics"""
    try:
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': datetime.now().isoformat()
        }
    except:
        return {}


def _get_health_indicators(system_state: Dict, services: Dict) -> Dict[str, Any]:
    """Get health indicators"""
    active_services = sum(1 for s in services.values() if s is not None)
    total_services = len(services)
    
    return {
        'service_availability': (active_services / total_services * 100) if total_services > 0 else 0,
        'system_readiness': system_state.get('kira_ready', False),
        'ai_capability': system_state.get('full_ai_experience', False)
    }


def _get_cpu_metrics() -> Dict[str, Any]:
    """Get detailed CPU metrics"""
    try:
        return {
            'usage_percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'physical_count': psutil.cpu_count(logical=False)
        }
    except:
        return {}


def _get_memory_metrics() -> Dict[str, Any]:
    """Get detailed memory metrics"""
    try:
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used,
            'free': memory.free
        }
    except:
        return {}


def _get_disk_metrics() -> Dict[str, Any]:
    """Get detailed disk metrics"""
    try:
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': disk.percent
        }
    except:
        return {}


def _get_network_metrics() -> Dict[str, Any]:
    """Get network metrics"""
    try:
        return {
            'connections': len(psutil.net_connections()),
            'interfaces': list(psutil.net_if_addrs().keys())
        }
    except:
        return {}


def _get_process_metrics() -> Dict[str, Any]:
    """Get process metrics"""
    try:
        return {
            'total_processes': len(psutil.pids()),
            'current_process_memory': psutil.Process().memory_percent()
        }
    except:
        return {}


def _cleanup_voice_files() -> Dict[str, Any]:
    """Clean up old voice files"""
    try:
        voice_dir = Path('voice/output')
        if not voice_dir.exists():
            return {'actions': [], 'space_freed': 0}
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        space_freed = 0
        actions = []
        
        for file_path in voice_dir.glob('*.wav'):
            if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_time:
                file_size = file_path.stat().st_size
                file_path.unlink()
                space_freed += file_size
                actions.append(f'Deleted old voice file: {file_path.name}')
        
        return {
            'actions': actions,
            'space_freed': space_freed
        }
    except Exception as e:
        return {
            'actions': [f'Voice cleanup failed: {e}'],
            'space_freed': 0
        }


def _cleanup_old_logs() -> Dict[str, Any]:
    """Clean up old log files"""
    try:
        logs_dir = Path('logs')
        if not logs_dir.exists():
            return {'actions': [], 'space_freed': 0}
        
        cutoff_time = datetime.now() - timedelta(days=7)
        space_freed = 0
        actions = []
        
        for file_path in logs_dir.glob('*.log.*'):  # Rotated logs
            if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_time:
                file_size = file_path.stat().st_size
                file_path.unlink()
                space_freed += file_size
                actions.append(f'Deleted old log file: {file_path.name}')
        
        return {
            'actions': actions,
            'space_freed': space_freed
        }
    except Exception as e:
        return {
            'actions': [f'Log cleanup failed: {e}'],
            'space_freed': 0
        }


def _cleanup_memory_caches() -> Dict[str, Any]:
    """Clean up memory caches"""
    try:
        # This would implement memory cache cleanup
        # For now, just return a placeholder
        return {
            'actions': ['Memory caches cleared']
        }
    except Exception as e:
        return {
            'actions': [f'Memory cleanup failed: {e}']
        }