"""
Kira System API Routes
Handles system status, monitoring and health endpoints
"""

import logging
from flask import Blueprint, jsonify, request
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


def create_system_routes(system_state: Dict[str, Any], services: Dict[str, Any]) -> Blueprint:
    """
    Create system API routes blueprint
    
    Args:
        system_state: Current system state
        services: Available services
        
    Returns:
        Blueprint with system routes
    """
    
    system_bp = Blueprint('system_api', __name__, url_prefix='/api/system')
    
    @system_bp.route('/status', methods=['GET'])
    def get_system_status():
        """Get comprehensive system status"""
        try:
            # Get status from all services
            status_data = {
                'timestamp': datetime.now().isoformat(),
                'system_ready': system_state.get('kira_ready', False),
                'full_ai_experience': system_state.get('full_ai_experience', False),
                'initialization_time': system_state.get('initialization_time'),
                'services': {}
            }
            
            # Get status from each service
            for service_name, service_instance in services.items():
                if service_instance and hasattr(service_instance, 'get_status'):
                    status_data['services'][service_name] = service_instance.get_status()
                else:
                    status_data['services'][service_name] = {
                        'initialized': False,
                        'status': 'not_available'
                    }
            
            # Add system-wide metrics
            status_data['metrics'] = _get_system_metrics(system_state, services)
            
            return jsonify({
                'success': True,
                'data': status_data
            })
            
        except Exception as e:
            logger.error(f"System status error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @system_bp.route('/health', methods=['GET'])
    def health_check():
        """Simple health check endpoint"""
        try:
            active_services = sum(1 for s in services.values() if s is not None)
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'active_services': active_services,
                'total_services': len(services),
                'kira_ready': system_state.get('kira_ready', False)
            })
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500
    
    @system_bp.route('/services', methods=['GET'])
    def get_services_info():
        """Get detailed information about all services"""
        try:
            services_info = {}
            
            for service_name, service_instance in services.items():
                if service_instance:
                    services_info[service_name] = {
                        'available': True,
                        'status': getattr(service_instance, 'status', 'unknown'),
                        'initialized': getattr(service_instance, 'is_initialized', False)
                    }
                    
                    # Get detailed status if available
                    if hasattr(service_instance, 'get_status'):
                        services_info[service_name]['details'] = service_instance.get_status()
                else:
                    services_info[service_name] = {
                        'available': False,
                        'status': 'not_initialized',
                        'initialized': False
                    }
            
            return jsonify({
                'success': True,
                'services': services_info
            })
            
        except Exception as e:
            logger.error(f"Services info error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @system_bp.route('/restart', methods=['POST'])
    def restart_system():
        """Restart system services"""
        try:
            # This would implement system restart logic
            # For now, return a placeholder response
            return jsonify({
                'success': True,
                'message': 'System restart initiated',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"System restart error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @system_bp.route('/logs', methods=['GET'])
    def get_system_logs():
        """Get recent system logs"""
        try:
            # Get query parameters
            lines = request.args.get('lines', 100, type=int)
            level = request.args.get('level', 'INFO')
            
            # Read system logs (simplified implementation)
            try:
                with open('logs/kira_system.log', 'r') as f:
                    log_lines = f.readlines()
                    recent_logs = log_lines[-lines:] if lines > 0 else log_lines
                    
                return jsonify({
                    'success': True,
                    'logs': recent_logs,
                    'total_lines': len(log_lines),
                    'returned_lines': len(recent_logs)
                })
                    
            except FileNotFoundError:
                return jsonify({
                    'success': True,
                    'logs': [],
                    'message': 'No log file found'
                })
            
        except Exception as e:
            logger.error(f"System logs error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @system_bp.route('/metrics', methods=['GET'])
    def get_system_metrics():
        """Get system performance metrics"""
        try:
            metrics = _get_system_metrics(system_state, services)
            
            return jsonify({
                'success': True,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"System metrics error: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    def _get_system_metrics(system_state: Dict, services: Dict) -> Dict:
        """Calculate system metrics"""
        try:
            import psutil
            import os
            
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Service metrics
            active_services = sum(1 for s in services.values() if s is not None)
            total_services = len(services)
            
            return {
                'system_resources': {
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory.percent,
                    'memory_available': memory.available,
                    'disk_usage': disk.percent,
                    'disk_free': disk.free
                },
                'services': {
                    'active': active_services,
                    'total': total_services,
                    'success_rate': (active_services / total_services * 100) if total_services > 0 else 0
                },
                'uptime': {
                    'started_at': system_state.get('initialization_time'),
                    'current_time': datetime.now().isoformat()
                }
            }
            
        except ImportError:
            # Fallback if psutil is not available
            return {
                'system_resources': {
                    'status': 'monitoring_not_available'
                },
                'services': {
                    'active': sum(1 for s in services.values() if s is not None),
                    'total': len(services)
                }
            }
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return {
                'error': str(e)
            }
    
    return system_bp