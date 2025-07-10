from flask import Blueprint, jsonify
import psutil
import platform
import time
from datetime import datetime

system_bp = Blueprint('system', __name__, url_prefix='/system')

@system_bp.route('/uptime', methods=['GET'])
def get_uptime():
    """Gibt die System-Uptime zurück."""
    try:
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        
        return jsonify({
            'success': True, 
            'uptime_seconds': uptime_seconds,
            'uptime_formatted': str(datetime.fromtimestamp(uptime_seconds)).split('.')[0],
            'boot_time': datetime.fromtimestamp(boot_time).isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@system_bp.route('/status', methods=['GET'])
def get_status():
    """Gibt den umfassenden Systemstatus zurück."""
    try:
        # CPU Usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory Usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk Usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # Network (simplified check)
        network_status = 'Connected'  # Could be enhanced with actual network tests
        
        # System Info
        system_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'hostname': platform.node(),
            'cpu_count': psutil.cpu_count(logical=True),
            'memory_total_gb': round(memory.total / (1024**3), 2),
            'disk_total_gb': round(disk.total / (1024**3), 2)
        }
        
        status_data = {
            'cpu_usage': round(cpu_usage, 1),
            'memory_usage': round(memory_usage, 1),
            'disk_usage': round(disk_usage, 1),
            'network_status': network_status,
            'system_info': system_info,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True, 
            'status': status_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500