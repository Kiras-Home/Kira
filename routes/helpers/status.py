import psutil
import platform
from datetime import datetime, timedelta
import os
from pathlib import Path
import time

def get_system_uptime():
    """Gibt echte System-Uptime zurück"""
    try:
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        uptime_delta = timedelta(seconds=uptime_seconds)
        
        return {
            'uptime_seconds': uptime_seconds,
            'uptime_formatted': str(uptime_delta).split('.')[0],  # Ohne Mikrosekunden
            'boot_time': datetime.fromtimestamp(boot_time).isoformat(),
            'current_time': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'uptime_seconds': 0,
            'uptime_formatted': '0:00:00',
            'boot_time': datetime.now().isoformat(),
            'current_time': datetime.now().isoformat(),
            'error': str(e)
        }

def get_system_status():
    """Gibt umfassenden System-Status zurück"""
    try:
        return {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'hostname': platform.node(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(logical=True),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'disk_total_gb': round(psutil.disk_usage('/').total / (1024**3), 2) if os.name != 'nt' else round(psutil.disk_usage('C:').total / (1024**3), 2)
        }
    except Exception as e:
        return {
            'platform': 'Unknown',
            'error': str(e)
        }