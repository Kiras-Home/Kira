import psutil
import time
from datetime import datetime

def collect_cpu_metrics():
    """Sammelt echte CPU-Metriken"""
    try:
        # CPU-Auslastung über 1 Sekunde messen für genaue Werte
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
            'cpu_freq_max': cpu_freq.max if cpu_freq else 0,
            'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
    except Exception as e:
        return {
            'cpu_percent': 0,
            'cpu_count_physical': 1,
            'cpu_count_logical': 1,
            'cpu_freq_current': 0,
            'cpu_freq_max': 0,
            'load_avg': [0, 0, 0],
            'error': str(e)
        }

def collect_memory_metrics():
    """Sammelt echte Speicher-Metriken"""
    try:
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()
        
        return {
            'memory_total': virtual_mem.total,
            'memory_available': virtual_mem.available,
            'memory_percent': virtual_mem.percent,
            'memory_used': virtual_mem.used,
            'memory_free': virtual_mem.free,
            'swap_total': swap_mem.total,
            'swap_used': swap_mem.used,
            'swap_percent': swap_mem.percent
        }
    except Exception as e:
        return {
            'memory_total': 0,
            'memory_available': 0,
            'memory_percent': 0,
            'memory_used': 0,
            'memory_free': 0,
            'swap_total': 0,
            'swap_used': 0,
            'swap_percent': 0,
            'error': str(e)
        }