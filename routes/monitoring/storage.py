from flask import Blueprint, jsonify
import psutil
import os
from datetime import datetime
from pathlib import Path

storage_bp = Blueprint('storage', __name__, url_prefix='/storage')

@storage_bp.route('/status', methods=['GET'])
def get_storage_status():
    """Gibt den aktuellen Speicherstatus zurück."""
    try:
        # Windows-kompatible Pfade
        if os.name == 'nt':  # Windows
            disk_usage = psutil.disk_usage('C:')
        else:  # Linux/Mac
            disk_usage = psutil.disk_usage('/')
            
        storage_status = {
            'total': disk_usage.total,
            'used': disk_usage.used,
            'free': disk_usage.free,
            'percent': disk_usage.percent,
            'total_gb': round(disk_usage.total / (1024**3), 2),
            'used_gb': round(disk_usage.used / (1024**3), 2),
            'free_gb': round(disk_usage.free / (1024**3), 2)
        }
        return jsonify({'success': True, 'storage_status': storage_status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@storage_bp.route('/files', methods=['GET'])
def get_file_list():
    """Gibt eine Liste von Dateien im aktuellen Verzeichnis zurück."""
    try:
        current_dir = Path('.')
        files = []
        
        for f in current_dir.iterdir():
            if f.is_file():
                try:
                    stat = f.stat()
                    files.append({
                        'name': f.name,
                        'size': stat.st_size,
                        'size_mb': round(stat.st_size / (1024**2), 2),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': f.suffix or 'file'
                    })
                except Exception:
                    continue  # Datei überspringen wenn Fehler
                    
        return jsonify({'success': True, 'files': files, 'total_files': len(files)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@storage_bp.route('/usage', methods=['GET'])
def get_disk_usage():
    """Detaillierte Festplatten-Nutzung"""
    try:
        disks = []
        
        # Alle verfügbaren Laufwerke
        if os.name == 'nt':  # Windows
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disks.append({
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': usage.percent
                    })
                except PermissionError:
                    continue
        else:  # Linux/Mac
            usage = psutil.disk_usage('/')
            disks.append({
                'device': '/',
                'mountpoint': '/',
                'total': usage.total,
                'used': usage.used,
                'free': usage.free,
                'percent': usage.percent
            })
            
        return jsonify({'success': True, 'disks': disks})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500