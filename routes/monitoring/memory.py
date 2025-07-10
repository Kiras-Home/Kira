from flask import Blueprint, jsonify
from ..helpers.metrics import collect_memory_metrics
import psutil

memory_bp = Blueprint('memory', __name__, url_prefix='/memory')

@memory_bp.route('/usage', methods=['GET'])
def get_memory_usage():
    """Gibt die aktuelle Speicherauslastung und den freien Speicher zurück."""
    try:
        memory_metrics = collect_memory_metrics()
        memory_metrics['memory_free'] = psutil.virtual_memory().available
        return jsonify({'success': True, 'memory_metrics': memory_metrics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500