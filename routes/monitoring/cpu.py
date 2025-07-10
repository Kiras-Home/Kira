from flask import Blueprint, jsonify
from ..helpers.metrics import collect_cpu_metrics
import psutil


cpu_bp = Blueprint('cpu', __name__, url_prefix='/cpu')

@cpu_bp.route('/usage', methods=['GET'])
def get_cpu_usage():
    """Gibt die aktuelle CPU-Auslastung und die Anzahl der Kerne zurück."""
    try:
        cpu_metrics = collect_cpu_metrics()
        cpu_metrics['cpu_count'] = psutil.cpu_count(logical=True)  # Anzahl der logischen Kerne hinzufügen
        return jsonify({'success': True, 'cpu_metrics': cpu_metrics})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500