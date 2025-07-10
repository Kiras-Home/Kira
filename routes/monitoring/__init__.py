from flask import Blueprint
from .cpu import cpu_bp
from .memory import memory_bp
from .system import system_bp
from .network import network_bp
from .storage import storage_bp
# ✅ WICHTIG: system_logs NICHT hier importieren (hat anderen URL-Präfix)

monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/api/monitoring')

# Registriere alle Sub-Blueprints
monitoring_bp.register_blueprint(cpu_bp)
monitoring_bp.register_blueprint(memory_bp)
monitoring_bp.register_blueprint(system_bp)
monitoring_bp.register_blueprint(network_bp)
monitoring_bp.register_blueprint(storage_bp)

print("✅ Monitoring sub-blueprints registered:")
print("   - /api/monitoring/cpu/usage")
print("   - /api/monitoring/memory/usage")
print("   - /api/monitoring/network/status")
print("   - /api/monitoring/storage/status")
print("   - /api/monitoring/system/uptime")
print("   - /api/monitoring/system/status")