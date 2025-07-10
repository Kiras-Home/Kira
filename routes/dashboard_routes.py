from flask import Blueprint, render_template
import logging
import os

__dashboard_routes__ = "routes.dashboard_routes"
logger = logging.getLogger(__dashboard_routes__)

dashboard_bp = Blueprint('dashboard', __name__, url_prefix='')

@dashboard_bp.route('/')
def index():
    logger.info("📊 Dashboard index route called")
    return render_template('pages/main_dashboard.html')

@dashboard_bp.route('/chat')
def chat_dashboard():
    return render_template('pages/chat_dashboard.html')

@dashboard_bp.route('/memory')
def memory_dashboard():
    return render_template('pages/memory_dashboard.html')

@dashboard_bp.route('/todo')
def todo_dashboard():
    return render_template('pages/todo_dashboard.html')

@dashboard_bp.route('/system')
def system_dashboard():
    return render_template('pages/system_dashboard.html')

@dashboard_bp.route('/health')
def health_dashboard():
    return render_template('pages/health_dashboard.html')

@dashboard_bp.route('/config')
def config_dashboard():
    return render_template('pages/config_dashboard.html')

@dashboard_bp.route('/storage')
def storage_dashboard():
    return render_template('pages/storage_dashboard.html')

@dashboard_bp.route('/settings')
def settings_dashboard():
    return render_template('pages/settings_content.html')

@dashboard_bp.route('/voice-test')
def voice_test():
    return render_template('voice_test.html')  # Bleibt unverändert

@dashboard_bp.route('/brain-memory-test')
def brain_memory_test():
    return render_template('pages/brain_memory_test.html')

print("✅ Dashboard Blueprint registered")