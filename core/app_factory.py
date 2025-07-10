"""
Kira Flask App Factory
Creates and configures the complete Flask application
"""

import logging
import os
from pathlib import Path
from flask import Flask, render_template
from flask_cors import CORS
from flask_sock import Sock

from api.system_routes import create_system_routes
from api.kira_routes import create_kira_routes
# from api.chat_routes import create_chat_routes  # Old API route
from routes.dashboard_routes import dashboard_bp

logger = logging.getLogger(__name__)


def create_kira_app(init_result: dict) -> Flask:
    """
    Create complete Kira Flask application
    
    Args:
        init_result: System initialization results
        
    Returns:
        Configured Flask application
    """
    print("\n🌐 Creating Kira Application...")
    
    # Bestimme das korrekte Projekt-Root-Verzeichnis
    project_root = Path(__file__).parent.parent.absolute()
    template_folder = project_root / "templates"
    static_folder = project_root / "static"
    
    print(f"📁 Project root: {project_root}")
    print(f"📄 Template folder: {template_folder}")
    print(f"🎨 Static folder: {static_folder}")
    
    # Prüfe ob Template-Ordner existiert
    if not template_folder.exists():
        print(f"❌ Template folder nicht gefunden: {template_folder}")
        raise FileNotFoundError(f"Template folder nicht gefunden: {template_folder}")
    
    # Create Flask app mit korrekten Pfaden
    app = Flask(
        __name__,
        template_folder=str(template_folder),
        static_folder=str(static_folder)
    )
    app.config['SECRET_KEY'] = 'kira-production-key-change-in-production'
    app.config['DEBUG'] = True
    
    # Flask Konfiguration für bessere Template-Unterstützung
    app.config['SERVER_NAME'] = 'localhost:5001'
    app.config['APPLICATION_ROOT'] = '/'
    app.config['PREFERRED_URL_SCHEME'] = 'http'
    
    # Enable CORS and WebSocket support
    CORS(app)
    sock = Sock(app)
    
    # Register core blueprints
    _register_core_blueprints(app, init_result)
    
    # Register API routes
    _register_api_routes(app, init_result, sock)
    
    # Register additional blueprints
    _register_additional_blueprints(app, init_result)
    
    # Setup static file serving
    _setup_static_routes(app)
    
    print("✅ Kira Application created successfully")
    return app


def _register_core_blueprints(app: Flask, init_result: dict):
    """Register core dashboard blueprints"""
    try:
        app.register_blueprint(dashboard_bp)
        print("✅ Dashboard Blueprint registered")
    except Exception as e:
        print(f"❌ Dashboard Blueprint Error: {e}")


def _register_api_routes(app: Flask, init_result: dict, sock: Sock):
    """Register all API route blueprints"""
    
    # System Routes
    try:
        system_bp = create_system_routes(init_result['system_state'], init_result['services'])
        app.register_blueprint(system_bp)
        print("✅ System API Routes registered")
    except Exception as e:
        print(f"❌ System Routes Error: {e}")
    
    # Kira-specific Routes
    try:
        kira_bp = create_kira_routes(init_result['system_state'], init_result['services'])
        app.register_blueprint(kira_bp)
        print("✅ Kira API Routes registered")
    except Exception as e:
        print(f"❌ Kira Routes Error: {e}")
    
    # Memory API Routes
    try:
        from routes.memory.memory_api import create_memory_routes
        memory_bp = create_memory_routes(init_result['system_state'], init_result['services'])
        app.register_blueprint(memory_bp)
        print("✅ Memory API Routes registered")
    except Exception as e:
        print(f"❌ Memory API Routes Error: {e}")
    
    # Chat Routes (Enhanced with memory integration)
    try:
        from routes.chat.chat_routes import create_chat_routes
        chat_bp = create_chat_routes(init_result['system_state'], init_result['services'])
        app.register_blueprint(chat_bp)
        print("✅ Enhanced Chat API Routes registered")
    except Exception as e:
        print(f"❌ Enhanced Chat Routes Error: {e}")
        # Fallback to standard API chat routes
        try:
            chat_bp = create_chat_routes(init_result['system_state'], init_result['services'])
            app.register_blueprint(chat_bp)
            print("✅ Standard Chat API Routes registered (fallback)")
        except Exception as e2:
            print(f"❌ All Chat Routes Error: {e2}")

    # Modern Chat Page Route
    @app.route('/chat')
    def chat_page():
        """Modern chat page"""
        return render_template('modern-chat.html')
    
    print("✅ Modern Chat Page registered")


def _register_additional_blueprints(app: Flask, init_result: dict):
    """Register additional feature blueprints"""
    
    # Chat Dashboard API
    try:
        from routes.chat.dashboard_api import chat_dashboard_bp
        app.register_blueprint(chat_dashboard_bp)
        print("✅ Chat Dashboard API registered")
    except ImportError as e:
        print(f"⚠️ Chat Dashboard API not available: {e}")
    
    # Todo API
    try:
        from routes.todo_api import todo_api
        app.register_blueprint(todo_api)
        print("✅ Todo API registered")
    except ImportError as e:
        print(f"⚠️ Todo API not available: {e}")
    
    # Voice Routes
    try:
        from routes.voice_routes.kira_voice_routes import voice_api
        app.register_blueprint(voice_api)
        print("✅ Voice API registered")
    except ImportError as e:
        print(f"⚠️ Voice API not available: {e}")
    
    # Monitoring
    try:
        from routes.monitoring import monitoring_bp
        from routes.monitoring.system_logs import logs_bp
        app.register_blueprint(monitoring_bp)
        app.register_blueprint(logs_bp)
        print("✅ Monitoring Blueprints registered")
    except ImportError as e:
        print(f"⚠️ Monitoring not available: {e}")
    
    # Emotion API
    try:
        from routes.emotion_api import create_emotion_api_blueprint
        memory_service = init_result['services'].get('memory')
        if memory_service and hasattr(memory_service, 'memory_manager'):
            emotion_api_bp = create_emotion_api_blueprint(memory_service.memory_manager)
            app.register_blueprint(emotion_api_bp, url_prefix='/api')
            print("✅ Emotion API registered")
    except Exception as e:
        print(f"❌ Emotion API registration failed: {e}")
    
    # Configuration API
    try:
        from routes.config_api import create_config_api_blueprint
        config_api_bp = create_config_api_blueprint()
        app.register_blueprint(config_api_bp, url_prefix='/api')
        print("✅ Configuration API registered")
    except Exception as e:
        print(f"❌ Configuration API registration failed: {e}")
    
    # Memory Routes
    try:
        from routes.memory.routes import create_memory_blueprint
        memory_bp = create_memory_blueprint()
        app.register_blueprint(memory_bp, url_prefix='/api/memory')
        print("✅ Memory Blueprint registered")
    except Exception as e:
        print(f"⚠️ Memory Blueprint fallback: {e}")


def _setup_static_routes(app: Flask):
    """Setup static file serving routes"""
    from flask import send_from_directory, send_file, jsonify
    from pathlib import Path
    import os
    
    @app.route('/favicon.ico')
    def favicon():
        return send_from_directory(
            os.path.join(app.root_path, 'static'), 
            'favicon.ico', 
            mimetype='image/vnd.microsoft.icon'
        )
    
    @app.route('/api/audio/<path:filename>')
    def serve_audio(filename):
        """Serve generated audio files"""
        try:
            safe_path = Path("voice/output").resolve()
            file_path = (safe_path / filename).resolve()
            
            if safe_path in file_path.parents and file_path.exists():
                return send_file(
                    file_path,
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name=filename
                )
            else:
                return jsonify({'error': 'Audio file not found'}), 404
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    print("✅ Static routes configured")