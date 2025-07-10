"""
Routes Package - Zentrale Blueprint Registrierung
Sammelt alle verfügbaren Routes und macht sie importierbar
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ====================================
# 📋 VERFÜGBARE BLUEPRINTS
# ====================================

def get_available_blueprints() -> Dict[str, Any]:
    """
    Sammelt alle verfügbaren Blueprints mit sicherer Import-Behandlung
    
    Returns:
        Dict: Blueprint-Name -> Blueprint-Info
    """
    blueprints = {}
    
    # 1. 🧠 KIRA ROUTES (Hauptsystem)
    try:
        from .kira_main import kira_bp, init_kira_routes
        blueprints['kira'] = {
            'blueprint': kira_bp,
            'init_function': init_kira_routes,
            'url_prefix': '/api/kira',
            'description': 'Kira Brain Routes - Memory, Personality, Analytics',
            'status': 'available',
            'priority': 1
        }
        logger.info("✅ Kira Routes verfügbar")
    except ImportError as e:
        logger.warning(f"⚠️ Kira Routes nicht verfügbar: {e}")
        blueprints['kira'] = {'status': 'unavailable', 'error': str(e)}
    
    # 2. 🎤 LM STUDIO ROUTES (AI Assistant)
    try:
        from .lm_studio_routes import lm_studio_bp
        blueprints['lm_studio'] = {
            'blueprint': lm_studio_bp,
            'url_prefix': '/api',
            'description': 'LM Studio AI Chat Interface',
            'status': 'available',
            'priority': 2
        }
        logger.info("✅ LM Studio Routes verfügbar")
    except ImportError as e:
        logger.warning(f"⚠️ LM Studio Routes nicht verfügbar: {e}")
        blueprints['lm_studio'] = {'status': 'unavailable', 'error': str(e)}
    
    # 3. 📚 LEARNING ROUTES (Learning Progress)
    try:
        from .learning_routes import learning_bp
        blueprints['learning'] = {
            'blueprint': learning_bp,
            'url_prefix': '/api/learning',
            'description': 'Learning Progress & Statistics',
            'status': 'available',
            'priority': 3
        }
        logger.info("✅ Learning Routes verfügbar")
    except ImportError as e:
        logger.warning(f"⚠️ Learning Routes nicht verfügbar: {e}")
        blueprints['learning'] = {'status': 'unavailable', 'error': str(e)}
    
    # 4. 🔄 SYNC ROUTES (Device Synchronization)
    try:
        from .sync_routes import sync_bp
        blueprints['sync'] = {
            'blueprint': sync_bp,
            'url_prefix': '/api/sync',
            'description': 'Device Synchronization & Heartbeat',
            'status': 'available',
            'priority': 4
        }
        logger.info("✅ Sync Routes verfügbar")
    except ImportError as e:
        logger.warning(f"⚠️ Sync Routes nicht verfügbar: {e}")
        blueprints['sync'] = {'status': 'unavailable', 'error': str(e)}
    
    # 5. 🔧 DEVICE ROUTES (Hardware Control)
    try:
        from .device_routes import device_bp
        blueprints['device'] = {
            'blueprint': device_bp,
            'url_prefix': '/api/device',
            'description': 'Hardware Device Control',
            'status': 'available',
            'priority': 5
        }
        logger.info("✅ Device Routes verfügbar")
    except ImportError as e:
        logger.warning(f"⚠️ Device Routes nicht verfügbar: {e}")
        blueprints['device'] = {'status': 'unavailable', 'error': str(e)}
    
    # 6. 📊 STATUS ROUTES (System Status)
    try:
        from .status_routes import status_bp
        blueprints['status'] = {
            'blueprint': status_bp,
            'url_prefix': '/api/status',
            'description': 'System Status & Health',
            'status': 'available',
            'priority': 6
        }
        logger.info("✅ Status Routes verfügbar")
    except ImportError as e:
        logger.warning(f"⚠️ Status Routes nicht verfügbar: {e}")
        blueprints['status'] = {'status': 'unavailable', 'error': str(e)}
    
    # 7. ⚙️ CONFIG ROUTES (Configuration)
    try:
        from .config_routes import config_bp
        blueprints['config'] = {
            'blueprint': config_bp,
            'url_prefix': '/api/config',
            'description': 'System Configuration',
            'status': 'available',
            'priority': 7
        }
        logger.info("✅ Config Routes verfügbar")
    except ImportError as e:
        logger.warning(f"⚠️ Config Routes nicht verfügbar: {e}")
        blueprints['config'] = {'status': 'unavailable', 'error': str(e)}
    
    # 8. 📝 LOGS ROUTES (Logging System)
    try:
        from .logs_routes import logs_bp
        blueprints['logs'] = {
            'blueprint': logs_bp,
            'url_prefix': '/api/logs',
            'description': 'System Logs & Monitoring',
            'status': 'available',
            'priority': 8
        }
        logger.info("✅ Logs Routes verfügbar")
    except ImportError as e:
        logger.warning(f"⚠️ Logs Routes nicht verfügbar: {e}")
        blueprints['logs'] = {'status': 'unavailable', 'error': str(e)}
    
    # 9. 🎮 DISCORD API (Discord Integration)
    try:
        from .discord_api import discord_api
        blueprints['discord'] = {
            'blueprint': discord_api,
            'url_prefix': '/api/discord',
            'description': 'Discord Bot Integration',
            'status': 'available',
            'priority': 9
        }
        logger.info("✅ Discord API verfügbar")
    except ImportError as e:
        logger.warning(f"⚠️ Discord API nicht verfügbar: {e}")
        blueprints['discord'] = {'status': 'unavailable', 'error': str(e)}
    
    return blueprints

def get_available_blueprint_list() -> List[Dict[str, Any]]:
    """
    Gibt Liste der verfügbaren Blueprints zurück, sortiert nach Priorität
    
    Returns:
        List: Verfügbare Blueprint-Infos
    """
    all_blueprints = get_available_blueprints()
    available = [
        {
            'name': name,
            **info
        }
        for name, info in all_blueprints.items()
        if info.get('status') == 'available'
    ]
    
    # Sortiere nach Priorität
    available.sort(key=lambda x: x.get('priority', 999))
    
    return available

def get_unavailable_blueprint_list() -> List[Dict[str, Any]]:
    """
    Gibt Liste der nicht verfügbaren Blueprints zurück
    
    Returns:
        List: Nicht verfügbare Blueprint-Infos
    """
    all_blueprints = get_available_blueprints()
    unavailable = [
        {
            'name': name,
            **info
        }
        for name, info in all_blueprints.items()
        if info.get('status') == 'unavailable'
    ]
    
    return unavailable

# ====================================
# 🚀 BLUEPRINT REGISTRIERUNG
# ====================================

def register_all_blueprints(app, data_dir: str = "data") -> Dict[str, Any]:
    """
    Registriert alle verfügbaren Blueprints in Flask App
    
    Args:
        app: Flask Application
        data_dir: Datenverzeichnis für Initialisierung
        
    Returns:
        Dict: Registrierungs-Ergebnisse
    """
    results = {
        'registered': [],
        'failed': [],
        'total_available': 0,
        'total_registered': 0
    }
    
    available_blueprints = get_available_blueprint_list()
    results['total_available'] = len(available_blueprints)
    
    logger.info(f"🚀 Starte Blueprint-Registrierung ({len(available_blueprints)} verfügbar)")
    
    for bp_info in available_blueprints:
        bp_name = bp_info['name']
        
        try:
            # Blueprint registrieren
            blueprint = bp_info['blueprint']
            url_prefix = bp_info.get('url_prefix')
            
            if url_prefix:
                app.register_blueprint(blueprint, url_prefix=url_prefix)
            else:
                app.register_blueprint(blueprint)
            
            # Spezielle Initialisierung für Kira Routes
            if bp_name == 'kira' and 'init_function' in bp_info:
                try:
                    init_result = bp_info['init_function'](data_dir)
                    if init_result:
                        logger.info(f"✅ Kira Routes initialisiert")
                    else:
                        logger.warning(f"⚠️ Kira Routes Initialisierung teilweise fehlgeschlagen")
                except Exception as init_error:
                    logger.error(f"❌ Kira Routes Initialisierung Fehler: {init_error}")
            
            results['registered'].append({
                'name': bp_name,
                'url_prefix': url_prefix,
                'description': bp_info.get('description', ''),
                'status': 'success'
            })
            
            logger.info(f"✅ {bp_name} registriert: {url_prefix or '/'}")
            
        except Exception as e:
            results['failed'].append({
                'name': bp_name,
                'error': str(e),
                'status': 'failed'
            })
            logger.error(f"❌ {bp_name} Registrierung fehlgeschlagen: {e}")
    
    results['total_registered'] = len(results['registered'])
    
    # Zusammenfassung
    logger.info(f"📊 Blueprint-Registrierung abgeschlossen:")
    logger.info(f"   ✅ Erfolgreich: {results['total_registered']}")
    logger.info(f"   ❌ Fehlgeschlagen: {len(results['failed'])}")
    
    return results

def print_blueprint_summary():
    """Druckt Zusammenfassung aller Blueprints"""
    available = get_available_blueprint_list()
    unavailable = get_unavailable_blueprint_list()
    
    print("\n🧩 BLUEPRINT ÜBERSICHT")
    print("=" * 50)
    
    if available:
        print("✅ VERFÜGBARE BLUEPRINTS:")
        for bp in available:
            url = bp.get('url_prefix', '/')
            desc = bp.get('description', 'Keine Beschreibung')
            print(f"   {bp['name']:12} {url:15} - {desc}")
    
    if unavailable:
        print("\n❌ NICHT VERFÜGBARE BLUEPRINTS:")
        for bp in unavailable:
            error = bp.get('error', 'Unbekannter Fehler')[:50]
            print(f"   {bp['name']:12} - {error}")
    
    print(f"\n📊 GESAMT: {len(available)} verfügbar, {len(unavailable)} nicht verfügbar")

# ====================================
# 🔍 BLUEPRINT DIAGNOSTICS
# ====================================

def diagnose_blueprints() -> Dict[str, Any]:
    """
    Führt Diagnose aller Blueprints durch
    
    Returns:
        Dict: Diagnose-Ergebnisse
    """
    diagnosis = {
        'timestamp': datetime.now().isoformat(),
        'available_blueprints': {},
        'unavailable_blueprints': {},
        'system_health': {},
        'recommendations': []
    }
    
    # Verfügbare Blueprints analysieren
    available = get_available_blueprint_list()
    for bp in available:
        diagnosis['available_blueprints'][bp['name']] = {
            'url_prefix': bp.get('url_prefix'),
            'description': bp.get('description'),
            'priority': bp.get('priority'),
            'has_init_function': 'init_function' in bp
        }
    
    # Nicht verfügbare Blueprints analysieren
    unavailable = get_unavailable_blueprint_list()
    for bp in unavailable:
        diagnosis['unavailable_blueprints'][bp['name']] = {
            'error': bp.get('error'),
            'error_type': type(bp.get('error', '')).__name__
        }
    
    # System Health Assessment
    total_blueprints = len(available) + len(unavailable)
    health_score = len(available) / total_blueprints if total_blueprints > 0 else 0
    
    diagnosis['system_health'] = {
        'total_blueprints': total_blueprints,
        'available_count': len(available),
        'unavailable_count': len(unavailable),
        'health_score': health_score,
        'status': 'healthy' if health_score > 0.7 else 'degraded' if health_score > 0.5 else 'critical'
    }
    
    # Empfehlungen generieren
    if unavailable:
        diagnosis['recommendations'].append("Prüfen Sie die Import-Abhängigkeiten der fehlenden Blueprints")
    
    if len(available) < 3:
        diagnosis['recommendations'].append("Zu wenige Blueprints verfügbar - System-Setup prüfen")
    
    if 'kira' not in [bp['name'] for bp in available]:
        diagnosis['recommendations'].append("Kira Routes fehlen - Haupt-Funktionalität nicht verfügbar")
    
    return diagnosis

# ====================================
# 📤 PUBLIC EXPORTS
# ====================================

__all__ = [
    'get_available_blueprints',
    'get_available_blueprint_list', 
    'get_unavailable_blueprint_list',
    'register_all_blueprints',
    'print_blueprint_summary',
    'diagnose_blueprints'
]

# Beim Import dieses Moduls Blueprint-Übersicht ausgeben
if __name__ != '__main__':
    try:
        import sys
        if hasattr(sys, '_getframe') and sys._getframe(1).f_globals.get('__name__') == '__main__':
            print_blueprint_summary()
    except:
        pass  # Fail silently wenn Frame-Introspection nicht funktioniert