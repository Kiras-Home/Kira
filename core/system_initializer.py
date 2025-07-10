"""
Kira System Initializer
Handles initialization of all Kira subsystems
"""

import logging
from datetime import datetime
from typing import Dict, Any

from config.system_config import get_system_config, KiraSystemConfig
from services.memory_service import MemoryService
from services.voice_system import VoiceService  
from services.lm_studio_service import LMStudioService

logger = logging.getLogger(__name__)

# Global system state
SYSTEM_STATE = {
    'initialization_time': None,
    'systems_status': {},
    'kira_ready': False,
    'full_ai_experience': False,
    'configuration': {}
}

# Global service instances
_services = {
    'memory': None,
    'voice': None,
    'lm_studio': None
}


def initialize_kira_system() -> Dict[str, Any]:
    """
    Initialize complete Kira system with all components
    
    Returns:
        Dictionary with initialization results
    """
    global SYSTEM_STATE, _services
    
    print("🚀 INITIALIZING COMPLETE KIRA SYSTEM")
    print("=" * 60)
    
    # Set initialization time
    SYSTEM_STATE['initialization_time'] = datetime.now().isoformat()
    
    # 1. Load system configuration
    system_config = _load_system_configuration()
    
    # 2. Initialize all services
    services_results = _initialize_all_services(system_config)
    
    # 3. Update system state
    _update_system_state(services_results, system_config)
    
    # 4. Calculate final system status
    result = _calculate_system_status()
    
    print(f"\n📊 SYSTEM INITIALIZATION COMPLETE")
    print(f"   🏗️ Initialized: {result['initialized_systems']}/5 systems")
    print(f"   ✅ Available: {result['available_systems']}/5 systems")
    print(f"   🎛️ Configuration: {'Valid' if result['configuration_valid'] else 'Invalid'}")
    
    return result


def _load_system_configuration() -> KiraSystemConfig:
    """Load and validate system configuration"""
    print("\n🎛️ Loading System Configuration...")
    
    try:
        system_config = get_system_config()
        config_validation = system_config.validate_configuration()
        
        if config_validation['valid']:
            print("✅ System configuration loaded and validated")
            print(f"   📄 Config version: {system_config.system_info['version']}")
            print(f"   🏗️ Environment: {system_config.system_info['environment']}")
            
            SYSTEM_STATE['configuration'] = {
                'loaded': True,
                'valid': True,
                'config_file': system_config.config_file,
                'version': system_config.system_info['version'],
                'components_enabled': {
                    'lm_studio': True,
                    'voice_system': system_config.voice.enable_voice,
                    'memory_system': system_config.memory.enable_memory,
                    'emotion_engine': system_config.emotion.enable_emotion_analysis,
                    'database_storage': system_config.memory.enable_storage
                }
            }
        else:
            print("⚠️ Configuration validation failed:")
            for error in config_validation['errors']:
                print(f"   ❌ {error}")
            SYSTEM_STATE['configuration'] = {
                'loaded': True,
                'valid': False,
                'errors': config_validation['errors']
            }
            
        return system_config
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        SYSTEM_STATE['configuration'] = {
            'loaded': False,
            'valid': False,
            'error': str(e)
        }
        return KiraSystemConfig()  # Use defaults


def _initialize_all_services(system_config: KiraSystemConfig) -> Dict[str, Dict]:
    """Initialize all Kira services"""
    results = {}
    
    # Initialize Memory Service
    print("\n🧠 Initializing Memory Service...")
    memory_service = MemoryService(system_config)
    memory_result = memory_service.initialize()
    results['memory'] = memory_result
    if memory_result['success']:
        _services['memory'] = memory_service
        print("✅ Memory Service initialized successfully")
    else:
        print(f"❌ Memory Service failed: {memory_result.get('error')}")
    
    # Initialize Voice Service (ERWEITERT)
    print("\n🎤 Initializing Enterprise Voice Service...")
    voice_service = VoiceService(system_config)
    voice_result = voice_service.initialize()
    results['voice'] = voice_result
    
    if voice_result['success']:
        _services['voice'] = voice_service
        
        # 🔗 WICHTIG: Connect Memory Service to Voice
        if memory_service and memory_result['success']:
            voice_service.set_memory_service(memory_service)
            print("🔗 Voice-Memory integration established")
        
        # Display voice backend info
        backend_type = voice_result.get('backend_type', 'unknown')
        wsl_env = voice_result.get('wsl_environment', False)
        
        print("✅ Enterprise Voice Service initialized successfully")
        print(f"   🌉 Backend: {backend_type}")
        print(f"   🐧 WSL Environment: {wsl_env}")
        
        if backend_type == 'wsl_bridge':
            print("   💡 Using WSL Audio Bridge for audio processing")
    else:
        print(f"❌ Voice Service failed: {voice_result.get('error')}")
        error = voice_result.get('error', '')
        if 'bridge' in error.lower():
            print("   💡 If using WSL, make sure Windows Audio Bridge is running:")
            print("      1. Open Windows Command Prompt")
            print("      2. cd C:\\Users\\LeonT\\Desktop\\Kira_Home")
            print("      3. python voice_bridge.py")
    
    # Initialize LM Studio Service
    print("\n🤖 Initializing LM Studio Service...")
    lm_studio_service = LMStudioService(system_config)
    lm_studio_result = lm_studio_service.initialize()
    results['lm_studio'] = lm_studio_result
    if lm_studio_result['success']:
        _services['lm_studio'] = lm_studio_service
        print("✅ LM Studio Service initialized successfully")
    else:
        print(f"❌ LM Studio Service failed: {lm_studio_result.get('error')}")
    
    # Initialize Storage Service (placeholder)
    results['storage'] = {'success': True, 'available': True, 'status': 'active'}
    
    # Initialize Chat Service (depends on LM Studio)
    results['chat'] = {
        'success': lm_studio_result['success'],
        'available': lm_studio_result['success'],
        'status': 'active' if lm_studio_result['success'] else 'offline'
    }
    
    return results


def _update_system_state(services_results: Dict, system_config: KiraSystemConfig):
    """Update global system state based on service initialization results"""
    SYSTEM_STATE['systems_status'] = {
        'memory_system': {
            'available': services_results['memory']['success'],
            'initialized': services_results['memory']['success'],
            'status': services_results['memory'].get('status', 'unknown'),
            'config': system_config.memory
        },
        'voice_system': {
            'available': services_results['voice']['success'],
            'initialized': services_results['voice']['success'],
            'status': services_results['voice'].get('status', 'unknown'),
            'config': system_config.voice
        },
        'lm_studio': {
            'available': services_results['lm_studio']['success'],
            'initialized': services_results['lm_studio']['success'],
            'status': services_results['lm_studio'].get('status', 'unknown'),
            'config': system_config.lm_studio
        },
        'storage_system': {
            'available': services_results['storage']['success'],
            'initialized': services_results['storage']['success'],
            'status': services_results['storage'].get('status', 'unknown'),
            'config': system_config.database
        },
        'chat_system': {
            'available': services_results['chat']['success'],
            'initialized': services_results['chat']['success'],
            'status': services_results['chat'].get('status', 'unknown'),
            'config': system_config.api
        }
    }


def _calculate_system_status() -> Dict[str, Any]:
    """Calculate final system status and capabilities"""
    initialized_systems = sum(1 for sys in SYSTEM_STATE['systems_status'].values() if sys['initialized'])
    available_systems = sum(1 for sys in SYSTEM_STATE['systems_status'].values() if sys['available'])
    
    # Determine Kira capabilities
    SYSTEM_STATE['kira_ready'] = available_systems >= 1
    SYSTEM_STATE['full_ai_experience'] = (
        SYSTEM_STATE['systems_status']['lm_studio']['available'] and 
        SYSTEM_STATE['systems_status']['voice_system']['available'] and
        SYSTEM_STATE['systems_status']['memory_system']['available']
    )
    SYSTEM_STATE['configuration_loaded'] = SYSTEM_STATE['configuration']['valid']
    
    return {
        'success': initialized_systems > 0,
        'initialized_systems': initialized_systems,
        'available_systems': available_systems,
        'systems_status': SYSTEM_STATE['systems_status'],
        'kira_ready': SYSTEM_STATE['kira_ready'],
        'full_ai_experience': SYSTEM_STATE['full_ai_experience'],
        'configuration_loaded': SYSTEM_STATE['configuration_loaded'],
        'configuration_valid': SYSTEM_STATE['configuration']['valid'],
        'system_state': SYSTEM_STATE,
        'services': _services
    }


def get_system_state() -> Dict[str, Any]:
    """Get current system state"""
    return SYSTEM_STATE


def get_service(service_name: str):
    """Get a specific service instance"""
    return _services.get(service_name)


def get_all_services() -> Dict[str, Any]:
    """Get all service instances"""
    return _services.copy()