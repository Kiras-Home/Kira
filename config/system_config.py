"""
🎛️ KIRA SYSTEM CONFIGURATION
Zentrale Konfigurationsverwaltung für alle Kira-Komponenten
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

@dataclass
class LMStudioConfig:
    """LM Studio Konfiguration"""
    url: str = "http://192.168.178.44:1234/v1"
    model_name: str = "auto"
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3
    enable_streaming: bool = False
    context_window: int = 4096
    
    # Add backward compatibility properties
    @property
    def host(self) -> str:
        """Extract host from URL for backward compatibility"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            return parsed.hostname or 'localhost'
        except:
            return 'localhost'
    
    @property 
    def port(self) -> int:
        """Extract port from URL for backward compatibility"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            return parsed.port or 1234
        except:
            return 1234

@dataclass
class VoiceConfig:
    """Voice System Konfiguration - Erweitert für deutsche weibliche Stimme"""
    enable_voice: bool = True
    
    # ✅ DEUTSCHE WEIBLICHE STIMME KONFIGURATION
    default_emotion: str = "calm"
    voice_language: str = "de"  # Deutsch
    voice_gender: str = "female"  # Weiblich
    voice_speed: float = 1.0
    voice_quality: str = "high"
    
    # ✅ BARK SPEZIFISCHE EINSTELLUNGEN
    bark_model: str = "suno/bark"
    bark_voice_preset: str = "v2/de_speaker_6"  # Deutsche weibliche Stimme
    bark_text_temp: float = 0.7
    bark_waveform_temp: float = 0.7
    
    # ✅ HARDWARE DETECTION & ANPASSUNG
    enable_hardware_detection: bool = True
    auto_select_devices: bool = True
    microphone_device: Optional[str] = None
    speaker_device: Optional[str] = None
    audio_buffer_size: int = 1024
    sample_rate: int = 24000
    channels: int = 1  # Mono für Sprache
    
    # ✅ WAKE WORD DETECTION
    enable_wake_word: bool = True
    wake_words: List[str] = field(default_factory=lambda: ["kira", "hey kira"])
    wake_word_threshold: float = 0.8
    wake_word_timeout: int = 30  # Sekunden
    
    # ✅ VOICE COMMANDS ERWEITERT
    enable_voice_commands: bool = True
    command_confidence_threshold: float = 0.7
    enable_continuous_listening: bool = True
    voice_activation_threshold: float = 0.02  # RMS threshold
    silence_timeout_ms: int = 2000
    
    # ✅ AUDIO PROCESSING
    enable_noise_reduction: bool = True
    enable_echo_cancellation: bool = True
    enable_auto_gain_control: bool = True
    
    # ✅ PERFORMANCE OPTIMIERUNG
    enable_voice_activity_detection: bool = True
    processing_chunk_size: int = 1024
    max_recording_duration: int = 30  # Sekunden
    
    output_format: str = "wav"
    enable_emotion_synthesis: bool = True
    whisper_model: str = "base"
    enable_speech_recognition: bool = True

@dataclass
class MemoryConfig:
    """Memory System Konfiguration"""
    enable_memory: bool = True
    data_dir: str = "data/kira_memory"
    stm_capacity: int = 50
    ltm_capacity: int = 1000
    consolidation_threshold: float = 0.7
    importance_threshold: int = 5
    enable_storage: bool = True
    storage_backend: str = "sqlite"
    enable_vector_search: bool = False
    enable_caching: bool = True
    auto_consolidation: bool = True
    consolidation_interval: int = 3600  # seconds
    memory_file_path: str = "memory/data/kira_memory.json"

@dataclass
class EmotionConfig:
    """Emotion Engine Konfiguration"""
    enable_emotion_analysis: bool = True
    enable_personality_profiling: bool = True
    emotion_confidence_threshold: float = 0.3
    personality_learning_rate: float = 0.3
    enable_response_personalization: bool = True
    emotion_history_limit: int = 100
    personality_update_frequency: int = 10  # conversations
    enable_emotional_voice: bool = True

@dataclass
class SecurityConfig:
    """Sicherheitskonfiguration"""
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 60
    enable_input_validation: bool = True
    max_input_length: int = 5000
    enable_content_filtering: bool = True
    log_conversations: bool = True
    enable_user_authentication: bool = False
    session_timeout: int = 3600

@dataclass
class PerformanceConfig:
    """Performance Konfiguration"""
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    enable_caching: bool = True
    cache_ttl: int = 300
    enable_compression: bool = True
    log_level: str = "INFO"
    enable_profiling: bool = False
    memory_limit_mb: int = 2048

@dataclass
class DatabaseConfig:
    """Datenbank Konfiguration"""
    database_type: str = "sqlite"
    database_url: str = "sqlite:///data/kira.db"
    connection_pool_size: int = 5
    enable_migrations: bool = True
    backup_enabled: bool = True
    backup_interval: int = 86400  # 24 hours
    retention_days: int = 30

@dataclass
class APIConfig:
    """API Konfiguration"""
    host: str = "0.0.0.0"
    port: int = 5001
    debug: bool = True
    enable_cors: bool = True
    api_key_required: bool = False
    enable_swagger: bool = True
    request_logging: bool = True
    enable_websockets: bool = True

class KiraSystemConfig:
    """
    🎛️ ZENTRALE KIRA SYSTEM KONFIGURATION
    Verwaltet alle Komponenten-Konfigurationen
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize System Configuration
        
        Args:
            config_file: Path to configuration file (JSON/YAML)
        """
        self.config_file = config_file or "config/kira_config.yaml"
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize default configurations
        self.lm_studio = LMStudioConfig()
        self.voice = VoiceConfig()
        self.memory = MemoryConfig()
        self.emotion = EmotionConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        self.database = DatabaseConfig()
        self.api = APIConfig()
        
        # System metadata
        self.system_info = {
            'version': '2.0.0',
            'build': 'production',
            'environment': 'development',
            'created_at': None,
            'last_updated': None
        }
        
        # Load configuration if file exists
        self.load_configuration()
        
        logger.info(f"🎛️ Kira System Configuration initialized from {self.config_file}")
    
    def load_configuration(self) -> bool:
        """
        ✅ LOAD CONFIGURATION FROM FILE
        
        Returns:
            Success status
        """
        try:
            config_path = Path(self.config_file)
            
            if not config_path.exists():
                logger.info(f"Config file not found: {config_path}, using defaults")
                self.save_configuration()  # Create default config
                return True
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update configurations
            if 'lm_studio' in config_data:
                self._update_dataclass_from_dict(self.lm_studio, config_data['lm_studio'])
            
            if 'voice' in config_data:
                self._update_dataclass_from_dict(self.voice, config_data['voice'])
            
            if 'memory' in config_data:
                self._update_dataclass_from_dict(self.memory, config_data['memory'])
            
            if 'emotion' in config_data:
                self._update_dataclass_from_dict(self.emotion, config_data['emotion'])
            
            if 'security' in config_data:
                self._update_dataclass_from_dict(self.security, config_data['security'])
            
            if 'performance' in config_data:
                self._update_dataclass_from_dict(self.performance, config_data['performance'])
            
            if 'database' in config_data:
                self._update_dataclass_from_dict(self.database, config_data['database'])
            
            if 'api' in config_data:
                self._update_dataclass_from_dict(self.api, config_data['api'])
            
            if 'system_info' in config_data:
                self.system_info.update(config_data['system_info'])
            
            logger.info(f"✅ Configuration loaded successfully from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            return False
    
    def save_configuration(self) -> bool:
        """
        ✅ SAVE CONFIGURATION TO FILE
        
        Returns:
            Success status
        """
        try:
            config_path = Path(self.config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare configuration data
            config_data = {
                'lm_studio': asdict(self.lm_studio),
                'voice': asdict(self.voice),
                'memory': asdict(self.memory),
                'emotion': asdict(self.emotion),
                'security': asdict(self.security),
                'performance': asdict(self.performance),
                'database': asdict(self.database),
                'api': asdict(self.api),
                'system_info': self.system_info
            }
            
            # Update metadata
            from datetime import datetime
            config_data['system_info']['last_updated'] = datetime.now().isoformat()
            if not config_data['system_info']['created_at']:
                config_data['system_info']['created_at'] = datetime.now().isoformat()
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration saving failed: {e}")
            return False
    
    def get_environment_overrides(self) -> Dict[str, Any]:
        """
        ✅ GET ENVIRONMENT VARIABLE OVERRIDES
        
        Returns:
            Dictionary of environment overrides
        """
        overrides = {}
        
        # LM Studio overrides
        if os.getenv('KIRA_LM_STUDIO_URL'):
            overrides['lm_studio_url'] = os.getenv('KIRA_LM_STUDIO_URL')
        
        if os.getenv('KIRA_LM_STUDIO_MODEL'):
            overrides['lm_studio_model'] = os.getenv('KIRA_LM_STUDIO_MODEL')
        
        # Voice overrides
        if os.getenv('KIRA_VOICE_ENABLED'):
            overrides['voice_enabled'] = os.getenv('KIRA_VOICE_ENABLED').lower() == 'true'
        
        # Memory overrides
        if os.getenv('KIRA_MEMORY_DIR'):
            overrides['memory_dir'] = os.getenv('KIRA_MEMORY_DIR')
        
        # Database overrides
        if os.getenv('KIRA_DATABASE_URL'):
            overrides['database_url'] = os.getenv('KIRA_DATABASE_URL')
        
        # API overrides
        if os.getenv('KIRA_API_PORT'):
            overrides['api_port'] = int(os.getenv('KIRA_API_PORT'))
        
        if os.getenv('KIRA_API_HOST'):
            overrides['api_host'] = os.getenv('KIRA_API_HOST')
        
        # Debug mode
        if os.getenv('KIRA_DEBUG'):
            overrides['debug'] = os.getenv('KIRA_DEBUG').lower() == 'true'
        
        return overrides
    
    def apply_environment_overrides(self):
        """
        ✅ APPLY ENVIRONMENT OVERRIDES
        """
        try:
            overrides = self.get_environment_overrides()
            
            for key, value in overrides.items():
                if key == 'lm_studio_url':
                    self.lm_studio.url = value
                elif key == 'lm_studio_model':
                    self.lm_studio.model_name = value
                elif key == 'voice_enabled':
                    self.voice.enable_voice = value
                elif key == 'memory_dir':
                    self.memory.data_dir = value
                elif key == 'database_url':
                    self.database.database_url = value
                elif key == 'api_port':
                    self.api.port = value
                elif key == 'api_host':
                    self.api.host = value
                elif key == 'debug':
                    self.api.debug = value
            
            if overrides:
                logger.info(f"✅ Applied {len(overrides)} environment overrides")
            
        except Exception as e:
            logger.error(f"❌ Environment overrides failed: {e}")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        ✅ VALIDATE CONFIGURATION
        
        Returns:
            Validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'components': {}
        }
        
        try:
            # Validate LM Studio Config
            lm_validation = self._validate_lm_studio_config()
            validation_results['components']['lm_studio'] = lm_validation
            if not lm_validation['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend(lm_validation['errors'])
            
            # Validate Voice Config
            voice_validation = self._validate_voice_config()
            validation_results['components']['voice'] = voice_validation
            if voice_validation['warnings']:
                validation_results['warnings'].extend(voice_validation['warnings'])
            
            # Validate Memory Config
            memory_validation = self._validate_memory_config()
            validation_results['components']['memory'] = memory_validation
            if not memory_validation['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend(memory_validation['errors'])
            
            # Validate Database Config
            db_validation = self._validate_database_config()
            validation_results['components']['database'] = db_validation
            if not db_validation['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend(db_validation['errors'])
            
            logger.info(f"🔍 Configuration validation completed: {'✅ Valid' if validation_results['valid'] else '❌ Invalid'}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            validation_results.update({
                'valid': False,
                'errors': [f"Validation error: {str(e)}"]
            })
            return validation_results
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        ✅ GET CONFIGURATION SUMMARY
        
        Returns:
            Configuration summary for dashboard
        """
        try:
            return {
                'system_info': self.system_info,
                'components_enabled': {
                    'lm_studio': True,  # Always enabled
                    'voice_system': self.voice.enable_voice,
                    'memory_system': self.memory.enable_memory,
                    'emotion_engine': self.emotion.enable_emotion_analysis,
                    'personality_profiling': self.emotion.enable_personality_profiling,
                    'database_storage': self.memory.enable_storage
                },
                'security_features': {
                    'rate_limiting': self.security.enable_rate_limiting,
                    'input_validation': self.security.enable_input_validation,
                    'content_filtering': self.security.enable_content_filtering,
                    'conversation_logging': self.security.log_conversations
                },
                'performance_settings': {
                    'caching_enabled': self.performance.enable_caching,
                    'compression_enabled': self.performance.enable_compression,
                    'max_concurrent_requests': self.performance.max_concurrent_requests,
                    'memory_limit_mb': self.performance.memory_limit_mb
                },
                'api_settings': {
                    'host': self.api.host,
                    'port': self.api.port,
                    'debug_mode': self.api.debug,
                    'cors_enabled': self.api.enable_cors,
                    'websockets_enabled': self.api.enable_websockets
                },
                'storage_settings': {
                    'database_type': self.database.database_type,
                    'backup_enabled': self.database.backup_enabled,
                    'retention_days': self.database.retention_days
                }
            }
            
        except Exception as e:
            logger.error(f"Configuration summary failed: {e}")
            return {'error': str(e)}
    
    # ✅ PRIVATE VALIDATION METHODS
    
    def _validate_lm_studio_config(self) -> Dict[str, Any]:
        """Validate LM Studio configuration"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        if not self.lm_studio.url.startswith(('http://', 'https://')):
            result['valid'] = False
            result['errors'].append("LM Studio URL must start with http:// or https://")
        
        if self.lm_studio.max_tokens < 100:
            result['warnings'].append("Max tokens is very low (< 100)")
        
        if self.lm_studio.temperature < 0 or self.lm_studio.temperature > 2:
            result['warnings'].append("Temperature should be between 0 and 2")
        
        return result
    
    def _validate_voice_config(self) -> Dict[str, Any]:
        """Validate Voice configuration"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        if self.voice.voice_speed < 0.5 or self.voice.voice_speed > 2.0:
            result['warnings'].append("Voice speed should be between 0.5 and 2.0")
        
        valid_emotions = ['calm', 'excited', 'empathetic', 'thoughtful', 'playful']
        if self.voice.default_emotion not in valid_emotions:
            result['warnings'].append(f"Default emotion should be one of: {valid_emotions}")
        
        return result
    
    def _validate_memory_config(self) -> Dict[str, Any]:
        """Validate Memory configuration"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            memory_path = Path(self.memory.data_dir)
            memory_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Cannot create memory directory: {e}")
        
        if self.memory.stm_capacity < 10:
            result['warnings'].append("STM capacity is very low (< 10)")
        
        if self.memory.ltm_capacity < 100:
            result['warnings'].append("LTM capacity is very low (< 100)")
        
        return result
    
    def _validate_database_config(self) -> Dict[str, Any]:
        """Validate Database configuration"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        if self.database.database_type == 'sqlite':
            # Validate SQLite path
            if 'sqlite:///' in self.database.database_url:
                db_path = self.database.database_url.replace('sqlite:///', '')
                try:
                    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    result['valid'] = False
                    result['errors'].append(f"Cannot create database directory: {e}")
        
        if self.database.retention_days < 1:
            result['warnings'].append("Retention days should be at least 1")
        
        return result
    
    def _update_dataclass_from_dict(self, dataclass_instance, data_dict):
        """Update dataclass instance from dictionary"""
        for key, value in data_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)

# ✅ GLOBAL CONFIGURATION INSTANCE
_global_config = None

def get_system_config() -> KiraSystemConfig:
    """
    Get global system configuration instance
    
    Returns:
        KiraSystemConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = KiraSystemConfig()
        _global_config.apply_environment_overrides()
    return _global_config

def reload_system_config(config_file: Optional[str] = None) -> KiraSystemConfig:
    """
    Reload system configuration
    
    Args:
        config_file: Optional new config file path
        
    Returns:
        New KiraSystemConfig instance
    """
    global _global_config
    _global_config = KiraSystemConfig(config_file)
    _global_config.apply_environment_overrides()
    return _global_config

# Export
__all__ = [
    'KiraSystemConfig',
    'LMStudioConfig',
    'VoiceConfig', 
    'MemoryConfig',
    'EmotionConfig',
    'SecurityConfig',
    'PerformanceConfig',
    'DatabaseConfig',
    'APIConfig',
    'get_system_config',
    'reload_system_config'
]