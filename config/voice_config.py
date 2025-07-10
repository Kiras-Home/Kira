"""
🎤 ENTERPRISE VOICE CONFIGURATION
Professional Voice System Configuration für Kira
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio Hardware Configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    buffer_size: int = 8192
    
    # Professional Audio Settings
    enable_noise_reduction: bool = True
    enable_echo_cancellation: bool = True
    enable_auto_gain_control: bool = True
    noise_threshold: float = 0.01
    silence_threshold: float = 0.015
    
    # Hardware Settings
    input_device_id: Optional[int] = None
    output_device_id: Optional[int] = None
    audio_backend: str = "pyaudio"  # "pyaudio", "sounddevice", "alsa"


@dataclass
class WakeWordConfig:
    """Wake Word Detection Configuration"""
    enabled: bool = True
    wake_words: List[str] = field(default_factory=lambda: ["kira", "kiera"])
    confidence_threshold: float = 0.7
    
    # Multi-Model Detection
    primary_model: str = "picovoice"  # "picovoice", "snowboy", "custom"
    fallback_model: str = "simple"
    enable_multi_model: bool = True
    
    # Performance Settings
    detection_sensitivity: float = 0.5
    false_positive_filter: bool = True
    adaptive_threshold: bool = True
    
    # Model Paths
    model_path: Optional[str] = None
    custom_model_path: Optional[str] = None
    
    # Advanced Settings
    pre_trigger_buffer: float = 0.5  # Seconds before wake word
    post_trigger_timeout: float = 8.0  # Seconds to listen after wake word


@dataclass
class SpeechRecognitionConfig:
    """Speech Recognition Configuration"""
    enabled: bool = True
    engine: str = "whisper"  # "whisper", "google", "azure"
    
    # Whisper Settings
    whisper_model: str = "base"  # "tiny", "base", "small", "medium", "large"
    language: str = "de"  # German
    enable_vad: bool = True  # Voice Activity Detection
    enable_preprocessing: bool = True
    
    # Performance Settings
    max_duration: float = 30.0
    timeout: float = 10.0
    phrase_timeout: float = 2.0
    
    # Quality Settings
    enable_denoising: bool = True
    enable_normalization: bool = True
    confidence_threshold: float = 0.6


@dataclass
class TextToSpeechConfig:
    """Text-to-Speech Configuration"""
    enabled: bool = True
    engine: str = "bark"  # "bark", "elevenlabs", "azure", "system"
    
    # Bark Settings
    bark_voice_preset: str = "v2/de/speaker_6"  # Deutsche weibliche Stimme
    bark_model_size: str = "small"  # "small", "medium", "large"
    enable_emotion_modulation: bool = True
    enable_speed_control: bool = True
    
    # Voice Personality
    base_emotion: str = "neutral"
    voice_speed: float = 1.0
    voice_pitch: float = 1.0
    voice_energy: float = 1.0
    
    # Audio Enhancement
    enable_audio_enhancement: bool = True
    enable_noise_gate: bool = True
    output_quality: str = "high"  # "low", "medium", "high"
    
    # Caching
    enable_voice_caching: bool = True
    cache_size_mb: int = 500
    cache_duration_days: int = 7


@dataclass
class PersonalityConfig:
    """Voice Personality Configuration"""
    personality_type: str = "friendly_assistant"
    cultural_context: str = "german"
    
    # Emotional Expression
    emotion_expressiveness: float = 0.7  # 0.0 = monotone, 1.0 = very expressive
    emotional_memory: bool = True
    emotional_adaptation: bool = True
    
    # Response Style
    response_length: str = "medium"  # "short", "medium", "long"
    formality_level: str = "casual"  # "formal", "casual", "friendly"
    humor_level: float = 0.3  # 0.0 = serious, 1.0 = very humorous
    
    # German-specific
    use_regional_expressions: bool = True
    politeness_level: str = "standard"  # "formal", "standard", "casual"


@dataclass
class CommandConfig:
    """Voice Command Configuration"""
    enabled: bool = True
    
    # Command Processing
    enable_fuzzy_matching: bool = True
    command_confidence_threshold: float = 0.6
    enable_context_awareness: bool = True
    
    # Command Categories
    enable_system_commands: bool = True
    enable_memory_commands: bool = True
    enable_smart_home_commands: bool = True
    enable_general_chat: bool = True
    
    # Advanced Features
    enable_multi_step_commands: bool = True
    enable_command_chaining: bool = True
    command_timeout: float = 30.0


@dataclass
class IntegrationConfig:
    """System Integration Configuration"""
    
    # Memory System Integration
    enable_memory_integration: bool = True
    store_voice_interactions: bool = True
    memory_importance_threshold: int = 5
    
    # Emotion System Integration
    enable_emotion_integration: bool = True
    emotion_feedback_loop: bool = True
    
    # Smart Home Integration
    enable_smart_home_integration: bool = False
    smart_home_confidence_threshold: float = 0.8
    
    # Learning Integration
    enable_learning_integration: bool = True
    adapt_to_user_speech: bool = True
    personalization_level: float = 0.7


@dataclass
class PerformanceConfig:
    """Performance and Optimization Configuration"""
    
    # Threading
    enable_async_processing: bool = True
    max_worker_threads: int = 4
    audio_buffer_threads: int = 2
    
    # Caching
    enable_model_caching: bool = True
    enable_response_caching: bool = True
    cache_warmup: bool = True
    
    # Resource Management
    max_memory_usage_mb: int = 2048
    gpu_acceleration: bool = True
    cpu_optimization: bool = True
    
    # Monitoring
    enable_performance_monitoring: bool = True
    log_performance_metrics: bool = True
    performance_alert_threshold: float = 2.0  # seconds


@dataclass
class SecurityConfig:
    """Security and Privacy Configuration"""
    
    # Privacy
    store_audio_locally: bool = False
    audio_retention_days: int = 0  # 0 = don't store
    anonymize_transcripts: bool = True
    
    # Security
    enable_user_authentication: bool = False
    require_wake_word: bool = True
    command_rate_limiting: bool = True
    max_commands_per_minute: int = 30
    
    # Data Protection
    encrypt_stored_data: bool = True
    secure_communication: bool = True


@dataclass
class EnterpriseVoiceConfig:
    """
    🎤 ENTERPRISE VOICE CONFIGURATION
    Comprehensive configuration for professional voice system
    """
    
    # Core Configuration Sections
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    speech_recognition: SpeechRecognitionConfig = field(default_factory=SpeechRecognitionConfig)
    text_to_speech: TextToSpeechConfig = field(default_factory=TextToSpeechConfig)
    personality: PersonalityConfig = field(default_factory=PersonalityConfig)
    commands: CommandConfig = field(default_factory=CommandConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # System Settings
    debug_mode: bool = False
    log_level: str = "INFO"
    config_version: str = "2.0.0"
    
    # File Paths
    data_dir: Path = field(default_factory=lambda: Path("data/voice"))
    model_dir: Path = field(default_factory=lambda: Path("models/voice"))
    cache_dir: Path = field(default_factory=lambda: Path("cache/voice"))
    output_dir: Path = field(default_factory=lambda: Path("output/voice"))
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Ensure directories exist
        for directory in [self.data_dir, self.model_dir, self.cache_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values"""
        # Audio validation
        if self.audio.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            logger.warning(f"Unusual sample rate: {self.audio.sample_rate}")
        
        # Wake word validation
        if self.wake_word.confidence_threshold < 0.1 or self.wake_word.confidence_threshold > 1.0:
            raise ValueError("Wake word confidence threshold must be between 0.1 and 1.0")
        
        # TTS validation
        if self.text_to_speech.voice_speed < 0.5 or self.text_to_speech.voice_speed > 2.0:
            logger.warning(f"Voice speed {self.text_to_speech.voice_speed} might be too extreme")
        
        # Memory validation
        if self.performance.max_memory_usage_mb < 512:
            logger.warning("Low memory limit might affect performance")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        
        for section_name in ['audio', 'wake_word', 'speech_recognition', 'text_to_speech', 
                           'personality', 'commands', 'integration', 'performance', 'security']:
            section = getattr(self, section_name)
            if hasattr(section, '__dict__'):
                result[section_name] = {}
                for key, value in section.__dict__.items():
                    if isinstance(value, Path):
                        result[section_name][key] = str(value)
                    elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                        result[section_name][key] = value
                    else:
                        result[section_name][key] = value
        
        # Add system settings
        result['debug_mode'] = self.debug_mode
        result['log_level'] = self.log_level
        result['config_version'] = self.config_version
        
        # Add paths
        result['paths'] = {
            'data_dir': str(self.data_dir),
            'model_dir': str(self.model_dir),
            'cache_dir': str(self.cache_dir),
            'output_dir': str(self.output_dir)
        }
        
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnterpriseVoiceConfig':
        """Create configuration from dictionary"""
        
        # Extract sections
        sections = {}
        
        if 'audio' in config_dict:
            sections['audio'] = AudioConfig(**config_dict['audio'])
        
        if 'wake_word' in config_dict:
            sections['wake_word'] = WakeWordConfig(**config_dict['wake_word'])
        
        if 'speech_recognition' in config_dict:
            sections['speech_recognition'] = SpeechRecognitionConfig(**config_dict['speech_recognition'])
        
        if 'text_to_speech' in config_dict:
            sections['text_to_speech'] = TextToSpeechConfig(**config_dict['text_to_speech'])
        
        if 'personality' in config_dict:
            sections['personality'] = PersonalityConfig(**config_dict['personality'])
        
        if 'commands' in config_dict:
            sections['commands'] = CommandConfig(**config_dict['commands'])
        
        if 'integration' in config_dict:
            sections['integration'] = IntegrationConfig(**config_dict['integration'])
        
        if 'performance' in config_dict:
            sections['performance'] = PerformanceConfig(**config_dict['performance'])
        
        if 'security' in config_dict:
            sections['security'] = SecurityConfig(**config_dict['security'])
        
        # System settings
        system_settings = {
            'debug_mode': config_dict.get('debug_mode', False),
            'log_level': config_dict.get('log_level', 'INFO'),
            'config_version': config_dict.get('config_version', '2.0.0')
        }
        
        # Paths
        if 'paths' in config_dict:
            paths = config_dict['paths']
            system_settings.update({
                'data_dir': Path(paths.get('data_dir', 'data/voice')),
                'model_dir': Path(paths.get('model_dir', 'models/voice')),
                'cache_dir': Path(paths.get('cache_dir', 'cache/voice')),
                'output_dir': Path(paths.get('output_dir', 'output/voice'))
            })
        
        return cls(**sections, **system_settings)
    
    def get_legacy_config(self) -> Dict[str, Any]:
        """
        Convert to legacy VoiceConfig format for backward compatibility
        """
        return {
            # Basic settings
            'sample_rate': self.audio.sample_rate,
            'channels': self.audio.channels,
            'language': self.speech_recognition.language,
            'whisper_model': self.speech_recognition.whisper_model,
            'bark_voice': self.text_to_speech.bark_voice_preset,
            'output_dir': str(self.output_dir),
            'max_duration': self.speech_recognition.max_duration,
            
            # Wake word
            'wake_word': self.wake_word.wake_words[0] if self.wake_word.wake_words else "kira",
            'enable_wake_word': self.wake_word.enabled,
            'wake_word_confidence': self.wake_word.confidence_threshold,
            
            # Features
            'enable_speech_recognition': self.speech_recognition.enabled,
            'enable_voice_commands': self.commands.enabled,
            'enable_emotion_synthesis': self.text_to_speech.enable_emotion_modulation,
            
            # Performance
            'command_timeout': self.commands.command_timeout,
            'silence_threshold': self.audio.silence_threshold,
            'chunk_size': self.audio.chunk_size
        }
    
    def optimize_for_hardware(self, cpu_cores: int = 4, ram_gb: int = 8, has_gpu: bool = False):
        """Optimize configuration based on hardware specs"""
        
        # CPU optimization
        if cpu_cores >= 8:
            self.performance.max_worker_threads = min(8, cpu_cores)
            self.speech_recognition.whisper_model = "medium"
        elif cpu_cores >= 4:
            self.performance.max_worker_threads = 4
            self.speech_recognition.whisper_model = "base"
        else:
            self.performance.max_worker_threads = 2
            self.speech_recognition.whisper_model = "tiny"
        
        # RAM optimization
        if ram_gb >= 16:
            self.performance.max_memory_usage_mb = 4096
            self.text_to_speech.cache_size_mb = 1000
        elif ram_gb >= 8:
            self.performance.max_memory_usage_mb = 2048
            self.text_to_speech.cache_size_mb = 500
        else:
            self.performance.max_memory_usage_mb = 1024
            self.text_to_speech.cache_size_mb = 200
            self.performance.enable_model_caching = False
        
        # GPU optimization
        self.performance.gpu_acceleration = has_gpu
        
        logger.info(f"Voice config optimized for: {cpu_cores} cores, {ram_gb}GB RAM, GPU: {has_gpu}")


# Default Enterprise Configuration
DEFAULT_ENTERPRISE_VOICE_CONFIG = EnterpriseVoiceConfig()

# Preset Configurations

def get_development_config() -> EnterpriseVoiceConfig:
    """Configuration for development environment"""
    config = EnterpriseVoiceConfig()
    
    # Development optimizations
    config.debug_mode = True
    config.log_level = "DEBUG"
    config.speech_recognition.whisper_model = "tiny"  # Faster for development
    config.text_to_speech.bark_model_size = "small"
    config.performance.enable_model_caching = True
    config.security.store_audio_locally = True  # For debugging
    config.security.audio_retention_days = 1
    
    return config


def get_production_config() -> EnterpriseVoiceConfig:
    """Configuration for production environment"""
    config = EnterpriseVoiceConfig()
    
    # Production optimizations
    config.debug_mode = False
    config.log_level = "INFO"
    config.speech_recognition.whisper_model = "base"  # Good balance
    config.text_to_speech.bark_model_size = "medium"
    config.performance.enable_performance_monitoring = True
    config.security.store_audio_locally = False  # Privacy
    config.security.encrypt_stored_data = True
    config.security.command_rate_limiting = True
    
    return config


def get_high_performance_config() -> EnterpriseVoiceConfig:
    """Configuration for high-performance systems"""
    config = EnterpriseVoiceConfig()
    
    # High performance optimizations
    config.speech_recognition.whisper_model = "large"
    config.text_to_speech.bark_model_size = "large"
    config.text_to_speech.output_quality = "high"
    config.performance.max_worker_threads = 8
    config.performance.max_memory_usage_mb = 4096
    config.performance.gpu_acceleration = True
    config.audio.enable_noise_reduction = True
    config.audio.enable_echo_cancellation = True
    
    return config


def get_privacy_focused_config() -> EnterpriseVoiceConfig:
    """Configuration with maximum privacy protection"""
    config = EnterpriseVoiceConfig()
    
    # Privacy optimizations
    config.security.store_audio_locally = False
    config.security.audio_retention_days = 0
    config.security.anonymize_transcripts = True
    config.security.encrypt_stored_data = True
    config.text_to_speech.enable_voice_caching = False  # No voice caching
    config.integration.store_voice_interactions = False
    
    return config


# Configuration Factory
class VoiceConfigFactory:
    """Factory for creating voice configurations"""
    
    @staticmethod
    def create_config(
        preset: str = "default",
        hardware_optimization: bool = True,
        **overrides
    ) -> EnterpriseVoiceConfig:
        """
        Create voice configuration with preset and overrides
        
        Args:
            preset: "default", "development", "production", "high_performance", "privacy"
            hardware_optimization: Auto-optimize for detected hardware
            **overrides: Configuration overrides
        """
        
        # Get preset configuration
        if preset == "development":
            config = get_development_config()
        elif preset == "production":
            config = get_production_config()
        elif preset == "high_performance":
            config = get_high_performance_config()
        elif preset == "privacy":
            config = get_privacy_focused_config()
        else:
            config = EnterpriseVoiceConfig()
        
        # Hardware optimization
        if hardware_optimization:
            try:
                import psutil
                cpu_count = psutil.cpu_count()
                memory_gb = psutil.virtual_memory().total // (1024**3)
                
                # Simple GPU detection
                has_gpu = False
                try:
                    import torch
                    has_gpu = torch.cuda.is_available()
                except ImportError:
                    pass
                
                config.optimize_for_hardware(cpu_count, memory_gb, has_gpu)
                
            except ImportError:
                logger.warning("psutil not available, skipping hardware optimization")
        
        # Apply overrides
        for key, value in overrides.items():
            if '.' in key:
                # Nested attribute (e.g., 'audio.sample_rate')
                section, attr = key.split('.', 1)
                if hasattr(config, section):
                    section_obj = getattr(config, section)
                    if hasattr(section_obj, attr):
                        setattr(section_obj, attr, value)
            else:
                # Top-level attribute
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config


# Export
__all__ = [
    'EnterpriseVoiceConfig',
    'AudioConfig',
    'WakeWordConfig', 
    'SpeechRecognitionConfig',
    'TextToSpeechConfig',
    'PersonalityConfig',
    'CommandConfig',
    'IntegrationConfig',
    'PerformanceConfig',
    'SecurityConfig',
    'VoiceConfigFactory',
    'DEFAULT_ENTERPRISE_VOICE_CONFIG',
    'get_development_config',
    'get_production_config',
    'get_high_performance_config',
    'get_privacy_focused_config'
]