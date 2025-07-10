"""
Voice System Configuration mit Enhanced Features
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

@dataclass
class VoiceConfig:
    """
    Enhanced Voice Configuration mit erweiterten Parametern
    """
    # ✅ BASIC AUDIO SETTINGS
    sample_rate: int = 16000
    channels: int = 1
    language: str = "de"
    
    # ✅ ENGINE SETTINGS
    whisper_model: str = "base"
    bark_voice: str = "v2/de_speaker_0"
    bark_model: str = "suno/bark"
    
    # ✅ FILE SETTINGS
    output_dir: str = "voice/output"
    max_duration: float = 10.0
    
    # ✅ ENHANCED VOICE SETTINGS (NEW)
    enable_voice: bool = True
    default_emotion: str = "calm"
    voice_speed: float = 1.0
    voice_quality: str = "high"
    output_format: str = "wav"
    enable_emotion_synthesis: bool = True
    
    # ✅ TTS CONFIGURATION
    tts_engine: str = "bark"
    tts_voice_id: Optional[str] = None
    tts_rate: int = 200
    tts_volume: float = 0.9
    tts_pitch: int = 0
    
    # ✅ STT CONFIGURATION
    stt_engine: str = "whisper"
    enable_speech_recognition: bool = True
    stt_confidence_threshold: float = 0.7
    stt_enable_partial_results: bool = True
    stt_processing_timeout_ms: int = 5000
    stt_continuous_listening: bool = False
    
    # ✅ HARDWARE CONFIGURATION
    microphone_device: Optional[str] = None
    speaker_device: Optional[str] = None
    
    # ✅ WAKE WORD DETECTION
    enable_wake_word: bool = True
    wake_words: List[str] = field(default_factory=lambda: ['kira', 'hey kira'])
    wake_word_threshold: float = 0.8
    wake_word_sensitivity: float = 0.7
    
    # ✅ COMMAND PROCESSING
    enable_voice_commands: bool = True
    command_confidence_threshold: float = 0.7
    enable_fuzzy_matching: bool = True
    enable_intent_classification: bool = True
    enable_entity_extraction: bool = True
    max_command_history: int = 100
    
    # ✅ AUDIO PROCESSING
    enable_real_time_processing: bool = True
    enable_audio_streaming: bool = True
    audio_output_dir: str = "voice/output"
    noise_reduction: bool = True
    auto_gain_control: bool = True
    echo_cancellation: bool = True
    voice_activation_threshold: float = 0.3
    silence_timeout_ms: int = 2000
    max_recording_duration_ms: int = 30000
    
    # ✅ PERFORMANCE & CACHING
    audio_cache_enabled: bool = True
    audio_cache_ttl: int = 3600
    max_concurrent_sessions: int = 10
    enable_rate_limiting: bool = True
    session_timeout: int = 1800
    
    # ✅ INTEGRATION FEATURES
    enable_context_awareness: bool = True
    enable_conversation_memory: bool = True
    enable_emotion_detection: bool = True
    enable_personality_adaptation: bool = True
    
    # ✅ WEBSOCKET CONFIGURATION
    enable_websocket_server: bool = True
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    
    # ✅ LOGGING & DEBUG
    log_level: str = "INFO"
    enable_profiling: bool = False
    conversation_logging: bool = True
    data_retention_days: int = 30
    
    # ✅ STORAGE INTEGRATION
    enable_persistent_storage: bool = False
    storage_backend: str = "sqlite"
    database_url: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        try:
            # ✅ CREATE OUTPUT DIRECTORIES
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            Path(self.audio_output_dir).mkdir(parents=True, exist_ok=True)
            
            # ✅ VALIDATE SAMPLE RATE
            if self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
                logger.warning(f"⚠️ Unusual sample rate: {self.sample_rate}")
            
            # ✅ VALIDATE VOICE SPEED
            if not 0.5 <= self.voice_speed <= 2.0:
                logger.warning(f"⚠️ Voice speed {self.voice_speed} may cause issues")
                self.voice_speed = max(0.5, min(2.0, self.voice_speed))
            
            # ✅ VALIDATE THRESHOLDS
            if not 0.0 <= self.wake_word_threshold <= 1.0:
                logger.warning(f"⚠️ Invalid wake word threshold: {self.wake_word_threshold}")
                self.wake_word_threshold = 0.8
                
            if not 0.0 <= self.command_confidence_threshold <= 1.0:
                logger.warning(f"⚠️ Invalid command confidence threshold: {self.command_confidence_threshold}")
                self.command_confidence_threshold = 0.7
            
            logger.info(f"✅ Enhanced VoiceConfig initialized")
            logger.info(f"   🎭 Emotion synthesis: {self.enable_emotion_synthesis}")
            logger.info(f"   🎤 Wake word: {self.enable_wake_word}")
            logger.info(f"   🗣️ Speech recognition: {self.enable_speech_recognition}")
            logger.info(f"   📁 Output: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ VoiceConfig post-init failed: {e}")

    def validate(self) -> Dict[str, Any]:
        """Validates the configuration"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # ✅ REQUIRED FIELDS VALIDATION
            if not self.output_dir:
                validation_result['errors'].append("output_dir is required")
                validation_result['valid'] = False
            
            if not self.language:
                validation_result['errors'].append("language is required")
                validation_result['valid'] = False
            
            # ✅ ENGINE VALIDATION
            if self.tts_engine not in ['bark', 'espeak', 'festival']:
                validation_result['warnings'].append(f"Unknown TTS engine: {self.tts_engine}")
            
            if self.stt_engine not in ['whisper', 'google', 'sphinx']:
                validation_result['warnings'].append(f"Unknown STT engine: {self.stt_engine}")
            
            # ✅ PERFORMANCE VALIDATION
            if self.max_concurrent_sessions > 50:
                validation_result['warnings'].append("High max_concurrent_sessions may impact performance")
            
            # ✅ STORAGE VALIDATION
            if self.enable_persistent_storage and not self.database_url:
                validation_result['warnings'].append("Persistent storage enabled but no database_url provided")
            
            return validation_result
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            return validation_result

    def to_dict(self) -> Dict[str, Any]:
        """Converts config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VoiceConfig':
        """Creates VoiceConfig from dictionary with error handling"""
        try:
            # Filter out unknown parameters
            valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
            
            # Log filtered parameters
            filtered_out = set(config_dict.keys()) - valid_fields
            if filtered_out:
                logger.info(f"🔧 Filtered unknown parameters: {filtered_out}")
            
            return cls(**filtered_dict)
            
        except Exception as e:
            logger.error(f"❌ VoiceConfig.from_dict failed: {e}")
            logger.info("🔄 Using default VoiceConfig")
            return cls()

# ✅ DEFAULT ENHANCED CONFIG
DEFAULT_CONFIG = VoiceConfig(
    sample_rate=16000,
    channels=1,
    language="de",
    whisper_model="base",
    bark_voice="v2/de_speaker_0",
    output_dir="voice/output",
    max_duration=10.0,
    enable_voice=True,
    default_emotion="calm",
    enable_emotion_synthesis=True,
    enable_wake_word=True,
    enable_speech_recognition=True,
    enable_voice_commands=True
)

logger.info("📦 Enhanced Voice Config Module geladen")