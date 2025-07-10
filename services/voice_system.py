"""
🎤 KIRA VOICE SERVICE
Integration des Enterprise Voice Systems mit WSL Support
"""

import logging
from typing import Dict, Any, Optional
from config.system_config import KiraSystemConfig

# Import des neuen Enterprise Voice Managers
from voice.core.enterprise_voice_manager import EnterpriseVoiceManager

logger = logging.getLogger(__name__)

class VoiceService:
    """Kira Voice Service mit Enterprise Voice Manager"""
    
    def __init__(self, system_config: KiraSystemConfig):
        self.system_config = system_config
        self.voice_manager = None
        self.initialized = False
        self.status = {
            'available': False,
            'backend_type': 'unknown',
            'wsl_environment': False,
            'error': None
        }
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize Voice Service with Enterprise Voice Manager"""
        try:
            logger.info("🎤 Initializing Enterprise Voice Service...")
            
            # Check if voice is enabled in config
            if not self.system_config.voice.enable_voice:
                logger.info("🔇 Voice system disabled in configuration")
                return {
                    'success': False,
                    'available': False,
                    'status': 'disabled',
                    'reason': 'Voice system disabled in configuration'
                }
            
            # Create Enterprise Voice Manager
            voice_config = self._build_voice_config()
            
            # 🌉 AUTO-DETECT WSL and create manager accordingly
            self.voice_manager = EnterpriseVoiceManager(
                config=voice_config,
                memory_service=None,  # Will be set later
                command_processor=None,  # Will be set later
                force_wsl_mode=None  # Auto-detect
            )
            
            # Initialize the voice system
            success = self.voice_manager.initialize_voice_system()
            
            if success:
                self.initialized = True
                self.status = {
                    'available': True,
                    'backend_type': self.voice_manager.status.audio_backend_type,
                    'wsl_environment': self.voice_manager.status.wsl_environment,
                    'windows_bridge_connected': self.voice_manager.status.windows_bridge_connected,
                    'components_loaded': self.voice_manager.status.components_loaded,
                    'status': 'active'
                }
                
                logger.info("✅ Enterprise Voice Service initialized successfully")
                logger.info(f"   🌉 Backend: {self.status['backend_type']}")
                logger.info(f"   🐧 WSL Environment: {self.status['wsl_environment']}")
                
                return {
                    'success': True,
                    'available': True,
                    'status': 'active',
                    'backend_type': self.status['backend_type'],
                    'wsl_environment': self.status['wsl_environment']
                }
            else:
                error_msg = "Enterprise Voice Manager initialization failed"
                logger.error(f"❌ {error_msg}")
                self.status['error'] = error_msg
                
                return {
                    'success': False,
                    'available': False,
                    'status': 'failed',
                    'error': error_msg
                }
        
        except Exception as e:
            error_msg = f"Voice Service initialization error: {e}"
            logger.error(f"❌ {error_msg}")
            self.status['error'] = error_msg
            
            return {
                'success': False,
                'available': False,
                'status': 'error',
                'error': error_msg
            }
    
    def _build_voice_config(self) -> Dict[str, Any]:
        """Build voice configuration from system config"""
        voice_config = self.system_config.voice
        
        # Convert Kira config to Enterprise Voice Manager format
        enterprise_config = {
            "audio": {
                "input_device": getattr(voice_config, 'input_device', None),
                "output_device": getattr(voice_config, 'output_device', None),
                "sample_rate": getattr(voice_config, 'sample_rate', 16000),
                "chunk_size": getattr(voice_config, 'chunk_size', 1024),
                "channels": 1,
                "buffer_size": 8192,
                "enable_noise_suppression": getattr(voice_config, 'enable_noise_suppression', True),
                "enable_echo_cancellation": getattr(voice_config, 'enable_echo_cancellation', True),
                "enable_auto_gain_control": getattr(voice_config, 'enable_auto_gain_control', True),
                "input_gain": getattr(voice_config, 'input_gain', 1.0),
                "output_volume": getattr(voice_config, 'output_volume', 0.8)
            },
            "wake_word": {
                "wake_words": getattr(voice_config, 'wake_words', ["kira", "kiera"]),
                "threshold": getattr(voice_config, 'wake_word_threshold', 0.7),
                "enable_verification": True,
                "continuous_listening": getattr(voice_config, 'continuous_listening', True)
            },
            "speech_recognition": {
                "model_size": getattr(voice_config, 'whisper_model_size', "base"),
                "language": "de",
                "enable_vad": True,
                "max_duration": 10.0
            },
            "voice_synthesis": {
                "voice_preset": getattr(voice_config, 'voice_preset', "v2/de/speaker_6"),
                "enable_emotion_modulation": True,
                "enable_caching": True,
                "default_emotion": "neutral"
            },
            "personality": {
                "formality_level": 0.7,
                "friendliness": 0.85,
                "response_style": "professional_friendly"
            }
        }
        
        return enterprise_config
    
    def start_voice_system(self) -> bool:
        """Start the voice system"""
        if not self.initialized or not self.voice_manager:
            logger.error("❌ Voice service not initialized")
            return False
        
        try:
            success = self.voice_manager.start_voice_system()
            if success:
                logger.info("✅ Voice system started successfully")
            else:
                logger.error("❌ Voice system start failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Voice system start error: {e}")
            return False
    
    def stop_voice_system(self) -> bool:
        """Stop the voice system"""
        if not self.voice_manager:
            return True
        
        try:
            success = self.voice_manager.stop_voice_system()
            if success:
                logger.info("🛑 Voice system stopped successfully")
            else:
                logger.error("❌ Voice system stop failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Voice system stop error: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get voice service status"""
        if not self.voice_manager:
            return {
                'initialized': False,
                'running': False,
                'backend_type': 'none',
                'error': 'Voice manager not initialized'
            }
        
        # Get detailed status from Enterprise Voice Manager
        detailed_status = self.voice_manager.get_detailed_status()
        
        return {
            'initialized': self.initialized,
            'service_status': self.status,
            'voice_manager_status': detailed_status,
            'quick_status': {
                'running': detailed_status['system']['running'],
                'listening': detailed_status['system']['listening'],
                'backend_type': detailed_status['system']['audio_backend_type'],
                'wsl_environment': detailed_status['system']['wsl_environment'],
                'components_loaded': len([k for k, v in detailed_status['components'].items() if v])
            }
        }
    
    def speak(self, text: str, emotion: str = "neutral") -> Dict[str, Any]:
        """Speak text with emotion"""
        if not self.voice_manager:
            return {
                'success': False,
                'error': 'Voice manager not available'
            }
        
        try:
            result = self.voice_manager.speak(text, emotion=emotion)
            if isinstance(result, dict):
                return result
            elif result:  # Boolean True
                return {
                    'success': True,
                    'audio_generated': True
                }
            else:
                return {
                    'success': False,
                    'error': 'Speech synthesis failed'
                }
        except Exception as e:
            logger.error(f"❌ Speak error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def listen_for_wake_word(self, callback=None) -> bool:
        """Start listening for wake word"""
        if not self.voice_manager:
            return False
        
        try:
            return self.voice_manager.start_wake_word_detection(callback)
        except Exception as e:
            logger.error(f"❌ Wake word detection error: {e}")
            return False
    
    def record_voice_command(self, duration: float = 5.0) -> Optional[str]:
        """Record and transcribe voice command"""
        if not self.voice_manager:
            return None
        
        try:
            return self.voice_manager.record_and_transcribe_command(duration)
        except Exception as e:
            logger.error(f"❌ Voice command recording error: {e}")
            return None
    
    def set_memory_service(self, memory_service):
        """Set memory service for voice manager"""
        if self.voice_manager:
            self.voice_manager.memory_service = memory_service
    
    def set_command_processor(self, command_processor):
        """Set command processor for voice manager"""
        if self.voice_manager:
            self.voice_manager.command_processor = command_processor
    
    def cleanup(self):
        """Cleanup voice service"""
        try:
            if self.voice_manager:
                self.voice_manager.cleanup()
            
            logger.info("🧹 Voice Service cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Voice service cleanup error: {e}")