"""
🎯 ENTERPRISE VOICE MANAGER
Hauptkoordinator für das gesamte Voice System mit allen Enterprise Komponenten
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import os


logger = logging.getLogger(__name__)


@dataclass
class VoiceSystemStatus:
    """Voice system status information"""
    initialized: bool = False
    running: bool = False
    listening: bool = False
    components_loaded: Dict[str, bool] = None
    current_session: Optional[str] = None
    total_interactions: int = 0
    uptime_seconds: float = 0.0
    last_interaction: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = None
    error_count: int = 0
    
    # 🌉 WSL-specific status
    wsl_environment: bool = False
    windows_bridge_connected: bool = False
    audio_backend_type: str = "unknown"
    
    def __post_init__(self):
        if self.components_loaded is None:
            self.components_loaded = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}


class EnterpriseVoiceManager:
    """
    🎯 ENTERPRISE VOICE MANAGER
    Zentraler Manager für das komplette Voice System mit WSL Backend Support
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        memory_service=None,
        command_processor=None,
        force_wsl_mode: bool = None  # 🌉 NEU: Force WSL mode
    ):
        self.config = config or {}
        self.memory_service = memory_service
        self.command_processor = command_processor
        self.force_wsl_mode = force_wsl_mode
        
        # 🌉 WSL Environment Detection
        self.is_wsl_environment = self._detect_wsl_environment()
        self.use_wsl_backend = self._should_use_wsl_backend()

        if not self.config or len(self.config) == 0:
            logger.info("🔧 Loading default configuration...")
            self._load_configuration()
        else:
            logger.info("🔧 Using provided configuration")
            # ✅ FIX: Merge with defaults to ensure all required keys exist
            self._merge_with_defaults()
        
        # System status
        self.status = VoiceSystemStatus()
        self.status.wsl_environment = self.is_wsl_environment
        self.status.audio_backend_type = "wsl_bridge" if self.use_wsl_backend else "native"
        self.startup_time = time.time()
        self._lock = threading.Lock()
        
        # Core components
        self.audio_manager = None
        self.wake_word_detector = None
        self.speech_recognizer = None
        self.memory_bridge = None
        self.personality_engine = None
        self.bark_engine = None
        self.voice_pipeline = None
        
        # 🌉 WSL-specific components
        self.wsl_audio_backend = None
        self.windows_bridge_connection = None
        
        # Component health monitoring
        self.component_health = {}
        self.health_check_interval = 30  # seconds
        self.last_health_check = 0
        
        # Event callbacks
        self.initialization_callbacks = []
        self.interaction_callbacks = []
        self.error_callbacks = []
        self.status_callbacks = []
        
        logger.info(f"🎯 Enterprise Voice Manager initialized")
        logger.info(f"   🐧 WSL Environment: {self.is_wsl_environment}")
        logger.info(f"   🌉 Using WSL Backend: {self.use_wsl_backend}")
    
    def _detect_wsl_environment(self) -> bool:
        """Detect if running in WSL environment"""
        try:
            # Check for WSL indicators
            if os.path.exists('/proc/version'):
                with open('/proc/version', 'r') as f:
                    version_info = f.read().lower()
                    if 'microsoft' in version_info or 'wsl' in version_info:
                        logger.info("🐧 WSL Environment detected via /proc/version")
                        return True
            
            # Check environment variables
            if 'WSL_DISTRO_NAME' in os.environ or 'WSL_INTEROP' in os.environ:
                logger.info("🐧 WSL Environment detected via environment variables")
                return True
            
            # Check for WSL filesystem
            if os.path.exists('/mnt/c'):
                logger.info("🐧 WSL Environment detected via /mnt/c")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"❌ WSL detection error: {e}")
            return False
    
    def _should_use_wsl_backend(self) -> bool:
        """Determine if WSL backend should be used"""
        if self.force_wsl_mode is not None:
            logger.info(f"🔧 Force WSL mode: {self.force_wsl_mode}")
            return self.force_wsl_mode
        
        if self.is_wsl_environment:
            logger.info("🌉 WSL environment detected - will use WSL backend")
            return True
        
        logger.info("🖥️ Native environment - will use native backend")
        return False
    
    def _merge_with_defaults(self):
        """Merge provided config with defaults to ensure all keys exist"""
        try:
            # Get default config
            default_config = self._get_default_config()
            
            # Deep merge: provided config overrides defaults
            merged_config = {}
            for key, default_value in default_config.items():
                if key in self.config:
                    if isinstance(default_value, dict) and isinstance(self.config[key], dict):
                        # Merge nested dictionaries
                        merged_config[key] = {**default_value, **self.config[key]}
                    else:
                        # Use provided value
                        merged_config[key] = self.config[key]
                else:
                    # Use default value
                    merged_config[key] = default_value
            
            self.config = merged_config
            logger.info("🔧 Configuration merged with defaults")
            
        except Exception as e:
            logger.error(f"❌ Config merge failed: {e}")
            # Fallback: use defaults
            self._load_configuration()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "audio": {
                "input_device": None,
                "output_device": None,
                "sample_rate": 16000,
                "chunk_size": 1024,
                "channels": 1,
                "buffer_size": 8192,
                "enable_noise_suppression": True,
                "enable_echo_cancellation": True,
                "enable_auto_gain_control": True,
                "input_gain": 1.0,
                "output_volume": 0.8
            },
            "wake_word": {
                "wake_words": ["kira", "kiera"],
                "model_path": "models/wake_word/kira_model.onnx",
                "threshold": 0.7,
                "enable_verification": True,
                "enable_noise_filtering": True,
                "enable_adaptive_threshold": True,
                "porcupine_access_key": None,
                "continuous_listening": True
            },
            "speech_recognition": {
                "model_size": "base",
                "language": "de",
                "device": None,
                "compute_type": "float32",
                "enable_vad": True,
                "vad_threshold": 0.5,
                "max_duration": 10.0
            },
            "voice_synthesis": {
                "voice_preset": "v2/de/speaker_6",
                "enable_emotion_modulation": True,
                "enable_speed_control": True,
                "enable_audio_enhancement": True,
                "enable_caching": True,
                "cache_dir": None,
                "model_cache_dir": None,
                "cache_size_mb": 500,
                "default_emotion": "neutral"
            },
            "personality": {
                "formality_level": 0.7,
                "friendliness": 0.85,
                "enable_cultural_adaptation": True,
                "response_style": "professional_friendly"
            },
            "memory": {
                "enable_memory_integration": True,
                "min_importance_for_storage": 5,
                "session_timeout_minutes": 30,
                "enable_context_enhancement": True
            },
            "pipeline": {
                "max_queue_size": 100,
                "enable_async_processing": True,
                "enable_performance_monitoring": True,
                "processing_timeout": 30.0,
                "enable_memory_integration": True
            },
            # 🌉 WSL-specific configuration
            "wsl": {
                "bridge_host": None,  # Auto-detect
                "bridge_port": 7777,
                "connection_timeout": 5.0,
                "retry_attempts": 3,
                "enable_audio_optimization": True,
                "enable_latency_monitoring": True
            }
        }
    
    async def check_windows_bridge_availability(self) -> bool:
        """Check if Windows Audio Bridge is available"""
        if not self.use_wsl_backend:
            return True  # Not needed for native mode
        
        try:
            import socket
            import subprocess
            import asyncio
            
            # Get Windows Host IP
            result = subprocess.run(
                "ip route | grep default | awk '{print $3}'",
                shell=True,
                capture_output=True,
                text=True
            )
            windows_ip = result.stdout.strip()
            
            if not windows_ip:
                logger.error("❌ Could not determine Windows host IP")
                return False
            
            # Test connection to bridge
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(windows_ip, 7777),
                    timeout=3.0
                )
                writer.close()
                await writer.wait_closed()
                
                logger.info(f"✅ Windows Audio Bridge available at {windows_ip}:7777")
                self.status.windows_bridge_connected = True
                return True
                
            except asyncio.TimeoutError:
                logger.error(f"⏰ Windows Audio Bridge timeout at {windows_ip}:7777")
                return False
            except Exception as e:
                logger.error(f"❌ Windows Audio Bridge connection failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Bridge availability check failed: {e}")
            return False
    
    def _check_windows_bridge_sync(self) -> bool:
        """Synchronous check for Windows Bridge (for initialization)"""
        try:
            import socket
            import subprocess
            
            # Get Windows Host IP
            result = subprocess.run(
                "ip route | grep default | awk '{print $3}'",
                shell=True,
                capture_output=True,
                text=True
            )
            windows_ip = result.stdout.strip()
            
            if not windows_ip:
                return False
            
            # Test connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            
            try:
                sock.connect((windows_ip, 7777))
                sock.close()
                self.status.windows_bridge_connected = True
                return True
            except:
                self.status.windows_bridge_connected = False
                return False
                
        except Exception:
            return False
    
    def initialize_voice_system(self) -> bool:
        """Initialize complete voice system - IMPROVED ERROR HANDLING"""
        try:
            logger.info("🚀 ENTERPRISE VOICE SYSTEM INITIALIZATION")
            logger.info("=" * 60)
            
            # ✅ FIX: Initialize components SEPARATELY with error isolation
            success_components = []
            failed_components = []
            
            # Audio System (critical)
            logger.info("🔊 Initializing Audio System...")
            try:
                if self._initialize_audio_system():
                    success_components.append("audio_system")
                    logger.info("✅ Audio System initialized")
                else:
                    failed_components.append("audio_system")
                    logger.error("❌ Audio System failed")
            except Exception as e:
                failed_components.append("audio_system")
                logger.error(f"❌ Audio System exception: {e}")
            
            # Voice Synthesis (critical)
            logger.info("🗣️ Initializing Voice Synthesis...")
            try:
                if self._initialize_voice_synthesis():
                    success_components.append("voice_synthesis")
                    logger.info("✅ Voice Synthesis initialized")
                else:
                    failed_components.append("voice_synthesis")
                    logger.error("❌ Voice Synthesis failed")
            except Exception as e:
                failed_components.append("voice_synthesis")
                logger.error(f"❌ Voice Synthesis exception: {e}")
            
            # Voice Recognition (important but not critical)
            logger.info("🎤 Initializing Voice Recognition...")
            try:
                if self._initialize_voice_recognition():
                    success_components.append("voice_recognition")
                    logger.info("✅ Voice Recognition initialized")
                else:
                    failed_components.append("voice_recognition")
                    logger.warning("⚠️ Voice Recognition failed (non-critical)")
            except Exception as e:
                failed_components.append("voice_recognition")
                logger.warning(f"⚠️ Voice Recognition exception (non-critical): {e}")
            
            # Wake Word Detection (optional)
            logger.info("👂 Initializing Wake Word Detection...")
            try:
                if self._initialize_wake_word_detection():
                    success_components.append("wake_word_detection")
                    logger.info("✅ Wake Word Detection initialized")
                else:
                    failed_components.append("wake_word_detection")
                    logger.warning("⚠️ Wake Word Detection failed (optional)")
            except Exception as e:
                failed_components.append("wake_word_detection")
                logger.warning(f"⚠️ Wake Word Detection exception (optional): {e}")
            
            # ✅ FIX: Check if critical components succeeded
            critical_components = ["audio_system", "voice_synthesis"]
            critical_success = all(comp in success_components for comp in critical_components)
            
            if critical_success:
                self.is_initialized = True
                logger.info("✅ VOICE SYSTEM INITIALIZATION SUCCESSFUL")
                logger.info(f"   Success: {success_components}")
                if failed_components:
                    logger.warning(f"   Failed (non-critical): {failed_components}")
                return True
            else:
                self.is_initialized = False
                logger.error("❌ VOICE SYSTEM INITIALIZATION FAILED")
                logger.error(f"   Critical failures: {[c for c in failed_components if c in critical_components]}")
                logger.error(f"   All failures: {failed_components}")
                
                # ✅ FIX: Cleanup failed initialization
                self._cleanup_failed_initialization()
                return False
            
        except Exception as e:
            logger.error(f"Voice system initialization error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # ✅ FIX: Cleanup on exception
            self._cleanup_failed_initialization()
            return False
        
    def _cleanup_failed_initialization(self):
        """Cleanup after failed initialization"""
        try:
            logger.info("🧹 Cleaning up failed initialization...")
            
            # Cleanup WSL backend if created
            if hasattr(self, 'wsl_audio_backend') and self.wsl_audio_backend:
                self.wsl_audio_backend.cleanup()
                self.wsl_audio_backend = None
            
            # Cleanup audio manager if created
            if hasattr(self, 'audio_manager') and self.audio_manager:
                if hasattr(self.audio_manager, 'cleanup'):
                    self.audio_manager.cleanup()
                self.audio_manager = None
            
            # Reset initialization state
            self.is_initialized = False
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def _initialize_audio_system(self) -> bool:
        """Initialize audio system with WSL backend support - FIXED"""
        try:
            # ✅ FIX: Ensure audio config exists
            if "audio" not in self.config:
                logger.error("❌ Audio configuration missing!")
                logger.info("🔧 Loading default configuration...")
                self._load_configuration()
                
                if "audio" not in self.config:
                    logger.error("❌ Audio configuration still missing after loading defaults!")
                    return False
            
            # ✅ FIX: Check WSL backend decision and log it
            logger.info(f"🔍 Audio System Decision:")
            logger.info(f"   WSL Environment: {self.is_wsl_environment}")
            logger.info(f"   Use WSL Backend: {self.use_wsl_backend}")
            
            # ✅ FIX: ACTUALLY use WSL backend when detected!
            if self.use_wsl_backend:
                logger.info("🌉 Initializing WSL Audio Backend...")
                success = self._initialize_wsl_audio_backend()
                logger.info(f"🌉 WSL Audio Backend Result: {success}")
                return success
            else:
                logger.info("🖥️ Initializing Native Audio Backend...")
                success = self._initialize_native_audio_backend()
                logger.info(f"🖥️ Native Audio Backend Result: {success}")
                return success
                
        except Exception as e:
            logger.error(f"Audio system initialization error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _initialize_wsl_audio_backend(self) -> bool:
        """Initialize WSL Audio Backend with automatic client integration - COMPLETE SOLUTION"""
        try:
            logger.info("🌉 Initializing WSL Audio Backend with client integration...")
            
            # ✅ FIX: Ensure audio config exists
            if "audio" not in self.config:
                logger.error("❌ Audio config missing for WSL backend")
                return False
            
            audio_config = self.config["audio"]
            logger.info(f"🔧 Using audio config: {audio_config}")
            
            # ✅ FIX 1: FIRST ensure Windows Bridge is running
            logger.info("🔍 Checking Windows Voice Bridge availability...")
            if not self._check_bridge_availability():
                logger.warning("⚠️ Windows Voice Bridge not running - attempting to start...")
                # Note: Bridge must be started manually on Windows or via auto-start script
                logger.error("❌ Please start Windows Voice Bridge manually:")
                logger.error("   On Windows: python voice_bridge.py")
                return False
            
            # ✅ FIX 2: Start WSL Client and connect to bridge
            logger.info("🔌 Starting WSL Voice Client...")
            try:
                import sys
                import os
                
                # Add project root to path to find wsl_client
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                from wsl_client import WSLVoiceClient
                
                # Create and connect WSL client
                self.wsl_client = WSLVoiceClient()
                
                logger.info(f"🔌 Connecting to bridge at {self.wsl_client.bridge_host}:{self.wsl_client.bridge_port}")
                
                if not self.wsl_client.connect():
                    logger.error("❌ WSL Client connection failed - bridge not available!")
                    logger.error("   Ensure Windows Voice Bridge is running on Windows")
                    return False
                
                logger.info("✅ WSL Client connected successfully")
                
                # ✅ FIX 3: Test if audio devices are available through bridge
                devices = self.wsl_client.get_audio_devices()
                input_count = len(devices.get('input_devices', []))
                output_count = len(devices.get('output_devices', []))
                
                logger.info(f"🎧 Bridge audio devices: {input_count} input, {output_count} output")
                
                if input_count == 0 and output_count == 0:
                    logger.error("❌ No audio devices available through Windows bridge!")
                    self.wsl_client.disconnect()
                    return False
                
            except ImportError as e:
                logger.error(f"❌ WSL Client module not found: {e}")
                logger.error("   Ensure wsl_client.py is in project root")
                return False
            except Exception as e:
                logger.error(f"❌ WSL Client initialization failed: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return False
            
            # ✅ FIX 4: Create WSL Audio Backend with active client
            try:
                # Import or create WSL Audio Backend
                try:
                    from voice.core.wsl_audio_backend import WSLAudioBackend
                except ImportError:
                    logger.warning("⚠️ WSL Audio Backend not found, creating it...")
                    self._create_wsl_audio_backend_file()
                    from voice.core.wsl_audio_backend import WSLAudioBackend
                
                self.wsl_audio_backend = WSLAudioBackend(
                    sample_rate=audio_config["sample_rate"],
                    channels=audio_config["channels"],
                    chunk_size=audio_config["chunk_size"],
                    wsl_client=self.wsl_client  # ✅ Pass active client
                )
                
                # Initialize WSL backend
                if not self.wsl_audio_backend.initialize():
                    logger.error("❌ WSL Audio Backend initialization failed")
                    self.wsl_client.disconnect()
                    return False
                
                logger.info("✅ WSL Audio Backend created with active client")
                
            except Exception as e:
                logger.error(f"❌ WSL Audio Backend creation failed: {e}")
                self.wsl_client.disconnect()
                return False
            
            # ✅ FIX 5: Create Audio Manager that uses WSL backend
            from voice.audio.enterprise_audio import EnterpriseAudioManager
            
            self.audio_manager = EnterpriseAudioManager(
                sample_rate=audio_config["sample_rate"],
                channels=audio_config["channels"],
                chunk_size=audio_config["chunk_size"],
                buffer_size=audio_config.get("buffer_size", 8192),
                enable_noise_reduction=audio_config.get("enable_noise_suppression", True),
                enable_echo_cancellation=audio_config.get("enable_echo_cancellation", True),
                enable_auto_gain_control=audio_config.get("enable_auto_gain_control", True),
                input_device_id=None,  # WSL backend handles devices
                output_device_id=None,
                use_wsl_bridge=True,  # ✅ Enable WSL mode
                wsl_backend=self.wsl_audio_backend  # ✅ Pass WSL backend
            )
            
            logger.info("✅ WSL Audio Manager created with active WSL client")
            logger.info(f"✅ WSL Audio System fully initialized with {input_count} input, {output_count} output devices")
            return True
            
        except Exception as e:
            logger.error(f"WSL Audio Backend initialization error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # ✅ FIX 6: Cleanup on failure
            if hasattr(self, 'wsl_client') and self.wsl_client:
                self.wsl_client.disconnect()
            
            return False
    
    def _initialize_native_audio_backend(self) -> bool:
        """Initialize native audio backend (fallback)"""
        try:
            logger.info("🖥️ Initializing native audio backend...")
            
            audio_config = self.config["audio"]
            
            from voice.audio.enterprise_audio import EnterpriseAudioManager
            
            self.audio_manager = EnterpriseAudioManager(
                sample_rate=audio_config["sample_rate"],
                channels=audio_config["channels"],
                chunk_size=audio_config["chunk_size"],
                buffer_size=audio_config.get("buffer_size", 8192),
                enable_noise_reduction=audio_config.get("enable_noise_suppression", True),
                enable_echo_cancellation=audio_config.get("enable_echo_cancellation", True),
                enable_auto_gain_control=audio_config.get("enable_auto_gain_control", True),
                input_device_id=audio_config.get("input_device"),
                output_device_id=audio_config.get("output_device"),
                use_wsl_bridge=False  # ✅ Native mode
            )
            
            logger.info("✅ Native audio backend initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Native audio backend initialization failed: {e}")
            return False
    
    def stop_voice_system(self) -> bool:
        """Stop the voice system"""
        try:
            logger.info("🛑 Stopping Enterprise Voice System...")
            
            # Stop voice pipeline
            if self.voice_pipeline:
                self.voice_pipeline.stop_pipeline()
            
            # Update status
            with self._lock:
                self.status.running = False
                self.status.listening = False
            
            logger.info("🛑 Enterprise Voice System stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop voice system: {e}")
            self.status.error_count += 1
            return False
    
    def _load_configuration(self):
        """Load voice system configuration with WSL-specific defaults"""
        try:
            # Get default configuration
            default_config = self._get_default_config()
            
            # 🌉 WSL-specific adjustments
            if self.use_wsl_backend:
                # Optimize for WSL environment
                default_config["audio"]["chunk_size"] = 512  # Smaller chunks for lower latency
                default_config["pipeline"]["enable_async_processing"] = True  # Better for network audio
                logger.info("🌉 Applied WSL-specific configuration optimizations")
            
            # Merge with provided config
            self.config = {**default_config, **self.config}
            
            logger.info("📋 Voice system configuration loaded")
            if self.use_wsl_backend:
                logger.info("🌉 WSL-optimized configuration applied")
            
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")

    def _check_bridge_availability(self) -> bool:
        """Check if Windows Voice Bridge is available"""
        try:
            import socket
            import subprocess
            
            # Get Windows Host IP from WSL
            result = subprocess.run(
                "ip route | grep default | awk '{print $3}'",
                shell=True,
                capture_output=True,
                text=True
            )
            windows_ip = result.stdout.strip()
            
            if not windows_ip:
                logger.error("❌ Could not determine Windows IP from WSL")
                return False
            
            logger.info(f"🔍 Testing bridge connection to {windows_ip}:7777...")
            
            # Test connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            
            try:
                sock.connect((windows_ip, 7777))
                sock.close()
                logger.info("✅ Windows Voice Bridge is available")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Bridge connection failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Bridge availability check failed: {e}")
            return False
    
    
    def _initialize_wake_word_detection(self) -> bool:
        """Initialize wake word detection - WITH SAFE PARAMETER MAPPING"""
        try:
            from voice.recognition.enterprise_wake_word import EnterpriseWakeWordDetector
            
            wake_config = self.config["wake_word"]
            
            # 🔧 SAFE Parameter mapping
            wake_params = {}
            
            param_mappings = {
                "wake_words": ("wake_words", ["kira", "kiera"]),
                "confidence_threshold": ("threshold", 0.7),  # MAPPING!
                "enable_noise_filtering": ("enable_noise_filtering", True),
                "model_path": ("model_path", None),
                "enable_adaptive_threshold": ("enable_adaptive_threshold", True),
                "porcupine_access_key": ("porcupine_access_key", None)
            }
            
            # Validate what EnterpriseWakeWordDetector actually expects
            import inspect
            sig = inspect.signature(EnterpriseWakeWordDetector.__init__)
            expected_params = list(sig.parameters.keys())[1:]  # Skip 'self'
            
            for param_name, (config_key, default_value) in param_mappings.items():
                if param_name in expected_params:
                    wake_params[param_name] = wake_config.get(config_key, default_value)
                else:
                    logger.warning(f"⚠️ Parameter {param_name} not expected by EnterpriseWakeWordDetector")
            
            logger.info(f"🔧 Wake word parameters: {wake_params}")
            
            self.wake_word_detector = EnterpriseWakeWordDetector(**wake_params)
            
            if hasattr(self.wake_word_detector, 'initialize'):
                return self.wake_word_detector.initialize()
            else:
                return True
                
        except Exception as e:
            logger.error(f"Wake word detection initialization error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _initialize_speech_recognition(self) -> bool:
        """Initialize speech recognition - CORRECTED WITH EXACT PARAMETERS"""
        try:
            # ✅ FIX: Ensure speech_recognition config exists
            if "speech_recognition" not in self.config:
                logger.error("❌ Speech recognition configuration missing!")
                logger.info("🔧 Loading default configuration...")
                self._load_configuration()
                
                if "speech_recognition" not in self.config:
                    logger.error("❌ Speech recognition configuration still missing after loading defaults!")
                    return False
            
            from voice.recognition.whisper_engine import WhisperEngine
            
            speech_config = self.config["speech_recognition"]
            
            # ✅ EXAKTE Parameter basierend auf Debug-Ausgabe:
            self.speech_recognizer = WhisperEngine(
                model_size=speech_config["model_size"],          # ✅ KORREKT
                language=speech_config["language"],              # ✅ KORREKT
                device=speech_config.get("device"),             # ✅ NEU HINZUGEFÜGT
                compute_type=speech_config.get("compute_type", "float32")  # ✅ NEU HINZUGEFÜGT
            )
            
            if hasattr(self.speech_recognizer, 'initialize'):
                return self.speech_recognizer.initialize()
            else:
                return True
                
        except Exception as e:
            logger.error(f"Speech recognition initialization error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _initialize_memory_bridge(self) -> bool:
        """Initialize memory bridge"""
        try:
            from voice.integration.memory_bridge import VoiceMemoryBridge
            
            self.memory_bridge = VoiceMemoryBridge(
                memory_service=self.memory_service
            )
            
            # Configure memory settings
            self.memory_bridge.min_importance_for_memory = self.config["memory"]["min_importance_for_storage"]
            self.memory_bridge.session_timeout = timedelta(minutes=self.config["memory"]["session_timeout_minutes"])
            
            return True
            
        except Exception as e:
            logger.error(f"Memory bridge initialization error: {e}")
            return False
    
    def _initialize_personality_engine(self) -> bool:
        """Initialize German personality engine"""
        try:
            from voice.personality.german_personality import GermanPersonalityEngine
            
            self.personality_engine = GermanPersonalityEngine()
            
            # Configure personality traits based on config
            formality = self.config["personality"]["formality_level"]
            friendliness = self.config["personality"]["friendliness"]
            
            self.personality_engine.personality_traits.update({
                "formality": formality,
                "friendliness": friendliness
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Personality engine initialization error: {e}")
            return False
    
    def _initialize_voice_synthesis(self) -> bool:
        """Initialize voice synthesis engine - CORRECTED PARAMETER PASSING"""
        try:
            # ✅ FIX: Ensure voice_synthesis config exists
            if "voice_synthesis" not in self.config:
                logger.error("❌ Voice synthesis configuration missing!")
                logger.info("🔧 Loading default configuration...")
                self._load_configuration()
                
                if "voice_synthesis" not in self.config:
                    logger.error("❌ Voice synthesis configuration still missing after loading defaults!")
                    return False
            
            from voice.synthesis.enterprise_bark import EnterpriseBarkEngine
            
            synthesis_config = self.config["voice_synthesis"]
            
            # ✅ KORREKT: Übergebe gesamte Config als Dictionary
            bark_config = {
                "voice_preset": synthesis_config.get("voice_preset", "v2/de/speaker_6"),
                "enable_emotion_modulation": synthesis_config.get("enable_emotion_modulation", True),
                "enable_speed_control": synthesis_config.get("enable_speed_control", True),
                "enable_audio_enhancement": synthesis_config.get("enable_audio_enhancement", True),
                "cache_enabled": synthesis_config.get("enable_caching", True),
                "cache_dir": synthesis_config.get("cache_dir"),
                "model_cache_dir": synthesis_config.get("model_cache_dir")
            }
            
            logger.info(f"🔧 Voice synthesis config: {bark_config}")
            
            # ✅ Übergebe als config Dictionary
            self.bark_engine = EnterpriseBarkEngine(config=bark_config)
            
            # Initialize the engine
            if hasattr(self.bark_engine, 'initialize'):
                success = self.bark_engine.initialize()
                logger.info(f"🗣️ Bark engine initialized: {success}")
                return success
            else:
                logger.info("🗣️ Bark engine ready")
                return True
                
        except Exception as e:
            logger.error(f"Voice synthesis initialization error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _initialize_voice_pipeline(self) -> bool:
        """Initialize enterprise voice pipeline"""
        try:
            from voice.core.enterprise_pipeline import EnterpriseVoicePipeline
            
            self.voice_pipeline = EnterpriseVoicePipeline(
                audio_manager=self.audio_manager,
                wake_word_detector=self.wake_word_detector,
                speech_recognizer=self.speech_recognizer,
                memory_bridge=self.memory_bridge,
                personality_engine=self.personality_engine,
                bark_engine=self.bark_engine,
                command_processor=self.command_processor,
                max_queue_size=self.config["pipeline"]["max_queue_size"],
                enable_async_processing=self.config["pipeline"]["enable_async_processing"]
            )
            
            # Add interaction callback
            self.voice_pipeline.add_completion_callback(self._on_voice_interaction_complete)
            
            return True
            
        except Exception as e:
            logger.error(f"Voice pipeline initialization error: {e}")
            return False
    
    def _get_component_status(self) -> Dict[str, bool]:
        """Get status of all components"""
        return {
            "audio_manager": self.audio_manager is not None and getattr(self.audio_manager, 'initialized', False),
            "wake_word_detector": self.wake_word_detector is not None and getattr(self.wake_word_detector, 'initialized', False),
            "speech_recognizer": self.speech_recognizer is not None and getattr(self.speech_recognizer, 'initialized', False),
            "memory_bridge": self.memory_bridge is not None,
            "personality_engine": self.personality_engine is not None,
            "bark_engine": self.bark_engine is not None and getattr(self.bark_engine, 'models_loaded', False),
            "voice_pipeline": self.voice_pipeline is not None
        }
    
    def _start_health_monitoring(self):
        """Start component health monitoring"""
        def health_monitor():
            while self.status.running:
                try:
                    self._perform_health_check()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
        
        health_thread = threading.Thread(target=health_monitor, daemon=True)
        health_thread.start()
    
    def _perform_health_check(self):
        """Perform health check on all components"""
        try:
            current_time = time.time()
            
            # Check each component
            health_results = {}
            
            # Audio system health
            if self.audio_manager:
                audio_status = self.audio_manager.get_status()
                health_results["audio"] = audio_status.get("initialized", False)
            
            # Wake word detector health
            if self.wake_word_detector:
                detector_status = self.wake_word_detector.get_status()
                health_results["wake_word"] = detector_status.get("ready", False)
            
            # Speech recognizer health
            if self.speech_recognizer:
                recognizer_status = self.speech_recognizer.get_status()
                health_results["speech_recognition"] = recognizer_status.get("ready", False)
            
            # Voice synthesis health
            if self.bark_engine:
                synthesis_status = self.bark_engine.get_status()
                health_results["voice_synthesis"] = synthesis_status.get("models_loaded", False)
            
            # Pipeline health
            if self.voice_pipeline:
                pipeline_status = self.voice_pipeline.get_status()
                health_results["pipeline"] = pipeline_status.get("running", False)
            
            # Update component health
            self.component_health = health_results
            self.last_health_check = current_time
            
            # Check for critical issues
            critical_components = ["audio", "wake_word", "pipeline"]
            critical_issues = [comp for comp in critical_components if not health_results.get(comp, False)]
            
            if critical_issues:
                logger.warning(f"🚨 Critical component issues detected: {critical_issues}")
                # Could trigger automatic recovery here
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def _on_voice_interaction_complete(self, pipeline_data):
        """Callback for completed voice interactions"""
        try:
            with self._lock:
                self.status.total_interactions += 1
                self.status.last_interaction = datetime.now()
                
                # Update performance metrics
                if hasattr(pipeline_data, 'processing_times'):
                    total_time = sum(pipeline_data.processing_times.values())
                    if "total_latency" not in self.status.performance_metrics:
                        self.status.performance_metrics["total_latency"] = []
                    self.status.performance_metrics["total_latency"].append(total_time)
                    
                    # Keep only recent metrics
                    if len(self.status.performance_metrics["total_latency"]) > 100:
                        self.status.performance_metrics["total_latency"] = self.status.performance_metrics["total_latency"][-50:]
            
            # Call interaction callbacks
            for callback in self.interaction_callbacks:
                try:
                    callback(pipeline_data)
                except Exception as e:
                    logger.error(f"Interaction callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Interaction completion callback error: {e}")
    
    def get_system_status(self) -> VoiceSystemStatus:
        """Get complete system status"""
        with self._lock:
            # Update uptime
            if self.status.running:
                self.status.uptime_seconds = time.time() - self.startup_time
            
            # Update component status
            self.status.components_loaded = self._get_component_status()
            
            # Update pipeline session
            if self.voice_pipeline:
                pipeline_status = self.voice_pipeline.get_status()
                self.status.current_session = pipeline_status.get("current_session")
            
            return self.status
    
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed system status including WSL information"""
        status = self.get_system_status()
        
        detailed_status = {
            "system": {
                "initialized": status.initialized,
                "running": status.running,
                "listening": status.listening,
                "uptime_seconds": status.uptime_seconds,
                "total_interactions": status.total_interactions,
                "error_count": status.error_count,
                "last_interaction": status.last_interaction.isoformat() if status.last_interaction else None,
                # 🌉 WSL-specific status
                "wsl_environment": status.wsl_environment,
                "audio_backend_type": status.audio_backend_type,
                "windows_bridge_connected": status.windows_bridge_connected
            },
            "components": status.components_loaded,
            "component_health": self.component_health,
            "last_health_check": self.last_health_check,
            "performance": status.performance_metrics,
            "configuration": self.config
        }
        
        # Add WSL backend status if available
        if self.wsl_audio_backend:
            detailed_status["wsl_backend"] = self.wsl_audio_backend.get_status()
        
        # Add component-specific details
        if self.voice_pipeline:
            detailed_status["pipeline"] = self.voice_pipeline.get_status()
        
        if self.audio_manager:
            detailed_status["audio"] = self.audio_manager.get_status()
        
        if self.bark_engine:
            detailed_status["voice_synthesis"] = self.bark_engine.get_status()
        
        if self.memory_bridge:
            detailed_status["memory_bridge"] = self.memory_bridge.get_status()
        
        return detailed_status
    
    def add_initialization_callback(self, callback: Callable):
        """Add callback for system initialization"""
        self.initialization_callbacks.append(callback)
    
    def add_interaction_callback(self, callback: Callable):
        """Add callback for voice interactions"""
        self.interaction_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add callback for system errors"""
        self.error_callbacks.append(callback)
    
    def speak(self, text: str, emotion: str = "neutral", auto_play: bool = True) -> Dict[str, Any]:
        """
        Speak text using the Bark engine
        
        Args:
            text: Text to speak
            emotion: Emotion for speech synthesis
            auto_play: Whether to auto-play the audio (for backwards compatibility)
            
        Returns:
            Dict with success status and audio information
        """
        try:
            if not self.bark_engine:
                logger.warning("❌ No Bark engine available for speech synthesis")
                return {
                    'success': False,
                    'error': 'No Bark engine available'
                }
            
            logger.info(f"🗣️ Enterprise Voice Manager speaking: '{text[:50]}...' with emotion: {emotion}")
            
            # Check if bark engine has synthesize method
            if hasattr(self.bark_engine, 'synthesize'):
                result = self.bark_engine.synthesize(text, emotion=emotion)
                
                if result and result.get('success'):
                    logger.info("✅ Enterprise Voice Manager speech synthesis successful")
                    return {
                        'success': True,
                        'audio_url': result.get('audio_url'),
                        'filename': result.get('filename'),
                        'audio_path': result.get('audio_path'),
                        'duration_estimate': result.get('duration_estimate', len(text) * 0.1)
                    }
                else:
                    logger.warning(f"❌ Bark engine synthesis failed: {result}")
                    return {
                        'success': False,
                        'error': f"Bark synthesis failed: {result}"
                    }
            
            # Try alternative methods if synthesize doesn't exist
            elif hasattr(self.bark_engine, 'speak'):
                result = self.bark_engine.speak(text)
                if result:
                    return {
                        'success': True,
                        'audio_path': result,
                        'filename': result.split('/')[-1] if isinstance(result, str) else None,
                        'duration_estimate': len(text) * 0.1
                    }
            
            logger.warning("❌ No compatible speech synthesis method found in Bark engine")
            return {
                'success': False,
                'error': 'No compatible speech synthesis method found'
            }
                
        except Exception as e:
            logger.error(f"❌ Enterprise Voice Manager speak error: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e)
            }

    def cleanup(self):
        """Cleanup voice system including WSL components"""
        try:
            logger.info("🧹 Cleaning up Enterprise Voice System...")
            
            # Stop system if running
            if self.status.running:
                self.stop_voice_system()
            
            # Cleanup WSL backend
            if self.wsl_audio_backend:
                self.wsl_audio_backend.cleanup()
                logger.info("🌉 WSL Audio Backend cleaned up")
            
            # Cleanup other components
            if self.bark_engine:
                self.bark_engine.cleanup()
            
            if self.audio_manager:
                self.audio_manager.cleanup()
            
            logger.info("🧹 Voice system cleanup completed")
            
        except Exception as e:
            logger.error(f"Voice system cleanup error: {e}")

# Export classes
__all__ = [
    'EnterpriseVoiceManager',
    'VoiceSystemStatus'
]