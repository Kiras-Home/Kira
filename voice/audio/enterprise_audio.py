"""
🔊 ENTERPRISE AUDIO MANAGER
Professional Audio Processing für Kira Voice System
"""

import logging
import threading
import time
import queue
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import io
import wave

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    webrtcvad = None

try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False
    nr = None

logger = logging.getLogger(__name__)


@dataclass
class AudioBuffer:
    """Audio buffer with metadata"""
    data: np.ndarray
    sample_rate: int
    channels: int
    timestamp: float
    duration: float
    is_speech: bool = False
    energy_level: float = 0.0
    snr_db: float = 0.0  # Signal-to-Noise Ratio


@dataclass
class AudioDeviceInfo:
    """Audio device information"""
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_default_input: bool = False
    is_default_output: bool = False


class VoiceActivityDetector:
    """Advanced Voice Activity Detection"""
    
    def __init__(self, sample_rate: int = 16000, aggressiveness: int = 2):
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.vad = None
        self.energy_threshold = 0.01
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.speech_threshold = 3  # Frames needed to detect speech
        self.silence_threshold = 10  # Frames needed to detect silence
        
        # Initialize WebRTC VAD if available
        if WEBRTC_VAD_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(aggressiveness)
                logger.info(f"✅ WebRTC VAD initialized with aggressiveness {aggressiveness}")
            except Exception as e:
                logger.warning(f"WebRTC VAD initialization failed: {e}")
    
    def is_speech(self, audio_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if audio frame contains speech
        
        Returns:
            (is_speech, confidence)
        """
        try:
            # Energy-based detection
            energy = np.sqrt(np.mean(audio_frame ** 2))
            energy_speech = energy > self.energy_threshold
            
            # WebRTC VAD if available
            webrtc_speech = False
            if self.vad and len(audio_frame) in [160, 320, 480]:  # WebRTC supported frame sizes
                try:
                    # Convert to 16-bit PCM
                    audio_16bit = (audio_frame * 32767).astype(np.int16)
                    audio_bytes = audio_16bit.tobytes()
                    webrtc_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
                except Exception as e:
                    logger.debug(f"WebRTC VAD error: {e}")
            
            # Combine detection methods
            is_speech_detected = energy_speech or webrtc_speech
            
            # State tracking for stability
            if is_speech_detected:
                self.consecutive_speech_frames += 1
                self.consecutive_silence_frames = 0
            else:
                self.consecutive_silence_frames += 1
                self.consecutive_speech_frames = 0
            
            # Final decision with hysteresis
            final_is_speech = (
                self.consecutive_speech_frames >= self.speech_threshold or
                (self.consecutive_speech_frames > 0 and self.consecutive_silence_frames < self.silence_threshold)
            )
            
            confidence = min(1.0, max(0.0, energy / (self.energy_threshold * 2)))
            
            return final_is_speech, confidence
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False, 0.0


class NoiseReductionEngine:
    """Advanced Noise Reduction and Audio Enhancement"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.noise_profile = None
        self.noise_reduction_enabled = NOISE_REDUCE_AVAILABLE
        
        # Noise gate parameters
        self.noise_gate_threshold = 0.005
        self.gate_attack_time = 0.01  # seconds
        self.gate_release_time = 0.1  # seconds
        
        logger.info(f"✅ Noise Reduction Engine initialized (noisereduce: {NOISE_REDUCE_AVAILABLE})")
    
    def reduce_noise(self, audio: np.ndarray, stationary: bool = True) -> np.ndarray:
        """Apply noise reduction to audio"""
        try:
            if not self.noise_reduction_enabled or nr is None:
                return self._apply_noise_gate(audio)
            
            # Apply noisereduce library
            if stationary:
                # Stationary noise reduction (faster)
                reduced_audio = nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    stationary=True,
                    prop_decrease=0.8
                )
            else:
                # Non-stationary noise reduction (better quality)
                reduced_audio = nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    stationary=False
                )
            
            # Apply additional noise gate
            return self._apply_noise_gate(reduced_audio)
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return self._apply_noise_gate(audio)
    
    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply simple noise gate"""
        try:
            # Calculate RMS energy in windows
            window_size = int(0.02 * self.sample_rate)  # 20ms windows
            gated_audio = audio.copy()
            
            for i in range(0, len(audio) - window_size, window_size // 2):
                window = audio[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                
                if rms < self.noise_gate_threshold:
                    # Apply gate (reduce volume)
                    gate_factor = max(0.1, rms / self.noise_gate_threshold)
                    gated_audio[i:i + window_size] *= gate_factor
            
            return gated_audio
            
        except Exception as e:
            logger.error(f"Noise gate error: {e}")
            return audio
    
    def estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio"""
        try:
            # Simple SNR estimation
            signal_power = np.mean(audio ** 2)
            
            # Estimate noise from quieter portions
            sorted_audio = np.sort(np.abs(audio))
            noise_samples = sorted_audio[:len(sorted_audio) // 4]  # Bottom 25%
            noise_power = np.mean(noise_samples ** 2)
            
            if noise_power > 0:
                snr_linear = signal_power / noise_power
                snr_db = 10 * np.log10(snr_linear)
                return max(0.0, min(40.0, snr_db))  # Clamp between 0-40 dB
            else:
                return 40.0  # Very clean signal
                
        except Exception as e:
            logger.error(f"SNR estimation error: {e}")
            return 0.0


class AudioBufferManager:
    """Professional Audio Buffer Management"""
    
    def __init__(self, max_buffers: int = 100, max_duration: float = 30.0):
        self.max_buffers = max_buffers
        self.max_duration = max_duration
        self.buffers = queue.Queue(maxsize=max_buffers)
        self.overflow_count = 0
        self.total_duration = 0.0
        self.lock = threading.Lock()
    
    def add_buffer(self, buffer: AudioBuffer) -> bool:
        """Add audio buffer to queue"""
        try:
            with self.lock:
                # Check duration limit
                if self.total_duration + buffer.duration > self.max_duration:
                    # Remove old buffers
                    while not self.buffers.empty() and self.total_duration > self.max_duration * 0.8:
                        old_buffer = self.buffers.get_nowait()
                        self.total_duration -= old_buffer.duration
                
                # Add new buffer
                if not self.buffers.full():
                    self.buffers.put_nowait(buffer)
                    self.total_duration += buffer.duration
                    return True
                else:
                    self.overflow_count += 1
                    return False
                    
        except Exception as e:
            logger.error(f"Buffer add error: {e}")
            return False
    
    def get_buffer(self, timeout: float = 1.0) -> Optional[AudioBuffer]:
        """Get audio buffer from queue"""
        try:
            buffer = self.buffers.get(timeout=timeout)
            with self.lock:
                self.total_duration -= buffer.duration
            return buffer
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Buffer get error: {e}")
            return None
    
    def get_recent_audio(self, duration: float = 5.0) -> Optional[np.ndarray]:
        """Get recent audio data as concatenated array"""
        try:
            recent_buffers = []
            collected_duration = 0.0
            
            # Collect recent buffers without removing them
            temp_buffers = []
            while not self.buffers.empty() and collected_duration < duration:
                buffer = self.buffers.get_nowait()
                temp_buffers.append(buffer)
                recent_buffers.append(buffer.data)
                collected_duration += buffer.duration
            
            # Put buffers back
            for buffer in temp_buffers:
                self.buffers.put_nowait(buffer)
            
            if recent_buffers:
                return np.concatenate(recent_buffers)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Recent audio get error: {e}")
            return None


class EnterpriseAudioManager:
    """
    🔊 ENTERPRISE AUDIO MANAGER
    Professional Audio Processing für Kira Voice System
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        buffer_size: int = 8192,
        enable_noise_reduction: bool = True,
        enable_echo_cancellation: bool = True,
        enable_auto_gain_control: bool = True,
        input_device_id: Optional[int] = None,
        output_device_id: Optional[int] = None,
        use_wsl_bridge: bool = False,  # 🌉 NEU
        wsl_backend=None
    ):
        self.use_wsl_bridge = use_wsl_bridge
        self.wsl_backend = wsl_backend
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        
        # 🚨 DEBUG: Zeige was reingekommen ist
        logger.info(f"🚨 CONSTRUCTOR DEBUG: input_device_id = {input_device_id} (type: {type(input_device_id)})")
        logger.info(f"🚨 CONSTRUCTOR DEBUG: output_device_id = {output_device_id} (type: {type(output_device_id)})")
        
        # 🔧 AGGRESSIVE VALIDATION - DIREKT HIER:
        if input_device_id == -1:
            logger.warning(f"🚨 CONSTRUCTOR: Converting input_device_id -1 to None")
            input_device_id = None
        
        if output_device_id == -1:
            logger.warning(f"🚨 CONSTRUCTOR: Converting output_device_id -1 to None") 
            output_device_id = None
        
        # Validate device IDs
        self.input_device_id = self._validate_device_id(input_device_id, "input")
        self.output_device_id = self._validate_device_id(output_device_id, "output")
        
        # 🚨 FINAL SAFETY CHECK:
        if self.input_device_id == -1:
            logger.error(f"🚨 CRITICAL: self.input_device_id is STILL -1 after validation! FORCING to None!")
            self.input_device_id = None
        
        if self.output_device_id == -1:
            logger.error(f"🚨 CRITICAL: self.output_device_id is STILL -1 after validation! FORCING to None!")
            self.output_device_id = None
        
        # Log final device selection
        logger.info(f"🎤 FINAL Input device: {self.input_device_id}")
        logger.info(f"🔊 FINAL Output device: {self.output_device_id}")
        
        # Features
        self.enable_noise_reduction = enable_noise_reduction
        self.enable_echo_cancellation = enable_echo_cancellation
        self.enable_auto_gain_control = enable_auto_gain_control
        
        # Audio backend
        self.audio_backend = None
        self.input_stream = None
        self.output_stream = None
        self.is_recording = False
        self.is_playing = False
        
        # Processing components
        self.vad = VoiceActivityDetector(sample_rate)
        self.noise_reducer = NoiseReductionEngine(sample_rate)
        self.buffer_manager = AudioBufferManager()
        
        # Threading
        self.recording_thread = None
        self.processing_thread = None
        self._stop_event = threading.Event()
        self._audio_callback = None
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'speech_frames_detected': 0,
            'average_snr_db': 0.0,
            'buffer_overflows': 0,
            'last_activity_time': 0.0
        }
        
        logger.info("🔊 Enterprise Audio Manager initialized")
        self._initialize_audio_backend()

    def _validate_device_id(self, device_id: Optional[int], device_type: str) -> Optional[int]:
        """Aggressively validate device ID"""
        logger.info(f"🚨 _validate_device_id: Validating {device_type} device_id: {device_id}")
        
        # Rule 1: None is always valid (system default)
        if device_id is None:
            logger.info(f"✅ {device_type} device: None (system default)")
            return None
        
        # Rule 2: -1 always becomes None
        if device_id == -1:
            logger.info(f"🔄 {device_type} device: -1 → None (converted to system default)")
            return None
        
        # Rule 3: String "-1" becomes None
        if device_id == "-1":
            logger.info(f"🔄 {device_type} device: '-1' → None (string conversion)")
            return None
        
        # Rule 4: Negative numbers (except -1) are invalid
        if device_id < 0:
            logger.warning(f"⚠️ {device_type} device: {device_id} is invalid → None")
            return None
        
        # Rule 5: Very high numbers are suspicious
        if device_id > 100:
            logger.warning(f"⚠️ {device_type} device: {device_id} seems too high → None")
            return None
        
        # If we get here, device_id should be valid
        logger.info(f"✅ {device_type} device: {device_id} (explicit)")
        return device_id
    
    def _initialize_audio_backend(self):
        """Initialize audio backend (PyAudio or SoundDevice)"""
        try:
            if PYAUDIO_AVAILABLE:
                self.audio_backend = pyaudio.PyAudio()
                self._backend_type = "pyaudio"
                logger.info("✅ PyAudio backend initialized")
                
                # 🔧 TEST: Verify input device is valid
                if self.input_device_id is not None:
                    try:
                        device_info = self.audio_backend.get_device_info_by_index(self.input_device_id)
                        if device_info['maxInputChannels'] == 0:
                            logger.warning(f"⚠️ Device {self.input_device_id} has no input channels, using default")
                            self.input_device_id = None
                        else:
                            logger.info(f"✅ Input device {self.input_device_id} validated: {device_info['name']}")
                    except Exception as e:
                        logger.warning(f"⚠️ Invalid input device {self.input_device_id}: {e}, using default")
                        self.input_device_id = None
                
                # 🔧 TEST: Verify output device is valid
                if self.output_device_id is not None:
                    try:
                        device_info = self.audio_backend.get_device_info_by_index(self.output_device_id)
                        if device_info['maxOutputChannels'] == 0:
                            logger.warning(f"⚠️ Device {self.output_device_id} has no output channels, using default")
                            self.output_device_id = None
                        else:
                            logger.info(f"✅ Output device {self.output_device_id} validated: {device_info['name']}")
                    except Exception as e:
                        logger.warning(f"⚠️ Invalid output device {self.output_device_id}: {e}, using default")
                        self.output_device_id = None
                        
            elif SOUNDDEVICE_AVAILABLE:
                self._backend_type = "sounddevice"
                logger.info("✅ SoundDevice backend initialized")
                
                # 🔧 TEST: Verify devices for SoundDevice
                if self.input_device_id is not None or self.output_device_id is not None:
                    try:
                        devices = sd.query_devices()
                        
                        if self.input_device_id is not None:
                            if self.input_device_id >= len(devices):
                                logger.warning(f"⚠️ Invalid input device {self.input_device_id}, using default")
                                self.input_device_id = None
                            elif devices[self.input_device_id]['max_input_channels'] == 0:
                                logger.warning(f"⚠️ Device {self.input_device_id} has no input channels, using default")
                                self.input_device_id = None
                        
                        if self.output_device_id is not None:
                            if self.output_device_id >= len(devices):
                                logger.warning(f"⚠️ Invalid output device {self.output_device_id}, using default")
                                self.output_device_id = None
                            elif devices[self.output_device_id]['max_output_channels'] == 0:
                                logger.warning(f"⚠️ Device {self.output_device_id} has no output channels, using default")
                                self.output_device_id = None
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Device validation failed: {e}, using defaults")
                        self.input_device_id = None
                        self.output_device_id = None
            else:
                raise Exception("No audio backend available (install pyaudio or sounddevice)")
                
            # Get device information
            self.input_devices, self.output_devices = self._get_audio_devices()
            logger.info(f"📱 Found {len(self.input_devices)} input devices, {len(self.output_devices)} output devices")
            
        except Exception as e:
            logger.error(f"Audio backend initialization failed: {e}")
            self.audio_backend = None
    
    def _get_audio_devices(self) -> Tuple[List[AudioDeviceInfo], List[AudioDeviceInfo]]:
        """Get available audio devices"""
        input_devices = []
        output_devices = []
        
        try:
            if self._backend_type == "pyaudio" and self.audio_backend:
                device_count = self.audio_backend.get_device_count()
                default_input = self.audio_backend.get_default_input_device_info()
                default_output = self.audio_backend.get_default_output_device_info()
                
                for i in range(device_count):
                    try:
                        info = self.audio_backend.get_device_info_by_index(i)
                        device_info = AudioDeviceInfo(
                            index=i,
                            name=info['name'],
                            max_input_channels=info['maxInputChannels'],
                            max_output_channels=info['maxOutputChannels'],
                            default_sample_rate=info['defaultSampleRate'],
                            is_default_input=(i == default_input['index']),
                            is_default_output=(i == default_output['index'])
                        )
                        
                        if device_info.max_input_channels > 0:
                            input_devices.append(device_info)
                        if device_info.max_output_channels > 0:
                            output_devices.append(device_info)
                            
                    except Exception as e:
                        logger.debug(f"Error getting device {i}: {e}")
            
            elif self._backend_type == "sounddevice":
                devices = sd.query_devices()
                for i, device in enumerate(devices):
                    device_info = AudioDeviceInfo(
                        index=i,
                        name=device['name'],
                        max_input_channels=device['max_input_channels'],
                        max_output_channels=device['max_output_channels'],
                        default_sample_rate=device['default_samplerate'],
                        is_default_input=(i == sd.default.device[0]),
                        is_default_output=(i == sd.default.device[1])
                    )
                    
                    if device_info.max_input_channels > 0:
                        input_devices.append(device_info)
                    if device_info.max_output_channels > 0:
                        output_devices.append(device_info)
        
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
        
        return input_devices, output_devices
    
    def start_recording(self, callback: Optional[Callable[[AudioBuffer], None]] = None) -> bool:
        """Start continuous audio recording - WITH WSL SUPPORT"""
        try:
            # 🌉 WSL MODE
            if self.use_wsl_bridge and self.wsl_backend:
                logger.info("🌉 Starting WSL backend recording...")
                
                # Check WSL backend connection
                if not self.wsl_backend.connected:
                    logger.error("❌ WSL backend not connected!")
                    logger.error("💡 Make sure Windows Audio Bridge is running:")
                    logger.error("   1. On Windows: cd C:\\Users\\LeonT\\Desktop\\Kira_Home")
                    logger.error("   2. python voice_bridge.py")
                    return False
                
                # Start WSL recording with wrapped callback
                success = self.wsl_backend.start_recording(self._wsl_callback_wrapper(callback))
                
                if success:
                    self.is_recording = True
                    logger.info("✅ WSL recording started successfully")
                else:
                    logger.error("❌ WSL recording start failed")
                
                return success
            
            # 🖥️ NATIVE MODE (existing code enhanced)
            if self.is_recording:
                logger.warning("Already recording")
                return True
            
            if not self.audio_backend and self._backend_type != "sounddevice":
                logger.error("No audio backend available")
                return False

            self._audio_callback = callback
            self._stop_event.clear()
            
            # 🔧 BULLETPROOF DEVICE VALIDATION:
            final_input_device = self._get_safe_input_device()
            
            logger.info(f"🎤 STARTING RECORDING:")
            logger.info(f"   Backend: {self._backend_type}")
            logger.info(f"   Input device: {final_input_device}")
            logger.info(f"   Sample rate: {self.sample_rate}")
            logger.info(f"   Channels: {self.channels}")
            logger.info(f"   Chunk size: {self.chunk_size}")
            
            # Start recording stream based on backend
            if self._backend_type == "pyaudio":
                success = self._start_pyaudio_recording(final_input_device)
            elif self._backend_type == "sounddevice":
                success = self._start_sounddevice_recording(final_input_device)
            else:
                logger.error(f"Unknown backend type: {self._backend_type}")
                return False
            
            if not success:
                logger.error("Failed to start audio stream")
                return False
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._audio_processing_loop,
                name="AudioProcessing",
                daemon=True
            )
            self.processing_thread.start()
            
            self.is_recording = True
            logger.info("✅ Audio recording started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
        
    def _wsl_callback_wrapper(self, user_callback):
        """Wrapper to adapt WSL callback to AudioBuffer format"""
        def wrapper(audio_data, frames, timestamp, status):
            try:
                if status:
                    logger.warning(f"⚠️ WSL audio status: {status}")
                
                # Convert to AudioBuffer format
                buffer = AudioBuffer(
                    data=audio_data,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    timestamp=timestamp,
                    duration=len(audio_data) / self.sample_rate
                )
                
                # Process with existing audio processing pipeline
                processed_buffer = self._process_audio_buffer(buffer)
                
                # Update statistics
                self.stats['total_frames_processed'] += 1
                if processed_buffer.is_speech:
                    self.stats['speech_frames_detected'] += 1
                    self.stats['last_activity_time'] = processed_buffer.timestamp
                
                # Update average SNR
                if processed_buffer.snr_db > 0:
                    current_avg = self.stats['average_snr_db']
                    total_frames = self.stats['total_frames_processed']
                    self.stats['average_snr_db'] = (current_avg * (total_frames - 1) + processed_buffer.snr_db) / total_frames
                
                # Call user callback
                if user_callback:
                    user_callback(processed_buffer)
                    
            except Exception as e:
                logger.error(f"❌ WSL callback wrapper error: {e}")
        
        return wrapper
        
    def _get_safe_input_device(self) -> Optional[int]:
        """Get safe input device ID - BULLETPROOF VERSION"""
        # Final validation before use
        device_id = self.input_device_id
        
        logger.info(f"🚨 DEBUG _get_safe_input_device: Starting with device_id: {device_id}")
        
        # AGGRESSIVE -1 CONVERSION:
        if device_id == -1:
            logger.info("🔄 Converting input device -1 to None")
            return None
        
        # Convert any negative number to None
        if device_id is not None and device_id < 0:
            logger.warning(f"🔄 Converting negative device {device_id} to None")
            return None
        
        # Convert string "-1" to None (just in case)
        if device_id == "-1":
            logger.warning(f"🔄 Converting string '-1' to None")
            return None
        
        # Validate device exists if SoundDevice
        if self._backend_type == "sounddevice" and device_id is not None:
            try:
                devices = sd.query_devices()
                if device_id >= len(devices):
                    logger.warning(f"⚠️ Input device {device_id} doesn't exist, using None")
                    return None
                
                device_info = devices[device_id]
                if device_info['max_input_channels'] == 0:
                    logger.warning(f"⚠️ Input device {device_id} has no input channels, using None")
                    return None
                
                logger.info(f"✅ Input device {device_id} validated: {device_info['name']}")
                return device_id
                
            except Exception as e:
                logger.warning(f"⚠️ Input device validation failed: {e}, using None")
                return None
        
        # 🔧 FIX: If we get here, device_id should be safe to return
        # BUT: One more safety check!
        if device_id == -1:
            logger.error(f"🚨 CRITICAL: device_id is STILL -1 at end! Forcing to None!")
            return None
        
        logger.info(f"🚨 DEBUG _get_safe_input_device: Returning: {device_id}")
        return device_id
    
    def _get_safe_output_device(self) -> Optional[int]:
        """Get safe output device ID"""
        # Final validation before use
        device_id = self.output_device_id
        
        # Always convert -1 to None
        if device_id == -1:
            logger.info("🔄 Converting output device -1 to None")
            return None
        
        # Validate device exists if SoundDevice
        if self._backend_type == "sounddevice" and device_id is not None:
            try:
                devices = sd.query_devices()
                if device_id >= len(devices):
                    logger.warning(f"⚠️ Output device {device_id} doesn't exist, using None")
                    return None
                
                device_info = devices[device_id]
                if device_info['max_output_channels'] == 0:
                    logger.warning(f"⚠️ Output device {device_id} has no output channels, using None")
                    return None
                
                logger.info(f"✅ Output device {device_id} validated: {device_info['name']}")
                return device_id
                
            except Exception as e:
                logger.warning(f"⚠️ Output device validation failed: {e}, using None")
                return None
        
        return device_id

    def _start_pyaudio_recording(self, input_device: Optional[int]) -> bool:
        """Start PyAudio recording stream"""
        try:
            stream_kwargs = {
                'format': pyaudio.paFloat32,
                'channels': self.channels,
                'rate': self.sample_rate,
                'input': True,
                'frames_per_buffer': self.chunk_size,
                'stream_callback': self._pyaudio_callback
            }
            
            # Only add device_index if not None
            if input_device is not None:
                stream_kwargs['input_device_index'] = input_device
                logger.info(f"🔧 PyAudio using explicit device: {input_device}")
            else:
                logger.info("🔧 PyAudio using system default device")
            
            self.input_stream = self.audio_backend.open(**stream_kwargs)
            self.input_stream.start_stream()
            return True
            
        except Exception as e:
            logger.error(f"PyAudio recording start failed: {e}")
            return False

    def _start_sounddevice_recording(self, input_device: Optional[int]) -> bool:
        """Start SoundDevice recording stream - FINAL WORKING VERSION"""
        try:
            # 🚨 EMERGENCY DEBUG:
            logger.info(f"🚨 DEBUG: Received input_device parameter: {input_device}")
            
            # 🔧 WENN None, DANN FINDE EXPLIZIT EIN WORKING DEVICE:
            if input_device is None:
                logger.info("🔧 input_device is None, finding working device...")
                working_device = self._find_working_input_device()
                if working_device is not None:
                    input_device = working_device
                    logger.info(f"🔧 Using working device: {input_device}")
                else:
                    logger.error("❌ No working input device found!")
                    return False
            
            # 🚨 FINAL SAFETY CHECK:
            if input_device == -1:
                logger.error(f"🚨 CRITICAL: Device is STILL -1! Cannot proceed!")
                return False
            
            logger.info(f"🎤 Creating InputStream with device: {input_device}")
            
            # 🔧 EXPLICIT DEVICE VALIDATION BEFORE CREATION:
            try:
                if input_device is not None:
                    # Test if device works
                    sd.check_input_settings(
                        device=input_device,
                        channels=self.channels,
                        samplerate=self.sample_rate
                    )
                    logger.info(f"✅ Device {input_device} pre-validation PASSED")
            except Exception as e:
                logger.error(f"❌ Device {input_device} pre-validation FAILED: {e}")
                return False
            
            # CREATE STREAM:
            self.input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=input_device,  # Should be valid device ID or None
                callback=self._sounddevice_callback,
                blocksize=self.chunk_size,
                dtype=np.float32
            )
            self.input_stream.start()
            
            logger.info("✅ SoundDevice recording started successfully")
            return True
            
        except Exception as e:
            logger.error(f"SoundDevice recording start failed: {e}")
            return False
        
    def _find_working_input_device(self) -> Optional[int]:
        """Find a working input device"""
        try:
            devices = sd.query_devices()
            logger.info(f"🔍 Searching {len(devices)} devices for working input...")
            
            # 🔧 FIRST: Try to get system default
            try:
                default_input = sd.query_devices(kind='input')
                default_id = default_input.get('index', None)
                if default_id is not None:
                    logger.info(f"🔧 Found system default input: {default_id}")
                    # Test it
                    sd.check_input_settings(
                        device=default_id,
                        channels=self.channels,
                        samplerate=self.sample_rate
                    )
                    logger.info(f"✅ System default device {default_id} works!")
                    return default_id
            except Exception as e:
                logger.warning(f"⚠️ System default input failed: {e}")
            
            # 🔧 FALLBACK: Find any working input device
            input_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append(i)
                    logger.info(f"   📱 Device {i}: {device['name']} ({device['max_input_channels']} channels)")
            
            if not input_devices:
                logger.error("❌ No input devices found!")
                return None
            
            # Test each input device
            for device_id in input_devices:
                try:
                    logger.info(f"🧪 Testing device {device_id}...")
                    
                    sd.check_input_settings(
                        device=device_id,
                        channels=self.channels,
                        samplerate=self.sample_rate
                    )
                    
                    logger.info(f"✅ Device {device_id} works!")
                    return device_id
                    
                except Exception as e:
                    logger.warning(f"⚠️ Device {device_id} test failed: {e}")
                    continue
            
            logger.error("❌ No working input devices found!")
            return None
            
        except Exception as e:
            logger.error(f"❌ Device search failed: {e}")
            return None
    
    def stop_recording(self) -> bool:
        """Stop audio recording - ENHANCED WITH WSL SUPPORT"""
        try:
            logger.info("🛑 Stopping audio recording...")
            
            # 🌉 WSL MODE
            if self.use_wsl_bridge and self.wsl_backend:
                logger.info("🌉 Stopping WSL backend recording...")
                
                success = self.wsl_backend.stop_recording()
                
                if success:
                    self.is_recording = False
                    logger.info("✅ WSL recording stopped successfully")
                else:
                    logger.error("❌ WSL recording stop failed")
                
                return success
            
            # 🖥️ NATIVE MODE
            self.is_recording = False
            self._stop_event.set()
            
            # Stop streams
            if self.input_stream:
                try:
                    if self._backend_type == "pyaudio":
                        self.input_stream.stop_stream()
                        self.input_stream.close()
                    elif self._backend_type == "sounddevice":
                        self.input_stream.stop()
                        self.input_stream.close()
                    
                    logger.info("✅ Audio stream stopped")
                except Exception as e:
                    logger.warning(f"⚠️ Stream stop error: {e}")
                
                self.input_stream = None
            
            # Wait for processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                logger.info("⏳ Waiting for processing thread...")
                self.processing_thread.join(timeout=2.0)
                
                if self.processing_thread.is_alive():
                    logger.warning("⚠️ Processing thread did not stop gracefully")
                else:
                    logger.info("✅ Processing thread stopped")
            
            logger.info("🔇 Audio recording stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop recording: {e}")
            return False
    
    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio input"""
        try:
            if status:
                logger.warning(f"PyAudio callback status: {status}")
            
            # Convert to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Create audio buffer
            buffer = AudioBuffer(
                data=audio_data,
                sample_rate=self.sample_rate,
                channels=self.channels,
                timestamp=time.time(),
                duration=len(audio_data) / self.sample_rate
            )
            
            # Add to buffer manager
            self.buffer_manager.add_buffer(buffer)
            
            return (None, pyaudio.paContinue)
            
        except Exception as e:
            logger.error(f"PyAudio callback error: {e}")
            return (None, pyaudio.paAbort)
    
    def _sounddevice_callback(self, indata, frames, time, status):
        """SoundDevice callback for audio input"""
        try:
            if status:
                logger.warning(f"SoundDevice callback status: {status}")
            
            # Convert to float32 if needed
            audio_data = indata.flatten().astype(np.float32)
            
            # Create audio buffer
            buffer = AudioBuffer(
                data=audio_data,
                sample_rate=self.sample_rate,
                channels=self.channels,
                timestamp=time.inputBufferAdcTime,
                duration=len(audio_data) / self.sample_rate
            )
            
            # Add to buffer manager
            self.buffer_manager.add_buffer(buffer)
            
        except Exception as e:
            logger.error(f"SoundDevice callback error: {e}")
    
    def _audio_processing_loop(self):
        """Background audio processing loop"""
        logger.info("🔄 Audio processing loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get audio buffer
                buffer = self.buffer_manager.get_buffer(timeout=0.1)
                if not buffer:
                    continue
                
                # Process audio
                processed_buffer = self._process_audio_buffer(buffer)
                
                # Update statistics
                self.stats['total_frames_processed'] += 1
                if processed_buffer.is_speech:
                    self.stats['speech_frames_detected'] += 1
                    self.stats['last_activity_time'] = processed_buffer.timestamp
                
                # Call user callback
                if self._audio_callback and processed_buffer:
                    try:
                        self._audio_callback(processed_buffer)
                    except Exception as e:
                        logger.error(f"Audio callback error: {e}")
                
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                time.sleep(0.01)
        
        logger.info("🛑 Audio processing loop stopped")
    
    def _process_audio_buffer(self, buffer: AudioBuffer) -> AudioBuffer:
        """Process audio buffer with enhancements"""
        try:
            processed_data = buffer.data.copy()
            
            # Noise reduction
            if self.enable_noise_reduction:
                processed_data = self.noise_reducer.reduce_noise(processed_data)
            
            # Voice activity detection
            is_speech, confidence = self.vad.is_speech(processed_data)
            
            # Estimate SNR
            snr_db = self.noise_reducer.estimate_snr(processed_data)
            
            # Calculate energy level
            energy_level = np.sqrt(np.mean(processed_data ** 2))
            
            # Create processed buffer
            processed_buffer = AudioBuffer(
                data=processed_data,
                sample_rate=buffer.sample_rate,
                channels=buffer.channels,
                timestamp=buffer.timestamp,
                duration=buffer.duration,
                is_speech=is_speech,
                energy_level=energy_level,
                snr_db=snr_db
            )
            
            return processed_buffer
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return buffer
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get audio chunk for wake word detection - WSL COMPATIBLE"""
        try:
            # 🌉 WSL MODE
            if self.use_wsl_bridge and self.wsl_backend:
                # Get audio from WSL backend
                audio_data = self.wsl_backend.get_recent_audio(duration=0.1)  # Get 100ms of audio
                
                if audio_data is not None:
                    return audio_data
                else:
                    # Fallback: wait a bit and try again
                    time.sleep(timeout)
                    return self.wsl_backend.get_recent_audio(duration=0.1)
            
            # 🖥️ NATIVE MODE
            buffer = self.buffer_manager.get_buffer(timeout=timeout)
            if buffer:
                return buffer.data
            return None
            
        except Exception as e:
            logger.error(f"❌ Get audio chunk error: {e}")
            return None
    
    def record_command(self, duration: float = 5.0, silence_threshold: float = 0.01) -> Optional[np.ndarray]:
        """Record audio command with silence detection - WSL COMPATIBLE"""
        try:
            logger.info(f"🎙️ Recording command for {duration}s...")
            
            # 🌉 WSL MODE - Use WSL backend directly
            if self.use_wsl_bridge and self.wsl_backend:
                logger.info("🌉 Recording command through WSL backend...")
                
                audio_chunks = []
                start_time = time.time()
                silence_start = None
                max_silence_duration = 2.0
                
                # Temporary callback to collect audio
                def collect_audio(audio_data, frames, timestamp, status):
                    nonlocal silence_start
                    
                    # Calculate energy
                    energy = np.sqrt(np.mean(audio_data**2))
                    
                    audio_chunks.append(audio_data.copy())
                    
                    # Check for silence
                    if energy < silence_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > max_silence_duration:
                            logger.info("🔇 Silence detected, ending WSL recording")
                            return  # This won't stop the recording but signals end
                    else:
                        silence_start = None
                
                # Start temporary recording
                if not self.wsl_backend.start_recording(collect_audio):
                    logger.error("❌ Failed to start WSL command recording")
                    return None
                
                # Wait for recording duration
                while time.time() - start_time < duration:
                    time.sleep(0.1)
                    
                    # Check for silence break
                    if silence_start and time.time() - silence_start > max_silence_duration:
                        break
                
                # Stop recording
                self.wsl_backend.stop_recording()
                
                if audio_chunks:
                    command_audio = np.concatenate(audio_chunks)
                    logger.info(f"✅ WSL Command recorded: {len(command_audio)/self.sample_rate:.2f}s")
                    return command_audio
                else:
                    logger.warning("❌ No WSL audio recorded")
                    return None
            
            # 🖥️ NATIVE MODE (existing implementation)
            start_time = time.time()
            audio_chunks = []
            silence_start = None
            max_silence_duration = 2.0  # Max 2 seconds of silence
            
            while time.time() - start_time < duration:
                buffer = self.buffer_manager.get_buffer(timeout=0.1)
                if buffer:
                    audio_chunks.append(buffer.data)
                    
                    # Check for silence
                    if buffer.energy_level < silence_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > max_silence_duration:
                            logger.info("🔇 Silence detected, ending recording")
                            break
                    else:
                        silence_start = None
            
            if audio_chunks:
                command_audio = np.concatenate(audio_chunks)
                logger.info(f"✅ Command recorded: {len(command_audio)/self.sample_rate:.2f}s")
                return command_audio
            else:
                logger.warning("❌ No audio recorded")
                return None
                
        except Exception as e:
            logger.error(f"❌ Command recording failed: {e}")
            return None
    
    def play_audio(self, audio_data) -> bool:
        """Play audio data - WITH WSL SUPPORT"""
        try:
            # 🌉 WSL MODE
            if self.use_wsl_bridge and self.wsl_backend:
                logger.info("🌉 Playing audio through WSL backend...")
                
                # Convert bytes to numpy array if needed
                if isinstance(audio_data, bytes):
                    try:
                        audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    except ValueError:
                        # Try different data types
                        try:
                            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
                        except ValueError:
                            logger.error("❌ Cannot convert audio data to numpy array")
                            return False
                elif isinstance(audio_data, np.ndarray):
                    audio_array = audio_data.astype(np.float32)
                else:
                    logger.error(f"❌ Unsupported audio data type: {type(audio_data)}")
                    return False
                
                return self.wsl_backend.play_audio(audio_array)
            
            # 🖥️ NATIVE MODE (existing code)
            if self.is_playing:
                logger.warning("Already playing audio")
                return False
            
            self.is_playing = True  # ✅ WICHTIG: Status setzen!
            
            try:
                if self._backend_type == "pyaudio":
                    success = self._play_audio_pyaudio(audio_data)
                elif self._backend_type == "sounddevice":
                    success = self._play_audio_sounddevice(audio_data)
                else:
                    logger.error(f"❌ Unknown backend type: {self._backend_type}")
                    success = False
                
                return success
                
            finally:
                self.is_playing = False  # ✅ WICHTIG: Status zurücksetzen!
            
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            self.is_playing = False
            return False
        
    def _play_audio_pyaudio(self, audio_data) -> bool:
        """Play audio using PyAudio backend"""
        try:
            # Convert to numpy array
            if isinstance(audio_data, bytes):
                try:
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                except ValueError:
                    # Try int16 and convert
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            else:
                audio_array = np.array(audio_data, dtype=np.float32)
            
            # Get safe output device
            output_device = self._get_safe_output_device()
            
            # Create output stream
            stream_kwargs = {
                'format': pyaudio.paFloat32,
                'channels': self.channels,
                'rate': self.sample_rate,
                'output': True,
                'frames_per_buffer': self.chunk_size
            }
            
            if output_device is not None:
                stream_kwargs['output_device_index'] = output_device
                logger.info(f"🔊 PyAudio using device: {output_device}")
            else:
                logger.info("🔊 PyAudio using system default output")
            
            output_stream = self.audio_backend.open(**stream_kwargs)
            
            try:
                # Convert to bytes and write
                audio_bytes = audio_array.tobytes()
                output_stream.write(audio_bytes)
                
                logger.info("✅ PyAudio playback completed")
                return True
                
            finally:
                output_stream.stop_stream()
                output_stream.close()
            
        except Exception as e:
            logger.error(f"❌ PyAudio playback failed: {e}")
            return False

    def _play_audio_sounddevice(self, audio_data) -> bool:
        """Play audio using SoundDevice backend"""
        try:
            # Convert to numpy array
            if isinstance(audio_data, bytes):
                try:
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                except ValueError:
                    # Try int16 and convert
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            else:
                audio_array = np.array(audio_data, dtype=np.float32)
            
            # Reshape for channels if needed
            if self.channels > 1 and len(audio_array.shape) == 1:
                audio_array = audio_array.reshape(-1, self.channels)
            
            # Get safe output device
            output_device = self._get_safe_output_device()
            
            # Validate output device for SoundDevice
            if output_device is not None:
                try:
                    devices = sd.query_devices()
                    if output_device >= len(devices) or devices[output_device]['max_output_channels'] == 0:
                        logger.warning(f"⚠️ Invalid output device {output_device}, using default")
                        output_device = None
                except Exception as e:
                    logger.warning(f"⚠️ Output device validation failed: {e}, using default")
                    output_device = None
            
            logger.info(f"🔊 SoundDevice playing on device: {output_device}")
            
            # Play audio
            sd.play(
                audio_array,
                samplerate=self.sample_rate,
                device=output_device,
                blocking=True  # Wait until playback finishes
            )
            
            logger.info("✅ SoundDevice playback completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ SoundDevice playback failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get audio manager status - ENHANCED WITH WSL INFO"""
        base_status = {
            'initialized': self.audio_backend is not None or (self.use_wsl_bridge and self.wsl_backend),
            'backend': 'wsl_bridge' if self.use_wsl_bridge else self._backend_type,
            'recording': self.is_recording,
            'playing': self.is_playing,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'features': {
                'noise_reduction': self.enable_noise_reduction and NOISE_REDUCE_AVAILABLE,
                'voice_activity_detection': WEBRTC_VAD_AVAILABLE,
                'echo_cancellation': self.enable_echo_cancellation,
                'auto_gain_control': self.enable_auto_gain_control
            },
            'statistics': self.stats.copy()
        }
        
        # 🌉 WSL-specific status
        if self.use_wsl_bridge:
            base_status['wsl_backend'] = {
                'connected': self.wsl_backend.connected if self.wsl_backend else False,
                'bridge_status': self.wsl_backend.get_status() if self.wsl_backend else None
            }
        else:
            # Native device information
            base_status['devices'] = {
                'input_devices': len(self.input_devices) if hasattr(self, 'input_devices') else 0,
                'output_devices': len(self.output_devices) if hasattr(self, 'output_devices') else 0,
                'current_input': self.input_device_id,
                'current_output': self.output_device_id
            }
        
        # Buffer status (only for native mode)
        if not self.use_wsl_bridge:
            base_status['buffer_status'] = {
                'total_duration': self.buffer_manager.total_duration,
                'overflow_count': self.buffer_manager.overflow_count
            }
        
        return base_status

    def cleanup(self):
        """Cleanup audio manager - ENHANCED WITH WSL CLEANUP"""
        try:
            logger.info("🧹 Cleaning up Enterprise Audio Manager...")
            
            # Stop recording
            self.stop_recording()
            
            # 🌉 WSL CLEANUP
            if self.use_wsl_bridge and self.wsl_backend:
                logger.info("🌉 Cleaning up WSL backend...")
                self.wsl_backend.cleanup()
            
            # 🖥️ NATIVE CLEANUP
            if self.audio_backend and self._backend_type == "pyaudio":
                self.audio_backend.terminate()
            
            logger.info("✅ Enterprise Audio Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Audio manager cleanup failed: {e}")


    def test_wsl_connection(self) -> bool:
        """Test WSL backend connection"""
        if not self.use_wsl_bridge or not self.wsl_backend:
            return False
        
        try:
            # Test connection and audio capabilities
            status = self.wsl_backend.get_status()
            
            if status['connected']:
                logger.info("✅ WSL backend connection test passed")
                return True
            else:
                logger.error("❌ WSL backend not connected")
                return False
                
        except Exception as e:
            logger.error(f"❌ WSL connection test failed: {e}")
            return False

    def switch_to_wsl_mode(self, wsl_backend) -> bool:
        """Switch from native to WSL mode (runtime switch)"""
        try:
            logger.info("🔄 Switching to WSL mode...")
            
            # Stop current recording if active
            was_recording = self.is_recording
            if was_recording:
                self.stop_recording()
            
            # Switch to WSL
            self.use_wsl_bridge = True
            self.wsl_backend = wsl_backend
            
            # Test connection
            if not self.test_wsl_connection():
                logger.error("❌ WSL mode switch failed - connection test failed")
                self.use_wsl_bridge = False
                self.wsl_backend = None
                return False
            
            # Restart recording if it was active
            if was_recording:
                success = self.start_recording(self._audio_callback)
                if not success:
                    logger.error("❌ Failed to restart recording in WSL mode")
                    return False
            
            logger.info("✅ Successfully switched to WSL mode")
            return True
            
        except Exception as e:
            logger.error(f"❌ WSL mode switch failed: {e}")
            return False

    def switch_to_native_mode(self) -> bool:
        """Switch from WSL to native mode (runtime switch)"""
        try:
            logger.info("🔄 Switching to native mode...")
            
            # Stop current recording if active
            was_recording = self.is_recording
            if was_recording:
                self.stop_recording()
            
            # Cleanup WSL backend
            if self.wsl_backend:
                self.wsl_backend.cleanup()
            
            # Switch to native
            self.use_wsl_bridge = False
            self.wsl_backend = None
            
            # Reinitialize native backend if needed
            if not self.audio_backend:
                self._initialize_audio_backend()
            
            # Restart recording if it was active
            if was_recording:
                success = self.start_recording(self._audio_callback)
                if not success:
                    logger.error("❌ Failed to restart recording in native mode")
                    return False
            
            logger.info("✅ Successfully switched to native mode")
            return True
            
        except Exception as e:
            logger.error(f"❌ Native mode switch failed: {e}")
            return False

# Export classes
__all__ = [
    'EnterpriseAudioManager',
    'AudioBuffer',
    'AudioDeviceInfo',
    'VoiceActivityDetector',
    'NoiseReductionEngine',
    'AudioBufferManager'
]