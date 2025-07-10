"""
Einfacher Audio Recorder - ENHANCED WITH DEVICE DETECTION
"""

import sounddevice as sd
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AudioData:
    """Audio Daten Container"""
    data: np.ndarray
    sample_rate: int
    duration: float
    success: bool
    error: Optional[str] = None

class SimpleAudioRecorder:
    """Einfacher Audio Recorder - ENHANCED"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, remote_mode: bool = False):
        self.sample_rate = sample_rate
        self.channels = channels
        self.remote_mode = remote_mode
        self.buffer_size = 4096 if remote_mode else None
        self.audio_buffer = [] if remote_mode else None
        
        # 🔧 AUDIO DEVICE DETECTION
        self.input_device = self._detect_input_device()
        
        logger.info(f"🎤 Audio Recorder initialisiert: {sample_rate}Hz, {channels} Kanal(e)")
        logger.info(f"🎤 Input device: {self.input_device}")
    
    def _detect_input_device(self) -> Optional[int]:
        """Detect best available input device"""
        try:
            # Get all audio devices
            devices = sd.query_devices()
            logger.info(f"🔍 Available audio devices: {len(devices)}")
            
            # Find input devices
            input_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
                    logger.info(f"   📱 Device {i}: {device['name']} ({device['max_input_channels']} channels)")
            
            if not input_devices:
                logger.warning("⚠️ No input devices found!")
                return None
            
            # Try to find default input device
            try:
                default_device = sd.query_devices(kind='input')
                device_id = default_device['index'] if 'index' in default_device else None
                if device_id is not None:
                    logger.info(f"✅ Using default input device: {device_id}")
                    return device_id
            except Exception as e:
                logger.warning(f"⚠️ Could not get default input device: {e}")
            
            # Fallback: Use first available input device
            first_device = input_devices[0]
            logger.info(f"🔄 Fallback to first available device: {first_device['id']} - {first_device['name']}")
            return first_device['id']
            
        except Exception as e:
            logger.error(f"❌ Device detection failed: {e}")
            return None
    
    def record(self, duration: float) -> AudioData:
        """Nehme Audio auf - ENHANCED WITH DEVICE FALLBACK"""
        try:
            logger.info(f"🎤 Starte Aufnahme: {duration}s")
            
            # Try with detected device first
            if self.input_device is not None:
                try:
                    audio_data = sd.rec(
                        int(duration * self.sample_rate),
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype='float32',
                        device=self.input_device  # ✅ EXPLICIT DEVICE
                    )
                    sd.wait()
                    
                    # Process audio
                    audio_flat = audio_data.flatten() if self.channels == 1 else audio_data
                    rms = np.sqrt(np.mean(audio_flat**2))
                    logger.info(f"✅ Aufnahme erfolgreich: RMS={rms:.4f}")
                    
                    return AudioData(
                        data=audio_flat,
                        sample_rate=self.sample_rate,
                        duration=duration,
                        success=True
                    )
                    
                except Exception as e:
                    logger.warning(f"⚠️ Recording with device {self.input_device} failed: {e}")
            
            # Fallback: Try without explicit device
            logger.info("🔄 Trying fallback recording without explicit device...")
            
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
                # NO device parameter = use system default
            )
            sd.wait()
            
            audio_flat = audio_data.flatten() if self.channels == 1 else audio_data
            rms = np.sqrt(np.mean(audio_flat**2))
            logger.info(f"✅ Fallback aufnahme erfolgreich: RMS={rms:.4f}")
            
            return AudioData(
                data=audio_flat,
                sample_rate=self.sample_rate,
                duration=duration,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Aufnahme-Fehler: {e}")
            return AudioData(
                data=np.array([]),
                sample_rate=self.sample_rate,
                duration=0.0,
                success=False,
                error=str(e)
            )
    
    def record_until_silence(self, max_duration: float = 10.0, silence_threshold: float = 0.01, silence_duration: float = 2.0) -> AudioData:
        """Nehme auf bis Stille erkannt wird - ENHANCED"""
        try:
            logger.info(f"🎤 Starte Aufnahme bis Stille (max {max_duration}s)")
            
            chunk_size = int(0.1 * self.sample_rate)  # 100ms chunks
            audio_chunks = []
            silence_counter = 0
            silence_chunks = int(silence_duration / 0.1)
            total_chunks = 0
            max_chunks = int(max_duration / 0.1)
            
            # Enhanced InputStream with device selection
            stream_kwargs = {
                'samplerate': self.sample_rate,
                'channels': self.channels,
                'dtype': 'float32'
            }
            
            if self.input_device is not None:
                stream_kwargs['device'] = self.input_device
            
            with sd.InputStream(**stream_kwargs) as stream:
                while total_chunks < max_chunks:
                    chunk, overflowed = stream.read(chunk_size)
                    
                    if overflowed:
                        logger.warning("⚠️ Audio Overflow")
                    
                    chunk_flat = chunk.flatten()
                    audio_chunks.append(chunk_flat)
                    
                    # Prüfe auf Stille
                    rms = np.sqrt(np.mean(chunk_flat**2))
                    
                    if rms < silence_threshold:
                        silence_counter += 1
                    else:
                        silence_counter = 0
                    
                    # Stoppe bei Stille
                    if silence_counter >= silence_chunks:
                        logger.info("🔇 Stille erkannt - stoppe Aufnahme")
                        break
                    
                    total_chunks += 1
            
            # Kombiniere Chunks
            full_audio = np.concatenate(audio_chunks)
            actual_duration = len(full_audio) / self.sample_rate
            
            logger.info(f"✅ Aufnahme bis Stille erfolgreich: {actual_duration:.1f}s")
            
            return AudioData(
                data=full_audio,
                sample_rate=self.sample_rate,
                duration=actual_duration,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Aufnahme bis Stille Fehler: {e}")
            return AudioData(
                data=np.array([]),
                sample_rate=self.sample_rate,
                duration=0.0,
                success=False,
                error=str(e)
            )
    
    def test_audio_devices(self) -> Dict:
        """Test verfügbare Audio Devices"""
        try:
            devices = sd.query_devices()
            default_input = sd.query_devices(kind='input')
            default_output = sd.query_devices(kind='output')
            
            result = {
                'total_devices': len(devices),
                'input_devices': [],
                'output_devices': [],
                'default_input': default_input,
                'default_output': default_output
            }
            
            for i, device in enumerate(devices):
                device_info = {
                    'id': i,
                    'name': device['name'],
                    'input_channels': device['max_input_channels'],
                    'output_channels': device['max_output_channels'],
                    'default_samplerate': device['default_samplerate']
                }
                
                if device['max_input_channels'] > 0:
                    result['input_devices'].append(device_info)
                
                if device['max_output_channels'] > 0:
                    result['output_devices'].append(device_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Audio device test failed: {e}")
            return {'error': str(e)}

# Export
__all__ = ['SimpleAudioRecorder', 'AudioData']