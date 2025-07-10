"""
🌉 WSL AUDIO BACKEND für Kira Voice System
Ersetzt SoundDevice/PyAudio mit Windows Audio Bridge
"""

import numpy as np
import time
import threading
import queue
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from wsl_client import WSLVoiceClient

logger = logging.getLogger(__name__)

@dataclass
class AudioBuffer:
    """Audio Buffer Data Structure"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    channels: int
    duration: float

class WSLAudioBackend:
    """WSL Audio Backend für Kira Voice System"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        bridge_host: str = None,
        bridge_port: int = 7777,
        wsl_client=None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

        
        
        # WSL Client
        if wsl_client is not None:
            self.wsl_client = wsl_client
            self.connected = self.wsl_client.is_connected()
            logger.info("🌉 WSL Audio Backend using provided connected client")
        else:
            # ✅ FIX: Import path correction
            import sys
            import os
            
            # Add project root to path
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            from wsl_client import WSLVoiceClient
            
            # Create new WSL Client
            self.wsl_client = WSLVoiceClient(
                bridge_host=bridge_host,
                bridge_port=bridge_port
            )
            self.connected = False
            logger.info("🌉 WSL Audio Backend created new client")
        
        
        # Audio Streams
        self.is_recording = False
        self.is_playing = False
        
        # Buffers
        self.audio_buffers = queue.Queue(maxsize=100)
        self.playback_queue = queue.Queue(maxsize=50)
        
        # Callbacks
        self.audio_callback = None
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'buffer_overflows': 0,
            'average_latency': 0.0,
            'last_activity_time': 0.0
        }
        
        logger.info("🌉 WSL Audio Backend initialized")
    
    def initialize(self) -> bool:
        """Initialize WSL backend with existing client"""
        try:
            if not self.wsl_client:
                logger.error("❌ No WSL client provided to backend")
                return False
            
            if not self.wsl_client.is_connected():
                logger.error("❌ WSL client is not connected")
                return False
            
            self.initialized = True
            logger.info("✅ WSL Backend: Using existing connected client")
            return True
                
        except Exception as e:
            logger.error(f"❌ WSL Backend initialization failed: {e}")
            return False
    
    def start_recording(self, callback: Optional[Callable] = None) -> bool:
        """Start audio recording with callback"""
        if not self.connected:
            logger.error("❌ WSL Client not connected")
            return False
        
        try:
            self.audio_callback = callback
            self.is_recording = True
            
            # Start recording thread
            threading.Thread(target=self._recording_loop, daemon=True).start()
            
            logger.info("✅ WSL Audio recording started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start WSL recording: {e}")
            return False
    
    def _recording_loop(self):
        """Main recording loop"""
        logger.info("🎤 WSL Recording loop started")
        
        while self.is_recording and self.connected:
            try:
                # Get audio data from Windows
                audio_data = self.wsl_client.get_microphone_data(timeout=0.1)
                
                if audio_data is not None:
                    # Create audio buffer
                    buffer = AudioBuffer(
                        data=audio_data,
                        timestamp=time.time(),
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                        duration=len(audio_data) / self.sample_rate
                    )
                    
                    # Add to buffer queue
                    try:
                        self.audio_buffers.put_nowait(buffer)
                        self.stats['total_frames_processed'] += 1
                        self.stats['last_activity_time'] = time.time()
                    except queue.Full:
                        self.stats['buffer_overflows'] += 1
                        logger.warning("⚠️ Audio buffer overflow")
                    
                    # Call callback if provided
                    if self.audio_callback:
                        try:
                            self.audio_callback(audio_data, len(audio_data), time.time(), None)
                        except Exception as e:
                            logger.error(f"❌ Audio callback error: {e}")
                
            except Exception as e:
                logger.error(f"❌ Recording loop error: {e}")
                break
        
        logger.info("🛑 WSL Recording loop stopped")
    
    def stop_recording(self) -> bool:
        """Stop audio recording"""
        try:
            self.is_recording = False
            logger.info("🛑 WSL Audio recording stopped")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to stop WSL recording: {e}")
            return False
        
    def record_audio(self, duration: float = None) -> np.ndarray:
        """Record audio through WSL client"""
        try:
            if not self.initialized or not self.wsl_client:
                return None
            
            # Get microphone data from Windows through client
            audio_data = self.wsl_client.get_microphone_data(timeout=0.1)
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ WSL record audio error: {e}")
            return None
    
    def play_audio(self, audio_data: np.ndarray) -> bool:
        """Play audio through Windows speakers"""
        if not self.connected:
            logger.error("❌ WSL Client not connected")
            return False
        
        try:
            self.wsl_client.play_audio(audio_data)
            logger.debug("🔊 Audio played through WSL bridge")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to play audio: {e}")
            return False
    
    def get_status(self) -> dict:
        """Get backend status"""
        status = {
            "initialized": self.initialized,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size,
            "has_client": self.wsl_client is not None
        }
        
        if self.wsl_client:
            client_status = self.wsl_client.get_status()
            status.update({
                "client_connected": client_status.get('connected', False),
                "client_stable": client_status.get('connection_stable', False),
                "bridge_host": client_status.get('bridge_host'),
                "bridge_port": client_status.get('bridge_port')
            })
        
        return status
    
    def get_recent_audio(self, duration: float = 5.0) -> Optional[np.ndarray]:
        """Get recent audio data"""
        try:
            recent_buffers = []
            collected_duration = 0.0
            
            # Collect recent buffers
            temp_buffers = []
            while not self.audio_buffers.empty() and collected_duration < duration:
                buffer = self.audio_buffers.get_nowait()
                temp_buffers.append(buffer)
                recent_buffers.append(buffer.data)
                collected_duration += buffer.duration
            
            # Put buffers back
            for buffer in temp_buffers:
                try:
                    self.audio_buffers.put_nowait(buffer)
                except queue.Full:
                    break
            
            if recent_buffers:
                return np.concatenate(recent_buffers)
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Get recent audio error: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get backend status"""
        return {
            'backend_type': 'wsl_bridge',
            'connected': self.connected,
            'recording': self.is_recording,
            'playing': self.is_playing,
            'buffer_size': self.audio_buffers.qsize(),
            'statistics': self.stats.copy(),
            'audio_config': {
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'chunk_size': self.chunk_size
            }
        }
    
    def cleanup(self):
        """Cleanup WSL Audio Backend"""
        try:
            self.stop_recording()
            
            if self.wsl_client:
                self.wsl_client.disconnect()
            
            logger.info("🧹 WSL Audio Backend cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")