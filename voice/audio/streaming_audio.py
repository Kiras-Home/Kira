"""
🎵 REAL-TIME AUDIO STREAMING
Enterprise-grade real-time audio streaming for continuous voice processing
"""

import logging
import asyncio
import threading
import queue
import time
import numpy as np
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import pyaudio
import webrtcvad
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """Audio stream chunk"""
    data: np.ndarray
    timestamp: float
    chunk_id: int
    sample_rate: int
    energy_level: float
    is_speech: bool = False
    vad_confidence: float = 0.0
    processing_latency: float = 0.0


@dataclass
class StreamConfig:
    """Streaming configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: int = pyaudio.paInt16
    
    # VAD settings
    vad_mode: int = 3  # 0-3, 3 = most aggressive
    vad_frame_duration: int = 30  # ms (10, 20, or 30)
    
    # Buffer settings
    buffer_size: int = 50  # chunks
    overlap_chunks: int = 2
    
    # Performance settings
    max_latency_ms: float = 100.0
    enable_noise_suppression: bool = True
    enable_automatic_gain: bool = True


class AudioStreamProcessor:
    """
    🎵 REAL-TIME AUDIO STREAM PROCESSOR
    Processes continuous audio stream with VAD and noise reduction
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.is_streaming = False
        self.chunk_counter = 0
        
        # Audio components
        self.pyaudio = None
        self.stream = None
        self.vad = None
        
        # Buffers
        self.audio_buffer = deque(maxlen=config.buffer_size)
        self.speech_buffer = deque(maxlen=20)  # Last 20 speech chunks
        
        # Processing
        self.processing_queue = asyncio.Queue(maxsize=100)
        self.result_callbacks = []
        
        # Statistics
        self.stats = {
            'chunks_processed': 0,
            'speech_chunks': 0,
            'total_audio_time': 0.0,
            'average_latency': 0.0,
            'last_speech_time': None
        }
        
        logger.info("🎵 Audio Stream Processor initialized")
    
    async def initialize(self) -> bool:
        """Initialize streaming components"""
        try:
            # Initialize PyAudio
            self.pyaudio = pyaudio.PyAudio()
            
            # Initialize WebRTC VAD
            self.vad = webrtcvad.Vad(self.config.vad_mode)
            
            logger.info("✅ Audio streaming components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Streaming initialization failed: {e}")
            return False
    
    async def start_streaming(self) -> bool:
        """Start real-time audio streaming"""
        try:
            if self.is_streaming:
                logger.warning("Streaming already active")
                return True
            
            if not self.pyaudio:
                await self.initialize()
            
            # Open audio stream
            self.stream = self.pyaudio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback,
                start=False
            )
            
            # Start processing tasks
            asyncio.create_task(self._process_audio_chunks())
            
            # Start stream
            self.stream.start_stream()
            self.is_streaming = True
            
            logger.info("🎵 Real-time audio streaming started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    async def stop_streaming(self) -> bool:
        """Stop audio streaming"""
        try:
            self.is_streaming = False
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            logger.info("🛑 Audio streaming stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop streaming: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for incoming audio"""
        try:
            if not self.is_streaming:
                return (None, pyaudio.paComplete)
            
            # Convert to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate energy level
            energy = np.sqrt(np.mean(audio_data ** 2))
            
            # VAD check
            is_speech, vad_confidence = self._check_voice_activity(in_data)
            
            # Create stream chunk
            chunk = StreamChunk(
                data=audio_data,
                timestamp=time.time(),
                chunk_id=self.chunk_counter,
                sample_rate=self.config.sample_rate,
                energy_level=energy,
                is_speech=is_speech,
                vad_confidence=vad_confidence
            )
            
            self.chunk_counter += 1
            
            # Add to processing queue (non-blocking)
            try:
                self.processing_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                logger.warning("Processing queue full, dropping chunk")
            
            return (in_data, pyaudio.paContinue)
            
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
            return (None, pyaudio.paAbort)
    
    def _check_voice_activity(self, audio_data: bytes) -> tuple[bool, float]:
        """Check voice activity using WebRTC VAD"""
        try:
            if not self.vad:
                return False, 0.0
            
            # VAD requires specific frame sizes
            frame_size = int(self.config.sample_rate * self.config.vad_frame_duration / 1000)
            
            if len(audio_data) < frame_size * 2:  # 2 bytes per sample
                return False, 0.0
            
            # Take first frame for VAD
            frame = audio_data[:frame_size * 2]
            
            # Check voice activity
            is_speech = self.vad.is_speech(frame, self.config.sample_rate)
            confidence = 0.8 if is_speech else 0.2  # Simple confidence estimation
            
            return is_speech, confidence
            
        except Exception as e:
            logger.debug(f"VAD error: {e}")
            return False, 0.0
    
    async def _process_audio_chunks(self):
        """Process audio chunks from queue"""
        logger.info("🔄 Audio chunk processing started")
        
        while self.is_streaming:
            try:
                # Get chunk with timeout
                chunk = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=0.1
                )
                
                # Process chunk
                await self._handle_audio_chunk(chunk)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Chunk processing error: {e}")
        
        logger.info("🛑 Audio chunk processing stopped")
    
    async def _handle_audio_chunk(self, chunk: StreamChunk):
        """Handle individual audio chunk"""
        try:
            start_time = time.time()
            
            # Add to buffer
            self.audio_buffer.append(chunk)
            
            # If speech detected, add to speech buffer
            if chunk.is_speech:
                self.speech_buffer.append(chunk)
                self.stats['speech_chunks'] += 1
                self.stats['last_speech_time'] = chunk.timestamp
            
            # Apply noise suppression if enabled
            if self.config.enable_noise_suppression:
                chunk.data = self._apply_noise_suppression(chunk.data)
            
            # Apply automatic gain if enabled
            if self.config.enable_automatic_gain:
                chunk.data = self._apply_automatic_gain(chunk.data)
            
            # Calculate processing latency
            chunk.processing_latency = (time.time() - start_time) * 1000  # ms
            
            # Update statistics
            self._update_chunk_stats(chunk)
            
            # Notify callbacks
            await self._notify_chunk_callbacks(chunk)
            
            # Check for wake word or continuous speech
            await self._check_speech_patterns()
            
        except Exception as e:
            logger.error(f"Chunk handling error: {e}")
    
    def _apply_noise_suppression(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply simple noise suppression"""
        try:
            # Simple spectral subtraction approach
            # Calculate noise floor from recent non-speech segments
            noise_floor = 0.01  # Default noise floor
            
            # Get recent non-speech chunks for noise estimation
            recent_chunks = list(self.audio_buffer)[-10:]
            non_speech_chunks = [chunk for chunk in recent_chunks if not chunk.is_speech]
            
            if non_speech_chunks:
                noise_levels = [chunk.energy_level for chunk in non_speech_chunks]
                noise_floor = np.mean(noise_levels) * 1.5  # 1.5x noise floor
            
            # Apply noise gate
            if np.max(np.abs(audio_data)) < noise_floor:
                return audio_data * 0.1  # Reduce by 90%
            
            return audio_data
            
        except Exception as e:
            logger.debug(f"Noise suppression error: {e}")
            return audio_data
    
    def _apply_automatic_gain(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply automatic gain control"""
        try:
            # Target RMS level
            target_rms = 0.1
            current_rms = np.sqrt(np.mean(audio_data ** 2))
            
            if current_rms > 0:
                gain = target_rms / current_rms
                # Limit gain to reasonable range
                gain = np.clip(gain, 0.1, 10.0)
                return audio_data * gain
            
            return audio_data
            
        except Exception as e:
            logger.debug(f"Automatic gain error: {e}")
            return audio_data
    
    async def _check_speech_patterns(self):
        """Check for speech patterns (wake word, continuous speech)"""
        try:
            if len(self.speech_buffer) < 5:
                return
            
            # Check for continuous speech (potential wake word)
            recent_speech = list(self.speech_buffer)[-5:]
            if all(chunk.is_speech for chunk in recent_speech):
                # Potential wake word detected
                await self._handle_potential_wake_word(recent_speech)
            
            # Check for speech pause (end of utterance)
            if self._detect_speech_pause():
                await self._handle_speech_end()
                
        except Exception as e:
            logger.error(f"Speech pattern checking error: {e}")
    
    async def _handle_potential_wake_word(self, speech_chunks: List[StreamChunk]):
        """Handle potential wake word detection"""
        try:
            # Combine audio data from chunks
            combined_audio = np.concatenate([chunk.data for chunk in speech_chunks])
            
            # Notify callbacks about potential wake word
            for callback in self.result_callbacks:
                try:
                    await callback({
                        'type': 'potential_wake_word',
                        'audio_data': combined_audio,
                        'timestamp': speech_chunks[0].timestamp,
                        'duration': len(combined_audio) / self.config.sample_rate,
                        'confidence': np.mean([chunk.vad_confidence for chunk in speech_chunks])
                    })
                except Exception as e:
                    logger.error(f"Wake word callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Wake word handling error: {e}")
    
    def _detect_speech_pause(self) -> bool:
        """Detect pause in speech (end of utterance)"""
        try:
            if len(self.audio_buffer) < 10:
                return False
            
            # Check last 10 chunks for speech pause
            recent_chunks = list(self.audio_buffer)[-10:]
            non_speech_count = sum(1 for chunk in recent_chunks if not chunk.is_speech)
            
            # If more than 70% are non-speech, consider it a pause
            return non_speech_count > 7
            
        except Exception as e:
            logger.debug(f"Speech pause detection error: {e}")
            return False
    
    async def _handle_speech_end(self):
        """Handle end of speech utterance"""
        try:
            # Get all speech chunks from current utterance
            speech_chunks = [chunk for chunk in self.speech_buffer if chunk.is_speech]
            
            if len(speech_chunks) < 3:  # Too short to be meaningful
                return
            
            # Combine audio data
            combined_audio = np.concatenate([chunk.data for chunk in speech_chunks])
            
            # Notify callbacks about complete utterance
            for callback in self.result_callbacks:
                try:
                    await callback({
                        'type': 'speech_utterance',
                        'audio_data': combined_audio,
                        'timestamp': speech_chunks[0].timestamp,
                        'duration': len(combined_audio) / self.config.sample_rate,
                        'chunk_count': len(speech_chunks)
                    })
                except Exception as e:
                    logger.error(f"Speech end callback error: {e}")
            
            # Clear speech buffer for next utterance
            self.speech_buffer.clear()
            
        except Exception as e:
            logger.error(f"Speech end handling error: {e}")
    
    async def _notify_chunk_callbacks(self, chunk: StreamChunk):
        """Notify callbacks about new chunk"""
        for callback in self.result_callbacks:
            try:
                await callback({
                    'type': 'audio_chunk',
                    'chunk': chunk,
                    'timestamp': chunk.timestamp
                })
            except Exception as e:
                logger.error(f"Chunk callback error: {e}")
    
    def _update_chunk_stats(self, chunk: StreamChunk):
        """Update processing statistics"""
        self.stats['chunks_processed'] += 1
        self.stats['total_audio_time'] += len(chunk.data) / self.config.sample_rate
        
        # Update average latency
        total_chunks = self.stats['chunks_processed']
        current_avg = self.stats['average_latency']
        self.stats['average_latency'] = (
            (current_avg * (total_chunks - 1) + chunk.processing_latency) / total_chunks
        )
    
    def add_callback(self, callback: Callable):
        """Add result callback"""
        self.result_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove result callback"""
        if callback in self.result_callbacks:
            self.result_callbacks.remove(callback)
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            'is_streaming': self.is_streaming,
            'chunks_processed': self.stats['chunks_processed'],
            'speech_chunks': self.stats['speech_chunks'],
            'total_audio_time': self.stats['total_audio_time'],
            'average_latency_ms': self.stats['average_latency'],
            'last_speech_time': self.stats['last_speech_time'],
            'buffer_size': len(self.audio_buffer),
            'speech_buffer_size': len(self.speech_buffer),
            'speech_ratio': (
                self.stats['speech_chunks'] / self.stats['chunks_processed'] 
                if self.stats['chunks_processed'] > 0 else 0.0
            )
        }
    
    async def cleanup(self):
        """Cleanup streaming resources"""
        try:
            await self.stop_streaming()
            
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None
            
            self.result_callbacks.clear()
            self.audio_buffer.clear()
            self.speech_buffer.clear()
            
            logger.info("🧹 Audio streaming cleanup completed")
            
        except Exception as e:
            logger.error(f"Streaming cleanup error: {e}")


class StreamingVoiceProcessor:
    """
    🎵 STREAMING VOICE PROCESSOR
    High-level streaming voice processor with wake word and speech recognition
    """
    
    def __init__(
        self,
        wake_word_detector=None,
        speech_recognizer=None,
        stream_config: Optional[StreamConfig] = None
    ):
        self.wake_word_detector = wake_word_detector
        self.speech_recognizer = speech_recognizer
        self.stream_config = stream_config or StreamConfig()
        
        # Streaming components
        self.stream_processor = AudioStreamProcessor(self.stream_config)
        
        # State
        self.is_processing = False
        self.wake_word_active = False
        self.listening_for_command = False
        
        # Buffers for processing
        self.wake_word_buffer = deque(maxlen=20)  # ~1 second at 50fps
        self.command_buffer = deque(maxlen=100)   # ~5 seconds
        
        # Event callbacks
        self.on_wake_word_detected: Optional[Callable] = None
        self.on_speech_recognized: Optional[Callable] = None
        self.on_processing_complete: Optional[Callable] = None
        
        logger.info("🎵 Streaming Voice Processor initialized")
    
    async def start_processing(self) -> bool:
        """Start streaming voice processing"""
        try:
            if self.is_processing:
                logger.warning("Voice processing already active")
                return True
            
            # Initialize stream processor
            if not await self.stream_processor.initialize():
                logger.error("Failed to initialize stream processor")
                return False
            
            # Add callback for audio chunks
            self.stream_processor.add_callback(self._handle_audio_event)
            
            # Start streaming
            if not await self.stream_processor.start_streaming():
                logger.error("Failed to start audio streaming")
                return False
            
            self.is_processing = True
            logger.info("🎵 Streaming voice processing started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start voice processing: {e}")
            return False
    
    async def stop_processing(self) -> bool:
        """Stop streaming voice processing"""
        try:
            self.is_processing = False
            
            # Stop stream processor
            await self.stream_processor.stop_streaming()
            
            logger.info("🛑 Streaming voice processing stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop voice processing: {e}")
            return False
    
    async def _handle_audio_event(self, event: Dict[str, Any]):
        """Handle audio events from stream processor"""
        try:
            event_type = event.get('type')
            
            if event_type == 'audio_chunk':
                await self._handle_audio_chunk(event['chunk'])
            elif event_type == 'potential_wake_word':
                await self._handle_potential_wake_word(event)
            elif event_type == 'speech_utterance':
                await self._handle_speech_utterance(event)
                
        except Exception as e:
            logger.error(f"Audio event handling error: {e}")
    
    async def _handle_audio_chunk(self, chunk: StreamChunk):
        """Handle individual audio chunk"""
        try:
            # Always add to wake word buffer for continuous monitoring
            self.wake_word_buffer.append(chunk)
            
            # If listening for command, add to command buffer
            if self.listening_for_command:
                self.command_buffer.append(chunk)
                
        except Exception as e:
            logger.error(f"Audio chunk handling error: {e}")
    
    async def _handle_potential_wake_word(self, event: Dict[str, Any]):
        """Handle potential wake word detection"""
        try:
            if not self.wake_word_detector:
                return
            
            audio_data = event['audio_data']
            
            # Check with wake word detector
            detection_result = self.wake_word_detector.detect_wake_word(audio_data)
            
            if detection_result.detected:
                logger.info(f"🔔 Wake word detected! Confidence: {detection_result.confidence:.2f}")
                
                # Start listening for command
                self.wake_word_active = True
                self.listening_for_command = True
                self.command_buffer.clear()
                
                # Notify callback
                if self.on_wake_word_detected:
                    await self.on_wake_word_detected(detection_result)
                    
        except Exception as e:
            logger.error(f"Wake word handling error: {e}")
    
    async def _handle_speech_utterance(self, event: Dict[str, Any]):
        """Handle complete speech utterance"""
        try:
            if not self.listening_for_command or not self.speech_recognizer:
                return
            
            audio_data = event['audio_data']
            
            # Transcribe speech
            recognized_text = self.speech_recognizer.transcribe(
                audio_data, 
                self.stream_config.sample_rate
            )
            
            if recognized_text:
                logger.info(f"🗣️ Speech recognized: '{recognized_text}'")
                
                # Stop listening for this command
                self.listening_for_command = False
                self.wake_word_active = False
                
                # Notify callback
                if self.on_speech_recognized:
                    await self.on_speech_recognized(recognized_text)
                
                # Notify processing complete
                if self.on_processing_complete:
                    await self.on_processing_complete({
                        'recognized_text': recognized_text,
                        'audio_duration': event.get('duration', 0),
                        'processing_time': time.time() - event.get('timestamp', time.time())
                    })
                    
        except Exception as e:
            logger.error(f"Speech utterance handling error: {e}")
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get processing status"""
        stream_stats = self.stream_processor.get_streaming_stats()
        
        return {
            'is_processing': self.is_processing,
            'wake_word_active': self.wake_word_active,
            'listening_for_command': self.listening_for_command,
            'wake_word_buffer_size': len(self.wake_word_buffer),
            'command_buffer_size': len(self.command_buffer),
            'streaming_stats': stream_stats
        }
    
    async def cleanup(self):
        """Cleanup streaming processor"""
        try:
            await self.stop_processing()
            await self.stream_processor.cleanup()
            
            self.wake_word_buffer.clear()
            self.command_buffer.clear()
            
            logger.info("🧹 Streaming voice processor cleanup completed")
            
        except Exception as e:
            logger.error(f"Streaming processor cleanup error: {e}")


# Export classes
__all__ = [
    'AudioStreamProcessor',
    'StreamingVoiceProcessor', 
    'StreamChunk',
    'StreamConfig'
]