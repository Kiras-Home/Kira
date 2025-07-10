"""
🎤 VOICE SERVICE
Enterprise Voice Service ähnlich MemoryService - Zentraler Voice Service Manager
"""

import logging
import threading
import time
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json

from voice.synthesis.unified_bark_engine import UnifiedBarkTTSEngine

logger = logging.getLogger(__name__)


@dataclass
class VoiceServiceConfig:
    """Voice Service Configuration"""
    # Audio Settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    
    # Wake Word Settings
    wake_word: str = "kira"
    wake_word_threshold: float = 0.7
    
    # Speech Recognition
    whisper_model: str = "base"
    language: str = "de"
    
    # Voice Synthesis
    voice_preset: str = "v2/de/speaker_6"
    default_emotion: str = "neutral"
    
    # Service Settings
    enable_streaming: bool = True
    enable_memory_integration: bool = True
    enable_health_monitoring: bool = True
    health_check_interval: int = 30
    
    # Performance
    max_concurrent_requests: int = 5
    request_timeout: float = 30.0
    cache_enabled: bool = True
    cache_size_mb: int = 500


@dataclass
class VoiceRequest:
    """Voice Service Request"""
    request_id: str
    request_type: str  # 'listen', 'speak', 'process', 'stream'
    data: Any
    params: Dict[str, Any] = None
    callback: Optional[Callable] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.params is None:
            self.params = {}


@dataclass
class VoiceResponse:
    """Voice Service Response"""
    request_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VoiceService:
    """
    🎤 ENTERPRISE VOICE SERVICE
    Zentraler Voice Service Manager ähnlich MemoryService
    """
    
    def __init__(
        self,
        config: Optional[VoiceServiceConfig] = None,
        memory_service=None,
        command_processor=None,
        data_dir: Optional[Path] = None
    ):
        self.config = config or VoiceServiceConfig()
        self.memory_service = memory_service
        self.command_processor = command_processor
        self.data_dir = data_dir or Path("data/voice")
        
        # Service state
        self.is_initialized = False
        self.is_running = False
        self._lock = threading.Lock()
        
        # Core components
        self.voice_manager = None
        self.audio_manager = None
        self.voice_pipeline = None
        
        # Request handling
        self.request_queue = asyncio.Queue()
        self.active_requests = {}
        self.request_counter = 0
        
        # Service statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'uptime_start': None,
            'last_health_check': None
        }
        
        # Event handlers
        self.event_handlers = {
            'on_wake_word': [],
            'on_speech_recognized': [],
            'on_response_generated': [],
            'on_error': [],
            'on_status_change': []
        }
        
        # Health monitoring
        self.health_status = 'unknown'
        self.component_health = {}
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🎤 Voice Service initialized")
    
    async def initialize(self) -> bool:
        """Initialize Voice Service"""
        try:
            logger.info("🚀 Initializing Voice Service...")
            
            with self._lock:
                if self.is_initialized:
                    logger.warning("Voice Service already initialized")
                    return True
            
            # Initialize Enterprise Voice Manager
            success = await self._initialize_voice_manager()
            if not success:
                logger.error("Failed to initialize voice manager")
                return False
            
            # Start health monitoring
            if self.config.enable_health_monitoring:
                await self._start_health_monitoring()
            
            # Update state
            with self._lock:
                self.is_initialized = True
                self.stats['uptime_start'] = datetime.now()
            
            logger.info("✅ Voice Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Voice Service initialization failed: {e}")
            return False
    
    async def start(self) -> bool:
        """Start Voice Service"""
        try:
            if not self.is_initialized:
                logger.error("Voice Service not initialized")
                return False
            
            with self._lock:
                if self.is_running:
                    logger.warning("Voice Service already running")
                    return True
            
            logger.info("🚀 Starting Voice Service...")
            
            # Start voice system
            if self.voice_manager:
                success = self.voice_manager.start_voice_system()
                if not success:
                    logger.error("Failed to start voice system")
                    return False
            
            # Start request processing
            asyncio.create_task(self._process_requests())
            
            # Update state
            with self._lock:
                self.is_running = True
            
            # Notify status change
            await self._notify_status_change('running')
            
            logger.info("✅ Voice Service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Voice Service: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop Voice Service"""
        try:
            logger.info("🛑 Stopping Voice Service...")
            
            # Stop voice system
            if self.voice_manager:
                self.voice_manager.stop_voice_system()
            
            # Update state
            with self._lock:
                self.is_running = False
            
            # Notify status change
            await self._notify_status_change('stopped')
            
            logger.info("🛑 Voice Service stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Voice Service: {e}")
            return False
    
    async def listen_once(
        self, 
        duration: float = 5.0,
        return_audio: bool = False
    ) -> VoiceResponse:
        """Listen for speech once"""
        request = VoiceRequest(
            request_id=self._get_next_request_id(),
            request_type='listen',
            data={'duration': duration, 'return_audio': return_audio}
        )
        
        return await self._process_request(request)
    
    async def speak_text(
        self, 
        text: str, 
        emotion: str = "neutral",
        wait_for_completion: bool = True
    ) -> VoiceResponse:
        """Speak text using voice synthesis"""
        request = VoiceRequest(
            request_id=self._get_next_request_id(),
            request_type='speak',
            data={'text': text, 'emotion': emotion, 'wait': wait_for_completion}
        )
        
        return await self._process_request(request)
    
    async def process_audio(
        self, 
        audio_data: bytes, 
        sample_rate: int = 16000
    ) -> VoiceResponse:
        """Process audio data through speech recognition"""
        request = VoiceRequest(
            request_id=self._get_next_request_id(),
            request_type='process',
            data={'audio_data': audio_data, 'sample_rate': sample_rate}
        )
        
        return await self._process_request(request)
    
    async def start_streaming(self, callback: Callable) -> VoiceResponse:
        """Start real-time audio streaming"""
        if not self.config.enable_streaming:
            return VoiceResponse(
                request_id="stream_disabled",
                success=False,
                error="Streaming is disabled in configuration"
            )
        
        request = VoiceRequest(
            request_id=self._get_next_request_id(),
            request_type='stream',
            data={'callback': callback}
        )
        
        return await self._process_request(request)
    
    async def stop_streaming(self) -> VoiceResponse:
        """Stop real-time audio streaming"""
        # Find and cancel streaming requests
        streaming_requests = [req_id for req_id, req in self.active_requests.items() 
                            if req.request_type == 'stream']
        
        for req_id in streaming_requests:
            await self._cancel_request(req_id)
        
        return VoiceResponse(
            request_id="stop_stream",
            success=True,
            data={'stopped_streams': len(streaming_requests)}
        )
    
    async def get_voice_status(self) -> Dict[str, Any]:
        """Get comprehensive voice service status"""
        try:
            # Get voice manager status
            voice_status = {}
            if self.voice_manager:
                voice_status = self.voice_manager.get_detailed_status()
            
            # Service-specific status
            service_status = {
                'service': {
                    'initialized': self.is_initialized,
                    'running': self.is_running,
                    'health': self.health_status,
                    'uptime': self._get_uptime(),
                    'active_requests': len(self.active_requests),
                    'queue_size': self.request_queue.qsize()
                },
                'statistics': self._get_service_statistics(),
                'configuration': {
                    'sample_rate': self.config.sample_rate,
                    'wake_word': self.config.wake_word,
                    'streaming_enabled': self.config.enable_streaming,
                    'memory_integration': self.config.enable_memory_integration
                },
                'component_health': self.component_health
            }
            
            # Merge voice system and service status
            return {**voice_status, **service_status}
            
        except Exception as e:
            logger.error(f"Failed to get voice status: {e}")
            return {'error': str(e)}
    
    def add_event_handler(self, event: str, handler: Callable):
        """Add event handler"""
        if event in self.event_handlers:
            self.event_handlers[event].append(handler)
        else:
            logger.warning(f"Unknown event type: {event}")
    
    def remove_event_handler(self, event: str, handler: Callable):
        """Remove event handler"""
        if event in self.event_handlers and handler in self.event_handlers[event]:
            self.event_handlers[event].remove(handler)
    
    async def _initialize_voice_manager(self) -> bool:
        """Initialize Enterprise Voice Manager"""
        try:
            from voice.core.enterprise_voice_manager import EnterpriseVoiceManager
            
            # Convert service config to voice manager config
            voice_config = {
                'audio': {
                    'sample_rate': self.config.sample_rate,
                    'channels': self.config.channels,
                    'chunk_size': self.config.chunk_size
                },
                'wake_word': {
                    'wake_word': self.config.wake_word,
                    'threshold': self.config.wake_word_threshold
                },
                'speech_recognition': {
                    'model_size': self.config.whisper_model,
                    'language': self.config.language
                },
                'voice_synthesis': {
                    'voice_preset': self.config.voice_preset,
                    'default_emotion': self.config.default_emotion
                }
            }
            
            self.voice_manager = EnterpriseVoiceManager(
                config=voice_config,
                memory_service=self.memory_service,
                command_processor=self.command_processor
            )
            
            # Add service callbacks
            self.voice_manager.add_interaction_callback(self._on_voice_interaction)
            self.voice_manager.add_error_callback(self._on_voice_error)
            
            # Initialize voice manager
            success = self.voice_manager.initialize_voice_system()
            return success
            
        except Exception as e:
            logger.error(f"Voice manager initialization failed: {e}")
            return False
    
    async def _process_requests(self):
        """Process incoming requests"""
        while self.is_running:
            try:
                # Get request with timeout
                request = await asyncio.wait_for(
                    self.request_queue.get(), 
                    timeout=1.0
                )
                
                # Process request
                asyncio.create_task(self._handle_request(request))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Request processing error: {e}")
    
    async def _process_request(self, request: VoiceRequest) -> VoiceResponse:
        """Process a voice request"""
        start_time = time.time()
        
        try:
            # Add to queue
            await asyncio.wait_for(
                self.request_queue.put(request),
                timeout=5.0
            )
            
            # Wait for response (with timeout)
            response = await asyncio.wait_for(
                self._wait_for_response(request.request_id),
                timeout=self.config.request_timeout
            )
            
            return response
            
        except asyncio.TimeoutError:
            return VoiceResponse(
                request_id=request.request_id,
                success=False,
                error="Request timeout",
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return VoiceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _handle_request(self, request: VoiceRequest):
        """Handle individual request"""
        start_time = time.time()
        
        try:
            # Add to active requests
            self.active_requests[request.request_id] = request
            
            # Process based on type
            if request.request_type == 'listen':
                response = await self._handle_listen_request(request)
            elif request.request_type == 'speak':
                response = await self._handle_speak_request(request)
            elif request.request_type == 'process':
                response = await self._handle_process_request(request)
            elif request.request_type == 'stream':
                response = await self._handle_stream_request(request)
            else:
                response = VoiceResponse(
                    request_id=request.request_id,
                    success=False,
                    error=f"Unknown request type: {request.request_type}"
                )
            
            # Set processing time
            response.processing_time = time.time() - start_time
            
            # Update statistics
            self._update_request_stats(response.success, response.processing_time)
            
            # Execute callback if provided
            if request.callback:
                try:
                    await request.callback(response)
                except Exception as e:
                    logger.error(f"Request callback error: {e}")
            
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            response = VoiceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
        
        finally:
            # Remove from active requests
            self.active_requests.pop(request.request_id, None)
    
    async def _handle_listen_request(self, request: VoiceRequest) -> VoiceResponse:
        """Handle listen request"""
        try:
            data = request.data
            duration = data.get('duration', 5.0)
            return_audio = data.get('return_audio', False)
            
            # Use voice manager for listening
            if not self.voice_manager or not self.voice_manager.audio_manager:
                return VoiceResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Audio manager not available"
                )
            
            # Record audio
            audio_data = self.voice_manager.audio_manager.record_command(duration)
            
            if audio_data is None:
                return VoiceResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Audio recording failed"
                )
            
            # Speech recognition
            recognized_text = ""
            if self.voice_manager.speech_recognizer:
                recognized_text = self.voice_manager.speech_recognizer.transcribe(
                    audio_data, self.config.sample_rate
                )
            
            response_data = {
                'recognized_text': recognized_text,
                'audio_duration': duration
            }
            
            if return_audio:
                response_data['audio_data'] = audio_data
            
            return VoiceResponse(
                request_id=request.request_id,
                success=True,
                data=response_data
            )
            
        except Exception as e:
            return VoiceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_speak_request(self, request: VoiceRequest) -> VoiceResponse:
        """Handle speak request"""
        try:
            data = request.data
            text = data.get('text', '')
            emotion = data.get('emotion', 'neutral')
            wait = data.get('wait', True)
            
            if not text:
                return VoiceResponse(
                    request_id=request.request_id,
                    success=False,
                    error="No text provided"
                )
            
            # Use voice manager for speaking
            if not self.voice_manager or not self.voice_manager.bark_engine:
                return VoiceResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Voice synthesis not available"
                )
            
            # Synthesize speech
            audio_bytes = self.voice_manager.bark_engine.synthesize_german_female(
                text=text,
                emotion=emotion,
                use_cache=self.config.cache_enabled
            )
            
            if not audio_bytes:
                return VoiceResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Voice synthesis failed"
                )
            
            # Play audio if requested
            if wait and self.voice_manager.audio_manager:
                self.voice_manager.audio_manager.play_audio(audio_bytes)
            
            return VoiceResponse(
                request_id=request.request_id,
                success=True,
                data={
                    'text': text,
                    'emotion': emotion,
                    'audio_generated': True,
                    'audio_played': wait
                }
            )
            
        except Exception as e:
            return VoiceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_process_request(self, request: VoiceRequest) -> VoiceResponse:
        """Handle audio processing request"""
        try:
            data = request.data
            audio_data = data.get('audio_data')
            sample_rate = data.get('sample_rate', 16000)
            
            if audio_data is None:
                return VoiceResponse(
                    request_id=request.request_id,
                    success=False,
                    error="No audio data provided"
                )
            
            # Speech recognition
            recognized_text = ""
            if self.voice_manager and self.voice_manager.speech_recognizer:
                recognized_text = self.voice_manager.speech_recognizer.transcribe(
                    audio_data, sample_rate
                )
            
            # Command processing if available
            command_response = ""
            if recognized_text and self.command_processor:
                command_response = self.command_processor.process_command(recognized_text)
            
            return VoiceResponse(
                request_id=request.request_id,
                success=True,
                data={
                    'recognized_text': recognized_text,
                    'command_response': command_response,
                    'sample_rate': sample_rate
                }
            )
            
        except Exception as e:
            return VoiceResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_stream_request(self, request: VoiceRequest) -> VoiceResponse:
        """Handle streaming request - NOW IMPLEMENTED"""
        try:
            data = request.data
            callback = data.get('callback')
            
            if not callback:
                return VoiceResponse(
                    request_id=request.request_id,
                    success=False,
                    error="No callback provided for streaming"
                )
            
            # Import streaming components
            from voice.audio.streaming_audio import StreamingVoiceProcessor, StreamConfig
            
            # Create streaming processor
            stream_config = StreamConfig(
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
                chunk_size=self.config.chunk_size
            )
            
            streaming_processor = StreamingVoiceProcessor(
                wake_word_detector=self.voice_manager.wake_word_detector if self.voice_manager else None,
                speech_recognizer=self.voice_manager.speech_recognizer if self.voice_manager else None,
                stream_config=stream_config
            )
            
            # Set up streaming callbacks
            streaming_processor.on_wake_word_detected = lambda result: callback({
                'type': 'wake_word',
                'confidence': result.confidence,
                'timestamp': time.time()
            })
            
            streaming_processor.on_speech_recognized = lambda text: callback({
                'type': 'speech_recognized', 
                'text': text,
                'timestamp': time.time()
            })
            
            streaming_processor.on_processing_complete = lambda data: callback({
                'type': 'processing_complete',
                'data': data,
                'timestamp': time.time()
            })
            
            # Start streaming
            success = await streaming_processor.start_processing()
            
            if success:
                # Store processor for cleanup
                self.active_requests[request.request_id] = {
                    'processor': streaming_processor,
                    'type': 'streaming'
                }
                
                return VoiceResponse(
                    request_id=request.request_id,
                    success=True,
                    data={
                        'streaming_started': True,
                        'processor_id': request.request_id
                    }
                )
            else:
                return VoiceResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Failed to start streaming processor"
                )
                
        except Exception as e:
            return VoiceResponse(
                request_id=request.request_id,
                success=False,
                error=f"Streaming error: {e}"
            )
    
    async def _wait_for_response(self, request_id: str) -> VoiceResponse:
        """Wait for request response"""
        # This would wait for the response from the request handler
        # For now, implement a simple wait mechanism
        timeout = self.config.request_timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id not in self.active_requests:
                # Request completed, but we need the response
                # For now, return a placeholder
                break
            await asyncio.sleep(0.1)
        
        # Placeholder implementation
        return VoiceResponse(
            request_id=request_id,
            success=True,
            data="Response placeholder"
        )
    
    async def _cancel_request(self, request_id: str):
        """Cancel active request"""
        self.active_requests.pop(request_id, None)
    
    async def _start_health_monitoring(self):
        """Start health monitoring"""
        async def health_monitor():
            while self.is_running:
                try:
                    await self._perform_health_check()
                    await asyncio.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
        
        asyncio.create_task(health_monitor())
    
    async def _perform_health_check(self):
        """Perform health check"""
        try:
            # Check voice manager health
            if self.voice_manager:
                voice_status = self.voice_manager.get_system_status()
                self.component_health['voice_manager'] = voice_status.running
            
            # Overall health assessment
            healthy_components = sum(1 for status in self.component_health.values() if status)
            total_components = len(self.component_health)
            
            if total_components == 0:
                self.health_status = 'unknown'
            elif healthy_components == total_components:
                self.health_status = 'healthy'
            elif healthy_components > total_components * 0.5:
                self.health_status = 'degraded'
            else:
                self.health_status = 'unhealthy'
            
            self.stats['last_health_check'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.health_status = 'error'
    
    def _on_voice_interaction(self, pipeline_data):
        """Handle voice interaction events"""
        try:
            # Notify speech recognition handlers
            if hasattr(pipeline_data, 'recognized_text') and pipeline_data.recognized_text:
                for handler in self.event_handlers['on_speech_recognized']:
                    handler(pipeline_data.recognized_text)
            
            # Notify response generation handlers
            if hasattr(pipeline_data, 'response_text') and pipeline_data.response_text:
                for handler in self.event_handlers['on_response_generated']:
                    handler(pipeline_data.response_text)
            
            # Notify wake word handlers
            if hasattr(pipeline_data, 'wake_word_detected') and pipeline_data.wake_word_detected:
                for handler in self.event_handlers['on_wake_word']:
                    handler()
                    
        except Exception as e:
            logger.error(f"Voice interaction event handling error: {e}")
    
    def _on_voice_error(self, error_info):
        """Handle voice error events"""
        try:
            for handler in self.event_handlers['on_error']:
                handler(error_info)
        except Exception as e:
            logger.error(f"Voice error event handling error: {e}")
    
    async def _notify_status_change(self, new_status: str):
        """Notify status change handlers"""
        try:
            for handler in self.event_handlers['on_status_change']:
                handler(new_status)
        except Exception as e:
            logger.error(f"Status change notification error: {e}")
    
    def _get_next_request_id(self) -> str:
        """Get next request ID"""
        self.request_counter += 1
        return f"voice_req_{self.request_counter}_{int(time.time())}"
    
    def _update_request_stats(self, success: bool, processing_time: float):
        """Update request statistics"""
        with self._lock:
            self.stats['total_requests'] += 1
            
            if success:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
            
            # Update average response time
            total_requests = self.stats['total_requests']
            current_avg = self.stats['average_response_time']
            self.stats['average_response_time'] = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )
    
    def _get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        with self._lock:
            return dict(self.stats)
    
    def _get_uptime(self) -> float:
        """Get service uptime in seconds"""
        if self.stats['uptime_start']:
            return (datetime.now() - self.stats['uptime_start']).total_seconds()
        return 0.0
    
    async def cleanup(self):
        """Cleanup voice service"""
        try:
            logger.info("🧹 Cleaning up Voice Service...")
            
            # Stop service
            await self.stop()
            
            # Cleanup voice manager
            if self.voice_manager:
                self.voice_manager.cleanup()
            
            # Clear queues and active requests
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                except:
                    break
            
            self.active_requests.clear()
            
            logger.info("🧹 Voice Service cleanup completed")
            
        except Exception as e:
            logger.error(f"Voice Service cleanup error: {e}")


# Export classes
__all__ = [
    'VoiceService',
    'VoiceServiceConfig',
    'VoiceRequest',
    'VoiceResponse'
]