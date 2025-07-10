"""
🚀 ENTERPRISE VOICE PIPELINE
Professional Voice Processing Pipeline für Kira
"""

import logging
import asyncio
import threading
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import queue

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages"""
    AUDIO_CAPTURE = "audio_capture"
    WAKE_WORD_DETECTION = "wake_word_detection"
    SPEECH_RECOGNITION = "speech_recognition"
    CONTEXT_ENHANCEMENT = "context_enhancement"
    COMMAND_PROCESSING = "command_processing"
    RESPONSE_GENERATION = "response_generation"
    VOICE_SYNTHESIS = "voice_synthesis"
    AUDIO_OUTPUT = "audio_output"


@dataclass
class PipelineData:
    """Data flowing through the pipeline"""
    stage: PipelineStage
    timestamp: float
    session_id: str
    
    # Audio data
    raw_audio: Optional[bytes] = None
    processed_audio: Optional[bytes] = None
    audio_quality: float = 0.0
    
    # Recognition data
    wake_word_detected: bool = False
    wake_word_confidence: float = 0.0
    recognized_text: str = ""
    recognition_confidence: float = 0.0
    user_emotion: Optional[str] = None
    
    # Context data
    conversation_context: Dict[str, Any] = None
    enhanced_prompt: str = ""
    user_preferences: Dict[str, Any] = None
    
    # Response data
    response_text: str = ""
    response_emotion: str = "neutral"
    command_executed: bool = False
    command_result: Dict[str, Any] = None
    
    # Output data
    synthesized_audio: Optional[bytes] = None
    output_quality: float = 0.0
    
    # Processing metadata
    processing_times: Dict[str, float] = None
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = {}
        if self.error_messages is None:
            self.error_messages = []
        if self.conversation_context is None:
            self.conversation_context = {}
        if self.user_preferences is None:
            self.user_preferences = {}


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    total_processed: int = 0
    successful_completions: int = 0
    average_latency: float = 0.0
    stage_latencies: Dict[str, float] = None
    error_counts: Dict[str, int] = None
    quality_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.stage_latencies is None:
            self.stage_latencies = {}
        if self.error_counts is None:
            self.error_counts = {}
        if self.quality_scores is None:
            self.quality_scores = {}


class EnterpriseVoicePipeline:
    """
    🚀 ENTERPRISE VOICE PIPELINE
    Professional Voice Processing Pipeline mit allen Komponenten
    """
    
    def __init__(
        self,
        audio_manager=None,
        wake_word_detector=None,
        speech_recognizer=None,
        memory_bridge=None,
        personality_engine=None,
        bark_engine=None,
        command_processor=None,
        max_queue_size: int = 100,
        enable_async_processing: bool = True
    ):
        # Core components
        self.audio_manager = audio_manager
        self.wake_word_detector = wake_word_detector
        self.speech_recognizer = speech_recognizer
        self.memory_bridge = memory_bridge
        self.personality_engine = personality_engine
        self.bark_engine = bark_engine
        self.command_processor = command_processor
        
        # Pipeline configuration
        self.max_queue_size = max_queue_size
        self.enable_async_processing = enable_async_processing
        
        # Processing queues
        self.audio_queue = queue.Queue(maxsize=max_queue_size)
        self.processing_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        
        # Pipeline state
        self.is_running = False
        self.is_listening = False
        self.current_session = None
        
        # Worker threads
        self.audio_worker = None
        self.processing_worker = None
        self.output_worker = None
        self._stop_event = threading.Event()
        
        # Performance metrics
        self.metrics = PipelineMetrics()
        self.performance_history = []
        
        # Pipeline callbacks
        self.stage_callbacks = {}  # stage -> [callbacks]
        self.error_callbacks = []
        self.completion_callbacks = []
        
        logger.info("🚀 Enterprise Voice Pipeline initialized")
        logger.info(f"   Async processing: {enable_async_processing}")
        logger.info(f"   Max queue size: {max_queue_size}")
    
    def start_pipeline(self) -> bool:
        """Start the voice processing pipeline"""
        try:
            if self.is_running:
                logger.warning("Pipeline already running")
                return True
            
            # Validate components
            missing_components = self._validate_components()
            if missing_components:
                logger.error(f"Missing pipeline components: {missing_components}")
                return False
            
            # Start memory bridge session
            if self.memory_bridge:
                self.current_session = self.memory_bridge.start_conversation_session()
            
            # Clear stop event
            self._stop_event.clear()
            
            # Start worker threads
            self.audio_worker = threading.Thread(
                target=self._audio_processing_loop,
                name="VoicePipelineAudio",
                daemon=True
            )
            
            self.processing_worker = threading.Thread(
                target=self._main_processing_loop,
                name="VoicePipelineProcessing",
                daemon=True
            )
            
            self.output_worker = threading.Thread(
                target=self._output_processing_loop,
                name="VoicePipelineOutput",
                daemon=True
            )
            
            # Start threads
            self.audio_worker.start()
            self.processing_worker.start()
            self.output_worker.start()
            
            # Start audio capture
            if self.audio_manager:
                success = self.audio_manager.start_recording(callback=self._audio_callback)
                if not success:
                    logger.error("Failed to start audio recording")
                    return False
            
            self.is_running = True
            self.is_listening = True
            
            logger.info("🚀 Voice pipeline started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start voice pipeline: {e}")
            return False
    
    def stop_pipeline(self) -> bool:
        """Stop the voice processing pipeline"""
        try:
            self.is_running = False
            self.is_listening = False
            
            # Stop audio recording
            if self.audio_manager:
                self.audio_manager.stop_recording()
            
            # Signal stop to all threads
            self._stop_event.set()
            
            # Wait for threads to finish
            for worker in [self.audio_worker, self.processing_worker, self.output_worker]:
                if worker and worker.is_alive():
                    worker.join(timeout=2.0)
            
            # End memory session
            if self.memory_bridge and self.current_session:
                self.memory_bridge.end_conversation_session(self.current_session)
                self.current_session = None
            
            # Clear queues
            self._clear_queues()
            
            logger.info("🛑 Voice pipeline stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop voice pipeline: {e}")
            return False
    
    def _validate_components(self) -> List[str]:
        """Validate required pipeline components"""
        missing = []
        
        required_components = [
            ("audio_manager", self.audio_manager),
            ("wake_word_detector", self.wake_word_detector),
            ("speech_recognizer", self.speech_recognizer),
            ("bark_engine", self.bark_engine)
        ]
        
        for name, component in required_components:
            if component is None:
                missing.append(name)
        
        return missing
    
    def _audio_callback(self, audio_buffer):
        """Callback for incoming audio data"""
        try:
            if not self.is_listening:
                return
            
            # Create pipeline data
            pipeline_data = PipelineData(
                stage=PipelineStage.AUDIO_CAPTURE,
                timestamp=time.time(),
                session_id=self.current_session or "no_session",
                raw_audio=audio_buffer.data.tobytes() if hasattr(audio_buffer, 'data') else audio_buffer,
                audio_quality=getattr(audio_buffer, 'snr_db', 0.0) / 40.0,  # Normalize to 0-1
                user_emotion=self._detect_emotion_from_audio(audio_buffer)
            )
            
            # Add to processing queue
            try:
                self.audio_queue.put_nowait(pipeline_data)
            except queue.Full:
                logger.warning("Audio queue full, dropping audio buffer")
                
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
    
    def _audio_processing_loop(self):
        """Audio processing worker loop"""
        logger.info("🎤 Audio processing loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get audio data
                pipeline_data = self.audio_queue.get(timeout=0.1)
                
                # Process wake word detection
                pipeline_data = self._process_wake_word_detection(pipeline_data)
                
                # If wake word detected, add to main processing queue
                if pipeline_data.wake_word_detected:
                    try:
                        self.processing_queue.put_nowait(pipeline_data)
                        logger.info(f"🔔 Wake word detected, queued for processing")
                    except queue.Full:
                        logger.warning("Processing queue full, dropping wake word detection")
                
                # Call stage callbacks
                self._call_stage_callbacks(PipelineStage.WAKE_WORD_DETECTION, pipeline_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing loop error: {e}")
        
        logger.info("🛑 Audio processing loop stopped")
    
    def _main_processing_loop(self):
        """Main processing worker loop"""
        logger.info("🔄 Main processing loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get processing data
                pipeline_data = self.processing_queue.get(timeout=0.1)
                
                # Process through all stages
                pipeline_data = self._process_full_pipeline(pipeline_data)
                
                # Add to output queue
                try:
                    self.output_queue.put_nowait(pipeline_data)
                except queue.Full:
                    logger.warning("Output queue full, dropping processed data")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Main processing loop error: {e}")
        
        logger.info("🛑 Main processing loop stopped")
    
    def _output_processing_loop(self):
        """Output processing worker loop"""
        logger.info("🔊 Output processing loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get output data
                pipeline_data = self.output_queue.get(timeout=0.1)
                
                # Play audio output
                if pipeline_data.synthesized_audio and self.audio_manager:
                    self.audio_manager.play_audio(pipeline_data.synthesized_audio)
                
                # Update metrics
                self._update_metrics(pipeline_data)
                
                # Call completion callbacks
                for callback in self.completion_callbacks:
                    callback(pipeline_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Output processing loop error: {e}")
        
        logger.info("🛑 Output processing loop stopped")
    
    def _process_wake_word_detection(self, pipeline_data: PipelineData) -> PipelineData:
        """Process wake word detection stage"""
        start_time = time.time()
        
        try:
            if not self.wake_word_detector:
                pipeline_data.error_messages.append("No wake word detector available")
                return pipeline_data
            
            # Convert audio for wake word detection
            import numpy as np
            audio_array = np.frombuffer(pipeline_data.raw_audio, dtype=np.float32)
            
            # Detect wake word
            result = self.wake_word_detector.detect_wake_word(audio_array)
            
            pipeline_data.wake_word_detected = result.detected
            pipeline_data.wake_word_confidence = result.confidence
            pipeline_data.stage = PipelineStage.WAKE_WORD_DETECTION
            
            processing_time = time.time() - start_time
            pipeline_data.processing_times["wake_word_detection"] = processing_time
            
            logger.debug(f"Wake word detection: {result.detected} (confidence: {result.confidence:.2f})")
            
        except Exception as e:
            pipeline_data.error_messages.append(f"Wake word detection error: {e}")
            logger.error(f"Wake word detection error: {e}")
        
        return pipeline_data
    
    def _process_full_pipeline(self, pipeline_data: PipelineData) -> PipelineData:
        """Process data through full pipeline"""
        try:
            # Speech Recognition
            pipeline_data = self._process_speech_recognition(pipeline_data)
            
            # Context Enhancement
            pipeline_data = self._process_context_enhancement(pipeline_data)
            
            # Command Processing
            pipeline_data = self._process_command_processing(pipeline_data)
            
            # Response Generation
            pipeline_data = self._process_response_generation(pipeline_data)
            
            # Voice Synthesis
            pipeline_data = self._process_voice_synthesis(pipeline_data)
            
            # Record interaction
            if self.memory_bridge:
                self._record_interaction(pipeline_data)
            
        except Exception as e:
            pipeline_data.error_messages.append(f"Pipeline processing error: {e}")
            logger.error(f"Pipeline processing error: {e}")
        
        return pipeline_data
    
    def _process_speech_recognition(self, pipeline_data: PipelineData) -> PipelineData:
        """Process speech recognition stage"""
        start_time = time.time()
        
        try:
            if not self.speech_recognizer:
                pipeline_data.error_messages.append("No speech recognizer available")
                return pipeline_data
            
            # Capture additional audio for speech recognition
            if self.audio_manager:
                command_audio = self.audio_manager.record_command(duration=5.0)
                if command_audio is not None:
                    # Transcribe audio
                    recognized_text = self.speech_recognizer.transcribe(command_audio, 16000)
                    pipeline_data.recognized_text = recognized_text or ""
                    pipeline_data.recognition_confidence = 0.8  # Default confidence
            
            pipeline_data.stage = PipelineStage.SPEECH_RECOGNITION
            
            processing_time = time.time() - start_time
            pipeline_data.processing_times["speech_recognition"] = processing_time
            
            logger.info(f"🗣️ Recognized: '{pipeline_data.recognized_text}'")
            
        except Exception as e:
            pipeline_data.error_messages.append(f"Speech recognition error: {e}")
            logger.error(f"Speech recognition error: {e}")
        
        return pipeline_data
    
    def _process_context_enhancement(self, pipeline_data: PipelineData) -> PipelineData:
        """Process context enhancement stage"""
        start_time = time.time()
        
        try:
            if not self.memory_bridge:
                pipeline_data.conversation_context = {"context": "no_memory_bridge"}
                return pipeline_data
            
            # Get contextual enhancement
            context = self.memory_bridge.get_contextual_prompt_enhancement(
                pipeline_data.session_id,
                pipeline_data.recognized_text
            )
            
            pipeline_data.conversation_context = context
            pipeline_data.user_preferences = context.get("user_preferences", {})
            
            # Enhance prompt with context
            if pipeline_data.recognized_text:
                pipeline_data.enhanced_prompt = self._create_enhanced_prompt(pipeline_data, context)
            
            pipeline_data.stage = PipelineStage.CONTEXT_ENHANCEMENT
            
            processing_time = time.time() - start_time
            pipeline_data.processing_times["context_enhancement"] = processing_time
            
            logger.debug(f"Context enhancement completed")
            
        except Exception as e:
            pipeline_data.error_messages.append(f"Context enhancement error: {e}")
            logger.error(f"Context enhancement error: {e}")
        
        return pipeline_data
    
    def _process_command_processing(self, pipeline_data: PipelineData) -> PipelineData:
        """Process command processing stage"""
        start_time = time.time()
        
        try:
            if not pipeline_data.recognized_text:
                return pipeline_data
            
            # Try command processing first
            if self.command_processor:
                command_result = self.command_processor.process_command(
                    pipeline_data.recognized_text,
                    context=pipeline_data.conversation_context
                )
                
                if command_result and command_result.get("success"):
                    pipeline_data.command_executed = True
                    pipeline_data.command_result = command_result
                    pipeline_data.response_text = command_result.get("response", "")
            
            # If no command was executed, prepare for general response
            if not pipeline_data.command_executed:
                pipeline_data.response_text = pipeline_data.enhanced_prompt or pipeline_data.recognized_text
            
            pipeline_data.stage = PipelineStage.COMMAND_PROCESSING
            
            processing_time = time.time() - start_time
            pipeline_data.processing_times["command_processing"] = processing_time
            
        except Exception as e:
            pipeline_data.error_messages.append(f"Command processing error: {e}")
            logger.error(f"Command processing error: {e}")
        
        return pipeline_data
    
    def _process_response_generation(self, pipeline_data: PipelineData) -> PipelineData:
        """Process response generation stage"""
        start_time = time.time()
        
        try:
            # Enhance response with personality if available
            if self.personality_engine and pipeline_data.response_text:
                personality_result = self.personality_engine.enhance_response(
                    pipeline_data.response_text,
                    context=pipeline_data.conversation_context,
                    user_emotion=pipeline_data.user_emotion
                )
                
                pipeline_data.response_text = personality_result.enhanced_text
                pipeline_data.response_emotion = personality_result.emotion
            else:
                # Default response if no text was generated
                if not pipeline_data.response_text:
                    pipeline_data.response_text = "Entschuldigung, ich habe Sie nicht verstanden."
                    pipeline_data.response_emotion = "apologetic"
            
            pipeline_data.stage = PipelineStage.RESPONSE_GENERATION
            
            processing_time = time.time() - start_time
            pipeline_data.processing_times["response_generation"] = processing_time
            
            logger.info(f"💬 Response: '{pipeline_data.response_text}' (emotion: {pipeline_data.response_emotion})")
            
        except Exception as e:
            pipeline_data.error_messages.append(f"Response generation error: {e}")
            logger.error(f"Response generation error: {e}")
        
        return pipeline_data
    
    def _process_voice_synthesis(self, pipeline_data: PipelineData) -> PipelineData:
        """Process voice synthesis stage"""
        start_time = time.time()
        
        try:
            if not self.bark_engine or not pipeline_data.response_text:
                return pipeline_data
            
            # Synthesize German female voice
            audio_bytes = self.bark_engine.synthesize_german_female(
                text=pipeline_data.response_text,
                emotion=pipeline_data.response_emotion,
                enhance_pronunciation=True,
                use_cache=True
            )
            
            if audio_bytes:
                pipeline_data.synthesized_audio = audio_bytes
                pipeline_data.output_quality = 0.8  # Default quality score
            
            pipeline_data.stage = PipelineStage.VOICE_SYNTHESIS
            
            processing_time = time.time() - start_time
            pipeline_data.processing_times["voice_synthesis"] = processing_time
            
            logger.info(f"🗣️ Voice synthesized for response")
            
        except Exception as e:
            pipeline_data.error_messages.append(f"Voice synthesis error: {e}")
            logger.error(f"Voice synthesis error: {e}")
        
        return pipeline_data
    
    def _create_enhanced_prompt(self, pipeline_data: PipelineData, context: Dict[str, Any]) -> str:
        """Create enhanced prompt with context"""
        try:
            base_prompt = pipeline_data.recognized_text
            
            # Add context information for better responses
            context_parts = []
            
            # Add conversation context
            conv_context = context.get("conversation_context", {})
            if conv_context.get("current_topic"):
                context_parts.append(f"Aktuelles Thema: {conv_context['current_topic']}")
            
            if conv_context.get("user_mood"):
                context_parts.append(f"Nutzerstimmung: {conv_context['user_mood']}")
            
            # Add relevant memories
            memories = context.get("relevant_memories", [])
            if memories:
                context_parts.append("Relevante Erinnerungen:")
                for memory in memories[:2]:  # Top 2 memories
                    context_parts.append(f"- {memory['content']}")
            
            if context_parts:
                enhanced_prompt = f"{base_prompt}\n\nKontext:\n" + "\n".join(context_parts)
            else:
                enhanced_prompt = base_prompt
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Enhanced prompt creation error: {e}")
            return pipeline_data.recognized_text
    
    def _record_interaction(self, pipeline_data: PipelineData):
        """Record interaction to memory system"""
        try:
            if not self.memory_bridge:
                return
            
            # Determine interaction type
            interaction_type = "conversation"
            if pipeline_data.command_executed:
                interaction_type = "command"
            elif "?" in pipeline_data.recognized_text:
                interaction_type = "question"
            
            # Record interaction
            self.memory_bridge.record_voice_interaction(
                session_id=pipeline_data.session_id,
                user_input=pipeline_data.recognized_text,
                kira_response=pipeline_data.response_text,
                user_emotion=pipeline_data.user_emotion,
                response_emotion=pipeline_data.response_emotion,
                confidence_score=pipeline_data.recognition_confidence,
                processing_time=sum(pipeline_data.processing_times.values()),
                interaction_type=interaction_type
            )
            
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")
    
    def _detect_emotion_from_audio(self, audio_buffer) -> Optional[str]:
        """Simple emotion detection from audio properties"""
        try:
            if not hasattr(audio_buffer, 'energy_level'):
                return None
            
            energy = audio_buffer.energy_level
            
            # Very simple emotion detection based on energy
            if energy > 0.1:
                return "excited"
            elif energy > 0.05:
                return "normal"
            elif energy > 0.01:
                return "calm"
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Emotion detection error: {e}")
            return None
    
    def _update_metrics(self, pipeline_data: PipelineData):
        """Update pipeline performance metrics"""
        try:
            self.metrics.total_processed += 1
            
            # Check if processing was successful
            if pipeline_data.synthesized_audio and not pipeline_data.error_messages:
                self.metrics.successful_completions += 1
            
            # Update stage latencies
            for stage, latency in pipeline_data.processing_times.items():
                if stage not in self.metrics.stage_latencies:
                    self.metrics.stage_latencies[stage] = []
                self.metrics.stage_latencies[stage].append(latency)
            
            # Update error counts
            for error in pipeline_data.error_messages:
                error_type = error.split(":")[0] if ":" in error else "unknown"
                if error_type not in self.metrics.error_counts:
                    self.metrics.error_counts[error_type] = 0
                self.metrics.error_counts[error_type] += 1
            
            # Calculate average latency
            total_latency = sum(pipeline_data.processing_times.values())
            if self.metrics.total_processed > 0:
                self.metrics.average_latency = (
                    (self.metrics.average_latency * (self.metrics.total_processed - 1) + total_latency) /
                    self.metrics.total_processed
                )
            
        except Exception as e:
            logger.error(f"Metrics update error: {e}")
    
    def _call_stage_callbacks(self, stage: PipelineStage, pipeline_data: PipelineData):
        """Call callbacks for pipeline stage"""
        try:
            stage_callbacks = self.stage_callbacks.get(stage, [])
            for callback in stage_callbacks:
                callback(pipeline_data)
        except Exception as e:
            logger.error(f"Stage callback error: {e}")
    
    def _clear_queues(self):
        """Clear all processing queues"""
        for q in [self.audio_queue, self.processing_queue, self.output_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
    
    def add_stage_callback(self, stage: PipelineStage, callback: Callable):
        """Add callback for pipeline stage"""
        if stage not in self.stage_callbacks:
            self.stage_callbacks[stage] = []
        self.stage_callbacks[stage].append(callback)
    
    def add_completion_callback(self, callback: Callable):
        """Add callback for pipeline completion"""
        self.completion_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            "running": self.is_running,
            "listening": self.is_listening,
            "current_session": self.current_session,
            "queue_sizes": {
                "audio": self.audio_queue.qsize(),
                "processing": self.processing_queue.qsize(),
                "output": self.output_queue.qsize()
            },
            "components": {
                "audio_manager": self.audio_manager is not None,
                "wake_word_detector": self.wake_word_detector is not None,
                "speech_recognizer": self.speech_recognizer is not None,
                "memory_bridge": self.memory_bridge is not None,
                "personality_engine": self.personality_engine is not None,
                "bark_engine": self.bark_engine is not None,
                "command_processor": self.command_processor is not None
            },
            "metrics": {
                "total_processed": self.metrics.total_processed,
                "successful_completions": self.metrics.successful_completions,
                "success_rate": (
                    self.metrics.successful_completions / self.metrics.total_processed
                    if self.metrics.total_processed > 0 else 0.0
                ),
                "average_latency": self.metrics.average_latency,
                "error_counts": self.metrics.error_counts
            }
        }


# Export classes
__all__ = [
    'EnterpriseVoicePipeline',
    'PipelineData',
    'PipelineStage',
    'PipelineMetrics'
]