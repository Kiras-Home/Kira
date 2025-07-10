"""
🎯 ENTERPRISE WAKE WORD DETECTOR
Multi-Model "Kira" Detection System
"""

import logging
import threading
import time
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    pvporcupine = None

try:
    import snowboy
    SNOWBOY_AVAILABLE = True
except ImportError:
    SNOWBOY_AVAILABLE = False
    snowboy = None

logger = logging.getLogger(__name__)


@dataclass
class WakeWordResult:
    """Wake word detection result"""
    detected: bool
    confidence: float
    keyword: str
    timestamp: float
    model_used: str
    processing_time: float
    audio_quality: float = 0.0


class SimpleWakeWordDetector:
    """Simple text-based wake word detection fallback"""
    
    def __init__(self, wake_words: List[str]):
        self.wake_words = [word.lower() for word in wake_words]
        self.confidence_threshold = 0.6
        
    def detect(self, text: str) -> WakeWordResult:
        """Detect wake word in text"""
        start_time = time.time()
        text_lower = text.lower()
        
        best_match = None
        best_confidence = 0.0
        
        for wake_word in self.wake_words:
            if wake_word in text_lower:
                # Simple confidence based on word boundaries
                words = text_lower.split()
                if wake_word in words:
                    confidence = 0.9  # Exact word match
                else:
                    confidence = 0.7  # Substring match
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = wake_word
        
        detected = best_confidence >= self.confidence_threshold
        processing_time = time.time() - start_time
        
        return WakeWordResult(
            detected=detected,
            confidence=best_confidence,
            keyword=best_match or "",
            timestamp=time.time(),
            model_used="simple_text",
            processing_time=processing_time
        )


class PorcupineWakeWordDetector:
    """Picovoice Porcupine wake word detector"""
    
    def __init__(self, access_key: str, keyword_paths: List[str], sensitivity: float = 0.7):
        self.access_key = access_key
        self.keyword_paths = keyword_paths
        self.sensitivity = sensitivity
        self.porcupine = None
        self.is_initialized = False
        
        try:
            if PORCUPINE_AVAILABLE:
                self.porcupine = pvporcupine.create(
                    access_key=access_key,
                    keyword_paths=keyword_paths,
                    sensitivities=[sensitivity] * len(keyword_paths)
                )
                self.is_initialized = True
                logger.info("✅ Porcupine wake word detector initialized")
            else:
                logger.warning("Porcupine not available")
        except Exception as e:
            logger.error(f"Porcupine initialization failed: {e}")
    
    def detect(self, audio_frame: np.ndarray) -> WakeWordResult:
        """Detect wake word in audio frame"""
        start_time = time.time()
        
        if not self.is_initialized or not self.porcupine:
            return WakeWordResult(
                detected=False,
                confidence=0.0,
                keyword="",
                timestamp=time.time(),
                model_used="porcupine_unavailable",
                processing_time=time.time() - start_time
            )
        
        try:
            # Convert to int16 format expected by Porcupine
            audio_int16 = (audio_frame * 32767).astype(np.int16)
            
            # Process frame
            keyword_index = self.porcupine.process(audio_int16)
            
            processing_time = time.time() - start_time
            
            if keyword_index >= 0:
                return WakeWordResult(
                    detected=True,
                    confidence=0.8,  # Porcupine doesn't provide confidence scores
                    keyword=f"keyword_{keyword_index}",
                    timestamp=time.time(),
                    model_used="porcupine",
                    processing_time=processing_time
                )
            else:
                return WakeWordResult(
                    detected=False,
                    confidence=0.0,
                    keyword="",
                    timestamp=time.time(),
                    model_used="porcupine",
                    processing_time=processing_time
                )
                
        except Exception as e:
            logger.error(f"Porcupine detection error: {e}")
            return WakeWordResult(
                detected=False,
                confidence=0.0,
                keyword="",
                timestamp=time.time(),
                model_used="porcupine_error",
                processing_time=time.time() - start_time
            )
    
    def cleanup(self):
        """Cleanup Porcupine resources"""
        if self.porcupine:
            self.porcupine.delete()


class AdaptiveThresholdManager:
    """Adaptive threshold management for wake word detection"""
    
    def __init__(self, initial_threshold: float = 0.7):
        self.current_threshold = initial_threshold
        self.initial_threshold = initial_threshold
        self.recent_scores = []
        self.false_positives = 0
        self.true_positives = 0
        self.adaptation_rate = 0.1
        self.max_history = 100
        
    def update_threshold(self, score: float, was_true_positive: bool):
        """Update threshold based on detection results"""
        self.recent_scores.append(score)
        if len(self.recent_scores) > self.max_history:
            self.recent_scores.pop(0)
        
        if was_true_positive:
            self.true_positives += 1
            # Lower threshold slightly if we're getting good detections
            self.current_threshold = max(0.3, self.current_threshold - self.adaptation_rate * 0.1)
        else:
            self.false_positives += 1
            # Raise threshold if we're getting false positives
            self.current_threshold = min(0.9, self.current_threshold + self.adaptation_rate * 0.2)
        
        # Don't deviate too far from initial threshold
        self.current_threshold = max(
            self.initial_threshold - 0.2,
            min(self.initial_threshold + 0.2, self.current_threshold)
        )
    
    def get_threshold(self) -> float:
        """Get current adaptive threshold"""
        return self.current_threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        total_detections = self.true_positives + self.false_positives
        accuracy = self.true_positives / total_detections if total_detections > 0 else 0.0
        
        return {
            'current_threshold': self.current_threshold,
            'initial_threshold': self.initial_threshold,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'accuracy': accuracy,
            'recent_scores_avg': np.mean(self.recent_scores) if self.recent_scores else 0.0
        }


class EnterpriseWakeWordDetector:
    """
    🎯 ENTERPRISE WAKE WORD DETECTOR
    Multi-Model "Kira" Detection with Adaptive Thresholding
    """
    
    def __init__(
        self,
        wake_words: List[str] = ["kira", "kiera"],
        confidence_threshold: float = 0.7,
        enable_noise_filtering: bool = True,
        model_path: Optional[str] = None,
        enable_adaptive_threshold: bool = True,
        porcupine_access_key: Optional[str] = None
    ):
        self.wake_words = [word.lower() for word in wake_words]
        self.confidence_threshold = confidence_threshold
        self.enable_noise_filtering = enable_noise_filtering
        self.model_path = Path(model_path) if model_path else None
        self.enable_adaptive_threshold = enable_adaptive_threshold
        
        # Detection models
        self.porcupine_detector = None
        self.simple_detector = SimpleWakeWordDetector(self.wake_words)
        
        # Adaptive threshold management
        if enable_adaptive_threshold:
            self.threshold_manager = AdaptiveThresholdManager(confidence_threshold)
        else:
            self.threshold_manager = None
        
        # Statistics
        self.stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'false_positives': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0,
            'models_used': {},
            'last_detection_time': 0.0
        }
        
        # Multi-model detection
        self.detection_models = []
        self._initialize_detection_models(porcupine_access_key)
        
        # Noise filtering
        self.noise_threshold = 0.01
        self.energy_threshold = 0.005
        
        logger.info(f"🎯 Enterprise Wake Word Detector initialized with {len(self.detection_models)} models")
        logger.info(f"   Wake words: {', '.join(self.wake_words)}")
        logger.info(f"   Adaptive threshold: {enable_adaptive_threshold}")
    
    def _initialize_detection_models(self, porcupine_access_key: Optional[str]):
        """Initialize available detection models"""
        
        # 1. Porcupine (if available and configured)
        if PORCUPINE_AVAILABLE and porcupine_access_key:
            try:
                # Check for custom keyword files
                keyword_paths = []
                if self.model_path and self.model_path.exists():
                    for wake_word in self.wake_words:
                        keyword_file = self.model_path / f"{wake_word}.ppn"
                        if keyword_file.exists():
                            keyword_paths.append(str(keyword_file))
                
                if keyword_paths:
                    self.porcupine_detector = PorcupineWakeWordDetector(
                        access_key=porcupine_access_key,
                        keyword_paths=keyword_paths,
                        sensitivity=0.7
                    )
                    self.detection_models.append(("porcupine", self.porcupine_detector))
                    logger.info("✅ Porcupine detector added to models")
                else:
                    logger.warning("Porcupine available but no keyword files found")
            except Exception as e:
                logger.warning(f"Porcupine initialization failed: {e}")
        
        # 2. Simple text detector (always available as fallback)
        self.detection_models.append(("simple", self.simple_detector))
        logger.info("✅ Simple text detector added to models")
    
    def detect_wake_word(self, audio_chunk: np.ndarray, transcribed_text: Optional[str] = None) -> WakeWordResult:
        """
        Multi-model wake word detection
        
        Args:
            audio_chunk: Audio data for audio-based detection
            transcribed_text: Transcribed text for text-based detection
            
        Returns:
            WakeWordResult with detection information
        """
        start_time = time.time()
        
        try:
            # Pre-filtering
            if self.enable_noise_filtering and not self._passes_noise_filter(audio_chunk):
                return WakeWordResult(
                    detected=False,
                    confidence=0.0,
                    keyword="",
                    timestamp=time.time(),
                    model_used="noise_filtered",
                    processing_time=time.time() - start_time
                )
            
            # Try each detection model
            best_result = None
            best_confidence = 0.0
            
            for model_name, detector in self.detection_models:
                try:
                    if model_name == "porcupine" and audio_chunk is not None:
                        # Audio-based detection
                        result = detector.detect(audio_chunk)
                    elif model_name == "simple" and transcribed_text:
                        # Text-based detection
                        result = detector.detect(transcribed_text)
                    else:
                        continue
                    
                    # Track model usage
                    if model_name not in self.stats['models_used']:
                        self.stats['models_used'][model_name] = 0
                    self.stats['models_used'][model_name] += 1
                    
                    # Keep best result
                    if result.confidence > best_confidence:
                        best_confidence = result.confidence
                        best_result = result
                        
                except Exception as e:
                    logger.error(f"Detection model {model_name} error: {e}")
                    continue
            
            # Use simple detector as final fallback
            if not best_result and transcribed_text:
                best_result = self.simple_detector.detect(transcribed_text)
            
            # Apply adaptive threshold
            current_threshold = self.confidence_threshold
            if self.threshold_manager:
                current_threshold = self.threshold_manager.get_threshold()
            
            # Final detection decision
            if best_result and best_result.confidence >= current_threshold:
                detected = True
                self.stats['successful_detections'] += 1
                self.stats['last_detection_time'] = time.time()
                
                # Log detection
                logger.info(f"🔔 Wake word '{best_result.keyword}' detected!")
                logger.info(f"   Confidence: {best_result.confidence:.2f}")
                logger.info(f"   Model: {best_result.model_used}")
                logger.info(f"   Threshold: {current_threshold:.2f}")
            else:
                detected = False
            
            # Update statistics
            self.stats['total_detections'] += 1
            if best_result:
                self._update_stats(best_result, detected)
            
            # Return result
            if best_result:
                best_result.detected = detected
                return best_result
            else:
                return WakeWordResult(
                    detected=False,
                    confidence=0.0,
                    keyword="",
                    timestamp=time.time(),
                    model_used="no_detection",
                    processing_time=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return WakeWordResult(
                detected=False,
                confidence=0.0,
                keyword="",
                timestamp=time.time(),
                model_used="error",
                processing_time=time.time() - start_time
            )
    
    def _passes_noise_filter(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio passes noise filtering"""
        try:
            # Energy-based filtering
            energy = np.sqrt(np.mean(audio_chunk ** 2))
            if energy < self.energy_threshold:
                return False
            
            # Simple noise detection (very basic)
            # Check for excessive noise (might indicate interference)
            if energy > 1.0:  # Clipping or very loud noise
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Noise filter error: {e}")
            return True  # Pass through on error
    
    def _update_stats(self, result: WakeWordResult, was_detected: bool):
        """Update detection statistics"""
        try:
            # Update average confidence
            total_conf = self.stats['average_confidence'] * (self.stats['total_detections'] - 1)
            self.stats['average_confidence'] = (total_conf + result.confidence) / self.stats['total_detections']
            
            # Update average processing time
            total_time = self.stats['average_processing_time'] * (self.stats['total_detections'] - 1)
            self.stats['average_processing_time'] = (total_time + result.processing_time) / self.stats['total_detections']
            
            # Update adaptive threshold if enabled
            if self.threshold_manager and was_detected:
                # For now, assume all detections are true positives
                # In a real system, you'd have user feedback
                self.threshold_manager.update_threshold(result.confidence, True)
            
        except Exception as e:
            logger.error(f"Stats update error: {e}")
    
    def report_false_positive(self):
        """Report a false positive detection for adaptive learning"""
        self.stats['false_positives'] += 1
        if self.threshold_manager:
            # Increase threshold to reduce false positives
            self.threshold_manager.update_threshold(self.confidence_threshold, False)
        logger.info("🚫 False positive reported, adjusting threshold")
    
    def get_status(self) -> Dict[str, Any]:
        """Get wake word detector status"""
        status = {
            'initialized': len(self.detection_models) > 0,
            'wake_words': self.wake_words,
            'confidence_threshold': self.confidence_threshold,
            'adaptive_threshold_enabled': self.enable_adaptive_threshold,
            'available_models': [name for name, _ in self.detection_models],
            'statistics': self.stats.copy()
        }
        
        if self.threshold_manager:
            status['adaptive_threshold'] = self.threshold_manager.get_stats()
        
        return status
    
    def cleanup(self):
        """Cleanup wake word detector"""
        try:
            if self.porcupine_detector:
                self.porcupine_detector.cleanup()
            logger.info("🧹 Enterprise Wake Word Detector cleanup completed")
        except Exception as e:
            logger.error(f"Wake word detector cleanup failed: {e}")


# Export classes
__all__ = [
    'EnterpriseWakeWordDetector',
    'WakeWordResult',
    'SimpleWakeWordDetector',
    'PorcupineWakeWordDetector',
    'AdaptiveThresholdManager'
]