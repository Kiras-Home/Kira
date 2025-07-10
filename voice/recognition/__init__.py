"""
Recognition Module für Kira Voice System
Enthält Speech Recognition (Whisper) und Wake Word Detection
"""

import logging

logger = logging.getLogger(__name__)

# Import Speech Recognition
try:
    from .whisper_engine import WhisperEngine
    logger.info("✅ Whisper Engine geladen")
    WHISPER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Whisper Engine Import Fehler: {e}")
    WhisperEngine = None
    WHISPER_AVAILABLE = False

# Import Wake Word Detection
try:
    from .simple_detector import SimpleWakeWordDetector, DetectionResult
    logger.info("✅ Simple Wake Word Detector geladen")
    WAKE_WORD_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Wake Word Detector Import Fehler: {e}")
    SimpleWakeWordDetector = None
    DetectionResult = None
    WAKE_WORD_AVAILABLE = False

# Combined Recognition System
class KiraRecognitionSystem:
    """Kombiniertes Recognition System: Wake Word + Speech Recognition"""
    
    def __init__(self, 
                 wake_word: str = "kira", 
                 wake_sensitivity: float = 0.6,
                 whisper_model: str = "base",
                 language: str = "de"):
        
        self.wake_word = wake_word
        self.wake_sensitivity = wake_sensitivity
        self.whisper_model = whisper_model
        self.language = language
        
        # Engines
        self.wake_detector = None
        self.speech_engine = None
        
        self.is_initialized = False
        
        logger.info(f"🎯 Kira Recognition System: Wake='{wake_word}', STT={whisper_model}")
    
    def initialize(self) -> bool:
        """Initialisiert Recognition System"""
        try:
            logger.info("🚀 Initialisiere Kira Recognition System...")
            
            success = True
            
            # Wake Word Detector
            if WAKE_WORD_AVAILABLE:
                self.wake_detector = SimpleWakeWordDetector(
                    wake_word=self.wake_word,
                    sensitivity=self.wake_sensitivity
                )
                logger.info("✅ Wake Word Detector bereit")
            else:
                logger.warning("⚠️ Wake Word Detector nicht verfügbar")
                success = False
            
            # Speech Recognition
            if WHISPER_AVAILABLE:
                self.speech_engine = WhisperEngine(
                    model_size=self.whisper_model,
                    language=self.language
                )
                
                if self.speech_engine.initialize():
                    logger.info("✅ Speech Recognition bereit")
                else:
                    logger.error("❌ Speech Recognition Initialisierung fehlgeschlagen")
                    success = False
            else:
                logger.error("❌ Speech Recognition nicht verfügbar")
                success = False
            
            if success:
                self.is_initialized = True
                logger.info("🎉 Kira Recognition System vollständig initialisiert!")
                return True
            else:
                logger.error("❌ Kira Recognition System Initialisierung fehlgeschlagen")
                return False
                
        except Exception as e:
            logger.error(f"❌ Recognition System Initialisierung Fehler: {e}")
            return False
    
    def detect_wake_word(self, audio_data, sample_rate: int = 16000):
        """Erkennt Wake Word in Audio"""
        if not self.wake_detector:
            return DetectionResult(
                detected=False, 
                confidence=0.0, 
                wake_word=None, 
                detection_time=0.0, 
                method="not_available"
            )
        
        return self.wake_detector.detect_wake_word(audio_data, sample_rate)
    
    def transcribe_speech(self, audio_data, sample_rate: int = 16000):
        """Transkribiert Sprache zu Text"""
        if not self.speech_engine:
            return None
        
        return self.speech_engine.transcribe(audio_data, sample_rate)
    
    def transcribe_with_confidence(self, audio_data, sample_rate: int = 16000):
        """Transkribiert mit Confidence-Informationen"""
        if not self.speech_engine:
            return {
                'text': None,
                'success': False,
                'confidence': 0.0,
                'error': 'Speech Engine nicht verfügbar'
            }
        
        return self.speech_engine.transcribe_with_confidence(audio_data, sample_rate)
    
    def get_system_info(self) -> dict:
        """Gibt System-Informationen zurück"""
        return {
            'whisper_available': WHISPER_AVAILABLE,
            'wake_word_available': WAKE_WORD_AVAILABLE,
            'is_initialized': self.is_initialized,
            'wake_word': self.wake_word,
            'wake_sensitivity': self.wake_sensitivity,
            'whisper_model': self.whisper_model,
            'language': self.language
        }
    
    def get_performance_stats(self) -> dict:
        """Gibt Performance-Statistiken zurück"""
        stats = {}
        
        if self.speech_engine:
            stats['speech'] = self.speech_engine.get_performance_stats()
        
        if self.wake_detector:
            stats['wake_detection'] = self.wake_detector.get_detection_stats()
        
        return stats
    
    def cleanup(self):
        """Cleanup Recognition System"""
        try:
            logger.info("🧹 Recognition System Cleanup...")
            
            if self.speech_engine:
                self.speech_engine.cleanup()
            
            if self.wake_detector:
                self.wake_detector.reset_history()
            
            self.is_initialized = False
            logger.info("✅ Recognition System Cleanup abgeschlossen")
            
        except Exception as e:
            logger.error(f"❌ Recognition Cleanup Fehler: {e}")

# Test-Funktion für komplettes Recognition System
def test_recognition_system():
    """Testet das komplette Recognition System"""
    print("🎯 === KIRA RECOGNITION SYSTEM TEST ===")
    
    # Initialisiere System
    recognition = KiraRecognitionSystem(
        wake_word="kira",
        wake_sensitivity=0.6,
        whisper_model="base",
        language="de"
    )
    
    if not recognition.initialize():
        print("❌ Recognition System Initialisierung fehlgeschlagen")
        return False
    
    # System Info
    info = recognition.get_system_info()
    print(f"Whisper verfügbar: {info['whisper_available']}")
    print(f"Wake Word verfügbar: {info['wake_word_available']}")
    print(f"Wake Word: {info['wake_word']}")
    print(f"Sensitivität: {info['wake_sensitivity']}")
    
    try:
        from ..audio.recorder import SimpleAudioRecorder
        
        recorder = SimpleAudioRecorder()
        
        # Test 1: Wake Word Detection
        print(f"\n🔍 TEST 1: Wake Word Detection")
        print(f"Sagen Sie '{info['wake_word']}' (3 Sekunden)...")
        
        for i in range(3, 0, -1):
            print(f"Start in {i}...")
            time.sleep(1)
        
        audio_result = recorder.record(3.0)
        
        if audio_result.success:
            wake_result = recognition.detect_wake_word(
                audio_result.data, 
                audio_result.sample_rate
            )
            
            print(f"Wake Word erkannt: {wake_result.detected}")
            print(f"Confidence: {wake_result.confidence:.3f}")
        
        # Test 2: Speech Recognition
        print(f"\n🎧 TEST 2: Speech Recognition")
        print("Sprechen Sie einen Satz (5 Sekunden)...")
        
        for i in range(3, 0, -1):
            print(f"Start in {i}...")
            time.sleep(1)
        
        audio_result2 = recorder.record(5.0)
        
        if audio_result2.success:
            speech_result = recognition.transcribe_with_confidence(
                audio_result2.data,
                audio_result2.sample_rate
            )
            
            print(f"Erkannter Text: '{speech_result['text']}'")
            print(f"Confidence: {speech_result['confidence']:.3f}")
            print(f"Erfolg: {speech_result['success']}")
        
        # Performance Statistiken
        print(f"\n📊 PERFORMANCE STATISTIKEN:")
        stats = recognition.get_performance_stats()
        
        if 'speech' in stats:
            speech_stats = stats['speech']
            print(f"Speech Recognition:")
            print(f"   Erkennungen: {speech_stats['recognition_count']}")
            print(f"   Ø Zeit: {speech_stats['average_time']:.2f}s")
        
        if 'wake_detection' in stats:
            wake_stats = stats['wake_detection']
            print(f"Wake Word Detection:")
            print(f"   Tests: {wake_stats['total_detections']}")
            print(f"   Erfolgsrate: {wake_stats['success_rate']:.2%}")
        
        recognition.cleanup()
        
        print(f"\n🎉 Recognition System Test abgeschlossen!")
        return True
        
    except ImportError:
        print("❌ Audio Recorder nicht verfügbar")
        recognition.cleanup()
        return False
    except Exception as e:
        print(f"❌ Test Fehler: {e}")
        recognition.cleanup() 
        return False

# Export
__all__ = [
    'WhisperEngine',
    'SimpleWakeWordDetector',
    'DetectionResult',
    'KiraRecognitionSystem',
    'WHISPER_AVAILABLE',
    'WAKE_WORD_AVAILABLE',
    'test_recognition_system'
]

# Log beim Import
logger.info("📦 Kira Recognition Module geladen")