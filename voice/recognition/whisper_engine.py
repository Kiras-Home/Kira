"""
Optimierte Whisper Speech Recognition Engine für Kira
"""

import numpy as np
import logging
import time
import whisper
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
import os

# Logging Setup
logging.basicConfig(level=logging.INFO)
__whisper_engien__ = "voice.recognition.whisper_engine"
logger = logging.getLogger(__whisper_engien__)

def check_mps_compatibility() -> bool:
    """
    Prüft MPS-Kompatibilität für Whisper
    Retourniert True wenn MPS sicher verwendet werden kann
    """
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return False
    
    try:
        # Teste sparse tensor operations (häufiges Problem mit Whisper auf MPS)
        test_tensor = torch.sparse_coo_tensor(
            indices=torch.zeros(1, 1, dtype=torch.long),
            values=torch.ones(1),
            size=(1,),
            device='mps'
        )
        
        # Teste grundlegende Operationen
        test_dense = torch.randn(10, 10, device='mps')
        _ = test_dense @ test_dense.T
        
        return True
    except Exception as e:
        logger.debug(f"MPS-Kompatibilitätstest fehlgeschlagen: {e}")
        return False

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class WhisperResult:
    """Whisper Ergebnis-Klasse"""
    success: bool
    text: Optional[str] = None
    language: Optional[str] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    error: Optional[str] = None

class WhisperEngine:
    """Optimierte Whisper Speech Recognition Engine"""
    
    def __init__(self, 
                 model_size: str = "base", 
                 language: str = "de",
                 device: str = None,
                 compute_type: str = "float32"):
        
        self.model_size = model_size
        self.language = language
        self.model = None
        self.is_initialized = False
        
        # Optimale Device-Auswahl mit Umgebungsvariablen-Override
        if device:
            self.device = device
        else:
            # Prüfe Umgebungsvariablen für Device-Override
            env_device = os.getenv('WHISPER_DEVICE', '').lower()
            if env_device in ['cpu', 'cuda', 'mps']:
                self.device = env_device
                logger.info(f"🔧 Device per Umgebungsvariable gesetzt: {self.device}")
            else:
                self.device = self._detect_optimal_device()
        
        self.compute_type = compute_type
        
        # Cache Directory
        self.cache_dir = Path(os.getenv('WHISPER_CACHE_DIR', 'voice/models/whisper'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio preprocessing statistics
        self.audio_mean = None
        self.audio_std = None
        
        # Performance Tracking
        self.recognition_count = 0
        self.total_recognition_time = 0.0
        self.last_recognition_time = 0.0
        
        logger.info(f"🧠 Whisper Engine: {model_size} Model, Sprache: {language}, Device: {self.device}")
    
    def _detect_optimal_device(self) -> str:
        """Erkennt optimales Compute Device mit verbesserter MPS-Kompatibilitätsprüfung"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"🎯 CUDA verfügbar: {torch.cuda.get_device_name(0)}")
        elif check_mps_compatibility():
            device = "mps"
            logger.info("🎯 Apple MPS verfügbar und kompatibel")
        else:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.warning("⚠️ MPS verfügbar aber nicht kompatibel mit Whisper - verwende CPU")
            device = "cpu"
            logger.info("🎯 Verwende CPU")
        return device

    def initialize(self) -> bool:
        """Initialisiert Whisper Model mit MPS-Kompatibilitätsprüfungen"""
        try:
            start_time = time.time()
            logger.info(f"📥 Lade Whisper Model '{self.model_size}' auf {self.device}...")
            
            # Lade Model mit optimalen Einstellungen und MPS-Fallback
            try:
                self.model = whisper.load_model(
                    self.model_size,
                    device=self.device,
                    download_root=str(self.cache_dir),
                    in_memory=True
                )
            except Exception as device_error:
                if self.device == "mps":
                    logger.warning(f"⚠️ MPS-Laden fehlgeschlagen: {device_error}")
                    logger.info("🔄 Versuche Fallback auf CPU...")
                    self.device = "cpu"
                    
                    self.model = whisper.load_model(
                        self.model_size,
                        device=self.device,
                        download_root=str(self.cache_dir),
                        in_memory=True
                    )
                else:
                    raise device_error
            
            # Optimiere Model
            if self.device == "cuda":
                if self.compute_type == "float16":
                    self.model.half()  # Reduziere Präzision für bessere Performance
            
            load_time = time.time() - start_time
            logger.info(f"✅ Whisper Model geladen auf {self.device} ({load_time:.1f}s)")
            
            # Warm-up mit Test Recognition
            if self._warm_up_model():
                self.is_initialized = True
                return True
            else:
                logger.error("❌ Whisper Model Warm-up fehlgeschlagen")
                return False
            
        except Exception as e:
            logger.error(f"❌ Whisper Initialisierung fehlgeschlagen: {e}")
            return False
    
    def _warm_up_model(self) -> bool:
        """Warm-up des Models für bessere erste Performance mit MPS-Fallback"""
        try:
            # Erstelle 2 Sekunden Test-Audio
            dummy_audio = np.zeros(32000, dtype=np.float32)
            
            # Warm-up Recognition mit Fehlerbehandlung
            try:
                with torch.no_grad():
                    _ = self.model.transcribe(
                        dummy_audio,
                        language=self.language,
                        task='transcribe',
                        verbose=False,
                        fp16=False  # Deaktiviere fp16 für MPS-Kompatibilität
                    )
            except Exception as transcribe_error:
                if self.device == "mps" and "sparse" in str(transcribe_error).lower():
                    logger.warning(f"⚠️ MPS-Transkription fehlgeschlagen: {transcribe_error}")
                    logger.info("🔄 Fallback auf CPU für bessere Kompatibilität...")
                    
                    # Lade Model neu auf CPU
                    del self.model
                    self.device = "cpu"
                    
                    self.model = whisper.load_model(
                        self.model_size,
                        device=self.device,
                        download_root=str(self.cache_dir),
                        in_memory=True
                    )
                    
                    # Wiederhole Warm-up auf CPU
                    with torch.no_grad():
                        _ = self.model.transcribe(
                            dummy_audio,
                            language=self.language,
                            task='transcribe',
                            verbose=False
                        )
                else:
                    raise transcribe_error
            
            logger.info(f"✅ Whisper Model Warm-up erfolgreich auf {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Whisper Warm-up Fehler: {e}")
            return False
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> WhisperResult:
        """Transkribiert Audio zu Text"""
        if not self.is_initialized:
            return WhisperResult(
                success=False,
                error="Whisper nicht initialisiert"
            )
        
        start_time = time.time()
        
        try:
            # Audio Preprocessing
            processed_audio = self._preprocess_audio(audio_data, sample_rate)
            
            # Transkription
            result = self._transcribe_audio(processed_audio)
            
            # Verarbeitung des Ergebnisses
            if result and 'text' in result:
                text = result['text'].strip()
                confidence = result.get('confidence', 0.0)
                detected_language = result.get('language', self.language)
                
                processing_time = time.time() - start_time
                
                # Performance Tracking
                self.recognition_count += 1
                self.total_recognition_time += processing_time
                self.last_recognition_time = processing_time
                
                if text:
                    logger.info(f"✅ Erkannt ({processing_time:.2f}s): '{text}'")
                    return WhisperResult(
                        success=True,
                        text=text,
                        language=detected_language,
                        confidence=confidence,
                        processing_time=processing_time
                    )
                else:
                    logger.warning("⚠️ Keine Sprache erkannt")
                    return WhisperResult(
                        success=False,
                        error="Keine Sprache erkannt",
                        processing_time=processing_time
                    )
            
            return WhisperResult(
                success=False,
                error="Transkription fehlgeschlagen",
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Transkription Fehler: {e}")
            return WhisperResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
        
    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Optimiertes Audio Preprocessing"""
        try:
            # Normalisierung
            if len(audio_data) == 0:
                return np.zeros(16000, dtype=np.float32)
            
            audio_data = audio_data.astype(np.float32)
            
            # Resampling wenn nötig
            if sample_rate != 16000:
                # TODO: Implement resampling
                logger.warning("⚠️ Resampling nicht implementiert")
            
            # Normalisierung
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Statistiken aktualisieren
            if self.audio_mean is None:
                self.audio_mean = np.mean(audio_data)
                self.audio_std = np.std(audio_data)
            else:
                # Exponentielles Moving Average
                alpha = 0.01
                self.audio_mean = (1 - alpha) * self.audio_mean + alpha * np.mean(audio_data)
                self.audio_std = (1 - alpha) * self.audio_std + alpha * np.std(audio_data)
            
            # Standardisierung
            audio_data = (audio_data - self.audio_mean) / (self.audio_std + 1e-10)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Preprocessing Fehler: {e}")
            return audio_data
    
    def _transcribe_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Interne Transkriptionsmethode mit MPS-Kompatibilität"""
        try:
            with torch.no_grad():
                # Whisper Transkription mit Device-spezifischen Einstellungen
                result = self.model.transcribe(
                    audio_data,
                    language=self.language,
                    task='transcribe',
                    fp16=self.device == 'cuda',  # Nur fp16 für CUDA, nicht für MPS
                    verbose=False
                )
                return result
                
        except Exception as e:
            # Bei MPS-Fehlern versuche CPU-Fallback
            if self.device == "mps" and ("sparse" in str(e).lower() or "mps" in str(e).lower()):
                logger.warning(f"⚠️ MPS-Transkription fehlgeschlagen, versuche CPU-Fallback")
                try:
                    # Temporär auf CPU wechseln für diese Transkription
                    original_device = self.device
                    self.device = "cpu"
                    
                    # Model auf CPU laden falls nötig
                    if str(next(self.model.parameters()).device) != "cpu":
                        self.model = self.model.cpu()
                    
                    with torch.no_grad():
                        result = self.model.transcribe(
                            audio_data,
                            language=self.language,
                            task='transcribe',
                            verbose=False
                        )
                    
                    logger.info("✅ CPU-Fallback erfolgreich")
                    return result
                    
                except Exception as cpu_error:
                    logger.error(f"❌ Auch CPU-Fallback fehlgeschlagen: {cpu_error}")
                    return {}
            else:
                logger.error(f"❌ Whisper Transkription Fehler: {e}")
                return {}
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gibt Performance Statistiken zurück"""
        if self.recognition_count == 0:
            return {
                'average_time': 0,
                'total_time': 0,
                'recognition_count': 0,
                'last_time': 0
            }
            
        return {
            'average_time': self.total_recognition_time / self.recognition_count,
            'total_time': self.total_recognition_time,
            'recognition_count': self.recognition_count,
            'last_time': self.last_recognition_time
        }
    
    def _prepare_audio(self, audio_data: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Verbesserte Audio-Vorbereitung"""
        try:
            # Konvertierung und Validierung
            audio_data = np.asarray(audio_data, dtype=np.float32)
            
            if len(audio_data) == 0:
                logger.warning("⚠️ Leeres Audio Array")
                return None
            
            # Normalisierung wenn nötig
            max_abs = np.max(np.abs(audio_data))
            if max_abs > 1.0:
                audio_data = audio_data / max_abs
            
            # Resampling wenn nötig
            if sample_rate != 16000:
                try:
                    import librosa
                    audio_data = librosa.resample(
                        audio_data, 
                        orig_sr=sample_rate, 
                        target_sr=16000
                    )
                except ImportError:
                    logger.warning("⚠️ librosa nicht verfügbar - Resampling übersprungen")
            
            # Qualitätsprüfungen
            duration = len(audio_data) / 16000
            rms = np.sqrt(np.mean(audio_data**2))
            
            if duration < 0.1:
                logger.warning("⚠️ Audio zu kurz (<0.1s)")
                return None
                
            if rms < 0.001:
                logger.warning("⚠️ Audio zu leise")
                return None
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Audio Vorbereitung Fehler: {e}")
            return None
    
    def cleanup(self):
        """Verbesserte Cleanup-Routine"""
        try:
            logger.info("🧹 Whisper Engine Cleanup...")
            
            # Performance Report
            if self.recognition_count > 0:
                avg_time = self.total_recognition_time / self.recognition_count
                logger.info(f"📊 Statistik: {self.recognition_count} Erkennungen, "
                          f"⌀ {avg_time:.2f}s pro Erkennung")
            
            # Explizites Memory Cleanup
            if self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.model = None
            self.is_initialized = False
            
            logger.info("✅ Whisper Engine Cleanup abgeschlossen")
            
        except Exception as e:
            logger.error(f"❌ Whisper Cleanup Fehler: {e}")

def test_whisper():
    """Test-Funktion"""
    from ..audio.recorder import SimpleAudioRecorder
    
    print("\n🎤 === WHISPER ENGINE TEST ===")
    
    # Engine initialisieren
    engine = WhisperEngine(model_size="base", language="de")
    if not engine.initialize():
        print("❌ Whisper Initialisierung fehlgeschlagen")
        return
    
    # Audio aufnehmen
    recorder = SimpleAudioRecorder()
    
    try:
        print("\nSprich einen Satz (3 Sekunden)...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
            
        print("\n🎤 AUFNAHME LÄUFT...")
        audio = recorder.record(3.0)
        
        if not audio.success:
            print(f"❌ Aufnahme fehlgeschlagen: {audio.error}")
            return
            
        print("✅ Aufnahme erfolgreich")
        
        # Transkription
        print("\n🔍 Transkribiere...")
        result = engine.transcribe(audio.data)
        
        # Ergebnisse
        print("\n📊 ERGEBNIS:")
        print(f"Erfolg: {result.success}")
        print(f"Text: {result.text}")
        print(f"Sprache: {result.language}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Zeit: {result.processing_time:.2f}s")
        
        if result.error:
            print(f"Fehler: {result.error}")
        
        # Performance Stats
        stats = engine.get_performance_stats()
        print("\n⚡ PERFORMANCE:")
        print(f"Durchschnittszeit: {stats['average_time']:.2f}s")
        print(f"Erkennungen: {stats['recognition_count']}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Test Fehler: {e}")
        return False
    finally:
        engine.cleanup()

if __name__ == "__main__":
    test_whisper()