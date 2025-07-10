"""
Verbesserter Wake Word Detector für Kira
Verwendet Audio-Pattern Recognition ohne ML-Overhead
"""

import numpy as np
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import scipy.signal as signal
from collections import deque

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Ergebnis der Wake Word Detection"""
    detected: bool
    confidence: float
    wake_word: Optional[str]
    detection_time: float
    method: str
    audio_stats: Dict[str, float] = None

class SimpleWakeWordDetector:
    """Verbesserter Wake Word Detector ohne ML-Dependencies"""
    
    def __init__(self, wake_word: str = "kira", sensitivity: float = 0.6):
        self.wake_word = wake_word.lower()
        self.sensitivity = sensitivity
        self.sample_rate = 16000
        
        # Verbesserte Detection Historie mit Deque
        self.detection_history = deque(maxlen=10)
        
        # Optimierte Audio-Pattern für "Kira"
        self.kira_patterns = {
            'duration_range': (0.4, 0.8),     # Typische Dauer für "Kira"
            'syllable_count': 2,              # Zwei Silben: Ki-ra
            'freq_range': (150, 400),         # Typischer Frequenzbereich für Sprache
            'energy_threshold': 0.01,         # Minimum Energie
            'silence_threshold': 0.1,         # Stille-Erkennungsschwelle
            'peak_distance': 0.15             # Minimaler Abstand zwischen Silben
        }
        
        logger.info(f"🎤 Wake Word Detector initialisiert: '{wake_word}' (Sensitivität: {sensitivity})")

    def detect_wake_word(self, audio_data: np.ndarray, sample_rate: int = 16000) -> DetectionResult:
        """Hauptmethode für Wake Word Erkennung"""
        start_time = time.time()
        
        try:
            # Normalisierung
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
            
            # Audio Analyse
            audio_stats = self._analyze_audio(audio_data, sample_rate)
            
            if not audio_stats:
                return DetectionResult(False, 0.0, None, time.time() - start_time, "error")
            
            # Pattern Detection
            confidence = self._pattern_detection(audio_stats)
            
            # Historie-basierte Verbesserung
            history_boost = self._evaluate_history()
            final_confidence = min(1.0, confidence + history_boost)
            
            # Detection Entscheidung
            detected = final_confidence >= self.sensitivity
            
            # Update Historie
            self._update_history(final_confidence, detected)
            
            result = DetectionResult(
                detected=detected,
                confidence=final_confidence,
                wake_word=self.wake_word if detected else None,
                detection_time=time.time() - start_time,
                method="enhanced_pattern",
                audio_stats=audio_stats
            )
            
            if detected:
                logger.info(f"✅ Wake Word erkannt! (Confidence: {final_confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Detection Fehler: {e}")
            return DetectionResult(False, 0.0, None, time.time() - start_time, "error")

    def _analyze_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Verbesserte Audio-Analyse"""
        try:
            # Basis-Analyse
            duration = len(audio_data) / sample_rate
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Fensterung
            window = signal.hamming(len(audio_data))
            windowed_audio = audio_data * window
            
            # Spektralanalyse
            frequencies, times, spectrogram = signal.spectrogram(
                windowed_audio,
                fs=sample_rate,
                nperseg=int(sample_rate * 0.025),  # 25ms Fenster
                noverlap=int(sample_rate * 0.015)  # 15ms Überlappung
            )
            
            # Energie-Analyse
            energy_envelope = np.sum(spectrogram, axis=0)
            peaks, _ = signal.find_peaks(
                energy_envelope,
                distance=int(self.kira_patterns['peak_distance'] * len(energy_envelope)),
                height=np.mean(energy_envelope) * 0.5
            )
            
            # Frequenz-Analyse
            dominant_freq = frequencies[np.argmax(np.mean(spectrogram, axis=1))]
            speech_mask = (frequencies >= 150) & (frequencies <= 400)
            speech_energy = np.sum(spectrogram[speech_mask, :])
            
            # Spektrale Flachheit
            spectral_flatness = self._compute_spectral_flatness(spectrogram)
            
            return {
                'duration': duration,
                'rms': rms,
                'dominant_frequency': dominant_freq,
                'speech_energy': speech_energy,
                'syllable_count': len(peaks),
                'spectral_flatness': spectral_flatness,
                'energy_envelope': energy_envelope.tolist(),
                'peak_positions': (peaks / len(energy_envelope) * duration).tolist()
            }
            
        except Exception as e:
            logger.error(f"❌ Audio Analyse Fehler: {e}")
            return {}

    def _compute_spectral_flatness(self, spectrogram: np.ndarray) -> float:
        """Berechnet spektrale Flachheit"""
        spectrum = np.mean(spectrogram, axis=1) + 1e-10
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        return geometric_mean / arithmetic_mean

    def _pattern_detection(self, audio_stats: Dict[str, float]) -> float:
        """Verbesserte Pattern-Detection"""
        if not audio_stats:
            return 0.0
            
        try:
            scores = []
            
            # 1. Dauer-Check
            duration = audio_stats['duration']
            min_dur, max_dur = self.kira_patterns['duration_range']
            if min_dur <= duration <= max_dur:
                scores.append(1.0)
            else:
                scores.append(max(0, 1 - abs(duration - 0.6) / 0.4))
            
            # 2. Silben-Check
            syllables = audio_stats['syllable_count']
            scores.append(1.0 if syllables == 2 else 0.5 if syllables == 3 else 0.0)
            
            # 3. Frequenz-Check
            freq = audio_stats['dominant_frequency']
            min_freq, max_freq = self.kira_patterns['freq_range']
            if min_freq <= freq <= max_freq:
                scores.append(1.0)
            else:
                scores.append(max(0, 1 - abs(freq - 275) / 275))
            
            # 4. Energie-Check
            energy = audio_stats['speech_energy']
            energy_score = min(energy / 100, 1.0)
            scores.append(energy_score)
            
            # 5. Spektrale Flachheit
            flatness = audio_stats['spectral_flatness']
            if 0.1 <= flatness <= 0.5:
                scores.append(1.0)
            else:
                scores.append(0.0)
            
            # Gewichteter Durchschnitt
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            final_score = sum(s * w for s, w in zip(scores, weights))
            
            return final_score
            
        except Exception as e:
            logger.error(f"❌ Pattern Detection Fehler: {e}")
            return 0.0

    def _evaluate_history(self) -> float:
        """Verbesserte Historie-Auswertung"""
        if len(self.detection_history) < 2:
            return 0.0
            
        try:
            recent = list(self.detection_history)[-3:]
            confidences = [h['confidence'] for h in recent]
            
            # Trend-basierter Boost
            if len(confidences) >= 2 and all(c > 0.3 for c in confidences):
                trend = (confidences[-1] - confidences[0]) / len(confidences)
                if trend > 0:
                    return min(0.1, trend)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Historie Auswertung Fehler: {e}")
            return 0.0

    def _update_history(self, confidence: float, detected: bool):
        """Update Detection-Historie"""
        try:
            self.detection_history.append({
                'timestamp': time.time(),
                'confidence': confidence,
                'detected': detected
            })
        except Exception as e:
            logger.error(f"❌ Historie Update Fehler: {e}")

    def calibrate(self, audio_samples: List[np.ndarray], sample_rate: int) -> Dict[str, Any]:
        """Kalibriert den Detector"""
        logger.info("🎯 Starte Kalibrierung...")
        
        results = []
        for i, sample in enumerate(audio_samples):
            stats = self._analyze_audio(sample, sample_rate)
            score = self._pattern_detection(stats)
            results.append({'stats': stats, 'score': score})
            logger.info(f"Sample {i+1}: Score = {score:.3f}")
        
        scores = [r['score'] for r in results]
        suggested_sensitivity = np.percentile(scores, 25)
        
        self.sensitivity = suggested_sensitivity
        
        return {
            'samples_analyzed': len(results),
            'average_score': np.mean(scores),
            'suggested_sensitivity': suggested_sensitivity,
            'score_distribution': {
                'min': min(scores),
                'max': max(scores),
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
        }

    def reset(self):
        """Reset Detector"""
        self.detection_history.clear()
        logger.info("🔄 Detector zurückgesetzt")

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Detector Statistiken zurück"""
        history = list(self.detection_history)
        total = len(history)
        detected = sum(1 for h in history if h['detected'])
        
        return {
            'wake_word': self.wake_word,
            'sensitivity': self.sensitivity,
            'total_detections': total,
            'successful_detections': detected,
            'success_rate': detected / total if total > 0 else 0,
            'average_confidence': np.mean([h['confidence'] for h in history]) if history else 0
        }

def test_detector():
    """Test-Funktion"""
    from ..audio.recorder import AudioRecorder
    
    print("\n🎤 === WAKE WORD DETECTOR TEST ===")
    
    detector = SimpleWakeWordDetector("kira", sensitivity=0.6)
    recorder = AudioRecorder(sample_rate=16000)
    
    try:
        print(f"\nSage '{detector.wake_word}' (3 Sekunden)...")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
            
        print("\n🎤 AUFNAHME LÄUFT...")
        audio = recorder.record(3.0)
        
        if not audio.success:
            print(f"❌ Aufnahme fehlgeschlagen: {audio.error}")
            return
            
        print("✅ Aufnahme erfolgreich")
        
        result = detector.detect_wake_word(audio.data, audio.sample_rate)
        
        print("\n📊 ERGEBNIS:")
        print(f"Erkannt: {result.detected}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Zeit: {result.detection_time:.3f}s")
        
        if result.audio_stats:
            print("\n📈 AUDIO ANALYSE:")
            stats = result.audio_stats
            print(f"Dauer: {stats['duration']:.2f}s")
            print(f"Dominante Frequenz: {stats['dominant_frequency']:.0f}Hz")
            print(f"Silben: {stats['syllable_count']}")
            
        return result.detected
        
    except Exception as e:
        print(f"❌ Test Fehler: {e}")
        return False

if __name__ == "__main__":
    test_detector()