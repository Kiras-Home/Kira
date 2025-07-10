"""
Verbesserte Wake Word Detection mit Machine Learning Ansatz
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import librosa
from scipy import signal
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class ImprovedDetectionResult:
    """Verbessertes Detection Result"""
    detected: bool
    confidence: float
    wake_word: str
    detection_time: float
    method: str
    audio_features: Dict[str, Any]
    probability_score: float
    threshold_used: float

class ImprovedWakeWordDetector:
    """
    Verbesserte Wake Word Detection mit:
    - MFCC Feature Extraction
    - Template Matching
    - Energy-based Pre-filtering
    - Confidence Scoring
    """
    
    def __init__(self, 
                 wake_word: str = "kira", 
                 sensitivity: float = 0.7,
                 sample_rate: int = 16000):
        self.wake_word = wake_word.lower()
        self.sensitivity = sensitivity
        self.sample_rate = sample_rate
        
        # ✅ MFCC Configuration
        self.mfcc_config = {
            'n_mfcc': 13,
            'n_fft': 2048,
            'hop_length': 512,
            'n_mels': 40,
            'fmin': 80,
            'fmax': 8000
        }
        
        # ✅ Detection Thresholds
        self.energy_threshold = 0.01
        self.mfcc_similarity_threshold = sensitivity
        self.duration_range = (0.3, 1.0)  # Kira sollte 0.3-1.0s dauern
        
        # ✅ Template Storage (würde normalerweise trainiert)
        self.wake_word_templates = self._create_kira_templates()
        
        logger.info(f"🔊 Improved Wake Word Detector: '{wake_word}' (sensitivity: {sensitivity})")
    
    def _create_kira_templates(self) -> List[np.ndarray]:
        """
        Erstellt MFCC-Templates für 'Kira'
        In echter Implementierung würden diese aus Training-Daten erstellt
        """
        # ✅ SIMULIERTE KIRA MFCC TEMPLATES
        # Basierend auf typischen deutschen Aussprache-Patterns
        templates = []
        
        # Template 1: Standard Aussprache
        template1 = np.array([
            # Ki- (hohe Frequenz, kurz)
            [2.5, -1.2, 0.8, -0.3, 0.1, -0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.8, -0.9, 0.6, -0.2, 0.08, -0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # -ra (tiefere Frequenz, länger)
            [1.2, -0.5, 0.3, -0.1, 0.05, -0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, -0.3, 0.2, -0.05, 0.02, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        templates.append(template1)
        
        # Template 2: Etwas andere Betonung
        template2 = np.array([
            [2.2, -1.0, 0.7, -0.25, 0.09, -0.04, 0.015, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.5, -0.7, 0.5, -0.15, 0.06, -0.025, 0.008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, -0.4, 0.25, -0.08, 0.03, -0.015, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, -0.2, 0.15, -0.03, 0.01, -0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        templates.append(template2)
        
        return templates
    
    def detect_wake_word(self, audio_data: np.ndarray, sample_rate: int = None) -> ImprovedDetectionResult:
        """
        Verbesserte Wake Word Detection
        """
        start_time = time.time()
        
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            # ✅ 1. PREPROCESSING
            processed_audio = self._preprocess_audio(audio_data, sample_rate)
            if processed_audio is None:
                return ImprovedDetectionResult(
                    detected=False, confidence=0.0, wake_word=self.wake_word,
                    detection_time=time.time() - start_time, method="preprocessing_failed",
                    audio_features={}, probability_score=0.0, threshold_used=self.sensitivity
                )
            
            # ✅ 2. ENERGY-BASED PRE-FILTERING
            if not self._has_sufficient_energy(processed_audio):
                return ImprovedDetectionResult(
                    detected=False, confidence=0.0, wake_word=self.wake_word,
                    detection_time=time.time() - start_time, method="insufficient_energy",
                    audio_features={'energy': 'too_low'}, probability_score=0.0, 
                    threshold_used=self.sensitivity
                )
            
            # ✅ 3. MFCC FEATURE EXTRACTION
            mfcc_features = self._extract_mfcc_features(processed_audio, sample_rate)
            if mfcc_features is None:
                return ImprovedDetectionResult(
                    detected=False, confidence=0.0, wake_word=self.wake_word,
                    detection_time=time.time() - start_time, method="mfcc_extraction_failed",
                    audio_features={}, probability_score=0.0, threshold_used=self.sensitivity
                )
            
            # ✅ 4. TEMPLATE MATCHING
            best_similarity, best_confidence = self._match_templates(mfcc_features)
            
            # ✅ 5. DURATION CHECK
            duration = len(processed_audio) / sample_rate
            duration_valid = self.duration_range[0] <= duration <= self.duration_range[1]
            
            # ✅ 6. CONFIDENCE CALCULATION
            final_confidence = self._calculate_final_confidence(
                mfcc_similarity=best_similarity,
                duration_valid=duration_valid,
                energy_level=np.sqrt(np.mean(processed_audio**2)),
                mfcc_features=mfcc_features
            )
            
            # ✅ 7. DETECTION DECISION
            detected = final_confidence >= self.sensitivity
            
            # ✅ 8. DETAILED AUDIO FEATURES
            audio_features = {
                'duration': duration,
                'energy_rms': float(np.sqrt(np.mean(processed_audio**2))),
                'mfcc_shape': mfcc_features.shape,
                'template_similarity': float(best_similarity),
                'duration_valid': duration_valid,
                'spectral_features': self._extract_spectral_features(processed_audio, sample_rate)
            }
            
            result = ImprovedDetectionResult(
                detected=detected,
                confidence=final_confidence,
                wake_word=self.wake_word,
                detection_time=time.time() - start_time,
                method="improved_mfcc_template_matching",
                audio_features=audio_features,
                probability_score=best_similarity,
                threshold_used=self.sensitivity
            )
            
            logger.info(f"🔊 Wake word detection: {detected} (confidence: {final_confidence:.3f}, "
                       f"similarity: {best_similarity:.3f}, duration: {duration:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Improved wake word detection failed: {e}")
            return ImprovedDetectionResult(
                detected=False, confidence=0.0, wake_word=self.wake_word,
                detection_time=time.time() - start_time, method="error",
                audio_features={'error': str(e)}, probability_score=0.0,
                threshold_used=self.sensitivity
            )
    
    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Verbesserte Audio-Vorverarbeitung"""
        try:
            # Resample falls nötig
            if sample_rate != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # Normalisierung
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # High-pass Filter (entfernt tiefe Störgeräusche)
            sos = signal.butter(4, 80, btype='highpass', fs=self.sample_rate, output='sos')
            audio_data = signal.sosfilt(sos, audio_data)
            
            # Pre-emphasis (verstärkt hohe Frequenzen)
            audio_data = signal.lfilter([1, -0.97], [1], audio_data)
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return None
    
    def _has_sufficient_energy(self, audio_data: np.ndarray) -> bool:
        """Prüft ob Audio genügend Energie für Sprache hat"""
        rms_energy = np.sqrt(np.mean(audio_data**2))
        return rms_energy > self.energy_threshold
    
    def _extract_mfcc_features(self, audio_data: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Extrahiert MFCC Features"""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                **self.mfcc_config
            )
            
            # Delta features (Änderungsrate)
            mfcc_delta = librosa.feature.delta(mfcc)
            
            # Kombiniere MFCC + Delta
            features = np.vstack([mfcc, mfcc_delta])
            
            # Normalisierung per Feature
            features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-6)
            
            return features.T  # (time_frames, features)
            
        except Exception as e:
            logger.error(f"MFCC extraction failed: {e}")
            return None
    
    def _match_templates(self, mfcc_features: np.ndarray) -> tuple:
        """Template Matching mit DTW (Dynamic Time Warping)"""
        try:
            best_similarity = 0.0
            best_confidence = 0.0
            
            for template in self.wake_word_templates:
                # Einfache Distanz (in echter Implementierung: DTW)
                min_len = min(len(mfcc_features), len(template))
                
                if min_len > 0:
                    # Feature-weise Cosine Similarity
                    similarities = []
                    for i in range(min_len):
                        mfcc_frame = mfcc_features[i][:template.shape[1]]  # Erste 13 MFCCs
                        template_frame = template[min(i, len(template)-1)]
                        
                        # Cosine Similarity
                        dot_product = np.dot(mfcc_frame, template_frame)
                        norm_product = np.linalg.norm(mfcc_frame) * np.linalg.norm(template_frame)
                        
                        if norm_product > 0:
                            similarity = dot_product / norm_product
                            similarities.append(max(0, similarity))  # Nur positive
                    
                    if similarities:
                        avg_similarity = np.mean(similarities)
                        if avg_similarity > best_similarity:
                            best_similarity = avg_similarity
                            best_confidence = avg_similarity
            
            return best_similarity, best_confidence
            
        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            return 0.0, 0.0
    
    def _calculate_final_confidence(self, mfcc_similarity: float, duration_valid: bool, 
                                  energy_level: float, mfcc_features: np.ndarray) -> float:
        """Berechnet finale Confidence mit mehreren Faktoren"""
        try:
            confidence = mfcc_similarity * 0.7  # MFCC ist Hauptfaktor
            
            # Duration Bonus
            if duration_valid:
                confidence += 0.1
            
            # Energy Bonus
            if energy_level > self.energy_threshold * 2:
                confidence += 0.1
            
            # Feature Consistency Bonus
            if mfcc_features is not None and len(mfcc_features) > 2:
                # Prüfe ob Features konsistent sind (nicht zu viel Variation)
                feature_std = np.std(mfcc_features, axis=0)
                consistency = 1.0 / (1.0 + np.mean(feature_std))
                confidence += consistency * 0.1
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return mfcc_similarity
    
    def _extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extrahiert zusätzliche spektrale Features"""
        try:
            # Spektrales Zentrum
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # Spektrale Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'zero_crossing_rate_mean': float(np.mean(zcr)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff))
            }
            
        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}")
            return {}

# ✅ INTEGRATION HELPER
def create_improved_wake_detector(config: dict = None) -> ImprovedWakeWordDetector:
    """Factory function für improved wake detector"""
    if config is None:
        config = {
            'wake_word': 'kira',
            'sensitivity': 0.7,
            'sample_rate': 16000
        }
    
    return ImprovedWakeWordDetector(**config)