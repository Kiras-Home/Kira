"""
Audio Utilities für Kira Voice System
Hilfsfunktionen für Audio-Verarbeitung, -Analyse und -Konvertierung
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, List
import tempfile
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio-Verarbeitungsklasse mit nützlichen Funktionen"""
    
    def __init__(self):
        self.sample_rate = 16000
        logger.debug("🔧 Audio Processor initialisiert")
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray, target_level: float = 0.7) -> np.ndarray:
        """Normalisiert Audio auf Target-Level"""
        try:
            if len(audio_data) == 0:
                return audio_data
            
            # Finde Maximum
            max_val = np.max(np.abs(audio_data))
            
            if max_val == 0:
                return audio_data
            
            # Normalisiere
            normalized = audio_data * (target_level / max_val)
            
            logger.debug(f"🔧 Audio normalisiert: {max_val:.4f} → {target_level}")
            return normalized
            
        except Exception as e:
            logger.error(f"❌ Audio Normalisierung Fehler: {e}")
            return audio_data
    
    @staticmethod
    def remove_silence(audio_data: np.ndarray, 
                      sample_rate: int = 16000,
                      silence_threshold: float = 0.01,
                      min_silence_duration: float = 0.1) -> np.ndarray:
        """Entfernt Stille am Anfang und Ende"""
        try:
            if len(audio_data) == 0:
                return audio_data
            
            # Berechne RMS in kleinen Fenstern
            window_size = int(sample_rate * 0.01)  # 10ms Fenster
            
            rms_values = []
            for i in range(0, len(audio_data), window_size):
                window = audio_data[i:i+window_size]
                rms = np.sqrt(np.mean(window**2))
                rms_values.append(rms)
            
            rms_array = np.array(rms_values)
            
            # Finde Start (erstes Fenster über Threshold)
            start_idx = 0
            for i, rms in enumerate(rms_array):
                if rms > silence_threshold:
                    start_idx = max(0, i - 2)  # 2 Fenster Vorlauf
                    break
            
            # Finde Ende (letztes Fenster über Threshold)
            end_idx = len(rms_array)
            for i in range(len(rms_array) - 1, -1, -1):
                if rms_array[i] > silence_threshold:
                    end_idx = min(len(rms_array), i + 3)  # 3 Fenster Nachlauf
                    break
            
            # Konvertiere zu Audio-Indizes
            start_sample = start_idx * window_size
            end_sample = end_idx * window_size
            
            # Trimme Audio
            trimmed = audio_data[start_sample:end_sample]
            
            logger.debug(f"🔧 Stille entfernt: {len(audio_data)} → {len(trimmed)} Samples")
            return trimmed
            
        except Exception as e:
            logger.error(f"❌ Stille-Entfernung Fehler: {e}")
            return audio_data
    
    @staticmethod
    def apply_noise_gate(audio_data: np.ndarray, 
                        threshold: float = 0.01,
                        ratio: float = 10.0) -> np.ndarray:
        """Wendet Noise Gate an"""
        try:
            if len(audio_data) == 0:
                return audio_data
            
            # Berechne RMS für jedes Sample (gleitender Durchschnitt)
            window_size = 512
            half_window = window_size // 2
            
            gated_audio = audio_data.copy()
            
            for i in range(len(audio_data)):
                start = max(0, i - half_window)
                end = min(len(audio_data), i + half_window)
                window = audio_data[start:end]
                
                rms = np.sqrt(np.mean(window**2))
                
                if rms < threshold:
                    # Unterhalb Threshold: reduziere um Ratio
                    reduction = 1.0 / ratio
                    gated_audio[i] *= reduction
            
            logger.debug(f"🔧 Noise Gate angewendet (Threshold: {threshold})")
            return gated_audio
            
        except Exception as e:
            logger.error(f"❌ Noise Gate Fehler: {e}")
            return audio_data
    
    @staticmethod
    def detect_speech_segments(audio_data: np.ndarray,
                             sample_rate: int = 16000,
                             min_speech_duration: float = 0.1,
                             speech_threshold: float = 0.02) -> List[Tuple[int, int]]:
        """Erkennt Sprach-Segmente im Audio"""
        try:
            if len(audio_data) == 0:
                return []
            
            # Berechne RMS in kleinen Fenstern
            window_size = int(sample_rate * 0.02)  # 20ms Fenster
            rms_values = []
            
            for i in range(0, len(audio_data), window_size):
                window = audio_data[i:i+window_size]
                if len(window) > 0:
                    rms = np.sqrt(np.mean(window**2))
                    rms_values.append((i, rms))
            
            # Finde Sprach-Segmente
            segments = []
            in_speech = False
            speech_start = 0
            
            min_speech_samples = int(min_speech_duration * sample_rate)
            
            for sample_idx, rms in rms_values:
                if rms > speech_threshold:
                    if not in_speech:
                        speech_start = sample_idx
                        in_speech = True
                else:
                    if in_speech:
                        speech_end = sample_idx
                        
                        # Prüfe Mindestdauer
                        if speech_end - speech_start >= min_speech_samples:
                            segments.append((speech_start, speech_end))
                        
                        in_speech = False
            
            # Letztes Segment falls noch aktiv
            if in_speech:
                speech_end = len(audio_data)
                if speech_end - speech_start >= min_speech_samples:
                    segments.append((speech_start, speech_end))
            
            logger.debug(f"🔧 {len(segments)} Sprach-Segmente erkannt")
            return segments
            
        except Exception as e:
            logger.error(f"❌ Sprach-Segmentierung Fehler: {e}")
            return []
    
    @staticmethod
    def calculate_audio_features(audio_data: np.ndarray, 
                               sample_rate: int = 16000) -> Dict[str, float]:
        """Berechnet Audio-Features für Analyse"""
        try:
            if len(audio_data) == 0:
                return {}
            
            # Basis-Features
            duration = len(audio_data) / sample_rate
            rms = np.sqrt(np.mean(audio_data**2))
            max_amplitude = np.max(np.abs(audio_data))
            
            # Zero-Crossing Rate
            zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
            zcr = zero_crossings / len(audio_data)
            
            # Spectral Features (vereinfacht)
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft[:len(fft)//2])
            frequencies = np.fft.fftfreq(len(fft), 1/sample_rate)[:len(fft)//2]
            
            # Spectral Centroid (gewichteter Durchschnitt der Frequenzen)
            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
            else:
                spectral_centroid = 0.0
            
            # Spectral Rolloff (95% der Energie)
            cumulative_magnitude = np.cumsum(magnitude)
            total_magnitude = cumulative_magnitude[-1]
            
            if total_magnitude > 0:
                rolloff_idx = np.where(cumulative_magnitude >= 0.95 * total_magnitude)[0]
                spectral_rolloff = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0
            else:
                spectral_rolloff = 0.0
            
            # MFCC-ähnliche Features (vereinfacht)
            # Mel-Scale Approximation
            mel_frequencies = np.linspace(0, sample_rate//2, 13)
            mel_energies = []
            
            for i in range(len(mel_frequencies)-1):
                start_freq = mel_frequencies[i]
                end_freq = mel_frequencies[i+1]
                
                freq_mask = (frequencies >= start_freq) & (frequencies < end_freq)
                mel_energy = np.sum(magnitude[freq_mask])
                mel_energies.append(mel_energy)
            
            # Features Dictionary
            features = {
                'duration': duration,
                'rms': rms,
                'max_amplitude': max_amplitude,
                'zero_crossing_rate': zcr,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'mel_energy_mean': np.mean(mel_energies),
                'mel_energy_std': np.std(mel_energies),
                'energy': np.sum(audio_data**2),
                'dynamic_range': max_amplitude - np.min(np.abs(audio_data))
            }
            
            logger.debug(f"🔧 Audio Features berechnet: {len(features)} Features")
            return features
            
        except Exception as e:
            logger.error(f"❌ Audio Features Berechnung Fehler: {e}")
            return {}
    
    @staticmethod
    def resample_audio(audio_data: np.ndarray, 
                      original_rate: int, 
                      target_rate: int) -> np.ndarray:
        """Resampelt Audio zu anderer Sample Rate"""
        try:
            if original_rate == target_rate:
                return audio_data
            
            if len(audio_data) == 0:
                return audio_data
            
            # Einfaches Linear Resampling
            original_length = len(audio_data)
            target_length = int(original_length * target_rate / original_rate)
            
            # Neue Zeit-Indizes
            original_indices = np.arange(original_length)
            target_indices = np.linspace(0, original_length - 1, target_length)
            
            # Interpoliere
            resampled = np.interp(target_indices, original_indices, audio_data)
            
            logger.debug(f"🔧 Audio resampelt: {original_rate}Hz → {target_rate}Hz")
            return resampled.astype(audio_data.dtype)
            
        except Exception as e:
            logger.error(f"❌ Audio Resampling Fehler: {e}")
            return audio_data
    
    @staticmethod
    def save_audio_to_wav(audio_data: np.ndarray, 
                         sample_rate: int, 
                         filename: str) -> bool:
        """Speichert Audio als WAV-Datei"""
        try:
            import soundfile as sf
            
            # Normalisiere für WAV
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Clamp zu [-1, 1]
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            sf.write(filename, audio_data, sample_rate)
            
            logger.info(f"💾 Audio gespeichert: {filename}")
            return True
            
        except ImportError:
            logger.error("❌ soundfile nicht verfügbar für WAV Export")
            return False
        except Exception as e:
            logger.error(f"❌ WAV Speichern Fehler: {e}")
            return False

class VoiceActivityDetector:
    """Voice Activity Detection (VAD) für Wake Word und Speech"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_duration: float = 0.02,  # 20ms Frames
                 speech_threshold: float = 0.03):
        
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration)
        self.speech_threshold = speech_threshold
        
        # Historische Daten für bessere Erkennung
        self.energy_history = []
        self.max_history_length = 50
        
        logger.debug(f"🎙️ VAD initialisiert: {frame_duration*1000}ms Frames, Threshold: {speech_threshold}")
    
    def is_speech(self, audio_frame: np.ndarray) -> Tuple[bool, float]:
        """Erkennt ob Frame Sprache enthält"""
        try:
            if len(audio_frame) == 0:
                return False, 0.0
            
            # Berechne Energie-Features
            rms = np.sqrt(np.mean(audio_frame**2))
            
            # Zero-Crossing Rate
            zero_crossings = np.sum(np.diff(np.sign(audio_frame)) != 0)
            zcr = zero_crossings / len(audio_frame) if len(audio_frame) > 0 else 0.0
            
            # Spectral Features
            fft = np.fft.fft(audio_frame)
            magnitude = np.abs(fft[:len(fft)//2])
            
            # Energie in Sprach-Frequenzen (300-3400 Hz)
            frequencies = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]
            speech_mask = (frequencies >= 300) & (frequencies <= 3400)
            speech_energy = np.sum(magnitude[speech_mask])
            total_energy = np.sum(magnitude)
            
            speech_ratio = speech_energy / total_energy if total_energy > 0 else 0.0
            
            # Kombiniere Features für Speech Score
            speech_score = 0.0
            
            # RMS Score (40% Gewichtung)
            if rms > self.speech_threshold:
                rms_score = min(1.0, rms / (self.speech_threshold * 3))
                speech_score += rms_score * 0.4
            
            # ZCR Score (20% Gewichtung) - Sprache hat moderate ZCR
            if 0.05 <= zcr <= 0.3:
                zcr_score = 1.0
            elif zcr < 0.05:
                zcr_score = zcr / 0.05
            else:
                zcr_score = max(0.0, 1.0 - (zcr - 0.3) / 0.2)
            
            speech_score += zcr_score * 0.2
            
            # Speech Frequency Ratio (40% Gewichtung)
            if speech_ratio > 0.3:
                freq_score = min(1.0, speech_ratio / 0.7)
                speech_score += freq_score * 0.4
            
            # Aktualisiere Historie
            self.energy_history.append(rms)
            if len(self.energy_history) > self.max_history_length:
                self.energy_history.pop(0)
            
            # Entscheide basierend auf Score
            is_speech_detected = speech_score > 0.5
            
            return is_speech_detected, speech_score
            
        except Exception as e:
            logger.error(f"❌ VAD Fehler: {e}")
            return False, 0.0
    
    def process_audio_stream(self, audio_data: np.ndarray) -> List[Dict[str, Any]]:
        """Verarbeitet Audio-Stream und gibt VAD-Ergebnisse zurück"""
        try:
            results = []
            
            # Teile in Frames auf
            for i in range(0, len(audio_data), self.frame_size):
                frame = audio_data[i:i+self.frame_size]
                
                if len(frame) < self.frame_size:
                    # Padding für letzten Frame
                    frame = np.pad(frame, (0, self.frame_size - len(frame)))
                
                is_speech, speech_score = self.is_speech(frame)
                
                results.append({
                    'frame_start': i,
                    'frame_end': i + self.frame_size,
                    'timestamp': i / self.sample_rate,
                    'is_speech': is_speech,
                    'speech_score': speech_score,
                    'rms': np.sqrt(np.mean(frame**2))
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ VAD Stream Processing Fehler: {e}")
            return []
    
    def get_speech_segments(self, vad_results: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """Extrahiert kontinuierliche Speech-Segmente aus VAD-Ergebnissen"""
        try:
            segments = []
            in_speech = False
            segment_start = 0.0
            
            for result in vad_results:
                timestamp = result['timestamp']
                is_speech = result['is_speech']
                
                if is_speech and not in_speech:
                    # Start neues Speech-Segment
                    segment_start = timestamp
                    in_speech = True
                elif not is_speech and in_speech:
                    # Ende Speech-Segment
                    segments.append((segment_start, timestamp))
                    in_speech = False
            
            # Letztes Segment falls noch aktiv
            if in_speech and vad_results:
                last_timestamp = vad_results[-1]['timestamp']
                segments.append((segment_start, last_timestamp))
            
            return segments
            
        except Exception as e:
            logger.error(f"❌ Speech Segmentierung Fehler: {e}")
            return []

# Utility Functions (Standalone)
def create_silence(duration_seconds: float, sample_rate: int = 16000) -> np.ndarray:
    """Erstellt Stille-Audio"""
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)

def create_tone(frequency: float, 
               duration_seconds: float, 
               sample_rate: int = 16000,
               amplitude: float = 0.3) -> np.ndarray:
    """Erstellt Sinus-Ton"""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    return tone.astype(np.float32)

def mix_audio(audio1: np.ndarray, audio2: np.ndarray, mix_ratio: float = 0.5) -> np.ndarray:
    """Mischt zwei Audio-Streams"""
    try:
        # Gleiche Länge
        min_length = min(len(audio1), len(audio2))
        audio1_trimmed = audio1[:min_length]
        audio2_trimmed = audio2[:min_length]
        
        # Mische
        mixed = (audio1_trimmed * mix_ratio + 
                audio2_trimmed * (1.0 - mix_ratio))
        
        return mixed
        
    except Exception as e:
        logger.error(f"❌ Audio Mixing Fehler: {e}")
        return audio1

def convert_to_mono(audio_data: np.ndarray) -> np.ndarray:
    """Konvertiert Stereo zu Mono"""
    try:
        if len(audio_data.shape) == 1:
            return audio_data  # Bereits Mono
        
        if audio_data.shape[1] == 2:  # Stereo
            return np.mean(audio_data, axis=1)
        else:
            return audio_data[:, 0]  # Ersten Kanal nehmen
            
    except Exception as e:
        logger.error(f"❌ Mono Konvertierung Fehler: {e}")
        return audio_data

# Export
__all__ = [
    'AudioProcessor',
    'VoiceActivityDetector',
    'create_silence',
    'create_tone', 
    'mix_audio',
    'convert_to_mono'
]