"""
Brain Wave Generators
Generiert einzelne Wellentypen (Delta, Alpha, Beta, Theta, Gamma)
"""

import logging
import math
import random
import statistics
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def generate_delta_waves(duration: int, sample_rate: int, characteristics: Dict) -> Dict[str, Any]:
    """Generiert Delta Waves (0.5-4 Hz)"""
    try:
        samples = []
        total_samples = duration * sample_rate
        
        # Delta wave parameters
        base_frequency = 2.0  # 2 Hz
        amplitude = characteristics.get('base_amplitude', 1.0) * 0.8
        noise_level = characteristics.get('noise_level', 0.05)
        
        for i in range(total_samples):
            t = i / sample_rate
            
            # Main delta wave
            wave_value = amplitude * math.sin(2 * math.pi * base_frequency * t)
            
            # Add harmonics
            wave_value += amplitude * 0.3 * math.sin(2 * math.pi * base_frequency * 2 * t)
            
            # Add noise
            noise = random.uniform(-noise_level, noise_level)
            wave_value += noise
            
            # Add slight frequency modulation
            freq_mod = 1.0 + 0.1 * math.sin(2 * math.pi * 0.1 * t)
            wave_value *= freq_mod
            
            samples.append(wave_value)
        
        return {
            'wave_type': 'delta',
            'frequency_range': (0.5, 4.0),
            'samples': samples,
            'sample_rate': sample_rate,
            'duration': duration,
            'characteristics': {
                'base_frequency': base_frequency,
                'amplitude': amplitude,
                'dominant_frequency': _calculate_dominant_frequency(samples, sample_rate)
            }
        }
        
    except Exception as e:
        logger.debug(f"Delta wave generation failed: {e}")
        return {'wave_type': 'delta', 'samples': [], 'error': str(e)}

def generate_alpha_waves(duration: int, sample_rate: int, characteristics: Dict) -> Dict[str, Any]:
    """Generiert Alpha Waves (8-12 Hz)"""
    try:
        samples = []
        total_samples = duration * sample_rate
        
        # Alpha wave parameters
        base_frequency = 10.0  # 10 Hz
        amplitude = characteristics.get('base_amplitude', 1.0) * 1.2
        noise_level = characteristics.get('noise_level', 0.05)
        variability = characteristics.get('variability', 0.1)
        
        for i in range(total_samples):
            t = i / sample_rate
            
            # Main alpha wave with slight frequency variation
            freq_variation = 1.0 + variability * math.sin(2 * math.pi * 0.05 * t)
            current_frequency = base_frequency * freq_variation
            
            wave_value = amplitude * math.sin(2 * math.pi * current_frequency * t)
            
            # Add harmonics
            wave_value += amplitude * 0.2 * math.sin(2 * math.pi * current_frequency * 1.5 * t)
            
            # Add amplitude modulation (alpha rhythm)
            amp_mod = 1.0 + 0.3 * math.sin(2 * math.pi * 0.2 * t)
            wave_value *= amp_mod
            
            # Add noise
            noise = random.uniform(-noise_level, noise_level)
            wave_value += noise
            
            samples.append(wave_value)
        
        return {
            'wave_type': 'alpha',
            'frequency_range': (8.0, 12.0),
            'samples': samples,
            'sample_rate': sample_rate,
            'duration': duration,
            'characteristics': {
                'base_frequency': base_frequency,
                'amplitude': amplitude,
                'variability': variability,
                'dominant_frequency': _calculate_dominant_frequency(samples, sample_rate)
            }
        }
        
    except Exception as e:
        logger.debug(f"Alpha wave generation failed: {e}")
        return {'wave_type': 'alpha', 'samples': [], 'error': str(e)}

def generate_beta_waves(duration: int, sample_rate: int, characteristics: Dict) -> Dict[str, Any]:
    """Generiert Beta Waves (12-30 Hz)"""
    try:
        samples = []
        total_samples = duration * sample_rate
        
        # Beta wave parameters
        base_frequency = 20.0  # 20 Hz
        amplitude = characteristics.get('base_amplitude', 1.0) * 0.9
        noise_level = characteristics.get('noise_level', 0.05) * 1.2  # Beta waves are more noisy
        
        for i in range(total_samples):
            t = i / sample_rate
            
            # Main beta wave
            wave_value = amplitude * math.sin(2 * math.pi * base_frequency * t)
            
            # Add multiple frequency components (beta is more complex)
            wave_value += amplitude * 0.4 * math.sin(2 * math.pi * (base_frequency * 0.8) * t)
            wave_value += amplitude * 0.3 * math.sin(2 * math.pi * (base_frequency * 1.2) * t)
            
            # Add burst pattern (characteristic of beta waves)
            burst_pattern = 1.0 + 0.5 * (math.sin(2 * math.pi * 2 * t) > 0.5)
            wave_value *= burst_pattern
            
            # Add noise
            noise = random.uniform(-noise_level, noise_level)
            wave_value += noise
            
            samples.append(wave_value)
        
        return {
            'wave_type': 'beta',
            'frequency_range': (12.0, 30.0),
            'samples': samples,
            'sample_rate': sample_rate,
            'duration': duration,
            'characteristics': {
                'base_frequency': base_frequency,
                'amplitude': amplitude,
                'complexity': 'high',
                'dominant_frequency': _calculate_dominant_frequency(samples, sample_rate)
            }
        }
        
    except Exception as e:
        logger.debug(f"Beta wave generation failed: {e}")
        return {'wave_type': 'beta', 'samples': [], 'error': str(e)}

def generate_theta_waves(duration: int, sample_rate: int, characteristics: Dict) -> Dict[str, Any]:
    """Generiert Theta Waves (4-8 Hz)"""
    try:
        samples = []
        total_samples = duration * sample_rate
        
        # Theta wave parameters
        base_frequency = 6.0  # 6 Hz
        amplitude = characteristics.get('base_amplitude', 1.0) * 1.0
        noise_level = characteristics.get('noise_level', 0.05)
        
        for i in range(total_samples):
            t = i / sample_rate
            
            # Main theta wave
            wave_value = amplitude * math.sin(2 * math.pi * base_frequency * t)
            
            # Add theta rhythm modulation
            theta_mod = 1.0 + 0.4 * math.sin(2 * math.pi * 0.15 * t)
            wave_value *= theta_mod
            
            # Add harmonics
            wave_value += amplitude * 0.25 * math.sin(2 * math.pi * base_frequency * 1.5 * t)
            
            # Add noise
            noise = random.uniform(-noise_level, noise_level)
            wave_value += noise
            
            samples.append(wave_value)
        
        return {
            'wave_type': 'theta',
            'frequency_range': (4.0, 8.0),
            'samples': samples,
            'sample_rate': sample_rate,
            'duration': duration,
            'characteristics': {
                'base_frequency': base_frequency,
                'amplitude': amplitude,
                'rhythm_modulation': True,
                'dominant_frequency': _calculate_dominant_frequency(samples, sample_rate)
            }
        }
        
    except Exception as e:
        logger.debug(f"Theta wave generation failed: {e}")
        return {'wave_type': 'theta', 'samples': [], 'error': str(e)}

def generate_gamma_waves(duration: int, sample_rate: int, characteristics: Dict) -> Dict[str, Any]:
    """Generiert Gamma Waves (30-100 Hz)"""
    try:
        samples = []
        total_samples = duration * sample_rate
        
        # Gamma wave parameters
        base_frequency = 40.0  # 40 Hz
        amplitude = characteristics.get('base_amplitude', 1.0) * 0.6  # Gamma waves are typically smaller
        noise_level = characteristics.get('noise_level', 0.05) * 1.5  # More noisy
        
        for i in range(total_samples):
            t = i / sample_rate
            
            # Main gamma wave
            wave_value = amplitude * math.sin(2 * math.pi * base_frequency * t)
            
            # Add gamma burst pattern
            burst_envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * t)
            wave_value *= burst_envelope
            
            # Add high-frequency components
            wave_value += amplitude * 0.3 * math.sin(2 * math.pi * (base_frequency * 1.5) * t)
            wave_value += amplitude * 0.2 * math.sin(2 * math.pi * (base_frequency * 2.0) * t)
            
            # Add noise
            noise = random.uniform(-noise_level, noise_level)
            wave_value += noise
            
            samples.append(wave_value)
        
        return {
            'wave_type': 'gamma',
            'frequency_range': (30.0, 100.0),
            'samples': samples,
            'sample_rate': sample_rate,
            'duration': duration,
            'characteristics': {
                'base_frequency': base_frequency,
                'amplitude': amplitude,
                'burst_pattern': True,
                'dominant_frequency': _calculate_dominant_frequency(samples, sample_rate)
            }
        }
        
    except Exception as e:
        logger.debug(f"Gamma wave generation failed: {e}")
        return {'wave_type': 'gamma', 'samples': [], 'error': str(e)}

def _calculate_dominant_frequency(samples: List[float], sample_rate: int) -> float:
    """Berechnet dominante Frequenz aus Samples"""
    try:
        if not samples or len(samples) < 10:
            return 0.0
        
        # Simple frequency estimation using zero crossings
        zero_crossings = 0
        for i in range(1, len(samples)):
            if (samples[i-1] >= 0 and samples[i] < 0) or (samples[i-1] < 0 and samples[i] >= 0):
                zero_crossings += 1
        
        # Estimate frequency
        duration = len(samples) / sample_rate
        estimated_frequency = zero_crossings / (2 * duration) if duration > 0 else 0.0
        
        return estimated_frequency
        
    except Exception as e:
        logger.debug(f"Dominant frequency calculation failed: {e}")
        return 0.0

__all__ = [
    'generate_delta_waves',
    'generate_alpha_waves', 
    'generate_beta_waves',
    'generate_theta_waves',
    'generate_gamma_waves'
]