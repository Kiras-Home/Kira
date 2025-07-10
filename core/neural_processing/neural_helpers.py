"""
Neural Processing Helper Functions
Gemeinsame Hilfsfunktionen für Neural Processing
"""

import logging
import math
import random
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def determine_wave_characteristics(memory_manager=None, personality_data: Dict = None, system_state: Dict = None) -> Dict[str, Any]:
    """Bestimmt Wave Characteristics basierend auf aktuellem State"""
    try:
        characteristics = {
            'base_amplitude': 1.0,
            'noise_level': 0.05,
            'variability': 0.1,
            'coherence_level': 0.7,
            'dominant_frequency_bias': 'balanced'
        }
        
        # Memory manager influence
        if memory_manager:
            try:
                if hasattr(memory_manager, 'get_current_cognitive_load'):
                    cognitive_load = memory_manager.get_current_cognitive_load()
                    characteristics['base_amplitude'] *= (1.0 + cognitive_load * 0.3)
                    characteristics['variability'] *= (1.0 + cognitive_load * 0.2)
            except Exception as e:
                logger.debug(f"Memory manager influence extraction failed: {e}")
        
        # Personality influence
        if personality_data:
            current_state = personality_data.get('current_state', {})
            emotional_stability = current_state.get('emotional_stability', 0.7)
            characteristics['coherence_level'] *= emotional_stability
            characteristics['noise_level'] *= (2.0 - emotional_stability)
        
        # ✅ NEUE: System State influence
        if system_state:
            try:
                available_systems = sum(1 for sys in system_state.get('systems_status', {}).values() if sys.get('available', False))
                system_health = available_systems / 5.0  # 0.0 - 1.0
                
                # System health beeinflusst wave characteristics
                characteristics['base_amplitude'] *= (0.8 + system_health * 0.4)  # 0.8 - 1.2
                characteristics['coherence_level'] *= (0.6 + system_health * 0.4)  # 0.6 - 1.0
                characteristics['noise_level'] *= (1.5 - system_health * 0.5)  # 1.0 - 1.5
                
                logger.debug(f"System state influence applied: health={system_health:.2f}")
                
            except Exception as e:
                logger.debug(f"System state influence extraction failed: {e}")
        
        return characteristics
        
    except Exception as e:
        logger.debug(f"Wave characteristics determination failed: {e}")
        return characteristics

def generate_composite_brain_wave(brain_wave_patterns: Dict, wave_characteristics: Dict) -> Dict[str, Any]:
    """Generiert Composite Brain Wave - VEREINFACHT"""
    try:
        if not brain_wave_patterns:
            return {'samples': [], 'composite_properties': {}}
        
        # Find the longest sample set to use as base
        max_samples = 0
        base_wave_data = None
        
        for wave_type, wave_data in brain_wave_patterns.items():
            if isinstance(wave_data, dict) and 'samples' in wave_data:
                samples = wave_data.get('samples', [])
                if len(samples) > max_samples:
                    max_samples = len(samples)
                    base_wave_data = wave_data
        
        if not base_wave_data or not base_wave_data.get('samples'):
            return {'samples': [], 'composite_properties': {}}
        
        # Simple composite: weighted combination of all waves
        composite_samples = []
        sample_count = len(base_wave_data['samples'])
        
        # Define wave weights
        wave_weights = {
            'delta_waves': 0.15,
            'theta_waves': 0.20,
            'alpha_waves': 0.25,
            'beta_waves': 0.25,
            'gamma_waves': 0.15
        }
        
        for i in range(sample_count):
            composite_value = 0.0
            total_weight = 0.0
            
            for wave_type, wave_data in brain_wave_patterns.items():
                if isinstance(wave_data, dict) and 'samples' in wave_data:
                    samples = wave_data['samples']
                    if i < len(samples):
                        weight = wave_weights.get(wave_type, 0.1)
                        composite_value += samples[i] * weight
                        total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                composite_value /= total_weight
            
            composite_samples.append(composite_value)
        
        return {
            'samples': composite_samples,
            'sample_rate': base_wave_data.get('sample_rate', 100),
            'duration': base_wave_data.get('duration', 30),
            'composite_properties': {
                'generation_method': 'weighted_combination',
                'component_waves': list(brain_wave_patterns.keys()),
                'total_samples': len(composite_samples),
                'wave_weights': wave_weights
            }
        }
        
    except Exception as e:
        logger.debug(f"Composite brain wave generation failed: {e}")
        return {'samples': [], 'composite_properties': {}}

def generate_fallback_brain_wave_data(duration: int, sample_rate: int) -> Dict[str, Any]:
    """Generiert Fallback Brain Wave Data"""
    try:
        # Simple alpha wave as fallback
        samples = []
        total_samples = duration * sample_rate
        base_frequency = 10.0  # Alpha frequency
        
        for i in range(total_samples):
            t = i / sample_rate
            sample_value = math.sin(2 * math.pi * base_frequency * t)
            samples.append(sample_value)
        
        return {
            'fallback_mode': True,
            'wave_patterns': {
                'alpha_waves': {
                    'wave_type': 'alpha',
                    'samples': samples,
                    'sample_rate': sample_rate,
                    'duration': duration,
                    'frequency_range': (8.0, 12.0)
                }
            },
            'wave_analysis': {
                'dominant_frequencies': {'alpha_waves': base_frequency},
                'wave_coherence': {'global_coherence_score': 0.7}
            },
            'composite_wave': {
                'samples': samples,
                'sample_rate': sample_rate,
                'duration': duration
            },
            'generation_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'fallback_mode': True,
                'fallback_reason': 'normal_operation_fallback'
            }
        }
        
    except Exception as e:
        logger.error(f"Fallback brain wave data generation failed: {e}")
        return {
            'fallback_mode': True,
            'error': str(e),
            'wave_patterns': {},
            'generation_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'fallback_mode': True,
                'fallback_reason': 'error_fallback'
            }
        }

def validate_simulation_params(simulation_params: Dict) -> Dict[str, Any]:
    """Validiert und normalisiert Simulation Parameters"""
    try:
        validated_params = {
            'duration': 30,
            'sample_rate': 100,
            'noise_level': 0.05,
            'variability': 0.1
        }
        
        if simulation_params:
            # Duration validation
            if 'duration' in simulation_params:
                duration = simulation_params['duration']
                if isinstance(duration, (int, float)) and 1 <= duration <= 300:
                    validated_params['duration'] = int(duration)
            
            # Sample rate validation
            if 'sample_rate' in simulation_params:
                sample_rate = simulation_params['sample_rate']
                if isinstance(sample_rate, (int, float)) and 10 <= sample_rate <= 1000:
                    validated_params['sample_rate'] = int(sample_rate)
            
            # Noise level validation
            if 'noise_level' in simulation_params:
                noise_level = simulation_params['noise_level']
                if isinstance(noise_level, (int, float)) and 0.0 <= noise_level <= 1.0:
                    validated_params['noise_level'] = float(noise_level)
            
            # Variability validation
            if 'variability' in simulation_params:
                variability = simulation_params['variability']
                if isinstance(variability, (int, float)) and 0.0 <= variability <= 1.0:
                    validated_params['variability'] = float(variability)
        
        return validated_params
        
    except Exception as e:
        logger.debug(f"Simulation params validation failed: {e}")
        return {
            'duration': 30,
            'sample_rate': 100,
            'noise_level': 0.05,
            'variability': 0.1
        }

def calculate_wave_statistics(samples: List[float]) -> Dict[str, Any]:
    """Berechnet grundlegende Wave Statistics"""
    try:
        if not samples:
            return {'error': 'no_samples'}
        
        # Basic statistics
        mean_value = statistics.mean(samples)
        max_value = max(samples)
        min_value = min(samples)
        std_dev = statistics.stdev(samples) if len(samples) > 1 else 0.0
        
        # RMS (Root Mean Square)
        rms = math.sqrt(sum(s**2 for s in samples) / len(samples))
        
        # Peak-to-peak
        peak_to_peak = max_value - min_value
        
        return {
            'sample_count': len(samples),
            'mean': mean_value,
            'max': max_value,
            'min': min_value,
            'std_dev': std_dev,
            'rms': rms,
            'peak_to_peak': peak_to_peak,
            'dynamic_range': std_dev / max(abs(mean_value), 0.001)  # Normalized dynamic range
        }
        
    except Exception as e:
        logger.debug(f"Wave statistics calculation failed: {e}")
        return {'error': str(e)}

def apply_wave_filter(samples: List[float], filter_type: str = 'none') -> List[float]:
    """Wendet einfache Filter auf Wave Samples an"""
    try:
        if not samples or filter_type == 'none':
            return samples
        
        filtered_samples = samples.copy()
        
        if filter_type == 'smooth':
            # Simple moving average smoothing
            window_size = min(5, len(samples) // 10)
            if window_size > 1:
                smoothed = []
                for i in range(len(samples)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(samples), i + window_size // 2 + 1)
                    window_samples = samples[start_idx:end_idx]
                    smoothed.append(sum(window_samples) / len(window_samples))
                filtered_samples = smoothed
        
        elif filter_type == 'amplify':
            # Simple amplification
            max_abs = max(abs(s) for s in samples) if samples else 1.0
            if max_abs > 0:
                amplification_factor = 1.0 / max_abs
                filtered_samples = [s * amplification_factor for s in samples]
        
        elif filter_type == 'normalize':
            # Zero-mean normalization
            if samples:
                mean_val = statistics.mean(samples)
                filtered_samples = [s - mean_val for s in samples]
        
        return filtered_samples
        
    except Exception as e:
        logger.debug(f"Wave filtering failed: {e}")
        return samples

def generate_wave_metadata(wave_data: Dict, generation_params: Dict = None) -> Dict[str, Any]:
    """Generiert Metadata für Wave Data"""
    try:
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'wave_type': wave_data.get('wave_type', 'unknown'),
            'sample_count': len(wave_data.get('samples', [])),
            'duration_seconds': wave_data.get('duration', 0),
            'sample_rate_hz': wave_data.get('sample_rate', 0),
            'frequency_range_hz': wave_data.get('frequency_range', (0, 0))
        }
        
        if generation_params:
            metadata['generation_params'] = generation_params
        
        # Add wave statistics if samples available
        samples = wave_data.get('samples', [])
        if samples:
            wave_stats = calculate_wave_statistics(samples)
            if 'error' not in wave_stats:
                metadata['wave_statistics'] = wave_stats
        
        return metadata
        
    except Exception as e:
        logger.debug(f"Wave metadata generation failed: {e}")
        return {
            'generation_timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

def classify_mental_state_from_patterns(wave_patterns: Dict) -> str:
    """Klassifiziert Mental State basierend auf Wave Patterns - VEREINFACHT"""
    try:
        if not wave_patterns:
            return 'unknown'
        
        # Simple classification based on dominant wave amplitudes
        wave_amplitudes = {}
        
        for wave_type, wave_data in wave_patterns.items():
            if isinstance(wave_data, dict) and 'samples' in wave_data:
                samples = wave_data['samples']
                if samples:
                    avg_amplitude = statistics.mean([abs(s) for s in samples])
                    wave_amplitudes[wave_type] = avg_amplitude
        
        if not wave_amplitudes:
            return 'unknown'
        
        # Find dominant wave type
        dominant_wave = max(wave_amplitudes.items(), key=lambda x: x[1])[0]
        
        # Simple state classification
        if dominant_wave == 'alpha_waves':
            return 'relaxed'
        elif dominant_wave == 'beta_waves':
            return 'focused'
        elif dominant_wave == 'theta_waves':
            return 'creative'
        elif dominant_wave == 'delta_waves':
            return 'deep_rest'
        elif dominant_wave == 'gamma_waves':
            return 'highly_active'
        else:
            return 'balanced'
        
    except Exception as e:
        logger.debug(f"Mental state classification failed: {e}")
        return 'unknown'

__all__ = [
    'determine_wave_characteristics',
    'generate_composite_brain_wave',
    'generate_fallback_brain_wave_data',
    'validate_simulation_params',
    'calculate_wave_statistics',
    'apply_wave_filter',
    'generate_wave_metadata',
    'classify_mental_state_from_patterns'
]