"""
Brain Wave Analyzers
Analysiert Frequenzen, Kohärenz und Muster von Brain Waves
"""

import logging
import statistics
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def analyze_wave_frequencies(brain_wave_patterns: Dict) -> Dict[str, Any]:
    """Analysiert Wave Frequencies - VEREINFACHT"""
    try:
        frequency_analysis = {}
        
        # Analyze each wave type
        for wave_type, wave_data in brain_wave_patterns.items():
            if isinstance(wave_data, dict) and 'samples' in wave_data:
                samples = wave_data['samples']
                sample_rate = wave_data.get('sample_rate', 100)
                
                if samples:
                    # Vereinfachte Frequenz-Analyse
                    frequency_analysis[wave_type] = {
                        'dominant_frequency': _calculate_dominant_frequency_simple(samples, sample_rate),
                        'average_amplitude': statistics.mean([abs(s) for s in samples]),
                        'signal_strength': 'high' if len(samples) > 1000 else 'moderate'
                    }
        
        # Overall analysis
        dominant_wave = _identify_dominant_wave_simple(frequency_analysis)
        
        return {
            'wave_specific_analysis': frequency_analysis,
            'overall_analysis': {
                'dominant_wave_type': dominant_wave,
                'frequency_balance': 'balanced',  # Simplified
                'analysis_quality': 'simplified_analysis'
            }
        }
        
    except Exception as e:
        logger.error(f"Wave frequency analysis failed: {e}")
        return {'error': str(e)}

def calculate_wave_coherence(brain_wave_patterns: Dict) -> Dict[str, Any]:
    """Berechnet Wave Coherence - VEREINFACHT"""
    try:
        coherence_metrics = {}
        
        # Simplified coherence calculation
        wave_types = list(brain_wave_patterns.keys())
        total_coherence = 0.0
        pair_count = 0
        
        for i, wave_type1 in enumerate(wave_types):
            for wave_type2 in wave_types[i+1:]:
                wave_data1 = brain_wave_patterns[wave_type1]
                wave_data2 = brain_wave_patterns[wave_type2]
                
                if isinstance(wave_data1, dict) and isinstance(wave_data2, dict):
                    samples1 = wave_data1.get('samples', [])
                    samples2 = wave_data2.get('samples', [])
                    
                    if samples1 and samples2:
                        # Simplified coherence calculation
                        coherence = _calculate_simple_coherence(samples1, samples2)
                        coherence_key = f"{wave_type1}_{wave_type2}"
                        coherence_metrics[coherence_key] = coherence
                        total_coherence += coherence
                        pair_count += 1
        
        global_coherence = total_coherence / pair_count if pair_count > 0 else 0.5
        
        return {
            'pairwise_coherence': coherence_metrics,
            'overall_coherence': {
                'global_coherence_score': global_coherence,
                'coherence_quality': 'good' if global_coherence > 0.7 else 'moderate',
                'analysis_method': 'simplified_cross_correlation'
            }
        }
        
    except Exception as e:
        logger.error(f"Wave coherence calculation failed: {e}")
        return {'error': str(e)}

def analyze_wave_patterns(brain_wave_patterns: Dict) -> Dict[str, Any]:
    """Analysiert Wave Patterns - VEREINFACHT"""
    try:
        pattern_analysis = {}
        
        # Count available patterns
        available_patterns = len([w for w in brain_wave_patterns.values() 
                                if isinstance(w, dict) and w.get('samples')])
        
        # Simplified pattern analysis
        if available_patterns >= 4:
            pattern_complexity = 'high'
            pattern_type = 'multi_wave_complex'
        elif available_patterns >= 2:
            pattern_complexity = 'moderate'
            pattern_type = 'dual_wave_balanced'
        else:
            pattern_complexity = 'low'
            pattern_type = 'single_wave_dominant'
        
        return {
            'pattern_type': pattern_type,
            'complexity': pattern_complexity,
            'available_wave_types': available_patterns,
            'analysis_method': 'simplified_pattern_recognition'
        }
        
    except Exception as e:
        logger.debug(f"Wave pattern analysis failed: {e}")
        return {'pattern_type': 'unknown', 'complexity': 'moderate'}

def calculate_wave_synchronization(brain_wave_patterns: Dict) -> Dict[str, Any]:
    """Berechnet Wave Synchronization - VEREINFACHT"""
    try:
        # Simplified synchronization calculation
        synchronization_score = 0.7  # Default moderate sync
        
        # Check if we have multiple wave types
        wave_count = len([w for w in brain_wave_patterns.values() 
                         if isinstance(w, dict) and w.get('samples')])
        
        if wave_count >= 4:
            synchronization_score = 0.8  # Good sync with multiple waves
        elif wave_count >= 2:
            synchronization_score = 0.75  # Moderate sync
        else:
            synchronization_score = 0.6  # Lower sync with single wave
        
        return {
            'synchronization_score': synchronization_score,
            'sync_quality': 'good' if synchronization_score > 0.75 else 'moderate',
            'participating_waves': wave_count,
            'analysis_method': 'simplified_multi_wave_sync'
        }
        
    except Exception as e:
        logger.debug(f"Wave synchronization calculation failed: {e}")
        return {'synchronization_score': 0.5, 'sync_quality': 'unknown'}

# Helper Functions
def _calculate_dominant_frequency_simple(samples: List[float], sample_rate: int) -> float:
    """Vereinfachte dominante Frequenz-Berechnung"""
    try:
        if not samples or len(samples) < 10:
            return 0.0
        
        # Count zero crossings
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

def _identify_dominant_wave_simple(frequency_analysis: Dict) -> str:
    """Identifiziert dominanten Wellentyp basierend auf Amplitude"""
    try:
        if not frequency_analysis:
            return 'unknown'
        
        max_amplitude = 0.0
        dominant_wave = 'unknown'
        
        for wave_type, analysis in frequency_analysis.items():
            amplitude = analysis.get('average_amplitude', 0.0)
            if amplitude > max_amplitude:
                max_amplitude = amplitude
                dominant_wave = wave_type
        
        return dominant_wave
        
    except Exception as e:
        logger.debug(f"Dominant wave identification failed: {e}")
        return 'unknown'

def _calculate_simple_coherence(samples1: List[float], samples2: List[float]) -> float:
    """Vereinfachte Coherence-Berechnung"""
    try:
        if len(samples1) != len(samples2) or not samples1:
            return 0.0
        
        # Simple correlation coefficient
        n = min(len(samples1), len(samples2), 1000)  # Limit for performance
        
        # Calculate means
        mean1 = sum(samples1[:n]) / n
        mean2 = sum(samples2[:n]) / n
        
        # Calculate correlation
        numerator = sum((samples1[i] - mean1) * (samples2[i] - mean2) for i in range(n))
        
        sum_sq1 = sum((samples1[i] - mean1) ** 2 for i in range(n))
        sum_sq2 = sum((samples2[i] - mean2) ** 2 for i in range(n))
        
        denominator = (sum_sq1 * sum_sq2) ** 0.5
        
        if denominator > 0:
            correlation = abs(numerator / denominator)
        else:
            correlation = 0.0
        
        return max(0.0, min(1.0, correlation))
        
    except Exception as e:
        logger.debug(f"Simple coherence calculation failed: {e}")
        return 0.0

__all__ = [
    'analyze_wave_frequencies',
    'calculate_wave_coherence',
    'analyze_wave_patterns', 
    'calculate_wave_synchronization'
]