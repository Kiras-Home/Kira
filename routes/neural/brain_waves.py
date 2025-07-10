"""
Neural Brain Waves Module - Main Interface
Brain Wave Simulation, Pattern Analysis und Frequency Management
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from core.neural_processing.wave_generators import (
    generate_delta_waves, generate_alpha_waves, generate_beta_waves,
    generate_theta_waves, generate_gamma_waves
)
from core.neural_processing.pattern_simulators import (
    simulate_meditative_patterns, simulate_focused_patterns,
    simulate_creative_patterns, simulate_learning_patterns,
    simulate_stressed_patterns, simulate_dynamic_patterns
)
from core.neural_processing.wave_analyzers import (
    analyze_wave_frequencies, calculate_wave_coherence,
    analyze_wave_patterns, calculate_wave_synchronization
)
from core.neural_processing.neural_helpers import (
    determine_wave_characteristics, generate_composite_brain_wave,
    generate_fallback_brain_wave_data, validate_simulation_params
)

logger = logging.getLogger(__name__)

def generate_brain_wave_data(memory_manager=None,
                           personality_data: Dict = None,
                           wave_duration: int = 60,
                           sample_rate: int = 100) -> Dict[str, Any]:
    """
    Generiert Brain Wave Data für Visualization
    
    HAUPTINTERFACE für Brain Wave Generation
    """
    try:
        # Determine brain wave characteristics based on current state
        wave_characteristics = determine_wave_characteristics(memory_manager, personality_data)
        
        # Generate brain wave patterns
        brain_wave_patterns = {
            'delta_waves': generate_delta_waves(wave_duration, sample_rate, wave_characteristics),
            'theta_waves': generate_theta_waves(wave_duration, sample_rate, wave_characteristics),
            'alpha_waves': generate_alpha_waves(wave_duration, sample_rate, wave_characteristics),
            'beta_waves': generate_beta_waves(wave_duration, sample_rate, wave_characteristics),
            'gamma_waves': generate_gamma_waves(wave_duration, sample_rate, wave_characteristics)
        }
        
        # Analyze wave patterns
        wave_analysis = {
            'dominant_frequencies': analyze_wave_frequencies(brain_wave_patterns),
            'wave_coherence': calculate_wave_coherence(brain_wave_patterns),
            'pattern_analysis': analyze_wave_patterns(brain_wave_patterns),
            'synchronization_metrics': calculate_wave_synchronization(brain_wave_patterns)
        }
        
        # Generate composite brain wave
        composite_wave = generate_composite_brain_wave(brain_wave_patterns, wave_characteristics)
        
        # Compile complete brain wave data
        brain_wave_data = {
            'wave_patterns': brain_wave_patterns,
            'wave_analysis': wave_analysis,
            'composite_wave': composite_wave,
            'wave_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'wave_duration_seconds': wave_duration,
                'sample_rate_hz': sample_rate,
                'total_samples': wave_duration * sample_rate,
                'wave_characteristics': wave_characteristics
            }
        }
        
        return brain_wave_data
        
    except Exception as e:
        logger.error(f"Brain wave data generation failed: {e}")
        return generate_fallback_brain_wave_data(wave_duration, sample_rate)

def simulate_brain_wave_patterns(pattern_type: str = 'dynamic',
                               memory_manager=None,
                               personality_data: Dict = None,
                               simulation_params: Dict = None) -> Dict[str, Any]:
    """
    Simuliert spezifische Brain Wave Patterns
    
    HAUPTINTERFACE für Pattern Simulation
    """
    try:
        # Validate and normalize simulation parameters
        validated_params = validate_simulation_params(simulation_params or {})
        
        # Pattern-specific simulation
        if pattern_type == 'meditative':
            pattern_simulation = simulate_meditative_patterns(validated_params, memory_manager, personality_data)
        elif pattern_type == 'focused':
            pattern_simulation = simulate_focused_patterns(validated_params, memory_manager, personality_data)
        elif pattern_type == 'creative':
            pattern_simulation = simulate_creative_patterns(validated_params, memory_manager, personality_data)
        elif pattern_type == 'learning':
            pattern_simulation = simulate_learning_patterns(validated_params, memory_manager, personality_data)
        elif pattern_type == 'stressed':
            pattern_simulation = simulate_stressed_patterns(validated_params, memory_manager, personality_data)
        else:  # dynamic
            pattern_simulation = simulate_dynamic_patterns(validated_params, memory_manager, personality_data)
        
        return {
            'pattern_type': pattern_type,
            'simulation_data': pattern_simulation,
            'simulation_metadata': {
                'simulation_timestamp': datetime.now().isoformat(),
                'simulation_params': validated_params
            }
        }
        
    except Exception as e:
        logger.error(f"Brain wave pattern simulation failed: {e}")
        return {'error': str(e), 'pattern_type': pattern_type}

__all__ = [
    'generate_brain_wave_data',
    'simulate_brain_wave_patterns'
]