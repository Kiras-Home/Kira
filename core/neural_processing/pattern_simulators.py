"""
Brain Wave Pattern Simulators
Simuliert spezifische mentale Zustände (Meditation, Fokus, Kreativität, etc.)
"""

import logging
from typing import Dict, Any
from .wave_generators import (
    generate_delta_waves, generate_alpha_waves, generate_beta_waves,
    generate_theta_waves, generate_gamma_waves
)

logger = logging.getLogger(__name__)

def simulate_meditative_patterns(simulation_params: Dict, memory_manager=None, 
                               personality_data: Dict = None) -> Dict[str, Any]:
    """Simuliert meditative Hirnwellen-Muster - VEREINFACHT"""
    try:
        duration = simulation_params.get('duration', 30)
        sample_rate = simulation_params.get('sample_rate', 100)
        
        # Meditative Charakteristiken - Alpha und Theta dominant
        meditative_characteristics = {
            'base_amplitude': 1.0,
            'noise_level': 0.02,  # Sehr wenig Rauschen
            'variability': 0.05   # Sehr regelmäßig
        }
        
        # Generiere meditative Wellen-Muster
        meditative_patterns = {
            'alpha_waves': generate_alpha_waves(duration, sample_rate, {
                **meditative_characteristics, 'base_amplitude': 1.5  # Alpha dominant
            }),
            'theta_waves': generate_theta_waves(duration, sample_rate, {
                **meditative_characteristics, 'base_amplitude': 1.2  # Theta erhöht
            }),
            'beta_waves': generate_beta_waves(duration, sample_rate, {
                **meditative_characteristics, 'base_amplitude': 0.3  # Beta unterdrückt
            }),
            'gamma_waves': generate_gamma_waves(duration, sample_rate, {
                **meditative_characteristics, 'base_amplitude': 0.4  # Gamma reduziert
            }),
            'delta_waves': generate_delta_waves(duration, sample_rate, {
                **meditative_characteristics, 'base_amplitude': 1.0  # Delta normal
            })
        }
        
        return {
            'pattern_type': 'meditative',
            'wave_patterns': meditative_patterns,
            'pattern_analysis': {
                'meditation_depth': 0.8,
                'mindfulness_index': 0.85,
                'relaxation_level': 0.9
            },
            'simulation_info': {
                'is_real_eeg': False,
                'simulation_type': 'simplified_model',
                'characteristics': 'alpha_theta_dominant'
            }
        }
        
    except Exception as e:
        logger.error(f"Meditative patterns simulation failed: {e}")
        return {'error': str(e), 'pattern_type': 'meditative'}

def simulate_focused_patterns(simulation_params: Dict, memory_manager=None, 
                            personality_data: Dict = None) -> Dict[str, Any]:
    """Simuliert fokussierte Aufmerksamkeits-Muster - VEREINFACHT"""
    try:
        duration = simulation_params.get('duration', 30)
        sample_rate = simulation_params.get('sample_rate', 100)
        
        # Fokus-Charakteristiken - Beta und Gamma dominant
        focused_characteristics = {
            'base_amplitude': 1.0,
            'noise_level': 0.06,  # Mehr Aktivität = mehr Rauschen
            'variability': 0.15   # Variable Aufmerksamkeit
        }
        
        # Generiere fokussierte Wellen-Muster
        focused_patterns = {
            'beta_waves': generate_beta_waves(duration, sample_rate, {
                **focused_characteristics, 'base_amplitude': 1.8  # Beta sehr dominant
            }),
            'gamma_waves': generate_gamma_waves(duration, sample_rate, {
                **focused_characteristics, 'base_amplitude': 1.2  # Gamma erhöht
            }),
            'alpha_waves': generate_alpha_waves(duration, sample_rate, {
                **focused_characteristics, 'base_amplitude': 0.5  # Alpha unterdrückt
            }),
            'theta_waves': generate_theta_waves(duration, sample_rate, {
                **focused_characteristics, 'base_amplitude': 0.6  # Theta reduziert
            }),
            'delta_waves': generate_delta_waves(duration, sample_rate, {
                **focused_characteristics, 'base_amplitude': 0.3  # Delta minimal
            })
        }
        
        return {
            'pattern_type': 'focused',
            'wave_patterns': focused_patterns,
            'pattern_analysis': {
                'attention_level': 0.9,
                'focus_intensity': 0.85,
                'concentration_stability': 0.8
            },
            'simulation_info': {
                'is_real_eeg': False,
                'simulation_type': 'simplified_model',
                'characteristics': 'beta_gamma_dominant'
            }
        }
        
    except Exception as e:
        logger.error(f"Focused patterns simulation failed: {e}")
        return {'error': str(e), 'pattern_type': 'focused'}

def simulate_creative_patterns(simulation_params: Dict, memory_manager=None, 
                             personality_data: Dict = None) -> Dict[str, Any]:
    """Simuliert kreative Denk-Muster - VEREINFACHT"""
    try:
        duration = simulation_params.get('duration', 30)
        sample_rate = simulation_params.get('sample_rate', 100)
        
        # Kreativitäts-Charakteristiken - Alpha-Theta Balance mit Gamma-Bursts
        creative_characteristics = {
            'base_amplitude': 1.0,
            'noise_level': 0.07,  # Kreative Unregelmäßigkeit
            'variability': 0.2    # Hohe Variabilität für Ideenfluss
        }
        
        # Generiere kreative Wellen-Muster
        creative_patterns = {
            'alpha_waves': generate_alpha_waves(duration, sample_rate, {
                **creative_characteristics, 'base_amplitude': 1.3  # Alpha hoch für Entspannung
            }),
            'theta_waves': generate_theta_waves(duration, sample_rate, {
                **creative_characteristics, 'base_amplitude': 1.1  # Theta für Imagination
            }),
            'gamma_waves': generate_gamma_waves(duration, sample_rate, {
                **creative_characteristics, 'base_amplitude': 0.9  # Gamma für Aha-Momente
            }),
            'beta_waves': generate_beta_waves(duration, sample_rate, {
                **creative_characteristics, 'base_amplitude': 0.7  # Beta moderat
            }),
            'delta_waves': generate_delta_waves(duration, sample_rate, {
                **creative_characteristics, 'base_amplitude': 0.5  # Delta niedrig
            })
        }
        
        return {
            'pattern_type': 'creative',
            'wave_patterns': creative_patterns,
            'pattern_analysis': {
                'creativity_index': 0.85,
                'imagination_level': 0.8,
                'idea_flow_rate': 0.75
            },
            'simulation_info': {
                'is_real_eeg': False,
                'simulation_type': 'simplified_model',
                'characteristics': 'alpha_theta_gamma_balanced'
            }
        }
        
    except Exception as e:
        logger.error(f"Creative patterns simulation failed: {e}")
        return {'error': str(e), 'pattern_type': 'creative'}

def simulate_learning_patterns(simulation_params: Dict, memory_manager=None, 
                             personality_data: Dict = None) -> Dict[str, Any]:
    """Simuliert Lern-Muster - VEREINFACHT"""
    try:
        duration = simulation_params.get('duration', 30)
        sample_rate = simulation_params.get('sample_rate', 100)
        
        # Lern-Charakteristiken - Theta dominant mit Alpha und Gamma
        learning_characteristics = {
            'base_amplitude': 1.0,
            'noise_level': 0.05,  # Moderate Aktivität
            'variability': 0.12   # Anpassungsfähigkeit
        }
        
        # Generiere Lern-Wellen-Muster
        learning_patterns = {
            'theta_waves': generate_theta_waves(duration, sample_rate, {
                **learning_characteristics, 'base_amplitude': 1.4  # Theta dominant für Lernen
            }),
            'alpha_waves': generate_alpha_waves(duration, sample_rate, {
                **learning_characteristics, 'base_amplitude': 1.1  # Alpha für Integration
            }),
            'gamma_waves': generate_gamma_waves(duration, sample_rate, {
                **learning_characteristics, 'base_amplitude': 0.8  # Gamma für Encoding
            }),
            'beta_waves': generate_beta_waves(duration, sample_rate, {
                **learning_characteristics, 'base_amplitude': 0.9  # Beta für Verarbeitung
            }),
            'delta_waves': generate_delta_waves(duration, sample_rate, {
                **learning_characteristics, 'base_amplitude': 0.7  # Delta für Consolidation
            })
        }
        
        return {
            'pattern_type': 'learning',
            'wave_patterns': learning_patterns,
            'pattern_analysis': {
                'learning_efficiency': 0.8,
                'memory_encoding_strength': 0.75,
                'knowledge_integration': 0.7
            },
            'simulation_info': {
                'is_real_eeg': False,
                'simulation_type': 'simplified_model',
                'characteristics': 'theta_dominant_learning'
            }
        }
        
    except Exception as e:
        logger.error(f"Learning patterns simulation failed: {e}")
        return {'error': str(e), 'pattern_type': 'learning'}

def simulate_stressed_patterns(simulation_params: Dict, memory_manager=None, 
                             personality_data: Dict = None) -> Dict[str, Any]:
    """Simuliert Stress-Muster - VEREINFACHT"""
    try:
        duration = simulation_params.get('duration', 30)
        sample_rate = simulation_params.get('sample_rate', 100)
        
        # Stress-Charakteristiken - High Beta, chaotisch
        stressed_characteristics = {
            'base_amplitude': 1.0,
            'noise_level': 0.12,  # Viel Rauschen bei Stress
            'variability': 0.25   # Hohe Unregelmäßigkeit
        }
        
        # Generiere Stress-Wellen-Muster
        stressed_patterns = {
            'beta_waves': generate_beta_waves(duration, sample_rate, {
                **stressed_characteristics, 'base_amplitude': 2.0  # Beta sehr hoch
            }),
            'gamma_waves': generate_gamma_waves(duration, sample_rate, {
                **stressed_characteristics, 'base_amplitude': 1.3  # Gamma Spitzen
            }),
            'alpha_waves': generate_alpha_waves(duration, sample_rate, {
                **stressed_characteristics, 'base_amplitude': 0.2  # Alpha stark unterdrückt
            }),
            'theta_waves': generate_theta_waves(duration, sample_rate, {
                **stressed_characteristics, 'base_amplitude': 0.4  # Theta gestört
            }),
            'delta_waves': generate_delta_waves(duration, sample_rate, {
                **stressed_characteristics, 'base_amplitude': 0.8  # Delta erhöht (Erschöpfung)
            })
        }
        
        return {
            'pattern_type': 'stressed',
            'wave_patterns': stressed_patterns,
            'pattern_analysis': {
                'stress_level': 0.9,
                'anxiety_index': 0.8,
                'mental_fatigue': 0.7
            },
            'simulation_info': {
                'is_real_eeg': False,
                'simulation_type': 'simplified_model',
                'characteristics': 'high_beta_chaotic'
            }
        }
        
    except Exception as e:
        logger.error(f"Stressed patterns simulation failed: {e}")
        return {'error': str(e), 'pattern_type': 'stressed'}

def simulate_dynamic_patterns(simulation_params: Dict, memory_manager=None, 
                            personality_data: Dict = None) -> Dict[str, Any]:
    """Simuliert dynamische Muster basierend auf aktuellem AI-Zustand - VEREINFACHT"""
    try:
        duration = simulation_params.get('duration', 30)
        sample_rate = simulation_params.get('sample_rate', 100)
        
        # Bestimme Zustand basierend auf verfügbaren Daten
        cognitive_load = 0.6  # Default
        emotional_stability = 0.7  # Default
        
        if memory_manager:
            try:
                if hasattr(memory_manager, 'get_current_cognitive_load'):
                    cognitive_load = memory_manager.get_current_cognitive_load()
            except:
                pass
        
        if personality_data:
            current_state = personality_data.get('current_state', {})
            emotional_stability = current_state.get('emotional_stability', 0.7)
        
        # Dynamische Charakteristiken
        dynamic_characteristics = {
            'base_amplitude': 1.0,
            'noise_level': 0.05 * (2.0 - emotional_stability),  # Mehr Rauschen bei niedrigerer Stabilität
            'variability': 0.1 + cognitive_load * 0.1  # Mehr Variabilität bei höherer Last
        }
        
        # Anpasse Amplituden basierend auf Zustand
        alpha_amp = 1.0 * emotional_stability  # Alpha korreliert mit emotionaler Stabilität
        beta_amp = 0.5 + cognitive_load * 1.0   # Beta korreliert mit kognitiver Last
        
        # Generiere dynamische Wellen-Muster
        dynamic_patterns = {
            'alpha_waves': generate_alpha_waves(duration, sample_rate, {
                **dynamic_characteristics, 'base_amplitude': alpha_amp
            }),
            'beta_waves': generate_beta_waves(duration, sample_rate, {
                **dynamic_characteristics, 'base_amplitude': beta_amp
            }),
            'theta_waves': generate_theta_waves(duration, sample_rate, {
                **dynamic_characteristics, 'base_amplitude': 0.8
            }),
            'gamma_waves': generate_gamma_waves(duration, sample_rate, {
                **dynamic_characteristics, 'base_amplitude': 0.6 + cognitive_load * 0.4
            }),
            'delta_waves': generate_delta_waves(duration, sample_rate, {
                **dynamic_characteristics, 'base_amplitude': 1.0 - cognitive_load * 0.3
            })
        }
        
        return {
            'pattern_type': 'dynamic',
            'wave_patterns': dynamic_patterns,
            'pattern_analysis': {
                'cognitive_load': cognitive_load,
                'emotional_stability': emotional_stability,
                'system_balance': (alpha_amp + (1.0 - beta_amp/2.0)) / 2.0
            },
            'simulation_info': {
                'is_real_eeg': False,
                'simulation_type': 'simplified_adaptive_model',
                'characteristics': f'cognitive_load_{cognitive_load:.1f}_stability_{emotional_stability:.1f}'
            }
        }
        
    except Exception as e:
        logger.error(f"Dynamic patterns simulation failed: {e}")
        return {'error': str(e), 'pattern_type': 'dynamic'}

__all__ = [
    'simulate_meditative_patterns',
    'simulate_focused_patterns', 
    'simulate_creative_patterns',
    'simulate_learning_patterns',
    'simulate_stressed_patterns',
    'simulate_dynamic_patterns'
]