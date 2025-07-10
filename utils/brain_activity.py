"""
Kira Brain Activity Utilities
Advanced brain wave simulation and neural processing functions
"""

import logging
import math
import time
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def generate_brain_activity(system_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate realistic brain activity data based on system state
    
    Args:
        system_state: Current system state dictionary
        
    Returns:
        Dictionary with comprehensive brain activity data
    """
    try:
        # Check if neural processing system is available
        neural_data = _try_get_neural_processing_data()
        
        if neural_data:
            return convert_neural_to_dashboard_values(neural_data, system_state)
        else:
            return generate_system_based_brain_activity(system_state)
            
    except Exception as e:
        logger.error(f"Brain activity generation failed: {e}")
        return _get_emergency_fallback_brain_data()


def _try_get_neural_processing_data() -> Dict[str, Any]:
    """Try to get data from neural processing system"""
    try:
        # This would connect to actual neural processing system
        # For now, simulate realistic neural data
        return _simulate_neural_processing_data()
    except:
        return None


def _simulate_neural_processing_data() -> Dict[str, Any]:
    """Simulate realistic neural processing data"""
    current_time = time.time()
    
    # Generate realistic wave patterns
    wave_patterns = {
        'alpha_waves': {
            'frequency': 10.5,  # Hz
            'amplitude': random.uniform(0.5, 1.5),
            'samples': [math.sin(current_time * 10.5 + i) * random.uniform(15, 35) + 25 
                       for i in range(10)]
        },
        'beta_waves': {
            'frequency': 20.0,  # Hz
            'amplitude': random.uniform(0.8, 2.0),
            'samples': [math.sin(current_time * 20.0 + i) * random.uniform(20, 50) + 35 
                       for i in range(10)]
        },
        'theta_waves': {
            'frequency': 6.0,   # Hz
            'amplitude': random.uniform(0.3, 1.0),
            'samples': [math.sin(current_time * 6.0 + i) * random.uniform(5, 25) + 15 
                       for i in range(10)]
        },
        'gamma_waves': {
            'frequency': 40.0,  # Hz
            'amplitude': random.uniform(0.2, 0.8),
            'samples': [math.sin(current_time * 40.0 + i) * random.uniform(3, 15) + 8 
                       for i in range(10)]
        }
    }
    
    # Pattern analysis
    pattern_analysis = {
        'cognitive_load': random.uniform(0.3, 0.9),
        'emotional_stability': random.uniform(0.5, 0.95),
        'pattern_complexity': random.uniform(0.4, 0.8),
        'attention_level': random.uniform(0.4, 0.9)
    }
    
    return {
        'wave_patterns': wave_patterns,
        'pattern_analysis': pattern_analysis,
        'pattern_type': 'dynamic',
        'timestamp': current_time
    }


def convert_neural_to_dashboard_values(neural_data: Dict[str, Any], system_state: Dict[str, Any]) -> Dict[str, Any]:
    """Convert neural processing data to dashboard-compatible values"""
    try:
        wave_patterns = neural_data.get('wave_patterns', {})
        pattern_analysis = neural_data.get('pattern_analysis', {})

        # Realistic ranges for dashboard
        wave_ranges = {
            'alpha_waves': {'min': 5, 'max': 45, 'baseline': 25},
            'beta_waves': {'min': 10, 'max': 60, 'baseline': 35},
            'theta_waves': {'min': 2, 'max': 30, 'baseline': 15},
            'gamma_waves': {'min': 1, 'max': 20, 'baseline': 8}
        }

        brain_values = {}

        # Convert each wave type
        for wave_type, wave_data in wave_patterns.items():
            if isinstance(wave_data, dict) and 'samples' in wave_data:
                samples = wave_data['samples']
                if samples and len(samples) > 0:
                    # Use latest sample and apply realistic constraints
                    latest_sample = samples[-1]
                    wave_key = wave_type.replace('_waves', '')
                    
                    if wave_key in ['alpha', 'beta', 'theta', 'gamma']:
                        range_info = wave_ranges.get(f'{wave_key}_waves', {'min': 0, 'max': 100, 'baseline': 50})
                        
                        # Constrain to realistic range
                        constrained_value = max(range_info['min'], 
                                              min(range_info['max'], abs(latest_sample)))
                        
                        brain_values[wave_key] = round(constrained_value, 1)

        # Fallback if no values found
        if not brain_values:
            brain_values = {
                'alpha': 25.0,
                'beta': 35.0,
                'theta': 15.0,
                'gamma': 8.0
            }

        # Determine activity level
        activity_level = determine_activity_level(brain_values, pattern_analysis, system_state)

        # Calculate system load
        system_load = calculate_system_load_from_waves(brain_values)

        return {
            'alpha': brain_values.get('alpha', 25.0),
            'beta': brain_values.get('beta', 35.0),
            'theta': brain_values.get('theta', 15.0),
            'gamma': brain_values.get('gamma', 8.0),
            'activity_level': activity_level,
            'system_load': system_load,
            'timestamp': time.time(),
            'data_source': 'neural_processing_system',
            'pattern_type': neural_data.get('pattern_type', 'dynamic'),
            'neural_analysis': {
                'cognitive_load': pattern_analysis.get('cognitive_load', 0.5),
                'emotional_stability': pattern_analysis.get('emotional_stability', 0.7),
                'pattern_complexity': pattern_analysis.get('pattern_complexity', 0.6)
            }
        }

    except Exception as e:
        logger.error(f"Neural data conversion failed: {e}")
        return generate_system_based_brain_activity(system_state)


def generate_system_based_brain_activity(system_state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate system-based realistic brain activity"""
    try:
        # System health based on available systems
        available_systems = sum(1 for sys in system_state['systems_status'].values() if sys['available'])
        system_factor = available_systems / 5.0  # 0.0 - 1.0

        # Time-based base activity
        hour = datetime.now().hour
        if 6 <= hour < 10:
            # Morning - Beta dominant
            base_alpha, base_beta, base_theta, base_gamma = 20, 45, 12, 6
            activity_label = "Morning Alert"
        elif 10 <= hour < 14:
            # Forenoon - High concentration
            base_alpha, base_beta, base_theta, base_gamma = 15, 55, 10, 12
            activity_label = "Peak Performance"  
        elif 14 <= hour < 18:
            # Afternoon - Normal activity
            base_alpha, base_beta, base_theta, base_gamma = 25, 40, 15, 8
            activity_label = "Afternoon Active"
        elif 18 <= hour < 22:
            # Evening - Relaxation
            base_alpha, base_beta, base_theta, base_gamma = 35, 25, 20, 5
            activity_label = "Evening Calm"
        else:
            # Night - Theta/Alpha dominant
            base_alpha, base_beta, base_theta, base_gamma = 40, 15, 25, 3
            activity_label = "Night Rest"

        # System health influences values
        health_multiplier = 0.7 + (system_factor * 0.6)  # 0.7 - 1.3

        # Natural fluctuations
        current_time = time.time()
        sine_variation = math.sin(current_time / 20) * 0.15  # -0.15 to +0.15
        noise = random.uniform(-0.1, 0.1)

        total_variation = (sine_variation + noise) * health_multiplier

        # Calculate final values
        alpha = max(5, min(45, base_alpha * (1 + total_variation)))
        beta = max(10, min(60, base_beta * (1 + total_variation)))
        theta = max(2, min(30, base_theta * (1 + total_variation)))
        gamma = max(1, min(20, base_gamma * (1 + total_variation)))

        # System load
        system_load = calculate_system_load_from_waves({
            'alpha': alpha, 'beta': beta, 'theta': theta, 'gamma': gamma
        })

        return {
            'alpha': round(alpha, 1),
            'beta': round(beta, 1), 
            'theta': round(theta, 1),
            'gamma': round(gamma, 1),
            'activity_level': activity_label,
            'system_load': system_load,
            'timestamp': time.time(),
            'data_source': 'system_health_simulation',
            'system_factor': system_factor,
            'health_multiplier': health_multiplier,
            'available_systems': available_systems
        }

    except Exception as e:
        logger.error(f"System-based brain activity generation failed: {e}")
        return _get_emergency_fallback_brain_data()


def determine_activity_level(brain_values: Dict, pattern_analysis: Dict, system_state: Dict) -> str:
    """Determine activity level based on brain values and system state"""
    try:
        # Calculate total activity
        total_activity = sum(brain_values.values())

        # System health factor
        available_systems = sum(1 for sys in system_state['systems_status'].values() if sys['available'])
        system_factor = available_systems / 5.0  # 0.0 - 1.0

        # Cognitive load from pattern analysis
        cognitive_load = pattern_analysis.get('cognitive_load', 0.5)
        emotional_stability = pattern_analysis.get('emotional_stability', 0.7)

        # Time factor
        hour = datetime.now().hour
        if 6 <= hour < 10:
            time_label = "Morning Alert"
            activity_threshold = 80
        elif 10 <= hour < 14:
            time_label = "Peak Performance"
            activity_threshold = 100
        elif 14 <= hour < 18:
            time_label = "Afternoon Active"
            activity_threshold = 90
        elif 18 <= hour < 22:
            time_label = "Evening Calm"
            activity_threshold = 70
        else:
            time_label = "Night Rest"
            activity_threshold = 50

        # Determine final activity level
        if total_activity > activity_threshold * 1.2 or cognitive_load > 0.8:
            return f"{time_label} - Peak"
        elif total_activity > activity_threshold or cognitive_load > 0.6:
            return time_label
        elif total_activity > activity_threshold * 0.8 or emotional_stability > 0.8:
            return f"{time_label} - Relaxed"
        else:
            return f"{time_label} - Low"

    except Exception as e:
        logger.error(f"Activity level determination failed: {e}")
        return 'Normal Activity'


def calculate_system_load_from_waves(brain_values: Dict) -> float:
    """Calculate system load based on brain waves"""
    try:
        # Beta and Gamma = Primary system load indicators
        beta = brain_values.get('beta', 35.0)
        gamma = brain_values.get('gamma', 8.0)
        alpha = brain_values.get('alpha', 25.0)
        theta = brain_values.get('theta', 15.0)

        # System load = weighted combination
        # Beta: Alertness/concentration (moderate weight)
        # Gamma: High-frequency processing (high weight)
        # Alpha: Relaxation (negative weight)
        # Theta: Creativity/dreaming (slight weight)

        system_load = (
            beta * 0.6 +           # Beta - concentration
            gamma * 3.0 +          # Gamma - intensive processing  
            theta * 0.3 -          # Theta - creative processing
            alpha * 0.2            # Alpha - relaxation reduces load
        )

        # Normalize to 0-100%
        system_load = max(0, min(100, system_load))

        return round(system_load, 1)

    except Exception as e:
        logger.error(f"System load calculation failed: {e}")
        return 50.0


def get_activity_patterns(brain_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get detailed activity patterns from brain data"""
    try:
        patterns = {
            'dominant_frequency': determine_dominant_frequency(brain_data),
            'pattern_complexity': calculate_pattern_complexity(brain_data),
            'coherence_score': calculate_coherence_score(brain_data),
            'stability_index': calculate_stability_index(brain_data),
            'brain_state': get_brain_state_classification(brain_data),
            'insights': generate_brain_insights(brain_data, {})
        }
        
        return patterns
        
    except Exception as e:
        logger.error(f"Activity patterns analysis failed: {e}")
        return {
            'error': str(e),
            'patterns_available': False
        }


def determine_dominant_frequency(brain_data: Dict) -> str:
    """Determine dominant brain wave frequency"""
    try:
        # Extract wave values
        alpha = brain_data.get('alpha', 25.0)
        beta = brain_data.get('beta', 35.0)
        theta = brain_data.get('theta', 15.0)
        gamma = brain_data.get('gamma', 8.0)
        
        # Find highest value
        wave_values = {
            'alpha': alpha,
            'beta': beta,
            'theta': theta,
            'gamma': gamma
        }
        
        dominant_wave = max(wave_values, key=wave_values.get)
        dominant_value = wave_values[dominant_wave]
        
        # Determine intensity
        if dominant_value > 50:
            intensity = "Very High"
        elif dominant_value > 40:
            intensity = "High"
        elif dominant_value > 30:
            intensity = "Moderate"
        elif dominant_value > 20:
            intensity = "Low"
        else:
            intensity = "Very Low"
        
        # Wave-specific descriptions
        wave_descriptions = {
            'alpha': {
                'freq_range': '8-13 Hz',
                'state': 'Relaxed Awareness',
                'characteristics': 'Calm, meditative, creative flow'
            },
            'beta': {
                'freq_range': '13-30 Hz',
                'state': 'Active Concentration',
                'characteristics': 'Focused thinking, problem solving'
            },
            'theta': {
                'freq_range': '4-8 Hz',
                'state': 'Deep Creativity',
                'characteristics': 'Intuitive insights, memory formation'
            },
            'gamma': {
                'freq_range': '30-100 Hz',
                'state': 'High-Level Processing',
                'characteristics': 'Complex cognitive tasks, binding consciousness'
            }
        }
        
        description = wave_descriptions.get(dominant_wave, {})
        
        return f"{dominant_wave.capitalize()} ({description.get('freq_range', 'Unknown')}) - {intensity} {description.get('state', 'Activity')}"
        
    except Exception as e:
        logger.error(f"Dominant frequency determination failed: {e}")
        return "Mixed Frequencies - Balanced Activity"


def calculate_pattern_complexity(brain_data: Dict) -> float:
    """Calculate complexity of brain wave patterns"""
    try:
        # Extract wave values
        alpha = brain_data.get('alpha', 25.0)
        beta = brain_data.get('beta', 35.0)
        theta = brain_data.get('theta', 15.0)
        gamma = brain_data.get('gamma', 8.0)
        
        wave_values = [alpha, beta, theta, gamma]
        
        # 1. Variance of values (higher variance = higher complexity)
        mean_value = sum(wave_values) / len(wave_values)
        variance = sum((x - mean_value) ** 2 for x in wave_values) / len(wave_values)
        variance_complexity = min(1.0, variance / 400.0)  # Normalized to 0-1
        
        # 2. Number of active frequency bands (> 60% of average)
        active_bands = sum(1 for value in wave_values if value > mean_value * 0.6)
        band_complexity = active_bands / 4.0  # 0-1 based on 4 bands
        
        # 3. Gamma activity (indicator for complex processing)
        gamma_complexity = min(1.0, gamma / 20.0)  # Gamma > 20 = very complex
        
        # 4. Beta/Alpha ratio (activity vs. relaxation balance)
        if alpha > 0:
            beta_alpha_ratio = beta / alpha
            ratio_complexity = min(1.0, abs(beta_alpha_ratio - 1.4) / 2.0)  # Optimal ~1.4
        else:
            ratio_complexity = 0.5
        
        # Weighted total complexity
        total_complexity = (
            variance_complexity * 0.4 +      # Wave variance
            band_complexity * 0.3 +         # Active frequency bands
            gamma_complexity * 0.2 +        # Gamma activity
            ratio_complexity * 0.1          # Beta/Alpha balance
        )
        
        # Ensure value is between 0 and 1
        total_complexity = max(0.0, min(1.0, total_complexity))
        
        return round(total_complexity, 3)
        
    except Exception as e:
        logger.error(f"Pattern complexity calculation failed: {e}")
        return 0.5  # Medium complexity as fallback


def calculate_coherence_score(brain_data: Dict) -> float:
    """Calculate coherence between different brain wave frequencies"""
    try:
        alpha = brain_data.get('alpha', 25.0)
        beta = brain_data.get('beta', 35.0)
        theta = brain_data.get('theta', 15.0)
        gamma = brain_data.get('gamma', 8.0)
        
        # Calculate coherence between frequency bands
        coherence_scores = []
        
        # Alpha-Theta coherence (creativity + relaxation)
        if alpha > 0 and theta > 0:
            alpha_theta_coherence = min(alpha, theta) / max(alpha, theta)
            coherence_scores.append(alpha_theta_coherence)
        
        # Beta-Gamma coherence (concentration + processing)
        if beta > 0 and gamma > 0:
            beta_gamma_coherence = min(beta, gamma) / max(beta, gamma)
            coherence_scores.append(beta_gamma_coherence)
        
        # Overall balance
        wave_values = [alpha, beta, theta, gamma]
        max_wave = max(wave_values)
        min_wave = min(wave_values)
        
        if max_wave > 0:
            balance_score = min_wave / max_wave
            coherence_scores.append(balance_score)
        
        # Calculate average coherence
        if coherence_scores:
            total_coherence = sum(coherence_scores) / len(coherence_scores)
        else:
            total_coherence = 0.5
        
        return round(total_coherence, 3)
        
    except Exception as e:
        logger.error(f"Coherence score calculation failed: {e}")
        return 0.6  # Medium coherence as fallback


def calculate_stability_index(brain_data: Dict) -> float:
    """Calculate stability index of brain wave activity"""
    try:
        alpha = brain_data.get('alpha', 25.0)
        beta = brain_data.get('beta', 35.0)
        theta = brain_data.get('theta', 15.0)
        gamma = brain_data.get('gamma', 8.0)
        
        # Normal ranges for stability assessment
        normal_ranges = {
            'alpha': (15, 35),    # Normal Alpha range
            'beta': (20, 50),     # Normal Beta range  
            'theta': (8, 25),     # Normal Theta range
            'gamma': (3, 15)      # Normal Gamma range
        }
        
        stability_scores = []
        
        for wave_name, value in zip(['alpha', 'beta', 'theta', 'gamma'], [alpha, beta, theta, gamma]):
            min_range, max_range = normal_ranges[wave_name]
            
            if min_range <= value <= max_range:
                # Perfect stability in range
                stability_scores.append(1.0)
            else:
                # Reduced stability outside range
                if value < min_range:
                    stability = value / min_range
                else:
                    stability = max_range / value
                stability_scores.append(max(0.0, stability))
        
        # Calculate average stability
        total_stability = sum(stability_scores) / len(stability_scores)
        
        return round(total_stability, 3)
        
    except Exception as e:
        logger.error(f"Stability index calculation failed: {e}")
        return 0.7  # Moderate stability as fallback


def get_brain_state_classification(brain_data: Dict) -> Dict[str, Any]:
    """Classify current brain state based on wave patterns"""
    try:
        alpha = brain_data.get('alpha', 25.0)
        beta = brain_data.get('beta', 35.0)
        theta = brain_data.get('theta', 15.0)
        gamma = brain_data.get('gamma', 8.0)
        
        # State classification logic
        if beta > 45 and gamma > 10:
            state = "High Cognitive Load"
            description = "Intense mental processing and concentration"
            recommendations = ["Consider taking breaks", "Monitor stress levels"]
            
        elif alpha > 30 and theta > 18:
            state = "Creative Flow"
            description = "Optimal state for creative and intuitive thinking"
            recommendations = ["Great time for creative work", "Maintain this state"]
            
        elif beta > 40 and alpha < 20:
            state = "Active Focus"
            description = "High concentration and analytical thinking"
            recommendations = ["Good for problem solving", "Watch for mental fatigue"]
            
        elif alpha > 35 and beta < 25:
            state = "Relaxed Awareness"
            description = "Calm, meditative state with gentle awareness"
            recommendations = ["Perfect for meditation", "Consider learning activities"]
            
        elif theta > 20 and alpha > 25:
            state = "Deep Reflection"
            description = "Introspective state ideal for memory consolidation"
            recommendations = ["Good for reviewing information", "Allow natural flow"]
            
        elif gamma > 12:
            state = "Peak Performance"
            description = "High-level cognitive integration and awareness"
            recommendations = ["Optimal for complex tasks", "Maintain focus"]
            
        else:
            state = "Balanced Activity"
            description = "Normal, well-balanced brain activity"
            recommendations = ["All activities suitable", "Maintain balance"]
        
        # Calculate state confidence
        dominant_wave = max([alpha, beta, theta, gamma])
        state_confidence = min(1.0, dominant_wave / 50.0)
        
        return {
            'state': state,
            'description': description,
            'confidence': round(state_confidence, 2),
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Brain state classification failed: {e}")
        return {
            'state': 'Unknown State',
            'description': 'Unable to determine current brain state',
            'confidence': 0.0,
            'recommendations': ['Check system status'],
            'error': str(e)
        }


def generate_brain_insights(brain_data: Dict, system_state: Dict) -> Dict[str, Any]:
    """Generate extended insights based on brain activity"""
    try:
        # Basic classification
        state_classification = get_brain_state_classification(brain_data)
        
        # Performance insights
        performance_insights = []
        
        if brain_data.get('beta', 35) > 50:
            performance_insights.append("High analytical capability detected")
        
        if brain_data.get('alpha', 25) > 30:
            performance_insights.append("Excellent relaxation and focus balance")
        
        if brain_data.get('gamma', 8) > 10:
            performance_insights.append("Advanced cognitive processing active")
        
        # Wellness recommendations
        wellness_recommendations = []
        
        if brain_data.get('beta', 35) > 45 and brain_data.get('alpha', 25) < 20:
            wellness_recommendations.append("Consider relaxation techniques to balance high beta activity")
        
        if brain_data.get('theta', 15) < 10:
            wellness_recommendations.append("Increase creative activities to boost theta waves")
        
        if calculate_stability_index(brain_data) < 0.5:
            wellness_recommendations.append("Focus on activities that promote brain stability")
        
        return {
            'state_classification': state_classification,
            'performance_insights': performance_insights,
            'wellness_recommendations': wellness_recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Brain insights generation failed: {e}")
        return {
            'error': str(e),
            'basic_state': 'Analysis unavailable',
            'generated_at': datetime.now().isoformat()
        }


def _get_emergency_fallback_brain_data() -> Dict[str, Any]:
    """Emergency fallback brain data"""
    return {
        'alpha': 25.0, 
        'beta': 35.0, 
        'theta': 15.0, 
        'gamma': 8.0,
        'activity_level': 'Normal Activity', 
        'system_load': 50.0,
        'timestamp': time.time(), 
        'data_source': 'emergency_fallback'
    }