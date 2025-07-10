"""
Personality Evolution Module
Evolution Timeline, Predictions und Development History Management
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def generate_evolution_timeline(personality_data: Dict, 
                              timeframe: str = '30d',
                              granularity: str = 'daily') -> Dict[str, Any]:
    """
    Generiert Evolution Timeline
    
    Extrahiert aus kira_routes.py.backup Evolution Timeline Logic
    """
    try:
        development_history = personality_data.get('development_history', [])
        interaction_history = personality_data.get('interaction_history', [])
        
        if not development_history and not interaction_history:
            return _generate_synthetic_evolution_timeline(personality_data, timeframe, granularity)
        
        # Parse timeframe
        days_back = _parse_timeframe_to_days(timeframe)
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter histories by timeframe
        filtered_development = [h for h in development_history 
                              if _parse_timestamp(h.get('timestamp')) >= cutoff_date]
        filtered_interactions = [h for h in interaction_history 
                               if _parse_timestamp(h.get('timestamp')) >= cutoff_date]
        
        # Generate timeline points
        timeline_points = _generate_timeline_points(
            filtered_development, filtered_interactions, days_back, granularity
        )
        
        # Evolution analysis
        evolution_analysis = {
            'timeline_points': timeline_points,
            'evolution_summary': _analyze_evolution_summary(timeline_points),
            'development_phases': _identify_development_phases(timeline_points),
            'milestone_events': _identify_milestone_events(timeline_points),
            'evolution_velocity': calculate_evolution_velocity(timeline_points),
            'trend_analysis': _analyze_evolution_trends(timeline_points)
        }
        
        # Timeline metadata
        evolution_analysis['timeline_metadata'] = {
            'timeframe': timeframe,
            'granularity': granularity,
            'data_points': len(timeline_points),
            'development_events': len(filtered_development),
            'interaction_events': len(filtered_interactions),
            'analysis_confidence': _calculate_timeline_confidence(timeline_points)
        }
        
        return evolution_analysis
        
    except Exception as e:
        logger.error(f"Evolution timeline generation failed: {e}")
        return {
            'error': str(e),
            'fallback_timeline': _generate_fallback_evolution_timeline(timeframe)
        }

def predict_personality_evolution(personality_data: Dict,
                                prediction_horizon: str = '90d') -> Dict[str, Any]:
    """
    Vorhersage der Personality Evolution
    
    Basiert auf kira_routes.py.backup Evolution Prediction Logic
    """
    try:
        current_traits = personality_data.get('traits', {})
        development_history = personality_data.get('development_history', [])
        current_state = personality_data.get('current_state', {})
        
        if not current_traits:
            return {'available': False, 'reason': 'no_current_traits'}
        
        # Parse prediction horizon
        prediction_days = _parse_timeframe_to_days(prediction_horizon)
        
        # Evolution predictions
        predictions = {
            'trait_evolution_predictions': _predict_trait_evolution(current_traits, development_history, prediction_days),
            'state_evolution_predictions': _predict_state_evolution(current_state, development_history, prediction_days),
            'behavioral_evolution_predictions': _predict_behavioral_evolution(personality_data, prediction_days),
            'capability_evolution_predictions': _predict_capability_evolution(personality_data, prediction_days)
        }
        
        # Prediction scenarios
        predictions['evolution_scenarios'] = {
            'optimistic_scenario': _generate_optimistic_evolution_scenario(predictions, personality_data),
            'realistic_scenario': _generate_realistic_evolution_scenario(predictions, personality_data),
            'conservative_scenario': _generate_conservative_evolution_scenario(predictions, personality_data)
        }
        
        # Evolution milestones
        predictions['predicted_milestones'] = _predict_evolution_milestones(predictions, prediction_days)
        
        # Prediction metadata
        predictions['prediction_metadata'] = {
            'prediction_horizon': prediction_horizon,
            'prediction_days': prediction_days,
            'base_data_quality': _assess_prediction_base_data_quality(personality_data),
            'prediction_confidence': _calculate_evolution_prediction_confidence(personality_data, prediction_days),
            'methodology': 'trait_trajectory_extrapolation_with_behavioral_modeling'
        }
        
        return predictions
        
    except Exception as e:
        logger.error(f"Personality evolution prediction failed: {e}")
        return {
            'available': False,
            'error': str(e),
            'fallback_predictions': _generate_fallback_evolution_predictions(prediction_horizon)
        }

def analyze_development_patterns(personality_data: Dict,
                               pattern_analysis_depth: str = 'comprehensive') -> Dict[str, Any]:
    """
    Analysiert Development Patterns
    
    Extrahiert aus kira_routes.py.backup Pattern Analysis Logic
    """
    try:
        development_history = personality_data.get('development_history', [])
        interaction_history = personality_data.get('interaction_history', [])
        
        if not development_history and not interaction_history:
            return {'available': False, 'reason': 'insufficient_history_data'}
        
        # Pattern analysis components
        pattern_analysis = {
            'temporal_patterns': _analyze_temporal_development_patterns(development_history),
            'cyclical_patterns': _analyze_cyclical_development_patterns(development_history),
            'trigger_patterns': _analyze_development_trigger_patterns(development_history, interaction_history),
            'progression_patterns': _analyze_progression_patterns(development_history),
            'adaptation_patterns': _analyze_adaptation_patterns(interaction_history)
        }
        
        # Comprehensive analysis
        if pattern_analysis_depth == 'comprehensive':
            pattern_analysis.update({
                'correlation_patterns': _analyze_development_correlations(development_history, interaction_history),
                'environmental_influence_patterns': _analyze_environmental_influence_patterns(development_history),
                'learning_patterns': _analyze_learning_patterns(development_history),
                'consolidation_patterns': _analyze_consolidation_patterns(development_history)
            })
        
        # Pattern insights
        pattern_analysis['pattern_insights'] = {
            'dominant_patterns': _identify_dominant_development_patterns(pattern_analysis),
            'pattern_stability': _assess_pattern_stability(pattern_analysis),
            'pattern_optimization_opportunities': _identify_pattern_optimization_opportunities(pattern_analysis),
            'pattern_based_recommendations': _generate_pattern_based_recommendations(pattern_analysis)
        }
        
        return pattern_analysis
        
    except Exception as e:
        logger.error(f"Development patterns analysis failed: {e}")
        return {
            'available': False,
            'error': str(e)
        }

def calculate_evolution_velocity(timeline_points: List[Dict] = None,
                               personality_data: Dict = None) -> Dict[str, Any]:
    """
    Berechnet Evolution Velocity
    
    Basiert auf kira_routes.py.backup Velocity Calculations
    """
    try:
        if not timeline_points and personality_data:
            # Generate timeline points from personality data
            timeline_points = _extract_timeline_from_personality_data(personality_data)
        
        if not timeline_points or len(timeline_points) < 2:
            return {'available': False, 'reason': 'insufficient_timeline_data'}
        
        # Velocity calculations
        velocity_metrics = {
            'overall_evolution_velocity': _calculate_overall_evolution_velocity(timeline_points),
            'trait_specific_velocities': _calculate_trait_specific_velocities(timeline_points),
            'development_acceleration': _calculate_development_acceleration(timeline_points),
            'velocity_consistency': _calculate_velocity_consistency(timeline_points),
            'peak_development_periods': _identify_peak_development_periods(timeline_points)
        }
        
        # Velocity analysis
        velocity_metrics['velocity_analysis'] = {
            'velocity_trend': _determine_velocity_trend(velocity_metrics),
            'velocity_stability': _assess_velocity_stability(velocity_metrics),
            'velocity_optimization_potential': _calculate_velocity_optimization_potential(velocity_metrics),
            'factors_affecting_velocity': _identify_velocity_factors(timeline_points)
        }
        
        # Velocity predictions
        velocity_metrics['velocity_predictions'] = {
            'short_term_velocity_forecast': _forecast_short_term_velocity(velocity_metrics, timeline_points),
            'optimal_development_windows': _identify_optimal_development_windows(velocity_metrics),
            'velocity_enhancement_strategies': _suggest_velocity_enhancement_strategies(velocity_metrics)
        }
        
        return velocity_metrics
        
    except Exception as e:
        logger.error(f"Evolution velocity calculation failed: {e}")
        return {
            'available': False,
            'error': str(e)
        }

# ====================================
# PRIVATE HELPER FUNCTIONS
# ====================================

def _generate_synthetic_evolution_timeline(personality_data: Dict, 
                                         timeframe: str, granularity: str) -> Dict[str, Any]:
    """Generiert synthetische Evolution Timeline wenn keine History verfügbar"""
    try:
        current_traits = personality_data.get('traits', {})
        current_state = personality_data.get('current_state', {})
        
        days_back = _parse_timeframe_to_days(timeframe)
        
        # Generate synthetic timeline points
        synthetic_points = []
        
        if granularity == 'daily':
            interval = 1
        elif granularity == 'weekly':
            interval = 7
        else:  # hourly or custom
            interval = 1
        
        for day in range(0, days_back, interval):
            timestamp = datetime.now() - timedelta(days=day)
            
            # Simulate evolution based on current state
            synthetic_point = {
                'timestamp': timestamp.isoformat(),
                'day_offset': day,
                'synthetic_data': True,
                'traits_snapshot': _simulate_historical_traits(current_traits, day),
                'state_snapshot': _simulate_historical_state(current_state, day),
                'development_events': _simulate_development_events(day),
                'confidence': max(0.1, 1.0 - (day / days_back))  # Confidence decreases with time
            }
            
            synthetic_points.append(synthetic_point)
        
        synthetic_points.reverse()  # Chronological order
        
        return {
            'timeline_points': synthetic_points,
            'synthetic_data': True,
            'evolution_summary': {
                'data_type': 'synthetic',
                'confidence': 'low',
                'recommendation': 'Enable development history tracking for accurate evolution analysis'
            },
            'timeline_metadata': {
                'timeframe': timeframe,
                'granularity': granularity,
                'data_points': len(synthetic_points),
                'synthetic_generation': True
            }
        }
        
    except Exception as e:
        logger.debug(f"Synthetic evolution timeline generation failed: {e}")
        return {
            'error': str(e),
            'fallback_timeline': _generate_fallback_evolution_timeline(timeframe)
        }

def _generate_timeline_points(development_history: List, interaction_history: List,
                            days_back: int, granularity: str) -> List[Dict]:
    """Generiert Timeline Points aus History Data"""
    try:
        timeline_points = []
        
        # Group events by time intervals based on granularity
        if granularity == 'daily':
            interval_hours = 24
        elif granularity == 'weekly':
            interval_hours = 168
        elif granularity == 'hourly':
            interval_hours = 1
        else:
            interval_hours = 24
        
        # Create time buckets
        current_time = datetime.now()
        for i in range(0, days_back * 24, interval_hours):
            bucket_start = current_time - timedelta(hours=i + interval_hours)
            bucket_end = current_time - timedelta(hours=i)
            
            # Find events in this bucket
            bucket_developments = [
                d for d in development_history 
                if bucket_start <= _parse_timestamp(d.get('timestamp')) < bucket_end
            ]
            
            bucket_interactions = [
                i for i in interaction_history 
                if bucket_start <= _parse_timestamp(i.get('timestamp')) < bucket_end
            ]
            
            if bucket_developments or bucket_interactions:
                timeline_point = {
                    'timestamp': bucket_end.isoformat(),
                    'period_start': bucket_start.isoformat(),
                    'period_end': bucket_end.isoformat(),
                    'development_events': bucket_developments,
                    'interaction_events': bucket_interactions,
                    'development_count': len(bucket_developments),
                    'interaction_count': len(bucket_interactions),
                    'activity_level': _calculate_activity_level(bucket_developments, bucket_interactions),
                    'key_developments': _extract_key_developments(bucket_developments),
                    'evolution_indicators': _calculate_evolution_indicators(bucket_developments, bucket_interactions)
                }
                
                timeline_points.append(timeline_point)
        
        timeline_points.reverse()  # Chronological order
        return timeline_points
        
    except Exception as e:
        logger.debug(f"Timeline points generation failed: {e}")
        return []

def _predict_trait_evolution(current_traits: Dict, development_history: List, 
                           prediction_days: int) -> Dict[str, Any]:
    """Vorhersage der Trait Evolution"""
    try:
        trait_predictions = {}
        
        for trait_name, trait_data in current_traits.items():
            if isinstance(trait_data, dict):
                current_strength = trait_data.get('current_strength', 0.5)
                development_rate = trait_data.get('development_rate', 0.01)
                base_strength = trait_data.get('base_strength', 0.5)
                
                # Calculate historical development velocity
                historical_velocity = _calculate_trait_historical_velocity(trait_name, development_history)
                
                # Predict future evolution
                daily_predictions = []
                for day in range(1, min(prediction_days + 1, 91)):  # Max 90 days
                    # Apply development rate with diminishing returns
                    growth_factor = _calculate_trait_growth_factor(current_strength, day, development_rate)
                    predicted_strength = current_strength + (development_rate * day * growth_factor)
                    
                    # Apply bounds
                    predicted_strength = max(0.0, min(1.0, predicted_strength))
                    
                    daily_predictions.append({
                        'day': day,
                        'predicted_strength': predicted_strength,
                        'confidence': _calculate_trait_prediction_confidence(day, historical_velocity),
                        'development_phase': _determine_trait_development_phase(predicted_strength)
                    })
                
                trait_predictions[trait_name] = {
                    'current_strength': current_strength,
                    'predicted_final_strength': daily_predictions[-1]['predicted_strength'] if daily_predictions else current_strength,
                    'expected_growth': daily_predictions[-1]['predicted_strength'] - current_strength if daily_predictions else 0,
                    'development_trajectory': 'growth' if daily_predictions and daily_predictions[-1]['predicted_strength'] > current_strength else 'stable',
                    'daily_predictions': daily_predictions[-30:],  # Last 30 days
                    'key_milestones': _identify_trait_evolution_milestones(daily_predictions)
                }
        
        return trait_predictions
        
    except Exception as e:
        logger.debug(f"Trait evolution prediction failed: {e}")
        return {}

def _predict_state_evolution(current_state: Dict, development_history: List,
                           prediction_days: int) -> Dict[str, Any]:
    """Vorhersage der State Evolution"""
    try:
        state_predictions = {}
        
        # Key state components to predict
        state_components = [
            'emotional_stability', 'adaptability', 'empathy_level',
            'social_confidence', 'learning_motivation', 'creativity_level'
        ]
        
        for component in state_components:
            current_value = current_state.get(component, 0.7)
            
            # Analyze historical changes for this component
            historical_changes = _extract_state_component_history(component, development_history)
            trend = _calculate_state_component_trend(historical_changes)
            
            # Predict evolution
            predictions = []
            for day in range(1, min(prediction_days + 1, 61)):  # Max 60 days for state
                # State changes are typically more gradual than trait changes
                daily_change = trend * 0.1  # Reduce volatility
                predicted_value = current_value + (daily_change * day)
                
                # Apply realistic bounds for state components
                predicted_value = max(0.1, min(0.95, predicted_value))
                
                predictions.append({
                    'day': day,
                    'predicted_value': predicted_value,
                    'confidence': _calculate_state_prediction_confidence(day, len(historical_changes))
                })
            
            state_predictions[component] = {
                'current_value': current_value,
                'predicted_final_value': predictions[-1]['predicted_value'] if predictions else current_value,
                'trend_direction': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
                'predictions': predictions[-14:],  # Last 14 days
                'stability_forecast': _forecast_component_stability(predictions)
            }
        
        return state_predictions
        
    except Exception as e:
        logger.debug(f"State evolution prediction failed: {e}")
        return {}

def _analyze_temporal_development_patterns(development_history: List) -> Dict[str, Any]:
    """Analysiert temporale Development Patterns"""
    try:
        if not development_history:
            return {'available': False, 'reason': 'no_development_history'}
        
        # Extract timestamps and analyze patterns
        timestamps = [_parse_timestamp(event.get('timestamp')) for event in development_history]
        timestamps = [ts for ts in timestamps if ts is not None]
        
        if len(timestamps) < 3:
            return {'available': False, 'reason': 'insufficient_temporal_data'}
        
        # Analyze temporal patterns
        hourly_distribution = _analyze_hourly_distribution(timestamps)
        daily_distribution = _analyze_daily_distribution(timestamps)
        weekly_distribution = _analyze_weekly_distribution(timestamps)
        
        # Peak activity periods
        peak_hours = _identify_peak_development_hours(hourly_distribution)
        peak_days = _identify_peak_development_days(daily_distribution)
        
        return {
            'available': True,
            'temporal_patterns': {
                'hourly_distribution': hourly_distribution,
                'daily_distribution': daily_distribution,
                'weekly_distribution': weekly_distribution,
                'peak_development_hours': peak_hours,
                'peak_development_days': peak_days,
                'activity_consistency': _calculate_temporal_consistency(timestamps)
            },
            'temporal_insights': {
                'most_active_period': _identify_most_active_period(hourly_distribution, daily_distribution),
                'development_rhythm': _characterize_development_rhythm(timestamps),
                'temporal_optimization_suggestions': _suggest_temporal_optimizations(hourly_distribution, daily_distribution)
            }
        }
        
    except Exception as e:
        logger.debug(f"Temporal development patterns analysis failed: {e}")
        return {'available': False, 'error': str(e)}

# Additional helper functions...

def _parse_timeframe_to_days(timeframe: str) -> int:
    """Parst Timeframe zu Tagen"""
    timeframe_map = {
        '7d': 7, '1w': 7, '14d': 14, '2w': 14,
        '30d': 30, '1m': 30, '60d': 60, '2m': 60,
        '90d': 90, '3m': 90, '180d': 180, '6m': 180,
        '365d': 365, '1y': 365
    }
    return timeframe_map.get(timeframe, 30)

def _parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parst Timestamp String zu datetime"""
    try:
        if isinstance(timestamp_str, str):
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return None
    except Exception:
        return None

def _generate_fallback_evolution_timeline(timeframe: str) -> Dict[str, Any]:
    """Generiert Fallback Evolution Timeline"""
    return {
        'fallback_mode': True,
        'basic_timeline': {
            'timeframe': timeframe,
            'status': 'limited_data_available',
            'recommendation': 'Enable personality development tracking for detailed evolution analysis'
        },
        'timestamp': datetime.now().isoformat()
    }

__all__ = [
    'generate_evolution_timeline',
    'predict_personality_evolution',
    'analyze_development_patterns',
    'calculate_evolution_velocity'
]