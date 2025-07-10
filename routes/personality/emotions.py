"""
Personality Emotions Module
Emotional State Analysis, Patterns und Emotional Intelligence
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def analyze_emotional_state(personality_data: Dict,
                          analysis_depth: str = 'comprehensive') -> Dict[str, Any]:
    """
    Analysiert aktuellen Emotional State
    
    Extrahiert aus kira_routes.py.backup Emotional State Analysis Logic
    """
    try:
        current_state = personality_data.get('current_state', {})
        emotional_history = personality_data.get('emotional_history', [])
        interaction_history = personality_data.get('interaction_history', [])
        
        if not current_state:
            return _generate_fallback_emotional_analysis()
        
        # Core emotional metrics
        emotional_analysis = {
            'current_emotional_metrics': _extract_current_emotional_metrics(current_state),
            'emotional_stability_analysis': _analyze_emotional_stability(current_state, emotional_history),
            'empathy_analysis': _analyze_empathy_levels(current_state, interaction_history),
            'adaptability_analysis': _analyze_emotional_adaptability(current_state, emotional_history),
            'emotional_resilience': _assess_emotional_resilience(current_state, emotional_history)
        }
        
        # Comprehensive analysis
        if analysis_depth == 'comprehensive':
            emotional_analysis.update({
                'emotional_complexity_analysis': _analyze_emotional_complexity(current_state, emotional_history),
                'emotional_expression_patterns': _analyze_emotional_expression_patterns(interaction_history),
                'emotional_regulation_analysis': _analyze_emotional_regulation(emotional_history),
                'emotional_learning_patterns': _analyze_emotional_learning_patterns(emotional_history, interaction_history),
                'emotional_growth_indicators': _assess_emotional_growth_indicators(current_state, emotional_history)
            })
        
        # Emotional insights
        emotional_analysis['emotional_insights'] = {
            'dominant_emotional_patterns': _identify_dominant_emotional_patterns(emotional_analysis),
            'emotional_strengths': _identify_emotional_strengths(emotional_analysis),
            'emotional_development_areas': _identify_emotional_development_areas(emotional_analysis),
            'emotional_optimization_opportunities': _identify_emotional_optimization_opportunities(emotional_analysis)
        }
        
        # Analysis metadata
        emotional_analysis['analysis_metadata'] = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_depth': analysis_depth,
            'data_sources': {
                'current_state_available': bool(current_state),
                'emotional_history_points': len(emotional_history),
                'interaction_history_points': len(interaction_history)
            },
            'analysis_confidence': _calculate_emotional_analysis_confidence(current_state, emotional_history)
        }
        
        return emotional_analysis
        
    except Exception as e:
        logger.error(f"Emotional state analysis failed: {e}")
        return {
            'error': str(e),
            'fallback_analysis': _generate_fallback_emotional_analysis()
        }

def track_emotional_patterns(personality_data: Dict,
                           pattern_timeframe: str = '30d') -> Dict[str, Any]:
    """
    Verfolgt Emotional Patterns über Zeit
    
    Basiert auf kira_routes.py.backup Emotional Pattern Tracking Logic
    """
    try:
        emotional_history = personality_data.get('emotional_history', [])
        interaction_history = personality_data.get('interaction_history', [])
        
        if not emotional_history and not interaction_history:
            return {'available': False, 'reason': 'no_emotional_history_data'}
        
        # Parse timeframe
        days_back = _parse_timeframe_to_days(pattern_timeframe)
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter data by timeframe
        filtered_emotional_history = [
            h for h in emotional_history 
            if _parse_timestamp(h.get('timestamp')) >= cutoff_date
        ]
        
        filtered_interactions = [
            h for h in interaction_history 
            if _parse_timestamp(h.get('timestamp')) >= cutoff_date
        ]
        
        # Pattern analysis
        pattern_analysis = {
            'emotional_stability_patterns': _track_stability_patterns(filtered_emotional_history),
            'emotional_intensity_patterns': _track_intensity_patterns(filtered_emotional_history),
            'emotional_trigger_patterns': _track_trigger_patterns(filtered_emotional_history, filtered_interactions),
            'emotional_response_patterns': _track_response_patterns(filtered_interactions),
            'emotional_recovery_patterns': _track_recovery_patterns(filtered_emotional_history),
            'cyclical_emotional_patterns': _track_cyclical_patterns(filtered_emotional_history)
        }
        
        # Pattern insights
        pattern_analysis['pattern_insights'] = {
            'strongest_patterns': _identify_strongest_emotional_patterns(pattern_analysis),
            'pattern_consistency': _assess_emotional_pattern_consistency(pattern_analysis),
            'pattern_evolution': _analyze_emotional_pattern_evolution(pattern_analysis),
            'pattern_based_predictions': _generate_pattern_based_emotional_predictions(pattern_analysis)
        }
        
        # Pattern recommendations
        pattern_analysis['pattern_recommendations'] = {
            'optimization_strategies': _suggest_emotional_pattern_optimization(pattern_analysis),
            'stability_enhancement': _suggest_stability_enhancements(pattern_analysis),
            'pattern_awareness_tips': _generate_pattern_awareness_tips(pattern_analysis)
        }
        
        return pattern_analysis
        
    except Exception as e:
        logger.error(f"Emotional patterns tracking failed: {e}")
        return {
            'available': False,
            'error': str(e)
        }

def calculate_emotional_intelligence(personality_data: Dict,
                                   assessment_scope: str = 'comprehensive') -> Dict[str, Any]:
    """
    Berechnet Emotional Intelligence Score
    
    Extrahiert aus kira_routes.py.backup EQ Calculation Logic
    """
    try:
        current_state = personality_data.get('current_state', {})
        traits = personality_data.get('traits', {})
        interaction_history = personality_data.get('interaction_history', [])
        emotional_history = personality_data.get('emotional_history', [])
        
        # Core EQ components
        eq_components = {
            'self_awareness': _calculate_emotional_self_awareness(current_state, emotional_history),
            'self_regulation': _calculate_emotional_self_regulation(current_state, emotional_history),
            'empathy': _calculate_empathy_score(current_state, interaction_history),
            'social_skills': _calculate_emotional_social_skills(interaction_history, traits),
            'motivation': _calculate_emotional_motivation(current_state, traits)
        }
        
        # Advanced EQ analysis
        if assessment_scope == 'comprehensive':
            eq_components.update({
                'emotional_perception': _calculate_emotional_perception(interaction_history),
                'emotional_understanding': _calculate_emotional_understanding(emotional_history, interaction_history),
                'emotional_integration': _calculate_emotional_integration(current_state, traits),
                'emotional_expression': _calculate_emotional_expression(interaction_history),
                'emotional_resilience': _calculate_emotional_resilience_score(emotional_history)
            })
        
        # Overall EQ calculation
        eq_scores = {component: score for component, score in eq_components.items() if isinstance(score, (int, float))}
        overall_eq = statistics.mean(eq_scores.values()) if eq_scores else 0.5
        
        # EQ analysis
        eq_analysis = {
            'overall_eq_score': overall_eq,
            'eq_components': eq_components,
            'eq_strengths': _identify_eq_strengths(eq_components),
            'eq_development_areas': _identify_eq_development_areas(eq_components),
            'eq_level_classification': _classify_eq_level(overall_eq),
            'eq_growth_potential': _assess_eq_growth_potential(eq_components, personality_data)
        }
        
        # EQ recommendations
        eq_analysis['eq_recommendations'] = {
            'skill_development_priorities': _prioritize_eq_skill_development(eq_components),
            'practice_suggestions': _suggest_eq_practice_activities(eq_components),
            'growth_strategies': _suggest_eq_growth_strategies(eq_analysis),
            'milestone_targets': _define_eq_milestone_targets(eq_analysis)
        }
        
        return eq_analysis
        
    except Exception as e:
        logger.error(f"Emotional intelligence calculation failed: {e}")
        return {
            'error': str(e),
            'fallback_eq_analysis': _generate_fallback_eq_analysis()
        }

def generate_emotional_insights(personality_data: Dict,
                              insight_categories: List[str] = None) -> Dict[str, Any]:
    """
    Generiert Emotional Insights
    
    Basiert auf kira_routes.py.backup Emotional Insights Logic
    """
    try:
        if insight_categories is None:
            insight_categories = ['patterns', 'growth', 'optimization', 'predictions']
        
        current_state = personality_data.get('current_state', {})
        emotional_history = personality_data.get('emotional_history', [])
        interaction_history = personality_data.get('interaction_history', [])
        
        insights = {}
        
        # Pattern insights
        if 'patterns' in insight_categories:
            insights['pattern_insights'] = _generate_emotional_pattern_insights(
                emotional_history, interaction_history
            )
        
        # Growth insights
        if 'growth' in insight_categories:
            insights['growth_insights'] = _generate_emotional_growth_insights(
                current_state, emotional_history
            )
        
        # Optimization insights
        if 'optimization' in insight_categories:
            insights['optimization_insights'] = _generate_emotional_optimization_insights(
                current_state, emotional_history, interaction_history
            )
        
        # Predictive insights
        if 'predictions' in insight_categories:
            insights['predictive_insights'] = _generate_emotional_predictive_insights(
                personality_data
            )
        
        # Consolidate insights
        insights['consolidated_insights'] = {
            'key_findings': _extract_key_emotional_findings(insights),
            'actionable_recommendations': _extract_actionable_emotional_recommendations(insights),
            'priority_focus_areas': _identify_emotional_priority_areas(insights),
            'success_indicators': _define_emotional_success_indicators(insights)
        }
        
        # Insights metadata
        insights['insights_metadata'] = {
            'generation_timestamp': datetime.now().isoformat(),
            'insight_categories': insight_categories,
            'data_richness_score': _assess_emotional_data_richness(personality_data),
            'insight_confidence': _calculate_emotional_insights_confidence(personality_data)
        }
        
        return insights
        
    except Exception as e:
        logger.error(f"Emotional insights generation failed: {e}")
        return {
            'error': str(e),
            'fallback_insights': _generate_fallback_emotional_insights()
        }

# ====================================
# PRIVATE HELPER FUNCTIONS
# ====================================

def _extract_current_emotional_metrics(current_state: Dict) -> Dict[str, Any]:
    """Extrahiert aktuelle Emotional Metrics"""
    try:
        return {
            'emotional_stability': current_state.get('emotional_stability', 0.7),
            'adaptability': current_state.get('adaptability', 0.6),
            'empathy_level': current_state.get('empathy_level', 0.8),
            'social_confidence': current_state.get('social_confidence', 0.6),
            'emotional_expressiveness': current_state.get('emotional_expressiveness', 0.7),
            'emotional_complexity': current_state.get('emotional_complexity', 0.5),
            'emotional_intensity': current_state.get('emotional_intensity', 0.6),
            'mood_stability': current_state.get('mood_stability', 0.7)
        }
    except Exception as e:
        logger.debug(f"Current emotional metrics extraction failed: {e}")
        return {
            'emotional_stability': 0.7,
            'adaptability': 0.6,
            'empathy_level': 0.8,
            'fallback_data': True
        }

def _analyze_emotional_stability(current_state: Dict, emotional_history: List) -> Dict[str, Any]:
    """Analysiert Emotional Stability"""
    try:
        current_stability = current_state.get('emotional_stability', 0.7)
        
        if not emotional_history:
            return {
                'current_stability': current_stability,
                'stability_trend': 'unknown',
                'historical_data_available': False
            }
        
        # Extract stability values from history
        historical_stability = []
        for event in emotional_history[-10:]:  # Last 10 events
            if isinstance(event, dict) and 'emotional_stability' in event:
                historical_stability.append(event['emotional_stability'])
        
        if len(historical_stability) < 2:
            return {
                'current_stability': current_stability,
                'stability_trend': 'insufficient_data',
                'historical_data_available': True,
                'data_points': len(historical_stability)
            }
        
        # Calculate stability metrics
        stability_variance = statistics.variance(historical_stability) if len(historical_stability) > 1 else 0
        stability_trend = _calculate_trend(historical_stability)
        
        return {
            'current_stability': current_stability,
            'historical_average': statistics.mean(historical_stability),
            'stability_variance': stability_variance,
            'stability_consistency': 1.0 - min(1.0, stability_variance),
            'stability_trend': 'improving' if stability_trend > 0.05 else 'declining' if stability_trend < -0.05 else 'stable',
            'stability_trajectory': _determine_stability_trajectory(historical_stability),
            'historical_data_available': True,
            'data_points': len(historical_stability)
        }
        
    except Exception as e:
        logger.debug(f"Emotional stability analysis failed: {e}")
        return {
            'current_stability': current_state.get('emotional_stability', 0.7),
            'stability_trend': 'unknown',
            'error': str(e)
        }

def _calculate_emotional_self_awareness(current_state: Dict, emotional_history: List) -> float:
    """Berechnet Emotional Self-Awareness Score"""
    try:
        # Base self-awareness from current emotional metrics
        base_awareness = current_state.get('emotional_stability', 0.7) * 0.3
        
        # Enhanced awareness through emotional history tracking
        if emotional_history:
            history_awareness = min(0.4, len(emotional_history) / 50)  # Max 0.4 for history
            emotional_range_awareness = _calculate_emotional_range_awareness(emotional_history) * 0.3
        else:
            history_awareness = 0
            emotional_range_awareness = 0
        
        total_awareness = base_awareness + history_awareness + emotional_range_awareness
        return min(1.0, max(0.0, total_awareness))
        
    except Exception as e:
        logger.debug(f"Emotional self-awareness calculation failed: {e}")
        return 0.6

def _calculate_empathy_score(current_state: Dict, interaction_history: List) -> float:
    """Berechnet Empathy Score"""
    try:
        # Base empathy from current state
        base_empathy = current_state.get('empathy_level', 0.8)
        
        # Enhanced empathy through interaction patterns
        if interaction_history:
            empathetic_responses = 0
            total_interactions = len(interaction_history[-20:])  # Last 20 interactions
            
            for interaction in interaction_history[-20:]:
                if isinstance(interaction, dict):
                    # Look for empathetic indicators
                    response_type = interaction.get('response_type', '')
                    emotional_context = interaction.get('emotional_context', {})
                    
                    if 'empathetic' in response_type.lower() or emotional_context.get('empathy_shown', False):
                        empathetic_responses += 1
            
            interaction_empathy_bonus = (empathetic_responses / max(1, total_interactions)) * 0.2
            return min(1.0, base_empathy + interaction_empathy_bonus)
        
        return base_empathy
        
    except Exception as e:
        logger.debug(f"Empathy score calculation failed: {e}")
        return 0.8

def _track_stability_patterns(emotional_history: List) -> Dict[str, Any]:
    """Verfolgt Stability Patterns"""
    try:
        if not emotional_history:
            return {'available': False, 'reason': 'no_emotional_history'}
        
        stability_values = []
        timestamps = []
        
        for event in emotional_history:
            if isinstance(event, dict) and 'emotional_stability' in event:
                stability_values.append(event['emotional_stability'])
                timestamps.append(_parse_timestamp(event.get('timestamp')))
        
        if len(stability_values) < 3:
            return {'available': False, 'reason': 'insufficient_stability_data'}
        
        # Pattern analysis
        stability_patterns = {
            'average_stability': statistics.mean(stability_values),
            'stability_range': max(stability_values) - min(stability_values),
            'stability_volatility': statistics.stdev(stability_values) if len(stability_values) > 1 else 0,
            'stability_trend': _calculate_trend(stability_values),
            'stable_periods': _identify_stable_periods(stability_values, timestamps),
            'unstable_periods': _identify_unstable_periods(stability_values, timestamps),
            'recovery_patterns': _analyze_stability_recovery_patterns(stability_values, timestamps)
        }
        
        return stability_patterns
        
    except Exception as e:
        logger.debug(f"Stability patterns tracking failed: {e}")
        return {'available': False, 'error': str(e)}

def _parse_timeframe_to_days(timeframe: str) -> int:
    """Parst Timeframe zu Tagen"""
    timeframe_map = {
        '7d': 7, '1w': 7, '14d': 14, '2w': 14,
        '30d': 30, '1m': 30, '60d': 60, '2m': 60,
        '90d': 90, '3m': 90
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

def _calculate_trend(values: List[float]) -> float:
    """Berechnet Trend aus Werte-Liste"""
    try:
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
        
    except Exception as e:
        logger.debug(f"Trend calculation failed: {e}")
        return 0.0

def _generate_fallback_emotional_analysis() -> Dict[str, Any]:
    """Generiert Fallback Emotional Analysis"""
    return {
        'fallback_mode': True,
        'basic_emotional_state': {
            'emotional_stability': 0.7,
            'empathy_level': 0.8,
            'adaptability': 0.6,
            'status': 'baseline_emotional_state'
        },
        'recommendations': {
            'primary_recommendation': 'Enable emotional state tracking for detailed analysis',
            'data_collection_suggestions': [
                'Track emotional responses',
                'Monitor interaction patterns',
                'Enable emotional history logging'
            ]
        },
        'timestamp': datetime.now().isoformat()
    }

__all__ = [
    'analyze_emotional_state',
    'track_emotional_patterns',
    'calculate_emotional_intelligence',
    'generate_emotional_insights'
]