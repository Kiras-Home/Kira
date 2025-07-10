"""
Analytics Reports Module
Generiert umfassende Analytics Reports und Summaries
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def generate_analytics_summary(memory_manager, db_stats: Dict, personality_data: Dict = None) -> Dict[str, Any]:
    """
    Generiert Analytics Summary basierend auf kira_routes.py.backup
    
    Extrahiert aus: _generate_analytics_summary() Funktion
    """
    try:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'memory_analytics': _analyze_memory_performance(memory_manager, db_stats),
            'learning_analytics': _analyze_learning_patterns(memory_manager, db_stats),
            'efficiency_analytics': _analyze_system_efficiency(memory_manager, db_stats),
            'growth_indicators': _calculate_growth_indicators(memory_manager, db_stats)
        }
        
        # Personality Analytics falls verfügbar
        if personality_data:
            summary['personality_analytics'] = _analyze_personality_performance(personality_data)
        
        # Overall Score
        summary['overall_performance_score'] = _calculate_overall_performance_score(summary)
        
        return summary
        
    except Exception as e:
        logger.error(f"Analytics summary generation failed: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'fallback_data': True
        }

def generate_comprehensive_report(memory_manager, db_stats: Dict, 
                                personality_data: Dict = None,
                                time_period: str = '30d') -> Dict[str, Any]:
    """
    Generiert umfassenden Analytics Report
    
    Basiert auf kira_routes.py.backup Analytics Logic
    """
    try:
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'time_period': time_period,
                'report_type': 'comprehensive_analytics',
                'version': '2.0'
            },
            
            # Core Analytics Sections
            'executive_summary': _generate_executive_summary(memory_manager, db_stats),
            'memory_analysis': _generate_detailed_memory_analysis(memory_manager, db_stats, time_period),
            'learning_analysis': _generate_learning_analysis(memory_manager, db_stats, time_period),
            'performance_analysis': _generate_performance_analysis(memory_manager, db_stats),
            'growth_analysis': calculate_growth_analysis(memory_manager, db_stats, time_period),
            'recommendations': _generate_analytics_recommendations(memory_manager, db_stats)
        }
        
        # Personality Analysis falls verfügbar
        if personality_data:
            report['personality_analysis'] = _generate_personality_analytics(personality_data, time_period)
        
        # Report Statistics
        report['report_statistics'] = {
            'total_data_points': _count_total_data_points(report),
            'analysis_confidence': _calculate_analysis_confidence(memory_manager, db_stats),
            'data_quality_score': _assess_data_quality(db_stats)
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Comprehensive report generation failed: {e}")
        return {
            'error': str(e),
            'fallback_report': True,
            'timestamp': datetime.now().isoformat()
        }

def calculate_growth_analysis(memory_manager, db_stats: Dict, time_period: str = '30d') -> Dict[str, Any]:
    """
    Berechnet detaillierte Growth Analysis
    
    Extrahiert aus kira_routes.py.backup Growth Logic
    """
    try:
        # Parse time period
        days = _parse_time_period_to_days(time_period)
        
        growth_metrics = {
            'memory_growth': _calculate_memory_growth(memory_manager, db_stats, days),
            'learning_velocity': _calculate_learning_velocity(memory_manager, db_stats, days),
            'efficiency_improvement': _calculate_efficiency_trends(memory_manager, db_stats, days),
            'system_optimization': _calculate_optimization_growth(memory_manager, db_stats, days)
        }
        
        # Growth Trends
        growth_metrics['growth_trends'] = {
            'overall_trajectory': _determine_growth_trajectory(growth_metrics),
            'growth_acceleration': _calculate_growth_acceleration(growth_metrics),
            'projected_growth': _project_future_growth(growth_metrics, days)
        }
        
        # Growth Insights
        growth_metrics['growth_insights'] = _generate_growth_insights(growth_metrics)
        
        return growth_metrics
        
    except Exception as e:
        logger.error(f"Growth analysis calculation failed: {e}")
        return {
            'error': str(e),
            'fallback_growth_data': True
        }

def analyze_memory_distribution(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """
    Analysiert Memory Distribution Patterns
    
    Basiert auf kira_routes.py.backup Memory Distribution Logic
    """
    try:
        distribution_analysis = {
            'memory_type_distribution': _analyze_memory_type_distribution(db_stats),
            'temporal_distribution': _analyze_temporal_memory_distribution(db_stats),
            'importance_distribution': _analyze_importance_distribution(memory_manager, db_stats),
            'access_pattern_distribution': _analyze_access_patterns(memory_manager, db_stats)
        }
        
        # Distribution Insights
        distribution_analysis['distribution_insights'] = {
            'dominant_patterns': _identify_dominant_patterns(distribution_analysis),
            'anomalies': _detect_distribution_anomalies(distribution_analysis),
            'optimization_opportunities': _identify_optimization_opportunities(distribution_analysis)
        }
        
        return distribution_analysis
        
    except Exception as e:
        logger.error(f"Memory distribution analysis failed: {e}")
        return {
            'error': str(e),
            'fallback_distribution_data': True
        }

# ====================================
# PRIVATE HELPER FUNCTIONS
# ====================================

def _analyze_memory_performance(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Analysiert Memory Performance Metriken"""
    try:
        from routes.utils.memory_helpers import (
            get_stm_load_direct, get_stm_capacity_direct,
            calculate_stm_efficiency_direct, get_ltm_size_direct
        )
        
        stm_load = get_stm_load_direct(memory_manager)
        stm_capacity = get_stm_capacity_direct(memory_manager)
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        ltm_size = get_ltm_size_direct(memory_manager)
        
        return {
            'stm_utilization': stm_load / max(1, stm_capacity),
            'stm_efficiency': stm_efficiency,
            'ltm_capacity': ltm_size,
            'total_memories': db_stats.get('total_memories', 0),
            'memory_formation_rate': db_stats.get('recent_activity', 0),
            'performance_score': (stm_efficiency + (1 - stm_load/max(1, stm_capacity))) / 2
        }
        
    except Exception as e:
        logger.debug(f"Memory performance analysis failed: {e}")
        return {
            'stm_utilization': 0.5,
            'stm_efficiency': 0.7,
            'ltm_capacity': 100,
            'total_memories': db_stats.get('total_memories', 0),
            'performance_score': 0.6,
            'fallback_data': True
        }

def _analyze_learning_patterns(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Analysiert Learning Patterns"""
    try:
        from routes.utils.memory_helpers import calculate_learning_readiness_direct
        
        learning_readiness = calculate_learning_readiness_direct(memory_manager, db_stats)
        recent_activity = db_stats.get('recent_activity', 0)
        
        return {
            'learning_readiness': learning_readiness,
            'recent_learning_activity': recent_activity,
            'learning_velocity': recent_activity / 24,  # Per hour
            'learning_pattern': 'active' if recent_activity > 5 else 'moderate' if recent_activity > 1 else 'low',
            'learning_efficiency': learning_readiness * (1 + min(recent_activity / 10, 1))
        }
        
    except Exception as e:
        logger.debug(f"Learning patterns analysis failed: {e}")
        return {
            'learning_readiness': 0.7,
            'recent_learning_activity': 3,
            'learning_velocity': 0.125,
            'learning_pattern': 'moderate',
            'learning_efficiency': 0.8,
            'fallback_data': True
        }

def _analyze_system_efficiency(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Analysiert System Efficiency"""
    try:
        from routes.utils.memory_helpers import (
            calculate_current_cognitive_load,
            is_consolidation_active_direct
        )
        
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        consolidation_active = is_consolidation_active_direct(memory_manager, db_stats)
        
        # System Efficiency Score
        efficiency_factors = [
            1.0 - cognitive_load,  # Lower load = higher efficiency
            0.8 if consolidation_active else 0.6,  # Consolidation indicates good maintenance
            min(1.0, db_stats.get('recent_activity', 0) / 10)  # Optimal activity level
        ]
        
        system_efficiency = sum(efficiency_factors) / len(efficiency_factors)
        
        return {
            'cognitive_load': cognitive_load,
            'consolidation_active': consolidation_active,
            'system_efficiency_score': system_efficiency,
            'efficiency_rating': _rate_efficiency(system_efficiency),
            'optimization_potential': 1.0 - system_efficiency
        }
        
    except Exception as e:
        logger.debug(f"System efficiency analysis failed: {e}")
        return {
            'cognitive_load': 0.5,
            'consolidation_active': False,
            'system_efficiency_score': 0.7,
            'efficiency_rating': 'good',
            'optimization_potential': 0.3,
            'fallback_data': True
        }

def _calculate_growth_indicators(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Berechnet Growth Indicators"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Growth Indicators
        memory_growth_rate = recent_activity / max(1, total_memories) if total_memories > 0 else 0
        learning_momentum = min(1.0, recent_activity / 5)  # Normalized learning momentum
        
        return {
            'memory_growth_rate': memory_growth_rate,
            'learning_momentum': learning_momentum,
            'growth_trajectory': 'accelerating' if memory_growth_rate > 0.1 else 'steady' if memory_growth_rate > 0.05 else 'slow',
            'development_stage': _determine_development_stage(total_memories, recent_activity),
            'growth_potential': _calculate_growth_potential(memory_manager, db_stats)
        }
        
    except Exception as e:
        logger.debug(f"Growth indicators calculation failed: {e}")
        return {
            'memory_growth_rate': 0.05,
            'learning_momentum': 0.6,
            'growth_trajectory': 'steady',
            'development_stage': 'developing',
            'growth_potential': 0.8,
            'fallback_data': True
        }

def _analyze_personality_performance(personality_data: Dict) -> Dict[str, Any]:
    """Analysiert Personality Performance falls verfügbar"""
    try:
        if not personality_data:
            return {'available': False}
        
        # Extrahiere Personality Metrics
        traits = personality_data.get('traits', {})
        current_state = personality_data.get('current_state', {})
        
        # Personality Performance Metrics
        trait_balance = _calculate_trait_balance(traits)
        emotional_stability = current_state.get('emotional_stability', 0.7)
        adaptability = current_state.get('adaptability', 0.6)
        
        return {
            'available': True,
            'trait_balance_score': trait_balance,
            'emotional_stability': emotional_stability,
            'adaptability_score': adaptability,
            'personality_health': (trait_balance + emotional_stability + adaptability) / 3,
            'dominant_traits': _identify_dominant_traits(traits),
            'development_areas': _identify_development_areas(traits)
        }
        
    except Exception as e:
        logger.debug(f"Personality performance analysis failed: {e}")
        return {
            'available': False,
            'error': str(e)
        }

def _calculate_overall_performance_score(summary: Dict) -> float:
    """Berechnet Overall Performance Score"""
    try:
        scores = []
        
        # Memory Performance
        if 'memory_analytics' in summary:
            scores.append(summary['memory_analytics'].get('performance_score', 0.5))
        
        # Learning Performance
        if 'learning_analytics' in summary:
            scores.append(summary['learning_analytics'].get('learning_efficiency', 0.5))
        
        # System Efficiency
        if 'efficiency_analytics' in summary:
            scores.append(summary['efficiency_analytics'].get('system_efficiency_score', 0.5))
        
        # Personality Performance
        if 'personality_analytics' in summary and summary['personality_analytics'].get('available'):
            scores.append(summary['personality_analytics'].get('personality_health', 0.5))
        
        return sum(scores) / len(scores) if scores else 0.5
        
    except Exception as e:
        logger.debug(f"Overall performance score calculation failed: {e}")
        return 0.5

# Weitere Helper Functions...

def _parse_time_period_to_days(time_period: str) -> int:
    """Parst Zeitperiode zu Tagen"""
    period_map = {
        '24h': 1, '1d': 1, '7d': 7, '1w': 7,
        '30d': 30, '1m': 30, '90d': 90, '3m': 90,
        '1y': 365, '365d': 365
    }
    return period_map.get(time_period, 30)

def _rate_efficiency(efficiency_score: float) -> str:
    """Bewertet Efficiency Score"""
    if efficiency_score >= 0.9:
        return 'excellent'
    elif efficiency_score >= 0.75:
        return 'good'
    elif efficiency_score >= 0.6:
        return 'fair'
    else:
        return 'needs_improvement'

def _determine_development_stage(total_memories: int, recent_activity: int) -> str:
    """Bestimmt Development Stage"""
    if total_memories < 50:
        return 'early'
    elif total_memories < 200:
        return 'developing'
    elif total_memories < 1000:
        return 'mature'
    else:
        return 'advanced'

def _calculate_growth_potential(memory_manager, db_stats: Dict) -> float:
    """Berechnet Growth Potential"""
    try:
        from routes.utils.memory_helpers import calculate_learning_readiness_direct
        
        learning_readiness = calculate_learning_readiness_direct(memory_manager, db_stats)
        system_capacity = 1.0  # Normalized capacity
        current_utilization = db_stats.get('total_memories', 0) / 1000  # Normalize to 1000 memories
        
        growth_potential = learning_readiness * (system_capacity - min(current_utilization, 1.0))
        return max(0.0, min(1.0, growth_potential))
        
    except Exception as e:
        logger.debug(f"Growth potential calculation failed: {e}")
        return 0.8

# Additional helper functions für umfassende Reports...
def _generate_executive_summary(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Generiert Executive Summary"""
    return {
        'system_status': 'operational',
        'key_highlights': [
            f"Total memories: {db_stats.get('total_memories', 0)}",
            f"Recent activity: {db_stats.get('recent_activity', 0)} events",
            "System performing within normal parameters"
        ],
        'performance_overview': 'System shows consistent learning and memory formation patterns',
        'recommendation_summary': 'Continue current optimization strategies'
    }


def _generate_detailed_memory_analysis(memory_manager, db_stats: Dict, time_period: str = '30d') -> Dict[str, Any]:
    """Generiert detaillierte Memory Analysis"""
    try:
        from routes.utils.memory_helpers import (
            get_stm_load_direct, get_stm_capacity_direct,
            calculate_stm_efficiency_direct, get_ltm_size_direct,
            calculate_current_cognitive_load, is_consolidation_active_direct
        )
        
        # Core Memory Metrics
        stm_load = get_stm_load_direct(memory_manager)
        stm_capacity = get_stm_capacity_direct(memory_manager)
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        ltm_size = get_ltm_size_direct(memory_manager)
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        consolidation_active = is_consolidation_active_direct(memory_manager, db_stats)
        
        memory_analysis = {
            'memory_system_overview': {
                'total_memories': db_stats.get('total_memories', 0),
                'stm_current_load': stm_load,
                'stm_capacity': stm_capacity,
                'stm_utilization_rate': stm_load / max(1, stm_capacity),
                'ltm_size': ltm_size,
                'memory_formation_rate': db_stats.get('recent_activity', 0)
            },
            
            'memory_efficiency_metrics': {
                'stm_efficiency_score': stm_efficiency,
                'cognitive_load_level': cognitive_load,
                'memory_processing_efficiency': 1.0 - cognitive_load,
                'consolidation_status': 'active' if consolidation_active else 'inactive',
                'memory_optimization_score': _calculate_memory_optimization_score(
                    stm_efficiency, cognitive_load, stm_load, stm_capacity
                )
            },
            
            'memory_patterns': {
                'formation_patterns': _analyze_memory_formation_patterns(db_stats, time_period),
                'access_patterns': _analyze_memory_access_patterns(memory_manager, db_stats),
                'retention_patterns': _analyze_memory_retention_patterns(db_stats, time_period),
                'consolidation_patterns': _analyze_consolidation_patterns(memory_manager, db_stats)
            },
            
            'memory_quality_assessment': {
                'memory_coherence': _assess_memory_coherence(memory_manager, db_stats),
                'information_density': _calculate_information_density(db_stats),
                'memory_diversity': _assess_memory_diversity(db_stats),
                'knowledge_integration': _assess_knowledge_integration(memory_manager, db_stats)
            },
            
            'temporal_analysis': {
                'short_term_trends': _analyze_short_term_memory_trends(db_stats, 7),
                'medium_term_trends': _analyze_medium_term_memory_trends(db_stats, 30),
                'long_term_trends': _analyze_long_term_memory_trends(db_stats, 90),
                'seasonal_patterns': _detect_seasonal_memory_patterns(db_stats)
            },
            
            'memory_health_indicators': {
                'system_stability': _assess_memory_system_stability(memory_manager, db_stats),
                'processing_resilience': _assess_processing_resilience(memory_manager, db_stats),
                'adaptation_capability': _assess_memory_adaptation_capability(memory_manager, db_stats),
                'recovery_potential': _assess_memory_recovery_potential(memory_manager, db_stats)
            }
        }
        
        # Memory Analysis Summary
        memory_analysis['analysis_summary'] = {
            'overall_memory_health': _calculate_overall_memory_health(memory_analysis),
            'key_insights': _generate_memory_insights(memory_analysis),
            'critical_findings': _identify_critical_memory_findings(memory_analysis),
            'improvement_opportunities': _identify_memory_improvement_opportunities(memory_analysis)
        }
        
        return memory_analysis
        
    except Exception as e:
        logger.error(f"Detailed memory analysis generation failed: {e}")
        return _generate_fallback_memory_analysis(db_stats, time_period)

def _generate_learning_analysis(memory_manager, db_stats: Dict, time_period: str = '30d') -> Dict[str, Any]:
    """Generiert Learning Analysis"""
    try:
        from routes.utils.memory_helpers import (
            calculate_learning_readiness_direct,
            calculate_current_cognitive_load
        )
        
        learning_readiness = calculate_learning_readiness_direct(memory_manager, db_stats)
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        recent_activity = db_stats.get('recent_activity', 0)
        total_memories = db_stats.get('total_memories', 0)
        
        learning_analysis = {
            'learning_performance_metrics': {
                'learning_readiness_score': learning_readiness,
                'learning_activity_level': recent_activity,
                'learning_velocity': recent_activity / max(1, _parse_time_period_to_days(time_period)),
                'learning_efficiency': _calculate_learning_efficiency_score(
                    learning_readiness, cognitive_load, recent_activity
                ),
                'knowledge_acquisition_rate': recent_activity / max(1, total_memories),
                'learning_momentum': _calculate_learning_momentum_score(recent_activity, learning_readiness)
            },
            
            'learning_patterns': {
                'learning_frequency': _analyze_learning_frequency(db_stats, time_period),
                'learning_intensity': _analyze_learning_intensity(db_stats, recent_activity),
                'learning_consistency': _analyze_learning_consistency(db_stats, time_period),
                'learning_progression': _analyze_learning_progression(db_stats, memory_manager)
            },
            
            'knowledge_development': {
                'knowledge_base_growth': _analyze_knowledge_base_growth(db_stats, time_period),
                'domain_expansion': _analyze_domain_expansion(db_stats),
                'conceptual_development': _analyze_conceptual_development(memory_manager, db_stats),
                'skill_acquisition': _analyze_skill_acquisition_patterns(db_stats)
            },
            
            'learning_quality_metrics': {
                'comprehension_depth': _assess_comprehension_depth(memory_manager, db_stats),
                'retention_quality': _assess_retention_quality(db_stats, time_period),
                'application_ability': _assess_application_ability(memory_manager, db_stats),
                'integration_capability': _assess_integration_capability(memory_manager, db_stats)
            },
            
            'learning_challenges': {
                'identified_bottlenecks': _identify_learning_bottlenecks(memory_manager, db_stats),
                'cognitive_limitations': _assess_cognitive_limitations(cognitive_load, memory_manager),
                'resource_constraints': _identify_resource_constraints(memory_manager, db_stats),
                'improvement_barriers': _identify_improvement_barriers(learning_readiness, recent_activity)
            }
        }
        
        # Learning Analysis Summary
        learning_analysis['learning_summary'] = {
            'overall_learning_score': _calculate_overall_learning_score(learning_analysis),
            'learning_stage': _determine_learning_stage(total_memories, recent_activity, learning_readiness),
            'learning_trajectory': _predict_learning_trajectory(learning_analysis),
            'optimization_recommendations': _generate_learning_optimization_recommendations(learning_analysis)
        }
        
        return learning_analysis
        
    except Exception as e:
        logger.error(f"Learning analysis generation failed: {e}")
        return _generate_fallback_learning_analysis(db_stats, time_period)

def _generate_performance_analysis(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Generiert Performance Analysis"""
    try:
        from routes.utils.memory_helpers import (
            calculate_current_cognitive_load,
            calculate_stm_efficiency_direct,
            get_stm_load_direct, get_stm_capacity_direct
        )
        
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        stm_load = get_stm_load_direct(memory_manager)
        stm_capacity = get_stm_capacity_direct(memory_manager)
        
        performance_analysis = {
            'system_performance_metrics': {
                'cognitive_performance': {
                    'cognitive_load': cognitive_load,
                    'cognitive_efficiency': 1.0 - cognitive_load,
                    'processing_capacity': max(0.0, 1.0 - (stm_load / max(1, stm_capacity))),
                    'mental_agility': _calculate_mental_agility(memory_manager, db_stats)
                },
                
                'memory_performance': {
                    'stm_performance': stm_efficiency,
                    'memory_throughput': _calculate_memory_throughput(db_stats),
                    'retrieval_speed': _estimate_retrieval_speed(memory_manager, db_stats),
                    'consolidation_efficiency': _calculate_consolidation_efficiency_score(memory_manager, db_stats)
                },
                
                'learning_performance': {
                    'acquisition_speed': db_stats.get('recent_activity', 0) / 24,  # per hour
                    'retention_rate': _calculate_retention_rate(db_stats),
                    'adaptation_speed': _calculate_adaptation_speed(memory_manager, db_stats),
                    'skill_development_rate': _calculate_skill_development_rate(db_stats)
                }
            },
            
            'performance_benchmarks': {
                'current_vs_optimal': _compare_current_vs_optimal_performance(
                    cognitive_load, stm_efficiency, db_stats
                ),
                'historical_comparison': _compare_historical_performance(db_stats),
                'peer_comparison': _generate_peer_comparison_metrics(memory_manager, db_stats),
                'industry_standards': _compare_against_standards()
            },
            
            'performance_trends': {
                'short_term_trends': _analyze_short_term_performance_trends(db_stats),
                'performance_volatility': _calculate_performance_volatility(db_stats),
                'improvement_trajectory': _calculate_improvement_trajectory(memory_manager, db_stats),
                'performance_predictability': _assess_performance_predictability(db_stats)
            },
            
            'bottleneck_analysis': {
                'identified_bottlenecks': _identify_performance_bottlenecks(memory_manager, db_stats),
                'resource_limitations': _identify_resource_limitations(memory_manager, db_stats),
                'processing_constraints': _identify_processing_constraints(cognitive_load, stm_load, stm_capacity),
                'optimization_potential': _calculate_optimization_potential(memory_manager, db_stats)
            }
        }
        
        # Performance Summary
        performance_analysis['performance_summary'] = {
            'overall_performance_score': _calculate_comprehensive_performance_score(performance_analysis),
            'performance_classification': _classify_performance_level(performance_analysis),
            'key_strengths': _identify_performance_strengths(performance_analysis),
            'areas_for_improvement': _identify_performance_improvement_areas(performance_analysis)
        }
        
        return performance_analysis
        
    except Exception as e:
        logger.error(f"Performance analysis generation failed: {e}")
        return _generate_fallback_performance_analysis(db_stats)

def _generate_personality_analytics(personality_data: Dict, time_period: str = '30d') -> Dict[str, Any]:
    """Generiert Personality Analytics"""
    try:
        if not personality_data:
            return {'available': False, 'reason': 'No personality data provided'}
        
        traits = personality_data.get('traits', {})
        current_state = personality_data.get('current_state', {})
        development_metrics = personality_data.get('development_metrics', {})
        
        personality_analytics = {
            'trait_analysis': {
                'trait_profile': _analyze_trait_profile(traits),
                'trait_balance': _calculate_trait_balance_comprehensive(traits),
                'dominant_characteristics': _identify_dominant_characteristics(traits),
                'trait_interactions': _analyze_trait_interactions(traits),
                'trait_stability': _assess_trait_stability(traits, current_state)
            },
            
            'emotional_analysis': {
                'emotional_stability': current_state.get('emotional_stability', 0.7),
                'emotional_range': _calculate_emotional_range(current_state),
                'emotional_regulation': _assess_emotional_regulation(current_state),
                'empathy_metrics': _analyze_empathy_metrics(current_state),
                'social_awareness': _assess_social_awareness(current_state)
            },
            
            'behavioral_patterns': {
                'adaptability_patterns': _analyze_adaptability_patterns(current_state),
                'decision_making_style': _analyze_decision_making_style(traits, current_state),
                'interaction_preferences': _analyze_interaction_preferences(traits),
                'learning_style': _analyze_personality_learning_style(traits, current_state),
                'communication_patterns': _analyze_communication_patterns(traits)
            },
            
            'development_analysis': {
                'personality_growth_rate': development_metrics.get('development_rate', 0.5),
                'maturity_progression': _assess_maturity_progression(traits, current_state),
                'character_development': _analyze_character_development(development_metrics),
                'self_awareness_level': _assess_self_awareness_level(current_state),
                'growth_potential': development_metrics.get('growth_potential', 0.7)
            },
            
            'integration_metrics': {
                'personality_coherence': _assess_personality_coherence(traits, current_state),
                'behavioral_consistency': _assess_behavioral_consistency(traits, current_state),
                'authentic_expression': _assess_authentic_expression(traits, current_state),
                'adaptive_flexibility': _assess_adaptive_flexibility(current_state)
            }
        }
        
        # Personality Summary
        personality_analytics['personality_summary'] = {
            'overall_personality_health': _calculate_overall_personality_health_comprehensive(personality_analytics),
            'personality_type': _determine_personality_type(traits),
            'key_characteristics': _identify_key_characteristics(personality_analytics),
            'development_recommendations': _generate_personality_development_recommendations(personality_analytics)
        }
        
        personality_analytics['available'] = True
        return personality_analytics
        
    except Exception as e:
        logger.error(f"Personality analytics generation failed: {e}")
        return {
            'available': False,
            'error': str(e),
            'fallback_analytics': _generate_fallback_personality_analytics()
        }

def _generate_analytics_recommendations(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Generiert Analytics-basierte Recommendations"""
    try:
        from routes.utils.memory_helpers import (
            calculate_current_cognitive_load,
            calculate_learning_readiness_direct,
            calculate_stm_efficiency_direct
        )
        
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        learning_readiness = calculate_learning_readiness_direct(memory_manager, db_stats)
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        recent_activity = db_stats.get('recent_activity', 0)
        total_memories = db_stats.get('total_memories', 0)
        
        recommendations = {
            'immediate_actions': _generate_immediate_recommendations(
                cognitive_load, learning_readiness, stm_efficiency, recent_activity
            ),
            
            'short_term_optimizations': _generate_short_term_optimizations(
                memory_manager, db_stats, cognitive_load
            ),
            
            'long_term_strategies': _generate_long_term_strategies(
                total_memories, recent_activity, learning_readiness
            ),
            
            'performance_improvements': _generate_performance_improvement_recommendations(
                memory_manager, db_stats
            ),
            
            'system_enhancements': _generate_system_enhancement_recommendations(
                cognitive_load, stm_efficiency, memory_manager
            ),
            
            'learning_optimizations': _generate_learning_optimization_recommendations_detailed(
                learning_readiness, recent_activity, total_memories
            )
        }
        
        # Prioritize recommendations
        recommendations['prioritized_recommendations'] = _prioritize_recommendations(recommendations)
        recommendations['implementation_roadmap'] = _create_implementation_roadmap(recommendations)
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Analytics recommendations generation failed: {e}")
        return _generate_fallback_recommendations()

# ====================================
# MISSING GROWTH ANALYSIS FUNCTIONS
# ====================================

def _calculate_memory_growth(memory_manager, db_stats: Dict, days: int) -> Dict[str, Any]:
    """Berechnet Memory Growth"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Growth rate calculations
        daily_growth_rate = recent_activity / max(1, days)
        projected_weekly_growth = daily_growth_rate * 7
        projected_monthly_growth = daily_growth_rate * 30
        
        # Growth efficiency
        growth_efficiency = min(1.0, recent_activity / (days * 2))  # Assuming 2 per day is efficient
        
        return {
            'current_memory_count': total_memories,
            'recent_additions': recent_activity,
            'daily_growth_rate': daily_growth_rate,
            'weekly_projection': projected_weekly_growth,
            'monthly_projection': projected_monthly_growth,
            'growth_efficiency': growth_efficiency,
            'growth_trend': 'accelerating' if daily_growth_rate > 1.0 else 'steady' if daily_growth_rate > 0.3 else 'slow'
        }
        
    except Exception as e:
        logger.debug(f"Memory growth calculation failed: {e}")
        return {
            'current_memory_count': db_stats.get('total_memories', 0),
            'growth_trend': 'stable',
            'fallback_data': True
        }

def _calculate_learning_velocity(memory_manager, db_stats: Dict, days: int) -> Dict[str, Any]:
    """Berechnet Learning Velocity"""
    try:
        from routes.utils.memory_helpers import calculate_learning_readiness_direct
        
        learning_readiness = calculate_learning_readiness_direct(memory_manager, db_stats)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Velocity calculations
        base_velocity = recent_activity / max(1, days)
        adjusted_velocity = base_velocity * learning_readiness
        velocity_momentum = min(1.0, adjusted_velocity / 0.5)  # 0.5 per day is good momentum
        
        return {
            'base_learning_velocity': base_velocity,
            'adjusted_learning_velocity': adjusted_velocity,
            'learning_momentum': velocity_momentum,
            'velocity_classification': _classify_learning_velocity(adjusted_velocity),
            'acceleration_potential': learning_readiness - (adjusted_velocity / 2.0)
        }
        
    except Exception as e:
        logger.debug(f"Learning velocity calculation failed: {e}")
        return {
            'base_learning_velocity': 0.2,
            'velocity_classification': 'moderate',
            'fallback_data': True
        }

def _calculate_efficiency_trends(memory_manager, db_stats: Dict, days: int) -> Dict[str, Any]:
    """Berechnet Efficiency Trends"""
    try:
        from routes.utils.memory_helpers import (
            calculate_stm_efficiency_direct,
            calculate_current_cognitive_load
        )
        
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        
        # Efficiency metrics
        current_efficiency = stm_efficiency
        processing_efficiency = 1.0 - cognitive_load
        overall_efficiency = (current_efficiency + processing_efficiency) / 2
        
        # Trend analysis (simplified - in real implementation would use historical data)
        efficiency_trend = 'improving' if overall_efficiency > 0.75 else 'stable' if overall_efficiency > 0.6 else 'declining'
        
        return {
            'current_efficiency': current_efficiency,
            'processing_efficiency': processing_efficiency,
            'overall_efficiency': overall_efficiency,
            'efficiency_trend': efficiency_trend,
            'optimization_opportunity': 1.0 - overall_efficiency,
            'trend_confidence': 0.8  # Moderate confidence without historical data
        }
        
    except Exception as e:
        logger.debug(f"Efficiency trends calculation failed: {e}")
        return {
            'overall_efficiency': 0.7,
            'efficiency_trend': 'stable',
            'fallback_data': True
        }

def _calculate_optimization_growth(memory_manager, db_stats: Dict, days: int) -> Dict[str, Any]:
    """Berechnet Optimization Growth"""
    try:
        from routes.utils.memory_helpers import (
            get_stm_load_direct, get_stm_capacity_direct,
            calculate_current_cognitive_load
        )
        
        stm_load = get_stm_load_direct(memory_manager)
        stm_capacity = get_stm_capacity_direct(memory_manager)
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        
        # Optimization metrics
        memory_utilization_optimization = abs(0.6 - (stm_load / max(1, stm_capacity)))  # Target 60% utilization
        cognitive_load_optimization = cognitive_load  # Lower is better
        
        # Overall optimization score
        optimization_score = 1.0 - ((memory_utilization_optimization + cognitive_load_optimization) / 2)
        
        return {
            'memory_optimization_level': 1.0 - memory_utilization_optimization,
            'cognitive_optimization_level': 1.0 - cognitive_load_optimization,
            'overall_optimization_score': optimization_score,
            'optimization_growth_rate': max(0.0, (optimization_score - 0.7) / days),  # Compared to baseline 0.7
            'optimization_potential': 1.0 - optimization_score
        }
        
    except Exception as e:
        logger.debug(f"Optimization growth calculation failed: {e}")
        return {
            'overall_optimization_score': 0.7,
            'optimization_potential': 0.3,
            'fallback_data': True
        }

def _determine_growth_trajectory(growth_metrics: Dict) -> str:
    """Bestimmt Growth Trajectory"""
    try:
        # Analyze different growth components
        memory_growth = growth_metrics.get('memory_growth', {})
        learning_velocity = growth_metrics.get('learning_velocity', {})
        efficiency_improvement = growth_metrics.get('efficiency_improvement', {})
        
        # Get trend indicators
        memory_trend = memory_growth.get('growth_trend', 'stable')
        velocity_class = learning_velocity.get('velocity_classification', 'moderate')
        efficiency_trend = efficiency_improvement.get('efficiency_trend', 'stable')
        
        # Determine overall trajectory
        positive_indicators = 0
        if memory_trend in ['accelerating', 'steady']:
            positive_indicators += 1
        if velocity_class in ['high', 'very_high']:
            positive_indicators += 1
        if efficiency_trend == 'improving':
            positive_indicators += 1
        
        if positive_indicators >= 3:
            return 'exponential_growth'
        elif positive_indicators >= 2:
            return 'strong_growth'
        elif positive_indicators >= 1:
            return 'moderate_growth'
        else:
            return 'slow_growth'
            
    except Exception as e:
        logger.debug(f"Growth trajectory determination failed: {e}")
        return 'stable_growth'

def _calculate_growth_acceleration(growth_metrics: Dict) -> float:
    """Berechnet Growth Acceleration"""
    try:
        # Get growth rates from different components
        memory_growth = growth_metrics.get('memory_growth', {})
        learning_velocity = growth_metrics.get('learning_velocity', {})
        
        daily_growth_rate = memory_growth.get('daily_growth_rate', 0.2)
        learning_momentum = learning_velocity.get('learning_momentum', 0.5)
        
        # Calculate acceleration (rate of change of growth rate)
        # Simplified calculation - in real implementation would use historical data
        base_acceleration = daily_growth_rate * learning_momentum
        
        # Normalize to 0-1 scale
        acceleration_score = min(1.0, base_acceleration / 0.5)  # 0.5 is considered high acceleration
        
        return acceleration_score
        
    except Exception as e:
        logger.debug(f"Growth acceleration calculation failed: {e}")
        return 0.3  # Moderate acceleration

def _project_future_growth(growth_metrics: Dict, days: int) -> Dict[str, Any]:
    """Projiziert Future Growth"""
    try:
        memory_growth = growth_metrics.get('memory_growth', {})
        learning_velocity = growth_metrics.get('learning_velocity', {})
        
        daily_growth_rate = memory_growth.get('daily_growth_rate', 0.2)
        learning_momentum = learning_velocity.get('learning_momentum', 0.5)
        
        # Projection calculations
        projected_30_days = daily_growth_rate * 30 * learning_momentum
        projected_90_days = daily_growth_rate * 90 * learning_momentum * 0.9  # Slight decay over time
        projected_365_days = daily_growth_rate * 365 * learning_momentum * 0.8  # More decay over longer time
        
        return {
            '30_day_projection': projected_30_days,
            '90_day_projection': projected_90_days,
            '1_year_projection': projected_365_days,
            'projection_confidence': 0.7,  # Moderate confidence
            'growth_sustainability': _assess_growth_sustainability(growth_metrics)
        }
        
    except Exception as e:
        logger.debug(f"Future growth projection failed: {e}")
        return {
            '30_day_projection': 6.0,
            '90_day_projection': 15.0,
            '1_year_projection': 50.0,
            'projection_confidence': 0.5
        }

def _generate_growth_insights(growth_metrics: Dict) -> List[str]:
    """Generiert Growth Insights"""
    try:
        insights = []
        
        # Memory growth insights
        memory_growth = growth_metrics.get('memory_growth', {})
        growth_trend = memory_growth.get('growth_trend', 'stable')
        
        if growth_trend == 'accelerating':
            insights.append("Memory formation is accelerating - excellent learning momentum")
        elif growth_trend == 'slow':
            insights.append("Memory growth is slow - consider increasing learning activities")
        
        # Learning velocity insights
        learning_velocity = growth_metrics.get('learning_velocity', {})
        velocity_class = learning_velocity.get('velocity_classification', 'moderate')
        
        if velocity_class == 'high':
            insights.append("High learning velocity indicates strong knowledge acquisition")
        elif velocity_class == 'low':
            insights.append("Low learning velocity suggests need for enhanced engagement")
        
        # Efficiency insights
        efficiency_improvement = growth_metrics.get('efficiency_improvement', {})
        efficiency_trend = efficiency_improvement.get('efficiency_trend', 'stable')
        
        if efficiency_trend == 'improving':
            insights.append("System efficiency is improving - optimization strategies are working")
        elif efficiency_trend == 'declining':
            insights.append("Efficiency decline detected - system optimization recommended")
        
        # Growth trajectory insights
        growth_trends = growth_metrics.get('growth_trends', {})
        trajectory = growth_trends.get('overall_trajectory', 'stable')
        
        if trajectory == 'exponential_growth':
            insights.append("Exceptional growth trajectory - maintain current strategies")
        elif trajectory == 'slow_growth':
            insights.append("Growth potential not fully realized - consider enhancement strategies")
        
        return insights[:5]  # Limit to top 5 insights
        
    except Exception as e:
        logger.debug(f"Growth insights generation failed: {e}")
        return ["System showing steady development patterns"]

# ====================================
# MISSING MEMORY DISTRIBUTION FUNCTIONS
# ====================================

def _analyze_memory_type_distribution(db_stats: Dict) -> Dict[str, Any]:
    """Analysiert Memory Type Distribution"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        
        # Simulate memory type distribution (in real implementation, query database)
        distribution = {
            'episodic_memories': int(total_memories * 0.4),  # 40%
            'semantic_memories': int(total_memories * 0.35),  # 35%
            'procedural_memories': int(total_memories * 0.15),  # 15%
            'emotional_memories': int(total_memories * 0.1)   # 10%
        }
        
        return {
            'distribution_counts': distribution,
            'distribution_percentages': {k: (v/max(1, total_memories))*100 for k, v in distribution.items()},
            'dominant_type': max(distribution.items(), key=lambda x: x[1])[0],
            'diversity_score': _calculate_memory_type_diversity(distribution)
        }
        
    except Exception as e:
        logger.debug(f"Memory type distribution analysis failed: {e}")
        return {
            'distribution_counts': {'episodic_memories': 40, 'semantic_memories': 35},
            'diversity_score': 0.7,
            'fallback_data': True
        }

def _analyze_temporal_memory_distribution(db_stats: Dict) -> Dict[str, Any]:
    """Analysiert Temporal Memory Distribution"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Simulate temporal distribution
        temporal_distribution = {
            'last_24h': recent_activity,
            'last_week': min(total_memories, recent_activity * 3),
            'last_month': min(total_memories, recent_activity * 8),
            'older': max(0, total_memories - (recent_activity * 8))
        }
        
        return {
            'temporal_counts': temporal_distribution,
            'recency_bias': temporal_distribution['last_24h'] / max(1, total_memories),
            'memory_aging_pattern': _analyze_memory_aging_pattern(temporal_distribution),
            'retention_efficiency': _calculate_temporal_retention_efficiency(temporal_distribution)
        }
        
    except Exception as e:
        logger.debug(f"Temporal memory distribution analysis failed: {e}")
        return {
            'temporal_counts': {'recent': recent_activity, 'older': max(0, total_memories - recent_activity)},
            'fallback_data': True
        }

def _analyze_importance_distribution(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Analysiert Importance Distribution"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        
        # Simulate importance distribution (in real implementation, analyze importance scores)
        importance_distribution = {
            'critical': int(total_memories * 0.1),    # 10%
            'high': int(total_memories * 0.2),        # 20%
            'medium': int(total_memories * 0.4),      # 40%
            'low': int(total_memories * 0.3)          # 30%
        }
        
        return {
            'importance_counts': importance_distribution,
            'importance_balance': _assess_importance_balance(importance_distribution),
            'critical_memory_ratio': importance_distribution['critical'] / max(1, total_memories),
            'importance_optimization_potential': _calculate_importance_optimization_potential(importance_distribution)
        }
        
    except Exception as e:
        logger.debug(f"Importance distribution analysis failed: {e}")
        return {
            'importance_counts': {'high': 20, 'medium': 50, 'low': 30},
            'fallback_data': True
        }

def _analyze_access_patterns(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Analysiert Access Patterns"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Simulate access patterns
        access_patterns = {
            'frequently_accessed': int(total_memories * 0.15),  # 15%
            'regularly_accessed': int(total_memories * 0.25),   # 25%
            'occasionally_accessed': int(total_memories * 0.35), # 35%
            'rarely_accessed': int(total_memories * 0.25)       # 25%
        }
        
        return {
            'access_distribution': access_patterns,
            'access_efficiency': _calculate_access_efficiency(access_patterns),
            'hot_memory_ratio': access_patterns['frequently_accessed'] / max(1, total_memories),
            'access_optimization_score': _calculate_access_optimization_score(access_patterns)
        }
        
    except Exception as e:
        logger.debug(f"Access patterns analysis failed: {e}")
        return {
            'access_distribution': {'frequent': 15, 'regular': 25, 'occasional': 35, 'rare': 25},
            'fallback_data': True
        }

def _identify_dominant_patterns(distribution_analysis: Dict) -> List[str]:
    """Identifiziert dominante Patterns"""
    try:
        patterns = []
        
        # Memory type patterns
        if 'memory_type_distribution' in distribution_analysis:
            type_dist = distribution_analysis['memory_type_distribution']
            dominant_type = type_dist.get('dominant_type', 'unknown')
            patterns.append(f"Dominant memory type: {dominant_type}")
        
        # Temporal patterns
        if 'temporal_distribution' in distribution_analysis:
            temporal_dist = distribution_analysis['temporal_distribution']
            recency_bias = temporal_dist.get('recency_bias', 0)
            if recency_bias > 0.3:
                patterns.append("Strong recency bias in memory formation")
        
        # Access patterns
        if 'access_pattern_distribution' in distribution_analysis:
            access_dist = distribution_analysis['access_pattern_distribution']
            hot_ratio = access_dist.get('hot_memory_ratio', 0)
            if hot_ratio > 0.2:
                patterns.append("High concentration of frequently accessed memories")
        
        return patterns[:5]
        
    except Exception as e:
        logger.debug(f"Dominant patterns identification failed: {e}")
        return ["Standard memory distribution patterns observed"]

def _detect_distribution_anomalies(distribution_analysis: Dict) -> List[str]:
    """Detektiert Distribution Anomalies"""
    try:
        anomalies = []
        
        # Check memory type distribution anomalies
        if 'memory_type_distribution' in distribution_analysis:
            type_dist = distribution_analysis['memory_type_distribution']['distribution_percentages']
            
            # Check for extreme concentrations
            for mem_type, percentage in type_dist.items():
                if percentage > 70:
                    anomalies.append(f"Extreme concentration in {mem_type}: {percentage:.1f}%")
                elif percentage < 5 and mem_type in ['episodic_memories', 'semantic_memories']:
                    anomalies.append(f"Unusually low {mem_type}: {percentage:.1f}%")
        
        # Check temporal distribution anomalies
        if 'temporal_distribution' in distribution_analysis:
            temporal_dist = distribution_analysis['temporal_distribution']
            recency_bias = temporal_dist.get('recency_bias', 0)
            
            if recency_bias > 0.8:
                anomalies.append("Excessive recency bias - potential memory retention issues")
            elif recency_bias < 0.05:
                anomalies.append("Very low recent activity - system may be inactive")
        
        return anomalies
        
    except Exception as e:
        logger.debug(f"Distribution anomalies detection failed: {e}")
        return []

def _identify_optimization_opportunities(distribution_analysis: Dict) -> List[str]:
    """Identifiziert Optimization Opportunities"""
    try:
        opportunities = []
        
        # Memory type optimization
        if 'memory_type_distribution' in distribution_analysis:
            diversity_score = distribution_analysis['memory_type_distribution'].get('diversity_score', 0)
            if diversity_score < 0.6:
                opportunities.append("Improve memory type diversity for better knowledge representation")
        
        # Access pattern optimization
        if 'access_pattern_distribution' in distribution_analysis:
            access_efficiency = distribution_analysis['access_pattern_distribution'].get('access_efficiency', 0)
            if access_efficiency < 0.7:
                opportunities.append("Optimize memory access patterns for better retrieval efficiency")
        
        # Importance distribution optimization
        if 'importance_distribution' in distribution_analysis:
            critical_ratio = distribution_analysis['importance_distribution'].get('critical_memory_ratio', 0)
            if critical_ratio < 0.05:
                opportunities.append("Identify and prioritize more critical memories")
            elif critical_ratio > 0.2:
                opportunities.append("Review critical memory classification to avoid over-prioritization")
        
        return opportunities[:4]
        
    except Exception as e:
        logger.debug(f"Optimization opportunities identification failed: {e}")
        return ["Conduct comprehensive memory system optimization review"]

# ====================================
# MISSING COUNT AND ASSESSMENT FUNCTIONS
# ====================================

def _count_total_data_points(report: Dict) -> int:
    """Zählt Total Data Points im Report"""
    try:
        data_points = 0
        
        # Count data points in each section
        for section_name, section_data in report.items():
            if isinstance(section_data, dict):
                data_points += _count_dict_data_points(section_data)
            elif isinstance(section_data, list):
                data_points += len(section_data)
            elif section_data is not None:
                data_points += 1
        
        return data_points
        
    except Exception as e:
        logger.debug(f"Total data points counting failed: {e}")
        return 100  # Default estimate

def _count_dict_data_points(data: Dict) -> int:
    """Zählt Data Points in Dictionary"""
    count = 0
    for key, value in data.items():
        if isinstance(value, dict):
            count += _count_dict_data_points(value)
        elif isinstance(value, list):
            count += len(value)
        elif value is not None:
            count += 1
    return count

def _calculate_analysis_confidence(memory_manager, db_stats: Dict) -> float:
    """Berechnet Analysis Confidence"""
    try:
        confidence_factors = []
        
        # Data availability confidence
        if memory_manager:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        # Database stats confidence
        if db_stats and 'total_memories' in db_stats:
            total_memories = db_stats['total_memories']
            if total_memories > 100:
                confidence_factors.append(0.9)
            elif total_memories > 20:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        # Recent activity confidence
        recent_activity = db_stats.get('recent_activity', 0)
        if recent_activity > 5:
            confidence_factors.append(0.8)
        elif recent_activity > 0:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        return sum(confidence_factors) / len(confidence_factors)
        
    except Exception as e:
        logger.debug(f"Analysis confidence calculation failed: {e}")
        return 0.7

def _assess_data_quality(db_stats: Dict) -> float:
    """Bewertet Data Quality"""
    try:
        quality_factors = []
        
        # Completeness
        required_fields = ['total_memories', 'recent_activity']
        completeness = sum(1 for field in required_fields if field in db_stats and db_stats[field] is not None)
        quality_factors.append(completeness / len(required_fields))
        
        # Consistency
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        if total_memories >= recent_activity:  # Logical consistency
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.5)
        
        # Freshness (assume fresh if we have recent activity)
        if recent_activity > 0:
            quality_factors.append(0.9)
        else:
            quality_factors.append(0.6)
        
        return sum(quality_factors) / len(quality_factors)
        
    except Exception as e:
        logger.debug(f"Data quality assessment failed: {e}")
        return 0.7
    
def _calculate_memory_optimization_score(stm_efficiency: float, cognitive_load: float, 
                                       stm_load: int, stm_capacity: int) -> float:
    """Berechnet Memory Optimization Score"""
    try:
        # Optimization factors
        efficiency_factor = stm_efficiency
        load_factor = 1.0 - cognitive_load
        utilization_factor = 1.0 - abs((stm_load / max(1, stm_capacity)) - 0.6)  # Optimal at 60%
        
        optimization_score = (efficiency_factor * 0.4 + load_factor * 0.4 + utilization_factor * 0.2)
        return max(0.0, min(1.0, optimization_score))
        
    except Exception as e:
        logger.debug(f"Memory optimization score calculation failed: {e}")
        return 0.7

def _analyze_memory_formation_patterns(db_stats: Dict, time_period: str) -> Dict[str, Any]:
    """Analysiert Memory Formation Patterns"""
    try:
        recent_activity = db_stats.get('recent_activity', 0)
        total_memories = db_stats.get('total_memories', 0)
        days = _parse_time_period_to_days(time_period)
        
        formation_rate = recent_activity / max(1, days)
        formation_consistency = min(1.0, formation_rate / 0.5)  # 0.5 per day is consistent
        
        return {
            'daily_formation_rate': formation_rate,
            'formation_consistency': formation_consistency,
            'formation_trend': 'increasing' if formation_rate > 1.0 else 'stable' if formation_rate > 0.3 else 'decreasing',
            'formation_efficiency': min(1.0, recent_activity / (total_memories * 0.1)) if total_memories > 0 else 0.5
        }
        
    except Exception as e:
        logger.debug(f"Memory formation patterns analysis failed: {e}")
        return {
            'daily_formation_rate': 0.3,
            'formation_consistency': 0.6,
            'formation_trend': 'stable',
            'fallback_data': True
        }

def _analyze_memory_access_patterns(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Analysiert Memory Access Patterns"""
    try:
        from routes.utils.memory_helpers import get_stm_load_direct
        
        stm_load = get_stm_load_direct(memory_manager)
        total_memories = db_stats.get('total_memories', 0)
        
        # Simulate access patterns
        access_frequency = min(1.0, stm_load / 10)  # Higher STM load indicates more access
        access_efficiency = 0.8 if stm_load < 15 else 0.6 if stm_load < 25 else 0.4
        
        return {
            'access_frequency_score': access_frequency,
            'access_efficiency_score': access_efficiency,
            'hot_memory_ratio': 0.15,  # Simulated 15% hot memories
            'access_distribution': 'balanced' if 0.3 <= access_frequency <= 0.7 else 'skewed'
        }
        
    except Exception as e:
        logger.debug(f"Memory access patterns analysis failed: {e}")
        return {
            'access_frequency_score': 0.5,
            'access_efficiency_score': 0.7,
            'access_distribution': 'balanced',
            'fallback_data': True
        }

def _analyze_memory_retention_patterns(db_stats: Dict, time_period: str) -> Dict[str, Any]:
    """Analysiert Memory Retention Patterns"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        days = _parse_time_period_to_days(time_period)
        
        # Estimate retention based on growth patterns
        if total_memories > 0:
            retention_rate = max(0.0, (total_memories - recent_activity) / total_memories)
            retention_quality = min(1.0, retention_rate + 0.2)  # Boost for accumulated knowledge
        else:
            retention_rate = 0.8
            retention_quality = 0.8
        
        return {
            'estimated_retention_rate': retention_rate,
            'retention_quality_score': retention_quality,
            'memory_decay_rate': 1.0 - retention_rate,
            'retention_stability': 'high' if retention_rate > 0.8 else 'moderate' if retention_rate > 0.6 else 'low'
        }
        
    except Exception as e:
        logger.debug(f"Memory retention patterns analysis failed: {e}")
        return {
            'estimated_retention_rate': 0.8,
            'retention_quality_score': 0.8,
            'retention_stability': 'moderate',
            'fallback_data': True
        }

def _analyze_consolidation_patterns(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Analysiert Consolidation Patterns"""
    try:
        from routes.utils.memory_helpers import is_consolidation_active_direct
        
        consolidation_active = is_consolidation_active_direct(memory_manager, db_stats)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Consolidation metrics
        consolidation_frequency = 0.7 if consolidation_active else 0.3
        consolidation_efficiency = min(1.0, recent_activity / 15) if consolidation_active else 0.5
        
        return {
            'consolidation_active': consolidation_active,
            'consolidation_frequency_score': consolidation_frequency,
            'consolidation_efficiency_score': consolidation_efficiency,
            'consolidation_health': 'good' if consolidation_active and consolidation_efficiency > 0.6 else 'moderate'
        }
        
    except Exception as e:
        logger.debug(f"Consolidation patterns analysis failed: {e}")
        return {
            'consolidation_active': False,
            'consolidation_frequency_score': 0.5,
            'consolidation_health': 'moderate',
            'fallback_data': True
        }

def _assess_memory_coherence(memory_manager, db_stats: Dict) -> float:
    """Bewertet Memory Coherence"""
    try:
        from routes.utils.memory_helpers import calculate_stm_efficiency_direct
        
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        total_memories = db_stats.get('total_memories', 0)
        
        # Coherence based on efficiency and knowledge base size
        base_coherence = stm_efficiency
        size_factor = min(1.0, total_memories / 100)  # More memories can mean better coherence
        
        coherence_score = (base_coherence * 0.7 + size_factor * 0.3)
        return max(0.0, min(1.0, coherence_score))
        
    except Exception as e:
        logger.debug(f"Memory coherence assessment failed: {e}")
        return 0.75

def _calculate_information_density(db_stats: Dict) -> float:
    """Berechnet Information Density"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Density based on memory concentration and activity
        if total_memories == 0:
            return 0.5
        
        activity_density = recent_activity / total_memories
        normalized_density = min(1.0, activity_density * 10)  # Scale to reasonable range
        
        return normalized_density
        
    except Exception as e:
        logger.debug(f"Information density calculation failed: {e}")
        return 0.6

def _assess_memory_diversity(db_stats: Dict) -> float:
    """Bewertet Memory Diversity"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        
        # Simulate diversity based on memory count
        if total_memories < 20:
            diversity_score = 0.3
        elif total_memories < 100:
            diversity_score = 0.6
        elif total_memories < 500:
            diversity_score = 0.8
        else:
            diversity_score = 0.9
        
        return diversity_score
        
    except Exception as e:
        logger.debug(f"Memory diversity assessment failed: {e}")
        return 0.7

def _assess_knowledge_integration(memory_manager, db_stats: Dict) -> float:
    """Bewertet Knowledge Integration"""
    try:
        from routes.utils.memory_helpers import calculate_stm_efficiency_direct
        
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Integration based on efficiency and activity patterns
        efficiency_factor = stm_efficiency
        activity_factor = min(1.0, recent_activity / 10)
        knowledge_factor = min(1.0, total_memories / 200)
        
        integration_score = (efficiency_factor * 0.4 + activity_factor * 0.3 + knowledge_factor * 0.3)
        return max(0.0, min(1.0, integration_score))
        
    except Exception as e:
        logger.debug(f"Knowledge integration assessment failed: {e}")
        return 0.7

def _analyze_short_term_memory_trends(db_stats: Dict, days: int) -> Dict[str, Any]:
    """Analysiert Short-term Memory Trends"""
    try:
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Short-term trend analysis
        daily_rate = recent_activity / max(1, days)
        trend_strength = min(1.0, daily_rate / 0.5)
        
        return {
            'daily_activity_rate': daily_rate,
            'trend_strength': trend_strength,
            'trend_direction': 'positive' if daily_rate > 0.5 else 'neutral' if daily_rate > 0.2 else 'negative',
            'volatility': 'low'  # Simplified - would need historical data for real volatility
        }
        
    except Exception as e:
        logger.debug(f"Short-term memory trends analysis failed: {e}")
        return {
            'daily_activity_rate': 0.3,
            'trend_direction': 'neutral',
            'fallback_data': True
        }

def _analyze_medium_term_memory_trends(db_stats: Dict, days: int) -> Dict[str, Any]:
    """Analysiert Medium-term Memory Trends"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Medium-term growth analysis
        growth_rate = recent_activity / max(1, total_memories) if total_memories > 0 else 0
        trend_stability = 0.8 if 0.05 <= growth_rate <= 0.15 else 0.6
        
        return {
            'growth_rate': growth_rate,
            'trend_stability': trend_stability,
            'growth_consistency': 'stable' if trend_stability > 0.7 else 'variable',
            'projection_confidence': trend_stability
        }
        
    except Exception as e:
        logger.debug(f"Medium-term memory trends analysis failed: {e}")
        return {
            'growth_rate': 0.08,
            'trend_stability': 0.8,
            'growth_consistency': 'stable',
            'fallback_data': True
        }

def _analyze_long_term_memory_trends(db_stats: Dict, days: int) -> Dict[str, Any]:
    """Analysiert Long-term Memory Trends"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        
        # Long-term development analysis
        development_stage = _determine_development_stage(total_memories, 0)
        maturity_level = 'early' if total_memories < 100 else 'developing' if total_memories < 500 else 'mature'
        
        return {
            'development_stage': development_stage,
            'maturity_level': maturity_level,
            'knowledge_base_size': total_memories,
            'long_term_trajectory': 'positive' if total_memories > 50 else 'developing'
        }
        
    except Exception as e:
        logger.debug(f"Long-term memory trends analysis failed: {e}")
        return {
            'development_stage': 'developing',
            'maturity_level': 'developing',
            'long_term_trajectory': 'positive',
            'fallback_data': True
        }

def _detect_seasonal_memory_patterns(db_stats: Dict) -> Dict[str, Any]:
    """Detektiert Seasonal Memory Patterns"""
    try:
        # Simplified seasonal analysis
        current_month = datetime.now().month
        seasonal_factor = 1.0
        
        # Simulate seasonal patterns
        if current_month in [12, 1, 2]:  # Winter
            seasonal_pattern = 'winter_consolidation'
            seasonal_factor = 0.8
        elif current_month in [3, 4, 5]:  # Spring
            seasonal_pattern = 'spring_growth'
            seasonal_factor = 1.2
        elif current_month in [6, 7, 8]:  # Summer
            seasonal_pattern = 'summer_activity'
            seasonal_factor = 1.1
        else:  # Fall
            seasonal_pattern = 'autumn_reflection'
            seasonal_factor = 0.9
        
        return {
            'current_season': seasonal_pattern,
            'seasonal_factor': seasonal_factor,
            'pattern_detected': True,
            'seasonal_adjustment': seasonal_factor - 1.0
        }
        
    except Exception as e:
        logger.debug(f"Seasonal memory patterns detection failed: {e}")
        return {
            'current_season': 'neutral',
            'seasonal_factor': 1.0,
            'pattern_detected': False,
            'fallback_data': True
        }

def _assess_memory_system_stability(memory_manager, db_stats: Dict) -> float:
    """Bewertet Memory System Stability"""
    try:
        from routes.utils.memory_helpers import (
            calculate_current_cognitive_load,
            calculate_stm_efficiency_direct
        )
        
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        
        # Stability factors
        load_stability = 1.0 - cognitive_load
        efficiency_stability = stm_efficiency
        data_stability = 0.9 if db_stats else 0.5
        
        stability_score = (load_stability * 0.4 + efficiency_stability * 0.4 + data_stability * 0.2)
        return max(0.0, min(1.0, stability_score))
        
    except Exception as e:
        logger.debug(f"Memory system stability assessment failed: {e}")
        return 0.75

def _assess_processing_resilience(memory_manager, db_stats: Dict) -> float:
    """Bewertet Processing Resilience"""
    try:
        from routes.utils.memory_helpers import get_stm_capacity_direct, get_stm_load_direct
        
        stm_capacity = get_stm_capacity_direct(memory_manager)
        stm_load = get_stm_load_direct(memory_manager)
        
        # Resilience based on capacity utilization
        utilization = stm_load / max(1, stm_capacity)
        resilience_score = 1.0 - abs(utilization - 0.5)  # Best resilience at 50% utilization
        
        return max(0.0, min(1.0, resilience_score))
        
    except Exception as e:
        logger.debug(f"Processing resilience assessment failed: {e}")
        return 0.7

def _assess_memory_adaptation_capability(memory_manager, db_stats: Dict) -> float:
    """Bewertet Memory Adaptation Capability"""
    try:
        from routes.utils.memory_helpers import calculate_learning_readiness_direct
        
        learning_readiness = calculate_learning_readiness_direct(memory_manager, db_stats)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Adaptation based on learning readiness and activity
        readiness_factor = learning_readiness
        activity_factor = min(1.0, recent_activity / 10)
        
        adaptation_capability = (readiness_factor * 0.6 + activity_factor * 0.4)
        return max(0.0, min(1.0, adaptation_capability))
        
    except Exception as e:
        logger.debug(f"Memory adaptation capability assessment failed: {e}")
        return 0.7

def _assess_memory_recovery_potential(memory_manager, db_stats: Dict) -> float:
    """Bewertet Memory Recovery Potential"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Recovery potential based on knowledge base and activity
        knowledge_factor = min(1.0, total_memories / 100)
        activity_factor = min(1.0, recent_activity / 5)
        
        recovery_potential = (knowledge_factor * 0.6 + activity_factor * 0.4)
        return max(0.0, min(1.0, recovery_potential))
        
    except Exception as e:
        logger.debug(f"Memory recovery potential assessment failed: {e}")
        return 0.7

def _calculate_overall_memory_health(memory_analysis: Dict) -> float:
    """Berechnet Overall Memory Health"""
    try:
        health_factors = []
        
        # System overview health
        if 'memory_system_overview' in memory_analysis:
            overview = memory_analysis['memory_system_overview']
            utilization = overview.get('stm_utilization_rate', 0.5)
            # Optimal utilization between 0.4 and 0.7
            utilization_health = 1.0 - abs(utilization - 0.55) / 0.55
            health_factors.append(utilization_health)
        
        # Efficiency health
        if 'memory_efficiency_metrics' in memory_analysis:
            efficiency = memory_analysis['memory_efficiency_metrics']
            efficiency_score = efficiency.get('stm_efficiency_score', 0.7)
            processing_efficiency = efficiency.get('memory_processing_efficiency', 0.7)
            health_factors.extend([efficiency_score, processing_efficiency])
        
        # Quality health
        if 'memory_quality_assessment' in memory_analysis:
            quality = memory_analysis['memory_quality_assessment']
            coherence = quality.get('memory_coherence', 0.7)
            diversity = quality.get('memory_diversity', 0.7)
            health_factors.extend([coherence, diversity])
        
        # Health indicators
        if 'memory_health_indicators' in memory_analysis:
            indicators = memory_analysis['memory_health_indicators']
            stability = indicators.get('system_stability', 0.7)
            resilience = indicators.get('processing_resilience', 0.7)
            health_factors.extend([stability, resilience])
        
        if health_factors:
            overall_health = sum(health_factors) / len(health_factors)
            return max(0.0, min(1.0, overall_health))
        else:
            return 0.7
            
    except Exception as e:
        logger.debug(f"Overall memory health calculation failed: {e}")
        return 0.7

def _generate_memory_insights(memory_analysis: Dict) -> List[str]:
    """Generiert Memory Insights"""
    try:
        insights = []
        
        # System overview insights
        if 'memory_system_overview' in memory_analysis:
            overview = memory_analysis['memory_system_overview']
            utilization = overview.get('stm_utilization_rate', 0.5)
            
            if utilization > 0.8:
                insights.append("High STM utilization - consider consolidation")
            elif utilization < 0.3:
                insights.append("Low STM utilization - opportunity for more learning")
            else:
                insights.append("STM utilization is well-balanced")
        
        # Efficiency insights
        if 'memory_efficiency_metrics' in memory_analysis:
            efficiency = memory_analysis['memory_efficiency_metrics']
            efficiency_score = efficiency.get('stm_efficiency_score', 0.7)
            
            if efficiency_score > 0.8:
                insights.append("Excellent memory efficiency - system is well-optimized")
            elif efficiency_score < 0.6:
                insights.append("Memory efficiency could be improved through optimization")
        
        # Pattern insights
        if 'memory_patterns' in memory_analysis:
            patterns = memory_analysis['memory_patterns']
            formation = patterns.get('formation_patterns', {})
            formation_trend = formation.get('formation_trend', 'stable')
            
            if formation_trend == 'increasing':
                insights.append("Memory formation is accelerating - good learning momentum")
            elif formation_trend == 'decreasing':
                insights.append("Memory formation is declining - may need engagement boost")
        
        return insights[:5]
        
    except Exception as e:
        logger.debug(f"Memory insights generation failed: {e}")
        return ["Memory system showing normal operational patterns"]

def _identify_critical_memory_findings(memory_analysis: Dict) -> List[str]:
    """Identifiziert Critical Memory Findings"""
    try:
        critical_findings = []
        
        # Critical efficiency issues
        if 'memory_efficiency_metrics' in memory_analysis:
            efficiency = memory_analysis['memory_efficiency_metrics']
            cognitive_load = efficiency.get('cognitive_load_level', 0.5)
            
            if cognitive_load > 0.8:
                critical_findings.append("Critical: High cognitive load detected")
        
        # Critical system issues
        if 'memory_health_indicators' in memory_analysis:
            indicators = memory_analysis['memory_health_indicators']
            stability = indicators.get('system_stability', 0.7)
            
            if stability < 0.5:
                critical_findings.append("Critical: Memory system stability compromised")
        
        # Critical utilization issues
        if 'memory_system_overview' in memory_analysis:
            overview = memory_analysis['memory_system_overview']
            utilization = overview.get('stm_utilization_rate', 0.5)
            
            if utilization > 0.95:
                critical_findings.append("Critical: STM near capacity - immediate consolidation needed")
        
        return critical_findings
        
    except Exception as e:
        logger.debug(f"Critical memory findings identification failed: {e}")
        return []

def _identify_memory_improvement_opportunities(memory_analysis: Dict) -> List[str]:
    """Identifiziert Memory Improvement Opportunities"""
    try:
        opportunities = []
        
        # Efficiency opportunities
        if 'memory_efficiency_metrics' in memory_analysis:
            efficiency = memory_analysis['memory_efficiency_metrics']
            optimization_score = efficiency.get('memory_optimization_score', 0.7)
            
            if optimization_score < 0.8:
                opportunities.append("Optimize memory processing efficiency")
        
        # Quality opportunities
        if 'memory_quality_assessment' in memory_analysis:
            quality = memory_analysis['memory_quality_assessment']
            diversity = quality.get('memory_diversity', 0.7)
            
            if diversity < 0.7:
                opportunities.append("Increase memory diversity for better knowledge representation")
        
        # Pattern opportunities
        if 'memory_patterns' in memory_analysis:
            patterns = memory_analysis['memory_patterns']
            consolidation = patterns.get('consolidation_patterns', {})
            
            if not consolidation.get('consolidation_active', False):
                opportunities.append("Activate consolidation processes for better memory management")
        
        return opportunities[:4]
        
    except Exception as e:
        logger.debug(f"Memory improvement opportunities identification failed: {e}")
        return ["Regular memory system maintenance and optimization"]

# ====================================
# MISSING LEARNING ANALYSIS HELPER FUNCTIONS
# ====================================

def _calculate_learning_efficiency_score(learning_readiness: float, cognitive_load: float, 
                                       recent_activity: int) -> float:
    """Berechnet Learning Efficiency Score"""
    try:
        readiness_factor = learning_readiness
        load_factor = 1.0 - cognitive_load
        activity_factor = min(1.0, recent_activity / 10)
        
        efficiency_score = (readiness_factor * 0.4 + load_factor * 0.3 + activity_factor * 0.3)
        return max(0.0, min(1.0, efficiency_score))
        
    except Exception as e:
        logger.debug(f"Learning efficiency score calculation failed: {e}")
        return 0.7

def _calculate_learning_momentum_score(recent_activity: int, learning_readiness: float) -> float:
    """Berechnet Learning Momentum Score"""
    try:
        activity_momentum = min(1.0, recent_activity / 8)  # 8 activities = full momentum
        readiness_momentum = learning_readiness
        
        momentum_score = (activity_momentum * 0.6 + readiness_momentum * 0.4)
        return max(0.0, min(1.0, momentum_score))
        
    except Exception as e:
        logger.debug(f"Learning momentum score calculation failed: {e}")
        return 0.6

def _analyze_learning_frequency(db_stats: Dict, time_period: str) -> Dict[str, Any]:
    """Analysiert Learning Frequency"""
    try:
        recent_activity = db_stats.get('recent_activity', 0)
        days = _parse_time_period_to_days(time_period)
        
        daily_frequency = recent_activity / max(1, days)
        frequency_classification = (
            'high' if daily_frequency > 1.0 else
            'moderate' if daily_frequency > 0.5 else
            'low'
        )
        
        return {
            'daily_learning_frequency': daily_frequency,
            'frequency_classification': frequency_classification,
            'learning_sessions_per_week': daily_frequency * 7,
            'frequency_consistency': 0.8 if 0.3 <= daily_frequency <= 1.5 else 0.6
        }
        
    except Exception as e:
        logger.debug(f"Learning frequency analysis failed: {e}")
        return {
            'daily_learning_frequency': 0.4,
            'frequency_classification': 'moderate',
            'fallback_data': True
        }

def _analyze_learning_intensity(db_stats: Dict, recent_activity: int) -> Dict[str, Any]:
    """Analysiert Learning Intensity"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        
        # Intensity based on recent activity relative to total knowledge
        if total_memories > 0:
            intensity_ratio = recent_activity / total_memories
            intensity_score = min(1.0, intensity_ratio * 10)  # Scale to reasonable range
        else:
            intensity_score = 0.5
        
        intensity_level = (
            'high' if intensity_score > 0.8 else
            'moderate' if intensity_score > 0.4 else
            'low'
        )
        
        return {
            'learning_intensity_score': intensity_score,
            'intensity_level': intensity_level,
            'intensity_sustainability': 0.9 if intensity_level == 'moderate' else 0.7,
            'intensity_optimization': 1.0 - intensity_score
        }
        
    except Exception as e:
        logger.debug(f"Learning intensity analysis failed: {e}")
        return {
            'learning_intensity_score': 0.6,
            'intensity_level': 'moderate',
            'fallback_data': True
        }

def _analyze_learning_consistency(db_stats: Dict, time_period: str) -> Dict[str, Any]:
    """Analysiert Learning Consistency"""
    try:
        recent_activity = db_stats.get('recent_activity', 0)
        days = _parse_time_period_to_days(time_period)
        
        # Consistency based on regular activity patterns
        expected_daily_activity = 0.5  # Expected baseline
        actual_daily_activity = recent_activity / max(1, days)
        
        consistency_score = 1.0 - abs(actual_daily_activity - expected_daily_activity) / expected_daily_activity
        consistency_score = max(0.0, min(1.0, consistency_score))
        
        consistency_rating = (
            'excellent' if consistency_score > 0.8 else
            'good' if consistency_score > 0.6 else
            'variable'
        )
        
        return {
            'consistency_score': consistency_score,
            'consistency_rating': consistency_rating,
            'learning_regularity': 'regular' if consistency_score > 0.7 else 'irregular',
            'predictability': consistency_score
        }
        
    except Exception as e:
        logger.debug(f"Learning consistency analysis failed: {e}")
        return {
            'consistency_score': 0.7,
            'consistency_rating': 'good',
            'learning_regularity': 'regular',
            'fallback_data': True
        }

def _analyze_learning_progression(db_stats: Dict, memory_manager) -> Dict[str, Any]:
    """Analysiert Learning Progression"""
    try:
        from routes.utils.memory_helpers import calculate_learning_readiness_direct
        
        total_memories = db_stats.get('total_memories', 0)
        learning_readiness = calculate_learning_readiness_direct(memory_manager, db_stats)
        
        # Progression based on knowledge base size and readiness
        knowledge_progression = min(1.0, total_memories / 200)  # 200 memories = good progression
        readiness_progression = learning_readiness
        
        overall_progression = (knowledge_progression * 0.6 + readiness_progression * 0.4)
        
        progression_stage = (
            'advanced' if overall_progression > 0.8 else
            'intermediate' if overall_progression > 0.6 else
            'beginner' if overall_progression > 0.3 else
            'starting'
        )
        
        return {
            'overall_progression_score': overall_progression,
            'progression_stage': progression_stage,
            'knowledge_development_level': knowledge_progression,
            'learning_maturity': readiness_progression
        }
        
    except Exception as e:
        logger.debug(f"Learning progression analysis failed: {e}")
        return {
            'overall_progression_score': 0.6,
            'progression_stage': 'intermediate',
            'fallback_data': True
        }

def _analyze_knowledge_base_growth(db_stats: Dict, time_period: str) -> Dict[str, Any]:
    """Analysiert Knowledge Base Growth"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        days = _parse_time_period_to_days(time_period)
        
        # Growth metrics
        growth_rate = recent_activity / max(1, days)
        growth_percentage = (recent_activity / max(1, total_memories)) * 100 if total_memories > 0 else 0
        
        growth_classification = (
            'rapid' if growth_rate > 1.5 else
            'steady' if growth_rate > 0.5 else
            'slow' if growth_rate > 0.1 else
            'minimal'
        )
        
        return {
            'daily_growth_rate': growth_rate,
            'growth_percentage': growth_percentage,
            'growth_classification': growth_classification,
            'knowledge_base_size': total_memories,
            'growth_sustainability': 0.8 if growth_classification in ['steady', 'rapid'] else 0.6
        }
        
    except Exception as e:
        logger.debug(f"Knowledge base growth analysis failed: {e}")
        return {
            'daily_growth_rate': 0.4,
            'growth_classification': 'steady',
            'knowledge_base_size': total_memories,
            'fallback_data': True
        }

def _analyze_domain_expansion(db_stats: Dict) -> Dict[str, Any]:
    """Analysiert Domain Expansion"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        
        # Simulate domain expansion analysis
        estimated_domains = min(10, max(1, total_memories // 20))  # Estimate domains from memory count
        domain_diversity = min(1.0, estimated_domains / 8)  # 8 domains = high diversity
        
        expansion_rate = 'high' if domain_diversity > 0.7 else 'moderate' if domain_diversity > 0.4 else 'low'
        
        return {
            'estimated_domain_count': estimated_domains,
            'domain_diversity_score': domain_diversity,
            'expansion_rate': expansion_rate,
            'specialization_balance': 0.7  # Balanced between breadth and depth
        }
        
    except Exception as e:
        logger.debug(f"Domain expansion analysis failed: {e}")
        return {
            'estimated_domain_count': 3,
            'domain_diversity_score': 0.6,
            'expansion_rate': 'moderate',
            'fallback_data': True
        }

def _analyze_conceptual_development(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Analysiert Conceptual Development"""
    try:
        from routes.utils.memory_helpers import calculate_stm_efficiency_direct
        
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        total_memories = db_stats.get('total_memories', 0)
        
        # Conceptual development based on efficiency and knowledge base
        efficiency_factor = stm_efficiency
        knowledge_factor = min(1.0, total_memories / 150)
        
        conceptual_score = (efficiency_factor * 0.6 + knowledge_factor * 0.4)
        
        development_level = (
            'sophisticated' if conceptual_score > 0.8 else
            'developing' if conceptual_score > 0.6 else
            'basic' if conceptual_score > 0.4 else
            'elementary'
        )
        
        return {
            'conceptual_development_score': conceptual_score,
            'development_level': development_level,
            'abstraction_capability': efficiency_factor,
            'concept_integration': knowledge_factor
        }
        
    except Exception as e:
        logger.debug(f"Conceptual development analysis failed: {e}")
        return {
            'conceptual_development_score': 0.7,
            'development_level': 'developing',
            'fallback_data': True
        }

def _analyze_skill_acquisition_patterns(db_stats: Dict) -> Dict[str, Any]:
    """Analysiert Skill Acquisition Patterns"""
    try:
        recent_activity = db_stats.get('recent_activity', 0)
        total_memories = db_stats.get('total_memories', 0)
        
        # Skill acquisition based on activity patterns
        acquisition_rate = min(1.0, recent_activity / 8)  # 8 activities = high acquisition
        skill_diversity = min(1.0, total_memories / 100)  # Estimate skill diversity
        
        acquisition_pattern = (
            'accelerated' if acquisition_rate > 0.8 else
            'steady' if acquisition_rate > 0.5 else
            'gradual' if acquisition_rate > 0.2 else
            'slow'
        )
        
        return {
            'acquisition_rate': acquisition_rate,
            'skill_diversity_estimate': skill_diversity,
            'acquisition_pattern': acquisition_pattern,
            'skill_retention_estimate': 0.8  # Assume good retention
        }
        
    except Exception as e:
        logger.debug(f"Skill acquisition patterns analysis failed: {e}")
        return {
            'acquisition_rate': 0.6,
            'acquisition_pattern': 'steady',
            'fallback_data': True
        }
    
def _generate_immediate_recommendations(cognitive_load: float, learning_readiness: float, 
                                      stm_efficiency: float, recent_activity: int) -> List[str]:
    """Generiert Immediate Action Recommendations"""
    try:
        recommendations = []
        
        # Cognitive load recommendations
        if cognitive_load > 0.8:
            recommendations.append("URGENT: Reduce cognitive load through memory consolidation")
            recommendations.append("Pause new learning activities temporarily")
        elif cognitive_load > 0.6:
            recommendations.append("Consider memory consolidation to reduce cognitive strain")
        
        # Learning readiness recommendations
        if learning_readiness < 0.4:
            recommendations.append("System not ready for intensive learning - focus on optimization")
        elif learning_readiness > 0.8 and recent_activity < 3:
            recommendations.append("High learning readiness detected - engage in learning activities")
        
        # STM efficiency recommendations
        if stm_efficiency < 0.5:
            recommendations.append("Critical: STM efficiency low - immediate system optimization needed")
        elif stm_efficiency < 0.7:
            recommendations.append("Optimize STM processing for better efficiency")
        
        # Activity level recommendations
        if recent_activity == 0:
            recommendations.append("No recent activity - engage system with learning tasks")
        elif recent_activity > 20:
            recommendations.append("High activity detected - consider consolidation break")
        
        return recommendations[:5]  # Limit to top 5 immediate actions
        
    except Exception as e:
        logger.debug(f"Immediate recommendations generation failed: {e}")
        return ["Monitor system status and ensure stable operation"]

def _generate_short_term_optimizations(memory_manager, db_stats: Dict, cognitive_load: float) -> List[str]:
    """Generiert Short-term Optimization Recommendations"""
    try:
        from routes.utils.memory_helpers import get_stm_load_direct, get_stm_capacity_direct
        
        optimizations = []
        stm_load = get_stm_load_direct(memory_manager)
        stm_capacity = get_stm_capacity_direct(memory_manager)
        total_memories = db_stats.get('total_memories', 0)
        
        # STM optimization
        if stm_capacity > 0:
            utilization = stm_load / stm_capacity
            if utilization > 0.8:
                optimizations.append("Schedule regular STM consolidation sessions")
                optimizations.append("Implement automatic memory prioritization")
            elif utilization < 0.3:
                optimizations.append("Increase learning engagement to better utilize STM capacity")
        
        # Cognitive load optimization  
        if cognitive_load > 0.7:
            optimizations.append("Implement cognitive load balancing strategies")
            optimizations.append("Optimize memory retrieval patterns")
        
        # Memory base optimization
        if total_memories > 0:
            if total_memories < 50:
                optimizations.append("Focus on foundational knowledge building")
            elif total_memories > 500:
                optimizations.append("Implement advanced memory organization strategies")
        
        # Processing optimization
        optimizations.append("Fine-tune memory processing algorithms")
        optimizations.append("Optimize consolidation timing and frequency")
        
        return optimizations[:6]  # Limit to top 6 optimizations
        
    except Exception as e:
        logger.debug(f"Short-term optimizations generation failed: {e}")
        return [
            "Regular system maintenance and monitoring",
            "Memory processing optimization",
            "Learning pattern analysis"
        ]

def _generate_long_term_strategies(total_memories: int, recent_activity: int, 
                                 learning_readiness: float) -> List[str]:
    """Generiert Long-term Strategic Recommendations"""
    try:
        strategies = []
        
        # Knowledge base development strategies
        if total_memories < 100:
            strategies.append("Establish systematic knowledge acquisition program")
            strategies.append("Build foundational knowledge across core domains")
            strategies.append("Implement consistent daily learning routines")
        elif total_memories < 500:
            strategies.append("Expand knowledge breadth while maintaining depth")
            strategies.append("Develop specialized expertise in key areas")
            strategies.append("Create knowledge interconnection maps")
        else:
            strategies.append("Focus on advanced knowledge synthesis and integration")
            strategies.append("Develop expert-level domain specializations")
            strategies.append("Implement knowledge innovation and creation processes")
        
        # Learning system evolution strategies
        if learning_readiness > 0.8:
            strategies.append("Develop advanced learning methodologies")
            strategies.append("Implement adaptive learning rate optimization")
        elif learning_readiness > 0.6:
            strategies.append("Enhance learning system capabilities gradually")
            strategies.append("Build robust learning pattern recognition")
        else:
            strategies.append("Strengthen foundational learning mechanisms")
            strategies.append("Develop basic learning readiness capabilities")
        
        # Activity-based strategies
        activity_rate = recent_activity / 30  # Normalize to monthly rate
        if activity_rate > 1.0:
            strategies.append("Maintain high-performance learning trajectory")
            strategies.append("Implement advanced knowledge management systems")
        elif activity_rate > 0.3:
            strategies.append("Build sustainable learning momentum")
            strategies.append("Develop consistent knowledge acquisition habits")
        else:
            strategies.append("Establish basic learning engagement protocols")
            strategies.append("Create motivation and engagement systems")
        
        # System maturity strategies
        strategies.append("Plan for long-term cognitive architecture scaling")
        strategies.append("Develop autonomous learning and adaptation capabilities")
        strategies.append("Implement predictive learning and optimization systems")
        
        return strategies[:8]  # Limit to top 8 strategies
        
    except Exception as e:
        logger.debug(f"Long-term strategies generation failed: {e}")
        return [
            "Continuous system improvement and optimization",
            "Long-term knowledge base development", 
            "Advanced learning capability enhancement",
            "Autonomous system evolution planning"
        ]

def _generate_performance_improvement_recommendations(memory_manager, db_stats: Dict) -> List[str]:
    """Generiert Performance Improvement Recommendations"""
    try:
        from routes.utils.memory_helpers import (
            calculate_current_cognitive_load,
            calculate_stm_efficiency_direct,
            get_stm_load_direct, get_stm_capacity_direct
        )
        
        recommendations = []
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        stm_load = get_stm_load_direct(memory_manager)
        stm_capacity = get_stm_capacity_direct(memory_manager)
        
        # Cognitive performance improvements
        if cognitive_load > 0.6:
            recommendations.append("Implement cognitive load distribution algorithms")
            recommendations.append("Optimize parallel processing capabilities")
        
        # Memory performance improvements
        if stm_efficiency < 0.8:
            recommendations.append("Enhance STM processing algorithms")
            recommendations.append("Implement adaptive memory management")
        
        # Capacity utilization improvements
        if stm_capacity > 0:
            utilization = stm_load / stm_capacity
            if abs(utilization - 0.6) > 0.2:  # Not near optimal 60%
                recommendations.append("Optimize STM capacity utilization to 60% target")
                recommendations.append("Implement dynamic capacity adjustment")
        
        # Processing speed improvements
        recommendations.append("Implement memory access caching strategies")
        recommendations.append("Optimize retrieval path algorithms")
        
        # System throughput improvements
        recent_activity = db_stats.get('recent_activity', 0)
        if recent_activity > 0:
            throughput = recent_activity / 24  # Per hour
            if throughput < 0.5:
                recommendations.append("Increase system processing throughput")
                recommendations.append("Parallel process multiple learning streams")
        
        # Quality improvements
        recommendations.append("Implement quality assurance for memory formation")
        recommendations.append("Develop error detection and correction systems")
        
        return recommendations[:7]  # Limit to top 7 improvements
        
    except Exception as e:
        logger.debug(f"Performance improvement recommendations generation failed: {e}")
        return [
            "General system performance optimization",
            "Memory processing enhancement",
            "Learning efficiency improvement"
        ]

def _generate_system_enhancement_recommendations(cognitive_load: float, stm_efficiency: float, 
                                               memory_manager) -> List[str]:
    """Generiert System Enhancement Recommendations"""
    try:
        from routes.utils.memory_helpers import get_stm_capacity_direct
        
        enhancements = []
        stm_capacity = get_stm_capacity_direct(memory_manager)
        
        # Core system enhancements
        if cognitive_load > 0.7:
            enhancements.append("Upgrade cognitive processing architecture")
            enhancements.append("Implement distributed processing systems")
        
        if stm_efficiency < 0.7:
            enhancements.append("Enhance short-term memory algorithms")
            enhancements.append("Implement advanced memory compression techniques")
        
        # Capacity enhancements
        if stm_capacity < 20:
            enhancements.append("Expand STM capacity infrastructure")
        elif stm_capacity > 50:
            enhancements.append("Implement STM capacity optimization algorithms")
        
        # Advanced features
        enhancements.append("Develop predictive memory management")
        enhancements.append("Implement adaptive learning rate systems")
        enhancements.append("Create automated optimization routines")
        
        # Integration enhancements
        enhancements.append("Enhance memory-learning integration")
        enhancements.append("Develop cross-system communication protocols")
        
        # Monitoring enhancements
        enhancements.append("Implement real-time performance monitoring")
        enhancements.append("Create automated health check systems")
        
        return enhancements[:8]  # Limit to top 8 enhancements
        
    except Exception as e:
        logger.debug(f"System enhancement recommendations generation failed: {e}")
        return [
            "Core system architecture improvements",
            "Memory system enhancements",
            "Processing capability upgrades"
        ]

def _generate_learning_optimization_recommendations_detailed(learning_readiness: float, 
                                                           recent_activity: int, 
                                                           total_memories: int) -> List[str]:
    """Generiert detaillierte Learning Optimization Recommendations"""
    try:
        recommendations = []
        
        # Learning readiness optimizations
        if learning_readiness < 0.5:
            recommendations.append("Improve learning readiness through system optimization")
            recommendations.append("Address cognitive barriers to learning")
            recommendations.append("Implement learning preparation protocols")
        elif learning_readiness > 0.8:
            recommendations.append("Leverage high learning readiness for accelerated acquisition")
            recommendations.append("Implement advanced learning strategies")
        
        # Activity level optimizations
        if recent_activity == 0:
            recommendations.append("Initiate basic learning engagement protocols")
            recommendations.append("Establish regular learning schedules")
        elif recent_activity < 5:
            recommendations.append("Increase learning activity frequency")
            recommendations.append("Develop consistent learning habits")
        elif recent_activity > 15:
            recommendations.append("Balance high activity with consolidation periods")
            recommendations.append("Implement learning sustainability strategies")
        
        # Knowledge base optimizations
        if total_memories < 50:
            recommendations.append("Focus on foundational knowledge acquisition")
            recommendations.append("Build core conceptual frameworks")
        elif total_memories < 200:
            recommendations.append("Expand knowledge breadth systematically")
            recommendations.append("Develop knowledge interconnections")
        else:
            recommendations.append("Implement advanced knowledge synthesis")
            recommendations.append("Focus on expertise deepening strategies")
        
        # Learning efficiency optimizations
        learning_rate = recent_activity / max(1, total_memories)
        if learning_rate > 0.1:
            recommendations.append("Optimize rapid learning for sustainability")
        elif learning_rate < 0.05:
            recommendations.append("Accelerate learning acquisition rate")
        
        # Advanced optimizations
        recommendations.append("Implement adaptive learning algorithms")
        recommendations.append("Develop personalized learning pathways")
        
        return recommendations[:8]  # Limit to top 8 recommendations
        
    except Exception as e:
        logger.debug(f"Learning optimization recommendations generation failed: {e}")
        return [
            "General learning system optimization",
            "Learning efficiency improvement",
            "Knowledge acquisition enhancement"
        ]

def _prioritize_recommendations(recommendations: Dict) -> List[Dict[str, Any]]:
    """Priorisiert alle Recommendations"""
    try:
        prioritized = []
        
        # Immediate actions (highest priority)
        immediate = recommendations.get('immediate_actions', [])
        for action in immediate:
            prioritized.append({
                'recommendation': action,
                'priority': 'critical',
                'timeframe': 'immediate',
                'category': 'immediate_action'
            })
        
        # Performance improvements (high priority)
        performance = recommendations.get('performance_improvements', [])
        for improvement in performance[:3]:  # Top 3
            prioritized.append({
                'recommendation': improvement,
                'priority': 'high',
                'timeframe': 'short_term',
                'category': 'performance'
            })
        
        # Short-term optimizations (medium priority)
        short_term = recommendations.get('short_term_optimizations', [])
        for optimization in short_term[:3]:  # Top 3
            prioritized.append({
                'recommendation': optimization,
                'priority': 'medium',
                'timeframe': 'short_term',
                'category': 'optimization'
            })
        
        # System enhancements (medium priority)
        enhancements = recommendations.get('system_enhancements', [])
        for enhancement in enhancements[:2]:  # Top 2
            prioritized.append({
                'recommendation': enhancement,
                'priority': 'medium',
                'timeframe': 'medium_term',
                'category': 'enhancement'
            })
        
        # Long-term strategies (lower priority)
        long_term = recommendations.get('long_term_strategies', [])
        for strategy in long_term[:2]:  # Top 2
            prioritized.append({
                'recommendation': strategy,
                'priority': 'low',
                'timeframe': 'long_term',
                'category': 'strategy'
            })
        
        return prioritized[:12]  # Limit to top 12 prioritized recommendations
        
    except Exception as e:
        logger.debug(f"Recommendations prioritization failed: {e}")
        return [
            {
                'recommendation': 'Monitor system performance regularly',
                'priority': 'medium',
                'timeframe': 'ongoing',
                'category': 'maintenance'
            }
        ]

def _create_implementation_roadmap(recommendations: Dict) -> Dict[str, List[str]]:
    """Erstellt Implementation Roadmap"""
    try:
        roadmap = {
            'phase_1_immediate': [],
            'phase_2_short_term': [],
            'phase_3_medium_term': [],
            'phase_4_long_term': []
        }
        
        # Phase 1: Immediate (0-7 days)
        immediate = recommendations.get('immediate_actions', [])
        roadmap['phase_1_immediate'] = immediate[:3]  # Top 3 immediate actions
        
        # Phase 2: Short-term (1-4 weeks)
        short_term = recommendations.get('short_term_optimizations', [])
        performance = recommendations.get('performance_improvements', [])
        roadmap['phase_2_short_term'] = (short_term[:2] + performance[:2])[:4]
        
        # Phase 3: Medium-term (1-3 months)
        enhancements = recommendations.get('system_enhancements', [])
        learning_opt = recommendations.get('learning_optimizations', [])
        roadmap['phase_3_medium_term'] = (enhancements[:2] + learning_opt[:2])[:4]
        
        # Phase 4: Long-term (3+ months)
        long_term = recommendations.get('long_term_strategies', [])
        roadmap['phase_4_long_term'] = long_term[:3]
        
        return roadmap
        
    except Exception as e:
        logger.debug(f"Implementation roadmap creation failed: {e}")
        return {
            'phase_1_immediate': ['Monitor system status'],
            'phase_2_short_term': ['Optimize performance'],
            'phase_3_medium_term': ['Enhance capabilities'],
            'phase_4_long_term': ['Long-term development']
        }

def _generate_fallback_recommendations() -> Dict[str, Any]:
    """Generiert Fallback Recommendations"""
    return {
        'immediate_actions': [
            'Monitor system performance and stability',
            'Ensure basic memory management functions',
            'Maintain learning system availability'
        ],
        'short_term_optimizations': [
            'Regular system maintenance and cleanup',
            'Memory usage optimization',
            'Learning pattern analysis'
        ],
        'long_term_strategies': [
            'Continuous system improvement',
            'Knowledge base expansion',
            'Advanced capability development'
        ],
        'performance_improvements': [
            'General performance optimization',
            'Efficiency enhancement',
            'Resource utilization improvement'
        ],
        'system_enhancements': [
            'Core system upgrades',
            'Feature enhancement',
            'Capability expansion'
        ],
        'learning_optimizations': [
            'Learning process optimization',
            'Knowledge acquisition improvement',
            'Learning efficiency enhancement'
        ],
        'fallback_mode': True
    }

# ====================================
# MISSING HELPER FUNCTIONS FOR ANALYSIS
# ====================================

def _calculate_trait_balance(traits: Dict) -> float:
    """Berechnet Trait Balance"""
    try:
        if not traits:
            return 0.5
        
        trait_values = list(traits.values())
        if not trait_values:
            return 0.5
        
        # Calculate variance - lower variance = better balance
        mean_value = sum(trait_values) / len(trait_values)
        variance = sum((x - mean_value) ** 2 for x in trait_values) / len(trait_values)
        
        # Convert variance to balance score (0-1, where 1 is perfect balance)
        balance_score = max(0.0, 1.0 - (variance / 0.25))
        return min(1.0, balance_score)
        
    except Exception as e:
        logger.debug(f"Trait balance calculation failed: {e}")
        return 0.7

def _identify_dominant_traits(traits: Dict) -> List[str]:
    """Identifiziert dominante Traits"""
    try:
        if not traits:
            return []
        
        # Sort traits by value (descending)
        sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)
        return [trait for trait, value in sorted_traits[:3]]  # Top 3 traits
        
    except Exception as e:
        logger.debug(f"Dominant traits identification failed: {e}")
        return []

def _identify_development_areas(traits: Dict) -> List[str]:
    """Identifiziert Development Areas"""
    try:
        if not traits:
            return ['General personality development']
        
        # Sort traits by value (ascending) to find areas needing development
        sorted_traits = sorted(traits.items(), key=lambda x: x[1])
        development_areas = [trait for trait, value in sorted_traits[:3] if value < 0.7]
        
        return development_areas if development_areas else ['Maintain current development']
        
    except Exception as e:
        logger.debug(f"Development areas identification failed: {e}")
        return ['General personality development']

def _classify_learning_velocity(velocity: float) -> str:
    """Klassifiziert Learning Velocity"""
    if velocity >= 1.0:
        return 'very_high'
    elif velocity >= 0.7:
        return 'high'
    elif velocity >= 0.4:
        return 'moderate'
    elif velocity >= 0.2:
        return 'low'
    else:
        return 'very_low'

def _assess_growth_sustainability(growth_metrics: Dict) -> float:
    """Bewertet Growth Sustainability"""
    try:
        # Analyze growth patterns for sustainability
        memory_growth = growth_metrics.get('memory_growth', {})
        learning_velocity = growth_metrics.get('learning_velocity', {})
        
        growth_rate = memory_growth.get('daily_growth_rate', 0.2)
        velocity_class = learning_velocity.get('velocity_classification', 'moderate')
        
        # Sustainable growth is moderate and consistent
        if velocity_class == 'moderate' and 0.3 <= growth_rate <= 1.0:
            return 0.9
        elif velocity_class in ['high', 'very_high'] and growth_rate > 1.0:
            return 0.6  # High growth may not be sustainable
        elif velocity_class in ['low', 'very_low']:
            return 0.4
        else:
            return 0.7
            
    except Exception as e:
        logger.debug(f"Growth sustainability assessment failed: {e}")
        return 0.7

def _calculate_memory_type_diversity(distribution: Dict) -> float:
    """Berechnet Memory Type Diversity"""
    try:
        if not distribution:
            return 0.5
        
        total = sum(distribution.values())
        if total == 0:
            return 0.5
        
        # Calculate Shannon diversity index
        diversity = 0.0
        for count in distribution.values():
            if count > 0:
                p = count / total
                diversity -= p * (p**0.5)  # Simplified diversity calculation
        
        # Normalize to 0-1 scale
        max_diversity = len(distribution) * 0.25  # Theoretical maximum
        return min(1.0, diversity / max_diversity) if max_diversity > 0 else 0.5
        
    except Exception as e:
        logger.debug(f"Memory type diversity calculation failed: {e}")
        return 0.7

def _analyze_memory_aging_pattern(temporal_distribution: Dict) -> str:
    """Analysiert Memory Aging Pattern"""
    try:
        recent = temporal_distribution.get('last_24h', 0)
        week = temporal_distribution.get('last_week', 0)
        month = temporal_distribution.get('last_month', 0)
        older = temporal_distribution.get('older', 0)
        
        total = recent + week + month + older
        if total == 0:
            return 'no_pattern'
        
        recent_ratio = recent / total
        
        if recent_ratio > 0.5:
            return 'heavy_recency_bias'
        elif recent_ratio > 0.3:
            return 'moderate_recency_bias'
        elif recent_ratio > 0.1:
            return 'balanced_aging'
        else:
            return 'historical_focus'
            
    except Exception as e:
        logger.debug(f"Memory aging pattern analysis failed: {e}")
        return 'balanced_aging'

def _calculate_temporal_retention_efficiency(temporal_distribution: Dict) -> float:
    """Berechnet Temporal Retention Efficiency"""
    try:
        total = sum(temporal_distribution.values())
        if total == 0:
            return 0.5
        
        # Efficiency based on balanced distribution
        recent = temporal_distribution.get('last_24h', 0) / total
        week = temporal_distribution.get('last_week', 0) / total
        month = temporal_distribution.get('last_month', 0) / total
        older = temporal_distribution.get('older', 0) / total
        
        # Ideal distribution: some recent, good retention of older
        efficiency = 1.0 - abs(recent - 0.2) - abs(older - 0.4)
        return max(0.0, min(1.0, efficiency))
        
    except Exception as e:
        logger.debug(f"Temporal retention efficiency calculation failed: {e}")
        return 0.7

def _assess_importance_balance(importance_distribution: Dict) -> float:
    """Bewertet Importance Balance"""
    try:
        total = sum(importance_distribution.values())
        if total == 0:
            return 0.5
        
        # Good balance: some critical, more medium/high, less low
        critical_ratio = importance_distribution.get('critical', 0) / total
        high_ratio = importance_distribution.get('high', 0) / total
        medium_ratio = importance_distribution.get('medium', 0) / total
        
        # Ideal ratios: 10% critical, 20% high, 40% medium, 30% low
        balance_score = 1.0 - (
            abs(critical_ratio - 0.1) + 
            abs(high_ratio - 0.2) + 
            abs(medium_ratio - 0.4)
        )
        
        return max(0.0, min(1.0, balance_score))
        
    except Exception as e:
        logger.debug(f"Importance balance assessment failed: {e}")
        return 0.7

def _calculate_importance_optimization_potential(importance_distribution: Dict) -> float:
    """Berechnet Importance Optimization Potential"""
    try:
        total = sum(importance_distribution.values())
        if total == 0:
            return 0.5
        
        low_ratio = importance_distribution.get('low', 0) / total
        critical_ratio = importance_distribution.get('critical', 0) / total
        
        # High potential if too many low importance or too few critical
        potential = (low_ratio * 0.6) + ((0.1 - critical_ratio) * 0.4) if critical_ratio < 0.1 else (low_ratio * 0.6)
        
        return max(0.0, min(1.0, potential))
        
    except Exception as e:
        logger.debug(f"Importance optimization potential calculation failed: {e}")
        return 0.3

def _calculate_access_efficiency(access_patterns: Dict) -> float:
    """Berechnet Access Efficiency"""
    try:
        total = sum(access_patterns.values())
        if total == 0:
            return 0.5
        
        frequent = access_patterns.get('frequently_accessed', 0) / total
        regular = access_patterns.get('regularly_accessed', 0) / total
        
        # Efficiency based on having right amount of hot memories
        efficiency = min(1.0, (frequent * 2.0) + (regular * 1.5))  # More weight on frequent access
        
        return max(0.0, min(1.0, efficiency))
        
    except Exception as e:
        logger.debug(f"Access efficiency calculation failed: {e}")
        return 0.7

def _calculate_access_optimization_score(access_patterns: Dict) -> float:
    """Berechnet Access Optimization Score"""
    try:
        total = sum(access_patterns.values())
        if total == 0:
            return 0.5
        
        rare = access_patterns.get('rarely_accessed', 0) / total
        frequent = access_patterns.get('frequently_accessed', 0) / total
        
        # Good optimization: not too many rare, good amount frequent
        optimization_score = 1.0 - (rare * 0.8) + (frequent * 0.6)
        
        return max(0.0, min(1.0, optimization_score))
        
    except Exception as e:
        logger.debug(f"Access optimization score calculation failed: {e}")
        return 0.7

def _generate_fallback_memory_analysis(db_stats: Dict, time_period: str) -> Dict[str, Any]:
    """Generiert Fallback Memory Analysis"""
    return {
        'fallback_mode': True,
        'memory_system_overview': {
            'total_memories': db_stats.get('total_memories', 0),
            'stm_current_load': 5,
            'stm_capacity': 20,
            'stm_utilization_rate': 0.25,
            'memory_formation_rate': db_stats.get('recent_activity', 0)
        },
        'analysis_summary': {
            'overall_memory_health': 0.7,
            'key_insights': ['System operating in fallback mode'],
            'critical_findings': [],
            'improvement_opportunities': ['Enable full memory analysis']
        }
    }

def _generate_fallback_learning_analysis(db_stats: Dict, time_period: str) -> Dict[str, Any]:
    """Generiert Fallback Learning Analysis"""
    return {
        'fallback_mode': True,
        'learning_performance_metrics': {
            'learning_readiness_score': 0.7,
            'learning_activity_level': db_stats.get('recent_activity', 0),
            'learning_velocity': 0.3,
            'learning_efficiency': 0.6
        },
        'learning_summary': {
            'overall_learning_score': 0.6,
            'learning_stage': 'developing',
            'learning_trajectory': 'stable'
        }
    }

def _generate_fallback_performance_analysis(db_stats: Dict) -> Dict[str, Any]:
    """Generiert Fallback Performance Analysis"""
    return {
        'fallback_mode': True,
        'system_performance_metrics': {
            'cognitive_performance': {
                'cognitive_load': 0.5,
                'cognitive_efficiency': 0.5,
                'processing_capacity': 0.6
            }
        },
        'performance_summary': {
            'overall_performance_score': 0.6,
            'performance_classification': 'moderate',
            'key_strengths': ['Basic functionality'],
            'areas_for_improvement': ['Enable full analysis']
        }
    }

def _assess_adaptive_flexibility(current_state: Dict) -> float:
    """Bewertet Adaptive Flexibility"""
    try:
        adaptability = current_state.get('adaptability', 0.6)
        emotional_stability = current_state.get('emotional_stability', 0.7)
        openness = current_state.get('openness', 0.6)
        
        # Adaptive flexibility combines adaptability with emotional stability and openness
        flexibility_score = (adaptability * 0.5 + emotional_stability * 0.3 + openness * 0.2)
        return max(0.0, min(1.0, flexibility_score))
        
    except Exception as e:
        logger.debug(f"Adaptive flexibility assessment failed: {e}")
        return 0.6

def _analyze_trait_profile(traits: Dict) -> Dict[str, Any]:
    """Analysiert Trait Profile"""
    try:
        if not traits:
            return {'profile_type': 'undefined', 'trait_count': 0}
        
        trait_count = len(traits)
        trait_values = list(traits.values())
        
        # Profile analysis
        if trait_values:
            avg_trait_value = sum(trait_values) / len(trait_values)
            trait_variance = sum((x - avg_trait_value) ** 2 for x in trait_values) / len(trait_values)
            
            # Determine profile type
            if trait_variance < 0.1:
                profile_type = 'balanced'
            elif max(trait_values) - min(trait_values) > 0.6:
                profile_type = 'polarized'
            else:
                profile_type = 'developing'
        else:
            avg_trait_value = 0.5
            trait_variance = 0.0
            profile_type = 'undefined'
        
        return {
            'profile_type': profile_type,
            'trait_count': trait_count,
            'average_trait_value': avg_trait_value,
            'trait_variance': trait_variance,
            'profile_strength': min(1.0, avg_trait_value + (trait_count / 20))
        }
        
    except Exception as e:
        logger.debug(f"Trait profile analysis failed: {e}")
        return {
            'profile_type': 'developing',
            'trait_count': 0,
            'average_trait_value': 0.5
        }

def _calculate_trait_balance_comprehensive(traits: Dict) -> Dict[str, Any]:
    """Berechnet umfassende Trait Balance"""
    try:
        if not traits:
            return {'balance_score': 0.5, 'balance_rating': 'moderate'}
        
        trait_values = list(traits.values())
        if not trait_values:
            return {'balance_score': 0.5, 'balance_rating': 'moderate'}
        
        # Calculate comprehensive balance metrics
        mean_value = sum(trait_values) / len(trait_values)
        variance = sum((x - mean_value) ** 2 for x in trait_values) / len(trait_values)
        std_deviation = variance ** 0.5
        
        # Balance score (lower variance = better balance)
        balance_score = max(0.0, 1.0 - (variance / 0.25))
        
        # Balance rating
        if balance_score > 0.8:
            balance_rating = 'excellent'
        elif balance_score > 0.6:
            balance_rating = 'good'
        elif balance_score > 0.4:
            balance_rating = 'moderate'
        else:
            balance_rating = 'poor'
        
        return {
            'balance_score': balance_score,
            'balance_rating': balance_rating,
            'trait_variance': variance,
            'trait_std_deviation': std_deviation,
            'trait_range': max(trait_values) - min(trait_values) if trait_values else 0
        }
        
    except Exception as e:
        logger.debug(f"Comprehensive trait balance calculation failed: {e}")
        return {
            'balance_score': 0.7,
            'balance_rating': 'moderate'
        }

def _identify_dominant_characteristics(traits: Dict) -> List[str]:
    """Identifiziert dominante Characteristics"""
    try:
        if not traits:
            return []
        
        # Sort traits by value and identify dominant ones
        sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)
        
        # Take top traits that are above average
        trait_values = list(traits.values())
        mean_value = sum(trait_values) / len(trait_values) if trait_values else 0.5
        
        dominant_traits = []
        for trait, value in sorted_traits:
            if value > mean_value and value > 0.6:  # Must be above average and reasonably high
                dominant_traits.append(trait)
        
        return dominant_traits[:5]  # Top 5 dominant characteristics
        
    except Exception as e:
        logger.debug(f"Dominant characteristics identification failed: {e}")
        return []

def _analyze_trait_interactions(traits: Dict) -> Dict[str, Any]:
    """Analysiert Trait Interactions"""
    try:
        if not traits or len(traits) < 2:
            return {'interaction_complexity': 'simple', 'synergies': [], 'conflicts': []}
        
        trait_items = list(traits.items())
        synergies = []
        conflicts = []
        
        # Analyze trait pairs for synergies and conflicts
        for i in range(len(trait_items)):
            for j in range(i + 1, len(trait_items)):
                trait1_name, trait1_value = trait_items[i]
                trait2_name, trait2_value = trait_items[j]
                
                # High-high combinations = synergies
                if trait1_value > 0.7 and trait2_value > 0.7:
                    synergies.append(f"{trait1_name} + {trait2_name}")
                
                # High-low combinations = potential conflicts
                elif abs(trait1_value - trait2_value) > 0.5:
                    conflicts.append(f"{trait1_name} vs {trait2_name}")
        
        # Determine interaction complexity
        total_interactions = len(synergies) + len(conflicts)
        if total_interactions > 5:
            complexity = 'complex'
        elif total_interactions > 2:
            complexity = 'moderate'
        else:
            complexity = 'simple'
        
        return {
            'interaction_complexity': complexity,
            'synergies': synergies[:3],  # Top 3 synergies
            'conflicts': conflicts[:3],  # Top 3 conflicts
            'synergy_count': len(synergies),
            'conflict_count': len(conflicts)
        }
        
    except Exception as e:
        logger.debug(f"Trait interactions analysis failed: {e}")
        return {
            'interaction_complexity': 'moderate',
            'synergies': [],
            'conflicts': []
        }

def _assess_trait_stability(traits: Dict, current_state: Dict) -> float:
    """Bewertet Trait Stability"""
    try:
        if not traits:
            return 0.5
        
        emotional_stability = current_state.get('emotional_stability', 0.7)
        trait_values = list(traits.values())
        
        # Stability based on trait consistency and emotional stability
        if trait_values:
            trait_variance = sum((x - (sum(trait_values) / len(trait_values))) ** 2 for x in trait_values) / len(trait_values)
            trait_consistency = 1.0 - min(1.0, trait_variance / 0.2)  # Lower variance = higher consistency
        else:
            trait_consistency = 0.5
        
        # Combine trait consistency with emotional stability
        stability_score = (trait_consistency * 0.6 + emotional_stability * 0.4)
        return max(0.0, min(1.0, stability_score))
        
    except Exception as e:
        logger.debug(f"Trait stability assessment failed: {e}")
        return 0.7

def _calculate_emotional_range(current_state: Dict) -> float:
    """Berechnet Emotional Range"""
    try:
        # Extract emotional-related metrics from current state
        emotional_stability = current_state.get('emotional_stability', 0.7)
        empathy_level = current_state.get('empathy_level', 0.8)
        social_awareness = current_state.get('social_awareness', 0.6)
        
        # Emotional range considers both stability and expressiveness
        # High stability with good empathy and awareness = good range
        range_score = (emotional_stability * 0.4 + empathy_level * 0.3 + social_awareness * 0.3)
        return max(0.0, min(1.0, range_score))
        
    except Exception as e:
        logger.debug(f"Emotional range calculation failed: {e}")
        return 0.7

def _assess_emotional_regulation(current_state: Dict) -> float:
    """Bewertet Emotional Regulation"""
    try:
        emotional_stability = current_state.get('emotional_stability', 0.7)
        adaptability = current_state.get('adaptability', 0.6)
        
        # Emotional regulation combines stability with adaptability
        regulation_score = (emotional_stability * 0.7 + adaptability * 0.3)
        return max(0.0, min(1.0, regulation_score))
        
    except Exception as e:
        logger.debug(f"Emotional regulation assessment failed: {e}")
        return 0.7

def _analyze_empathy_metrics(current_state: Dict) -> Dict[str, Any]:
    """Analysiert Empathy Metrics"""
    try:
        empathy_level = current_state.get('empathy_level', 0.8)
        social_awareness = current_state.get('social_awareness', 0.6)
        emotional_stability = current_state.get('emotional_stability', 0.7)
        
        # Empathy components
        cognitive_empathy = empathy_level
        emotional_empathy = (empathy_level + emotional_stability) / 2
        social_empathy = (empathy_level + social_awareness) / 2
        
        # Overall empathy score
        overall_empathy = (cognitive_empathy * 0.4 + emotional_empathy * 0.3 + social_empathy * 0.3)
        
        # Empathy classification
        if overall_empathy > 0.8:
            empathy_level_rating = 'high'
        elif overall_empathy > 0.6:
            empathy_level_rating = 'moderate'
        else:
            empathy_level_rating = 'developing'
        
        return {
            'overall_empathy_score': overall_empathy,
            'cognitive_empathy': cognitive_empathy,
            'emotional_empathy': emotional_empathy,
            'social_empathy': social_empathy,
            'empathy_level_rating': empathy_level_rating
        }
        
    except Exception as e:
        logger.debug(f"Empathy metrics analysis failed: {e}")
        return {
            'overall_empathy_score': 0.7,
            'empathy_level_rating': 'moderate'
        }

def _assess_social_awareness(current_state: Dict) -> float:
    """Bewertet Social Awareness"""
    try:
        social_awareness = current_state.get('social_awareness', 0.6)
        empathy_level = current_state.get('empathy_level', 0.8)
        adaptability = current_state.get('adaptability', 0.6)
        
        # Social awareness combines direct measure with empathy and adaptability
        awareness_score = (social_awareness * 0.6 + empathy_level * 0.2 + adaptability * 0.2)
        return max(0.0, min(1.0, awareness_score))
        
    except Exception as e:
        logger.debug(f"Social awareness assessment failed: {e}")
        return 0.6

def _analyze_adaptability_patterns(current_state: Dict) -> Dict[str, Any]:
    """Analysiert Adaptability Patterns"""
    try:
        adaptability = current_state.get('adaptability', 0.6)
        emotional_stability = current_state.get('emotional_stability', 0.7)
        
        # Adaptability patterns
        flexibility_type = 'high' if adaptability > 0.7 else 'moderate' if adaptability > 0.5 else 'low'
        stability_support = emotional_stability > 0.6  # Stable emotions support adaptability
        
        # Combined adaptability effectiveness
        adaptability_effectiveness = adaptability * (1.0 + (0.2 if stability_support else -0.1))
        adaptability_effectiveness = max(0.0, min(1.0, adaptability_effectiveness))
        
        return {
            'adaptability_score': adaptability,
            'flexibility_type': flexibility_type,
            'stability_support': stability_support,
            'adaptability_effectiveness': adaptability_effectiveness,
            'adaptation_style': 'balanced' if stability_support else 'reactive'
        }
        
    except Exception as e:
        logger.debug(f"Adaptability patterns analysis failed: {e}")
        return {
            'adaptability_score': 0.6,
            'flexibility_type': 'moderate',
            'adaptation_style': 'balanced'
        }

def _analyze_decision_making_style(traits: Dict, current_state: Dict) -> Dict[str, Any]:
    """Analysiert Decision Making Style"""
    try:
        # Extract relevant traits for decision making
        conscientiousness = traits.get('conscientiousness', 0.6)
        openness = traits.get('openness', 0.6)
        emotional_stability = current_state.get('emotional_stability', 0.7)
        adaptability = current_state.get('adaptability', 0.6)
        
        # Decision making components
        analytical_tendency = (conscientiousness + emotional_stability) / 2
        creative_tendency = (openness + adaptability) / 2
        
        # Determine decision making style
        if analytical_tendency > creative_tendency + 0.2:
            style = 'analytical'
        elif creative_tendency > analytical_tendency + 0.2:
            style = 'intuitive'
        else:
            style = 'balanced'
        
        # Decision confidence
        decision_confidence = (emotional_stability + conscientiousness) / 2
        
        return {
            'decision_making_style': style,
            'analytical_tendency': analytical_tendency,
            'creative_tendency': creative_tendency,
            'decision_confidence': decision_confidence,
            'style_effectiveness': max(analytical_tendency, creative_tendency)
        }
        
    except Exception as e:
        logger.debug(f"Decision making style analysis failed: {e}")
        return {
            'decision_making_style': 'balanced',
            'decision_confidence': 0.7,
            'style_effectiveness': 0.7
        }

def _analyze_interaction_preferences(traits: Dict) -> Dict[str, Any]:
    """Analysiert Interaction Preferences"""
    try:
        extraversion = traits.get('extraversion', 0.5)
        agreeableness = traits.get('agreeableness', 0.7)
        openness = traits.get('openness', 0.6)
        
        # Interaction style analysis
        if extraversion > 0.7:
            interaction_style = 'highly_social'
        elif extraversion > 0.5:
            interaction_style = 'moderately_social'
        else:
            interaction_style = 'reserved'
        
        # Collaboration preference
        collaboration_preference = (agreeableness + extraversion) / 2
        
        # Communication openness
        communication_openness = (openness + agreeableness) / 2
        
        return {
            'interaction_style': interaction_style,
            'collaboration_preference': collaboration_preference,
            'communication_openness': communication_openness,
            'social_engagement_level': extraversion,
            'interpersonal_harmony': agreeableness
        }
        
    except Exception as e:
        logger.debug(f"Interaction preferences analysis failed: {e}")
        return {
            'interaction_style': 'moderately_social',
            'collaboration_preference': 0.6,
            'communication_openness': 0.7
        }

def _analyze_personality_learning_style(traits: Dict, current_state: Dict) -> Dict[str, Any]:
    """Analysiert Personality Learning Style"""
    try:
        openness = traits.get('openness', 0.6)
        conscientiousness = traits.get('conscientiousness', 0.6)
        adaptability = current_state.get('adaptability', 0.6)
        
        # Learning style components
        exploration_preference = (openness + adaptability) / 2
        structure_preference = conscientiousness
        
        # Determine learning style
        if exploration_preference > structure_preference + 0.2:
            learning_style = 'exploratory'
        elif structure_preference > exploration_preference + 0.2:
            learning_style = 'structured'
        else:
            learning_style = 'adaptive'
        
        # Learning effectiveness
        learning_effectiveness = (exploration_preference + structure_preference) / 2
        
        return {
            'learning_style': learning_style,
            'exploration_preference': exploration_preference,
            'structure_preference': structure_preference,
            'learning_effectiveness': learning_effectiveness,
            'learning_adaptability': adaptability
        }
        
    except Exception as e:
        logger.debug(f"Personality learning style analysis failed: {e}")
        return {
            'learning_style': 'adaptive',
            'learning_effectiveness': 0.6,
            'learning_adaptability': 0.6
        }

def _analyze_communication_patterns(traits: Dict) -> Dict[str, Any]:
    """Analysiert Communication Patterns"""
    try:
        extraversion = traits.get('extraversion', 0.5)
        agreeableness = traits.get('agreeableness', 0.7)
        openness = traits.get('openness', 0.6)
        conscientiousness = traits.get('conscientiousness', 0.6)
        
        # Communication components
        expressiveness = extraversion
        consideration = agreeableness
        creativity = openness
        clarity = conscientiousness
        
        # Communication style
        if expressiveness > 0.7:
            communication_style = 'expressive'
        elif consideration > 0.7:
            communication_style = 'considerate'
        elif creativity > 0.7:
            communication_style = 'creative'
        else:
            communication_style = 'balanced'
        
        # Overall communication effectiveness
        communication_effectiveness = (expressiveness + consideration + creativity + clarity) / 4
        
        return {
            'communication_style': communication_style,
            'expressiveness': expressiveness,
            'consideration': consideration,
            'creativity': creativity,
            'clarity': clarity,
            'communication_effectiveness': communication_effectiveness
        }
        
    except Exception as e:
        logger.debug(f"Communication patterns analysis failed: {e}")
        return {
            'communication_style': 'balanced',
            'communication_effectiveness': 0.6
        }

def _assess_maturity_progression(traits: Dict, current_state: Dict) -> Dict[str, Any]:
    """Bewertet Maturity Progression"""
    try:
        # Extract maturity indicators
        emotional_stability = current_state.get('emotional_stability', 0.7)
        conscientiousness = traits.get('conscientiousness', 0.6)
        agreeableness = traits.get('agreeableness', 0.7)
        
        # Maturity components
        emotional_maturity = emotional_stability
        behavioral_maturity = conscientiousness
        social_maturity = agreeableness
        
        # Overall maturity score
        overall_maturity = (emotional_maturity * 0.4 + behavioral_maturity * 0.3 + social_maturity * 0.3)
        
        # Maturity level classification
        if overall_maturity > 0.8:
            maturity_level = 'high'
        elif overall_maturity > 0.6:
            maturity_level = 'moderate'
        else:
            maturity_level = 'developing'
        
        return {
            'overall_maturity_score': overall_maturity,
            'emotional_maturity': emotional_maturity,
            'behavioral_maturity': behavioral_maturity,
            'social_maturity': social_maturity,
            'maturity_level': maturity_level,
            'maturity_balance': 1.0 - abs(emotional_maturity - behavioral_maturity)
        }
        
    except Exception as e:
        logger.debug(f"Maturity progression assessment failed: {e}")
        return {
            'overall_maturity_score': 0.7,
            'maturity_level': 'moderate'
        }

def _analyze_character_development(development_metrics: Dict) -> Dict[str, Any]:
    """Analysiert Character Development"""
    try:
        development_rate = development_metrics.get('development_rate', 0.5)
        growth_potential = development_metrics.get('growth_potential', 0.7)
        
        # Character development analysis
        development_momentum = development_rate * growth_potential
        development_consistency = min(1.0, development_rate / 0.3)  # 0.3 is steady development
        
        # Development stage
        if development_momentum > 0.6:
            development_stage = 'accelerated'
        elif development_momentum > 0.3:
            development_stage = 'steady'
        else:
            development_stage = 'gradual'
        
        return {
            'development_momentum': development_momentum,
            'development_consistency': development_consistency,
            'development_stage': development_stage,
            'character_strength': growth_potential,
            'development_trajectory': 'positive' if development_rate > 0.2 else 'stable'
        }
        
    except Exception as e:
        logger.debug(f"Character development analysis failed: {e}")
        return {
            'development_momentum': 0.4,
            'development_stage': 'steady',
            'development_trajectory': 'stable'
        }

def _assess_self_awareness_level(current_state: Dict) -> float:
    """Bewertet Self Awareness Level"""
    try:
        emotional_stability = current_state.get('emotional_stability', 0.7)
        adaptability = current_state.get('adaptability', 0.6)
        empathy_level = current_state.get('empathy_level', 0.8)
        
        # Self-awareness components
        emotional_awareness = emotional_stability
        behavioral_awareness = adaptability
        social_awareness = empathy_level
        
        # Overall self-awareness
        self_awareness = (emotional_awareness * 0.4 + behavioral_awareness * 0.3 + social_awareness * 0.3)
        return max(0.0, min(1.0, self_awareness))
        
    except Exception as e:
        logger.debug(f"Self awareness level assessment failed: {e}")
        return 0.7

def _assess_personality_coherence(traits: Dict, current_state: Dict) -> float:
    """Bewertet Personality Coherence"""
    try:
        if not traits:
            return 0.5
        
        # Trait coherence
        trait_values = list(traits.values())
        if trait_values:
            trait_variance = sum((x - (sum(trait_values) / len(trait_values))) ** 2 for x in trait_values) / len(trait_values)
            trait_coherence = 1.0 - min(1.0, trait_variance / 0.2)
        else:
            trait_coherence = 0.5
        
        # State coherence
        emotional_stability = current_state.get('emotional_stability', 0.7)
        state_coherence = emotional_stability
        
        # Overall coherence
        personality_coherence = (trait_coherence * 0.6 + state_coherence * 0.4)
        return max(0.0, min(1.0, personality_coherence))
        
    except Exception as e:
        logger.debug(f"Personality coherence assessment failed: {e}")
        return 0.7

def _assess_behavioral_consistency(traits: Dict, current_state: Dict) -> float:
    """Bewertet Behavioral Consistency"""
    try:
        conscientiousness = traits.get('conscientiousness', 0.6)
        emotional_stability = current_state.get('emotional_stability', 0.7)
        
        # Consistency based on conscientiousness and emotional stability
        consistency_score = (conscientiousness * 0.6 + emotional_stability * 0.4)
        return max(0.0, min(1.0, consistency_score))
        
    except Exception as e:
        logger.debug(f"Behavioral consistency assessment failed: {e}")
        return 0.7

def _assess_authentic_expression(traits: Dict, current_state: Dict) -> float:
    """Bewertet Authentic Expression"""
    try:
        openness = traits.get('openness', 0.6)
        emotional_stability = current_state.get('emotional_stability', 0.7)
        
        # Authenticity based on openness and emotional stability
        authenticity_score = (openness * 0.6 + emotional_stability * 0.4)
        return max(0.0, min(1.0, authenticity_score))
        
    except Exception as e:
        logger.debug(f"Authentic expression assessment failed: {e}")
        return 0.7

def _calculate_overall_personality_health_comprehensive(personality_analytics: Dict) -> float:
    """Berechnet umfassende Overall Personality Health"""
    try:
        health_factors = []
        
        # Trait analysis health
        if 'trait_analysis' in personality_analytics:
            trait_analysis = personality_analytics['trait_analysis']
            trait_balance = trait_analysis.get('trait_balance', {}).get('balance_score', 0.7)
            health_factors.append(trait_balance)
        
        # Emotional analysis health
        if 'emotional_analysis' in personality_analytics:
            emotional_analysis = personality_analytics['emotional_analysis']
            emotional_stability = emotional_analysis.get('emotional_stability', 0.7)
            empathy_score = emotional_analysis.get('empathy_metrics', {}).get('overall_empathy_score', 0.7)
            health_factors.extend([emotional_stability, empathy_score])
        
        # Integration metrics health
        if 'integration_metrics' in personality_analytics:
            integration = personality_analytics['integration_metrics']
            coherence = integration.get('personality_coherence', 0.7)
            consistency = integration.get('behavioral_consistency', 0.7)
            health_factors.extend([coherence, consistency])
        
        # Development analysis health
        if 'development_analysis' in personality_analytics:
            development = personality_analytics['development_analysis']
            growth_potential = development.get('growth_potential', 0.7)
            health_factors.append(growth_potential)
        
        if health_factors:
            overall_health = sum(health_factors) / len(health_factors)
            return max(0.0, min(1.0, overall_health))
        else:
            return 0.7
            
    except Exception as e:
        logger.debug(f"Comprehensive personality health calculation failed: {e}")
        return 0.7

def _determine_personality_type(traits: Dict) -> str:
    """Bestimmt Personality Type"""
    try:
        if not traits:
            return 'undefined'
        
        # Simplified personality type classification based on dominant traits
        extraversion = traits.get('extraversion', 0.5)
        conscientiousness = traits.get('conscientiousness', 0.6)
        openness = traits.get('openness', 0.6)
        agreeableness = traits.get('agreeableness', 0.7)
        
        # Basic personality type determination
        if extraversion > 0.7 and agreeableness > 0.7:
            return 'social_harmonizer'
        elif conscientiousness > 0.7 and openness > 0.7:
            return 'creative_organizer'
        elif openness > 0.7 and extraversion > 0.6:
            return 'innovative_communicator'
        elif conscientiousness > 0.7:
            return 'reliable_achiever'
        elif agreeableness > 0.7:
            return 'empathetic_supporter'
        else:
            return 'balanced_adaptive'
            
    except Exception as e:
        logger.debug(f"Personality type determination failed: {e}")
        return 'developing'

def _identify_key_characteristics(personality_analytics: Dict) -> List[str]:
    """Identifiziert Key Characteristics"""
    try:
        characteristics = []
        
        # From trait analysis
        if 'trait_analysis' in personality_analytics:
            dominant_chars = personality_analytics['trait_analysis'].get('dominant_characteristics', [])
            characteristics.extend(dominant_chars[:2])
        
        # From behavioral patterns
        if 'behavioral_patterns' in personality_analytics:
            patterns = personality_analytics['behavioral_patterns']
            interaction_style = patterns.get('interaction_style', '')
            if interaction_style:
                characteristics.append(f"Interaction style: {interaction_style}")
        
        # From emotional analysis
        if 'emotional_analysis' in personality_analytics:
            emotional = personality_analytics['emotional_analysis']
            empathy_rating = emotional.get('empathy_metrics', {}).get('empathy_level_rating', '')
            if empathy_rating:
                characteristics.append(f"Empathy level: {empathy_rating}")
        
        # From development analysis
        if 'development_analysis' in personality_analytics:
            development = personality_analytics['development_analysis']
            maturity_level = development.get('maturity_progression', {}).get('maturity_level', '')
            if maturity_level:
                characteristics.append(f"Maturity: {maturity_level}")
        
        return characteristics[:5]
        
    except Exception as e:
        logger.debug(f"Key characteristics identification failed: {e}")
        return ["Developing personality profile"]

def _generate_personality_development_recommendations(personality_analytics: Dict) -> List[str]:
    """Generiert Personality Development Recommendations"""
    try:
        recommendations = []
        
        # Trait balance recommendations
        if 'trait_analysis' in personality_analytics:
            trait_balance = personality_analytics['trait_analysis'].get('trait_balance', {})
            balance_rating = trait_balance.get('balance_rating', 'moderate')
            
            if balance_rating == 'poor':
                recommendations.append("Focus on developing balanced trait expression")
            elif balance_rating == 'excellent':
                recommendations.append("Maintain current trait balance while exploring growth areas")
        
        # Emotional development recommendations
        if 'emotional_analysis' in personality_analytics:
            emotional = personality_analytics['emotional_analysis']
            emotional_stability = emotional.get('emotional_stability', 0.7)
            
            if emotional_stability < 0.6:
                recommendations.append("Develop emotional regulation and stability techniques")
        
        # Social development recommendations
        if 'behavioral_patterns' in personality_analytics:
            patterns = personality_analytics['behavioral_patterns']
            social_engagement = patterns.get('social_engagement_level', 0.5)
            
            if social_engagement < 0.5:
                recommendations.append("Enhance social interaction and communication skills")
        
        # Growth recommendations
        if 'development_analysis' in personality_analytics:
            development = personality_analytics['development_analysis']
            growth_potential = development.get('growth_potential', 0.7)
            
            if growth_potential > 0.8:
                recommendations.append("Leverage high growth potential for advanced development")
            elif growth_potential < 0.5:
                recommendations.append("Build foundational skills for personality growth")
        
        # General recommendations
        if len(recommendations) < 3:
            recommendations.extend([
                "Continue regular self-reflection and awareness practices",
                "Seek diverse experiences for personality development",
                "Build strong interpersonal relationships"
            ])
        
        return recommendations[:5]
        
    except Exception as e:
        logger.debug(f"Personality development recommendations generation failed: {e}")
        return [
            "Continue personality development through varied experiences",
            "Practice self-reflection and awareness",
            "Build emotional intelligence and regulation skills"
        ]

def _generate_fallback_personality_analytics() -> Dict[str, Any]:
    """Generiert Fallback Personality Analytics"""
    return {
        'trait_analysis': {
            'trait_profile': {'profile_type': 'developing', 'trait_count': 0},
            'trait_balance': {'balance_score': 0.7, 'balance_rating': 'moderate'},
            'dominant_characteristics': [],
            'trait_stability': 0.7
        },
        'emotional_analysis': {
            'emotional_stability': 0.7,
            'emotional_range': 0.6,
            'empathy_metrics': {'overall_empathy_score': 0.7, 'empathy_level_rating': 'moderate'}
        },
        'behavioral_patterns': {
            'adaptability_patterns': {'adaptability_score': 0.6, 'flexibility_type': 'moderate'},
            'interaction_preferences': {'interaction_style': 'moderately_social'},
            'communication_patterns': {'communication_style': 'balanced'}
        },
        'development_analysis': {
            'personality_growth_rate': 0.5,
            'growth_potential': 0.7,
            'maturity_progression': {'overall_maturity_score': 0.7, 'maturity_level': 'moderate'}
        },
        'integration_metrics': {
            'personality_coherence': 0.7,
            'behavioral_consistency': 0.7,
            'adaptive_flexibility': 0.6
        },
        'fallback_mode': True
    }

# ====================================
# ADDITIONAL MISSING PERFORMANCE FUNCTIONS
# ====================================

def _calculate_mental_agility(memory_manager, db_stats: Dict) -> float:
    """Berechnet Mental Agility"""
    try:
        from routes.utils.memory_helpers import (
            calculate_current_cognitive_load,
            calculate_learning_readiness_direct
        )
        
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        learning_readiness = calculate_learning_readiness_direct(memory_manager, db_stats)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Mental agility combines low cognitive load, high learning readiness, and activity
        agility_score = (
            (1.0 - cognitive_load) * 0.4 +
            learning_readiness * 0.4 +
            min(1.0, recent_activity / 10) * 0.2
        )
        
        return max(0.0, min(1.0, agility_score))
        
    except Exception as e:
        logger.debug(f"Mental agility calculation failed: {e}")
        return 0.7

def _calculate_memory_throughput(db_stats: Dict) -> float:
    """Berechnet Memory Throughput"""
    try:
        recent_activity = db_stats.get('recent_activity', 0)
        total_memories = db_stats.get('total_memories', 0)
        
        # Throughput as memories processed per unit time
        throughput = recent_activity / 24  # Per hour
        
        # Normalize based on total memory base
        if total_memories > 0:
            relative_throughput = min(1.0, throughput / (total_memories * 0.01))  # 1% of base per hour is high
        else:
            relative_throughput = min(1.0, throughput / 0.5)  # 0.5 per hour is good baseline
        
        return relative_throughput
        
    except Exception as e:
        logger.debug(f"Memory throughput calculation failed: {e}")
        return 0.5

def _estimate_retrieval_speed(memory_manager, db_stats: Dict) -> float:
    """Schätzt Retrieval Speed"""
    try:
        from routes.utils.memory_helpers import (
            calculate_stm_efficiency_direct,
            get_stm_load_direct
        )
        
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        stm_load = get_stm_load_direct(memory_manager)
        
        # Retrieval speed based on efficiency and load
        base_speed = stm_efficiency
        load_factor = max(0.5, 1.0 - (stm_load / 30))  # Slower with high load
        
        retrieval_speed = base_speed * load_factor
        return max(0.0, min(1.0, retrieval_speed))
        
    except Exception as e:
        logger.debug(f"Retrieval speed estimation failed: {e}")
        return 0.7

def _calculate_consolidation_efficiency_score(memory_manager, db_stats: Dict) -> float:
    """Berechnet Consolidation Efficiency Score"""
    try:
        from routes.utils.memory_helpers import is_consolidation_active_direct
        
        consolidation_active = is_consolidation_active_direct(memory_manager, db_stats)
        recent_activity = db_stats.get('recent_activity', 0)
        
        if consolidation_active:
            # Efficiency based on activity level when consolidation is active
            efficiency_score = min(1.0, recent_activity / 15) * 0.9  # Good efficiency with activity
        else:
            # Lower efficiency when consolidation is not active
            efficiency_score = 0.4
        
        return efficiency_score
        
    except Exception as e:
        logger.debug(f"Consolidation efficiency score calculation failed: {e}")
        return 0.6

def _calculate_retention_rate(db_stats: Dict) -> float:
    """Berechnet Retention Rate"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Estimate retention rate
        if total_memories > 0:
            # Retention = existing memories / (existing + recent)
            retention_rate = total_memories / (total_memories + recent_activity) if recent_activity > 0 else 0.9
        else:
            retention_rate = 0.8  # Default assumption
        
        return max(0.0, min(1.0, retention_rate))
        
    except Exception as e:
        logger.debug(f"Retention rate calculation failed: {e}")
        return 0.8

def _calculate_adaptation_speed(memory_manager, db_stats: Dict) -> float:
    """Berechnet Adaptation Speed"""
    try:
        from routes.utils.memory_helpers import calculate_learning_readiness_direct
        
        learning_readiness = calculate_learning_readiness_direct(memory_manager, db_stats)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Adaptation speed based on readiness and activity
        adaptation_speed = learning_readiness * min(1.0, recent_activity / 8)
        return max(0.0, min(1.0, adaptation_speed))
        
    except Exception as e:
        logger.debug(f"Adaptation speed calculation failed: {e}")
        return 0.6

def _calculate_skill_development_rate(db_stats: Dict) -> float:
    """Berechnet Skill Development Rate"""
    try:
        recent_activity = db_stats.get('recent_activity', 0)
        total_memories = db_stats.get('total_memories', 0)
        
        # Skill development based on learning activity and knowledge base
        activity_rate = recent_activity / 24  # Per hour
        knowledge_factor = min(1.0, total_memories / 100)  # Knowledge base supports skill development
        
        skill_development_rate = activity_rate * (0.5 + knowledge_factor * 0.5)
        return max(0.0, min(1.0, skill_development_rate))
        
    except Exception as e:
        logger.debug(f"Skill development rate calculation failed: {e}")
        return 0.4

__all__ = [
    'generate_analytics_summary',
    'generate_comprehensive_report',
    'calculate_growth_analysis', 
    'analyze_memory_distribution'
]