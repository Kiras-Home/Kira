"""
Personality Traits Module
Trait Analysis, Development und Growth Management
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def analyze_personality_traits(personality_data: Dict = None, 
                             memory_manager=None,
                             analysis_depth: str = 'comprehensive') -> Dict[str, Any]:
    """
    Umfassende Personality Traits Analyse
    
    Extrahiert aus kira_routes.py.backup Trait Analysis Logic
    """
    try:
        if not personality_data and memory_manager:
            # Versuche Personality Data aus Memory Manager zu extrahieren
            personality_data = _extract_personality_from_memory_manager(memory_manager)
        
        if not personality_data:
            return _generate_fallback_trait_analysis()
        
        # Core Trait Analysis
        trait_analysis = {
            'trait_overview': _analyze_trait_overview(personality_data),
            'trait_strengths': _identify_trait_strengths(personality_data),
            'trait_development_areas': _identify_development_areas(personality_data),
            'trait_balance_assessment': assess_trait_balance(personality_data),
            'trait_synergies': _analyze_trait_synergies(personality_data),
            'trait_conflicts': _analyze_trait_conflicts(personality_data)
        }
        
        # Detailed Analysis basierend auf depth
        if analysis_depth == 'comprehensive':
            trait_analysis.update({
                'detailed_trait_profiles': _generate_detailed_trait_profiles(personality_data),
                'trait_expression_patterns': _analyze_trait_expression_patterns(personality_data),
                'trait_activation_triggers': _identify_trait_activation_triggers(personality_data),
                'trait_stability_analysis': _analyze_trait_stability(personality_data),
                'trait_growth_predictions': _predict_trait_growth(personality_data)
            })
        
        # Trait Recommendations
        trait_analysis['recommendations'] = generate_trait_recommendations(trait_analysis, personality_data)
        
        # Analysis Metadata
        trait_analysis['analysis_metadata'] = {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_depth': analysis_depth,
            'traits_analyzed': len(personality_data.get('traits', {})),
            'data_quality_score': _assess_trait_data_quality(personality_data)
        }
        
        return trait_analysis
        
    except Exception as e:
        logger.error(f"Personality traits analysis failed: {e}")
        return {
            'error': str(e),
            'fallback_analysis': _generate_fallback_trait_analysis()
        }

def calculate_trait_development(personality_data: Dict, 
                              timeframe: str = '30d') -> Dict[str, Any]:
    """
    Berechnet Trait Development über Zeit
    
    Basiert auf kira_routes.py.backup Development Calculations
    """
    try:
        traits = personality_data.get('traits', {})
        development_history = personality_data.get('development_history', [])
        
        if not traits:
            return {'available': False, 'reason': 'no_traits_data'}
        
        # Development Analysis für jeden Trait
        trait_developments = {}
        for trait_name, trait_data in traits.items():
            if isinstance(trait_data, dict):
                trait_developments[trait_name] = _calculate_individual_trait_development(
                    trait_name, trait_data, development_history, timeframe
                )
        
        # Overall Development Metrics
        development_summary = {
            'total_traits_analyzed': len(trait_developments),
            'traits_showing_growth': len([t for t in trait_developments.values() if t.get('growth_rate', 0) > 0]),
            'fastest_growing_trait': _identify_fastest_growing_trait(trait_developments),
            'most_stable_trait': _identify_most_stable_trait(trait_developments),
            'average_development_rate': _calculate_average_development_rate(trait_developments),
            'development_momentum': _calculate_development_momentum(trait_developments)
        }
        
        return {
            'available': True,
            'timeframe': timeframe,
            'trait_developments': trait_developments,
            'development_summary': development_summary,
            'development_insights': _generate_development_insights(trait_developments, development_summary),
            'future_projections': _project_future_trait_development(trait_developments)
        }
        
    except Exception as e:
        logger.error(f"Trait development calculation failed: {e}")
        return {
            'available': False,
            'error': str(e)
        }

def generate_trait_recommendations(trait_analysis: Dict, 
                                 personality_data: Dict) -> Dict[str, Any]:
    """
    Generiert Trait-basierte Empfehlungen
    
    Extrahiert aus kira_routes.py.backup Recommendation Logic
    """
    try:
        recommendations = {
            'trait_development_recommendations': _generate_trait_development_recommendations(trait_analysis, personality_data),
            'balance_optimization_recommendations': _generate_balance_recommendations(trait_analysis),
            'expression_enhancement_recommendations': _generate_expression_recommendations(trait_analysis),
            'synergy_maximization_recommendations': _generate_synergy_recommendations(trait_analysis),
            'growth_acceleration_recommendations': _generate_growth_recommendations(trait_analysis)
        }
        
        # Prioritize Recommendations
        recommendations['prioritized_actions'] = _prioritize_trait_recommendations(recommendations)
        
        # Implementation Guidance
        recommendations['implementation_guidance'] = {
            'immediate_actions': _identify_immediate_trait_actions(recommendations),
            'short_term_goals': _identify_short_term_trait_goals(recommendations),
            'long_term_objectives': _identify_long_term_trait_objectives(recommendations),
            'success_metrics': _define_trait_success_metrics(recommendations)
        }
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Trait recommendations generation failed: {e}")
        return {
            'error': str(e),
            'fallback_recommendations': _generate_fallback_trait_recommendations()
        }

def assess_trait_balance(personality_data: Dict) -> Dict[str, Any]:
    """
    Bewertet Trait Balance und Harmonie
    
    Basiert auf kira_routes.py.backup Balance Assessment Logic
    """
    try:
        traits = personality_data.get('traits', {})
        
        if not traits:
            return {'available': False, 'reason': 'no_traits_data'}
        
        # Balance Metrics
        balance_assessment = {
            'overall_balance_score': _calculate_overall_trait_balance(traits),
            'category_balance': _assess_trait_category_balance(traits),
            'strength_distribution': _analyze_trait_strength_distribution(traits),
            'harmony_score': _calculate_trait_harmony_score(traits),
            'diversity_index': _calculate_trait_diversity_index(traits)
        }
        
        # Balance Analysis
        balance_assessment['balance_analysis'] = {
            'dominant_categories': _identify_dominant_trait_categories(traits),
            'underrepresented_categories': _identify_underrepresented_categories(traits),
            'balance_outliers': _identify_balance_outliers(traits),
            'optimization_opportunities': _identify_balance_optimization_opportunities(balance_assessment)
        }
        
        # Balance Recommendations
        balance_assessment['balance_recommendations'] = {
            'rebalancing_strategies': _generate_rebalancing_strategies(balance_assessment, traits),
            'focus_areas': _identify_balance_focus_areas(balance_assessment),
            'expected_outcomes': _predict_balance_improvement_outcomes(balance_assessment)
        }
        
        return balance_assessment
        
    except Exception as e:
        logger.error(f"Trait balance assessment failed: {e}")
        return {
            'available': False,
            'error': str(e)
        }

# ====================================
# PRIVATE HELPER FUNCTIONS
# ====================================

def _extract_personality_from_memory_manager(memory_manager) -> Optional[Dict]:
    """Extrahiert Personality Data aus Memory Manager"""
    try:
        if hasattr(memory_manager, 'kira_personality'):
            personality_system = memory_manager.kira_personality
            
            # Extract core personality data
            personality_data = {
                'traits': {},
                'current_state': {},
                'development_history': [],
                'interaction_history': []
            }
            
            # Extract traits
            if hasattr(personality_system, 'traits'):
                personality_data['traits'] = _convert_traits_to_dict(personality_system.traits)
            
            # Extract current state
            if hasattr(personality_system, 'current_state'):
                personality_data['current_state'] = _convert_state_to_dict(personality_system.current_state)
            
            # Extract histories
            if hasattr(personality_system, 'development_history'):
                personality_data['development_history'] = personality_system.development_history
            
            if hasattr(personality_system, 'interaction_history'):
                personality_data['interaction_history'] = personality_system.interaction_history
            
            return personality_data
            
        return None
        
    except Exception as e:
        logger.debug(f"Personality extraction from memory manager failed: {e}")
        return None

def _analyze_trait_overview(personality_data: Dict) -> Dict[str, Any]:
    """Analysiert Trait Overview"""
    try:
        traits = personality_data.get('traits', {})
        
        if not traits:
            return {'available': False}
        
        # Basic Overview Metrics
        total_traits = len(traits)
        active_traits = len([t for t in traits.values() if isinstance(t, dict) and t.get('current_strength', 0) > 0.3])
        
        # Strength Analysis
        trait_strengths = []
        for trait_name, trait_data in traits.items():
            if isinstance(trait_data, dict):
                strength = trait_data.get('current_strength', 0.5)
                trait_strengths.append((trait_name, strength))
        
        trait_strengths.sort(key=lambda x: x[1], reverse=True)
        
        # Category Distribution
        category_distribution = _calculate_trait_category_distribution(traits)
        
        return {
            'available': True,
            'total_traits': total_traits,
            'active_traits': active_traits,
            'strongest_traits': trait_strengths[:5],  # Top 5
            'weakest_traits': trait_strengths[-3:],   # Bottom 3
            'category_distribution': category_distribution,
            'average_trait_strength': statistics.mean([s[1] for s in trait_strengths]) if trait_strengths else 0.5
        }
        
    except Exception as e:
        logger.debug(f"Trait overview analysis failed: {e}")
        return {'available': False, 'error': str(e)}

def _identify_trait_strengths(personality_data: Dict) -> List[Dict[str, Any]]:
    """Identifiziert Trait Strengths"""
    try:
        traits = personality_data.get('traits', {})
        strengths = []
        
        for trait_name, trait_data in traits.items():
            if isinstance(trait_data, dict):
                current_strength = trait_data.get('current_strength', 0.5)
                base_strength = trait_data.get('base_strength', 0.5)
                
                # Consider a trait a strength if it's above average or showing growth
                if current_strength > 0.7 or (current_strength - base_strength) > 0.1:
                    strengths.append({
                        'trait_name': trait_name,
                        'current_strength': current_strength,
                        'growth': current_strength - base_strength,
                        'strength_type': 'natural' if base_strength > 0.7 else 'developed',
                        'expression_frequency': trait_data.get('expression_count', 0)
                    })
        
        # Sort by strength
        strengths.sort(key=lambda x: x['current_strength'], reverse=True)
        return strengths
        
    except Exception as e:
        logger.debug(f"Trait strengths identification failed: {e}")
        return []

def _identify_development_areas(personality_data: Dict) -> List[Dict[str, Any]]:
    """Identifiziert Development Areas"""
    try:
        traits = personality_data.get('traits', {})
        development_areas = []
        
        for trait_name, trait_data in traits.items():
            if isinstance(trait_data, dict):
                current_strength = trait_data.get('current_strength', 0.5)
                base_strength = trait_data.get('base_strength', 0.5)
                development_rate = trait_data.get('development_rate', 0.01)
                
                # Consider a trait a development area if it's below average or stagnant
                if current_strength < 0.4 or development_rate < 0.005:
                    development_areas.append({
                        'trait_name': trait_name,
                        'current_strength': current_strength,
                        'development_potential': 1.0 - current_strength,
                        'stagnation_risk': development_rate < 0.005,
                        'priority_level': _calculate_development_priority(trait_data),
                        'recommended_focus': _recommend_trait_focus(trait_data)
                    })
        
        # Sort by priority
        development_areas.sort(key=lambda x: x['priority_level'], reverse=True)
        return development_areas
        
    except Exception as e:
        logger.debug(f"Development areas identification failed: {e}")
        return []

def _analyze_trait_synergies(personality_data: Dict) -> Dict[str, Any]:
    """Analysiert Trait Synergies"""
    try:
        traits = personality_data.get('traits', {})
        synergies = {}
        
        trait_names = list(traits.keys())
        for i, trait1_name in enumerate(trait_names):
            for trait2_name in trait_names[i+1:]:
                trait1_data = traits[trait1_name]
                trait2_data = traits[trait2_name]
                
                if isinstance(trait1_data, dict) and isinstance(trait2_data, dict):
                    synergy_score = _calculate_trait_synergy_score(trait1_data, trait2_data)
                    
                    if synergy_score > 0.6:  # Significant synergy
                        synergy_key = f"{trait1_name}_{trait2_name}"
                        synergies[synergy_key] = {
                            'trait1': trait1_name,
                            'trait2': trait2_name,
                            'synergy_score': synergy_score,
                            'synergy_type': _classify_synergy_type(trait1_data, trait2_data),
                            'enhancement_potential': _calculate_enhancement_potential(trait1_data, trait2_data)
                        }
        
        return {
            'available': True,
            'synergy_count': len(synergies),
            'synergies': synergies,
            'strongest_synergies': sorted(synergies.items(), key=lambda x: x[1]['synergy_score'], reverse=True)[:3],
            'synergy_optimization_opportunities': _identify_synergy_optimization_opportunities(synergies)
        }
        
    except Exception as e:
        logger.debug(f"Trait synergies analysis failed: {e}")
        return {'available': False, 'error': str(e)}

def _analyze_trait_conflicts(personality_data: Dict) -> Dict[str, Any]:
    """Analysiert Trait Conflicts"""
    try:
        traits = personality_data.get('traits', {})
        conflicts = {}
        
        trait_names = list(traits.keys())
        for i, trait1_name in enumerate(trait_names):
            for trait2_name in trait_names[i+1:]:
                trait1_data = traits[trait1_name]
                trait2_data = traits[trait2_name]
                
                if isinstance(trait1_data, dict) and isinstance(trait2_data, dict):
                    conflict_score = _calculate_trait_conflict_score(trait1_data, trait2_data)
                    
                    if conflict_score > 0.6:  # Significant conflict
                        conflict_key = f"{trait1_name}_{trait2_name}"
                        conflicts[conflict_key] = {
                            'trait1': trait1_name,
                            'trait2': trait2_name,
                            'conflict_score': conflict_score,
                            'conflict_type': _classify_conflict_type(trait1_data, trait2_data),
                            'resolution_strategies': _suggest_conflict_resolution_strategies(trait1_data, trait2_data)
                        }
        
        return {
            'available': True,
            'conflict_count': len(conflicts),
            'conflicts': conflicts,
            'most_problematic_conflicts': sorted(conflicts.items(), key=lambda x: x[1]['conflict_score'], reverse=True)[:3],
            'conflict_resolution_priorities': _prioritize_conflict_resolution(conflicts)
        }
        
    except Exception as e:
        logger.debug(f"Trait conflicts analysis failed: {e}")
        return {'available': False, 'error': str(e)}

def _calculate_individual_trait_development(trait_name: str, trait_data: Dict, 
                                          development_history: List, timeframe: str) -> Dict[str, Any]:
    """Berechnet Individual Trait Development"""
    try:
        current_strength = trait_data.get('current_strength', 0.5)
        base_strength = trait_data.get('base_strength', 0.5)
        development_rate = trait_data.get('development_rate', 0.01)
        
        # Calculate development metrics
        growth_rate = current_strength - base_strength
        relative_growth = growth_rate / max(base_strength, 0.1)  # Avoid division by zero
        
        # Analyze development history for this trait
        trait_history = [h for h in development_history if h.get('trait_name') == trait_name]
        
        return {
            'trait_name': trait_name,
            'current_strength': current_strength,
            'base_strength': base_strength,
            'growth_rate': growth_rate,
            'relative_growth': relative_growth,
            'development_rate': development_rate,
            'development_velocity': _calculate_trait_development_velocity(trait_history),
            'development_consistency': _calculate_trait_development_consistency(trait_history),
            'future_projection': _project_trait_future_strength(current_strength, development_rate, 30)
        }
        
    except Exception as e:
        logger.debug(f"Individual trait development calculation failed for {trait_name}: {e}")
        return {
            'trait_name': trait_name,
            'error': str(e),
            'fallback_data': True
        }

# Additional helper functions...

def _calculate_trait_synergy_score(trait1_data: Dict, trait2_data: Dict) -> float:
    """Berechnet Synergy Score zwischen zwei Traits"""
    try:
        # Extract trait categories and characteristics
        trait1_category = trait1_data.get('category', 'unknown')
        trait2_category = trait2_data.get('category', 'unknown')
        
        trait1_strength = trait1_data.get('current_strength', 0.5)
        trait2_strength = trait2_data.get('current_strength', 0.5)
        
        # Check for explicit synergies in trait data
        trait1_synergies = trait1_data.get('synergies', [])
        trait2_synergies = trait2_data.get('synergies', [])
        
        synergy_score = 0.0
        
        # Direct synergy references
        if trait2_data.get('name', '') in trait1_synergies:
            synergy_score += 0.8
        if trait1_data.get('name', '') in trait2_synergies:
            synergy_score += 0.8
        
        # Category-based synergies
        complementary_categories = {
            'social': ['empathy', 'communication'],
            'cognitive': ['creativity', 'analytical'],
            'emotional': ['stability', 'expression']
        }
        
        if trait1_category in complementary_categories:
            if trait2_category in complementary_categories[trait1_category]:
                synergy_score += 0.6
        
        # Strength-based synergy (both strong traits work well together)
        if trait1_strength > 0.7 and trait2_strength > 0.7:
            synergy_score += 0.4
        
        return min(1.0, synergy_score)
        
    except Exception as e:
        logger.debug(f"Trait synergy score calculation failed: {e}")
        return 0.3  # Default neutral synergy

def _generate_fallback_trait_analysis() -> Dict[str, Any]:
    """Generiert Fallback Trait Analysis"""
    return {
        'fallback_mode': True,
        'basic_analysis': {
            'status': 'limited_data_available',
            'traits_detected': 0,
            'analysis_confidence': 'low'
        },
        'recommendations': {
            'primary_recommendation': 'Increase personality system integration for detailed analysis',
            'data_collection_suggestions': [
                'Enable personality tracking',
                'Increase interaction diversity',
                'Allow longer observation period'
            ]
        },
        'timestamp': datetime.now().isoformat()
    }

__all__ = [
    'analyze_personality_traits',
    'calculate_trait_development',
    'generate_trait_recommendations',
    'assess_trait_balance'
]