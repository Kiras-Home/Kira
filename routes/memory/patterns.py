"""
Memory Patterns Module
Memory Pattern Detection, Learning Patterns, Behavioral Analysis und Trend Identification
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import json
import re

logger = logging.getLogger(__name__)

def detect_learning_patterns(memory_manager=None,
                           pattern_scope: str = 'comprehensive',
                           time_window: Optional[timedelta] = None) -> Dict[str, Any]:
    """
    Erkennt Learning Patterns
    
    Extrahiert aus kira_routes.py.backup Learning Pattern Detection Logic
    """
    try:
        if not memory_manager:
            return {
                'available': False,
                'reason': 'no_memory_manager',
                'fallback_patterns': _generate_fallback_learning_patterns()
            }
        
        if time_window is None:
            time_window = timedelta(days=30)  # Default: last 30 days
        
        # Collect learning-related memories
        learning_memories = _collect_learning_memories(memory_manager, time_window)
        
        # Core learning pattern detection
        learning_patterns = {
            'knowledge_acquisition_patterns': _detect_knowledge_acquisition_patterns(learning_memories),
            'skill_development_patterns': _detect_skill_development_patterns(learning_memories),
            'learning_frequency_patterns': _detect_learning_frequency_patterns(learning_memories, time_window),
            'concept_mastery_patterns': _detect_concept_mastery_patterns(learning_memories),
            'learning_difficulty_patterns': _detect_learning_difficulty_patterns(learning_memories)
        }
        
        # Comprehensive pattern analysis
        if pattern_scope == 'comprehensive':
            learning_patterns.update({
                'learning_style_patterns': _detect_learning_style_patterns(learning_memories),
                'retention_patterns': _detect_retention_patterns(learning_memories),
                'forgetting_curve_patterns': _detect_forgetting_curve_patterns(learning_memories),
                'learning_context_patterns': _detect_learning_context_patterns(learning_memories),
                'cross_domain_learning_patterns': _detect_cross_domain_learning_patterns(learning_memories)
            })
        
        # Pattern analysis and insights
        pattern_analysis = {
            'dominant_learning_patterns': _identify_dominant_learning_patterns(learning_patterns),
            'learning_effectiveness_indicators': _analyze_learning_effectiveness(learning_patterns),
            'learning_bottlenecks': _identify_learning_bottlenecks(learning_patterns),
            'learning_acceleration_opportunities': _identify_learning_acceleration_opportunities(learning_patterns)
        }
        
        # Learning recommendations
        learning_recommendations = {
            'learning_optimization_suggestions': _generate_learning_optimization_suggestions(learning_patterns),
            'personalized_learning_strategies': _generate_personalized_learning_strategies(learning_patterns),
            'skill_development_roadmap': _generate_skill_development_roadmap(learning_patterns)
        }
        
        return {
            'available': True,
            'pattern_scope': pattern_scope,
            'time_window_days': time_window.days,
            'learning_memories_analyzed': len(learning_memories),
            'learning_patterns': learning_patterns,
            'pattern_analysis': pattern_analysis,
            'learning_recommendations': learning_recommendations,
            'detection_metadata': {
                'detection_timestamp': datetime.now().isoformat(),
                'pattern_confidence_score': _calculate_pattern_confidence_score(learning_patterns),
                'pattern_diversity_score': _calculate_pattern_diversity_score(learning_patterns)
            }
        }
        
    except Exception as e:
        logger.error(f"Learning pattern detection failed: {e}")
        return {
            'available': False,
            'error': str(e),
            'fallback_patterns': _generate_fallback_learning_patterns()
        }

def analyze_behavioral_patterns(memory_manager=None,
                              behavior_categories: List[str] = None,
                              analysis_depth: str = 'standard') -> Dict[str, Any]:
    """
    Analysiert Behavioral Patterns
    
    Basiert auf kira_routes.py.backup Behavioral Pattern Analysis Logic
    """
    try:
        if not memory_manager:
            return {
                'available': False,
                'reason': 'no_memory_manager',
                'fallback_behavioral_analysis': _generate_fallback_behavioral_analysis()
            }
        
        if behavior_categories is None:
            behavior_categories = [
                'interaction_patterns', 'decision_patterns', 'problem_solving_patterns',
                'communication_patterns', 'adaptation_patterns'
            ]
        
        # Collect behavioral data from memories
        behavioral_data = _collect_behavioral_data(memory_manager)
        
        # Analyze behavioral patterns by category
        behavioral_patterns = {}
        
        if 'interaction_patterns' in behavior_categories:
            behavioral_patterns['interaction_patterns'] = _analyze_interaction_patterns(behavioral_data)
        
        if 'decision_patterns' in behavior_categories:
            behavioral_patterns['decision_patterns'] = _analyze_decision_patterns(behavioral_data)
        
        if 'problem_solving_patterns' in behavior_categories:
            behavioral_patterns['problem_solving_patterns'] = _analyze_problem_solving_patterns(behavioral_data)
        
        if 'communication_patterns' in behavior_categories:
            behavioral_patterns['communication_patterns'] = _analyze_communication_patterns(behavioral_data)
        
        if 'adaptation_patterns' in behavior_categories:
            behavioral_patterns['adaptation_patterns'] = _analyze_adaptation_patterns(behavioral_data)
        
        # Deep analysis for comprehensive depth
        if analysis_depth == 'deep':
            behavioral_patterns.update({
                'emotional_response_patterns': _analyze_emotional_response_patterns(behavioral_data),
                'stress_response_patterns': _analyze_stress_response_patterns(behavioral_data),
                'learning_preference_patterns': _analyze_learning_preference_patterns(behavioral_data),
                'social_interaction_patterns': _analyze_social_interaction_patterns(behavioral_data)
            })
        
        # Behavioral insights
        behavioral_insights = {
            'behavioral_consistency': _assess_behavioral_consistency(behavioral_patterns),
            'behavioral_adaptability': _assess_behavioral_adaptability(behavioral_patterns),
            'behavioral_strengths': _identify_behavioral_strengths(behavioral_patterns),
            'behavioral_improvement_areas': _identify_behavioral_improvement_areas(behavioral_patterns)
        }
        
        # Behavioral predictions
        behavioral_predictions = {
            'likely_future_behaviors': _predict_future_behaviors(behavioral_patterns),
            'behavioral_trend_projections': _project_behavioral_trends(behavioral_patterns),
            'adaptation_likelihood': _assess_adaptation_likelihood(behavioral_patterns)
        }
        
        return {
            'available': True,
            'behavior_categories': behavior_categories,
            'analysis_depth': analysis_depth,
            'behavioral_data_points': len(behavioral_data),
            'behavioral_patterns': behavioral_patterns,
            'behavioral_insights': behavioral_insights,
            'behavioral_predictions': behavioral_predictions,
            'analysis_metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'behavioral_complexity_score': _calculate_behavioral_complexity_score(behavioral_patterns),
                'pattern_reliability_score': _calculate_pattern_reliability_score(behavioral_patterns)
            }
        }
        
    except Exception as e:
        logger.error(f"Behavioral pattern analysis failed: {e}")
        return {
            'available': False,
            'error': str(e),
            'fallback_behavioral_analysis': _generate_fallback_behavioral_analysis()
        }

def identify_memory_trends(memory_manager=None,
                         trend_types: List[str] = None,
                         trend_analysis_period: timedelta = None) -> Dict[str, Any]:
    """
    Identifiziert Memory Trends
    
    Extrahiert aus kira_routes.py.backup Memory Trend Identification Logic
    """
    try:
        if not memory_manager:
            return {
                'available': False,
                'reason': 'no_memory_manager',
                'fallback_trends': _generate_fallback_memory_trends()
            }
        
        if trend_types is None:
            trend_types = [
                'memory_formation_trends', 'content_evolution_trends', 'access_pattern_trends',
                'consolidation_trends', 'forgetting_trends'
            ]
        
        if trend_analysis_period is None:
            trend_analysis_period = timedelta(days=60)  # Default: 60 days
        
        # Collect time-series memory data
        memory_time_series = _collect_memory_time_series(memory_manager, trend_analysis_period)
        
        # Analyze trends by type
        memory_trends = {}
        
        if 'memory_formation_trends' in trend_types:
            memory_trends['memory_formation_trends'] = _analyze_memory_formation_trends(memory_time_series)
        
        if 'content_evolution_trends' in trend_types:
            memory_trends['content_evolution_trends'] = _analyze_content_evolution_trends(memory_time_series)
        
        if 'access_pattern_trends' in trend_types:
            memory_trends['access_pattern_trends'] = _analyze_access_pattern_trends(memory_time_series)
        
        if 'consolidation_trends' in trend_types:
            memory_trends['consolidation_trends'] = _analyze_consolidation_trends(memory_time_series)
        
        if 'forgetting_trends' in trend_types:
            memory_trends['forgetting_trends'] = _analyze_forgetting_trends(memory_time_series)
        
        # Trend analysis and forecasting
        trend_analysis = {
            'trend_directions': _determine_trend_directions(memory_trends),
            'trend_strengths': _assess_trend_strengths(memory_trends),
            'trend_stability': _assess_trend_stability(memory_trends),
            'trend_correlations': _identify_trend_correlations(memory_trends)
        }
        
        # Trend forecasting
        trend_forecasting = {
            'short_term_projections': _generate_short_term_trend_projections(memory_trends),
            'long_term_projections': _generate_long_term_trend_projections(memory_trends),
            'trend_break_predictions': _predict_trend_breaks(memory_trends),
            'intervention_recommendations': _recommend_trend_interventions(memory_trends)
        }
        
        return {
            'available': True,
            'trend_types': trend_types,
            'analysis_period_days': trend_analysis_period.days,
            'memory_time_series_points': len(memory_time_series),
            'memory_trends': memory_trends,
            'trend_analysis': trend_analysis,
            'trend_forecasting': trend_forecasting,
            'trend_metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'trend_data_quality_score': _assess_trend_data_quality(memory_time_series),
                'forecasting_confidence_score': _calculate_forecasting_confidence_score(memory_trends)
            }
        }
        
    except Exception as e:
        logger.error(f"Memory trend identification failed: {e}")
        return {
            'available': False,
            'error': str(e),
            'fallback_trends': _generate_fallback_memory_trends()
        }

def calculate_pattern_significance(patterns: Dict,
                                 significance_criteria: Dict = None,
                                 statistical_confidence: float = 0.95) -> Dict[str, Any]:
    """
    Berechnet Pattern Significance
    
    Basiert auf kira_routes.py.backup Pattern Significance Calculation Logic
    """
    try:
        if not patterns:
            return {
                'available': False,
                'reason': 'no_patterns_provided'
            }
        
        if significance_criteria is None:
            significance_criteria = {
                'minimum_occurrences': 5,
                'minimum_consistency': 0.7,
                'minimum_effect_size': 0.3,
                'temporal_stability_required': True
            }
        
        # Calculate significance for each pattern type
        pattern_significance = {}
        
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict):
                significance_analysis = {
                    'statistical_significance': _calculate_statistical_significance(pattern_data, significance_criteria),
                    'practical_significance': _calculate_practical_significance(pattern_data, significance_criteria),
                    'temporal_stability': _assess_temporal_stability(pattern_data, significance_criteria),
                    'effect_size': _calculate_effect_size(pattern_data),
                    'confidence_interval': _calculate_confidence_interval(pattern_data, statistical_confidence)
                }
                
                # Overall significance score
                significance_scores = [
                    significance_analysis['statistical_significance'],
                    significance_analysis['practical_significance'],
                    significance_analysis['temporal_stability'],
                    min(significance_analysis['effect_size'] * 2, 1.0)  # Normalize effect size
                ]
                
                overall_significance = statistics.mean([s for s in significance_scores if isinstance(s, (int, float))])
                significance_analysis['overall_significance_score'] = overall_significance
                significance_analysis['significance_rating'] = _rate_significance(overall_significance)
                
                pattern_significance[pattern_type] = significance_analysis
        
        # Cross-pattern significance analysis
        cross_pattern_analysis = {
            'most_significant_patterns': _identify_most_significant_patterns(pattern_significance),
            'pattern_significance_distribution': _analyze_significance_distribution(pattern_significance),
            'significance_correlations': _identify_significance_correlations(pattern_significance),
            'actionable_patterns': _identify_actionable_patterns(pattern_significance, significance_criteria)
        }
        
        # Significance insights and recommendations
        significance_insights = {
            'key_findings': _extract_key_significance_findings(pattern_significance, cross_pattern_analysis),
            'reliability_assessment': _assess_pattern_reliability(pattern_significance),
            'generalizability_assessment': _assess_pattern_generalizability(pattern_significance),
            'intervention_priorities': _prioritize_pattern_interventions(pattern_significance)
        }
        
        return {
            'available': True,
            'significance_criteria': significance_criteria,
            'statistical_confidence': statistical_confidence,
            'patterns_analyzed': len(patterns),
            'pattern_significance': pattern_significance,
            'cross_pattern_analysis': cross_pattern_analysis,
            'significance_insights': significance_insights,
            'calculation_metadata': {
                'calculation_timestamp': datetime.now().isoformat(),
                'analysis_thoroughness_score': _calculate_analysis_thoroughness_score(pattern_significance),
                'methodological_rigor_score': _assess_methodological_rigor(significance_criteria)
            }
        }
        
    except Exception as e:
        logger.error(f"Pattern significance calculation failed: {e}")
        return {
            'available': False,
            'error': str(e)
        }

# ====================================
# PRIVATE HELPER FUNCTIONS
# ====================================

def _collect_learning_memories(memory_manager, time_window: timedelta) -> List[Dict]:
    """Sammelt Learning-related Memories"""
    try:
        learning_memories = []
        cutoff_time = datetime.now() - time_window
        
        # Keywords that indicate learning content
        learning_keywords = [
            'learn', 'understand', 'knowledge', 'skill', 'concept', 'study',
            'practice', 'master', 'improve', 'develop', 'acquire', 'grasp'
        ]
        
        # Collect from different memory systems
        for memory_system_name in ['short_term_memory', 'long_term_memory', 'working_memory']:
            if hasattr(memory_manager, memory_system_name):
                memory_system = getattr(memory_manager, memory_system_name)
                memories = _extract_memories_safely_patterns(memory_system)
                
                for memory in memories:
                    memory_dict = memory if isinstance(memory, dict) else {'content': memory}
                    
                    # Check if memory is learning-related
                    content_str = json.dumps(memory_dict).lower() if isinstance(memory_dict, dict) else str(memory_dict).lower()
                    
                    if any(keyword in content_str for keyword in learning_keywords):
                        # Check time window
                        memory_time = _extract_memory_timestamp_safe_patterns(memory_dict)
                        if not memory_time or memory_time >= cutoff_time:
                            memory_dict['learning_relevance_score'] = _calculate_learning_relevance_score(memory_dict)
                            memory_dict['memory_system'] = memory_system_name
                            learning_memories.append(memory_dict)
        
        return learning_memories
        
    except Exception as e:
        logger.debug(f"Learning memory collection failed: {e}")
        return []

def _detect_knowledge_acquisition_patterns(learning_memories: List[Dict]) -> Dict[str, Any]:
    """Erkennt Knowledge Acquisition Patterns"""
    try:
        if not learning_memories:
            return {'pattern_strength': 0.0, 'patterns': []}
        
        # Analyze knowledge acquisition patterns
        knowledge_domains = defaultdict(list)
        acquisition_timeline = []
        acquisition_methods = Counter()
        
        for memory in learning_memories:
            # Extract knowledge domain
            domain = _extract_knowledge_domain(memory)
            knowledge_domains[domain].append(memory)
            
            # Timeline analysis
            memory_time = _extract_memory_timestamp_safe_patterns(memory)
            if memory_time:
                acquisition_timeline.append({
                    'timestamp': memory_time,
                    'domain': domain,
                    'content': memory.get('content', '')
                })
            
            # Acquisition method analysis
            method = _identify_acquisition_method(memory)
            acquisition_methods[method] += 1
        
        # Identify patterns
        patterns = []
        
        # Domain clustering pattern
        if len(knowledge_domains) > 1:
            patterns.append({
                'pattern_type': 'domain_clustering',
                'description': f'Knowledge acquisition spans {len(knowledge_domains)} domains',
                'domains': list(knowledge_domains.keys()),
                'pattern_strength': min(len(knowledge_domains) / 5, 1.0)  # Normalize to 0-1
            })
        
        # Sequential learning pattern
        if len(acquisition_timeline) >= 3:
            timeline_sorted = sorted(acquisition_timeline, key=lambda x: x['timestamp'])
            domain_sequence = [item['domain'] for item in timeline_sorted]
            
            # Check for sequential domain learning
            domain_changes = sum(1 for i in range(1, len(domain_sequence)) if domain_sequence[i] != domain_sequence[i-1])
            sequential_strength = domain_changes / len(domain_sequence) if domain_sequence else 0
            
            patterns.append({
                'pattern_type': 'sequential_learning',
                'description': f'Sequential learning across domains with {domain_changes} transitions',
                'sequence_strength': sequential_strength,
                'pattern_strength': sequential_strength
            })
        
        # Method preference pattern
        if acquisition_methods:
            dominant_method = acquisition_methods.most_common(1)[0]
            method_preference_strength = dominant_method[1] / sum(acquisition_methods.values())
            
            patterns.append({
                'pattern_type': 'method_preference',
                'description': f'Preference for {dominant_method[0]} learning method',
                'preferred_method': dominant_method[0],
                'preference_strength': method_preference_strength,
                'pattern_strength': method_preference_strength
            })
        
        # Overall pattern strength
        overall_strength = statistics.mean([p['pattern_strength'] for p in patterns]) if patterns else 0.0
        
        return {
            'pattern_strength': overall_strength,
            'patterns': patterns,
            'knowledge_domains': dict(knowledge_domains),
            'acquisition_methods': dict(acquisition_methods),
            'total_acquisitions': len(learning_memories)
        }
        
    except Exception as e:
        logger.debug(f"Knowledge acquisition pattern detection failed: {e}")
        return {'pattern_strength': 0.0, 'patterns': [], 'error': str(e)}

def _analyze_interaction_patterns(behavioral_data: List[Dict]) -> Dict[str, Any]:
    """Analysiert Interaction Patterns"""
    try:
        if not behavioral_data:
            return {'pattern_strength': 0.0, 'patterns': []}
        
        # Filter interaction-related data
        interaction_data = [
            data for data in behavioral_data
            if _is_interaction_related(data)
        ]
        
        if not interaction_data:
            return {'pattern_strength': 0.0, 'patterns': []}
        
        # Analyze interaction patterns
        interaction_types = Counter()
        interaction_frequency = defaultdict(list)
        response_patterns = []
        
        for data in interaction_data:
            # Classify interaction type
            interaction_type = _classify_interaction_type(data)
            interaction_types[interaction_type] += 1
            
            # Frequency analysis
            timestamp = _extract_memory_timestamp_safe_patterns(data)
            if timestamp:
                hour = timestamp.hour
                interaction_frequency[hour].append(data)
            
            # Response pattern analysis
            response_pattern = _analyze_response_pattern(data)
            if response_pattern:
                response_patterns.append(response_pattern)
        
        patterns = []
        
        # Interaction type preference pattern
        if interaction_types:
            dominant_type = interaction_types.most_common(1)[0]
            type_preference_strength = dominant_type[1] / sum(interaction_types.values())
            
            patterns.append({
                'pattern_type': 'interaction_type_preference',
                'description': f'Preference for {dominant_type[0]} interactions',
                'preferred_type': dominant_type[0],
                'preference_strength': type_preference_strength,
                'pattern_strength': type_preference_strength
            })
        
        # Temporal interaction pattern
        if interaction_frequency:
            peak_hours = sorted(interaction_frequency.items(), key=lambda x: len(x[1]), reverse=True)[:3]
            temporal_concentration = sum(len(hour_data[1]) for hour_data in peak_hours) / len(interaction_data)
            
            patterns.append({
                'pattern_type': 'temporal_interaction',
                'description': f'Peak interaction hours: {[hour[0] for hour in peak_hours]}',
                'peak_hours': [hour[0] for hour in peak_hours],
                'concentration_strength': temporal_concentration,
                'pattern_strength': temporal_concentration
            })
        
        # Response consistency pattern
        if response_patterns:
            response_consistency = _calculate_response_consistency(response_patterns)
            
            patterns.append({
                'pattern_type': 'response_consistency',
                'description': f'Response consistency score: {response_consistency:.2f}',
                'consistency_score': response_consistency,
                'pattern_strength': response_consistency
            })
        
        overall_strength = statistics.mean([p['pattern_strength'] for p in patterns]) if patterns else 0.0
        
        return {
            'pattern_strength': overall_strength,
            'patterns': patterns,
            'interaction_types': dict(interaction_types),
            'temporal_distribution': {hour: len(data) for hour, data in interaction_frequency.items()},
            'total_interactions': len(interaction_data)
        }
        
    except Exception as e:
        logger.debug(f"Interaction pattern analysis failed: {e}")
        return {'pattern_strength': 0.0, 'patterns': [], 'error': str(e)}

def _collect_memory_time_series(memory_manager, analysis_period: timedelta) -> List[Dict]:
    """Sammelt Memory Time Series Data"""
    try:
        cutoff_time = datetime.now() - analysis_period
        time_series_data = []
        
        # Collect timestamped memories from all systems
        for memory_system_name in ['short_term_memory', 'long_term_memory', 'working_memory']:
            if hasattr(memory_manager, memory_system_name):
                memory_system = getattr(memory_manager, memory_system_name)
                memories = _extract_memories_safely_patterns(memory_system)
                
                for memory in memories:
                    memory_dict = memory if isinstance(memory, dict) else {'content': memory}
                    memory_time = _extract_memory_timestamp_safe_patterns(memory_dict)
                    
                    if memory_time and memory_time >= cutoff_time:
                        time_series_point = {
                            'timestamp': memory_time,
                            'memory_system': memory_system_name,
                            'content': memory_dict.get('content', ''),
                            'memory_type': _classify_memory_type(memory_dict),
                            'content_length': len(str(memory_dict.get('content', ''))),
                            'importance_score': memory_dict.get('importance', 0.5)
                        }
                        time_series_data.append(time_series_point)
        
        # Sort by timestamp
        time_series_data.sort(key=lambda x: x['timestamp'])
        
        return time_series_data
        
    except Exception as e:
        logger.debug(f"Memory time series collection failed: {e}")
        return []

def _calculate_statistical_significance(pattern_data: Dict, criteria: Dict) -> float:
    """Berechnet Statistical Significance"""
    try:
        # Extract pattern occurrences
        occurrences = pattern_data.get('total_occurrences', 0)
        if isinstance(pattern_data.get('patterns'), list):
            occurrences = len(pattern_data['patterns'])
        
        min_occurrences = criteria.get('minimum_occurrences', 5)
        
        # Simple significance based on occurrence count
        if occurrences >= min_occurrences:
            # Calculate significance score based on how much we exceed minimum
            excess_ratio = (occurrences - min_occurrences) / min_occurrences
            significance_score = min(1.0, 0.7 + (excess_ratio * 0.3))  # 0.7 base + up to 0.3 bonus
        else:
            significance_score = occurrences / min_occurrences  # Partial significance
        
        return significance_score
        
    except Exception as e:
        logger.debug(f"Statistical significance calculation failed: {e}")
        return 0.0

def _extract_memories_safely_patterns(memory_system) -> List:
    """Extrahiert Memories sicher für Pattern Analysis"""
    try:
        if hasattr(memory_system, 'get_memories'):
            return memory_system.get_memories()
        elif hasattr(memory_system, 'memories'):
            memories = memory_system.memories
            if isinstance(memories, list):
                return memories
            elif hasattr(memories, '__iter__') and not isinstance(memories, str):
                return list(memories)
            else:
                return [memories]
        elif hasattr(memory_system, 'get_all'):
            return memory_system.get_all()
        elif isinstance(memory_system, list):
            return memory_system
        elif isinstance(memory_system, dict):
            return [memory_system]
        else:
            return []
    except Exception as e:
        logger.debug(f"Memory extraction for patterns failed: {e}")
        return []

def _extract_memory_timestamp_safe_patterns(memory) -> Optional[datetime]:
    """Extrahiert Memory Timestamp sicher für Patterns"""
    try:
        if isinstance(memory, dict):
            for field in ['timestamp', 'created_at', 'time', 'date']:
                if field in memory:
                    timestamp_value = memory[field]
                    if isinstance(timestamp_value, datetime):
                        return timestamp_value
                    elif isinstance(timestamp_value, str):
                        try:
                            # Try different datetime formats
                            for fmt in ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                                try:
                                    return datetime.strptime(timestamp_value.replace('Z', ''), fmt)
                                except ValueError:
                                    continue
                            # Try ISO format with timezone
                            return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                        except:
                            continue
        return None
    except Exception as e:
        logger.debug(f"Timestamp extraction for patterns failed: {e}")
        return None

def _calculate_learning_relevance_score(memory_dict: Dict) -> float:
    """Berechnet Learning Relevance Score"""
    try:
        content = memory_dict.get('content', '')
        content_str = json.dumps(content).lower() if isinstance(content, dict) else str(content).lower()
        
        learning_indicators = [
            'learn', 'understand', 'knowledge', 'skill', 'concept', 'study',
            'practice', 'master', 'improve', 'develop', 'acquire', 'tutorial'
        ]
        
        relevance_score = 0.0
        for indicator in learning_indicators:
            if indicator in content_str:
                relevance_score += 1.0 / len(learning_indicators)
        
        # Bonus for educational context
        if any(word in content_str for word in ['explain', 'example', 'how to', 'why']):
            relevance_score += 0.2
        
        return min(1.0, relevance_score)
        
    except Exception as e:
        logger.debug(f"Learning relevance score calculation failed: {e}")
        return 0.5

def _extract_knowledge_domain(memory: Dict) -> str:
    """Extrahiert Knowledge Domain"""
    try:
        content = memory.get('content', '')
        content_str = json.dumps(content).lower() if isinstance(content, dict) else str(content).lower()
        
        # Domain keywords mapping
        domain_keywords = {
            'programming': ['code', 'program', 'function', 'variable', 'algorithm', 'debug'],
            'language': ['word', 'grammar', 'sentence', 'translate', 'language', 'vocabulary'],
            'mathematics': ['math', 'equation', 'number', 'calculate', 'formula', 'solve'],
            'science': ['experiment', 'hypothesis', 'theory', 'research', 'analysis', 'data'],
            'general': []  # Default category
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in content_str for keyword in keywords):
                return domain
        
        return 'general'
        
    except Exception as e:
        logger.debug(f"Knowledge domain extraction failed: {e}")
        return 'general'

def _identify_acquisition_method(memory: Dict) -> str:
    """Identifiziert Acquisition Method"""
    try:
        content = memory.get('content', '')
        content_str = json.dumps(content).lower() if isinstance(content, dict) else str(content).lower()
        
        method_keywords = {
            'reading': ['read', 'book', 'article', 'text', 'document'],
            'practice': ['practice', 'exercise', 'try', 'attempt', 'do'],
            'explanation': ['explain', 'understand', 'clarify', 'describe'],
            'example': ['example', 'instance', 'case', 'sample', 'demo'],
            'discussion': ['discuss', 'talk', 'conversation', 'chat', 'ask']
        }
        
        for method, keywords in method_keywords.items():
            if any(keyword in content_str for keyword in keywords):
                return method
        
        return 'general'
        
    except Exception as e:
        logger.debug(f"Acquisition method identification failed: {e}")
        return 'general'

def _is_interaction_related(data: Dict) -> bool:
    """Prüft ob Data Interaction-related ist"""
    try:
        content = data.get('content', '')
        content_str = json.dumps(content).lower() if isinstance(content, dict) else str(content).lower()
        
        interaction_keywords = [
            'question', 'ask', 'answer', 'response', 'reply', 'conversation',
            'chat', 'discuss', 'talk', 'communicate', 'interact'
        ]
        
        return any(keyword in content_str for keyword in interaction_keywords)
        
    except Exception as e:
        logger.debug(f"Interaction relation check failed: {e}")
        return False

def _classify_memory_type(memory_dict: Dict) -> str:
    """Klassifiziert Memory Type"""
    try:
        content = memory_dict.get('content', '')
        content_str = json.dumps(content).lower() if isinstance(content, dict) else str(content).lower()
        
        type_keywords = {
            'procedural': ['how to', 'step', 'process', 'procedure', 'method'],
            'declarative': ['fact', 'information', 'data', 'knowledge', 'definition'],
            'episodic': ['happened', 'event', 'experience', 'remember', 'recall'],
            'semantic': ['concept', 'meaning', 'relationship', 'category', 'classification']
        }
        
        for memory_type, keywords in type_keywords.items():
            if any(keyword in content_str for keyword in keywords):
                return memory_type
        
        return 'general'
        
    except Exception as e:
        logger.debug(f"Memory type classification failed: {e}")
        return 'general'

def _generate_fallback_learning_patterns() -> Dict[str, Any]:
    """Generiert Fallback Learning Patterns"""
    return {
        'fallback_mode': True,
        'knowledge_acquisition_patterns': {
            'pattern_strength': 0.7,
            'patterns': [
                {
                    'pattern_type': 'sequential_learning',
                    'description': 'Sequential learning across domains',
                    'pattern_strength': 0.7
                }
            ]
        },
        'skill_development_patterns': {
            'pattern_strength': 0.6,
            'patterns': [
                {
                    'pattern_type': 'incremental_improvement',
                    'description': 'Gradual skill improvement over time',
                    'pattern_strength': 0.6
                }
            ]
        },
        'learning_frequency_patterns': {
            'pattern_strength': 0.8,
            'daily_learning_average': 5,
            'peak_learning_hours': [10, 14, 16]
        }
    }

def _generate_fallback_behavioral_analysis() -> Dict[str, Any]:
    """Generiert Fallback Behavioral Analysis"""
    return {
        'fallback_mode': True,
        'interaction_patterns': {
            'pattern_strength': 0.7,
            'preferred_interaction_type': 'question_answer',
            'interaction_consistency': 0.8
        },
        'decision_patterns': {
            'pattern_strength': 0.6,
            'decision_style': 'analytical',
            'decision_speed': 'moderate'
        },
        'problem_solving_patterns': {
            'pattern_strength': 0.8,
            'approach': 'systematic',
            'persistence_level': 'high'
        }
    }

def _generate_fallback_memory_trends() -> Dict[str, Any]:
    """Generiert Fallback Memory Trends"""
    return {
        'fallback_mode': True,
        'memory_formation_trends': {
            'trend_direction': 'increasing',
            'trend_strength': 0.6,
            'weekly_formation_rate': 25
        },
        'content_evolution_trends': {
            'trend_direction': 'diversifying',
            'trend_strength': 0.7,
            'content_complexity_trend': 'increasing'
        },
        'access_pattern_trends': {
            'trend_direction': 'stable',
            'trend_strength': 0.8,
            'access_frequency_trend': 'consistent'
        }
    }

__all__ = [
    'detect_learning_patterns',
    'analyze_behavioral_patterns',
    'identify_memory_trends',
    'calculate_pattern_significance'
]