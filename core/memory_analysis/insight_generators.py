"""
Memory Insight Generators
Generiert intelligente Insights und Empfehlungen basierend auf Memory Analysis
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from .analysis_helpers import (
    collect_memory_data, calculate_memory_age_distribution,
    classify_content_type, extract_memory_content_safe
)

logger = logging.getLogger(__name__)

def generate_memory_insights(memory_manager=None,
                           insight_categories: List[str] = None,
                           analysis_depth: str = 'standard') -> Dict[str, Any]:
    """Hauptfunktion für Memory Insights Generation"""
    try:
        if not memory_manager:
            return {
                'available': False,
                'reason': 'no_memory_manager',
                'fallback_insights': _generate_fallback_insights()
            }
        
        if insight_categories is None:
            insight_categories = ['patterns', 'trends', 'optimization', 'behavior']
        
        # Collect memory data
        memory_data = collect_memory_data(memory_manager, timedelta(days=30))
        
        # Generate insights
        generated_insights = {}
        
        if 'patterns' in insight_categories:
            generated_insights['pattern_insights'] = _generate_pattern_insights(memory_data)
        
        if 'trends' in insight_categories:
            generated_insights['trend_insights'] = _generate_trend_insights(memory_data)
        
        if 'optimization' in insight_categories:
            generated_insights['optimization_insights'] = _generate_optimization_insights(memory_data)
        
        if 'behavior' in insight_categories:
            generated_insights['behavioral_insights'] = _generate_behavioral_insights(memory_data)
        
        if 'learning' in insight_categories:
            generated_insights['learning_insights'] = _generate_learning_insights(memory_data)
        
        # Meta insights
        meta_insights = _generate_meta_insights(generated_insights, memory_data)
        
        return {
            'available': True,
            'analysis_depth': analysis_depth,
            'insight_categories': insight_categories,
            'generated_insights': generated_insights,
            'meta_insights': meta_insights,
            'key_findings': _extract_key_findings(generated_insights),
            'actionable_recommendations': _generate_actionable_recommendations(generated_insights),
            'insights_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'memories_analyzed': len(memory_data.get('all_memories', [])),
                'analysis_timeframe_days': 30
            }
        }
        
    except Exception as e:
        logger.error(f"Memory insights generation failed: {e}")
        return {'available': False, 'error': str(e)}

def _generate_pattern_insights(memory_data: Dict) -> Dict[str, Any]:
    """Generiert Pattern-basierte Insights"""
    try:
        all_memories = memory_data.get('all_memories', [])
        temporal_distribution = memory_data.get('temporal_distribution', {})
        memory_types = memory_data.get('memory_types', {})
        
        insights = []
        
        # Temporal pattern insights
        if temporal_distribution:
            daily_counts = list(temporal_distribution.values())
            if daily_counts:
                avg_daily = statistics.mean(daily_counts)
                max_daily = max(daily_counts)
                
                if max_daily > avg_daily * 2:
                    insights.append({
                        'type': 'temporal_spike',
                        'description': f'Detected high memory formation days with {max_daily} memories (avg: {avg_daily:.1f})',
                        'significance': 'high',
                        'recommendation': 'Investigate factors contributing to high memory formation periods'
                    })
                
                if len([c for c in daily_counts if c == 0]) > len(daily_counts) * 0.3:
                    insights.append({
                        'type': 'temporal_gaps',
                        'description': 'Frequent gaps in memory formation detected',
                        'significance': 'moderate',
                        'recommendation': 'Consider more consistent memory formation patterns'
                    })
        
        # Content pattern insights
        content_types = {}
        for memory in all_memories:
            content = extract_memory_content_safe(memory)
            content_type = classify_content_type(content)
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        if content_types:
            dominant_type = max(content_types.items(), key=lambda x: x[1])
            if dominant_type[1] > len(all_memories) * 0.6:
                insights.append({
                    'type': 'content_dominance',
                    'description': f'Strong preference for {dominant_type[0]} content ({dominant_type[1]} memories)',
                    'significance': 'moderate',
                    'recommendation': 'Consider diversifying memory content types'
                })
        
        # Memory type distribution insights
        if memory_types:
            type_counts = {k: len(v) for k, v in memory_types.items()}
            total_typed = sum(type_counts.values())
            
            if total_typed > 0:
                for mem_type, count in type_counts.items():
                    percentage = (count / total_typed) * 100
                    if percentage > 70:
                        insights.append({
                            'type': 'memory_type_imbalance',
                            'description': f'{mem_type} memories dominate ({percentage:.1f}% of total)',
                            'significance': 'moderate',
                            'recommendation': f'Consider balancing {mem_type} with other memory types'
                        })
        
        return {
            'insights_count': len(insights),
            'insights': insights,
            'pattern_summary': {
                'dominant_content_type': max(content_types.items(), key=lambda x: x[1])[0] if content_types else 'unknown',
                'memory_type_distribution': {k: len(v) for k, v in memory_types.items()},
                'temporal_consistency': _calculate_temporal_consistency(temporal_distribution)
            }
        }
        
    except Exception as e:
        logger.debug(f"Pattern insights generation failed: {e}")
        return {'insights_count': 0, 'insights': [], 'error': str(e)}

def _generate_trend_insights(memory_data: Dict) -> Dict[str, Any]:
    """Generiert Trend-basierte Insights"""
    try:
        temporal_distribution = memory_data.get('temporal_distribution', {})
        all_memories = memory_data.get('all_memories', [])
        
        insights = []
        
        if temporal_distribution and len(temporal_distribution) >= 7:
            # Analyze trends over time
            sorted_dates = sorted(temporal_distribution.keys())
            recent_week = sorted_dates[-7:]
            older_week = sorted_dates[-14:-7] if len(sorted_dates) >= 14 else sorted_dates[:-7]
            
            if recent_week and older_week:
                recent_avg = statistics.mean([temporal_distribution[date] for date in recent_week])
                older_avg = statistics.mean([temporal_distribution[date] for date in older_week])
                
                if recent_avg > older_avg * 1.5:
                    insights.append({
                        'type': 'increasing_activity',
                        'description': f'Memory formation increasing: {recent_avg:.1f} vs {older_avg:.1f} memories/day',
                        'significance': 'high',
                        'trend_direction': 'upward',
                        'recommendation': 'Positive trend - maintain current patterns'
                    })
                elif recent_avg < older_avg * 0.7:
                    insights.append({
                        'type': 'decreasing_activity',
                        'description': f'Memory formation declining: {recent_avg:.1f} vs {older_avg:.1f} memories/day',
                        'significance': 'moderate',
                        'trend_direction': 'downward',
                        'recommendation': 'Consider factors affecting memory formation'
                    })
        
        # Content complexity trends
        if len(all_memories) >= 20:
            recent_memories = all_memories[-10:]
            older_memories = all_memories[-20:-10]
            
            recent_complexity = []
            older_complexity = []
            
            for memory in recent_memories:
                content = extract_memory_content_safe(memory)
                recent_complexity.append(len(content))
            
            for memory in older_memories:
                content = extract_memory_content_safe(memory)
                older_complexity.append(len(content))
            
            if recent_complexity and older_complexity:
                recent_avg_complexity = statistics.mean(recent_complexity)
                older_avg_complexity = statistics.mean(older_complexity)
                
                if recent_avg_complexity > older_avg_complexity * 1.3:
                    insights.append({
                        'type': 'complexity_increase',
                        'description': 'Memory content becoming more complex',
                        'significance': 'moderate',
                        'trend_direction': 'upward',
                        'recommendation': 'Positive development - richer memory content'
                    })
        
        return {
            'insights_count': len(insights),
            'insights': insights,
            'trend_analysis': {
                'temporal_trend': _analyze_temporal_trend(temporal_distribution),
                'activity_level': _determine_activity_level(temporal_distribution),
                'trend_stability': _calculate_trend_stability(temporal_distribution)
            }
        }
        
    except Exception as e:
        logger.debug(f"Trend insights generation failed: {e}")
        return {'insights_count': 0, 'insights': [], 'error': str(e)}

def _generate_optimization_insights(memory_data: Dict) -> Dict[str, Any]:
    """Generiert Optimierungs-Insights"""
    try:
        all_memories = memory_data.get('all_memories', [])
        memory_types = memory_data.get('memory_types', {})
        
        insights = []
        
        # Memory consolidation optimization
        short_term_count = len(memory_types.get('short_term', []))
        long_term_count = len(memory_types.get('long_term', []))
        working_count = len(memory_types.get('working', []))
        
        if short_term_count > 50:
            insights.append({
                'type': 'consolidation_opportunity',
                'description': f'Large number of short-term memories ({short_term_count}) could be consolidated',
                'significance': 'moderate',
                'optimization_type': 'consolidation',
                'recommendation': 'Consider consolidating older short-term memories to long-term storage'
            })
        
        if working_count > 15:
            insights.append({
                'type': 'working_memory_cleanup',
                'description': f'Working memory appears overloaded ({working_count} items)',
                'significance': 'high',
                'optimization_type': 'cleanup',
                'recommendation': 'Clear completed working memory items to improve performance'
            })
        
        # Content optimization
        content_lengths = []
        for memory in all_memories[:100]:  # Sample for performance
            content = extract_memory_content_safe(memory)
            content_lengths.append(len(content))
        
        if content_lengths:
            avg_length = statistics.mean(content_lengths)
            very_short = len([l for l in content_lengths if l < 10])
            very_long = len([l for l in content_lengths if l > 500])
            
            if very_short > len(content_lengths) * 0.3:
                insights.append({
                    'type': 'content_fragmentation',
                    'description': f'Many very short memories detected ({very_short} of {len(content_lengths)})',
                    'significance': 'moderate',
                    'optimization_type': 'content',
                    'recommendation': 'Consider combining related short memories'
                })
            
            if very_long > len(content_lengths) * 0.1:
                insights.append({
                    'type': 'content_bloat',
                    'description': f'Some very long memories detected ({very_long} of {len(content_lengths)})',
                    'significance': 'low',
                    'optimization_type': 'content',
                    'recommendation': 'Consider summarizing or segmenting long memories'
                })
        
        return {
            'insights_count': len(insights),
            'insights': insights,
            'optimization_potential': {
                'consolidation_candidates': short_term_count,
                'cleanup_candidates': working_count,
                'content_optimization_score': _calculate_content_optimization_score(content_lengths)
            }
        }
        
    except Exception as e:
        logger.debug(f"Optimization insights generation failed: {e}")
        return {'insights_count': 0, 'insights': [], 'error': str(e)}

def _generate_behavioral_insights(memory_data: Dict) -> Dict[str, Any]:
    """Generiert Verhaltens-Insights"""
    try:
        all_memories = memory_data.get('all_memories', [])
        temporal_distribution = memory_data.get('temporal_distribution', {})
        
        insights = []
        
        # Activity pattern insights
        if temporal_distribution:
            daily_counts = list(temporal_distribution.values())
            if daily_counts:
                most_active = max(daily_counts)
                least_active = min(daily_counts)
                
                if most_active > least_active * 5:
                    insights.append({
                        'type': 'irregular_activity',
                        'description': f'Highly irregular memory formation pattern (range: {least_active}-{most_active})',
                        'significance': 'moderate',
                        'behavioral_pattern': 'irregular',
                        'recommendation': 'Consider establishing more consistent memory formation routines'
                    })
                
                # Consistency analysis
                if len(daily_counts) >= 7:
                    recent_consistency = statistics.stdev(daily_counts[-7:])
                    if recent_consistency < 2:
                        insights.append({
                            'type': 'consistent_behavior',
                            'description': 'Very consistent memory formation pattern in recent period',
                            'significance': 'positive',
                            'behavioral_pattern': 'consistent',
                            'recommendation': 'Maintain current consistent patterns'
                        })
        
        # Content behavior insights
        content_types = {}
        for memory in all_memories:
            content = extract_memory_content_safe(memory)
            content_type = classify_content_type(content)
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        if content_types:
            total_content = sum(content_types.values())
            learning_ratio = content_types.get('learning', 0) / total_content
            
            if learning_ratio > 0.4:
                insights.append({
                    'type': 'learning_focused',
                    'description': f'Strong learning orientation ({learning_ratio:.1%} of memories)',
                    'significance': 'positive',
                    'behavioral_pattern': 'learning_oriented',
                    'recommendation': 'Excellent learning behavior - continue current approach'
                })
        
        return {
            'insights_count': len(insights),
            'insights': insights,
            'behavioral_profile': {
                'activity_pattern': _classify_activity_pattern(temporal_distribution),
                'content_preference': max(content_types.items(), key=lambda x: x[1])[0] if content_types else 'unknown',
                'consistency_score': _calculate_behavioral_consistency(temporal_distribution)
            }
        }
        
    except Exception as e:
        logger.debug(f"Behavioral insights generation failed: {e}")
        return {'insights_count': 0, 'insights': [], 'error': str(e)}

def _generate_learning_insights(memory_data: Dict) -> Dict[str, Any]:
    """Generiert Lern-spezifische Insights"""
    try:
        all_memories = memory_data.get('all_memories', [])
        
        insights = []
        learning_memories = []
        
        # Identify learning-related memories
        for memory in all_memories:
            content = extract_memory_content_safe(memory).lower()
            if any(keyword in content for keyword in ['learn', 'understand', 'knowledge', 'study', 'discover']):
                learning_memories.append(memory)
        
        learning_ratio = len(learning_memories) / len(all_memories) if all_memories else 0
        
        if learning_ratio > 0.3:
            insights.append({
                'type': 'high_learning_activity',
                'description': f'High learning activity detected ({learning_ratio:.1%} of memories)',
                'significance': 'high',
                'learning_metric': 'activity_level',
                'recommendation': 'Excellent learning engagement - maintain current patterns'
            })
        elif learning_ratio < 0.1:
            insights.append({
                'type': 'low_learning_activity',
                'description': f'Limited learning activity detected ({learning_ratio:.1%} of memories)',
                'significance': 'moderate',
                'learning_metric': 'activity_level',
                'recommendation': 'Consider increasing learning-focused activities'
            })
        
        # Learning complexity analysis
        if learning_memories:
            learning_content_lengths = []
            for memory in learning_memories:
                content = extract_memory_content_safe(memory)
                learning_content_lengths.append(len(content))
            
            avg_learning_complexity = statistics.mean(learning_content_lengths)
            
            if avg_learning_complexity > 200:
                insights.append({
                    'type': 'complex_learning',
                    'description': 'Learning memories show high complexity',
                    'significance': 'positive',
                    'learning_metric': 'complexity',
                    'recommendation': 'Great depth in learning content - continue detailed approach'
                })
        
        return {
            'insights_count': len(insights),
            'insights': insights,
            'learning_profile': {
                'learning_ratio': learning_ratio,
                'learning_memory_count': len(learning_memories),
                'learning_engagement': 'high' if learning_ratio > 0.3 else 'moderate' if learning_ratio > 0.1 else 'low'
            }
        }
        
    except Exception as e:
        logger.debug(f"Learning insights generation failed: {e}")
        return {'insights_count': 0, 'insights': [], 'error': str(e)}

def _generate_meta_insights(generated_insights: Dict, memory_data: Dict) -> Dict[str, Any]:
    """Generiert Meta-Insights über alle Insights"""
    try:
        total_insights = sum(
            insight_category.get('insights_count', 0) 
            for insight_category in generated_insights.values()
            if isinstance(insight_category, dict)
        )
        
        high_significance_insights = []
        for category, insights_data in generated_insights.items():
            if isinstance(insights_data, dict) and 'insights' in insights_data:
                for insight in insights_data['insights']:
                    if insight.get('significance') == 'high':
                        high_significance_insights.append(insight)
        
        return {
            'total_insights_generated': total_insights,
            'high_significance_insights': len(high_significance_insights),
            'insight_distribution': {
                category: data.get('insights_count', 0) 
                for category, data in generated_insights.items()
                if isinstance(data, dict)
            },
            'overall_memory_health_indicator': _calculate_overall_health_from_insights(generated_insights),
            'priority_areas': _identify_priority_areas(generated_insights)
        }
        
    except Exception as e:
        logger.debug(f"Meta insights generation failed: {e}")
        return {'total_insights_generated': 0}

def _extract_key_findings(generated_insights: Dict) -> List[str]:
    """Extrahiert Key Findings aus allen Insights"""
    key_findings = []
    
    try:
        for category, insights_data in generated_insights.items():
            if isinstance(insights_data, dict) and 'insights' in insights_data:
                for insight in insights_data['insights']:
                    if insight.get('significance') in ['high', 'positive']:
                        key_findings.append(insight.get('description', ''))
        
        return key_findings[:5]  # Top 5 key findings
        
    except Exception as e:
        logger.debug(f"Key findings extraction failed: {e}")
        return ["Unable to extract key findings"]

def _generate_actionable_recommendations(generated_insights: Dict) -> List[str]:
    """Generiert Actionable Recommendations"""
    recommendations = []
    
    try:
        for category, insights_data in generated_insights.items():
            if isinstance(insights_data, dict) and 'insights' in insights_data:
                for insight in insights_data['insights']:
                    if insight.get('significance') in ['high', 'moderate']:
                        recommendation = insight.get('recommendation', '')
                        if recommendation and recommendation not in recommendations:
                            recommendations.append(recommendation)
        
        return recommendations[:7]  # Top 7 recommendations
        
    except Exception as e:
        logger.debug(f"Actionable recommendations generation failed: {e}")
        return ["Consider reviewing memory system performance"]

# Helper functions
def _calculate_temporal_consistency(temporal_distribution: Dict) -> float:
    if not temporal_distribution:
        return 0.5
    
    daily_counts = list(temporal_distribution.values())
    if len(daily_counts) < 2:
        return 0.5
    
    avg = statistics.mean(daily_counts)
    std_dev = statistics.stdev(daily_counts)
    
    consistency = max(0.0, 1.0 - (std_dev / max(avg, 1)))
    return min(1.0, consistency)

def _analyze_temporal_trend(temporal_distribution: Dict) -> str:
    if not temporal_distribution or len(temporal_distribution) < 3:
        return 'insufficient_data'
    
    sorted_dates = sorted(temporal_distribution.keys())
    recent_half = sorted_dates[len(sorted_dates)//2:]
    older_half = sorted_dates[:len(sorted_dates)//2]
    
    recent_avg = statistics.mean([temporal_distribution[date] for date in recent_half])
    older_avg = statistics.mean([temporal_distribution[date] for date in older_half])
    
    if recent_avg > older_avg * 1.2:
        return 'increasing'
    elif recent_avg < older_avg * 0.8:
        return 'decreasing'
    else:
        return 'stable'

def _determine_activity_level(temporal_distribution: Dict) -> str:
    if not temporal_distribution:
        return 'unknown'
    
    daily_counts = list(temporal_distribution.values())
    avg_daily = statistics.mean(daily_counts)
    
    if avg_daily > 10:
        return 'high'
    elif avg_daily > 3:
        return 'moderate'
    else:
        return 'low'

def _calculate_trend_stability(temporal_distribution: Dict) -> float:
    return _calculate_temporal_consistency(temporal_distribution)

def _calculate_content_optimization_score(content_lengths: List[int]) -> float:
    if not content_lengths:
        return 0.5
    
    avg_length = statistics.mean(content_lengths)
    std_dev = statistics.stdev(content_lengths) if len(content_lengths) > 1 else 0
    
    # Optimal range: 50-200 characters
    length_score = 1.0 - abs(avg_length - 125) / 125
    consistency_score = max(0.0, 1.0 - (std_dev / max(avg_length, 1)))
    
    return (length_score + consistency_score) / 2.0

def _classify_activity_pattern(temporal_distribution: Dict) -> str:
    if not temporal_distribution:
        return 'unknown'
    
    daily_counts = list(temporal_distribution.values())
    if not daily_counts:
        return 'unknown'
    
    std_dev = statistics.stdev(daily_counts) if len(daily_counts) > 1 else 0
    avg = statistics.mean(daily_counts)
    
    if std_dev / max(avg, 1) < 0.3:
        return 'consistent'
    elif std_dev / max(avg, 1) > 1.0:
        return 'irregular'
    else:
        return 'moderate'

def _calculate_behavioral_consistency(temporal_distribution: Dict) -> float:
    return _calculate_temporal_consistency(temporal_distribution)

def _calculate_overall_health_from_insights(generated_insights: Dict) -> str:
    positive_insights = 0
    negative_insights = 0
    
    for category, insights_data in generated_insights.items():
        if isinstance(insights_data, dict) and 'insights' in insights_data:
            for insight in insights_data['insights']:
                if insight.get('significance') == 'positive':
                    positive_insights += 1
                elif insight.get('significance') in ['high', 'moderate']:
                    negative_insights += 1
    
    if positive_insights > negative_insights:
        return 'good'
    elif negative_insights > positive_insights * 2:
        return 'concerning'
    else:
        return 'moderate'

def _identify_priority_areas(generated_insights: Dict) -> List[str]:
    priority_areas = []
    
    for category, insights_data in generated_insights.items():
        if isinstance(insights_data, dict) and 'insights' in insights_data:
            high_significance = [
                insight for insight in insights_data['insights'] 
                if insight.get('significance') == 'high'
            ]
            if high_significance:
                priority_areas.append(category.replace('_insights', ''))
    
    return priority_areas[:3]  # Top 3 priority areas

def _generate_fallback_insights() -> Dict[str, Any]:
    """Generiert Fallback Insights"""
    return {
        'pattern_insights': {
            'insights_count': 2,
            'insights': [
                {
                    'type': 'fallback_pattern',
                    'description': 'Standard memory formation patterns detected',
                    'significance': 'moderate',
                    'recommendation': 'Continue current memory management approach'
                }
            ]
        },
        'trend_insights': {
            'insights_count': 1,
            'insights': [
                {
                    'type': 'stable_trend',
                    'description': 'Memory formation appears stable',
                    'significance': 'positive',
                    'recommendation': 'Maintain current patterns'
                }
            ]
        },
        'fallback_mode': True
    }

__all__ = [
    'generate_memory_insights'
]