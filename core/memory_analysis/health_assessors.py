"""
Memory Health Assessors
Bewertet die Gesundheit und Stabilität des Memory Systems
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from .analysis_helpers import (
    collect_memory_data, calculate_memory_age_distribution,
    extract_memory_content_safe, filter_memories_by_timeframe
)

logger = logging.getLogger(__name__)

def assess_memory_health(memory_manager=None, 
                        health_checks: List[str] = None) -> Dict[str, Any]:
    """Hauptfunktion für Memory Health Assessment"""
    try:
        if not memory_manager:
            return {
                'available': False,
                'reason': 'no_memory_manager',
                'fallback_health': _generate_fallback_health_assessment()
            }
        
        if health_checks is None:
            health_checks = ['integrity', 'stability', 'performance', 'capacity']
        
        # Collect memory data
        memory_data = collect_memory_data(memory_manager, timedelta(days=30))
        
        # Perform health assessments
        health_assessment = {}
        
        if 'integrity' in health_checks:
            health_assessment['integrity_check'] = _assess_memory_integrity(memory_data)
        
        if 'stability' in health_checks:
            health_assessment['stability_check'] = _assess_memory_stability(memory_data)
        
        if 'performance' in health_checks:
            health_assessment['performance_check'] = _assess_memory_performance(memory_data)
        
        if 'capacity' in health_checks:
            health_assessment['capacity_check'] = _assess_memory_capacity(memory_data)
        
        if 'fragmentation' in health_checks:
            health_assessment['fragmentation_check'] = _assess_memory_fragmentation(memory_data)
        
        # Overall health score
        health_scores = [v.get('health_score', 0.5) for v in health_assessment.values() 
                        if isinstance(v, dict)]
        overall_health = statistics.mean(health_scores) if health_scores else 0.5
        
        return {
            'available': True,
            'overall_health_score': overall_health,
            'health_status': _determine_health_status(overall_health),
            'health_assessment': health_assessment,
            'recommendations': _generate_health_recommendations(health_assessment, overall_health),
            'assessment_metadata': {
                'assessment_timestamp': datetime.now().isoformat(),
                'checks_performed': health_checks,
                'memories_analyzed': len(memory_data.get('all_memories', []))
            }
        }
        
    except Exception as e:
        logger.error(f"Memory health assessment failed: {e}")
        return {'available': False, 'error': str(e)}

def _assess_memory_integrity(memory_data: Dict) -> Dict[str, Any]:
    """Bewertet Memory Integrity - VEREINFACHT"""
    try:
        all_memories = memory_data.get('all_memories', [])
        
        if not all_memories:
            return {'health_score': 0.5, 'integrity_status': 'unknown'}
        
        # Integrity checks
        valid_memories = 0
        corrupted_memories = 0
        incomplete_memories = 0
        
        for memory in all_memories:
            try:
                # Check if memory has essential components
                content = extract_memory_content_safe(memory)
                has_content = bool(content and len(content.strip()) > 0)
                
                # Check for timestamp
                if isinstance(memory, dict):
                    has_timestamp = any(key in memory for key in ['timestamp', 'created_at', 'date'])
                else:
                    has_timestamp = hasattr(memory, 'timestamp') or hasattr(memory, 'created_at')
                
                if has_content and has_timestamp:
                    valid_memories += 1
                elif has_content or has_timestamp:
                    incomplete_memories += 1
                else:
                    corrupted_memories += 1
                    
            except Exception as e:
                corrupted_memories += 1
                logger.debug(f"Memory integrity check failed for memory: {e}")
        
        total_memories = len(all_memories)
        integrity_ratio = valid_memories / total_memories if total_memories > 0 else 0
        
        return {
            'health_score': integrity_ratio,
            'integrity_status': _rate_integrity(integrity_ratio),
            'valid_memories': valid_memories,
            'corrupted_memories': corrupted_memories,
            'incomplete_memories': incomplete_memories,
            'total_memories_checked': total_memories
        }
        
    except Exception as e:
        logger.debug(f"Memory integrity assessment failed: {e}")
        return {'health_score': 0.5, 'integrity_status': 'unknown', 'error': str(e)}

def _assess_memory_stability(memory_data: Dict) -> Dict[str, Any]:
    """Bewertet Memory Stability - VEREINFACHT"""
    try:
        temporal_distribution = memory_data.get('temporal_distribution', {})
        
        if not temporal_distribution:
            return {'health_score': 0.5, 'stability_status': 'unknown'}
        
        # Stability based on consistent memory formation
        daily_counts = list(temporal_distribution.values())
        
        if len(daily_counts) < 2:
            return {'health_score': 0.5, 'stability_status': 'insufficient_data'}
        
        # Calculate stability metrics
        avg_daily = statistics.mean(daily_counts)
        std_dev = statistics.stdev(daily_counts)
        
        # Stability = low variation relative to mean
        if avg_daily > 0:
            coefficient_of_variation = std_dev / avg_daily
            stability_score = max(0.0, 1.0 - coefficient_of_variation)
        else:
            stability_score = 0.0
        
        # Check for concerning patterns
        zero_days = daily_counts.count(0)
        high_variation_days = len([count for count in daily_counts if abs(count - avg_daily) > 2 * std_dev])
        
        return {
            'health_score': max(0.0, min(1.0, stability_score)),
            'stability_status': _rate_stability(stability_score),
            'average_daily_memories': avg_daily,
            'variation_coefficient': coefficient_of_variation if avg_daily > 0 else 0,
            'zero_activity_days': zero_days,
            'high_variation_days': high_variation_days
        }
        
    except Exception as e:
        logger.debug(f"Memory stability assessment failed: {e}")
        return {'health_score': 0.5, 'stability_status': 'unknown', 'error': str(e)}

def _assess_memory_performance(memory_data: Dict) -> Dict[str, Any]:
    """Bewertet Memory Performance - VEREINFACHT"""
    try:
        all_memories = memory_data.get('all_memories', [])
        memory_types = memory_data.get('memory_types', {})
        
        if not all_memories:
            return {'health_score': 0.5, 'performance_status': 'unknown'}
        
        # Performance metrics
        total_memories = len(all_memories)
        
        # Age distribution for performance assessment
        age_distribution = calculate_memory_age_distribution(all_memories)
        
        # Recent memory activity indicates good performance
        recent_activity = age_distribution.get('very_recent', 0) + age_distribution.get('recent', 0)
        recent_ratio = recent_activity / total_memories if total_memories > 0 else 0
        
        # Memory type distribution efficiency
        type_diversity = len(memory_types)
        type_balance = min(1.0, type_diversity / 3.0)  # Ideal: 3+ types
        
        # Content processing efficiency (simplified)
        content_lengths = []
        for memory in all_memories[:100]:  # Sample first 100 for performance
            content = extract_memory_content_safe(memory)
            content_lengths.append(len(content))
        
        avg_content_length = statistics.mean(content_lengths) if content_lengths else 0
        content_efficiency = min(1.0, avg_content_length / 100.0)  # Normalize to 100 chars
        
        # Overall performance score
        performance_score = (recent_ratio * 0.4) + (type_balance * 0.3) + (content_efficiency * 0.3)
        
        return {
            'health_score': max(0.0, min(1.0, performance_score)),
            'performance_status': _rate_performance(performance_score),
            'recent_activity_ratio': recent_ratio,
            'memory_type_diversity': type_diversity,
            'average_content_length': avg_content_length,
            'total_memories_processed': total_memories
        }
        
    except Exception as e:
        logger.debug(f"Memory performance assessment failed: {e}")
        return {'health_score': 0.5, 'performance_status': 'unknown', 'error': str(e)}

def _assess_memory_capacity(memory_data: Dict) -> Dict[str, Any]:
    """Bewertet Memory Capacity - VEREINFACHT"""
    try:
        all_memories = memory_data.get('all_memories', [])
        memory_types = memory_data.get('memory_types', {})
        
        total_memories = len(all_memories)
        
        # Capacity assessment based on memory distribution
        working_memories = len(memory_types.get('working', []))
        short_term_memories = len(memory_types.get('short_term', []))
        long_term_memories = len(memory_types.get('long_term', []))
        
        # Capacity utilization
        # Assume healthy limits: Working < 10, Short-term < 50, Long-term unlimited
        working_utilization = min(1.0, working_memories / 10.0)
        short_term_utilization = min(1.0, short_term_memories / 50.0)
        
        # Capacity health: good utilization without overload
        if working_utilization > 0.9 or short_term_utilization > 0.9:
            capacity_health = 0.4  # Near overload
        elif working_utilization > 0.7 or short_term_utilization > 0.7:
            capacity_health = 0.6  # High utilization
        else:
            capacity_health = 0.8  # Healthy utilization
        
        # Adjust for total memory count
        if total_memories > 1000:
            capacity_health *= 0.9  # Slight penalty for very high counts
        
        return {
            'health_score': max(0.0, min(1.0, capacity_health)),
            'capacity_status': _rate_capacity(capacity_health),
            'total_memory_count': total_memories,
            'working_memory_utilization': working_utilization,
            'short_term_utilization': short_term_utilization,
            'long_term_memory_count': long_term_memories,
            'capacity_distribution': {
                'working': working_memories,
                'short_term': short_term_memories,
                'long_term': long_term_memories
            }
        }
        
    except Exception as e:
        logger.debug(f"Memory capacity assessment failed: {e}")
        return {'health_score': 0.5, 'capacity_status': 'unknown', 'error': str(e)}

def _assess_memory_fragmentation(memory_data: Dict) -> Dict[str, Any]:
    """Bewertet Memory Fragmentation - VEREINFACHT"""
    try:
        all_memories = memory_data.get('all_memories', [])
        temporal_distribution = memory_data.get('temporal_distribution', {})
        
        if not all_memories:
            return {'health_score': 0.5, 'fragmentation_status': 'unknown'}
        
        # Fragmentation assessment based on temporal gaps
        if not temporal_distribution:
            return {'health_score': 0.5, 'fragmentation_status': 'insufficient_temporal_data'}
        
        # Check for temporal gaps (days with no memory formation)
        sorted_dates = sorted(temporal_distribution.keys())
        if len(sorted_dates) < 2:
            return {'health_score': 0.7, 'fragmentation_status': 'insufficient_data'}
        
        # Count gaps between memory formation dates
        gaps = []
        for i in range(1, len(sorted_dates)):
            try:
                date1 = datetime.strptime(sorted_dates[i-1], '%Y-%m-%d')
                date2 = datetime.strptime(sorted_dates[i], '%Y-%m-%d')
                gap_days = (date2 - date1).days - 1
                if gap_days > 0:
                    gaps.append(gap_days)
            except ValueError:
                continue
        
        # Fragmentation score based on gaps
        if not gaps:
            fragmentation_score = 0.9  # No gaps = low fragmentation
        else:
            avg_gap = statistics.mean(gaps)
            max_gap = max(gaps)
            
            # Penalize large gaps
            fragmentation_score = max(0.1, 1.0 - (avg_gap / 7.0) - (max_gap / 30.0))
        
        return {
            'health_score': max(0.0, min(1.0, fragmentation_score)),
            'fragmentation_status': _rate_fragmentation(fragmentation_score),
            'temporal_gaps_count': len(gaps),
            'average_gap_days': statistics.mean(gaps) if gaps else 0,
            'max_gap_days': max(gaps) if gaps else 0,
            'continuous_days': len(sorted_dates)
        }
        
    except Exception as e:
        logger.debug(f"Memory fragmentation assessment failed: {e}")
        return {'health_score': 0.5, 'fragmentation_status': 'unknown', 'error': str(e)}

# Rating functions
def _rate_integrity(score: float) -> str:
    if score > 0.9: return 'excellent'
    elif score > 0.7: return 'good'
    elif score > 0.5: return 'moderate'
    elif score > 0.3: return 'poor'
    else: return 'critical'

def _rate_stability(score: float) -> str:
    if score > 0.8: return 'very_stable'
    elif score > 0.6: return 'stable'
    elif score > 0.4: return 'moderate'
    elif score > 0.2: return 'unstable'
    else: return 'very_unstable'

def _rate_performance(score: float) -> str:
    if score > 0.8: return 'high_performance'
    elif score > 0.6: return 'good_performance'
    elif score > 0.4: return 'moderate_performance'
    elif score > 0.2: return 'poor_performance'
    else: return 'very_poor_performance'

def _rate_capacity(score: float) -> str:
    if score > 0.8: return 'healthy_capacity'
    elif score > 0.6: return 'moderate_capacity'
    elif score > 0.4: return 'high_utilization'
    elif score > 0.2: return 'near_capacity'
    else: return 'over_capacity'

def _rate_fragmentation(score: float) -> str:
    if score > 0.8: return 'low_fragmentation'
    elif score > 0.6: return 'moderate_fragmentation'
    elif score > 0.4: return 'high_fragmentation'
    else: return 'severe_fragmentation'

def _determine_health_status(overall_health: float) -> str:
    """Bestimmt Overall Health Status"""
    if overall_health > 0.8:
        return 'excellent_health'
    elif overall_health > 0.6:
        return 'good_health'
    elif overall_health > 0.4:
        return 'moderate_health'
    elif overall_health > 0.2:
        return 'poor_health'
    else:
        return 'critical_health'

def _generate_health_recommendations(health_assessment: Dict, overall_health: float) -> List[str]:
    """Generiert Health Recommendations"""
    recommendations = []
    
    try:
        # Integrity recommendations
        integrity_check = health_assessment.get('integrity_check', {})
        if integrity_check.get('health_score', 0.5) < 0.6:
            recommendations.append("Consider memory validation and cleanup procedures")
        
        # Stability recommendations
        stability_check = health_assessment.get('stability_check', {})
        if stability_check.get('health_score', 0.5) < 0.6:
            recommendations.append("Implement more consistent memory formation patterns")
        
        # Performance recommendations
        performance_check = health_assessment.get('performance_check', {})
        if performance_check.get('health_score', 0.5) < 0.6:
            recommendations.append("Optimize memory processing and retrieval mechanisms")
        
        # Capacity recommendations
        capacity_check = health_assessment.get('capacity_check', {})
        if capacity_check.get('health_score', 0.5) < 0.6:
            recommendations.append("Review memory capacity limits and consolidation strategies")
        
        # Fragmentation recommendations
        fragmentation_check = health_assessment.get('fragmentation_check', {})
        if fragmentation_check.get('health_score', 0.5) < 0.6:
            recommendations.append("Address memory fragmentation through better temporal organization")
        
        # Overall health recommendations
        if overall_health < 0.4:
            recommendations.append("URGENT: Comprehensive memory system review required")
        elif overall_health < 0.6:
            recommendations.append("Consider implementing memory optimization procedures")
        
        return recommendations if recommendations else ["Memory system appears healthy"]
        
    except Exception as e:
        logger.debug(f"Health recommendations generation failed: {e}")
        return ["Unable to generate specific recommendations"]

def _generate_fallback_health_assessment() -> Dict[str, Any]:
    """Generiert Fallback Health Assessment"""
    return {
        'integrity_check': {'health_score': 0.7, 'integrity_status': 'good'},
        'stability_check': {'health_score': 0.6, 'stability_status': 'stable'},
        'performance_check': {'health_score': 0.6, 'performance_status': 'good_performance'},
        'capacity_check': {'health_score': 0.7, 'capacity_status': 'healthy_capacity'},
        'overall_health_score': 0.65,
        'health_status': 'good_health',
        'recommendations': ["No specific recommendations - fallback assessment"],
        'fallback_mode': True
    }

__all__ = [
    'assess_memory_health'
]