"""
Memory Efficiency Calculators
Berechnet verschiedene Memory Efficiency Metriken
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from .analysis_helpers import (
    collect_memory_data, calculate_memory_age_distribution,
    extract_memory_content_safe, calculate_content_complexity
)

logger = logging.getLogger(__name__)

def calculate_memory_efficiency(memory_manager=None, 
                              efficiency_metrics: List[str] = None) -> Dict[str, Any]:
    """Hauptfunktion für Memory Efficiency Berechnung"""
    try:
        if not memory_manager:
            return {
                'available': False,
                'reason': 'no_memory_manager',
                'fallback_efficiency': _generate_fallback_efficiency()
            }
        
        if efficiency_metrics is None:
            efficiency_metrics = ['storage', 'retrieval', 'consolidation', 'utilization']
        
        # Collect memory data
        memory_data = collect_memory_data(memory_manager, timedelta(days=7))
        
        # Calculate efficiency metrics
        efficiency_analysis = {}
        
        if 'storage' in efficiency_metrics:
            efficiency_analysis['storage_efficiency'] = _calculate_storage_efficiency(memory_data)
        
        if 'retrieval' in efficiency_metrics:
            efficiency_analysis['retrieval_efficiency'] = _calculate_retrieval_efficiency(memory_data)
        
        if 'consolidation' in efficiency_metrics:
            efficiency_analysis['consolidation_efficiency'] = _calculate_consolidation_efficiency(memory_data)
        
        if 'utilization' in efficiency_metrics:
            efficiency_analysis['utilization_efficiency'] = _calculate_utilization_efficiency(memory_data)
        
        # Overall efficiency score
        efficiency_scores = [v.get('efficiency_score', 0.5) for v in efficiency_analysis.values() 
                           if isinstance(v, dict)]
        overall_efficiency = statistics.mean(efficiency_scores) if efficiency_scores else 0.5
        
        return {
            'available': True,
            'overall_efficiency': overall_efficiency,
            'efficiency_rating': _rate_efficiency(overall_efficiency),
            'efficiency_analysis': efficiency_analysis,
            'analysis_metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'metrics_calculated': efficiency_metrics,
                'memories_analyzed': len(memory_data.get('all_memories', []))
            }
        }
        
    except Exception as e:
        logger.error(f"Memory efficiency calculation failed: {e}")
        return {'available': False, 'error': str(e)}

def _calculate_storage_efficiency(memory_data: Dict) -> Dict[str, Any]:
    """Berechnet Storage Efficiency - VEREINFACHT"""
    try:
        all_memories = memory_data.get('all_memories', [])
        
        if not all_memories:
            return {'efficiency_score': 0.5, 'storage_utilization': 'unknown'}
        
        # Simple storage efficiency metrics
        total_memories = len(all_memories)
        
        # Content complexity analysis
        content_complexities = []
        for memory in all_memories:
            content = extract_memory_content_safe(memory)
            complexity = calculate_content_complexity(content)
            content_complexities.append(complexity)
        
        avg_complexity = statistics.mean(content_complexities) if content_complexities else 0.5
        
        # Storage efficiency based on content density and organization
        # High complexity + reasonable count = good efficiency
        if total_memories > 50:
            count_factor = min(1.0, total_memories / 100.0)  # Normalize
        else:
            count_factor = total_memories / 50.0
        
        storage_efficiency = (avg_complexity + count_factor) / 2.0
        
        return {
            'efficiency_score': max(0.0, min(1.0, storage_efficiency)),
            'total_memories_stored': total_memories,
            'average_content_complexity': avg_complexity,
            'storage_utilization': 'high' if storage_efficiency > 0.7 else 'moderate' if storage_efficiency > 0.4 else 'low'
        }
        
    except Exception as e:
        logger.debug(f"Storage efficiency calculation failed: {e}")
        return {'efficiency_score': 0.5, 'error': str(e)}

def _calculate_retrieval_efficiency(memory_data: Dict) -> Dict[str, Any]:
    """Berechnet Retrieval Efficiency - VEREINFACHT"""
    try:
        all_memories = memory_data.get('all_memories', [])
        memory_types = memory_data.get('memory_types', {})
        
        if not all_memories:
            return {'efficiency_score': 0.5, 'retrieval_speed': 'unknown'}
        
        # Retrieval efficiency based on memory organization
        total_memories = len(all_memories)
        organized_memories = sum(len(memories) for memories in memory_types.values())
        
        organization_ratio = organized_memories / max(total_memories, 1)
        
        # Age distribution for retrieval efficiency
        age_distribution = calculate_memory_age_distribution(all_memories)
        recent_memories = age_distribution.get('very_recent', 0) + age_distribution.get('recent', 0)
        total_age_memories = sum(age_distribution.values())
        
        recency_factor = recent_memories / max(total_age_memories, 1)
        
        # Retrieval efficiency combines organization and recency
        retrieval_efficiency = (organization_ratio * 0.6) + (recency_factor * 0.4)
        
        return {
            'efficiency_score': max(0.0, min(1.0, retrieval_efficiency)),
            'organization_ratio': organization_ratio,
            'recent_memory_ratio': recency_factor,
            'retrieval_speed': 'fast' if retrieval_efficiency > 0.7 else 'moderate' if retrieval_efficiency > 0.4 else 'slow'
        }
        
    except Exception as e:
        logger.debug(f"Retrieval efficiency calculation failed: {e}")
        return {'efficiency_score': 0.5, 'error': str(e)}

def _calculate_consolidation_efficiency(memory_data: Dict) -> Dict[str, Any]:
    """Berechnet Consolidation Efficiency - VEREINFACHT"""
    try:
        memory_types = memory_data.get('memory_types', {})
        
        # Consolidation efficiency based on memory type distribution
        short_term_count = len(memory_types.get('short_term', []))
        long_term_count = len(memory_types.get('long_term', []))
        working_count = len(memory_types.get('working', []))
        
        total_count = short_term_count + long_term_count + working_count
        
        if total_count == 0:
            return {'efficiency_score': 0.5, 'consolidation_status': 'unknown'}
        
        # Good consolidation = healthy balance with progression to long-term
        long_term_ratio = long_term_count / total_count
        short_term_ratio = short_term_count / total_count
        
        # Ideal: some progression to long-term, but not all (need active short-term)
        consolidation_efficiency = long_term_ratio * 0.7 + min(short_term_ratio, 0.5) * 0.3
        
        return {
            'efficiency_score': max(0.0, min(1.0, consolidation_efficiency)),
            'long_term_ratio': long_term_ratio,
            'short_term_ratio': short_term_ratio,
            'working_memory_count': working_count,
            'consolidation_status': 'good' if consolidation_efficiency > 0.6 else 'moderate' if consolidation_efficiency > 0.3 else 'poor'
        }
        
    except Exception as e:
        logger.debug(f"Consolidation efficiency calculation failed: {e}")
        return {'efficiency_score': 0.5, 'error': str(e)}

def _calculate_utilization_efficiency(memory_data: Dict) -> Dict[str, Any]:
    """Berechnet Utilization Efficiency - VEREINFACHT"""
    try:
        all_memories = memory_data.get('all_memories', [])
        temporal_distribution = memory_data.get('temporal_distribution', {})
        
        if not all_memories:
            return {'efficiency_score': 0.5, 'utilization_rate': 'unknown'}
        
        # Utilization efficiency based on consistent memory usage
        total_memories = len(all_memories)
        
        # Temporal consistency
        if temporal_distribution:
            daily_counts = list(temporal_distribution.values())
            if len(daily_counts) > 1:
                avg_daily = statistics.mean(daily_counts)
                consistency = 1.0 - (statistics.stdev(daily_counts) / max(avg_daily, 1))
                consistency = max(0.0, min(1.0, consistency))
            else:
                consistency = 0.5
        else:
            consistency = 0.5
        
        # Memory count factor
        count_factor = min(1.0, total_memories / 50.0)  # Normalized to 50 memories
        
        # Utilization combines consistency and usage volume
        utilization_efficiency = (consistency * 0.7) + (count_factor * 0.3)
        
        return {
            'efficiency_score': max(0.0, min(1.0, utilization_efficiency)),
            'memory_count_factor': count_factor,
            'usage_consistency': consistency,
            'utilization_rate': 'high' if utilization_efficiency > 0.7 else 'moderate' if utilization_efficiency > 0.4 else 'low'
        }
        
    except Exception as e:
        logger.debug(f"Utilization efficiency calculation failed: {e}")
        return {'efficiency_score': 0.5, 'error': str(e)}

def _rate_efficiency(efficiency_score: float) -> str:
    """Bewertet Efficiency Score"""
    if efficiency_score > 0.8:
        return 'excellent'
    elif efficiency_score > 0.6:
        return 'good'
    elif efficiency_score > 0.4:
        return 'moderate'
    elif efficiency_score > 0.2:
        return 'poor'
    else:
        return 'very_poor'

def _generate_fallback_efficiency() -> Dict[str, Any]:
    """Generiert Fallback Efficiency Data"""
    return {
        'storage_efficiency': {'efficiency_score': 0.6, 'storage_utilization': 'moderate'},
        'retrieval_efficiency': {'efficiency_score': 0.7, 'retrieval_speed': 'moderate'},
        'consolidation_efficiency': {'efficiency_score': 0.5, 'consolidation_status': 'moderate'},
        'utilization_efficiency': {'efficiency_score': 0.6, 'utilization_rate': 'moderate'},
        'overall_efficiency': 0.6,
        'efficiency_rating': 'moderate',
        'fallback_mode': True
    }

__all__ = [
    'calculate_memory_efficiency'
]