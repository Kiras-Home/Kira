"""
Memory Pattern Analyzers
Spezialisierte Funktionen für Memory Pattern Analysis
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict

from .analysis_helpers import (
    collect_memory_data, extract_memory_timestamp_safe,
    classify_content_type, generate_fallback_patterns
)

logger = logging.getLogger(__name__)

def analyze_memory_patterns(memory_manager=None,
                          analysis_scope: str = 'comprehensive', 
                          time_window: Optional[timedelta] = None) -> Dict[str, Any]:
    """Hauptfunktion für Memory Pattern Analysis"""
    try:
        if not memory_manager:
            return {
                'available': False,
                'reason': 'no_memory_manager',
                'fallback_analysis': generate_fallback_patterns()
            }
        
        if time_window is None:
            time_window = timedelta(days=7)
        
        # Collect memory data
        memory_data = collect_memory_data(memory_manager, time_window)
        
        # Core pattern analysis
        pattern_analysis = {
            'temporal_patterns': analyze_temporal_patterns(memory_data, time_window),
            'content_patterns': analyze_content_patterns(memory_data),
            'usage_patterns': analyze_usage_patterns(memory_data)
        }
        
        # Comprehensive analysis if requested
        if analysis_scope == 'comprehensive':
            pattern_analysis.update({
                'learning_patterns': analyze_learning_patterns(memory_data),
                'consolidation_patterns': analyze_consolidation_patterns(memory_data),
                'retrieval_patterns': analyze_retrieval_patterns(memory_data)
            })
        
        return {
            'available': True,
            'analysis_scope': analysis_scope,
            'time_window_days': time_window.days,
            'pattern_analysis': pattern_analysis,
            'analysis_metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'memories_analyzed': len(memory_data.get('all_memories', []))
            }
        }
        
    except Exception as e:
        logger.error(f"Memory pattern analysis failed: {e}")
        return {'available': False, 'error': str(e)}

def analyze_temporal_patterns(memory_data: Dict, time_window: timedelta) -> Dict[str, Any]:
    """Analysiert zeitliche Muster - VEREINFACHT"""
    try:
        temporal_dist = memory_data.get('temporal_distribution', {})
        
        if not temporal_dist:
            return {
                'daily_patterns': {'peak_hours': [10, 14, 16]},
                'weekly_patterns': {'active_days': 5},
                'formation_rate': 0.5
            }
        
        # Einfache zeitliche Analyse
        daily_counts = list(temporal_dist.values())
        avg_daily = statistics.mean(daily_counts) if daily_counts else 0
        
        return {
            'daily_patterns': {
                'average_memories_per_day': avg_daily,
                'peak_days': max(daily_counts) if daily_counts else 0,
                'quiet_days': min(daily_counts) if daily_counts else 0
            },
            'formation_rate': avg_daily / 10.0,  # Normalized
            'activity_consistency': 1.0 - (statistics.stdev(daily_counts) / max(avg_daily, 1)) if len(daily_counts) > 1 else 0.5
        }
        
    except Exception as e:
        logger.debug(f"Temporal pattern analysis failed: {e}")
        return {'formation_rate': 0.5, 'error': str(e)}

def analyze_content_patterns(memory_data: Dict) -> Dict[str, Any]:
    """Analysiert Inhalts-Muster - VEREINFACHT"""
    try:
        all_memories = memory_data.get('all_memories', [])
        
        if not all_memories:
            return {
                'content_type_distribution': {'general': 10, 'learning': 5},
                'dominant_themes': {'interaction': 3, 'processing': 2}
            }
        
        # Content type classification
        content_types = Counter()
        content_lengths = []
        
        for memory in all_memories:
            content = memory.get('content', '')
            content_str = str(content)
            
            content_types[classify_content_type(content)] += 1
            content_lengths.append(len(content_str))
        
        return {
            'content_type_distribution': dict(content_types),
            'content_statistics': {
                'total_memories': len(all_memories),
                'average_content_length': statistics.mean(content_lengths) if content_lengths else 0,
                'content_variety': len(content_types)
            },
            'dominant_content_type': content_types.most_common(1)[0][0] if content_types else 'general'
        }
        
    except Exception as e:
        logger.debug(f"Content pattern analysis failed: {e}")
        return {'content_type_distribution': {}, 'error': str(e)}

def analyze_usage_patterns(memory_data: Dict) -> Dict[str, Any]:
    """Analysiert Nutzungs-Muster - VEREINFACHT"""
    try:
        all_memories = memory_data.get('all_memories', [])
        memory_types = memory_data.get('memory_types', {})
        
        # Memory type distribution
        type_counts = {mem_type: len(memories) for mem_type, memories in memory_types.items()}
        
        # Usage frequency estimation
        total_memories = len(all_memories)
        usage_intensity = 'high' if total_memories > 100 else 'moderate' if total_memories > 20 else 'low'
        
        return {
            'memory_type_distribution': type_counts,
            'usage_intensity': usage_intensity,
            'total_memories': total_memories,
            'primary_memory_system': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else 'unknown'
        }
        
    except Exception as e:
        logger.debug(f"Usage pattern analysis failed: {e}")
        return {'usage_intensity': 'moderate', 'error': str(e)}

def analyze_learning_patterns(memory_data: Dict) -> Dict[str, Any]:
    """Analysiert Lern-Muster - VEREINFACHT"""
    try:
        all_memories = memory_data.get('all_memories', [])
        
        # Count learning-related memories
        learning_memories = [
            memory for memory in all_memories
            if 'learn' in str(memory.get('content', '')).lower() or
               'understand' in str(memory.get('content', '')).lower() or
               'knowledge' in str(memory.get('content', '')).lower()
        ]
        
        learning_rate = len(learning_memories) / max(len(all_memories), 1)
        
        return {
            'learning_memory_count': len(learning_memories),
            'learning_rate': learning_rate,
            'learning_activity': 'high' if learning_rate > 0.3 else 'moderate' if learning_rate > 0.1 else 'low'
        }
        
    except Exception as e:
        logger.debug(f"Learning pattern analysis failed: {e}")
        return {'learning_rate': 0.2, 'error': str(e)}

def analyze_consolidation_patterns(memory_data: Dict) -> Dict[str, Any]:
    """Analysiert Konsolidierungs-Muster - VEREINFACHT"""
    try:
        memory_types = memory_data.get('memory_types', {})
        
        short_term_count = len(memory_types.get('short_term', []))
        long_term_count = len(memory_types.get('long_term', []))
        
        # Consolidation rate estimation
        if short_term_count + long_term_count > 0:
            consolidation_ratio = long_term_count / (short_term_count + long_term_count)
        else:
            consolidation_ratio = 0.5
        
        return {
            'consolidation_ratio': consolidation_ratio,
            'short_term_memories': short_term_count,
            'long_term_memories': long_term_count,
            'consolidation_health': 'good' if consolidation_ratio > 0.6 else 'moderate'
        }
        
    except Exception as e:
        logger.debug(f"Consolidation pattern analysis failed: {e}")
        return {'consolidation_ratio': 0.5, 'error': str(e)}

def analyze_retrieval_patterns(memory_data: Dict) -> Dict[str, Any]:
    """Analysiert Abruf-Muster - VEREINFACHT"""
    try:
        all_memories = memory_data.get('all_memories', [])
        
        # Simple retrieval pattern analysis
        return {
            'total_retrievable_memories': len(all_memories),
            'retrieval_complexity': 'low',  # Simplified assumption
            'access_patterns': 'sequential'  # Simplified assumption
        }
        
    except Exception as e:
        logger.debug(f"Retrieval pattern analysis failed: {e}")
        return {'retrieval_complexity': 'moderate', 'error': str(e)}

__all__ = [
    'analyze_memory_patterns',
    'analyze_temporal_patterns',
    'analyze_content_patterns',
    'analyze_usage_patterns',
    'analyze_learning_patterns',
    'analyze_consolidation_patterns',
    'analyze_retrieval_patterns'
]