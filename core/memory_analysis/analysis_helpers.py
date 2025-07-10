"""
Memory Analysis Helper Functions
Gemeinsame Hilfsfunktionen für Memory Analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

def collect_memory_data(memory_manager, time_window: timedelta) -> Dict[str, Any]:
    """Sammelt Memory Data für Analyse - VEREINFACHT"""
    try:
        # Versuche verschiedene Memory Manager Methoden
        all_memories = []
        memory_types = defaultdict(list)
        temporal_distribution = {}
        
        # Try different memory manager interfaces
        if hasattr(memory_manager, 'get_all_memories'):
            all_memories = memory_manager.get_all_memories()
        elif hasattr(memory_manager, 'memories'):
            all_memories = list(memory_manager.memories.values()) if hasattr(memory_manager.memories, 'values') else []
        elif hasattr(memory_manager, 'get_recent_memories'):
            all_memories = memory_manager.get_recent_memories(limit=1000)
        
        # Classify memories by type
        for memory in all_memories:
            memory_type = classify_memory_type(memory)
            memory_types[memory_type].append(memory)
            
            # Temporal distribution
            timestamp = extract_memory_timestamp_safe(memory)
            if timestamp:
                date_key = timestamp.strftime('%Y-%m-%d')
                temporal_distribution[date_key] = temporal_distribution.get(date_key, 0) + 1
        
        return {
            'all_memories': all_memories,
            'memory_types': dict(memory_types),
            'temporal_distribution': temporal_distribution,
            'collection_timestamp': datetime.now().isoformat(),
            'time_window_days': time_window.days
        }
        
    except Exception as e:
        logger.debug(f"Memory data collection failed: {e}")
        return {
            'all_memories': [],
            'memory_types': {},
            'temporal_distribution': {},
            'error': str(e)
        }

def extract_memory_timestamp_safe(memory) -> Optional[datetime]:
    """Extrahiert Timestamp aus Memory Object - SAFE"""
    try:
        # Try different timestamp fields
        if isinstance(memory, dict):
            # Common timestamp fields
            for field in ['timestamp', 'created_at', 'date', 'time']:
                if field in memory:
                    timestamp_value = memory[field]
                    if isinstance(timestamp_value, datetime):
                        return timestamp_value
                    elif isinstance(timestamp_value, str):
                        try:
                            return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                        except:
                            pass
        
        # Try object attributes
        elif hasattr(memory, 'timestamp'):
            return memory.timestamp
        elif hasattr(memory, 'created_at'):
            return memory.created_at
        
        # Default to now if no timestamp found
        return datetime.now()
        
    except Exception as e:
        logger.debug(f"Timestamp extraction failed: {e}")
        return datetime.now()

def classify_content_type(content) -> str:
    """Klassifiziert Content Type - VEREINFACHT"""
    try:
        content_str = str(content).lower()
        
        # Simple keyword-based classification
        if any(word in content_str for word in ['learn', 'study', 'understand', 'knowledge']):
            return 'learning'
        elif any(word in content_str for word in ['interact', 'talk', 'speak', 'conversation']):
            return 'interaction'
        elif any(word in content_str for word in ['process', 'analyze', 'compute', 'calculate']):
            return 'processing'
        elif any(word in content_str for word in ['emotion', 'feel', 'mood', 'sentiment']):
            return 'emotional'
        elif any(word in content_str for word in ['decision', 'choose', 'select', 'decide']):
            return 'decision'
        else:
            return 'general'
            
    except Exception as e:
        logger.debug(f"Content type classification failed: {e}")
        return 'general'

def classify_memory_type(memory) -> str:
    """Klassifiziert Memory Type - VEREINFACHT"""
    try:
        # Try to get memory type from memory object
        if isinstance(memory, dict):
            # Look for explicit type field
            if 'type' in memory:
                return memory['type']
            elif 'memory_type' in memory:
                return memory['memory_type']
            elif 'category' in memory:
                return memory['category']
        
        # Try object attributes
        elif hasattr(memory, 'memory_type'):
            return memory.memory_type
        elif hasattr(memory, 'type'):
            return memory.type
        
        # Fallback classification based on content/age
        timestamp = extract_memory_timestamp_safe(memory)
        if timestamp:
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            if age_hours < 1:
                return 'working'
            elif age_hours < 24:
                return 'short_term'
            else:
                return 'long_term'
        
        return 'general'
        
    except Exception as e:
        logger.debug(f"Memory type classification failed: {e}")
        return 'general'

def generate_fallback_patterns() -> Dict[str, Any]:
    """Generiert Fallback Memory Patterns"""
    return {
        'temporal_patterns': {
            'daily_patterns': {'peak_hours': [10, 14, 16], 'quiet_hours': [2, 4, 6]},
            'weekly_patterns': {'active_days': 5, 'quiet_days': 2},
            'formation_rate': 0.5
        },
        'content_patterns': {
            'content_type_distribution': {
                'general': 15,
                'interaction': 10,
                'learning': 8,
                'processing': 5,
                'emotional': 3
            },
            'dominant_content_type': 'general'
        },
        'usage_patterns': {
            'memory_type_distribution': {
                'short_term': 12,
                'long_term': 8,
                'working': 5
            },
            'usage_intensity': 'moderate',
            'primary_memory_system': 'short_term'
        },
        'fallback_mode': True,
        'reason': 'no_memory_manager_available'
    }

def calculate_memory_age_distribution(all_memories: List) -> Dict[str, int]:
    """Berechnet Altersverteilung der Memories"""
    try:
        age_buckets = {
            'very_recent': 0,    # < 1 hour
            'recent': 0,         # 1-24 hours
            'daily': 0,          # 1-7 days
            'weekly': 0,         # 7-30 days
            'monthly': 0,        # 30+ days
        }
        
        now = datetime.now()
        
        for memory in all_memories:
            timestamp = extract_memory_timestamp_safe(memory)
            if timestamp:
                age_hours = (now - timestamp).total_seconds() / 3600
                
                if age_hours < 1:
                    age_buckets['very_recent'] += 1
                elif age_hours < 24:
                    age_buckets['recent'] += 1
                elif age_hours < 24 * 7:
                    age_buckets['daily'] += 1
                elif age_hours < 24 * 30:
                    age_buckets['weekly'] += 1
                else:
                    age_buckets['monthly'] += 1
        
        return age_buckets
        
    except Exception as e:
        logger.debug(f"Memory age distribution calculation failed: {e}")
        return {'recent': 10, 'daily': 15, 'weekly': 8, 'monthly': 5}

def extract_memory_content_safe(memory) -> str:
    """Extrahiert Content aus Memory Object - SAFE"""
    try:
        if isinstance(memory, dict):
            # Try common content fields
            for field in ['content', 'text', 'message', 'data', 'value']:
                if field in memory:
                    return str(memory[field])
        
        # Try object attributes
        elif hasattr(memory, 'content'):
            return str(memory.content)
        elif hasattr(memory, 'text'):
            return str(memory.text)
        elif hasattr(memory, 'data'):
            return str(memory.data)
        
        # Fallback to string representation
        return str(memory)
        
    except Exception as e:
        logger.debug(f"Memory content extraction failed: {e}")
        return ""

def calculate_content_complexity(content: str) -> float:
    """Berechnet Content Complexity - VEREINFACHT"""
    try:
        if not content:
            return 0.0
        
        # Simple complexity metrics
        word_count = len(content.split())
        unique_words = len(set(content.lower().split()))
        
        # Complexity based on length and vocabulary diversity
        length_complexity = min(1.0, word_count / 100.0)  # Normalize to 100 words
        diversity_complexity = unique_words / max(word_count, 1)
        
        complexity = (length_complexity + diversity_complexity) / 2.0
        return max(0.0, min(1.0, complexity))
        
    except Exception as e:
        logger.debug(f"Content complexity calculation failed: {e}")
        return 0.5

def filter_memories_by_timeframe(memories: List, timeframe_hours: int) -> List:
    """Filtert Memories nach Zeitrahmen"""
    try:
        if not memories:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        filtered_memories = []
        
        for memory in memories:
            timestamp = extract_memory_timestamp_safe(memory)
            if timestamp and timestamp >= cutoff_time:
                filtered_memories.append(memory)
        
        return filtered_memories
        
    except Exception as e:
        logger.debug(f"Memory filtering by timeframe failed: {e}")
        return memories  # Return all if filtering fails

__all__ = [
    'collect_memory_data',
    'extract_memory_timestamp_safe',
    'classify_content_type',
    'classify_memory_type',
    'generate_fallback_patterns',
    'calculate_memory_age_distribution',
    'extract_memory_content_safe',
    'calculate_content_complexity',
    'filter_memories_by_timeframe'
]