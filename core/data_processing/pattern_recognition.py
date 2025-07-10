"""
Pattern Recognition Module
Erkennt Muster, Trends und Anomalien in verschiedenen Datentypen
"""

import logging
import statistics
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Pattern:
    """Erkanntes Muster"""
    pattern_type: str
    pattern_id: str
    confidence: float
    description: str
    data_points: List[Any]
    metadata: Dict[str, Any]
    detected_at: str

@dataclass
class Trend:
    """Erkannter Trend"""
    trend_type: str  # increasing, decreasing, stable, cyclical
    trend_strength: float  # 0.0 to 1.0
    time_span: str
    start_value: Any
    end_value: Any
    trend_data: List[Any]
    confidence: float

@dataclass
class Anomaly:
    """Erkannte Anomalie"""
    anomaly_type: str
    severity: str  # low, medium, high, critical
    data_point: Any
    expected_range: Tuple[float, float]
    actual_value: float
    deviation_score: float
    context: Dict[str, Any]

def detect_patterns(data: Any, pattern_types: List[str] = None, 
                   sensitivity: float = 0.7) -> Dict[str, Any]:
    """
    Hauptfunktion für Mustererkennung
    """
    try:
        if pattern_types is None:
            pattern_types = ['sequence', 'frequency', 'clustering', 'temporal', 'behavioral']
        
        detection_result = {
            'detection_timestamp': datetime.now().isoformat(),
            'data_summary': _get_pattern_data_summary(data),
            'patterns_detected': [],
            'detection_settings': {
                'pattern_types': pattern_types,
                'sensitivity': sensitivity
            }
        }
        
        patterns = []
        
        # Sequence Pattern Detection
        if 'sequence' in pattern_types:
            sequence_patterns = _detect_sequence_patterns(data, sensitivity)
            patterns.extend(sequence_patterns)
        
        # Frequency Pattern Detection
        if 'frequency' in pattern_types:
            frequency_patterns = _detect_frequency_patterns(data, sensitivity)
            patterns.extend(frequency_patterns)
        
        # Clustering Pattern Detection
        if 'clustering' in pattern_types:
            clustering_patterns = _detect_clustering_patterns(data, sensitivity)
            patterns.extend(clustering_patterns)
        
        # Temporal Pattern Detection
        if 'temporal' in pattern_types:
            temporal_patterns = _detect_temporal_patterns(data, sensitivity)
            patterns.extend(temporal_patterns)
        
        # Behavioral Pattern Detection
        if 'behavioral' in pattern_types:
            behavioral_patterns = _detect_behavioral_patterns(data, sensitivity)
            patterns.extend(behavioral_patterns)
        
        detection_result['patterns_detected'] = [pattern.__dict__ for pattern in patterns]
        detection_result['pattern_summary'] = {
            'total_patterns': len(patterns),
            'high_confidence_patterns': len([p for p in patterns if p.confidence > 0.8]),
            'pattern_types_found': list(set(p.pattern_type for p in patterns))
        }
        
        return detection_result
        
    except Exception as e:
        logger.error(f"Pattern detection failed: {e}")
        return {
            'patterns_detected': [],
            'error': str(e),
            'detection_timestamp': datetime.now().isoformat()
        }

def detect_trends(data: Any, trend_window: int = 10, 
                 trend_threshold: float = 0.1) -> Dict[str, Any]:
    """
    Erkennt Trends in Daten
    """
    try:
        trend_result = {
            'detection_timestamp': datetime.now().isoformat(),
            'trends_detected': [],
            'trend_settings': {
                'window_size': trend_window,
                'threshold': trend_threshold
            }
        }
        
        trends = []
        
        # Numeric Trend Detection
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            numeric_trends = _detect_numeric_trends(data, trend_window, trend_threshold)
            trends.extend(numeric_trends)
        
        # Time Series Trend Detection
        elif _is_time_series_data(data):
            timeseries_trends = _detect_timeseries_trends(data, trend_window, trend_threshold)
            trends.extend(timeseries_trends)
        
        # Categorical Trend Detection
        elif isinstance(data, list):
            categorical_trends = _detect_categorical_trends(data, trend_window)
            trends.extend(categorical_trends)
        
        # Volume/Frequency Trends
        volume_trends = _detect_volume_trends(data, trend_window)
        trends.extend(volume_trends)
        
        trend_result['trends_detected'] = [trend.__dict__ for trend in trends]
        trend_result['trend_summary'] = {
            'total_trends': len(trends),
            'strong_trends': len([t for t in trends if t.trend_strength > 0.7]),
            'trend_types': list(set(t.trend_type for t in trends))
        }
        
        return trend_result
        
    except Exception as e:
        logger.error(f"Trend detection failed: {e}")
        return {
            'trends_detected': [],
            'error': str(e),
            'detection_timestamp': datetime.now().isoformat()
        }

def detect_anomalies(data: Any, anomaly_threshold: float = 2.0,
                    context_window: int = 10) -> Dict[str, Any]:
    """
    Erkennt Anomalien in Daten
    """
    try:
        anomaly_result = {
            'detection_timestamp': datetime.now().isoformat(),
            'anomalies_detected': [],
            'anomaly_settings': {
                'threshold': anomaly_threshold,
                'context_window': context_window
            }
        }
        
        anomalies = []
        
        # Statistical Anomalies (für numerische Daten)
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            statistical_anomalies = _detect_statistical_anomalies(data, anomaly_threshold)
            anomalies.extend(statistical_anomalies)
        
        # Pattern-based Anomalies
        pattern_anomalies = _detect_pattern_anomalies(data, context_window)
        anomalies.extend(pattern_anomalies)
        
        # Frequency Anomalies
        frequency_anomalies = _detect_frequency_anomalies(data, anomaly_threshold)
        anomalies.extend(frequency_anomalies)
        
        # Behavioral Anomalies
        behavioral_anomalies = _detect_behavioral_anomalies(data, context_window)
        anomalies.extend(behavioral_anomalies)
        
        anomaly_result['anomalies_detected'] = [anomaly.__dict__ for anomaly in anomalies]
        anomaly_result['anomaly_summary'] = {
            'total_anomalies': len(anomalies),
            'critical_anomalies': len([a for a in anomalies if a.severity == 'critical']),
            'anomaly_types': list(set(a.anomaly_type for a in anomalies))
        }
        
        return anomaly_result
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        return {
            'anomalies_detected': [],
            'error': str(e),
            'detection_timestamp': datetime.now().isoformat()
        }

def analyze_data_relationships(data: Any, relationship_types: List[str] = None) -> Dict[str, Any]:
    """
    Analysiert Beziehungen zwischen Datenpunkten
    """
    try:
        if relationship_types is None:
            relationship_types = ['correlation', 'causation', 'dependency', 'similarity']
        
        relationship_result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'relationships_found': {},
            'relationship_settings': {
                'types_analyzed': relationship_types
            }
        }
        
        # Correlation Analysis
        if 'correlation' in relationship_types:
            correlations = _analyze_correlations(data)
            relationship_result['relationships_found']['correlations'] = correlations
        
        # Causation Analysis (simplified)
        if 'causation' in relationship_types:
            causations = _analyze_causations(data)
            relationship_result['relationships_found']['causations'] = causations
        
        # Dependency Analysis
        if 'dependency' in relationship_types:
            dependencies = _analyze_dependencies(data)
            relationship_result['relationships_found']['dependencies'] = dependencies
        
        # Similarity Analysis
        if 'similarity' in relationship_types:
            similarities = _analyze_similarities(data)
            relationship_result['relationships_found']['similarities'] = similarities
        
        relationship_result['relationship_summary'] = {
            'total_relationships': sum(len(v) if isinstance(v, list) else 1 
                                     for v in relationship_result['relationships_found'].values()),
            'relationship_types_found': list(relationship_result['relationships_found'].keys())
        }
        
        return relationship_result
        
    except Exception as e:
        logger.error(f"Relationship analysis failed: {e}")
        return {
            'relationships_found': {},
            'error': str(e),
            'analysis_timestamp': datetime.now().isoformat()
        }

def find_recurring_patterns(data: Any, min_occurrences: int = 3,
                          pattern_length_range: Tuple[int, int] = (2, 10)) -> Dict[str, Any]:
    """
    Findet wiederkehrende Muster
    """
    try:
        recurring_result = {
            'detection_timestamp': datetime.now().isoformat(),
            'recurring_patterns': [],
            'detection_settings': {
                'min_occurrences': min_occurrences,
                'pattern_length_range': pattern_length_range
            }
        }
        
        recurring_patterns = []
        
        # Sequence-based recurring patterns
        if isinstance(data, list):
            sequence_recurring = _find_recurring_sequences(data, min_occurrences, pattern_length_range)
            recurring_patterns.extend(sequence_recurring)
        
        # Time-based recurring patterns
        if _is_time_series_data(data):
            temporal_recurring = _find_recurring_temporal_patterns(data, min_occurrences)
            recurring_patterns.extend(temporal_recurring)
        
        # Frequency-based recurring patterns
        frequency_recurring = _find_recurring_frequency_patterns(data, min_occurrences)
        recurring_patterns.extend(frequency_recurring)
        
        recurring_result['recurring_patterns'] = [pattern.__dict__ for pattern in recurring_patterns]
        recurring_result['pattern_summary'] = {
            'total_recurring_patterns': len(recurring_patterns),
            'high_frequency_patterns': len([p for p in recurring_patterns if p.confidence > 0.8]),
            'pattern_types': list(set(p.pattern_type for p in recurring_patterns))
        }
        
        return recurring_result
        
    except Exception as e:
        logger.error(f"Recurring pattern detection failed: {e}")
        return {
            'recurring_patterns': [],
            'error': str(e),
            'detection_timestamp': datetime.now().isoformat()
        }

# Helper Functions für Pattern Detection

def _get_pattern_data_summary(data: Any) -> Dict[str, Any]:
    """Erstellt Data Summary für Pattern Detection"""
    try:
        return {
            'data_type': type(data).__name__,
            'data_size': len(data) if hasattr(data, '__len__') else 1,
            'is_numeric': isinstance(data, list) and all(isinstance(x, (int, float)) for x in data),
            'is_temporal': _is_time_series_data(data),
            'has_structure': isinstance(data, (list, dict))
        }
    except:
        return {'data_type': 'unknown', 'data_size': 0}

def _is_time_series_data(data: Any) -> bool:
    """Prüft ob Daten Time Series sind"""
    try:
        if isinstance(data, list) and data:
            # Check if items have timestamp-like attributes
            first_item = data[0]
            return (hasattr(first_item, 'timestamp') or 
                   hasattr(first_item, 'created_at') or
                   hasattr(first_item, 'time') or
                   (isinstance(first_item, dict) and 
                    any('time' in str(k).lower() or 'date' in str(k).lower() 
                        for k in first_item.keys())))
        return False
    except:
        return False

# Sequence Pattern Detection
def _detect_sequence_patterns(data: Any, sensitivity: float) -> List[Pattern]:
    """Erkennt Sequenz-Muster"""
    try:
        patterns = []
        
        if isinstance(data, list) and len(data) >= 3:
            # Arithmetic sequences
            arithmetic_patterns = _find_arithmetic_sequences(data, sensitivity)
            patterns.extend(arithmetic_patterns)
            
            # Repeating subsequences
            repeating_patterns = _find_repeating_subsequences(data, sensitivity)
            patterns.extend(repeating_patterns)
            
            # Alternating patterns
            alternating_patterns = _find_alternating_patterns(data, sensitivity)
            patterns.extend(alternating_patterns)
        
        return patterns
    except:
        return []

def _find_arithmetic_sequences(data: List, sensitivity: float) -> List[Pattern]:
    """Findet arithmetische Sequenzen"""
    patterns = []
    try:
        if all(isinstance(x, (int, float)) for x in data) and len(data) >= 3:
            differences = [data[i+1] - data[i] for i in range(len(data)-1)]
            
            # Check for consistent differences
            if len(set(differences)) == 1:  # Perfect arithmetic sequence
                patterns.append(Pattern(
                    pattern_type='arithmetic_sequence',
                    pattern_id=f'arith_seq_{datetime.now().timestamp()}',
                    confidence=1.0,
                    description=f'Arithmetic sequence with difference {differences[0]}',
                    data_points=data,
                    metadata={'difference': differences[0], 'length': len(data)},
                    detected_at=datetime.now().isoformat()
                ))
            elif statistics.stdev(differences) < sensitivity:  # Nearly arithmetic
                patterns.append(Pattern(
                    pattern_type='near_arithmetic_sequence',
                    pattern_id=f'near_arith_seq_{datetime.now().timestamp()}',
                    confidence=1.0 - statistics.stdev(differences),
                    description=f'Nearly arithmetic sequence with average difference {statistics.mean(differences):.2f}',
                    data_points=data,
                    metadata={
                        'average_difference': statistics.mean(differences),
                        'difference_std': statistics.stdev(differences),
                        'length': len(data)
                    },
                    detected_at=datetime.now().isoformat()
                ))
    except:
        pass
    
    return patterns

def _find_repeating_subsequences(data: List, sensitivity: float) -> List[Pattern]:
    """Findet sich wiederholende Subsequenzen"""
    patterns = []
    try:
        for seq_length in range(2, min(len(data) // 2 + 1, 6)):  # Limit sequence length
            for start in range(len(data) - seq_length + 1):
                subsequence = data[start:start + seq_length]
                
                # Count occurrences of this subsequence
                occurrences = 0
                positions = []
                
                for i in range(len(data) - seq_length + 1):
                    if data[i:i + seq_length] == subsequence:
                        occurrences += 1
                        positions.append(i)
                
                if occurrences >= 2:  # Found repeating pattern
                    confidence = min(1.0, occurrences * 0.3)
                    patterns.append(Pattern(
                        pattern_type='repeating_subsequence',
                        pattern_id=f'repeat_subseq_{start}_{seq_length}_{datetime.now().timestamp()}',
                        confidence=confidence,
                        description=f'Subsequence {subsequence} repeats {occurrences} times',
                        data_points=subsequence,
                        metadata={
                            'occurrences': occurrences,
                            'positions': positions,
                            'sequence_length': seq_length
                        },
                        detected_at=datetime.now().isoformat()
                    ))
    except:
        pass
    
    return patterns

def _find_alternating_patterns(data: List, sensitivity: float) -> List[Pattern]:
    """Findet alternierende Muster"""
    patterns = []
    try:
        if len(data) >= 4:
            # Check for simple alternating pattern (A-B-A-B...)
            alternating_detected = True
            for i in range(2, len(data)):
                if data[i] != data[i-2]:
                    alternating_detected = False
                    break
            
            if alternating_detected:
                patterns.append(Pattern(
                    pattern_type='alternating_pattern',
                    pattern_id=f'alternating_{datetime.now().timestamp()}',
                    confidence=0.9,
                    description=f'Alternating pattern between {data[0]} and {data[1]}',
                    data_points=data,
                    metadata={
                        'alternating_values': [data[0], data[1]],
                        'pattern_length': len(data)
                    },
                    detected_at=datetime.now().isoformat()
                ))
    except:
        pass
    
    return patterns

# Frequency Pattern Detection
def _detect_frequency_patterns(data: Any, sensitivity: float) -> List[Pattern]:
    """Erkennt Häufigkeits-Muster"""
    patterns = []
    try:
        if isinstance(data, list):
            # Count frequencies
            frequency_counter = Counter(str(item) for item in data)
            
            # Find dominant elements
            total_items = len(data)
            for item, count in frequency_counter.most_common(5):  # Top 5
                frequency = count / total_items
                if frequency > sensitivity:
                    patterns.append(Pattern(
                        pattern_type='frequency_dominance',
                        pattern_id=f'freq_dom_{item}_{datetime.now().timestamp()}',
                        confidence=frequency,
                        description=f'Item "{item}" appears with {frequency:.1%} frequency',
                        data_points=[item] * count,
                        metadata={
                            'item': item,
                            'count': count,
                            'frequency': frequency,
                            'total_items': total_items
                        },
                        detected_at=datetime.now().isoformat()
                    ))
            
            # Check for equal distribution
            unique_items = len(frequency_counter)
            expected_frequency = 1.0 / unique_items
            actual_frequencies = [count / total_items for count in frequency_counter.values()]
            
            if all(abs(freq - expected_frequency) < 0.1 for freq in actual_frequencies):
                patterns.append(Pattern(
                    pattern_type='uniform_distribution',
                    pattern_id=f'uniform_dist_{datetime.now().timestamp()}',
                    confidence=0.8,
                    description=f'Uniform distribution across {unique_items} unique items',
                    data_points=list(frequency_counter.keys()),
                    metadata={
                        'unique_items': unique_items,
                        'expected_frequency': expected_frequency,
                        'distribution_type': 'uniform'
                    },
                    detected_at=datetime.now().isoformat()
                ))
    except:
        pass
    
    return patterns

# Clustering Pattern Detection
def _detect_clustering_patterns(data: Any, sensitivity: float) -> List[Pattern]:
    """Erkennt Clustering-Muster (vereinfacht)"""
    patterns = []
    try:
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            # Simple clustering based on value ranges
            sorted_data = sorted(data)
            clusters = []
            current_cluster = [sorted_data[0]]
            
            threshold = statistics.stdev(data) * sensitivity if len(data) > 1 else 1.0
            
            for i in range(1, len(sorted_data)):
                if abs(sorted_data[i] - sorted_data[i-1]) <= threshold:
                    current_cluster.append(sorted_data[i])
                else:
                    if len(current_cluster) >= 2:
                        clusters.append(current_cluster)
                    current_cluster = [sorted_data[i]]
            
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
            
            # Report significant clusters
            for i, cluster in enumerate(clusters):
                if len(cluster) >= max(2, len(data) * 0.1):  # At least 10% of data
                    patterns.append(Pattern(
                        pattern_type='value_cluster',
                        pattern_id=f'cluster_{i}_{datetime.now().timestamp()}',
                        confidence=len(cluster) / len(data),
                        description=f'Cluster of {len(cluster)} values around {statistics.mean(cluster):.2f}',
                        data_points=cluster,
                        metadata={
                            'cluster_size': len(cluster),
                            'cluster_center': statistics.mean(cluster),
                            'cluster_range': (min(cluster), max(cluster))
                        },
                        detected_at=datetime.now().isoformat()
                    ))
    except:
        pass
    
    return patterns

# Temporal Pattern Detection
def _detect_temporal_patterns(data: Any, sensitivity: float) -> List[Pattern]:
    """Erkennt zeitliche Muster"""
    patterns = []
    try:
        if _is_time_series_data(data):
            # Extract temporal information
            timestamps = []
            values = []
            
            for item in data:
                if hasattr(item, 'timestamp'):
                    timestamps.append(item.timestamp)
                    values.append(getattr(item, 'value', item))
                elif isinstance(item, dict):
                    # Look for timestamp-like keys
                    for key in item.keys():
                        if 'time' in str(key).lower() or 'date' in str(key).lower():
                            timestamps.append(item[key])
                            values.append(item.get('value', item))
                            break
            
            if len(timestamps) >= 3:
                # Check for periodic patterns
                time_intervals = []
                for i in range(1, len(timestamps)):
                    try:
                        if isinstance(timestamps[i], str) and isinstance(timestamps[i-1], str):
                            t1 = datetime.fromisoformat(timestamps[i].replace('Z', '+00:00'))
                            t2 = datetime.fromisoformat(timestamps[i-1].replace('Z', '+00:00'))
                            interval = (t1 - t2).total_seconds()
                            time_intervals.append(interval)
                    except:
                        continue
                
                if time_intervals and len(set(time_intervals)) == 1:
                    patterns.append(Pattern(
                        pattern_type='regular_temporal_interval',
                        pattern_id=f'temp_interval_{datetime.now().timestamp()}',
                        confidence=0.9,
                        description=f'Regular time interval of {time_intervals[0]} seconds',
                        data_points=timestamps,
                        metadata={
                            'interval_seconds': time_intervals[0],
                            'data_points_count': len(timestamps)
                        },
                        detected_at=datetime.now().isoformat()
                    ))
    except:
        pass
    
    return patterns

# Behavioral Pattern Detection
def _detect_behavioral_patterns(data: Any, sensitivity: float) -> List[Pattern]:
    """Erkennt Verhaltensmuster"""
    patterns = []
    try:
        # This is a simplified behavioral pattern detection
        # In practice, this would analyze user interactions, system responses, etc.
        
        if isinstance(data, list) and len(data) >= 5:
            # Look for increasing/decreasing patterns
            if all(isinstance(x, (int, float)) for x in data):
                if all(data[i] <= data[i+1] for i in range(len(data)-1)):
                    patterns.append(Pattern(
                        pattern_type='monotonic_increasing',
                        pattern_id=f'mono_inc_{datetime.now().timestamp()}',
                        confidence=0.8,
                        description='Monotonically increasing behavioral pattern',
                        data_points=data,
                        metadata={'behavior_type': 'growth', 'consistency': 'perfect'},
                        detected_at=datetime.now().isoformat()
                    ))
                elif all(data[i] >= data[i+1] for i in range(len(data)-1)):
                    patterns.append(Pattern(
                        pattern_type='monotonic_decreasing',
                        pattern_id=f'mono_dec_{datetime.now().timestamp()}',
                        confidence=0.8,
                        description='Monotonically decreasing behavioral pattern',
                        data_points=data,
                        metadata={'behavior_type': 'decline', 'consistency': 'perfect'},
                        detected_at=datetime.now().isoformat()
                    ))
            
            # Look for cyclical behavior
            if len(data) >= 8:
                # Simple cycle detection by looking for repeating subsequences
                for cycle_length in [2, 3, 4]:
                    if len(data) >= cycle_length * 2:
                        is_cyclical = True
                        for i in range(cycle_length, len(data)):
                            if data[i] != data[i % cycle_length]:
                                is_cyclical = False
                                break
                        
                        if is_cyclical:
                            patterns.append(Pattern(
                                pattern_type='cyclical_behavior',
                                pattern_id=f'cycle_{cycle_length}_{datetime.now().timestamp()}',
                                confidence=0.9,
                                description=f'Cyclical behavior with period {cycle_length}',
                                data_points=data[:cycle_length],
                                metadata={
                                    'cycle_length': cycle_length,
                                    'repetitions': len(data) // cycle_length,
                                    'behavior_type': 'cyclical'
                                },
                                detected_at=datetime.now().isoformat()
                            ))
                            break
    except:
        pass
    
    return patterns

__all__ = [
    'Pattern',
    'Trend', 
    'Anomaly',
    'detect_patterns',
    'detect_trends',
    'detect_anomalies',
    'analyze_data_relationships',
    'find_recurring_patterns'
]