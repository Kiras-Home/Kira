"""
Data Processing Helper Functions
Grundlegende Datenverarbeitungs-Utilities und Formatierungen
"""

import logging
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class DataPoint:
    """Standardisierter Datenpunkt"""
    timestamp: str
    value: Any
    data_type: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DataSeries:
    """Serie von Datenpunkten"""
    name: str
    data_points: List[DataPoint]
    series_type: str
    created_at: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

def normalize_data_structure(raw_data: Any, data_type: str = 'auto') -> Dict[str, Any]:
    """
    Normalisiert verschiedene Datenstrukturen zu einheitlichem Format
    """
    try:
        if raw_data is None:
            return {'normalized_data': None, 'data_type': 'null', 'status': 'empty'}
        
        # Auto-detect data type
        if data_type == 'auto':
            data_type = _detect_data_type(raw_data)
        
        normalized = {
            'original_type': type(raw_data).__name__,
            'detected_type': data_type,
            'normalization_timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        # Normalize based on detected type
        if data_type == 'list':
            normalized['normalized_data'] = _normalize_list_data(raw_data)
        elif data_type == 'dict':
            normalized['normalized_data'] = _normalize_dict_data(raw_data)
        elif data_type == 'string':
            normalized['normalized_data'] = _normalize_string_data(raw_data)
        elif data_type == 'numeric':
            normalized['normalized_data'] = _normalize_numeric_data(raw_data)
        elif data_type == 'memory_object':
            normalized['normalized_data'] = _normalize_memory_object(raw_data)
        else:
            # Fallback: convert to string representation
            normalized['normalized_data'] = str(raw_data)
            normalized['status'] = 'fallback_conversion'
        
        return normalized
        
    except Exception as e:
        logger.error(f"Data normalization failed: {e}")
        return {
            'normalized_data': str(raw_data) if raw_data is not None else '',
            'data_type': 'error',
            'status': 'failed',
            'error': str(e)
        }

def extract_data_features(data: Any, feature_types: List[str] = None) -> Dict[str, Any]:
    """
    Extrahiert Features aus Daten für weitere Verarbeitung
    """
    try:
        if feature_types is None:
            feature_types = ['basic', 'statistical', 'structural']
        
        features = {
            'extraction_timestamp': datetime.now().isoformat(),
            'data_summary': _get_data_summary(data)
        }
        
        if 'basic' in feature_types:
            features['basic_features'] = _extract_basic_features(data)
        
        if 'statistical' in feature_types:
            features['statistical_features'] = _extract_statistical_features(data)
        
        if 'structural' in feature_types:
            features['structural_features'] = _extract_structural_features(data)
        
        if 'temporal' in feature_types:
            features['temporal_features'] = _extract_temporal_features(data)
        
        if 'content' in feature_types:
            features['content_features'] = _extract_content_features(data)
        
        return features
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return {'error': str(e), 'extraction_timestamp': datetime.now().isoformat()}

def clean_data(data: Any, cleaning_options: Dict[str, bool] = None) -> Dict[str, Any]:
    """
    Reinigt Daten von Inkonsistenzen und Fehlern
    """
    try:
        if cleaning_options is None:
            cleaning_options = {
                'remove_nulls': True,
                'remove_duplicates': True,
                'normalize_whitespace': True,
                'validate_types': True,
                'standardize_formats': True
            }
        
        cleaning_result = {
            'original_data': data,
            'cleaning_applied': [],
            'cleaning_timestamp': datetime.now().isoformat()
        }
        
        cleaned_data = data
        
        # Remove nulls
        if cleaning_options.get('remove_nulls', False):
            cleaned_data = _remove_null_values(cleaned_data)
            cleaning_result['cleaning_applied'].append('null_removal')
        
        # Remove duplicates
        if cleaning_options.get('remove_duplicates', False):
            cleaned_data = _remove_duplicates(cleaned_data)
            cleaning_result['cleaning_applied'].append('duplicate_removal')
        
        # Normalize whitespace
        if cleaning_options.get('normalize_whitespace', False):
            cleaned_data = _normalize_whitespace(cleaned_data)
            cleaning_result['cleaning_applied'].append('whitespace_normalization')
        
        # Validate types
        if cleaning_options.get('validate_types', False):
            cleaned_data = _validate_and_fix_types(cleaned_data)
            cleaning_result['cleaning_applied'].append('type_validation')
        
        # Standardize formats
        if cleaning_options.get('standardize_formats', False):
            cleaned_data = _standardize_formats(cleaned_data)
            cleaning_result['cleaning_applied'].append('format_standardization')
        
        cleaning_result['cleaned_data'] = cleaned_data
        cleaning_result['cleaning_summary'] = {
            'operations_applied': len(cleaning_result['cleaning_applied']),
            'data_size_before': _calculate_data_size(data),
            'data_size_after': _calculate_data_size(cleaned_data)
        }
        
        return cleaning_result
        
    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        return {
            'cleaned_data': data,
            'error': str(e),
            'cleaning_timestamp': datetime.now().isoformat()
        }

def validate_data_quality(data: Any, quality_checks: List[str] = None) -> Dict[str, Any]:
    """
    Validiert Datenqualität
    """
    try:
        if quality_checks is None:
            quality_checks = ['completeness', 'consistency', 'accuracy', 'validity']
        
        quality_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_quality_score': 0.0,
            'quality_checks': {}
        }
        
        total_score = 0.0
        checks_performed = 0
        
        if 'completeness' in quality_checks:
            completeness_score = _check_data_completeness(data)
            quality_report['quality_checks']['completeness'] = completeness_score
            total_score += completeness_score['score']
            checks_performed += 1
        
        if 'consistency' in quality_checks:
            consistency_score = _check_data_consistency(data)
            quality_report['quality_checks']['consistency'] = consistency_score
            total_score += consistency_score['score']
            checks_performed += 1
        
        if 'accuracy' in quality_checks:
            accuracy_score = _check_data_accuracy(data)
            quality_report['quality_checks']['accuracy'] = accuracy_score
            total_score += accuracy_score['score']
            checks_performed += 1
        
        if 'validity' in quality_checks:
            validity_score = _check_data_validity(data)
            quality_report['quality_checks']['validity'] = validity_score
            total_score += validity_score['score']
            checks_performed += 1
        
        # Calculate overall quality score
        quality_report['overall_quality_score'] = total_score / checks_performed if checks_performed > 0 else 0.0
        quality_report['quality_rating'] = _rate_data_quality(quality_report['overall_quality_score'])
        
        return quality_report
        
    except Exception as e:
        logger.error(f"Data quality validation failed: {e}")
        return {
            'overall_quality_score': 0.5,
            'quality_rating': 'unknown',
            'error': str(e),
            'validation_timestamp': datetime.now().isoformat()
        }

def transform_data_format(data: Any, target_format: str, transformation_options: Dict = None) -> Dict[str, Any]:
    """
    Transformiert Daten in verschiedene Formate
    """
    try:
        transformation_options = transformation_options or {}
        
        transformation_result = {
            'original_format': type(data).__name__,
            'target_format': target_format,
            'transformation_timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        if target_format == 'json':
            transformed_data = _transform_to_json(data, transformation_options)
        elif target_format == 'csv':
            transformed_data = _transform_to_csv(data, transformation_options)
        elif target_format == 'list':
            transformed_data = _transform_to_list(data, transformation_options)
        elif target_format == 'dict':
            transformed_data = _transform_to_dict(data, transformation_options)
        elif target_format == 'datapoints':
            transformed_data = _transform_to_datapoints(data, transformation_options)
        elif target_format == 'timeseries':
            transformed_data = _transform_to_timeseries(data, transformation_options)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
        
        transformation_result['transformed_data'] = transformed_data
        transformation_result['success'] = True
        
        return transformation_result
        
    except Exception as e:
        logger.error(f"Data format transformation failed: {e}")
        return {
            'transformed_data': data,
            'success': False,
            'error': str(e),
            'transformation_timestamp': datetime.now().isoformat()
        }

def aggregate_data(data: Any, aggregation_method: str = 'auto', grouping_key: str = None) -> Dict[str, Any]:
    """
    Aggregiert Daten nach verschiedenen Methoden
    """
    try:
        aggregation_result = {
            'aggregation_method': aggregation_method,
            'aggregation_timestamp': datetime.now().isoformat(),
            'grouping_key': grouping_key
        }
        
        # Auto-detect best aggregation method
        if aggregation_method == 'auto':
            aggregation_method = _detect_best_aggregation_method(data)
            aggregation_result['detected_method'] = aggregation_method
        
        if aggregation_method == 'sum':
            aggregated_data = _aggregate_sum(data, grouping_key)
        elif aggregation_method == 'average':
            aggregated_data = _aggregate_average(data, grouping_key)
        elif aggregation_method == 'count':
            aggregated_data = _aggregate_count(data, grouping_key)
        elif aggregation_method == 'group':
            aggregated_data = _aggregate_group(data, grouping_key)
        elif aggregation_method == 'temporal':
            aggregated_data = _aggregate_temporal(data, grouping_key)
        else:
            aggregated_data = _aggregate_simple(data)
        
        aggregation_result['aggregated_data'] = aggregated_data
        aggregation_result['aggregation_summary'] = {
            'original_size': _calculate_data_size(data),
            'aggregated_size': _calculate_data_size(aggregated_data),
            'compression_ratio': _calculate_compression_ratio(data, aggregated_data)
        }
        
        return aggregation_result
        
    except Exception as e:
        logger.error(f"Data aggregation failed: {e}")
        return {
            'aggregated_data': data,
            'error': str(e),
            'aggregation_timestamp': datetime.now().isoformat()
        }

# Helper Functions

def _detect_data_type(data: Any) -> str:
    """Erkennt Datentyp automatisch"""
    try:
        if isinstance(data, list):
            return 'list'
        elif isinstance(data, dict):
            return 'dict'
        elif isinstance(data, str):
            return 'string'
        elif isinstance(data, (int, float)):
            return 'numeric'
        elif hasattr(data, '__dict__') and hasattr(data, 'content'):
            return 'memory_object'
        else:
            return 'unknown'
    except:
        return 'unknown'

def _normalize_list_data(data: List) -> List[Dict]:
    """Normalisiert Listen-Daten"""
    normalized = []
    for i, item in enumerate(data):
        normalized.append({
            'index': i,
            'value': item,
            'type': type(item).__name__,
            'normalized_at': datetime.now().isoformat()
        })
    return normalized

def _normalize_dict_data(data: Dict) -> Dict:
    """Normalisiert Dictionary-Daten"""
    normalized = {}
    for key, value in data.items():
        normalized[str(key)] = {
            'value': value,
            'type': type(value).__name__,
            'key_type': type(key).__name__
        }
    return normalized

def _normalize_string_data(data: str) -> Dict:
    """Normalisiert String-Daten"""
    return {
        'content': data,
        'length': len(data),
        'word_count': len(data.split()),
        'encoding': 'utf-8'
    }

def _normalize_numeric_data(data: Union[int, float]) -> Dict:
    """Normalisiert numerische Daten"""
    return {
        'value': data,
        'type': 'integer' if isinstance(data, int) else 'float',
        'is_positive': data > 0,
        'absolute_value': abs(data)
    }

def _normalize_memory_object(data: Any) -> Dict:
    """Normalisiert Memory-Objekte"""
    try:
        normalized = {
            'object_type': type(data).__name__,
            'has_content': hasattr(data, 'content'),
            'has_timestamp': hasattr(data, 'timestamp') or hasattr(data, 'created_at')
        }
        
        if hasattr(data, 'content'):
            normalized['content'] = str(data.content)[:200]  # Limit content length
        
        if hasattr(data, 'timestamp'):
            normalized['timestamp'] = str(data.timestamp)
        elif hasattr(data, 'created_at'):
            normalized['timestamp'] = str(data.created_at)
        
        return normalized
    except:
        return {'object_type': 'unknown_memory_object'}

def _get_data_summary(data: Any) -> Dict[str, Any]:
    """Erstellt Daten-Summary"""
    return {
        'data_type': type(data).__name__,
        'size': _calculate_data_size(data),
        'is_empty': data is None or (hasattr(data, '__len__') and len(data) == 0),
        'summary_timestamp': datetime.now().isoformat()
    }

def _extract_basic_features(data: Any) -> Dict[str, Any]:
    """Extrahiert grundlegende Features"""
    return {
        'data_type': type(data).__name__,
        'size': _calculate_data_size(data),
        'is_numeric': isinstance(data, (int, float)),
        'is_text': isinstance(data, str),
        'is_structured': isinstance(data, (list, dict))
    }

def _extract_statistical_features(data: Any) -> Dict[str, Any]:
    """Extrahiert statistische Features"""
    try:
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            return {
                'count': len(data),
                'mean': statistics.mean(data),
                'median': statistics.median(data),
                'std_dev': statistics.stdev(data) if len(data) > 1 else 0,
                'min': min(data),
                'max': max(data)
            }
        else:
            return {'statistical_analysis': 'not_applicable', 'reason': 'non_numeric_data'}
    except:
        return {'statistical_analysis': 'failed'}

def _extract_structural_features(data: Any) -> Dict[str, Any]:
    """Extrahiert strukturelle Features"""
    try:
        features = {'structure_type': type(data).__name__}
        
        if isinstance(data, list):
            features.update({
                'length': len(data),
                'element_types': list(set(type(x).__name__ for x in data)),
                'is_homogeneous': len(set(type(x).__name__ for x in data)) == 1
            })
        elif isinstance(data, dict):
            features.update({
                'key_count': len(data),
                'key_types': list(set(type(k).__name__ for k in data.keys())),
                'value_types': list(set(type(v).__name__ for v in data.values())),
                'nesting_detected': any(isinstance(v, (dict, list)) for v in data.values())
            })
        elif isinstance(data, str):
            features.update({
                'character_count': len(data),
                'word_count': len(data.split()),
                'line_count': len(data.split('\n'))
            })
        
        return features
    except:
        return {'structure_analysis': 'failed'}

def _extract_temporal_features(data: Any) -> Dict[str, Any]:
    """Extrahiert zeitliche Features"""
    try:
        features = {'temporal_analysis': 'attempted'}
        
        # Look for timestamp-like data
        if isinstance(data, list):
            timestamp_candidates = []
            for item in data:
                if hasattr(item, 'timestamp') or hasattr(item, 'created_at'):
                    timestamp_candidates.append(item)
            
            features['items_with_timestamps'] = len(timestamp_candidates)
            features['temporal_data_available'] = len(timestamp_candidates) > 0
        
        elif isinstance(data, dict):
            timestamp_keys = [k for k in data.keys() if 'time' in str(k).lower() or 'date' in str(k).lower()]
            features['timestamp_keys'] = timestamp_keys
            features['temporal_data_available'] = len(timestamp_keys) > 0
        
        else:
            features['temporal_data_available'] = False
        
        return features
    except:
        return {'temporal_analysis': 'failed'}

def _extract_content_features(data: Any) -> Dict[str, Any]:
    """Extrahiert Inhalts-Features"""
    try:
        features = {'content_analysis': 'attempted'}
        
        if isinstance(data, str):
            features.update({
                'contains_numbers': any(c.isdigit() for c in data),
                'contains_punctuation': any(c in '.,!?;:' for c in data),
                'contains_uppercase': any(c.isupper() for c in data),
                'contains_lowercase': any(c.islower() for c in data),
                'language_indicators': _detect_language_indicators(data)
            })
        elif isinstance(data, list):
            content_types = []
            for item in data:
                if hasattr(item, 'content'):
                    content_types.append('has_content')
                elif isinstance(item, str):
                    content_types.append('string_content')
                else:
                    content_types.append('other_content')
            
            features['content_types'] = list(set(content_types))
        
        return features
    except:
        return {'content_analysis': 'failed'}

def _detect_language_indicators(text: str) -> List[str]:
    """Erkennt Sprach-Indikatoren"""
    indicators = []
    
    # Simple language detection based on common words
    german_words = ['der', 'die', 'das', 'und', 'ist', 'zu', 'in', 'mit']
    english_words = ['the', 'and', 'is', 'to', 'in', 'with', 'a', 'of']
    
    text_lower = text.lower()
    
    german_count = sum(1 for word in german_words if word in text_lower)
    english_count = sum(1 for word in english_words if word in text_lower)
    
    if german_count > english_count:
        indicators.append('german')
    elif english_count > german_count:
        indicators.append('english')
    
    return indicators

def _calculate_data_size(data: Any) -> int:
    """Berechnet Datengröße"""
    try:
        if hasattr(data, '__len__'):
            return len(data)
        elif isinstance(data, (int, float)):
            return 1
        elif data is None:
            return 0
        else:
            return len(str(data))
    except:
        return 0

# Data Cleaning Helper Functions
def _remove_null_values(data: Any) -> Any:
    """Entfernt Null-Werte"""
    try:
        if isinstance(data, list):
            return [item for item in data if item is not None]
        elif isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        else:
            return data
    except:
        return data

def _remove_duplicates(data: Any) -> Any:
    """Entfernt Duplikate"""
    try:
        if isinstance(data, list):
            seen = set()
            result = []
            for item in data:
                if isinstance(item, (dict, list)):
                    # For complex objects, use string representation
                    item_key = str(item)
                else:
                    item_key = item
                
                if item_key not in seen:
                    seen.add(item_key)
                    result.append(item)
            return result
        else:
            return data
    except:
        return data

def _normalize_whitespace(data: Any) -> Any:
    """Normalisiert Whitespace"""
    try:
        if isinstance(data, str):
            return ' '.join(data.split())
        elif isinstance(data, list):
            return [_normalize_whitespace(item) for item in data]
        elif isinstance(data, dict):
            return {k: _normalize_whitespace(v) for k, v in data.items()}
        else:
            return data
    except:
        return data

def _validate_and_fix_types(data: Any) -> Any:
    """Validiert und korrigiert Datentypen"""
    # Simplified type validation
    return data

def _standardize_formats(data: Any) -> Any:
    """Standardisiert Formate"""
    # Simplified format standardization
    return data

# Data Quality Check Functions
def _check_data_completeness(data: Any) -> Dict[str, Any]:
    """Prüft Datenvollständigkeit"""
    try:
        if data is None:
            return {'score': 0.0, 'reason': 'data_is_null'}
        elif hasattr(data, '__len__') and len(data) == 0:
            return {'score': 0.0, 'reason': 'data_is_empty'}
        elif isinstance(data, list):
            null_count = sum(1 for item in data if item is None)
            completeness = 1.0 - (null_count / len(data))
            return {'score': completeness, 'null_items': null_count, 'total_items': len(data)}
        else:
            return {'score': 1.0, 'reason': 'single_value_complete'}
    except:
        return {'score': 0.5, 'reason': 'completeness_check_failed'}

def _check_data_consistency(data: Any) -> Dict[str, Any]:
    """Prüft Datenkonsistenz"""
    try:
        if isinstance(data, list):
            if not data:
                return {'score': 1.0, 'reason': 'empty_list_consistent'}
            
            # Check type consistency
            types = [type(item).__name__ for item in data]
            unique_types = set(types)
            type_consistency = 1.0 - (len(unique_types) - 1) / len(data)
            
            return {'score': max(0.0, type_consistency), 'unique_types': len(unique_types)}
        else:
            return {'score': 1.0, 'reason': 'single_value_consistent'}
    except:
        return {'score': 0.5, 'reason': 'consistency_check_failed'}

def _check_data_accuracy(data: Any) -> Dict[str, Any]:
    """Prüft Datengenauigkeit (vereinfacht)"""
    try:
        # Simplified accuracy check
        if isinstance(data, (list, dict)) and len(data) > 0:
            return {'score': 0.8, 'reason': 'structured_data_assumed_accurate'}
        elif isinstance(data, str) and len(data) > 0:
            return {'score': 0.7, 'reason': 'text_data_assumed_moderately_accurate'}
        elif isinstance(data, (int, float)):
            return {'score': 0.9, 'reason': 'numeric_data_assumed_accurate'}
        else:
            return {'score': 0.5, 'reason': 'unknown_accuracy'}
    except:
        return {'score': 0.5, 'reason': 'accuracy_check_failed'}

def _check_data_validity(data: Any) -> Dict[str, Any]:
    """Prüft Datenvalidität"""
    try:
        # Basic validity checks
        if data is None:
            return {'score': 0.0, 'reason': 'null_data_invalid'}
        elif isinstance(data, (list, dict)) and len(data) == 0:
            return {'score': 0.5, 'reason': 'empty_container_partially_valid'}
        else:
            return {'score': 0.8, 'reason': 'data_appears_valid'}
    except:
        return {'score': 0.5, 'reason': 'validity_check_failed'}

def _rate_data_quality(score: float) -> str:
    """Bewertet Datenqualität"""
    if score >= 0.9:
        return 'excellent'
    elif score >= 0.7:
        return 'good'
    elif score >= 0.5:
        return 'moderate'
    elif score >= 0.3:
        return 'poor'
    else:
        return 'very_poor'

# Data Transformation Functions
def _transform_to_json(data: Any, options: Dict) -> str:
    """Transformiert zu JSON"""
    try:
        return json.dumps(data, default=str, indent=2)
    except:
        return json.dumps({'data': str(data)}, indent=2)

def _transform_to_csv(data: Any, options: Dict) -> str:
    """Transformiert zu CSV (vereinfacht)"""
    try:
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                # Convert list of dicts to CSV
                headers = list(data[0].keys())
                csv_lines = [','.join(headers)]
                for item in data:
                    values = [str(item.get(header, '')) for header in headers]
                    csv_lines.append(','.join(values))
                return '\n'.join(csv_lines)
        
        return str(data)  # Fallback
    except:
        return str(data)

def _transform_to_list(data: Any, options: Dict) -> List:
    """Transformiert zu Liste"""
    try:
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return list(data.values())
        elif isinstance(data, str):
            return list(data.split())
        else:
            return [data]
    except:
        return [str(data)]

def _transform_to_dict(data: Any, options: Dict) -> Dict:
    """Transformiert zu Dictionary"""
    try:
        if isinstance(data, dict):
            return data
        elif isinstance(data, list):
            return {str(i): item for i, item in enumerate(data)}
        else:
            return {'value': data, 'type': type(data).__name__}
    except:
        return {'data': str(data)}

def _transform_to_datapoints(data: Any, options: Dict) -> List[DataPoint]:
    """Transformiert zu DataPoint Objekten"""
    try:
        datapoints = []
        timestamp = datetime.now().isoformat()
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                datapoints.append(DataPoint(
                    timestamp=timestamp,
                    value=item,
                    data_type=type(item).__name__,
                    confidence=1.0,
                    metadata={'index': i}
                ))
        else:
            datapoints.append(DataPoint(
                timestamp=timestamp,
                value=data,
                data_type=type(data).__name__,
                confidence=1.0
            ))
        
        return datapoints
    except:
        return []

def _transform_to_timeseries(data: Any, options: Dict) -> DataSeries:
    """Transformiert zu TimeSeries"""
    try:
        datapoints = _transform_to_datapoints(data, options)
        return DataSeries(
            name=options.get('series_name', 'unnamed_series'),
            data_points=datapoints,
            series_type='general',
            created_at=datetime.now().isoformat()
        )
    except:
        return DataSeries(
            name='error_series',
            data_points=[],
            series_type='error',
            created_at=datetime.now().isoformat()
        )

# Data Aggregation Functions
def _detect_best_aggregation_method(data: Any) -> str:
    """Erkennt beste Aggregations-Methode"""
    try:
        if isinstance(data, list):
            if all(isinstance(x, (int, float)) for x in data):
                return 'average'
            else:
                return 'count'
        elif isinstance(data, dict):
            return 'group'
        else:
            return 'simple'
    except:
        return 'simple'

def _aggregate_sum(data: Any, grouping_key: str) -> Any:
    """Aggregiert durch Summierung"""
    try:
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            return {'sum': sum(data), 'count': len(data)}
        else:
            return {'sum': 'not_applicable', 'reason': 'non_numeric_data'}
    except:
        return {'sum': 0, 'error': 'aggregation_failed'}

def _aggregate_average(data: Any, grouping_key: str) -> Any:
    """Aggregiert durch Durchschnittsbildung"""
    try:
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            return {
                'average': sum(data) / len(data),
                'count': len(data),
                'min': min(data),
                'max': max(data)
            }
        else:
            return {'average': 'not_applicable', 'reason': 'non_numeric_data'}
    except:
        return {'average': 0, 'error': 'aggregation_failed'}

def _aggregate_count(data: Any, grouping_key: str) -> Any:
    """Aggregiert durch Zählung"""
    try:
        if isinstance(data, list):
            return {'count': len(data), 'unique_items': len(set(str(x) for x in data))}
        elif isinstance(data, dict):
            return {'count': len(data), 'keys': list(data.keys())}
        else:
            return {'count': 1, 'type': type(data).__name__}
    except:
        return {'count': 0, 'error': 'aggregation_failed'}

def _aggregate_group(data: Any, grouping_key: str) -> Any:
    """Aggregiert durch Gruppierung"""
    try:
        if isinstance(data, list):
            groups = {}
            for item in data:
                if isinstance(item, dict) and grouping_key in item:
                    key = item[grouping_key]
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(item)
                else:
                    # Group by type
                    item_type = type(item).__name__
                    if item_type not in groups:
                        groups[item_type] = []
                    groups[item_type].append(item)
            
            return {'groups': groups, 'group_count': len(groups)}
        else:
            return {'groups': {'single_item': [data]}, 'group_count': 1}
    except:
        return {'groups': {}, 'error': 'grouping_failed'}

def _aggregate_temporal(data: Any, grouping_key: str) -> Any:
    """Aggregiert zeitlich"""
    # Simplified temporal aggregation
    return {'temporal_aggregation': 'simplified', 'data_points': _calculate_data_size(data)}

def _aggregate_simple(data: Any) -> Any:
    """Einfache Aggregation"""
    return {
        'type': type(data).__name__,
        'size': _calculate_data_size(data),
        'summary': str(data)[:100] if len(str(data)) > 100 else str(data)
    }

def _calculate_compression_ratio(original_data: Any, aggregated_data: Any) -> float:
    """Berechnet Kompressionsrate"""
    try:
        original_size = _calculate_data_size(original_data)
        aggregated_size = _calculate_data_size(aggregated_data)
        
        if original_size == 0:
            return 0.0
        
        return aggregated_size / original_size
    except:
        return 1.0

__all__ = [
    'DataPoint',
    'DataSeries',
    'normalize_data_structure',
    'extract_data_features',
    'clean_data',
    'validate_data_quality',
    'transform_data_format',
    'aggregate_data'
]