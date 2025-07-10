"""
Statistical Analysis Module - Part 1
Grundlegende statistische Analysen, deskriptive Statistik und Verteilungsanalysen
"""

import logging
import statistics
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import Counter
import random

logger = logging.getLogger(__name__)

@dataclass
class StatisticalSummary:
    """Statistische Zusammenfassung"""
    data_type: str
    sample_size: int
    measures_of_central_tendency: Dict[str, float]
    measures_of_dispersion: Dict[str, float]
    measures_of_shape: Dict[str, float]
    outliers: List[Any]
    confidence_intervals: Dict[str, Tuple[float, float]]
    analysis_timestamp: str

@dataclass
class DistributionAnalysis:
    """Verteilungsanalyse"""
    distribution_type: str
    distribution_parameters: Dict[str, float]
    goodness_of_fit: float
    distribution_properties: Dict[str, Any]
    histogram_data: Dict[str, List]
    statistical_tests: Dict[str, Dict]

@dataclass
class CorrelationResult:
    """Korrelationsresultat"""
    correlation_type: str
    correlation_coefficient: float
    p_value: float
    significance_level: str
    sample_size: int
    confidence_interval: Tuple[float, float]
    interpretation: str

def calculate_descriptive_statistics(data: Any, confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Berechnet umfassende deskriptive Statistiken
    """
    try:
        if not isinstance(data, list) or not data:
            return {'error': 'invalid_data', 'message': 'Data must be a non-empty list'}
        
        # Separate numeric and non-numeric data
        numeric_data = [x for x in data if isinstance(x, (int, float)) and not math.isnan(x)]
        non_numeric_data = [x for x in data if not isinstance(x, (int, float)) or math.isnan(x)]
        
        result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_observations': len(data),
                'numeric_observations': len(numeric_data),
                'non_numeric_observations': len(non_numeric_data),
                'missing_values': len([x for x in data if x is None or (isinstance(x, float) and math.isnan(x))])
            }
        }
        
        # Analyze numeric data
        if numeric_data:
            result['numeric_statistics'] = _calculate_numeric_statistics(numeric_data, confidence_level)
        
        # Analyze non-numeric data
        if non_numeric_data:
            result['categorical_statistics'] = _calculate_categorical_statistics(non_numeric_data)
        
        # Overall data quality assessment
        result['data_quality'] = _assess_data_quality(data, numeric_data, non_numeric_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Descriptive statistics calculation failed: {e}")
        return {
            'error': str(e),
            'analysis_timestamp': datetime.now().isoformat()
        }

def analyze_distribution(data: List[Union[int, float]], 
                        distribution_types: List[str] = None) -> Dict[str, Any]:
    """
    Analysiert die Verteilung der Daten
    """
    try:
        if not data or not all(isinstance(x, (int, float)) for x in data):
            return {'error': 'invalid_numeric_data', 'message': 'Data must be numeric'}
        
        if distribution_types is None:
            distribution_types = ['normal', 'uniform', 'exponential', 'binomial']
        
        distribution_result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'sample_size': len(data),
            'data_range': (min(data), max(data)),
            'distribution_analyses': {}
        }
        
        # Basic distribution properties
        distribution_result['basic_properties'] = _calculate_distribution_properties(data)
        
        # Test different distributions
        for dist_type in distribution_types:
            if dist_type == 'normal':
                distribution_result['distribution_analyses']['normal'] = _test_normal_distribution(data)
            elif dist_type == 'uniform':
                distribution_result['distribution_analyses']['uniform'] = _test_uniform_distribution(data)
            elif dist_type == 'exponential':
                distribution_result['distribution_analyses']['exponential'] = _test_exponential_distribution(data)
            elif dist_type == 'binomial':
                distribution_result['distribution_analyses']['binomial'] = _test_binomial_distribution(data)
        
        # Generate histogram data
        distribution_result['histogram'] = _generate_histogram_data(data)
        
        # Best fit distribution
        distribution_result['best_fit'] = _determine_best_fit_distribution(
            distribution_result['distribution_analyses']
        )
        
        return distribution_result
        
    except Exception as e:
        logger.error(f"Distribution analysis failed: {e}")
        return {
            'error': str(e),
            'analysis_timestamp': datetime.now().isoformat()
        }

def calculate_correlation(data1: List[Union[int, float]], 
                         data2: List[Union[int, float]],
                         correlation_types: List[str] = None) -> Dict[str, Any]:
    """
    Berechnet verschiedene Korrelationsmaße
    """
    try:
        if len(data1) != len(data2):
            return {'error': 'mismatched_lengths', 'message': 'Data arrays must have same length'}
        
        if not data1 or not data2:
            return {'error': 'empty_data', 'message': 'Data arrays cannot be empty'}
        
        if correlation_types is None:
            correlation_types = ['pearson', 'spearman', 'kendall']
        
        correlation_result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'sample_size': len(data1),
            'data1_summary': _get_basic_stats(data1),
            'data2_summary': _get_basic_stats(data2),
            'correlations': {}
        }
        
        # Calculate different correlation types
        if 'pearson' in correlation_types:
            correlation_result['correlations']['pearson'] = _calculate_pearson_correlation(data1, data2)
        
        if 'spearman' in correlation_types:
            correlation_result['correlations']['spearman'] = _calculate_spearman_correlation(data1, data2)
        
        if 'kendall' in correlation_types:
            correlation_result['correlations']['kendall'] = _calculate_kendall_correlation(data1, data2)
        
        # Overall correlation assessment
        correlation_result['correlation_summary'] = _summarize_correlations(
            correlation_result['correlations']
        )
        
        return correlation_result
        
    except Exception as e:
        logger.error(f"Correlation calculation failed: {e}")
        return {
            'error': str(e),
            'analysis_timestamp': datetime.now().isoformat()
        }

def perform_hypothesis_testing(data: List[Union[int, float]], 
                             test_type: str,
                             test_parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Führt Hypothesentests durch
    """
    try:
        if not data:
            return {'error': 'empty_data', 'message': 'Data cannot be empty'}
        
        test_parameters = test_parameters or {}
        
        hypothesis_result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'test_type': test_type,
            'sample_size': len(data),
            'test_parameters': test_parameters
        }
        
        if test_type == 'one_sample_t_test':
            hypothesis_result['test_result'] = _perform_one_sample_t_test(data, test_parameters)
        elif test_type == 'normality_test':
            hypothesis_result['test_result'] = _perform_normality_test(data, test_parameters)
        elif test_type == 'variance_test':
            hypothesis_result['test_result'] = _perform_variance_test(data, test_parameters)
        elif test_type == 'mean_comparison':
            hypothesis_result['test_result'] = _perform_mean_comparison(data, test_parameters)
        else:
            return {'error': 'unsupported_test_type', 'message': f'Test type {test_type} not supported'}
        
        return hypothesis_result
        
    except Exception as e:
        logger.error(f"Hypothesis testing failed: {e}")
        return {
            'error': str(e),
            'analysis_timestamp': datetime.now().isoformat()
        }

def detect_outliers(data: List[Union[int, float]], 
                   methods: List[str] = None,
                   sensitivity: float = 1.5) -> Dict[str, Any]:
    """
    Erkennt Ausreißer mit verschiedenen Methoden
    """
    try:
        if not data or not all(isinstance(x, (int, float)) for x in data):
            return {'error': 'invalid_data', 'message': 'Data must be numeric'}
        
        if methods is None:
            methods = ['iqr', 'z_score', 'modified_z_score', 'isolation_forest']
        
        outlier_result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'sample_size': len(data),
            'data_summary': _get_basic_stats(data),
            'outlier_analyses': {},
            'sensitivity': sensitivity
        }
        
        # Apply different outlier detection methods
        if 'iqr' in methods:
            outlier_result['outlier_analyses']['iqr'] = _detect_outliers_iqr(data, sensitivity)
        
        if 'z_score' in methods:
            outlier_result['outlier_analyses']['z_score'] = _detect_outliers_z_score(data, sensitivity)
        
        if 'modified_z_score' in methods:
            outlier_result['outlier_analyses']['modified_z_score'] = _detect_outliers_modified_z_score(data)
        
        if 'isolation_forest' in methods:
            outlier_result['outlier_analyses']['isolation_forest'] = _detect_outliers_isolation_forest(data)
        
        # Consensus outliers (outliers detected by multiple methods)
        outlier_result['consensus_outliers'] = _find_consensus_outliers(
            outlier_result['outlier_analyses']
        )
        
        return outlier_result
        
    except Exception as e:
        logger.error(f"Outlier detection failed: {e}")
        return {
            'error': str(e),
            'analysis_timestamp': datetime.now().isoformat()
        }

def calculate_confidence_intervals(data: List[Union[int, float]], 
                                 confidence_levels: List[float] = None,
                                 parameter_types: List[str] = None) -> Dict[str, Any]:
    """
    Berechnet Konfidenzintervalle für verschiedene Parameter
    """
    try:
        if not data or not all(isinstance(x, (int, float)) for x in data):
            return {'error': 'invalid_data', 'message': 'Data must be numeric'}
        
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        if parameter_types is None:
            parameter_types = ['mean', 'proportion', 'variance']
        
        ci_result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'sample_size': len(data),
            'confidence_intervals': {}
        }
        
        for confidence_level in confidence_levels:
            ci_result['confidence_intervals'][f'{confidence_level:.0%}'] = {}
            
            for param_type in parameter_types:
                if param_type == 'mean':
                    ci_result['confidence_intervals'][f'{confidence_level:.0%}']['mean'] = \
                        _calculate_mean_confidence_interval(data, confidence_level)
                elif param_type == 'proportion':
                    ci_result['confidence_intervals'][f'{confidence_level:.0%}']['proportion'] = \
                        _calculate_proportion_confidence_interval(data, confidence_level)
                elif param_type == 'variance':
                    ci_result['confidence_intervals'][f'{confidence_level:.0%}']['variance'] = \
                        _calculate_variance_confidence_interval(data, confidence_level)
        
        return ci_result
        
    except Exception as e:
        logger.error(f"Confidence interval calculation failed: {e}")
        return {
            'error': str(e),
            'analysis_timestamp': datetime.now().isoformat()
        }

# Helper Functions für Numeric Statistics

def _calculate_numeric_statistics(data: List[Union[int, float]], confidence_level: float) -> Dict[str, Any]:
    """Berechnet numerische Statistiken"""
    try:
        n = len(data)
        
        # Central Tendency Measures
        mean_val = statistics.mean(data)
        median_val = statistics.median(data)
        try:
            mode_val = statistics.mode(data)
        except statistics.StatisticsError:
            mode_val = None
        
        # Dispersion Measures
        variance_val = statistics.variance(data) if n > 1 else 0
        std_dev_val = statistics.stdev(data) if n > 1 else 0
        range_val = max(data) - min(data)
        
        # Quartiles and IQR
        q1 = _calculate_percentile(data, 25)
        q3 = _calculate_percentile(data, 75)
        iqr_val = q3 - q1
        
        # Shape Measures
        skewness_val = _calculate_skewness(data)
        kurtosis_val = _calculate_kurtosis(data)
        
        # Confidence Interval for Mean
        mean_ci = _calculate_mean_confidence_interval(data, confidence_level)
        
        # Outliers
        outliers = _detect_outliers_iqr(data, 1.5)['outliers']
        
        return {
            'central_tendency': {
                'mean': mean_val,
                'median': median_val,
                'mode': mode_val,
                'geometric_mean': _calculate_geometric_mean(data),
                'harmonic_mean': _calculate_harmonic_mean(data)
            },
            'dispersion': {
                'variance': variance_val,
                'standard_deviation': std_dev_val,
                'range': range_val,
                'interquartile_range': iqr_val,
                'coefficient_of_variation': std_dev_val / mean_val if mean_val != 0 else 0,
                'mean_absolute_deviation': _calculate_mad(data)
            },
            'shape': {
                'skewness': skewness_val,
                'kurtosis': kurtosis_val,
                'distribution_type': _classify_distribution_shape(skewness_val, kurtosis_val)
            },
            'percentiles': {
                'min': min(data),
                'q1': q1,
                'median': median_val,
                'q3': q3,
                'max': max(data),
                'p10': _calculate_percentile(data, 10),
                'p90': _calculate_percentile(data, 90)
            },
            'confidence_intervals': {
                f'{confidence_level:.0%}_mean': mean_ci
            },
            'outliers': {
                'outlier_values': outliers,
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(data) * 100
            }
        }
        
    except Exception as e:
        logger.debug(f"Numeric statistics calculation failed: {e}")
        return {'error': str(e)}

def _calculate_categorical_statistics(data: List[Any]) -> Dict[str, Any]:
    """Berechnet kategorische Statistiken"""
    try:
        # Convert to strings for consistent handling
        str_data = [str(x) for x in data if x is not None]
        
        if not str_data:
            return {'error': 'no_valid_categorical_data'}
        
        # Frequency analysis
        frequency_counter = Counter(str_data)
        total_count = len(str_data)
        
        # Most common values
        most_common = frequency_counter.most_common(10)
        
        # Calculate diversity measures
        unique_values = len(frequency_counter)
        diversity_index = _calculate_shannon_diversity(frequency_counter, total_count)
        
        return {
            'frequency_analysis': {
                'unique_values': unique_values,
                'total_observations': total_count,
                'most_common': [{'value': val, 'count': count, 'percentage': count/total_count*100} 
                               for val, count in most_common],
                'least_common': frequency_counter.most_common()[-5:] if len(frequency_counter) > 5 else []
            },
            'diversity_measures': {
                'shannon_diversity_index': diversity_index,
                'simpson_diversity_index': _calculate_simpson_diversity(frequency_counter, total_count),
                'uniformity_index': diversity_index / math.log(unique_values) if unique_values > 1 else 0
            },
            'distribution_properties': {
                'mode': frequency_counter.most_common(1)[0][0],
                'mode_frequency': frequency_counter.most_common(1)[0][1],
                'mode_percentage': frequency_counter.most_common(1)[0][1] / total_count * 100,
                'is_uniform': len(set(frequency_counter.values())) == 1
            }
        }
        
    except Exception as e:
        logger.debug(f"Categorical statistics calculation failed: {e}")
        return {'error': str(e)}

def _assess_data_quality(all_data: List, numeric_data: List, non_numeric_data: List) -> Dict[str, Any]:
    """Bewertet Datenqualität"""
    try:
        total_count = len(all_data)
        
        if total_count == 0:
            return {'quality_score': 0.0, 'quality_rating': 'no_data'}
        
        # Completeness
        missing_count = len([x for x in all_data if x is None])
        completeness_score = (total_count - missing_count) / total_count
        
        # Consistency (simplified)
        consistency_score = 0.8  # Default assumption
        if numeric_data:
            # Check for reasonable numeric values
            outliers = _detect_outliers_iqr(numeric_data, 2.0)['outliers']
            consistency_score = max(0.5, 1.0 - len(outliers) / len(numeric_data))
        
        # Validity (simplified)
        validity_score = 0.9 if numeric_data or non_numeric_data else 0.1
        
        # Overall quality score
        quality_score = (completeness_score + consistency_score + validity_score) / 3
        
        return {
            'quality_score': quality_score,
            'quality_rating': _rate_data_quality(quality_score),
            'quality_components': {
                'completeness': {'score': completeness_score, 'missing_values': missing_count},
                'consistency': {'score': consistency_score},
                'validity': {'score': validity_score}
            },
            'recommendations': _generate_quality_recommendations(
                completeness_score, consistency_score, validity_score
            )
        }
        
    except Exception as e:
        logger.debug(f"Data quality assessment failed: {e}")
        return {'quality_score': 0.5, 'quality_rating': 'unknown', 'error': str(e)}

# Mathematical Helper Functions

def _calculate_percentile(data: List[float], percentile: float) -> float:
    """Berechnet Perzentil"""
    try:
        sorted_data = sorted(data)
        n = len(sorted_data)
        k = (n - 1) * (percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_data[int(k)]
        
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1
    except:
        return 0.0

def _calculate_skewness(data: List[float]) -> float:
    """Berechnet Schiefe (Skewness)"""
    try:
        n = len(data)
        if n < 3:
            return 0.0
        
        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data)
        
        if std_val == 0:
            return 0.0
        
        skew_sum = sum(((x - mean_val) / std_val) ** 3 for x in data)
        return (n / ((n - 1) * (n - 2))) * skew_sum
    except:
        return 0.0

def _calculate_kurtosis(data: List[float]) -> float:
    """Berechnet Kurtosis (Wölbung)"""
    try:
        n = len(data)
        if n < 4:
            return 0.0
        
        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data)
        
        if std_val == 0:
            return 0.0
        
        # Excess kurtosis (normal distribution has kurtosis of 3)
        kurt_sum = sum(((x - mean_val) / std_val) ** 4 for x in data)
        kurtosis_val = (n * (n + 1) * kurt_sum) / ((n - 1) * (n - 2) * (n - 3))
        excess_kurtosis = kurtosis_val - 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
        
        return excess_kurtosis
    except:
        return 0.0

def _calculate_geometric_mean(data: List[float]) -> Optional[float]:
    """Berechnet geometrisches Mittel"""
    try:
        if any(x <= 0 for x in data):
            return None  # Geometric mean undefined for non-positive values
        
        product = 1.0
        for x in data:
            product *= x
        
        return product ** (1.0 / len(data))
    except:
        return None

def _calculate_harmonic_mean(data: List[float]) -> Optional[float]:
    """Berechnet harmonisches Mittel"""
    try:
        if any(x <= 0 for x in data):
            return None  # Harmonic mean undefined for non-positive values
        
        reciprocal_sum = sum(1.0 / x for x in data)
        return len(data) / reciprocal_sum
    except:
        return None

def _calculate_mad(data: List[float]) -> float:
    """Berechnet Mean Absolute Deviation"""
    try:
        mean_val = statistics.mean(data)
        return sum(abs(x - mean_val) for x in data) / len(data)
    except:
        return 0.0

def _classify_distribution_shape(skewness: float, kurtosis: float) -> str:
    """Klassifiziert Verteilungsform"""
    try:
        shape_desc = []
        
        # Skewness classification
        if abs(skewness) < 0.5:
            shape_desc.append('approximately_symmetric')
        elif skewness > 0.5:
            shape_desc.append('right_skewed')
        elif skewness < -0.5:
            shape_desc.append('left_skewed')
        
        # Kurtosis classification
        if abs(kurtosis) < 0.5:
            shape_desc.append('mesokurtic')
        elif kurtosis > 0.5:
            shape_desc.append('leptokurtic')
        elif kurtosis < -0.5:
            shape_desc.append('platykurtic')
        
        return '_'.join(shape_desc)
    except:
        return 'unknown_shape'

def _calculate_shannon_diversity(frequency_counter: Counter, total_count: int) -> float:
    """Berechnet Shannon Diversity Index"""
    try:
        diversity = 0.0
        for count in frequency_counter.values():
            if count > 0:
                proportion = count / total_count
                diversity -= proportion * math.log(proportion)
        return diversity
    except:
        return 0.0

def _calculate_simpson_diversity(frequency_counter: Counter, total_count: int) -> float:
    """Berechnet Simpson Diversity Index"""
    try:
        diversity = 0.0
        for count in frequency_counter.values():
            if count > 0:
                proportion = count / total_count
                diversity += proportion ** 2
        return 1.0 - diversity
    except:
        return 0.0

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

def _generate_quality_recommendations(completeness: float, consistency: float, validity: float) -> List[str]:
    """Generiert Empfehlungen zur Datenqualität"""
    recommendations = []
    
    if completeness < 0.8:
        recommendations.append('Consider addressing missing values through imputation or collection')
    
    if consistency < 0.7:
        recommendations.append('Review data for outliers and inconsistencies')
    
    if validity < 0.8:
        recommendations.append('Validate data sources and collection methods')
    
    if not recommendations:
        recommendations.append('Data quality appears satisfactory')
    
    return recommendations

# Basic Statistics Helper
def _get_basic_stats(data: List[Union[int, float]]) -> Dict[str, float]:
    """Gibt grundlegende Statistiken zurück"""
    try:
        return {
            'count': len(data),
            'mean': statistics.mean(data),
            'median': statistics.median(data),
            'std_dev': statistics.stdev(data) if len(data) > 1 else 0.0,
            'min': min(data),
            'max': max(data)
        }
    except:
        return {'count': 0, 'mean': 0, 'median': 0, 'std_dev': 0, 'min': 0, 'max': 0}

__all__ = [
    'StatisticalSummary',
    'DistributionAnalysis', 
    'CorrelationResult',
    'calculate_descriptive_statistics',
    'analyze_distribution',
    'calculate_correlation',
    'perform_hypothesis_testing',
    'detect_outliers',
    'calculate_confidence_intervals'
]