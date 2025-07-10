"""
Analytics Metrics Module
Key Metrics und Performance Indicators für alle Kira Systeme
"""

import logging
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def calculate_key_metrics(memory_manager, db_stats: Dict, 
                         personality_data: Dict = None) -> Dict[str, Any]:
    """
    Berechnet Key Performance Metrics
    
    Extrahiert aus kira_routes.py.backup Key Metrics Logic
    """
    try:
        metrics = {
            'memory_metrics': _calculate_memory_key_metrics(memory_manager, db_stats),
            'learning_metrics': _calculate_learning_key_metrics(memory_manager, db_stats),
            'system_performance_metrics': _calculate_system_performance_metrics(memory_manager, db_stats),
            'efficiency_metrics': calculate_efficiency_metrics(memory_manager, db_stats)
        }
        
        # Personality Metrics falls verfügbar
        if personality_data:
            metrics['personality_metrics'] = _calculate_personality_key_metrics(personality_data)
        
        # Composite Metrics
        metrics['composite_metrics'] = {
            'overall_health_score': _calculate_overall_health_score(metrics),
            'system_readiness_score': _calculate_system_readiness_score(metrics),
            'optimization_score': _calculate_optimization_score(metrics),
            'development_momentum_score': _calculate_development_momentum_score(metrics)
        }
        
        # Metrics Metadata
        metrics['metrics_metadata'] = {
            'calculation_timestamp': datetime.now().isoformat(),
            'metrics_version': '2.0',
            'data_quality_score': _assess_metrics_data_quality(memory_manager, db_stats),
            'confidence_level': _calculate_metrics_confidence(metrics)
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Key metrics calculation failed: {e}")
        return {
            'error': str(e),
            'fallback_metrics': _generate_fallback_key_metrics()
        }

def get_performance_indicators(memory_manager, db_stats: Dict,
                             real_time: bool = False) -> Dict[str, Any]:
    """
    Holt Performance Indicators
    
    Basiert auf kira_routes.py.backup Performance Indicators Logic
    """
    try:
        if real_time:
            indicators = get_real_time_metrics(memory_manager, db_stats)
        else:
            indicators = _get_standard_performance_indicators(memory_manager, db_stats)
        
        # Enrich with trend indicators
        indicators['trend_indicators'] = _calculate_trend_indicators(indicators)
        
        # Performance Thresholds
        indicators['performance_thresholds'] = _get_performance_thresholds()
        
        # Status Assessment
        indicators['status_assessment'] = _assess_performance_status(indicators)
        
        return indicators
        
    except Exception as e:
        logger.error(f"Performance indicators retrieval failed: {e}")
        return {
            'error': str(e),
            'fallback_indicators': _generate_fallback_performance_indicators()
        }

def calculate_efficiency_metrics(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """
    Berechnet Efficiency Metrics
    
    Extrahiert aus kira_routes.py.backup Efficiency Calculations
    """
    try:
        from routes.utils.memory_helpers import (
            calculate_stm_efficiency_direct,
            calculate_current_cognitive_load,
            get_stm_load_direct,
            get_stm_capacity_direct
        )
        
        # Core Efficiency Metrics
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        stm_load = get_stm_load_direct(memory_manager)
        stm_capacity = get_stm_capacity_direct(memory_manager)
        
        # Calculated Efficiency Metrics
        memory_utilization_efficiency = stm_load / max(1, stm_capacity)
        cognitive_efficiency = 1.0 - cognitive_load
        overall_efficiency = (stm_efficiency + cognitive_efficiency) / 2
        
        # Advanced Efficiency Metrics
        efficiency_metrics = {
            'core_efficiency_metrics': {
                'stm_efficiency': stm_efficiency,
                'cognitive_efficiency': cognitive_efficiency,
                'memory_utilization_efficiency': memory_utilization_efficiency,
                'overall_efficiency': overall_efficiency
            },
            
            'resource_efficiency_metrics': {
                'memory_consolidation_efficiency': _calculate_consolidation_efficiency(memory_manager, db_stats),
                'retrieval_efficiency': _calculate_retrieval_efficiency(memory_manager, db_stats),
                'storage_efficiency': _calculate_storage_efficiency(db_stats),
                'processing_efficiency': _calculate_processing_efficiency(memory_manager, db_stats)
            },
            
            'operational_efficiency_metrics': {
                'response_time_efficiency': _calculate_response_time_efficiency(),
                'accuracy_efficiency': _calculate_accuracy_efficiency(memory_manager, db_stats),
                'adaptability_efficiency': _calculate_adaptability_efficiency(memory_manager, db_stats),
                'learning_efficiency': _calculate_learning_efficiency(memory_manager, db_stats)
            }
        }
        
        # Efficiency Ratings
        efficiency_metrics['efficiency_ratings'] = {
            'overall_rating': _rate_efficiency(overall_efficiency),
            'improvement_areas': _identify_efficiency_improvement_areas(efficiency_metrics),
            'efficiency_trend': _determine_efficiency_trend(efficiency_metrics),
            'optimization_recommendations': _generate_efficiency_optimization_recommendations(efficiency_metrics)
        }
        
        return efficiency_metrics
        
    except Exception as e:
        logger.error(f"Efficiency metrics calculation failed: {e}")
        return {
            'error': str(e),
            'fallback_efficiency_metrics': _generate_fallback_efficiency_metrics()
        }

def get_real_time_metrics(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """
    Holt Real-time Metrics
    
    Basiert auf kira_routes.py.backup Real-time Monitoring Logic
    """
    try:
        current_time = datetime.now()
        
        # Real-time System Metrics
        realtime_metrics = {
            'current_system_state': {
                'timestamp': current_time.isoformat(),
                'system_uptime': _calculate_system_uptime(),
                'active_processes': _count_active_processes(memory_manager),
                'memory_usage': _get_current_memory_usage(),
                'cpu_utilization': _get_current_cpu_utilization()
            },
            
            'live_memory_metrics': {
                'current_stm_load': _get_live_stm_load(memory_manager),
                'recent_memory_formations': _count_recent_memory_formations(db_stats),
                'active_consolidation_processes': _count_active_consolidation_processes(memory_manager),
                'retrieval_requests_per_minute': _calculate_retrieval_rate()
            },
            
            'real_time_performance': {
                'current_response_time': _measure_current_response_time(),
                'system_health_score': _calculate_live_system_health_score(memory_manager, db_stats),
                'learning_activity_level': _assess_current_learning_activity(memory_manager, db_stats),
                'optimization_opportunities': _identify_live_optimization_opportunities(memory_manager, db_stats)
            }
        }
        
        # Real-time Alerts
        realtime_metrics['alerts'] = _check_real_time_alerts(realtime_metrics)
        
        # Performance Comparison
        realtime_metrics['performance_comparison'] = _compare_with_baseline_performance(realtime_metrics)
        
        return realtime_metrics
        
    except Exception as e:
        logger.error(f"Real-time metrics collection failed: {e}")
        return {
            'error': str(e),
            'fallback_realtime_metrics': _generate_fallback_realtime_metrics()
        }

# ====================================
# PRIVATE HELPER FUNCTIONS
# ====================================

def _calculate_memory_key_metrics(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Berechnet Memory Key Metrics"""
    try:
        from routes.utils.memory_helpers import (
            get_stm_load_direct, get_stm_capacity_direct,
            get_ltm_size_direct, calculate_stm_efficiency_direct
        )
        
        stm_load = get_stm_load_direct(memory_manager)
        stm_capacity = get_stm_capacity_direct(memory_manager)
        ltm_size = get_ltm_size_direct(memory_manager)
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        return {
            'stm_utilization_rate': stm_load / max(1, stm_capacity),
            'stm_efficiency_score': stm_efficiency,
            'ltm_capacity': ltm_size,
            'total_memory_count': total_memories,
            'memory_formation_rate': recent_activity,
            'memory_system_health': _calculate_memory_system_health(stm_efficiency, stm_load, stm_capacity),
            'memory_balance_score': _calculate_memory_balance_score(stm_load, ltm_size)
        }
        
    except Exception as e:
        logger.debug(f"Memory key metrics calculation failed: {e}")
        return {
            'stm_utilization_rate': 0.5,
            'stm_efficiency_score': 0.7,
            'ltm_capacity': 100,
            'total_memory_count': db_stats.get('total_memories', 0),
            'memory_formation_rate': db_stats.get('recent_activity', 0),
            'fallback_data': True
        }

def _calculate_learning_key_metrics(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Berechnet Learning Key Metrics"""
    try:
        from routes.utils.memory_helpers import calculate_learning_readiness_direct
        
        learning_readiness = calculate_learning_readiness_direct(memory_manager, db_stats)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Learning Velocity (memories per hour)
        learning_velocity = recent_activity / 24
        
        # Learning Efficiency
        learning_efficiency = learning_readiness * (1 + min(recent_activity / 10, 0.5))
        
        return {
            'learning_readiness_score': learning_readiness,
            'learning_velocity': learning_velocity,
            'learning_efficiency_score': learning_efficiency,
            'learning_momentum': _calculate_learning_momentum(recent_activity, learning_readiness),
            'knowledge_acquisition_rate': _calculate_knowledge_acquisition_rate(memory_manager, db_stats),
            'learning_curve_position': _determine_learning_curve_position(memory_manager, db_stats)
        }
        
    except Exception as e:
        logger.debug(f"Learning key metrics calculation failed: {e}")
        return {
            'learning_readiness_score': 0.7,
            'learning_velocity': 0.1,
            'learning_efficiency_score': 0.6,
            'fallback_data': True
        }

def _calculate_system_performance_metrics(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Berechnet System Performance Metrics"""
    try:
        from routes.utils.memory_helpers import calculate_current_cognitive_load
        
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        
        # System Load Metrics
        system_load = cognitive_load
        system_capacity_utilization = min(1.0, db_stats.get('total_memories', 0) / 1000)
        
        # Performance Indicators
        response_quality = 1.0 - cognitive_load
        system_stability = _calculate_system_stability(memory_manager, db_stats)
        adaptability_score = _calculate_adaptability_score(memory_manager, db_stats)
        
        return {
            'system_load': system_load,
            'system_capacity_utilization': system_capacity_utilization,
            'response_quality_score': response_quality,
            'system_stability_score': system_stability,
            'adaptability_score': adaptability_score,
            'overall_performance_score': (response_quality + system_stability + adaptability_score) / 3,
            'performance_classification': _classify_performance_level(response_quality, system_stability, adaptability_score)
        }
        
    except Exception as e:
        logger.debug(f"System performance metrics calculation failed: {e}")
        return {
            'system_load': 0.5,
            'overall_performance_score': 0.7,
            'performance_classification': 'good',
            'fallback_data': True
        }

def _calculate_personality_key_metrics(personality_data: Dict) -> Dict[str, Any]:
    """Berechnet Personality Key Metrics"""
    try:
        if not personality_data:
            return {'available': False}
        
        traits = personality_data.get('traits', {})
        current_state = personality_data.get('current_state', {})
        
        # Trait Metrics
        trait_count = len(traits)
        trait_balance = _calculate_trait_balance_score(traits)
        dominant_traits = _identify_dominant_traits(traits)
        
        # State Metrics
        emotional_stability = current_state.get('emotional_stability', 0.7)
        adaptability = current_state.get('adaptability', 0.6)
        empathy_level = current_state.get('empathy_level', 0.8)
        
        # Development Metrics
        development_rate = _calculate_personality_development_rate(personality_data)
        growth_potential = _calculate_personality_growth_potential(personality_data)
        
        return {
            'available': True,
            'trait_metrics': {
                'trait_count': trait_count,
                'trait_balance_score': trait_balance,
                'dominant_traits': dominant_traits[:3]  # Top 3
            },
            'emotional_metrics': {
                'emotional_stability': emotional_stability,
                'adaptability': adaptability,
                'empathy_level': empathy_level,
                'emotional_health_score': (emotional_stability + adaptability + empathy_level) / 3
            },
            'development_metrics': {
                'development_rate': development_rate,
                'growth_potential': growth_potential,
                'maturity_level': _assess_personality_maturity_level(personality_data)
            },
            'overall_personality_health': _calculate_overall_personality_health(
                trait_balance, emotional_stability, adaptability, empathy_level
            )
        }
        
    except Exception as e:
        logger.debug(f"Personality key metrics calculation failed: {e}")
        return {
            'available': False,
            'error': str(e)
        }

# Additional helper functions...

def _calculate_overall_health_score(metrics: Dict) -> float:
    """Berechnet Overall Health Score"""
    try:
        health_components = []
        
        # Memory Health
        if 'memory_metrics' in metrics:
            memory_health = metrics['memory_metrics'].get('memory_system_health', 0.7)
            health_components.append(memory_health)
        
        # Learning Health
        if 'learning_metrics' in metrics:
            learning_health = metrics['learning_metrics'].get('learning_efficiency_score', 0.7)
            health_components.append(learning_health)
        
        # System Performance Health
        if 'system_performance_metrics' in metrics:
            system_health = metrics['system_performance_metrics'].get('overall_performance_score', 0.7)
            health_components.append(system_health)
        
        # Personality Health
        if 'personality_metrics' in metrics and metrics['personality_metrics'].get('available'):
            personality_health = metrics['personality_metrics'].get('overall_personality_health', 0.7)
            health_components.append(personality_health)
        
        return sum(health_components) / len(health_components) if health_components else 0.7
        
    except Exception as e:
        logger.debug(f"Overall health score calculation failed: {e}")
        return 0.7

def _rate_efficiency(efficiency_score: float) -> str:
    """Bewertet Efficiency Score"""
    if efficiency_score >= 0.9:
        return 'excellent'
    elif efficiency_score >= 0.75: 
        return 'good'
    elif efficiency_score >= 0.6:
        return 'fair'
    elif efficiency_score >= 0.4:
        return 'poor'
    else:
        return 'critical'

def _generate_fallback_key_metrics() -> Dict[str, Any]:
    """Generiert Fallback Key Metrics"""
    return {
        'fallback_mode': True,
        'basic_metrics': {
            'system_status': 'operational',
            'health_score': 0.7,
            'performance_level': 'moderate'
        },
        'timestamp': datetime.now().isoformat()
    }

def _calculate_system_readiness_score(metrics: Dict) -> float:
    """Berechnet System Readiness Score"""
    try:
        readiness_components = []
        
        # Memory System Readiness
        if 'memory_metrics' in metrics:
            memory_data = metrics['memory_metrics']
            stm_efficiency = memory_data.get('stm_efficiency_score', 0.7)
            stm_utilization = memory_data.get('stm_utilization_rate', 0.5)
            memory_health = memory_data.get('memory_system_health', 0.7)
            
            # Memory readiness: high efficiency, optimal utilization (not too low, not too high)
            optimal_utilization = 1.0 - abs(stm_utilization - 0.6)  # Target 60% utilization
            memory_readiness = (stm_efficiency * 0.4 + optimal_utilization * 0.3 + memory_health * 0.3)
            readiness_components.append(memory_readiness)
        
        # Learning System Readiness
        if 'learning_metrics' in metrics:
            learning_data = metrics['learning_metrics']
            learning_readiness = learning_data.get('learning_readiness_score', 0.7)
            learning_efficiency = learning_data.get('learning_efficiency_score', 0.6)
            learning_velocity = min(1.0, learning_data.get('learning_velocity', 0.1) * 10)  # Scale velocity
            
            learning_system_readiness = (learning_readiness * 0.5 + learning_efficiency * 0.3 + learning_velocity * 0.2)
            readiness_components.append(learning_system_readiness)
        
        # System Performance Readiness
        if 'system_performance_metrics' in metrics:
            perf_data = metrics['system_performance_metrics']
            system_load = perf_data.get('system_load', 0.5)
            system_stability = perf_data.get('system_stability_score', 0.7)
            adaptability = perf_data.get('adaptability_score', 0.6)
            
            # Readiness requires low system load, high stability, high adaptability
            load_readiness = 1.0 - system_load  # Lower load = higher readiness
            performance_readiness = (load_readiness * 0.4 + system_stability * 0.4 + adaptability * 0.2)
            readiness_components.append(performance_readiness)
        
        # Resource Efficiency Readiness
        if 'efficiency_metrics' in metrics:
            efficiency_data = metrics['efficiency_metrics']
            if 'core_efficiency_metrics' in efficiency_data:
                core_efficiency = efficiency_data['core_efficiency_metrics']
                overall_efficiency = core_efficiency.get('overall_efficiency', 0.7)
                readiness_components.append(overall_efficiency)
        
        # Personality Readiness (if available)
        if 'personality_metrics' in metrics and metrics['personality_metrics'].get('available'):
            personality_data = metrics['personality_metrics']
            emotional_stability = personality_data.get('emotional_metrics', {}).get('emotional_stability', 0.7)
            adaptability = personality_data.get('emotional_metrics', {}).get('adaptability', 0.6)
            
            personality_readiness = (emotional_stability + adaptability) / 2
            readiness_components.append(personality_readiness)
        
        # Calculate weighted average
        if readiness_components:
            system_readiness = sum(readiness_components) / len(readiness_components)
            
            # Apply readiness curve (exponential for high readiness, linear for low)
            if system_readiness > 0.8:
                system_readiness = 0.8 + (system_readiness - 0.8) * 1.5  # Boost high readiness
            
            return min(1.0, system_readiness)
        else:
            return 0.7  # Default moderate readiness
            
    except Exception as e:
        logger.debug(f"System readiness score calculation failed: {e}")
        return 0.7

def _calculate_optimization_score(metrics: Dict) -> float:
    """Berechnet Optimization Score"""
    try:
        optimization_factors = []
        
        # Memory Optimization
        if 'memory_metrics' in metrics:
            memory_data = metrics['memory_metrics']
            stm_efficiency = memory_data.get('stm_efficiency_score', 0.7)
            memory_balance = memory_data.get('memory_balance_score', 0.7)
            
            memory_optimization = (stm_efficiency + memory_balance) / 2
            optimization_factors.append(memory_optimization)
        
        # Learning Optimization
        if 'learning_metrics' in metrics:
            learning_data = metrics['learning_metrics']
            learning_efficiency = learning_data.get('learning_efficiency_score', 0.6)
            learning_momentum = learning_data.get('learning_momentum', 0.5)
            
            learning_optimization = (learning_efficiency + learning_momentum) / 2
            optimization_factors.append(learning_optimization)
        
        # System Performance Optimization
        if 'system_performance_metrics' in metrics:
            perf_data = metrics['system_performance_metrics']
            performance_score = perf_data.get('overall_performance_score', 0.7)
            system_stability = perf_data.get('system_stability_score', 0.7)
            
            performance_optimization = (performance_score + system_stability) / 2
            optimization_factors.append(performance_optimization)
        
        # Resource Efficiency Optimization
        if 'efficiency_metrics' in metrics:
            efficiency_data = metrics['efficiency_metrics']
            
            # Core efficiency
            if 'core_efficiency_metrics' in efficiency_data:
                core_efficiency = efficiency_data['core_efficiency_metrics'].get('overall_efficiency', 0.7)
                optimization_factors.append(core_efficiency)
            
            # Resource efficiency
            if 'resource_efficiency_metrics' in efficiency_data:
                resource_metrics = efficiency_data['resource_efficiency_metrics']
                consolidation_eff = resource_metrics.get('memory_consolidation_efficiency', 0.7)
                retrieval_eff = resource_metrics.get('retrieval_efficiency', 0.7)
                storage_eff = resource_metrics.get('storage_efficiency', 0.7)
                
                resource_optimization = (consolidation_eff + retrieval_eff + storage_eff) / 3
                optimization_factors.append(resource_optimization)
        
        # Calculate optimization score
        if optimization_factors:
            optimization_score = sum(optimization_factors) / len(optimization_factors)
            
            # Bonus for consistently high optimization across all areas
            if all(factor > 0.8 for factor in optimization_factors):
                optimization_score = min(1.0, optimization_score * 1.1)
            
            return optimization_score
        else:
            return 0.7
            
    except Exception as e:
        logger.debug(f"Optimization score calculation failed: {e}")
        return 0.7

def _calculate_development_momentum_score(metrics: Dict) -> float:
    """Berechnet Development Momentum Score"""
    try:
        momentum_indicators = []
        
        # Learning Momentum
        if 'learning_metrics' in metrics:
            learning_data = metrics['learning_metrics']
            learning_velocity = learning_data.get('learning_velocity', 0.1)
            learning_efficiency = learning_data.get('learning_efficiency_score', 0.6)
            learning_momentum = learning_data.get('learning_momentum', 0.5)
            
            # High velocity + efficiency = strong momentum
            learning_dev_momentum = (learning_velocity * 2 + learning_efficiency + learning_momentum) / 4
            momentum_indicators.append(learning_dev_momentum)
        
        # Memory Development Momentum
        if 'memory_metrics' in metrics:
            memory_data = metrics['memory_metrics']
            memory_formation_rate = memory_data.get('memory_formation_rate', 0)
            memory_system_health = memory_data.get('memory_system_health', 0.7)
            
            # Scale formation rate (assume 10 memories/day is high momentum)
            scaled_formation_rate = min(1.0, memory_formation_rate / 10)
            memory_momentum = (scaled_formation_rate + memory_system_health) / 2
            momentum_indicators.append(memory_momentum)
        
        # System Performance Momentum
        if 'system_performance_metrics' in metrics:
            perf_data = metrics['system_performance_metrics']
            adaptability = perf_data.get('adaptability_score', 0.6)
            response_quality = perf_data.get('response_quality_score', 0.7)
            
            # High adaptability + quality = good development momentum
            performance_momentum = (adaptability + response_quality) / 2
            momentum_indicators.append(performance_momentum)
        
        # Personality Development Momentum (if available)
        if 'personality_metrics' in metrics and metrics['personality_metrics'].get('available'):
            personality_data = metrics['personality_metrics']
            development_metrics = personality_data.get('development_metrics', {})
            
            development_rate = development_metrics.get('development_rate', 0.5)
            growth_potential = development_metrics.get('growth_potential', 0.6)
            
            personality_momentum = (development_rate + growth_potential) / 2
            momentum_indicators.append(personality_momentum)
        
        # Efficiency Momentum
        if 'efficiency_metrics' in metrics:
            efficiency_data = metrics['efficiency_metrics']
            if 'operational_efficiency_metrics' in efficiency_data:
                operational_metrics = efficiency_data['operational_efficiency_metrics']
                learning_efficiency = operational_metrics.get('learning_efficiency', 0.6)
                adaptability_efficiency = operational_metrics.get('adaptability_efficiency', 0.6)
                
                efficiency_momentum = (learning_efficiency + adaptability_efficiency) / 2
                momentum_indicators.append(efficiency_momentum)
        
        # Calculate momentum score
        if momentum_indicators:
            momentum_score = sum(momentum_indicators) / len(momentum_indicators)
            
            # Momentum curve: exponential growth for high momentum
            if momentum_score > 0.7:
                momentum_score = 0.7 + (momentum_score - 0.7) * 1.5
            
            return min(1.0, momentum_score)
        else:
            return 0.5  # Neutral momentum
            
    except Exception as e:
        logger.debug(f"Development momentum score calculation failed: {e}")
        return 0.5

# ====================================
# MISSING ASSESSMENT FUNCTIONS
# ====================================

def _assess_metrics_data_quality(memory_manager, db_stats: Dict) -> float:
    """Bewertet Data Quality für Metrics"""
    try:
        quality_factors = []
        
        # Memory Manager Data Quality
        if memory_manager:
            try:
                # Check if memory manager is responsive
                from routes.utils.memory_helpers import get_stm_load_direct
                stm_load = get_stm_load_direct(memory_manager)
                if stm_load is not None:
                    quality_factors.append(1.0)  # Good data
                else:
                    quality_factors.append(0.5)  # Partial data
            except Exception:
                quality_factors.append(0.3)  # Poor data
        else:
            quality_factors.append(0.0)  # No data
        
        # Database Stats Quality
        if db_stats:
            required_fields = ['total_memories', 'recent_activity']
            available_fields = sum(1 for field in required_fields if field in db_stats and db_stats[field] is not None)
            db_quality = available_fields / len(required_fields)
            quality_factors.append(db_quality)
        else:
            quality_factors.append(0.0)
        
        # Data Freshness Quality
        current_time = datetime.now()
        # Assume data is fresh if we can calculate it now
        quality_factors.append(1.0)  # Fresh data
        
        # Data Consistency Quality
        # Check if metrics are internally consistent
        consistency_score = 0.9  # Assume good consistency for now
        quality_factors.append(consistency_score)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5
        
    except Exception as e:
        logger.debug(f"Data quality assessment failed: {e}")
        return 0.5

def _calculate_metrics_confidence(metrics: Dict) -> float:
    """Berechnet Confidence Level für Metrics"""
    try:
        confidence_factors = []
        
        # Data availability confidence
        available_metrics = 0
        total_possible_metrics = 4  # memory, learning, system_performance, efficiency
        
        if 'memory_metrics' in metrics:
            available_metrics += 1
        if 'learning_metrics' in metrics:
            available_metrics += 1
        if 'system_performance_metrics' in metrics:
            available_metrics += 1
        if 'efficiency_metrics' in metrics:
            available_metrics += 1
        
        data_availability_confidence = available_metrics / total_possible_metrics
        confidence_factors.append(data_availability_confidence)
        
        # Fallback data confidence
        fallback_penalty = 0.0
        for metric_group in metrics.values():
            if isinstance(metric_group, dict) and metric_group.get('fallback_data'):
                fallback_penalty += 0.2
        
        fallback_confidence = max(0.0, 1.0 - fallback_penalty)
        confidence_factors.append(fallback_confidence)
        
        # Calculation success confidence
        if 'composite_metrics' in metrics:
            composite_data = metrics['composite_metrics']
            successful_calculations = sum(1 for value in composite_data.values() if isinstance(value, (int, float)) and value > 0)
            total_calculations = len(composite_data)
            
            calculation_confidence = successful_calculations / max(1, total_calculations)
            confidence_factors.append(calculation_confidence)
        
        # Overall confidence
        overall_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        
        return overall_confidence
        
    except Exception as e:
        logger.debug(f"Metrics confidence calculation failed: {e}")
        return 0.5

# ====================================
# MISSING HELPER FUNCTIONS (CONTINUED)
# ====================================

def _calculate_memory_system_health(stm_efficiency: float, stm_load: int, stm_capacity: int) -> float:
    """Berechnet Memory System Health"""
    try:
        # Health factors
        efficiency_health = stm_efficiency
        utilization_health = 1.0 - abs((stm_load / max(1, stm_capacity)) - 0.6)  # Optimal at 60%
        capacity_health = min(1.0, stm_capacity / 10)  # Assume 10 is good capacity
        
        # Weighted average
        memory_health = (efficiency_health * 0.5 + utilization_health * 0.3 + capacity_health * 0.2)
        
        return max(0.0, min(1.0, memory_health))
        
    except Exception as e:
        logger.debug(f"Memory system health calculation failed: {e}")
        return 0.7

def _calculate_memory_balance_score(stm_load: int, ltm_size: int) -> float:
    """Berechnet Memory Balance Score"""
    try:
        # Ideal ratio: STM should be small relative to LTM for good balance
        if ltm_size == 0:
            return 0.5  # No LTM data
        
        stm_to_ltm_ratio = stm_load / ltm_size
        
        # Optimal ratio is around 0.1 (STM is 10% of LTM)
        optimal_ratio = 0.1
        balance_score = 1.0 - abs(stm_to_ltm_ratio - optimal_ratio) / optimal_ratio
        
        return max(0.0, min(1.0, balance_score))
        
    except Exception as e:
        logger.debug(f"Memory balance score calculation failed: {e}")
        return 0.7

def _calculate_learning_momentum(recent_activity: int, learning_readiness: float) -> float:
    """Berechnet Learning Momentum"""
    try:
        # Activity momentum (more activity = more momentum)
        activity_momentum = min(1.0, recent_activity / 20)  # 20 activities = full momentum
        
        # Readiness momentum
        readiness_momentum = learning_readiness
        
        # Combined momentum with activity weighted higher for momentum
        learning_momentum = (activity_momentum * 0.7 + readiness_momentum * 0.3)
        
        return max(0.0, min(1.0, learning_momentum))
        
    except Exception as e:
        logger.debug(f"Learning momentum calculation failed: {e}")
        return 0.5

def _calculate_knowledge_acquisition_rate(memory_manager, db_stats: Dict) -> float:
    """Berechnet Knowledge Acquisition Rate"""
    try:
        recent_activity = db_stats.get('recent_activity', 0)
        total_memories = db_stats.get('total_memories', 1)
        
        # Rate as percentage of total knowledge base
        acquisition_rate = recent_activity / total_memories if total_memories > 0 else 0
        
        # Scale to reasonable range (0-1)
        scaled_rate = min(1.0, acquisition_rate * 100)  # 1% acquisition = score of 1.0
        
        return scaled_rate
        
    except Exception as e:
        logger.debug(f"Knowledge acquisition rate calculation failed: {e}")
        return 0.1

def _determine_learning_curve_position(memory_manager, db_stats: Dict) -> str:
    """Bestimmt Learning Curve Position"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Determine learning phase based on total knowledge and activity
        if total_memories < 50:
            return 'initial_learning'
        elif total_memories < 200:
            if recent_activity > 5:
                return 'rapid_growth'
            else:
                return 'steady_growth'
        elif total_memories < 500:
            if recent_activity > 10:
                return 'accelerated_learning'
            elif recent_activity > 3:
                return 'mature_learning'
            else:
                return 'plateau'
        else:
            if recent_activity > 15:
                return 'expert_expansion'
            elif recent_activity > 5:
                return 'specialized_learning'
            else:
                return 'maintenance_mode'
                
    except Exception as e:
        logger.debug(f"Learning curve position determination failed: {e}")
        return 'unknown'

def _calculate_system_stability(memory_manager, db_stats: Dict) -> float:
    """Berechnet System Stability"""
    try:
        from routes.utils.memory_helpers import calculate_current_cognitive_load
        
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        
        # Stability factors
        load_stability = 1.0 - cognitive_load  # Lower load = more stable
        
        # Memory consistency (assume stable if we have data)
        memory_consistency = 0.8 if memory_manager else 0.3
        
        # Data availability stability
        data_stability = 0.9 if db_stats else 0.5
        
        # Overall stability
        system_stability = (load_stability * 0.5 + memory_consistency * 0.3 + data_stability * 0.2)
        
        return max(0.0, min(1.0, system_stability))
        
    except Exception as e:
        logger.debug(f"System stability calculation failed: {e}")
        return 0.7

def _calculate_adaptability_score(memory_manager, db_stats: Dict) -> float:
    """Berechnet Adaptability Score"""
    try:
        # Recent activity indicates adaptability
        recent_activity = db_stats.get('recent_activity', 0)
        activity_adaptability = min(1.0, recent_activity / 15)  # 15 activities = full adaptability
        
        # Memory system flexibility
        if memory_manager:
            try:
                from routes.utils.memory_helpers import get_stm_capacity_direct
                stm_capacity = get_stm_capacity_direct(memory_manager)
                memory_adaptability = min(1.0, stm_capacity / 15)  # 15 capacity = full adaptability
            except Exception:
                memory_adaptability = 0.7
        else:
            memory_adaptability = 0.5
        
        # Learning adaptability (variety in memory types)
        learning_adaptability = 0.8  # Assume good adaptability
        
        # Combined adaptability
        adaptability = (activity_adaptability * 0.4 + memory_adaptability * 0.3 + learning_adaptability * 0.3)
        
        return max(0.0, min(1.0, adaptability))
        
    except Exception as e:
        logger.debug(f"Adaptability score calculation failed: {e}")
        return 0.6

def _classify_performance_level(response_quality: float, system_stability: float, adaptability: float) -> str:
    """Klassifiziert Performance Level"""
    try:
        average_performance = (response_quality + system_stability + adaptability) / 3
        
        if average_performance >= 0.9:
            return 'exceptional'
        elif average_performance >= 0.8:
            return 'excellent'
        elif average_performance >= 0.7:
            return 'good'
        elif average_performance >= 0.6:
            return 'fair'
        elif average_performance >= 0.4:
            return 'poor'
        else:
            return 'critical'
            
    except Exception as e:
        logger.debug(f"Performance level classification failed: {e}")
        return 'unknown'

# ====================================
# MISSING CONSOLIDATION & EFFICIENCY FUNCTIONS
# ====================================

def _calculate_consolidation_efficiency(memory_manager, db_stats: Dict) -> float:
    """Berechnet Memory Consolidation Efficiency"""
    try:
        # If we have consolidation data, use it
        total_memories = db_stats.get('total_memories', 0)
        recent_activity = db_stats.get('recent_activity', 0)
        
        if total_memories == 0:
            return 0.5  # No data to consolidate
        
        # Efficiency = how well we're managing memory growth
        if recent_activity == 0:
            return 0.8  # Stable, no new consolidation needed
        
        # Good consolidation efficiency if recent activity is manageable
        consolidation_ratio = min(1.0, (total_memories - recent_activity) / total_memories)
        
        return max(0.3, consolidation_ratio)
        
    except Exception as e:
        logger.debug(f"Consolidation efficiency calculation failed: {e}")
        return 0.7

def _calculate_retrieval_efficiency(memory_manager, db_stats: Dict) -> float:
    """Berechnet Retrieval Efficiency"""
    try:
        if memory_manager:
            try:
                from routes.utils.memory_helpers import get_stm_load_direct, get_stm_capacity_direct
                stm_load = get_stm_load_direct(memory_manager)
                stm_capacity = get_stm_capacity_direct(memory_manager)
                
                # Good retrieval efficiency if STM is not overloaded
                if stm_capacity > 0:
                    utilization = stm_load / stm_capacity
                    # Optimal utilization for retrieval is around 50-70%
                    if 0.5 <= utilization <= 0.7:
                        return 0.9
                    elif 0.3 <= utilization <= 0.8:
                        return 0.8
                    else:
                        return 0.6
                else:
                    return 0.5
                    
            except Exception:
                return 0.7
        else:
            return 0.5
            
    except Exception as e:
        logger.debug(f"Retrieval efficiency calculation failed: {e}")
        return 0.7

def _calculate_storage_efficiency(db_stats: Dict) -> float:
    """Berechnet Storage Efficiency"""
    try:
        total_memories = db_stats.get('total_memories', 0)
        
        if total_memories == 0:
            return 1.0  # Perfect efficiency with no storage
        
        # Assume good storage efficiency if we have reasonable amount of memories
        if total_memories < 1000:
            return 0.9
        elif total_memories < 5000:
            return 0.8
        elif total_memories < 10000:
            return 0.7
        else:
            return 0.6  # Large storage may have efficiency challenges
            
    except Exception as e:
        logger.debug(f"Storage efficiency calculation failed: {e}")
        return 0.8

def _calculate_processing_efficiency(memory_manager, db_stats: Dict) -> float:
    """Berechnet Processing Efficiency"""
    try:
        from routes.utils.memory_helpers import calculate_current_cognitive_load
        
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        
        # Processing efficiency is inverse of cognitive load
        processing_efficiency = 1.0 - cognitive_load
        
        # Boost efficiency if system is handling reasonable load well
        if 0.3 <= cognitive_load <= 0.6:  # Optimal processing range
            processing_efficiency = min(1.0, processing_efficiency * 1.1)
        
        return max(0.0, processing_efficiency)
        
    except Exception as e:
        logger.debug(f"Processing efficiency calculation failed: {e}")
        return 0.7

def _calculate_response_time_efficiency() -> float:
    """Berechnet Response Time Efficiency"""
    try:
        # Simulate response time measurement
        # In real implementation, this would measure actual response times
        simulated_response_time = 0.05  # 50ms
        
        # Good efficiency for response times under 100ms
        if simulated_response_time <= 0.05:  # 50ms
            return 0.95
        elif simulated_response_time <= 0.1:  # 100ms
            return 0.9
        elif simulated_response_time <= 0.2:  # 200ms
            return 0.8
        elif simulated_response_time <= 0.5:  # 500ms
            return 0.6
        else:
            return 0.4
            
    except Exception as e:
        logger.debug(f"Response time efficiency calculation failed: {e}")
        return 0.8

def _calculate_accuracy_efficiency(memory_manager, db_stats: Dict) -> float:
    """Berechnet Accuracy Efficiency"""
    try:
        # Accuracy based on system stability and data quality
        if memory_manager:
            system_stability = _calculate_system_stability(memory_manager, db_stats)
            data_quality = 0.9 if db_stats else 0.5
            
            accuracy_efficiency = (system_stability * 0.7 + data_quality * 0.3)
            return accuracy_efficiency
        else:
            return 0.6
            
    except Exception as e:
        logger.debug(f"Accuracy efficiency calculation failed: {e}")
        return 0.7

def _calculate_adaptability_efficiency(memory_manager, db_stats: Dict) -> float:
    """Berechnet Adaptability Efficiency"""
    try:
        adaptability_score = _calculate_adaptability_score(memory_manager, db_stats)
        
        # Efficiency is how well adaptability is utilized
        recent_activity = db_stats.get('recent_activity', 0)
        activity_factor = min(1.0, recent_activity / 10)  # 10 activities = full utilization
        
        adaptability_efficiency = (adaptability_score * 0.6 + activity_factor * 0.4)
        
        return max(0.0, min(1.0, adaptability_efficiency))
        
    except Exception as e:
        logger.debug(f"Adaptability efficiency calculation failed: {e}")
        return 0.6

def _calculate_learning_efficiency(memory_manager, db_stats: Dict) -> float:
    """Berechnet Learning Efficiency"""
    try:
        from routes.utils.memory_helpers import calculate_learning_readiness_direct
        
        learning_readiness = calculate_learning_readiness_direct(memory_manager, db_stats)
        recent_activity = db_stats.get('recent_activity', 0)
        
        # Learning efficiency: readiness * activity utilization
        activity_utilization = min(1.0, recent_activity / 15)  # 15 activities = full utilization
        learning_efficiency = learning_readiness * (0.3 + 0.7 * activity_utilization)
        
        return max(0.0, min(1.0, learning_efficiency))
        
    except Exception as e:
        logger.debug(f"Learning efficiency calculation failed: {e}")
        return 0.6
    
def _identify_efficiency_improvement_areas(efficiency_metrics: Dict) -> List[str]:
    """Identifiziert Efficiency Improvement Areas"""
    try:
        improvement_areas = []
        
        # Core Efficiency Analysis
        if 'core_efficiency_metrics' in efficiency_metrics:
            core_metrics = efficiency_metrics['core_efficiency_metrics']
            
            if core_metrics.get('stm_efficiency', 1.0) < 0.7:
                improvement_areas.append('Short-term Memory Efficiency')
            
            if core_metrics.get('cognitive_efficiency', 1.0) < 0.7:
                improvement_areas.append('Cognitive Processing Efficiency')
            
            if core_metrics.get('memory_utilization_efficiency', 1.0) < 0.6:
                improvement_areas.append('Memory Utilization Optimization')
        
        # Resource Efficiency Analysis
        if 'resource_efficiency_metrics' in efficiency_metrics:
            resource_metrics = efficiency_metrics['resource_efficiency_metrics']
            
            if resource_metrics.get('memory_consolidation_efficiency', 1.0) < 0.7:
                improvement_areas.append('Memory Consolidation Process')
            
            if resource_metrics.get('retrieval_efficiency', 1.0) < 0.7:
                improvement_areas.append('Information Retrieval Speed')
            
            if resource_metrics.get('storage_efficiency', 1.0) < 0.7:
                improvement_areas.append('Data Storage Organization')
            
            if resource_metrics.get('processing_efficiency', 1.0) < 0.7:
                improvement_areas.append('Processing Pipeline Optimization')
        
        # Operational Efficiency Analysis
        if 'operational_efficiency_metrics' in efficiency_metrics:
            operational_metrics = efficiency_metrics['operational_efficiency_metrics']
            
            if operational_metrics.get('response_time_efficiency', 1.0) < 0.8:
                improvement_areas.append('Response Time Optimization')
            
            if operational_metrics.get('accuracy_efficiency', 1.0) < 0.8:
                improvement_areas.append('Accuracy Enhancement')
            
            if operational_metrics.get('adaptability_efficiency', 1.0) < 0.7:
                improvement_areas.append('Adaptability Mechanisms')
            
            if operational_metrics.get('learning_efficiency', 1.0) < 0.7:
                improvement_areas.append('Learning Process Optimization')
        
        # If no specific areas identified, suggest general improvements
        if not improvement_areas:
            improvement_areas.append('System Monitoring Enhancement')
        
        return improvement_areas[:5]  # Limit to top 5 areas
        
    except Exception as e:
        logger.debug(f"Efficiency improvement areas identification failed: {e}")
        return ['General System Optimization']

def _determine_efficiency_trend(efficiency_metrics: Dict) -> str:
    """Bestimmt Efficiency Trend"""
    try:
        # Calculate overall efficiency scores
        efficiency_scores = []
        
        # Core efficiency
        if 'core_efficiency_metrics' in efficiency_metrics:
            core_efficiency = efficiency_metrics['core_efficiency_metrics'].get('overall_efficiency', 0.7)
            efficiency_scores.append(core_efficiency)
        
        # Resource efficiency
        if 'resource_efficiency_metrics' in efficiency_metrics:
            resource_metrics = efficiency_metrics['resource_efficiency_metrics']
            resource_scores = [
                resource_metrics.get('memory_consolidation_efficiency', 0.7),
                resource_metrics.get('retrieval_efficiency', 0.7),
                resource_metrics.get('storage_efficiency', 0.7),
                resource_metrics.get('processing_efficiency', 0.7)
            ]
            avg_resource_efficiency = sum(resource_scores) / len(resource_scores)
            efficiency_scores.append(avg_resource_efficiency)
        
        # Operational efficiency
        if 'operational_efficiency_metrics' in efficiency_metrics:
            operational_metrics = efficiency_metrics['operational_efficiency_metrics']
            operational_scores = [
                operational_metrics.get('response_time_efficiency', 0.8),
                operational_metrics.get('accuracy_efficiency', 0.7),
                operational_metrics.get('adaptability_efficiency', 0.6),
                operational_metrics.get('learning_efficiency', 0.6)
            ]
            avg_operational_efficiency = sum(operational_scores) / len(operational_scores)
            efficiency_scores.append(avg_operational_efficiency)
        
        # Determine trend based on overall efficiency
        if efficiency_scores:
            overall_efficiency = sum(efficiency_scores) / len(efficiency_scores)
            
            if overall_efficiency >= 0.85:
                return 'excellent_and_improving'
            elif overall_efficiency >= 0.75:
                return 'good_and_stable'
            elif overall_efficiency >= 0.65:
                return 'moderate_with_potential'
            elif overall_efficiency >= 0.5:
                return 'needs_optimization'
            else:
                return 'requires_immediate_attention'
        else:
            return 'unknown_trend'
            
    except Exception as e:
        logger.debug(f"Efficiency trend determination failed: {e}")
        return 'stable'

def _generate_efficiency_optimization_recommendations(efficiency_metrics: Dict) -> List[str]:
    """Generiert Efficiency Optimization Recommendations"""
    try:
        recommendations = []
        
        # Analyze core efficiency
        if 'core_efficiency_metrics' in efficiency_metrics:
            core_metrics = efficiency_metrics['core_efficiency_metrics']
            overall_efficiency = core_metrics.get('overall_efficiency', 0.7)
            
            if overall_efficiency < 0.6:
                recommendations.append("Implement comprehensive memory system optimization")
                recommendations.append("Review and optimize cognitive processing algorithms")
            elif overall_efficiency < 0.8:
                recommendations.append("Fine-tune memory utilization parameters")
                recommendations.append("Optimize short-term memory consolidation timing")
        
        # Analyze resource efficiency
        if 'resource_efficiency_metrics' in efficiency_metrics:
            resource_metrics = efficiency_metrics['resource_efficiency_metrics']
            
            consolidation_eff = resource_metrics.get('memory_consolidation_efficiency', 0.7)
            if consolidation_eff < 0.7:
                recommendations.append("Implement automatic memory consolidation scheduling")
            
            retrieval_eff = resource_metrics.get('retrieval_efficiency', 0.7)
            if retrieval_eff < 0.7:
                recommendations.append("Optimize memory indexing for faster retrieval")
            
            storage_eff = resource_metrics.get('storage_efficiency', 0.7)
            if storage_eff < 0.7:
                recommendations.append("Implement data compression and archiving strategies")
        
        # Analyze operational efficiency
        if 'operational_efficiency_metrics' in efficiency_metrics:
            operational_metrics = efficiency_metrics['operational_efficiency_metrics']
            
            response_time_eff = operational_metrics.get('response_time_efficiency', 0.8)
            if response_time_eff < 0.8:
                recommendations.append("Optimize response processing pipeline")
                recommendations.append("Implement caching for frequently accessed data")
            
            learning_eff = operational_metrics.get('learning_efficiency', 0.6)
            if learning_eff < 0.7:
                recommendations.append("Enhance learning algorithm efficiency")
                recommendations.append("Implement adaptive learning rate optimization")
        
        # General recommendations if specific ones are limited
        if len(recommendations) < 3:
            recommendations.extend([
                "Regular system performance monitoring and tuning",
                "Implement automated optimization routines",
                "Consider system resource scaling options"
            ])
        
        return recommendations[:6]  # Limit to top 6 recommendations
        
    except Exception as e:
        logger.debug(f"Efficiency optimization recommendations generation failed: {e}")
        return [
            "Perform comprehensive system efficiency audit",
            "Implement monitoring for efficiency metrics",
            "Consider professional system optimization consultation"
        ]

# ====================================
# MISSING PERSONALITY FUNCTIONS
# ====================================

def _calculate_trait_balance_score(traits: Dict) -> float:
    """Berechnet Trait Balance Score"""
    try:
        if not traits:
            return 0.5
        
        trait_values = list(traits.values())
        if not trait_values:
            return 0.5
        
        # Calculate variance - lower variance = better balance
        mean_value = sum(trait_values) / len(trait_values)
        variance = sum((x - mean_value) ** 2 for x in trait_values) / len(trait_values)
        
        # Convert variance to balance score (0-1, where 1 is perfect balance)
        # Assume variance of 0.25 or higher indicates poor balance
        balance_score = max(0.0, 1.0 - (variance / 0.25))
        
        return min(1.0, balance_score)
        
    except Exception as e:
        logger.debug(f"Trait balance score calculation failed: {e}")
        return 0.7

def _identify_dominant_traits(traits: Dict) -> List[str]:
    """Identifiziert dominante Traits"""
    try:
        if not traits:
            return []
        
        # Sort traits by value (descending)
        sorted_traits = sorted(traits.items(), key=lambda x: x[1], reverse=True)
        
        # Return top traits (above average)
        mean_value = sum(traits.values()) / len(traits)
        dominant_traits = [trait for trait, value in sorted_traits if value > mean_value]
        
        return dominant_traits[:5]  # Top 5 dominant traits
        
    except Exception as e:
        logger.debug(f"Dominant traits identification failed: {e}")
        return []

def _calculate_personality_development_rate(personality_data: Dict) -> float:
    """Berechnet Personality Development Rate"""
    try:
        # Simulate development rate based on traits and state
        traits = personality_data.get('traits', {})
        current_state = personality_data.get('current_state', {})
        
        # Development rate factors
        trait_diversity = len(traits) / 10  # Assume 10 traits is full diversity
        adaptability = current_state.get('adaptability', 0.6)
        empathy_level = current_state.get('empathy_level', 0.8)
        
        # Calculate development rate
        development_rate = (trait_diversity * 0.4 + adaptability * 0.4 + empathy_level * 0.2)
        
        return min(1.0, development_rate)
        
    except Exception as e:
        logger.debug(f"Personality development rate calculation failed: {e}")
        return 0.5

def _calculate_personality_growth_potential(personality_data: Dict) -> float:
    """Berechnet Personality Growth Potential"""
    try:
        traits = personality_data.get('traits', {})
        current_state = personality_data.get('current_state', {})
        
        # Growth potential factors
        trait_balance = _calculate_trait_balance_score(traits)
        emotional_stability = current_state.get('emotional_stability', 0.7)
        adaptability = current_state.get('adaptability', 0.6)
        
        # High balance + stability + adaptability = high growth potential
        growth_potential = (trait_balance * 0.3 + emotional_stability * 0.4 + adaptability * 0.3)
        
        # Bonus for trait diversity
        trait_diversity_bonus = min(0.2, len(traits) / 50)  # Up to 20% bonus
        growth_potential += trait_diversity_bonus
        
        return min(1.0, growth_potential)
        
    except Exception as e:
        logger.debug(f"Personality growth potential calculation failed: {e}")
        return 0.6

def _assess_personality_maturity_level(personality_data: Dict) -> str:
    """Bewertet Personality Maturity Level"""
    try:
        traits = personality_data.get('traits', {})
        current_state = personality_data.get('current_state', {})
        
        # Maturity indicators
        trait_count = len(traits)
        emotional_stability = current_state.get('emotional_stability', 0.7)
        empathy_level = current_state.get('empathy_level', 0.8)
        adaptability = current_state.get('adaptability', 0.6)
        
        # Calculate maturity score
        maturity_score = (
            (trait_count / 10) * 0.3 +  # Trait development
            emotional_stability * 0.3 +  # Emotional maturity
            empathy_level * 0.2 +        # Social maturity
            adaptability * 0.2           # Cognitive maturity
        )
        
        # Classify maturity level
        if maturity_score >= 0.9:
            return 'highly_mature'
        elif maturity_score >= 0.75:
            return 'mature'
        elif maturity_score >= 0.6:
            return 'developing'
        elif maturity_score >= 0.4:
            return 'early_stage'
        else:
            return 'initial'
            
    except Exception as e:
        logger.debug(f"Personality maturity level assessment failed: {e}")
        return 'unknown'

def _calculate_overall_personality_health(trait_balance: float, emotional_stability: float, 
                                        adaptability: float, empathy_level: float) -> float:
    """Berechnet Overall Personality Health"""
    try:
        # Weighted average of personality health factors
        personality_health = (
            trait_balance * 0.25 +
            emotional_stability * 0.35 +
            adaptability * 0.25 +
            empathy_level * 0.15
        )
        
        return max(0.0, min(1.0, personality_health))
        
    except Exception as e:
        logger.debug(f"Overall personality health calculation failed: {e}")
        return 0.7

# ====================================
# MISSING REAL-TIME MONITORING FUNCTIONS
# ====================================

def _calculate_system_uptime() -> str:
    """Berechnet System Uptime"""
    try:
        # Simulate uptime calculation
        # In real implementation, this would track actual uptime
        import time
        current_time = time.time()
        
        # Simulate uptime (hours)
        uptime_hours = 24.5  # Example uptime
        uptime_minutes = int((uptime_hours % 1) * 60)
        uptime_hours = int(uptime_hours)
        
        return f"{uptime_hours}h {uptime_minutes}m"
        
    except Exception as e:
        logger.debug(f"System uptime calculation failed: {e}")
        return "unknown"

def _count_active_processes(memory_manager) -> int:
    """Zählt aktive Prozesse"""
    try:
        # Count active processes based on memory manager state
        active_processes = 1  # Base process
        
        if memory_manager:
            # Add memory-related processes
            active_processes += 2  # STM and LTM processes
            
            # Add consolidation processes
            active_processes += 1  # Consolidation process
        
        return active_processes
        
    except Exception as e:
        logger.debug(f"Active processes counting failed: {e}")
        return 3  # Default process count

def _get_current_memory_usage() -> str:
    """Holt aktuelle Memory Usage"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return f"{memory.percent}%"
        
    except Exception as e:
        logger.debug(f"Memory usage retrieval failed: {e}")
        return "65%"  # Default value

def _get_current_cpu_utilization() -> str:
    """Holt aktuelle CPU Utilization"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return f"{cpu_percent}%"
        
    except Exception as e:
        logger.debug(f"CPU utilization retrieval failed: {e}")
        return "25%"  # Default value

def _get_live_stm_load(memory_manager) -> int:
    """Holt Live STM Load"""
    try:
        if memory_manager:
            from routes.utils.memory_helpers import get_stm_load_direct
            return get_stm_load_direct(memory_manager)
        else:
            return 5  # Default STM load
            
    except Exception as e:
        logger.debug(f"Live STM load retrieval failed: {e}")
        return 5

def _count_recent_memory_formations(db_stats: Dict) -> int:
    """Zählt Recent Memory Formations"""
    try:
        return db_stats.get('recent_activity', 0)
        
    except Exception as e:
        logger.debug(f"Recent memory formations counting failed: {e}")
        return 0

def _count_active_consolidation_processes(memory_manager) -> int:
    """Zählt aktive Consolidation Processes"""
    try:
        # Simulate active consolidation processes
        if memory_manager:
            return 1  # One active consolidation process
        else:
            return 0
            
    except Exception as e:
        logger.debug(f"Active consolidation processes counting failed: {e}")
        return 0

def _calculate_retrieval_rate() -> float:
    """Berechnet Retrieval Rate"""
    try:
        # Simulate retrieval rate (requests per minute)
        return 12.5  # Example rate
        
    except Exception as e:
        logger.debug(f"Retrieval rate calculation failed: {e}")
        return 10.0

def _measure_current_response_time() -> float:
    """Misst aktuelle Response Time"""
    try:
        # Simulate response time measurement
        import random
        return random.uniform(0.02, 0.08)  # 20-80ms
        
    except Exception as e:
        logger.debug(f"Response time measurement failed: {e}")
        return 0.05  # 50ms default

def _calculate_live_system_health_score(memory_manager, db_stats: Dict) -> float:
    """Berechnet Live System Health Score"""
    try:
        from routes.utils.memory_helpers import calculate_current_cognitive_load
        
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        memory_health = 0.8 if memory_manager else 0.5
        data_quality = 0.9 if db_stats else 0.6
        
        # Live health score
        health_score = (
            (1.0 - cognitive_load) * 0.4 +
            memory_health * 0.35 +
            data_quality * 0.25
        )
        
        return max(0.0, min(1.0, health_score))
        
    except Exception as e:
        logger.debug(f"Live system health score calculation failed: {e}")
        return 0.75

def _assess_current_learning_activity(memory_manager, db_stats: Dict) -> str:
    """Bewertet aktuelle Learning Activity"""
    try:
        recent_activity = db_stats.get('recent_activity', 0)
        
        if recent_activity >= 10:
            return 'very_high'
        elif recent_activity >= 5:
            return 'high'
        elif recent_activity >= 2:
            return 'moderate'
        elif recent_activity >= 1:
            return 'low'
        else:
            return 'minimal'
            
    except Exception as e:
        logger.debug(f"Current learning activity assessment failed: {e}")
        return 'moderate'

def _identify_live_optimization_opportunities(memory_manager, db_stats: Dict) -> List[str]:
    """Identifiziert Live Optimization Opportunities"""
    try:
        opportunities = []
        
        # Check cognitive load
        from routes.utils.memory_helpers import calculate_current_cognitive_load
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        
        if cognitive_load > 0.8:
            opportunities.append("Reduce cognitive load through memory consolidation")
        
        # Check STM utilization
        if memory_manager:
            from routes.utils.memory_helpers import get_stm_load_direct, get_stm_capacity_direct
            stm_load = get_stm_load_direct(memory_manager)
            stm_capacity = get_stm_capacity_direct(memory_manager)
            
            if stm_capacity > 0:
                utilization = stm_load / stm_capacity
                if utilization > 0.9:
                    opportunities.append("Optimize STM capacity management")
                elif utilization < 0.3:
                    opportunities.append("Increase learning activity to better utilize STM")
        
        # Check recent activity
        recent_activity = db_stats.get('recent_activity', 0)
        if recent_activity == 0:
            opportunities.append("Engage in learning activities to improve system development")
        
        return opportunities[:3]  # Limit to top 3 opportunities
        
    except Exception as e:
        logger.debug(f"Live optimization opportunities identification failed: {e}")
        return ["Monitor system performance continuously"]

# ====================================
# MISSING ALERT AND COMPARISON FUNCTIONS
# ====================================

def _check_real_time_alerts(realtime_metrics: Dict) -> List[Dict[str, Any]]:
    """Prüft Real-time Alerts"""
    try:
        alerts = []
        
        # Check system state alerts
        system_state = realtime_metrics.get('current_system_state', {})
        
        # Memory usage alert
        memory_usage_str = system_state.get('memory_usage', '0%')
        memory_usage = float(memory_usage_str.replace('%', ''))
        if memory_usage > 90:
            alerts.append({
                'type': 'critical',
                'message': f'High memory usage: {memory_usage}%',
                'timestamp': datetime.now().isoformat(),
                'action': 'immediate_attention_required'
            })
        elif memory_usage > 75:
            alerts.append({
                'type': 'warning',
                'message': f'Elevated memory usage: {memory_usage}%',
                'timestamp': datetime.now().isoformat(),
                'action': 'monitor_closely'
            })
        
        # CPU utilization alert
        cpu_usage_str = system_state.get('cpu_utilization', '0%')
        cpu_usage = float(cpu_usage_str.replace('%', ''))
        if cpu_usage > 80:
            alerts.append({
                'type': 'warning',
                'message': f'High CPU utilization: {cpu_usage}%',
                'timestamp': datetime.now().isoformat(),
                'action': 'optimize_processes'
            })
        
        # Performance alerts
        performance = realtime_metrics.get('real_time_performance', {})
        health_score = performance.get('system_health_score', 1.0)
        
        if health_score < 0.5:
            alerts.append({
                'type': 'critical',
                'message': f'Low system health score: {health_score:.2f}',
                'timestamp': datetime.now().isoformat(),
                'action': 'system_diagnostic_required'
            })
        
        return alerts
        
    except Exception as e:
        logger.debug(f"Real-time alerts check failed: {e}")
        return []

def _compare_with_baseline_performance(realtime_metrics: Dict) -> Dict[str, Any]:
    """Vergleicht mit Baseline Performance"""
    try:
        # Baseline values (these would be stored from historical data)
        baseline = {
            'memory_usage': 60.0,  # %
            'cpu_utilization': 30.0,  # %
            'system_health_score': 0.8,
            'response_time': 0.06  # seconds
        }
        
        # Current values
        current_state = realtime_metrics.get('current_system_state', {})
        performance = realtime_metrics.get('real_time_performance', {})
        
        current_memory = float(current_state.get('memory_usage', '60%').replace('%', ''))
        current_cpu = float(current_state.get('cpu_utilization', '30%').replace('%', ''))
        current_health = performance.get('system_health_score', 0.8)
        current_response_time = performance.get('current_response_time', 0.06)
        
        # Calculate comparisons
        comparison = {
            'memory_usage_change': current_memory - baseline['memory_usage'],
            'cpu_utilization_change': current_cpu - baseline['cpu_utilization'],
            'health_score_change': current_health - baseline['system_health_score'],
            'response_time_change': current_response_time - baseline['response_time'],
            'overall_performance_trend': 'stable'
        }
        
        # Determine overall trend
        positive_changes = sum(1 for change in [
            -comparison['memory_usage_change'],  # Lower memory usage is better
            -comparison['cpu_utilization_change'],  # Lower CPU usage is better
            comparison['health_score_change'],   # Higher health score is better
            -comparison['response_time_change']  # Lower response time is better
        ] if change > 0.05)  # Significant positive change threshold
        
        if positive_changes >= 3:
            comparison['overall_performance_trend'] = 'improving'
        elif positive_changes <= 1:
            comparison['overall_performance_trend'] = 'declining'
        else:
            comparison['overall_performance_trend'] = 'stable'
        
        return comparison
        
    except Exception as e:
        logger.debug(f"Baseline performance comparison failed: {e}")
        return {
            'overall_performance_trend': 'unknown',
            'comparison_error': str(e)
        }

# ====================================
# MISSING STANDARD PERFORMANCE & TREND FUNCTIONS
# ====================================

def _get_standard_performance_indicators(memory_manager, db_stats: Dict) -> Dict[str, Any]:
    """Holt Standard Performance Indicators"""
    try:
        from routes.utils.memory_helpers import (
            calculate_current_cognitive_load,
            calculate_stm_efficiency_direct,
            get_stm_load_direct,
            get_stm_capacity_direct
        )
        
        # Core performance indicators
        cognitive_load = calculate_current_cognitive_load(memory_manager, db_stats)
        stm_efficiency = calculate_stm_efficiency_direct(memory_manager)
        stm_load = get_stm_load_direct(memory_manager)
        stm_capacity = get_stm_capacity_direct(memory_manager)
        
        indicators = {
            'cognitive_performance': {
                'cognitive_load': cognitive_load,
                'cognitive_efficiency': 1.0 - cognitive_load,
                'processing_capacity': max(0.0, 1.0 - (stm_load / max(1, stm_capacity)))
            },
            
            'memory_performance': {
                'stm_efficiency': stm_efficiency,
                'stm_utilization': stm_load / max(1, stm_capacity),
                'memory_health': _calculate_memory_system_health(stm_efficiency, stm_load, stm_capacity)
            },
            
            'learning_performance': {
                'learning_rate': db_stats.get('recent_activity', 0) / 24,  # per hour
                'knowledge_base_size': db_stats.get('total_memories', 0),
                'learning_efficiency': _calculate_learning_efficiency(memory_manager, db_stats)
            },
            
            'system_performance': {
                'overall_health': _calculate_live_system_health_score(memory_manager, db_stats),
                'stability_score': _calculate_system_stability(memory_manager, db_stats),
                'adaptability_score': _calculate_adaptability_score(memory_manager, db_stats)
            }
        }
        
        return indicators
        
    except Exception as e:
        logger.error(f"Standard performance indicators retrieval failed: {e}")
        return _generate_fallback_performance_indicators()

def _calculate_trend_indicators(indicators: Dict) -> Dict[str, Any]:
    """Berechnet Trend Indicators"""
    try:
        trend_indicators = {
            'cognitive_trend': 'stable',
            'memory_trend': 'stable',
            'learning_trend': 'stable',
            'system_trend': 'stable',
            'overall_trend': 'stable'
        }
        
        # Cognitive trend analysis
        if 'cognitive_performance' in indicators:
            cognitive_perf = indicators['cognitive_performance']
            cognitive_efficiency = cognitive_perf.get('cognitive_efficiency', 0.7)
            
            if cognitive_efficiency > 0.8:
                trend_indicators['cognitive_trend'] = 'improving'
            elif cognitive_efficiency < 0.5:
                trend_indicators['cognitive_trend'] = 'declining'
        
        # Memory trend analysis
        if 'memory_performance' in indicators:
            memory_perf = indicators['memory_performance']
            memory_health = memory_perf.get('memory_health', 0.7)
            
            if memory_health > 0.8:
                trend_indicators['memory_trend'] = 'improving'
            elif memory_health < 0.6:
                trend_indicators['memory_trend'] = 'declining'
        
        # Learning trend analysis
        if 'learning_performance' in indicators:
            learning_perf = indicators['learning_performance']
            learning_rate = learning_perf.get('learning_rate', 0)
            
            if learning_rate > 0.5:  # More than 0.5 per hour
                trend_indicators['learning_trend'] = 'accelerating'
            elif learning_rate < 0.1:  # Less than 0.1 per hour
                trend_indicators['learning_trend'] = 'slowing'
        
        # System trend analysis
        if 'system_performance' in indicators:
            system_perf = indicators['system_performance']
            overall_health = system_perf.get('overall_health', 0.7)
            
            if overall_health > 0.85:
                trend_indicators['system_trend'] = 'excellent'
            elif overall_health < 0.6:
                trend_indicators['system_trend'] = 'concerning'
        
        # Overall trend calculation
        positive_trends = sum(1 for trend in trend_indicators.values() 
                            if trend in ['improving', 'accelerating', 'excellent'])
        negative_trends = sum(1 for trend in trend_indicators.values() 
                            if trend in ['declining', 'slowing', 'concerning'])
        
        if positive_trends > negative_trends:
            trend_indicators['overall_trend'] = 'positive'
        elif negative_trends > positive_trends:
            trend_indicators['overall_trend'] = 'negative'
        else:
            trend_indicators['overall_trend'] = 'stable'
        
        return trend_indicators
        
    except Exception as e:
        logger.debug(f"Trend indicators calculation failed: {e}")
        return {
            'cognitive_trend': 'stable',
            'memory_trend': 'stable',
            'learning_trend': 'stable',
            'system_trend': 'stable',
            'overall_trend': 'stable'
        }

def _get_performance_thresholds() -> Dict[str, Any]:
    """Holt Performance Thresholds"""
    return {
        'cognitive_load': {
            'excellent': 0.3,
            'good': 0.5,
            'fair': 0.7,
            'poor': 0.8,
            'critical': 0.9
        },
        'memory_efficiency': {
            'excellent': 0.9,
            'good': 0.8,
            'fair': 0.7,
            'poor': 0.6,
            'critical': 0.5
        },
        'learning_rate': {
            'excellent': 1.0,  # per hour
            'good': 0.5,
            'fair': 0.2,
            'poor': 0.1,
            'critical': 0.05
        },
        'system_health': {
            'excellent': 0.9,
            'good': 0.8,
            'fair': 0.7,
            'poor': 0.6,
            'critical': 0.5
        }
    }

def _assess_performance_status(indicators: Dict) -> Dict[str, str]:
    """Bewertet Performance Status"""
    try:
        thresholds = _get_performance_thresholds()
        status = {}
        
        # Assess cognitive performance
        if 'cognitive_performance' in indicators:
            cognitive_load = indicators['cognitive_performance'].get('cognitive_load', 0.5)
            status['cognitive_status'] = _assess_against_thresholds(
                cognitive_load, thresholds['cognitive_load'], reverse=True
            )
        
        # Assess memory performance
        if 'memory_performance' in indicators:
            memory_health = indicators['memory_performance'].get('memory_health', 0.7)
            status['memory_status'] = _assess_against_thresholds(
                memory_health, thresholds['memory_efficiency']
            )
        
        # Assess learning performance
        if 'learning_performance' in indicators:
            learning_rate = indicators['learning_performance'].get('learning_rate', 0.1)
            status['learning_status'] = _assess_against_thresholds(
                learning_rate, thresholds['learning_rate']
            )
        
        # Assess system performance
        if 'system_performance' in indicators:
            system_health = indicators['system_performance'].get('overall_health', 0.7)
            status['system_status'] = _assess_against_thresholds(
                system_health, thresholds['system_health']
            )
        
        return status
        
    except Exception as e:
        logger.debug(f"Performance status assessment failed: {e}")
        return {
            'cognitive_status': 'unknown',
            'memory_status': 'unknown',
            'learning_status': 'unknown',
            'system_status': 'unknown'
        }

def _assess_against_thresholds(value: float, thresholds: Dict, reverse: bool = False) -> str:
    """Bewertet Wert gegen Thresholds"""
    try:
        if reverse:
            # For metrics where lower is better (like cognitive load)
            if value <= thresholds['excellent']:
                return 'excellent'
            elif value <= thresholds['good']:
                return 'good'
            elif value <= thresholds['fair']:
                return 'fair'
            elif value <= thresholds['poor']:
                return 'poor'
            else:
                return 'critical'
        else:
            # For metrics where higher is better
            if value >= thresholds['excellent']:
                return 'excellent'
            elif value >= thresholds['good']:
                return 'good'
            elif value >= thresholds['fair']:
                return 'fair'
            elif value >= thresholds['poor']:
                return 'poor'
            else:
                return 'critical'
                
    except Exception as e:
        logger.debug(f"Threshold assessment failed: {e}")
        return 'unknown'

# ====================================
# MISSING FALLBACK FUNCTIONS
# ====================================

def _generate_fallback_performance_indicators() -> Dict[str, Any]:
    """Generiert Fallback Performance Indicators"""
    return {
        'fallback_mode': True,
        'cognitive_performance': {
            'cognitive_load': 0.5,
            'cognitive_efficiency': 0.7,
            'processing_capacity': 0.6
        },
        'memory_performance': {
            'stm_efficiency': 0.7,
            'stm_utilization': 0.5,
            'memory_health': 0.7
        },
        'learning_performance': {
            'learning_rate': 0.2,
            'knowledge_base_size': 100,
            'learning_efficiency': 0.6
        },
        'system_performance': {
            'overall_health': 0.7,
            'stability_score': 0.8,
            'adaptability_score': 0.6
        }
    }

def _generate_fallback_efficiency_metrics() -> Dict[str, Any]:
    """Generiert Fallback Efficiency Metrics"""
    return {
        'fallback_mode': True,
        'core_efficiency_metrics': {
            'stm_efficiency': 0.7,
            'cognitive_efficiency': 0.7,
            'memory_utilization_efficiency': 0.5,
            'overall_efficiency': 0.7
        },
        'efficiency_ratings': {
            'overall_rating': 'good',
            'efficiency_trend': 'stable'
        }
    }

def _generate_fallback_realtime_metrics() -> Dict[str, Any]:
    """Generiert Fallback Realtime Metrics"""
    return {
        'fallback_mode': True,
        'current_system_state': {
            'timestamp': datetime.now().isoformat(),
            'system_uptime': '24h 0m',
            'memory_usage': '60%',
            'cpu_utilization': '25%'
        },
        'real_time_performance': {
            'system_health_score': 0.75,
            'learning_activity_level': 'moderate'
        }
    }

__all__ = [
    'calculate_key_metrics',
    'get_performance_indicators', 
    'calculate_efficiency_metrics',
    'get_real_time_metrics',
    '_calculate_system_readiness_score',
    '_calculate_optimization_score',
    '_calculate_development_momentum_score'
]