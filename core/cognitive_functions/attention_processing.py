"""
Attention Processing Engine
Verwaltet Aufmerksamkeit, Focus, und kognitive Ressourcen-Allokation
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AttentionManager:
    """Verwaltet Aufmerksamkeits-Allokation und Focus"""
    
    def __init__(self):
        self.current_focus_targets = []
        self.attention_history = deque(maxlen=100)
        self.cognitive_load_threshold = 0.8
        self.attention_span = 300  # 5 minutes default
        
    def allocate_attention(self, targets: List[Dict], context: Dict = None) -> Dict[str, Any]:
        """Allokiert Aufmerksamkeit auf verschiedene Ziele"""
        try:
            if not targets:
                return {'allocated_attention': {}, 'total_allocation': 0.0}
            
            # Berechne Prioritäten
            prioritized_targets = self._calculate_priorities(targets, context)
            
            # Allokiere Aufmerksamkeit
            attention_allocation = self._distribute_attention(prioritized_targets)
            
            # Update current focus
            self.current_focus_targets = attention_allocation['focus_targets']
            
            # Log attention change
            self.attention_history.append({
                'timestamp': datetime.now(),
                'allocation': attention_allocation,
                'context': context or {}
            })
            
            return attention_allocation
            
        except Exception as e:
            logger.error(f"Attention allocation failed: {e}")
            return {'allocated_attention': {}, 'total_allocation': 0.0, 'error': str(e)}
    
    def _calculate_priorities(self, targets: List[Dict], context: Dict) -> List[Dict]:
        """Berechnet Prioritäten für Attention Targets"""
        try:
            prioritized = []
            
            for target in targets:
                priority_score = 0.0
                
                # Base priority
                priority_score += target.get('base_priority', 0.5)
                
                # Urgency factor
                urgency = target.get('urgency', 0.5)
                priority_score += urgency * 0.3
                
                # Importance factor
                importance = target.get('importance', 0.5)
                priority_score += importance * 0.4
                
                # Context relevance
                if context:
                    current_task = context.get('current_task', '')
                    target_task = target.get('task_type', '')
                    if current_task and target_task and current_task == target_task:
                        priority_score += 0.2  # Boost for task relevance
                
                # Novelty factor
                novelty = target.get('novelty', 0.5)
                priority_score += novelty * 0.1
                
                target_with_priority = target.copy()
                target_with_priority['calculated_priority'] = min(1.0, priority_score)
                prioritized.append(target_with_priority)
            
            return sorted(prioritized, key=lambda x: x['calculated_priority'], reverse=True)
            
        except Exception as e:
            logger.debug(f"Priority calculation failed: {e}")
            return targets
    
    def _distribute_attention(self, prioritized_targets: List[Dict]) -> Dict[str, Any]:
        """Verteilt verfügbare Aufmerksamkeit auf Targets"""
        try:
            total_attention_budget = 1.0
            allocated_attention = {}
            focus_targets = []
            
            # Verwende eine einfache Prioritäts-basierte Verteilung
            remaining_budget = total_attention_budget
            
            for i, target in enumerate(prioritized_targets):
                if remaining_budget <= 0:
                    break
                
                target_id = target.get('id', f'target_{i}')
                priority = target.get('calculated_priority', 0.5)
                
                # Allokiere Aufmerksamkeit basierend auf Priorität
                if i == 0:  # Höchste Priorität bekommt den größten Anteil
                    allocation = min(remaining_budget, priority * 0.6)
                elif i == 1:  # Zweithöchste Priorität
                    allocation = min(remaining_budget, priority * 0.3)
                else:  # Restliche Targets teilen sich den Rest
                    allocation = min(remaining_budget / max(len(prioritized_targets) - 2, 1), priority * 0.2)
                
                if allocation > 0.05:  # Minimum threshold für Focus
                    allocated_attention[target_id] = {
                        'allocation_percentage': allocation,
                        'priority_score': priority,
                        'target_info': target
                    }
                    focus_targets.append(target)
                    remaining_budget -= allocation
            
            return {
                'allocated_attention': allocated_attention,
                'focus_targets': focus_targets,
                'total_allocation': total_attention_budget - remaining_budget,
                'remaining_capacity': remaining_budget,
                'allocation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.debug(f"Attention distribution failed: {e}")
            return {'allocated_attention': {}, 'focus_targets': [], 'total_allocation': 0.0}

def process_attention_requests(attention_requests: List[Dict], 
                             current_context: Dict = None,
                             cognitive_load: float = 0.5) -> Dict[str, Any]:
    """Verarbeitet Attention Requests - HAUPTINTERFACE"""
    try:
        attention_manager = AttentionManager()
        
        if not attention_requests:
            return {
                'attention_allocation': {},
                'processing_result': 'no_requests',
                'recommendations': ['No attention requests to process']
            }
        
        # Allokiere Aufmerksamkeit
        allocation_result = attention_manager.allocate_attention(attention_requests, current_context)
        
        # Analyse cognitive load impact
        load_analysis = _analyze_cognitive_load_impact(allocation_result, cognitive_load)
        
        # Generiere Empfehlungen
        recommendations = _generate_attention_recommendations(allocation_result, load_analysis)
        
        return {
            'attention_allocation': allocation_result,
            'cognitive_load_analysis': load_analysis,
            'processing_result': 'success',
            'recommendations': recommendations,
            'processing_metadata': {
                'requests_processed': len(attention_requests),
                'processing_timestamp': datetime.now().isoformat(),
                'context_considered': bool(current_context)
            }
        }
        
    except Exception as e:
        logger.error(f"Attention request processing failed: {e}")
        return {
            'attention_allocation': {},
            'processing_result': 'error',
            'error': str(e),
            'recommendations': ['Error processing attention requests']
        }

def calculate_attention_metrics(memory_manager=None, 
                              time_window: timedelta = None) -> Dict[str, Any]:
    """Berechnet Attention Metrics"""
    try:
        if time_window is None:
            time_window = timedelta(hours=1)
        
        attention_metrics = {
            'focus_stability': _calculate_focus_stability(),
            'attention_span_estimate': _estimate_attention_span(),
            'distraction_resistance': _calculate_distraction_resistance(),
            'task_switching_frequency': _calculate_task_switching_frequency(),
            'cognitive_load_distribution': _analyze_cognitive_load_distribution()
        }
        
        # Overall attention health
        metric_scores = [v for v in attention_metrics.values() if isinstance(v, (int, float))]
        overall_attention_health = statistics.mean(metric_scores) if metric_scores else 0.5
        
        return {
            'attention_metrics': attention_metrics,
            'overall_attention_health': overall_attention_health,
            'attention_quality': _rate_attention_quality(overall_attention_health),
            'analysis_metadata': {
                'time_window_hours': time_window.total_seconds() / 3600,
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Attention metrics calculation failed: {e}")
        return {'error': str(e), 'attention_metrics': {}}

def manage_cognitive_load(current_tasks: List[Dict], 
                        cognitive_capacity: float = 1.0) -> Dict[str, Any]:
    """Verwaltet Cognitive Load und Task Prioritization"""
    try:
        if not current_tasks:
            return {
                'load_management': 'no_tasks',
                'cognitive_load': 0.0,
                'recommendations': ['No tasks to manage']
            }
        
        # Berechne aktuelle cognitive load
        total_load = sum(task.get('cognitive_weight', 0.5) for task in current_tasks)
        current_load_ratio = total_load / cognitive_capacity
        
        # Load management strategy
        if current_load_ratio > 1.0:
            # Overload - need to prioritize/defer
            management_strategy = _handle_cognitive_overload(current_tasks, cognitive_capacity)
        elif current_load_ratio > 0.8:
            # High load - optimize
            management_strategy = _optimize_high_cognitive_load(current_tasks)
        else:
            # Normal load - maintain
            management_strategy = _maintain_normal_cognitive_load(current_tasks)
        
        return {
            'load_management': management_strategy,
            'cognitive_load': current_load_ratio,
            'cognitive_capacity': cognitive_capacity,
            'load_status': _classify_load_status(current_load_ratio),
            'task_recommendations': _generate_task_recommendations(current_tasks, current_load_ratio),
            'management_metadata': {
                'total_tasks': len(current_tasks),
                'management_timestamp': datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Cognitive load management failed: {e}")
        return {'error': str(e), 'load_management': 'error'}

# Helper Functions
def _analyze_cognitive_load_impact(allocation_result: Dict, cognitive_load: float) -> Dict[str, Any]:
    """Analysiert Impact von Attention Allocation auf Cognitive Load"""
    try:
        total_allocation = allocation_result.get('total_allocation', 0.0)
        
        # Simple impact analysis
        load_increase = total_allocation * 0.3  # Attention allocation increases load
        projected_load = cognitive_load + load_increase
        
        return {
            'current_cognitive_load': cognitive_load,
            'projected_cognitive_load': min(1.0, projected_load),
            'load_increase': load_increase,
            'overload_risk': 'high' if projected_load > 0.9 else 'moderate' if projected_load > 0.7 else 'low'
        }
        
    except Exception as e:
        logger.debug(f"Cognitive load impact analysis failed: {e}")
        return {'current_cognitive_load': cognitive_load, 'projected_cognitive_load': cognitive_load}

def _generate_attention_recommendations(allocation_result: Dict, load_analysis: Dict) -> List[str]:
    """Generiert Attention Management Recommendations"""
    recommendations = []
    
    try:
        remaining_capacity = allocation_result.get('remaining_capacity', 0.0)
        overload_risk = load_analysis.get('overload_risk', 'low')
        
        if remaining_capacity > 0.3:
            recommendations.append("Good attention capacity available - can take on additional tasks")
        elif remaining_capacity < 0.1:
            recommendations.append("Attention capacity nearly exhausted - consider task prioritization")
        
        if overload_risk == 'high':
            recommendations.append("High cognitive overload risk - reduce concurrent attention targets")
        elif overload_risk == 'moderate':
            recommendations.append("Moderate load - monitor attention allocation carefully")
        
        focus_targets = allocation_result.get('focus_targets', [])
        if len(focus_targets) > 3:
            recommendations.append("Many focus targets - consider consolidating related tasks")
        
        return recommendations if recommendations else ["Attention allocation appears optimal"]
        
    except Exception as e:
        logger.debug(f"Attention recommendations generation failed: {e}")
        return ["Unable to generate specific recommendations"]

def _calculate_focus_stability() -> float:
    """Berechnet Focus Stability - VEREINFACHT"""
    # Simplified focus stability calculation
    return 0.75  # Moderate stability as baseline

def _estimate_attention_span() -> int:
    """Schätzt aktuelle Attention Span in Sekunden"""
    # Simplified attention span estimation
    return 240  # 4 minutes baseline

def _calculate_distraction_resistance() -> float:
    """Berechnet Distraction Resistance"""
    # Simplified distraction resistance
    return 0.6  # Moderate resistance

def _calculate_task_switching_frequency() -> float:
    """Berechnet Task Switching Frequency"""
    # Simplified task switching frequency (switches per hour)
    return 8.0  # 8 switches per hour baseline

def _analyze_cognitive_load_distribution() -> Dict[str, float]:
    """Analysiert Cognitive Load Distribution"""
    return {
        'working_memory': 0.4,
        'attention_control': 0.3,
        'processing_speed': 0.2,
        'executive_function': 0.1
    }

def _rate_attention_quality(health_score: float) -> str:
    """Bewertet Attention Quality"""
    if health_score > 0.8:
        return 'excellent'
    elif health_score > 0.6:
        return 'good'
    elif health_score > 0.4:
        return 'moderate'
    else:
        return 'needs_improvement'

def _handle_cognitive_overload(tasks: List[Dict], capacity: float) -> Dict[str, Any]:
    """Handles Cognitive Overload Situation"""
    # Sort tasks by priority and defer low-priority ones
    sorted_tasks = sorted(tasks, key=lambda x: x.get('priority', 0.5), reverse=True)
    
    active_tasks = []
    deferred_tasks = []
    current_load = 0.0
    
    for task in sorted_tasks:
        task_weight = task.get('cognitive_weight', 0.5)
        if current_load + task_weight <= capacity:
            active_tasks.append(task)
            current_load += task_weight
        else:
            deferred_tasks.append(task)
    
    return {
        'strategy': 'prioritize_and_defer',
        'active_tasks': active_tasks,
        'deferred_tasks': deferred_tasks,
        'load_after_management': current_load / capacity
    }

def _optimize_high_cognitive_load(tasks: List[Dict]) -> Dict[str, Any]:
    """Optimiert High Cognitive Load"""
    return {
        'strategy': 'optimize_and_batch',
        'recommendation': 'Group similar tasks and use time-boxing',
        'suggested_batch_size': min(3, len(tasks)),
        'break_frequency': 'every_45_minutes'
    }

def _maintain_normal_cognitive_load(tasks: List[Dict]) -> Dict[str, Any]:
    """Maintains Normal Cognitive Load"""
    return {
        'strategy': 'maintain_current_approach',
        'recommendation': 'Current load is manageable - continue with current task distribution',
        'capacity_utilization': 'optimal'
    }

def _classify_load_status(load_ratio: float) -> str:
    """Klassifiziert Load Status"""
    if load_ratio > 1.0:
        return 'overloaded'
    elif load_ratio > 0.8:
        return 'high_load'
    elif load_ratio > 0.5:
        return 'moderate_load'
    else:
        return 'low_load'

def _generate_task_recommendations(tasks: List[Dict], load_ratio: float) -> List[str]:
    """Generiert Task-spezifische Recommendations"""
    recommendations = []
    
    if load_ratio > 1.0:
        recommendations.append("Prioritize critical tasks and defer non-essential ones")
        recommendations.append("Consider breaking complex tasks into smaller chunks")
    elif load_ratio > 0.8:
        recommendations.append("Monitor cognitive fatigue and schedule regular breaks")
        recommendations.append("Avoid taking on additional complex tasks")
    else:
        recommendations.append("Current task load is manageable")
        recommendations.append("Capacity available for additional tasks if needed")
    
    return recommendations

__all__ = [
    'AttentionManager',
    'process_attention_requests',
    'calculate_attention_metrics',
    'manage_cognitive_load'
]