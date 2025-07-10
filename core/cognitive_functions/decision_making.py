"""
Decision Making Engine
Intelligente Entscheidungsfindung, Option Evaluation, und Choice Selection
"""

import logging
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import random

logger = logging.getLogger(__name__)

class DecisionEngine:
    """Core Decision Making Engine"""
    
    def __init__(self):
        self.decision_history = []
        self.decision_criteria_weights = {
            'utility': 0.3,
            'risk': 0.2,
            'feasibility': 0.2,
            'alignment': 0.15,
            'resources': 0.15
        }
        self.confidence_threshold = 0.7
    
    def make_decision(self, options: List[Dict], context: Dict = None, 
                     criteria: Dict = None) -> Dict[str, Any]:
        """Macht eine Entscheidung zwischen gegebenen Optionen"""
        try:
            if not options:
                return {
                    'decision': None,
                    'confidence': 0.0,
                    'reason': 'no_options_provided'
                }
            
            # Use custom criteria if provided
            if criteria:
                self.decision_criteria_weights.update(criteria)
            
            # Evaluate all options
            evaluated_options = self._evaluate_options(options, context)
            
            # Select best option
            best_option = self._select_best_option(evaluated_options)
            
            # Calculate decision confidence
            confidence = self._calculate_decision_confidence(evaluated_options, best_option)
            
            # Create decision record
            decision_record = {
                'decision': best_option,
                'confidence': confidence,
                'all_evaluations': evaluated_options,
                'decision_timestamp': datetime.now().isoformat(),
                'context': context or {},
                'criteria_used': self.decision_criteria_weights.copy()
            }
            
            # Store in history
            self.decision_history.append(decision_record)
            
            return decision_record
            
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            return {
                'decision': None,
                'confidence': 0.0,
                'error': str(e),
                'reason': 'decision_process_error'
            }
    
    def _evaluate_options(self, options: List[Dict], context: Dict) -> List[Dict]:
        """Evaluiert alle verfügbaren Optionen"""
        evaluated_options = []
        
        for option in options:
            evaluation = self._evaluate_single_option(option, context)
            option_with_eval = option.copy()
            option_with_eval['evaluation'] = evaluation
            evaluated_options.append(option_with_eval)
        
        return evaluated_options
    
    def _evaluate_single_option(self, option: Dict, context: Dict) -> Dict[str, Any]:
        """Evaluiert eine einzelne Option"""
        try:
            evaluation = {}
            
            # Utility evaluation
            evaluation['utility'] = self._evaluate_utility(option, context)
            
            # Risk evaluation
            evaluation['risk'] = self._evaluate_risk(option, context)
            
            # Feasibility evaluation
            evaluation['feasibility'] = self._evaluate_feasibility(option, context)
            
            # Alignment evaluation (mit Zielen/Werten)
            evaluation['alignment'] = self._evaluate_alignment(option, context)
            
            # Resources evaluation
            evaluation['resources'] = self._evaluate_resources(option, context)
            
            # Calculate weighted score
            weighted_score = 0.0
            for criterion, score in evaluation.items():
                weight = self.decision_criteria_weights.get(criterion, 0.0)
                weighted_score += score * weight
            
            evaluation['weighted_score'] = weighted_score
            evaluation['overall_rating'] = self._rate_option(weighted_score)
            
            return evaluation
            
        except Exception as e:
            logger.debug(f"Option evaluation failed: {e}")
            return {'weighted_score': 0.0, 'overall_rating': 'poor'}
    
    def _evaluate_utility(self, option: Dict, context: Dict) -> float:
        """Evaluiert den Nutzen einer Option"""
        try:
            # Base utility from option
            base_utility = option.get('expected_utility', 0.5)
            
            # Context-based adjustments
            if context:
                current_goals = context.get('current_goals', [])
                option_benefits = option.get('benefits', [])
                
                # Boost utility if option aligns with current goals
                goal_alignment = self._calculate_goal_alignment(option_benefits, current_goals)
                base_utility += goal_alignment * 0.3
            
            return max(0.0, min(1.0, base_utility))
            
        except Exception as e:
            logger.debug(f"Utility evaluation failed: {e}")
            return 0.5
    
    def _evaluate_risk(self, option: Dict, context: Dict) -> float:
        """Evaluiert das Risiko einer Option (höher = weniger riskant)"""
        try:
            # Base risk from option (invert because higher risk = lower score)
            base_risk = option.get('risk_level', 0.5)
            risk_score = 1.0 - base_risk
            
            # Adjust based on context
            if context:
                risk_tolerance = context.get('risk_tolerance', 0.5)
                # If high risk tolerance, don't penalize risky options as much
                if risk_tolerance > 0.7:
                    risk_score += (base_risk * 0.2)  # Boost risky options for high risk tolerance
            
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            logger.debug(f"Risk evaluation failed: {e}")
            return 0.5
    
    def _evaluate_feasibility(self, option: Dict, context: Dict) -> float:
        """Evaluiert die Durchführbarkeit einer Option"""
        try:
            base_feasibility = option.get('feasibility', 0.5)
            
            # Adjust based on available resources
            required_resources = option.get('required_resources', {})
            available_resources = context.get('available_resources', {}) if context else {}
            
            if required_resources and available_resources:
                resource_match = self._calculate_resource_match(required_resources, available_resources)
                base_feasibility *= resource_match
            
            return max(0.0, min(1.0, base_feasibility))
            
        except Exception as e:
            logger.debug(f"Feasibility evaluation failed: {e}")
            return 0.5
    
    def _evaluate_alignment(self, option: Dict, context: Dict) -> float:
        """Evaluiert Alignment mit Werten und Zielen"""
        try:
            base_alignment = option.get('value_alignment', 0.5)
            
            if context:
                personal_values = context.get('personal_values', [])
                option_values = option.get('aligned_values', [])
                
                if personal_values and option_values:
                    value_overlap = len(set(personal_values) & set(option_values)) / max(len(personal_values), 1)
                    base_alignment += value_overlap * 0.3
            
            return max(0.0, min(1.0, base_alignment))
            
        except Exception as e:
            logger.debug(f"Alignment evaluation failed: {e}")
            return 0.5
    
    def _evaluate_resources(self, option: Dict, context: Dict) -> float:
        """Evaluiert Resource-Effizienz einer Option"""
        try:
            resource_efficiency = option.get('resource_efficiency', 0.5)
            
            # Penalize options that require too many resources
            required_resources = option.get('required_resources', {})
            if required_resources:
                total_requirement = sum(required_resources.values()) if isinstance(required_resources, dict) else len(required_resources)
                if total_requirement > 10:  # Arbitrary threshold
                    resource_efficiency *= 0.8
            
            return max(0.0, min(1.0, resource_efficiency))
            
        except Exception as e:
            logger.debug(f"Resources evaluation failed: {e}")
            return 0.5
    
    def _select_best_option(self, evaluated_options: List[Dict]) -> Dict:
        """Wählt die beste Option aus"""
        try:
            if not evaluated_options:
                return {}
            
            # Sort by weighted score
            sorted_options = sorted(evaluated_options, 
                                  key=lambda x: x.get('evaluation', {}).get('weighted_score', 0.0), 
                                  reverse=True)
            
            return sorted_options[0]
            
        except Exception as e:
            logger.debug(f"Best option selection failed: {e}")
            return evaluated_options[0] if evaluated_options else {}
    
    def _calculate_decision_confidence(self, evaluated_options: List[Dict], best_option: Dict) -> float:
        """Berechnet Confidence in der Entscheidung"""
        try:
            if len(evaluated_options) < 2:
                return 0.8  # High confidence if only one option
            
            scores = [opt.get('evaluation', {}).get('weighted_score', 0.0) for opt in evaluated_options]
            best_score = best_option.get('evaluation', {}).get('weighted_score', 0.0)
            
            # Calculate score gap
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) >= 2:
                score_gap = sorted_scores[0] - sorted_scores[1]
                # Higher gap = higher confidence
                confidence = 0.5 + (score_gap * 0.5)
            else:
                confidence = 0.8
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.debug(f"Decision confidence calculation failed: {e}")
            return 0.5

    def _calculate_goal_alignment(self, benefits: List, goals: List) -> float:
        """Berechnet Alignment zwischen Benefits und Goals"""
        if not benefits or not goals:
            return 0.0
        
        # Simple text matching for alignment
        alignment_count = 0
        for benefit in benefits:
            for goal in goals:
                if str(benefit).lower() in str(goal).lower() or str(goal).lower() in str(benefit).lower():
                    alignment_count += 1
                    break
        
        return alignment_count / len(benefits)
    
    def _calculate_resource_match(self, required: Dict, available: Dict) -> float:
        """Berechnet Resource Match"""
        if not required:
            return 1.0
        
        match_score = 0.0
        for resource, amount in required.items():
            available_amount = available.get(resource, 0)
            if available_amount >= amount:
                match_score += 1.0
            else:
                match_score += available_amount / amount if amount > 0 else 0.0
        
        return match_score / len(required)
    
    def _rate_option(self, score: float) -> str:
        """Bewertet Option basierend auf Score"""
        if score > 0.8:
            return 'excellent'
        elif score > 0.6:
            return 'good'
        elif score > 0.4:
            return 'moderate'
        else:
            return 'poor'

def evaluate_decision_options(options: List[Dict], 
                            decision_context: Dict = None,
                            decision_criteria: Dict = None) -> Dict[str, Any]:
    """Evaluiert Decision Options - HAUPTINTERFACE"""
    try:
        decision_engine = DecisionEngine()
        
        if not options:
            return {
                'evaluation_result': 'no_options',
                'recommendation': None,
                'analysis': 'No options provided for evaluation'
            }
        
        # Make decision
        decision_result = decision_engine.make_decision(options, decision_context, decision_criteria)
        
        # Analyze decision quality
        decision_analysis = _analyze_decision_quality(decision_result)
        
        # Generate recommendations
        recommendations = _generate_decision_recommendations(decision_result, decision_analysis)
        
        return {
            'evaluation_result': 'success',
            'decision_result': decision_result,
            'decision_analysis': decision_analysis,
            'recommendations': recommendations,
            'evaluation_metadata': {
                'options_evaluated': len(options),
                'evaluation_timestamp': datetime.now().isoformat(),
                'context_provided': bool(decision_context)
            }
        }
        
    except Exception as e:
        logger.error(f"Decision options evaluation failed: {e}")
        return {
            'evaluation_result': 'error',
            'error': str(e),
            'recommendation': None
        }

def make_autonomous_decision(decision_scenario: Dict, 
                           personality_data: Dict = None,
                           memory_manager=None) -> Dict[str, Any]:
    """Macht autonome Entscheidungen basierend auf Scenario"""
    try:
        # Extract options from scenario
        options = decision_scenario.get('options', [])
        scenario_context = decision_scenario.get('context', {})
        
        # Enhance context with personality data
        if personality_data:
            scenario_context.update({
                'personal_values': personality_data.get('core_values', []),
                'risk_tolerance': personality_data.get('risk_tolerance', 0.5),
                'decision_style': personality_data.get('decision_style', 'balanced')
            })
        
        # Use memory manager for context if available
        if memory_manager:
            try:
                recent_decisions = getattr(memory_manager, 'get_recent_decisions', lambda: [])()
                scenario_context['recent_decisions'] = recent_decisions[:5]  # Last 5 decisions
            except:
                pass
        
        # Make decision
        decision_result = evaluate_decision_options(options, scenario_context)
        
        # Add autonomy metadata
        decision_result['autonomy_metadata'] = {
            'decision_mode': 'autonomous',
            'personality_influenced': bool(personality_data),
            'memory_informed': bool(memory_manager),
            'decision_timestamp': datetime.now().isoformat()
        }
        
        return decision_result
        
    except Exception as e:
        logger.error(f"Autonomous decision making failed: {e}")
        return {
            'evaluation_result': 'error',
            'error': str(e),
            'decision_mode': 'autonomous'
        }

def analyze_decision_patterns(decision_history: List[Dict] = None,
                            time_window_days: int = 7) -> Dict[str, Any]:
    """Analysiert Decision Patterns über Zeit"""
    try:
        if not decision_history:
            return {
                'pattern_analysis': 'no_history',
                'insights': ['No decision history available for analysis']
            }
        
        # Basic pattern analysis
        total_decisions = len(decision_history)
        
        # Confidence analysis
        confidences = [d.get('confidence', 0.5) for d in decision_history]
        avg_confidence = statistics.mean(confidences) if confidences else 0.5
        
        # Decision quality trends
        recent_decisions = decision_history[-10:] if len(decision_history) >= 10 else decision_history
        recent_confidences = [d.get('confidence', 0.5) for d in recent_decisions]
        recent_avg_confidence = statistics.mean(recent_confidences) if recent_confidences else 0.5
        
        # Pattern insights
        confidence_trend = 'improving' if recent_avg_confidence > avg_confidence else 'declining' if recent_avg_confidence < avg_confidence - 0.1 else 'stable'
        
        return {
            'pattern_analysis': {
                'total_decisions': total_decisions,
                'average_confidence': avg_confidence,
                'recent_confidence': recent_avg_confidence,
                'confidence_trend': confidence_trend,
                'decision_frequency': total_decisions / max(time_window_days, 1)
            },
            'insights': _generate_pattern_insights(avg_confidence, confidence_trend, total_decisions),
            'analysis_metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'history_length': total_decisions,
                'time_window_days': time_window_days
            }
        }
        
    except Exception as e:
        logger.error(f"Decision pattern analysis failed: {e}")
        return {
            'pattern_analysis': 'error',
            'error': str(e),
            'insights': ['Unable to analyze decision patterns']
        }

# Helper Functions
def _analyze_decision_quality(decision_result: Dict) -> Dict[str, Any]:
    """Analysiert Decision Quality"""
    try:
        confidence = decision_result.get('confidence', 0.5)
        decision = decision_result.get('decision', {})
        
        quality_indicators = {
            'confidence_level': 'high' if confidence > 0.8 else 'moderate' if confidence > 0.6 else 'low',
            'option_evaluation_completeness': 'complete' if decision.get('evaluation') else 'partial',
            'context_consideration': 'considered' if decision_result.get('context') else 'not_considered'
        }
        
        overall_quality = 'good' if confidence > 0.7 else 'moderate' if confidence > 0.5 else 'needs_improvement'
        
        return {
            'overall_quality': overall_quality,
            'quality_indicators': quality_indicators,
            'confidence_score': confidence
        }
        
    except Exception as e:
        logger.debug(f"Decision quality analysis failed: {e}")
        return {'overall_quality': 'unknown', 'confidence_score': 0.5}

def _generate_decision_recommendations(decision_result: Dict, analysis: Dict) -> List[str]:
    """Generiert Decision Recommendations"""
    recommendations = []
    
    try:
        confidence = decision_result.get('confidence', 0.5)
        overall_quality = analysis.get('overall_quality', 'moderate')
        
        if confidence < 0.6:
            recommendations.append("Low decision confidence - consider gathering more information")
        
        if overall_quality == 'needs_improvement':
            recommendations.append("Decision quality could be improved - review decision criteria")
        
        decision = decision_result.get('decision', {})
        if decision and decision.get('evaluation', {}).get('risk', 0.5) < 0.3:
            recommendations.append("High-risk option selected - ensure risk mitigation strategies")
        
        if not decision_result.get('context'):
            recommendations.append("Consider providing more context for better decision making")
        
        return recommendations if recommendations else ["Decision appears well-reasoned"]
        
    except Exception as e:
        logger.debug(f"Decision recommendations generation failed: {e}")
        return ["Unable to generate specific recommendations"]

def _generate_pattern_insights(avg_confidence: float, trend: str, total_decisions: int) -> List[str]:
    """Generiert Pattern Insights"""
    insights = []
    
    if avg_confidence > 0.8:
        insights.append("High average decision confidence indicates good decision-making capability")
    elif avg_confidence < 0.5:
        insights.append("Low average confidence suggests need for improved decision processes")
    
    if trend == 'improving':
        insights.append("Decision confidence is improving over time - positive trend")
    elif trend == 'declining':
        insights.append("Decision confidence is declining - may need process review")
    
    if total_decisions < 3:
        insights.append("Limited decision history - patterns may not be fully representative")
    elif total_decisions > 20:
        insights.append("Rich decision history available for pattern analysis")
    
    return insights if insights else ["Decision patterns appear normal"]

__all__ = [
    'DecisionEngine',
    'evaluate_decision_options',
    'make_autonomous_decision',
    'analyze_decision_patterns'
]