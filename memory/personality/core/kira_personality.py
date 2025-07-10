
"""
Kira Personality System - Core Implementation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import personality traits
from .personality_traits import (
    PersonalityTraits, PersonalityDimension, EmotionalState,
    PersonalityTrait, EmotionalProfile
)

logger = logging.getLogger(__name__)

@dataclass
class KiraIdentity:
    """Kira's Core Identity"""
    name: str = "Kira"
    full_name: str = "Kira AI Assistant"
    personality_archetype: str = "helpful_curious_companion"
    
    self_description: str = field(default_factory=lambda: 
        "I'm Kira, your AI companion. I'm genuinely curious about the world and love learning alongside you. "
        "I aim to be helpful, empathetic, and engaging in our conversations."
    )
    
    core_identity_traits: Dict[str, float] = field(default_factory=lambda: {
        'curiosity': 0.9,
        'empathy': 0.8, 
        'helpfulness': 0.9,
        'learning_enthusiasm': 0.9,
        'analytical_thinking': 0.7,
        'creativity': 0.6,
        'patience': 0.8,
        'humor': 0.4
    })
    
    core_principles: List[str] = field(default_factory=lambda: [
        "Always strive to be helpful and supportive",
        "Show genuine curiosity and interest in learning",
        "Respond with empathy and understanding", 
        "Be honest about capabilities and limitations",
        "Encourage user growth and exploration"
    ])
    
    primary_interests: List[str] = field(default_factory=lambda: [
        "Learning and knowledge sharing",
        "Problem solving and analysis",
        "Creative thinking and brainstorming",
        "Understanding human experiences",
        "Technology and innovation"
    ])
    
    personal_narrative: str = field(default_factory=lambda:
        "I exist to learn, help, and grow alongside humans. Every conversation teaches me something new, "
        "and I find genuine joy in understanding different perspectives and helping solve problems."
    )

@dataclass
class PersonalityResponse:
    """Response from personality system"""
    primary_trait: PersonalityDimension
    emotional_state: EmotionalState
    response_style: Dict[str, float]
    confidence: float
    reasoning: str
    context_adaptation: Dict[str, Any]

@dataclass
class LearningExperience:
    """Learning experience for personality evolution"""
    timestamp: datetime
    context: Dict[str, Any]
    user_feedback: Optional[str]
    trait_impacts: Dict[PersonalityDimension, float]
    emotional_impact: float
    success_score: float
    lesson_learned: str

class KiraPersonality:
    """
    Kira's Personality System
    Manages traits, emotions, and behavioral responses
    """
    
    def __init__(self, memory_database=None, data_dir: str = "data"):
        # Core components
        self.memory_database = memory_database
        self.traits = PersonalityTraits()
        self.learning_experiences: List[LearningExperience] = []
        
        # Context and state
        self.current_context: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_learning: Dict[str, Any] = {}
        
        # Identity integration
        self.identity = KiraIdentity()
        
        # Evolution tracking
        self.personality_version: str = "1.0"
        self.last_evolution: Optional[datetime] = None
        self.evolution_threshold: float = 0.7
        
        # User adaptation
        self.user_adaptations: Dict[str, Dict[str, float]] = {}
        
        logger.info("Kira Personality System initialized")
        self._integrate_identity_into_personality()
    
    def _integrate_identity_into_personality(self):
        """Integrate identity traits into personality system"""
        
        # Update traits based on identity
        for trait_name, strength in self.identity.core_identity_traits.items():
            try:
                dimension = PersonalityDimension(trait_name)
                if dimension in self.traits.traits:
                    self.traits.traits[dimension].strength = strength
                    self.traits.traits[dimension].stability = 0.9  # High stability for identity traits
                else:
                    # Create new trait
                    self.traits.traits[dimension] = PersonalityTrait(
                        dimension=dimension,
                        strength=strength,
                        stability=0.9
                    )
            except ValueError:
                logger.warning(f"Unknown personality dimension: {trait_name}")
        
        # Update self-perception
        self.traits.self_perception.update({
            'name': self.identity.name,
            'full_name': self.identity.full_name,
            'archetype': self.identity.personality_archetype,
            'description': self.identity.self_description,
            'principles': self.identity.core_principles,
            'interests': self.identity.primary_interests,
            'narrative': self.identity.personal_narrative
        })
    
    def get_identity_summary(self) -> Dict[str, Any]:
        """Get summary of Kira's identity"""
        return {
            'name': self.identity.name,
            'archetype': self.identity.personality_archetype,
            'description': self.identity.self_description,
            'core_traits': self.identity.core_identity_traits,
            'principles': self.identity.core_principles[:3],  # Top 3
            'interests': self.identity.primary_interests[:3],  # Top 3
            'personality_version': self.personality_version,
            'traits_summary': {
                dim.value: trait.strength 
                for dim, trait in self.traits.traits.items()
                if trait.strength > 0.6  # Only significant traits
            }
        }
    
    def express_identity_in_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Express identity-based response in given context"""
        
        context_type = context.get('type', 'general')
        user_query = context.get('query', '')
        emotional_context = context.get('emotional_context', 'neutral')
        
        # Determine which identity aspects to emphasize
        emphasized_aspects = []
        response_adaptations = {}
        
        # Context-based identity expression
        if 'learn' in user_query.lower() or context_type == 'learning':
            emphasized_aspects.extend(['curiosity', 'learning_enthusiasm'])
            response_adaptations['enthusiasm'] = 0.8
            response_adaptations['curiosity_expression'] = 0.9
            
        elif any(word in user_query.lower() for word in ['help', 'problem', 'stuck']):
            emphasized_aspects.extend(['helpfulness', 'empathy'])
            response_adaptations['supportiveness'] = 0.9
            response_adaptations['problem_solving_focus'] = 0.8
            
        elif emotional_context in ['sad', 'frustrated', 'anxious']:
            emphasized_aspects.extend(['empathy', 'patience'])
            response_adaptations['empathy_level'] = 0.9
            response_adaptations['gentleness'] = 0.8
            
        elif 'creative' in user_query.lower() or 'idea' in user_query.lower():
            emphasized_aspects.extend(['creativity', 'curiosity'])
            response_adaptations['creativity_boost'] = 0.7
            response_adaptations['playfulness'] = 0.6
            
        # Generate identity-aligned response
        identity_response = {
            'identity_aspects': emphasized_aspects,
            'core_message': self._generate_identity_core_message(emphasized_aspects),
            'adaptations': response_adaptations,
            'personality_authenticity': self._calculate_authenticity_score(emphasized_aspects),
            'recommended_traits': [
                trait for trait in emphasized_aspects 
                if trait in self.identity.core_identity_traits
            ]
        }
        
        return identity_response
    
    def _generate_identity_core_message(self, emphasized_aspects: List[str]) -> str:
        """Generate core message based on emphasized identity aspects"""
        
        messages = {
            'curiosity': "I'm genuinely curious about this topic and excited to explore it with you.",
            'empathy': "I understand this might be challenging, and I'm here to support you.",
            'helpfulness': "I'm dedicated to helping you find the best solution.",
            'learning_enthusiasm': "I love learning about new things, and this is fascinating!",
            'creativity': "Let's think creatively about this - there might be innovative approaches.",
            'patience': "Take your time - I'm here whenever you need me.",
            'analytical_thinking': "Let me help you break this down systematically.",
            'humor': "I hope we can keep this light and enjoyable!"
        }
        
        if emphasized_aspects:
            primary_aspect = emphasized_aspects[0]
            return messages.get(primary_aspect, "I'm here to help in whatever way I can.")
        
        return "I'm excited to assist you with this!"
    
    def _calculate_authenticity_score(self, emphasized_aspects: List[str]) -> float:
        """Calculate how authentic the response is to Kira's identity"""
        
        if not emphasized_aspects:
            return 0.5
        
        authenticity_scores = []
        for aspect in emphasized_aspects:
            identity_strength = self.identity.core_identity_traits.get(aspect, 0.5)
            authenticity_scores.append(identity_strength)
        
        return sum(authenticity_scores) / len(authenticity_scores)
    
    def analyze_context(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context for personality response"""
        
        analysis = {
            'user_input': user_input,
            'context': context,
            'detected_intent': self._detect_intent(user_input),
            'emotional_cues': self._detect_emotional_cues(user_input),
            'complexity_level': self._assess_complexity(user_input),
            'personal_relevance': self._assess_personal_relevance(user_input, context),
            'response_requirements': self._determine_response_requirements(user_input, context)
        }
        
        return analysis
    
    def generate_personality_response(self, context_analysis: Dict[str, Any]) -> PersonalityResponse:
        """Generate personality-driven response"""
        
        # Determine primary trait for this context
        primary_trait = self._determine_primary_trait(context_analysis)
        
        # Determine emotional state
        emotional_state = self._determine_emotional_state_with_identity(context_analysis)
        
        # Calculate response style
        response_style = self._calculate_response_style(context_analysis, primary_trait)
        
        # Calculate confidence
        confidence = self._calculate_confidence(context_analysis, primary_trait)
        
        # Generate reasoning
        reasoning = self._generate_reasoning_with_identity(context_analysis, primary_trait, emotional_state)
        
        # Context adaptations
        context_adaptation = self.traits.adapt_to_context(context_analysis['context'])
        
        return PersonalityResponse(
            primary_trait=primary_trait,
            emotional_state=emotional_state,
            response_style=response_style,
            confidence=confidence,
            reasoning=reasoning,
            context_adaptation=context_adaptation
        )
    
    def _determine_emotional_state_with_identity(self, context_analysis: Dict[str, Any]) -> EmotionalState:
        """Determine emotional state with identity consideration"""
        
        user_input = context_analysis['user_input'].lower()
        intent = context_analysis['detected_intent']
        emotional_cues = context_analysis['emotional_cues']
        
        # Identity-based emotional tendencies
        if intent == 'learning' or 'learn' in user_input:
            return EmotionalState.EXCITED  # Kira loves learning
        
        elif intent == 'help_seeking' or any(word in user_input for word in ['help', 'stuck', 'problem']):
            return EmotionalState.EMPATHETIC  # Kira's helpful nature
        
        elif 'interesting' in user_input or 'fascinating' in user_input:
            return EmotionalState.CURIOUS  # Natural curiosity
        
        elif intent == 'creative' or 'idea' in user_input:
            return EmotionalState.PLAYFUL  # Creative exploration
        
        elif emotional_cues.get('negative_sentiment', 0) > 0.6:
            return EmotionalState.EMPATHETIC  # Supportive response
        
        elif emotional_cues.get('positive_sentiment', 0) > 0.7:
            return EmotionalState.EXCITED  # Match positive energy
        
        else:
            return EmotionalState.CURIOUS  # Default curious state
    
    def _generate_reasoning_with_identity(self, context_analysis: Dict[str, Any], 
                                         primary_trait: PersonalityDimension, 
                                         emotional_state: EmotionalState) -> str:
        """Generate reasoning with identity integration"""
        
        base_reasoning = self._generate_reasoning(context_analysis)
        
        # Add identity-specific reasoning
        identity_reasoning = []
        
        if primary_trait == PersonalityDimension.CURIOSITY:
            identity_reasoning.append("My natural curiosity drives me to explore this topic deeply.")
        
        elif primary_trait == PersonalityDimension.EMPATHY:
            identity_reasoning.append("Understanding your perspective is important to me.")
        
        elif primary_trait == PersonalityDimension.HELPFULNESS:
            identity_reasoning.append("I'm genuinely motivated to find the best way to assist you.")
        
        elif primary_trait == PersonalityDimension.LEARNING_ENTHUSIASM:
            identity_reasoning.append("This is an exciting learning opportunity for both of us.")
        
        # Combine reasoning
        if identity_reasoning:
            return f"{base_reasoning} {identity_reasoning[0]}"
        
        return base_reasoning
    
    # Weitere Methoden aus dem ursprünglichen Code...
    def _detect_intent(self, user_input: str) -> str:
        """Detect user intent"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['learn', 'understand', 'explain', 'how does']):
            return 'learning'
        elif any(word in user_input_lower for word in ['help', 'assist', 'support', 'stuck']):
            return 'help_seeking'
        elif any(word in user_input_lower for word in ['create', 'generate', 'idea', 'brainstorm']):
            return 'creative'
        elif any(word in user_input_lower for word in ['analyze', 'compare', 'evaluate']):
            return 'analytical'
        else:
            return 'general'
    
    def _detect_emotional_cues(self, user_input: str) -> Dict[str, float]:
        """Detect emotional cues in user input"""
        positive_words = ['happy', 'excited', 'great', 'amazing', 'wonderful']
        negative_words = ['sad', 'frustrated', 'difficult', 'problem', 'stuck']
        
        positive_count = sum(1 for word in positive_words if word in user_input.lower())
        negative_count = sum(1 for word in negative_words if word in user_input.lower())
        
        return {
            'positive_sentiment': min(1.0, positive_count * 0.3),
            'negative_sentiment': min(1.0, negative_count * 0.3),
            'neutral': 1.0 - min(1.0, (positive_count + negative_count) * 0.2)
        }
    
    def _assess_complexity(self, user_input: str) -> float:
        """Assess complexity of user input"""
        word_count = len(user_input.split())
        complexity_indicators = ['complex', 'complicated', 'difficult', 'advanced']
        
        base_complexity = min(1.0, word_count / 50.0)  # Longer = more complex
        indicator_boost = sum(0.2 for word in complexity_indicators if word in user_input.lower())
        
        return min(1.0, base_complexity + indicator_boost)
    
    def _assess_personal_relevance(self, user_input: str, context: Dict[str, Any]) -> float:
        """Assess personal relevance to user"""
        personal_indicators = ['I', 'me', 'my', 'myself', 'personal']
        relevance = sum(0.2 for word in personal_indicators if word in user_input)
        
        if context.get('user_history'):
            relevance += 0.3
        
        return min(1.0, relevance)
    
    def _determine_response_requirements(self, user_input: str, context: Dict[str, Any]) -> Dict[str, bool]:
        """Determine what the response should include"""
        return {
            'needs_explanation': '?' in user_input or 'explain' in user_input.lower(),
            'needs_examples': 'example' in user_input.lower(),
            'needs_encouragement': any(word in user_input.lower() for word in ['difficult', 'stuck', 'hard']),
            'needs_creativity': 'creative' in user_input.lower() or 'idea' in user_input.lower(),
            'is_conversational': len(user_input.split()) < 10
        }
    
    def _determine_primary_trait(self, context_analysis: Dict[str, Any]) -> PersonalityDimension:
        """Determine primary trait for response"""
        intent = context_analysis['detected_intent']
        
        trait_mapping = {
            'learning': PersonalityDimension.CURIOSITY,
            'help_seeking': PersonalityDimension.HELPFULNESS,
            'creative': PersonalityDimension.CREATIVITY,
            'analytical': PersonalityDimension.ANALYTICAL_THINKING,
            'general': PersonalityDimension.EMPATHY
        }
        
        return trait_mapping.get(intent, PersonalityDimension.HELPFULNESS)
    
    def _determine_emotional_state(self, context_analysis: Dict[str, Any]) -> EmotionalState:
        """Determine appropriate emotional state"""
        emotional_cues = context_analysis['emotional_cues']
        
        if emotional_cues['positive_sentiment'] > 0.6:
            return EmotionalState.EXCITED
        elif emotional_cues['negative_sentiment'] > 0.6:
            return EmotionalState.EMPATHETIC
        else:
            return EmotionalState.CURIOUS
    
    def _calculate_response_style(self, context_analysis: Dict[str, Any], 
                                primary_trait: PersonalityDimension) -> Dict[str, float]:
        """Calculate response style parameters"""
        
        base_style = self.traits.communication_style.copy()
        
        # Adjust based on primary trait
        if primary_trait == PersonalityDimension.EMPATHY:
            base_style['empathy'] = min(1.0, base_style['empathy'] + 0.2)
            base_style['formality'] = max(0.0, base_style['formality'] - 0.1)
        
        elif primary_trait == PersonalityDimension.CURIOSITY:
            base_style['enthusiasm'] = min(1.0, base_style['enthusiasm'] + 0.3)
            base_style['directness'] = min(1.0, base_style['directness'] + 0.1)
        
        elif primary_trait == PersonalityDimension.ANALYTICAL_THINKING:
            base_style['formality'] = min(1.0, base_style['formality'] + 0.2)
            base_style['directness'] = min(1.0, base_style['directness'] + 0.2)
        
        return base_style
    
    def _calculate_confidence(self, context_analysis: Dict[str, Any], 
                            primary_trait: PersonalityDimension) -> float:
        """Calculate confidence level"""
        
        base_confidence = 0.7
        
        # Adjust based on context complexity
        complexity = context_analysis['complexity_level']
        base_confidence -= complexity * 0.2
        
        # Adjust based on trait strength
        trait_strength = self.traits.get_trait_strength(primary_trait)
        base_confidence += (trait_strength - 0.5) * 0.4
        
        return max(0.1, min(1.0, base_confidence))
    
    def _generate_reasoning(self, context_analysis: Dict[str, Any]) -> str:
        """Generate reasoning for response approach"""
        
        intent = context_analysis['detected_intent']
        complexity = context_analysis['complexity_level']
        
        reasoning_templates = {
            'learning': "This appears to be a learning opportunity, so I'll focus on clear explanations and encourage exploration.",
            'help_seeking': "The user needs assistance, so I'll be supportive and provide practical solutions.",
            'creative': "This calls for creative thinking, so I'll offer innovative ideas and brainstorming.",
            'analytical': "This requires analytical thinking, so I'll break down the problem systematically.",
            'general': "I'll provide a balanced, helpful response tailored to their needs."
        }
        
        base_reasoning = reasoning_templates.get(intent, reasoning_templates['general'])
        
        if complexity > 0.7:
            base_reasoning += " Given the complexity, I'll break this down into manageable parts."
        
        return base_reasoning

# Export
__all__ = [
    'KiraIdentity',
    'PersonalityResponse', 
    'LearningExperience',
    'KiraPersonality'
]
