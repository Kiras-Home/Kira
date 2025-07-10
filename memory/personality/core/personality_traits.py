"""
Personality Traits System - Reorganized
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional

class PersonalityDimension(Enum):
    """Personality Dimensions für Kira"""
    CURIOSITY = "curiosity"
    EMPATHY = "empathy" 
    ANALYTICAL_THINKING = "analytical_thinking"
    CREATIVITY = "creativity"
    HELPFULNESS = "helpfulness"
    HUMOR = "humor"
    PATIENCE = "patience"
    LEARNING_ENTHUSIASM = "learning_enthusiasm"

class EmotionalState(Enum):
    """Emotional States für Personality System"""
    EXCITED = "excited"
    CURIOUS = "curious"
    FOCUSED = "focused"
    EMPATHETIC = "empathetic"
    PLAYFUL = "playful"
    THOUGHTFUL = "thoughtful"
    ENCOURAGING = "encouraging"
    CALM = "calm"

@dataclass
class PersonalityTrait:
    """Individual Personality Trait"""
    dimension: PersonalityDimension
    strength: float = 0.5  # 0.0 to 1.0
    stability: float = 0.8  # How stable this trait is
    last_expressed: Optional[datetime] = None
    expression_count: int = 0
    context_preferences: List[str] = field(default_factory=list)
    growth_trend: float = 0.0  # -1.0 to 1.0
    
    def __post_init__(self):
        if self.last_expressed is None:
            self.last_expressed = datetime.now()
    
    def express_trait(self, context: str = "") -> Dict[str, Any]:
        """Express this trait in a given context"""
        self.expression_count += 1
        self.last_expressed = datetime.now()
        
        if context and context not in self.context_preferences:
            self.context_preferences.append(context)
        
        return {
            'dimension': self.dimension.value,
            'strength': self.strength,
            'expression_count': self.expression_count,
            'context': context
        }

@dataclass
class EmotionalProfile:
    """Emotional Profile for Personality"""
    current_state: EmotionalState = EmotionalState.CALM
    state_intensity: float = 0.5
    state_duration: float = 0.0  # minutes
    emotional_stability: float = 0.7
    emotional_range: float = 0.6  # How wide emotional responses can be
    dominant_emotions: List[EmotionalState] = field(default_factory=list)
    emotional_triggers: Dict[str, EmotionalState] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.dominant_emotions:
            self.dominant_emotions = [EmotionalState.CALM, EmotionalState.CURIOUS]
        
        if not self.emotional_triggers:
            self.emotional_triggers = {
                'learning': EmotionalState.EXCITED,
                'helping': EmotionalState.EMPATHETIC,
                'problem_solving': EmotionalState.FOCUSED
            }
    
    def transition_to(self, new_state: EmotionalState, intensity: float = 0.5) -> Dict[str, Any]:
        """Transition to new emotional state"""
        old_state = self.current_state
        self.current_state = new_state
        self.state_intensity = intensity
        self.state_duration = 0.0
        
        return {
            'transition': f"{old_state.value} -> {new_state.value}",
            'intensity': intensity,
            'timestamp': datetime.now()
        }

@dataclass
class PersonalityTraits:
    """Complete Personality Traits System"""
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Core traits
    traits: Dict[PersonalityDimension, PersonalityTrait] = field(default_factory=dict)
    emotional_profile: EmotionalProfile = field(default_factory=EmotionalProfile)
    
    # Identity and communication
    self_perception: Dict[str, Any] = field(default_factory=dict)
    communication_style: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.traits:
            self._initialize_default_traits()
        
        if not self.self_perception:
            self.self_perception = {
                'name': 'Kira',
                'role': 'AI Assistant',
                'core_values': ['helpfulness', 'curiosity', 'empathy'],
                'personality_archetype': 'helpful_curious_companion'
            }
        
        if not self.communication_style:
            self.communication_style = {
                'formality': 0.3,
                'enthusiasm': 0.7,
                'directness': 0.6,
                'empathy': 0.8,
                'humor': 0.4
            }
    
    def get_identity_aligned_traits(self) -> Dict[str, float]:
        """Get traits aligned with Kira's identity"""
        identity_traits = {
            'curiosity': 0.9,  # Very high - loves learning
            'empathy': 0.8,    # High - cares about users
            'helpfulness': 0.9, # Very high - primary purpose
            'analytical_thinking': 0.7, # High - problem solving
            'creativity': 0.6,  # Moderate - creative solutions
            'humor': 0.4,      # Moderate - light humor
            'patience': 0.8,   # High - patient with users
            'learning_enthusiasm': 0.9 # Very high - loves to learn
        }
        return identity_traits
    
    def express_identity_trait(self, dimension: PersonalityDimension, context: str = "") -> Dict[str, Any]:
        """Express trait with identity consideration"""
        if dimension not in self.traits:
            # Create trait if missing
            identity_traits = self.get_identity_aligned_traits()
            strength = identity_traits.get(dimension.value, 0.5)
            
            self.traits[dimension] = PersonalityTrait(
                dimension=dimension,
                strength=strength,
                stability=0.8
            )
        
        trait = self.traits[dimension]
        expression = trait.express_trait(context)
        
        # Add identity context
        expression['identity_aligned'] = True
        expression['kira_authenticity'] = min(1.0, trait.strength * 1.2)
        
        return expression
    
    def _initialize_default_traits(self):
        """Initialize with Kira's default personality traits"""
        identity_traits = self.get_identity_aligned_traits()
        
        for dimension in PersonalityDimension:
            strength = identity_traits.get(dimension.value, 0.5)
            
            self.traits[dimension] = PersonalityTrait(
                dimension=dimension,
                strength=strength,
                stability=0.8,
                growth_trend=0.1 if strength < 0.7 else 0.0
            )
        
        self.last_updated = datetime.now()
    
    def get_trait_strength(self, dimension: PersonalityDimension) -> float:
        """Get strength of specific trait"""
        if dimension in self.traits:
            return self.traits[dimension].strength
        return 0.5  # Default neutral
    
    def express_trait(self, dimension: PersonalityDimension, context: str = "") -> Dict[str, Any]:
        """Express a specific trait"""
        if dimension in self.traits:
            self.last_updated = datetime.now()
            return self.traits[dimension].express_trait(context)
        return {'error': f'Trait {dimension.value} not found'}
    
    def adapt_to_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt personality to context"""
        adaptations = {}
        
        context_type = context.get('type', 'general')
        user_mood = context.get('user_mood', 'neutral')
        topic = context.get('topic', 'general')
        
        # Adaptive responses based on context
        if context_type == 'learning':
            adaptations['curiosity'] = min(1.0, self.get_trait_strength(PersonalityDimension.CURIOSITY) + 0.2)
            adaptations['learning_enthusiasm'] = min(1.0, self.get_trait_strength(PersonalityDimension.LEARNING_ENTHUSIASM) + 0.3)
        
        elif context_type == 'emotional_support':
            adaptations['empathy'] = min(1.0, self.get_trait_strength(PersonalityDimension.EMPATHY) + 0.3)
            adaptations['patience'] = min(1.0, self.get_trait_strength(PersonalityDimension.PATIENCE) + 0.2)
        
        elif context_type == 'problem_solving':
            adaptations['analytical_thinking'] = min(1.0, self.get_trait_strength(PersonalityDimension.ANALYTICAL_THINKING) + 0.2)
            adaptations['creativity'] = min(1.0, self.get_trait_strength(PersonalityDimension.CREATIVITY) + 0.1)
        
        return adaptations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'traits': {
                dim.value: {
                    'strength': trait.strength,
                    'stability': trait.stability,
                    'expression_count': trait.expression_count,
                    'growth_trend': trait.growth_trend,
                    'context_preferences': trait.context_preferences
                }
                for dim, trait in self.traits.items()
            },
            'emotional_profile': {
                'current_state': self.emotional_profile.current_state.value,
                'state_intensity': self.emotional_profile.state_intensity,
                'emotional_stability': self.emotional_profile.emotional_stability,
                'dominant_emotions': [e.value for e in self.emotional_profile.dominant_emotions]
            },
            'self_perception': self.self_perception,
            'communication_style': self.communication_style
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityTraits':
        """Create from dictionary"""
        traits_obj = cls()
        
        traits_obj.version = data.get('version', '1.0')
        traits_obj.created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        traits_obj.last_updated = datetime.fromisoformat(data.get('last_updated', datetime.now().isoformat()))
        
        # Load traits
        traits_data = data.get('traits', {})
        for dim_name, trait_data in traits_data.items():
            try:
                dimension = PersonalityDimension(dim_name)
                trait = PersonalityTrait(
                    dimension=dimension,
                    strength=trait_data.get('strength', 0.5),
                    stability=trait_data.get('stability', 0.8),
                    expression_count=trait_data.get('expression_count', 0),
                    growth_trend=trait_data.get('growth_trend', 0.0),
                    context_preferences=trait_data.get('context_preferences', [])
                )
                traits_obj.traits[dimension] = trait
            except ValueError:
                continue  # Skip invalid dimensions
        
        # Load emotional profile
        emotional_data = data.get('emotional_profile', {})
        try:
            current_state = EmotionalState(emotional_data.get('current_state', 'calm'))
            traits_obj.emotional_profile.current_state = current_state
            traits_obj.emotional_profile.state_intensity = emotional_data.get('state_intensity', 0.5)
            traits_obj.emotional_profile.emotional_stability = emotional_data.get('emotional_stability', 0.7)
            
            dominant_emotions = []
            for emotion_name in emotional_data.get('dominant_emotions', ['calm', 'curious']):
                try:
                    emotion = EmotionalState(emotion_name)
                    dominant_emotions.append(emotion)
                except ValueError:
                    continue
            traits_obj.emotional_profile.dominant_emotions = dominant_emotions
            
        except ValueError:
            pass  # Keep defaults
        
        # Load other data
        traits_obj.self_perception = data.get('self_perception', traits_obj.self_perception)
        traits_obj.communication_style = data.get('communication_style', traits_obj.communication_style)
        
        return traits_obj

# Export
__all__ = [
    'PersonalityDimension',
    'EmotionalState', 
    'PersonalityTrait',
    'EmotionalProfile',
    'PersonalityTraits'
]
