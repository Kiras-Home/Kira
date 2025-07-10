"""
Personality Package - Kira AI System
"""

# Export core personality components
from .core.personality_traits import (
    PersonalityDimension,
    EmotionalState,
    PersonalityTrait,
    EmotionalProfile,
    PersonalityTraits
)

from .core.kira_personality import (
    KiraIdentity,
    PersonalityResponse,
    LearningExperience,
    KiraPersonality
)

__all__ = [
    # Personality Traits
    'PersonalityDimension',
    'EmotionalState',
    'PersonalityTrait',
    'EmotionalProfile',
    'PersonalityTraits',

    # Kira Personality
    'KiraIdentity',
    'PersonalityResponse',
    'LearningExperience',
    'KiraPersonality'
]

__version__ = "1.0.0"