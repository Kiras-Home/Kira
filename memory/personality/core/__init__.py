"""
Personality Core Package
"""

from .personality_traits import (
    PersonalityDimension,
    EmotionalState,
    PersonalityTrait,
    EmotionalProfile,
    PersonalityTraits
)

from .kira_personality import (
    KiraIdentity,
    PersonalityResponse,
    LearningExperience,
    KiraPersonality
)

__all__ = [
    'PersonalityDimension',
    'EmotionalState',
    'PersonalityTrait',
    'EmotionalProfile',
    'PersonalityTraits',
    'KiraIdentity',
    'PersonalityResponse',
    'LearningExperience',
    'KiraPersonality'
]