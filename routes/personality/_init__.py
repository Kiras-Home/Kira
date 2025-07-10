"""
Kira Personality Module
Comprehensive Personality System für Trait Management und Evolution

Module:
- traits.py: Trait Analysis, Development und Growth Trends
- evolution.py: Evolution Timeline, Predictions und History
- emotions.py: Emotional State, Patterns und Intensity
- direct_integration.py: Direkte PersonalitySystem Integration
"""

from .traits import (
    analyze_personality_traits,
    calculate_trait_development,
    generate_trait_recommendations,
    assess_trait_balance
)

from .evolution import (
    generate_evolution_timeline,
    predict_personality_evolution,
    analyze_development_patterns,
    calculate_evolution_velocity
)

from .emotions import (
    analyze_emotional_state,
    track_emotional_patterns,
    calculate_emotional_intelligence,
    generate_emotional_insights
)

from .direct_integration import (
    get_direct_personality_data,
    integrate_with_personality_system,
    sync_personality_state,
    validate_personality_integration
)

__all__ = [
    # Traits
    'analyze_personality_traits',
    'calculate_trait_development',
    'generate_trait_recommendations',
    'assess_trait_balance',
    
    # Evolution
    'generate_evolution_timeline',
    'predict_personality_evolution',
    'analyze_development_patterns',
    'calculate_evolution_velocity',
    
    # Emotions
    'analyze_emotional_state',
    'track_emotional_patterns',
    'calculate_emotional_intelligence',
    'generate_emotional_insights',
    
    # Direct Integration
    'get_direct_personality_data',
    'integrate_with_personality_system',
    'sync_personality_state',
    'validate_personality_integration'
]