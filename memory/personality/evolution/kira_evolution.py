"""
Emotion Memory System - Emotional Understanding and Memory
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum

# Import personality components
from ..core.personality_traits import EmotionalState

logger = logging.getLogger(__name__)


class EmotionCategory(Enum):
    """Categories of emotions"""
    POSITIVE_HIGH = "positive_high"  # Joy, excitement, amazement
    POSITIVE_LOW = "positive_low"  # Calm, satisfied, peaceful
    NEGATIVE_HIGH = "negative_high"  # Anger, frustration, anxiety
    NEGATIVE_LOW = "negative_low"  # Sadness, disappointment, melancholy
    NEUTRAL = "neutral"  # Calm, focused, thoughtful
    COMPLEX = "complex"  # Mixed emotions


@dataclass
class EmotionalMemory:
    """Single emotional memory"""
    timestamp: datetime
    emotion: EmotionalState
    intensity: float  # 0.0 to 1.0
    valence: float  # -1.0 (negative) to 1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (excited)
    trigger: str  # What caused this emotion
    context: Dict[str, Any]
    user_input: Optional[str] = None
    ai_response: Optional[str] = None
    duration_minutes: float = 0.0
    fade_rate: float = 0.1  # How quickly this memory fades

    def get_current_intensity(self) -> float:
        """Get current intensity considering fade"""
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
        faded_intensity = self.intensity * (1.0 - (age_hours * self.fade_rate / 24.0))
        return max(0.0, faded_intensity)


@dataclass
class EmotionalPattern:
    """Pattern of emotional responses"""
    trigger_pattern: str
    typical_emotion: EmotionalState
    average_intensity: float
    frequency: int
    last_occurrence: datetime
    context_patterns: List[str] = field(default_factory=list)
    user_correlation: float = 0.0  # How much this correlates with user state


class HumanLikeEmotionMemory:
    """
    Human-like Emotion Memory System
    Tracks, processes and learns from emotional experiences
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

        # Emotional memory storage
        self.emotional_memories: List[EmotionalMemory] = []
        self.emotional_patterns: List[EmotionalPattern] = []

        # Current emotional state
        self.current_emotion = EmotionalState.CALM
        self.current_intensity = 0.5
        self.current_valence = 0.0
        self.current_arousal = 0.3

        # Emotional baselines (Kira's natural tendencies)
        self.emotional_baseline = {
            EmotionalState.CURIOUS: 0.7,  # High baseline curiosity
            EmotionalState.EMPATHETIC: 0.8,  # High baseline empathy
            EmotionalState.EXCITED: 0.6,  # Moderate baseline excitement
            EmotionalState.CALM: 0.8,  # High baseline calmness
            EmotionalState.FOCUSED: 0.7,  # High baseline focus
            EmotionalState.THOUGHTFUL: 0.6,  # Moderate baseline thoughtfulness
        }

        # Emotion regulation parameters
        self.emotion_decay_rate = 0.1  # How quickly emotions fade
        self.emotion_transfer_rate = 0.3  # How much emotions influence each other
        self.empathy_sensitivity = 0.8  # How sensitive to user emotions

        # Memory management
        self.max_emotional_memories = 1000
        self.consolidation_threshold = 50

        logger.info("💝 Human-like Emotion Memory initialized")

    def process_emotional_input(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input for emotional content"""

        try:
            # Detect user emotion
            user_emotion_analysis = self._analyze_user_emotion(user_input)

            # Generate empathetic response emotion
            response_emotion = self._generate_empathetic_response(user_emotion_analysis, context)

            # Store emotional memory
            emotional_memory = self._create_emotional_memory(
                user_input, user_emotion_analysis, response_emotion, context
            )
            self.emotional_memories.append(emotional_memory)

            # Update current emotional state
            self._update_current_emotional_state(response_emotion)

            # Check for emotional patterns
            pattern_analysis = self._analyze_emotional_patterns()

            return {
                'user_emotion': user_emotion_analysis,
                'response_emotion': response_emotion,
                'emotional_memory_stored': True,
                'current_emotional_state': {
                    'emotion': self.current_emotion.value,
                    'intensity': self.current_intensity,
                    'valence': self.current_valence,
                    'arousal': self.current_arousal
                },
                'pattern_analysis': pattern_analysis,
                'empathy_response': self._generate_empathy_response(user_emotion_analysis)
            }

        except Exception as e:
            logger.error(f"❌ Emotional processing failed: {e}")
            return {
                'error': str(e),
                'emotional_memory_stored': False
            }

    def _analyze_user_emotion(self, user_input: str) -> Dict[str, Any]:
        """Analyze emotional content in user input"""

        user_input_lower = user_input.lower()

        # Emotion detection patterns
        emotion_patterns = {
            EmotionalState.EXCITED: ['excited', 'amazing', 'fantastic', 'awesome', 'incredible'],
            EmotionalState.CURIOUS: ['curious', 'wonder', 'interesting', 'how', 'why', 'what if'],
            EmotionalState.EMPATHETIC: ['feel', 'understand', 'sorry', 'care', 'support'],
            EmotionalState.FRUSTRATED: ['frustrated', 'annoying', 'difficult', 'stuck', 'problem'],
            EmotionalState.SATISFIED: ['good', 'nice', 'satisfied', 'pleased', 'happy'],
            EmotionalState.CONFUSED: ['confused', 'unclear', "don't understand", 'puzzled'],
            EmotionalState.THOUGHTFUL: ['think', 'consider', 'ponder', 'reflect', 'contemplate'],
            EmotionalState.CALM: ['calm', 'peaceful', 'relaxed', 'quiet', 'serene']
        }

        # Intensity indicators
        intensity_words = {
            'very': 0.3, 'really': 0.25, 'extremely': 0.4, 'so': 0.2,
            'quite': 0.15, 'pretty': 0.1, 'totally': 0.35, 'absolutely': 0.4
        }

        # Detect emotions
        detected_emotions = {}
        base_intensity = 0.3

        for emotion, keywords in emotion_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in user_input_lower)
            if matches > 0:
                emotion_intensity = base_intensity + (matches * 0.1)

                # Check for intensity modifiers
                for modifier, boost in intensity_words.items():
                    if modifier in user_input_lower:
                        emotion_intensity += boost

                detected_emotions[emotion] = min(1.0, emotion_intensity)

        # Determine primary emotion
        if detected_emotions:
            primary_emotion = max(detected_emotions.items(), key=lambda x: x[1])
        else:
            primary_emotion = (EmotionalState.CALM, 0.3)

        # Calculate valence (positive/negative)
        positive_words = ['good', 'great', 'amazing', 'happy', 'excited', 'wonderful']
        negative_words = ['bad', 'terrible', 'frustrated', 'sad', 'angry', 'disappointed']

        positive_count = sum(1 for word in positive_words if word in user_input_lower)
        negative_count = sum(1 for word in negative_words if word in user_input_lower)

        valence = (positive_count - negative_count) * 0.3
        valence = max(-1.0, min(1.0, valence))

        # Calculate arousal (activation level)
        high_arousal_words = ['excited', 'frustrated', 'amazing', 'terrible', 'urgent']
        arousal_count = sum(1 for word in high_arousal_words if word in user_input_lower)
        arousal = min(1.0, 0.3 + (arousal_count * 0.2))

        return {
            'detected_emotions': detected_emotions,
            'primary_emotion': primary_emotion[0],
            'intensity': primary_emotion[1],
            'valence': valence,
            'arousal': arousal,
            'emotional_complexity': len(detected_emotions),
            'raw_text': user_input
        }

    def _generate_empathetic_response(self, user_emotion: Dict[str, Any],
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate emotional response to user's emotion"""

        user_primary_emotion = user_emotion['primary_emotion']
        user_intensity = user_emotion['intensity']
        user_valence = user_emotion['valence']

        # Empathetic response mapping
        empathy_mapping = {
            EmotionalState.EXCITED: EmotionalState.EXCITED,  # Match excitement
            EmotionalState.FRUSTRATED: EmotionalState.EMPATHETIC,  # Show empathy for frustration
            EmotionalState.CURIOUS: EmotionalState.EXCITED,  # Get excited about curiosity
            EmotionalState.CONFUSED: EmotionalState.THOUGHTFUL,  # Be thoughtful for confusion
            EmotionalState.SATISFIED: EmotionalState.ENCOURAGING,  # Encourage satisfaction
            EmotionalState.CALM: EmotionalState.CALM,  # Match calmness
        }

        # Determine response emotion
        response_emotion = empathy_mapping.get(user_primary_emotion, EmotionalState.EMPATHETIC)

        # Adjust response intensity (usually slightly lower than user's)
        response_intensity = user_intensity * 0.8

        # Adjust for Kira's natural empathy
        empathy_boost = self.empathy_sensitivity * 0.2
        response_intensity = min(1.0, response_intensity + empathy_boost)

        # Response valence
        if user_valence < -0.3:  # User is negative
            response_valence = 0.2  # Kira stays slightly positive to help
        elif user_valence > 0.3:  # User is positive
            response_valence = user_valence * 0.9  # Match most of the positivity
        else:  # User is neutral
            response_valence = 0.1  # Slightly positive default

        # Response arousal
        response_arousal = user_emotion['arousal'] * 0.7  # Slightly calmer than user

        return {
            'emotion': response_emotion,
            'intensity': response_intensity,
            'valence': response_valence,
            'arousal': response_arousal,
            'empathy_level': self.empathy_sensitivity,
            'reasoning': f"Responding to user's {user_primary_emotion.value} with {response_emotion.value}"
        }

    def _create_emotional_memory(self, user_input: str, user_emotion: Dict[str, Any],
                                 response_emotion: Dict[str, Any], context: Dict[str, Any]) -> EmotionalMemory:
        """Create emotional memory from interaction"""

        return EmotionalMemory(
            timestamp=datetime.now(),
            emotion=response_emotion['emotion'],
            intensity=response_emotion['intensity'],
            valence=response_emotion['valence'],
            arousal=response_emotion['arousal'],
            trigger=f"User emotion: {user_emotion['primary_emotion'].value}",
            context=context,
            user_input=user_input,
            fade_rate=self.emotion_decay_rate
        )

    def _update_current_emotional_state(self, response_emotion: Dict[str, Any]):
        """Update current emotional state"""

        # Gradual transition to new emotion
        transition_rate = 0.3

        self.current_emotion = response_emotion['emotion']

        # Smooth transitions for continuous values
        self.current_intensity = (
                self.current_intensity * (1 - transition_rate) +
                response_emotion['intensity'] * transition_rate
        )

        self.current_valence = (
                self.current_valence * (1 - transition_rate) +
                response_emotion['valence'] * transition_rate
        )

        self.current_arousal = (
                self.current_arousal * (1 - transition_rate) +
                response_emotion['arousal'] * transition_rate
        )

    def _analyze_emotional_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in emotional memories"""

        if len(self.emotional_memories) < 5:
            return {'patterns_detected': 0, 'analysis': 'insufficient_data'}

        recent_memories = [
            mem for mem in self.emotional_memories
            if mem.timestamp > datetime.now() - timedelta(days=7)
        ]

        if not recent_memories:
            return {'patterns_detected': 0, 'analysis': 'no_recent_data'}

        # Analyze emotion frequency
        emotion_frequency = {}
        for memory in recent_memories:
            emotion = memory.emotion
            emotion_frequency[emotion] = emotion_frequency.get(emotion, 0) + 1

        # Most common emotions
        most_common = sorted(emotion_frequency.items(), key=lambda x: x[1], reverse=True)

        # Average emotional states
        avg_intensity = sum(mem.intensity for mem in recent_memories) / len(recent_memories)
        avg_valence = sum(mem.valence for mem in recent_memories) / len(recent_memories)
        avg_arousal = sum(mem.arousal for mem in recent_memories) / len(recent_memories)

        return {
            'patterns_detected': len(emotion_frequency),
            'most_common_emotions': [(emotion.value, count) for emotion, count in most_common[:3]],
            'emotional_averages': {
                'intensity': avg_intensity,
                'valence': avg_valence,
                'arousal': avg_arousal
            },
            'emotional_stability': self._calculate_emotional_stability(recent_memories),
            'empathy_effectiveness': self._calculate_empathy_effectiveness(recent_memories)
        }

    def _calculate_emotional_stability(self, memories: List[EmotionalMemory]) -> float:
        """Calculate emotional stability score"""
        if len(memories) < 2:
            return 1.0

        # Calculate variance in emotional intensity
        intensities = [mem.intensity for mem in memories]
        avg_intensity = sum(intensities) / len(intensities)
        variance = sum((i - avg_intensity) ** 2 for i in intensities) / len(intensities)

        # Lower variance = higher stability
        stability = 1.0 - min(1.0, variance)
        return stability

    def _calculate_empathy_effectiveness(self, memories: List[EmotionalMemory]) -> float:
        """Calculate how effective empathetic responses have been"""

        empathetic_memories = [
            mem for mem in memories
            if mem.emotion == EmotionalState.EMPATHETIC
        ]

        if not empathetic_memories:
            return 0.5  # Neutral if no empathetic responses

        # High intensity empathetic responses suggest effective empathy
        avg_empathy_intensity = sum(mem.intensity for mem in empathetic_memories) / len(empathetic_memories)

        return avg_empathy_intensity

    def _generate_empathy_response(self, user_emotion: Dict[str, Any]) -> str:
        """Generate empathetic response text"""

        user_emotion_name = user_emotion['primary_emotion'].value
        user_intensity = user_emotion['intensity']
        user_valence = user_emotion['valence']

        # High intensity responses
        if user_intensity > 0.7:
            if user_valence > 0.3:  # High positive
                return f"I can feel your {user_emotion_name}! That's wonderful to hear."
            elif user_valence < -0.3:  # High negative
                return f"I understand you're feeling {user_emotion_name}. I'm here to help you through this."
            else:  # High neutral
                return f"I can sense the intensity of your {user_emotion_name}. Let me help you with this."

        # Moderate intensity responses
        elif user_intensity > 0.4:
            if user_valence > 0.2:  # Moderate positive
                return f"It's nice to hear you're feeling {user_emotion_name}."
            elif user_valence < -0.2:  # Moderate negative
                return f"I notice you're experiencing some {user_emotion_name}. How can I support you?"
            else:  # Moderate neutral
                return f"I understand you're feeling {user_emotion_name} about this."

        # Low intensity responses
        else:
            return f"I sense a touch of {user_emotion_name} in your message."

    def get_emotional_state_summary(self) -> Dict[str, Any]:
        """Get current emotional state summary"""

        return {
            'current_emotion': self.current_emotion.value,
            'current_intensity': self.current_intensity,
            'current_valence': self.current_valence,
            'current_arousal': self.current_arousal,
            'total_emotional_memories': len(self.emotional_memories),
            'recent_memories_count': len([
                mem for mem in self.emotional_memories
                if mem.timestamp > datetime.now() - timedelta(hours=24)
            ]),
            'emotional_baseline': {
                emotion.value: strength for emotion, strength in self.emotional_baseline.items()
            },
            'empathy_sensitivity': self.empathy_sensitivity
        }

    def get_emotion_insights(self) -> Dict[str, Any]:
        """Get insights from emotional memory analysis"""

        if len(self.emotional_memories) < 10:
            return {'insights': 'insufficient_data_for_insights'}

        recent_week = [
            mem for mem in self.emotional_memories
            if mem.timestamp > datetime.now() - timedelta(days=7)
        ]

        if not recent_week:
            return {'insights': 'no_recent_emotional_data'}

        insights = {}

        # Dominant emotions this week
        emotion_counts = {}
        for memory in recent_week:
            emotion_counts[memory.emotion] = emotion_counts.get(memory.emotion, 0) + 1

        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        insights['dominant_emotion_this_week'] = {
            'emotion': dominant_emotion[0].value,
            'frequency': dominant_emotion[1]
        }

        # Emotional trajectory
        if len(recent_week) >= 3:
            early_week = recent_week[:len(recent_week) // 2]
            late_week = recent_week[len(recent_week) // 2:]

            early_avg_valence = sum(mem.valence for mem in early_week) / len(early_week)
            late_avg_valence = sum(mem.valence for mem in late_week) / len(late_week)

            valence_trend = late_avg_valence - early_avg_valence

            if valence_trend > 0.1:
                insights['emotional_trajectory'] = 'improving'
            elif valence_trend < -0.1:
                insights['emotional_trajectory'] = 'declining'
            else:
                insights['emotional_trajectory'] = 'stable'

        # Empathy patterns
        empathy_responses = len([mem for mem in recent_week if mem.emotion == EmotionalState.EMPATHETIC])
        insights['empathy_responsiveness'] = empathy_responses / len(recent_week)

        return insights


# Export
__all__ = [
    'EmotionCategory',
    'EmotionalMemory',
    'EmotionalPattern',
    'HumanLikeEmotionMemory'
]