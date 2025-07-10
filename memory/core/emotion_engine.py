"""
🎭 EMOTION & PERSONALITY ENGINE
Erweiterte Emotionserkennung und Persönlichkeitsanalyse für Memory System
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Basis-Emotionen nach Ekman-Modell erweitert"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    LOVE = "love"
    EXCITEMENT = "excitement"
    CALM = "calm"
    CONFUSION = "confusion"
    FRUSTRATION = "frustration"
    SATISFACTION = "satisfaction"
    CURIOSITY = "curiosity"
    NEUTRAL = "neutral"

class PersonalityTrait(Enum):
    """Big Five Persönlichkeitsmerkmale + erweiterte Traits"""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"
    # Erweiterte Traits
    CREATIVITY = "creativity"
    INTELLIGENCE = "intelligence"
    HUMOR = "humor"
    EMPATHY = "empathy"
    LEADERSHIP = "leadership"

@dataclass
class EmotionAnalysis:
    """Emotion Analysis Result"""
    primary_emotion: EmotionType
    emotion_intensity: float  # 0.0 - 1.0
    emotion_confidence: float  # 0.0 - 1.0
    secondary_emotions: List[Tuple[EmotionType, float]]
    emotion_triggers: List[str]
    context_factors: Dict[str, Any]
    timestamp: datetime

@dataclass
class PersonalityProfile:
    """Personality Profile"""
    user_id: str
    traits: Dict[PersonalityTrait, float]  # 0.0 - 1.0
    preferences: Dict[str, Any]
    communication_style: Dict[str, float]
    learned_patterns: List[Dict[str, Any]]
    confidence_level: float
    last_updated: datetime

class EmotionEngine:
    """
    🎭 ADVANCED EMOTION & PERSONALITY ENGINE
    Analysiert Emotionen und lernt Persönlichkeitsmuster
    """
    
    def __init__(self):
        """Initialize Emotion Engine"""
        self.emotion_patterns = self._initialize_emotion_patterns()
        self.personality_indicators = self._initialize_personality_indicators()
        self.user_profiles = {}  # user_id -> PersonalityProfile
        self.emotion_history = {}  # user_id -> List[EmotionAnalysis]
        
        # Advanced features
        self.context_weights = {
            'conversation_length': 0.1,
            'response_time': 0.05,
            'topic_complexity': 0.15,
            'question_type': 0.1,
            'emotional_words': 0.3,
            'sentence_structure': 0.2,
            'punctuation_usage': 0.1
        }
        
        # Statistics
        self.analysis_stats = {
            'total_analyses': 0,
            'emotion_detections': {},
            'personality_updates': 0,
            'confidence_improvements': 0
        }
        
        logger.info("🎭 Emotion & Personality Engine initialized")
    
    def analyze_emotion(self, 
                       text: str, 
                       user_id: str = None,
                       context: Dict[str, Any] = None) -> EmotionAnalysis:
        """
        ✅ COMPREHENSIVE EMOTION ANALYSIS
        
        Args:
            text: Text to analyze
            user_id: User identifier for personalization
            context: Additional context information
            
        Returns:
            Detailed emotion analysis
        """
        try:
            self.analysis_stats['total_analyses'] += 1
            context = context or {}
            
            # 1. Lexical Emotion Detection
            lexical_emotions = self._analyze_lexical_emotions(text)
            
            # 2. Pattern-based Emotion Detection
            pattern_emotions = self._analyze_emotion_patterns(text)
            
            # 3. Context-based Emotion Analysis
            context_emotions = self._analyze_contextual_emotions(text, context)
            
            # 4. Combine emotion signals
            combined_emotions = self._combine_emotion_signals(
                lexical_emotions, pattern_emotions, context_emotions
            )
            
            # 5. Determine primary emotion
            primary_emotion, intensity = self._determine_primary_emotion(combined_emotions)
            
            # 6. Calculate confidence
            confidence = self._calculate_emotion_confidence(
                combined_emotions, text, context
            )
            
            # 7. Find secondary emotions
            secondary_emotions = self._find_secondary_emotions(
                combined_emotions, primary_emotion
            )
            
            # 8. Identify triggers
            triggers = self._identify_emotion_triggers(text, primary_emotion)
            
            # 9. Personal context (if user known)
            if user_id and user_id in self.user_profiles:
                intensity, confidence = self._apply_personal_context(
                    user_id, primary_emotion, intensity, confidence
                )
            
            # Create analysis result
            analysis = EmotionAnalysis(
                primary_emotion=primary_emotion,
                emotion_intensity=intensity,
                emotion_confidence=confidence,
                secondary_emotions=secondary_emotions,
                emotion_triggers=triggers,
                context_factors={
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'exclamation_count': text.count('!'),
                    'question_count': text.count('?'),
                    'capital_ratio': self._calculate_capital_ratio(text),
                    'sentiment_words': self._count_sentiment_words(text),
                    **context
                },
                timestamp=datetime.now()
            )
            
            # Store in emotion history
            if user_id:
                if user_id not in self.emotion_history:
                    self.emotion_history[user_id] = []
                self.emotion_history[user_id].append(analysis)
                
                # Keep only last 100 analyses per user
                self.emotion_history[user_id] = self.emotion_history[user_id][-100:]
            
            # Update statistics
            emotion_str = primary_emotion.value
            if emotion_str not in self.analysis_stats['emotion_detections']:
                self.analysis_stats['emotion_detections'][emotion_str] = 0
            self.analysis_stats['emotion_detections'][emotion_str] += 1
            
            logger.info(f"🎭 Emotion analyzed: {primary_emotion.value} (intensity: {intensity:.2f}, confidence: {confidence:.2f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return EmotionAnalysis(
                primary_emotion=EmotionType.NEUTRAL,
                emotion_intensity=0.0,
                emotion_confidence=0.0,
                secondary_emotions=[],
                emotion_triggers=[],
                context_factors={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def analyze_personality(self, 
                           user_id: str,
                           conversation_history: List[str] = None,
                           emotion_history: List[EmotionAnalysis] = None) -> PersonalityProfile:
        """
        ✅ PERSONALITY ANALYSIS & LEARNING
        
        Args:
            user_id: User identifier
            conversation_history: Recent conversations
            emotion_history: Recent emotion analyses
            
        Returns:
            Updated personality profile
        """
        try:
            # Get existing profile or create new
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
            else:
                profile = PersonalityProfile(
                    user_id=user_id,
                    traits={trait: 0.5 for trait in PersonalityTrait},  # Start neutral
                    preferences={},
                    communication_style={},
                    learned_patterns=[],
                    confidence_level=0.0,
                    last_updated=datetime.now()
                )
            
            # Use provided or stored emotion history
            if emotion_history is None:
                emotion_history = self.emotion_history.get(user_id, [])
            
            if conversation_history is None:
                conversation_history = []
            
            # Analyze personality traits
            trait_updates = {}
            
            # 1. Openness to Experience
            trait_updates[PersonalityTrait.OPENNESS] = self._analyze_openness(
                conversation_history, emotion_history
            )
            
            # 2. Conscientiousness
            trait_updates[PersonalityTrait.CONSCIENTIOUSNESS] = self._analyze_conscientiousness(
                conversation_history, emotion_history
            )
            
            # 3. Extraversion
            trait_updates[PersonalityTrait.EXTRAVERSION] = self._analyze_extraversion(
                conversation_history, emotion_history
            )
            
            # 4. Agreeableness
            trait_updates[PersonalityTrait.AGREEABLENESS] = self._analyze_agreeableness(
                conversation_history, emotion_history
            )
            
            # 5. Neuroticism
            trait_updates[PersonalityTrait.NEUROTICISM] = self._analyze_neuroticism(
                conversation_history, emotion_history
            )
            
            # 6. Extended traits
            trait_updates[PersonalityTrait.CREATIVITY] = self._analyze_creativity(
                conversation_history, emotion_history
            )
            
            trait_updates[PersonalityTrait.EMPATHY] = self._analyze_empathy(
                conversation_history, emotion_history
            )
            
            # Update profile with weighted averaging
            for trait, new_value in trait_updates.items():
                if trait in profile.traits:
                    # Weighted average: 70% old, 30% new
                    profile.traits[trait] = (profile.traits[trait] * 0.7) + (new_value * 0.3)
                else:
                    profile.traits[trait] = new_value
            
            # Analyze communication style
            profile.communication_style = self._analyze_communication_style(
                conversation_history, emotion_history
            )
            
            # Update preferences
            profile.preferences = self._analyze_preferences(
                conversation_history, emotion_history, profile.preferences
            )
            
            # Calculate confidence level
            profile.confidence_level = self._calculate_personality_confidence(
                len(conversation_history), len(emotion_history), profile
            )
            
            # Add learned patterns
            new_patterns = self._identify_behavioral_patterns(
                conversation_history, emotion_history
            )
            profile.learned_patterns.extend(new_patterns)
            
            # Keep only recent patterns
            profile.learned_patterns = profile.learned_patterns[-20:]
            
            profile.last_updated = datetime.now()
            
            # Store updated profile
            self.user_profiles[user_id] = profile
            self.analysis_stats['personality_updates'] += 1
            
            logger.info(f"🧠 Personality profile updated for {user_id} (confidence: {profile.confidence_level:.2f})")
            
            return profile
            
        except Exception as e:
            logger.error(f"Personality analysis failed: {e}")
            return self.user_profiles.get(user_id, PersonalityProfile(
                user_id=user_id,
                traits={trait: 0.5 for trait in PersonalityTrait},
                preferences={},
                communication_style={},
                learned_patterns=[],
                confidence_level=0.0,
                last_updated=datetime.now()
            ))
    
    def get_personalized_response_style(self, user_id: str) -> Dict[str, Any]:
        """
        ✅ PERSONALIZED RESPONSE RECOMMENDATIONS
        
        Args:
            user_id: User identifier
            
        Returns:
            Recommended response style for this user
        """
        try:
            if user_id not in self.user_profiles:
                return self._get_default_response_style()
            
            profile = self.user_profiles[user_id]
            
            # Base response style on personality traits
            response_style = {
                'formality_level': 0.5,  # 0.0 = very casual, 1.0 = very formal
                'detail_level': 0.5,     # 0.0 = brief, 1.0 = detailed
                'emotional_tone': 'neutral',
                'humor_usage': 0.3,      # 0.0 = no humor, 1.0 = lots of humor
                'empathy_level': 0.7,    # 0.0 = factual, 1.0 = very empathetic
                'directness': 0.5,       # 0.0 = indirect, 1.0 = very direct
                'encouragement': 0.6,    # 0.0 = neutral, 1.0 = very encouraging
                'complexity': 0.5        # 0.0 = simple, 1.0 = complex
            }
            
            # Adjust based on personality traits
            traits = profile.traits
            
            # Openness affects complexity and detail
            if traits[PersonalityTrait.OPENNESS] > 0.7:
                response_style['complexity'] = 0.8
                response_style['detail_level'] = 0.7
            elif traits[PersonalityTrait.OPENNESS] < 0.3:
                response_style['complexity'] = 0.3
                response_style['detail_level'] = 0.4
            
            # Extraversion affects formality and humor
            if traits[PersonalityTrait.EXTRAVERSION] > 0.7:
                response_style['formality_level'] = 0.3
                response_style['humor_usage'] = 0.7
            elif traits[PersonalityTrait.EXTRAVERSION] < 0.3:
                response_style['formality_level'] = 0.7
                response_style['humor_usage'] = 0.2
            
            # Agreeableness affects empathy and encouragement
            if traits[PersonalityTrait.AGREEABLENESS] > 0.7:
                response_style['empathy_level'] = 0.9
                response_style['encouragement'] = 0.8
            elif traits[PersonalityTrait.AGREEABLENESS] < 0.3:
                response_style['empathy_level'] = 0.4
                response_style['directness'] = 0.8
            
            # Conscientiousness affects detail and structure
            if traits[PersonalityTrait.CONSCIENTIOUSNESS] > 0.7:
                response_style['detail_level'] = 0.8
                response_style['formality_level'] = 0.6
            
            # Neuroticism affects emotional tone and empathy
            if traits[PersonalityTrait.NEUROTICISM] > 0.7:
                response_style['empathy_level'] = 0.9
                response_style['encouragement'] = 0.9
                response_style['emotional_tone'] = 'supportive'
            elif traits[PersonalityTrait.NEUROTICISM] < 0.3:
                response_style['emotional_tone'] = 'confident'
            
            # Add communication preferences
            comm_style = profile.communication_style
            if 'prefers_detailed_explanations' in comm_style:
                response_style['detail_level'] = max(response_style['detail_level'], 0.7)
            if 'prefers_casual_tone' in comm_style:
                response_style['formality_level'] = min(response_style['formality_level'], 0.4)
            
            # Recent emotion context
            recent_emotions = self.emotion_history.get(user_id, [])[-5:]  # Last 5
            if recent_emotions:
                avg_intensity = sum(e.emotion_intensity for e in recent_emotions) / len(recent_emotions)
                if avg_intensity > 0.7:
                    response_style['empathy_level'] = 0.9
                    response_style['emotional_tone'] = 'caring'
            
            return {
                'response_style': response_style,
                'personality_context': {
                    'dominant_traits': self._get_dominant_traits(profile),
                    'communication_preferences': profile.communication_style,
                    'recent_emotional_state': self._get_recent_emotional_state(user_id),
                    'confidence_level': profile.confidence_level
                },
                'recommendations': self._generate_response_recommendations(response_style, profile)
            }
            
        except Exception as e:
            logger.error(f"Personalized response style failed: {e}")
            return self._get_default_response_style()
    
    # ✅ EMOTION PATTERN METHODS
    
    def _initialize_emotion_patterns(self) -> Dict[str, Any]:
        """Initialize emotion detection patterns"""
        return {
            'joy_patterns': [
                r'\b(freu[e|t]|glücklich|toll|super|großartig|wunderbar|fantastisch)\b',
                r'\b(yeah|yes|yay|hurra|juhu)\b',
                r'[!]{2,}',  # Multiple exclamation marks
                r'\b(lach[e|t]|smile|grins)\b'
            ],
            'sadness_patterns': [
                r'\b(traurig|deprimiert|niedergeschlagen|betrübt|melancholisch)\b',
                r'\b(wein[e|t]|heul[e|t])\b',
                r'\b(schlimm|schrecklich|furchtbar|schlecht)\b',
                r'\b(verlust|verloren|vermiss[e|t])\b'
            ],
            'anger_patterns': [
                r'\b(ärger[lich|n]|wütend|sauer|böse|zornig)\b',
                r'\b(verdammt|scheiße|mist|fuck)\b',
                r'\b(hass[e|t]|verabscheu[e|t])\b',
                r'[!]{3,}',  # Many exclamation marks
                r'\b(WARUM|WTF|NEIN)\b'  # Capitals indicate strong emotion
            ],
            'fear_patterns': [
                r'\b(angst|ängstlich|furcht|besorgt|nervös)\b',
                r'\b(panik|sorge|befürcht[e|et])\b',
                r'\b(schock|erschrocken|erschreckt)\b'
            ],
            'surprise_patterns': [
                r'\b(überrascht|erstaunt|verwundert|verblüfft)\b',
                r'\b(wow|omg|krass|unglaublich)\b',
                r'\b(hätte nie gedacht|unerwartet)\b'
            ],
            'love_patterns': [
                r'\b(lieb[e|t]|verliebt|romantisch|zärtlich)\b',
                r'\b(herz|hearts?|<3|♥)\b',
                r'\b(küss[e|t]|umarm[e|t])\b'
            ],
            'excitement_patterns': [
                r'\b(aufgeregt|begeistert|enthusiastisch)\b',
                r'\b(kann es kaum erwarten|so gespannt)\b',
                r'\b(energy|energie|power)\b'
            ],
            'curiosity_patterns': [
                r'\b(neugierig|interessant|faszinierend|spannend)\b',
                r'\b(wie funktioniert|warum ist|was passiert wenn)\b',
                r'\b(erkläre|erzähl|zeig mir)\b',
                r'[?]{2,}'  # Multiple question marks
            ],
            'confusion_patterns': [
                r'\b(verwirrt|durcheinander|unklar|verstehe nicht)\b',
                r'\b(häh|hä|what|wat)\b',
                r'\b(keine ahnung|weiß nicht|bin verloren)\b'
            ],
            'frustration_patterns': [
                r'\b(frustriert|genervt|gestresst)\b',
                r'\b(funktioniert nicht|klappt nicht|geht nicht)\b',
                r'\b(schon wieder|immer noch|ständig)\b'
            ]
        }
    
    def _initialize_personality_indicators(self) -> Dict[str, Any]:
        """Initialize personality trait indicators"""
        return {
            'openness_indicators': {
                'high': ['kreativ', 'künstlerisch', 'philosophie', 'reisen', 'kultur', 'experiment'],
                'low': ['routine', 'gewohnheit', 'traditional', 'bekannt', 'sicher', 'standard']
            },
            'conscientiousness_indicators': {
                'high': ['plan', 'organisation', 'struktur', 'pünktlich', 'zuverlässig', 'ordnung'],
                'low': ['spontan', 'flexibel', 'chaos', 'vergessen', 'später', 'egal']
            },
            'extraversion_indicators': {
                'high': ['party', 'menschen', 'social', 'gespräch', 'teamwork', 'laut'],
                'low': ['allein', 'ruhe', 'privat', 'introvertiert', 'leise', 'einzeln']
            },
            'agreeableness_indicators': {
                'high': ['hilfe', 'freundlich', 'kooperation', 'team', 'harmonie', 'verstehen'],
                'low': ['konkurrenz', 'kritik', 'streit', 'konflikt', 'durchsetzen', 'gewinnen']
            },
            'neuroticism_indicators': {
                'high': ['stress', 'sorge', 'nervös', 'angst', 'problem', 'schwierig'],
                'low': ['entspannt', 'gelassen', 'ruhig', 'stabil', 'sicher', 'optimistisch']
            }
        }
    
    # ✅ ANALYSIS HELPER METHODS (Simplified versions - full implementations would be much longer)
    
    def _analyze_lexical_emotions(self, text: str) -> Dict[EmotionType, float]:
        """Analyze emotions based on word content"""
        emotions = {emotion: 0.0 for emotion in EmotionType}
        text_lower = text.lower()
        
        for emotion_name, patterns in self.emotion_patterns.items():
            emotion_type = EmotionType(emotion_name.split('_')[0])
            score = 0.0
            
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches * 0.3
            
            emotions[emotion_type] = min(score, 1.0)
        
        return emotions
    
    def _analyze_emotion_patterns(self, text: str) -> Dict[EmotionType, float]:
        """Analyze emotions based on patterns"""
        emotions = {emotion: 0.0 for emotion in EmotionType}
        
        # Simple pattern-based scoring
        if '!' in text:
            emotions[EmotionType.EXCITEMENT] += 0.3
        if '?' in text:
            emotions[EmotionType.CURIOSITY] += 0.2
        if text.isupper():
            emotions[EmotionType.ANGER] += 0.4
        
        return emotions
    
    def _analyze_contextual_emotions(self, text: str, context: Dict) -> Dict[EmotionType, float]:
        """Analyze emotions based on context"""
        emotions = {emotion: 0.0 for emotion in EmotionType}
        
        # Context-based emotion inference
        if context.get('conversation_type') == 'problem_solving':
            emotions[EmotionType.FRUSTRATION] += 0.2
        if context.get('topic_category') == 'personal':
            emotions[EmotionType.LOVE] += 0.1
        
        return emotions
    
    def _get_default_response_style(self) -> Dict[str, Any]:
        """Default response style for unknown users"""
        return {
            'response_style': {
                'formality_level': 0.5,
                'detail_level': 0.6,
                'emotional_tone': 'friendly',
                'humor_usage': 0.4,
                'empathy_level': 0.7,
                'directness': 0.5,
                'encouragement': 0.6,
                'complexity': 0.5
            },
            'personality_context': {
                'dominant_traits': ['neutral'],
                'communication_preferences': {},
                'recent_emotional_state': 'unknown',
                'confidence_level': 0.0
            },
            'recommendations': [
                'Use balanced, friendly tone',
                'Provide moderate detail level',
                'Be empathetic and supportive'
            ]
        }
    
    # ✅ Placeholder implementations for complex methods
    def _combine_emotion_signals(self, lexical, pattern, context):
        """Combine different emotion signals"""
        combined = {}
        for emotion in EmotionType:
            combined[emotion] = (
                lexical.get(emotion, 0) * 0.5 +
                pattern.get(emotion, 0) * 0.3 +
                context.get(emotion, 0) * 0.2
            )
        return combined
    
    def _determine_primary_emotion(self, emotions):
        """Find primary emotion"""
        if not emotions:
            return EmotionType.NEUTRAL, 0.0
        
        primary = max(emotions.items(), key=lambda x: x[1])
        return primary[0], primary[1]
    
    def _calculate_emotion_confidence(self, emotions, text, context):
        """Calculate confidence in emotion detection"""
        max_score = max(emotions.values()) if emotions else 0
        text_length_factor = min(len(text) / 100, 1.0)
        return max_score * text_length_factor * 0.8
    
    def _find_secondary_emotions(self, emotions, primary):
        """Find secondary emotions"""
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        return [(e, s) for e, s in sorted_emotions[1:3] if s > 0.1 and e != primary]
    
    def _identify_emotion_triggers(self, text, emotion):
        """Identify what triggered the emotion"""
        # Simplified trigger identification
        words = text.lower().split()
        triggers = []
        
        if emotion == EmotionType.JOY:
            triggers = [w for w in words if w in ['toll', 'super', 'great', 'fantastic']]
        elif emotion == EmotionType.ANGER:
            triggers = [w for w in words if w in ['ärgerlich', 'wütend', 'damn', 'shit']]
        
        return triggers[:5]  # Max 5 triggers
    
    # Placeholder personality analysis methods
    def _analyze_openness(self, conversations, emotions):
        return 0.5  # Placeholder
    
    def _analyze_conscientiousness(self, conversations, emotions):
        return 0.5  # Placeholder
    
    def _analyze_extraversion(self, conversations, emotions):
        return 0.5  # Placeholder
    
    def _analyze_agreeableness(self, conversations, emotions):
        return 0.5  # Placeholder
    
    def _analyze_neuroticism(self, conversations, emotions):
        return 0.5  # Placeholder
    
    def _analyze_creativity(self, conversations, emotions):
        return 0.5  # Placeholder
    
    def _analyze_empathy(self, conversations, emotions):
        return 0.5  # Placeholder
    
    def _analyze_communication_style(self, conversations, emotions):
        return {'style': 'balanced'}
    
    def _analyze_preferences(self, conversations, emotions, existing_prefs):
        return existing_prefs
    
    def _calculate_personality_confidence(self, conv_count, emotion_count, profile):
        return min((conv_count + emotion_count) / 50, 1.0)
    
    def _identify_behavioral_patterns(self, conversations, emotions):
        return []
    
    def _apply_personal_context(self, user_id, emotion, intensity, confidence):
        return intensity, confidence
    
    def _calculate_capital_ratio(self, text):
        if not text:
            return 0.0
        return sum(1 for c in text if c.isupper()) / len(text)
    
    def _count_sentiment_words(self, text):
        positive_words = ['gut', 'toll', 'super', 'fantastisch', 'great']
        negative_words = ['schlecht', 'schlimm', 'terrible', 'awful']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        return {'positive': positive_count, 'negative': negative_count}
    
    def _get_dominant_traits(self, profile):
        sorted_traits = sorted(profile.traits.items(), key=lambda x: x[1], reverse=True)
        return [trait.value for trait, score in sorted_traits[:3] if score > 0.6]
    
    def _get_recent_emotional_state(self, user_id):
        if user_id not in self.emotion_history:
            return 'unknown'
        
        recent = self.emotion_history[user_id][-3:]  # Last 3 emotions
        if not recent:
            return 'neutral'
        
        # Most common recent emotion
        emotions = [e.primary_emotion.value for e in recent]
        return max(set(emotions), key=emotions.count)
    
    def _generate_response_recommendations(self, style, profile):
        recommendations = []
        
        if style['empathy_level'] > 0.8:
            recommendations.append("Use highly empathetic language")
        if style['humor_usage'] > 0.6:
            recommendations.append("Include appropriate humor")
        if style['detail_level'] > 0.7:
            recommendations.append("Provide detailed explanations")
        if style['formality_level'] < 0.3:
            recommendations.append("Use casual, friendly tone")
        
        return recommendations
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get emotion engine statistics"""
        return {
            'analysis_statistics': self.analysis_stats.copy(),
            'active_users': len(self.user_profiles),
            'total_emotion_history': sum(len(hist) for hist in self.emotion_history.values()),
            'supported_emotions': [e.value for e in EmotionType],
            'supported_traits': [t.value for t in PersonalityTrait],
            'features': {
                'emotion_analysis': True,
                'personality_profiling': True,
                'personalized_responses': True,
                'pattern_learning': True
            }
        }
    
class EmotionState(Enum):
    """Emotion State Enumeration"""
    NEUTRAL = "neutral"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    STORED = "stored"
    ERROR = "error"

@dataclass
class EmotionContext:
    """Emotion Analysis Context"""
    user_id: str
    session_id: str
    timestamp: datetime
    text_analyzed: str
    emotion_result: Dict[str, Any] = None
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.emotion_result is None:
            self.emotion_result = {}

# Export
__all__ = ['EmotionEngine', 'EmotionType', 'PersonalityTrait', 'EmotionAnalysis', 'PersonalityProfile']