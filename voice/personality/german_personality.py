"""
🎭 GERMAN VOICE PERSONALITY
Deutsche Persönlichkeit für Kira's weibliche Stimme
"""

import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time

logger = logging.getLogger(__name__)


@dataclass
class PersonalityResponse:
    """Personality-enhanced response"""
    original_text: str
    enhanced_text: str
    emotion: str
    personality_traits: List[str]
    cultural_elements: List[str]
    confidence: float


class GermanCulturalContext:
    """German cultural context and expressions"""
    
    def __init__(self):
        # German greetings based on time of day
        self.greetings = {
            'morning': ['Guten Morgen', 'Schönen guten Morgen', 'Morgen'],
            'day': ['Guten Tag', 'Hallo', 'Tag'],
            'evening': ['Guten Abend', 'Schönen Abend'],
            'night': ['Gute Nacht', 'Schöne Nacht']
        }
        
        # Polite expressions
        self.polite_expressions = {
            'please': ['bitte', 'bitte schön'],
            'thank_you': ['danke', 'vielen Dank', 'herzlichen Dank', 'dankeschön'],
            'excuse_me': ['entschuldigung', 'verzeihung', 'tut mir leid'],
            'you_welcome': ['gern geschehen', 'bitte schön', 'keine Ursache', 'gerne']
        }
        
        # German emotional expressions
        self.emotional_expressions = {
            'joy': ['wunderbar', 'herrlich', 'toll', 'super', 'fantastisch', 'prima'],
            'surprise': ['ach so', 'tatsächlich', 'interessant', 'oh', 'aha'],
            'concern': ['oh je', 'ach du liebe Zeit', 'das ist aber schade'],
            'approval': ['genau', 'stimmt', 'richtig', 'so ist es', 'absolut'],
            'understanding': ['ich verstehe', 'ach so', 'klar', 'natürlich']
        }
        
        # Regional expressions (keeping it standard German)
        self.regional_expressions = {
            'standard': {
                'small_talk': ['wie geht es Ihnen?', 'alles in Ordnung?', 'was gibt es Neues?'],
                'comfort': ['alles wird gut', 'kopf hoch', 'das schaffen wir schon'],
                'encouragement': ['Sie schaffen das', 'nur Mut', 'das wird schon']
            }
        }
        
        # German conversation starters
        self.conversation_starters = [
            'Was kann ich für Sie tun?',
            'Womit kann ich Ihnen helfen?',
            'Was beschäftigt Sie?',
            'Wie kann ich behilflich sein?'
        ]
        
        # German filler words and connectors
        self.connectors = {
            'transition': ['also', 'nun', 'gut', 'so', 'dann'],
            'emphasis': ['durchaus', 'wirklich', 'tatsächlich', 'ziemlich'],
            'uncertainty': ['vielleicht', 'möglicherweise', 'eventuell', 'unter Umständen'],
            'certainty': ['sicher', 'bestimmt', 'definitiv', 'ganz klar']
        }
    
    def get_time_appropriate_greeting(self) -> str:
        """Get time-appropriate German greeting"""
        current_time = datetime.now().time()
        
        if time(5, 0) <= current_time < time(12, 0):
            return random.choice(self.greetings['morning'])
        elif time(12, 0) <= current_time < time(18, 0):
            return random.choice(self.greetings['day'])
        elif time(18, 0) <= current_time < time(22, 0):
            return random.choice(self.greetings['evening'])
        else:
            return random.choice(self.greetings['night'])
    
    def add_polite_expression(self, text: str, expression_type: str) -> str:
        """Add polite expression to text"""
        if expression_type in self.polite_expressions:
            expression = random.choice(self.polite_expressions[expression_type])
            return f"{expression}, {text}"
        return text
    
    def enhance_with_emotion(self, text: str, emotion: str) -> str:
        """Enhance text with German emotional expressions"""
        if emotion in self.emotional_expressions:
            expressions = self.emotional_expressions[emotion]
            if random.random() < 0.3:  # 30% chance to add expression
                expression = random.choice(expressions)
                return f"{expression}! {text}"
        return text


class GermanPersonalityEngine:
    """German personality engine for voice responses"""
    
    def __init__(self):
        self.cultural_context = GermanCulturalContext()
        
        # Personality traits for German female assistant
        self.personality_traits = {
            'friendliness': 0.85,     # Warm and welcoming
            'professionalism': 0.75,  # Professional but not cold
            'helpfulness': 0.90,      # Very helpful
            'patience': 0.80,         # Patient with users
            'humor': 0.40,            # Subtle, appropriate humor
            'empathy': 0.85,          # Understanding and caring
            'directness': 0.60,       # German directness but softened
            'formality': 0.70         # Appropriately formal
        }
        
        # Conversation styles
        self.conversation_styles = {
            'first_interaction': {
                'formality': 0.8,
                'warmth': 0.7,
                'introduction_needed': True
            },
            'regular_conversation': {
                'formality': 0.6,
                'warmth': 0.8,
                'familiarity': 0.7
            },
            'problem_solving': {
                'formality': 0.7,
                'focus': 0.9,
                'empathy': 0.8
            },
            'casual_chat': {
                'formality': 0.4,
                'warmth': 0.9,
                'humor': 0.6
            }
        }
        
        # Response patterns
        self.response_patterns = {
            'acknowledgment': [
                'Das verstehe ich.',
                'Ich kann nachvollziehen, was Sie meinen.',
                'Das ist verständlich.',
                'Ich verstehe Ihr Anliegen.'
            ],
            'assistance_offer': [
                'Gerne helfe ich Ihnen dabei.',
                'Das kann ich für Sie machen.',
                'Dabei kann ich Ihnen behilflich sein.',
                'Das lässt sich sicher lösen.'
            ],
            'clarification': [
                'Könnten Sie das näher erläutern?',
                'Was genau meinen Sie damit?',
                'Können Sie mir mehr Details geben?',
                'Das würde ich gerne besser verstehen.'
            ],
            'encouragement': [
                'Das schaffen Sie bestimmt.',
                'Lassen Sie uns das gemeinsam angehen.',
                'Das bekommen wir hin.',
                'Sie sind auf dem richtigen Weg.'
            ]
        }
    
    def enhance_response(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None,
        conversation_type: str = "regular_conversation",
        user_emotion: Optional[str] = None
    ) -> PersonalityResponse:
        """
        Enhance response with German personality
        
        Args:
            text: Original response text
            context: Conversation context
            conversation_type: Type of conversation
            user_emotion: Detected user emotion
            
        Returns:
            PersonalityResponse with enhanced text
        """
        try:
            # Initialize enhancement
            enhanced_text = text
            applied_traits = []
            cultural_elements = []
            emotion = "neutral"
            
            # Determine response emotion based on user emotion and content
            emotion = self._determine_response_emotion(text, user_emotion, context)
            
            # Apply conversation style
            style = self.conversation_styles.get(conversation_type, self.conversation_styles['regular_conversation'])
            
            # Add greeting if appropriate
            if self._should_add_greeting(context):
                greeting = self.cultural_context.get_time_appropriate_greeting()
                enhanced_text = f"{greeting}! {enhanced_text}"
                cultural_elements.append("time_appropriate_greeting")
            
            # Add politeness
            if style.get('formality', 0.5) > 0.6:
                enhanced_text = self._add_politeness(enhanced_text, style['formality'])
                applied_traits.append("politeness")
            
            # Add empathy if user seems distressed
            if user_emotion in ['sad', 'frustrated', 'angry'] and self.personality_traits['empathy'] > 0.7:
                enhanced_text = self._add_empathy(enhanced_text, user_emotion)
                applied_traits.append("empathy")
            
            # Add encouragement if appropriate
            if self._needs_encouragement(text, context):
                enhanced_text = self._add_encouragement(enhanced_text)
                applied_traits.append("encouragement")
            
            # Add cultural expressions
            enhanced_text = self.cultural_context.enhance_with_emotion(enhanced_text, emotion)
            cultural_elements.append("emotional_expression")
            
            # Apply professional touch
            if style.get('formality', 0.5) > 0.7:
                enhanced_text = self._add_professional_tone(enhanced_text)
                applied_traits.append("professionalism")
            
            # Ensure German sentence structure
            enhanced_text = self._improve_german_structure(enhanced_text)
            cultural_elements.append("german_structure")
            
            # Calculate confidence
            confidence = self._calculate_enhancement_confidence(applied_traits, cultural_elements)
            
            return PersonalityResponse(
                original_text=text,
                enhanced_text=enhanced_text,
                emotion=emotion,
                personality_traits=applied_traits,
                cultural_elements=cultural_elements,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Personality enhancement error: {e}")
            return PersonalityResponse(
                original_text=text,
                enhanced_text=text,
                emotion="neutral",
                personality_traits=[],
                cultural_elements=[],
                confidence=0.5
            )
    
    def _determine_response_emotion(self, text: str, user_emotion: Optional[str], context: Optional[Dict]) -> str:
        """Determine appropriate response emotion"""
        # Emotional mirroring with German cultural adaptation
        if user_emotion:
            if user_emotion in ['happy', 'excited']:
                return 'happy'
            elif user_emotion in ['sad', 'disappointed']:
                return 'caring'
            elif user_emotion in ['angry', 'frustrated']:
                return 'calm'
            elif user_emotion in ['confused', 'uncertain']:
                return 'helpful'
            elif user_emotion in ['surprised']:
                return 'understanding'
        
        # Content-based emotion detection
        text_lower = text.lower()
        if any(word in text_lower for word in ['toll', 'super', 'wunderbar', 'gut']):
            return 'happy'
        elif any(word in text_lower for word in ['problem', 'fehler', 'schwierig']):
            return 'helpful'
        elif any(word in text_lower for word in ['danke', 'dank']):
            return 'warm'
        
        return 'neutral'
    
    def _should_add_greeting(self, context: Optional[Dict]) -> bool:
        """Check if greeting should be added"""
        if not context:
            return False
        
        return (
            context.get('is_first_interaction', False) or
            context.get('conversation_started_recently', False) or
            context.get('long_pause_before', False)
        )
    
    def _add_politeness(self, text: str, formality_level: float) -> str:
        """Add German politeness markers"""
        if formality_level > 0.8:
            # Very formal
            if not any(formal in text.lower() for formal in ['sie', 'ihnen', 'ihr']):
                text = text.replace('du ', 'Sie ')
                text = text.replace('dir ', 'Ihnen ')
                text = text.replace('dein', 'Ihr')
        
        # Add polite phrases
        if random.random() < formality_level * 0.5:
            if text.endswith('.'):
                text = text[:-1] + ', bitte.'
        
        return text
    
    def _add_empathy(self, text: str, user_emotion: str) -> str:
        """Add empathetic responses"""
        empathy_starters = {
            'sad': ['Das tut mir leid zu hören.', 'Das verstehe ich gut.'],
            'frustrated': ['Das kann ich verstehen.', 'Das ist wirklich ärgerlich.'],
            'angry': ['Ich verstehe Ihren Ärger.', 'Das ist verständlich.']
        }
        
        if user_emotion in empathy_starters:
            starter = random.choice(empathy_starters[user_emotion])
            text = f"{starter} {text}"
        
        return text
    
    def _needs_encouragement(self, text: str, context: Optional[Dict]) -> bool:
        """Check if encouragement is needed"""
        if not context:
            return False
        
        return (
            context.get('user_struggling', False) or
            context.get('task_difficult', False) or
            any(word in text.lower() for word in ['schwierig', 'problem', 'nicht verstehen'])
        )
    
    def _add_encouragement(self, text: str) -> str:
        """Add encouraging phrases"""
        encouragement = random.choice(self.response_patterns['encouragement'])
        return f"{encouragement} {text}"
    
    def _add_professional_tone(self, text: str) -> str:
        """Add professional tone markers"""
        # Ensure proper formal address
        text = text.replace('du kannst', 'Sie können')
        text = text.replace('du solltest', 'Sie sollten')
        text = text.replace('du musst', 'Sie müssen')
        
        return text
    
    def _improve_german_structure(self, text: str) -> str:
        """Improve German sentence structure"""
        # Basic German structure improvements
        sentences = text.split('. ')
        improved_sentences = []
        
        for sentence in sentences:
            # Ensure proper punctuation
            if sentence and not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            # Basic word order improvements (simplified)
            if sentence.startswith('Ich kann'):
                sentence = sentence.replace('Ich kann', 'Gerne kann ich')
            
            improved_sentences.append(sentence)
        
        return ' '.join(improved_sentences)
    
    def _calculate_enhancement_confidence(self, traits: List[str], cultural_elements: List[str]) -> float:
        """Calculate confidence in personality enhancement"""
        base_confidence = 0.7
        
        # Increase confidence based on applied enhancements
        enhancement_bonus = len(traits) * 0.05 + len(cultural_elements) * 0.03
        
        return min(1.0, base_confidence + enhancement_bonus)
    
    def get_conversation_starter(self, context: Optional[Dict] = None) -> str:
        """Get appropriate German conversation starter"""
        if context and context.get('returning_user', False):
            starters = [
                'Schön, Sie wieder zu sehen!',
                'Freut mich, dass Sie wieder da sind.',
                'Willkommen zurück!'
            ]
        else:
            starters = self.cultural_context.conversation_starters
        
        return random.choice(starters)
    
    def get_status(self) -> Dict[str, Any]:
        """Get personality engine status"""
        return {
            'personality_traits': self.personality_traits,
            'conversation_styles': list(self.conversation_styles.keys()),
            'cultural_features': {
                'greetings': len(self.cultural_context.greetings),
                'polite_expressions': len(self.cultural_context.polite_expressions),
                'emotional_expressions': len(self.cultural_context.emotional_expressions),
                'regional_support': list(self.cultural_context.regional_expressions.keys())
            },
            'response_patterns': list(self.response_patterns.keys())
        }


# Export classes
__all__ = [
    'GermanPersonalityEngine',
    'GermanCulturalContext',
    'PersonalityResponse'
]