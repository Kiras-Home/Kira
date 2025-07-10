"""
Enhanced Memory Models - Erweiterte Datenstrukturen für intelligentes Gedächtnis
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from enum import Enum
import json
import hashlib
import logging


logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Memory Typen"""
    CONVERSATION = "conversation"
    PERSONAL_FACT = "personal_fact"      # Wer bin ich, Name, Alter, etc.
    SYSTEM_FACT = "system_fact"          # Was ist Kira, Fähigkeiten, etc.
    USER_PREFERENCE = "user_preference"   # Vorlieben, Abneigungen
    RELATIONSHIP = "relationship"         # Beziehung zu Personen
    EXPERIENCE = "experience"             # Gemeinsame Erlebnisse
    KNOWLEDGE = "knowledge"               # Allgemeinwissen
    SKILL = "skill"                       # Erlernte Fähigkeiten
    EMOTION = "emotion"                   # Emotionale Memories
    SMART_HOME = "smart_home"             # Smart Home Kontext
    LEARNING = "learning"                 # Lern-Feedback

class ImportanceLevel(Enum):
    """Wichtigkeits-Level"""
    TRIVIAL = 1      # Unwichtig, schnell vergessen
    LOW = 3          # Wenig wichtig
    MEDIUM = 5       # Standard-Wichtigkeit
    HIGH = 7         # Wichtig, behalten
    CRITICAL = 9     # Sehr wichtig, niemals vergessen
    CORE = 10        # Kern-Identität, permanent

class EmotionType(Enum):
    """Basis-Emotionstypen für Memory Models"""
    
    # Primäre Emotionen
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    
    # Sekundäre Emotionen
    JOY = "joy"
    EXCITEMENT = "excitement"
    CONTENTMENT = "contentment"
    GRATITUDE = "gratitude"
    PRIDE = "pride"
    LOVE = "love"
    EMPATHY = "empathy"
    COMPASSION = "compassion"
    
    # Negative Emotionen
    FRUSTRATION = "frustration"
    ANXIETY = "anxiety"
    WORRY = "worry"
    DISAPPOINTMENT = "disappointment"
    JEALOUSY = "jealousy"
    GUILT = "guilt"
    SHAME = "shame"
    LONELINESS = "loneliness"
    
    # Neutrale/Gemischte
    CURIOSITY = "curiosity"
    CONFUSION = "confusion"
    BOREDOM = "boredom"
    NEUTRAL = "neutral"
    AMBIVALENT = "ambivalent"

@dataclass
class EmojiEmotionResult:
    """Result für Emoji Emotion Prediction"""
    primary_emotion: EmotionType
    confidence: float
    secondary_emotions: Dict[EmotionType, float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.secondary_emotions is None:
            self.secondary_emotions = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass
class EmotionPrediction:
    """Result für Text Emotion Prediction"""
    primary_emotion: EmotionType
    confidence: float
    emotion_distribution: Dict[EmotionType, float] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.emotion_distribution is None:
            self.emotion_distribution = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass 
class TrainingPerformance:
    """Training Performance Tracking"""
    model_name: str
    accuracy: float
    training_samples: int
    validation_samples: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class TrainingConfig:
    """Training Configuration"""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "emoji_confidence_threshold": 0.5,
            "text_confidence_threshold": 0.5,
            "max_training_samples": 10000,
            "validation_split": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
    
class EmotionMappings:
    """Vordefinierte Emotion Mappings"""
    
    EMOJI_TO_EMOTION = {
        # Happiness
        "😀": (EmotionType.HAPPINESS, 0.9),
        "😁": (EmotionType.HAPPINESS, 0.9),
        "😂": (EmotionType.JOY, 0.9),
        "🤣": (EmotionType.JOY, 0.9),
        "😃": (EmotionType.HAPPINESS, 0.8),
        "😄": (EmotionType.HAPPINESS, 0.8),
        "😅": (EmotionType.HAPPINESS, 0.7),
        "😆": (EmotionType.JOY, 0.8),
        "😊": (EmotionType.HAPPINESS, 0.9),
        "☺️": (EmotionType.HAPPINESS, 0.8),
        "🙂": (EmotionType.HAPPINESS, 0.7),
        "😇": (EmotionType.HAPPINESS, 0.8),
        
        # Love
        "🥰": (EmotionType.LOVE, 0.9),
        "😍": (EmotionType.LOVE, 0.9),
        "🤩": (EmotionType.EXCITEMENT, 0.8),
        "😘": (EmotionType.LOVE, 0.8),
        "😗": (EmotionType.LOVE, 0.7),
        "😙": (EmotionType.LOVE, 0.7),
        "😚": (EmotionType.LOVE, 0.7),
        "❤️": (EmotionType.LOVE, 0.9),
        "💕": (EmotionType.LOVE, 0.8),
        "💖": (EmotionType.LOVE, 0.8),
        "💗": (EmotionType.LOVE, 0.8),
        "💘": (EmotionType.LOVE, 0.8),
        "💝": (EmotionType.LOVE, 0.7),
        "💞": (EmotionType.LOVE, 0.8),
        "💟": (EmotionType.LOVE, 0.7),
        "❣️": (EmotionType.LOVE, 0.8),
        "💋": (EmotionType.LOVE, 0.7),
        
        # Sadness
        "😢": (EmotionType.SADNESS, 0.9),
        "😭": (EmotionType.SADNESS, 0.9),
        "😿": (EmotionType.SADNESS, 0.8),
        "😾": (EmotionType.SADNESS, 0.7),
        "😞": (EmotionType.SADNESS, 0.8),
        "😔": (EmotionType.SADNESS, 0.8),
        "😟": (EmotionType.SADNESS, 0.7),
        "🙁": (EmotionType.SADNESS, 0.7),
        "☹️": (EmotionType.SADNESS, 0.7),
        
        # Anger
        "😠": (EmotionType.ANGER, 0.9),
        "😡": (EmotionType.ANGER, 0.9),
        "🤬": (EmotionType.ANGER, 0.9),
        "😤": (EmotionType.FRUSTRATION, 0.8),
        "💢": (EmotionType.ANGER, 0.8),
        "👿": (EmotionType.ANGER, 0.8),
        "😈": (EmotionType.ANGER, 0.7),
        
        # Fear
        "😰": (EmotionType.FEAR, 0.8),
        "😨": (EmotionType.FEAR, 0.9),
        "😱": (EmotionType.FEAR, 0.9),
        "🙀": (EmotionType.FEAR, 0.8),
        "😧": (EmotionType.ANXIETY, 0.7),
        "😦": (EmotionType.FEAR, 0.6),
        "😮": (EmotionType.SURPRISE, 0.6),
        
        # Surprise
        "😲": (EmotionType.SURPRISE, 0.9),
        "😯": (EmotionType.SURPRISE, 0.8),
        "🤯": (EmotionType.SURPRISE, 0.9),
        "😵": (EmotionType.SURPRISE, 0.7),
        
        # Disgust
        "🤢": (EmotionType.DISGUST, 0.9),
        "🤮": (EmotionType.DISGUST, 0.9),
        "😖": (EmotionType.DISGUST, 0.7),
        "😣": (EmotionType.FRUSTRATION, 0.7),
        "😫": (EmotionType.FRUSTRATION, 0.8),
        "🙄": (EmotionType.BOREDOM, 0.7),
        
        # Neutral
        "😐": (EmotionType.NEUTRAL, 0.8),
        "😑": (EmotionType.NEUTRAL, 0.8),
        "😶": (EmotionType.NEUTRAL, 0.7),
        "😏": (EmotionType.NEUTRAL, 0.6),
        
        # Celebration/Excitement
        "🎉": (EmotionType.EXCITEMENT, 0.9),
        "🎊": (EmotionType.EXCITEMENT, 0.8),
        "🥳": (EmotionType.EXCITEMENT, 0.9),
        "🎈": (EmotionType.JOY, 0.7),
        
        # Thumbs up/down
        "👍": (EmotionType.HAPPINESS, 0.7),
        "👎": (EmotionType.SADNESS, 0.6),
        "👌": (EmotionType.HAPPINESS, 0.6),
        "✌️": (EmotionType.HAPPINESS, 0.6),
        "🤟": (EmotionType.LOVE, 0.7),
        "🤘": (EmotionType.EXCITEMENT, 0.7),
        
        # Thinking/Curious
        "🤔": (EmotionType.CURIOSITY, 0.8),
        "🧐": (EmotionType.CURIOSITY, 0.7),
        "🤓": (EmotionType.PRIDE, 0.6),
        
        # Grateful
        "🙏": (EmotionType.GRATITUDE, 0.8),
        "🤲": (EmotionType.GRATITUDE, 0.7),
        
        # Sleepy/Tired
        "😴": (EmotionType.BOREDOM, 0.6),
        "🥱": (EmotionType.BOREDOM, 0.7),
        "😪": (EmotionType.SADNESS, 0.5),
    }
    
    EMOTION_KEYWORDS = {
        EmotionType.HAPPINESS: [
            "happy", "glad", "joyful", "pleased", "delighted", "cheerful", "content", 
            "glücklich", "froh", "fröhlich", "zufrieden", "erfreut", "heiter"
        ],
        EmotionType.SADNESS: [
            "sad", "unhappy", "depressed", "down", "blue", "melancholy", "gloomy",
            "traurig", "niedergeschlagen", "deprimiert", "betrübt", "schwermütig"
        ],
        EmotionType.ANGER: [
            "angry", "mad", "furious", "irritated", "annoyed", "frustrated", "rage",
            "wütend", "sauer", "verärgert", "zornig", "erbost", "aufgebracht"
        ],
        EmotionType.FEAR: [
            "afraid", "scared", "frightened", "anxious", "worried", "nervous", "terrified",
            "ängstlich", "besorgt", "verängstigt", "bange", "beunruhigt"
        ],
        EmotionType.SURPRISE: [
            "surprised", "shocked", "amazed", "astonished", "stunned", "bewildered",
            "überrascht", "erstaunt", "verblüfft", "verdutzt", "perplex"
        ],
        EmotionType.LOVE: [
            "love", "adore", "cherish", "affection", "devoted", "care", "fond",
            "liebe", "lieben", "mögen", "vergöttern", "verehren", "schätzen"
        ],
        EmotionType.JOY: [
            "joy", "elated", "ecstatic", "thrilled", "overjoyed", "blissful",
            "freude", "jubel", "entzückt", "begeistert", "überwältigt"
        ],
        EmotionType.EXCITEMENT: [
            "excited", "thrilled", "elated", "enthusiastic", "pumped", "energetic",
            "aufgeregt", "begeistert", "enthusiastisch", "elektrisiert"
        ],
        EmotionType.GRATITUDE: [
            "grateful", "thankful", "appreciative", "blessed", "acknowledge",
            "dankbar", "erkenntlich", "verbunden", "zu schätzen wissen"
        ],
        EmotionType.CURIOSITY: [
            "curious", "interested", "wondering", "intrigued", "fascinated",
            "neugierig", "interessiert", "wissbegierig", "fasziniert"
        ],
        EmotionType.EMPATHY: [
            "empathy", "compassionate", "understanding", "sympathetic", "caring",
            "empathie", "mitfühlend", "verständnisvoll", "teilnehmend"
        ],
        EmotionType.PRIDE: [
            "proud", "accomplished", "achievement", "success", "satisfied",
            "stolz", "erfolgreich", "zufrieden", "errungen"
        ],
        EmotionType.SHAME: [
            "ashamed", "embarrassed", "humiliated", "mortified", "disgraced",
            "beschämt", "peinlich", "verlegen", "gedemütigt"
        ],
        EmotionType.GUILT: [
            "guilty", "remorseful", "regret", "sorry", "contrite",
            "schuldig", "reumütig", "bereuen", "bedauern"
        ],
        EmotionType.ANXIETY: [
            "anxious", "worried", "nervous", "uneasy", "apprehensive", "tense",
            "ängstlich", "besorgt", "nervös", "unruhig", "angespannt"
        ],
        EmotionType.FRUSTRATION: [
            "frustrated", "annoyed", "irritated", "exasperated", "aggravated",
            "frustriert", "genervt", "gereizt", "verärgert"
        ],
        EmotionType.CONFUSION: [
            "confused", "puzzled", "unclear", "bewildered", "perplexed",
            "verwirrt", "ratlos", "perplex", "durcheinander"
        ],
        EmotionType.BOREDOM: [
            "bored", "boring", "dull", "tedious", "uninteresting", "monotonous",
            "langweilig", "öde", "eintönig", "fade"
        ],
        EmotionType.NEUTRAL: [
            "neutral", "calm", "balanced", "stable", "ordinary", "normal",
            "neutral", "ruhig", "ausgeglichen", "normal", "gewöhnlich"
        ]
    }

class EmojiProcessor:
    """Emoji Processing Component"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.emoji_mappings = EmotionMappings.EMOJI_TO_EMOTION.copy()
        
    def analyze_emoji(self, emoji: str) -> Dict[str, Any]:
        """Analyze emoji for emotion"""
        if emoji in self.emoji_mappings:
            emotion, confidence = self.emoji_mappings[emoji]
            return {
                "primary_emotion": emotion.value,
                "confidence": confidence,
                "source": "emoji_mapping"
            }
        return None

class TextAnalyzer:
    """Text Analysis Component"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.keyword_mappings = EmotionMappings.EMOTION_KEYWORDS.copy()
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for emotion"""
        text_lower = text.lower()
        emotion_scores = defaultdict(float)
        
        for emotion, keywords in self.keyword_mappings.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            
            if score > 0:
                confidence = min(0.9, score * 0.2 + 0.3)
                emotion_scores[emotion] = confidence
        
        if emotion_scores:
            primary_emotion = max(emotion_scores.keys(), key=emotion_scores.get)
            return {
                "primary_emotion": primary_emotion.value,
                "confidence": emotion_scores[primary_emotion],
                "emotion_distribution": {e.value: s for e, s in emotion_scores.items()},
                "source": "text_analysis"
            }
        
        return {
            "primary_emotion": EmotionType.NEUTRAL.value,
            "confidence": 0.5,
            "source": "text_fallback"
        }



@dataclass
class PersonProfile:
    """Profil einer bekannten Person"""
    user_id: str
    name: Optional[str] = None
    relationship_type: str = "acquaintance"  # friend, family, colleague, etc.
    first_met: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0
    personality_traits: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    facts: List[str] = field(default_factory=list)
    emotional_pattern: Dict[str, float] = field(default_factory=dict)
    trust_level: float = 0.5  # 0-1
    familiarity_score: float = 0.0  # Wie gut kennt Kira diese Person
    
    # Erweiterte Felder
    conversation_topics: List[str] = field(default_factory=list)
    communication_style: str = "neutral"  # formal, casual, friendly, etc.
    timezone: Optional[str] = None
    preferred_language: str = "de"
    context_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def update_interaction(self):
        """Update bei neuer Interaktion"""
        self.last_interaction = datetime.now()
        self.interaction_count += 1
        
        # Erhöhe Vertrautheit basierend auf Interaktionen
        interaction_bonus = min(0.1, self.interaction_count * 0.01)
        self.familiarity_score = min(1.0, self.familiarity_score + interaction_bonus)
    
    def add_personality_trait(self, trait: str):
        """Fügt Persönlichkeitsmerkmal hinzu"""
        if trait not in self.personality_traits:
            self.personality_traits.append(trait)
    
    def update_emotional_pattern(self, emotion: str, intensity: float):
        """Updated emotionales Muster"""
        if emotion in self.emotional_pattern:
            # Gewichteter Durchschnitt
            current = self.emotional_pattern[emotion]
            self.emotional_pattern[emotion] = (current * 0.8) + (intensity * 0.2)
        else:
            self.emotional_pattern[emotion] = intensity
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für Serialisierung"""
        return {
            'user_id': self.user_id,
            'name': self.name,
            'relationship_type': self.relationship_type,
            'first_met': self.first_met.isoformat(),
            'last_interaction': self.last_interaction.isoformat(),
            'interaction_count': self.interaction_count,
            'personality_traits': self.personality_traits,
            'preferences': self.preferences,
            'facts': self.facts,
            'emotional_pattern': self.emotional_pattern,
            'trust_level': self.trust_level,
            'familiarity_score': self.familiarity_score,
            'conversation_topics': self.conversation_topics,
            'communication_style': self.communication_style,
            'timezone': self.timezone,
            'preferred_language': self.preferred_language,
            'context_preferences': self.context_preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonProfile':
        """Erstellt PersonProfile aus Dictionary"""
        # Parse datetime fields
        first_met = datetime.fromisoformat(data.get('first_met', datetime.now().isoformat()))
        last_interaction = datetime.fromisoformat(data.get('last_interaction', datetime.now().isoformat()))
        
        return cls(
            user_id=data['user_id'],
            name=data.get('name'),
            relationship_type=data.get('relationship_type', 'acquaintance'),
            first_met=first_met,
            last_interaction=last_interaction,
            interaction_count=data.get('interaction_count', 0),
            personality_traits=data.get('personality_traits', []),
            preferences=data.get('preferences', {}),
            facts=data.get('facts', []),
            emotional_pattern=data.get('emotional_pattern', {}),
            trust_level=data.get('trust_level', 0.5),
            familiarity_score=data.get('familiarity_score', 0.0),
            conversation_topics=data.get('conversation_topics', []),
            communication_style=data.get('communication_style', 'neutral'),
            timezone=data.get('timezone'),
            preferred_language=data.get('preferred_language', 'de'),
            context_preferences=data.get('context_preferences', {})
        )

@dataclass
class MemoryEntry:
    """Erweiterte Memory Entry"""
    id: Optional[int] = None
    session_id: str = "main"
    user_id: str = "default"
    memory_type: MemoryType = MemoryType.CONVERSATION
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    content_hash: str = ""
    
    # Enhanced fields
    emotional_weight: float = 0.5  # Emotionale Bedeutung
    learning_context: str = ""     # Wie wurde das gelernt
    related_memories: List[int] = field(default_factory=list)  # Verknüpfte Memories
    consolidation_score: float = 0.0  # Wahrscheinlichkeit für Langzeit-Konsolidierung
    person_context: Optional[str] = None  # Zu welcher Person gehört das
    
    # Neue Enhanced Felder
    user_context: str = ""                    # User-spezifischer Kontext
    conversation_context: str = ""            # Conversation Flow Kontext
    emotion_type: Optional[str] = None        # Assoziierte Emotion
    emotion_intensity: float = 0.5            # Emotionsintensität
    learning_weight: float = 1.0              # Lern-Gewichtung
    device_context: str = ""                  # Smart Home Device Kontext
    intent_detected: str = ""                 # Erkannte Absicht
    confidence_score: float = 1.0             # Vertrauenswürdigkeit
    voice_context: str = ""                   # Stimm-/Audio-Kontext
    personality_aspect: str = ""              # Bezug zu Persönlichkeit
    source_system: str = "general"            # Quell-System
    memory_category: str = ""                 # Memory-Kategorie
    semantic_vector: Optional[str] = None     # Semantischer Vektor (JSON)
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Generiere Content Hash falls nicht vorhanden
        if not self.content_hash:
            self.content_hash = self._generate_content_hash()
        
        # Setze Standard-Expires falls nicht definiert
        if self.expires_at is None and self.importance in [ImportanceLevel.TRIVIAL, ImportanceLevel.LOW]:
            # Trivial: 1 Tag, Low: 7 Tage
            days = 1 if self.importance == ImportanceLevel.TRIVIAL else 7
            self.expires_at = self.created_at + timedelta(days=days)
    
    def _generate_content_hash(self) -> str:
        """Generiert Hash für Content-Duplikat-Erkennung"""
        content_for_hash = f"{self.user_id}:{self.content}:{self.memory_type.value}"
        return hashlib.md5(content_for_hash.encode()).hexdigest()

    def age_in_days(self) -> int:
        """Alter des Memory in Tagen"""
        return (datetime.now() - self.created_at).days
    
    def age_in_hours(self) -> float:
        """Alter des Memory in Stunden"""
        return (datetime.now() - self.created_at).total_seconds() / 3600
    
    def is_expired(self) -> bool:
        """Prüft ob Memory abgelaufen ist"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def is_recent(self, hours: int = 24) -> bool:
        """Prüft ob Memory recent ist"""
        return self.age_in_hours() <= hours
    
    def calculate_retention_strength(self) -> float:
        """Berechnet Behaltensstärke (0-1)"""
        base_strength = self.importance.value / 10.0
        access_bonus = min(self.access_count * 0.1, 0.5)
        age_penalty = max(0, self.age_in_days() * 0.01)
        emotional_bonus = self.emotional_weight * 0.3
        learning_bonus = (self.learning_weight - 1.0) * 0.2
        confidence_factor = self.confidence_score * 0.1
        
        total_strength = (
            base_strength + 
            access_bonus + 
            emotional_bonus + 
            learning_bonus + 
            confidence_factor - 
            age_penalty
        )
        
        return max(0, min(1, total_strength))
    
    def calculate_consolidation_score(self) -> float:
        """Berechnet Konsolidierungs-Score für STM->LTM Transfer"""
        # Basis-Score auf Wichtigkeit
        base_score = self.importance.value / 10.0
        
        # Access-basierter Bonus
        access_factor = min(1.0, self.access_count / 5.0) * 0.3
        
        # Emotionaler Faktor
        emotion_factor = self.emotional_weight * 0.2
        
        # Lern-Gewichtung
        learning_factor = (self.learning_weight - 1.0) * 0.2
        
        # Zeit-Faktor (neuere Memories sind relevanter)
        age_factor = max(0, 1.0 - (self.age_in_hours() / 168))  # 1 Woche Normalisation
        age_factor *= 0.1
        
        # Confidence-Faktor
        confidence_factor = self.confidence_score * 0.2
        
        consolidation_score = (
            base_score + 
            access_factor + 
            emotion_factor + 
            learning_factor + 
            age_factor + 
            confidence_factor
        )
        
        self.consolidation_score = max(0, min(1, consolidation_score))
        return self.consolidation_score
    
    def should_consolidate(self, threshold: float = 0.7) -> bool:
        """Prüft ob Memory konsolidiert werden sollte"""
        if self.consolidation_score == 0:
            self.calculate_consolidation_score()
        
        return (
            self.consolidation_score >= threshold or
            self.importance.value >= 7 or
            self.access_count >= 3 or
            self.emotional_weight > 0.8
        )
    
    def access(self):
        """Registriert Memory-Access"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        
        # Erhöhe Emotional Weight bei häufigem Zugriff
        if self.access_count > 1:
            access_boost = min(0.1, self.access_count * 0.02)
            self.emotional_weight = min(1.0, self.emotional_weight + access_boost)
    
    def add_tag(self, tag: str):
        """Fügt Tag hinzu (falls nicht vorhanden)"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str):
        """Entfernt Tag"""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def update_metadata(self, key: str, value: Any):
        """Updated Metadata-Eintrag"""
        self.metadata[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für DB-Storage"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'memory_type': self.memory_type.value,
            'content': self.content,
            'metadata': json.dumps(self.metadata),
            'importance': self.importance.value,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'tags': ','.join(self.tags),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'content_hash': self.content_hash,
            'emotional_weight': self.emotional_weight,
            'learning_context': self.learning_context,
            'related_memories': json.dumps(self.related_memories),
            'consolidation_score': self.consolidation_score,
            'person_context': self.person_context,
            'user_context': self.user_context,
            'conversation_context': self.conversation_context,
            'emotion_type': self.emotion_type,
            'emotion_intensity': self.emotion_intensity,
            'learning_weight': self.learning_weight,
            'device_context': self.device_context,
            'intent_detected': self.intent_detected,
            'confidence_score': self.confidence_score,
            'voice_context': self.voice_context,
            'personality_aspect': self.personality_aspect,
            'source_system': self.source_system,
            'memory_category': self.memory_category,
            'semantic_vector': self.semantic_vector
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Erstellt MemoryEntry aus Dictionary"""
        # Parse JSON fields
        metadata = json.loads(data.get('metadata', '{}')) if data.get('metadata') else {}
        related_memories = json.loads(data.get('related_memories', '[]')) if data.get('related_memories') else []
        
        # Parse datetime fields
        created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        last_accessed = datetime.fromisoformat(data.get('last_accessed', datetime.now().isoformat()))
        expires_at = datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None
        
        # Parse enum fields
        memory_type = MemoryType(data.get('memory_type', MemoryType.CONVERSATION.value))
        importance = ImportanceLevel(data.get('importance', ImportanceLevel.MEDIUM.value))
        
        # Parse tags
        tags = data.get('tags', '').split(',') if data.get('tags') else []
        tags = [tag.strip() for tag in tags if tag.strip()]
        
        return cls(
            id=data.get('id'),
            session_id=data.get('session_id', 'main'),
            user_id=data.get('user_id', 'default'),
            memory_type=memory_type,
            content=data.get('content', ''),
            metadata=metadata,
            importance=importance,
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=data.get('access_count', 0),
            tags=tags,
            expires_at=expires_at,
            content_hash=data.get('content_hash', ''),
            emotional_weight=data.get('emotional_weight', 0.5),
            learning_context=data.get('learning_context', ''),
            related_memories=related_memories,
            consolidation_score=data.get('consolidation_score', 0.0),
            person_context=data.get('person_context'),
            user_context=data.get('user_context', ''),
            conversation_context=data.get('conversation_context', ''),
            emotion_type=data.get('emotion_type'),
            emotion_intensity=data.get('emotion_intensity', 0.5),
            learning_weight=data.get('learning_weight', 1.0),
            device_context=data.get('device_context', ''),
            intent_detected=data.get('intent_detected', ''),
            confidence_score=data.get('confidence_score', 1.0),
            voice_context=data.get('voice_context', ''),
            personality_aspect=data.get('personality_aspect', ''),
            source_system=data.get('source_system', 'general'),
            memory_category=data.get('memory_category', ''),
            semantic_vector=data.get('semantic_vector')
        )

@dataclass
class ConversationContext:
    """Kontext für Conversation Flow"""
    session_id: str
    user_id: str
    conversation_turn: int
    user_input: str
    ai_response: str
    intent_detected: str = ""
    entities_extracted: Dict[str, Any] = field(default_factory=dict)
    emotion_flow: Dict[str, float] = field(default_factory=dict)
    context_carried: Dict[str, Any] = field(default_factory=dict)
    response_quality_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'conversation_turn': self.conversation_turn,
            'user_input': self.user_input,
            'ai_response': self.ai_response,
            'intent_detected': self.intent_detected,
            'entities_extracted': json.dumps(self.entities_extracted),
            'emotion_flow': json.dumps(self.emotion_flow),
            'context_carried': json.dumps(self.context_carried),
            'response_quality_score': self.response_quality_score,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class SmartHomeContext:
    """Smart Home Kontext für Memory"""
    device_id: str
    device_type: str
    action_performed: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 1.0
    user_satisfaction: float = 0.5
    learning_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'action_performed': self.action_performed,
            'context_data': json.dumps(self.context_data),
            'success_rate': self.success_rate,
            'user_satisfaction': self.user_satisfaction,
            'learning_data': json.dumps(self.learning_data),
            'created_at': self.created_at.isoformat()
        }

@dataclass
class LearningFeedback:
    """Learning Feedback für Memory System"""
    memory_id: Optional[int]
    user_id: str
    feedback_type: str  # positive, negative, correction, enhancement
    feedback_value: float  # -1 bis 1
    feedback_text: str = ""
    improvement_suggestion: str = ""
    applied: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    
    def apply_feedback(self):
        """Markiert Feedback als angewendet"""
        self.applied = True
        self.applied_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            'memory_id': self.memory_id,
            'user_id': self.user_id,
            'feedback_type': self.feedback_type,
            'feedback_value': self.feedback_value,
            'feedback_text': self.feedback_text,
            'improvement_suggestion': self.improvement_suggestion,
            'applied': self.applied,
            'created_at': self.created_at.isoformat(),
            'applied_at': self.applied_at.isoformat() if self.applied_at else None
        }

# Utility Functions für Memory Models

def create_memory_from_conversation(
    user_id: str,
    content: str,
    session_id: str = "main",
    importance: ImportanceLevel = ImportanceLevel.MEDIUM,
    emotion_context: Optional[Dict[str, Any]] = None
) -> MemoryEntry:
    """Erstellt Memory aus Conversation Content"""
    
    memory = MemoryEntry(
        session_id=session_id,
        user_id=user_id,
        memory_type=MemoryType.CONVERSATION,
        content=content,
        importance=importance
    )
    
    # Füge Emotion-Kontext hinzu falls vorhanden
    if emotion_context:
        memory.emotion_type = emotion_context.get('emotion_type')
        memory.emotion_intensity = emotion_context.get('intensity', 0.5)
        memory.emotional_weight = emotion_context.get('intensity', 0.5)
    
    return memory

def create_personal_fact_memory(
    user_id: str,
    fact: str,
    confidence: float = 1.0,
    importance: ImportanceLevel = ImportanceLevel.HIGH
) -> MemoryEntry:
    """Erstellt Personal Fact Memory"""
    
    return MemoryEntry(
        session_id="personal_facts",
        user_id=user_id,
        memory_type=MemoryType.PERSONAL_FACT,
        content=fact,
        importance=importance,
        confidence_score=confidence,
        learning_weight=1.2,  # Personal Facts sind wichtiger für Learning
        consolidation_score=0.8  # Hohe Wahrscheinlichkeit für LTM
    )

def create_preference_memory(
    user_id: str,
    preference: str,
    preference_strength: float = 0.7,
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
) -> MemoryEntry:
    """Erstellt User Preference Memory"""
    
    return MemoryEntry(
        session_id="preferences",
        user_id=user_id,
        memory_type=MemoryType.USER_PREFERENCE,
        content=preference,
        importance=importance,
        emotional_weight=preference_strength,
        learning_weight=1.1
    )

# Constants für Memory Management
DEFAULT_STM_RETENTION_HOURS = 24
DEFAULT_LTM_CONSOLIDATION_THRESHOLD = 0.7
DEFAULT_MEMORY_CLEANUP_DAYS = 30
MAX_RELATED_MEMORIES = 5
MAX_TAGS_PER_MEMORY = 10