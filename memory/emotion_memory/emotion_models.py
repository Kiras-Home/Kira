"""
Enhanced Emotion Memory - Intelligentes emotionales Gedächtnis
Versteht, speichert und lernt aus emotionalen Mustern wie ein menschliches Gehirn
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import math

from ..storage.memory_database import MemoryDatabase

logger = logging.getLogger(__name__)

class EmotionIntensity(Enum):
    """Emotions-Intensitäts-Level"""
    VERY_LOW = 0.1      # Kaum wahrnehmbar
    LOW = 0.3           # Schwach
    MODERATE = 0.5      # Mittel
    HIGH = 0.7          # Stark
    VERY_HIGH = 0.9     # Sehr stark
    EXTREME = 1.0       # Extrem

class EmotionCategory(Enum):
    """Emotions-Kategorien"""
    PRIMARY = "primary"         # Grundemotionen (Freude, Trauer, Angst, etc.)
    SECONDARY = "secondary"     # Komplexe Emotionen (Stolz, Scham, etc.)
    SOCIAL = "social"          # Soziale Emotionen (Empathie, Neid, etc.)
    COGNITIVE = "cognitive"    # Kognitive Emotionen (Interesse, Verwirrung, etc.)
    MIXED = "mixed"           # Gemischte Emotionen

@dataclass
class EmotionEntry:
    """Einzelner Emotions-Eintrag"""
    user_id: str
    emotion_type: str
    intensity: float
    context: str = ""
    triggers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Auto-generated fields
    id: Optional[int] = None
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    source: str = "user_input"
    
    # Processing fields
    category: Optional[str] = None
    related_emotions: List[str] = field(default_factory=list)
    emotional_weight: float = 0.5
    decay_rate: float = 0.9
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'emotion_type': self.emotion_type,
            'intensity': self.intensity,
            'context': self.context,
            'triggers': self.triggers,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class EmotionPattern:
    """Erkanntes Emotions-Muster"""
    pattern_type: str
    user_id: str
    emotion_type: str
    pattern_data: Dict[str, Any]
    confidence: float
    occurrences: int = 1
    
    # Timestamps
    first_detected: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_type': self.pattern_type,
            'user_id': self.user_id,
            'emotion_type': self.emotion_type,
            'pattern_data': self.pattern_data,
            'confidence': self.confidence,
            'occurrences': self.occurrences
        }

@dataclass
class EmotionState:
    """Aktueller emotionaler Zustand eines Users"""
    user_id: str
    primary_emotion: str
    intensity: float
    secondary_emotions: Dict[str, float] = field(default_factory=dict)
    mood_baseline: str = "neutral"
    emotional_stability: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_emotion(self, emotion_type: str, intensity: float, decay_factor: float = 0.8):
        """Updated emotionalen Zustand"""
        if intensity > self.intensity:
            self.primary_emotion = emotion_type
            self.intensity = intensity
        
        self.secondary_emotions[emotion_type] = intensity
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'primary_emotion': self.primary_emotion,
            'intensity': self.intensity,
            'secondary_emotions': self.secondary_emotions,
            'mood_baseline': self.mood_baseline,
            'emotional_stability': self.emotional_stability,
            'last_updated': self.last_updated.isoformat()
        }

@dataclass
class EmotionTrigger:
    """Emotion-Trigger Erkennung"""
    trigger_text: str
    emotion_type: str
    confidence: float
    context: str = ""
    trigger_pattern: str = ""
    user_specific: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trigger_text': self.trigger_text,
            'emotion_type': self.emotion_type,
            'confidence': self.confidence,
            'context': self.context,
            'trigger_pattern': self.trigger_pattern,
            'user_specific': self.user_specific
        }

class EmotionMemory:
    """Intelligentes emotionales Gedächtnis für Kira"""
    
    def __init__(self, database: MemoryDatabase):
        self.db = database
        
        # Emotion-Mapping und -Verständnis
        self.emotion_taxonomy = self._build_emotion_taxonomy()
        self.emotion_triggers = self._load_emotion_triggers()
        self.user_emotion_profiles = {}  # Cache für User-spezifische Profile
        
        # Intelligente Analyse-Parameter
        self.EMOTIONAL_MEMORY_DECAY = 0.95  # Wie Emotionen über Zeit verblassen
        self.PATTERN_DETECTION_THRESHOLD = 3  # Min. Occurrences für Pattern
        self.EMOTIONAL_CONTAGION_FACTOR = 0.3  # Wie Emotionen sich "übertragen"
        
        # Initialisiere Emotion-Tabelle falls nötig
        self._init_emotion_tables()
        
        logger.info("💝 Enhanced Emotion Memory initialisiert")
    
    def _build_emotion_taxonomy(self) -> Dict[str, Dict[str, Any]]:
        """Baut intelligente Emotions-Taxonomie auf"""
        
        return {
            # Primäre Emotionen (Basisemotionen)
            'happiness': {
                'category': EmotionCategory.PRIMARY,
                'intensity_range': (0.3, 1.0),
                'related_emotions': ['joy', 'delight', 'contentment', 'euphoria'],
                'opposite': 'sadness',
                'triggers': ['achievement', 'surprise_positive', 'social_connection'],
                'physical_markers': ['laughter', 'smiling', 'positive_language'],
                'decay_rate': 0.9  # Glück verblasst langsamer
            },
            'sadness': {
                'category': EmotionCategory.PRIMARY,
                'intensity_range': (0.2, 1.0),
                'related_emotions': ['grief', 'melancholy', 'despair', 'sorrow'],
                'opposite': 'happiness',
                'triggers': ['loss', 'disappointment', 'rejection'],
                'physical_markers': ['crying', 'negative_language', 'withdrawal'],
                'decay_rate': 0.85  # Trauer kann länger anhalten
            },
            'anger': {
                'category': EmotionCategory.PRIMARY,
                'intensity_range': (0.4, 1.0),
                'related_emotions': ['rage', 'irritation', 'frustration', 'annoyance'],
                'opposite': 'calmness',
                'triggers': ['injustice', 'blocking', 'threat', 'disrespect'],
                'physical_markers': ['exclamation', 'harsh_language', 'criticism'],
                'decay_rate': 0.8  # Wut verblasst relativ schnell
            },
            'fear': {
                'category': EmotionCategory.PRIMARY,
                'intensity_range': (0.3, 1.0),
                'related_emotions': ['anxiety', 'worry', 'panic', 'nervousness'],
                'opposite': 'confidence',
                'triggers': ['threat', 'unknown', 'uncertainty', 'danger'],
                'physical_markers': ['hesitation', 'questions', 'seeking_reassurance'],
                'decay_rate': 0.88
            },
            'surprise': {
                'category': EmotionCategory.PRIMARY,
                'intensity_range': (0.2, 0.9),
                'related_emotions': ['amazement', 'astonishment', 'shock'],
                'opposite': 'expectation',
                'triggers': ['unexpected_event', 'new_information'],
                'physical_markers': ['exclamation', 'questions', 'pause'],
                'decay_rate': 0.7  # Überraschung verblasst schnell
            },
            'disgust': {
                'category': EmotionCategory.PRIMARY,
                'intensity_range': (0.3, 1.0),
                'related_emotions': ['revulsion', 'contempt', 'aversion'],
                'opposite': 'acceptance',
                'triggers': ['unpleasant_stimulus', 'moral_violation'],
                'physical_markers': ['negative_expressions', 'rejection_words'],
                'decay_rate': 0.82
            },
            
            # Sekundäre/Komplexe Emotionen
            'pride': {
                'category': EmotionCategory.SECONDARY,
                'intensity_range': (0.4, 0.9),
                'related_emotions': ['accomplishment', 'satisfaction', 'triumph'],
                'triggers': ['achievement', 'recognition', 'success'],
                'composite_of': ['happiness', 'confidence'],
                'decay_rate': 0.92
            },
            'shame': {
                'category': EmotionCategory.SECONDARY,
                'intensity_range': (0.3, 1.0),
                'related_emotions': ['embarrassment', 'humiliation', 'guilt'],
                'triggers': ['failure', 'exposure', 'criticism'],
                'composite_of': ['sadness', 'fear'],
                'decay_rate': 0.75  # Scham kann lange anhalten
            },
            'empathy': {
                'category': EmotionCategory.SOCIAL,
                'intensity_range': (0.2, 0.8),
                'related_emotions': ['compassion', 'sympathy', 'understanding'],
                'triggers': ['others_emotion', 'shared_experience'],
                'decay_rate': 0.85
            },
            'gratitude': {
                'category': EmotionCategory.SOCIAL,
                'intensity_range': (0.3, 0.9),
                'related_emotions': ['thankfulness', 'appreciation'],
                'triggers': ['help_received', 'kindness', 'gift'],
                'composite_of': ['happiness', 'humility'],
                'decay_rate': 0.88
            },
            'curiosity': {
                'category': EmotionCategory.COGNITIVE,
                'intensity_range': (0.2, 0.8),
                'related_emotions': ['interest', 'wonder', 'fascination'],
                'triggers': ['unknown', 'mystery', 'learning_opportunity'],
                'decay_rate': 0.85
            },
            'confusion': {
                'category': EmotionCategory.COGNITIVE,
                'intensity_range': (0.2, 0.7),
                'related_emotions': ['bewilderment', 'perplexity'],
                'triggers': ['complexity', 'contradiction', 'ambiguity'],
                'decay_rate': 0.9  # Verwirrung löst sich meist schnell
            },
            
            # Neutrale/Baseline
            'neutral': {
                'category': EmotionCategory.PRIMARY,
                'intensity_range': (0.0, 0.3),
                'related_emotions': ['calm', 'balanced', 'stable'],
                'decay_rate': 1.0  # Neutral ist der Baseline
            }
        }
    
    def _load_emotion_triggers(self) -> Dict[str, List[str]]:
        """Lädt Emotion-Trigger-Patterns"""
        
        return {
            # Textuelle Trigger für Emotionserkennung
            'happiness_triggers': [
                'danke', 'toll', 'super', 'großartig', 'perfekt', 'liebe',
                'freue mich', 'bin glücklich', 'macht spaß', 'wunderbar'
            ],
            'sadness_triggers': [
                'traurig', 'schlecht', 'deprimiert', 'niedergeschlagen',
                'tut mir leid', 'schade', 'enttäuscht', 'verletzt'
            ],
            'anger_triggers': [
                'wütend', 'ärgerlich', 'sauer', 'genervt', 'frustriert',
                'unfair', 'blöd', 'dumm', 'hasse', 'furchtbar'
            ],
            'fear_triggers': [
                'angst', 'ängstlich', 'nervös', 'besorgt', 'unsicher',
                'befürchte', 'sorge mich', 'panik', 'schrecklich'
            ],
            'surprise_triggers': [
                'wow', 'überraschend', 'hätte nicht gedacht', 'krass',
                'unglaublich', 'echt?', 'wirklich?', 'amazing'
            ],
            'gratitude_triggers': [
                'dankbar', 'vielen dank', 'bin dir dankbar', 'schätze',
                'appreciate', 'hilfreich', 'freundlich'
            ],
            'curiosity_triggers': [
                'interessant', 'wie funktioniert', 'erkläre mir', 'warum',
                'wieso', 'neugierig', 'möchte wissen', 'frage mich'
            ]
        }
    
    def _init_emotion_tables(self):
        """Initialisiert spezielle Emotion-Tabellen"""
        
        try:
            with self.db.get_connection() as conn:
                # Emotions-spezifische Tabelle für bessere Performance
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS emotion_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        emotion_type TEXT NOT NULL,
                        intensity REAL NOT NULL,
                        category TEXT NOT NULL,
                        context TEXT,
                        triggers TEXT,  -- JSON array
                        related_memory_id INTEGER,
                        created_at TEXT NOT NULL,
                        expires_at TEXT,
                        confidence REAL DEFAULT 0.8,
                        source TEXT DEFAULT 'text_analysis',
                        metadata TEXT,  -- JSON
                        
                        FOREIGN KEY (related_memory_id) REFERENCES memory_entries (id)
                    )
                ''')
                
                # Emotion-Pattern Tabelle
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS emotion_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        pattern_type TEXT NOT NULL,  -- daily, weekly, situational
                        pattern_data TEXT NOT NULL,  -- JSON
                        confidence REAL NOT NULL,
                        detected_at TEXT NOT NULL,
                        last_updated TEXT NOT NULL,
                        occurrences INTEGER DEFAULT 1
                    )
                ''')
                
                # Emotion-Links (Emotionale Verknüpfungen)
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS emotion_links (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        from_emotion_id INTEGER NOT NULL,
                        to_emotion_id INTEGER NOT NULL,
                        link_type TEXT NOT NULL,  -- sequence, trigger, opposite
                        strength REAL NOT NULL,
                        created_at TEXT NOT NULL,
                        
                        FOREIGN KEY (from_emotion_id) REFERENCES emotion_entries (id),
                        FOREIGN KEY (to_emotion_id) REFERENCES emotion_entries (id)
                    )
                ''')
                
                conn.commit()
                logger.info("💾 Emotion-Tabellen initialisiert")
                
        except Exception as e:
            logger.error(f"❌ Emotion-Tabellen Init Error: {e}")
    
    def store_emotion(
        self,
        user_id: str,
        emotion_type: str,
        intensity: float = 0.5,
        context: str = "",
        memory_entry_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
        source: str = "text_analysis"
    ) -> Optional[int]:
        """Erweiterte Emotions-Speicherung mit intelligenter Analyse"""
        
        try:
            # 1. Normalisiere und validiere Emotion
            normalized_emotion = self._normalize_emotion(emotion_type)
            validated_intensity = self._validate_intensity(normalized_emotion, intensity)
            
            # 2. Bestimme Emotion-Kategorie
            emotion_info = self.emotion_taxonomy.get(normalized_emotion, {})
            category = emotion_info.get('category', EmotionCategory.PRIMARY).value
            
            # 3. Extrahiere Trigger aus Kontext
            detected_triggers = self._extract_triggers_from_context(context, normalized_emotion)
            
            # 4. Berechne Confidence basierend auf verschiedenen Faktoren
            confidence = self._calculate_emotion_confidence(
                normalized_emotion, context, detected_triggers, source
            )
            
            # 5. Bestimme Ablaufzeit basierend auf Emotion-Typ
            expires_at = self._calculate_emotion_expiry(normalized_emotion, validated_intensity)
            
            # 6. Speichere in Emotion-Tabelle
            with self.db.get_connection() as conn:
                cursor = conn.execute('''
                    INSERT INTO emotion_entries (
                        user_id, emotion_type, intensity, category, context,
                        triggers, related_memory_id, created_at, expires_at,
                        confidence, source, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    normalized_emotion,
                    validated_intensity,
                    category,
                    context,
                    json.dumps(detected_triggers),
                    memory_entry_id,
                    datetime.now().isoformat(),
                    expires_at.isoformat() if expires_at else None,
                    confidence,
                    source,
                    json.dumps(metadata or {})
                ))
                
                emotion_id = cursor.lastrowid
                conn.commit()
            
            # 7. Update User Emotion Profile
            self._update_user_emotion_profile(user_id, normalized_emotion, validated_intensity)
            
            # 8. Erkenne und speichere Patterns
            self._detect_and_store_patterns(user_id, normalized_emotion, validated_intensity)
            
            # 9. Erstelle emotionale Links
            self._create_emotion_links(emotion_id, user_id, normalized_emotion)
            
            logger.info(f"💝 Emotion gespeichert: {normalized_emotion} ({validated_intensity:.2f}) "
                       f"für {user_id} [ID: {emotion_id}]")
            
            return emotion_id
            
        except Exception as e:
            logger.error(f"❌ Emotion Storage Error: {e}")
            return None
    
    def _normalize_emotion(self, emotion_type: str) -> str:
        """Normalisiert Emotion-Type zu Standard-Kategorien"""
        
        emotion_lower = emotion_type.lower().strip()
        
        # Mapping von ähnlichen Emotionen
        emotion_mappings = {
            # Happiness family
            'glücklich': 'happiness', 'froh': 'happiness', 'fröhlich': 'happiness',
            'zufrieden': 'happiness', 'euphorie': 'happiness', 'freude': 'happiness',
            'begeistert': 'happiness', 'erfreut': 'happiness',
            
            # Sadness family
            'traurig': 'sadness', 'niedergeschlagen': 'sadness', 'deprimiert': 'sadness',
            'melancholisch': 'sadness', 'betrübt': 'sadness', 'wehmütig': 'sadness',
            
            # Anger family
            'wütend': 'anger', 'sauer': 'anger', 'ärgerlich': 'anger',
            'frustriert': 'anger', 'genervt': 'anger', 'verärgert': 'anger',
            
            # Fear family
            'ängstlich': 'fear', 'nervös': 'fear', 'besorgt': 'fear',
            'unsicher': 'fear', 'panisch': 'fear', 'angst': 'fear',
            
            # Other emotions
            'überrascht': 'surprise', 'erstaunt': 'surprise', 'verwundert': 'surprise',
            'angewidert': 'disgust', 'abgestoßen': 'disgust',
            'stolz': 'pride', 'beschämt': 'shame', 'schuldig': 'shame',
            'dankbar': 'gratitude', 'neugierig': 'curiosity',
            'verwirrt': 'confusion', 'mitfühlend': 'empathy'
        }
        
        # Direkte Mappings
        if emotion_lower in emotion_mappings:
            return emotion_mappings[emotion_lower]
        
        # Prüfe auf Teilstrings
        for key, value in emotion_mappings.items():
            if key in emotion_lower or emotion_lower in key:
                return value
        
        # Prüfe ob schon normalisiert
        if emotion_lower in self.emotion_taxonomy:
            return emotion_lower
        
        # Default: Fallback zu ähnlichster Emotion
        return self._find_closest_emotion(emotion_lower)
    
    def _find_closest_emotion(self, emotion: str) -> str:
        """Findet ähnlichste bekannte Emotion"""
        
        # Einfache String-Ähnlichkeit
        best_match = 'neutral'
        best_score = 0
        
        for known_emotion in self.emotion_taxonomy.keys():
            # Berechne Ähnlichkeit (vereinfacht)
            score = len(set(emotion) & set(known_emotion)) / len(set(emotion) | set(known_emotion))
            
            if score > best_score:
                best_score = score
                best_match = known_emotion
        
        return best_match if best_score > 0.3 else 'neutral'
    
    def _validate_intensity(self, emotion_type: str, intensity: float) -> float:
        """Validiert Intensität basierend auf Emotion-Typ"""
        
        # Klemme auf 0.0 - 1.0
        intensity = max(0.0, min(1.0, intensity))
        
        # Prüfe Emotion-spezifische Bereiche
        emotion_info = self.emotion_taxonomy.get(emotion_type, {})
        intensity_range = emotion_info.get('intensity_range', (0.0, 1.0))
        
        min_intensity, max_intensity = intensity_range
        return max(min_intensity, min(max_intensity, intensity))
    
    def _extract_triggers_from_context(self, context: str, emotion_type: str) -> List[str]:
        """Extrahiert Emotion-Trigger aus Kontext"""
        
        triggers = []
        context_lower = context.lower()
        
        # Emotion-spezifische Trigger
        trigger_key = f"{emotion_type}_triggers"
        emotion_triggers = self.emotion_triggers.get(trigger_key, [])
        
        for trigger in emotion_triggers:
            if trigger in context_lower:
                triggers.append(trigger)
        
        # Allgemeine Trigger-Patterns
        general_triggers = {
            'achievement': ['geschafft', 'erreicht', 'erfolgreich', 'gelungen'],
            'loss': ['verloren', 'weg', 'tot', 'vorbei', 'ende'],
            'surprise': ['plötzlich', 'unerwartet', 'überraschend', 'auf einmal'],
            'social_positive': ['freunde', 'zusammen', 'team', 'gemeinschaft'],
            'social_negative': ['allein', 'einsam', 'ausgeschlossen', 'ignoriert']
        }
        
        for trigger_type, keywords in general_triggers.items():
            if any(keyword in context_lower for keyword in keywords):
                triggers.append(trigger_type)
        
        return triggers
    
    def _calculate_emotion_confidence(
        self, 
        emotion_type: str, 
        context: str, 
        triggers: List[str], 
        source: str
    ) -> float:
        """Berechnet Confidence-Score für Emotion"""
        
        base_confidence = 0.5
        
        # Source-basierte Confidence
        source_bonuses = {
            'direct_input': 0.9,      # User sagt direkt "ich bin traurig"
            'text_analysis': 0.7,     # Aus Text analysiert
            'voice_analysis': 0.8,    # Aus Stimme analysiert
            'facial_analysis': 0.85,  # Aus Gesicht analysiert
            'behavioral': 0.6,        # Aus Verhalten abgeleitet
            'pattern_match': 0.75     # Aus Pattern erkannt
        }
        
        confidence = source_bonuses.get(source, base_confidence)
        
        # Trigger-Bonus
        if triggers:
            confidence += len(triggers) * 0.05  # +5% pro erkanntem Trigger
        
        # Context-Länge Bonus (mehr Kontext = höhere Confidence)
        if len(context) > 50:
            confidence += 0.1
        elif len(context) > 20:
            confidence += 0.05
        
        # Emotion-spezifische Adjustments
        emotion_info = self.emotion_taxonomy.get(emotion_type, {})
        if emotion_info.get('category') == EmotionCategory.PRIMARY:
            confidence += 0.1  # Primäre Emotionen leichter zu erkennen
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_emotion_expiry(self, emotion_type: str, intensity: float) -> Optional[datetime]:
        """Berechnet wann Emotion 'verblasst'"""
        
        emotion_info = self.emotion_taxonomy.get(emotion_type, {})
        decay_rate = emotion_info.get('decay_rate', 0.9)
        
        # Basis-Lebensdauer: 1-24 Stunden basierend auf Intensität und Typ
        base_hours = 2 + (intensity * 22)  # 2-24 Stunden
        
        # Adjustiere basierend auf Decay Rate
        adjusted_hours = base_hours * (2 - decay_rate)  # Niedrige Decay Rate = länger anhaltend
        
        # Spezielle Emotionen
        if emotion_type == 'neutral':
            return None  # Neutral verblasst nicht
        elif emotion_type in ['trauma', 'grief']:
            adjusted_hours *= 7  # Trauma hält länger an
        elif emotion_type == 'surprise':
            adjusted_hours *= 0.1  # Überraschung verblasst sehr schnell
        
        return datetime.now() + timedelta(hours=adjusted_hours)
    
    def _update_user_emotion_profile(self, user_id: str, emotion_type: str, intensity: float):
        """Updated User-spezifisches Emotion-Profil"""
        
        if user_id not in self.user_emotion_profiles:
            self.user_emotion_profiles[user_id] = {
                'user_id': user_id,
                'emotion_counts': Counter(),
                'intensity_patterns': defaultdict(list),
                'last_emotions': [],
                'emotional_baseline': 'neutral',
                'emotional_volatility': 0.5,
                'dominant_emotions': [],
                'last_updated': datetime.now(),
                'total_emotions': 0
            }
        
        profile = self.user_emotion_profiles[user_id]
        
        # Update Counters
        profile['emotion_counts'][emotion_type] += 1
        profile['intensity_patterns'][emotion_type].append(intensity)
        profile['total_emotions'] += 1
        
        # Update Last Emotions (Ring Buffer)
        profile['last_emotions'].append({
            'emotion': emotion_type,
            'intensity': intensity,
            'timestamp': datetime.now()
        })
        
        if len(profile['last_emotions']) > 10:
            profile['last_emotions'] = profile['last_emotions'][-10:]
        
        # Update Dominant Emotions
        profile['dominant_emotions'] = profile['emotion_counts'].most_common(5)
        
        # Berechne Emotional Volatility
        if len(profile['last_emotions']) >= 3:
            recent_intensities = [e['intensity'] for e in profile['last_emotions'][-5:]]
            variance = self._calculate_variance(recent_intensities)
            profile['emotional_volatility'] = min(1.0, variance * 2)
        
        # Update Baseline
        if profile['total_emotions'] > 5:
            most_common = profile['emotion_counts'].most_common(1)[0][0]
            if profile['emotion_counts'][most_common] > profile['total_emotions'] * 0.3:
                profile['emotional_baseline'] = most_common
        
        profile['last_updated'] = datetime.now()
    
    def _detect_and_store_patterns(self, user_id: str, emotion_type: str, intensity: float):
        """Erkennt und speichert emotionale Muster"""
        
        try:
            patterns_detected = []
            
            # 1. Zeitliche Muster (Tageszeit, Wochentag)
            temporal_patterns = self._detect_temporal_patterns(user_id, emotion_type)
            patterns_detected.extend(temporal_patterns)
            
            # 2. Sequenz-Muster (welche Emotionen folgen aufeinander)
            sequence_patterns = self._detect_sequence_patterns(user_id, emotion_type)
            patterns_detected.extend(sequence_patterns)
            
            # 3. Intensitäts-Muster
            intensity_patterns = self._detect_intensity_patterns(user_id, emotion_type, intensity)
            patterns_detected.extend(intensity_patterns)
            
            # 4. Trigger-Muster
            trigger_patterns = self._detect_trigger_patterns(user_id, emotion_type)
            patterns_detected.extend(trigger_patterns)
            
            # Speichere neue Patterns
            for pattern in patterns_detected:
                self._store_emotion_pattern(user_id, pattern)
            
        except Exception as e:
            logger.error(f"❌ Pattern Detection Error: {e}")
    
    def _detect_temporal_patterns(self, user_id: str, emotion_type: str) -> List[Dict[str, Any]]:
        """Erkennt zeitliche emotionale Muster"""
        
        patterns = []
        
        try:
            with self.db.get_connection() as conn:
                # Hole Emotionen der letzten 30 Tage
                cutoff = datetime.now() - timedelta(days=30)
                
                cursor = conn.execute('''
                    SELECT emotion_type, created_at FROM emotion_entries
                    WHERE user_id = ? AND emotion_type = ? AND created_at > ?
                ''', (user_id, emotion_type, cutoff.isoformat()))
                
                emotions = cursor.fetchall()
                
                if len(emotions) < self.PATTERN_DETECTION_THRESHOLD:
                    return patterns
                
                # Analysiere Tageszeit-Muster
                hourly_counts = defaultdict(int)
                daily_counts = defaultdict(int)
                
                for emotion in emotions:
                    dt = datetime.fromisoformat(emotion['created_at'])
                    hourly_counts[dt.hour] += 1
                    daily_counts[dt.weekday()] += 1
                
                # Erkenne Peak-Zeiten
                max_hour = max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
                max_day = max(daily_counts, key=daily_counts.get) if daily_counts else None
                
                if max_hour is not None and hourly_counts[max_hour] >= self.PATTERN_DETECTION_THRESHOLD:
                    patterns.append({
                        'type': 'temporal_hourly',
                        'pattern_data': {
                            'emotion_type': emotion_type,
                            'peak_hour': max_hour,
                            'occurrences': hourly_counts[max_hour],
                            'confidence': min(0.95, hourly_counts[max_hour] / len(emotions))
                        }
                    })
                
                if max_day is not None and daily_counts[max_day] >= self.PATTERN_DETECTION_THRESHOLD:
                    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    patterns.append({
                        'type': 'temporal_daily',
                        'pattern_data': {
                            'emotion_type': emotion_type,
                            'peak_day': day_names[max_day],
                            'peak_day_index': max_day,
                            'occurrences': daily_counts[max_day],
                            'confidence': min(0.95, daily_counts[max_day] / len(emotions))
                        }
                    })
            
        except Exception as e:
            logger.error(f"❌ Temporal Pattern Error: {e}")
        
        return patterns
    
    def _detect_sequence_patterns(self, user_id: str, current_emotion: str) -> List[Dict[str, Any]]:
        """Erkennt Sequenz-Muster (A->B Emotionen)"""
        
        patterns = []
        
        try:
            with self.db.get_connection() as conn:
                # Hole letzte 20 Emotionen
                cursor = conn.execute('''
                    SELECT emotion_type, created_at FROM emotion_entries
                    WHERE user_id = ?
                    ORDER BY created_at DESC LIMIT 20
                ''', (user_id,))
                
                emotions = [row['emotion_type'] for row in cursor.fetchall()]
                
                if len(emotions) < 3:
                    return patterns
                
                # Analysiere Sequenzen
                sequences = defaultdict(int)
                for i in range(len(emotions) - 1):
                    sequence = f"{emotions[i+1]}->{emotions[i]}"  # Reverse because DESC order
                    sequences[sequence] += 1
                
                # Finde signifikante Sequenzen
                for sequence, count in sequences.items():
                    if count >= self.PATTERN_DETECTION_THRESHOLD:
                        from_emotion, to_emotion = sequence.split('->')
                        if to_emotion == current_emotion:
                            patterns.append({
                                'type': 'sequence',
                                'pattern_data': {
                                    'from_emotion': from_emotion,
                                    'to_emotion': to_emotion,
                                    'occurrences': count,
                                    'confidence': min(0.9, count / len(emotions))
                                }
                            })
            
        except Exception as e:
            logger.error(f"❌ Sequence Pattern Error: {e}")
        
        return patterns
    
    def _create_emotion_links(self, emotion_id: int, user_id: str, emotion_type: str):
        """Erstellt Links zwischen Emotionen"""
        
        try:
            with self.db.get_connection() as conn:
                # Finde kürzlich aufgetretene Emotionen
                cursor = conn.execute('''
                    SELECT id, emotion_type, created_at FROM emotion_entries
                    WHERE user_id = ? AND id != ?
                    ORDER BY created_at DESC LIMIT 5
                ''', (user_id, emotion_id))
                
                recent_emotions = cursor.fetchall()
                
                for emotion in recent_emotions:
                    # Berechne zeitlichen Abstand
                    time_diff = datetime.now() - datetime.fromisoformat(emotion['created_at'])
                    
                    if time_diff.total_seconds() < 3600:  # Innerhalb einer Stunde
                        # Bestimme Link-Typ
                        link_type = self._determine_link_type(emotion['emotion_type'], emotion_type)
                        
                        if link_type:
                            # Berechne Stärke basierend auf Zeit und Beziehung
                            strength = max(0.1, 1.0 - (time_diff.total_seconds() / 3600))
                            
                            # Erstelle Link
                            conn.execute('''
                                INSERT INTO emotion_links 
                                (from_emotion_id, to_emotion_id, link_type, strength, created_at)
                                VALUES (?, ?, ?, ?, ?)
                            ''', (
                                emotion['id'],
                                emotion_id,
                                link_type,
                                strength,
                                datetime.now().isoformat()
                            ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Emotion Link Error: {e}")
    
    def _determine_link_type(self, from_emotion: str, to_emotion: str) -> Optional[str]:
        """Bestimmt Art der emotionalen Verknüpfung"""
        
        # Hole Emotion-Infos
        from_info = self.emotion_taxonomy.get(from_emotion, {})
        to_info = self.emotion_taxonomy.get(to_emotion, {})
        
        # Prüfe auf Gegensätze
        if from_info.get('opposite') == to_emotion or to_info.get('opposite') == from_emotion:
            return 'opposite'
        
        # Prüfe auf verwandte Emotionen
        if to_emotion in from_info.get('related_emotions', []):
            return 'related'
        
        # Prüfe auf typische Sequenzen
        common_sequences = {
            ('surprise', 'happiness'): 'positive_surprise',
            ('surprise', 'fear'): 'negative_surprise',
            ('anger', 'sadness'): 'emotional_cooling',
            ('fear', 'relief'): 'resolution',
            ('sadness', 'acceptance'): 'grief_processing',
            ('confusion', 'understanding'): 'learning'
        }
        
        sequence_key = (from_emotion, to_emotion)
        if sequence_key in common_sequences:
            return common_sequences[sequence_key]
        
        # Default: Zeitliche Sequenz
        return 'sequence'
    
    def get_emotion_insights(self, user_id: str = "default", days: int = 30) -> Dict[str, Any]:
        """Generiert umfassende Emotion-Insights"""
        
        try:
            insights = {
                'user_id': user_id,
                'analysis_period_days': days,
                'emotional_profile': self._generate_emotional_profile(user_id, days),
                'patterns': self._get_emotion_patterns_detailed(user_id, days),
                'trends': self._analyze_emotion_trends(user_id, days),
                'recommendations': self._generate_emotion_recommendations(user_id),
                'social_dynamics': self._analyze_social_emotion_dynamics(user_id, days),
                'emotional_intelligence_score': self._calculate_ei_score(user_id),
                'generated_at': datetime.now().isoformat()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Emotion Insights Error: {e}")
            return {'error': str(e), 'user_id': user_id}

    def _calculate_variance(self, values: List[float]) -> float:
        """Berechnet Varianz einer Werte-Liste"""
        
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _detect_intensity_patterns(self, user_id: str, emotion_type: str, intensity: float) -> List[Dict[str, Any]]:
        """Erkennt Intensitäts-Muster"""
        
        patterns = []
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT intensity, created_at FROM emotion_entries
                    WHERE user_id = ? AND emotion_type = ?
                    ORDER BY created_at DESC LIMIT 20
                ''', (user_id, emotion_type))
                
                intensities = [row['intensity'] for row in cursor.fetchall()]
                
                if len(intensities) >= self.PATTERN_DETECTION_THRESHOLD:
                    avg_intensity = sum(intensities) / len(intensities)
                    
                    # Prüfe auf Intensitäts-Trends
                    if len(intensities) >= 5:
                        recent_avg = sum(intensities[:5]) / 5
                        older_avg = sum(intensities[5:]) / max(1, len(intensities[5:]))
                        
                        if recent_avg > older_avg * 1.2:
                            patterns.append({
                                'type': 'intensity_increasing',
                                'pattern_data': {
                                    'emotion_type': emotion_type,
                                    'trend': 'increasing',
                                    'recent_avg': recent_avg,
                                    'older_avg': older_avg,
                                    'confidence': min(0.9, abs(recent_avg - older_avg))
                                }
                            })
                        elif recent_avg < older_avg * 0.8:
                            patterns.append({
                                'type': 'intensity_decreasing',
                                'pattern_data': {
                                    'emotion_type': emotion_type,
                                    'trend': 'decreasing',
                                    'recent_avg': recent_avg,
                                    'older_avg': older_avg,
                                    'confidence': min(0.9, abs(older_avg - recent_avg))
                                }
                            })
            
        except Exception as e:
            logger.error(f"❌ Intensity Pattern Error: {e}")
        
        return patterns
    
    def _detect_trigger_patterns(self, user_id: str, emotion_type: str) -> List[Dict[str, Any]]:
        """Erkennt Trigger-Muster"""
        
        patterns = []
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT triggers FROM emotion_entries
                    WHERE user_id = ? AND emotion_type = ?
                    AND triggers IS NOT NULL AND triggers != '[]'
                ''', (user_id, emotion_type))
                
                all_triggers = []
                for row in cursor.fetchall():
                    try:
                        triggers = json.loads(row['triggers'])
                        all_triggers.extend(triggers)
                    except:
                        continue
                
                if all_triggers:
                    trigger_counts = Counter(all_triggers)
                    common_triggers = trigger_counts.most_common(5)
                    
                    for trigger, count in common_triggers:
                        if count >= self.PATTERN_DETECTION_THRESHOLD:
                            patterns.append({
                                'type': 'trigger_pattern',
                                'pattern_data': {
                                    'emotion_type': emotion_type,
                                    'trigger': trigger,
                                    'occurrences': count,
                                    'confidence': min(0.9, count / len(all_triggers))
                                }
                            })
            
        except Exception as e:
            logger.error(f"❌ Trigger Pattern Error: {e}")
        
        return patterns
    
    def _store_emotion_pattern(self, user_id: str, pattern: Dict[str, Any]):
        """Speichert erkanntes Emotions-Pattern"""
        
        try:
            with self.db.get_connection() as conn:
                # Prüfe ob Pattern bereits existiert
                existing = conn.execute('''
                    SELECT id, occurrences FROM emotion_patterns
                    WHERE user_id = ? AND pattern_type = ? 
                    AND json_extract(pattern_data, '$.emotion_type') = ?
                ''', (
                    user_id, 
                    pattern['type'],
                    pattern['pattern_data'].get('emotion_type', '')
                )).fetchone()
                
                if existing:
                    # Update existierendes Pattern
                    conn.execute('''
                        UPDATE emotion_patterns 
                        SET pattern_data = ?, 
                            last_updated = ?,
                            occurrences = occurrences + 1
                        WHERE id = ?
                    ''', (
                        json.dumps(pattern['pattern_data']),
                        datetime.now().isoformat(),
                        existing['id']
                    ))
                else:
                    # Erstelle neues Pattern
                    conn.execute('''
                        INSERT INTO emotion_patterns 
                        (user_id, pattern_type, pattern_data, confidence, detected_at, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        user_id,
                        pattern['type'],
                        json.dumps(pattern['pattern_data']),
                        pattern['pattern_data'].get('confidence', 0.7),
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
            
        except Exception as e:
            logger.error(f"❌ Pattern Storage Error: {e}")
    
    def search_emotions(
        self,
        user_id: str = "default",
        query: str = "",
        emotion_types: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        min_intensity: float = 0.0,
        max_intensity: float = 1.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Intelligente Emotion-Suche"""
        
        results = []
        
        try:
            with self.db.get_connection() as conn:
                # Basis-Query
                sql = '''
                    SELECT e.*, 
                           (e.intensity * 0.3 + e.confidence * 0.7) as relevance_score
                    FROM emotion_entries e
                    WHERE e.user_id = ?
                '''
                params = [user_id]
                
                # Query-Filter
                if query:
                    sql += ' AND (e.context LIKE ? OR e.triggers LIKE ?)'
                    params.extend([f'%{query}%', f'%{query}%'])
                
                # Emotion-Typ Filter
                if emotion_types:
                    placeholders = ','.join(['?' for _ in emotion_types])
                    sql += f' AND e.emotion_type IN ({placeholders})'
                    params.extend(emotion_types)
                
                # Datum-Filter
                if date_range:
                    start_date, end_date = date_range
                    sql += ' AND e.created_at BETWEEN ? AND ?'
                    params.extend([start_date.isoformat(), end_date.isoformat()])
                
                # Intensitäts-Filter
                sql += ' AND e.intensity BETWEEN ? AND ?'
                params.extend([min_intensity, max_intensity])
                
                # Sortierung und Limit
                sql += ' ORDER BY relevance_score DESC, e.created_at DESC LIMIT ?'
                params.append(limit)
                
                cursor = conn.execute(sql, params)
                
                for row in cursor.fetchall():
                    try:
                        result = {
                            'id': row['id'],
                            'emotion_type': row['emotion_type'],
                            'intensity': row['intensity'],
                            'category': row['category'],
                            'context': row['context'],
                            'triggers': json.loads(row['triggers']) if row['triggers'] else [],
                            'created_at': row['created_at'],
                            'confidence': row['confidence'],
                            'source': row['source'],
                            'relevance_score': row['relevance_score'],
                            'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                        }
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"⚠️ Result parsing error: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"❌ Emotion Search Error: {e}")
        
        return results
    
    def _generate_emotional_profile(self, user_id: str, days: int) -> Dict[str, Any]:
        """Generiert detailliertes emotionales Profil"""
        
        profile = {
            'dominant_emotions': [],
            'emotional_stability': 0.0,
            'emotional_diversity': 0.0,
            'average_intensity': 0.0,
            'emotional_peaks': [],
            'baseline_emotion': 'neutral',
            'volatility_score': 0.0
        }
        
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            with self.db.get_connection() as conn:
                # Hole alle Emotionen im Zeitraum
                cursor = conn.execute('''
                    SELECT emotion_type, intensity, created_at FROM emotion_entries
                    WHERE user_id = ? AND created_at > ?
                    ORDER BY created_at ASC
                ''', (user_id, cutoff.isoformat()))
                
                emotions = cursor.fetchall()
                
                if not emotions:
                    return profile
                
                # Analysiere Emotionen
                emotion_counts = Counter()
                intensities = []
                daily_emotions = defaultdict(list)
                
                for emotion in emotions:
                    emotion_counts[emotion['emotion_type']] += 1
                    intensities.append(emotion['intensity'])
                    
                    # Gruppiere nach Tag
                    date = datetime.fromisoformat(emotion['created_at']).date()
                    daily_emotions[date].append(emotion)
                
                # Berechne Statistiken
                profile['dominant_emotions'] = [
                    {'emotion': emotion, 'count': count, 'percentage': count/len(emotions)*100}
                    for emotion, count in emotion_counts.most_common(5)
                ]
                
                profile['average_intensity'] = sum(intensities) / len(intensities)
                profile['volatility_score'] = self._calculate_variance(intensities)
                
                # Emotionale Diversität (Shannon-Diversity)
                total = len(emotions)
                profile['emotional_diversity'] = -sum(
                    (count/total) * math.log2(count/total) 
                    for count in emotion_counts.values()
                )
                
                # Baseline-Emotion
                if emotion_counts:
                    profile['baseline_emotion'] = emotion_counts.most_common(1)[0][0]
                
                # Emotional Peaks (Tage mit hoher Intensität)
                daily_intensities = {}
                for date, day_emotions in daily_emotions.items():
                    avg_intensity = sum(e['intensity'] for e in day_emotions) / len(day_emotions)
                    daily_intensities[date] = avg_intensity
                
                if daily_intensities:
                    sorted_days = sorted(daily_intensities.items(), key=lambda x: x[1], reverse=True)
                    profile['emotional_peaks'] = [
                        {'date': date.isoformat(), 'intensity': intensity}
                        for date, intensity in sorted_days[:3]
                    ]
                
                # Emotional Stability (niedrige Varianz = hohe Stabilität)
                profile['emotional_stability'] = max(0, 1.0 - profile['volatility_score'])
            
        except Exception as e:
            logger.error(f"❌ Emotional Profile Error: {e}")
        
        return profile
    
    def _get_emotion_patterns_detailed(self, user_id: str, days: int) -> Dict[str, Any]:
        """Holt detaillierte Emotion-Patterns"""
        
        patterns = {
            'temporal_patterns': [],
            'sequence_patterns': [],
            'trigger_patterns': [],
            'intensity_patterns': [],
            'pattern_confidence': 0.0
        }
        
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT * FROM emotion_patterns
                    WHERE user_id = ? AND last_updated > ?
                    ORDER BY confidence DESC
                ''', (user_id, cutoff.isoformat()))
                
                for row in cursor.fetchall():
                    try:
                        pattern_data = json.loads(row['pattern_data'])
                        pattern_info = {
                            'type': row['pattern_type'],
                            'data': pattern_data,
                            'confidence': row['confidence'],
                            'occurrences': row['occurrences'],
                            'last_updated': row['last_updated']
                        }
                        
                        # Kategorisiere Pattern
                        if 'temporal' in row['pattern_type']:
                            patterns['temporal_patterns'].append(pattern_info)
                        elif 'sequence' in row['pattern_type']:
                            patterns['sequence_patterns'].append(pattern_info)
                        elif 'trigger' in row['pattern_type']:
                            patterns['trigger_patterns'].append(pattern_info)
                        elif 'intensity' in row['pattern_type']:
                            patterns['intensity_patterns'].append(pattern_info)
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Pattern parsing error: {e}")
                        continue
                
                # Berechne durchschnittliche Pattern-Confidence
                all_patterns = (
                    patterns['temporal_patterns'] + 
                    patterns['sequence_patterns'] + 
                    patterns['trigger_patterns'] + 
                    patterns['intensity_patterns']
                )
                
                if all_patterns:
                    patterns['pattern_confidence'] = sum(p['confidence'] for p in all_patterns) / len(all_patterns)
            
        except Exception as e:
            logger.error(f"❌ Pattern Details Error: {e}")
        
        return patterns
    
    def _analyze_emotion_trends(self, user_id: str, days: int) -> Dict[str, Any]:
        """Analysiert emotionale Trends"""
        
        trends = {
            'overall_trend': 'stable',
            'happiness_trend': 'stable',
            'stress_indicators': [],
            'improvement_areas': [],
            'positive_developments': [],
            'weekly_comparison': {}
        }
        
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            with self.db.get_connection() as conn:
                # Hole Emotionen chronologisch
                cursor = conn.execute('''
                    SELECT emotion_type, intensity, created_at FROM emotion_entries
                    WHERE user_id = ? AND created_at > ?
                    ORDER BY created_at ASC
                ''', (user_id, cutoff.isoformat()))
                
                emotions = cursor.fetchall()
                
                if len(emotions) < 7:  # Mindestens eine Woche Daten
                    return trends
                
                # Teile in Wochen auf
                weekly_data = defaultdict(lambda: defaultdict(list))
                
                for emotion in emotions:
                    dt = datetime.fromisoformat(emotion['created_at'])
                    week = dt.isocalendar()[1]  # ISO Woche
                    
                    weekly_data[week][emotion['emotion_type']].append(emotion['intensity'])
                
                # Analysiere Trends
                weeks = sorted(weekly_data.keys())
                if len(weeks) >= 2:
                    # Vergleiche erste und letzte Woche
                    first_week = weekly_data[weeks[0]]
                    last_week = weekly_data[weeks[-1]]
                    
                    # Happiness Trend
                    happiness_emotions = ['happiness', 'joy', 'contentment', 'gratitude']
                    first_happiness = sum(
                        sum(first_week[emotion]) / len(first_week[emotion])
                        for emotion in happiness_emotions if emotion in first_week
                    )
                    last_happiness = sum(
                        sum(last_week[emotion]) / len(last_week[emotion])
                        for emotion in happiness_emotions if emotion in last_week
                    )
                    
                    if last_happiness > first_happiness * 1.1:
                        trends['happiness_trend'] = 'improving'
                        trends['positive_developments'].append('Increased happiness levels')
                    elif last_happiness < first_happiness * 0.9:
                        trends['happiness_trend'] = 'declining'
                        trends['improvement_areas'].append('Focus on positive activities')
                    
                    # Stress Indicators
                    stress_emotions = ['anger', 'fear', 'anxiety', 'frustration']
                    for emotion in stress_emotions:
                        if emotion in last_week:
                            avg_intensity = sum(last_week[emotion]) / len(last_week[emotion])
                            if avg_intensity > 0.7:
                                trends['stress_indicators'].append(f'High {emotion} levels detected')
                
                # Weekly Comparison
                for week in weeks[-4:]:  # Letzte 4 Wochen
                    week_emotions = weekly_data[week]
                    week_summary = {}
                    
                    for emotion_type, intensities in week_emotions.items():
                        week_summary[emotion_type] = {
                            'avg_intensity': sum(intensities) / len(intensities),
                            'count': len(intensities)
                        }
                    
                    trends['weekly_comparison'][f'week_{week}'] = week_summary
            
        except Exception as e:
            logger.error(f"❌ Trend Analysis Error: {e}")
        
        return trends
    
    def _generate_emotion_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Generiert personalisierte Emotion-Empfehlungen"""
        
        recommendations = []
        
        try:
            # Hole User Profile aus Cache
            profile = self.user_emotion_profiles.get(user_id)
            if not profile:
                return recommendations
            
            # Analysiere dominante Emotionen
            dominant_emotions = profile.get('dominant_emotions', [])
            volatility = profile.get('emotional_volatility', 0.5)
            baseline = profile.get('emotional_baseline', 'neutral')
            
            # Empfehlungen basierend auf Patterns
            if volatility > 0.7:
                recommendations.append({
                    'type': 'stability',
                    'priority': 'high',
                    'title': 'Emotional Stability',
                    'description': 'Your emotional patterns show high volatility. Consider mindfulness or relaxation techniques.',
                    'actionable_steps': [
                        'Practice daily meditation',
                        'Keep an emotion journal',
                        'Identify triggers for emotional swings'
                    ]
                })
            
            # Empfehlungen basierend auf dominanten Emotionen
            if dominant_emotions:
                top_emotion = dominant_emotions[0][0]
                
                if top_emotion in ['sadness', 'fear', 'anger']:
                    recommendations.append({
                        'type': 'mood_improvement',
                        'priority': 'medium',
                        'title': f'Managing {top_emotion.title()}',
                        'description': f'You frequently experience {top_emotion}. Here are some strategies to help.',
                        'actionable_steps': self._get_coping_strategies(top_emotion)
                    })
                
                elif top_emotion == 'happiness':
                    recommendations.append({
                        'type': 'maintain_positivity',
                        'priority': 'low',
                        'title': 'Maintaining Positive Emotions',
                        'description': 'Great job maintaining positive emotions! Here\'s how to sustain this.',
                        'actionable_steps': [
                            'Share positive experiences with others',
                            'Practice gratitude regularly',
                            'Engage in activities that bring you joy'
                        ]
                    })
            
            # Baseline-basierte Empfehlungen
            if baseline in ['sadness', 'fear']:
                recommendations.append({
                    'type': 'baseline_shift',
                    'priority': 'high',
                    'title': 'Shifting Emotional Baseline',
                    'description': f'Your emotional baseline appears to be {baseline}. Let\'s work on shifting this.',
                    'actionable_steps': [
                        'Engage in regular physical exercise',
                        'Connect with supportive friends and family',
                        'Consider professional support if needed',
                        'Practice positive self-talk'
                    ]
                })
        
        except Exception as e:
            logger.error(f"❌ Recommendations Error: {e}")
        
        return recommendations
    
    def _get_coping_strategies(self, emotion_type: str) -> List[str]:
        """Holt Bewältigungsstrategien für spezifische Emotionen"""
        
        strategies = {
            'sadness': [
                'Reach out to friends or family for support',
                'Engage in physical activity',
                'Practice self-compassion',
                'Consider what you\'re grateful for'
            ],
            'anger': [
                'Take deep breaths before responding',
                'Use "I" statements to express feelings',
                'Take a break from the situation',
                'Channel energy into physical activity'
            ],
            'fear': [
                'Identify what specifically you\'re afraid of',
                'Challenge negative thought patterns',
                'Take small steps toward what you fear',
                'Practice relaxation techniques'
            ],
            'anxiety': [
                'Use grounding techniques (5-4-3-2-1 method)',
                'Practice progressive muscle relaxation',
                'Focus on what you can control',
                'Limit caffeine and alcohol'
            ]
        }
        
        return strategies.get(emotion_type, [
            'Take time to acknowledge and accept the emotion',
            'Consider what this emotion might be telling you',
            'Seek support from others when needed'
        ])
    
    def _analyze_social_emotion_dynamics(self, user_id: str, days: int) -> Dict[str, Any]:
        """Analysiert soziale Emotions-Dynamiken"""
        
        dynamics = {
            'emotional_contagion': 0.0,
            'empathy_indicators': [],
            'social_triggers': [],
            'relationship_emotions': {}
        }
        
        try:
            cutoff = datetime.now() - timedelta(days=days)
            
            with self.db.get_connection() as conn:
                # Suche nach sozialen Emotion-Triggern
                cursor = conn.execute('''
                    SELECT emotion_type, triggers, context FROM emotion_entries
                    WHERE user_id = ? AND created_at > ?
                    AND (triggers LIKE '%social%' OR context LIKE '%friend%' 
                         OR context LIKE '%family%' OR context LIKE '%colleague%')
                ''', (user_id, cutoff.isoformat()))
                
                social_emotions = cursor.fetchall()
                
                # Analysiere soziale Trigger
                social_trigger_counts = Counter()
                for emotion in social_emotions:
                    try:
                        triggers = json.loads(emotion['triggers']) if emotion['triggers'] else []
                        for trigger in triggers:
                            if 'social' in trigger:
                                social_trigger_counts[trigger] += 1
                    except:
                        continue
                
                dynamics['social_triggers'] = [
                    {'trigger': trigger, 'count': count}
                    for trigger, count in social_trigger_counts.most_common(5)
                ]
                
                # Empathie-Indikatoren
                empathy_emotions = ['empathy', 'compassion', 'sympathy']
                empathy_count = sum(
                    1 for emotion in social_emotions 
                    if emotion['emotion_type'] in empathy_emotions
                )
                
                if empathy_count > 0:
                    dynamics['empathy_indicators'].append({
                        'type': 'emotional_responsiveness',
                        'score': min(1.0, empathy_count / len(social_emotions)) if social_emotions else 0,
                        'description': 'Shows emotional responsiveness to others'
                    })
            
        except Exception as e:
            logger.error(f"❌ Social Dynamics Error: {e}")
        
        return dynamics
    
    def _calculate_ei_score(self, user_id: str) -> Dict[str, Any]:
        """Berechnet Emotional Intelligence Score"""
        
        ei_score = {
            'overall_score': 0.0,
            'self_awareness': 0.0,
            'self_regulation': 0.0,
            'empathy': 0.0,
            'social_skills': 0.0,
            'assessment_date': datetime.now().isoformat()
        }
        
        try:
            profile = self.user_emotion_profiles.get(user_id)
            if not profile:
                return ei_score
            
            # Self-Awareness (Emotionale Vielfalt und Bewusstsein)
            emotion_count = len(profile['emotion_counts'])
            total_emotions = profile['total_emotions']
            
            if total_emotions > 0:
                ei_score['self_awareness'] = min(1.0, emotion_count / 8.0)  # Max bei 8 verschiedenen Emotionen
            
            # Self-Regulation (Emotionale Stabilität)
            volatility = profile.get('emotional_volatility', 0.5)
            ei_score['self_regulation'] = max(0.0, 1.0 - volatility)
            
            # Empathy (Anzahl empathischer Emotionen)
            empathy_emotions = ['empathy', 'compassion', 'sympathy', 'understanding']
            empathy_count = sum(
                profile['emotion_counts'].get(emotion, 0) 
                for emotion in empathy_emotions
            )
            
            if total_emotions > 0:
                ei_score['empathy'] = min(1.0, empathy_count / (total_emotions * 0.1))
            
            # Social Skills (soziale Emotion-Trigger)
            # Vereinfacht: basierend auf Anzahl sozialer Interaktionen
            social_emotions = ['gratitude', 'empathy', 'happiness', 'pride']
            social_count = sum(
                profile['emotion_counts'].get(emotion, 0) 
                for emotion in social_emotions
            )
            
            if total_emotions > 0:
                ei_score['social_skills'] = min(1.0, social_count / (total_emotions * 0.3))
            
            # Overall Score (gewichteter Durchschnitt)
            scores = [
                ei_score['self_awareness'] * 0.3,
                ei_score['self_regulation'] * 0.3,
                ei_score['empathy'] * 0.2,
                ei_score['social_skills'] * 0.2
            ]
            
            ei_score['overall_score'] = sum(scores)
        
        except Exception as e:
            logger.error(f"❌ EI Score Error: {e}")
        
        return ei_score
    
    def get_current_emotional_state(self, user_id: str) -> Dict[str, Any]:
        """Gibt aktuellen emotionalen Zustand zurück"""
        
        state = {
            'primary_emotion': 'neutral',
            'intensity': 0.5,
            'confidence': 0.5,
            'duration': 0,
            'triggers': [],
            'related_emotions': [],
            'prediction': {},
            'recommendations': []
        }
        
        try:
            with self.db.get_connection() as conn:
                # Hole neueste Emotion
                cursor = conn.execute('''
                    SELECT * FROM emotion_entries
                    WHERE user_id = ? 
                    ORDER BY created_at DESC LIMIT 1
                ''', (user_id,))
                
                latest = cursor.fetchone()
                
                if latest:
                    created_at = datetime.fromisoformat(latest['created_at'])
                    duration_minutes = (datetime.now() - created_at).total_seconds() / 60
                    
                    # Berechne aktuelle Intensität mit Decay
                    emotion_info = self.emotion_taxonomy.get(latest['emotion_type'], {})
                    decay_rate = emotion_info.get('decay_rate', 0.9)
                    
                    # Exponential decay basierend auf Zeit
                    time_factor = math.exp(-duration_minutes / (24 * 60))  # 24h Halbwertszeit
                    current_intensity = latest['intensity'] * decay_rate * time_factor
                    
                    state.update({
                        'primary_emotion': latest['emotion_type'],
                        'intensity': max(0.1, current_intensity),
                        'confidence': latest['confidence'],
                        'duration': duration_minutes,
                        'triggers': json.loads(latest['triggers']) if latest['triggers'] else [],
                        'related_emotions': emotion_info.get('related_emotions', [])
                    })
                    
                    # Vorhersage nächster wahrscheinlicher Emotionen
                    state['prediction'] = self._predict_next_emotions(user_id, latest['emotion_type'])
                    
                    # Situationsbasierte Empfehlungen
                    state['recommendations'] = self._get_immediate_recommendations(
                        latest['emotion_type'], current_intensity
                    )
        
        except Exception as e:
            logger.error(f"❌ Current State Error: {e}")
        
        return state
    
    def _predict_next_emotions(self, user_id: str, current_emotion: str) -> Dict[str, Any]:
        """Vorhersage wahrscheinlicher nächster Emotionen"""
        
        prediction = {
            'likely_next': [],
            'confidence': 0.0,
            'time_window': '1-2 hours'
        }
        
        try:
            with self.db.get_connection() as conn:
                # Finde historische Sequenzen
                cursor = conn.execute('''
                    SELECT el.to_emotion_id, e2.emotion_type, AVG(el.strength) as avg_strength
                    FROM emotion_links el
                    JOIN emotion_entries e1 ON el.from_emotion_id = e1.id
                    JOIN emotion_entries e2 ON el.to_emotion_id = e2.id
                    WHERE e1.user_id = ? AND e1.emotion_type = ?
                    GROUP BY e2.emotion_type
                    ORDER BY avg_strength DESC
                    LIMIT 3
                ''', (user_id, current_emotion))
                
                predictions = cursor.fetchall()
                
                for pred in predictions:
                    prediction['likely_next'].append({
                        'emotion': pred['emotion_type'],
                        'probability': pred['avg_strength'],
                        'basis': 'historical_pattern'
                    })
                
                if predictions:
                    prediction['confidence'] = sum(p['avg_strength'] for p in predictions) / len(predictions)
        
        except Exception as e:
            logger.error(f"❌ Prediction Error: {e}")
        
        return prediction
    
    def _get_immediate_recommendations(self, emotion_type: str, intensity: float) -> List[str]:
        """Gibt sofortige Empfehlungen basierend auf aktueller Emotion"""
        
        recommendations = []
        
        if intensity > 0.8:  # Hohe Intensität
            if emotion_type in ['anger', 'fear', 'sadness']:
                recommendations.extend([
                    "Take slow, deep breaths",
                    "Consider taking a short break",
                    "Practice grounding techniques"
                ])
            elif emotion_type == 'happiness':
                recommendations.extend([
                    "Share this positive moment with someone",
                    "Take note of what made you happy",
                    "Consider doing something creative"
                ])
        
        elif intensity < 0.3:  # Niedrige Intensität
            recommendations.extend([
                "Check in with yourself about how you're feeling",
                "Consider what might improve your mood",
                "Engage in a activity you enjoy"
            ])
        
        # Emotion-spezifische Empfehlungen
        specific_recommendations = {
            'confusion': ["Break down complex problems into smaller parts", "Seek clarification or help"],
            'curiosity': ["Explore this interest further", "Ask questions and learn more"],
            'gratitude': ["Express thanks to someone who helped you", "Write down what you're grateful for"],
            'empathy': ["Reach out to offer support", "Practice active listening"]
        }
        
        if emotion_type in specific_recommendations:
            recommendations.extend(specific_recommendations[emotion_type])
        
        return recommendations[:3]  # Maximal 3 Empfehlungen