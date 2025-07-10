"""
Enhanced Emotion Memory System - Menschenähnliches emotionales Gedächtnis
Integriert mit Enhanced Memory Database und Human-Like Memory System
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics

# Enhanced Memory Integration
from ..storage.memory_database import EnhancedMemoryDatabase, MemorySearchFilter
from ..storage.memory_models import (
    EnhancedMemoryEntry, EmotionMemory, EmotionType, 
    MemoryType, PersonalityPattern
)

logger = logging.getLogger(__name__)

@dataclass
class EmotionalContext:
    """Kontext für emotionale Verarbeitung"""
    current_emotion: Optional[EmotionType] = None
    emotion_intensity: float = 0.5
    emotion_valence: float = 0.0
    mood_state: str = "neutral"
    emotional_trajectory: List[Dict] = field(default_factory=list)
    trigger_events: List[str] = field(default_factory=list)
    social_context: Optional[str] = None
    environmental_factors: List[str] = field(default_factory=list)

@dataclass
class EmotionalInsight:
    """Emotionale Einsichten und Patterns"""
    dominant_emotions: List[str]
    emotional_volatility: float
    average_intensity: float
    positive_negative_ratio: float
    peak_emotional_moments: List[Dict]
    emotional_triggers: Dict[str, float]
    mood_patterns: Dict[str, Any]
    emotional_intelligence_score: float
    recommendations: List[str]

class HumanLikeEmotionMemory:
    """
    Enhanced Emotion Memory System für menschenähnliche emotionale Intelligenz
    
    Features:
    - Emotionale Gedächtnisbildung und -abruf
    - Mood-Tracking und emotionale Muster
    - Empathie und emotionale Anpassung
    - Emotionale Lernfähigkeit
    - Integration mit STM/LTM System
    """
    
    def __init__(self, memory_database: EnhancedMemoryDatabase, user_id: str = "default"):
        self.memory_database = memory_database
        self.user_id = user_id
        
        # Emotionale Zustandsverfolgung
        self.current_emotional_context = EmotionalContext()
        self.emotional_history = []
        self.mood_buffer = []
        
        # Konfiguration
        self.config = {
            'emotion_threshold': 0.3,
            'peak_emotion_threshold': 0.8,
            'mood_window_minutes': 30,
            'emotional_memory_boost': 1.5,
            'empathy_factor': 0.7,
            'emotional_decay_rate': 0.05,
            'mood_influence_strength': 0.4
        }
        
        # Emotionale Intelligence
        self.emotional_patterns = {}
        self.empathy_model = {}
        self.emotional_associations = defaultdict(list)
        
        logger.info(f"🧠❤️ Enhanced Emotion Memory System initialisiert für User: {user_id}")
    
    def process_emotional_input(
        self,
        content: str,
        detected_emotion: Union[str, EmotionType],
        emotion_intensity: float,
        emotion_valence: float = 0.0,
        session_id: str = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Verarbeitet emotionale Eingabe und erstellt emotionale Erinnerung
        
        Args:
            content: Textinhalt der emotionalen Erfahrung
            detected_emotion: Erkannte Emotion
            emotion_intensity: Intensität der Emotion (0.0-1.0)
            emotion_valence: Emotionale Valenz (-1.0 bis 1.0)
            session_id: Session ID
            context: Zusätzlicher Kontext
            
        Returns:
            Dictionary mit Verarbeitungsresultaten
        """
        
        try:
            # Normalisiere Emotion
            if isinstance(detected_emotion, str):
                try:
                    emotion_type = EmotionType(detected_emotion.lower())
                except ValueError:
                    emotion_type = EmotionType.NEUTRAL
                    logger.warning(f"⚠️ Unbekannte Emotion: {detected_emotion}, nutze NEUTRAL")
            else:
                emotion_type = detected_emotion
            
            # Analysiere emotionalen Kontext
            emotional_analysis = self._analyze_emotional_context(
                emotion_type, emotion_intensity, emotion_valence, context
            )
            
            # Bestimme ob Peak Emotional Moment
            is_peak_moment = emotion_intensity >= self.config['peak_emotion_threshold']
            
            # Berechne emotionale Memory-Stärke
            emotional_memory_strength = self._calculate_emotional_memory_strength(
                emotion_intensity, emotion_valence, is_peak_moment
            )
            
            # Erweiterte Metadaten
            enhanced_metadata = {
                'emotional_analysis': emotional_analysis,
                'mood_state': self.current_emotional_context.mood_state,
                'emotional_triggers': emotional_analysis.get('triggers', []),
                'empathy_response': self._generate_empathy_response(emotion_type, emotion_intensity),
                'emotional_associations': self._find_emotional_associations(emotion_type),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            if context:
                enhanced_metadata.update(context)
            
            # Speichere Enhanced Memory Entry
            memory_id = self.memory_database.store_enhanced_memory(
                session_id=session_id or f"emotion_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=self.user_id,
                memory_type=MemoryType.EMOTION,
                content=content,
                metadata=enhanced_metadata,
                importance=self._calculate_emotional_importance(emotion_intensity, is_peak_moment),
                tags=self._generate_emotion_tags(emotion_type, emotion_intensity, context),
                
                # Emotion Fields
                emotion_type=emotion_type,
                emotion_intensity=emotion_intensity,
                emotion_valence=emotion_valence,
                
                # Context
                user_context=emotional_analysis.get('user_context'),
                conversation_context=emotional_analysis.get('conversation_context'),
                personality_aspect=self._identify_personality_aspect(emotion_type, context),
                
                # Learning Fields
                learning_weight=emotional_memory_strength,
                memory_category='emotional_experience',
                attention_weight=min(1.0, emotion_intensity + 0.3),
                
                # STM/LTM Integration
                stm_activation_level=emotion_intensity,
                ltm_significance_score=self._calculate_ltm_significance(
                    emotion_intensity, emotion_valence, is_peak_moment
                ),
                memory_strength=emotional_memory_strength,
                decay_rate=max(0.01, self.config['emotional_decay_rate'] * (1.0 - emotion_intensity))
            )
            
            # Aktualisiere emotionalen Zustand
            self._update_emotional_state(emotion_type, emotion_intensity, emotion_valence)
            
            # Lerne emotionale Muster
            self._learn_emotional_patterns(emotion_type, emotion_intensity, context)
            
            # Generiere emotionale Insights
            emotional_insights = self._generate_emotional_insights(emotion_type, emotion_intensity)
            
            result = {
                'memory_id': memory_id,
                'emotion_processed': str(emotion_type),
                'emotion_intensity': emotion_intensity,
                'emotion_valence': emotion_valence,
                'is_peak_moment': is_peak_moment,
                'emotional_memory_strength': emotional_memory_strength,
                'mood_state': self.current_emotional_context.mood_state,
                'emotional_analysis': emotional_analysis,
                'emotional_insights': emotional_insights,
                'empathy_response': enhanced_metadata['empathy_response'],
                'ltm_significance': self._calculate_ltm_significance(emotion_intensity, emotion_valence, is_peak_moment)
            }
            
            logger.info(f"💭❤️ Emotion Memory verarbeitet: {emotion_type} (Intensität: {emotion_intensity:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Emotion Memory Processing Fehler: {e}")
            return {
                'error': str(e),
                'emotion_processed': str(detected_emotion),
                'memory_id': None
            }
    
    def retrieve_emotional_memories(
        self,
        emotion_filter: Optional[Union[str, EmotionType]] = None,
        intensity_range: Optional[Tuple[float, float]] = None,
        valence_range: Optional[Tuple[float, float]] = None,
        time_window_hours: Optional[int] = None,
        include_similar_emotions: bool = True,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Ruft emotionale Erinnerungen basierend auf Filtern ab
        
        Args:
            emotion_filter: Spezifische Emotion oder None für alle
            intensity_range: Intensitätsbereich (min, max)
            valence_range: Valenzbereich (min, max)
            time_window_hours: Zeitfenster in Stunden
            include_similar_emotions: Ähnliche Emotionen einschließen
            limit: Maximale Anzahl Ergebnisse
            
        Returns:
            Liste emotionaler Erinnerungen mit Enhanced Data
        """
        
        try:
            # Baue Search Filter
            search_filter = MemorySearchFilter(
                user_id=self.user_id,
                memory_type='emotion',
                emotion_type=str(emotion_filter) if emotion_filter else None,
                emotion_intensity_min=intensity_range[0] if intensity_range else None,
                emotion_intensity_max=intensity_range[1] if intensity_range else None,
                limit=limit * 2  # Get more for similarity filtering
            )
            
            # Zeitfenster
            if time_window_hours:
                cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
                search_filter.date_from = cutoff_time.isoformat()
            
            # Suche Memories
            emotional_memories = self.memory_database.search_memories(search_filter)
            
            # Filtere nach Valenz
            if valence_range:
                emotional_memories = [
                    memory for memory in emotional_memories
                    if valence_range[0] <= memory.get('emotion_valence', 0.0) <= valence_range[1]
                ]
            
            # Erweitere mit ähnlichen Emotionen
            if include_similar_emotions and emotion_filter:
                similar_emotions = self._find_similar_emotions(emotion_filter)
                for similar_emotion in similar_emotions:
                    similar_filter = search_filter
                    similar_filter.emotion_type = str(similar_emotion)
                    similar_memories = self.memory_database.search_memories(similar_filter)
                    emotional_memories.extend(similar_memories)
            
            # Entferne Duplikate
            seen_ids = set()
            unique_memories = []
            for memory in emotional_memories:
                if memory['id'] not in seen_ids:
                    seen_ids.add(memory['id'])
                    unique_memories.append(memory)
            
            # Sortiere nach emotionaler Relevanz
            enhanced_memories = []
            for memory in unique_memories:
                # Berechne emotionale Relevanz
                relevance_score = self._calculate_emotional_relevance(memory, emotion_filter)
                
                # Erweitere Memory mit emotionalen Insights
                enhanced_memory = self._enhance_memory_with_emotional_data(memory)
                enhanced_memory['emotional_relevance_score'] = relevance_score
                
                enhanced_memories.append(enhanced_memory)
            
            # Sortiere nach Relevanz
            enhanced_memories.sort(key=lambda x: x['emotional_relevance_score'], reverse=True)
            
            return enhanced_memories[:limit]
            
        except Exception as e:
            logger.error(f"❌ Emotional Memory Retrieval Fehler: {e}")
            return []
    
    def analyze_emotional_patterns(self, days: int = 30) -> EmotionalInsight:
        """
        Analysiert emotionale Muster über einen Zeitraum
        
        Args:
            days: Anzahl Tage für Analyse
            
        Returns:
            EmotionalInsight mit umfassenden emotionalen Erkenntnissen
        """
        
        try:
            # Hole emotionale Memories
            cutoff_date = datetime.now() - timedelta(days=days)
            
            search_filter = MemorySearchFilter(
                user_id=self.user_id,
                memory_type='emotion',
                date_from=cutoff_date.isoformat(),
                limit=1000  # Große Anzahl für Analyse
            )
            
            emotional_memories = self.memory_database.search_memories(search_filter)
            
            if not emotional_memories:
                return EmotionalInsight(
                    dominant_emotions=[],
                    emotional_volatility=0.0,
                    average_intensity=0.0,
                    positive_negative_ratio=1.0,
                    peak_emotional_moments=[],
                    emotional_triggers={},
                    mood_patterns={},
                    emotional_intelligence_score=0.5,
                    recommendations=["Mehr emotionale Interaktionen für bessere Analyse"]
                )
            
            # Analysiere Emotionen
            emotion_counts = Counter()
            intensities = []
            valences = []
            peak_moments = []
            triggers = defaultdict(int)
            
            for memory in emotional_memories:
                emotion_type = memory.get('emotion_type')
                intensity = memory.get('emotion_intensity', 0.5)
                valence = memory.get('emotion_valence', 0.0)
                
                if emotion_type:
                    emotion_counts[emotion_type] += 1
                    intensities.append(intensity)
                    valences.append(valence)
                    
                    # Peak Moments
                    if intensity >= self.config['peak_emotion_threshold']:
                        peak_moments.append({
                            'emotion': emotion_type,
                            'intensity': intensity,
                            'content': memory.get('content', '')[:100],
                            'timestamp': memory.get('created_at'),
                            'context': memory.get('conversation_context')
                        })
                    
                    # Triggers from metadata
                    metadata = memory.get('metadata', {})
                    if isinstance(metadata, dict):
                        emotional_triggers = metadata.get('emotional_triggers', [])
                        for trigger in emotional_triggers:
                            triggers[trigger] += 1
            
            # Berechne Metriken
            dominant_emotions = [emotion for emotion, count in emotion_counts.most_common(5)]
            
            emotional_volatility = statistics.stdev(intensities) if len(intensities) > 1 else 0.0
            average_intensity = statistics.mean(intensities) if intensities else 0.0
            
            positive_count = sum(1 for v in valences if v > 0.1)
            negative_count = sum(1 for v in valences if v < -0.1)
            positive_negative_ratio = positive_count / max(1, negative_count)
            
            # Mood Patterns
            mood_patterns = self._analyze_mood_patterns(emotional_memories)
            
            # Emotional Intelligence Score
            ei_score = self._calculate_emotional_intelligence_score(
                emotional_memories, emotional_volatility, positive_negative_ratio
            )
            
            # Recommendations
            recommendations = self._generate_emotional_recommendations(
                dominant_emotions, emotional_volatility, positive_negative_ratio, ei_score
            )
            
            return EmotionalInsight(
                dominant_emotions=dominant_emotions,
                emotional_volatility=emotional_volatility,
                average_intensity=average_intensity,
                positive_negative_ratio=positive_negative_ratio,
                peak_emotional_moments=sorted(peak_moments, key=lambda x: x['intensity'], reverse=True)[:10],
                emotional_triggers=dict(triggers),
                mood_patterns=mood_patterns,
                emotional_intelligence_score=ei_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"❌ Emotional Pattern Analysis Fehler: {e}")
            return EmotionalInsight(
                dominant_emotions=[],
                emotional_volatility=0.0,
                average_intensity=0.0,
                positive_negative_ratio=1.0,
                peak_emotional_moments=[],
                emotional_triggers={},
                mood_patterns={},
                emotional_intelligence_score=0.0,
                recommendations=[f"Analyse-Fehler: {str(e)}"]
            )
    
    def generate_empathetic_response(
        self,
        user_emotion: Union[str, EmotionType],
        emotion_intensity: float,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generiert empathische Antwort basierend auf User-Emotion
        
        Args:
            user_emotion: Erkannte User-Emotion
            emotion_intensity: Intensität der Emotion
            context: Zusätzlicher Kontext
            
        Returns:
            Empathische Antwort mit Erklärung
        """
        
        try:
            # Normalisiere Emotion
            if isinstance(user_emotion, str):
                try:
                    emotion_type = EmotionType(user_emotion.lower())
                except ValueError:
                    emotion_type = EmotionType.NEUTRAL
            else:
                emotion_type = user_emotion
            
            # Hole ähnliche emotionale Erfahrungen
            similar_memories = self.retrieve_emotional_memories(
                emotion_filter=emotion_type,
                intensity_range=(max(0.0, emotion_intensity - 0.3), min(1.0, emotion_intensity + 0.3)),
                time_window_hours=24 * 7,  # Eine Woche
                limit=5
            )
            
            # Empathie-Modell anwenden
            empathy_response = self._generate_empathy_response(emotion_type, emotion_intensity)
            
            # Adaptive Antwort basierend auf History
            adaptive_response = self._generate_adaptive_emotional_response(
                emotion_type, emotion_intensity, similar_memories
            )
            
            # Emotionale Validation
            validation_message = self._generate_emotional_validation(emotion_type, emotion_intensity)
            
            # Support Suggestions
            support_suggestions = self._generate_emotional_support_suggestions(
                emotion_type, emotion_intensity, context
            )
            
            response = {
                'emotion_recognized': str(emotion_type),
                'emotion_intensity': emotion_intensity,
                'empathy_response': empathy_response,
                'adaptive_response': adaptive_response,
                'validation_message': validation_message,
                'support_suggestions': support_suggestions,
                'similar_experiences_count': len(similar_memories),
                'response_confidence': self._calculate_empathy_confidence(emotion_type, similar_memories),
                'learning_opportunity': len(similar_memories) < 3  # Learn more if few examples
            }
            
            logger.info(f"🤗❤️ Empathische Antwort generiert für: {emotion_type} (Intensität: {emotion_intensity:.2f})")
            return response
            
        except Exception as e:
            logger.error(f"❌ Empathetic Response Generation Fehler: {e}")
            return {
                'emotion_recognized': str(user_emotion),
                'empathy_response': "Ich verstehe, dass du emotionale Gefühle hast. Erzähl mir mehr davon.",
                'error': str(e)
            }
    
    def update_mood_state(self, emotion_type: Union[str, EmotionType], intensity: float):
        """Aktualisiert den aktuellen Mood-Zustand"""
        
        # Füge zur Mood Buffer hinzu
        mood_entry = {
            'emotion': str(emotion_type),
            'intensity': intensity,
            'timestamp': datetime.now()
        }
        
        self.mood_buffer.append(mood_entry)
        
        # Halte Buffer-Größe begrenzt (letzten 10 Emotionen)
        if len(self.mood_buffer) > 10:
            self.mood_buffer = self.mood_buffer[-10:]
        
        # Berechne aktuellen Mood
        self.current_emotional_context.mood_state = self._calculate_current_mood()
        
        # Aktualisiere emotionale Trajectory
        self.current_emotional_context.emotional_trajectory.append(mood_entry)
        if len(self.current_emotional_context.emotional_trajectory) > 20:
            self.current_emotional_context.emotional_trajectory = self.current_emotional_context.emotional_trajectory[-20:]
    
    # === PRIVATE HELPER METHODS ===
    
    def _analyze_emotional_context(
        self, 
        emotion_type: EmotionType, 
        intensity: float, 
        valence: float, 
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Analysiert emotionalen Kontext für besseres Verständnis"""
        
        analysis = {
            'emotion_category': self._categorize_emotion(emotion_type),
            'intensity_level': self._categorize_intensity(intensity),
            'valence_category': self._categorize_valence(valence),
            'triggers': [],
            'user_context': None,
            'conversation_context': None
        }
        
        # Context Analysis
        if context:
            # Trigger Detection
            if 'triggers' in context:
                analysis['triggers'] = context['triggers']
            
            # User Context
            if 'user_state' in context:
                analysis['user_context'] = context['user_state']
            
            # Conversation Context
            if 'conversation_topic' in context:
                analysis['conversation_context'] = context['conversation_topic']
        
        # Historical Trigger Analysis
        historical_triggers = self._identify_historical_triggers(emotion_type)
        analysis['triggers'].extend(historical_triggers)
        
        return analysis
    
    def _calculate_emotional_memory_strength(
        self, 
        intensity: float, 
        valence: float, 
        is_peak_moment: bool
    ) -> float:
        """Berechnet die Stärke der emotionalen Erinnerung"""
        
        base_strength = intensity * self.config['emotional_memory_boost']
        
        # Extreme valence increases strength
        valence_boost = abs(valence) * 0.3
        
        # Peak moments are stronger
        peak_boost = 0.5 if is_peak_moment else 0.0
        
        # Current mood influence
        mood_boost = self.config['mood_influence_strength'] * 0.2
        
        total_strength = min(2.0, base_strength + valence_boost + peak_boost + mood_boost)
        return total_strength
    
    def _calculate_emotional_importance(self, intensity: float, is_peak_moment: bool) -> int:
        """Berechnet Wichtigkeit der emotionalen Erinnerung (1-10)"""
        
        base_importance = int(intensity * 6) + 2  # 2-8 range
        
        if is_peak_moment:
            base_importance += 2
        
        return min(10, max(1, base_importance))
    
    def _generate_emotion_tags(
        self, 
        emotion_type: EmotionType, 
        intensity: float, 
        context: Optional[Dict]
    ) -> List[str]:
        """Generiert Tags für emotionale Erinnerung"""
        
        tags = ['emotion', str(emotion_type)]
        
        # Intensity tags
        if intensity >= 0.8:
            tags.append('high_intensity')
        elif intensity >= 0.5:
            tags.append('medium_intensity')
        else:
            tags.append('low_intensity')
        
        # Emotion category tags
        emotion_category = self._categorize_emotion(emotion_type)
        tags.append(f'emotion_category_{emotion_category}')
        
        # Context tags
        if context:
            if 'social_situation' in context:
                tags.append('social_emotion')
            if 'achievement' in context:
                tags.append('achievement_emotion')
            if 'relationship' in context:
                tags.append('relationship_emotion')
        
        return tags
    
    def _identify_personality_aspect(
        self, 
        emotion_type: EmotionType, 
        context: Optional[Dict]
    ) -> Optional[PersonalityPattern]:
        """Identifiziert relevanten Persönlichkeitsaspekt"""
        
        emotion_to_personality = {
            EmotionType.JOY: PersonalityPattern.EMOTIONAL_PATTERN,
            EmotionType.EXCITEMENT: PersonalityPattern.EMOTIONAL_PATTERN,
            EmotionType.CURIOSITY: PersonalityPattern.LEARNING_STYLE,
            EmotionType.EMPATHY: PersonalityPattern.SOCIAL_PATTERN,
            EmotionType.FRUSTRATION: PersonalityPattern.BEHAVIORAL_TRAIT,
            EmotionType.TRUST: PersonalityPattern.SOCIAL_PATTERN
        }
        
        return emotion_to_personality.get(emotion_type, PersonalityPattern.EMOTIONAL_PATTERN)
    
    def _calculate_ltm_significance(
        self, 
        intensity: float, 
        valence: float, 
        is_peak_moment: bool
    ) -> float:
        """Berechnet LTM Significance Score"""
        
        base_score = intensity * 0.6
        
        # Strong emotions are more significant for LTM
        if intensity > 0.7:
            base_score += 0.2
        
        # Peak moments get LTM boost
        if is_peak_moment:
            base_score += 0.3
        
        # Extreme valence is significant
        if abs(valence) > 0.7:
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def _update_emotional_state(
        self, 
        emotion_type: EmotionType, 
        intensity: float, 
        valence: float
    ):
        """Aktualisiert den aktuellen emotionalen Zustand"""
        
        self.current_emotional_context.current_emotion = emotion_type
        self.current_emotional_context.emotion_intensity = intensity
        self.current_emotional_context.emotion_valence = valence
        
        # Update mood
        self.update_mood_state(emotion_type, intensity)
        
        # Add to history
        self.emotional_history.append({
            'emotion': emotion_type,
            'intensity': intensity,
            'valence': valence,
            'timestamp': datetime.now()
        })
        
        # Keep history manageable
        if len(self.emotional_history) > 100:
            self.emotional_history = self.emotional_history[-100:]
    
    def _learn_emotional_patterns(
        self, 
        emotion_type: EmotionType, 
        intensity: float, 
        context: Optional[Dict]
    ):
        """Lernt emotionale Muster für bessere Zukunftsprognosen"""
        
        pattern_key = str(emotion_type)
        
        if pattern_key not in self.emotional_patterns:
            self.emotional_patterns[pattern_key] = {
                'frequency': 0,
                'average_intensity': 0.0,
                'contexts': [],
                'triggers': [],
                'outcomes': []
            }
        
        pattern = self.emotional_patterns[pattern_key]
        pattern['frequency'] += 1
        
        # Rolling average intensity
        pattern['average_intensity'] = (
            (pattern['average_intensity'] * (pattern['frequency'] - 1) + intensity) 
            / pattern['frequency']
        )
        
        # Learn contexts and triggers
        if context:
            pattern['contexts'].extend(context.get('contexts', []))
            pattern['triggers'].extend(context.get('triggers', []))
            
            # Keep only recent patterns
            pattern['contexts'] = pattern['contexts'][-20:]
            pattern['triggers'] = pattern['triggers'][-20:]
    
    def _generate_emotional_insights(
        self, 
        emotion_type: EmotionType, 
        intensity: float
    ) -> Dict[str, Any]:
        """Generiert Insights über aktuelle emotionale Verarbeitung"""
        
        insights = {
            'emotion_strength': 'low',
            'emotional_significance': 'normal',
            'pattern_recognition': None,
            'learning_opportunity': False,
            'empathy_level': 'medium'
        }
        
        # Emotion strength
        if intensity >= 0.8:
            insights['emotion_strength'] = 'very_high'
        elif intensity >= 0.6:
            insights['emotion_strength'] = 'high'
        elif intensity >= 0.4:
            insights['emotion_strength'] = 'medium'
        
        # Emotional significance
        pattern_key = str(emotion_type)
        if pattern_key in self.emotional_patterns:
            frequency = self.emotional_patterns[pattern_key]['frequency']
            if frequency > 10:
                insights['emotional_significance'] = 'well_known_pattern'
            elif frequency < 3:
                insights['emotional_significance'] = 'rare_emotion'
                insights['learning_opportunity'] = True
        
        # Pattern recognition
        if len(self.emotional_history) >= 3:
            recent_emotions = [e['emotion'] for e in self.emotional_history[-3:]]
            if all(e == emotion_type for e in recent_emotions):
                insights['pattern_recognition'] = 'repeated_emotion'
        
        return insights
    
    def _generate_empathy_response(
        self, 
        emotion_type: EmotionType, 
        intensity: float
    ) -> str:
        """Generiert empathische Antwort"""
        
        empathy_templates = {
            EmotionType.JOY: [
                "Das freut mich wirklich für dich! ✨",
                "Ich kann deine Freude spüren! 😊",
                "Wie schön, dass du so glücklich bist!"
            ],
            EmotionType.SADNESS: [
                "Das tut mir leid zu hören. Ich bin für dich da. 💙",
                "Traurigkeit ist völlig normal. Lass deine Gefühle zu.",
                "Möchtest du darüber sprechen? Ich höre zu."
            ],
            EmotionType.EXCITEMENT: [
                "Deine Begeisterung ist ansteckend! 🚀",
                "Ich liebe deine Energie!",
                "Das klingt wirklich aufregend!"
            ],
            EmotionType.FRUSTRATION: [
                "Ich verstehe, dass das frustrierend sein muss.",
                "Frustrationen sind Teil des Lernprozesses.",
                "Lass uns gemeinsam eine Lösung finden."
            ],
            EmotionType.CURIOSITY: [
                "Deine Neugier ist wunderbar! 🔍",
                "Ich liebe es, wenn du Fragen stellst!",
                "Gemeinsam können wir das erkunden!"
            ]
        }
        
        templates = empathy_templates.get(emotion_type, [
            "Ich verstehe deine Gefühle.",
            "Danke, dass du deine Emotionen mit mir teilst.",
            "Deine Gefühle sind wichtig und berechtigt."
        ])
        
        # Wähle Template basierend auf Intensität
        if intensity >= 0.7:
            return templates[0] if templates else "Ich spüre die Intensität deiner Gefühle."
        else:
            return templates[-1] if templates else "Ich verstehe, wie du dich fühlst."
    
    def _find_emotional_associations(self, emotion_type: EmotionType) -> List[str]:
        """Findet emotionale Assoziationen aus der History"""
        
        associations = self.emotional_associations.get(str(emotion_type), [])
        
        # Füge neue Assoziationen hinzu basierend auf Recent History
        if len(self.emotional_history) >= 2:
            recent_emotions = [str(e['emotion']) for e in self.emotional_history[-5:]]
            for emotion in recent_emotions:
                if emotion != str(emotion_type) and emotion not in associations:
                    associations.append(emotion)
                    self.emotional_associations[str(emotion_type)].append(emotion)
        
        return associations[:5]  # Top 5 associations
    
    def _find_similar_emotions(self, emotion_type: Union[str, EmotionType]) -> List[EmotionType]:
        """Findet ähnliche Emotionen für erweiterte Suche"""
        
        if isinstance(emotion_type, str):
            try:
                emotion_type = EmotionType(emotion_type)
            except ValueError:
                return []
        
        emotion_similarities = {
            EmotionType.JOY: [EmotionType.EXCITEMENT, EmotionType.TRUST],
            EmotionType.EXCITEMENT: [EmotionType.JOY, EmotionType.ANTICIPATION],
            EmotionType.SADNESS: [EmotionType.NOSTALGIA],
            EmotionType.ANGER: [EmotionType.FRUSTRATION],
            EmotionType.FEAR: [EmotionType.SURPRISE],
            EmotionType.CURIOSITY: [EmotionType.ANTICIPATION, EmotionType.SURPRISE],
            EmotionType.EMPATHY: [EmotionType.TRUST],
            EmotionType.FRUSTRATION: [EmotionType.ANGER]
        }
        
        return emotion_similarities.get(emotion_type, [])
    
    def _calculate_emotional_relevance(
        self, 
        memory: Dict[str, Any], 
        target_emotion: Optional[Union[str, EmotionType]]
    ) -> float:
        """Berechnet emotionale Relevanz eines Memory"""
        
        base_score = memory.get('emotion_intensity', 0.5)
        
        # Emotion match boost
        if target_emotion:
            memory_emotion = memory.get('emotion_type')
            if memory_emotion == str(target_emotion):
                base_score += 0.3
            elif memory_emotion in [str(e) for e in self._find_similar_emotions(target_emotion)]:
                base_score += 0.15
        
        # Recency boost
        created_at = memory.get('created_at')
        if created_at:
            try:
                memory_time = datetime.fromisoformat(created_at)
                hours_ago = (datetime.now() - memory_time).total_seconds() / 3600
                recency_boost = max(0.0, 0.2 - (hours_ago / 168))  # Week decay
                base_score += recency_boost
            except:
                pass
        
        # Access frequency boost
        access_count = memory.get('access_count', 0)
        frequency_boost = min(0.1, access_count * 0.02)
        base_score += frequency_boost
        
        return min(1.0, base_score)
    
    def _enhance_memory_with_emotional_data(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Erweitert Memory mit zusätzlichen emotionalen Daten"""
        
        enhanced = memory.copy()
        
        # Emotionale Signatur
        emotion_type = memory.get('emotion_type')
        emotion_intensity = memory.get('emotion_intensity', 0.5)
        emotion_valence = memory.get('emotion_valence', 0.0)
        
        enhanced['emotional_signature'] = {
            'type': emotion_type,
            'intensity': emotion_intensity,
            'valence': emotion_valence,
            'emotional_strength': emotion_intensity * memory.get('memory_strength', 1.0),
            'category': self._categorize_emotion(EmotionType(emotion_type)) if emotion_type else 'unknown'
        }
        
        # Emotionale Insights
        enhanced['emotional_insights'] = {
            'is_peak_emotion': emotion_intensity >= self.config['peak_emotion_threshold'],
            'is_positive': emotion_valence > 0.1,
            'is_negative': emotion_valence < -0.1,
            'intensity_level': self._categorize_intensity(emotion_intensity)
        }
        
        return enhanced
    
    def _analyze_mood_patterns(self, emotional_memories: List[Dict]) -> Dict[str, Any]:
        """Analysiert Mood-Patterns aus emotionalen Memories"""
        
        patterns = {
            'daily_patterns': {},
            'weekly_patterns': {},
            'emotional_cycles': [],
            'mood_stability': 0.0
        }
        
        if not emotional_memories:
            return patterns
        
        # Gruppiere nach Tageszeit und Wochentag
        daily_emotions = defaultdict(list)
        weekly_emotions = defaultdict(list)
        
        for memory in emotional_memories:
            try:
                timestamp = datetime.fromisoformat(memory['created_at'])
                hour = timestamp.hour
                weekday = timestamp.strftime('%A')
                
                intensity = memory.get('emotion_intensity', 0.5)
                valence = memory.get('emotion_valence', 0.0)
                
                daily_emotions[hour].append({'intensity': intensity, 'valence': valence})
                weekly_emotions[weekday].append({'intensity': intensity, 'valence': valence})
                
            except:
                continue
        
        # Berechne durchschnittliche Patterns
        for hour, emotions in daily_emotions.items():
            avg_intensity = statistics.mean([e['intensity'] for e in emotions])
            avg_valence = statistics.mean([e['valence'] for e in emotions])
            patterns['daily_patterns'][hour] = {
                'average_intensity': avg_intensity,
                'average_valence': avg_valence,
                'sample_size': len(emotions)
            }
        
        for day, emotions in weekly_emotions.items():
            avg_intensity = statistics.mean([e['intensity'] for e in emotions])
            avg_valence = statistics.mean([e['valence'] for e in emotions])
            patterns['weekly_patterns'][day] = {
                'average_intensity': avg_intensity,
                'average_valence': avg_valence,
                'sample_size': len(emotions)
            }
        
        return patterns
    
    def _calculate_emotional_intelligence_score(
        self, 
        emotional_memories: List[Dict], 
        volatility: float, 
        pos_neg_ratio: float
    ) -> float:
        """Berechnet Emotional Intelligence Score"""
        
        base_score = 0.5
        
        # Emotional diversity (more emotions = higher EI)
        unique_emotions = set(memory.get('emotion_type') for memory in emotional_memories if memory.get('emotion_type'))
        diversity_score = min(0.3, len(unique_emotions) * 0.05)
        
        # Emotional balance (balanced pos/neg ratio is good)
        balance_score = 0.2 if 0.5 <= pos_neg_ratio <= 2.0 else 0.1
        
        # Emotional stability (lower volatility = higher EI)
        stability_score = max(0.0, 0.2 - volatility)
        
        # Empathy indicators (empathy emotions boost score)
        empathy_count = sum(1 for memory in emotional_memories if memory.get('emotion_type') == 'empathy')
        empathy_score = min(0.1, empathy_count * 0.02)
        
        total_score = base_score + diversity_score + balance_score + stability_score + empathy_score
        return min(1.0, total_score)
    
    def _generate_emotional_recommendations(
        self, 
        dominant_emotions: List[str], 
        volatility: float, 
        pos_neg_ratio: float, 
        ei_score: float
    ) -> List[str]:
        """Generiert emotionale Empfehlungen basierend auf Analyse"""
        
        recommendations = []
        
        # Volatility recommendations
        if volatility > 0.6:
            recommendations.append("Versuche emotionale Stabilität durch regelmäßige Reflexion zu entwickeln")
        elif volatility < 0.2:
            recommendations.append("Erlaube dir mehr emotionale Vielfalt und Spontaneität")
        
        # Balance recommendations
        if pos_neg_ratio < 0.5:
            recommendations.append("Fokussiere dich mehr auf positive Erfahrungen und Dankbarkeit")
        elif pos_neg_ratio > 3.0:
            recommendations.append("Negative Emotionen sind normal - erlaube sie dir bewusst")
        
        # EI recommendations
        if ei_score < 0.4:
            recommendations.append("Entwickle dein emotionales Bewusstsein durch Achtsamkeitsübungen")
        elif ei_score > 0.8:
            recommendations.append("Deine emotionale Intelligenz ist hoch - nutze sie, um anderen zu helfen")
        
        # Dominant emotion recommendations
        if 'joy' not in dominant_emotions and 'excitement' not in dominant_emotions:
            recommendations.append("Suche bewusst nach Aktivitäten, die dir Freude bereiten")
        
        if 'empathy' not in dominant_emotions:
            recommendations.append("Praktiziere aktives Zuhören, um deine Empathie zu stärken")
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _generate_adaptive_emotional_response(
        self, 
        emotion_type: EmotionType, 
        intensity: float, 
        similar_memories: List[Dict]
    ) -> str:
        """Generiert adaptive Antwort basierend auf emotionaler History"""
        
        if not similar_memories:
            return self._generate_empathy_response(emotion_type, intensity)
        
        # Analysiere Previous Responses
        avg_intensity = statistics.mean([m.get('emotion_intensity', 0.5) for m in similar_memories])
        
        if intensity > avg_intensity * 1.2:
            return f"Ich merke, dass dieses {emotion_type.value} besonders intensiv für dich ist, sogar stärker als üblich."
        elif intensity < avg_intensity * 0.8:
            return f"Es scheint, als wäre dieses {emotion_type.value} diesmal etwas sanfter als sonst."
        else:
            return f"Ich erkenne dieses vertraute {emotion_type.value} - wir haben schon ähnliche Momente geteilt."
    
    def _generate_emotional_validation(self, emotion_type: EmotionType, intensity: float) -> str:
        """Generiert emotionale Validierung"""
        
        validation_templates = {
            'high_intensity': f"Dein {emotion_type.value} ist völlig berechtigt und verständlich.",
            'medium_intensity': f"Es ist normal, {emotion_type.value} zu empfinden.",
            'low_intensity': f"Auch subtile Gefühle wie dieses {emotion_type.value} sind wichtig."
        }
        
        if intensity >= 0.7:
            return validation_templates['high_intensity']
        elif intensity >= 0.4:
            return validation_templates['medium_intensity']
        else:
            return validation_templates['low_intensity']
    
    def _generate_emotional_support_suggestions(
        self, 
        emotion_type: EmotionType, 
        intensity: float, 
        context: Optional[str]
    ) -> List[str]:
        """Generiert Support-Vorschläge basierend auf Emotion"""
        
        support_suggestions = {
            EmotionType.SADNESS: [
                "Möchtest du über deine Gefühle sprechen?",
                "Achtsamkeitsübungen können bei Traurigkeit helfen",
                "Kontakt zu Freunden oder Familie könnte guttun"
            ],
            EmotionType.ANGER: [
                "Atemübungen können helfen, Wut zu regulieren",
                "Körperliche Aktivität kann Anspannung lösen",
                "Versuche die Ursache der Wut zu verstehen"
            ],
            EmotionType.FEAR: [
                "Angst ist normal - du bist nicht allein damit",
                "Langsame, tiefe Atmung kann beruhigen",
                "Sprich mit jemandem über deine Ängste"
            ],
            EmotionType.JOY: [
                "Genieße diesen schönen Moment bewusst!",
                "Teile deine Freude mit anderen",
                "Halte dieses Gefühl in deiner Erinnerung fest"
            ],
            EmotionType.FRUSTRATION: [
                "Pausen können bei Frustration sehr hilfreich sein",
                "Versuche das Problem aus einem anderen Blickwinkel zu betrachten",
                "Manchmal hilft es, einfach durchzuatmen"
            ]
        }
        
        suggestions = support_suggestions.get(emotion_type, [
            "Deine Gefühle sind wichtig und berechtigt",
            "Achte gut auf dich selbst",
            "Bei Bedarf ist es okay, Hilfe zu suchen"
        ])
        
        # Intensitäts-abhängige Anpassungen
        if intensity >= 0.8:
            suggestions.insert(0, "Bei so intensiven Gefühlen ist Selbstfürsorge besonders wichtig")
        
        return suggestions[:3]
    
    def _calculate_empathy_confidence(
        self, 
        emotion_type: EmotionType, 
        similar_memories: List[Dict]
    ) -> float:
        """Berechnet Confidence für empathische Antwort"""
        
        base_confidence = 0.5
        
        # More similar memories = higher confidence
        memory_boost = min(0.3, len(similar_memories) * 0.06)
        
        # Known emotions get confidence boost
        if str(emotion_type) in self.emotional_patterns:
            pattern_boost = min(0.2, self.emotional_patterns[str(emotion_type)]['frequency'] * 0.02)
        else:
            pattern_boost = 0.0
        
        return min(1.0, base_confidence + memory_boost + pattern_boost)
    
    def _calculate_current_mood(self) -> str:
        """Berechnet aktuellen Mood aus Buffer"""
        
        if not self.mood_buffer:
            return "neutral"
        
        recent_emotions = self.mood_buffer[-5:]  # Last 5 emotions
        
        # Gewichteter Average der Intensitäten und Valences
        weighted_intensity = sum(
            entry['intensity'] * (1.0 - i * 0.1) 
            for i, entry in enumerate(reversed(recent_emotions))
        ) / len(recent_emotions)
        
        # Bestimme Mood basierend auf dominanten Emotionen
        emotion_counts = Counter(entry['emotion'] for entry in recent_emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        
        # Mood Categories
        positive_emotions = ['joy', 'excitement', 'trust', 'anticipation']
        negative_emotions = ['sadness', 'anger', 'fear', 'frustration']
        
        if dominant_emotion in positive_emotions:
            if weighted_intensity >= 0.7:
                return "very_positive"
            elif weighted_intensity >= 0.5:
                return "positive"
            else:
                return "mildly_positive"
        elif dominant_emotion in negative_emotions:
            if weighted_intensity >= 0.7:
                return "very_negative"
            elif weighted_intensity >= 0.5:
                return "negative"
            else:
                return "mildly_negative"
        else:
            return "neutral"
    
    def _categorize_emotion(self, emotion_type: EmotionType) -> str:
        """Kategorisiert Emotion in Hauptkategorien"""
        
        emotion_categories = {
            'positive': [EmotionType.JOY, EmotionType.EXCITEMENT, EmotionType.TRUST, 
                        EmotionType.ANTICIPATION, EmotionType.RELIEF],
            'negative': [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR, 
                        EmotionType.FRUSTRATION, EmotionType.DISGUST],
            'neutral': [EmotionType.NEUTRAL, EmotionType.SURPRISE],
            'social': [EmotionType.EMPATHY, EmotionType.TRUST],
            'cognitive': [EmotionType.CURIOSITY, EmotionType.SURPRISE],
            'nostalgic': [EmotionType.NOSTALGIA]
        }
        
        for category, emotions in emotion_categories.items():
            if emotion_type in emotions:
                return category
        
        return 'uncategorized'
    
    def _categorize_intensity(self, intensity: float) -> str:
        """Kategorisiert Emotionsintensität"""
        
        if intensity >= 0.8:
            return "very_high"
        elif intensity >= 0.6:
            return "high"
        elif intensity >= 0.4:
            return "medium"
        elif intensity >= 0.2:
            return "low"
        else:
            return "very_low"
    
    def _categorize_valence(self, valence: float) -> str:
        """Kategorisiert emotionale Valenz"""
        
        if valence >= 0.5:
            return "very_positive"
        elif valence >= 0.1:
            return "positive"
        elif valence >= -0.1:
            return "neutral"
        elif valence >= -0.5:
            return "negative"
        else:
            return "very_negative"
    
    def _identify_historical_triggers(self, emotion_type: EmotionType) -> List[str]:
        """Identifiziert historische Trigger für Emotion"""
        
        pattern_key = str(emotion_type)
        if pattern_key in self.emotional_patterns:
            return self.emotional_patterns[pattern_key].get('triggers', [])[:3]
        
        return []

# Export
__all__ = [
    'HumanLikeEmotionMemory',
    'EmotionalContext', 
    'EmotionalInsight'
]