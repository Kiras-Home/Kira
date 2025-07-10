"""
🔗 VOICE-MEMORY INTEGRATION BRIDGE
Verbindung zwischen Voice System und Memory System für kontextbewusste Gespräche
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class VoiceInteraction:
    """Voice interaction record for memory storage"""
    timestamp: datetime
    user_input: str
    user_emotion: Optional[str]
    kira_response: str
    response_emotion: str
    confidence_score: float
    processing_time: float
    context_used: Dict[str, Any]
    interaction_type: str  # 'wake_word', 'command', 'conversation', 'question'
    session_id: str
    importance_score: int  # 1-10 scale for memory retention


@dataclass
class ConversationContext:
    """Current conversation context"""
    session_id: str
    user_preferences: Dict[str, Any]
    conversation_history: List[VoiceInteraction]
    current_topic: Optional[str]
    user_mood: Optional[str]
    conversation_style: str
    last_interaction_time: datetime
    interaction_count: int


class VoiceMemoryBridge:
    """
    🔗 VOICE-MEMORY INTEGRATION BRIDGE
    Manages integration between voice interactions and memory system
    """
    
    def __init__(self, memory_service=None):
        self.memory_service = memory_service
        self.current_sessions = {}  # session_id -> ConversationContext
        self.interaction_history = []
        
        # Memory integration settings
        self.store_interactions = True
        self.min_importance_for_memory = 5  # Only store interactions with importance >= 5
        self.max_context_history = 10  # Keep last 10 interactions for context
        self.session_timeout = timedelta(minutes=30)  # Session expires after 30 min
        
        # Context tracking
        self.conversation_topics = []
        self.user_preferences = {}
        self.learned_patterns = {}
        
        logger.info("🔗 Voice-Memory Bridge initialized")
    
    def start_conversation_session(self, user_id: str = "default") -> str:
        """Start new conversation session"""
        try:
            session_id = f"voice_session_{int(time.time())}"
            
            # Load user preferences from memory if available
            user_prefs = self._load_user_preferences(user_id)
            
            # Create conversation context
            context = ConversationContext(
                session_id=session_id,
                user_preferences=user_prefs,
                conversation_history=[],
                current_topic=None,
                user_mood=None,
                conversation_style="friendly",
                last_interaction_time=datetime.now(),
                interaction_count=0
            )
            
            self.current_sessions[session_id] = context
            
            logger.info(f"🆕 Started conversation session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start conversation session: {e}")
            return "fallback_session"
    
    def end_conversation_session(self, session_id: str):
        """End conversation session and store to memory"""
        try:
            if session_id not in self.current_sessions:
                return
            
            context = self.current_sessions[session_id]
            
            # Store conversation summary to memory
            if self.memory_service and len(context.conversation_history) > 0:
                self._store_conversation_summary(context)
            
            # Store learned preferences
            self._update_user_preferences(context)
            
            # Cleanup
            del self.current_sessions[session_id]
            
            logger.info(f"🔚 Ended conversation session: {session_id} ({context.interaction_count} interactions)")
            
        except Exception as e:
            logger.error(f"Failed to end conversation session: {e}")
    
    def record_voice_interaction(
        self,
        session_id: str,
        user_input: str,
        kira_response: str,
        user_emotion: Optional[str] = None,
        response_emotion: str = "neutral",
        confidence_score: float = 1.0,
        processing_time: float = 0.0,
        interaction_type: str = "conversation"
    ) -> VoiceInteraction:
        """Record voice interaction for memory and context"""
        try:
            # Create interaction record
            interaction = VoiceInteraction(
                timestamp=datetime.now(),
                user_input=user_input,
                user_emotion=user_emotion,
                kira_response=kira_response,
                response_emotion=response_emotion,
                confidence_score=confidence_score,
                processing_time=processing_time,
                context_used=self._get_current_context(session_id),
                interaction_type=interaction_type,
                session_id=session_id,
                importance_score=self._calculate_importance_score(user_input, interaction_type, confidence_score)
            )
            
            # Update session context
            if session_id in self.current_sessions:
                context = self.current_sessions[session_id]
                context.conversation_history.append(interaction)
                context.last_interaction_time = datetime.now()
                context.interaction_count += 1
                
                # Keep only recent history for context
                if len(context.conversation_history) > self.max_context_history:
                    context.conversation_history = context.conversation_history[-self.max_context_history:]
                
                # Update conversation analysis
                self._update_conversation_analysis(context, interaction)
            
            # Store to memory if important enough
            if self.store_interactions and interaction.importance_score >= self.min_importance_for_memory:
                self._store_interaction_to_memory(interaction)
            
            logger.debug(f"📝 Recorded voice interaction (importance: {interaction.importance_score})")
            return interaction
            
        except Exception as e:
            logger.error(f"Failed to record voice interaction: {e}")
            return None
    
    def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get current conversation context"""
        return self.current_sessions.get(session_id)
    
    def get_contextual_prompt_enhancement(self, session_id: str, current_input: str) -> Dict[str, Any]:
        """Get context-enhanced prompt for better responses"""
        try:
            context = self.current_sessions.get(session_id)
            if not context:
                return {"context": "no_session"}
            
            # Analyze recent conversation
            recent_topics = self._extract_recent_topics(context)
            conversation_mood = self._analyze_conversation_mood(context)
            user_style = self._determine_user_communication_style(context)
            
            # Get relevant memories from memory system
            relevant_memories = []
            if self.memory_service:
                relevant_memories = self._get_relevant_memories(current_input, context)
            
            enhancement = {
                "conversation_context": {
                    "session_id": session_id,
                    "interaction_count": context.interaction_count,
                    "recent_topics": recent_topics,
                    "conversation_mood": conversation_mood,
                    "user_communication_style": user_style,
                    "last_interaction_time": context.last_interaction_time.isoformat(),
                    "current_topic": context.current_topic
                },
                "user_preferences": context.user_preferences,
                "relevant_memories": relevant_memories,
                "suggested_response_style": self._suggest_response_style(context, current_input),
                "emotional_context": {
                    "user_mood": context.user_mood,
                    "recommended_response_emotion": self._recommend_response_emotion(context, current_input)
                }
            }
            
            logger.debug(f"🎯 Generated contextual enhancement for session {session_id}")
            return enhancement
            
        except Exception as e:
            logger.error(f"Failed to get contextual enhancement: {e}")
            return {"context": "error"}
    
    def _load_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences from memory system"""
        try:
            if not self.memory_service:
                return {}
            
            # Get user preferences from memory
            preferences_query = f"user preferences {user_id} voice interaction style"
            memories = self.memory_service.search_memories(preferences_query, limit=5)
            
            preferences = {
                "formality_level": "medium",  # formal, medium, casual
                "response_length": "medium",   # short, medium, long
                "preferred_topics": [],
                "communication_style": "friendly",
                "language_preference": "german",
                "time_preferences": {},
                "learned_from_interactions": True
            }
            
            # Extract preferences from memories
            for memory in memories:
                if "preference" in memory.content.lower():
                    # Simple preference extraction
                    if "formal" in memory.content.lower():
                        preferences["formality_level"] = "formal"
                    elif "casual" in memory.content.lower():
                        preferences["formality_level"] = "casual"
            
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to load user preferences: {e}")
            return {}
    
    def _store_conversation_summary(self, context: ConversationContext):
        """Store conversation summary to memory"""
        try:
            if not self.memory_service or len(context.conversation_history) == 0:
                return
            
            # Create conversation summary
            summary_parts = []
            summary_parts.append(f"Sprachkonversation mit {context.interaction_count} Interaktionen")
            
            if context.current_topic:
                summary_parts.append(f"Hauptthema: {context.current_topic}")
            
            if context.user_mood:
                summary_parts.append(f"Benutzerstimmung: {context.user_mood}")
            
            # Add key interactions
            important_interactions = [i for i in context.conversation_history if i.importance_score >= 7]
            if important_interactions:
                summary_parts.append("Wichtige Interaktionen:")
                for interaction in important_interactions[-3:]:  # Last 3 important ones
                    summary_parts.append(f"- Benutzer: {interaction.user_input[:100]}")
                    summary_parts.append(f"  Kira: {interaction.kira_response[:100]}")
            
            summary_text = "\n".join(summary_parts)
            
            # Store to memory
            self.memory_service.store_interaction(
                input_text=f"Gespräch vom {context.last_interaction_time.strftime('%d.%m.%Y %H:%M')}",
                response=summary_text,
                context={"type": "voice_conversation_summary", "session_id": context.session_id},
                importance=6  # Medium importance for summaries
            )
            
            logger.info(f"💾 Stored conversation summary to memory (session: {context.session_id})")
            
        except Exception as e:
            logger.error(f"Failed to store conversation summary: {e}")
    
    def _store_interaction_to_memory(self, interaction: VoiceInteraction):
        """Store individual interaction to memory"""
        try:
            if not self.memory_service:
                return
            
            # Store interaction
            self.memory_service.store_interaction(
                input_text=interaction.user_input,
                response=interaction.kira_response,
                context={
                    "type": "voice_interaction",
                    "session_id": interaction.session_id,
                    "user_emotion": interaction.user_emotion,
                    "response_emotion": interaction.response_emotion,
                    "interaction_type": interaction.interaction_type,
                    "confidence": interaction.confidence_score
                },
                importance=interaction.importance_score
            )
            
            logger.debug(f"💾 Stored interaction to memory (importance: {interaction.importance_score})")
            
        except Exception as e:
            logger.error(f"Failed to store interaction to memory: {e}")
    
    def _calculate_importance_score(self, user_input: str, interaction_type: str, confidence: float) -> int:
        """Calculate importance score for memory storage (1-10)"""
        try:
            score = 5  # Base score
            
            # Interaction type modifiers
            type_scores = {
                "wake_word": 2,
                "command": 6,
                "question": 7,
                "conversation": 5,
                "problem_solving": 8,
                "learning": 9
            }
            score = type_scores.get(interaction_type, 5)
            
            # Content-based modifiers
            user_lower = user_input.lower()
            
            # Questions are more important
            if "?" in user_input or any(word in user_lower for word in ["wie", "was", "warum", "wo", "wann", "wer"]):
                score += 1
            
            # Personal information is very important
            if any(word in user_lower for word in ["mein", "ich", "mir", "mich", "persönlich", "privat"]):
                score += 2
            
            # Problems/errors are important
            if any(word in user_lower for word in ["problem", "fehler", "hilfe", "nicht verstehen"]):
                score += 2
            
            # Preferences are important for learning
            if any(word in user_lower for word in ["mag", "gefällt", "bevorzuge", "lieber", "möchte"]):
                score += 2
            
            # Confidence modifier
            if confidence < 0.5:
                score -= 1
            elif confidence > 0.8:
                score += 1
            
            # Length modifier (very short or very long inputs might be less important)
            if len(user_input) < 10:
                score -= 1
            elif len(user_input) > 200:
                score += 1
            
            return max(1, min(10, score))
            
        except Exception as e:
            logger.error(f"Failed to calculate importance score: {e}")
            return 5
    
    def _get_current_context(self, session_id: str) -> Dict[str, Any]:
        """Get current context for interaction"""
        context = self.current_sessions.get(session_id, {})
        if not context:
            return {}
        
        return {
            "session_id": session_id,
            "interaction_count": getattr(context, 'interaction_count', 0),
            "current_topic": getattr(context, 'current_topic', None),
            "user_mood": getattr(context, 'user_mood', None),
            "conversation_style": getattr(context, 'conversation_style', "friendly")
        }
    
    def _update_conversation_analysis(self, context: ConversationContext, interaction: VoiceInteraction):
        """Update conversation analysis based on new interaction"""
        try:
            # Update user mood based on detected emotion
            if interaction.user_emotion:
                context.user_mood = interaction.user_emotion
            
            # Extract and update current topic
            potential_topic = self._extract_topic_from_interaction(interaction)
            if potential_topic:
                context.current_topic = potential_topic
            
            # Update conversation style based on user communication
            style = self._analyze_communication_style(interaction.user_input)
            if style:
                context.conversation_style = style
            
        except Exception as e:
            logger.error(f"Failed to update conversation analysis: {e}")
    
    def _extract_recent_topics(self, context: ConversationContext) -> List[str]:
        """Extract recent conversation topics"""
        topics = []
        for interaction in context.conversation_history[-5:]:  # Last 5 interactions
            topic = self._extract_topic_from_interaction(interaction)
            if topic and topic not in topics:
                topics.append(topic)
        return topics
    
    def _extract_topic_from_interaction(self, interaction: VoiceInteraction) -> Optional[str]:
        """Extract topic from interaction"""
        try:
            user_input = interaction.user_input.lower()
            
            # Simple topic extraction based on keywords
            topic_keywords = {
                "weather": ["wetter", "temperatur", "regen", "sonne", "warm", "kalt"],
                "time": ["zeit", "uhr", "datum", "heute", "morgen", "gestern"],
                "technology": ["computer", "software", "app", "internet", "technologie"],
                "health": ["gesundheit", "krankheit", "arzt", "medizin", "schmerzen"],
                "work": ["arbeit", "job", "beruf", "büro", "meeting", "projekt"],
                "family": ["familie", "kinder", "eltern", "partner", "verwandte"],
                "hobbies": ["hobby", "sport", "musik", "film", "buch", "spiel"],
                "food": ["essen", "kochen", "restaurant", "rezept", "hunger"]
            }
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in user_input for keyword in keywords):
                    return topic
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract topic: {e}")
            return None
    
    def _analyze_conversation_mood(self, context: ConversationContext) -> str:
        """Analyze overall conversation mood"""
        if not context.conversation_history:
            return "neutral"
        
        # Analyze emotions from recent interactions
        recent_emotions = []
        for interaction in context.conversation_history[-5:]:
            if interaction.user_emotion:
                recent_emotions.append(interaction.user_emotion)
        
        if not recent_emotions:
            return "neutral"
        
        # Simple mood calculation
        positive_emotions = ["happy", "excited", "confident", "satisfied"]
        negative_emotions = ["sad", "frustrated", "angry", "worried"]
        
        positive_count = sum(1 for emotion in recent_emotions if emotion in positive_emotions)
        negative_count = sum(1 for emotion in recent_emotions if emotion in negative_emotions)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _determine_user_communication_style(self, context: ConversationContext) -> str:
        """Determine user's communication style"""
        if not context.conversation_history:
            return "standard"
        
        # Analyze recent messages
        total_length = 0
        question_count = 0
        formal_indicators = 0
        casual_indicators = 0
        
        for interaction in context.conversation_history[-5:]:
            user_input = interaction.user_input
            total_length += len(user_input)
            
            if "?" in user_input:
                question_count += 1
            
            # Check for formal language
            if any(word in user_input.lower() for word in ["sie", "ihnen", "bitte", "danke"]):
                formal_indicators += 1
            
            # Check for casual language
            if any(word in user_input.lower() for word in ["du", "dir", "hey", "ok", "cool"]):
                casual_indicators += 1
        
        avg_length = total_length / len(context.conversation_history[-5:])
        
        # Determine style
        if formal_indicators > casual_indicators:
            return "formal"
        elif casual_indicators > formal_indicators:
            return "casual"
        elif avg_length > 100:
            return "detailed"
        elif question_count > 2:
            return "inquisitive"
        else:
            return "standard"
    
    def _analyze_communication_style(self, user_input: str) -> Optional[str]:
        """Analyze communication style from single input"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["sie", "ihnen", "bitte schön"]):
            return "formal"
        elif any(word in user_lower for word in ["du", "hey", "cool", "ok"]):
            return "casual"
        elif len(user_input) > 100:
            return "detailed"
        elif "?" in user_input:
            return "questioning"
        else:
            return None
    
    def _get_relevant_memories(self, current_input: str, context: ConversationContext) -> List[Dict[str, Any]]:
        """Get relevant memories from memory system"""
        try:
            if not self.memory_service:
                return []
            
            # Search for relevant memories
            memories = self.memory_service.search_memories(current_input, limit=3)
            
            # Also search by current topic
            if context.current_topic:
                topic_memories = self.memory_service.search_memories(context.current_topic, limit=2)
                memories.extend(topic_memories)
            
            # Convert to serializable format
            relevant_memories = []
            for memory in memories[:5]:  # Limit to 5 most relevant
                relevant_memories.append({
                    "content": memory.content[:200],  # Truncate for context
                    "importance": memory.importance,
                    "timestamp": memory.timestamp.isoformat() if hasattr(memory, 'timestamp') else None,
                    "relevance_score": getattr(memory, 'relevance_score', 0.5)
                })
            
            return relevant_memories
            
        except Exception as e:
            logger.error(f"Failed to get relevant memories: {e}")
            return []
    
    def _suggest_response_style(self, context: ConversationContext, current_input: str) -> str:
        """Suggest response style based on context"""
        # Base style on conversation context
        user_style = self._determine_user_communication_style(context)
        
        if context.user_mood == "negative":
            return "empathetic"
        elif context.user_mood == "positive":
            return "enthusiastic"
        elif user_style == "formal":
            return "professional"
        elif user_style == "casual":
            return "friendly"
        elif "?" in current_input:
            return "helpful"
        else:
            return "standard"
    
    def _recommend_response_emotion(self, context: ConversationContext, current_input: str) -> str:
        """Recommend response emotion based on context"""
        # Emotion mirroring and context awareness
        if context.user_mood:
            mood_to_emotion = {
                "positive": "happy",
                "negative": "caring",
                "neutral": "neutral"
            }
            base_emotion = mood_to_emotion.get(context.user_mood, "neutral")
        else:
            base_emotion = "neutral"
        
        # Adjust based on current input
        current_lower = current_input.lower()
        if any(word in current_lower for word in ["hilfe", "problem", "verstehe nicht"]):
            return "helpful"
        elif any(word in current_lower for word in ["toll", "super", "danke"]):
            return "happy"
        elif any(word in current_lower for word in ["traurig", "schlecht", "ärger"]):
            return "caring"
        else:
            return base_emotion
    
    def _update_user_preferences(self, context: ConversationContext):
        """Update user preferences based on conversation"""
        try:
            # Learn preferences from conversation patterns
            if context.interaction_count < 3:
                return  # Need enough interactions to learn
            
            # Update formality preference
            formal_count = 0
            casual_count = 0
            
            for interaction in context.conversation_history:
                if any(word in interaction.user_input.lower() for word in ["sie", "ihnen"]):
                    formal_count += 1
                elif any(word in interaction.user_input.lower() for word in ["du", "dir"]):
                    casual_count += 1
            
            if formal_count > casual_count:
                context.user_preferences["formality_level"] = "formal"
            elif casual_count > formal_count:
                context.user_preferences["formality_level"] = "casual"
            
            # Store learned preferences
            self.user_preferences.update(context.user_preferences)
            
        except Exception as e:
            logger.error(f"Failed to update user preferences: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get memory bridge status"""
        active_sessions = len(self.current_sessions)
        total_interactions = sum(len(ctx.conversation_history) for ctx in self.current_sessions.values())
        
        return {
            "memory_service_connected": self.memory_service is not None,
            "active_sessions": active_sessions,
            "total_interactions_in_memory": len(self.interaction_history),
            "total_current_interactions": total_interactions,
            "settings": {
                "store_interactions": self.store_interactions,
                "min_importance_for_memory": self.min_importance_for_memory,
                "max_context_history": self.max_context_history,
                "session_timeout_minutes": int(self.session_timeout.total_seconds() / 60)
            },
            "learned_preferences": len(self.user_preferences),
            "conversation_topics": len(self.conversation_topics)
        }
    
    def cleanup_expired_sessions(self):
        """Cleanup expired conversation sessions"""
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, context in self.current_sessions.items():
                if current_time - context.last_interaction_time > self.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self.end_conversation_session(session_id)
            
            if expired_sessions:
                logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")


# Export classes
__all__ = [
    'VoiceMemoryBridge',
    'VoiceInteraction',
    'ConversationContext'
]