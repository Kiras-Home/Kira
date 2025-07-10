from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import logging
import uuid


from .memory_types import Memory, MemoryType, EmotionalState, MemoryImportance, create_memory
from .short_term_memory import HumanLikeShortTermMemory
from .long_term_memory import HumanLikeLongTermMemory

# ✅ NEUE STORAGE IMPORTS
try:
    from ..storage.postgresql_storage import PostgreSQLMemoryStorage
    from ..storage.memory_database import EnhancedMemoryDatabase
    STORAGE_AVAILABLE = True
except ImportError:
    PostgreSQLMemoryStorage = None
    EnhancedMemoryDatabase = None
    STORAGE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Context für eine laufende Conversation"""
    conversation_id: str = field(default_factory=lambda: f"conv_{uuid.uuid4().hex[:8]}")
    user_id: str = "default"
    session_id: str = "main"
    interaction_count: int = 0
    emotional_tone: str = "neutral"
    user_engagement: float = 0.5
    topic_category: str = "general"
    conversation_type: str = "casual"
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
        self.interaction_count += 1

class ConversationImportance(Enum):
    """Importance levels für Conversations"""
    TRIVIAL = 1
    LOW = 3
    MEDIUM = 5
    HIGH = 7
    CRITICAL = 9
    PERMANENT = 10

class ConversationTopic(Enum):
    """Topic categories für Conversations"""
    GENERAL = "general"
    PERSONAL = "personal"
    TECHNICAL = "technical"
    EMOTIONAL = "emotional"
    LEARNING = "learning"
    PLANNING = "planning"
    PROBLEM_SOLVING = "problem_solving"
    ENTERTAINMENT = "entertainment"
    WORK = "work"
    RELATIONSHIPS = "relationships"


class ConversationMemorySystem:
    """
    Intelligentes System für Gespräch-Speicherung
    Entscheidet automatisch zwischen STM und LTM
    """
    
    def __init__(self, 
                 stm_system: Optional[HumanLikeShortTermMemory] = None,
                 ltm_system: Optional[HumanLikeLongTermMemory] = None,
                 memory_database=None,
                 conversation_storage=None):  # ✅ OPTIONAL PARAMS
        """Initialisiert Conversation Memory System"""
        
        # ✅ AUTO-INITIALIZE STM/LTM wenn nicht gegeben
        if stm_system is None:
            try:
                self.stm = HumanLikeShortTermMemory(capacity=7)
                logger.info("✅ STM auto-initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not auto-initialize STM: {e}")
                self.stm = None
        else:
            self.stm = stm_system
            
        if ltm_system is None:
            try:
                self.ltm = HumanLikeLongTermMemory()
                logger.info("✅ LTM auto-initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not auto-initialize LTM: {e}")
                self.ltm = None
        else:
            self.ltm = ltm_system
            
        self.memory_database = memory_database
        
        # ✅ CONVERSATION STORAGE SYSTEM
        self.conversation_storage = conversation_storage
        if not self.conversation_storage and STORAGE_AVAILABLE:
            try:
                # Auto-initialize conversation storage
                self.conversation_storage = PostgreSQLMemoryStorage()
                logger.info("✅ Conversation storage auto-initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not auto-initialize conversation storage: {e}")
        
        # Conversation Tracking
        self.current_conversation: Optional[ConversationContext] = None
        self.conversation_buffer: List[Dict[str, Any]] = []
        self.conversation_history: Dict[str, List[Memory]] = {}
        
        # ✅ CONVERSATION PERSISTENCE SETTINGS
        self.persistence_settings = {
            'store_all_conversations': True,      # Speichere alle Gespräche
            'store_important_only': False,       # Oder nur wichtige
            'importance_threshold': 5.0,         # Schwellenwert für wichtige Gespräche
            'buffer_flush_size': 10,             # Buffer größe vor Database flush
            'auto_cleanup_days': 30              # Auto-cleanup nach X Tagen
        }
        
        # Intelligence Thresholds
        self.ltm_thresholds = {
            'importance_min': 6,
            'emotional_intensity_min': 0.6,
            'personal_relevance_min': 0.7,
            'learning_value_min': 0.6,
            'follow_up_threshold': 2
        }
        
        # Pattern Recognition
        self.importance_keywords = {
            'high': ['important', 'critical', 'urgent', 'remember', 'significant', 'key', 'essential'],
            'personal': ['feel', 'think', 'believe', 'want', 'need', 'hope', 'fear', 'love'],
            'learning': ['learn', 'understand', 'explain', 'teach', 'know', 'discover', 'realize'],
            'planning': ['plan', 'future', 'goal', 'project', 'tomorrow', 'next', 'schedule'],
            'problem': ['problem', 'issue', 'challenge', 'difficulty', 'solve', 'fix', 'help']
        }
        
        logger.info("✅ Conversation Memory System initialized with database storage")


    async def _store_memories_intelligently(self, 
                                          user_memory: Memory, 
                                          kira_memory: Memory,
                                          storage_decisions: Dict[str, str]) -> Dict[str, Any]:
        """Speichert Memories basierend auf intelligenten Entscheidungen"""
        
        result = {
            'user_memory_stored': False,
            'kira_memory_stored': False,
            'storage_location': 'none',
            'memory_ids': [],
            'database_stored': False
        }
        
        try:
            # Store User Memory
            user_decision = storage_decisions['user_memory']
            
            if user_decision == 'ltm' and self.ltm:
                # Direkt zu LTM (bypass STM)
                ltm_result = await self.ltm.consolidate_from_stm([user_memory])
                if ltm_result['consolidated'] > 0:
                    result['user_memory_stored'] = True
                    result['memory_ids'].append(f"ltm_user_{user_memory.memory_id}")
                    
            elif user_decision == 'stm_priority' and self.stm:
                # STM mit hoher Priorität für spätere Konsolidierung
                stm_result = self.stm.process_experience(
                    user_memory.content, 
                    {**user_memory.context, 'consolidation_priority': True}
                )
                if stm_result['working_memory_updated']:
                    result['user_memory_stored'] = True
                    result['memory_ids'].append(f"stm_priority_user_{user_memory.memory_id}")
                    
            else:  # Regular STM
                if self.stm:
                    stm_result = self.stm.process_experience(user_memory.content, user_memory.context)
                    if stm_result['working_memory_updated']:
                        result['user_memory_stored'] = True
                        result['memory_ids'].append(f"stm_user_{user_memory.memory_id}")
            
            # Store Kira Memory
            kira_decision = storage_decisions['kira_memory']
            
            if kira_decision == 'ltm' and self.ltm:
                ltm_result = await self.ltm.consolidate_from_stm([kira_memory])
                if ltm_result['consolidated'] > 0:
                    result['kira_memory_stored'] = True
                    result['memory_ids'].append(f"ltm_kira_{kira_memory.memory_id}")
                    
            else:  # STM
                if self.stm:
                    stm_result = self.stm.process_experience(kira_memory.content, kira_memory.context)
                    if stm_result['working_memory_updated']:
                        result['kira_memory_stored'] = True
                        result['memory_ids'].append(f"stm_kira_{kira_memory.memory_id}")
            
            # Determine overall storage location
            if 'ltm' in [user_decision, kira_decision]:
                if user_decision == 'ltm' and kira_decision == 'ltm':
                    result['storage_location'] = 'both_ltm'
                else:
                    result['storage_location'] = 'mixed'
            else:
                result['storage_location'] = 'stm'
            
            # ✅ STORE IN DATABASE - ERWEITERT
            database_result = await self._store_conversation_in_database(
                user_memory, kira_memory, storage_decisions, result
            )
            result['database_stored'] = database_result
            
            logger.debug(f"✅ Memories stored: {result['storage_location']}, IDs: {result['memory_ids']}, DB: {database_result}")
            
        except Exception as e:
            logger.error(f"❌ Memory storage failed: {e}")
            result['error'] = str(e)
            
        return result

    # ✅ NEUE ERWEITERTE DATABASE STORAGE METHODE
    async def _store_conversation_in_database(self, 
                                            user_memory: Memory, 
                                            kira_memory: Memory,
                                            storage_decisions: Dict[str, str],
                                            processing_result: Dict[str, Any]) -> bool:
        """Erweiterte Database Storage für Conversations"""
        try:
            # 1. Check if we should store this conversation
            should_store = self._should_store_conversation(processing_result)
            
            if not should_store:
                logger.debug("Conversation not stored (below threshold)")
                return False
            
            # 2. Prepare conversation record
            conversation_record = await self._prepare_conversation_record(
                user_memory, kira_memory, storage_decisions, processing_result
            )
            
            # 3. Store in conversation storage if available
            if self.conversation_storage:
                storage_result = await self._store_in_conversation_storage(conversation_record)
                if storage_result:
                    logger.info(f"✅ Conversation stored in database: {conversation_record['conversation_id']}")
                    return True
            
            # 4. Fallback: Store in regular memory database
            if self.memory_database:
                await self._store_in_memory_database(user_memory, kira_memory, storage_decisions)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"❌ Database conversation storage failed: {e}")
            return False

    def _should_store_conversation(self, processing_result: Dict[str, Any]) -> bool:
        """Entscheidet ob Conversation gespeichert werden soll"""
        if self.persistence_settings['store_all_conversations']:
            return True
            
        if self.persistence_settings['store_important_only']:
            importance = processing_result.get('importance_score', 0)
            return importance >= self.persistence_settings['importance_threshold']
            
        return True  # Default: store everything

    async def _prepare_conversation_record(self, 
                                         user_memory: Memory, 
                                         kira_memory: Memory,
                                         storage_decisions: Dict[str, str],
                                         processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Bereitet Conversation Record für Database vor"""
        
        conversation_id = user_memory.context.get('conversation_id', f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        return {
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'user_input': user_memory.content.replace("User said: ", ""),
            'kira_response': kira_memory.content.replace("Kira responded: ", ""),
            'importance_score': processing_result.get('importance_score', 0),
            'emotional_impact': processing_result.get('emotional_impact', 0),
            'storage_location': processing_result.get('storage_location', 'stm'),
            'storage_reason': storage_decisions.get('reason', 'unknown'),
            'topic_category': user_memory.context.get('topic_category', 'smalltalk'),
            'conversation_type': user_memory.context.get('conversation_type', 'casual'),
            'key_indicators': user_memory.context.get('key_indicators', []),
            'user_memory_id': user_memory.memory_id,
            'kira_memory_id': kira_memory.memory_id,
            'personal_relevance': user_memory.context.get('personal_relevance', 0),
            'learning_value': user_memory.context.get('learning_value', 0),
            'user_name': user_memory.context.get('user_name', 'User'),
            'session_context': {
                'interaction_count': self.current_conversation.interaction_count if self.current_conversation else 1,
                'emotional_tone': self.current_conversation.emotional_tone if self.current_conversation else 'neutral',
                'user_engagement': self.current_conversation.user_engagement if self.current_conversation else 0.5
            }
        }

    async def _store_in_conversation_storage(self, conversation_record: Dict[str, Any]) -> bool:
        """Speichert in spezialisierter Conversation Storage"""
        try:
            if hasattr(self.conversation_storage, 'store_conversation'):
                await self.conversation_storage.store_conversation(conversation_record)
                return True
            elif hasattr(self.conversation_storage, 'store_enhanced_memory'):
                # Konvertiere zu Memory für Storage
                conversation_memory = self._conversation_record_to_memory(conversation_record)
                await self.conversation_storage.store_enhanced_memory(conversation_memory)
                return True
            else:
                logger.warning("Conversation storage has no compatible storage method")
                return False
                
        except Exception as e:
            logger.error(f"Conversation storage failed: {e}")
            return False

    def _conversation_record_to_memory(self, record: Dict[str, Any]) -> Memory:
        """Konvertiert Conversation Record zu Memory für Storage"""
        return create_memory(
            content=f"Conversation: User: {record['user_input']} | Kira: {record['kira_response']}",
            memory_type=MemoryType.CONVERSATION,
            importance=int(record['importance_score']),
            emotional_intensity=record['emotional_impact'],
            context={
                'conversation_id': record['conversation_id'],
                'storage_type': 'conversation_record',
                'topic_category': record['topic_category'],
                'conversation_type': record['conversation_type'],
                'storage_location': record['storage_location'],
                'storage_reason': record['storage_reason'],
                'user_name': record['user_name'],
                'session_context': record['session_context']
            },
            tags=['conversation', 'full_exchange', record['topic_category'], record['conversation_type']]
        )

    async def _store_in_memory_database(self, user_memory: Memory, kira_memory: Memory, storage_decisions: Dict[str, str]):
        """Speichert in regulärer Memory Database (Fallback)"""
        try:
            # Erweitere Memories mit Storage-Informationen
            user_memory.context['storage_decision'] = storage_decisions['user_memory']
            user_memory.context['storage_reason'] = storage_decisions['reason']
            user_memory.context['stored_at'] = datetime.now().isoformat()
            
            kira_memory.context['storage_decision'] = storage_decisions['kira_memory']  
            kira_memory.context['storage_reason'] = storage_decisions['reason']
            kira_memory.context['stored_at'] = datetime.now().isoformat()
            
            # Speichere in Database
            if hasattr(self.memory_database, 'store_enhanced_memory'):
                await self.memory_database.store_enhanced_memory(user_memory)
                await self.memory_database.store_enhanced_memory(kira_memory)
            else:
                # Fallback für einfachere Database
                await self.memory_database.store_memory(user_memory)
                await self.memory_database.store_memory(kira_memory)
            
        except Exception as e:
            logger.error(f"Memory database storage failed: {e}")

    # ✅ ERWEITERTE CONVERSATION BUFFER MIT DATABASE SYNC
    async def _update_conversation_buffer(self, user_input: str, kira_response: str, processing_result: Dict[str, Any]):
        """Aktualisiert den Conversation Buffer und synct mit Database"""
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'kira_response': kira_response,
            'importance_score': processing_result['importance_score'],
            'emotional_impact': processing_result.get('emotional_impact', 0),
            'storage_location': processing_result['storage_location'],
            'memory_ids': processing_result['memory_ids'],
            'database_stored': processing_result.get('database_stored', False)
        }
        
        self.conversation_buffer.append(conversation_entry)
        
        # Buffer-based database flush
        if len(self.conversation_buffer) >= self.persistence_settings['buffer_flush_size']:
            await self._flush_conversation_buffer()
        
        # Keep buffer size manageable
        if len(self.conversation_buffer) > 50:
            self.conversation_buffer = self.conversation_buffer[-30:]

    async def _flush_conversation_buffer(self):
        """Flusht Conversation Buffer zur Database"""
        try:
            if not self.conversation_storage:
                return
                
            # Batch store conversations that aren't stored yet
            unstored_conversations = [
                conv for conv in self.conversation_buffer 
                if not conv.get('database_stored', False)
            ]
            
            if unstored_conversations and hasattr(self.conversation_storage, 'store_conversations_batch'):
                # Batch storage wenn verfügbar
                batch_records = []
                for conv in unstored_conversations:
                    record = {
                        'conversation_id': f"conv_{conv['timestamp']}",
                        'timestamp': conv['timestamp'],
                        'user_input': conv['user_input'],
                        'kira_response': conv['kira_response'],
                        'importance_score': conv['importance_score'],
                        'emotional_impact': conv['emotional_impact'],
                        'storage_location': conv['storage_location']
                    }
                    batch_records.append(record)
                
                await self.conversation_storage.store_conversations_batch(batch_records)
                
                # Mark as stored
                for conv in unstored_conversations:
                    conv['database_stored'] = True
                    
                logger.info(f"✅ Flushed {len(batch_records)} conversations to database")
                
        except Exception as e:
            logger.error(f"Conversation buffer flush failed: {e}")

    # ✅ NEUE PUBLIC API METHODS FÜR DATABASE INTEGRATION

    def start_conversation(self, user_id: str = "default", session_id: str = "main", 
                          topic_category: str = "general") -> ConversationContext:
        """Startet eine neue Conversation"""
        self.current_conversation = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            topic_category=topic_category
        )
        logger.info(f"✅ Started conversation: {self.current_conversation.conversation_id}")
        return self.current_conversation
    
    async def process_conversation_exchange(self, 
                                          user_input: str, 
                                          kira_response: str,
                                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Verarbeitet einen Conversation Exchange"""
        try:
            # Ensure we have a conversation context
            if not self.current_conversation:
                self.start_conversation()
            
            # Update conversation activity
            self.current_conversation.update_activity()
            
            # Create memories
            user_memory = self._create_user_memory(user_input, context or {})
            kira_memory = self._create_kira_memory(kira_response, context or {})
            
            # Analyze importance and make storage decisions
            importance_analysis = self._analyze_conversation_importance(user_input, kira_response)
            storage_decisions = self._make_storage_decisions(importance_analysis)
            
            # Store memories intelligently
            storage_result = await self._store_memories_intelligently(
                user_memory, kira_memory, storage_decisions
            )
            
            # Update conversation buffer
            await self._update_conversation_buffer(user_input, kira_response, {
                **importance_analysis,
                **storage_result
            })
            
            result = {
                'conversation_id': self.current_conversation.conversation_id,
                'user_memory_id': user_memory.memory_id,
                'kira_memory_id': kira_memory.memory_id,
                'importance_analysis': importance_analysis,
                'storage_decisions': storage_decisions,
                'storage_result': storage_result,
                'interaction_count': self.current_conversation.interaction_count
            }
            
            logger.info(f"✅ Processed conversation exchange: {self.current_conversation.conversation_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process conversation exchange: {e}")
            return {
                'error': str(e),
                'conversation_id': self.current_conversation.conversation_id if self.current_conversation else None
            }

    def _create_user_memory(self, user_input: str, context: Dict[str, Any]) -> Memory:
        """Erstellt Memory für User Input"""
        return create_memory(
            content=f"User said: {user_input}",
            memory_type=MemoryType.CONVERSATION,
            importance=context.get('importance', 5),
            emotional_intensity=context.get('emotional_intensity', 0.0),
            context={
                **context,
                'conversation_id': self.current_conversation.conversation_id,
                'speaker': 'user',
                'user_name': context.get('user_name', 'User'),
                'timestamp': datetime.now().isoformat()
            },
            tags=['user_input', 'conversation', context.get('topic_category', 'general')]
        )
    
    def _analyze_conversation_importance(self, user_input: str, kira_response: str) -> Dict[str, Any]:
        """Analysiert die Wichtigkeit einer Conversation"""
        
        # Basic keyword matching
        importance_score = 5  # Default
        emotional_impact = 0.0
        topic_category = "general"
        key_indicators = []
        
        combined_text = f"{user_input} {kira_response}".lower()
        
        # Check for importance keywords
        for category, keywords in self.importance_keywords.items():
            matches = [kw for kw in keywords if kw in combined_text]
            if matches:
                key_indicators.extend(matches)
                if category == 'high':
                    importance_score += 2
                    emotional_impact += 0.2
                elif category == 'personal':
                    importance_score += 1
                    emotional_impact += 0.3
                    topic_category = "personal"
                elif category == 'learning':
                    importance_score += 1
                    topic_category = "learning"
                elif category == 'planning':
                    importance_score += 1
                    topic_category = "planning"
                elif category == 'problem':
                    importance_score += 2
                    topic_category = "problem_solving"
        
        # Length-based importance
        if len(user_input) > 100 or len(kira_response) > 100:
            importance_score += 1
        
        # Question patterns
        if '?' in user_input:
            importance_score += 1
            key_indicators.append('question')
        
        # Clamp values
        importance_score = min(10, max(1, importance_score))
        emotional_impact = min(1.0, max(0.0, emotional_impact))
        
        return {
            'importance_score': importance_score,
            'emotional_impact': emotional_impact,
            'topic_category': topic_category,
            'key_indicators': key_indicators,
            'personal_relevance': emotional_impact * 0.8,
            'learning_value': 0.5 if topic_category == "learning" else 0.2,
            'conversation_type': 'detailed' if importance_score >= 7 else 'casual'
        }
    
    def _make_storage_decisions(self, importance_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Entscheidet über Storage Location"""
        
        importance = importance_analysis['importance_score']
        emotional_impact = importance_analysis['emotional_impact']
        personal_relevance = importance_analysis['personal_relevance']
        learning_value = importance_analysis['learning_value']
        
        # Decision logic
        if (importance >= self.ltm_thresholds['importance_min'] or
            emotional_impact >= self.ltm_thresholds['emotional_intensity_min'] or
            personal_relevance >= self.ltm_thresholds['personal_relevance_min'] or
            learning_value >= self.ltm_thresholds['learning_value_min']):
            
            user_decision = 'ltm'
            kira_decision = 'ltm'
            reason = 'High importance/emotional/personal relevance'
            
        elif importance >= 6:
            user_decision = 'stm_priority'
            kira_decision = 'stm_priority'
            reason = 'Medium importance - STM priority for consolidation'
            
        else:
            user_decision = 'stm'
            kira_decision = 'stm'
            reason = 'Standard importance - STM storage'
        
        return {
            'user_memory': user_decision,
            'kira_memory': kira_decision,
            'reason': reason
        }
    
    def _create_kira_memory(self, kira_response: str, context: Dict[str, Any]) -> Memory:
        """Erstellt Memory für Kira Response"""
        return create_memory(
            content=f"Kira responded: {kira_response}",
            memory_type=MemoryType.CONVERSATION,
            importance=context.get('importance', 5),
            emotional_intensity=context.get('emotional_intensity', 0.0),
            context={
                **context,
                'conversation_id': self.current_conversation.conversation_id,
                'speaker': 'kira',
                'timestamp': datetime.now().isoformat()
            },
            tags=['kira_response', 'conversation', context.get('topic_category', 'general')]
        )

    async def get_conversation_history(self, 
                                     user_name: str = None, 
                                     limit: int = 50,
                                     importance_min: float = 0.0) -> List[Dict[str, Any]]:
        """Holt Conversation History aus Database"""
        try:
            if not self.conversation_storage:
                # Fallback zu Buffer
                filtered_conversations = [
                    conv for conv in self.conversation_buffer
                    if conv.get('importance_score', 0) >= importance_min
                ]
                return filtered_conversations[-limit:]
            
            # Database query
            if hasattr(self.conversation_storage, 'get_conversation_history'):
                return await self.conversation_storage.get_conversation_history(
                    user_name=user_name,
                    limit=limit,
                    importance_min=importance_min
                )
            else:
                # Fallback: Search memories
                results = await self.conversation_storage.search_memories(
                    query="conversation",
                    limit=limit,
                    filters={'importance_min': importance_min}
                )
                
                # Convert to conversation format
                conversations = []
                for result in results:
                    if result.memory_type == MemoryType.CONVERSATION:
                        conversations.append({
                            'conversation_id': result.context.get('conversation_id'),
                            'timestamp': result.timestamp.isoformat(),
                            'content': result.content,
                            'importance_score': result.importance,
                            'emotional_impact': result.emotional_intensity
                        })
                
                return conversations
                
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    async def search_conversation_database(self, 
                                         query: str, 
                                         limit: int = 20,
                                         filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Durchsucht Conversation Database"""
        try:
            if not self.conversation_storage:
                # Fallback: Search buffer
                return [
                    conv for conv in self.conversation_buffer
                    if query.lower() in conv.get('user_input', '').lower() or 
                       query.lower() in conv.get('kira_response', '').lower()
                ][:limit]
            
            # Database search
            if hasattr(self.conversation_storage, 'search_conversations'):
                return await self.conversation_storage.search_conversations(
                    query=query,
                    limit=limit,
                    filters=filters or {}
                )
            else:
                # Fallback: Memory search
                results = await self.conversation_storage.search_memories(
                    query=query,
                    limit=limit,
                    filters=filters or {}
                )
                
                return [self._memory_to_conversation_result(result) for result in results]
                
        except Exception as e:
            logger.error(f"Conversation database search failed: {e}")
            return []
        
    async def save_conversation_state(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        ✅ NEUE: Speichert kompletten Conversation State persistent
        
        Args:
            filepath: Optional custom save path
            
        Returns:
            Save operation result
        """
        try:
            import pickle
            import os
            from pathlib import Path
            
            # Default save location
            if not filepath:
                save_dir = Path("memory/data/conversation_states")
                save_dir.mkdir(parents=True, exist_ok=True)
                filepath = save_dir / f"conversation_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            # Prepare state for saving
            save_state = {
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0',
                'current_conversation': self.current_conversation.__dict__ if self.current_conversation else None,
                'conversation_buffer': [
                    {
                        'conversation_id': conv.get('conversation_id'),
                        'user_input': conv.get('user_input'),
                        'kira_response': conv.get('kira_response'),
                        'timestamp': conv.get('timestamp'),
                        'importance_score': conv.get('importance_score'),
                        'topic_category': conv.get('topic_category'),
                        'storage_location': conv.get('storage_location'),
                        'context': conv.get('context', {})
                    }
                    for conv in self.conversation_buffer
                ],
                'conversation_stats': self.conversation_stats.copy(),
                'memory_stats': self.get_memory_stats()
            }
            
            # Include STM/LTM state if available
            if self.stm:
                save_state['stm_memories'] = [
                    {
                        'memory_id': memory.memory_id,
                        'content': memory.content,
                        'memory_type': memory.memory_type.value,
                        'importance': memory.importance,
                        'context': memory.context,
                        'created_at': memory.created_at.isoformat(),
                        'last_accessed': memory.last_accessed.isoformat() if memory.last_accessed else None
                    }
                    for memory in self.stm.get_all_memories()
                ]
            
            if self.ltm:
                save_state['ltm_memories'] = [
                    {
                        'memory_id': memory.memory_id,
                        'content': memory.content,
                        'memory_type': memory.memory_type.value,
                        'importance': memory.importance,
                        'context': memory.context,
                        'created_at': memory.created_at.isoformat(),
                        'last_accessed': memory.last_accessed.isoformat() if memory.last_accessed else None
                    }
                    for memory in self.ltm.get_all_memories()
                ]
            
            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(save_state, f)
            
            logger.info(f"✅ Conversation state saved to {filepath}")
            
            return {
                'success': True,
                'filepath': str(filepath),
                'conversations_saved': len(save_state['conversation_buffer']),
                'stm_memories_saved': len(save_state.get('stm_memories', [])),
                'ltm_memories_saved': len(save_state.get('ltm_memories', [])),
                'timestamp': save_state['timestamp']
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to save conversation state: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def load_conversation_state(self, filepath: str) -> Dict[str, Any]:
        """
        ✅ NEUE: Lädt Conversation State aus Datei
        
        Args:
            filepath: Path to saved state file
            
        Returns:
            Load operation result
        """
        try:
            import pickle
            from pathlib import Path
            
            if not Path(filepath).exists():
                return {
                    'success': False,
                    'error': f'File not found: {filepath}'
                }
            
            # Load state from file
            with open(filepath, 'rb') as f:
                saved_state = pickle.load(f)
            
            # Validate state format
            if 'version' not in saved_state or 'timestamp' not in saved_state:
                return {
                    'success': False,
                    'error': 'Invalid saved state format'
                }
            
            # Restore conversation buffer
            if 'conversation_buffer' in saved_state:
                self.conversation_buffer = saved_state['conversation_buffer']
                logger.info(f"Restored {len(self.conversation_buffer)} conversations from buffer")
            
            # Restore conversation stats
            if 'conversation_stats' in saved_state:
                self.conversation_stats.update(saved_state['conversation_stats'])
            
            # Restore current conversation
            if saved_state.get('current_conversation'):
                from .conversation_memory import ConversationContext
                self.current_conversation = ConversationContext(**saved_state['current_conversation'])
                logger.info(f"Restored current conversation: {self.current_conversation.conversation_id}")
            
            # Restore STM memories
            stm_restored = 0
            if saved_state.get('stm_memories') and self.stm:
                for memory_data in saved_state['stm_memories']:
                    try:
                        memory = create_memory(
                            content=memory_data['content'],
                            memory_type=MemoryType(memory_data['memory_type']),
                            importance=memory_data['importance'],
                            context=memory_data.get('context', {})
                        )
                        
                        # Restore timestamps
                        memory.created_at = datetime.fromisoformat(memory_data['created_at'])
                        if memory_data.get('last_accessed'):
                            memory.last_accessed = datetime.fromisoformat(memory_data['last_accessed'])
                        
                        if self.stm.store_memory(memory):
                            stm_restored += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to restore STM memory: {e}")
            
            # Restore LTM memories
            ltm_restored = 0
            if saved_state.get('ltm_memories') and self.ltm:
                for memory_data in saved_state['ltm_memories']:
                    try:
                        memory = create_memory(
                            content=memory_data['content'],
                            memory_type=MemoryType(memory_data['memory_type']),
                            importance=memory_data['importance'],
                            context=memory_data.get('context', {})
                        )
                        
                        # Restore timestamps
                        memory.created_at = datetime.fromisoformat(memory_data['created_at'])
                        if memory_data.get('last_accessed'):
                            memory.last_accessed = datetime.fromisoformat(memory_data['last_accessed'])
                        
                        if self.ltm.store_memory(memory):
                            ltm_restored += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to restore LTM memory: {e}")
            
            logger.info(f"✅ Conversation state loaded from {filepath}")
            
            return {
                'success': True,
                'filepath': filepath,
                'conversations_restored': len(self.conversation_buffer),
                'stm_memories_restored': stm_restored,
                'ltm_memories_restored': ltm_restored,
                'saved_timestamp': saved_state['timestamp'],
                'loaded_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load conversation state: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_available_saved_states(self) -> List[Dict[str, Any]]:
        """
        ✅ NEUE: Listet verfügbare gespeicherte States auf
        
        Returns:
            List of available saved states
        """
        try:
            from pathlib import Path
            import os
            
            save_dir = Path("memory/data/conversation_states")
            if not save_dir.exists():
                return []
            
            saved_states = []
            
            for file_path in save_dir.glob("*.pkl"):
                try:
                    stat = os.stat(file_path)
                    saved_states.append({
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'size_bytes': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Failed to read state file {file_path}: {e}")
            
            # Sort by creation time (newest first)
            saved_states.sort(key=lambda x: x['created'], reverse=True)
            
            return saved_states
            
        except Exception as e:
            logger.error(f"Failed to list saved states: {e}")
            return []
    
    async def auto_save_conversation_state(self) -> Dict[str, Any]:
        """
        ✅ NEUE: Automatische Speicherung bei wichtigen Ereignissen
        
        Returns:
            Auto-save result
        """
        try:
            # Check if auto-save is needed
            if not self._should_auto_save():
                return {
                    'success': True,
                    'reason': 'auto_save_not_needed',
                    'skipped': True
                }
            
            # Create auto-save filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"auto_save_conversation_{timestamp}.pkl"
            
            # Perform save
            result = await self.save_conversation_state(filename)
            
            if result['success']:
                # Update last auto-save time
                self.conversation_stats['last_auto_save'] = datetime.now().isoformat()
                logger.info(f"✅ Auto-saved conversation state: {filename}")
            
            return result
            
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _should_auto_save(self) -> bool:
        """Entscheidet ob Auto-Save nötig ist"""
        try:
            # Auto-save triggers:
            # 1. Buffer has significant conversations
            if len(self.conversation_buffer) >= 10:
                return True
            
            # 2. High-importance conversations
            high_importance_count = sum(
                1 for conv in self.conversation_buffer 
                if conv.get('importance_score', 0) >= 7
            )
            if high_importance_count >= 3:
                return True
            
            # 3. Time-based (every hour with activity)
            last_auto_save = self.conversation_stats.get('last_auto_save')
            if last_auto_save:
                last_save_time = datetime.fromisoformat(last_auto_save)
                if (datetime.now() - last_save_time).total_seconds() > 3600:  # 1 hour
                    return True
            else:
                # First time - save if there's any content
                return len(self.conversation_buffer) > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Auto-save decision failed: {e}")
            return False

    def _memory_to_conversation_result(self, memory: Memory) -> Dict[str, Any]:
        """Konvertiert Memory zu Conversation Search Result"""
        return {
            'memory_id': memory.memory_id,
            'content': memory.content,
            'importance': memory.importance,
            'timestamp': memory.timestamp.isoformat(),
            'emotional_intensity': memory.emotional_intensity,
            'context': memory.context,
            'tags': memory.tags
        }

    # ✅ ERWEITERTE MEMORY STATS MIT DATABASE INFO
    def get_memory_stats(self) -> Dict[str, Any]:
        """Gibt Memory System Statistiken zurück"""
        base_stats = {
            'stm_capacity': len(self.stm.working_memory),
            'stm_max_capacity': self.stm.capacity,
            'ltm_total_memories': len(self.ltm.consolidated_memories),
            'conversation_buffer_size': len(self.conversation_buffer),
            'current_conversation_exchanges': self.current_conversation.interaction_count if self.current_conversation else 0,
            'ltm_thresholds': self.ltm_thresholds
        }
        
        # ✅ DATABASE STATS
        if self.conversation_storage:
            base_stats.update({
                'database_available': True,
                'storage_type': type(self.conversation_storage).__name__,
                'conversations_stored_db': len([c for c in self.conversation_buffer if c.get('database_stored', False)]),
                'conversations_pending_db': len([c for c in self.conversation_buffer if not c.get('database_stored', False)])
            })
        else:
            base_stats.update({
                'database_available': False,
                'storage_type': 'memory_only',
                'conversations_stored_db': 0,
                'conversations_pending_db': len(self.conversation_buffer)
            })
            
        return base_stats

    # ✅ CLEANUP METHODS
    async def cleanup_old_conversations(self, days_old: int = None):
        """Bereinigt alte Conversations"""
        try:
            days_old = days_old or self.persistence_settings['auto_cleanup_days']
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            if self.conversation_storage and hasattr(self.conversation_storage, 'cleanup_old_conversations'):
                cleaned = await self.conversation_storage.cleanup_old_conversations(cutoff_date)
                logger.info(f"✅ Cleaned up {cleaned} old conversations from database")
                return cleaned
            else:
                # Cleanup buffer
                original_size = len(self.conversation_buffer)
                self.conversation_buffer = [
                    conv for conv in self.conversation_buffer
                    if datetime.fromisoformat(conv['timestamp']) > cutoff_date
                ]
                cleaned = original_size - len(self.conversation_buffer)
                logger.info(f"✅ Cleaned up {cleaned} old conversations from buffer")
                return cleaned
                
        except Exception as e:
            logger.error(f"Conversation cleanup failed: {e}")
            return 0
        
    def health_check(self) -> Dict[str, Any]:
        """Health Check für Tests und Monitoring"""
        try:
            current_time = datetime.now()
            
            health = {
                'conversation_memory_system': True,
                'timestamp': current_time.isoformat(),
                'components': {},
                'statistics': {},
                'issues': [],
                'recommendations': []
            }
            
            # Check STM
            if self.stm:
                stm_utilization = len(self.stm.working_memory) / self.stm.capacity
                health['components']['stm'] = {
                    'available': True,
                    'capacity': self.stm.capacity,
                    'current_size': len(self.stm.working_memory),
                    'utilization': stm_utilization
                }
                
                if stm_utilization > 0.9:
                    health['issues'].append('STM near capacity')
                    health['recommendations'].append('Consider STM consolidation')
            else:
                health['components']['stm'] = {'available': False}
                health['issues'].append('STM not available')
            
            # Check LTM
            if self.ltm:
                ltm_utilization = len(self.ltm.consolidated_memories) / self.ltm.max_memories
                health['components']['ltm'] = {
                    'available': True,
                    'total_memories': len(self.ltm.consolidated_memories),
                    'max_capacity': self.ltm.max_memories,
                    'utilization': ltm_utilization
                }
                
                if ltm_utilization > 0.95:
                    health['issues'].append('LTM near capacity')
                    health['recommendations'].append('Consider memory pruning')
            else:
                health['components']['ltm'] = {'available': False}
                health['issues'].append('LTM not available')
            
            # Check Conversation Storage
            if self.conversation_storage:
                health['components']['conversation_storage'] = {
                    'available': True,
                    'type': type(self.conversation_storage).__name__
                }
            else:
                health['components']['conversation_storage'] = {'available': False}
                health['recommendations'].append('Enable database storage for persistence')
            
            # Check Memory Database
            if self.memory_database:
                health['components']['memory_database'] = {
                    'available': True,
                    'type': type(self.memory_database).__name__
                }
            else:
                health['components']['memory_database'] = {'available': False}
            
            # Statistics
            health['statistics'] = {
                'conversation_buffer_size': len(self.conversation_buffer),
                'current_conversation_active': self.current_conversation is not None,
                'total_conversations_processed': self.conversation_stats.get('conversations_processed', 0),
                'storage_available': STORAGE_AVAILABLE,
                'conversation_buffer_utilization': len(self.conversation_buffer) / 100  # Assume max 100
            }
            
            # Overall health assessment
            healthy_components = sum(1 for comp in health['components'].values() if comp.get('available', False))
            total_components = len(health['components'])
            
            if len(health['issues']) == 0 and healthy_components == total_components:
                health['overall_status'] = 'excellent'
            elif len(health['issues']) <= 1 and healthy_components >= total_components * 0.75:
                health['overall_status'] = 'good'
            elif healthy_components >= total_components * 0.5:
                health['overall_status'] = 'limited'
            else:
                health['overall_status'] = 'critical'
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'conversation_memory_system': False,
                'overall_status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Export
__all__ = [
    'ConversationMemorySystem',
    'ConversationImportance', 
    'ConversationTopic',
    'ConversationContext'
]