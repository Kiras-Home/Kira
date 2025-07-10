# memory/core/memory_types.py
"""
Memory Types für Kira Memory System
Vollständige Enum-Definitionen ohne Abhängigkeiten
"""

from enum import Enum, auto
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class MemoryType(Enum):
    """Memory Type Classifications"""
    CONVERSATION = "conversation"
    SHORT_TERM_EXPERIENCE = "short_term_experience"
    SIGNIFICANT_MEMORY = "significant_memory"
    PERSONAL_INFORMATION = "personal_information"
    SKILL_KNOWLEDGE = "skill_knowledge"
    EMOTIONAL_MEMORY = "emotional_memory"
    PERSONALITY_TRAIT = "personality_trait"
    SYSTEM_EVENT = "system_event"
    USER_PREFERENCE = "user_preference"
    CONTEXT_INFORMATION = "context_information"
    GENERAL = "general"
    LEARNING = "learning"
    OBSERVATION = "observation"
    FACT = "fact"
    PREFERENCE = "preference"
    # Zusätzliche Typen für STM/LTM
    EXPERIENCE = "experience"
    EMOTIONAL = "emotional"
    REFLECTION = "reflection"
    PROCEDURAL = "procedural"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PERSONALITY = "personality"
    RELATIONSHIP = "relationship"
    SKILL = "skill"
    INSIGHT = "insight"


class ImportanceLevel(Enum):
    """Importance Level Classifications"""
    CRITICAL = 10
    HIGH = 8
    MEDIUM_HIGH = 6
    MEDIUM = 5
    MEDIUM_LOW = 4
    LOW = 2
    MINIMAL = 1


# 🔧 ALIAS für Kompatibilität
class MemoryImportance(Enum):
    """Alias für ImportanceLevel - für Kompatibilität"""
    TRIVIAL = 1
    LOW = 3
    MEDIUM = 5
    HIGH = 7
    CRITICAL = 9
    CORE = 10


class EmotionalType(Enum):
    """Emotional Classifications"""
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    EXCITED = "excited"
    CALM = "calm"
    ANXIOUS = "anxious"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CURIOUS = "curious"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"


# 🔧 ALIAS für Kompatibilität
class EmotionalState(Enum):
    """Alias für EmotionalType - für Kompatibilität"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    ANXIOUS = "anxious"
    CURIOUS = "curious"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"


class ConsolidationState(Enum):
    """Memory Consolidation States"""
    PENDING = "pending"
    CONSOLIDATING = "consolidating"
    CONSOLIDATED = "consolidated"
    ARCHIVED = "archived"


@dataclass
class MemoryContext:
    """Memory Context Information"""
    session_id: str = "main"
    user_id: str = "default"
    device_context: str = "unknown"
    conversation_context: str = ""
    temporal_context: str = ""
    emotional_context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.emotional_context is None:
            self.emotional_context = {}


@dataclass
class MemoryMetadata:
    """Enhanced Memory Metadata"""
    source: str = "system"
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    relationships: List[int] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    access_count: int = 0

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


# 🔧 HAUPTKLASSE Memory - FEHLTE!
@dataclass
class Memory:
    """Basis Memory-Struktur"""
    content: str
    memory_type: MemoryType = MemoryType.SEMANTIC
    importance: int = 5
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str = "default"
    session_id: Optional[str] = None
    emotional_intensity: float = 0.5
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    memory_id: Optional[int] = None
    is_core_memory: bool = False
    decay_rate: float = 0.1

    def __post_init__(self):
        """Post-initialization processing"""
        if self.memory_id is None:
            # Generate unique ID
            self.memory_id = int(str(uuid.uuid4().int)[:10])

        if not self.context:
            self.context = {}

        if not self.tags:
            self.tags = []


# 🔧 MemoryConnection Klasse - FEHLTE!
@dataclass
class MemoryConnection:
    """Verbindung zwischen Memories"""
    source_memory_id: int
    target_memory_id: int
    connection_type: str = "related"
    strength: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)


# 🔧 create_memory Funktion - FEHLTE!
def create_memory(content: str,
                  memory_type: MemoryType = MemoryType.SEMANTIC,
                  importance: int = 5,
                  user_id: str = "default",
                  **kwargs) -> Memory:
    """Erstellt eine neue Memory"""
    return Memory(
        content=content,
        memory_type=memory_type,
        importance=importance,
        user_id=user_id,
        **kwargs
    )


# Utility Functions
def get_memory_type_by_name(name: str) -> Optional[MemoryType]:
    """Get MemoryType by string name"""
    try:
        return MemoryType(name.lower())
    except ValueError:
        return None


def get_importance_level(value: int) -> ImportanceLevel:
    """Get ImportanceLevel by integer value"""
    for level in ImportanceLevel:
        if level.value == value:
            return level
    # Return closest match
    if value >= 10:
        return ImportanceLevel.CRITICAL
    elif value >= 8:
        return ImportanceLevel.HIGH
    elif value >= 6:
        return ImportanceLevel.MEDIUM_HIGH
    elif value >= 4:
        return ImportanceLevel.MEDIUM_LOW
    elif value >= 2:
        return ImportanceLevel.LOW
    else:
        return ImportanceLevel.MINIMAL


def get_emotional_type_by_name(name: str) -> Optional[EmotionalType]:
    """Get EmotionalType by string name"""
    try:
        return EmotionalType(name.lower())
    except ValueError:
        return None


# 🔧 VOLLSTÄNDIGE EXPORT LISTE
__all__ = [
    # Haupt-Klassen
    'Memory',
    'MemoryConnection',

    # Enums - Original
    'MemoryType',
    'ImportanceLevel',
    'EmotionalType',
    'ConsolidationState',

    # Enums - Kompatibilitäts-Aliases
    'MemoryImportance',
    'EmotionalState',

    # Context/Metadata
    'MemoryContext',
    'MemoryMetadata',

    # Funktionen
    'create_memory',
    'get_memory_type_by_name',
    'get_importance_level',
    'get_emotional_type_by_name'
]