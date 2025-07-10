"""
Brain-Like Memory System - Gehirnähnliche Speicherung und Gewichtung
Implementiert neurowissenschaftliche Prinzipien für Memory-Management
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import random
from collections import defaultdict

# Debug-Logger Integration
from memory.debug_integration import (
    MemoryDebugMixin, 
    memory_debug_decorator,
    log_person_extraction,
    setup_debug_logging
)

logger = logging.getLogger(__name__)

class MemoryStrength(Enum):
    """Gedächtnisstärke basierend auf neurowissenschaftlichen Prinzipien"""
    FRAGILE = 0.1      # Schwaches Gedächtnis, schnell vergessen
    WEAK = 0.3         # Schwach, braucht Wiederholung
    MODERATE = 0.5     # Mittlere Stärke
    STRONG = 0.7       # Starkes Gedächtnis
    VIVID = 0.9        # Lebhaftes Gedächtnis
    TRAUMATIC = 1.0    # Traumatisches/unvergessliches Gedächtnis

class MemoryConsolidationLevel(Enum):
    """Konsolidierungsgrad wie im echten Gehirn"""
    ENCODING = "encoding"           # Gerade erst eingegangen
    INITIAL = "initial"             # Erste Verarbeitung
    CONSOLIDATING = "consolidating" # Wird gefestigt
    CONSOLIDATED = "consolidated"   # Gefestigt im LTM
    INTEGRATED = "integrated"       # Vollständig integriert

@dataclass
class NeuralConnection:
    """Neuronale Verbindung zwischen Memories"""
    source_memory_id: str
    target_memory_id: str
    connection_strength: float = 0.5
    connection_type: str = "associative"  # associative, causal, temporal, emotional
    activation_count: int = 0
    last_activation: datetime = field(default_factory=datetime.now)
    
    def strengthen(self, amount: float = 0.1):
        """Verstärkt die Verbindung (Hebbian Learning)"""
        self.connection_strength = min(1.0, self.connection_strength + amount)
        self.activation_count += 1
        self.last_activation = datetime.now()
    
    def weaken(self, amount: float = 0.05):
        """Schwächt die Verbindung ab"""
        self.connection_strength = max(0.0, self.connection_strength - amount)

@dataclass
class BrainLikeMemory:
    """Gehirnähnliche Memory-Repräsentation"""
    memory_id: str
    content: str
    memory_type: str
    
    # Neurowissenschaftliche Eigenschaften
    formation_time: datetime = field(default_factory=datetime.now)
    last_access: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    memory_strength: float = 0.5
    consolidation_level: MemoryConsolidationLevel = MemoryConsolidationLevel.ENCODING
    
    # Emotional weighting (Amygdala-Einfluss)
    emotional_valence: float = 0.0    # -1 (negativ) bis +1 (positiv)
    emotional_intensity: float = 0.0   # 0 (neutral) bis 1 (sehr emotional)
    
    # Contextual information (Hippocampus-Einfluss)
    spatial_context: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    social_context: Dict[str, Any] = field(default_factory=dict)
    
    # Synaptic connections
    incoming_connections: List[NeuralConnection] = field(default_factory=list)
    outgoing_connections: List[NeuralConnection] = field(default_factory=list)
    
    # Forgetting curve parameters
    forgetting_curve_slope: float = 0.1
    rehearsal_count: int = 0
    
    def calculate_retrieval_strength(self) -> float:
        """Berechnet die aktuelle Abrufstärke basierend auf der Ebbinghaus-Vergessenskurve"""
        time_since_last_access = (datetime.now() - self.last_access).total_seconds() / 3600  # Stunden
        
        # Grundlegende Vergessenskurve
        base_strength = self.memory_strength * math.exp(-self.forgetting_curve_slope * time_since_last_access)
        
        # Emotional boost (Amygdala-Effekt)
        emotional_boost = self.emotional_intensity * 0.3
        
        # Rehearsal bonus
        rehearsal_bonus = min(0.2, self.rehearsal_count * 0.05)
        
        # Connection strength bonus
        connection_bonus = min(0.1, len(self.incoming_connections) * 0.02)
        
        return min(1.0, base_strength + emotional_boost + rehearsal_bonus + connection_bonus)
    
    def access_memory(self):
        """Aktiviert das Memory (wie beim echten Abruf)"""
        self.last_access = datetime.now()
        self.access_count += 1
        
        # Strengthening bei Abruf (wie bei echten Neuronen)
        self.memory_strength = min(1.0, self.memory_strength + 0.05)
        
        # Aktiviere verbundene Memories
        for connection in self.outgoing_connections:
            connection.strengthen(0.02)
    
    def consolidate(self):
        """Konsolidiert das Memory (überführt es in stabilere Form)"""
        if self.consolidation_level == MemoryConsolidationLevel.ENCODING:
            self.consolidation_level = MemoryConsolidationLevel.INITIAL
        elif self.consolidation_level == MemoryConsolidationLevel.INITIAL:
            self.consolidation_level = MemoryConsolidationLevel.CONSOLIDATING
        elif self.consolidation_level == MemoryConsolidationLevel.CONSOLIDATING:
            self.consolidation_level = MemoryConsolidationLevel.CONSOLIDATED
        elif self.consolidation_level == MemoryConsolidationLevel.CONSOLIDATED:
            self.consolidation_level = MemoryConsolidationLevel.INTEGRATED
        
        # Stärkung durch Konsolidierung
        self.memory_strength = min(1.0, self.memory_strength + 0.1)

class BrainLikeMemorySystem(MemoryDebugMixin):
    """
    Gehirnähnliches Memory-System mit neurowissenschaftlichen Prinzipien und Debug-Logging
    """
    
    def __init__(self, storage_backend):
        """
        Initialisiert das Brain-Like Memory System
        
        Args:
            storage_backend: Das zugrundeliegende Storage-System
        """
        super().__init__()
        self.storage_backend = storage_backend
        self.memories: Dict[str, BrainLikeMemory] = {}
        self.neural_network: Dict[str, List[NeuralConnection]] = defaultdict(list)
        
        # Neurowissenschaftliche Parameter
        self.attention_threshold = 0.6
        self.consolidation_threshold = 0.7
        self.forgetting_threshold = 0.1
        
        # Simulation von Gehirnregionen
        self.hippocampus_capacity = 7  # Working memory capacity
        self.amygdala_emotional_boost = 0.3
        self.prefrontal_attention_filter = 0.5
        
        # Aktive Memories (wie im Working Memory)
        self.active_memories: List[str] = []
        
        # Debug-Logging initialisieren
        self.debug_enabled = True
        self.debug_stats = {
            'messages_stored': 0,
            'persons_extracted': 0,
            'memories_consolidated': 0,
            'connections_formed': 0,
            'storage_operations': 0
        }
        
        logger.info("🧠 Brain-Like Memory System initialized")
        
        # Debug-Log für System-Initialization
        if hasattr(self, 'debug_logger'):
            self.debug_logger.log_memory_operation(
                operation="system_initialization",
                data={
                    "memory_id": "system",
                    "content": "Brain-Like Memory System initialized",
                    "strength": 1.0,
                    "emotional_context": {
                        "emotional_intensity": 0.0,
                        "emotional_valence": 0.0,
                        "system_state": "initialized"
                    }
                }
            )
    
    @memory_debug_decorator("store_message")
    def store_message(self, 
                     content: str, 
                     user_id: str = "default",
                     conversation_id: str = "default",
                     message_type: str = "user",
                     emotional_context: Optional[Dict] = None) -> str:
        """
        Speichert eine Nachricht gehirnähnlich
        
        Args:
            content: Nachrichteninhalt
            user_id: Benutzer ID
            conversation_id: Konversations ID
            message_type: Typ der Nachricht (user/assistant)
            emotional_context: Emotionaler Kontext
            
        Returns:
            Memory ID
        """
        try:
            # Generiere Memory ID
            memory_id = f"brain_mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
            
            # Analysiere emotionalen Kontext
            emotional_valence, emotional_intensity = self._analyze_emotional_context(content, emotional_context)
            
            # Berechne initiale Memory-Stärke
            initial_strength = self._calculate_initial_strength(content, emotional_intensity)
            
            # Erstelle Brain-Like Memory
            brain_memory = BrainLikeMemory(
                memory_id=memory_id,
                content=content,
                memory_type="conversation",
                memory_strength=initial_strength,
                emotional_valence=emotional_valence,
                emotional_intensity=emotional_intensity,
                temporal_context={
                    "conversation_id": conversation_id,
                    "message_type": message_type,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Speichere in lokalem Memory
            self.memories[memory_id] = brain_memory
            
            # Speichere in Storage Backend
            storage_id = self.storage_backend.store_enhanced_memory(
                content=content,
                user_id=user_id,
                memory_type="conversation",
                session_id=conversation_id,
                metadata={
                    "brain_memory_id": memory_id,
                    "message_type": message_type,
                    "emotional_context": emotional_context or {}
                },
                importance=int(initial_strength * 10),
                emotion_intensity=emotional_intensity,
                stm_activation_level=1.0,
                memory_strength=initial_strength
            )
            
            # Verbinde mit ähnlichen Memories
            self._create_neural_connections(brain_memory)
            
            # Füge zu aktiven Memories hinzu
            self._add_to_active_memories(memory_id)
            
            # Debug-Logging
            if self.debug_enabled:
                self.debug_log_message_storage(
                    content=content,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message_type=message_type,
                    memory_id=memory_id,
                    metadata={
                        "brain_memory_id": memory_id,
                        "storage_id": storage_id,
                        "initial_strength": initial_strength,
                        "emotional_intensity": emotional_intensity
                    }
                )
                
                if hasattr(self, 'debug_logger'):
                    self.debug_logger.log_memory_operation(
                        operation="store",
                        data={
                            "memory_id": memory_id,
                            "content": content,
                            "strength": initial_strength,
                            "emotional_context": {
                                "emotional_intensity": emotional_intensity,
                                "emotional_valence": emotional_valence,
                                "message_type": message_type
                            }
                        }
                    )
                
                # Debug-Counter wurde durch neues Logging-System ersetzt
            
            logger.info(f"🧠 Message stored: {memory_id} | Strength: {initial_strength:.2f}")
            return memory_id
            
        except Exception as e:
            logger.error(f"❌ Error storing message: {e}")
            if self.debug_enabled:
                self.debug_log_storage_backend_operation(
                    operation="store_message",
                    table_name="brain_memory",
                    data_preview={"content_preview": content[:50], "user_id": user_id},
                    success=False,
                    error=str(e)
                )
            raise
    
    @log_person_extraction
    def extract_person_data(self, content: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Extrahiert Personen-Daten aus dem Content
        
        Args:
            content: Text-Inhalt
            user_id: Benutzer ID
            
        Returns:
            Dictionary mit extrahierten Personen-Daten
        """
        try:
            # Einfache Namen-Extraktion (kann durch NLP erweitert werden)
            extracted_names = self._extract_names_from_content(content)
            
            # Erweiterte Personen-Informationen
            extracted_info = {
                "names": extracted_names,
                "mention_count": len(extracted_names),
                "context": self._analyze_person_context(content, extracted_names),
                "extraction_confidence": self._calculate_extraction_confidence(content, extracted_names)
            }
            
            # Debug-Logging
            if self.debug_enabled:
                self.debug_log_person_data_extraction(
                    content=content,
                    extracted_names=extracted_names,
                    extracted_info=extracted_info,
                    user_id=user_id
                )
                
                # Debug-Counter wurde durch neues Logging-System ersetzt
            
            # Speichere Personen-Kontext
            for name in extracted_names:
                self._update_person_context(name, extracted_info["context"], user_id)
            
            logger.info(f"👤 Person data extracted: {len(extracted_names)} names")
            return extracted_info
            
        except Exception as e:
            logger.error(f"❌ Error extracting person data: {e}")
            if self.debug_enabled:
                self.debug_log_storage_backend_operation(
                    operation="extract_person_data",
                    table_name="person_data",
                    data_preview={"content_preview": content[:50]},
                    success=False,
                    error=str(e)
                )
            raise
    
    @memory_debug_decorator("consolidate_memory")
    def consolidate_memory(self, memory_id: str) -> bool:
        """
        Konsolidiert ein Memory von STM zu LTM
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True wenn erfolgreich konsolidiert
        """
        try:
            if memory_id not in self.memories:
                logger.warning(f"⚠️ Memory {memory_id} not found for consolidation")
                return False
            
            brain_memory = self.memories[memory_id]
            old_consolidation_level = brain_memory.consolidation_level
            old_strength = brain_memory.memory_strength
            
            # Konsolidiere das Memory
            brain_memory.consolidate()
            
            # Berechne Änderungen
            strength_change = brain_memory.memory_strength - old_strength
            connections_formed = len(brain_memory.outgoing_connections)
            
            # Update im Storage Backend
            self.storage_backend.update_memory_access(memory_id)
            
            # Debug-Logging
            if self.debug_enabled:
                self.debug_log_memory_consolidation(
                    memory_id=memory_id,
                    consolidation_level=brain_memory.consolidation_level.value,
                    strength_change=strength_change,
                    connections_formed=connections_formed,
                    trigger="manual_consolidation"
                )
                
                # Debug-Counter wurde durch neues Logging-System ersetzt
            
            logger.info(f"🧠 Memory consolidated: {memory_id} | Level: {brain_memory.consolidation_level.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error consolidating memory {memory_id}: {e}")
            if self.debug_enabled:
                self.debug_log_storage_backend_operation(
                    operation="consolidate_memory",
                    table_name="brain_memory",
                    data_preview={"memory_id": memory_id},
                    success=False,
                    error=str(e)
                )
            return False
    
    def _analyze_emotional_context(self, content: str, emotional_context: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Analysiert den emotionalen Kontext einer Nachricht
        
        Args:
            content: Nachrichteninhalt
            emotional_context: Zusätzlicher emotionaler Kontext
            
        Returns:
            Tuple von (valence, intensity)
        """
        # Einfache Emotionsanalyse (kann durch NLP erweitert werden)
        valence = 0.0
        intensity = 0.0
        
        if emotional_context:
            valence = emotional_context.get('valence', 0.0)
            intensity = emotional_context.get('intensity', 0.0)
        else:
            # Einfache Keyword-basierte Analyse
            positive_keywords = ['gut', 'toll', 'super', 'freude', 'liebe', 'glücklich']
            negative_keywords = ['schlecht', 'traurig', 'wütend', 'angst', 'hass', 'furcht']
            
            content_lower = content.lower()
            
            for keyword in positive_keywords:
                if keyword in content_lower:
                    valence += 0.2
                    intensity += 0.1
            
            for keyword in negative_keywords:
                if keyword in content_lower:
                    valence -= 0.2
                    intensity += 0.1
            
            # Clamp values
            valence = max(-1.0, min(1.0, valence))
            intensity = max(0.0, min(1.0, intensity))
        
        # Debug-Logging
        if self.debug_enabled:
            self.debug_log_emotional_memory_processing(
                memory_id="analysis",
                emotion_type="analysis",
                emotional_intensity=intensity,
                emotional_valence=valence,
                amygdala_activation=intensity * 0.5
            )
        
        return valence, intensity
    
    def _calculate_initial_strength(self, content: str, emotional_intensity: float) -> float:
        """
        Berechnet die initiale Memory-Stärke
        
        Args:
            content: Nachrichteninhalt
            emotional_intensity: Emotionale Intensität
            
        Returns:
            Memory-Stärke (0.0 - 1.0)
        """
        base_strength = 0.5  # Grundstärke
        
        # Emotional boost
        emotional_boost = emotional_intensity * self.amygdala_emotional_boost
        
        # Length boost (längere Nachrichten sind wichtiger)
        length_boost = min(0.2, len(content) / 1000)
        
        # Attention filter
        attention_boost = self.prefrontal_attention_filter * 0.1
        
        return min(1.0, base_strength + emotional_boost + length_boost + attention_boost)
    
    def _extract_names_from_content(self, content: str) -> List[str]:
        """
        Extrahiert Namen aus dem Content (einfache Implementierung)
        
        Args:
            content: Text-Inhalt
            
        Returns:
            Liste der extrahierten Namen
        """
        # Einfache Implementierung - kann durch NLP erweitert werden
        import re
        
        # Suche nach Wörtern mit Großbuchstaben (potenzielle Namen)
        name_pattern = r'\b[A-Z][a-zA-Z]{2,}\b'
        potential_names = re.findall(name_pattern, content)
        
        # Filtere häufige Nicht-Namen
        common_words = {'Der', 'Die', 'Das', 'Ein', 'Eine', 'Ich', 'Du', 'Sie', 'Er', 'Es', 'Wir', 'Ihr'}
        names = [name for name in potential_names if name not in common_words]
        
        return list(set(names))  # Entferne Duplikate
    
    def _analyze_person_context(self, content: str, names: List[str]) -> Dict[str, Any]:
        """
        Analysiert den Kontext der erwähnten Personen
        
        Args:
            content: Text-Inhalt
            names: Liste der Namen
            
        Returns:
            Kontext-Dictionary
        """
        context = {
            "mentioned_names": names,
            "context_sentences": [],
            "sentiment": "neutral",
            "relationships": []
        }
        
        # Erweiterte Kontext-Analyse kann hier implementiert werden
        for name in names:
            # Finde Sätze mit dem Namen
            sentences = content.split('.')
            for sentence in sentences:
                if name in sentence:
                    context["context_sentences"].append(sentence.strip())
        
        return context
    
    def _calculate_extraction_confidence(self, content: str, names: List[str]) -> float:
        """
        Berechnet die Konfidenz der Namen-Extraktion
        
        Args:
            content: Text-Inhalt
            names: Extrahierte Namen
            
        Returns:
            Konfidenz-Score (0.0 - 1.0)
        """
        if not names:
            return 0.0
        
        # Einfache Konfidenz-Berechnung
        base_confidence = 0.7
        
        # Bonus für mehrere Namen
        multiple_names_bonus = min(0.2, len(names) * 0.05)
        
        # Bonus für längeren Content
        content_length_bonus = min(0.1, len(content) / 500)
        
        return min(1.0, base_confidence + multiple_names_bonus + content_length_bonus)
    
    def _update_person_context(self, person_name: str, context: Dict[str, Any], user_id: str):
        """
        Aktualisiert den Kontext einer Person
        
        Args:
            person_name: Name der Person
            context: Neuer Kontext
            user_id: Benutzer ID
        """
        # Hier könnte eine Datenbank-Abfrage stehen
        old_context = {}  # Placeholder
        
        # Debug-Logging
        if self.debug_enabled:
            self.debug_log_person_context_update(
                person_name=person_name,
                context_type="conversation_context",
                old_context=old_context,
                new_context=context,
                confidence_score=context.get("confidence", 0.5)
            )
    
    def _create_neural_connections(self, brain_memory: BrainLikeMemory):
        """
        Erstellt neuronale Verbindungen zu ähnlichen Memories
        
        Args:
            brain_memory: Das Brain Memory
        """
        # Finde ähnliche Memories
        similar_memories = self._find_similar_memories(brain_memory)
        
        for similar_memory_id, similarity_score in similar_memories[:3]:  # Top 3
            if similar_memory_id != brain_memory.memory_id:
                # Erstelle Verbindung
                connection = NeuralConnection(
                    source_memory_id=brain_memory.memory_id,
                    target_memory_id=similar_memory_id,
                    connection_strength=similarity_score,
                    connection_type="similarity"
                )
                
                brain_memory.outgoing_connections.append(connection)
                self.neural_network[brain_memory.memory_id].append(connection)
                
                # Debug-Logging
                if self.debug_enabled:
                    self.debug_log_neural_connection(
                        source_memory_id=brain_memory.memory_id,
                        target_memory_id=similar_memory_id,
                        connection_strength=similarity_score,
                        connection_type="similarity",
                        activation_reason="memory_similarity"
                    )
                    
                    # Debug-Counter wurde durch neues Logging-System ersetzt
    
    def _find_similar_memories(self, brain_memory: BrainLikeMemory) -> List[Tuple[str, float]]:
        """
        Findet ähnliche Memories basierend auf Content
        
        Args:
            brain_memory: Das Brain Memory
            
        Returns:
            Liste von (memory_id, similarity_score) Tupeln
        """
        similarities = []
        
        for memory_id, memory in self.memories.items():
            if memory_id != brain_memory.memory_id:
                # Einfache Ähnlichkeitsberechnung
                similarity = self._calculate_content_similarity(
                    brain_memory.content, memory.content
                )
                
                if similarity > 0.3:  # Threshold
                    similarities.append((memory_id, similarity))
        
        # Sortiere nach Ähnlichkeit
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Berechnet die Ähnlichkeit zwischen zwei Texten
        
        Args:
            content1: Erster Text
            content2: Zweiter Text
            
        Returns:
            Ähnlichkeits-Score (0.0 - 1.0)
        """
        # Einfache Word-Overlap-Berechnung
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _add_to_active_memories(self, memory_id: str):
        """
        Fügt ein Memory zu den aktiven Memories hinzu
        
        Args:
            memory_id: Memory ID
        """
        # Entferne älteste Memories wenn Kapazität überschritten
        if len(self.active_memories) >= self.hippocampus_capacity:
            oldest_memory = self.active_memories.pop(0)
            logger.debug(f"🧠 Removed {oldest_memory} from active memories")
        
        self.active_memories.append(memory_id)
        logger.debug(f"🧠 Added {memory_id} to active memories")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Gibt den Gesundheitsstatus des Systems zurück
        
        Returns:
            System-Gesundheitsstatus
        """
        health_status = {
            "memory_count": len(self.memories),
            "active_memories": len(self.active_memories),
            "neural_connections": sum(len(connections) for connections in self.neural_network.values()),
            "storage_health": True,  # Kann durch Storage-Backend-Check erweitert werden
            "brain_memory_active": True,
            "consolidation_active": True,
            "debug_enabled": self.debug_enabled,
            "debug_stats": self.debug_stats
        }
        
        # Debug-Logging
        if hasattr(self, 'debug_logger'):
            self.debug_logger.log_memory_system_health(
                memory_count=health_status["memory_count"],
                storage_health=health_status["storage_health"],
                brain_memory_active=health_status["brain_memory_active"],
                consolidation_active=health_status["consolidation_active"]
            )
        
        return health_status
