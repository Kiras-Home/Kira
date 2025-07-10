"""
🧠 SHORT TERM MEMORY - Enhanced mit Konsolidierung
Erweiterte STM Implementation mit echtem Kapazitäts-Management
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import json

from .memory_types import Memory, MemoryType, MemoryImportance

logger = logging.getLogger(__name__)

class HumanLikeShortTermMemory:
    """
    🧠 HUMAN-LIKE SHORT TERM MEMORY
    Erweitert mit echtem Kapazitäts-Management und Konsolidierung
    """
    
    def __init__(self, capacity: int = 7, retention_minutes: int = 30):
        """
        Initialize Short Term Memory
        
        Args:
            capacity: Maximum number of memories to hold
            retention_minutes: How long memories stay in STM before decay
        """
        self.capacity = capacity
        self.retention_minutes = retention_minutes
        
        # Memory storage - ordered by insertion time
        self.working_memory: deque = deque(maxlen=capacity)
        
        # Memory index for fast lookup
        self._memory_index: Dict[int, Memory] = {}
        
        # Access tracking for LRU
        self._access_times: Dict[int, datetime] = {}
        self._access_counts: Dict[int, int] = {}
        
        # Statistics
        self.stats = {
            'memories_stored': 0,
            'memories_evicted': 0,
            'memories_accessed': 0,
            'consolidations_triggered': 0,
            'capacity_exceeded_count': 0
        }
        
        logger.info(f"Short Term Memory initialized with capacity: {capacity}")
    
    def store_memory(self, memory: Memory) -> bool:
        """
        Speichert Memory in STM mit intelligentem Kapazitäts-Management
        
        Args:
            memory: Memory object to store
            
        Returns:
            True if successfully stored
        """
        try:
            current_time = datetime.now()
            
            # Check if memory already exists
            if memory.memory_id in self._memory_index:
                logger.debug(f"Memory {memory.memory_id} already in STM, updating access time")
                self._access_times[memory.memory_id] = current_time
                self._access_counts[memory.memory_id] = self._access_counts.get(memory.memory_id, 0) + 1
                return True
            
            # Clean expired memories first
            self._clean_expired_memories()
            
            # Check capacity and make room if needed
            if len(self.working_memory) >= self.capacity:
                self._make_room_for_new_memory(memory)
            
            # Store the memory
            self.working_memory.append(memory)
            self._memory_index[memory.memory_id] = memory
            self._access_times[memory.memory_id] = current_time
            self._access_counts[memory.memory_id] = 1
            
            # Update memory with STM timestamp
            memory.context['stm_stored_at'] = current_time.isoformat()
            memory.context['stm_access_count'] = 1
            
            self.stats['memories_stored'] += 1
            
            logger.debug(f"Memory {memory.memory_id} stored in STM (utilization: {len(self.working_memory)}/{self.capacity})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory in STM: {e}")
            return False
    
    def get_memory(self, memory_id: int) -> Optional[Memory]:
        """
        Holt Memory aus STM und updated Access-Stats
        
        Args:
            memory_id: ID of memory to retrieve
            
        Returns:
            Memory object or None
        """
        try:
            if memory_id not in self._memory_index:
                return None
            
            memory = self._memory_index[memory_id]
            current_time = datetime.now()
            
            # Update access statistics
            self._access_times[memory_id] = current_time
            self._access_counts[memory_id] = self._access_counts.get(memory_id, 0) + 1
            self.stats['memories_accessed'] += 1
            
            # Update memory context
            memory.context['last_accessed'] = current_time.isoformat()
            memory.context['stm_access_count'] = self._access_counts[memory_id]
            
            logger.debug(f"Memory {memory_id} accessed from STM (access count: {self._access_counts[memory_id]})")
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to get memory from STM: {e}")
            return None
    
    def search_memories(self, query: str, limit: int = 5) -> List[Memory]:
        """
        Durchsucht STM Memories
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching memories
        """
        try:
            query_lower = query.lower()
            matching_memories = []
            
            for memory in self.working_memory:
                # Simple content matching
                if query_lower in memory.content.lower():
                    matching_memories.append(memory)
                
                # Check context for matches
                if hasattr(memory, 'context') and memory.context:
                    context_str = json.dumps(memory.context).lower()
                    if query_lower in context_str:
                        if memory not in matching_memories:
                            matching_memories.append(memory)
            
            # Sort by relevance (access count + importance)
            matching_memories.sort(
                key=lambda m: (
                    self._access_counts.get(m.memory_id, 0),
                    m.importance
                ),
                reverse=True
            )
            
            # Update access stats for searched memories
            current_time = datetime.now()
            for memory in matching_memories[:limit]:
                self._access_times[memory.memory_id] = current_time
                self._access_counts[memory.memory_id] = self._access_counts.get(memory.memory_id, 0) + 1
            
            return matching_memories[:limit]
            
        except Exception as e:
            logger.error(f"STM search failed: {e}")
            return []
    
    def get_consolidation_candidates(self) -> List[Memory]:
        """
        ✅ NEUE: Identifiziert Memories für LTM Konsolidierung
        
        Returns:
            List of memories ready for consolidation
        """
        try:
            candidates = []
            current_time = datetime.now()
            
            for memory in self.working_memory:
                # Criteria for consolidation:
                # 1. High importance (7+)
                # 2. High access count (3+)
                # 3. Old enough (30+ minutes)
                # 4. Marked as important in context
                
                should_consolidate = False
                consolidation_reason = []
                
                # High importance
                if memory.importance >= 7:
                    should_consolidate = True
                    consolidation_reason.append('high_importance')
                
                # High access count
                access_count = self._access_counts.get(memory.memory_id, 0)
                if access_count >= 3:
                    should_consolidate = True
                    consolidation_reason.append('frequently_accessed')
                
                # Time-based (older than retention period)
                if memory.memory_id in self._access_times:
                    time_in_stm = current_time - self._access_times[memory.memory_id]
                    if time_in_stm.total_seconds() > (self.retention_minutes * 60):
                        should_consolidate = True
                        consolidation_reason.append('time_based')
                
                # Explicit consolidation markers in context
                if memory.context.get('force_consolidation', False):
                    should_consolidate = True
                    consolidation_reason.append('explicit_marker')
                
                # Learning-related content
                if memory.memory_type in [MemoryType.LEARNING, MemoryType.SKILL]:
                    should_consolidate = True
                    consolidation_reason.append('learning_content')
                
                if should_consolidate:
                    # Add consolidation metadata
                    memory.context['consolidation_reasons'] = consolidation_reason
                    memory.context['consolidation_score'] = self._calculate_consolidation_score(memory)
                    memory.context['stm_duration_minutes'] = time_in_stm.total_seconds() / 60
                    
                    candidates.append(memory)
            
            # Sort by consolidation priority
            candidates.sort(
                key=lambda m: m.context.get('consolidation_score', 0),
                reverse=True
            )
            
            logger.info(f"Found {len(candidates)} consolidation candidates in STM")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to get consolidation candidates: {e}")
            return []
    
    def remove_memory(self, memory_id: int) -> bool:
        """
        ✅ NEUE: Entfernt Memory aus STM (für Konsolidierung)
        
        Args:
            memory_id: ID of memory to remove
            
        Returns:
            True if successfully removed
        """
        try:
            if memory_id not in self._memory_index:
                logger.warning(f"Memory {memory_id} not found in STM for removal")
                return False
            
            memory = self._memory_index[memory_id]
            
            # Remove from deque
            try:
                self.working_memory.remove(memory)
            except ValueError:
                logger.warning(f"Memory {memory_id} not in working_memory deque")
            
            # Remove from indexes
            del self._memory_index[memory_id]
            
            if memory_id in self._access_times:
                del self._access_times[memory_id]
            
            if memory_id in self._access_counts:
                del self._access_counts[memory_id]
            
            self.stats['memories_evicted'] += 1
            
            logger.debug(f"Memory {memory_id} removed from STM")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove memory from STM: {e}")
            return False
    
    def get_all_memories(self) -> List[Memory]:
        """
        ✅ ERWEITERT: Holt alle Memories mit aktualisierter Metadata
        
        Returns:
            List of all memories in STM
        """
        try:
            # Clean expired memories first
            self._clean_expired_memories()
            
            # Update metadata for all memories
            current_time = datetime.now()
            all_memories = []
            
            for memory in self.working_memory:
                # Update metadata
                memory.context['stm_access_count'] = self._access_counts.get(memory.memory_id, 0)
                memory.context['current_time_in_stm'] = current_time.isoformat()
                
                if memory.memory_id in self._access_times:
                    time_in_stm = current_time - self._access_times[memory.memory_id]
                    memory.context['stm_duration_minutes'] = time_in_stm.total_seconds() / 60
                
                all_memories.append(memory)
            
            return all_memories
            
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        ✅ ERWEITERT: Detaillierte STM Statistiken
        
        Returns:
            Detailed STM statistics
        """
        try:
            current_time = datetime.now()
            
            # Basic stats
            stats = self.stats.copy()
            stats.update({
                'current_capacity': len(self.working_memory),
                'max_capacity': self.capacity,
                'utilization_percent': (len(self.working_memory) / self.capacity) * 100,
                'retention_minutes': self.retention_minutes,
                'timestamp': current_time.isoformat()
            })
            
            # Memory analysis
            if self.working_memory:
                importances = [m.importance for m in self.working_memory]
                access_counts = [self._access_counts.get(m.memory_id, 0) for m in self.working_memory]
                
                stats['memory_analysis'] = {
                    'average_importance': sum(importances) / len(importances),
                    'max_importance': max(importances),
                    'min_importance': min(importances),
                    'average_access_count': sum(access_counts) / len(access_counts),
                    'max_access_count': max(access_counts),
                    'consolidation_candidates': len(self.get_consolidation_candidates())
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get STM stats: {e}")
            return self.stats.copy()
    
    # ✅ PRIVATE HELPER METHODS
    
    def _clean_expired_memories(self):
        """Entfernt abgelaufene Memories"""
        try:
            current_time = datetime.now()
            retention_delta = timedelta(minutes=self.retention_minutes)
            
            expired_ids = []
            
            for memory_id, access_time in self._access_times.items():
                if current_time - access_time > retention_delta:
                    expired_ids.append(memory_id)
            
            for memory_id in expired_ids:
                if memory_id in self._memory_index:
                    memory = self._memory_index[memory_id]
                    memory.context['expiry_reason'] = 'time_based'
                    self.remove_memory(memory_id)
                    logger.debug(f"Memory {memory_id} expired from STM")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    def _make_room_for_new_memory(self, new_memory: Memory):
        """Schafft Platz für neue Memory"""
        try:
            # Strategy: Remove least recently used, lowest importance
            if not self.working_memory:
                return
            
            self.stats['capacity_exceeded_count'] += 1
            
            # Calculate eviction scores (lower = more likely to evict)
            eviction_candidates = []
            current_time = datetime.now()
            
            for memory in self.working_memory:
                access_time = self._access_times.get(memory.memory_id, current_time)
                access_count = self._access_counts.get(memory.memory_id, 0)
                
                # Eviction score: importance + access_frequency - age_penalty
                time_since_access = (current_time - access_time).total_seconds() / 3600  # hours
                eviction_score = memory.importance + access_count - time_since_access
                
                eviction_candidates.append((eviction_score, memory))
            
            # Sort by eviction score (ascending - lowest first)
            eviction_candidates.sort(key=lambda x: x[0])
            
            # Evict the least valuable memory
            if eviction_candidates:
                _, memory_to_evict = eviction_candidates[0]
                memory_to_evict.context['eviction_reason'] = 'capacity_management'
                memory_to_evict.context['evicted_for'] = new_memory.memory_id
                
                self.remove_memory(memory_to_evict.memory_id)
                logger.debug(f"Memory {memory_to_evict.memory_id} evicted to make room for {new_memory.memory_id}")
            
        except Exception as e:
            logger.error(f"Failed to make room for new memory: {e}")
    
    def _calculate_consolidation_score(self, memory: Memory) -> float:
        """Berechnet Konsolidierungs-Score"""
        try:
            score = 0.0
            
            # Base importance
            score += memory.importance * 10
            
            # Access frequency
            access_count = self._access_counts.get(memory.memory_id, 0)
            score += access_count * 5
            
            # Time in STM
            if memory.memory_id in self._access_times:
                current_time = datetime.now()
                time_in_stm = current_time - self._access_times[memory.memory_id]
                hours_in_stm = time_in_stm.total_seconds() / 3600
                score += hours_in_stm * 2
            
            # Memory type bonus
            if memory.memory_type in [MemoryType.LEARNING, MemoryType.SKILL]:
                score += 20
            elif memory.memory_type in [MemoryType.PERSONAL, MemoryType.EMOTIONAL]:
                score += 15
            
            return score
            
        except Exception as e:
            logger.error(f"Consolidation score calculation failed: {e}")
            return 0.0

# Export
__all__ = ['HumanLikeShortTermMemory']