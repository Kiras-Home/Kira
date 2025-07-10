"""
🏛️ LONG TERM MEMORY - Enhanced mit Konsolidierung
Erweiterte LTM Implementation mit STM Integration
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import json

from .memory_types import Memory, MemoryType, MemoryImportance

logger = logging.getLogger(__name__)

class HumanLikeLongTermMemory:
    """
    🏛️ HUMAN-LIKE LONG TERM MEMORY
    Erweitert mit STM Konsolidierung und intelligentem Clustering
    """
    
    def __init__(self, 
                 max_memories: int = 10000,
                 consolidation_threshold: int = 6,
                 clustering_enabled: bool = True):
        """
        Initialize Long Term Memory
        
        Args:
            max_memories: Maximum number of memories to store
            consolidation_threshold: Minimum importance for auto-consolidation
            clustering_enabled: Enable memory clustering by topic/type
        """
        self.max_memories = max_memories
        self.consolidation_threshold = consolidation_threshold
        self.clustering_enabled = clustering_enabled
        
        # Memory storage
        self.consolidated_memories: Dict[int, Memory] = {}
        
        # Memory organization
        self._memory_clusters: Dict[str, Set[int]] = defaultdict(set)
        self._importance_index: Dict[int, Set[int]] = defaultdict(set)  # importance -> memory_ids
        self._type_index: Dict[MemoryType, Set[int]] = defaultdict(set)
        
        # Consolidation tracking
        self._consolidation_history: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            'memories_stored': 0,
            'consolidations_performed': 0,
            'memories_accessed': 0,
            'clusters_created': 0,
            'memories_pruned': 0
        }
        
        logger.info(f"Long Term Memory initialized for user: default")
    
    def store_memory(self, memory: Memory, source: str = 'direct') -> bool:
        """
        Speichert Memory in LTM mit automatischer Organisation
        
        Args:
            memory: Memory object to store
            source: Source of memory ('consolidation', 'direct', etc.)
            
        Returns:
            True if successfully stored
        """
        try:
            current_time = datetime.now()
            
            # Check if memory already exists
            if memory.memory_id in self.consolidated_memories:
                logger.debug(f"Memory {memory.memory_id} already in LTM, updating")
                existing_memory = self.consolidated_memories[memory.memory_id]
                
                # Update access metadata
                existing_memory.context['ltm_last_accessed'] = current_time.isoformat()
                existing_memory.context['ltm_access_count'] = existing_memory.context.get('ltm_access_count', 0) + 1
                
                return True
            
            # Check capacity and prune if necessary
            if len(self.consolidated_memories) >= self.max_memories:
                self._prune_old_memories()
            
            # Add LTM metadata
            memory.context.update({
                'ltm_stored_at': current_time.isoformat(),
                'ltm_source': source,
                'ltm_access_count': 0,
                'ltm_cluster': None
            })
            
            # Store memory
            self.consolidated_memories[memory.memory_id] = memory
            
            # Update indexes
            self._update_indexes(memory)
            
            # Perform clustering if enabled
            if self.clustering_enabled:
                self._cluster_memory(memory)
            
            self.stats['memories_stored'] += 1
            
            logger.debug(f"Memory {memory.memory_id} stored in LTM (source: {source}, total: {len(self.consolidated_memories)})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory in LTM: {e}")
            return False
    
    def consolidate_from_stm(self, stm_memories: List[Memory]) -> Dict[str, Any]:
        """
        ✅ NEUE: Konsolidiert Memories von STM zu LTM
        
        Args:
            stm_memories: List of memories from STM to consolidate
            
        Returns:
            Consolidation result
        """
        try:
            consolidation_start = datetime.now()
            
            # Filter memories that meet consolidation criteria
            consolidation_candidates = []
            for memory in stm_memories:
                if self._should_consolidate(memory):
                    consolidation_candidates.append(memory)
            
            if not consolidation_candidates:
                return {
                    'success': True,
                    'consolidated_count': 0,
                    'reason': 'no_candidates_met_criteria',
                    'candidates_reviewed': len(stm_memories)
                }
            
            # Perform consolidation
            consolidated_count = 0
            consolidation_details = []
            
            for memory in consolidation_candidates:
                # Enhance memory for LTM storage
                enhanced_memory = self._enhance_memory_for_ltm(memory)
                
                # Store in LTM
                if self.store_memory(enhanced_memory, source='stm_consolidation'):
                    consolidated_count += 1
                    
                    consolidation_details.append({
                        'memory_id': memory.memory_id,
                        'importance': memory.importance,
                        'memory_type': memory.memory_type.value,
                        'stm_access_count': memory.context.get('stm_access_count', 0),
                        'consolidation_reasons': memory.context.get('consolidation_reasons', [])
                    })
            
            # Record consolidation in history
            consolidation_record = {
                'timestamp': consolidation_start.isoformat(),
                'candidates_reviewed': len(stm_memories),
                'consolidated_count': consolidated_count,
                'consolidation_details': consolidation_details,
                'duration_seconds': (datetime.now() - consolidation_start).total_seconds()
            }
            
            self._consolidation_history.append(consolidation_record)
            self.stats['consolidations_performed'] += 1
            
            logger.info(f"✅ Consolidated {consolidated_count}/{len(stm_memories)} memories from STM to LTM")
            
            return {
                'success': True,
                'consolidated_count': consolidated_count,
                'candidates_reviewed': len(stm_memories),
                'consolidation_details': consolidation_details,
                'duration_seconds': consolidation_record['duration_seconds']
            }
            
        except Exception as e:
            logger.error(f"STM to LTM consolidation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'consolidated_count': 0
            }
    
    def search_memories(self, 
                       query: str, 
                       memory_types: Optional[List[MemoryType]] = None,
                       importance_min: int = 1,
                       limit: int = 10) -> List[Memory]:
        """
        Durchsucht LTM Memories mit erweiterten Filtern
        
        Args:
            query: Search query
            memory_types: Filter by memory types
            importance_min: Minimum importance level
            limit: Maximum results
            
        Returns:
            List of matching memories
        """
        try:
            query_lower = query.lower()
            matching_memories = []
            
            # Search in relevant clusters first
            relevant_clusters = self._find_relevant_clusters(query)
            
            # If clusters found, search in those first
            if relevant_clusters:
                for cluster_name in relevant_clusters:
                    memory_ids = self._memory_clusters[cluster_name]
                    for memory_id in memory_ids:
                        if memory_id in self.consolidated_memories:
                            memory = self.consolidated_memories[memory_id]
                            if self._matches_search_criteria(memory, query_lower, memory_types, importance_min):
                                matching_memories.append(memory)
            
            # If not enough results, search all memories
            if len(matching_memories) < limit:
                for memory in self.consolidated_memories.values():
                    if memory not in matching_memories:
                        if self._matches_search_criteria(memory, query_lower, memory_types, importance_min):
                            matching_memories.append(memory)
            
            # Sort by relevance and importance
            matching_memories.sort(
                key=lambda m: (
                    self._calculate_search_relevance(query_lower, m),
                    m.importance,
                    m.context.get('ltm_access_count', 0)
                ),
                reverse=True
            )
            
            # Update access stats
            current_time = datetime.now()
            for memory in matching_memories[:limit]:
                memory.context['ltm_last_accessed'] = current_time.isoformat()
                memory.context['ltm_access_count'] = memory.context.get('ltm_access_count', 0) + 1
                self.stats['memories_accessed'] += 1
            
            return matching_memories[:limit]
            
        except Exception as e:
            logger.error(f"LTM search failed: {e}")
            return []
    
    def get_all_memories(self) -> List[Memory]:
        """
        ✅ ERWEITERT: Holt alle LTM Memories mit aktueller Metadata
        
        Returns:
            List of all memories in LTM
        """
        try:
            current_time = datetime.now()
            all_memories = []
            
            for memory in self.consolidated_memories.values():
                # Update metadata
                memory.context['ltm_current_timestamp'] = current_time.isoformat()
                
                # Calculate time in LTM
                if 'ltm_stored_at' in memory.context:
                    stored_time = datetime.fromisoformat(memory.context['ltm_stored_at'])
                    time_in_ltm = current_time - stored_time
                    memory.context['ltm_duration_days'] = time_in_ltm.days
                
                all_memories.append(memory)
            
            # Sort by importance and access count
            all_memories.sort(
                key=lambda m: (m.importance, m.context.get('ltm_access_count', 0)),
                reverse=True
            )
            
            return all_memories
            
        except Exception as e:
            logger.error(f"Failed to get all LTM memories: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        ✅ ERWEITERT: Detaillierte LTM Statistiken
        
        Returns:
            Detailed LTM statistics
        """
        try:
            current_time = datetime.now()
            
            # Basic stats
            stats = self.stats.copy()
            stats.update({
                'total_memories': len(self.consolidated_memories),
                'max_capacity': self.max_memories,
                'utilization_percent': (len(self.consolidated_memories) / self.max_memories) * 100,
                'consolidation_threshold': self.consolidation_threshold,
                'clustering_enabled': self.clustering_enabled,
                'total_clusters': len(self._memory_clusters),
                'timestamp': current_time.isoformat()
            })
            
            # Memory analysis
            if self.consolidated_memories:
                importances = [m.importance for m in self.consolidated_memories.values()]
                access_counts = [m.context.get('ltm_access_count', 0) for m in self.consolidated_memories.values()]
                
                stats['memory_analysis'] = {
                    'average_importance': sum(importances) / len(importances),
                    'max_importance': max(importances),
                    'min_importance': min(importances),
                    'average_access_count': sum(access_counts) / len(access_counts) if access_counts else 0,
                    'max_access_count': max(access_counts) if access_counts else 0
                }
                
                # Type distribution
                type_counts = defaultdict(int)
                for memory in self.consolidated_memories.values():
                    type_counts[memory.memory_type.value] += 1
                
                stats['type_distribution'] = dict(type_counts)
            
            # Consolidation history summary
            if self._consolidation_history:
                recent_consolidations = [c for c in self._consolidation_history if 
                                       (current_time - datetime.fromisoformat(c['timestamp'])).days <= 7]
                
                stats['consolidation_summary'] = {
                    'total_consolidations': len(self._consolidation_history),
                    'recent_consolidations_7days': len(recent_consolidations),
                    'last_consolidation': self._consolidation_history[-1]['timestamp'] if self._consolidation_history else None
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get LTM stats: {e}")
            return self.stats.copy()
    
    # ✅ PRIVATE HELPER METHODS
    
    def _should_consolidate(self, memory: Memory) -> bool:
        """Entscheidet ob Memory konsolidiert werden soll"""
        try:
            # Basic importance check
            if memory.importance >= self.consolidation_threshold:
                return True
            
            # High access count in STM
            if memory.context.get('stm_access_count', 0) >= 3:
                return True
            
            # Learning/skill memories
            if memory.memory_type in [MemoryType.LEARNING, MemoryType.SKILL]:
                return True
            
            # Explicit consolidation marker
            if memory.context.get('force_consolidation', False):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Consolidation decision failed: {e}")
            return False
    
    def _enhance_memory_for_ltm(self, memory: Memory) -> Memory:
        """Erweitert Memory für LTM Storage"""
        try:
            # Add LTM enhancement metadata
            current_time = datetime.now()
            
            memory.context.update({
                'ltm_enhancement_timestamp': current_time.isoformat(),
                'ltm_enhanced_from_stm': True,
                'original_stm_context': memory.context.copy()
            })
            
            # Enhance content if it's learning-related
            if memory.memory_type in [MemoryType.LEARNING, MemoryType.SKILL]:
                memory.context['ltm_category'] = 'learning_enhanced'
            
            return memory
            
        except Exception as e:
            logger.error(f"Memory enhancement failed: {e}")
            return memory
    
    def _update_indexes(self, memory: Memory):
        """Updated Memory Indexes"""
        try:
            # Importance index
            self._importance_index[memory.importance].add(memory.memory_id)
            
            # Type index
            self._type_index[memory.memory_type].add(memory.memory_id)
            
        except Exception as e:
            logger.error(f"Index update failed: {e}")
    
    def _cluster_memory(self, memory: Memory):
        """Clustert Memory nach Thema/Typ"""
        try:
            # Simple clustering by type and keywords
            cluster_name = f"{memory.memory_type.value}"
            
            # Add keyword-based clusters
            content_lower = memory.content.lower()
            
            # Learning keywords
            if any(word in content_lower for word in ['learn', 'study', 'understand', 'remember']):
                cluster_name += "_learning"
            
            # Emotional keywords  
            elif any(word in content_lower for word in ['feel', 'emotion', 'happy', 'sad', 'angry']):
                cluster_name += "_emotional"
            
            # Technical keywords
            elif any(word in content_lower for word in ['code', 'program', 'system', 'technical']):
                cluster_name += "_technical"
            
            # Add to cluster
            self._memory_clusters[cluster_name].add(memory.memory_id)
            memory.context['ltm_cluster'] = cluster_name
            
            if cluster_name not in [cluster for cluster in self._memory_clusters.keys()]:
                self.stats['clusters_created'] += 1
            
        except Exception as e:
            logger.error(f"Memory clustering failed: {e}")
    
    def _find_relevant_clusters(self, query: str) -> List[str]:
        """Findet relevante Cluster für Query"""
        try:
            query_lower = query.lower()
            relevant_clusters = []
            
            for cluster_name in self._memory_clusters.keys():
                # Simple keyword matching
                if any(word in cluster_name.lower() for word in query_lower.split()):
                    relevant_clusters.append(cluster_name)
            
            return relevant_clusters
            
        except Exception as e:
            logger.error(f"Cluster search failed: {e}")
            return []
    
    def _matches_search_criteria(self, memory: Memory, query_lower: str, 
                                memory_types: Optional[List[MemoryType]], importance_min: int) -> bool:
        """Prüft ob Memory Search-Kriterien erfüllt"""
        try:
            # Content matching
            if query_lower not in memory.content.lower():
                # Check context for matches
                context_str = json.dumps(memory.context).lower()
                if query_lower not in context_str:
                    return False
            
            # Type filtering
            if memory_types and memory.memory_type not in memory_types:
                return False
            
            # Importance filtering
            if memory.importance < importance_min:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Search criteria check failed: {e}")
            return False
    
    def _calculate_search_relevance(self, query_lower: str, memory: Memory) -> float:
        """Berechnet Search Relevance Score"""
        try:
            relevance = 0.0
            
            # Content relevance
            content_lower = memory.content.lower()
            query_words = query_lower.split()
            content_words = content_lower.split()
            
            matching_words = len(set(query_words) & set(content_words))
            if query_words:
                relevance += (matching_words / len(query_words)) * 10
            
            # Exact phrase match bonus
            if query_lower in content_lower:
                relevance += 20
            
            return relevance
            
        except Exception as e:
            logger.error(f"Relevance calculation failed: {e}")
            return 0.0
    
    def _prune_old_memories(self):
        """Entfernt alte Memories bei Kapazitätsüberschreitung"""
        try:
            if len(self.consolidated_memories) < self.max_memories:
                return
            
            # Sort by least valuable (low importance + low access + old)
            memories_with_scores = []
            current_time = datetime.now()
            
            for memory in self.consolidated_memories.values():
                access_count = memory.context.get('ltm_access_count', 0)
                stored_time = datetime.fromisoformat(memory.context.get('ltm_stored_at', current_time.isoformat()))
                days_old = (current_time - stored_time).days
                
                # Pruning score (lower = more likely to prune)
                score = memory.importance + access_count - (days_old * 0.1)
                memories_with_scores.append((score, memory))
            
            # Sort by score and remove lowest 10%
            memories_with_scores.sort(key=lambda x: x[0])
            prune_count = max(1, len(memories_with_scores) // 10)
            
            for _, memory in memories_with_scores[:prune_count]:
                del self.consolidated_memories[memory.memory_id]
                
                # Remove from indexes
                self._importance_index[memory.importance].discard(memory.memory_id)
                self._type_index[memory.memory_type].discard(memory.memory_id)
                
                # Remove from clusters
                cluster_name = memory.context.get('ltm_cluster')
                if cluster_name:
                    self._memory_clusters[cluster_name].discard(memory.memory_id)
                
                self.stats['memories_pruned'] += 1
            
            logger.info(f"Pruned {prune_count} old memories from LTM")
            
        except Exception as e:
            logger.error(f"Memory pruning failed: {e}")

# Export
__all__ = ['HumanLikeLongTermMemory']