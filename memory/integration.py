"""
🔧 UNIFIED MEMORY SYSTEM
Zentrale Integration aller Memory-Komponenten
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

from .core.memory_types import Memory, MemoryType, create_memory
from .core.short_term_memory import HumanLikeShortTermMemory
from .core.long_term_memory import HumanLikeLongTermMemory
from .core.conversation_memory import ConversationMemorySystem

logger = logging.getLogger(__name__)

@dataclass
class MemorySystemConfig:
    """Configuration für Unified Memory System"""
    stm_capacity: int = 7
    enable_database: bool = True
    enable_conversations: bool = True
    auto_consolidation: bool = True
    consolidation_interval_minutes: int = 30
    database_config: Dict[str, Any] = None
    ltm_config: Dict[str, Any] = None
    conversation_config: Dict[str, Any] = None

class UnifiedMemorySystem:
    """
    🧠 UNIFIED MEMORY SYSTEM
    Zentrale Koordination aller Memory-Komponenten
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Unified Memory System"""
        
        # Parse configuration
        if isinstance(config, dict):
            self.config = MemorySystemConfig(**{
                k: v for k, v in config.items() 
                if k in MemorySystemConfig.__annotations__
            })
        else:
            self.config = MemorySystemConfig()
        
        # Initialize components
        self.stm: Optional[HumanLikeShortTermMemory] = None
        self.ltm: Optional[HumanLikeLongTermMemory] = None
        self.conversation_memory: Optional[ConversationMemorySystem] = None
        self.storage = None
        
        # System state
        self.initialized = False
        self.last_consolidation = None
        self.stats = {
            'memories_created': 0,
            'memories_consolidated': 0,
            'conversations_processed': 0,
            'errors': 0
        }
        
        logger.info("🧠 Unified Memory System created")
    
    def initialize(self) -> bool:
        """
        Initialisiert alle Memory-Komponenten
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🔧 Initializing Unified Memory System...")
            
            # 1. Initialize Short Term Memory
            self.stm = HumanLikeShortTermMemory(capacity=self.config.stm_capacity)
            logger.info(f"✅ STM initialized (capacity: {self.config.stm_capacity})")
            
            # 2. Initialize Long Term Memory
            ltm_config = self.config.ltm_config or {}
            self.ltm = HumanLikeLongTermMemory(**ltm_config)
            logger.info("✅ LTM initialized")
            
            # 3. Initialize Storage (if enabled)
            if self.config.enable_database:
                try:
                    from .storage.postgresql_storage import PostgreSQLMemoryStorage
                    self.storage = PostgreSQLMemoryStorage()
                    logger.info("✅ Database storage initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Database storage failed: {e}")
                    self.storage = None
            
            # 4. Initialize Conversation Memory (if enabled)
            if self.config.enable_conversations:
                self.conversation_memory = ConversationMemorySystem(
                    stm_system=self.stm,
                    ltm_system=self.ltm,
                    conversation_storage=self.storage
                )
                logger.info("✅ Conversation Memory initialized")
            
            # 5. Start auto-consolidation (if enabled)
            if self.config.auto_consolidation:
                self._schedule_auto_consolidation()
                logger.info("✅ Auto-consolidation scheduled")
            
            self.initialized = True
            logger.info("🎉 Unified Memory System fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory System initialization failed: {e}")
            self.initialized = False
            return False
    
    # ✅ UNIFIED MEMORY API
    
    def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.GENERAL,
        importance: int = 5,
        context: Optional[Dict[str, Any]] = None,
        force_ltm: bool = False
    ) -> Optional[Memory]:
        """
        Speichert Memory im geeigneten System (STM/LTM)
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance level (1-10)
            context: Additional context
            force_ltm: Force storage in LTM
        
        Returns:
            Created Memory object or None
        """
        try:
            if not self.initialized:
                logger.error("❌ Memory system not initialized")
                return None
            
            # Create memory object
            memory = create_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                context=context or {}
            )
            
            # Decide storage location
            if force_ltm or importance >= 7:
                # Store in LTM
                if self.ltm:
                    success = self.ltm.store_memory(memory)
                    if success:
                        logger.debug(f"💾 Memory stored in LTM: {memory.memory_id}")
                        self.stats['memories_created'] += 1
                        return memory
            else:
                # Store in STM
                if self.stm:
                    success = self.stm.store_memory(memory)
                    if success:
                        logger.debug(f"🧠 Memory stored in STM: {memory.memory_id}")
                        self.stats['memories_created'] += 1
                        return memory
            
            logger.warning(f"⚠️ Memory storage failed: {memory.memory_id}")
            self.stats['errors'] += 1
            return None
            
        except Exception as e:
            logger.error(f"❌ Memory storage error: {e}")
            self.stats['errors'] += 1
            return None
    
    def search_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        importance_min: int = 1,
        limit: int = 10
    ) -> List[Memory]:
        """
        Durchsucht alle Memory-Systeme
        
        Args:
            query: Search query
            memory_types: Filter by memory types
            importance_min: Minimum importance level
            limit: Maximum results
        
        Returns:
            List of matching memories
        """
        try:
            if not self.initialized:
                logger.error("❌ Memory system not initialized")
                return []
            
            all_memories = []
            
            # Search STM
            if self.stm:
                stm_memories = self.stm.search_memories(query, limit=limit//2)
                all_memories.extend(stm_memories)
            
            # Search LTM
            if self.ltm:
                ltm_memories = self.ltm.search_memories(query, limit=limit//2)
                all_memories.extend(ltm_memories)
            
            # Filter by criteria
            filtered_memories = []
            for memory in all_memories:
                # Filter by type
                if memory_types and memory.memory_type not in memory_types:
                    continue
                
                # Filter by importance
                if memory.importance < importance_min:
                    continue
                
                filtered_memories.append(memory)
            
            # Sort by relevance and importance
            filtered_memories.sort(
                key=lambda m: (
                    self._calculate_relevance(query, m.content),
                    m.importance
                ),
                reverse=True
            )
            
            return filtered_memories[:limit]
            
        except Exception as e:
            logger.error(f"❌ Memory search error: {e}")
            return []
    
    def consolidate_memories(self, force: bool = False) -> Dict[str, Any]:
        """
        Konsolidiert STM Memories zu LTM
        
        Args:
            force: Force consolidation even if not due
        
        Returns:
            Consolidation result
        """
        try:
            if not self.initialized or not self.stm or not self.ltm:
                return {'success': False, 'reason': 'system_not_ready'}
            
            # Check if consolidation is due
            if not force and self.last_consolidation:
                time_since_last = datetime.now() - self.last_consolidation
                if time_since_last.total_seconds() < self.config.consolidation_interval_minutes * 60:
                    return {'success': False, 'reason': 'too_soon'}
            
            # Get consolidation candidates from STM
            candidates = self.stm.get_consolidation_candidates()
            
            if not candidates:
                return {'success': True, 'consolidated': 0, 'reason': 'no_candidates'}
            
            # Transfer to LTM
            consolidated_count = 0
            for memory in candidates:
                if self.ltm.store_memory(memory):
                    self.stm.remove_memory(memory.memory_id)
                    consolidated_count += 1
            
            self.last_consolidation = datetime.now()
            self.stats['memories_consolidated'] += consolidated_count
            
            logger.info(f"✅ Consolidated {consolidated_count} memories to LTM")
            
            return {
                'success': True,
                'consolidated': consolidated_count,
                'candidates': len(candidates),
                'timestamp': self.last_consolidation.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Memory consolidation error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def process_conversation_exchange(
        self,
        user_input: str,
        ai_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verarbeitet einen kompletten Conversation Exchange
        
        Args:
            user_input: User's input
            ai_response: AI's response
            context: Additional context
        
        Returns:
            Processing result
        """
        try:
            if not self.initialized or not self.conversation_memory:
                return {'success': False, 'reason': 'conversation_system_not_ready'}
            
            # Process through conversation memory system
            result = await self.conversation_memory.process_conversation_exchange(
                user_input=user_input,
                kira_response=ai_response,
                context=context or {}
            )
            
            self.stats['conversations_processed'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Conversation processing error: {e}")
            self.stats['errors'] += 1
            return {'success': False, 'error': str(e)}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Holt detaillierte System-Statistiken
        
        Returns:
            System statistics
        """
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'initialized': self.initialized,
                'config': {
                    'stm_capacity': self.config.stm_capacity,
                    'enable_database': self.config.enable_database,
                    'enable_conversations': self.config.enable_conversations,
                    'auto_consolidation': self.config.auto_consolidation
                },
                'components': {},
                'statistics': self.stats.copy(),
                'memory_distribution': {}
            }
            
            # STM stats
            if self.stm:
                stm_stats = self.stm.get_stats()
                stats['components']['stm'] = {
                    'available': True,
                    'capacity': self.stm.capacity,
                    'current_size': len(self.stm.working_memory),
                    'utilization': len(self.stm.working_memory) / self.stm.capacity,
                    'details': stm_stats
                }
                stats['memory_distribution']['stm'] = len(self.stm.working_memory)
            else:
                stats['components']['stm'] = {'available': False}
                stats['memory_distribution']['stm'] = 0
            
            # LTM stats
            if self.ltm:
                ltm_stats = self.ltm.get_stats()
                stats['components']['ltm'] = {
                    'available': True,
                    'total_memories': len(self.ltm.consolidated_memories),
                    'details': ltm_stats
                }
                stats['memory_distribution']['ltm'] = len(self.ltm.consolidated_memories)
            else:
                stats['components']['ltm'] = {'available': False}
                stats['memory_distribution']['ltm'] = 0
            
            # Conversation memory stats
            if self.conversation_memory:
                conv_stats = self.conversation_memory.get_memory_stats()
                stats['components']['conversation_memory'] = {
                    'available': True,
                    'details': conv_stats
                }
            else:
                stats['components']['conversation_memory'] = {'available': False}
            
            # Storage stats
            if self.storage:
                stats['components']['storage'] = {
                    'available': True,
                    'type': type(self.storage).__name__
                }
            else:
                stats['components']['storage'] = {'available': False}
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ System stats error: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Überprüft System-Gesundheit
        
        Returns:
            Health status
        """
        try:
            health = {
                'overall_status': 'unknown',
                'timestamp': datetime.now().isoformat(),
                'components': {},
                'issues': [],
                'recommendations': []
            }
            
            # Check each component
            component_status = {}
            
            # STM health
            if self.stm:
                stm_utilization = len(self.stm.working_memory) / self.stm.capacity
                component_status['stm'] = {
                    'status': 'healthy',
                    'utilization': stm_utilization
                }
                
                if stm_utilization > 0.9:
                    health['issues'].append('STM near capacity')
                    health['recommendations'].append('Consider consolidation')
            else:
                component_status['stm'] = {'status': 'unavailable'}
                health['issues'].append('STM not available')
            
            # LTM health
            if self.ltm:
                component_status['ltm'] = {'status': 'healthy'}
            else:
                component_status['ltm'] = {'status': 'unavailable'}
                health['issues'].append('LTM not available')
            
            # Conversation memory health
            if self.conversation_memory:
                component_status['conversation_memory'] = {'status': 'healthy'}
            else:
                component_status['conversation_memory'] = {'status': 'unavailable'}
            
            # Storage health
            if self.storage:
                component_status['storage'] = {'status': 'healthy'}
            else:
                component_status['storage'] = {'status': 'unavailable'}
                health['recommendations'].append('Enable database storage for persistence')
            
            health['components'] = component_status
            
            # Determine overall status
            healthy_components = sum(1 for comp in component_status.values() if comp.get('status') == 'healthy')
            total_components = len(component_status)
            
            if healthy_components == total_components:
                health['overall_status'] = 'excellent'
            elif healthy_components >= total_components * 0.75:
                health['overall_status'] = 'good'
            elif healthy_components >= total_components * 0.5:
                health['overall_status'] = 'limited'
            else:
                health['overall_status'] = 'critical'
            
            return health
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # ✅ PRIVATE HELPER METHODS
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Berechnet Relevanz zwischen Query und Content"""
        try:
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if not query_words or not content_words:
                return 0.0
            
            # Simple word overlap calculation
            overlap = len(query_words.intersection(content_words))
            return overlap / len(query_words)
            
        except Exception:
            return 0.0
    
    def _schedule_auto_consolidation(self):
        """Plane automatische Konsolidierung"""
        try:
            # This would be implemented with a proper scheduler in production
            # For now, just log that it would be scheduled
            logger.info(f"Auto-consolidation scheduled every {self.config.consolidation_interval_minutes} minutes")
        except Exception as e:
            logger.error(f"Auto-consolidation scheduling failed: {e}")



class MemorySystemIntegration:
    """
    Legacy compatibility class for MemorySystemIntegration
    Wraps UnifiedMemorySystem for backward compatibility
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with backward compatibility"""
        self.unified_system = UnifiedMemorySystem(config)
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize the memory system"""
        try:
            success = self.unified_system.initialize()
            self.initialized = success
            return success
        except Exception as e:
            logger.error(f"MemorySystemIntegration initialization failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            if hasattr(self.unified_system, 'health_check'):
                health = self.unified_system.health_check()
                return {
                    'initialized': self.initialized,
                    'status': health.get('overall_status', 'unknown'),
                    'health': health
                }
            else:
                return {
                    'initialized': self.initialized,
                    'status': 'active' if self.initialized else 'offline'
                }
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {
                'initialized': False,
                'status': 'error',
                'error': str(e)
            }
    
    def store_memory(self, content: str, memory_type=None, importance: int = 5, context: Optional[Dict] = None):
        """Store memory - backward compatibility method"""
        try:
            from .core.memory_types import MemoryType
            if isinstance(memory_type, str):
                # Convert string to MemoryType if needed
                memory_type = getattr(MemoryType, memory_type.upper(), MemoryType.GENERAL)
            elif memory_type is None:
                memory_type = MemoryType.GENERAL
                
            return self.unified_system.store_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                context=context
            )
        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
            return None
    
    def search_memories(self, query: str, limit: int = 10):
        """Search memories - backward compatibility method"""
        try:
            return self.unified_system.search_memories(query=query, limit=limit)
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    def process_conversation(self, user_input: str, ai_response: str, context: Optional[Dict] = None):
        """Process conversation - backward compatibility method"""
        try:
            import asyncio
            if asyncio.iscoroutinefunction(self.unified_system.process_conversation_exchange):
                # If async, run in event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                return loop.run_until_complete(
                    self.unified_system.process_conversation_exchange(
                        user_input=user_input,
                        ai_response=ai_response,
                        context=context
                    )
                )
            else:
                # If sync, call directly
                return self.unified_system.process_conversation_exchange(
                    user_input=user_input,
                    ai_response=ai_response,
                    context=context
                )
        except Exception as e:
            logger.error(f"Conversation processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_stats(self):
        """Get system statistics"""
        try:
            return self.unified_system.get_system_stats()
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {'error': str(e)}
    
    def health_check(self):
        """Health check"""
        try:
            return self.unified_system.health_check()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    # Backward compatibility properties
    @property
    def is_initialized(self):
        return self.initialized
    
    @property
    def config(self):
        return getattr(self.unified_system, 'config', {})


# Export both classes
__all__ = ['UnifiedMemorySystem', 'MemorySystemConfig', 'MemorySystemIntegration']