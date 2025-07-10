"""
Debug Logger für Memory System
Spezielle Logging-Funktionen für Nachrichten und Personen-Daten
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from functools import wraps

# Erstelle Debug-Log-Verzeichnis
DEBUG_LOG_DIR = Path("logs/debug")
DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)

class MemoryDebugLogger:
    """
    Spezialisierter Logger für Memory-System Debugging
    """
    
    def __init__(self, name: str = "memory_debug"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Entferne vorhandene Handler
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Setup Debug-File Handler
        debug_file = DEBUG_LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(debug_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Setup Memory-spezifisches Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console Handler für wichtige Debug-Meldungen
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('🔍 %(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"🔍 Memory Debug Logger initialized: {debug_file}")
    
    def log_message_storage(self, 
                           message_content: str, 
                           user_id: str, 
                           conversation_id: str,
                           message_type: str,
                           memory_id: Optional[str] = None,
                           metadata: Optional[Dict] = None):
        """
        Loggt die Speicherung einer Nachricht
        """
        log_data = {
            "action": "message_storage",
            "timestamp": datetime.now().isoformat(),
            "message_content": message_content[:100] + "..." if len(message_content) > 100 else message_content,
            "message_length": len(message_content),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message_type": message_type,
            "memory_id": memory_id,
            "metadata": metadata or {}
        }
        
        self.logger.info(f"📝 MESSAGE STORED: {message_type} from {user_id} | ID: {memory_id}")
        self.logger.debug(f"MESSAGE_STORAGE_DETAIL: {json.dumps(log_data, ensure_ascii=False, indent=2)}")
    
    def log_person_data_extraction(self, 
                                 content: str, 
                                 extracted_names: List[str],
                                 extracted_info: Dict[str, Any],
                                 user_id: str):
        """
        Loggt die Extraktion von Personen-Daten
        """
        log_data = {
            "action": "person_data_extraction",
            "timestamp": datetime.now().isoformat(),
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "extracted_names": extracted_names,
            "extracted_info": extracted_info,
            "user_id": user_id,
            "names_count": len(extracted_names)
        }
        
        if extracted_names:
            self.logger.info(f"👤 PERSON DATA EXTRACTED: {len(extracted_names)} names from {user_id}")
            self.logger.info(f"   Names: {', '.join(extracted_names)}")
        
        self.logger.debug(f"PERSON_DATA_DETAIL: {json.dumps(log_data, ensure_ascii=False, indent=2)}")
    
    def log_brain_memory_operation(self, 
                                 operation: str,
                                 memory_id: str,
                                 content: str,
                                 strength: float,
                                 emotional_context: Dict[str, Any],
                                 connections: List[str] = None):
        """
        Loggt Brain Memory System Operationen
        """
        log_data = {
            "action": "brain_memory_operation",
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "memory_id": memory_id,
            "content_preview": content[:150] + "..." if len(content) > 150 else content,
            "content_length": len(content),
            "memory_strength": strength,
            "emotional_context": emotional_context,
            "neural_connections": connections or [],
            "connection_count": len(connections) if connections else 0
        }
        
        self.logger.info(f"🧠 BRAIN MEMORY {operation.upper()}: {memory_id} | Strength: {strength:.2f}")
        if connections:
            self.logger.info(f"   Neural connections: {len(connections)}")
        
        self.logger.debug(f"BRAIN_MEMORY_DETAIL: {json.dumps(log_data, ensure_ascii=False, indent=2)}")
    
    def log_storage_backend_operation(self, 
                                    operation: str,
                                    table_name: str,
                                    data_preview: Dict[str, Any],
                                    success: bool,
                                    error: Optional[str] = None):
        """
        Loggt Storage Backend Operationen
        """
        log_data = {
            "action": "storage_backend_operation",
            "operation": operation,
            "table_name": table_name,
            "timestamp": datetime.now().isoformat(),
            "data_preview": data_preview,
            "success": success,
            "error": error
        }
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.logger.info(f"💾 STORAGE {operation.upper()}: {table_name} | {status}")
        
        if error:
            self.logger.error(f"   Error: {error}")
        
        self.logger.debug(f"STORAGE_DETAIL: {json.dumps(log_data, ensure_ascii=False, indent=2)}")
    
    def log_conversation_flow(self, 
                            user_input: str, 
                            kira_response: str,
                            conversation_id: str,
                            user_id: str,
                            processing_time: float,
                            memory_operations: List[str]):
        """
        Loggt den kompletten Konversationsfluss
        """
        log_data = {
            "action": "conversation_flow",
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "user_id": user_id,
            "user_input": user_input,
            "kira_response": kira_response,
            "processing_time_ms": processing_time * 1000,
            "memory_operations": memory_operations
        }
        
        self.logger.info(f"💬 CONVERSATION: {user_id} | {len(memory_operations)} memory ops | {processing_time*1000:.0f}ms")
        self.logger.debug(f"CONVERSATION_DETAIL: {json.dumps(log_data, ensure_ascii=False, indent=2)}")
    
    def log_memory_search(self, 
                         query: str, 
                         results_count: int,
                         search_type: str,
                         processing_time: float,
                         context_cues: Optional[Dict] = None):
        """
        Loggt Memory-Suchen
        """
        log_data = {
            "action": "memory_search",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "search_type": search_type,
            "results_count": results_count,
            "processing_time_ms": processing_time * 1000,
            "context_cues": context_cues or {}
        }
        
        self.logger.info(f"🔍 MEMORY SEARCH: '{query}' | {results_count} results | {processing_time*1000:.0f}ms")
        self.logger.debug(f"SEARCH_DETAIL: {json.dumps(log_data, ensure_ascii=False, indent=2)}")

    def log_memory_consolidation(self, 
                               memory_id: str,
                               consolidation_level: str,
                               strength_change: float,
                               connections_formed: int,
                               consolidation_trigger: str):
        """
        Loggt Memory-Konsolidierung (STM -> LTM)
        """
        log_data = {
            "action": "memory_consolidation",
            "timestamp": datetime.now().isoformat(),
            "memory_id": memory_id,
            "consolidation_level": consolidation_level,
            "strength_change": strength_change,
            "connections_formed": connections_formed,
            "consolidation_trigger": consolidation_trigger
        }
        
        self.logger.info(f"🧠 MEMORY CONSOLIDATION: {memory_id} | Level: {consolidation_level} | Trigger: {consolidation_trigger}")
        self.logger.debug(f"CONSOLIDATION_DETAIL: {json.dumps(log_data, ensure_ascii=False, indent=2)}")

    def log_neural_connection(self, 
                            source_memory_id: str,
                            target_memory_id: str,
                            connection_strength: float,
                            connection_type: str,
                            activation_reason: str):
        """
        Loggt die Bildung neuronaler Verbindungen
        """
        log_data = {
            "action": "neural_connection",
            "timestamp": datetime.now().isoformat(),
            "source_memory_id": source_memory_id,
            "target_memory_id": target_memory_id,
            "connection_strength": connection_strength,
            "connection_type": connection_type,
            "activation_reason": activation_reason
        }
        
        self.logger.info(f"🔗 NEURAL CONNECTION: {source_memory_id} -> {target_memory_id} | Strength: {connection_strength:.2f}")
        self.logger.debug(f"CONNECTION_DETAIL: {json.dumps(log_data, ensure_ascii=False, indent=2)}")

    def log_emotional_memory_processing(self, 
                                      memory_id: str,
                                      emotion_type: str,
                                      emotional_intensity: float,
                                      emotional_valence: float,
                                      amygdala_activation: float):
        """
        Loggt emotionale Memory-Verarbeitung
        """
        log_data = {
            "action": "emotional_memory_processing",
            "timestamp": datetime.now().isoformat(),
            "memory_id": memory_id,
            "emotion_type": emotion_type,
            "emotional_intensity": emotional_intensity,
            "emotional_valence": emotional_valence,
            "amygdala_activation": amygdala_activation
        }
        
        self.logger.info(f"💭 EMOTIONAL MEMORY: {memory_id} | {emotion_type} | Intensity: {emotional_intensity:.2f}")
        self.logger.debug(f"EMOTIONAL_DETAIL: {json.dumps(log_data, ensure_ascii=False, indent=2)}")

    def log_person_context_update(self, 
                                 person_name: str,
                                 context_type: str,
                                 old_context: Dict[str, Any],
                                 new_context: Dict[str, Any],
                                 confidence_score: float):
        """
        Loggt Updates zu Personen-Kontexten
        """
        log_data = {
            "action": "person_context_update",
            "timestamp": datetime.now().isoformat(),
            "person_name": person_name,
            "context_type": context_type,
            "old_context": old_context,
            "new_context": new_context,
            "confidence_score": confidence_score,
            "context_changes": self._calculate_context_changes(old_context, new_context)
        }
        
        self.logger.info(f"👤 PERSON CONTEXT UPDATE: {person_name} | {context_type} | Confidence: {confidence_score:.2f}")
        self.logger.debug(f"PERSON_CONTEXT_DETAIL: {json.dumps(log_data, ensure_ascii=False, indent=2)}")

    def log_memory_retrieval_pattern(self, 
                                   user_id: str,
                                   retrieval_cues: List[str],
                                   retrieved_memories: List[str],
                                   retrieval_strength: float,
                                   retrieval_time: float):
        """
        Loggt Memory-Abruf-Muster
        """
        log_data = {
            "action": "memory_retrieval_pattern",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "retrieval_cues": retrieval_cues,
            "retrieved_memories": retrieved_memories,
            "retrieval_strength": retrieval_strength,
            "retrieval_time_ms": retrieval_time * 1000,
            "memory_count": len(retrieved_memories)
        }
        
        self.logger.info(f"🔍 MEMORY RETRIEVAL: {user_id} | {len(retrieved_memories)} memories | {retrieval_time*1000:.0f}ms")
        self.logger.debug(f"RETRIEVAL_DETAIL: {json.dumps(log_data, ensure_ascii=False, indent=2)}")

    def log_storage_performance_metrics(self, 
                                      operation: str,
                                      execution_time: float,
                                      memory_usage: float,
                                      records_processed: int,
                                      cache_hit_rate: float):
        """
        Loggt Storage Performance Metriken
        """
        log_data = {
            "action": "storage_performance_metrics",
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "execution_time_ms": execution_time * 1000,
            "memory_usage_mb": memory_usage,
            "records_processed": records_processed,
            "cache_hit_rate": cache_hit_rate,
            "throughput_records_per_sec": records_processed / execution_time if execution_time > 0 else 0
        }
        
        self.logger.info(f"📊 STORAGE PERFORMANCE: {operation} | {execution_time*1000:.0f}ms | {records_processed} records")
        self.logger.debug(f"PERFORMANCE_DETAIL: {json.dumps(log_data, ensure_ascii=False, indent=2)}")

    def _calculate_context_changes(self, old_context: Dict[str, Any], new_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Berechnet Änderungen zwischen altem und neuem Kontext
        """
        changes = {
            "added_keys": [],
            "removed_keys": [],
            "modified_keys": [],
            "unchanged_keys": []
        }
        
        all_keys = set(old_context.keys()) | set(new_context.keys())
        
        for key in all_keys:
            if key not in old_context:
                changes["added_keys"].append(key)
            elif key not in new_context:
                changes["removed_keys"].append(key)
            elif old_context[key] != new_context[key]:
                changes["modified_keys"].append(key)
            else:
                changes["unchanged_keys"].append(key)
        
        return changes


# Globaler Debug Logger
debug_logger = MemoryDebugLogger("memory_debug")

def debug_memory_operation(operation_name: str):
    """
    Decorator für automatisches Logging von Memory-Operationen
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                debug_logger.logger.info(f"✅ {operation_name} completed in {processing_time*1000:.0f}ms")
                
                return result
                
            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                debug_logger.logger.error(f"❌ {operation_name} failed after {processing_time*1000:.0f}ms: {e}")
                raise
                
        return wrapper
    return decorator

def log_data_flow(data_type: str, data: Any, context: str = ""):
    """
    Einfache Funktion zum Loggen von Datenflüssen
    """
    if isinstance(data, dict):
        data_preview = {k: str(v)[:50] + "..." if len(str(v)) > 50 else v for k, v in data.items()}
    elif isinstance(data, list):
        data_preview = f"List with {len(data)} items: {str(data[:3])}"
    else:
        data_preview = str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
    
    debug_logger.logger.debug(f"📊 DATA FLOW: {data_type} | {context} | {data_preview}")

def create_memory_debug_report() -> Dict[str, Any]:
    """
    Erstellt einen Debug-Report über das Memory-System
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "debug_log_files": [],
        "recent_operations": [],
        "system_status": "active"
    }
    
    # Sammle Debug-Log-Dateien
    for log_file in DEBUG_LOG_DIR.glob("*.log"):
        file_stats = log_file.stat()
        report["debug_log_files"].append({
            "filename": log_file.name,
            "size_mb": file_stats.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        })
    
    debug_logger.logger.info(f"📋 DEBUG REPORT CREATED: {len(report['debug_log_files'])} log files")
    return report

# Hilfsfunktionen für spezifische Logging-Szenarien
def log_name_extraction(content: str, names: List[str], user_id: str):
    """Einfache Funktion für Name-Extraktion Logging"""
    debug_logger.log_person_data_extraction(
        content=content,
        extracted_names=names,
        extracted_info={"names": names},
        user_id=user_id
    )

def log_message_save(content: str, user_id: str, conversation_id: str, message_type: str, memory_id: str = None):
    """Einfache Funktion für Message-Save Logging"""
    debug_logger.log_message_storage(
        message_content=content,
        user_id=user_id,
        conversation_id=conversation_id,
        message_type=message_type,
        memory_id=memory_id
    )

def log_brain_memory_store(memory_id: str, content: str, strength: float, emotional_context: Dict):
    """Einfache Funktion für Brain Memory Store Logging"""
    debug_logger.log_brain_memory_operation(
        operation="store",
        memory_id=memory_id,
        content=content,
        strength=strength,
        emotional_context=emotional_context
    )

def log_memory_consolidation(memory_id: str, consolidation_level: str, strength_change: float, connections_formed: int, trigger: str):
    """Einfache Funktion für Memory Consolidation Logging"""
    debug_logger.log_memory_consolidation(
        memory_id=memory_id,
        consolidation_level=consolidation_level,
        strength_change=strength_change,
        connections_formed=connections_formed,
        consolidation_trigger=trigger
    )

def log_person_context_update(person_name: str, context_type: str, old_context: Dict, new_context: Dict, confidence: float):
    """Einfache Funktion für Person Context Update Logging"""
    debug_logger.log_person_context_update(
        person_name=person_name,
        context_type=context_type,
        old_context=old_context,
        new_context=new_context,
        confidence_score=confidence
    )

def log_storage_operation(operation: str, table_name: str, data_preview: Dict, success: bool, error: str = None):
    """Einfache Funktion für Storage Operation Logging"""
    debug_logger.log_storage_backend_operation(
        operation=operation,
        table_name=table_name,
        data_preview=data_preview,
        success=success,
        error=error
    )

def log_neural_connection(source_id: str, target_id: str, strength: float, connection_type: str, reason: str):
    """Einfache Funktion für Neural Connection Logging"""
    debug_logger.log_neural_connection(
        source_memory_id=source_id,
        target_memory_id=target_id,
        connection_strength=strength,
        connection_type=connection_type,
        activation_reason=reason
    )

def log_emotional_processing(memory_id: str, emotion_type: str, intensity: float, valence: float, amygdala_activation: float):
    """Einfache Funktion für Emotional Processing Logging"""
    debug_logger.log_emotional_memory_processing(
        memory_id=memory_id,
        emotion_type=emotion_type,
        emotional_intensity=intensity,
        emotional_valence=valence,
        amygdala_activation=amygdala_activation
    )

def log_memory_retrieval(user_id: str, cues: List[str], retrieved: List[str], strength: float, retrieval_time: float):
    """Einfache Funktion für Memory Retrieval Logging"""
    debug_logger.log_memory_retrieval_pattern(
        user_id=user_id,
        retrieval_cues=cues,
        retrieved_memories=retrieved,
        retrieval_strength=strength,
        retrieval_time=retrieval_time
    )

def log_storage_performance(operation: str, execution_time: float, memory_usage: float, records: int, cache_hit_rate: float):
    """Einfache Funktion für Storage Performance Logging"""
    debug_logger.log_storage_performance_metrics(
        operation=operation,
        execution_time=execution_time,
        memory_usage=memory_usage,
        records_processed=records,
        cache_hit_rate=cache_hit_rate
    )
