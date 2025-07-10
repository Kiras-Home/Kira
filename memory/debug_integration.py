"""
Debug Integration Module für Memory System
Mixin-Klassen und Decorators für einfache Integration des Debug-Loggings
"""

from typing import Dict, Any, Optional, List, Callable
from functools import wraps
import json
import traceback
from datetime import datetime
from utils.debug_logger import MemoryDebugLogger

class MemoryDebugMixin:
    """
    Mixin-Klasse für Debug-Logging in Memory-Systemen
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_logger = MemoryDebugLogger(f"{self.__class__.__name__}_debug")
    
    def log_memory_operation(self, operation: str, data: Dict[str, Any], result: Any = None):
        """Log Memory-Operation mit strukturierten Daten"""
        self.debug_logger.log_memory_operation(operation, data, result)
    
    def log_person_data(self, person_name: str, data: Dict[str, Any]):
        """Log Personen-Daten"""
        self.debug_logger.log_person_data(person_name, data)
    
    def log_storage_operation(self, operation: str, data: Dict[str, Any], result: Any = None):
        """Log Storage-Operation"""
        self.debug_logger.log_storage_operation(operation, data, result)

class StorageDebugMixin:
    """
    Mixin-Klasse für Debug-Logging in Storage-Systemen
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_logger = MemoryDebugLogger(f"{self.__class__.__name__}_storage_debug")
    
    def log_storage_performance(self, operation: str, duration: float, data_size: int):
        """Log Storage-Performance"""
        self.debug_logger.log_storage_performance_metrics(operation, duration, data_size)
    
    def log_data_consolidation(self, consolidation_type: str, data: Dict[str, Any]):
        """Log Daten-Konsolidierung"""
        self.debug_logger.log_memory_consolidation(consolidation_type, data)

def memory_debug_decorator(operation_name: str = None):
    """
    Decorator für automatisches Debug-Logging von Memory-Operationen
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Erstelle Logger falls nicht vorhanden
            if not hasattr(self, 'debug_logger'):
                self.debug_logger = MemoryDebugLogger(f"{self.__class__.__name__}_debug")
            
            op_name = operation_name or func.__name__
            
            # Log Start der Operation
            self.debug_logger.logger.debug(f"Starting {op_name} with args: {args}, kwargs: {kwargs}")
            
            try:
                result = func(self, *args, **kwargs)
                
                # Log erfolgreiches Ergebnis
                self.debug_logger.logger.debug(f"Completed {op_name} successfully")
                
                return result
                
            except Exception as e:
                # Log Fehler
                self.debug_logger.logger.error(f"Error in {op_name}: {str(e)}")
                self.debug_logger.logger.debug(f"Traceback: {traceback.format_exc()}")
                raise
                
        return wrapper
    return decorator

def storage_debug_decorator(operation_name: str = None):
    """
    Decorator für automatisches Debug-Logging von Storage-Operationen
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Erstelle Logger falls nicht vorhanden
            if not hasattr(self, 'debug_logger'):
                self.debug_logger = MemoryDebugLogger(f"{self.__class__.__name__}_storage_debug")
            
            op_name = operation_name or func.__name__
            
            # Log Start der Operation
            self.debug_logger.logger.debug(f"Starting storage operation: {op_name}")
            
            try:
                result = func(self, *args, **kwargs)
                
                # Log erfolgreiches Ergebnis
                self.debug_logger.logger.debug(f"Storage operation {op_name} completed successfully")
                
                return result
                
            except Exception as e:
                # Log Fehler
                self.debug_logger.logger.error(f"Storage operation {op_name} failed: {str(e)}")
                self.debug_logger.logger.debug(f"Traceback: {traceback.format_exc()}")
                raise
                
        return wrapper
    return decorator

def log_person_extraction(func: Callable) -> Callable:
    """
    Decorator für Personen-Extraktion
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'debug_logger'):
            self.debug_logger = MemoryDebugLogger(f"{self.__class__.__name__}_person_debug")
        
        self.debug_logger.logger.debug("Starting person extraction...")
        
        try:
            result = func(self, *args, **kwargs)
            
            # Log extrahierte Personen
            if isinstance(result, list):
                for person in result:
                    if isinstance(person, dict) and 'name' in person:
                        self.debug_logger.log_person_data(person['name'], person)
            
            self.debug_logger.logger.debug(f"Person extraction completed: {len(result) if result else 0} persons found")
            
            return result
            
        except Exception as e:
            self.debug_logger.logger.error(f"Person extraction failed: {str(e)}")
            raise
            
    return wrapper

class DebugReportGenerator:
    """
    Generator für Debug-Reports
    """
    
    def __init__(self, logger: MemoryDebugLogger):
        self.logger = logger
    
    def generate_memory_report(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generiere Memory-Debug-Report
        """
        report = {
            "timestamp": str(datetime.now()),
            "memory_statistics": {
                "total_messages": len(memory_data.get('messages', [])),
                "total_persons": len(memory_data.get('persons', [])),
                "memory_size": len(str(memory_data))
            },
            "recent_operations": [],
            "health_status": "healthy"
        }
        
        self.logger.logger.info(f"Generated memory report: {json.dumps(report, indent=2)}")
        return report
    
    def generate_storage_report(self, storage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generiere Storage-Debug-Report
        """
        report = {
            "timestamp": str(datetime.now()),
            "storage_statistics": {
                "total_records": len(storage_data.get('records', [])),
                "storage_size": len(str(storage_data))
            },
            "performance_metrics": storage_data.get('performance', {}),
            "health_status": "healthy"
        }
        
        self.logger.logger.info(f"Generated storage report: {json.dumps(report, indent=2)}")
        return report

# Utility-Funktionen für Debug-Integration
def setup_debug_logging(class_instance, debug_name: str = None):
    """
    Setup Debug-Logging für eine Klassen-Instanz
    """
    if not hasattr(class_instance, 'debug_logger'):
        name = debug_name or f"{class_instance.__class__.__name__}_debug"
        class_instance.debug_logger = MemoryDebugLogger(name)
    
    return class_instance.debug_logger

def log_function_call(logger: MemoryDebugLogger, func_name: str, args: tuple, kwargs: dict, result: Any = None):
    """
    Log einen Funktionsaufruf mit Parametern und Ergebnis
    """
    logger.logger.debug(f"Function call: {func_name}")
    logger.logger.debug(f"Args: {args}")
    logger.logger.debug(f"Kwargs: {kwargs}")
    if result is not None:
        logger.logger.debug(f"Result: {result}")
