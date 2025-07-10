"""
Basis Command System für Kira Voice
Definiert Struktur für alle Voice Commands
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class CommandCategory(Enum):
    """Command Kategorien"""
    GENERAL = "general"         # Allgemeine Commands
    SYSTEM = "system"           # System-Informationen  
    VOICE = "voice"             # Voice-Einstellungen
    TIME = "time"               # Zeit/Datum
    WEATHER = "weather"         # Wetter (falls implementiert)
    HELP = "help"               # Hilfe-Commands

@dataclass
class CommandResponse:
    """Response von einem Command"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    emotion: str = "neutral"
    action: Optional[str] = None

@dataclass
class CommandMatch:
    """Matching-Ergebnis für Commands"""
    matched: bool
    confidence: float
    command_name: str
    extracted_params: Dict[str, Any]

class BaseCommand(ABC):
    """Basis-Klasse für alle Voice Commands"""
    
    def __init__(self, name: str, category: CommandCategory, voice_system=None):
        self.name = name
        self.category = category
        self.voice_system = voice_system
        
        # Command Eigenschaften
        self.keywords = []          # Erkennungs-Keywords
        self.patterns = []          # Erkennungs-Pattern
        self.aliases = []           # Alternative Namen
        self.description = ""       # Beschreibung
        self.examples = []          # Beispiele
        
        # Statistiken
        self.execution_count = 0
        self.last_execution = None
        self.total_execution_time = 0.0
        
        logger.debug(f"📝 Command erstellt: {name} ({category.value})")
    
    @abstractmethod
    def execute(self, user_input: str, params: Dict[str, Any] = None) -> CommandResponse:
        """Führt das Command aus"""
        pass
    
    def match(self, user_input: str) -> CommandMatch:
        """Prüft ob Command zu User Input passt"""
        
        user_input_lower = user_input.lower().strip()
        
        try:
            # Exakte Keyword-Suche
            keyword_matches = []
            for keyword in self.keywords:
                if keyword.lower() in user_input_lower:
                    keyword_matches.append(keyword)
            
            # Alias-Suche
            alias_matches = []
            for alias in self.aliases:
                if alias.lower() in user_input_lower:
                    alias_matches.append(alias)
            
            # Berechne Confidence
            total_matches = len(keyword_matches) + len(alias_matches)
            
            if total_matches == 0:
                return CommandMatch(
                    matched=False,
                    confidence=0.0,
                    command_name=self.name,
                    extracted_params={}
                )
            
            # Einfache Confidence-Berechnung
            confidence = min(1.0, total_matches / len(self.keywords + self.aliases))
            
            # Parameter-Extraktion (einfach)
            extracted_params = self._extract_parameters(user_input_lower)
            
            return CommandMatch(
                matched=confidence > 0.3,  # Threshold
                confidence=confidence,
                command_name=self.name,
                extracted_params=extracted_params
            )
            
        except Exception as e:
            logger.error(f"❌ Command Match Fehler für {self.name}: {e}")
            return CommandMatch(
                matched=False,
                confidence=0.0,
                command_name=self.name,
                extracted_params={}
            )
    
    def _extract_parameters(self, user_input: str) -> Dict[str, Any]:
        """Extrahiert Parameter aus User Input (Basis-Implementierung)"""
        # Kann in Subklassen überschrieben werden
        return {}
    
    def execute_with_stats(self, user_input: str, params: Dict[str, Any] = None) -> CommandResponse:
        """Führt Command mit Statistik-Tracking aus"""
        
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Führe Command aus: {self.name}")
            
            # Command ausführen
            response = self.execute(user_input, params)
            
            # Statistiken aktualisieren
            execution_time = time.time() - start_time
            self.execution_count += 1
            self.last_execution = time.time()
            self.total_execution_time += execution_time
            
            logger.info(f"✅ Command {self.name} erfolgreich ({execution_time:.2f}s)")
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Command {self.name} Fehler: {e}")
            
            return CommandResponse(
                success=False,
                message=f"Entschuldigung, beim Ausführen von '{self.name}' ist ein Fehler aufgetreten.",
                emotion="apologetic"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Gibt Command-Statistiken zurück"""
        avg_time = (
            self.total_execution_time / self.execution_count 
            if self.execution_count > 0 else 0.0
        )
        
        return {
            'name': self.name,
            'category': self.category.value,
            'execution_count': self.execution_count,
            'last_execution': self.last_execution,
            'average_execution_time': avg_time,
            'total_execution_time': self.total_execution_time
        }
    
    def get_help(self) -> str:
        """Gibt Hilfe-Text für Command zurück"""
        help_text = f"**{self.name}** ({self.category.value})\n"
        
        if self.description:
            help_text += f"Beschreibung: {self.description}\n"
        
        if self.keywords:
            help_text += f"Keywords: {', '.join(self.keywords)}\n"
        
        if self.examples:
            help_text += f"Beispiele:\n"
            for example in self.examples:
                help_text += f"  - {example}\n"
        
        return help_text

# Export
__all__ = ['BaseCommand', 'CommandCategory', 'CommandResponse', 'CommandMatch']