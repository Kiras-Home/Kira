"""
Complete Commands für Kira Voice System
Basic + Enhanced Commands mit Fallback-Kompatibilität
"""

import logging
from typing import Dict, Any, List, Optional
import time
import datetime
import platform
import psutil
import random
import re
import difflib
from dataclasses import dataclass
from enum import Enum

# ✅ FIX: Define missing classes locally if base_command import fails
try:
    from .base_command import BaseCommand, CommandCategory, CommandResponse
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ BaseCommand not found, creating fallback classes")
    
    class CommandCategory(Enum):
        GENERAL = "general"
        SYSTEM = "system"
        VOICE = "voice"
        TIME = "time"
        HELP = "help"
    
    @dataclass
    class CommandResponse:
        success: bool
        message: str
        emotion: str = "neutral"
        action: str = None
        data: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.data is None:
                self.data = {}
    
    class BaseCommand:
        def __init__(self, name: str, category: CommandCategory, voice_system=None):
            self.name = name
            self.category = category
            self.voice_system = voice_system
            self.enabled = True
            self.keywords = []
            self.aliases = []
            self.description = ""
            self.examples = []
        
        def execute_with_stats(self, user_input: str, params: Dict[str, Any] = None):
            return self.execute(user_input, params)
        
        def execute(self, user_input: str, params: Dict[str, Any] = None):
            return CommandResponse(
                success=True,
                message="Command executed",
                emotion="neutral"
            )
        
        def match(self, user_input: str):
            # Simple matching fallback
            user_lower = user_input.lower()
            confidence = 0.0
            
            for keyword in self.keywords:
                if keyword.lower() in user_lower:
                    confidence = max(confidence, 0.8)
            
            return type('Match', (), {
                'matched': confidence > 0.5,
                'confidence': confidence,
                'extracted_params': {}
            })()

logger = logging.getLogger(__name__)

class StatusCommand(BaseCommand):
    """Basic Status Command - Fallback für Kompatibilität"""
    
    def __init__(self, voice_system=None):
        super().__init__("status", CommandCategory.SYSTEM, voice_system)
        self.keywords = ["status", "wie geht es dir", "alles ok", "system status"]
        self.description = "Zeigt System-Status"
    
    def execute(self, user_input: str, params: Dict[str, Any] = None) -> CommandResponse:
        try:
            response = "Mir geht es gut! Alle Systeme laufen normal."
            
            return CommandResponse(
                success=True,
                message=response,
                emotion="helpful",
                data={
                    "system_status": "healthy",
                    "uptime": "läuft seit Start"
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Status command error: {e}")
            return CommandResponse(
                success=False,
                message="Entschuldigung, ich konnte den Status nicht abrufen.",
                emotion="apologetic"
            )
        
class WeatherCommand(BaseCommand):
    """Wetter-Command"""
    
    def __init__(self, voice_system=None):
        super().__init__("weather", CommandCategory.GENERAL, voice_system)
        self.keywords = ["wetter", "wettervorhersage", "wie ist das wetter", "regnet es"]
        self.description = "Gibt Wetterinformationen"
    
    def execute(self, user_input: str, params: Dict[str, Any] = None) -> CommandResponse:
        try:
            # Simulated weather info
            response = "Das Wetter ist heute schön mit 22 Grad und Sonnenschein."
            
            return CommandResponse(
                success=True,
                message=response,
                emotion="helpful",
                data={
                    "temperature": 22,
                    "condition": "sunny",
                    "location": "aktuelle Position"
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Weather command error: {e}")
            return CommandResponse(
                success=False,
                message="Entschuldigung, ich konnte das Wetter nicht abrufen.",
                emotion="apologetic"
            )

@dataclass
class EnhancedMatch:
    """Erweiterte Match Information"""
    matched: bool
    confidence: float
    method: str
    extracted_params: Dict[str, Any]
    intent: Optional[str] = None
    entities: Dict[str, Any] = None
    alternatives: List[str] = None

# ✅ FIX: Add basic GreetingCommand (alias for Enhanced)
class GreetingCommand(BaseCommand):
    """Basic Greeting Command - Fallback für Kompatibilität"""
    
    def __init__(self, voice_system=None):
        super().__init__("greeting", CommandCategory.GENERAL, voice_system)
        self.keywords = ["hallo", "hi", "hey", "guten", "morgen", "tag", "abend"]
        self.description = "Einfache Begrüßung"
    
    def execute(self, user_input: str, params: Dict[str, Any] = None) -> CommandResponse:
        current_hour = datetime.datetime.now().hour
        
        if 5 <= current_hour < 12:
            message = "Guten Morgen! Wie kann ich dir helfen?"
        elif 12 <= current_hour < 18:
            message = "Guten Tag! Was kann ich für dich tun?"
        else:
            message = "Guten Abend! Wie geht es dir?"
        
        return CommandResponse(
            success=True,
            message=message,
            emotion="friendly"
        )

class EnhancedGreetingCommand(BaseCommand):
    """Erweiterte Begrüßungs-Command mit Context Awareness"""
    
    def __init__(self, voice_system=None):
        super().__init__("greeting", CommandCategory.GENERAL, voice_system)
        
        # ✅ ERWEITERTE KEYWORDS mit Variations
        self.keywords = ["hallo", "hi", "hey", "guten", "morgen", "tag", "abend"]
        self.aliases = ["grüß dich", "servus", "moin", "wie geht's", "was machst du"]
        self.description = "Intelligente Begrüßung mit Tageszeit und Context"
        self.examples = [
            "Hallo Kira",
            "Guten Tag",
            "Hi, wie geht's?",
            "Morgen Kira",
            "Was machst du gerade?"
        ]
        
        # ✅ INTENT PATTERNS für bessere Erkennung
        self.intent_patterns = [
            r'hallo|hi|hey|grüß.*dich',
            r'guten\s*(morgen|tag|abend)',
            r'servus|moin|salü|tach',
            r'wie\s*geht\'?s|was\s*machst.*du',
            r'schön.*dich.*zu.*hören'
        ]
    
    def enhanced_match(self, user_input: str) -> EnhancedMatch:
        """Erweiterte Matching-Logik"""
        user_lower = user_input.lower()
        confidence = 0.0
        method = "none"
        intent = None
        entities = {}
        
        # ✅ 1. INTENT PATTERN MATCHING
        for pattern in self.intent_patterns:
            if re.search(pattern, user_lower):
                confidence = max(confidence, 0.9)
                method = "intent_pattern"
                intent = "greeting"
                break
        
        # ✅ 2. KEYWORD MATCHING mit Fuzzy
        keyword_matches = 0
        for keyword in self.keywords:
            if keyword in user_lower:
                keyword_matches += 1
            else:
                # Fuzzy matching für Tippfehler
                for word in user_lower.split():
                    similarity = difflib.SequenceMatcher(None, keyword, word).ratio()
                    if similarity > 0.8:
                        keyword_matches += 0.5
                        break
        
        if keyword_matches > 0:
            keyword_confidence = min(keyword_matches / len(self.keywords) * 2, 1.0)
            if keyword_confidence > confidence:
                confidence = keyword_confidence
                method = "keyword_fuzzy"
        
        # ✅ 3. ENTITY EXTRACTION
        if re.search(r'wie\s*geht\'?s|was.*machst', user_lower):
            entities['inquiry_type'] = 'wellbeing'
        
        time_match = re.search(r'(morgen|mittag|abend|nacht)', user_lower)
        if time_match:
            entities['time_of_day'] = time_match.group(1)
        
        return EnhancedMatch(
            matched=confidence > 0.4,
            confidence=confidence,
            method=method,
            extracted_params=entities,
            intent=intent,
            entities=entities
        )
    
    def execute(self, user_input: str, params: Dict[str, Any] = None) -> CommandResponse:
        """Erweiterte Begrüßung mit Context"""
        
        if params is None:
            params = {}
        
        current_hour = datetime.datetime.now().hour
        
        # ✅ CONTEXT-AWARE GREETING SELECTION
        inquiry_type = params.get('inquiry_type')
        time_of_day = params.get('time_of_day')
        
        if inquiry_type == 'wellbeing':
            responses = [
                "Mir geht es sehr gut, danke der Nachfrage! Wie geht es dir denn?",
                "Ich bin in bester Verfassung! Und wie läuft dein Tag?",
                "Prima! Alle Systeme laufen optimal. Wie steht es um dich?"
            ]
        elif time_of_day:
            if time_of_day == "morgen":
                responses = [
                    "Guten Morgen! Ich hoffe, du bist gut in den Tag gestartet!",
                    "Morgen! Bereit für einen produktiven Tag?",
                    "Guten Morgen! Was steht heute auf dem Programm?"
                ]
            elif time_of_day == "abend":
                responses = [
                    "Guten Abend! Wie war dein Tag?",
                    "Schönen Abend! Zeit zum Entspannen?",
                    "Guten Abend! Lass uns den Tag ausklingen lassen."
                ]
            else:
                responses = [f"Guten {time_of_day}! Schön, von dir zu hören!"]
        else:
            # Standard Tageszeit-abhängige Begrüßung
            if 5 <= current_hour < 12:
                responses = [
                    "Guten Morgen! Ich hoffe, du bist gut in den Tag gestartet.",
                    "Morgen! Wie kann ich dir heute helfen?",
                    "Guten Morgen! Schön, dass du da bist."
                ]
            elif 12 <= current_hour < 18:
                responses = [
                    "Hallo! Wie läuft dein Tag?",
                    "Hi! Schön, von dir zu hören.",
                    "Guten Tag! Was kann ich für dich tun?"
                ]
            else:
                responses = [
                    "Guten Abend! Wie war dein Tag?",
                    "Hallo! Schön, dass du noch da bist.",
                    "Guten Abend! Zeit für ein entspanntes Gespräch."
                ]
        
        message = random.choice(responses)
        
        return CommandResponse(
            success=True,
            message=message,
            emotion="friendly",
            data={
                'greeting_type': inquiry_type or 'standard',
                'time_of_day': time_of_day or f"hour_{current_hour}",
                'context_params': params
            }
        )

# ✅ FIX: Add missing TimeCommand
class TimeCommand(BaseCommand):
    """Zeit-Command"""
    
    def __init__(self, voice_system=None):
        super().__init__("time", CommandCategory.TIME, voice_system)
        self.keywords = ["zeit", "uhr", "uhrzeit", "spät", "wie viel uhr"]
        self.description = "Gibt die aktuelle Zeit an"
    
    def execute(self, user_input: str, params: Dict[str, Any] = None) -> CommandResponse:
        try:
            now = datetime.datetime.now()
            time_str = now.strftime("%H:%M")
            date_str = now.strftime("%d.%m.%Y")
            
            response = f"Es ist {time_str} Uhr am {date_str}."
            
            return CommandResponse(
                success=True,
                message=response,
                emotion="neutral",
                data={
                    "time": time_str,
                    "date": date_str,
                    "timestamp": now.timestamp()
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Time command error: {e}")
            return CommandResponse(
                success=False,
                message="Entschuldigung, ich konnte die Zeit nicht abrufen.",
                emotion="apologetic"
            )

# ✅ FIX: Add missing VoiceSettingsCommand
class VoiceSettingsCommand(BaseCommand):
    """Voice Settings Command"""
    
    def __init__(self, voice_system=None):
        super().__init__("voice_settings", CommandCategory.VOICE, voice_system)
        self.keywords = ["voice", "stimme", "einstellungen", "lautstärke", "geschwindigkeit"]
        self.description = "Voice-Einstellungen verwalten"
    
    def execute(self, user_input: str, params: Dict[str, Any] = None) -> CommandResponse:
        try:
            settings_info = "Voice-Einstellungen sind aktiv. Sprache: Deutsch, Geschwindigkeit: Normal."
            
            return CommandResponse(
                success=True,
                message=settings_info,
                emotion="helpful",
                data={
                    "language": "de",
                    "speed": "normal",
                    "volume": "medium"
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Voice settings command error: {e}")
            return CommandResponse(
                success=False,
                message="Entschuldigung, ich konnte die Voice-Einstellungen nicht abrufen.",
                emotion="apologetic"
            )

# ✅ FIX: Add missing HelpCommand
class HelpCommand(BaseCommand):
    """Hilfe-Command"""
    
    def __init__(self, voice_system=None, command_processor=None):
        super().__init__("help", CommandCategory.HELP, voice_system)
        self.keywords = ["hilfe", "help", "befehle", "kommandos", "was kannst du"]
        self.description = "Zeigt verfügbare Befehle"
        self.command_processor = command_processor
    
    def execute(self, user_input: str, params: Dict[str, Any] = None) -> CommandResponse:
        try:
            help_text = """Ich kann dir mit folgenden Befehlen helfen:

• Begrüßung: Sage 'Hallo' oder 'Hi'
• Zeit: Frage 'Wie spät ist es?'
• Status: Frage 'Wie geht es dir?'
• Voice-Einstellungen: Sage 'Voice Einstellungen'
• Hilfe: Sage 'Hilfe' für diese Liste
• Beenden: Sage 'Auf Wiedersehen'

Sprich einfach natürlich mit mir!"""
            
            return CommandResponse(
                success=True,
                message=help_text,
                emotion="helpful"
            )
            
        except Exception as e:
            logger.error(f"❌ Help command error: {e}")
            return CommandResponse(
                success=False,
                message="Entschuldigung, ich konnte die Hilfe nicht anzeigen.",
                emotion="apologetic"
            )

# ✅ FIX: Add missing ExitCommand
class ExitCommand(BaseCommand):
    """Exit/Beenden Command"""
    
    def __init__(self, voice_system=None):
        super().__init__("exit", CommandCategory.GENERAL, voice_system)
        self.keywords = ["auf wiedersehen", "tschüss", "beenden", "exit", "quit", "stop"]
        self.description = "Beendet die Voice-Session"
    
    def execute(self, user_input: str, params: Dict[str, Any] = None) -> CommandResponse:
        try:
            responses = [
                "Auf Wiedersehen! Es war schön, mit dir zu sprechen.",
                "Tschüss! Bis zum nächsten Mal.",
                "Auf Wiedersehen! Hab einen schönen Tag.",
                "Bis bald! Pass auf dich auf."
            ]
            
            return CommandResponse(
                success=True,
                message=random.choice(responses),
                emotion="friendly",
                action="exit"
            )
            
        except Exception as e:
            logger.error(f"❌ Exit command error: {e}")
            return CommandResponse(
                success=False,
                message="Auf Wiedersehen!",
                emotion="neutral",
                action="exit"
            )

# ✅ STATUS COMMAND bleibt wie es ist (bereits enhanced)
class EnhancedStatusCommand(BaseCommand):
    """Erweiterte System-Status Command mit detaillierter Analyse"""
    
    def __init__(self, voice_system=None):
        super().__init__("status", CommandCategory.SYSTEM, voice_system)
        
        self.keywords = ["status", "wie", "geht", "läuft", "system", "gesundheit"]
        self.aliases = ["zustand", "health", "check", "diagnose"]
        self.description = "Detaillierter System-Status mit Komponenten-Check"
        
        self.intent_patterns = [
            r'(wie\s*geht.*dir|wie.*läuft)',
            r'status|zustand|gesundheit',
            r'system.*check|health.*check',
            r'alles.*okay|funktioniert.*alles'
        ]
    
    def enhanced_match(self, user_input: str) -> EnhancedMatch:
        """Enhanced Status Matching"""
        user_lower = user_input.lower()
        confidence = 0.0
        method = "none"
        entities = {}
        
        # Intent matching
        for pattern in self.intent_patterns:
            if re.search(pattern, user_lower):
                confidence = 0.9
                method = "intent_pattern"
                break
        
        # Component-specific inquiry
        if re.search(r'memory|speicher|gedächtnis', user_lower):
            entities['component'] = 'memory'
            confidence = max(confidence, 0.8)
        elif re.search(r'voice|stimme|sprache', user_lower):
            entities['component'] = 'voice'
            confidence = max(confidence, 0.8)
        elif re.search(r'cpu|prozessor|performance', user_lower):
            entities['component'] = 'cpu'
            confidence = max(confidence, 0.8)
        
        return EnhancedMatch(
            matched=confidence > 0.4,
            confidence=confidence,
            method=method,
            extracted_params=entities,
            entities=entities
        )
    
    def execute(self, user_input: str, params: Dict[str, Any] = None) -> CommandResponse:
        """Erweiterte Status-Analyse"""
        
        if params is None:
            params = {}
        
        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            component = params.get('component')
            
            if component == 'memory':
                message = f"Memory läuft mit {memory_percent:.1f}% Auslastung."
                emotion = "analytical"
            elif component == 'cpu':
                message = f"CPU arbeitet mit {cpu_percent:.1f}% Auslastung."
                emotion = "analytical"
            elif component == 'voice':
                message = "Voice System ist aktiv und bereit."
                emotion = "helpful"
            else:
                # Comprehensive status
                if cpu_percent < 50 and memory_percent < 70:
                    message = "Mir geht es ausgezeichnet! Alle Systeme laufen optimal."
                    emotion = "happy"
                elif cpu_percent < 80 and memory_percent < 85:
                    message = "Mir geht es gut. Das System läuft stabil."
                    emotion = "content"
                else:
                    message = "Das System ist etwas belastet, funktioniert aber noch gut."
                    emotion = "concerned"
            
            return CommandResponse(
                success=True,
                message=message,
                emotion=emotion,
                data={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'component_focus': component
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Enhanced Status Command Fehler: {e}")
            return CommandResponse(
                success=False,
                message="Entschuldigung, ich kann den System-Status gerade nicht abrufen.",
                emotion="apologetic"
            )

# ✅ ENHANCED COMMAND PROCESSOR (vereinfacht)
class EnhancedCommandProcessor:
    """Enhanced Command Processor mit NLU und Context Awareness"""
    
    def __init__(self, voice_system=None):
        self.voice_system = voice_system
        self.commands = []
        self.conversation_history = []
        self.user_preferences = {}
        
        # Load commands
        self._load_enhanced_commands()
        
        logger.info(f"🧠 Enhanced Command Processor: {len(self.commands)} commands loaded")
    
    def _load_enhanced_commands(self):
        """Lädt Enhanced Commands"""
        
        self.commands = [
            EnhancedGreetingCommand(self.voice_system),
            EnhancedStatusCommand(self.voice_system),
            TimeCommand(self.voice_system),
            VoiceSettingsCommand(self.voice_system),
            HelpCommand(self.voice_system, self),
            ExitCommand(self.voice_system)
        ]
    
    def process_command(self, user_input: str) -> str:
        """Enhanced Command Processing"""
        
        if not user_input or not user_input.strip():
            return "Entschuldigung, ich habe nichts verstanden."
        
        try:
            logger.info(f"🧠 Enhanced processing: '{user_input}'")
            
            best_match = None
            best_confidence = 0.0
            
            for command in self.commands:
                # Try enhanced matching first
                if hasattr(command, 'enhanced_match'):
                    match = command.enhanced_match(user_input)
                    if match.matched and match.confidence > best_confidence:
                        best_match = (command, match)
                        best_confidence = match.confidence
                else:
                    # Fallback to basic matching
                    match = command.match(user_input)
                    if match.matched and match.confidence > best_confidence:
                        enhanced_match = EnhancedMatch(
                            matched=True,
                            confidence=match.confidence,
                            method="basic_pattern",
                            extracted_params=getattr(match, 'extracted_params', {})
                        )
                        best_match = (command, enhanced_match)
                        best_confidence = match.confidence
            
            if best_match:
                command, match = best_match
                
                logger.info(f"✅ Enhanced match: {command.name} "
                           f"(confidence: {match.confidence:.2f})")
                
                response = command.execute(user_input, match.extracted_params)
                
                # Handle actions
                if hasattr(response, 'action') and response.action == "exit" and self.voice_system:
                    if hasattr(self.voice_system, 'stop_listening'):
                        self.voice_system.stop_listening()
                
                return response.message
            
            else:
                return "Das verstehe ich nicht ganz. Versuche: 'Hallo', 'Status', 'Hilfe' oder 'Zeit'."
                
        except Exception as e:
            logger.error(f"❌ Enhanced command processing failed: {e}")
            return f"Entschuldigung, beim Verarbeiten ist ein Fehler aufgetreten: {str(e)}"

# ✅ FIX: Complete exports with all needed classes
__all__ = [
    # Basic Commands (for compatibility)
    'GreetingCommand',
    'TimeCommand',
    'VoiceSettingsCommand',
    'HelpCommand',
    'ExitCommand',
    
    # Enhanced Commands
    'EnhancedGreetingCommand',
    'EnhancedStatusCommand', 
    'EnhancedCommandProcessor',
    'EnhancedMatch',

    'StatusCommand',
    'WeatherCommand',
    
    # Support classes
    'CommandCategory',
    'CommandResponse'
]