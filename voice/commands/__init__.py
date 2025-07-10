"""
Commands Module für Kira Voice System
Enthält alle Voice Commands und Command Processing mit Enhanced Features
"""

import logging

logger = logging.getLogger(__name__)

# Import Base Command System
try:
    from .base_command import (
        BaseCommand, 
        CommandCategory, 
        CommandResponse, 
        CommandMatch
    )
    logger.info("✅ Base Command System geladen")
    BASE_COMMANDS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Base Command System Import Fehler: {e}")
    BASE_COMMANDS_AVAILABLE = False

# ✅ UPDATED: Proper Enhanced Commands Import with Fallback
ENHANCED_COMMANDS_AVAILABLE = False
SIMPLE_COMMANDS_AVAILABLE = False

try:
    # ✅ FIRST: Try to import Enhanced Commands
    from .simple_commands import (
        EnhancedGreetingCommand,
        EnhancedStatusCommand,
        EnhancedCommandProcessor,
        EnhancedMatch,
    )
    logger.info("✅ Enhanced Commands (Phase 1) geladen")
    ENHANCED_COMMANDS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Enhanced Commands nicht verfügbar: {e}")

try:
    # ✅ SECOND: Try to import Standard Commands
    from .simple_commands import (
        TimeCommand,
        VoiceSettingsCommand,
        HelpCommand,
        ExitCommand,
    )
    logger.info("✅ Standard Commands geladen")
except ImportError as e:
    logger.warning(f"⚠️ Standard Commands nicht verfügbar: {e}")

try:
    # ✅ THIRD: Try to import Legacy Commands (FALLBACK)
    from .simple_commands import (
        GreetingCommand,
        StatusCommand,
        SimpleCommandProcessor
    )
    logger.info("✅ Legacy Commands geladen")
    SIMPLE_COMMANDS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Alle Commands fehlgeschlagen: {e}")

# ✅ COMMAND FACTORY FUNCTION
def create_command_processor(voice_system=None, enhanced=True):
    """
    Factory Function für Command Processor mit Enhanced Fallback
    """
    try:
        if enhanced and ENHANCED_COMMANDS_AVAILABLE:
            logger.info("🚀 Creating Enhanced Command Processor")
            return EnhancedCommandProcessor(voice_system)
        elif SIMPLE_COMMANDS_AVAILABLE:
            logger.info("⚠️ Creating Simple Command Processor (Fallback)")
            return SimpleCommandProcessor(voice_system)
        else:
            # ✅ EMERGENCY FALLBACK - Create minimal processor
            logger.error("❌ No command processors available - creating minimal fallback")
            return MinimalCommandProcessor(voice_system)
    except Exception as e:
        logger.error(f"❌ Command Processor creation failed: {e}")
        # ✅ LAST RESORT FALLBACK
        return MinimalCommandProcessor(voice_system)

# ✅ MINIMAL FALLBACK COMMAND PROCESSOR
class MinimalCommandProcessor:
    """
    Minimal Command Processor als Fallback wenn nichts anderes funktioniert
    """
    def __init__(self, voice_system=None):
        self.voice_system = voice_system
        logger.warning("⚠️ Using Minimal Command Processor - Limited functionality")
    
    def process_command(self, user_input: str) -> str:
        """Minimal command processing"""
        user_lower = user_input.lower()
        
        # Basic hardcoded responses
        if any(word in user_lower for word in ['hallo', 'hi', 'hey']):
            return "Hallo! Schön dich zu hören."
        elif any(word in user_lower for word in ['status', 'wie geht']):
            return "Mir geht es gut, danke der Nachfrage!"
        elif any(word in user_lower for word in ['zeit', 'uhr', 'spät']):
            import datetime
            now = datetime.datetime.now()
            return f"Es ist {now.strftime('%H:%M')} Uhr."
        elif any(word in user_lower for word in ['hilfe', 'help']):
            return "Ich kann auf Begrüßungen antworten, die Zeit sagen und meinen Status mitteilen."
        elif any(word in user_lower for word in ['tschüss', 'bye', 'wiedersehen']):
            return "Auf Wiedersehen! Bis bald!"
        else:
            return f"Entschuldigung, '{user_input}' verstehe ich nicht. Versuche: Hallo, Status, Wie spät ist es, oder Hilfe."
    
    def get_all_commands(self):
        """Returns empty list for compatibility"""
        return []
    
    def get_processor_stats(self):
        """Returns minimal stats"""
        return {
            'total_commands': 5,
            'total_executions': 0,
            'processor_type': 'minimal_fallback'
        }

# ✅ ENHANCED TEST FUNCTION
def test_commands_system():
    """Testet das Enhanced Commands System"""
    print("🎯 === COMMANDS SYSTEM TEST ===")
    
    if not BASE_COMMANDS_AVAILABLE:
        print("❌ Base Commands System nicht verfügbar")
        return False
    
    # ✅ TEST COMMAND PROCESSOR CREATION
    try:
        processor = create_command_processor(enhanced=True)
        print(f"✅ Command Processor erstellt: {type(processor).__name__}")
    except Exception as e:
        print(f"❌ Command Processor Fehler: {e}")
        return False
    
    # ✅ BASIC TEST INPUTS
    test_inputs = [
        "Hallo Kira",
        "Status",
        "Wie spät ist es?",
        "Hilfe",
        "Tschüss",
        "Unbekannter Befehl"
    ]
    
    print(f"Teste {len(test_inputs)} Commands...")
    print("=" * 50)
    
    success_count = 0
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n🎯 Test {i}: '{test_input}'")
        
        try:
            response = processor.process_command(test_input)
            
            if isinstance(response, str) and response:
                print(f"✅ Antwort: {response}")
                success_count += 1
            else:
                print(f"⚠️ Unerwartete Antwort: {type(response)}")
            
        except Exception as e:
            print(f"❌ Fehler: {e}")
    
    # ✅ TEST RESULTS
    print(f"\n🎯 TEST ERGEBNISSE:")
    print(f"   ✅ Erfolgreiche Tests: {success_count}/{len(test_inputs)}")
    print(f"   📊 Erfolgsrate: {(success_count/len(test_inputs)*100):.1f}%")
    
    if ENHANCED_COMMANDS_AVAILABLE:
        print(f"   🚀 Enhanced Commands: VERFÜGBAR")
    elif SIMPLE_COMMANDS_AVAILABLE:
        print(f"   ⚠️ Simple Commands: VERFÜGBAR")
    else:
        print(f"   🆘 Minimal Fallback: AKTIV")
    
    print(f"\n🎉 Commands System Test abgeschlossen!")
    return success_count > len(test_inputs) * 0.7  # 70% Erfolgsrate erforderlich

# ✅ UPDATED EXPORT
__all__ = [
    # Base System
    'BaseCommand',
    'CommandCategory', 
    'CommandResponse',
    'CommandMatch',
    
    # Enhanced Commands (if available)
    'EnhancedGreetingCommand',
    'EnhancedStatusCommand',
    'EnhancedCommandProcessor', 
    'EnhancedMatch',
    
    # Standard Commands (if available)
    'TimeCommand',
    'VoiceSettingsCommand',
    'HelpCommand',
    'ExitCommand',
    
    # Legacy Commands (if available)
    'GreetingCommand',
    'StatusCommand',
    'SimpleCommandProcessor',
    
    # Fallback
    'MinimalCommandProcessor',
    
    # Utilities
    'test_commands_system',
    'create_command_processor',
    
    # Availability Flags
    'BASE_COMMANDS_AVAILABLE',
    'SIMPLE_COMMANDS_AVAILABLE',
    'ENHANCED_COMMANDS_AVAILABLE'
]

# ✅ ENHANCED LOGGING
logger.info("📦 Kira Commands Module geladen")
if ENHANCED_COMMANDS_AVAILABLE:
    logger.info("🚀 Enhanced Commands verfügbar")
elif SIMPLE_COMMANDS_AVAILABLE:
    logger.info("⚠️ Simple Commands verfügbar")
else:
    logger.warning("🆘 Nur Minimal Fallback verfügbar")