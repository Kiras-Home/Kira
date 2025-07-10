"""
Synthesis Module für Kira Voice System
Enthält TTS Engines: Bark (primär) und System TTS (Fallback)
"""

import logging

logger = logging.getLogger(__name__)

# Import TTS Engines
try:
    from .bark_engine import BarkTTSEngine
    logger.info("✅ Bark TTS Engine geladen")
    BARK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Bark TTS Engine nicht verfügbar: {e}")
    BarkTTSEngine = None
    BARK_AVAILABLE = False

try:
    from .system_tts import SystemTTSEngine
    logger.info("✅ System TTS Engine geladen")
    SYSTEM_TTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ System TTS Engine Import Fehler: {e}")
    SystemTTSEngine = None
    SYSTEM_TTS_AVAILABLE = False

# Smart TTS Engine Factory
class SmartTTSEngine:
    """Intelligente TTS Engine - wählt beste verfügbare Option"""
    
    def __init__(self, prefer_bark: bool = True, output_dir: str = "voice/output"):
        self.prefer_bark = prefer_bark
        self.output_dir = output_dir
        
        self.primary_engine = None
        self.fallback_engine = None
        self.active_engine = None
        
        logger.info(f"🧠 Smart TTS Engine (Bark bevorzugt: {prefer_bark})")
    
    def initialize(self) -> bool:
        """Initialisiert verfügbare TTS Engines"""
        try:
            success = False
            
            # Versuche Bark TTS (wenn bevorzugt und verfügbar)
            if self.prefer_bark and BARK_AVAILABLE:
                logger.info("🌳 Initialisiere Bark TTS...")
                self.primary_engine = BarkTTSEngine(output_dir=self.output_dir)
                
                if self.primary_engine.initialize():
                    logger.info("✅ Bark TTS als primäre Engine geladen")
                    self.active_engine = self.primary_engine
                    success = True
                else:
                    logger.warning("⚠️ Bark TTS Initialisierung fehlgeschlagen")
                    self.primary_engine = None
            
            # System TTS als Fallback
            if SYSTEM_TTS_AVAILABLE:
                logger.info("🖥️ Initialisiere System TTS...")
                self.fallback_engine = SystemTTSEngine(output_dir=self.output_dir)
                
                if self.fallback_engine.initialize():
                    logger.info("✅ System TTS als Fallback geladen")
                    
                    # Verwende System TTS wenn Bark nicht verfügbar
                    if not self.active_engine:
                        self.active_engine = self.fallback_engine
                        success = True
                else:
                    logger.warning("⚠️ System TTS Initialisierung fehlgeschlagen")
                    self.fallback_engine = None
            
            if success:
                engine_name = "Bark TTS" if self.active_engine == self.primary_engine else "System TTS"
                logger.info(f"🎉 Smart TTS Engine bereit mit: {engine_name}")
                return True
            else:
                logger.error("❌ Keine TTS Engine verfügbar")
                return False
                
        except Exception as e:
            logger.error(f"❌ Smart TTS Engine Initialisierung fehlgeschlagen: {e}")
            return False
    
    def speak(self, text: str, emotion: str = "neutral", auto_play: bool = True) -> bool:
        """Spricht Text mit bester verfügbarer Engine"""
        
        if not self.active_engine:
            logger.error("❌ Keine TTS Engine verfügbar")
            return False
        
        try:
            # Versuche aktive Engine
            success = self.active_engine.speak(text, emotion, auto_play)
            
            if success:
                return True
            
            # Fallback wenn primäre Engine fehlschlägt
            if self.active_engine == self.primary_engine and self.fallback_engine:
                logger.warning("⚠️ Primäre Engine fehlgeschlagen - verwende Fallback")
                return self.fallback_engine.speak(text, emotion, auto_play)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Smart TTS Speak Fehler: {e}")
            return False
    
    def get_engine_info(self) -> dict:
        """Gibt Engine-Informationen zurück"""
        return {
            'bark_available': BARK_AVAILABLE,
            'system_tts_available': SYSTEM_TTS_AVAILABLE,
            'primary_engine': type(self.primary_engine).__name__ if self.primary_engine else None,
            'fallback_engine': type(self.fallback_engine).__name__ if self.fallback_engine else None,
            'active_engine': type(self.active_engine).__name__ if self.active_engine else None,
            'prefer_bark': self.prefer_bark
        }
    
    def switch_to_fallback(self):
        """Wechselt zur Fallback Engine"""
        if self.fallback_engine:
            self.active_engine = self.fallback_engine
            logger.info("🔄 Gewechselt zu Fallback Engine")
        else:
            logger.warning("⚠️ Keine Fallback Engine verfügbar")
    
    def switch_to_primary(self):
        """Wechselt zur primären Engine"""
        if self.primary_engine:
            self.active_engine = self.primary_engine
            logger.info("🔄 Gewechselt zu primärer Engine")
        else:
            logger.warning("⚠️ Keine primäre Engine verfügbar")
    
    def cleanup(self):
        """Cleanup aller Engines"""
        try:
            if self.primary_engine:
                self.primary_engine.cleanup()
            if self.fallback_engine:
                self.fallback_engine.cleanup()
            
            self.active_engine = None
            logger.info("🧹 Smart TTS Engine Cleanup abgeschlossen")
            
        except Exception as e:
            logger.error(f"❌ Smart TTS Cleanup Fehler: {e}")

# Test-Funktion für komplettes Synthesis System
def test_synthesis_system():
    """Testet das komplette Synthesis System"""
    logger.info("🧪 Teste Synthesis System...")
    
    print("🎤 === SYNTHESIS SYSTEM TEST ===")
    
    # Test Smart TTS Engine
    smart_tts = SmartTTSEngine(prefer_bark=True)
    
    if not smart_tts.initialize():
        print("❌ Smart TTS Initialisierung fehlgeschlagen")
        return False
    
    # Engine Info
    info = smart_tts.get_engine_info()
    print(f"Bark verfügbar: {info['bark_available']}")
    print(f"System TTS verfügbar: {info['system_tts_available']}")
    print(f"Aktive Engine: {info['active_engine']}")
    
    # Test verschiedene Texte
    test_cases = [
        ("Hallo, ich bin Kira!", "neutral"),
        ("Das ist fantastisch!", "excited"),
        ("Ich verstehe dich sehr gut.", "empathetic")
    ]
    
    for text, emotion in test_cases:
        print(f"\n🗣️ Test: '{text}' ({emotion})")
        
        if smart_tts.speak(text, emotion, auto_play=True):
            print("✅ Erfolgreich")
        else:
            print("❌ Fehlgeschlagen")
        
        # Kurze Pause
        import time
        time.sleep(2)
    
    # Test Engine Switch (falls beide verfügbar)
    if info['bark_available'] and info['system_tts_available']:
        print(f"\n🔄 Teste Engine-Wechsel...")
        
        print("Wechsel zu System TTS...")
        smart_tts.switch_to_fallback()
        smart_tts.speak("Jetzt verwende ich System TTS.", auto_play=True)
        
        import time
        time.sleep(2)
        
        print("Wechsel zurück zu Bark...")
        smart_tts.switch_to_primary()
        smart_tts.speak("Und jetzt bin ich wieder Bark TTS.", auto_play=True)
    
    smart_tts.cleanup()
    
    print("\n🎉 Synthesis System Test abgeschlossen!")
    return True

# Export
__all__ = [
    'BarkTTSEngine',
    'SystemTTSEngine', 
    'SmartTTSEngine',
    'BARK_AVAILABLE',
    'SYSTEM_TTS_AVAILABLE',
    'test_synthesis_system'
]

# Log beim Import
logger.info("📦 Kira Synthesis Module geladen")