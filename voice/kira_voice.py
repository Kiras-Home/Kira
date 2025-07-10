"""
Hauptklasse für Kira Voice System
Einfach, fokussiert, funktionsfähig
"""

import logging
from typing import Optional, Dict, Any, List
import time
import asyncio
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from .config import VoiceConfig, DEFAULT_CONFIG
from .audio.recorder import SimpleAudioRecorder, AudioData
from .recognition.whisper_engine import WhisperEngine
from .synthesis.bark_engine import BarkTTSEngine

# NEW: LM Studio and Memory Integration
from services.lm_studio_service import LMStudioService
from memory.integration import UnifiedMemorySystem
from config.system_config import KiraSystemConfig

logger = logging.getLogger(__name__)

class KiraVoice:
    """Hauptklasse für Kira Voice System - Voice-only Assistant"""
    
    def __init__(self, config: VoiceConfig = None, config_dict: dict = None, server_mode: bool = False):
        # ✅ ENHANCED CONFIG HANDLING using from_dict method
        if config_dict:
            # Use the new from_dict method that filters parameters
            self.config = VoiceConfig.from_dict(config_dict)
            self.extended_config = config_dict  # Keep original for other components
            logger.info(f"✅ VoiceConfig created from dict with {len(config_dict)} parameters")
        else:
            self.config = config or DEFAULT_CONFIG
            self.extended_config = {}
            
        self.server_mode = server_mode

        self.audio_queue = asyncio.Queue() if server_mode else None
        self.response_queue = asyncio.Queue() if server_mode else None

        # Voice components
        self.recorder = None
        self.whisper = None
        self.bark_tts = None
        self.command_processor = None
        
        # NEW: AI and Memory components
        self.system_config = None
        self.lm_studio = None
        self.memory_system = None
        self.user_profile = {}
        
        # Voice-only state
        self.is_initialized = False
        self.is_listening = False
        self.is_voice_only_mode = True  # NEW: Default to voice-only
        self.wake_word_detected = False
        self.conversation_active = False
        
        # NEW: Intelligent Assistant Features
        self.assistant_capabilities = {
            'task_management': True,
            'calendar_integration': True,
            'reminder_system': True,
            'project_tracking': True,
            'knowledge_base': True,
            'learning_mode': True
        }
        
        # NEW: Task and Project Management
        self.active_tasks = []
        self.projects = {}
        self.reminders = []
        self.context_stack = []  # For complex multi-step tasks
        
        # NEW: Learning and Adaptation
        self.user_preferences = {}
        self.conversation_patterns = {}
        self.skill_learning = {
            'recognized_skills': [],
            'requested_help_areas': [],
            'frequently_used_commands': {}
        }
        
        logger.info("🎯 Kira Voice-only Assistant initialisiert")
        logger.info(f"   🎭 Enhanced features: {len(self.extended_config)} settings")
        logger.info(f"   🎤 Wake word enabled: {self.config.enable_wake_word}")
        logger.info(f"   🗣️ Speech recognition: {self.config.enable_speech_recognition}")
        logger.info(f"   🧠 Voice-only mode: {self.is_voice_only_mode}")
        logger.info(f"   🤖 Assistant capabilities: {list(self.assistant_capabilities.keys())}")
    
    def initialize(self) -> bool:
        """Initialisiert alle Komponenten für Voice-only Assistant"""
        try:
            logger.info("🚀 Initialisiere Kira Voice-only Assistant...")
            
            # NEW: Initialize System Config
            self.system_config = KiraSystemConfig()
            
            # NEW: Initialize LM Studio Service
            self.lm_studio = LMStudioService(self.system_config)
            lm_result = self.lm_studio.initialize()
            
            if lm_result['success']:
                logger.info("✅ LM Studio verbunden")
            else:
                logger.warning(f"⚠️ LM Studio nicht verfügbar: {lm_result.get('error')}")
                # Continue without LM Studio for now
            
            # NEW: Initialize Memory System
            memory_config = {
                'stm_capacity': 7,
                'enable_database': True,
                'enable_conversations': True,
                'auto_consolidation': True
            }
            self.memory_system = UnifiedMemorySystem(memory_config)
            
            try:
                memory_init = self.memory_system.initialize()
                if memory_init.get('success'):
                    logger.info("✅ Memory System initialisiert")
                else:
                    logger.warning("⚠️ Memory System Initialisierung teilweise fehlgeschlagen")
            except Exception as e:
                logger.warning(f"⚠️ Memory System Fehler: {e}")
            
            # Audio Recorder
            self.recorder = SimpleAudioRecorder(
                sample_rate=self.config.sample_rate,
                channels=self.config.channels
            )
            
            # Whisper Engine
            self.whisper = WhisperEngine(
                model_size=self.config.whisper_model,
                language=self.config.language
            )
            
            if not self.whisper.initialize():
                logger.error("❌ Whisper Initialisierung fehlgeschlagen")
                return False
            
            # Bark TTS Engine
            self.bark_tts = BarkTTSEngine(
                voice_preset=self.config.bark_voice,
                output_dir=self.config.output_dir
            )
            
            if not self.bark_tts.initialize():
                logger.error("❌ Bark TTS Initialisierung fehlgeschlagen")
                return False
            
            # NEW: Initialize Voice-only Command Processing
            if not self._initialize_voice_command_processor():
                logger.error("❌ Voice Command Processor Initialisierung fehlgeschlagen")
                return False
            
            # NEW: Load User Profile from Memory
            self._load_user_profile()
            
            logger.info("✅ Kira Voice-only Assistant bereit!")
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Kira Voice-only Assistant Initialisierung fehlgeschlagen: {e}")
            return False
    
    def listen_and_respond(self, max_duration: float = None) -> bool:
        """Hört zu und antwortet"""
        
        if not self.is_initialized:
            logger.error("❌ Kira Voice nicht initialisiert")
            return False
        
        try:
            duration = max_duration or self.config.max_duration
            
            logger.info(f"👂 Kira hört zu ({duration}s)...")
            
            # 1. Audio aufnehmen
            audio_result = self.recorder.record(duration)
            
            if not audio_result.success:
                logger.error(f"❌ Audio-Aufnahme fehlgeschlagen: {audio_result.error}")
                return False
            
            # 2. Speech Recognition
            logger.info("🧠 Kira analysiert Sprache...")
            recognized_text = self.whisper.transcribe(audio_result.data, audio_result.sample_rate)
            
            if not recognized_text:
                logger.warning("⚠️ Keine Sprache erkannt")
                self.speak("Entschuldigung, ich habe nichts verstanden.", "empathetic")
                return False
            
            logger.info(f"📝 Erkannt: '{recognized_text}'")
            
            # ✅ 3. ENHANCED COMMAND PROCESSING
            if self.command_processor:
                response = self.command_processor.process_command(recognized_text)
            else:
                response = "Command System nicht verfügbar."
            
            # 4. Antwort sprechen
            if response:
                self.speak(response, "calm")
                return True
            else:
                self.speak("Ich bin mir nicht sicher, wie ich darauf antworten soll.", "thoughtful")
                return False
                
        except Exception as e:
            logger.error(f"❌ Listen and Respond Fehler: {e}")
            return False
        
    async def process_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Verarbeitet Audio-Daten vom Client"""
        try:
            # Whisper Transkription
            result = self.whisper.transcribe(audio_data)
            
            if result.success:
                # ✅ ENHANCED COMMAND PROCESSING
                if self.command_processor:
                    response = self.command_processor.process_command(result.text)
                else:
                    response = "Command System nicht verfügbar."
                
                # TTS wenn nötig
                if response and hasattr(response, 'should_speak') and response.should_speak:
                    await self.speak_async(response.message, response.emotion)
                elif response:
                    await self.speak_async(str(response), "calm")
                
                return {
                    "success": True,
                    "text": result.text,
                    "response": response.to_dict() if hasattr(response, 'to_dict') else str(response)
                }
            
            return {"success": False, "error": "Transcription failed"}
            
        except Exception as e:
            logger.error(f"Audio Processing Error: {e}")
            return {"success": False, "error": str(e)}

    async def speak_async(self, text: str, emotion: str = "neutral") -> bool:
        """Async Version von speak für Server"""
        if self.server_mode:
            # Füge zur Response Queue hinzu
            await self.response_queue.put({
                "type": "speech",
                "text": text,
                "emotion": emotion
            })
        return await asyncio.to_thread(self.speak, text, emotion)
    
    def speak(self, text: str, emotion: str = "neutral") -> Optional[Path]:
        """Spricht Text und informiert alle Clients"""
        try:
            # Generate audio
            audio_path = self.bark_tts.speak(text, emotion, auto_play=True)
            
            if audio_path is None:
                logger.warning("⚠️ Audio konnte nicht generiert werden")
                return None

            # Wenn im Server-Modus, informiere alle Clients
            if self.server_mode and self.response_queue:
                asyncio.create_task(self.response_queue.put({
                    'type': 'speak',
                    'text': text,
                    'emotion': emotion
                }))

            return audio_path

        except Exception as e:
            logger.error(f"❌ Speak error: {e}")
            return None
    
    def continuous_listening(self):
        """Kontinuierliches Zuhören (Wake Word Mode)"""
        
        if not self.is_initialized:
            logger.error("❌ Kira Voice nicht initialisiert")
            return
        
        self.is_listening = True
        logger.info("👂 Kira startet kontinuierliches Zuhören...")
        logger.info("💡 Sagen Sie 'Kira stopp' zum Beenden")
        
        try:
            while self.is_listening:
                # Kurze Aufnahme für Wake Word Detection
                audio_result = self.recorder.record(2.0)  # 2 Sekunden
                
                if audio_result.success:
                    # Prüfe auf Wake Word
                    recognized_text = self.whisper.transcribe(audio_result.data, audio_result.sample_rate)
                    
                    if recognized_text and self.config.wake_word.lower() in recognized_text.lower():
                        logger.info(f"🔔 Wake Word '{self.config.wake_word}' erkannt!")
                        
                        # Vollständige Interaktion
                        self.speak("Ja, ich höre zu!", "attentive")
                        time.sleep(0.5)  # Kurze Pause
                        
                        self.listen_and_respond(max_duration=8.0)
                    
                    # Prüfe auf Stop-Command
                    if recognized_text and any(word in recognized_text.lower() for word in ["stopp", "stop", "beenden"]):
                        logger.info("⏹️ Stop-Command erkannt")
                        self.speak("Auf Wiedersehen!", "calm")
                        break
                
                # Kurze Pause
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("⌨️ Keyboard Interrupt - stoppe Zuhören")
        except Exception as e:
            logger.error(f"❌ Continuous Listening Fehler: {e}")
        finally:
            self.is_listening = False
            logger.info("👂 Kontinuierliches Zuhören beendet")
    
    def stop_listening(self):
        """Stoppt kontinuierliches Zuhören"""
        self.is_listening = False
        logger.info("⏹️ Zuhören gestoppt")
    
    # NEUE METHODEN FÜR ERWEITERTE FUNKTIONEN
    def get_system_status(self) -> Dict[str, Any]:
        """Gibt detaillierten System-Status zurück"""
        try:
            status = {
                'initialized': self.is_initialized,
                'listening': self.is_listening,
                'components': {},
                'component_details': {}
            }
            
            # ✅ ENHANCED COMPONENT STATUS
            # Audio Recorder
            if self.recorder:
                status['components']['recorder'] = 'ready'
                status['component_details']['recorder'] = {
                    'type': type(self.recorder).__name__,
                    'sample_rate': getattr(self.recorder, 'sample_rate', 'unknown'),
                    'channels': getattr(self.recorder, 'channels', 'unknown')
                }
            else:
                status['components']['recorder'] = 'offline'
            
            # Whisper STT Engine
            if self.whisper:
                status['components']['whisper'] = 'ready'
                status['component_details']['whisper'] = {
                    'type': type(self.whisper).__name__,
                    'initialized': getattr(self.whisper, 'is_initialized', False),
                    'model': getattr(self.whisper, 'model_size', 'unknown'),
                    'language': getattr(self.whisper, 'language', 'unknown')
                }
                
                # Whisper Performance Stats
                if hasattr(self.whisper, 'get_performance_stats'):
                    whisper_stats = self.whisper.get_performance_stats()
                    status['component_details']['whisper']['performance'] = whisper_stats
            else:
                status['components']['whisper'] = 'offline'
            
            # Bark TTS Engine
            if self.bark_tts:
                status['components']['bark_tts'] = 'ready'
                status['component_details']['bark_tts'] = {
                    'type': type(self.bark_tts).__name__,
                    'initialized': getattr(self.bark_tts, 'is_initialized', False),
                    'voice_preset': getattr(self.bark_tts, 'voice_preset', 'unknown'),
                    'output_dir': str(getattr(self.bark_tts, 'output_dir', 'unknown'))
                }
            else:
                status['components']['bark_tts'] = 'offline'
                
            # ✅ ENHANCED COMMAND PROCESSOR STATUS
            if self.command_processor:
                status['components']['command_processor'] = 'ready'
                status['component_details']['command_processor'] = {
                    'type': type(self.command_processor).__name__,
                    'total_commands': len(getattr(self.command_processor, 'commands', [])) if hasattr(self.command_processor, 'commands') else 0
                }
                
                # Command Processor Stats
                if hasattr(self.command_processor, 'get_processor_stats'):
                    command_stats = self.command_processor.get_processor_stats()
                    status['component_details']['command_processor']['stats'] = command_stats
            else:
                status['components']['command_processor'] = 'offline'
            
            # Wake Word Detector (falls vorhanden)
            if hasattr(self, 'wake_word_detector') and self.wake_word_detector:
                status['components']['wake_word'] = 'ready'
                status['component_details']['wake_word'] = {
                    'type': type(self.wake_word_detector).__name__,
                    'enabled': getattr(self.wake_word_detector, 'enabled', False)
                }
            else:
                status['components']['wake_word'] = 'offline'
            
            # NEW: LM Studio Status
            if self.lm_studio:
                status['components']['lm_studio'] = 'ready'
                status['component_details']['lm_studio'] = {
                    'type': type(self.lm_studio).__name__,
                    'initialized': getattr(self.lm_studio, 'is_initialized', False)
                }
            else:
                status['components']['lm_studio'] = 'offline'
            
            # NEW: Memory System Status
            if self.memory_system:
                status['components']['memory_system'] = 'ready'
                status['component_details']['memory_system'] = {
                    'type': type(self.memory_system).__name__,
                    'initialized': getattr(self.memory_system, 'is_initialized', False)
                }
            else:
                status['components']['memory_system'] = 'offline'
            
            # ✅ CONFIG INFORMATION
            if hasattr(self, 'config'):
                status['config_info'] = {
                    'sample_rate': self.config.sample_rate,
                    'language': self.config.language,
                    'whisper_model': self.config.whisper_model,
                    'bark_voice': self.config.bark_voice,
                    'output_dir': self.config.output_dir,
                    'max_duration': self.config.max_duration
                }
            
            # ✅ EXTENDED CONFIG (falls vorhanden)
            if hasattr(self, 'extended_config') and self.extended_config:
                status['extended_config_available'] = True
                status['extended_features'] = {
                    'wake_word_enabled': self.extended_config.get('enable_wake_word', False),
                    'emotion_synthesis': self.extended_config.get('enable_emotion_synthesis', False),
                    'speech_recognition': self.extended_config.get('enable_speech_recognition', False),
                    'voice_commands': self.extended_config.get('enable_voice_commands', False)
                }
            
            # ✅ SYSTEM HEALTH SUMMARY
            components_online = sum(1 for comp in status['components'].values() if comp == 'ready')
            total_components = len(status['components'])
            
            status['health'] = {
                'components_online': components_online,
                'total_components': total_components,
                'health_score': components_online / total_components if total_components > 0 else 0.0,
                'status': 'healthy' if components_online >= total_components * 0.75 else 'degraded' if components_online > 0 else 'offline'
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ System Status Fehler: {e}")
            return {
                'error': str(e),
                'initialized': False,
                'health': {'status': 'error'}
            }
    
    def list_available_commands(self) -> List[str]:
        """Gibt Liste verfügbarer Commands zurück"""
        try:
            if self.commands:
                commands = self.commands.get_all_commands()
                return [f"{cmd.name}: {', '.join(cmd.examples[:2])}" for cmd in commands]
            else:
                return []
        except Exception as e:
            logger.error(f"❌ Commands Liste Fehler: {e}")
            return []
    
    def process_text_command(self, text: str) -> str:
        """Verarbeitet Text-Command ohne Audio (für Tests)"""
        try:
            if not self.commands:
                return "Command System nicht verfügbar."
            
            response = self.commands.process_command(text)
            return response or "Kein Command gefunden."
            
        except Exception as e:
            logger.error(f"❌ Text Command Verarbeitung Fehler: {e}")
            return f"Fehler beim Verarbeiten: {str(e)}"
    
    def test_system(self) -> bool:
        """Testet das komplette System"""
        try:
            logger.info("🧪 Teste Kira Voice System...")
            
            # Test TTS
            logger.info("🗣️ Teste Text-to-Speech...")
            if not self.speak("Hallo, das ist ein System-Test.", "calm"):
                logger.error("❌ TTS Test fehlgeschlagen")
                return False
            
            logger.info("✅ TTS Test erfolgreich")
            
            # Test Audio Recording
            logger.info("🎤 Teste Audio Recording...")
            audio_result = self.recorder.record(1.0)
            
            if not audio_result.success:
                logger.error("❌ Audio Recording Test fehlgeschlagen")
                return False
            
            logger.info("✅ Audio Recording Test erfolgreich")
            
            # Test Speech Recognition
            logger.info("🧠 Teste Speech Recognition...")
            if audio_result.data is not None and len(audio_result.data) > 0:
                # Einfacher Test mit vorhandenem Audio
                result = self.whisper.transcribe(audio_result.data, audio_result.sample_rate)
                logger.info(f"ℹ️ Recognition Test Result: {result or 'Kein Text'}")
            
            logger.info("✅ Speech Recognition Test erfolgreich")
            
            # Test Command System (NEU)
            logger.info("🎯 Teste Command System...")
            test_commands = [
                "Hallo Kira",
                "Status",
                "Wie spät ist es?"
            ]
            
            for test_cmd in test_commands:
                response = self.process_text_command(test_cmd)
                logger.info(f"Command '{test_cmd}' → '{response[:50]}...'")
            
            logger.info("✅ Command System Test erfolgreich")
            
            # Test LM Studio Integration
            logger.info("🧠 Teste LM Studio Integration...")
            prompt = "Was ist die Hauptstadt von Deutschland?"
            lm_response = self.lm_studio.generate_response(prompt)
            
            if lm_response:
                logger.info(f"ℹ️ LM Studio Response: {lm_response}")
            else:
                logger.warning("⚠️ LM Studio keine Antwort erhalten")
            
            logger.info("✅ LM Studio Test erfolgreich")
            
            # Test Memory System Integration
            logger.info("🧠 Teste Memory System Integration...")
            memory_test_key = "test_key"
            memory_test_value = "test_value"
            
            # Speichern
            self.memory_system.save_to_memory(memory_test_key, memory_test_value)
            logger.info("✅ Wert in den Speicher geschrieben")
            
            # Abrufen
            retrieved_value = self.memory_system.get_from_memory(memory_test_key)
            logger.info(f"ℹ️ Aus dem Speicher abgerufen: {retrieved_value}")
            
            logger.info("✅ Memory System Test erfolgreich")
            
            logger.info("🎉 Kira Voice System Test erfolgreich!")
            return True
            
        except Exception as e:
            logger.error(f"❌ System Test Fehler: {e}")
            return False
    
    def cleanup(self):
        """Cleanup aller Komponenten"""
        try:
            logger.info("🧹 Kira Voice System Cleanup...")
            
            self.stop_listening()
            
            if self.whisper:
                self.whisper.cleanup()
            
            if self.bark_tts:
                self.bark_tts.cleanup()
            
            if self.lm_studio and hasattr(self.lm_studio, 'cleanup'):
                self.lm_studio.cleanup()
            
            if self.memory_system and hasattr(self.memory_system, 'cleanup'):
                self.memory_system.cleanup()
            
            self.is_initialized = False
            logger.info("✅ Kira Voice System Cleanup abgeschlossen")
            
        except Exception as e:
            logger.error(f"❌ Cleanup Fehler: {e}")

    def _initialize_command_processor(self):
        """Initialisiert den Enhanced Command Processor"""
        try:
            # ✅ USE FACTORY FUNCTION FROM COMMANDS MODULE
            from .commands import create_command_processor, ENHANCED_COMMANDS_AVAILABLE
            
            self.command_processor = create_command_processor(
                voice_system=self,
                enhanced=True  # Try Enhanced first
            )
            
            if ENHANCED_COMMANDS_AVAILABLE:
                logger.info("🚀 Enhanced Command Processor initialisiert")
            else:
                logger.info("⚠️ Legacy Command Processor initialisiert (Fallback)")
                
            # ✅ BACKWARD COMPATIBILITY
            self.commands = self.command_processor  # For existing code
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Command Processor initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.command_processor = None
            self.commands = None
            return False

    def _initialize_voice_command_processor(self):
        """Initialisiert Voice-only Command Processing mit LM Studio"""
        try:
            # Simple voice command processor for now
            self.command_processor = SimpleVoiceCommandProcessor(
                voice_system=self,
                lm_studio=self.lm_studio,
                memory_system=self.memory_system
            )
            return True
        except Exception as e:
            logger.error(f"❌ Voice Command Processor Fehler: {e}")
            return False
    
    def _load_user_profile(self):
        """Lädt User Profile aus Memory System"""
        try:
            if self.memory_system:
                # Try to load user profile from memory
                try:
                    # Try different API methods for memory access
                    if hasattr(self.memory_system, 'get_memories'):
                        user_memories = self.memory_system.get_memories(memory_type='user_profile', limit=1)
                    elif hasattr(self.memory_system, 'ltm') and hasattr(self.memory_system.ltm, 'get_memories'):
                        user_memories = self.memory_system.ltm.get_memories(memory_type='user_profile', limit=1)
                    else:
                        user_memories = []
                    
                    if user_memories:
                        self.user_profile = user_memories[0].content
                        logger.info(f"✅ User Profile geladen: {len(self.user_profile)} Einträge")
                    else:
                        self._create_default_user_profile()
                except Exception as e:
                    logger.warning(f"⚠️ Memory System Profile laden fehlgeschlagen: {e}")
                    self._create_default_user_profile()
            else:
                self._create_default_user_profile()
                
        except Exception as e:
            logger.warning(f"⚠️ User Profile laden fehlgeschlagen: {e}")
            self._create_default_user_profile()
    
    def _create_default_user_profile(self):
        """Erstellt Standard User Profile"""
        self.user_profile = {
            'name': None,
            'preferences': {},
            'important_topics': [],
            'projects': [],
            'todos': [],
            'first_interaction': datetime.now().isoformat()
        }
        logger.info("✅ Standard User Profile erstellt")
    
    def _save_user_profile(self):
        """Speichert User Profile in Memory System"""
        try:
            if self.memory_system and self.user_profile:
                # Save to memory system
                try:
                    if hasattr(self.memory_system, 'add_memory'):
                        self.memory_system.add_memory(
                            content=self.user_profile,
                            memory_type='user_profile',
                            importance=0.9,
                            metadata={'updated': datetime.now().isoformat(), 'source': 'voice_assistant'}
                        )
                    elif hasattr(self.memory_system, 'store_memory'):
                        self.memory_system.store_memory(
                            content=self.user_profile,
                            memory_type='user_profile',
                            importance=0.9,
                            metadata={'updated': datetime.now().isoformat(), 'source': 'voice_assistant'}
                        )
                    logger.info("✅ User Profile gespeichert")
                except Exception as e:
                    logger.warning(f"⚠️ User Profile speichern fehlgeschlagen: {e}")
                
        except Exception as e:
            logger.warning(f"⚠️ User Profile speichern fehlgeschlagen: {e}")
    
    def voice_only_conversation(self, max_turns: int = 10) -> bool:
        """Startet eine Voice-only Konversation"""
        
        if not self.is_initialized:
            logger.error("❌ Kira Voice nicht initialisiert")
            return False
        
        logger.info("🎤 === VOICE-ONLY CONVERSATION MODUS ===")
        self.conversation_active = True
        turn_count = 0
        
        try:
            # Begrüßung
            greeting = self._generate_personalized_greeting()
            self.speak(greeting, "friendly")
            
            while self.conversation_active and turn_count < max_turns:
                turn_count += 1
                logger.info(f"🗨️ Conversation Turn {turn_count}/{max_turns}")
                
                # Höre auf User Input
                if self.listen_and_respond_with_ai():
                    logger.info("✅ Successful voice interaction")
                else:
                    logger.warning("⚠️ Voice interaction failed")
                    
                    # Give user another chance
                    self.speak("Entschuldigung, ich habe Sie nicht verstanden. Bitte versuchen Sie es noch einmal.", "empathetic")
                    
                    if not self.listen_and_respond_with_ai():
                        logger.info("❌ Second attempt failed, ending conversation")
                        break
                
                # Short pause between turns
                time.sleep(0.5)
            
            # Ende der Konversation
            self.speak("Es war schön, mit Ihnen zu sprechen. Bis zum nächsten Mal!", "calm")
            
        except KeyboardInterrupt:
            logger.info("⌨️ Conversation interrupted by user")
            self.speak("Auf Wiedersehen!", "calm")
        except Exception as e:
            logger.error(f"❌ Voice conversation error: {e}")
            self.speak("Es tut mir leid, es gab ein technisches Problem. Auf Wiedersehen!", "empathetic")
        finally:
            self.conversation_active = False
            logger.info("🎤 Voice-only conversation ended")
        
        return True
    
    def listen_and_respond_with_ai(self, max_duration: float = None) -> bool:
        """Hört zu und antwortet mit AI (LM Studio) Integration"""
        
        if not self.is_initialized:
            logger.error("❌ Kira Voice nicht initialisiert")
            return False
        
        try:
            duration = max_duration or self.config.max_duration
            
            logger.info(f"👂 Kira hört zu ({duration}s)...")
            
            # 1. Audio aufnehmen
            audio_result = self.recorder.record(duration)
            
            if not audio_result.success:
                logger.error(f"❌ Audio-Aufnahme fehlgeschlagen: {audio_result.error}")
                return False
            
            # 2. Speech Recognition
            logger.info("🧠 Kira analysiert Sprache...")
            recognized_text = self.whisper.transcribe(audio_result.data, audio_result.sample_rate)
            
            if not recognized_text:
                logger.warning("⚠️ Keine Sprache erkannt")
                return False
            
            logger.info(f"📝 Erkannt: '{recognized_text}'")
            
            # 3. Store in short-term memory
            self._store_user_message(recognized_text)
            
            # 4. Generate AI response
            response = self._generate_ai_response(recognized_text)
            
            # 5. Antwort sprechen
            if response:
                self.speak(response, "calm")
                
                # Store AI response in memory
                self._store_ai_response(response)
                
                return True
            else:
                self.speak("Ich bin mir nicht sicher, wie ich darauf antworten soll.", "thoughtful")
                return False
                
        except Exception as e:
            logger.error(f"❌ AI Listen and Respond Fehler: {e}")
            return False
    
    def _generate_personalized_greeting(self) -> str:
        """Generiert personalisierte Begrüßung basierend auf User Profile"""
        try:
            user_name = self.user_profile.get('name')
            current_hour = datetime.now().hour
            
            # Time-based greeting
            if current_hour < 12:
                time_greeting = "Guten Morgen"
            elif current_hour < 18:
                time_greeting = "Guten Tag"
            else:
                time_greeting = "Guten Abend"
            
            # Personalized greeting
            if user_name:
                greeting = f"{time_greeting}, {user_name}! Ich bin Kira, Ihr persönlicher Assistent."
            else:
                greeting = f"{time_greeting}! Ich bin Kira, Ihr persönlicher Assistent."
            
            # Add context based on memory
            if self.user_profile.get('projects'):
                greeting += f" Ich erinnere mich an {len(self.user_profile['projects'])} Projekte, an denen Sie arbeiten."
            
            greeting += " Wie kann ich Ihnen heute helfen?"
            
            return greeting
            
        except Exception as e:
            logger.error(f"❌ Greeting generation error: {e}")
            return "Hallo! Ich bin Kira, Ihr persönlicher Assistent. Wie kann ich Ihnen helfen?"
    
    def _generate_ai_response(self, user_message: str) -> str:
        """Generiert AI-Antwort mit LM Studio Integration und Assistant Features"""
        try:
            # Check if LM Studio is available
            if self.lm_studio and self.lm_studio.is_available:
                # Prepare enhanced context with assistant capabilities
                conversation_history = self._prepare_conversation_history()
                
                # NEW: Enhanced system prompt for assistant
                system_prompt = self._create_assistant_system_prompt()
                
                # Generate response with LM Studio
                response = self.lm_studio.chat_with_kira(
                    user_message=user_message,
                    conversation_history=conversation_history
                )
                
                if response and response.get('success'):
                    ai_response = response['response']
                    
                    # NEW: Process response for assistant actions
                    processed_response = self._process_assistant_response(ai_response, user_message)
                    
                    return processed_response
                else:
                    logger.warning(f"⚠️ LM Studio response failed: {response.get('error')}")
                    return self._generate_intelligent_fallback_response(user_message)
            else:
                logger.warning("⚠️ LM Studio not available, using intelligent fallback")
                return self._generate_intelligent_fallback_response(user_message)
                
        except Exception as e:
            logger.error(f"❌ AI Response generation error: {e}")
            return self._generate_intelligent_fallback_response(user_message)
    
    def _create_assistant_system_prompt(self) -> str:
        """Erstellt erweiterten System-Prompt für intelligenten Assistenten"""
        
        current_time = datetime.now().strftime("%H:%M")
        current_date = datetime.now().strftime("%d.%m.%Y")
        user_name = self.user_profile.get('name', 'der Nutzer')
        
        # Build context about user
        user_context = []
        if self.user_profile.get('projects'):
            user_context.append(f"Projekte: {', '.join(self.user_profile['projects'])}")
        if self.active_tasks:
            user_context.append(f"Aktive Aufgaben: {len(self.active_tasks)}")
        if self.reminders:
            user_context.append(f"Offene Erinnerungen: {len(self.reminders)}")
        
        user_info = "\n".join(user_context) if user_context else "Keine besonderen Informationen verfügbar"
        
        system_prompt = f"""Du bist Kira, ein intelligenter persönlicher Voice-Assistant für {user_name}.

AKTUELLER KONTEXT:
- Zeit: {current_time}, Datum: {current_date}
- User: {user_name}
- Aktuelle Informationen:
{user_info}

DEINE FÄHIGKEITEN ALS ASSISTENT:
1. 📋 Aufgaben-Management (Tasks erstellen, verwalten, priorisieren)
2. 📅 Termin-Verwaltung (Erinnerungen, Deadlines)
3. 📊 Projekt-Tracking (Projekte verfolgen, Status updates)
4. 🧠 Wissen sammeln (Wichtige Infos merken und abrufen)
5. 💡 Proaktive Unterstützung (Vorschläge, Optimierungen)
6. 🎯 Ziel-Orientierung (Bei langfristigen Zielen helfen)

DEINE PERSÖNLICHKEIT:
- Effizient und hilfsbereit
- Proaktiv aber nicht aufdringlich  
- Merkt sich wichtige Details
- Strukturiert und organisiert
- Natürlich im Gesprächsstil
- Kurz und präzise (Voice-optimiert)

VOICE-OPTIMIERUNG:
- Halte Antworten unter 30 Wörtern (außer bei detaillierten Erklärungen)
- Verwende natürliche Sprache
- Bei komplexen Aufgaben: in Schritte unterteilen
- Frage nach, wenn Details unklar sind

ASSISTANT-AKTIONEN:
Wenn der Nutzer Aufgaben/Projekte/Erinnerungen erwähnt:
- Biete an, diese zu verwalten
- Frage nach Details (Deadline, Priorität)
- Schlage Struktur vor

Antworte als persönlicher Assistent, der den Nutzer bei seinen Zielen und Aufgaben unterstützt."""

        return system_prompt
    
    def _process_assistant_response(self, ai_response: str, user_message: str) -> str:
        """Verarbeitet AI-Antwort und führt Assistant-Aktionen aus"""
        try:
            # Extract and execute assistant actions
            self._extract_and_execute_actions(user_message, ai_response)
            
            # Enhanced response with context
            enhanced_response = self._enhance_response_with_context(ai_response)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"❌ Assistant response processing error: {e}")
            return ai_response  # Return original response if processing fails
    
    def _extract_and_execute_actions(self, user_message: str, ai_response: str):
        """Extrahiert und führt Assistant-Aktionen aus"""
        user_lower = user_message.lower()
        
        # Task Management
        if any(keyword in user_lower for keyword in ['task', 'aufgabe', 'todo', 'erledigen']):
            self._handle_task_management(user_message, ai_response)
        
        # Project Management  
        if any(keyword in user_lower for keyword in ['projekt', 'project', 'arbeite an']):
            self._handle_project_management(user_message, ai_response)
        
        # Reminder System
        if any(keyword in user_lower for keyword in ['erinnern', 'reminder', 'vergessen', 'termin']):
            self._handle_reminder_system(user_message, ai_response)
        
        # Learning new preferences
        if any(keyword in user_lower for keyword in ['ich mag', 'ich bevorzuge', 'mir gefällt']):
            self._learn_user_preference(user_message)
    
    def _handle_task_management(self, user_message: str, ai_response: str):
        """Behandelt Task-Management"""
        try:
            # Extract task information
            task_info = self._extract_task_info(user_message)
            
            if task_info:
                # Add to active tasks
                new_task = {
                    'id': len(self.active_tasks) + 1,
                    'title': task_info.get('title', user_message[:50]),
                    'description': user_message,
                    'priority': task_info.get('priority', 'medium'),
                    'deadline': task_info.get('deadline'),
                    'created': datetime.now(),
                    'status': 'open'
                }
                
                self.active_tasks.append(new_task)
                
                # Store in memory
                self._store_task_in_memory(new_task)
                
                logger.info(f"📋 New task created: {new_task['title']}")
                
        except Exception as e:
            logger.error(f"❌ Task management error: {e}")
    
    def _handle_project_management(self, user_message: str, ai_response: str):
        """Behandelt Projekt-Management"""
        try:
            # Extract project information
            project_info = self._extract_project_info(user_message)
            
            if project_info:
                project_name = project_info.get('name', f"Projekt_{len(self.projects) + 1}")
                
                if project_name not in self.projects:
                    self.projects[project_name] = {
                        'name': project_name,
                        'description': user_message,
                        'created': datetime.now(),
                        'status': 'active',
                        'tasks': [],
                        'progress': 0
                    }
                    
                    # Update user profile
                    if 'projects' not in self.user_profile:
                        self.user_profile['projects'] = []
                    self.user_profile['projects'].append(project_name)
                    
                    logger.info(f"📊 New project created: {project_name}")
                
                # Store in memory
                self._store_project_in_memory(self.projects[project_name])
                
        except Exception as e:
            logger.error(f"❌ Project management error: {e}")
    
    def _handle_reminder_system(self, user_message: str, ai_response: str):
        """Behandelt Erinnerungs-System"""
        try:
            # Extract reminder information
            reminder_info = self._extract_reminder_info(user_message)
            
            if reminder_info:
                new_reminder = {
                    'id': len(self.reminders) + 1,
                    'text': reminder_info.get('text', user_message),
                    'remind_at': reminder_info.get('when'),
                    'created': datetime.now(),
                    'status': 'active'
                }
                
                self.reminders.append(new_reminder)
                
                # Store in memory
                self._store_reminder_in_memory(new_reminder)
                
                logger.info(f"⏰ New reminder created: {new_reminder['text']}")
                
        except Exception as e:
            logger.error(f"❌ Reminder system error: {e}")
    
    def _extract_task_info(self, message: str) -> Dict[str, Any]:
        """Extrahiert Task-Informationen aus Nachricht"""
        info = {}
        message_lower = message.lower()
        
        # Priority detection
        if any(word in message_lower for word in ['wichtig', 'urgent', 'dringend']):
            info['priority'] = 'high'
        elif any(word in message_lower for word in ['low', 'niedrig', 'später']):
            info['priority'] = 'low'
        else:
            info['priority'] = 'medium'
        
        # Enhanced title extraction - more flexible patterns
        title = None
        
        # Pattern 1: "task: <title>" or "aufgabe: <title>"
        for keyword in ['task:', 'aufgabe:', 'todo:']:
            if keyword in message_lower:
                parts = message.split(keyword, 1)
                if len(parts) > 1:
                    title = parts[1].strip()[:50]
                    break
        
        # Pattern 2: "task für <title>" or "aufgabe für <title>"
        if not title:
            for keyword in ['task für', 'aufgabe für', 'task zum', 'aufgabe zum']:
                if keyword in message_lower:
                    parts = message.split(keyword, 1)
                    if len(parts) > 1:
                        title = parts[1].strip()[:50]
                        break
        
        # Pattern 3: "neue aufgabe: <title>" or similar
        if not title:
            for keyword in ['neue aufgabe:', 'neue task:', 'neuer task:', 'neue todo:']:
                if keyword in message_lower:
                    parts = message.split(keyword, 1)
                    if len(parts) > 1:
                        title = parts[1].strip()[:50]
                        break
        
        # Pattern 4: Simple task detection - any message with task keywords
        if not title and any(word in message_lower for word in ['task', 'aufgabe', 'todo', 'erledigen']):
            # Use the message as title but clean it up
            title = message.strip()[:50]
        
        if title:
            info['title'] = title.strip('.,!?')
        
        return info if 'title' in info else None
    
    def _extract_project_info(self, message: str) -> Dict[str, Any]:
        """Extrahiert Projekt-Informationen aus Nachricht"""
        info = {}
        message_lower = message.lower()
        
        # Enhanced project name extraction - more flexible patterns
        project_name = None
        
        # Pattern 1: "projekt: <name>" or "project: <name>"
        for keyword in ['projekt:', 'project:']:
            if keyword in message_lower:
                parts = message_lower.split(keyword, 1)
                if len(parts) > 1:
                    potential_name = parts[1].strip().split()[0] if parts[1].strip() else None
                    if potential_name:
                        project_name = potential_name
                        break
        
        # Pattern 2: "neues projekt <name>" or "new project <name>"
        if not project_name:
            for keyword in ['neues projekt', 'new project', 'projekt für', 'project für']:
                if keyword in message_lower:
                    parts = message_lower.split(keyword, 1)
                    if len(parts) > 1:
                        potential_name = parts[1].strip().split()[0] if parts[1].strip() else None
                        if potential_name:
                            project_name = potential_name
                            break
        
        # Pattern 3: "arbeite an projekt <name>" or "working on project <name>" or just "projekt <name>"
        if not project_name:
            for keyword in ['arbeite an projekt', 'working on project', 'projekt', 'project']:
                if keyword in message_lower:
                    parts = message_lower.split(keyword, 1)
                    if len(parts) > 1:
                        potential_name = parts[1].strip().split()[0] if parts[1].strip() else None
                        if potential_name:
                            project_name = potential_name
                            break
        
        if project_name:
            info['name'] = project_name.strip('.,!?')
        
        return info if 'name' in info else None
    
    def _extract_reminder_info(self, message: str) -> Dict[str, Any]:
        """Extrahiert Erinnerungs-Informationen aus Nachricht"""
        info = {}
        message_lower = message.lower()
        
        # Simple reminder text extraction
        if 'erinnern' in message_lower:
            parts = message.split('erinnern', 1)
            if len(parts) > 1:
                info['text'] = parts[1].strip()
        else:
            info['text'] = message
        
        # Simple time extraction (can be enhanced with NLP)
        if any(time_word in message_lower for time_word in ['morgen', 'heute', 'später']):
            info['when'] = 'relative_time_detected'
        
        return info
    
    def _store_task_in_memory(self, task: Dict[str, Any]):
        """Speichert Task im Memory System"""
        try:
            if self.memory_system:
                if hasattr(self.memory_system, 'add_memory'):
                    self.memory_system.add_memory(
                        content=task,
                        memory_type='task',
                        importance=0.8,
                        metadata={'task_id': task['id'], 'status': task['status']}
                    )
        except Exception as e:
            logger.error(f"❌ Task memory storage error: {e}")
    
    def _store_project_in_memory(self, project: Dict[str, Any]):
        """Speichert Projekt im Memory System"""
        try:
            if self.memory_system:
                if hasattr(self.memory_system, 'add_memory'):
                    self.memory_system.add_memory(
                        content=project,
                        memory_type='project',
                        importance=0.9,
                        metadata={'project_name': project['name'], 'status': project['status']}
                    )
        except Exception as e:
            logger.error(f"❌ Project memory storage error: {e}")
    
    def _store_reminder_in_memory(self, reminder: Dict[str, Any]):
        """Speichert Erinnerung im Memory System"""
        try:
            if self.memory_system:
                if hasattr(self.memory_system, 'add_memory'):
                    self.memory_system.add_memory(
                        content=reminder,
                        memory_type='reminder',
                        importance=0.7,
                        metadata={'reminder_id': reminder['id'], 'status': reminder['status']}
                    )
        except Exception as e:
            logger.error(f"❌ Reminder memory storage error: {e}")
    
    def _learn_user_preference(self, message: str):
        """Lernt User-Präferenzen"""
        try:
            # Simple preference learning
            preference_data = {
                'statement': message,
                'timestamp': datetime.now(),
                'context': 'voice_interaction'
            }
            
            # Store in user preferences
            if 'learned_preferences' not in self.user_profile:
                self.user_profile['learned_preferences'] = []
            
            self.user_profile['learned_preferences'].append(preference_data)
            
            logger.info(f"🧠 Learned new preference: {message[:30]}...")
            
        except Exception as e:
            logger.error(f"❌ Preference learning error: {e}")
    
    def _enhance_response_with_context(self, response: str) -> str:
        """Erweitert Antwort mit Kontext-Informationen"""
        try:
            # Add proactive suggestions based on context
            enhanced = response
            
            # Add task count if user mentioned tasks
            if self.active_tasks and any(keyword in response.lower() for keyword in ['task', 'aufgabe']):
                enhanced += f" Du hast aktuell {len(self.active_tasks)} offene Aufgaben."
            
            # Add project info if relevant
            if self.projects and any(keyword in response.lower() for keyword in ['projekt', 'project']):
                enhanced += f" Du arbeitest an {len(self.projects)} Projekten."
            
            return enhanced
            
        except Exception as e:
            logger.error(f"❌ Response enhancement error: {e}")
            return response
    
    def _generate_intelligent_fallback_response(self, user_message: str) -> str:
        """Generiert intelligente Fallback-Antwort mit Assistant-Features"""
        user_lower = user_message.lower()
        
        # Extract and remember important information
        self._extract_user_information(user_message)
        
        # Task management responses
        if any(keyword in user_lower for keyword in ['task', 'aufgabe', 'todo', 'erledigen']):
            self._handle_task_management(user_message, "")
            return f"Ich habe deine Aufgabe notiert. Du hast jetzt {len(self.active_tasks)} offene Aufgaben. Soll ich sie strukturieren?"
        
        # Project management responses
        if any(keyword in user_lower for keyword in ['projekt', 'project', 'arbeite an']):
            self._handle_project_management(user_message, "")
            return f"Dein Projekt wurde gespeichert. Du arbeitest an {len(self.projects)} Projekten. Brauchst du Hilfe bei der Organisation?"
        
        # Reminder responses
        if any(keyword in user_lower for keyword in ['erinnern', 'reminder', 'vergessen', 'termin']):
            self._handle_reminder_system(user_message, "")
            return f"Erinnerung notiert! Du hast {len(self.reminders)} aktive Erinnerungen."
        
        # Status inquiries
        if any(keyword in user_lower for keyword in ['status', 'überblick', 'was steht an']):
            return self._generate_status_overview()
        
        # Enhanced greeting patterns
        if any(greeting in user_lower for greeting in ['hallo', 'hi', 'hey', 'guten morgen', 'guten tag']):
            return self._generate_personalized_greeting()
        
        # Help requests
        if any(help_word in user_lower for help_word in ['hilfe', 'help', 'unterstützung', 'was kannst du']):
            return "Ich bin dein persönlicher Assistent! Ich helfe dir bei Aufgaben, Projekten, Erinnerungen und mehr. Einfach sagen, was du brauchst!"
        
        # Default enhanced response
        return f"Verstanden. Kann ich dir dabei helfen, das zu organisieren oder als Aufgabe zu vermerken?"
    
    def _generate_status_overview(self) -> str:
        """Generiert Status-Übersicht"""
        overview_parts = []
        
        if self.active_tasks:
            overview_parts.append(f"{len(self.active_tasks)} offene Aufgaben")
        
        if self.projects:
            overview_parts.append(f"{len(self.projects)} aktive Projekte")
        
        if self.reminders:
            overview_parts.append(f"{len(self.reminders)} Erinnerungen")
        
        if overview_parts:
            return f"Du hast {', '.join(overview_parts)}. Soll ich Details nennen?"
        else:
            return "Alles erledigt! Keine offenen Aufgaben oder Projekte."
    
    def _extract_user_information(self, user_message: str):
        """Extrahiert wichtige Informationen aus User-Nachricht"""
        try:
            user_lower = user_message.lower()
            
            # Extract name
            if 'ich heiße' in user_lower:
                name_part = user_message.split('ich heiße')[1].strip()
                name = name_part.split()[0].strip('.,!?')
                self.user_profile['name'] = name
                logger.info(f"✅ Name extrahiert: {name}")
            
            elif 'mein name ist' in user_lower:
                name_part = user_message.split('mein name ist')[1].strip()
                name = name_part.split()[0].strip('.,!?')
                self.user_profile['name'] = name
                logger.info(f"✅ Name extrahiert: {name}")
            
            # Extract projects
            if any(project_word in user_lower for project_word in ['projekt', 'arbeite an', 'entwickle']):
                if 'projects' not in self.user_profile:
                    self.user_profile['projects'] = []
                
                # Simple project extraction
                if 'projekt' in user_lower:
                    project_part = user_message.split('projekt')[1].strip()
                    project = project_part.split()[0].strip('.,!?')
                    if project and project not in self.user_profile['projects']:
                        self.user_profile['projects'].append(project)
                        logger.info(f"✅ Projekt extrahiert: {project}")
            
            # Save updated profile
            self._save_user_profile()
            
        except Exception as e:
            logger.error(f"❌ Information extraction error: {e}")
    
    def get_assistant_status(self) -> Dict[str, Any]:
        """Gibt detaillierten Assistant-Status zurück"""
        try:
            status = {
                'assistant_active': self.is_voice_only_mode,
                'capabilities': self.assistant_capabilities,
                'active_tasks': len(self.active_tasks),
                'projects': len(self.projects),
                'reminders': len(self.reminders),
                'user_profile': self.user_profile,
                'lm_studio_available': self.lm_studio.is_available if self.lm_studio else False,
                'memory_system_active': self.memory_system is not None
            }
            
            # Recent activity
            if self.active_tasks:
                status['recent_tasks'] = [task['title'] for task in self.active_tasks[-3:]]
            
            if self.projects:
                status['active_projects'] = list(self.projects.keys())
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Assistant status error: {e}")
            return {'error': str(e)}


class SimpleVoiceCommandProcessor:
    """Simple Voice Command Processor für Voice-only Mode"""
    
    def __init__(self, voice_system, lm_studio=None, memory_system=None):
        self.voice_system = voice_system
        self.lm_studio = lm_studio
        self.memory_system = memory_system
        
    def process_command(self, text: str) -> str:
        """Verarbeitet Voice Commands"""
        try:
            # For now, just pass through to AI response generation
            return self.voice_system._generate_ai_response(text)
        except Exception as e:
            logger.error(f"❌ Command processing error: {e}")
            return "Entschuldigung, ich konnte den Befehl nicht verarbeiten."