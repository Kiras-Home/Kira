"""
🎭 MOCK VOICE SERVICE for WSL Development
Audio-free voice service for development without hardware audio
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MockAudioData:
    """Mock audio data for testing"""
    data: np.ndarray
    sample_rate: int
    duration: float
    success: bool
    error: Optional[str] = None

@dataclass
class ServiceResponse:
    """Service response for mock operations"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class MockVoiceService:
    """Mock Voice Service for WSL development"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.initialized = False
        self.running = False
        self.mock_responses = [
            "Das ist eine Testnachricht vom Mock Voice Service.",
            "Hallo! Ich bin Kira's Mock-System für WSL-Entwicklung.",
            "Audio-Hardware wird simuliert da WSL keine Audio-Unterstützung hat.",
            "Diese Antwort würde normalerweise über Sprachsynthese ausgegeben."
        ]
        self.response_counter = 0
        
        logger.info("🎭 Mock Voice Service für WSL erstellt")
    
    async def initialize(self) -> bool:
        """Initialize mock voice service"""
        try:
            logger.info("🔧 Mock Voice Service wird initialisiert...")
            await asyncio.sleep(0.5)  # Simulate initialization time
            
            self.initialized = True
            logger.info("✅ Mock Voice Service erfolgreich initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Mock Voice Service Initialisierung fehlgeschlagen: {e}")
            return False
    
    async def start(self) -> bool:
        """Start mock voice service"""
        try:
            if not self.initialized:
                logger.error("Service nicht initialisiert")
                return False
            
            logger.info("🚀 Mock Voice Service wird gestartet...")
            await asyncio.sleep(0.2)
            
            self.running = True
            logger.info("✅ Mock Voice Service erfolgreich gestartet")
            return True
            
        except Exception as e:
            logger.error(f"❌ Mock Voice Service Start fehlgeschlagen: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop mock voice service"""
        try:
            logger.info("🛑 Mock Voice Service wird gestoppt...")
            self.running = False
            await asyncio.sleep(0.1)
            
            logger.info("✅ Mock Voice Service gestoppt")
            return True
            
        except Exception as e:
            logger.error(f"❌ Mock Voice Service Stop fehlgeschlagen: {e}")
            return False
    
    async def speak_text(self, text: str, emotion: str = "neutral", wait_for_completion: bool = True) -> ServiceResponse:
        """Mock text-to-speech"""
        try:
            logger.info(f"🗣️ MOCK TTS: '{text}' (Emotion: {emotion})")
            
            if wait_for_completion:
                # Simulate speech duration
                speech_duration = len(text) * 0.05  # ~50ms per character
                await asyncio.sleep(min(speech_duration, 2.0))  # Max 2 seconds
            
            return ServiceResponse(
                success=True,
                data={
                    "text": text,
                    "emotion": emotion,
                    "audio_generated": True,
                    "duration": len(text) * 0.05,
                    "mock_mode": True
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Mock TTS fehler: {e}")
            return ServiceResponse(success=False, error=str(e))
    
    async def listen_once(self, duration: float = 3.0, return_audio: bool = False) -> ServiceResponse:
        """Mock audio recording"""
        try:
            logger.info(f"🎤 MOCK RECORDING: {duration}s (simulated)")
            
            # Simulate recording time
            await asyncio.sleep(min(duration, 1.0))  # Max 1 second simulation
            
            # Generate mock response
            mock_text = self.mock_responses[self.response_counter % len(self.mock_responses)]
            self.response_counter += 1
            
            response_data = {
                "recognized_text": mock_text,
                "confidence": 0.95,
                "audio_duration": duration,
                "mock_mode": True
            }
            
            if return_audio:
                # Generate mock audio data
                mock_audio = np.random.normal(0, 0.1, int(16000 * duration)).astype(np.float32)
                response_data["audio_data"] = mock_audio
            
            logger.info(f"🎤 MOCK ERKANNT: '{mock_text}'")
            
            return ServiceResponse(
                success=True,
                data=response_data
            )
            
        except Exception as e:
            logger.error(f"❌ Mock Recording fehler: {e}")
            return ServiceResponse(success=False, error=str(e))
    
    async def get_voice_status(self) -> Dict[str, Any]:
        """Get mock voice status"""
        return {
            'service': {
                'initialized': self.initialized,
                'running': self.running,
                'health': 'healthy' if self.running else 'stopped',
                'uptime': 42.0,  # Mock uptime
                'mode': 'mock_wsl'
            },
            'components': {
                'audio_manager': True,  # Mock: Always "working"
                'wake_word_detector': True,
                'speech_recognition': True,
                'voice_synthesis': True,
                'pipeline': True
            },
            'component_health': {
                'audio_manager': True,
                'wake_word_detector': True,
                'speech_recognition': True,
                'voice_synthesis': True,
                'pipeline': True
            }
        }
    
    def _get_service_statistics(self) -> Dict[str, Any]:
        """Get mock service statistics"""
        return {
            'total_requests': 15,
            'successful_requests': 15,
            'failed_requests': 0,
            'average_response_time': 0.123
        }
    
    async def cleanup(self):
        """Cleanup mock service"""
        logger.info("🧹 Mock Voice Service cleanup...")
        self.running = False
        self.initialized = False