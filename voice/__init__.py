"""
Kira Voice System
"""

import logging
from pathlib import Path

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version
__version__ = "0.1.0"

# Export wichtiger Klassen
from .audio.recorder import SimpleAudioRecorder
from .audio.player import SimpleAudioPlayer
from .recognition.simple_detector import SimpleWakeWordDetector
from .recognition.whisper_engine import WhisperEngine

# Erstelle Verzeichnisse
def setup_directories():
    dirs = [
        "voice/output",
        "voice/models",
        "voice/models/whisper"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

setup_directories()
logger.info("📦 Kira Voice System initialisiert")

__all__ = [
    'SimpleAudioRecorder',
    'SimpleAudioPlayer',
    'SimpleWakeWordDetector',
    'WhisperEngine'
]