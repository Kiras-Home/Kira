"""
Kira Audio Module
"""

import logging
logger = logging.getLogger(__name__)

from .recorder import SimpleAudioRecorder
from .player import SimpleAudioPlayer

__all__ = ['SimpleAudioRecorder', 'SimpleAudioPlayer']

logger.info("📦 Kira Audio Module geladen")