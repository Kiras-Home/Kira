"""
Discord Chat Service - Extrahiert aus main.py
"""

from flask import jsonify
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def handle_discord_chat(message: str, user_id: str, context: dict):
    """Discord Chat Handler - Aus main.py extrahiert"""
    try:
        discord_username = context.get('discord_user', 'Unknown')
        discord_guild = context.get('discord_guild', 'Unknown Server')
        
        # Wake Word Detection
        wake_words = ['hey kira', 'hallo kira', 'hi kira', 'kira']
        message_lower = message.lower().strip()
        
        wake_word_detected = False
        used_wake_word = None
        for wake_word in wake_words:
            if wake_word in message_lower:
                wake_word_detected = True
                used_wake_word = wake_word
                break
        
        # Logging
        if wake_word_detected:
            logger.info(f"🎤 Discord Wake Word erkannt: '{used_wake_word}' von {discord_username} in {discord_guild}")
            logger.info(f"💬 Discord Message: '{message}' (User: {user_id})")
            
            # Command Extraction
            command_text = message
            for wake_word in wake_words:
                if wake_word in message_lower:
                    wake_word_pos = message_lower.find(wake_word)
                    command_start = wake_word_pos + len(wake_word)
                    command_text = message[command_start:].strip()
                    command_text = command_text.lstrip('.,!?:;- ')
                    break
            
            if command_text and command_text != message:
                logger.info(f"🎯 Discord Command extrahiert: '{command_text}'")
            else:
                logger.info(f"🎯 Discord Wake Word ohne spezifischen Command")
        else:
            logger.info(f"💬 Discord Chat: '{message[:50]}...' von {discord_username}")
        
        # Response Generation
        response = generate_discord_response(message, discord_username, wake_word_detected, used_wake_word)
        
        return jsonify({
            'success': True,
            'message': response,
            'context': {
                'platform': 'discord',
                'wake_word_detected': wake_word_detected,
                'used_wake_word': used_wake_word,
                'discord_username': discord_username,
                'discord_guild': discord_guild,
                'response_type': 'service'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Discord chat handling error: {e}")
        return jsonify({
            'success': True,
            'message': f"Hallo! Ich bin Kira. Du hast gesagt: '{message}' - Wie kann ich dir helfen?",
            'context': {
                'platform': 'discord',
                'fallback': True,
                'error': str(e)
            },
            'timestamp': datetime.now().isoformat()
        })

def generate_discord_response(message: str, discord_username: str, wake_word_detected: bool, used_wake_word: str = None) -> str:
    """Generiert Discord Response"""
    
    if wake_word_detected:
        return f"Hey {discord_username}! Du hast das Wake Word '{used_wake_word}' verwendet. Du sagtest: '{message}' - Was kann ich für dich tun? 🎤"
    else:
        return f"Hi {discord_username}! Du hast gesagt: '{message}' - Wie kann ich helfen? 🤖"