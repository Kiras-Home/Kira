from flask import Blueprint, request, jsonify, send_file, current_app
import io
import numpy as np
import base64
import soundfile as sf
from typing import Dict, Any, Optional
from flask_sock import Sock
from pathlib import Path
import logging
import time
import json



# Annahme: KiraVoice-Instanz ist global verfügbar
from voice.kira_voice import KiraVoice
from voice.config import DEFAULT_CONFIG

# Logger konfigurieren
__kira_voice_routing__ = "routes.voice_routes.kira_voice_routes"
logger = logging.getLogger(__kira_voice_routing__)

sock = Sock()

# Hole die globale Voice-Instanz (ggf. anpassen)
voice_system = KiraVoice(config=DEFAULT_CONFIG)
voice_system.initialize()

voice_api = Blueprint('voice_api', __name__, url_prefix='/api/voice')

@voice_api.route('/initialize', methods=['POST'])
def initialize_voice():
    """Initialisiert das Voice-System"""
    try:
        if not voice_system.is_initialized:
            success = voice_system.initialize()
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Voice system initialized successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to initialize voice system'
                }), 500
        return jsonify({
            'success': True,
            'message': 'Voice system already initialized'
        })
    except Exception as e:
        current_app.logger.error(f"Initialization error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@voice_api.route('/status', methods=['GET'])
def voice_status():
    """Gibt detaillierten Status des Voice-Systems zurück"""
    try:
        status: Dict[str, Any] = {
            'initialized': voice_system.is_initialized,
            'listening': voice_system.is_listening,
            'components': {
                'recorder': voice_system.recorder is not None,
                'whisper': voice_system.whisper is not None,
                'tts': voice_system.bark_tts is not None,
                'commands': voice_system.commands is not None
            },
            'config': {
                'sample_rate': voice_system.config.sample_rate,
                'language': voice_system.config.language,
                'whisper_model': voice_system.config.whisper_model
            }
        }
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        current_app.logger.error(f"Status check error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@voice_api.route('/recognize', methods=['POST'])
def recognize_speech():
    """Empfängt Audio und gibt erkannten Text zurück"""
    if not voice_system.is_initialized:
        return jsonify({'success': False, 'error': 'Voice system not initialized'}), 400
        
    try:
        if 'audio' in request.files:
            audio_file = request.files['audio']
            audio_bytes = audio_file.read()
            audio_data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        elif 'audio_base64' in request.json:
            audio_b64 = request.json['audio_base64']
            audio_bytes = base64.b64decode(audio_b64)
            audio_data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        else:
            return jsonify({'success': False, 'error': 'No audio provided'}), 400

        # Wake word detection
        if voice_system.config.wake_word:
            wake_word_detected = voice_system.detect_wake_word(audio_data)
            if not wake_word_detected:
                return jsonify({
                    'success': True,
                    'wake_word_detected': False,
                    'text': ''
                })

        # Sprache erkennen
        text = voice_system.whisper.transcribe(audio_data, samplerate)
        
        # Verarbeite Kommando wenn Text erkannt wurde
        response = None
        if text:
            response = voice_system.commands.process_command(text)

        return jsonify({
            'success': True,
            'wake_word_detected': True,
            'text': text,
            'response': response if response else None
        })

    except Exception as e:
        current_app.logger.error(f"Speech recognition error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@voice_api.route('/speak', methods=['POST'])
def speak_text():
    """Empfängt Text und gibt generiertes Audio zurück"""
    if not voice_system.is_initialized:
        return jsonify({'success': False, 'error': 'Voice system not initialized'}), 400
        
    try:
        data = request.json
        text = data.get('text')
        emotion = data.get('emotion', 'neutral')
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400

        # TTS generieren
        audio_path = voice_system.bark_tts.speak(text, emotion, auto_play=False)
        
        if audio_path is None:
            logger.error("❌ Audio-Generierung fehlgeschlagen")
            return jsonify({'success': False, 'error': 'TTS generation failed'}), 500

        # Warte kurz, um sicherzustellen, dass die Datei geschrieben wurde
        time.sleep(0.1)
        
        abs_audio_path = Path(audio_path).resolve()
        
        if not abs_audio_path.exists():
            logger.error(f"❌ Audio-Datei nicht gefunden: {abs_audio_path}")
            return jsonify({'success': False, 'error': 'Audio file not found'}), 500

        logger.info(f"📁 Sende Audio-Datei: {abs_audio_path}")
        
        try:
            return send_file(
                str(abs_audio_path),
                mimetype='audio/wav',
                as_attachment=True,
                download_name='response.wav'
            )
        finally:
            try:
                # Warte kurz bevor Löschung
                time.sleep(0.1)
                if abs_audio_path.exists():
                    abs_audio_path.unlink()
            except Exception as e:
                logger.warning(f"⚠️ Cleanup Fehler: {e}")

    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@voice_api.route('/ws')
def handle_websocket(ws):
    """WebSocket für Voice Streaming"""
    if not voice_api.sock:
        logger.error("WebSocket not initialized")
        return

    try:
        logger.info("🎤 New WebSocket connection established")
        while True:
            try:
                # Empfange Audio-Daten vom Client
                audio_data = ws.receive()
                
                if not audio_data:
                    continue
                    
                # Verarbeite Audio
                if voice_system and voice_system.is_initialized:
                    # Hier Audio verarbeiten und Antwort generieren
                    response = voice_system.process_audio_data(audio_data)
                    
                    # Sende Antwort zurück
                    ws.send(json.dumps({
                        'success': True,
                        'response': response
                    }))
                else:
                    ws.send(json.dumps({
                        'success': False,
                        'error': 'Voice system not initialized'
                    }))
                    
            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                ws.send(json.dumps({
                    'success': False,
                    'error': str(e)
                }))
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")

@voice_api.route('/reset', methods=['POST'])
def reset_voice():
    """Setzt das Voice-System zurück"""
    try:
        voice_system.is_initialized = False
        success = voice_system.initialize()
        return jsonify({
            'success': success,
            'message': 'Voice system reset successful' if success else 'Reset failed'
        })
    except Exception as e:
        current_app.logger.error(f"Reset error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@voice_api.route('/wake-up', methods=['POST'])
def voice_wake_up():
    """Handles voice wake-up events and returns greeting"""
    try:
        data = request.get_json()
        wake_word = data.get('wake_word', 'kira')
        
        # Generate dynamic greeting based on time of day
        import datetime
        current_hour = datetime.datetime.now().hour
        
        if 5 <= current_hour < 12:
            greetings = [
                'Guten Morgen! Wie kann ich dir helfen?',
                'Morgen! Was kann ich für dich tun?',
                'Hallo! Einen schönen Morgen wünsche ich dir!'
            ]
        elif 12 <= current_hour < 18:
            greetings = [
                'Hallo! Wie kann ich dir helfen?',
                'Ja, ich höre zu!',
                'Hallo, was kann ich für dich tun?',
                'Ich bin da! Wie kann ich helfen?'
            ]
        else:
            greetings = [
                'Guten Abend! Wie kann ich helfen?',
                'Hallo! Was kann ich für dich tun?',
                'Abends! Womit kann ich dir helfen?',
                'Ja, ich höre zu!'
            ]
        
        greeting = greetings[int(time.time()) % len(greetings)]
        
        # Log the wake-up event
        logger.info(f"🎤 Voice wake-up triggered with '{wake_word}' - sending greeting: {greeting}")
        
        return jsonify({
            'success': True,
            'greeting': greeting,
            'wake_word': wake_word,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Voice wake-up error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'greeting': 'Hallo! Wie kann ich dir helfen?'  # Fallback
        }), 500

@voice_api.route('/speak', methods=['POST'])
def voice_speak():
    """Text-to-Speech endpoint for server-side speech synthesis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        emotion = data.get('emotion', 'neutral')
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        
        if not voice_system.is_initialized:
            return jsonify({'success': False, 'error': 'Voice system not initialized'}), 400
        
        # Generate speech using backend TTS
        audio_path = voice_system.speak(text, emotion)
        
        if audio_path and Path(audio_path).exists():
            return send_file(
                audio_path,
                as_attachment=True,
                download_name=f'kira_speech_{int(time.time())}.wav',
                mimetype='audio/wav'
            )
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate speech'
            }), 500
            
    except Exception as e:
        logger.error(f"Voice speak error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@voice_api.route('/listen', methods=['POST'])
def voice_listen():
    """Voice recognition endpoint for processing audio input"""
    try:
        if not voice_system.is_initialized:
            return jsonify({'success': False, 'error': 'Voice system not initialized'}), 400
        
        # Get audio data from request
        if 'audio' in request.files:
            audio_file = request.files['audio']
            audio_bytes = audio_file.read()
            # Process audio and get text
            # This would need implementation in the voice system
            recognized_text = "Beispiel erkannter Text"  # Placeholder
        else:
            return jsonify({'success': False, 'error': 'No audio data provided'}), 400
        
        return jsonify({
            'success': True,
            'text': recognized_text,
            'confidence': 0.95
        })
        
    except Exception as e:
        logger.error(f"Voice listen error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
