from fastapi import FastAPI, WebSocket
from typing import Dict, Any
import asyncio
import numpy as np
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceServer:
    def __init__(self, kira_voice: 'KiraVoice', host: str = "0.0.0.0", port: int = 8765):
        self.app = FastAPI()
        self.kira_voice = kira_voice
        self.host = host
        self.port = port
        self.active_connections: Dict[str, WebSocket] = {}
        
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.websocket("/ws/audio")
        async def audio_websocket(websocket: WebSocket):
            await self.handle_audio_connection(websocket)
            
        @self.app.post("/api/speak")
        async def speak(text: str, emotion: str = "neutral"):
            success = self.kira_voice.speak(text, emotion)
            return {"success": success}
            
        @self.app.get("/api/status")
        async def get_status():
            return self.kira_voice.get_system_status()

    async def handle_audio_connection(self, websocket: WebSocket):
        await websocket.accept()
        client_id = str(id(websocket))
        self.active_connections[client_id] = websocket
        
        try:
            while True:
                # Empfange Audio vom Client
                audio_data = await websocket.receive_bytes()
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                
                # Verarbeite Audio
                self.kira_voice.process_audio(audio_array)
                
                # Sende Antwort zurück
                response = {
                    "status": "processed",
                    "timestamp": time.time()
                }
                await websocket.send_json(response)
                
        except Exception as e:
            logger.error(f"WebSocket Error: {e}")
        finally:
            del self.active_connections[client_id]