"""
Audio Player für lokale Audio-Wiedergabe
Unterstützt verschiedene Betriebssysteme
"""

import logging
import subprocess
import platform
import os
import tempfile
from pathlib import Path
from typing import Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

class SimpleAudioPlayer:
    """Einfacher Audio Player für lokale Wiedergabe"""
    
    def __init__(self, output_dir: str = "voice/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.system = platform.system()
        self.player_command = self._detect_audio_player()
        
        logger.info(f"🔊 Audio Player initialisiert: {self.system} ({self.player_command})")
    
    def _detect_audio_player(self) -> Optional[str]:
        """Erkennt verfügbaren Audio Player"""
        
        if self.system == "Darwin":  # macOS
            return "afplay"
        elif self.system == "Linux":
            # Teste verschiedene Linux Player
            players = ["aplay", "paplay", "play", "ffplay"]
            for player in players:
                try:
                    subprocess.run([player, "--version"], capture_output=True, timeout=2)
                    return player
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            return None
        elif self.system == "Windows":
            return "powershell"
        
        return None
    
    def play_file(self, file_path: Union[str, Path], timeout: int = 30) -> bool:
        """Spielt Audio-Datei ab"""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"❌ Audio-Datei nicht gefunden: {file_path}")
            return False
        
        try:
            logger.info(f"🔊 Spiele Audio ab: {file_path.name}")
            
            if self.system == "Darwin":  # macOS
                result = subprocess.run(
                    ["afplay", str(file_path)], 
                    timeout=timeout,
                    capture_output=True
                )
                return result.returncode == 0
                
            elif self.system == "Linux":
                if self.player_command == "aplay":
                    result = subprocess.run(
                        ["aplay", str(file_path)], 
                        timeout=timeout,
                        capture_output=True
                    )
                elif self.player_command == "paplay":
                    result = subprocess.run(
                        ["paplay", str(file_path)], 
                        timeout=timeout,
                        capture_output=True
                    )
                elif self.player_command == "play":
                    result = subprocess.run(
                        ["play", str(file_path)], 
                        timeout=timeout,
                        capture_output=True
                    )
                else:
                    logger.error("❌ Kein Audio Player verfügbar")
                    return False
                
                return result.returncode == 0
                
            elif self.system == "Windows":
                ps_command = f'(New-Object Media.SoundPlayer "{file_path}").PlaySync()'
                result = subprocess.run(
                    ["powershell", "-Command", ps_command], 
                    timeout=timeout,
                    capture_output=True
                )
                return result.returncode == 0
            
            else:
                logger.error(f"❌ Betriebssystem {self.system} nicht unterstützt")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Audio-Wiedergabe Timeout ({timeout}s)")
            return False
        except Exception as e:
            logger.error(f"❌ Audio-Wiedergabe Fehler: {e}")
            return False
    
    def play_numpy_array(self, audio_data: np.ndarray, sample_rate: int = 16000, format: str = "wav") -> bool:
        """Spielt NumPy Audio Array ab"""
        
        try:
            # Erstelle temporäre Datei
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f".{format}", 
                dir=self.output_dir,
                delete=False
            )
            temp_path = Path(temp_file.name)
            temp_file.close()
            
            # Speichere Audio als Datei
            if format == "wav":
                import soundfile as sf
                sf.write(str(temp_path), audio_data, sample_rate)
            else:
                logger.error(f"❌ Audio-Format {format} nicht unterstützt")
                return False
            
            # Spiele Datei ab
            success = self.play_file(temp_path)
            
            # Cleanup
            try:
                temp_path.unlink()
            except:
                pass
            
            return success
            
        except Exception as e:
            logger.error(f"❌ NumPy Array Wiedergabe Fehler: {e}")
            return False
    
    def test_audio_system(self) -> bool:
        """Testet das Audio-System"""
        
        try:
            logger.info("🧪 Teste Audio System...")
            
            # Erstelle Test-Audio (1 Sekunde Sinus-Ton)
            sample_rate = 16000
            duration = 1.0
            frequency = 440  # A4 Note
            
            t = np.linspace(0.0, duration, int(sample_rate * duration), False)
            test_audio = 0.3 * np.sin(frequency * 2.0 * np.pi * t)
            
            # Spiele Test-Audio ab
            logger.info("🔔 Spiele Test-Ton ab (440Hz, 1s)...")
            success = self.play_numpy_array(test_audio, sample_rate)
            
            if success:
                logger.info("✅ Audio System Test erfolgreich")
                return True
            else:
                logger.error("❌ Audio System Test fehlgeschlagen")
                return False
                
        except Exception as e:
            logger.error(f"❌ Audio System Test Fehler: {e}")
            return False
    
    def get_system_info(self) -> dict:
        """Gibt System-Informationen zurück"""
        return {
            "system": self.system,
            "player_command": self.player_command,
            "player_available": self.player_command is not None,
            "output_dir": str(self.output_dir)
        }

# Export
__all__ = ['SimpleAudioPlayer']

# Test-Funktion
def test_audio_player():
    """Test Audio Player"""
    print("🔊 === AUDIO PLAYER TEST ===")
    
    player = SimpleAudioPlayer()
    
    # System Info
    info = player.get_system_info()
    print(f"System: {info['system']}")
    print(f"Player: {info['player_command']}")
    print(f"Verfügbar: {info['player_available']}")
    
    # Audio Test
    if info['player_available']:
        success = player.test_audio_system()
        print(f"Audio Test: {'✅ Erfolgreich' if success else '❌ Fehlgeschlagen'}")
        return success
    else:
        print("❌ Kein Audio Player verfügbar")
        return False

if __name__ == "__main__":
    test_audio_player()