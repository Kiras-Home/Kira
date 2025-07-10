"""
Utils Module für Kira Voice System
Enthält Hilfsfunktionen für Audio-Verarbeitung und System-Utilities
"""

import logging

logger = logging.getLogger(__name__)

# Import Audio Utils
try:
    from .audio_utils import (
        AudioProcessor,
        VoiceActivityDetector,
        create_silence,
        create_tone,
        mix_audio,
        convert_to_mono
    )
    logger.info("✅ Audio Utils geladen")
    AUDIO_UTILS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Audio Utils Import Fehler: {e}")
    AUDIO_UTILS_AVAILABLE = False

# System Utils
import platform
import psutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

class SystemUtils:
    """System-Hilfsfunktionen"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Gibt System-Informationen zurück"""
        try:
            return {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'architecture': platform.architecture()[0],
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_total': psutil.disk_usage('/').total if platform.system() != 'Windows' else psutil.disk_usage('C:').total
            }
        except Exception as e:
            logger.error(f"❌ System Info Fehler: {e}")
            return {}
    
    @staticmethod
    def get_performance_info() -> Dict[str, Any]:
        """Gibt Performance-Informationen zurück"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'memory_used': memory.used,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                'boot_time': psutil.boot_time(),
                'uptime_seconds': time.time() - psutil.boot_time()
            }
        except Exception as e:
            logger.error(f"❌ Performance Info Fehler: {e}")
            return {}
    
    @staticmethod
    def ensure_directory(path: str) -> bool:
        """Stellt sicher, dass Verzeichnis existiert"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"❌ Verzeichnis erstellen Fehler: {e}")
            return False
    
    @staticmethod
    def cleanup_old_files(directory: str, max_age_days: int = 7) -> int:
        """Löscht alte Dateien"""
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return 0
            
            cutoff_time = time.time() - (max_age_days * 24 * 3600)
            deleted_count = 0
            
            for file_path in dir_path.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except:
                        pass
            
            logger.info(f"🧹 {deleted_count} alte Dateien gelöscht")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Cleanup Fehler: {e}")
            return 0

class LoggingUtils:
    """Logging-Hilfsfunktionen"""
    
    @staticmethod
    def setup_kira_logging(level: str = "INFO", 
                          log_file: Optional[str] = None) -> bool:
        """Richtet Kira-Logging ein"""
        try:
            log_level = getattr(logging, level.upper())
            
            # Format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            
            # Root Logger konfigurieren
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            root_logger.addHandler(console_handler)
            
            # File Handler (optional)
            if log_file:
                SystemUtils.ensure_directory(str(Path(log_file).parent))
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            
            logger.info(f"📝 Kira Logging eingerichtet: {level}")
            return True
            
        except Exception as e:
            print(f"❌ Logging Setup Fehler: {e}")
            return False
    
    @staticmethod
    def get_log_stats(log_file: str) -> Dict[str, Any]:
        """Analysiert Log-Datei"""
        try:
            log_path = Path(log_file)
            if not log_path.exists():
                return {}
            
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            stats = {
                'total_lines': len(lines),
                'file_size': log_path.stat().st_size,
                'created': log_path.stat().st_ctime,
                'modified': log_path.stat().st_mtime
            }
            
            # Level-Statistiken
            levels = {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0}
            for line in lines:
                for level in levels:
                    if level in line:
                        levels[level] += 1
                        break
            
            stats['levels'] = levels
            return stats
            
        except Exception as e:
            logger.error(f"❌ Log Stats Fehler: {e}")
            return {}

class ConfigUtils:
    """Konfigurations-Hilfsfunktionen"""
    
    @staticmethod
    def save_config(config_dict: Dict[str, Any], 
                   config_file: str) -> bool:
        """Speichert Konfiguration als JSON"""
        try:
            import json
            
            config_path = Path(config_file)
            SystemUtils.ensure_directory(str(config_path.parent))
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Konfiguration gespeichert: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Config speichern Fehler: {e}")
            return False
    
    @staticmethod
    def load_config(config_file: str) -> Dict[str, Any]:
        """Lädt Konfiguration aus JSON"""
        try:
            import json
            
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"⚠️ Config-Datei nicht gefunden: {config_file}")
                return {}
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"📂 Konfiguration geladen: {config_file}")
            return config
            
        except Exception as e:
            logger.error(f"❌ Config laden Fehler: {e}")
            return {}

# Performance Monitoring
class PerformanceMonitor:
    """Performance-Überwachung für Kira"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Startet Timer"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """Beendet Timer und gibt Dauer zurück"""
        if name not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(duration)
        del self.start_times[name]
        
        return duration
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Gibt Performance-Statistiken zurück"""
        stats = {}
        
        for name, durations in self.metrics.items():
            if durations:
                stats[name] = {
                    'count': len(durations),
                    'total': sum(durations),
                    'average': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'last': durations[-1]
                }
        
        return stats
    
    def reset(self):
        """Setzt Metriken zurück"""
        self.metrics.clear()
        self.start_times.clear()

# Test-Funktionen
def test_utils_system():
    """Testet das Utils System"""
    print("🔧 === UTILS SYSTEM TEST ===")
    
    success = True
    
    # Test Audio Utils
    if AUDIO_UTILS_AVAILABLE:
        try:
            print("🎵 Teste Audio Utils...")
            
            # Test Audio Processor
            processor = AudioProcessor()
            
            # Test Audio erstellen
            test_audio = create_tone(440, 1.0)  # 1s 440Hz Ton
            print(f"   ✅ Test Audio erstellt: {len(test_audio)} Samples")
            
            # Test Normalisierung
            normalized = processor.normalize_audio(test_audio)
            print(f"   ✅ Audio normalisiert")
            
            # Test Features
            features = processor.calculate_audio_features(test_audio)
            print(f"   ✅ Audio Features: {len(features)} Features")
            
            # Test VAD
            vad = VoiceActivityDetector()
            is_speech, score = vad.is_speech(test_audio[:320])  # Ein Frame
            print(f"   ✅ VAD Test: Speech={is_speech}, Score={score:.2f}")
            
        except Exception as e:
            print(f"   ❌ Audio Utils Fehler: {e}")
            success = False
    else:
        print("⚠️ Audio Utils nicht verfügbar")
    
    # Test System Utils
    try:
        print("\n🖥️ Teste System Utils...")
        
        sys_info = SystemUtils.get_system_info()
        print(f"   ✅ System Info: {sys_info.get('platform', 'Unknown')}")
        
        perf_info = SystemUtils.get_performance_info()
        print(f"   ✅ Performance: CPU {perf_info.get('cpu_percent', 0):.1f}%")
        
    except Exception as e:
        print(f"   ❌ System Utils Fehler: {e}")
        success = False
    
    # Test Performance Monitor
    try:
        print("\n⏱️ Teste Performance Monitor...")
        
        monitor = PerformanceMonitor()
        
        monitor.start_timer("test")
        time.sleep(0.1)
        duration = monitor.end_timer("test")
        
        stats = monitor.get_stats()
        print(f"   ✅ Performance Monitor: {duration:.3f}s gemessen")
        
    except Exception as e:
        print(f"   ❌ Performance Monitor Fehler: {e}")
        success = False
    
    if success:
        print(f"\n🎉 === UTILS SYSTEM FUNKTIONIERT! ===")
    else:
        print(f"\n❌ Utils System hat Probleme")
    
    return success

# Export
__all__ = [
    # Audio Utils
    'AudioProcessor',
    'VoiceActivityDetector',
    'create_silence',
    'create_tone',
    'mix_audio',
    'convert_to_mono',
    
    # System Utils
    'SystemUtils',
    'LoggingUtils',
    'ConfigUtils',
    'PerformanceMonitor',
    
    # Test
    'test_utils_system',
    
    # Flags
    'AUDIO_UTILS_AVAILABLE'
]

# Log beim Import
logger.info("📦 Kira Utils Module geladen")