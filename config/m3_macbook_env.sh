# Kira Voice System - Umgebungsvariablen für M3 MacBook
# Diese Datei in ~/.zshrc oder ~/.bashrc einbinden oder separat sourcing

# Whisper Engine Einstellungen
export WHISPER_DEVICE=cpu           # Forciert CPU-Nutzung (empfohlen für M3)
export WHISPER_CACHE_DIR=voice/models/whisper
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Aktiviert automatisches MPS-Fallback

# Performance Optimierungen für M3
export OMP_NUM_THREADS=8            # Optimiert für M3's CPU-Kerne
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Bark TTS Einstellungen
export BARK_DEVICE=cpu              # Auch Bark sollte CPU verwenden

# Python Optimierungen
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Deaktiviert MPS High Watermark

# Logging
export KIRA_LOG_LEVEL=INFO

echo "🍎 Kira Voice System - M3 MacBook Konfiguration geladen"
echo "💡 MPS wurde deaktiviert aufgrund von PyTorch-Inkompatibilität"
echo "🎤 Whisper läuft auf CPU für optimale Stabilität"
