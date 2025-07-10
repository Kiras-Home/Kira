"""
🎯 ENTERPRISE BARK ENGINE
Enhanced Bark TTS with German optimization
"""

import logging
import numpy as np
import torch
import threading
import time
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import from separate phonetic processor module
from voice.synthesis.phonetic_processors import GermanPhoneticProcessor

try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from bark.generation import SUPPORTED_LANGS
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    SAMPLE_RATE = 24000

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import scipy.io.wavfile as wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class VoicePreset:
    """Voice preset configuration"""
    name: str
    speaker_id: str
    emotion_weights: Dict[str, float]
    language: str = "de"

@dataclass
class VoiceProfile:
    """German Female Voice Profile"""
    name: str
    speaker_id: str
    description: str
    emotional_range: float
    naturalness: float
    preferred_emotions: List[str]
    sample_phrases: List[str]

@dataclass  
class SynthesisResult:
    """Synthesis result with metadata"""
    audio_data: Optional[np.ndarray]
    sample_rate: int
    duration: float
    text: str
    voice_preset: str
    processing_time: float
    success: bool = True
    error: Optional[str] = None
    model_used: str = "bark"
    quality_score: float = 0.0
    cache_hit: bool = False

class EmotionModulator:
    """Emotional Voice Modulation for German Female Voice"""
    
    def __init__(self):
        self.emotion_configs = {
            'neutral': {
                'speaker_prefix': '[Neutral female German speaker]',
                'tone_markers': '',
                'speed_modifier': 1.0,
                'pitch_hint': ''
            },
            'happy': {
                'speaker_prefix': '[Happy female German speaker]',
                'tone_markers': '😊',
                'speed_modifier': 1.1,
                'pitch_hint': '♪'
            },
            'sad': {
                'speaker_prefix': '[Sad female German speaker]',
                'tone_markers': '😢',
                'speed_modifier': 0.9,
                'pitch_hint': '♪'
            },
            'excited': {
                'speaker_prefix': '[Excited female German speaker]',
                'tone_markers': '🤗',
                'speed_modifier': 1.2,
                'pitch_hint': '♪♪'
            },
            'calm': {
                'speaker_prefix': '[Calm female German speaker]',
                'tone_markers': '😌',
                'speed_modifier': 0.95,
                'pitch_hint': ''
            },
            'friendly': {
                'speaker_prefix': '[Friendly female German speaker]',
                'tone_markers': '😊',
                'speed_modifier': 1.05,
                'pitch_hint': '♪'
            },
            'professional': {
                'speaker_prefix': '[Professional female German speaker]',
                'tone_markers': '',
                'speed_modifier': 1.0,
                'pitch_hint': ''
            },
            'empathetic': {
                'speaker_prefix': '[Empathetic female German speaker]',
                'tone_markers': '🤗',
                'speed_modifier': 0.95,
                'pitch_hint': '♪'
            },
            'surprised': {
                'speaker_prefix': '[Surprised female German speaker]',
                'tone_markers': '😲',
                'speed_modifier': 1.15,
                'pitch_hint': '♪!'
            },
            'thoughtful': {
                'speaker_prefix': '[Thoughtful female German speaker]',
                'tone_markers': '🤔',
                'speed_modifier': 0.9,
                'pitch_hint': '...'
            }
        }
        
        logger.info(f"🎭 Emotion Modulator initialized with {len(self.emotion_configs)} emotions")
    
    def apply_emotion(self, text: str, emotion: str = "neutral") -> str:
        """Apply emotional modulation to text"""
        try:
            if emotion not in self.emotion_configs:
                logger.warning(f"Unknown emotion '{emotion}', using neutral")
                emotion = "neutral"
            
            config = self.emotion_configs[emotion]
            
            # Build modulated text
            modulated_text = ""
            
            # Add speaker context
            if config['speaker_prefix']:
                modulated_text += config['speaker_prefix'] + " "
            
            # Add tone markers
            if config['tone_markers']:
                modulated_text += config['tone_markers'] + " "
            
            # Add the main text
            modulated_text += text
            
            # Add pitch hints
            if config['pitch_hint']:
                modulated_text += " " + config['pitch_hint']
            
            logger.debug(f"Emotion modulation ({emotion}): '{text}' -> '{modulated_text}'")
            return modulated_text
            
        except Exception as e:
            logger.error(f"Emotion modulation error: {e}")
            return text

class VoiceCache:
    """Voice Response Caching System"""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 500, max_age_days: int = 7):
        # ✅ FIX: Handle None cache_dir
        if cache_dir is None:
            cache_dir = Path("cache/voice/bark")  # Default fallback
            
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.max_age_days = max_age_days
        
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'size_mb': 0.0,
            'entries': 0
        }
        
        self._update_cache_stats()
        logger.info(f"🗄️ Voice Cache initialized: {self.stats['entries']} entries, {self.stats['size_mb']:.1f}MB")
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index"""
        try:
            if self.cache_index_file.exists():
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Cache index load error: {e}")
            return {}
    
    def _save_cache_index(self):
        """Save cache index"""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Cache index save error: {e}")
    
    def _generate_cache_key(self, text: str, emotion: str, voice_model: str) -> str:
        """Generate cache key for text/emotion/model combination"""
        content = f"{text}|{emotion}|{voice_model}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, text: str, emotion: str, voice_model: str) -> Optional[np.ndarray]:
        """Get cached audio"""
        try:
            cache_key = self._generate_cache_key(text, emotion, voice_model)
            
            if cache_key in self.cache_index:
                entry = self.cache_index[cache_key]
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                
                # Check if file exists and is not too old
                if cache_file.exists():
                    file_age = datetime.now() - datetime.fromisoformat(entry['created'])
                    if file_age.days <= self.max_age_days:
                        # Load cached audio
                        with open(cache_file, 'rb') as f:
                            audio_data = pickle.load(f)
                        
                        # Update access time
                        entry['last_accessed'] = datetime.now().isoformat()
                        self.stats['hits'] += 1
                        
                        logger.debug(f"Cache hit for: {text[:50]}...")
                        return audio_data
                    else:
                        # Remove expired entry
                        self._remove_cache_entry(cache_key)
            
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.stats['misses'] += 1
            return None
    
    def put(self, text: str, emotion: str, voice_model: str, audio_data: np.ndarray):
        """Store audio in cache"""
        try:
            cache_key = self._generate_cache_key(text, emotion, voice_model)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            # Save audio data
            with open(cache_file, 'wb') as f:
                pickle.dump(audio_data, f)
            
            # Update index
            self.cache_index[cache_key] = {
                'text': text[:100],  # Store first 100 chars for reference
                'emotion': emotion,
                'voice_model': voice_model,
                'created': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'file_size': cache_file.stat().st_size
            }
            
            self._save_cache_index()
            self._cleanup_if_needed()
            self._update_cache_stats()
            
            logger.debug(f"Cache stored for: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Cache put error: {e}")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove cache entry"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            if cache_key in self.cache_index:
                del self.cache_index[cache_key]
            
        except Exception as e:
            logger.error(f"Cache entry removal error: {e}")
    
    def _cleanup_if_needed(self):
        """Cleanup cache if size limit exceeded"""
        try:
            # Calculate current size
            total_size = sum(
                entry['file_size'] 
                for entry in self.cache_index.values()
            )
            
            if total_size > self.max_size_mb * 1024 * 1024:
                # Remove oldest entries
                sorted_entries = sorted(
                    self.cache_index.items(),
                    key=lambda x: x[1]['last_accessed']
                )
                
                removed_count = 0
                for cache_key, entry in sorted_entries:
                    self._remove_cache_entry(cache_key)
                    removed_count += 1
                    total_size -= entry['file_size']
                    
                    if total_size <= self.max_size_mb * 1024 * 1024 * 0.8:  # 80% of limit
                        break
                
                logger.info(f"Cache cleanup: removed {removed_count} entries")
                self._save_cache_index()
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    def _update_cache_stats(self):
        """Update cache statistics"""
        try:
            self.stats['entries'] = len(self.cache_index)
            self.stats['size_mb'] = sum(
                entry.get('file_size', 0) 
                for entry in self.cache_index.values()
            ) / (1024 * 1024)
        except Exception as e:
            logger.error(f"Cache stats update error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0.0
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'max_size_mb': self.max_size_mb,
            'max_age_days': self.max_age_days
        }

class EnterpriseBarkEngine:
    """
    🎯 ENTERPRISE BARK TTS ENGINE
    Enhanced Bark TTS with German optimization and enterprise features
    """
    
    def __init__(
        self, 
        config: Optional[Dict] = None,
        # ✅ NEUE: Direkte Parameter
        voice_preset: Optional[str] = None,
        enable_emotion_modulation: Optional[bool] = None,
        enable_speed_control: Optional[bool] = None,
        enable_audio_enhancement: Optional[bool] = None,
        cache_enabled: Optional[bool] = None,
        cache_dir: Optional[Path] = None,
        model_cache_dir: Optional[Path] = None
    ):
        # Merge direct parameters with config
        self.config = config or {}
        
        # Override config with direct parameters if provided
        if voice_preset is not None:
            self.config['voice_preset'] = voice_preset
        if enable_emotion_modulation is not None:
            self.config['enable_emotion_modulation'] = enable_emotion_modulation
        if enable_speed_control is not None:
            self.config['enable_speed_control'] = enable_speed_control
        if enable_audio_enhancement is not None:
            self.config['enable_audio_enhancement'] = enable_audio_enhancement
        if cache_enabled is not None:
            self.config['cache_enabled'] = cache_enabled
        if cache_dir is not None:
            self.config['cache_dir'] = cache_dir
        if model_cache_dir is not None:
            self.config['model_cache_dir'] = model_cache_dir
        
        self.device = self._get_device()
        
        # Voice settings (now from merged config)
        self.voice_preset = self.config.get('voice_preset', 'v2/en_speaker_6')
        self.enable_emotion_modulation = self.config.get('enable_emotion_modulation', True)
        self.enable_speed_control = self.config.get('enable_speed_control', True)
        self.enable_audio_enhancement = self.config.get('enable_audio_enhancement', True)
        self.cache_enabled = self.config.get('cache_enabled', True)
        
        # Initialize components
        self.phonetic_processor = GermanPhoneticProcessor()
        self.emotion_modulator = EmotionModulator()
        
        # Voice presets
        self.voice_presets = self._initialize_voice_presets()
        
        # ✅ FIX: Cache initialization with proper fallback
        if self.cache_enabled:
            cache_dir_config = self.config.get('cache_dir')
            if cache_dir_config is None:
                cache_dir_config = Path("cache/voice/bark")  # Default fallback
            else:
                cache_dir_config = Path(cache_dir_config)
            
            try:
                self.voice_cache = VoiceCache(cache_dir_config)
                logger.info(f"✅ Voice cache initialized at: {cache_dir_config}")
            except Exception as e:
                logger.error(f"Voice cache initialization failed: {e}")
                # Fallback: disable caching
                self.voice_cache = None
                self.cache_enabled = False
                logger.warning("🚫 Voice caching disabled due to initialization error")
        else:
            self.voice_cache = None
        
        # Bark initialization
        self.is_initialized = False
        self._lock = threading.Lock()
        
        # Performance tracking
        self.synthesis_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0,
            'total_audio_duration': 0.0,
            'cache_hits': 0
        }
        
        logger.info("🎯 Enterprise Bark Engine initialized")
    
    def _get_device(self) -> str:
        """Get optimal device for processing"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps" 
        else:
            return "cpu"
    
    def _initialize_voice_presets(self) -> Dict[str, VoicePreset]:
        """Initialize available voice presets"""
        return {
            "v2/en_speaker_6": VoicePreset(
                name="English Female Professional",
                speaker_id="v2/en_speaker_6",
                emotion_weights={
                    "neutral": 1.0,
                    "happy": 0.8,
                    "sad": 0.6,
                    "angry": 0.4,
                    "excited": 0.9
                }
            ),
            "v2/en_speaker_9": VoicePreset(
                name="English Female Friendly",
                speaker_id="v2/en_speaker_9", 
                emotion_weights={
                    "neutral": 1.0,
                    "happy": 1.2,
                    "cheerful": 1.1,
                    "calm": 0.9
                }
            )
        }
    
    def _get_valid_bark_preset(self, preset: str) -> Optional[str]:
        """Get valid Bark history prompt or fallback"""
        # ✅ ECHTE Bark History Prompts (English speakers work better)
        valid_bark_presets = {
            "v2/en_speaker_6": "v2/en_speaker_6",  
            "v2/en_speaker_9": "v2/en_speaker_9",  
            "v2/en_speaker_4": "v2/en_speaker_4",
            "v2/en_speaker_2": "v2/en_speaker_2",
            "v2/en_speaker_1": "v2/en_speaker_1",
            "v2/en_speaker_0": "v2/en_speaker_0",
            "german_female": "v2/en_speaker_6",    # Fallback zu English
            "default": None  # Kein History Prompt = Default Voice
        }
        
        return valid_bark_presets.get(preset, None)
    
    def initialize(self) -> bool:
        """Initialize the Bark engine"""
        try:
            with self._lock:
                if self.is_initialized:
                    return True
                
                logger.info("🚀 Initializing Enterprise Bark Engine...")
                
                if not BARK_AVAILABLE:
                    logger.error("Bark is not available - install with: pip install bark-voice")
                    return False
                
                # Setup environment
                self._setup_model_environment()
                
                # Preload models
                logger.info("🔄 Preloading Bark models...")
                start_time = time.time()
                
                preload_models()
                
                load_time = time.time() - start_time
                logger.info(f"✅ Bark models preloaded in {load_time:.1f}s")
                
                self.is_initialized = True
                
                # Run test
                self._run_initialization_test()
                
                logger.info("✅ Enterprise Bark Engine initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Enterprise Bark initialization failed: {e}")
            return False
    
    def _setup_model_environment(self):
        """Setup Bark model environment"""
        try:
            import os
            
            # Set model cache directory if specified
            model_cache_dir = self.config.get('model_cache_dir')
            if model_cache_dir:
                os.environ['BARK_CACHE_DIR'] = str(model_cache_dir)
            
            # Configure device usage
            if self.device == 'cpu':
                os.environ['BARK_FORCE_CPU'] = 'True'
            else:
                os.environ['BARK_FORCE_CPU'] = 'False'
            
            logger.info(f"✅ Bark environment configured for device: {self.device}")
            
        except Exception as e:
            logger.error(f"Bark environment setup failed: {e}")
    
    def _run_initialization_test(self):
        """Run initialization test synthesis"""
        try:
            test_text = "Hallo, ich bin Kira."
            result = self.synthesize_speech(test_text, emotion="neutral")
            
            if result.success:
                logger.info("✅ Bark initialization test successful")
            else:
                logger.warning(f"Bark initialization test failed: {result.error}")
                
        except Exception as e:
            logger.warning(f"Bark initialization test error: {e}")
    
    def synthesize_speech(
    self, 
    text: str, 
    voice_preset: str = None,
    emotion: str = "neutral",
    use_cache: bool = True
) -> SynthesisResult:
        """
        Synthesize speech with enterprise features - FIXED PRESET HANDLING
        """
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return SynthesisResult(
                        audio_data=None,
                        sample_rate=SAMPLE_RATE,
                        duration=0.0,
                        text=text,
                        voice_preset=voice_preset or self.voice_preset,
                        processing_time=time.time() - start_time,
                        success=False,
                        error="Engine not initialized"
                    )
            
            # Use provided voice preset or default
            preset = voice_preset or self.voice_preset
            
            # ✅ FIX: Get valid Bark preset
            bark_preset = self._get_valid_bark_preset(preset)
            
            # Check cache first
            if use_cache and self.voice_cache and self.cache_enabled:
                cached_audio = self.voice_cache.get(text, emotion, preset)
                if cached_audio is not None:
                    self.synthesis_stats['cache_hits'] += 1
                    return SynthesisResult(
                        audio_data=cached_audio,
                        sample_rate=SAMPLE_RATE,
                        duration=len(cached_audio) / SAMPLE_RATE,
                        text=text,
                        voice_preset=preset,
                        processing_time=time.time() - start_time,
                        success=True,
                        cache_hit=True
                    )
            
            # Process text
            processed_text = self._prepare_text_for_synthesis(text, emotion)
            
            # ✅ Generate audio with VALID Bark preset
            synthesis_start = time.time()
            
            if bark_preset:
                logger.info(f"🗣️ Using Bark history prompt: {bark_preset}")
                audio_array = generate_audio(
                    processed_text,
                    history_prompt=bark_preset,
                    text_temp=0.7,
                    waveform_temp=0.7
                )
            else:
                logger.info("🗣️ Using default Bark voice (no history prompt)")
                audio_array = generate_audio(
                    processed_text,
                    text_temp=0.7,
                    waveform_temp=0.7
                )
            
            synthesis_time = time.time() - synthesis_start
            
            # Post-process audio
            if self.enable_audio_enhancement:
                audio_array = self._enhance_audio(audio_array)
            
            # Store in cache
            if use_cache and self.voice_cache and self.cache_enabled:
                self.voice_cache.put(text, emotion, preset, audio_array)
            
            # Calculate metrics
            duration = len(audio_array) / SAMPLE_RATE
            quality_score = self._calculate_quality_score(audio_array, text)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(True, processing_time, duration)
            
            logger.info(f"🗣️ Speech synthesized: '{text[:50]}...' ({emotion}) in {synthesis_time:.2f}s")
            
            return SynthesisResult(
                audio_data=audio_array,
                sample_rate=SAMPLE_RATE,
                duration=duration,
                text=text,
                voice_preset=preset,
                processing_time=processing_time,
                success=True,
                model_used="bark_german_female",
                quality_score=quality_score
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(False, processing_time, 0.0)
            
            logger.error(f"Speech synthesis failed: {e}")
            return SynthesisResult(
                audio_data=None,
                sample_rate=SAMPLE_RATE,
                duration=0.0,
                text=text,
                voice_preset=voice_preset or self.voice_preset,
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def _prepare_text_for_synthesis(self, text: str, emotion: str) -> str:
        """Prepare text for synthesis with German optimization"""
        # German phonetic processing
        processed_text = self.phonetic_processor.process_text(text)
        
        # Emotion modulation
        if self.enable_emotion_modulation:
            processed_text = self.emotion_modulator.apply_emotion(processed_text, emotion)
        
        return processed_text
    
    def _enhance_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply audio enhancements"""
        try:
            enhanced_audio = audio_array.copy()
            
            # Basic audio normalization
            if np.max(np.abs(enhanced_audio)) > 0:
                enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.8
            
            # Remove DC component
            if len(enhanced_audio) > 100:
                enhanced_audio = enhanced_audio - np.mean(enhanced_audio)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Audio enhancement error: {e}")
            return audio_array
    
    def _calculate_quality_score(self, audio_array: np.ndarray, text: str) -> float:
        """Calculate audio quality score"""
        try:
            # Signal strength
            rms = np.sqrt(np.mean(audio_array ** 2))
            signal_strength = min(1.0, rms * 10)
            
            # Dynamic range
            dynamic_range = (np.max(audio_array) - np.min(audio_array)) / 2.0
            dynamic_score = min(1.0, dynamic_range)
            
            # Length appropriateness
            expected_duration = len(text) * 0.15
            actual_duration = len(audio_array) / SAMPLE_RATE
            length_ratio = min(actual_duration, expected_duration) / max(actual_duration, expected_duration)
            
            # Combined score
            quality_score = (signal_strength * 0.4 + dynamic_score * 0.3 + length_ratio * 0.3)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality score calculation error: {e}")
            return 0.5
    
    def _update_stats(self, success: bool, processing_time: float, duration: float):
        """Update synthesis statistics"""
        self.synthesis_stats['total_requests'] += 1
        
        if success:
            self.synthesis_stats['successful_requests'] += 1
            self.synthesis_stats['total_audio_duration'] += duration
        else:
            self.synthesis_stats['failed_requests'] += 1
        
        # Update average processing time
        total = self.synthesis_stats['total_requests']
        current_avg = self.synthesis_stats['average_processing_time']
        self.synthesis_stats['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voice presets"""
        return list(self.voice_presets.keys())
    
    def get_voice_info(self, voice_preset: str) -> Optional[VoicePreset]:
        """Get information about a voice preset"""
        return self.voice_presets.get(voice_preset)
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics"""
        stats = self.synthesis_stats.copy()
        
        if self.voice_cache:
            stats['cache'] = self.voice_cache.get_stats()
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            'initialized': self.is_initialized,
            'bark_available': BARK_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'device': self.device,
            'voice_preset': self.voice_preset,
            'features': {
                'emotion_modulation': self.enable_emotion_modulation,
                'speed_control': self.enable_speed_control,
                'audio_enhancement': self.enable_audio_enhancement,
                'caching': self.cache_enabled
            },
            'available_emotions': list(self.emotion_modulator.emotion_configs.keys()),
            'statistics': self.get_synthesis_stats()
        }
    
    def synthesize(self, text: str, emotion: str = "neutral") -> Dict[str, Any]:
        """
        Synthesize method expected by voice manager - converts SynthesisResult to dict
        
        Args:
            text: Text to synthesize  
            emotion: Emotion for the speech
            
        Returns:
            Dictionary with success status and audio info compatible with voice manager
        """
        try:
            result = self.synthesize_speech(text, emotion=emotion)
            
            if result.success:
                # Save audio data to file
                timestamp = int(time.time() * 1000)
                filename = f"enterprise_bark_{timestamp}_{emotion}.wav"
                output_dir = Path("voice/output")
                output_dir.mkdir(parents=True, exist_ok=True)
                audio_path = output_dir / filename
                
                # Save audio file
                from scipy.io.wavfile import write
                audio_array = result.audio_data.astype(np.float32)
                audio_array = audio_array / np.max(np.abs(audio_array))  # Normalize
                write(str(audio_path), result.sample_rate, audio_array)
                
                # Generate URL for frontend access
                audio_url = f"/api/audio/{filename}"
                
                return {
                    'success': True,
                    'audio_path': str(audio_path),
                    'audio_url': audio_url,
                    'filename': filename,
                    'text': text,
                    'emotion': emotion,
                    'duration_estimate': result.duration,
                    'processing_time': result.processing_time,
                    'quality_score': result.quality_score
                }
            else:
                return {
                    'success': False,
                    'error': result.error or 'Speech synthesis failed'
                }
                
        except Exception as e:
            logger.error(f"❌ Enterprise Bark synthesize error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def speak(self, text: str, emotion: str = "neutral", auto_play: bool = True) -> Optional[str]:
        """
        Speak method for backwards compatibility - returns audio file path
        
        Args:
            text: Text to speak
            emotion: Emotion for speech synthesis  
            auto_play: Whether to auto-play (not implemented in enterprise version)
            
        Returns:
            Path to generated audio file as string, or None if failed
        """
        try:
            result = self.synthesize(text, emotion)
            
            if result.get('success'):
                return result.get('audio_path')
            else:
                logger.warning(f"❌ Enterprise Bark speak failed: {result.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Enterprise Bark speak error: {e}")
            return None

    def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("🧹 Cleaning up Enterprise Bark Engine...")
            # Cleanup model resources here
            self.is_initialized = False
            logger.info("✅ Enterprise Bark Engine cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Export classes
__all__ = [
    'EnterpriseBarkEngine',
    'SynthesisResult',
    'VoiceProfile',
    'VoicePreset',
    'EmotionModulator',
    'VoiceCache'
]