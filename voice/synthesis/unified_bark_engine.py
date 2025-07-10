"""
🗣️ UNIFIED BARK TTS ENGINE
Kombination aus Smart Hardware Detection + Enterprise Features
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import os
import platform
import psutil
from pathlib import Path
import time

# Import Enterprise components
from voice.synthesis.enterprise_bark import (
    GermanPhoneticProcessor,
    EmotionModulator,
    VoiceCache,
    SynthesisResult,
    VoiceProfile
)

# Import German Voice support
from voice.synthesis.german_female_voice import (
    GermanFemaleVoiceManager,
    get_kira_voice_config,
    GERMAN_FEMALE_VOICES
)

logger = logging.getLogger(__name__)

class UnifiedBarkTTSEngine:
    """
    🧠 UNIFIED BARK TTS ENGINE
    Smart Hardware Detection + Enterprise German Voice Features
    """
    
    def __init__(
        self,
        voice_preset: Optional[str] = None,
        output_dir: str = "voice/output",
        enable_emotion_modulation: bool = True,
        enable_caching: bool = True,
        cache_dir: Optional[Path] = None
    ):
        # Initialize paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 🧠 SMART HARDWARE DETECTION (from bark_engine.py)
        self.device_info = self._detect_optimal_device()
        self.hardware_profile = self._analyze_hardware_capabilities()
        
        # 🇩🇪 GERMAN VOICE MANAGEMENT (from enterprise_bark.py)
        self.german_voice_manager = GermanFemaleVoiceManager()
        self.phonetic_processor = GermanPhoneticProcessor()
        self.emotion_modulator = EmotionModulator()
        
        # Voice Selection - combine both approaches
        if voice_preset is None:
            # Use hardware-optimized selection from bark_engine.py
            self.voice_preset = self._select_optimal_voice()
        elif voice_preset in GERMAN_FEMALE_VOICES:
            # Use German voice mapping from enterprise_bark.py
            self.voice_preset = GERMAN_FEMALE_VOICES[voice_preset]["speaker_id"]
        else:
            self.voice_preset = voice_preset
        
        # 🔧 SMART MODEL CONFIG (from bark_engine.py)
        self.model_config = self._get_optimal_model_config()
        
        # 🗄️ ENTERPRISE CACHING (from enterprise_bark.py)
        if enable_caching:
            cache_dir = cache_dir or Path("cache/voice/bark")
            self.voice_cache = VoiceCache(cache_dir)
        else:
            self.voice_cache = None
        
        # Engine State
        self.is_initialized = False
        self.system = platform.system()
        
        # Bark Module (loaded on demand)
        self.bark_generate_audio = None
        self.bark_sample_rate = None
        self.bark_preload_models = None
        
        # 📊 ENHANCED STATISTICS
        self.stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'cache_hits': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'total_audio_duration': 0.0,
            'last_generation_time': 0.0,
            'hardware_tier': self.hardware_profile['tier'],
            'device_type': self.device_info['type']
        }
        
        logger.info(f"🧠 Unified Bark TTS Engine initialized")
        logger.info(f"🖥️ Device: {self.device_info['device']} ({self.device_info['type']})")
        logger.info(f"⚡ Performance: {self.hardware_profile['tier']}")
        logger.info(f"🗣️ Voice: {self.voice_preset}")
        logger.info(f"🇩🇪 German Voice Features: {enable_emotion_modulation}")
    
    # 🖥️ HARDWARE DETECTION (from bark_engine.py)
    def _detect_optimal_device(self) -> Dict[str, Any]:
        """Automatische GPU/CPU Erkennung - Zukunftssicher"""
        device_info = {
            'device': 'cpu',
            'type': 'CPU',
            'available_devices': [],
            'cuda_available': False,
            'mps_available': False,
            'gpu_memory_gb': 0,
            'gpu_name': None
        }
        
        try:
            import torch
            
            # CUDA Detection
            if torch.cuda.is_available():
                device_info['cuda_available'] = True
                device_info['available_devices'].append('cuda')
                
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    device_info['gpu_name'] = torch.cuda.get_device_name(0)
                    device_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    if device_info['gpu_memory_gb'] >= 6:
                        device_info['device'] = 'cuda'
                        device_info['type'] = f"NVIDIA GPU ({device_info['gpu_memory_gb']:.1f}GB)"
                        logger.info(f"✅ CUDA GPU erkannt: {device_info['gpu_name']}")
            
            # MPS Detection
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_info['mps_available'] = True
                device_info['available_devices'].append('mps')
                device_info['device'] = 'mps'
                device_info['type'] = 'Apple Silicon GPU'
                logger.info("✅ Apple Silicon MPS erkannt")
            
            if device_info['device'] == 'cpu':
                logger.info("ℹ️ Verwende CPU (kein geeignetes GPU erkannt)")
                
        except ImportError:
            logger.warning("⚠️ PyTorch nicht verfügbar - verwende CPU")
        except Exception as e:
            logger.error(f"❌ Device Detection Fehler: {e}")
        
        return device_info
    
    def _analyze_hardware_capabilities(self) -> Dict[str, Any]:
        """Hardware-Analyse mit German Voice Optimization"""
        try:
            # Basic hardware info
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            ram_gb = round(psutil.virtual_memory().total / (1024**3))
            
            # Performance scoring
            performance_score = 0
            recommended_model_size = "small"
            
            if self.device_info['device'] == 'cuda':
                gpu_memory = self.device_info.get('gpu_memory_gb', 0)
                if gpu_memory >= 12:
                    performance_score += 100
                    recommended_model_size = "standard"
                elif gpu_memory >= 8:
                    performance_score += 80
                    recommended_model_size = "small"
                else:
                    performance_score += 60
                    recommended_model_size = "small"
            elif self.device_info['device'] == 'mps':
                performance_score += 90
                recommended_model_size = "small"  # Apple Silicon efficient with small models
            else:
                performance_score += 30
                recommended_model_size = "small"  # CPU always small
            
            # Thread/Memory bonus
            if cpu_threads >= 16: performance_score += 15
            elif cpu_threads >= 12: performance_score += 10
            elif cpu_threads >= 8: performance_score += 5
            
            if ram_gb >= 32: performance_score += 15
            elif ram_gb >= 16: performance_score += 10
            elif ram_gb >= 8: performance_score += 5
            
            # Determine tier
            if performance_score >= 120:
                tier = "HIGH"
            elif performance_score >= 90:
                tier = "MEDIUM"
            elif performance_score >= 60:
                tier = "LOW"
            else:
                tier = "BASIC"
            
            return {
                'cpu_cores': cpu_cores,
                'cpu_threads': cpu_threads,
                'ram_gb': ram_gb,
                'performance_score': performance_score,
                'tier': tier,
                'recommended_model_size': recommended_model_size,
                'estimated_performance': self._estimate_performance(tier, recommended_model_size)
            }
            
        except Exception as e:
            logger.error(f"❌ Hardware Analysis Fehler: {e}")
            return {
                'cpu_cores': 4, 'cpu_threads': 8, 'ram_gb': 8,
                'performance_score': 50, 'tier': 'BASIC',
                'recommended_model_size': 'small',
                'estimated_performance': '20-60s per sentence'
            }
    
    def _estimate_performance(self, tier: str, model_size: str) -> str:
        """Performance Estimation mit German Voice berücksichtigt"""
        if model_size == "small":
            estimates = {
                'HIGH': '2-8s per sentence',
                'MEDIUM': '5-15s per sentence',
                'LOW': '10-30s per sentence',
                'BASIC': '20-60s per sentence'
            }
        else:
            estimates = {
                'HIGH': '5-15s per sentence',
                'MEDIUM': '10-30s per sentence',
                'LOW': '20-60s per sentence',
                'BASIC': '60-120s per sentence'
            }
        return estimates.get(tier, '15-45s per sentence')
    
    def _select_optimal_voice(self) -> str:
        """Smart Voice Selection für German Female"""
        tier = self.hardware_profile['tier']
        
        # Hardware-optimierte deutsche Stimmenauswahl
        voice_selection = {
            'HIGH': 'v2/de/speaker_6',     # Kira - beste Qualität
            'MEDIUM': 'v2/de/speaker_4',   # Elegant - gute Balance
            'LOW': 'v2/de/speaker_2',      # Professional - speed optimiert
            'BASIC': 'v2/de/speaker_1'     # Speed optimiert
        }
        
        selected = voice_selection.get(tier, 'v2/de/speaker_6')
        logger.info(f"🎯 Voice Selection für {tier}: {selected}")
        return selected
    
    def _get_optimal_model_config(self) -> Dict[str, Any]:
        """Hardware-optimierte Model Config"""
        tier = self.hardware_profile['tier']
        device = self.device_info['device']
        
        configs = {
            'HIGH': {
                'text_temp': 0.7, 'waveform_temp': 0.7,
                'chunk_size': 150, 'preload_models': True
            },
            'MEDIUM': {
                'text_temp': 0.6, 'waveform_temp': 0.6,
                'chunk_size': 100, 'preload_models': False
            },
            'LOW': {
                'text_temp': 0.5, 'waveform_temp': 0.5,
                'chunk_size': 75, 'preload_models': False
            },
            'BASIC': {
                'text_temp': 0.4, 'waveform_temp': 0.4,
                'chunk_size': 50, 'preload_models': False
            }
        }
        
        return configs.get(tier, configs['BASIC'])
    
    # 🚀 INITIALIZATION
    def initialize(self) -> bool:
        """Unified Bark Initialization"""
        try:
            logger.info("🚀 Initializing Unified Bark TTS...")
            
            # Setup environment
            self._setup_optimal_environment()
            
            # Import Bark
            try:
                from bark import SAMPLE_RATE, generate_audio, preload_models
                
                self.bark_generate_audio = generate_audio
                self.bark_sample_rate = SAMPLE_RATE
                self.bark_preload_models = preload_models
                
                logger.info("✅ Bark Module imported successfully")
                
            except ImportError as e:
                logger.error(f"❌ Bark Import Error: {e}")
                return False
            
            # Model preloading based on hardware
            if self.model_config.get('preload_models', False):
                logger.info("🧠 Preloading Bark models...")
                self.bark_preload_models()
            
            # Test with German voice
            if self._test_german_voice():
                self.is_initialized = True
                logger.info("🎉 Unified Bark TTS initialized successfully!")
                self._log_initialization_summary()
                return True
            else:
                logger.error("❌ German voice test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Unified Bark initialization failed: {e}")
            return False
    
    def _setup_optimal_environment(self):
        """Setup optimal environment for German voice synthesis"""
        device = self.device_info['device']
        
        # Small model detection for German voices
        if self._should_use_small_models():
            os.environ['SUNO_USE_SMALL_MODELS'] = 'True'
            logger.info("📦 Using small Bark models for German voice")
        
        if device == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        elif device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['FORCE_CPU'] = '1'
            os.environ['SUNO_USE_SMALL_MODELS'] = 'True'
            
            # CPU optimization
            thread_count = str(self.hardware_profile['cpu_threads'])
            os.environ['OMP_NUM_THREADS'] = thread_count
            os.environ['TORCH_NUM_THREADS'] = thread_count
        
        os.environ['PYTORCH_JIT'] = '0'
    
    def _should_use_small_models(self) -> bool:
        """Decide whether to use small models"""
        device = self.device_info['device']
        tier = self.hardware_profile['tier']
        
        if device == 'cpu':
            return True
        elif device == 'cuda':
            gpu_memory = self.device_info.get('gpu_memory_gb', 0)
            return gpu_memory < 12 or tier in ['LOW', 'BASIC']
        elif device == 'mps':
            ram_gb = self.hardware_profile.get('ram_gb', 16)
            return ram_gb < 24 or tier in ['MEDIUM', 'LOW', 'BASIC']
        
        return tier in ['LOW', 'BASIC']
    
    def _test_german_voice(self) -> bool:
        """Test German voice synthesis"""
        try:
            test_text = "Hallo, ich bin Kira."
            result = self.synthesize_german_female(test_text, emotion="neutral")
            
            if result.success:
                logger.info("✅ German voice test successful")
                return True
            else:
                logger.error(f"❌ German voice test failed: {result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"❌ German voice test error: {e}")
            return False
    
    # 🗣️ MAIN SYNTHESIS METHOD
    def synthesize_german_female(
        self,
        text: str,
        emotion: str = "neutral",
        speed_factor: float = 1.0,
        use_cache: bool = True
    ) -> SynthesisResult:
        """
        🇩🇪 GERMAN FEMALE VOICE SYNTHESIS
        Combines Smart Hardware Optimization + Enterprise Features
        """
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                return SynthesisResult(
                    success=False,
                    audio_data=None,
                    sample_rate=self.bark_sample_rate or 24000,
                    duration=0.0,
                    text=text,
                    emotion=emotion,
                    processing_time=time.time() - start_time,
                    model_used="not_initialized",
                    error_message="Unified Bark not initialized"
                )
            
            # 🗄️ CHECK CACHE (Enterprise feature)
            if use_cache and self.voice_cache:
                cached_audio = self.voice_cache.get(text, emotion, self.voice_preset)
                if cached_audio is not None:
                    self.stats['cache_hits'] += 1
                    return SynthesisResult(
                        success=True,
                        audio_data=cached_audio,
                        sample_rate=self.bark_sample_rate,
                        duration=len(cached_audio) / self.bark_sample_rate,
                        text=text,
                        emotion=emotion,
                        processing_time=time.time() - start_time,
                        model_used="cached",
                        cache_hit=True
                    )
            
            # 🇩🇪 GERMAN TEXT PROCESSING (Enterprise feature)
            processed_text = self._prepare_german_text(text, emotion)
            
            # 🧠 SMART CHUNKING (Hardware-based from bark_engine.py)
            if self._should_chunk_text(text):
                return self._synthesize_chunked(text, emotion, speed_factor, use_cache)
            
            # 🎵 BARK SYNTHESIS
            synthesis_start = time.time()
            audio_array = self.bark_generate_audio(
                processed_text,
                history_prompt=self.voice_preset,
                text_temp=self.model_config['text_temp'],
                waveform_temp=self.model_config['waveform_temp'],
                silent=True
            )
            synthesis_time = time.time() - synthesis_start
            
            # 🔧 AUDIO ENHANCEMENT
            if speed_factor != 1.0:
                audio_array = self._apply_speed_control(audio_array, speed_factor)
            
            # 🗄️ CACHE STORAGE
            if use_cache and self.voice_cache:
                self.voice_cache.put(text, emotion, self.voice_preset, audio_array)
            
            # 📊 METRICS
            duration = len(audio_array) / self.bark_sample_rate
            quality_score = self._calculate_quality_score(audio_array, text)
            
            # Update statistics
            self._update_stats(time.time() - start_time, duration, True)
            
            logger.info(f"🗣️ German synthesis: '{text[:50]}...' ({emotion})")
            logger.info(f"   Duration: {duration:.2f}s, Processing: {synthesis_time:.2f}s")
            
            return SynthesisResult(
                success=True,
                audio_data=audio_array,
                sample_rate=self.bark_sample_rate,
                duration=duration,
                text=text,
                emotion=emotion,
                processing_time=time.time() - start_time,
                model_used="unified_bark_german",
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"❌ German female synthesis failed: {e}")
            self._update_stats(time.time() - start_time, 0.0, False)
            
            return SynthesisResult(
                success=False,
                audio_data=None,
                sample_rate=self.bark_sample_rate or 24000,
                duration=0.0,
                text=text,
                emotion=emotion,
                processing_time=time.time() - start_time,
                model_used="error",
                error_message=str(e)
            )
    
    def _prepare_german_text(self, text: str, emotion: str) -> str:
        """Prepare text with German processing + emotion"""
        # German phonetic processing
        processed_text = self.phonetic_processor.process_text_for_german_tts(text, emotion)
        
        # Emotion modulation
        processed_text = self.emotion_modulator.apply_emotion(processed_text, emotion)
        
        return processed_text
    
    def _should_chunk_text(self, text: str) -> bool:
        """Hardware-based chunking decision"""
        chunk_size = self.model_config['chunk_size']
        tier = self.hardware_profile['tier']
        
        if tier in ['BASIC', 'LOW']:
            return len(text) > chunk_size
        elif tier == 'MEDIUM':
            return len(text) > chunk_size * 1.5
        else:
            return len(text) > chunk_size * 2
    
    def _synthesize_chunked(self, text: str, emotion: str, speed_factor: float, use_cache: bool) -> SynthesisResult:
        """Chunked synthesis for long texts"""
        chunk_size = self.model_config['chunk_size']
        chunks = self._split_text_intelligently(text, chunk_size)
        
        audio_chunks = []
        total_processing_time = 0
        
        for chunk in chunks:
            result = self.synthesize_german_female(chunk, emotion, speed_factor, use_cache)
            if result.success:
                audio_chunks.append(result.audio_data)
                total_processing_time += result.processing_time
            else:
                # If a chunk fails, return the error
                return result
        
        # Combine audio chunks
        if audio_chunks:
            combined_audio = np.concatenate(audio_chunks)
            duration = len(combined_audio) / self.bark_sample_rate
            
            return SynthesisResult(
                success=True,
                audio_data=combined_audio,
                sample_rate=self.bark_sample_rate,
                duration=duration,
                text=text,
                emotion=emotion,
                processing_time=total_processing_time,
                model_used="unified_bark_chunked",
                quality_score=0.8  # Slightly lower for chunked
            )
        else:
            return SynthesisResult(
                success=False,
                audio_data=None,
                sample_rate=self.bark_sample_rate,
                duration=0.0,
                text=text,
                emotion=emotion,
                processing_time=total_processing_time,
                model_used="chunked_failed",
                error_message="All chunks failed"
            )
    
    def _split_text_intelligently(self, text: str, chunk_size: int) -> List[str]:
        """Split text at natural boundaries"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        remaining = text
        
        while len(remaining) > chunk_size:
            # Find best split point
            split_point = chunk_size
            
            # Look for sentence endings
            for i in range(min(chunk_size, len(remaining)), chunk_size // 2, -1):
                if remaining[i] in '.!?':
                    split_point = i + 1
                    break
            else:
                # Look for comma or space
                for i in range(min(chunk_size, len(remaining)), chunk_size // 2, -1):
                    if remaining[i] in ', ':
                        split_point = i
                        break
            
            chunk = remaining[:split_point].strip()
            if chunk:
                chunks.append(chunk)
            
            remaining = remaining[split_point:].strip()
        
        if remaining:
            chunks.append(remaining)
        
        return chunks
    
    def _apply_speed_control(self, audio_array: np.ndarray, speed_factor: float) -> np.ndarray:
        """Apply speed control to audio"""
        try:
            if speed_factor == 1.0:
                return audio_array
            
            # Simple resampling approach
            original_length = len(audio_array)
            new_length = int(original_length / speed_factor)
            
            # Linear interpolation for speed change
            indices = np.linspace(0, original_length - 1, new_length)
            modified_audio = np.interp(indices, np.arange(original_length), audio_array)
            
            return modified_audio
            
        except Exception as e:
            logger.error(f"Speed control error: {e}")
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
            actual_duration = len(audio_array) / self.bark_sample_rate
            length_ratio = min(actual_duration, expected_duration) / max(actual_duration, expected_duration)
            
            return (signal_strength * 0.4 + dynamic_score * 0.3 + length_ratio * 0.3)
            
        except:
            return 0.5
    
    def _update_stats(self, processing_time: float, duration: float, success: bool):
        """Update synthesis statistics"""
        self.stats['total_generations'] += 1
        if success:
            self.stats['successful_generations'] += 1
            self.stats['total_audio_duration'] += duration
        
        # Update average
        total_time = self.stats['average_time'] * (self.stats['total_generations'] - 1)
        self.stats['average_time'] = (total_time + processing_time) / self.stats['total_generations']
        self.stats['last_generation_time'] = time.time()
    
    def _log_initialization_summary(self):
        """Log initialization summary"""
        logger.info("=" * 60)
        logger.info("🎉 UNIFIED BARK TTS INITIALIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"🖥️ Device: {self.device_info['type']}")
        logger.info(f"⚡ Performance: {self.hardware_profile['tier']}")
        logger.info(f"🗣️ German Voice: {self.voice_preset}")
        logger.info(f"⏱️ Expected: {self.hardware_profile['estimated_performance']}")
        logger.info(f"🇩🇪 Features: Phonetics + Emotions + Caching")
        logger.info("=" * 60)
    
    # 📊 PUBLIC API
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        status = {
            'initialized': self.is_initialized,
            'device_info': self.device_info,
            'hardware_profile': self.hardware_profile,
            'voice_preset': self.voice_preset,
            'model_config': self.model_config,
            'features': {
                'german_phonetics': True,
                'emotion_modulation': True,
                'smart_chunking': True,
                'caching': self.voice_cache is not None,
                'speed_control': True
            },
            'available_emotions': self.emotion_modulator.get_available_emotions() if hasattr(self.emotion_modulator, 'get_available_emotions') else list(self.emotion_modulator.emotion_configs.keys()),
            'statistics': self.stats
        }
        
        if self.voice_cache:
            status['cache_stats'] = self.voice_cache.get_stats()
        
        return status
    
    def get_available_voices(self) -> List[str]:
        """Get available German voices"""
        return list(GERMAN_FEMALE_VOICES.keys())
    
    def set_voice(self, voice_name: str) -> bool:
        """Set German voice"""
        if voice_name in GERMAN_FEMALE_VOICES:
            old_voice = self.voice_preset
            self.voice_preset = GERMAN_FEMALE_VOICES[voice_name]["speaker_id"]
            logger.info(f"🗣️ Voice changed: {old_voice} → {self.voice_preset}")
            return True
        else:
            logger.warning(f"⚠️ Unknown German voice: {voice_name}")
            return False
    
    def cleanup(self):
        """Cleanup unified engine"""
        try:
            if self.voice_cache:
                logger.info("🧹 Voice cache cleanup...")
            
            self.bark_generate_audio = None
            self.bark_sample_rate = None
            self.bark_preload_models = None
            self.is_initialized = False
            
            logger.info("✅ Unified Bark TTS cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# 🔄 BACKWARD COMPATIBILITY
SmartBarkTTSEngine = UnifiedBarkTTSEngine
EnterpriseBarkEngine = UnifiedBarkTTSEngine

# Export
__all__ = ['UnifiedBarkTTSEngine', 'SmartBarkTTSEngine', 'EnterpriseBarkEngine']