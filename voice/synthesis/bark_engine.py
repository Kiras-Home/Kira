"""
Bark Text-to-Speech Engine - SMART AUTO-DETECTION
Automatische GPU/CPU Erkennung + Hardware-adaptive Optimierung
Zukunftssicher für System-Upgrades ohne Code-Änderungen
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import os
import subprocess
import platform
import psutil
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class SmartBarkTTSEngine:
    """Smart Bark TTS Engine mit automatischer Hardware-Erkennung"""
    
    def __init__(self, voice_preset: Optional[str] = None, output_dir: str = "voice/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Hardware Detection
        self.device_info = self._detect_optimal_device()
        self.hardware_profile = self._analyze_hardware_capabilities()
        
        # Smart Voice Selection
        self.voice_preset = voice_preset or self._select_optimal_voice()
        self.model_config = self._get_optimal_model_config()
        
        # Engine State
        self.is_initialized = False
        self.system = platform.system()
        
        # Bark Module (loaded on demand)
        self.bark_generate_audio = None
        self.bark_sample_rate = None
        self.bark_preload_models = None
        
        # Performance Tracking
        self.generation_stats = {
            'total_generations': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'last_generation_time': 0.0
        }
        
        logger.info(f"🧠 Smart Bark TTS Engine initialized")
        logger.info(f"🖥️ Device: {self.device_info['device']} ({self.device_info['type']})")
        logger.info(f"⚡ Performance: {self.hardware_profile['tier']}")
        logger.info(f"🗣️ Voice: {self.voice_preset}")
    
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
            # PyTorch Detection
            import torch
            
            # CUDA (NVIDIA) Detection
            if torch.cuda.is_available():
                device_info['cuda_available'] = True
                device_info['available_devices'].append('cuda')
                
                # GPU Details
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    device_info['gpu_name'] = torch.cuda.get_device_name(0)
                    device_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    # Smart GPU Selection basierend auf Memory
                    if device_info['gpu_memory_gb'] >= 6:  # RTX 4060 8GB, RTX 3070+ etc.
                        device_info['device'] = 'cuda'
                        device_info['type'] = f"NVIDIA GPU ({device_info['gpu_memory_gb']:.1f}GB)"
                        logger.info(f"✅ CUDA GPU erkannt: {device_info['gpu_name']} ({device_info['gpu_memory_gb']:.1f}GB)")
                    else:
                        logger.warning(f"⚠️ GPU Memory zu niedrig ({device_info['gpu_memory_gb']:.1f}GB < 6GB) - verwende CPU")
            
            # MPS (Apple Silicon) Detection
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_info['mps_available'] = True
                device_info['available_devices'].append('mps')
                device_info['device'] = 'mps'
                device_info['type'] = 'Apple Silicon GPU'
                logger.info("✅ Apple Silicon MPS erkannt")
            
            # CPU Fallback
            if device_info['device'] == 'cpu':
                logger.info("ℹ️ Verwende CPU (kein geeignetes GPU erkannt)")
                
        except ImportError:
            logger.warning("⚠️ PyTorch nicht verfügbar - verwende CPU")
        except Exception as e:
            logger.error(f"❌ Device Detection Fehler: {e} - verwende CPU")
        
        return device_info
    
    def _analyze_hardware_capabilities(self) -> Dict[str, Any]:
        """Analysiert Hardware-Fähigkeiten für optimale Konfiguration mit Small Model Detection"""
        
        try:
            # CPU Info
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_threads = psutil.cpu_count(logical=True)
            ram_gb = round(psutil.virtual_memory().total / (1024**3))
            cpu_freq_max = psutil.cpu_freq().max if psutil.cpu_freq() else 3000
            
            # CPU Brand Detection (verbesserter)
            cpu_brand = platform.processor().lower()
            cpu_family = "Unknown"
            cpu_generation = 0
            
            if any(x in cpu_brand for x in ['apple', 'm1', 'm2', 'm3', 'm4']):
                cpu_family = "Apple Silicon"
                if 'm4' in cpu_brand: cpu_generation = 4
                elif 'm3' in cpu_brand: cpu_generation = 3
                elif 'm2' in cpu_brand: cpu_generation = 2
                elif 'm1' in cpu_brand: cpu_generation = 1
                
            elif 'ryzen' in cpu_brand:
                cpu_family = "AMD Ryzen"
                # Extract generation from common patterns
                for gen in range(2, 8):  # Ryzen 2000-7000 series
                    if f'{gen}' in cpu_brand or f'{gen}000' in cpu_brand:
                        cpu_generation = gen
                        break
                        
            elif any(x in cpu_brand for x in ['intel', 'core']):
                cpu_family = "Intel"
                # Basic Intel generation detection
                if any(x in cpu_brand for x in ['13th', '12th', '11th']):
                    cpu_generation = 12 if '12th' in cpu_brand else 13 if '13th' in cpu_brand else 11
            
            # Performance Tier Calculation mit Model-Size Berücksichtigung
            performance_score = 0
            recommended_model_size = "small"  # Default to small for safety
            
            # Device Score (most important)
            if self.device_info['device'] == 'cuda':
                gpu_memory = self.device_info.get('gpu_memory_gb', 0)
                if gpu_memory >= 16:
                    performance_score += 120
                    recommended_model_size = "standard"  # 16GB+ can handle standard models
                elif gpu_memory >= 12:
                    performance_score += 100
                    recommended_model_size = "standard"  # 12GB can handle standard models
                elif gpu_memory >= 8:
                    performance_score += 80
                    recommended_model_size = "small"     # 8GB = Small Models optimal
                elif gpu_memory >= 6:
                    performance_score += 60
                    recommended_model_size = "small"     # 6GB = Small Models required
                else:
                    performance_score += 40
                    recommended_model_size = "small"     # <6GB = Small Models required
                    
            elif self.device_info['device'] == 'mps':
                performance_score += 90  # Apple Silicon is very efficient
                # Apple Silicon - abhängig von Generation und RAM
                if ram_gb >= 32 and cpu_generation >= 3:  # M3+ mit viel RAM
                    recommended_model_size = "standard"
                elif ram_gb >= 24 and cpu_generation >= 2:  # M2+ mit mittlerem RAM
                    recommended_model_size = "small"      # Small models für bessere Performance
                else:
                    recommended_model_size = "small"      # M1 oder weniger RAM = Small Models
                    
            else:  # CPU
                performance_score += 30
                recommended_model_size = "small"  # CPU immer kleine Models für Performance
            
            # CPU Score
            if cpu_family == "Apple Silicon":
                performance_score += 40 + (cpu_generation * 10)  # M1=50, M2=60, M3=70, M4=80
            elif cpu_family == "AMD Ryzen":
                performance_score += 20 + (cpu_generation * 5)   # Ryzen 2xxx=30, 5xxx=45, 7xxx=55
            elif cpu_family == "Intel":
                performance_score += 15 + (cpu_generation * 3)   # Modern Intel bonus
            
            # Thread/Memory Bonus
            if cpu_threads >= 16: performance_score += 15
            elif cpu_threads >= 12: performance_score += 10
            elif cpu_threads >= 8: performance_score += 5
            
            if ram_gb >= 32: performance_score += 15
            elif ram_gb >= 16: performance_score += 10
            elif ram_gb >= 8: performance_score += 5
            
            # Determine Performance Tier
            if performance_score >= 150:
                tier = "ULTRA"
            elif performance_score >= 120:
                tier = "HIGH"
            elif performance_score >= 90:
                tier = "MEDIUM" 
            elif performance_score >= 60:
                tier = "LOW"
            else:
                tier = "BASIC"
            
            # Final Model Size Decision basierend auf Tier + Hardware
            if tier in ['BASIC', 'LOW'] or self.device_info['device'] == 'cpu':
                recommended_model_size = "small"  # Force small for weaker hardware
            
            hardware_profile = {
                'cpu_family': cpu_family,
                'cpu_generation': cpu_generation,
                'cpu_cores': cpu_cores,
                'cpu_threads': cpu_threads,
                'cpu_freq_max': cpu_freq_max,
                'ram_gb': ram_gb,
                'performance_score': performance_score,
                'tier': tier,
                'recommended_model_size': recommended_model_size,  # NEU
                'estimated_performance': self._estimate_tier_performance(tier, recommended_model_size)  # Erweitert
            }
            
            logger.info(f"🔍 Hardware Analysis: {cpu_family} Gen{cpu_generation}, {cpu_threads}T, {ram_gb}GB RAM")
            logger.info(f"📊 Performance Score: {performance_score} → {tier} Tier")
            logger.info(f"📦 Recommended Model Size: {recommended_model_size}")
            
            return hardware_profile
            
        except Exception as e:
            logger.error(f"❌ Hardware Analysis Fehler: {e}")
            return {
                'cpu_family': 'Unknown',
                'cpu_generation': 0,
                'cpu_cores': 4,
                'cpu_threads': 8,
                'ram_gb': 8,
                'performance_score': 50,
                'tier': 'BASIC',
                'recommended_model_size': 'small',  # Safe default
                'estimated_performance': '40-80s per sentence (small models)'
            }
    
    def _estimate_tier_performance(self, tier: str, model_size: str = "standard") -> str:
        """Schätzt Performance basierend auf Tier und Model-Size"""
        
        if model_size == "small":
            # Kleine Models sind deutlich schneller!
            performance_estimates = {
                'ULTRA': '1-4s per sentence',     # High-end GPU + small models = sehr schnell
                'HIGH': '2-8s per sentence',      # Good GPU/Apple Silicon + small models  
                'MEDIUM': '5-15s per sentence',   # Mid-range + small models
                'LOW': '10-30s per sentence',     # Your Ryzen 5 2600 + small models!
                'BASIC': '20-60s per sentence'    # Older hardware + small models
            }
        else:  # standard models
            performance_estimates = {
                'ULTRA': '2-8s per sentence',     # High-end GPU + standard models
                'HIGH': '5-15s per sentence',     # Good GPU or Apple Silicon
                'MEDIUM': '10-30s per sentence',  # Mid-range hardware
                'LOW': '20-60s per sentence',     # Your Ryzen 5 2600
                'BASIC': '60-120s per sentence'   # Older hardware
            }
        
        return performance_estimates.get(tier, '15-45s per sentence')
    
    def _select_optimal_voice(self) -> str:
        """Intelligente Voice-Auswahl basierend auf Hardware-Fähigkeiten"""
        
        tier = self.hardware_profile['tier']
        
        # Voice Selection Matrix (Quality vs Speed) - FIXED: Correct Bark voice presets
        voice_matrix = {
            'ULTRA': {
                'primary': 'v2/en_speaker_6',    # Best quality (English speakers work better)
                'alternatives': ['v2/en_speaker_4', 'v2/en_speaker_2'],
                'reason': 'Hardware kann beste Qualität handhaben'
            },
            'HIGH': {
                'primary': 'v2/en_speaker_4',    # High quality
                'alternatives': ['v2/en_speaker_2', 'v2/en_speaker_6'],
                'reason': 'Gute Balance zwischen Qualität und Geschwindigkeit'
            },
            'MEDIUM': {
                'primary': 'v2/en_speaker_2',    # Balanced
                'alternatives': ['v2/en_speaker_1', 'v2/en_speaker_3'],
                'reason': 'Ausgewogene Qualität für Mid-Range Hardware'
            },
            'LOW': {
                'primary': 'v2/en_speaker_1',    # Speed optimized
                'alternatives': ['v2/en_speaker_0', 'v2/en_speaker_2'],
                'reason': 'Optimiert für bessere Performance'
            },
            'BASIC': {
                'primary': 'v2/en_speaker_0',    # Fastest
                'alternatives': ['v2/en_speaker_1'],
                'reason': 'Schnellste Voice für schwächste Hardware'
            }
        }
        
        voice_config = voice_matrix.get(tier, voice_matrix['BASIC'])
        selected_voice = voice_config['primary']
        
        logger.info(f"🎯 Voice Selection: {selected_voice}")
        logger.info(f"💡 Reason: {voice_config['reason']}")
        
        return selected_voice
    
    def _get_optimal_model_config(self) -> Dict[str, Any]:
        """Generiert optimale Model-Konfiguration basierend auf Hardware"""
        
        tier = self.hardware_profile['tier']
        device = self.device_info['device']
        
        # Base Configuration per Tier
        base_configs = {
            'ULTRA': {
                'text_temp': 0.7,
                'waveform_temp': 0.7,
                'chunk_size': 150,
                'batch_size': 4,
                'use_cache': True,
                'preload_all_models': True,
                'fp16': device == 'cuda',  # GPU can use FP16
                'max_context_length': 256,
                'max_text_length': 300
            },
            'HIGH': {
                'text_temp': 0.7,
                'waveform_temp': 0.7,
                'chunk_size': 100,
                'batch_size': 2,
                'use_cache': True,
                'preload_all_models': True,
                'fp16': device == 'cuda',
                'max_context_length': 200,
                'max_text_length': 250
            },
            'MEDIUM': {
                'text_temp': 0.6,
                'waveform_temp': 0.6,
                'chunk_size': 75,
                'batch_size': 1,
                'use_cache': True,
                'preload_all_models': False,
                'fp16': False,  # Safer for mid-range
                'max_context_length': 150,
                'max_text_length': 200
            },
            'LOW': {
                'text_temp': 0.5,          # Reduced for speed
                'waveform_temp': 0.5,      # Reduced for speed
                'chunk_size': 50,          # Smaller chunks
                'batch_size': 1,
                'use_cache': True,
                'preload_all_models': False,
                'fp16': False,
                'max_context_length': 100,
                'max_text_length': 150
            },
            'BASIC': {
                'text_temp': 0.4,          # Minimal for maximum speed
                'waveform_temp': 0.4,
                'chunk_size': 30,          # Very small chunks
                'batch_size': 1,
                'use_cache': True,
                'preload_all_models': False,
                'fp16': False,
                'max_context_length': 80,
                'max_text_length': 100
            }
        }
        
        config = base_configs.get(tier, base_configs['BASIC'])
        
        # Device-specific optimizations
        if device == 'cuda':
            config['device_specific'] = {
                'torch_threads': min(8, self.hardware_profile['cpu_threads']),
                'cuda_memory_fraction': 0.8,
                'use_gpu_acceleration': True
            }
        elif device == 'mps':
            config['device_specific'] = {
                'torch_threads': self.hardware_profile['cpu_threads'],
                'use_mps_acceleration': True
            }
        else:  # CPU
            config['device_specific'] = {
                'torch_threads': self.hardware_profile['cpu_threads'],
                'omp_threads': self.hardware_profile['cpu_threads'],
                'mkl_threads': self.hardware_profile['cpu_threads']
            }
        
        logger.info(f"⚙️ Model Config: {tier} tier, {device} device")
        logger.debug(f"Config details: {config}")
        
        return config
    
    def initialize(self) -> bool:
        """Intelligente Bark TTS Initialisierung"""
        try:
            logger.info("🚀 Initialisiere Smart Bark TTS...")
            
            # Environment Setup basierend auf Device
            self._setup_optimal_environment()
            
            # Import Bark
            try:
                from bark import SAMPLE_RATE, generate_audio, preload_models
                
                self.bark_generate_audio = generate_audio
                self.bark_sample_rate = SAMPLE_RATE
                self.bark_preload_models = preload_models
                
                # Try to import set_seed, but don't fail if it's not available
                try:
                    from bark.generation import set_seed
                    set_seed(42)
                    logger.info("✅ Seed set for consistent results")
                except ImportError:
                    logger.warning("⚠️ set_seed not available in this Bark version")
                    # Try alternative locations
                    try:
                        from bark import set_seed
                        set_seed(42)
                        logger.info("✅ Seed set from bark module")
                    except ImportError:
                        logger.warning("⚠️ set_seed not found - results may vary")
                
                logger.info("✅ Bark Module imported successfully")
                
            except ImportError as e:
                logger.error(f"❌ Bark Import Fehler: {e}")
                logger.info("💡 Installation: pip install bark")
                return False
            
            # Model Preloading basierend auf Konfiguration
            if self.model_config['preload_all_models']:
                logger.info("🧠 Preloading alle Bark Models (kann 1-5 Min dauern)...")
                start_time = time.time()
                
                self.bark_preload_models()
                
                preload_time = time.time() - start_time
                logger.info(f"✅ Models preloaded ({preload_time:.1f}s)")
            else:
                logger.info("🧠 Models werden on-demand geladen (Performance-optimiert)")
            
            # Test Model
            if self._test_model():
                self.is_initialized = True
                logger.info("🎉 Smart Bark TTS erfolgreich initialisiert!")
                
                # Log final configuration
                self._log_initialization_summary()
                
                return True
            else:
                logger.error("❌ Model Test fehlgeschlagen")
                return False
                
        except Exception as e:
            logger.error(f"❌ Smart Bark Initialisierung fehlgeschlagen: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _setup_optimal_environment(self):
        """Setup optimale Environment Variables basierend auf Hardware"""
        
        device = self.device_info['device']
        config = self.model_config
        
        logger.info(f"🔧 Setup Environment für {device}...")
        
        # NEU: Automatic Small Model Detection
        if self._should_use_small_models():
            os.environ['SUNO_USE_SMALL_MODELS'] = 'True'
            logger.info("📦 Verwende kleine Bark Models (8GB VRAM / CPU-optimiert)")
        else:
            logger.info("📦 Verwende Standard Bark Models")
        
        if device == 'cuda':
            # GPU Environment
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
            if config.get('device_specific', {}).get('cuda_memory_fraction'):
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f"max_split_size_mb:512"
            
        elif device == 'mps':
            # Apple Silicon Environment - kann auch von kleineren Models profitieren
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            if self._should_use_small_models():
                logger.info("🍎 Apple Silicon: Verwende kleine Models für bessere Performance")
            
        else:
            # CPU Environment - IMMER kleine Models
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['FORCE_CPU'] = '1'
            os.environ['SUNO_USE_SMALL_MODELS'] = 'True'  # CPU braucht immer kleine Models
            
            # CPU Thread Optimization
            device_config = config.get('device_specific', {})
            thread_count = str(device_config.get('torch_threads', 8))
            
            os.environ['OMP_NUM_THREADS'] = thread_count
            os.environ['MKL_NUM_THREADS'] = thread_count
            os.environ['TORCH_NUM_THREADS'] = thread_count
            
            logger.info(f"💻 CPU Modus: Kleine Models + {thread_count} Threads")
        
        # Memory Management
        os.environ['PYTORCH_JIT'] = '0'  # Disable JIT for stability

    # NEU: Intelligente Small Model Detection
    def _should_use_small_models(self) -> bool:
        """Entscheidet intelligent ob kleine Models verwendet werden sollen"""
        
        device = self.device_info['device']
        tier = self.hardware_profile['tier']
        
        # CPU = immer kleine Models
        if device == 'cpu':
            return True
        
        # GPU Memory Check
        if device == 'cuda':
            gpu_memory = self.device_info.get('gpu_memory_gb', 0)
            if gpu_memory < 12:  # Weniger als 12GB VRAM
                return True
            elif gpu_memory < 16 and tier in ['LOW', 'BASIC']:  # Auch bei 12-16GB wenn schwächere Hardware
                return True
            else:
                return False
        
        # Apple Silicon - intelligente Entscheidung
        if device == 'mps':
            # M1/M2 mit wenig RAM = kleine Models
            ram_gb = self.hardware_profile.get('ram_gb', 16)
            cpu_generation = self.hardware_profile.get('cpu_generation', 1)
            
            if ram_gb < 16 or cpu_generation <= 2:  # M1/M2 mit <16GB RAM
                return True
            elif tier in ['MEDIUM', 'LOW', 'BASIC']:
                return True
            else:
                return False
        
        # Fallback
        return tier in ['LOW', 'BASIC']
    
    def _test_model(self) -> bool:
        """Testet Bark Model mit Hardware-spezifischen Parametern"""
        try:
            logger.info("🧪 Teste Bark Model...")
            
            # Simple test text
            test_text = f"[{self.voice_preset}] Test"
            
            # Test generation with timeout
            start_time = time.time()
            
            try:
                # Direct bark generation test
                audio_array = self.bark_generate_audio(test_text)
                
                test_time = time.time() - start_time
                
                if audio_array is not None and len(audio_array) > 0:
                    logger.info(f"✅ Model Test erfolgreich ({test_time:.1f}s)")
                    return True
                else:
                    logger.error("❌ Model Test: Kein Audio generiert")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Model Test Generation Fehler: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model Test Fehler: {e}")
            return False
    
    def speak(self, text: str, emotion: str = "neutral", auto_play: bool = True) -> Optional[Path]:
        """
        Speak text with emotion using Bark TTS and return audio file path
        
        Args:
            text: Text to speak
            emotion: Emotion for the speech
            auto_play: Whether to automatically play the audio
            
        Returns:
            Path to generated audio file or None if failed
        """
        if not self.is_initialized:
            logger.error("❌ Bark TTS not initialisiert")
            return None
        
        try:
            start_time = time.time()
            
            # Clean and prepare text
            clean_text = self._clean_text_for_synthesis(text)
            if not clean_text:
                logger.warning("⚠️ Empty text after cleaning")
                return None
            
            # Check text length and use chunking if needed
            max_length = self.model_config.get('max_text_length', 200)
            if len(clean_text) > max_length:
                logger.info(f"📝 Text too long ({len(clean_text)} chars), using chunked speech")
                success = self.speak_long_text(text, emotion, auto_play)
                return Path(f"chunked_audio_{int(time.time())}.wav") if success else None
            
            # Enhance text with emotion and voice
            enhanced_text = self._enhance_text_with_emotion(clean_text, emotion)
            
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            emotion_suffix = f"_{emotion}" if emotion != "neutral" else ""
            filename = f"kira_voice_{timestamp}{emotion_suffix}.wav"
            output_path = self.output_dir / filename
            
            logger.info(f"🗣️ Generating speech: '{clean_text[:50]}...' ({emotion})")
            
            # Generate audio with Bark
            audio_array = self.bark_generate_audio(enhanced_text)
            
            # Save audio file
            if self._save_audio_array(audio_array, output_path):
                generation_time = time.time() - start_time
                self._update_performance_stats(generation_time)
                
                logger.info(f"✅ Audio generated: {output_path.name} ({generation_time:.1f}s)")
                
                # Auto-play if requested
                if auto_play:
                    self._play_audio(output_path)
                
                return output_path
            else:
                logger.error("❌ Failed to save audio file")
                return None
                
        except Exception as e:
            logger.error(f"❌ Speech generation failed: {e}")
            return None
    
    def synthesize(self, text: str, emotion: str = "neutral") -> Dict[str, Any]:
        """
        Synthesize method expected by voice manager - delegates to synthesize_speech
        
        Args:
            text: Text to synthesize  
            emotion: Emotion for the speech
            
        Returns:
            Dictionary with success status and audio info (same as synthesize_speech)
        """
        return self.synthesize_speech(text, emotion)
    
    def synthesize_speech(self, text: str, emotion: str = "neutral") -> Dict[str, Any]:
        """
        Synthesize speech and return API-compatible response
        
        Args:
            text: Text to synthesize
            emotion: Emotion for the speech
            
        Returns:
            Dictionary with success status and audio info
        """
        try:
            from datetime import datetime
            
            audio_path = self.speak(text, emotion, auto_play=False)
            
            if audio_path and audio_path.exists():
                # Generate URL for frontend access
                audio_url = f"/api/audio/{audio_path.name}"
                
                return {
                    'success': True,
                    'audio_path': str(audio_path),
                    'audio_url': audio_url,
                    'filename': audio_path.name,
                    'text': text,
                    'emotion': emotion,
                    'duration_estimate': len(text) * 0.1,  # Rough estimate
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to generate audio file'
                }
                
        except Exception as e:
            logger.error(f"❌ Speech synthesis API error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _enhance_text_with_emotion(self, text: str, emotion: str) -> str:
        """
        Enhance text with emotion and voice preset for Bark synthesis
        
        Args:
            text: Clean text to enhance
            emotion: Emotion to apply
            
        Returns:
            Enhanced text with Bark formatting
        """
        # Add voice preset to text for Bark
        enhanced_text = f"[{self.voice_preset}] {text}"
        
        # Add emotion markers if not neutral
        if emotion != "neutral":
            emotion_markers = {
                "happy": "😊",
                "sad": "😢", 
                "excited": "🎉",
                "angry": "😠",
                "surprised": "😲",
                "calm": "😌",
                "confident": "💪",
                "worried": "😟"
            }
            
            marker = emotion_markers.get(emotion.lower(), "")
            if marker:
                enhanced_text = f"[{self.voice_preset}] {marker} {text}"
        
        return enhanced_text
    
    def _clean_text_for_synthesis(self, text: str) -> str:
        """Clean text for optimal synthesis"""
        if not text:
            return ""
        
        # Remove markdown
        import re
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        
        # Clean special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'\"]+', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _save_audio_array(self, audio_array: np.ndarray, output_path: Path) -> bool:
        """Save audio array to file"""
        try:
            # Import scipy for saving
            from scipy.io.wavfile import write
            
            # Ensure audio is in correct format
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Normalize audio
            audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Save as WAV file
            write(str(output_path), self.bark_sample_rate, audio_array)
            
            return output_path.exists()
            
        except Exception as e:
            logger.error(f"❌ Audio save error: {e}")
            return False
    
    def _update_performance_stats(self, generation_time: float):
        """Update performance statistics"""
        self.generation_stats['total_generations'] += 1
        self.generation_stats['total_time'] += generation_time
        self.generation_stats['last_generation_time'] = generation_time
        self.generation_stats['average_time'] = (
            self.generation_stats['total_time'] / self.generation_stats['total_generations']
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            **self.generation_stats,
            'hardware_tier': self.hardware_profile['tier'],
            'device_type': self.device_info['device'],
            'voice_preset': self.voice_preset
        }
    
    def cleanup(self):
        """Cleanup Smart Bark Engine"""
        try:
            logger.info("🧹 Smart Bark TTS Cleanup...")
            
            # Performance Report
            if self.generation_stats['total_generations'] > 0:
                stats = self.get_performance_stats()
                logger.info(f"📊 Final Performance Stats:")
                logger.info(f"   Total Generations: {stats['total_generations']}")
                logger.info(f"   Average Time: {stats['average_time']:.1f}s")
                logger.info(f"   Hardware Tier: {stats['hardware_tier']}")
            
            # Clear references
            self.bark_generate_audio = None
            self.bark_sample_rate = None
            self.bark_preload_models = None
            self.is_initialized = False
            
            logger.info("✅ Smart Bark TTS Cleanup abgeschlossen")
            
        except Exception as e:
            logger.error(f"❌ Cleanup Fehler: {e}")
    
    def _generate_audio_optimized(self, text: str) -> Optional[np.ndarray]:
        """
        Optimierte Audio-Generierung basierend auf Hardware-Profil
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio array or None if failed
        """
        try:
            # Hardware-adaptive generation
            tier = self.hardware_profile['tier']
            
            if tier in ['ULTRA', 'HIGH']:
                # High-performance generation
                return self._generate_audio_high_performance(text)
            elif tier == 'MEDIUM':
                # Balanced generation
                return self._generate_audio_balanced(text)
            else:
                # Conservative generation for lower-end hardware
                return self._generate_audio_conservative(text)
                
        except Exception as e:
            logger.error(f"❌ Optimized audio generation failed: {e}")
            return None
    
    def _generate_audio_high_performance(self, text: str) -> Optional[np.ndarray]:
        """High-performance audio generation"""
        try:
            return self.bark_generate_audio(text)
        except Exception as e:
            logger.error(f"❌ High-performance generation failed: {e}")
            return None
    
    def _generate_audio_balanced(self, text: str) -> Optional[np.ndarray]:
        """Balanced audio generation"""
        try:
            # Add some memory management for medium-tier hardware
            import gc
            gc.collect()
            
            audio_array = self.bark_generate_audio(text)
            
            gc.collect()
            return audio_array
        except Exception as e:
            logger.error(f"❌ Balanced generation failed: {e}")
            return None
    
    def _generate_audio_conservative(self, text: str) -> Optional[np.ndarray]:
        """Conservative audio generation for lower-end hardware"""
        try:
            # Aggressive memory management
            import gc
            import torch
            
            # Clear cache before generation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate with lower precision if needed
            audio_array = self.bark_generate_audio(text)
            
            # Clean up after generation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return audio_array
        except Exception as e:
            logger.error(f"❌ Conservative generation failed: {e}")
            return None
    
    def _play_audio(self, audio_path: Path) -> bool:
        """
        Play audio file using platform-appropriate method
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not audio_path.exists():
                logger.error(f"❌ Audio file not found: {audio_path}")
                return False
            
            logger.info(f"🔊 Playing audio: {audio_path.name}")
            
            # Platform-specific audio playback
            if self.system == "Darwin":  # macOS
                return self._play_audio_macos(audio_path)
            elif self.system == "Linux":
                return self._play_audio_linux(audio_path)
            elif self.system == "Windows":
                return self._play_audio_windows(audio_path)
            else:
                logger.warning(f"⚠️ Unsupported platform: {self.system}")
                return self._play_audio_fallback(audio_path)
                
        except Exception as e:
            logger.error(f"❌ Audio playback error: {e}")
            return False
    
    def _play_audio_macos(self, audio_path: Path) -> bool:
        """macOS audio playback using afplay"""
        try:
            import subprocess
            subprocess.run(['afplay', str(audio_path)], check=True, timeout=60)
            return True
        except Exception as e:
            logger.error(f"❌ macOS audio playback failed: {e}")
            return False
    
    def _play_audio_linux(self, audio_path: Path) -> bool:
        """Linux audio playback"""
        try:
            import subprocess
            
            # Try different audio players
            players = ['aplay', 'paplay', 'play', 'ffplay']
            
            for player in players:
                try:
                    subprocess.run([player, str(audio_path)], check=True, timeout=60)
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
                    
            logger.error("❌ No suitable audio player found on Linux")
            return False
            
        except Exception as e:
            logger.error(f"❌ Linux audio playback failed: {e}")
            return False
    
    def _play_audio_windows(self, audio_path: Path) -> bool:
        """Windows audio playback"""
        try:
            import winsound
            winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
            return True
        except Exception as e:
            logger.error(f"❌ Windows audio playback failed: {e}")
            return False
    
    def _play_audio_fallback(self, audio_path: Path) -> bool:
        """Fallback audio playback using pygame or other libraries"""
        try:
            # Try pygame
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            pygame.mixer.quit()
            return True
            
        except ImportError:
            logger.debug("pygame not available")
        except Exception as e:
            logger.debug(f"pygame playback failed: {e}")
        
        logger.warning("⚠️ No audio playback method available")
        return False
    
    def speak_long_text(self, text: str, emotion: str = "neutral", auto_play: bool = True) -> bool:
        """
        Handle long text by chunking it into smaller pieces
        
        Args:
            text: Long text to speak
            emotion: Emotion for speech
            auto_play: Whether to play audio automatically
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Split text into manageable chunks
            max_chunk_size = self.model_config.get('max_text_length', 200)
            chunks = self._split_text_intelligent(text, max_chunk_size)
            
            logger.info(f"📝 Speaking long text in {len(chunks)} chunks")
            
            success_count = 0
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"🗣️ Chunk {i}/{len(chunks)}: '{chunk[:30]}...'")
                
                if self.speak(chunk, emotion, auto_play):
                    success_count += 1
                    
                    # Brief pause between chunks
                    if i < len(chunks) and auto_play:
                        time.sleep(0.5)
                else:
                    logger.warning(f"⚠️ Chunk {i} failed")
            
            success_rate = success_count / len(chunks)
            logger.info(f"📊 Long text success rate: {success_count}/{len(chunks)} ({success_rate:.1%})")
            
            return success_rate >= 0.7  # Consider successful if 70%+ chunks work
            
        except Exception as e:
            logger.error(f"❌ Long text speech failed: {e}")
            return False
    
    def _split_text_intelligent(self, text: str, max_chunk_size: int) -> List[str]:
        """
        Intelligently split text into chunks while preserving sentence structure
        
        Args:
            text: Text to split
            max_chunk_size: Maximum size per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed limit
            if len(current_chunk) + len(sentence) + 2 > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Sentence itself is too long, split by words
                    words = sentence.split()
                    word_chunk = ""
                    
                    for word in words:
                        if len(word_chunk) + len(word) + 1 > max_chunk_size:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                                word_chunk = word
                            else:
                                # Single word too long, just add it
                                chunks.append(word)
                        else:
                            word_chunk += " " + word if word_chunk else word
                    
                    if word_chunk:
                        chunks.append(word_chunk.strip())
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _log_initialization_summary(self):
        """Log initialization summary with system info"""
        try:
            logger.info("=" * 60)
            logger.info("🎤 SMART BARK TTS INITIALISIERUNG ABGESCHLOSSEN")
            logger.info("=" * 60)
            
            # Device Info
            device_info = self.device_info
            logger.info(f"🖥️  Device: {device_info.get('device', 'unknown')}")
            logger.info(f"💾 RAM: {device_info.get('ram_gb', 0):.1f} GB")
            
            if device_info.get('device') == 'cuda':
                gpu_info = device_info.get('gpu_info', {})
                logger.info(f"🎮 GPU: {gpu_info.get('name', 'Unknown')}")
                logger.info(f"📊 VRAM: {gpu_info.get('memory_gb', 0):.1f} GB")
            elif device_info.get('device') == 'mps':
                logger.info("🍎 Apple Silicon GPU detected")
            
            # Model Config
            config = self.model_config
            logger.info(f"🎯 Voice Preset: {self.voice_preset}")
            logger.info(f"📦 Model Type: {'Small' if config.get('use_small_models') else 'Standard'}")
            logger.info(f"⚡ Batch Size: {config.get('max_batch_size', 1)}")
            
            # Performance Profile
            hardware_profile = getattr(self, 'hardware_profile', {})
            performance_tier = hardware_profile.get('performance_tier', 'unknown')
            logger.info(f"🚀 Performance Tier: {performance_tier}")
            
            # Environment
            env_vars = []
            for var in ['SUNO_USE_SMALL_MODELS', 'CUDA_VISIBLE_DEVICES', 'PYTORCH_ENABLE_MPS_FALLBACK']:
                if os.environ.get(var):
                    env_vars.append(f"{var}={os.environ[var]}")
            
            if env_vars:
                logger.info(f"🔧 Environment: {', '.join(env_vars)}")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.warning(f"❌ Could not log initialization summary: {e}")

# Backward compatibility
BarkTTSEngine = SmartBarkTTSEngine

# Export
__all__ = ['SmartBarkTTSEngine', 'BarkTTSEngine']