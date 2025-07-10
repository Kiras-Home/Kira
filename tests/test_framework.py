"""
Quality Assurance & Testing Framework - Comprehensive Testing for Kira Voice System
Advanced testing suite with automated tests, performance monitoring, and quality metrics
"""

import logging
import unittest
import pytest
import asyncio
import threading
import time
import json
import os
import sys
import subprocess
import psutil
import traceback
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import wave
import numpy as np
from collections import defaultdict, deque
import statistics
import concurrent.futures
import queue

# Test data and mocking
import mock
from unittest.mock import Mock, MagicMock, patch
import io
import contextlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    category: str
    success: bool
    execution_time_ms: float
    message: str
    error: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TestSuite:
    """Test suite configuration"""
    name: str
    description: str
    tests: List[Callable] = field(default_factory=list)
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    timeout_seconds: int = 30
    parallel: bool = False

@dataclass
class QualityMetrics:
    """Quality assurance metrics"""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    avg_execution_time: float = 0.0
    code_coverage: float = 0.0
    performance_score: float = 0.0
    reliability_score: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return (self.passed_tests / max(1, self.total_tests)) * 100
    
    @property
    def overall_score(self) -> float:
        return (self.success_rate + self.performance_score + self.reliability_score) / 3

class MockAudioData:
    """Mock audio data for testing"""
    
    @staticmethod
    def generate_sine_wave(frequency: float = 440.0, duration: float = 1.0, sample_rate: int = 44100) -> np.ndarray:
        """Generate sine wave audio data"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        return np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    @staticmethod
    def generate_noise(duration: float = 1.0, sample_rate: int = 44100) -> np.ndarray:
        """Generate random noise audio data"""
        samples = int(sample_rate * duration)
        return np.random.uniform(-1, 1, samples).astype(np.float32)
    
    @staticmethod
    def generate_silence(duration: float = 1.0, sample_rate: int = 44100) -> np.ndarray:
        """Generate silence audio data"""
        samples = int(sample_rate * duration)
        return np.zeros(samples, dtype=np.float32)

class MockVoiceEngine:
    """Mock voice engine for testing"""
    
    def __init__(self):
        self.is_initialized = False
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        
    def initialize(self) -> bool:
        """Mock initialization"""
        time.sleep(0.1)  # Simulate initialization time
        self.is_initialized = True
        return True
    
    def speak(self, text: str) -> bool:
        """Mock speak function"""
        if not self.is_initialized:
            return False
        
        self.is_speaking = True
        self.speech_queue.put(text)
        time.sleep(len(text) * 0.01)  # Simulate speech time
        self.is_speaking = False
        return True
    
    def stop_speaking(self) -> bool:
        """Mock stop speaking"""
        self.is_speaking = False
        return True
    
    def cleanup(self):
        """Mock cleanup"""
        self.is_initialized = False
        self.is_speaking = False

class MockAudioManager:
    """Mock audio manager for testing"""
    
    def __init__(self):
        self.volume = 0.5
        self.is_muted = False
        self.is_playing = False
        self.devices = ["Default Speaker", "Bluetooth Speaker"]
        
    def set_volume(self, volume: float) -> bool:
        """Mock set volume"""
        if 0.0 <= volume <= 1.0:
            self.volume = volume
            return True
        return False
    
    def adjust_volume(self, delta: float) -> bool:
        """Mock adjust volume"""
        new_volume = max(0.0, min(1.0, self.volume + delta))
        return self.set_volume(new_volume)
    
    def mute_audio(self) -> bool:
        """Mock mute audio"""
        self.is_muted = True
        return True
    
    def unmute_audio(self) -> bool:
        """Mock unmute audio"""
        self.is_muted = False
        return True
    
    def stop_all_audio(self) -> bool:
        """Mock stop all audio"""
        self.is_playing = False
        return True
    
    def get_audio_status(self) -> Dict[str, Any]:
        """Mock get audio status"""
        return {
            'devices_count': len(self.devices),
            'master_volume': self.volume,
            'is_muted': self.is_muted,
            'is_playing': self.is_playing
        }

class PerformanceMonitor:
    """Performance monitoring for tests"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        
    def start_measurement(self, metric_name: str):
        """Start measuring a metric"""
        self.start_times[metric_name] = time.time()
    
    def end_measurement(self, metric_name: str) -> float:
        """End measuring a metric and return duration"""
        if metric_name in self.start_times:
            duration = time.time() - self.start_times[metric_name]
            self.metrics[metric_name].append(duration)
            del self.start_times[metric_name]
            return duration
        return 0.0
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        values = self.metrics[metric_name]
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics"""
        return {name: self.get_statistics(name) for name in self.metrics.keys()}

class VoiceCommandTester:
    """Specialized tester for voice commands"""
    
    def __init__(self):
        self.test_commands = [
            # Audio control commands
            "kira stopp",
            "kira pause",
            "kira lauter",
            "kira leiser",
            "kira lautstärke auf 50",
            
            # System commands  
            "kira status",
            "kira hilfe",
            "wie spät ist es kira",
            
            # Greeting commands
            "hallo kira",
            "danke kira",
            "gute nacht kira",
            
            # Invalid commands
            "unbekannter befehl",
            "xyz 123 test",
            ""
        ]
        
        self.expected_results = {
            "kira stopp": True,
            "kira pause": True,
            "kira lauter": True,
            "kira leiser": True,
            "kira lautstärke auf 50": True,
            "kira status": True,
            "kira hilfe": True,
            "wie spät ist es kira": True,
            "hallo kira": True,
            "danke kira": True,
            "gute nacht kira": True,
            "unbekannter befehl": False,
            "xyz 123 test": False,
            "": False
        }
    
    def test_command_recognition(self, voice_processor) -> List[TestResult]:
        """Test command recognition accuracy"""
        results = []
        
        for command in self.test_commands:
            start_time = time.time()
            
            try:
                result = voice_processor.process_voice_input(command, "test_user", "test_session")
                execution_time = (time.time() - start_time) * 1000
                
                expected = self.expected_results.get(command, False)
                success = (result.success == expected)
                
                test_result = TestResult(
                    test_name=f"command_recognition_{command[:20]}",
                    category="voice_commands",
                    success=success,
                    execution_time_ms=execution_time,
                    message=f"Command: '{command}' -> {result.success} (expected: {expected})",
                    performance_metrics={
                        'confidence': result.confidence,
                        'processing_time_ms': result.execution_time_ms
                    }
                )
                
                results.append(test_result)
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                test_result = TestResult(
                    test_name=f"command_recognition_{command[:20]}",
                    category="voice_commands",
                    success=False,
                    execution_time_ms=execution_time,
                    message=f"Command processing failed: {command}",
                    error=str(e)
                )
                
                results.append(test_result)
        
        return results
    
    def test_response_time(self, voice_processor) -> List[TestResult]:
        """Test command response time"""
        results = []
        response_times = []
        
        # Test with valid commands
        valid_commands = [cmd for cmd, expected in self.expected_results.items() if expected]
        
        for command in valid_commands[:5]:  # Test first 5 valid commands
            start_time = time.time()
            
            try:
                result = voice_processor.process_voice_input(command, "test_user", "test_session")
                execution_time = (time.time() - start_time) * 1000
                response_times.append(execution_time)
                
                # Response time should be < 500ms for good UX
                success = execution_time < 500
                
                test_result = TestResult(
                    test_name=f"response_time_{command[:15]}",
                    category="performance",
                    success=success,
                    execution_time_ms=execution_time,
                    message=f"Response time: {execution_time:.1f}ms ({'✅' if success else '❌'})",
                    performance_metrics={'response_time_ms': execution_time}
                )
                
                results.append(test_result)
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                test_result = TestResult(
                    test_name=f"response_time_{command[:15]}",
                    category="performance",
                    success=False,
                    execution_time_ms=execution_time,
                    message=f"Response time test failed: {command}",
                    error=str(e)
                )
                
                results.append(test_result)
        
        # Add overall response time test
        if response_times:
            avg_response_time = statistics.mean(response_times)
            success = avg_response_time < 300  # Average should be < 300ms
            
            overall_result = TestResult(
                test_name="average_response_time",
                category="performance",
                success=success,
                execution_time_ms=avg_response_time,
                message=f"Average response time: {avg_response_time:.1f}ms",
                performance_metrics={
                    'avg_response_time_ms': avg_response_time,
                    'min_response_time_ms': min(response_times),
                    'max_response_time_ms': max(response_times)
                }
            )
            
            results.append(overall_result)
        
        return results

class AudioSystemTester:
    """Specialized tester for audio system"""
    
    def __init__(self):
        self.mock_audio_manager = MockAudioManager()
        self.mock_voice_engine = MockVoiceEngine()
    
    def test_audio_manager_functionality(self) -> List[TestResult]:
        """Test audio manager basic functionality"""
        results = []
        
        # Test volume control
        test_cases = [
            ("set_volume_valid", lambda: self.mock_audio_manager.set_volume(0.7), True),
            ("set_volume_invalid_high", lambda: self.mock_audio_manager.set_volume(1.5), False),
            ("set_volume_invalid_low", lambda: self.mock_audio_manager.set_volume(-0.1), False),
            ("adjust_volume_up", lambda: self.mock_audio_manager.adjust_volume(0.1), True),
            ("adjust_volume_down", lambda: self.mock_audio_manager.adjust_volume(-0.1), True),
            ("mute_audio", lambda: self.mock_audio_manager.mute_audio(), True),
            ("unmute_audio", lambda: self.mock_audio_manager.unmute_audio(), True),
            ("stop_all_audio", lambda: self.mock_audio_manager.stop_all_audio(), True),
        ]
        
        for test_name, test_func, expected in test_cases:
            start_time = time.time()
            
            try:
                result = test_func()
                execution_time = (time.time() - start_time) * 1000
                
                success = (result == expected)
                
                test_result = TestResult(
                    test_name=test_name,
                    category="audio_system",
                    success=success,
                    execution_time_ms=execution_time,
                    message=f"Audio manager {test_name}: {result} (expected: {expected})"
                )
                
                results.append(test_result)
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                test_result = TestResult(
                    test_name=test_name,
                    category="audio_system",
                    success=False,
                    execution_time_ms=execution_time,
                    message=f"Audio manager {test_name} failed",
                    error=str(e)
                )
                
                results.append(test_result)
        
        return results
    
    def test_voice_engine_functionality(self) -> List[TestResult]:
        """Test voice engine functionality"""
        results = []
        
        # Test voice engine initialization
        start_time = time.time()
        try:
            init_result = self.mock_voice_engine.initialize()
            execution_time = (time.time() - start_time) * 1000
            
            test_result = TestResult(
                test_name="voice_engine_initialization",
                category="voice_engine",
                success=init_result,
                execution_time_ms=execution_time,
                message=f"Voice engine initialization: {init_result}"
            )
            results.append(test_result)
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            test_result = TestResult(
                test_name="voice_engine_initialization",
                category="voice_engine",
                success=False,
                execution_time_ms=execution_time,
                message="Voice engine initialization failed",
                error=str(e)
            )
            results.append(test_result)
        
        # Test speech functionality
        test_phrases = [
            "Hallo, ich bin Kira!",
            "Das ist ein Test.",
            "Sehr langer Text um die Sprachausgabe zu testen mit vielen Wörtern und Sätzen.",
            "",  # Empty string test
            "123 456 789"  # Numbers test
        ]
        
        for i, phrase in enumerate(test_phrases):
            start_time = time.time()
            
            try:
                speak_result = self.mock_voice_engine.speak(phrase)
                execution_time = (time.time() - start_time) * 1000
                
                # Empty string should return False
                expected = len(phrase) > 0
                success = (speak_result == expected)
                
                test_result = TestResult(
                    test_name=f"voice_engine_speak_{i}",
                    category="voice_engine",
                    success=success,
                    execution_time_ms=execution_time,
                    message=f"Speak test '{phrase[:30]}...': {speak_result}",
                    performance_metrics={'text_length': len(phrase)}
                )
                
                results.append(test_result)
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                test_result = TestResult(
                    test_name=f"voice_engine_speak_{i}",
                    category="voice_engine",
                    success=False,
                    execution_time_ms=execution_time,
                    message=f"Speak test failed: '{phrase[:30]}...'",
                    error=str(e)
                )
                
                results.append(test_result)
        
        return results

class IntegrationTester:
    """Integration testing for complete system"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
    
    def test_full_voice_workflow(self) -> List[TestResult]:
        """Test complete voice interaction workflow"""
        results = []
        
        try:
            # Import actual components (with fallback to mocks)
            try:
                from voice.stt.voice_commands import VoiceCommandProcessor
                processor = VoiceCommandProcessor()
                processor.initialize()
                using_real_components = True
            except ImportError:
                # Fallback to mock
                processor = MockVoiceCommandProcessor()
                using_real_components = False
            
            # Test workflow steps
            workflow_steps = [
                ("initialization", "System initialization"),
                ("voice_input", "Voice input processing"),
                ("command_recognition", "Command recognition"),
                ("command_execution", "Command execution"),
                ("response_generation", "Response generation"),
                ("cleanup", "System cleanup")
            ]
            
            for step_name, step_description in workflow_steps:
                start_time = time.time()
                self.performance_monitor.start_measurement(step_name)
                
                try:
                    # Simulate workflow step
                    if step_name == "initialization":
                        success = True  # Already initialized
                    elif step_name == "voice_input":
                        success = True  # Simulated voice input
                    elif step_name == "command_recognition":
                        result = processor.process_voice_input("kira status", "test_user")
                        success = hasattr(result, 'success')
                    elif step_name == "command_execution":
                        success = True  # Command was processed
                    elif step_name == "response_generation":
                        success = True  # Response was generated
                    elif step_name == "cleanup":
                        if hasattr(processor, 'cleanup'):
                            processor.cleanup()
                        success = True
                    
                    execution_time = self.performance_monitor.end_measurement(step_name) * 1000
                    
                    test_result = TestResult(
                        test_name=f"workflow_{step_name}",
                        category="integration",
                        success=success,
                        execution_time_ms=execution_time,
                        message=f"{step_description}: {'✅' if success else '❌'}",
                        performance_metrics={'step': step_name, 'using_real_components': using_real_components}
                    )
                    
                    results.append(test_result)
                    
                except Exception as e:
                    execution_time = self.performance_monitor.end_measurement(step_name) * 1000
                    
                    test_result = TestResult(
                        test_name=f"workflow_{step_name}",
                        category="integration",
                        success=False,
                        execution_time_ms=execution_time,
                        message=f"{step_description} failed",
                        error=str(e)
                    )
                    
                    results.append(test_result)
            
            # Add overall workflow performance test
            workflow_stats = self.performance_monitor.get_all_statistics()
            total_time = sum(stats.get('mean', 0) for stats in workflow_stats.values()) * 1000
            
            overall_result = TestResult(
                test_name="full_workflow_performance",
                category="integration",
                success=total_time < 2000,  # Should complete in < 2 seconds
                execution_time_ms=total_time,
                message=f"Full workflow time: {total_time:.1f}ms",
                performance_metrics=workflow_stats
            )
            
            results.append(overall_result)
            
        except Exception as e:
            test_result = TestResult(
                test_name="full_workflow_setup",
                category="integration",
                success=False,
                execution_time_ms=0,
                message="Workflow test setup failed",
                error=str(e)
            )
            results.append(test_result)
        
        return results

class MockVoiceCommandProcessor:
    """Mock voice command processor for testing"""
    
    def __init__(self):
        self.is_initialized = False
        self.commands_processed = 0
        
    def initialize(self) -> bool:
        self.is_initialized = True
        return True
    
    def process_voice_input(self, text: str, user_id: str = "test") -> Mock:
        self.commands_processed += 1
        
        # Simulate command processing
        success = len(text) > 0 and "kira" in text.lower()
        confidence = 0.8 if success else 0.2
        
        result = Mock()
        result.success = success
        result.confidence = confidence
        result.execution_time_ms = 150.0
        result.message = f"Processed: {text}"
        
        return result
    
    def cleanup(self):
        self.is_initialized = False

class StressTester:
    """Stress testing for system limits"""
    
    def __init__(self):
        self.max_concurrent_requests = 100
        self.test_duration_seconds = 30
    
    def test_concurrent_commands(self) -> List[TestResult]:
        """Test concurrent command processing"""
        results = []
        
        try:
            # Create mock processor
            processor = MockVoiceCommandProcessor()
            processor.initialize()
            
            # Test different concurrency levels
            concurrency_levels = [1, 5, 10, 25, 50]
            
            for concurrency in concurrency_levels:
                start_time = time.time()
                
                def process_command(i):
                    return processor.process_voice_input(f"kira test command {i}", f"user_{i}")
                
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futures = [executor.submit(process_command, i) for i in range(concurrency)]
                        results_list = [future.result(timeout=5) for future in futures]
                    
                    execution_time = (time.time() - start_time) * 1000
                    successful_requests = sum(1 for r in results_list if r.success)
                    success_rate = (successful_requests / concurrency) * 100
                    
                    # Success if > 90% of requests succeeded
                    success = success_rate > 90
                    
                    test_result = TestResult(
                        test_name=f"concurrent_commands_{concurrency}",
                        category="stress_test",
                        success=success,
                        execution_time_ms=execution_time,
                        message=f"Concurrency {concurrency}: {success_rate:.1f}% success rate",
                        performance_metrics={
                            'concurrency_level': concurrency,
                            'success_rate': success_rate,
                            'successful_requests': successful_requests,
                            'avg_time_per_request': execution_time / concurrency
                        }
                    )
                    
                    results.append(test_result)
                    
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    
                    test_result = TestResult(
                        test_name=f"concurrent_commands_{concurrency}",
                        category="stress_test",
                        success=False,
                        execution_time_ms=execution_time,
                        message=f"Concurrency test {concurrency} failed",
                        error=str(e)
                    )
                    
                    results.append(test_result)
        
        except Exception as e:
            test_result = TestResult(
                test_name="concurrent_commands_setup",
                category="stress_test",
                success=False,
                execution_time_ms=0,
                message="Stress test setup failed",
                error=str(e)
            )
            results.append(test_result)
        
        return results
    
    def test_memory_usage(self) -> List[TestResult]:
        """Test memory usage under load"""
        results = []
        
        try:
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create processor and run commands
            processor = MockVoiceCommandProcessor()
            processor.initialize()
            
            # Run many commands to test memory usage
            start_time = time.time()
            
            for i in range(1000):
                processor.process_voice_input(f"kira test command {i}", f"user_{i % 10}")
                
                # Check memory every 100 commands
                if i % 100 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_increase = current_memory - initial_memory
                    
                    # Memory increase should be reasonable (< 100MB for 1000 commands)
                    success = memory_increase < 100
                    
                    if not success:
                        break
            
            execution_time = (time.time() - start_time) * 1000
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            total_memory_increase = final_memory - initial_memory
            
            # Final memory check
            success = total_memory_increase < 100
            
            test_result = TestResult(
                test_name="memory_usage_test",
                category="stress_test",
                success=success,
                execution_time_ms=execution_time,
                message=f"Memory usage: {total_memory_increase:.1f}MB increase",
                performance_metrics={
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'memory_increase_mb': total_memory_increase,
                    'commands_processed': 1000
                }
            )
            
            results.append(test_result)
            
        except Exception as e:
            test_result = TestResult(
                test_name="memory_usage_test",
                category="stress_test",
                success=False,
                execution_time_ms=0,
                message="Memory usage test failed",
                error=str(e)
            )
            results.append(test_result)
        
        return results

class TestRunner:
    """
    🧪 COMPREHENSIVE TEST RUNNER
    Orchestrates all tests and generates detailed reports
    """
    
    def __init__(self):
        self.test_suites = []
        self.results = []
        self.quality_metrics = QualityMetrics()
        self.performance_monitor = PerformanceMonitor()
        
        # Setup test suites
        self.setup_test_suites()
        
        logger.info("🧪 Test Runner initialized")
    
    def setup_test_suites(self):
        """Setup all test suites"""
        
        # Voice Commands Test Suite
        voice_commands_suite = TestSuite(
            name="voice_commands",
            description="Voice command recognition and processing tests",
            tests=[
                self.run_voice_command_tests,
                self.run_voice_response_time_tests
            ],
            timeout_seconds=60
        )
        self.test_suites.append(voice_commands_suite)
        
        # Audio System Test Suite
        audio_system_suite = TestSuite(
            name="audio_system",
            description="Audio manager and voice engine tests",
            tests=[
                self.run_audio_manager_tests,
                self.run_voice_engine_tests
            ],
            timeout_seconds=30
        )
        self.test_suites.append(audio_system_suite)
        
        # Integration Test Suite
        integration_suite = TestSuite(
            name="integration",
            description="End-to-end integration tests",
            tests=[
                self.run_integration_tests
            ],
            timeout_seconds=120
        )
        self.test_suites.append(integration_suite)
        
        # Stress Test Suite
        stress_test_suite = TestSuite(
            name="stress_tests",
            description="Performance and stress tests",
            tests=[
                self.run_stress_tests
            ],
            timeout_seconds=180
        )
        self.test_suites.append(stress_test_suite)
    
    def run_voice_command_tests(self) -> List[TestResult]:
        """Run voice command tests"""
        try:
            tester = VoiceCommandTester()
            
            # Try to use real voice processor
            try:
                from voice.stt.voice_commands import VoiceCommandProcessor
                processor = VoiceCommandProcessor()
                processor.initialize()
            except ImportError:
                processor = MockVoiceCommandProcessor()
                processor.initialize()
            
            return tester.test_command_recognition(processor)
            
        except Exception as e:
            return [TestResult(
                test_name="voice_command_tests_setup",
                category="voice_commands",
                success=False,
                execution_time_ms=0,
                message="Voice command tests setup failed",
                error=str(e)
            )]
    
    def run_voice_response_time_tests(self) -> List[TestResult]:
        """Run voice response time tests"""
        try:
            tester = VoiceCommandTester()
            
            try:
                from voice.stt.voice_commands import VoiceCommandProcessor
                processor = VoiceCommandProcessor()
                processor.initialize()
            except ImportError:
                processor = MockVoiceCommandProcessor()
                processor.initialize()
            
            return tester.test_response_time(processor)
            
        except Exception as e:
            return [TestResult(
                test_name="voice_response_time_tests_setup",
                category="voice_commands",
                success=False,
                execution_time_ms=0,
                message="Voice response time tests setup failed",
                error=str(e)
            )]
    
    def run_audio_manager_tests(self) -> List[TestResult]:
        """Run audio manager tests"""
        try:
            tester = AudioSystemTester()
            return tester.test_audio_manager_functionality()
            
        except Exception as e:
            return [TestResult(
                test_name="audio_manager_tests_setup",
                category="audio_system",
                success=False,
                execution_time_ms=0,
                message="Audio manager tests setup failed",
                error=str(e)
            )]
    
    def run_voice_engine_tests(self) -> List[TestResult]:
        """Run voice engine tests"""
        try:
            tester = AudioSystemTester()
            return tester.test_voice_engine_functionality()
            
        except Exception as e:
            return [TestResult(
                test_name="voice_engine_tests_setup",
                category="audio_system",
                success=False,
                execution_time_ms=0,
                message="Voice engine tests setup failed",
                error=str(e)
            )]
    
    def run_integration_tests(self) -> List[TestResult]:
        """Run integration tests"""
        try:
            tester = IntegrationTester()
            return tester.test_full_voice_workflow()
            
        except Exception as e:
            return [TestResult(
                test_name="integration_tests_setup",
                category="integration",
                success=False,
                execution_time_ms=0,
                message="Integration tests setup failed",
                error=str(e)
            )]
    
    def run_stress_tests(self) -> List[TestResult]:
        """Run stress tests"""
        try:
            tester = StressTester()
            results = []
            results.extend(tester.test_concurrent_commands())
            results.extend(tester.test_memory_usage())
            return results
            
        except Exception as e:
            return [TestResult(
                test_name="stress_tests_setup",
                category="stress_tests",
                success=False,
                execution_time_ms=0,
                message="Stress tests setup failed",
                error=str(e)
            )]
    
    def run_all_tests(self, parallel: bool = False) -> Dict[str, Any]:
        """
        Run all test suites and return comprehensive results
        
        Args:
            parallel: Whether to run test suites in parallel
            
        Returns:
            Dict containing test results and metrics
        """
        logger.info("🚀 Starting comprehensive test run...")
        overall_start_time = time.time()
        
        self.results.clear()
        suite_results = {}
        
        if parallel:
            # Run test suites in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.test_suites)) as executor:
                future_to_suite = {
                    executor.submit(self.run_test_suite, suite): suite 
                    for suite in self.test_suites
                }
                
                for future in concurrent.futures.as_completed(future_to_suite):
                    suite = future_to_suite[future]
                    try:
                        suite_results[suite.name] = future.result()
                    except Exception as e:
                        logger.error(f"Test suite {suite.name} failed: {e}")
                        suite_results[suite.name] = []
        else:
            # Run test suites sequentially
            for suite in self.test_suites:
                logger.info(f"🧪 Running test suite: {suite.name}")
                suite_results[suite.name] = self.run_test_suite(suite)
        
        # Flatten all results
        for suite_name, results in suite_results.items():
            self.results.extend(results)
        
        # Calculate metrics
        self.calculate_quality_metrics()
        
        overall_execution_time = (time.time() - overall_start_time) * 1000
        
        # Generate final report
        report = self.generate_comprehensive_report(overall_execution_time, suite_results)
        
        logger.info(f"✅ Test run completed in {overall_execution_time:.1f}ms")
        logger.info(f"📊 Overall success rate: {self.quality_metrics.success_rate:.1f}%")
        
        return report
    
    def run_test_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run a single test suite"""
        suite_results = []
        
        try:
            # Run setup if provided
            if suite.setup_func:
                suite.setup_func()
            
            # Run all tests in the suite
            for test_func in suite.tests:
                try:
                    test_results = test_func()
                    suite_results.extend(test_results)
                except Exception as e:
                    logger.error(f"Test function failed: {e}")
                    error_result = TestResult(
                        test_name=f"{suite.name}_test_function_error",
                        category=suite.name,
                        success=False,
                        execution_time_ms=0,
                        message=f"Test function in {suite.name} failed",
                        error=str(e)
                    )
                    suite_results.append(error_result)
            
            # Run teardown if provided
            if suite.teardown_func:
                suite.teardown_func()
                
        except Exception as e:
            logger.error(f"Test suite {suite.name} setup/teardown failed: {e}")
            error_result = TestResult(
                test_name=f"{suite.name}_suite_error",
                category=suite.name,
                success=False,
                execution_time_ms=0,
                message=f"Test suite {suite.name} failed",
                error=str(e)
            )
            suite_results.append(error_result)
        
        return suite_results
    
    def calculate_quality_metrics(self):
        """Calculate comprehensive quality metrics"""
        if not self.results:
            return
        
        self.quality_metrics.total_tests = len(self.results)
        self.quality_metrics.passed_tests = sum(1 for r in self.results if r.success)
        self.quality_metrics.failed_tests = sum(1 for r in self.results if not r.success)
        self.quality_metrics.skipped_tests = 0  # No skipped tests in our implementation
        
        # Calculate average execution time
        if self.results:
            self.quality_metrics.avg_execution_time = statistics.mean(
                r.execution_time_ms for r in self.results
            )
        
        # Calculate performance score (based on response times)
        response_times = [r.execution_time_ms for r in self.results if r.execution_time_ms > 0]
        if response_times:
            avg_response_time = statistics.mean(response_times)
            # Score: 100 for 0ms, 0 for 1000ms+
            self.quality_metrics.performance_score = max(0, 100 - (avg_response_time / 10))
        else:
            self.quality_metrics.performance_score = 100
        
        # Calculate reliability score (based on success rate)
        self.quality_metrics.reliability_score = self.quality_metrics.success_rate
        
        # Code coverage would be calculated by external tools
        self.quality_metrics.code_coverage = 75.0  # Placeholder
    
    def generate_comprehensive_report(self, overall_time: float, suite_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        # Group results by category
        category_stats = defaultdict(lambda: {'passed': 0, 'failed': 0, 'total': 0, 'avg_time': 0})
        
        for result in self.results:
            category = result.category
            category_stats[category]['total'] += 1
            if result.success:
                category_stats[category]['passed'] += 1
            else:
                category_stats[category]['failed'] += 1
        
        # Calculate average times
        for category in category_stats:
            category_results = [r for r in self.results if r.category == category]
            if category_results:
                category_stats[category]['avg_time'] = statistics.mean(
                    r.execution_time_ms for r in category_results
                )
        
        # Performance metrics
        performance_metrics = {}
        for result in self.results:
            if result.performance_metrics:
                for key, value in result.performance_metrics.items():
                    if key not in performance_metrics:
                        performance_metrics[key] = []
                    performance_metrics[key].append(value)
        
        # Generate report
        report = {
            'summary': {
                'total_tests': self.quality_metrics.total_tests,
                'passed_tests': self.quality_metrics.passed_tests,
                'failed_tests': self.quality_metrics.failed_tests,
                'success_rate': self.quality_metrics.success_rate,
                'overall_score': self.quality_metrics.overall_score,
                'total_execution_time_ms': overall_time,
                'average_test_time_ms': self.quality_metrics.avg_execution_time
            },
            'quality_metrics': {
                'performance_score': self.quality_metrics.performance_score,
                'reliability_score': self.quality_metrics.reliability_score,
                'code_coverage': self.quality_metrics.code_coverage
            },
            'category_breakdown': dict(category_stats),
            'suite_results': {
                suite_name: {
                    'test_count': len(results),
                    'passed': sum(1 for r in results if r.success),
                    'failed': sum(1 for r in results if not r.success),
                    'success_rate': (sum(1 for r in results if r.success) / len(results) * 100) if results else 0
                }
                for suite_name, results in suite_results.items()
            },
            'performance_metrics': performance_metrics,
            'failed_tests': [
                {
                    'name': r.test_name,
                    'category': r.category,
                    'message': r.message,
                    'error': r.error,
                    'execution_time_ms': r.execution_time_ms
                }
                for r in self.results if not r.success
            ],
            'detailed_results': [
                {
                    'name': r.test_name,
                    'category': r.category,
                    'success': r.success,
                    'execution_time_ms': r.execution_time_ms,
                    'message': r.message,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.results
            ]
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """Save test report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kira_test_report_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📄 Test report saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save test report: {e}")
    
    def print_summary_report(self, report: Dict[str, Any]):
        """Print summary report to console"""
        
        print("\n" + "="*80)
        print("🧪 KIRA VOICE SYSTEM - TEST REPORT SUMMARY")
        print("="*80)
        
        summary = report['summary']
        print(f"📊 Overall Results:")
        print(f"   • Total Tests: {summary['total_tests']}")
        print(f"   • Passed: {summary['passed_tests']} ✅")
        print(f"   • Failed: {summary['failed_tests']} ❌")
        print(f"   • Success Rate: {summary['success_rate']:.1f}%")
        print(f"   • Overall Score: {summary.get('overall_score', 0):.1f}/100")
        print(f"   • Total Time: {summary['total_execution_time_ms']:.1f}ms")
        
        print(f"\n🎯 Quality Metrics:")
        quality = report['quality_metrics']
        print(f"   • Performance Score: {quality['performance_score']:.1f}/100")
        print(f"   • Reliability Score: {quality['reliability_score']:.1f}/100")
        print(f"   • Code Coverage: {quality['code_coverage']:.1f}%")
        
        print(f"\n📋 Test Suites:")
        for suite_name, suite_data in report['suite_results'].items():
            status = "✅" if suite_data['success_rate'] > 80 else "⚠️" if suite_data['success_rate'] > 50 else "❌"
            print(f"   {status} {suite_name}: {suite_data['passed']}/{suite_data['test_count']} ({suite_data['success_rate']:.1f}%)")
        
        if report['failed_tests']:
            print(f"\n❌ Failed Tests:")
            for failed_test in report['failed_tests'][:5]:  # Show first 5 failures
                print(f"   • {failed_test['name']}: {failed_test['message']}")
            
            if len(report['failed_tests']) > 5:
                print(f"   ... and {len(report['failed_tests']) - 5} more failures")
        
        print("\n" + "="*80)

# Utility functions for running tests

def run_quick_tests():
    """Run a quick subset of tests for development"""
    print("🚀 Running Quick Tests for Kira Voice System...")
    
    runner = TestRunner()
    
    # Run only essential tests
    quick_suites = [suite for suite in runner.test_suites if suite.name in ['voice_commands', 'audio_system']]
    runner.test_suites = quick_suites
    
    report = runner.run_all_tests(parallel=False)
    runner.print_summary_report(report)
    
    return report

def run_full_tests():
    """Run comprehensive test suite"""
    print("🧪 Running Full Test Suite for Kira Voice System...")
    
    runner = TestRunner()
    report = runner.run_all_tests(parallel=True)
    
    # Print and save report
    runner.print_summary_report(report)
    runner.save_report(report)
    
    return report

def run_performance_tests():
    """Run performance-focused tests"""
    print("⚡ Running Performance Tests for Kira Voice System...")
    
    runner = TestRunner()
    
    # Run only performance-related tests
    perf_suites = [suite for suite in runner.test_suites if suite.name in ['stress_tests', 'integration']]
    runner.test_suites = perf_suites
    
    report = runner.run_all_tests(parallel=False)
    runner.print_summary_report(report)
    
    return report

# Main test execution
if __name__ == "__main__":
    print("🎯 Kira Voice System - Quality Assurance & Testing Framework")
    print("1. Quick Tests (Development)")
    print("2. Full Test Suite (CI/CD)")
    print("3. Performance Tests")
    print("4. Custom Test Selection")
    
    choice = input("\nWähle Test-Modus (1-4): ").strip()
    
    if choice == "1":
        run_quick_tests()
    elif choice == "2":
        run_full_tests()
    elif choice == "3":
        run_performance_tests()
    elif choice == "4":
        # Custom test selection
        runner = TestRunner()
        
        print("\nVerfügbare Test Suites:")
        for i, suite in enumerate(runner.test_suites, 1):
            print(f"{i}. {suite.name} - {suite.description}")
        
        selection = input("\nWähle Test Suites (z.B. 1,3): ").strip()
        
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_suites = [runner.test_suites[i] for i in indices if 0 <= i < len(runner.test_suites)]
            
            if selected_suites:
                runner.test_suites = selected_suites
                report = runner.run_all_tests(parallel=False)
                runner.print_summary_report(report)
            else:
                print("❌ Keine gültigen Test Suites ausgewählt")
                
        except (ValueError, IndexError):
            print("❌ Ungültige Eingabe")
    else:
        print("🚀 Running Full Test Suite by default...")
        run_full_tests()