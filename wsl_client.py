"""
🐧 WSL VOICE CLIENT - COMPLETE FIXED VERSION
Verbindet sich zur Windows Voice Bridge
"""

import socket
import struct
import numpy as np
import time
import threading
import queue
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WSLVoiceClient:
    """WSL Voice Client für Windows Bridge - COMPLETE FIXED"""
    
    def __init__(self, bridge_host: str = None, bridge_port: int = 7777):
        # Auto-detect Windows Host IP
        if bridge_host is None:
            bridge_host = self._get_windows_ip()
        
        self.bridge_host = bridge_host
        self.bridge_port = bridge_port
        self.socket = None
        self.connected = False
        
        # Audio Config (muss mit Bridge übereinstimmen)
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.audio_format = np.float32
        
        # Audio Queues
        self.input_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=100)
        
        # Connection management
        self._receiver_thread = None
        self._keep_alive = False
        
        # ✅ FIX: Add connection state tracking
        self._last_heartbeat = time.time()
        self._connection_stable = False
    
    def _get_windows_ip(self) -> str:
        """Get Windows Host IP from WSL - IMPROVED"""
        try:
            import subprocess
            
            # Check if we're actually in WSL
            if os.path.exists('/proc/version'):
                with open('/proc/version', 'r') as f:
                    if 'microsoft' in f.read().lower():
                        # We're in WSL - use Linux commands
                        result = subprocess.run(
                            "ip route | grep default | awk '{print $3}'",
                            shell=True,
                            capture_output=True,
                            text=True
                        )
                        windows_ip = result.stdout.strip()
                        if windows_ip:
                            logger.info(f"🖥️ Windows IP detected: {windows_ip}")
                            return windows_ip
            
            # Fallback for non-WSL or if detection failed
            logger.warning("⚠️ Not in WSL or IP detection failed, using localhost")
            return "localhost"
            
        except Exception as e:
            logger.error(f"❌ IP detection error: {e}")
            return "localhost"
    
    def connect(self) -> bool:
        """Verbinde zur Voice Bridge - ROBUST CONNECTION WITH DEVICE TEST"""
        try:
            logger.info(f"🔌 Connecting to Windows Voice Bridge...")
            logger.info(f"   Host: {self.bridge_host}")
            logger.info(f"   Port: {self.bridge_port}")
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)
            
            # ✅ FIX: Add socket options for better connection
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            self.socket.connect((self.bridge_host, self.bridge_port))
            
            self.connected = True
            self._keep_alive = True
            self._connection_stable = False
            self._last_heartbeat = time.time()
            
            logger.info(f"✅ Connected to Voice Bridge: {self.bridge_host}:{self.bridge_port}")
            
            # ✅ NEW: Test audio device availability
            logger.info("🎧 Testing audio device availability...")
            if not self._test_audio_devices():
                logger.error("❌ No audio devices available through bridge!")
                self.disconnect()
                return False
            
            # ✅ FIX: Wait a bit before starting receiver
            time.sleep(0.5)
            
            # Starte Receiver Thread für PERSISTENT connection
            self._receiver_thread = threading.Thread(
                target=self._receive_audio_loop, 
                daemon=True,
                name="WSLAudioReceiver"
            )
            self._receiver_thread.start()
            
            # ✅ FIX: Wait for connection to stabilize
            time.sleep(1.0)
            self._connection_stable = True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection Error: {e}")
            self.connected = False
            return False
        
    def _test_audio_devices(self) -> bool:
        """Test if Windows Bridge provides audio devices"""
        try:
            # Send device info request
            self.socket.settimeout(5.0)
            
            # Request Type 4 = Device Info
            request = struct.pack('!II', 0, 4)  # Type 4 = Request Device Info
            self.socket.sendall(request)
            
            # Receive response
            header_data = self._receive_exactly(8)
            if not header_data:
                logger.error("❌ No response to device info request")
                return False
            
            data_length, data_type = struct.unpack('!II', header_data)
            
            if data_type == 5:  # Device Info Response
                if data_length > 0:
                    device_data = self._receive_exactly(data_length)
                    if device_data:
                        # Parse device info (simple format: num_input, num_output)
                        if len(device_data) >= 8:
                            num_input, num_output = struct.unpack('!II', device_data[:8])
                            logger.info(f"🎧 Bridge reports: {num_input} input devices, {num_output} output devices")
                            
                            if num_input > 0 or num_output > 0:
                                return True
                            else:
                                logger.warning("⚠️ Bridge has no audio devices!")
                                return False
                else:
                    logger.warning("⚠️ Empty device info response")
                    return False
            else:
                logger.warning(f"⚠️ Unexpected device info response type: {data_type}")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Audio device test failed: {e}")
            return False

    def _receive_exactly(self, num_bytes: int) -> bytes:
        """Receive exactly num_bytes from socket"""
        data = b''
        while len(data) < num_bytes:
            chunk = self.socket.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Socket closed during receive")
            data += chunk
        return data

    def get_audio_devices(self) -> dict:
        """Get audio devices from Windows Bridge"""
        try:
            if not self.connected:
                return {"input_devices": [], "output_devices": []}
            
            # Send device list request
            self.socket.settimeout(3.0)
            request = struct.pack('!II', 0, 6)  # Type 6 = Request Device List
            self.socket.sendall(request)
            
            # Receive response
            header_data = self._receive_exactly(8)
            if not header_data:
                return {"input_devices": [], "output_devices": []}
            
            data_length, data_type = struct.unpack('!II', header_data)
            
            if data_type == 7 and data_length > 0:  # Device List Response
                device_data = self._receive_exactly(data_length)
                
                # Parse device list (JSON format)
                import json
                try:
                    devices = json.loads(device_data.decode('utf-8'))
                    return devices
                except:
                    return {"input_devices": [], "output_devices": []}
            
            return {"input_devices": [], "output_devices": []}
            
        except Exception as e:
            logger.error(f"❌ Get audio devices failed: {e}")
            return {"input_devices": [], "output_devices": []}

    def _receive_audio_loop(self):
        """Empfange Audio Daten von Bridge - ROBUST LOOP"""
        logger.info("🔄 WSL Audio Receiver started")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.connected and self._keep_alive:
            try:
                # ✅ FIX: Wait for connection to stabilize first
                if not self._connection_stable:
                    time.sleep(0.1)
                    continue
                
                # ✅ FIX: Send heartbeat/request more carefully
                self.socket.settimeout(2.0)  # 2 second timeout
                
                # Send request for input data
                request = struct.pack('!II', 0, 2)  # Type 2 = Request Input
                self.socket.sendall(request)  # ✅ FIX: Use sendall instead of send
                
                # ✅ FIX: Receive header with proper error handling
                header_data = b''
                while len(header_data) < 8:
                    chunk = self.socket.recv(8 - len(header_data))
                    if not chunk:
                        raise ConnectionError("Socket closed during header receive")
                    header_data += chunk
                
                if len(header_data) == 8:
                    data_length, data_type = struct.unpack('!II', header_data)
                    
                    # ✅ FIX: Validate data length
                    if data_length > 0 and data_length < 1024 * 1024:  # Max 1MB
                        if data_type == 3:  # Input Data
                            # ✅ FIX: Receive audio data properly
                            audio_data = b''
                            while len(audio_data) < data_length:
                                chunk = self.socket.recv(min(4096, data_length - len(audio_data)))
                                if not chunk:
                                    raise ConnectionError("Socket closed during audio receive")
                                audio_data += chunk
                            
                            if len(audio_data) == data_length:
                                # Convert to numpy array
                                audio_array = np.frombuffer(audio_data, dtype=self.audio_format)
                                
                                # Put in input queue
                                try:
                                    self.input_queue.put_nowait(audio_array)
                                    consecutive_errors = 0  # ✅ Reset error counter
                                except queue.Full:
                                    # Remove oldest item and add new one
                                    try:
                                        self.input_queue.get_nowait()
                                        self.input_queue.put_nowait(audio_array)
                                    except queue.Empty:
                                        pass
                
                self._last_heartbeat = time.time()
                time.sleep(0.05)  # ✅ FIX: 50ms delay instead of 10ms
                
            except socket.timeout:
                # ✅ FIX: Timeout handling - less aggressive
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    logger.warning(f"⚠️ Too many consecutive timeouts ({consecutive_errors})")
                    break
                continue
                
            except (ConnectionError, ConnectionResetError, ConnectionAbortedError) as e:
                logger.error(f"❌ Connection lost: {e}")
                break
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"❌ Receive Error: {e}")
                
                if consecutive_errors > max_consecutive_errors:
                    logger.error(f"❌ Too many consecutive errors ({consecutive_errors}), stopping")
                    break
                
                time.sleep(0.1)  # Brief pause before retry
        
        self.connected = False
        logger.info("🛑 WSL Audio Receiver stopped")
    
    def play_audio(self, audio_data: np.ndarray) -> bool:
        """Spiele Audio über Windows Lautsprecher - ROBUST"""
        if not self.connected or not self._connection_stable:
            logger.error("❌ Not connected to bridge or connection not stable")
            return False
        
        try:
            # ✅ FIX: Ensure audio data is in correct format
            if audio_data.dtype != self.audio_format:
                audio_data = audio_data.astype(self.audio_format)
            
            # ✅ FIX: Flatten array to ensure 1D
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            audio_bytes = audio_data.tobytes()
            
            # ✅ FIX: Validate data size
            if len(audio_bytes) > 1024 * 1024:  # Max 1MB
                logger.error("❌ Audio data too large")
                return False
            
            header = struct.pack('!II', len(audio_bytes), 1)  # Type 1 = Output Data
            
            # ✅ FIX: Send with timeout and error handling
            self.socket.settimeout(5.0)
            self.socket.sendall(header + audio_bytes)
            
            logger.debug("🔊 Audio sent to Windows speakers")
            return True
            
        except Exception as e:
            logger.error(f"❌ Play Audio Error: {e}")
            return False
    
    def get_microphone_data(self, timeout: float = 0.1) -> np.ndarray:
        """Hole Mikrofon Daten von Windows"""
        try:
            audio_data = self.input_queue.get(timeout=timeout)
            return audio_data
        except queue.Empty:
            return None
    
    def is_connected(self) -> bool:
        """Check if connected and stable"""
        return self.connected and self._connection_stable
    
    def get_status(self) -> dict:
        """Get client status"""
        return {
            'connected': self.connected,
            'connection_stable': self._connection_stable,
            'bridge_host': self.bridge_host,
            'bridge_port': self.bridge_port,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'receiver_thread_alive': self._receiver_thread.is_alive() if self._receiver_thread else False,
            'last_heartbeat': time.time() - self._last_heartbeat
        }
    
    def disconnect(self):
        """Trenne Verbindung - GRACEFUL SHUTDOWN"""
        logger.info("🛑 Disconnecting from Voice Bridge...")
        
        self._keep_alive = False
        self.connected = False
        self._connection_stable = False
        
        # Wait for receiver thread to stop
        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=3.0)
        
        if self.socket:
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except:
                pass
        
        logger.info("✅ WSL Voice Client disconnected")

# ========================================
# 🧪 IMPROVED TEST FUNCTIONS
# ========================================

def test_connection_stability():
    """Test connection stability over time"""
    print("🔌 WSL VOICE CLIENT - CONNECTION STABILITY TEST")
    print("=" * 60)
    
    client = WSLVoiceClient()
    
    if client.connect():
        print("✅ Initial connection successful!")
        
        # Monitor connection for 30 seconds
        start_time = time.time()
        last_status_time = 0
        
        try:
            while time.time() - start_time < 30.0:
                current_time = time.time()
                
                # Print status every 5 seconds
                if current_time - last_status_time > 5.0:
                    status = client.get_status()
                    print(f"📊 Status: Connected={status['connected']}, "
                          f"Stable={status['connection_stable']}, "
                          f"Queue={status['input_queue_size']}, "
                          f"Heartbeat={status['last_heartbeat']:.1f}s ago")
                    last_status_time = current_time
                
                # Check if still connected
                if not client.is_connected():
                    print("❌ Connection lost!")
                    break
                
                time.sleep(1.0)
            
            print("✅ Connection stability test completed")
            
        except KeyboardInterrupt:
            print("⚠️ Test interrupted by user")
        
        finally:
            client.disconnect()
    else:
        print("❌ Initial connection failed!")

def test_simple_connection():
    """Simple connection test without audio processing"""
    print("🔌 WSL VOICE CLIENT - SIMPLE CONNECTION TEST")
    print("=" * 50)
    
    client = WSLVoiceClient()
    
    if client.connect():
        print("✅ Connection successful!")
        
        # Wait a bit and check status
        time.sleep(3.0)
        
        status = client.get_status()
        print(f"📊 Final Status: {status}")
        
        if client.is_connected():
            print("✅ Connection remains stable!")
        else:
            print("❌ Connection became unstable")
        
        client.disconnect()
        return True
    else:
        print("❌ Connection failed!")
        return False

# ========================================
# 🚀 MAIN FUNCTION
# ========================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "stability":
            test_connection_stability()
        elif sys.argv[1] == "simple":
            test_simple_connection()
        elif sys.argv[1] == "connect":
            client = WSLVoiceClient()
            if client.connect():
                try:
                    print("✅ Connected! Press Ctrl+C to disconnect...")
                    while client.is_connected():
                        time.sleep(1)
                except KeyboardInterrupt:
                    client.disconnect()
    else:
        print("🐧 WSL VOICE CLIENT - COMPLETE FIXED VERSION")
        print("Usage:")
        print("  python wsl_client.py simple     - Simple connection test")
        print("  python wsl_client.py stability  - Long connection test")
        print("  python wsl_client.py connect    - Manual connection")
        print("")
        print("🎯 For production use, import WSLVoiceClient class")