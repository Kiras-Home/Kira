/**
 * Modern Kira Chat Widget - Advanced Version
 * Features: Modern Design, Text-to-Speech, Voice Wake-up, Auto-Scroll
 */

class ModernKiraChatWidget {
    constructor() {
        this.isOpen = false;
        this.conversationId = null;
        this.sessionStartTime = Date.now();
        this.ttsEnabled = true;
        this.voiceWakeupEnabled = true;
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.isListening = false;
        this.lastKiraResponse = null;  // Store last response for audio fallback
        
        console.log('🚀 Initializing Modern Kira Chat Widget');
        this.initializeWidget();
        this.initializeVoiceFeatures();
    }

    initializeWidget() {
        this.createChatContainer();
        this.setupEventListeners();
        this.injectStyles();
        console.log('✅ Modern Kira Chat Widget initialized');
    }

    initializeVoiceFeatures() {
        // Text-to-Speech Setup
        if (this.synthesis) {
            console.log('🔊 Text-to-Speech initialized');
        }

        // Voice Wake-up Setup
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = true;
            this.recognition.interimResults = true;
            this.recognition.lang = 'de-DE';
            
            this.recognition.onresult = (event) => {
                const transcript = event.results[event.results.length - 1][0].transcript.toLowerCase();
                console.log('🎤 Voice input:', transcript);
                
                // Check for wake words
                if (transcript.includes('kira') || transcript.includes('hey kira')) {
                    this.handleVoiceWakeup(transcript);
                }
            };
            
            this.recognition.onerror = (event) => {
                console.warn('🎤 Voice recognition error:', event.error);
                if (event.error === 'not-allowed') {
                    this.voiceWakeupEnabled = false;
                    this.updateVoiceButton();
                }
            };
            
            this.recognition.onend = () => {
                if (this.voiceWakeupEnabled && this.isListening) {
                    setTimeout(() => {
                        this.startVoiceWakeup();
                    }, 1000);
                }
            };
            
            // Start listening
            this.startVoiceWakeup();
            console.log('🎤 Voice wake-up initialized');
        }
    }

    handleVoiceWakeup(transcript) {
        console.log('🎤 Voice wake-up triggered:', transcript);
        
        // Show visual feedback for activation
        this.showWakeUpFeedback();
        
        // Open chat widget if not already open
        if (!this.isOpen) {
            this.toggle();
        }
        
        // Start listening for follow-up command
        this.startListeningForCommand();
    }
    
    showWakeUpFeedback() {
        // Visual feedback for wake-up
        const toggle = document.getElementById('kiraChatToggle');
        if (toggle) {
            toggle.style.transform = 'scale(1.2)';
            toggle.style.background = 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)';
            
            setTimeout(() => {
                toggle.style.transform = 'scale(1)';
                toggle.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            }, 500);
        }
        
        console.log('✅ Kira activated - ready for command!');
    }

    sendGreetingMessage() {
        // Send greeting to backend API
        fetch('/api/voice/wake-up', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                wake_word: 'kira',
                timestamp: new Date().toISOString()
            })
        }).then(response => response.json())
        .then(data => {
            if (data.success && data.greeting) {
                // Add greeting message to chat
                this.addMessage(data.greeting, 'kira');
                // Speak the greeting
                this.speakText(data.greeting);
            } else {
                // Fallback greeting
                const greetings = [
                    'Hallo! Wie kann ich dir helfen?',
                    'Ja, ich höre zu!',
                    'Hallo, was kann ich für dich tun?',
                    'Ich bin da! Wie kann ich helfen?'
                ];
                const greeting = greetings[Math.floor(Math.random() * greetings.length)];
                this.addMessage(greeting, 'kira');
                this.speakText(greeting);
            }
        }).catch(error => {
            console.error('Voice wake-up API error:', error);
            // Fallback greeting
            const greeting = 'Hallo! Wie kann ich dir helfen?';
            this.addMessage(greeting, 'kira');
            this.speakText(greeting);
        });
    }

    startListeningForCommand() {
        // Setup continuous listening for user command
        if (this.recognition) {
            this.recognition.continuous = true;
            this.recognition.interimResults = true;
            
            let commandTimeout = setTimeout(() => {
                this.stopListeningForCommand();
                this.addMessage('Ich höre nichts mehr. Sage einfach "Kira" um mich wieder zu aktivieren.', 'kira');
                this.speakText('Ich höre nichts mehr. Sage einfach "Kira" um mich wieder zu aktivieren.');
            }, 10000); // 10 seconds timeout
            
            this.recognition.onresult = (event) => {
                const result = event.results[event.results.length - 1];
                const transcript = result[0].transcript;
                
                console.log('🎤 Command listening:', transcript);
                
                // If final result and not just wake word
                if (result.isFinal && transcript.trim().length > 0) {
                    const cleanTranscript = transcript.toLowerCase().trim();
                    
                    // Skip if it's just the wake word again
                    if (cleanTranscript === 'kira' || cleanTranscript === 'hey kira') {
                        return;
                    }
                    
                    clearTimeout(commandTimeout);
                    this.stopListeningForCommand();
                    
                    // Add user message and process it
                    this.addMessage(transcript, 'user');
                    this.sendToKiraAPI(transcript);
                }
            };
            
            // Update UI to show listening state
            this.showListeningIndicator();
        }
    }

    stopListeningForCommand() {
        if (this.recognition) {
            this.recognition.continuous = false;
            this.recognition.onresult = null;
        }
        this.hideListeningIndicator();
        
        // Restart wake-up listening
        setTimeout(() => {
            this.startVoiceWakeup();
        }, 1000);
    }

    showListeningIndicator() {
        const indicator = document.getElementById('voiceListeningIndicator');
        if (!indicator) {
            const chatMessages = document.getElementById('chatMessages');
            if (chatMessages) {
                const listenerDiv = document.createElement('div');
                listenerDiv.id = 'voiceListeningIndicator';
                listenerDiv.className = 'voice-listening-indicator';
                listenerDiv.innerHTML = `
                    <div class="message-avatar">🎤</div>
                    <div class="listening-animation">
                        <div class="listening-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                        <div class="listening-text">Ich höre zu...</div>
                    </div>
                `;
                chatMessages.appendChild(listenerDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
    }

    hideListeningIndicator() {
        const indicator = document.getElementById('voiceListeningIndicator');
        if (indicator) {
            indicator.remove();
        }
    }

    createChatContainer() {
        if (!document.getElementById('kiraChatWidget')) {
            const chatWidget = document.createElement('div');
            chatWidget.id = 'kiraChatWidget';
            chatWidget.className = 'kira-chat-widget';
            chatWidget.innerHTML = `
                <button class="kira-chat-toggle" id="kiraChatToggle" onclick="window.kiraChat.toggle()">
                    <span class="toggle-icon">💬</span>
                    <span class="notification-badge" id="notificationBadge">1</span>
                </button>
                <div class="kira-chat-window" id="kiraChatWindow">
                    <div class="chat-header">
                        <div class="chat-avatar">
                            <div class="avatar-img">🤖</div>
                            <div class="status-indicator online"></div>
                        </div>
                        <div class="chat-info">
                            <h4 class="chat-title">Kira Assistant</h4>
                            <p class="chat-status">Online • Bereit zu helfen</p>
                        </div>
                        <div class="chat-controls">
                            <button class="control-btn" id="ttsToggle" onclick="window.kiraChat.toggleTTS()" title="Text-to-Speech">🔊</button>
                            <button class="control-btn" id="voiceToggle" onclick="window.kiraChat.toggleVoiceWakeup()" title="Voice Wake-up">🎤</button>
                            <button class="control-btn minimize-btn" onclick="window.kiraChat.minimize()" title="Minimieren">−</button>
                            <button class="control-btn close-btn" onclick="window.kiraChat.close()" title="Schließen">✕</button>
                        </div>
                    </div>
                    <div class="chat-content">
                        <div class="chat-messages" id="chatMessages">
                            <div class="message kira-message">
                                <div class="message-avatar">🤖</div>
                                <div class="message-content">
                                    <div class="message-bubble">
                                        Hallo! Ich bin Kira, dein KI-Assistent. Sage "Kira" oder "Hey Kira" um mich zu aktivieren!
                                    </div>
                                    <div class="message-time">jetzt</div>
                                </div>
                            </div>
                        </div>
                        <div class="typing-indicator" id="typingIndicator" style="display: none;">
                            <div class="message-avatar">🤖</div>
                            <div class="typing-dots">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    </div>
                    <div class="chat-input">
                        <div class="input-container">
                            <input type="text" id="kiraChatInput" class="chat-input-field" placeholder="Nachricht eingeben..." onkeypress="handleChatKeyPress(event)">
                            <button class="voice-input-btn" onclick="window.kiraChat.startVoiceInput()" title="Spracheingabe">🎙️</button>
                            <button class="send-btn" onclick="window.kiraChat.sendMessage()" title="Senden">📤</button>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(chatWidget);
            console.log('💬 Modern chat container created');
        }
    }

    setupEventListeners() {
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'k') {
                e.preventDefault();
                this.toggle();
            }
        });

        // Auto-hide notification badge when chat opens
        const toggle = document.getElementById('kiraChatToggle');
        if (toggle) {
            toggle.addEventListener('click', () => {
                const badge = document.getElementById('notificationBadge');
                if (badge) {
                    badge.style.display = 'none';
                }
            });
        }
    }

    injectStyles() {
        const styleId = 'kira-chat-styles';
        if (!document.getElementById(styleId)) {
            const style = document.createElement('style');
            style.id = styleId;
            style.textContent = `
                .kira-chat-widget {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    z-index: 10000;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }

                .kira-chat-toggle {
                    width: 60px;
                    height: 60px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border: none;
                    border-radius: 50%;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
                    position: relative;
                }

                .kira-chat-toggle:hover {
                    transform: scale(1.05);
                    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
                }

                .kira-chat-toggle.active {
                    border-radius: 15px;
                    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
                }

                .toggle-icon {
                    font-size: 24px;
                    color: white;
                }

                .notification-badge {
                    position: absolute;
                    top: -5px;
                    right: -5px;
                    background: #ff4757;
                    color: white;
                    border-radius: 50%;
                    width: 20px;
                    height: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 12px;
                    font-weight: bold;
                    animation: pulse 2s infinite;
                }

                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                    100% { transform: scale(1); }
                }

                .kira-chat-window {
                    position: absolute;
                    bottom: 80px;
                    right: 0;
                    width: 400px;
                    height: 600px;
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
                    border: 1px solid #e0e0e0;
                    display: none;
                    flex-direction: column;
                    overflow: hidden;
                    transform: translateY(20px);
                    opacity: 0;
                    transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                }

                .kira-chat-window.open {
                    display: flex;
                    transform: translateY(0);
                    opacity: 1;
                }

                .chat-header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    display: flex;
                    align-items: center;
                    gap: 15px;
                }

                .chat-avatar {
                    position: relative;
                }

                .avatar-img {
                    width: 45px;
                    height: 45px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 20px;
                    border: 2px solid rgba(255, 255, 255, 0.3);
                }

                .status-indicator {
                    position: absolute;
                    bottom: 0;
                    right: 0;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    border: 2px solid white;
                }

                .status-indicator.online {
                    background: #2ed573;
                }

                .chat-info {
                    flex: 1;
                }

                .chat-title {
                    margin: 0 0 5px 0;
                    font-size: 16px;
                    font-weight: 600;
                }

                .chat-status {
                    margin: 0;
                    font-size: 12px;
                    opacity: 0.9;
                }

                .chat-controls {
                    display: flex;
                    gap: 8px;
                }

                .control-btn {
                    width: 32px;
                    height: 32px;
                    border: none;
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 14px;
                }

                .control-btn:hover {
                    background: rgba(255, 255, 255, 0.3);
                    transform: scale(1.05);
                }

                .chat-content {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                }

                .chat-messages {
                    flex: 1;
                    padding: 20px;
                    overflow-y: auto;
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                    scroll-behavior: smooth;
                }

                .chat-messages::-webkit-scrollbar {
                    width: 6px;
                }

                .chat-messages::-webkit-scrollbar-track {
                    background: #f1f1f1;
                    border-radius: 10px;
                }

                .chat-messages::-webkit-scrollbar-thumb {
                    background: #c1c1c1;
                    border-radius: 10px;
                }

                .chat-messages::-webkit-scrollbar-thumb:hover {
                    background: #a8a8a8;
                }

                .message {
                    display: flex;
                    gap: 12px;
                    animation: slideInUp 0.3s ease;
                }

                .message.user-message {
                    flex-direction: row-reverse;
                }

                @keyframes slideInUp {
                    from {
                        opacity: 0;
                        transform: translateY(20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }

                .message-avatar {
                    width: 35px;
                    height: 35px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 16px;
                    flex-shrink: 0;
                }

                .message-content {
                    max-width: 85%;
                    display: flex;
                    flex-direction: column;
                    gap: 5px;
                    min-width: 0;
                }

                .message.user-message .message-content {
                    align-items: flex-end;
                }

                .message-bubble {
                    padding: 12px 16px;
                    border-radius: 18px;
                    font-size: 14px;
                    line-height: 1.6;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    white-space: pre-wrap;
                    max-width: 100%;
                    box-sizing: border-box;
                }

                .message.kira-message .message-bubble {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-bottom-left-radius: 6px;
                }

                .message.user-message .message-bubble {
                    background: #f0f0f0;
                    color: #333;
                    border-bottom-right-radius: 6px;
                }

                .message-time {
                    font-size: 11px;
                    color: #999;
                    margin-left: 8px;
                }

                .message.user-message .message-time {
                    text-align: right;
                    margin-left: 0;
                    margin-right: 8px;
                }

                .typing-indicator {
                    display: flex;
                    gap: 12px;
                    align-items: center;
                    padding: 0 20px;
                }

                .typing-dots {
                    display: flex;
                    gap: 4px;
                    padding: 12px 16px;
                    background: #f0f0f0;
                    border-radius: 18px;
                    border-bottom-left-radius: 6px;
                }

                .typing-dots span {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #999;
                    animation: typing 1.5s infinite;
                }

                .typing-dots span:nth-child(2) {
                    animation-delay: 0.2s;
                }

                .typing-dots span:nth-child(3) {
                    animation-delay: 0.4s;
                }

                @keyframes typing {
                    0%, 60%, 100% { transform: scale(1); opacity: 0.5; }
                    30% { transform: scale(1.2); opacity: 1; }
                }

                .chat-input {
                    padding: 20px;
                    background: #f9f9f9;
                    border-top: 1px solid #e0e0e0;
                }

                .input-container {
                    display: flex;
                    gap: 10px;
                    align-items: center;
                }

                .chat-input-field {
                    flex: 1;
                    padding: 12px 16px;
                    border: 2px solid #e0e0e0;
                    border-radius: 25px;
                    font-size: 14px;
                    outline: none;
                    transition: all 0.2s ease;
                }

                .chat-input-field:focus {
                    border-color: #667eea;
                    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                }

                .voice-input-btn,
                .send-btn {
                    width: 40px;
                    height: 40px;
                    border: none;
                    border-radius: 50%;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.2s ease;
                    font-size: 16px;
                }

                .voice-input-btn {
                    background: #f0f0f0;
                    color: #666;
                }

                .voice-input-btn:hover {
                    background: #e0e0e0;
                    transform: scale(1.05);
                }

                .voice-input-btn.active {
                    background: #ff4757;
                    color: white;
                    animation: pulse 1s infinite;
                }

                .send-btn {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }

                .send-btn:hover {
                    transform: scale(1.05);
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                }

                .send-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                    transform: none;
                }

                .voice-listening-indicator {
                    display: flex;
                    gap: 12px;
                    align-items: center;
                    padding: 0 20px;
                    margin-bottom: 10px;
                }

                .listening-animation {
                    display: flex;
                    flex-direction: column;
                    gap: 5px;
                }

                .listening-dots {
                    display: flex;
                    gap: 4px;
                    padding: 12px 16px;
                    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                    border-radius: 18px;
                    border-bottom-left-radius: 6px;
                }

                .listening-dots span {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: white;
                    animation: listeningPulse 1.5s infinite;
                }

                .listening-dots span:nth-child(2) {
                    animation-delay: 0.2s;
                }

                .listening-dots span:nth-child(3) {
                    animation-delay: 0.4s;
                }

                @keyframes listeningPulse {
                    0%, 60%, 100% { transform: scale(1); opacity: 0.7; }
                    30% { transform: scale(1.3); opacity: 1; }
                }

                .listening-text {
                    font-size: 11px;
                    color: #ff6b6b;
                    margin-left: 8px;
                    font-weight: 500;
                    animation: fade 2s infinite;
                }

                @keyframes fade {
                    0%, 100% { opacity: 0.5; }
                    50% { opacity: 1; }
                }

                /* Responsive Design */
                @media (max-width: 480px) {
                    .kira-chat-window {
                        width: 100vw;
                        height: 100vh;
                        bottom: 0;
                        right: 0;
                        border-radius: 0;
                    }
                    
                    .kira-chat-widget {
                        bottom: 10px;
                        right: 10px;
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }

    toggle() {
        const chatWindow = document.getElementById('kiraChatWindow');
        const toggle = document.getElementById('kiraChatToggle');
        
        if (chatWindow && toggle) {
            this.isOpen = !this.isOpen;
            
            if (this.isOpen) {
                chatWindow.style.display = 'flex';
                setTimeout(() => {
                    chatWindow.classList.add('open');
                }, 10);
                toggle.classList.add('active');
                
                // Focus input
                setTimeout(() => {
                    const input = document.getElementById('kiraChatInput');
                    if (input) input.focus();
                }, 300);
                
                console.log('💬 Chat opened');
            } else {
                chatWindow.classList.remove('open');
                setTimeout(() => {
                    chatWindow.style.display = 'none';
                }, 300);
                toggle.classList.remove('active');
                
                console.log('💬 Chat closed');
            }
        }
    }

    close() {
        if (this.isOpen) {
            this.toggle();
        }
    }

    minimize() {
        if (this.isOpen) {
            this.toggle();
            
            // Visual feedback
            const toggle = document.getElementById('kiraChatToggle');
            if (toggle) {
                toggle.style.transform = 'scale(0.9)';
                setTimeout(() => {
                    toggle.style.transform = 'scale(1)';
                }, 200);
            }
        }
    }

    sendMessage(message = null) {
        const input = document.getElementById('kiraChatInput');
        const messageText = message || (input ? input.value.trim() : '');
        
        if (!messageText) {
            console.warn('⚠️ No message to send');
            return;
        }

        console.log('📤 Sending message:', messageText);
        
        // Add user message
        this.addMessage(messageText, 'user');
        
        // Clear input
        if (input) {
            input.value = '';
        }
        
        // Send to API
        this.sendToKiraAPI(messageText);
    }

    async sendToKiraAPI(messageText) {
        console.log('🚀 Sending to Kira API...');
        
        try {
            // Show typing indicator
            this.showTypingIndicator();
            
            const response = await fetch('/api/chat/message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: messageText,
                    conversation_id: this.conversationId || this.generateConversationId(),
                    user_name: 'WebUser',
                    session: {
                        duration_minutes: Math.floor((Date.now() - this.sessionStartTime) / 60000),
                        platform: 'web'
                    }
                })
            });

            this.hideTypingIndicator();

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            if (data.success && data.response) {
                // Add Kira's response
                this.addMessage(data.response, 'kira');
                
                // Play backend-generated audio if available
                if (data.audio && data.audio.success && data.audio.audio_url) {
                    console.log('🔊 Playing backend audio:', data.audio.audio_url);
                    this.playBackendAudio(data.audio.audio_url);
                } else {
                    // Fallback to browser TTS only if no backend audio
                    console.log('🔊 Using browser TTS fallback');
                    this.speakText(data.response);
                }
                
                // Update conversation ID
                if (data.conversation_id) {
                    this.conversationId = data.conversation_id;
                }
                
                console.log('✅ Kira response received');
            } else {
                throw new Error(data.error || 'Unknown API error');
            }
            
        } catch (error) {
            console.error('❌ Kira API error:', error);
            this.addMessage(
                `Entschuldigung, ich hatte ein Problem: ${error.message}`,
                'kira'
            );
        }
    }

    addMessage(text, sender) {
        const messages = document.getElementById('chatMessages');
        if (!messages) return;

        // Store last Kira response for audio fallback
        if (sender === 'kira') {
            this.lastKiraResponse = text;
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const isUser = sender === 'user';
        const avatar = isUser ? '👤' : '🤖';
        const time = new Date().toLocaleTimeString('de-DE', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        // Enhanced text formatting for longer messages
        const formattedText = this.formatMessageText(text);
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-bubble">${formattedText}</div>
                <div class="message-time">${time}</div>
            </div>
        `;
        
        messages.appendChild(messageDiv);
        
        // Enhanced auto-scroll with smooth animation
        this.smoothScrollToBottom(messages);
        
        console.log(`💬 Message added: ${sender} (${text.length} chars)`);
    }
    
    formatMessageText(text) {
        if (!text) return '';
        
        // Convert line breaks to proper HTML
        text = text.replace(/\n/g, '<br>');
        
        // Handle basic markdown-like formatting
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        return text;
    }
    
    smoothScrollToBottom(container) {
        // Smooth scroll to bottom
        const scrollOptions = {
            top: container.scrollHeight,
            behavior: 'smooth'
        };
        
        container.scrollTo(scrollOptions);
        
        // Fallback for older browsers
        if (container.scrollTop !== container.scrollHeight - container.clientHeight) {
            setTimeout(() => {
                container.scrollTop = container.scrollHeight;
            }, 100);
        }
    }

    showTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.style.display = 'flex';
            
            // Scroll to bottom
            const messages = document.getElementById('chatMessages');
            if (messages) {
                messages.scrollTop = messages.scrollHeight;
            }
        }
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.style.display = 'none';
        }
    }

    generateConversationId() {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        return `web_conv_${timestamp}`;
    }

    // Voice Features
    startVoiceWakeup() {
        if (this.recognition && this.voiceWakeupEnabled && !this.isListening) {
            try {
                this.recognition.start();
                this.isListening = true;
                console.log('🎤 Voice wake-up started');
            } catch (error) {
                console.warn('🎤 Voice wake-up failed:', error);
            }
        }
    }

    stopVoiceWakeup() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
            this.isListening = false;
            console.log('🎤 Voice wake-up stopped');
        }
    }

    handleVoiceWakeup(transcript) {
        console.log('🎤 Voice wake-up triggered:', transcript);
        
        // Send greeting message immediately
        this.sendGreetingMessage();
        
        // Open chat widget if not already open
        if (!this.isOpen) {
            this.toggle();
        }
        
        // Start listening for follow-up command
        this.startListeningForCommand();
    }

    speakText(text) {
        if (!this.ttsEnabled || !this.synthesis) return;
        
        // Cancel any ongoing speech
        this.synthesis.cancel();
        
        // Clean text for speech
        const cleanText = text.replace(/[🤖👤💬📤🎤🔊]/g, '').trim();
        
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.lang = 'de-DE';
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 0.8;
        
        // Find German voice
        const voices = this.synthesis.getVoices();
        const germanVoice = voices.find(voice => 
            voice.lang.startsWith('de') && voice.name.includes('Female')
        ) || voices.find(voice => voice.lang.startsWith('de'));
        
        if (germanVoice) {
            utterance.voice = germanVoice;
        }
        
        utterance.onstart = () => {
            console.log('🔊 Speaking:', cleanText.substring(0, 50) + '...');
        };
        
        this.synthesis.speak(utterance);
    }

    toggleTTS() {
        this.ttsEnabled = !this.ttsEnabled;
        console.log('🔊 TTS toggled:', this.ttsEnabled);
        
        const btn = document.getElementById('ttsToggle');
        if (btn) {
            btn.innerHTML = this.ttsEnabled ? '🔊' : '🔇';
            btn.title = this.ttsEnabled ? 'TTS deaktivieren' : 'TTS aktivieren';
        }
    }

    toggleVoiceWakeup() {
        this.voiceWakeupEnabled = !this.voiceWakeupEnabled;
        
        if (this.voiceWakeupEnabled) {
            this.startVoiceWakeup();
        } else {
            this.stopVoiceWakeup();
        }
        
        this.updateVoiceButton();
        console.log('🎤 Voice wake-up toggled:', this.voiceWakeupEnabled);
    }

    updateVoiceButton() {
        const btn = document.getElementById('voiceToggle');
        if (btn) {
            btn.style.opacity = this.voiceWakeupEnabled ? '1' : '0.5';
            btn.title = this.voiceWakeupEnabled ? 'Voice Wake-up aktiv' : 'Voice Wake-up inaktiv';
        }
    }

    startVoiceInput() {
        // TODO: Implement manual voice input
        console.log('🎙️ Manual voice input not implemented yet');
    }

    playBackendAudio(audioUrl) {
        try {
            console.log('🎵 Loading backend audio:', audioUrl);
            
            // Create audio element
            const audio = new Audio(audioUrl);
            
            // Set audio properties
            audio.volume = 0.8;
            audio.preload = 'auto';
            
            // Handle audio events
            audio.onloadstart = () => {
                console.log('⏳ Audio loading started');
            };
            
            audio.oncanplaythrough = () => {
                console.log('✅ Audio ready to play');
            };
            
            audio.onplay = () => {
                console.log('🔊 Audio playback started');
            };
            
            audio.onended = () => {
                console.log('✅ Audio playback finished');
            };
            
            audio.onerror = (e) => {
                console.error('❌ Audio playback error:', e);
                console.log('🔄 Falling back to browser TTS');
                // Fallback to browser TTS if audio fails
                this.speakText(this.lastKiraResponse || 'Audio playback failed');
            };
            
            // Play audio
            const playPromise = audio.play();
            
            if (playPromise !== undefined) {
                playPromise
                    .then(() => {
                        console.log('🎵 Audio playing successfully');
                    })
                    .catch((error) => {
                        console.error('❌ Audio play promise failed:', error);
                        console.log('🔄 Falling back to browser TTS');
                        this.speakText(this.lastKiraResponse || 'Audio playback failed');
                    });
            }
            
            return true;
            
        } catch (error) {
            console.error('❌ Backend audio error:', error);
            console.log('🔄 Falling back to browser TTS');
            this.speakText(this.lastKiraResponse || 'Audio playback failed');
            return false;
        }
    }
}

// Global functions
function handleChatKeyPress(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        if (window.kiraChat) {
            window.kiraChat.sendMessage();
        }
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 DOM loaded, initializing modern chat widget');
    window.kiraChat = new ModernKiraChatWidget();
});

// Fallback for manual initialization
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        if (!window.kiraChat) {
            window.kiraChat = new ModernKiraChatWidget();
        }
    });
} else {
    if (!window.kiraChat) {
        window.kiraChat = new ModernKiraChatWidget();
    }
}

console.log('✅ Modern Chat Widget JavaScript loaded');
