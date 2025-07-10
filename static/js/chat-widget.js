/**
 * Kira Chat Widget JavaScript - ULTRA MODERN VERSION 2.0
 * Chat-Funktionalität mit Text-to-Speech, Voice Wake-up und modernem Design
 */

class KiraChatWidget {
    constructor() {
        this.isOpen = false;
        this.isMinimized = false;
        this.container = null;
        this.conversationId = null;
        this.sessionStartTime = Date.now();
        this.messageCount = 0;
        this.ttsEnabled = true;
        this.voiceWakeupEnabled = true;
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.isListening = false;
        this.currentVoice = null;
        this.isKiraSpeaking = false;
        this.currentSpeechUtterance = null;
        this.messageQueue = [];
        this.isTyping = false;
        
        console.log('🚀 Initializing Ultra Modern Kira Chat Widget v2.0');
        this.initializeWidget();
        this.initializeVoiceFeatures();
    }

    initializeWidget() {
        this.createModernChatContainer();
        this.setupEventListeners();
        this.injectModernStyles();
        this.setupQuickActions();
        console.log('✅ Ultra Modern Kira Chat Widget v2.0 initialized');
    }

    initializeVoiceFeatures() {
        // Advanced Text-to-Speech Setup with German voice
        this.setupAdvancedTTS();
        
        // Voice Wake-up Setup
        this.setupVoiceWakeup();
        
        console.log('🔊 Advanced Voice Features initialized');
    }

    setupAdvancedTTS() {
        if (!this.synthesis) {
            console.warn('⚠️ Speech synthesis not supported');
            return;
        }

        // Wait for voices to load
        const setupVoices = () => {
            const voices = this.synthesis.getVoices();
            console.log('🔊 Available voices:', voices.length);
            
            // Find best German voice (prefer neural/premium voices)
            const germanVoices = voices.filter(voice => 
                voice.lang.startsWith('de') || 
                voice.lang.includes('DE') ||
                voice.name.toLowerCase().includes('german')
            );
            
            console.log('🇩🇪 German voices found:', germanVoices.length);
            
            // Prefer premium/neural voices
            this.currentVoice = germanVoices.find(voice => 
                voice.name.toLowerCase().includes('premium') ||
                voice.name.toLowerCase().includes('neural') ||
                voice.name.toLowerCase().includes('enhanced')
            ) || germanVoices[0] || voices[0];
            
            if (this.currentVoice) {
                console.log('🔊 Selected TTS voice:', this.currentVoice.name, this.currentVoice.lang);
            }
        };

        if (this.synthesis.getVoices().length === 0) {
            this.synthesis.addEventListener('voiceschanged', setupVoices);
        } else {
            setupVoices();
        }
    }

    setupVoiceWakeup() {
        if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
            console.warn('⚠️ Speech recognition not supported');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'de-DE';
        
        this.recognition.onresult = (event) => {
            const transcript = event.results[event.results.length - 1][0].transcript.toLowerCase();
            
            if (transcript.includes('kira') || transcript.includes('hey kira')) {
                this.handleVoiceWakeup(transcript);
            }
        };
        
        this.recognition.onerror = (event) => {
            console.warn('🎤 Voice recognition error:', event.error);
            // Auto-restart after error
            setTimeout(() => this.startVoiceWakeup(), 1000);
        };
        
        this.recognition.onend = () => {
            // Auto-restart if enabled
            if (this.voiceWakeupEnabled) {
                setTimeout(() => this.startVoiceWakeup(), 500);
            }
        };
        
        this.startVoiceWakeup();
    }

    startVoiceWakeup() {
        if (this.recognition && this.voiceWakeupEnabled) {
            try {
                this.recognition.start();
                console.log('🎤 Voice wake-up listening started');
            } catch (error) {
                console.warn('🎤 Voice wake-up start failed:', error);
            }
        }
    }

    stopVoiceWakeup() {
        if (this.recognition) {
            this.recognition.stop();
            console.log('🎤 Voice wake-up stopped');
        }
    }

    handleVoiceWakeup(transcript) {
        console.log('🎤 Voice wake-up triggered:', transcript);
        
        // Visual feedback
        this.showVoiceWakeupFeedback();
        
        // Öffne Chat-Widget
        if (!this.isOpen) {
            this.toggle();
        }
        
        // Fokussiere Input
        setTimeout(() => {
            const input = document.getElementById('kiraChatInput');
            if (input) {
                input.focus();
                input.placeholder = '🎤 Voice erkannt! Schreibe deine Nachricht...';
                setTimeout(() => {
                    input.placeholder = 'Nachricht eingeben...';
                }, 3000);
            }
        }, 300);
    }

    showVoiceWakeupFeedback() {
        const pulseRing = document.querySelector('.kira-pulse-ring');
        if (pulseRing) {
            pulseRing.style.animation = 'none';
            setTimeout(() => {
                pulseRing.style.animation = 'kira-pulse 0.5s ease-out';
            }, 10);
        }
    }

    speakText(text) {
        if (!this.ttsEnabled || !this.synthesis || !text || this.isKiraSpeaking) {
            return;
        }
        
        // Cancel any ongoing speech
        this.synthesis.cancel();
        
        // Clean text for TTS
        const cleanText = this.cleanTextForTTS(text);
        
        // Create utterance
        const utterance = new SpeechSynthesisUtterance(cleanText);
        utterance.lang = 'de-DE';
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 0.8;
        
        // Use selected German voice
        if (this.currentVoice) {
            utterance.voice = this.currentVoice;
        }
        
        // Visual feedback
        utterance.onstart = () => {
            this.isKiraSpeaking = true;
            console.log('🔊 Kira speaking:', cleanText.substring(0, 50) + '...');
            this.showSpeakingIndicator();
        };
        
        utterance.onend = () => {
            this.isKiraSpeaking = false;
            console.log('🔊 Kira finished speaking');
            this.hideSpeakingIndicator();
        };
        
        utterance.onerror = () => {
            this.isKiraSpeaking = false;
            this.hideSpeakingIndicator();
        };
        
        this.currentSpeechUtterance = utterance;
        this.synthesis.speak(utterance);
    }

    cleanTextForTTS(text) {
        return text
            .replace(/[🤖👤💬📤🔊🎤]/g, '') // Remove emojis
            .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markdown
            .replace(/\*(.*?)\*/g, '$1') // Remove italic markdown
            .replace(/```.*?```/gs, 'Code-Block') // Replace code blocks
            .replace(/`(.*?)`/g, '$1') // Remove inline code
            .trim();
    }

    showSpeakingIndicator() {
        const avatar = document.querySelector('.kira-avatar');
        if (avatar) {
            avatar.classList.add('speaking');
        }
        
        const ttsBtn = document.getElementById('ttsToggle');
        if (ttsBtn) {
            ttsBtn.innerHTML = '🔊';
            ttsBtn.classList.add('active');
        }
    }

    hideSpeakingIndicator() {
        const avatar = document.querySelector('.kira-avatar');
        if (avatar) {
            avatar.classList.remove('speaking');
        }
        
        const ttsBtn = document.getElementById('ttsToggle');
        if (ttsBtn) {
            ttsBtn.classList.remove('active');
        }
    }

    stopTTS() {
        if (this.synthesis) {
            this.synthesis.cancel();
            this.isKiraSpeaking = false;
            this.hideSpeakingIndicator();
        }
    }

    toggleTTS() {
        this.ttsEnabled = !this.ttsEnabled;
        console.log('🔊 TTS toggled:', this.ttsEnabled);
        
        const ttsBtn = document.getElementById('ttsToggle');
        if (ttsBtn) {
            ttsBtn.innerHTML = this.ttsEnabled ? '🔊' : '🔇';
            ttsBtn.title = this.ttsEnabled ? 'TTS deaktivieren' : 'TTS aktivieren';
        }
    }

    toggleVoiceWakeup() {
        this.voiceWakeupEnabled = !this.voiceWakeupEnabled;
        
        if (this.voiceWakeupEnabled) {
            this.startVoiceWakeup();
        } else {
            this.stopVoiceWakeup();
        }
        
        console.log('🎤 Voice wake-up toggled:', this.voiceWakeupEnabled);
        
        const voiceBtn = document.getElementById('voiceToggle');
        if (voiceBtn) {
            voiceBtn.innerHTML = this.voiceWakeupEnabled ? '🎤' : '🎤';
            voiceBtn.style.opacity = this.voiceWakeupEnabled ? '1' : '0.5';
            voiceBtn.title = this.voiceWakeupEnabled ? 'Voice Wake-up aktiv' : 'Voice Wake-up inaktiv';
        }
    }

    createModernChatContainer() {
        // Remove existing widget if present
        const existingWidget = document.getElementById('kiraChatWidget');
        if (existingWidget) {
            existingWidget.remove();
        }

        const chatWidget = document.createElement('div');
        chatWidget.id = 'kiraChatWidget';
        chatWidget.className = 'kira-chat-widget modern-style';
        chatWidget.innerHTML = `
            <!-- Chat Toggle Button with Pulse Ring -->
            <div class="kira-chat-toggle-container">
                <div class="kira-pulse-ring"></div>
                <button class="kira-chat-toggle modern-toggle" id="kiraChatToggle">
                    <div class="kira-avatar">
                        <div class="avatar-inner">🤖</div>
                        <div class="status-indicator online"></div>
                    </div>
                </button>
            </div>

            <!-- Modern Chat Window -->
            <div class="kira-chat-window modern-window" id="kiraChatWindow">
                <!-- Chat Header -->
                <div class="chat-header modern-header">
                    <div class="header-left">
                        <div class="kira-avatar-header">
                            <div class="avatar-inner">🤖</div>
                            <div class="status-indicator online"></div>
                        </div>
                        <div class="header-info">
                            <h4 class="kira-name">Kira Assistant</h4>
                            <div class="kira-status">Online • KI-Assistent</div>
                        </div>
                    </div>
                    <div class="header-controls">
                        <button class="control-btn" id="ttsToggle" title="Text-to-Speech">🔊</button>
                        <button class="control-btn" id="voiceToggle" title="Voice Wake-up">🎤</button>
                        <button class="control-btn minimize-btn" id="minimizeBtn" title="Minimieren">━</button>
                        <button class="control-btn close-btn" id="closeBtn" title="Schließen">✕</button>
                    </div>
                </div>

                <!-- Chat Content -->
                <div class="chat-content modern-content">
                    <!-- Messages Container -->
                    <div class="chat-messages modern-messages" id="chatMessages">
                        <!-- Welcome Message -->
                        <div class="message kira-message welcome-message">
                            <div class="message-avatar">
                                <div class="avatar-inner">🤖</div>
                            </div>
                            <div class="message-content">
                                <div class="message-bubble kira-bubble">
                                    <div class="message-text">
                                        Hallo! Ich bin <strong>Kira</strong>, dein persönlicher KI-Assistent. 
                                        Wie kann ich dir heute helfen?
                                    </div>
                                    <div class="message-time">Jetzt</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Quick Actions -->
                    <div class="quick-actions" id="quickActions">
                        <button class="quick-action" data-action="Wie ist das Wetter?">🌤️ Wetter</button>
                        <button class="quick-action" data-action="Hilfe">❓ Hilfe</button>
                        <button class="quick-action" data-action="Was kannst du?">🚀 Features</button>
                    </div>

                    <!-- Chat Input -->
                    <div class="chat-input modern-input">
                        <div class="input-container">
                            <input type="text" 
                                   id="kiraChatInput" 
                                   placeholder="Nachricht eingeben..." 
                                   class="message-input"
                                   autocomplete="off"
                                   spellcheck="true">
                            <button class="send-btn modern-send" id="sendBtn" title="Senden">
                                <span class="send-icon">📤</span>
                            </button>
                        </div>
                        <div class="input-footer">
                            <span class="typing-indicator" id="typingIndicator">Kira tippt...</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(chatWidget);
        this.container = chatWidget;
        
        console.log('✅ Modern chat container created');
    }

    setupEventListeners() {
        // Toggle button
        const toggleBtn = document.getElementById('kiraChatToggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggle());
        }

        // Header controls
        const ttsBtn = document.getElementById('ttsToggle');
        if (ttsBtn) {
            ttsBtn.addEventListener('click', () => this.toggleTTS());
        }

        const voiceBtn = document.getElementById('voiceToggle');
        if (voiceBtn) {
            voiceBtn.addEventListener('click', () => this.toggleVoiceWakeup());
        }

        const minimizeBtn = document.getElementById('minimizeBtn');
        if (minimizeBtn) {
            minimizeBtn.addEventListener('click', () => this.minimize());
        }

        const closeBtn = document.getElementById('closeBtn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.close());
        }

        // Message input
        const chatInput = document.getElementById('kiraChatInput');
        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            chatInput.addEventListener('focus', () => {
                this.hideQuickActions();
            });

            chatInput.addEventListener('blur', () => {
                if (!chatInput.value.trim()) {
                    this.showQuickActions();
                }
            });
        }

        // Send button
        const sendBtn = document.getElementById('sendBtn');
        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }

        // Quick actions
        this.setupQuickActions();

        console.log('✅ Event listeners set up');
    }

    setupQuickActions() {
        const quickActions = document.querySelectorAll('.quick-action');
        quickActions.forEach(action => {
            action.addEventListener('click', () => {
                const message = action.dataset.action;
                const input = document.getElementById('kiraChatInput');
                if (input) {
                    input.value = message;
                    this.sendMessage();
                }
            });
        });
    }

    hideQuickActions() {
        const quickActions = document.getElementById('quickActions');
        if (quickActions) {
            quickActions.style.display = 'none';
        }
    }

    showQuickActions() {
        const quickActions = document.getElementById('quickActions');
        if (quickActions && this.messageCount <= 1) {
            quickActions.style.display = 'flex';
        }
    }

    injectModernStyles() {
        if (document.getElementById('kiraModernStyles')) return;

        const style = document.createElement('style');
        style.id = 'kiraModernStyles';
        style.textContent = `
            /* Modern Kira Chat Widget Styles */
            .kira-chat-widget {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                position: fixed;
                bottom: 20px;
                right: 20px;
                z-index: 10000;
                --kira-primary: #667eea;
                --kira-secondary: #764ba2;
                --kira-accent: #f093fb;
                --kira-success: #4facfe;
                --kira-bg: #ffffff;
                --kira-surface: #f8fafc;
                --kira-text: #1a202c;
                --kira-text-light: #718096;
                --kira-border: #e2e8f0;
                --kira-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                --kira-shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            }

            /* Toggle Button Container */
            .kira-chat-toggle-container {
                position: relative;
                display: inline-block;
            }

            /* Pulse Ring Animation */
            .kira-pulse-ring {
                position: absolute;
                top: -4px;
                left: -4px;
                right: -4px;
                bottom: -4px;
                border: 2px solid var(--kira-primary);
                border-radius: 50%;
                animation: kira-pulse 2s infinite;
                opacity: 0.6;
            }

            @keyframes kira-pulse {
                0% { transform: scale(1); opacity: 0.6; }
                50% { transform: scale(1.1); opacity: 0.3; }
                100% { transform: scale(1); opacity: 0.6; }
            }

            /* Modern Toggle Button */
            .kira-chat-toggle.modern-toggle {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                border: none;
                background: linear-gradient(135deg, var(--kira-primary), var(--kira-secondary));
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
                box-shadow: var(--kira-shadow);
                position: relative;
                overflow: hidden;
            }

            .kira-chat-toggle.modern-toggle:hover {
                transform: translateY(-2px);
                box-shadow: var(--kira-shadow-lg);
            }

            .kira-chat-toggle.modern-toggle:active {
                transform: translateY(0);
            }

            .kira-chat-toggle.modern-toggle.active {
                background: linear-gradient(135deg, var(--kira-accent), var(--kira-success));
            }

            /* Avatar */
            .kira-avatar {
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .avatar-inner {
                font-size: 24px;
                transition: transform 0.3s ease;
            }

            .kira-avatar.speaking .avatar-inner {
                animation: kira-speaking 0.8s infinite;
            }

            @keyframes kira-speaking {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }

            .status-indicator {
                position: absolute;
                bottom: 2px;
                right: 2px;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                border: 2px solid white;
                background: #48bb78;
            }

            .status-indicator.online {
                background: #48bb78;
                animation: kira-online-pulse 2s infinite;
            }

            @keyframes kira-online-pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }

            /* Modern Chat Window */
            .kira-chat-window.modern-window {
                position: absolute;
                bottom: 80px;
                right: 0;
                width: 380px;
                height: 500px;
                background: var(--kira-bg);
                border-radius: 16px;
                box-shadow: var(--kira-shadow-lg);
                display: none;
                flex-direction: column;
                overflow: hidden;
                backdrop-filter: blur(10px);
                border: 1px solid var(--kira-border);
            }

            .kira-chat-window.modern-window.open {
                display: flex;
                animation: kira-slide-up 0.3s ease-out;
            }

            @keyframes kira-slide-up {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }

            /* Modern Header */
            .chat-header.modern-header {
                background: linear-gradient(135deg, var(--kira-primary), var(--kira-secondary));
                color: white;
                padding: 16px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }

            .header-left {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .kira-avatar-header {
                position: relative;
                width: 40px;
                height: 40px;
                background: rgba(255, 255, 255, 0.15);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
            }

            .header-info {
                flex: 1;
            }

            .kira-name {
                margin: 0;
                font-size: 16px;
                font-weight: 600;
            }

            .kira-status {
                font-size: 12px;
                opacity: 0.8;
                margin-top: 2px;
            }

            .header-controls {
                display: flex;
                gap: 8px;
            }

            .control-btn {
                width: 32px;
                height: 32px;
                border: none;
                background: rgba(255, 255, 255, 0.15);
                color: white;
                border-radius: 8px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
                font-size: 14px;
            }

            .control-btn:hover {
                background: rgba(255, 255, 255, 0.25);
                transform: translateY(-1px);
            }

            .control-btn.active {
                background: rgba(255, 255, 255, 0.3);
                animation: kira-btn-pulse 1s infinite;
            }

            @keyframes kira-btn-pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }

            /* Modern Content */
            .chat-content.modern-content {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: var(--kira-surface);
            }

            /* Modern Messages */
            .chat-messages.modern-messages {
                flex: 1;
                padding: 16px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .chat-messages.modern-messages::-webkit-scrollbar {
                width: 4px;
            }

            .chat-messages.modern-messages::-webkit-scrollbar-track {
                background: transparent;
            }

            .chat-messages.modern-messages::-webkit-scrollbar-thumb {
                background: var(--kira-border);
                border-radius: 2px;
            }

            .message {
                display: flex;
                align-items: flex-start;
                gap: 8px;
                animation: kira-message-slide 0.3s ease-out;
            }

            @keyframes kira-message-slide {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .message.user-message {
                flex-direction: row-reverse;
            }

            .message-avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                flex-shrink: 0;
            }

            .kira-message .message-avatar {
                background: linear-gradient(135deg, var(--kira-primary), var(--kira-secondary));
                color: white;
            }

            .user-message .message-avatar {
                background: var(--kira-border);
                color: var(--kira-text);
            }

            .message-content {
                flex: 1;
                max-width: 80%;
            }

            .message-bubble {
                padding: 12px 16px;
                border-radius: 16px;
                position: relative;
                word-wrap: break-word;
            }

            .kira-bubble {
                background: white;
                border: 1px solid var(--kira-border);
                border-bottom-left-radius: 4px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }

            .user-message .message-bubble {
                background: linear-gradient(135deg, var(--kira-primary), var(--kira-secondary));
                color: white;
                border-bottom-right-radius: 4px;
                border-bottom-left-radius: 16px;
            }

            .message-text {
                font-size: 14px;
                line-height: 1.4;
                margin-bottom: 4px;
            }

            .message-time {
                font-size: 11px;
                opacity: 0.6;
                text-align: right;
            }

            .user-message .message-time {
                color: rgba(255, 255, 255, 0.8);
            }

            .typing-indicator {
                display: none;
                color: var(--kira-primary);
                font-size: 12px;
                padding: 4px 8px;
                animation: kira-typing 1.5s infinite;
            }

            @keyframes kira-typing {
                0%, 100% { opacity: 0.5; }
                50% { opacity: 1; }
            }

            .typing-indicator.active {
                display: block;
            }

            /* Quick Actions */
            .quick-actions {
                padding: 12px 16px;
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                border-top: 1px solid var(--kira-border);
                background: white;
            }

            .quick-action {
                background: var(--kira-surface);
                border: 1px solid var(--kira-border);
                border-radius: 20px;
                padding: 6px 12px;
                font-size: 12px;
                cursor: pointer;
                transition: all 0.2s ease;
                color: var(--kira-text);
            }

            .quick-action:hover {
                background: var(--kira-primary);
                color: white;
                transform: translateY(-1px);
            }

            /* Modern Input */
            .chat-input.modern-input {
                background: white;
                border-top: 1px solid var(--kira-border);
                padding: 16px;
            }

            .input-container {
                display: flex;
                gap: 8px;
                align-items: center;
                background: var(--kira-surface);
                border-radius: 24px;
                padding: 4px;
                border: 1px solid var(--kira-border);
                transition: all 0.2s ease;
            }

            .input-container:focus-within {
                border-color: var(--kira-primary);
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }

            .message-input {
                flex: 1;
                border: none;
                background: transparent;
                padding: 12px 16px;
                font-size: 14px;
                outline: none;
                resize: none;
                color: var(--kira-text);
            }

            .message-input::placeholder {
                color: var(--kira-text-light);
            }

            .send-btn.modern-send {
                width: 40px;
                height: 40px;
                border: none;
                background: linear-gradient(135deg, var(--kira-primary), var(--kira-secondary));
                color: white;
                border-radius: 50%;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
                flex-shrink: 0;
            }

            .send-btn.modern-send:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }

            .send-btn.modern-send:active {
                transform: translateY(0);
            }

            .send-icon {
                font-size: 16px;
            }

            .input-footer {
                margin-top: 8px;
                height: 16px;
            }

            /* Responsive Design */
            @media (max-width: 480px) {
                .kira-chat-window.modern-window {
                    width: calc(100vw - 40px);
                    height: calc(100vh - 140px);
                    bottom: 80px;
                    right: 20px;
                }
            }

            /* Dark Mode Support */
            @media (prefers-color-scheme: dark) {
                .kira-chat-widget {
                    --kira-bg: #1a202c;
                    --kira-surface: #2d3748;
                    --kira-text: #f7fafc;
                    --kira-text-light: #a0aec0;
                    --kira-border: #4a5568;
                }

                .kira-bubble {
                    background: var(--kira-surface);
                    color: var(--kira-text);
                }

                .input-container {
                    background: var(--kira-surface);
                }

                .quick-actions {
                    background: var(--kira-surface);
                }

                .chat-input.modern-input {
                    background: var(--kira-surface);
                }
            }
        `;

        document.head.appendChild(style);
        console.log('✅ Modern styles injected');
    }

    toggle() {
        const chatWindow = document.getElementById('kiraChatWindow');
        const toggleBtn = document.getElementById('kiraChatToggle');
        
        if (chatWindow && toggleBtn) {
            this.isOpen = !this.isOpen;
            
            if (this.isOpen) {
                chatWindow.classList.add('open');
                chatWindow.style.display = 'flex';
                toggleBtn.classList.add('active');
                
                // Focus input
                setTimeout(() => {
                    const input = document.getElementById('kiraChatInput');
                    if (input) input.focus();
                }, 300);
                
                // Show quick actions if this is a new conversation
                if (this.messageCount <= 1) {
                    this.showQuickActions();
                }
            } else {
                chatWindow.classList.remove('open');
                setTimeout(() => {
                    chatWindow.style.display = 'none';
                }, 300);
                toggleBtn.classList.remove('active');
            }
            
            console.log(`💬 Chat ${this.isOpen ? 'opened' : 'closed'}`);
        }
    }

    close() {
        const chatWindow = document.getElementById('kiraChatWindow');
        const toggleBtn = document.getElementById('kiraChatToggle');
        
        if (chatWindow && toggleBtn) {
            this.isOpen = false;
            chatWindow.classList.remove('open');
            setTimeout(() => {
                chatWindow.style.display = 'none';
            }, 300);
            toggleBtn.classList.remove('active');
            
            // Stop any ongoing TTS
            this.stopTTS();
            
            console.log('💬 Chat closed');
        }
    }

    minimize() {
        const chatWindow = document.getElementById('kiraChatWindow');
        const toggleBtn = document.getElementById('kiraChatToggle');
        
        if (chatWindow && toggleBtn) {
            this.isOpen = false;
            this.isMinimized = true;
            
            chatWindow.classList.remove('open');
            setTimeout(() => {
                chatWindow.style.display = 'none';
            }, 300);
            
            toggleBtn.classList.remove('active');
            toggleBtn.classList.add('minimized');
            
            // Visual feedback
            setTimeout(() => {
                toggleBtn.classList.remove('minimized');
            }, 500);
            
            console.log('💬 Chat minimized');
        }
    }

    sendMessage(message = null) {
        const input = document.getElementById('kiraChatInput');
        const messageText = message || (input ? input.value.trim() : '');
        
        if (!messageText) {
            console.warn('⚠️ No message text to send');
            return;
        }
        
        console.log('� Sending message:', messageText);
        
        // Ensure chat window is open
        if (!this.isOpen) {
            this.toggle();
        }
        
        // Add user message
        this.addUserMessage(messageText);
        
        // Clear input
        if (input) {
            input.value = '';
        }
        
        // Hide quick actions after first message
        this.hideQuickActions();
        
        // Send to API
        this.sendToKiraAPI(messageText);
        
        this.messageCount++;
    }

    addUserMessage(text) {
        const messages = document.getElementById('chatMessages');
        if (!messages) return;
        
        const timestamp = this.formatTime(new Date());
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <div class="avatar-inner">👤</div>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    <div class="message-text">${this.escapeHtml(text)}</div>
                    <div class="message-time">${timestamp}</div>
                </div>
            </div>
        `;
        
        messages.appendChild(messageDiv);
        this.scrollToBottom();
        
        console.log('📤 User message added:', text);
    }

    addKiraMessage(text) {
        const messages = document.getElementById('chatMessages');
        if (!messages) return;
        
        const timestamp = this.formatTime(new Date());
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message kira-message';
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <div class="avatar-inner">🤖</div>
            </div>
            <div class="message-content">
                <div class="message-bubble kira-bubble">
                    <div class="message-text">${this.formatKiraText(text)}</div>
                    <div class="message-time">${timestamp}</div>
                </div>
            </div>
        `;
        
        messages.appendChild(messageDiv);
        this.scrollToBottom();
        
        console.log('🤖 Kira message added:', text);
        
        // TTS für Kira's Antwort
        this.speakText(text);
        
        return messageDiv;
    }

    showTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.classList.add('active');
        }
        this.isTyping = true;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.classList.remove('active');
        }
        this.isTyping = false;
    }

    formatTime(date) {
        return date.toLocaleTimeString('de-DE', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    formatKiraText(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom() {
        const messages = document.getElementById('chatMessages');
        if (messages) {
            messages.scrollTop = messages.scrollHeight;
        }
    }

    async sendToKiraAPI(messageText) {
        console.log('🚀 Sending message to Kira API...');
        
        try {
            // Show typing indicator
            this.showTypingIndicator();
            
            // API-Request an Kira
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

            // Hide typing indicator
            this.hideTypingIndicator();

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            if (data.success && data.response) {
                // Add Kira's response
                this.addKiraMessage(data.response);
                
                // Update conversation ID
                if (data.conversation_id) {
                    this.conversationId = data.conversation_id;
                }
                
                // Log additional data
                if (data.memory_processing) {
                    console.log('� Memory processing:', data.memory_processing);
                }
                
                if (data.ai_status) {
                    console.log('🤖 AI Status:', data.ai_status);
                }
                
                console.log('✅ Kira response received successfully');
            } else {
                throw new Error(data.error || 'Unknown API error');
            }
            
        } catch (error) {
            console.error('❌ Kira API error:', error);
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Show error message
            this.addKiraMessage(
                `Entschuldigung, ich hatte ein Problem bei der Verarbeitung deiner Nachricht: ${error.message}`
            );
        }
    }

    generateConversationId() {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const random = Math.random().toString(36).substr(2, 9);
        return `web_conv_${timestamp}_${random}`;
    }

    toggleTTS() {
        this.ttsEnabled = !this.ttsEnabled;
        
        if (!this.ttsEnabled) {
            this.stopTTS();
        }
        
        console.log('🔊 TTS toggled:', this.ttsEnabled);
        
        const ttsBtn = document.getElementById('ttsToggle');
        if (ttsBtn) {
            ttsBtn.innerHTML = this.ttsEnabled ? '🔊' : '🔇';
            ttsBtn.title = this.ttsEnabled ? 'TTS deaktivieren' : 'TTS aktivieren';
        }
    }

    toggleVoiceWakeup() {
        this.voiceWakeupEnabled = !this.voiceWakeupEnabled;
        
        if (this.voiceWakeupEnabled) {
            this.startVoiceWakeup();
        } else {
            this.stopVoiceWakeup();
        }
        
        console.log('🎤 Voice wake-up toggled:', this.voiceWakeupEnabled);
        
        const voiceBtn = document.getElementById('voiceToggle');
        if (voiceBtn) {
            voiceBtn.style.opacity = this.voiceWakeupEnabled ? '1' : '0.5';
            voiceBtn.title = this.voiceWakeupEnabled ? 'Voice Wake-up aktiv' : 'Voice Wake-up inaktiv';
        }
    }

    startVoiceWakeup() {
        if (this.recognition && this.voiceWakeupEnabled && !this.isListening) {
            try {
                this.recognition.start();
                this.isListening = true;
                console.log('🎤 Voice wake-up listening started');
            } catch (error) {
                console.warn('🎤 Voice wake-up start failed:', error);
            }
        }
    }

    stopVoiceWakeup() {
        if (this.recognition && this.isListening) {
            try {
                this.recognition.stop();
                this.isListening = false;
                console.log('🎤 Voice wake-up stopped');
            } catch (error) {
                console.warn('🎤 Voice wake-up stop failed:', error);
            }
        }
    }
}

// Globale Chat-Funktionen mit verbesserter Fehlerbehandlung
function sendKiraMessage() {
    console.log('📤 sendKiraMessage() called');
    
    if (!window.kiraChat) {
        console.log('💡 Kira Chat Widget nicht initialisiert, erstelle es...');
        window.kiraChat = new KiraChatWidget();
    }
    
    window.kiraChat.sendMessage();
}

function toggleKiraChat() {
    if (!window.kiraChat) {
        window.kiraChat = new KiraChatWidget();
    }
    
    window.kiraChat.toggle();
}

function closeKiraChat() {
    if (window.kiraChat) {
        window.kiraChat.close();
    }
}

function minimizeKiraChat() {
    if (window.kiraChat) {
        window.kiraChat.minimize();
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K to toggle chat
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        toggleKiraChat();
    }
    
    // Escape to close chat
    if (e.key === 'Escape' && window.kiraChat && window.kiraChat.isOpen) {
        e.preventDefault();
        closeKiraChat();
    }
});

// Stelle sicher, dass die Funktionen global verfügbar sind
window.KiraChatWidget = KiraChatWidget;
window.sendKiraMessage = sendKiraMessage;
window.toggleKiraChat = toggleKiraChat;
window.closeKiraChat = closeKiraChat;
window.minimizeKiraChat = minimizeKiraChat;

// Auto-Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 DOM loaded, initializing Kira Chat Widget...');
    
    // Initialize widget automatically
    if (!window.kiraChat) {
        window.kiraChat = new KiraChatWidget();
    }
    
    // Show welcome animation
    setTimeout(() => {
        const toggleBtn = document.getElementById('kiraChatToggle');
        if (toggleBtn) {
            toggleBtn.style.animation = 'kira-welcome 2s ease-out';
        }
    }, 1000);
    
    console.log('✅ Kira Chat Widget auto-initialized');
});

// Add welcome animation
const welcomeStyle = document.createElement('style');
welcomeStyle.textContent = `
    @keyframes kira-welcome {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
`;
document.head.appendChild(welcomeStyle);

console.log('✅ Ultra Modern Kira Chat Widget v2.0 loaded successfully');
