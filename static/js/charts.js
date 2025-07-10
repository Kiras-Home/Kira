
        class KiraChatWidget {
            constructor() {
                this.isOpen = false;
                this.sessionId = this.generateSessionId();
                this.userId = 'web_user_' + Math.random().toString(36).substr(2, 9);
                this.voiceEnabled = true;

                this.initializeElements();
                this.loadChatHistory();

                console.log('🤖 Kira Chat Widget initialisiert');
            }

            initializeElements() {
                this.chatToggle = document.getElementById('kiraChatToggle');
                this.chatWindow = document.getElementById('kiraChatWindow');
                this.chatMessages = document.getElementById('kiraChatMessages');
                this.chatInput = document.getElementById('kiraChatInput');
                this.sendBtn = document.getElementById('kiraChatSendBtn');
                this.typingIndicator = document.getElementById('kiraTyping');
                this.notificationBadge = document.getElementById('chatNotificationBadge');
            }

            filterText(text) {
                if (!text) return "";

                // Code-Blöcke entfernen
                text = text.replace(/```[\s\S]*?```/g, '');
                text = text.replace(/`[^`]*`/g, '');

                // JSON/Code-ähnliche Strukturen
                text = text.replace(/\{[^}]*\}/g, '');
                text = text.replace(/\[[^\]]*\]/g, '');

                // Emojis entfernen (außer einfache)
                text = text.replace(/[\u{1F600}-\u{1F6FF}]/gu, '');
                text = text.replace(/[\u{1F300}-\u{1F5FF}]/gu, '');
                text = text.replace(/[\u{2600}-\u{26FF}]/gu, '');
                text = text.replace(/[\u{2700}-\u{27BF}]/gu, '');

                // Nur erlaubte Zeichen behalten
                text = text.replace(/[^\w\s\.\,\?\!\-\äöüÄÖÜß]/g, '');

                return text.trim();
            }

            async sendMessage(message) {
                const filteredMessage = this.filterText(message);

                if (!filteredMessage || filteredMessage.length < 2) {
                    this.showError('Bitte gib eine gültige Nachricht ein (nur Text, keine Sonderzeichen).');
                    return;
                }

                // User-Nachricht anzeigen
                this.addMessage(filteredMessage, 'user');
                this.chatInput.value = '';
                this.showTyping();
                this.scrollToBottom();

                try {
                    const response = await fetch('/api/chat/kira', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            message: filteredMessage,
                            user_id: this.userId,
                            session_id: this.sessionId,
                            generate_audio: this.voiceEnabled,
                            context: {
                                timestamp: new Date().toISOString(),
                                interface: 'dropdown_chat'
                            }
                        })
                    });

                    const data = await response.json();
                    this.hideTyping();

                    if (data.success) {
                        const filteredResponse = this.filterText(data.message);
                        this.addMessage(filteredResponse, 'kira');

                        // Audio abspielen falls verfügbar
                        if (data.audio_url && this.voiceEnabled) {
                            this.playAudio(data.audio_url);
                        }

                        // Notification falls Chat geschlossen
                        if (!this.isOpen) {
                            this.showNotification();
                        }
                    } else {
                        this.addMessage('Entschuldigung, da ist etwas schiefgelaufen: ' + (data.error || 'Unbekannter Fehler'), 'kira');
                    }

                } catch (error) {
                    this.hideTyping();
                    this.addMessage('Verbindungsfehler. Bitte versuche es erneut.', 'kira');
                    console.error('Kira Chat Fehler:', error);
                }

                this.scrollToBottom();
                this.saveChatHistory();
            }

            addMessage(message, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `chat-message ${sender}`;

                const bubble = document.createElement('div');
                bubble.className = 'message-bubble';
                bubble.textContent = message;

                const time = document.createElement('div');
                time.className = 'message-time';
                time.textContent = new Date().toLocaleTimeString();

                messageDiv.appendChild(bubble);
                messageDiv.appendChild(time);
                this.chatMessages.appendChild(messageDiv);

                this.scrollToBottom();
            }

            showTyping() {
                this.typingIndicator.style.display = 'block';
                this.sendBtn.disabled = true;
                this.scrollToBottom();
            }

            hideTyping() {
                this.typingIndicator.style.display = 'none';
                this.sendBtn.disabled = false;
            }

            showError(message) {
                this.addMessage(`❌ ${message}`, 'kira');
            }

            playAudio(audioUrl) {
                try {
                    const audio = new Audio(audioUrl);
                    audio.play().catch(e => console.warn('Audio playback failed:', e));
                } catch (error) {
                    console.warn('Audio creation failed:', error);
                }
            }

            showNotification() {
                this.notificationBadge.style.display = 'flex';
                this.notificationBadge.textContent = '1';
            }

            hideNotification() {
                this.notificationBadge.style.display = 'none';
            }

            toggle() {
                this.isOpen = !this.isOpen;

                if (this.isOpen) {
                    this.chatWindow.classList.add('show');
                    this.chatToggle.innerHTML = '<i class="fas fa-chevron-down me-2"></i>Kira Chat';
                    this.hideNotification();
                    setTimeout(() => {
                        this.chatInput.focus();
                        this.scrollToBottom();
                    }, 300);
                } else {
                    this.chatWindow.classList.remove('show');
                    this.chatToggle.innerHTML = '<i class="fas fa-robot me-2"></i>Chat mit Kira';
                }
            }

            minimize() {
                this.isOpen = false;
                this.chatWindow.classList.remove('show');
                this.chatToggle.innerHTML = '<i class="fas fa-robot me-2"></i>Chat mit Kira';
            }

            close() {
                this.minimize();
                // Optional: Chat-Verlauf löschen
                if (confirm('Chat-Verlauf löschen?')) {
                    this.clearChat();
                }
            }

            clearChat() {
                this.chatMessages.innerHTML = `
                    <div class="chat-message kira">
                        <div class="message-bubble">
                            Hallo! Ich bin Kira, deine intelligente Assistentin. Wie kann ich dir heute helfen?
                        </div>
                        <div class="message-time">gerade eben</div>
                    </div>
                `;
                localStorage.removeItem('kira_chat_history');
            }

            scrollToBottom() {
                setTimeout(() => {
                    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
                }, 100);
            }

            saveChatHistory() {
                try {
                    const messages = Array.from(this.chatMessages.children).map(msg => ({
                        content: msg.querySelector('.message-bubble').textContent,
                        sender: msg.classList.contains('chat-message') && msg.classList.contains('user') ? 'user' : 'kira',
                        time: msg.querySelector('.message-time').textContent
                    }));
                    localStorage.setItem('kira_chat_history', JSON.stringify(messages.slice(-20))); // Nur letzte 20
                } catch (error) {
                    console.warn('Chat history save failed:', error);
                }
            }

            loadChatHistory() {
                try {
                    const history = localStorage.getItem('kira_chat_history');
                    if (history) {
                        const messages = JSON.parse(history);
                        this.chatMessages.innerHTML = ''; // Clear current

                        messages.forEach(msg => {
                            const messageDiv = document.createElement('div');
                            messageDiv.className = `chat-message ${msg.sender}`;
                            messageDiv.innerHTML = `
                                <div class="message-bubble">${msg.content}</div>
                                <div class="message-time">${msg.time}</div>
                            `;
                            this.chatMessages.appendChild(messageDiv);
                        });

                        this.scrollToBottom();
                    }
                } catch (error) {
                    console.warn('Chat history load failed:', error);
                }
            }

            generateSessionId() {
                return 'chat_widget_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            }
        }

        // Globale Funktionen für HTML onclick events
        let kiraChat = null;

        function toggleKiraChat() {
            if (!kiraChat) {
                kiraChat = new KiraChatWidget();
            }
            kiraChat.toggle();
        }

        function minimizeKiraChat() {
            if (kiraChat) kiraChat.minimize();
        }

        function closeKiraChat() {
            if (kiraChat) kiraChat.close();
        }

        function sendKiraMessage() {
            if (kiraChat) {
                const input = document.getElementById('kiraChatInput');
                const message = input.value.trim();
                if (message) {
                    kiraChat.sendMessage(message);
                }
            }
        }

        function handleChatKeyPress(event) {
            if (event.key === 'Enter') {
                sendKiraMessage();
            }
        }

        // Auto-initialize wenn DOM geladen
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🤖 Kira Chat Widget bereit');
        });

        // Bestehende Funktionen (Brain Waves etc.)
        function toggleBrainWaves() {
            const widget = document.getElementById('brainWaveWidget');
            if (widget) {
                widget.style.display = widget.style.display === 'none' ? 'block' : 'none';
            }
        }

        function pauseBrainWaves() {
            console.log('Brain waves paused/resumed');
        }

        function checkSystemHealth() {
            console.log('Checking system health...');
        }

        function checkMemoryHealth() {
            console.log('Checking memory health...');
        }

        function viewLogs() {
            console.log('Opening logs...');
        }

        function restartSystem() {
            if (confirm('System wirklich neustarten?')) {
                console.log('Restarting system...');
            }
        }
