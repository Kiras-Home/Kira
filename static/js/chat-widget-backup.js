/**
 * Kira Chat Widget JavaScript
 * Chat-Funktionalität und UI-Interaktionen
 */

class KiraChatWidget {
    constructor() {
        this.isOpen = false;
        this.container = null;
        this.initializeWidget();
    }

    initializeWidget() {
        console.log('Initializing Kira Chat Widget');
        this.createChatContainer();
        this.setupEventListeners();
    }

    createChatContainer() {
        // Erstelle Chat-Container falls nicht vorhanden
        if (!document.getElementById('kiraChatWidget')) {
            const chatWidget = document.createElement('div');
            chatWidget.id = 'kiraChatWidget';
            chatWidget.className = 'kira-chat-widget';
            chatWidget.innerHTML = `
                <button class="kira-chat-toggle" id="kiraChatToggle" onclick="toggleKiraChat()">
                    <i class="fas fa-comments"></i>
                </button>
                <div class="kira-chat-window" id="kiraChatWindow" style="display: none;">
                    <div class="chat-header">
                        <h4>Kira Assistant</h4>
                        <button onclick="closeKiraChat()" class="close-btn">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="chat-content">
                        <div class="chat-messages" id="chatMessages">
                            <div class="message kira-message">
                                <div class="message-avatar">
                                    <i class="fas fa-robot"></i>
                                </div>
                                <div class="message-content">
                                    Hallo! Ich bin Kira, dein digitaler Assistent. Wie kann ich dir helfen?
                                </div>
                            </div>
                        </div>
                        <div class="chat-input">
                            <input type="text" id="kiraChatInput" placeholder="Nachricht eingeben..." onkeypress="handleChatKeyPress(event)">
                            <button onclick="sendKiraMessage()" class="send-btn">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(chatWidget);
        }
        
        this.container = document.getElementById('kiraChatWidget');
    }

    setupEventListeners() {
        // Warte kurz, damit das DOM-Element verfügbar ist
        setTimeout(() => {
            const chatInput = document.getElementById('kiraChatInput') || document.getElementById('chatInput');
            if (chatInput) {
                console.log('✅ Chat input event listener registered');
                chatInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        this.sendMessage();
                    }
                });
            } else {
                console.warn('⚠️ Chat input element not found for event listener');
            }
        }, 100);
    }

    toggle() {
        const chatWindow = document.getElementById('kiraChatWindow');
        if (chatWindow) {
            this.isOpen = !this.isOpen;
            chatWindow.style.display = this.isOpen ? 'block' : 'none';
            
            const toggleBtn = document.getElementById('kiraChatToggle');
            if (toggleBtn) {
                toggleBtn.classList.toggle('active', this.isOpen);
            }
        }
    }

    close() {
        const chatWindow = document.getElementById('kiraChatWindow');
        if (chatWindow) {
            this.isOpen = false;
            chatWindow.style.display = 'none';
            
            const toggleBtn = document.getElementById('kiraChatToggle');
            if (toggleBtn) {
                toggleBtn.classList.remove('active');
            }
        }
    }

    minimize() {
        const chatWindow = document.getElementById('kiraChatWindow');
        if (chatWindow) {
            // Minimiere das Chat-Fenster (ähnlich wie close, aber vielleicht andere Animation)
            this.isOpen = false;
            chatWindow.style.display = 'none';
            
            const toggleBtn = document.getElementById('kiraChatToggle');
            if (toggleBtn) {
                toggleBtn.classList.remove('active');
                // Optional: Zeige minimize animation
                toggleBtn.classList.add('minimized');
                setTimeout(() => {
                    toggleBtn.classList.remove('minimized');
                }, 300);
            }
        }
    }

    sendMessage(message = null) {
        const input = document.getElementById('kiraChatInput') || document.getElementById('chatInput');
        const messages = document.getElementById('chatMessages');
        
        const messageText = message || (input ? input.value.trim() : '');
        
        console.log('💬 sendMessage called with:', messageText);
        
        if (messageText && messages) {
            // Benutzer-Nachricht hinzufügen
            this.addMessage(messageText, 'user');
            
            // Input leeren
            if (input) {
                input.value = '';
            }
            
            // Kira-Antwort simulieren
            setTimeout(() => {
                this.addMessage('Das ist eine Test-Antwort von Kira. Die echte AI-Integration folgt bald!', 'kira');
            }, 1000);
        } else {
            console.warn('⚠️ Keine Nachricht zum Senden oder Messages-Container nicht gefunden');
            console.log('   messageText:', messageText);
            console.log('   messages element:', messages);
        }
    }

    addMessage(text, sender) {
        const messages = document.getElementById('chatMessages');
        if (!messages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const isUser = sender === 'user';
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-${isUser ? 'user' : 'robot'}"></i>
            </div>
            <div class="message-content">${text}</div>
        `;
        
        messages.appendChild(messageDiv);
        messages.scrollTop = messages.scrollHeight;
    }
}

// Globale Chat-Funktionen
function sendMessage() {
    if (window.kiraChat) {
        window.kiraChat.sendMessage();
    }
}

function sendKiraMessage() {
    if (window.kiraChat) {
        const input = document.getElementById('kiraChatInput') || document.getElementById('chatInput');
        const message = input ? input.value.trim() : '';
        if (message) {
            window.kiraChat.sendMessage(message);
        }
    } else {
        console.error('❌ Kira Chat Widget ist nicht initialisiert');
    }
}

// Stelle sicher, dass die Funktionen global verfügbar sind
window.KiraChatWidget = KiraChatWidget;
window.sendMessage = sendMessage;
window.sendKiraMessage = sendKiraMessage;