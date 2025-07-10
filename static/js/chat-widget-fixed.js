/**
 * Kira Chat Widget JavaScript - FIXED VERSION
 * Chat-Funktionalität mit verbesserter Fehlerbehandlung
 */

class KiraChatWidget {
    constructor() {
        this.isOpen = false;
        this.container = null;
        console.log('🚀 Initializing Kira Chat Widget');
        this.initializeWidget();
    }

    initializeWidget() {
        this.createChatContainer();
        this.setupEventListeners();
        console.log('✅ Kira Chat Widget initialized');
    }

    createChatContainer() {
        // Erstelle Chat-Container falls nicht vorhanden
        if (!document.getElementById('kiraChatWidget')) {
            const chatWidget = document.createElement('div');
            chatWidget.id = 'kiraChatWidget';
            chatWidget.className = 'kira-chat-widget';
            chatWidget.innerHTML = `
                <button class="kira-chat-toggle" id="kiraChatToggle" onclick="toggleKiraChat()">
                    💬
                </button>
                <div class="kira-chat-window" id="kiraChatWindow" style="display: none;">
                    <div class="chat-header">
                        <h4>Kira Assistant</h4>
                        <button onclick="closeKiraChat()" class="close-btn">✕</button>
                    </div>
                    <div class="chat-content">
                        <div class="chat-messages" id="chatMessages">
                            <div class="message kira-message">
                                <div class="message-avatar">🤖</div>
                                <div class="message-content">
                                    Hallo! Ich bin Kira, dein digitaler Assistent. Wie kann ich dir helfen?
                                </div>
                            </div>
                        </div>
                        <div class="chat-input">
                            <input type="text" id="kiraChatInput" placeholder="Nachricht eingeben..." onkeypress="handleChatKeyPress(event)">
                            <button onclick="sendKiraMessage()" class="send-btn">📤</button>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(chatWidget);
            console.log('✅ Chat container created');
        }
        
        this.container = document.getElementById('kiraChatWidget');
    }

    setupEventListeners() {
        // Warte kurz, damit das DOM-Element verfügbar ist
        setTimeout(() => {
            const chatInput = document.getElementById('kiraChatInput');
            if (chatInput) {
                console.log('✅ Setting up chat input event listener');
                chatInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        this.sendMessage();
                    }
                });
            } else {
                console.error('❌ Chat input element not found for event listener');
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
            
            console.log(`💬 Chat ${this.isOpen ? 'opened' : 'closed'}`);
        } else {
            console.error('❌ Chat window not found');
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
            
            console.log('💬 Chat closed');
        }
    }

    minimize() {
        const chatWindow = document.getElementById('kiraChatWindow');
        if (chatWindow) {
            this.isOpen = false;
            chatWindow.style.display = 'none';
            
            const toggleBtn = document.getElementById('kiraChatToggle');
            if (toggleBtn) {
                toggleBtn.classList.remove('active');
                toggleBtn.classList.add('minimized');
                setTimeout(() => {
                    toggleBtn.classList.remove('minimized');
                }, 300);
            }
            
            console.log('💬 Chat minimized');
        }
    }

    sendMessage(message = null) {
        const input = document.getElementById('kiraChatInput');
        const messages = document.getElementById('chatMessages');
        
        const messageText = message || (input ? input.value.trim() : '');
        
        console.log('📤 Attempting to send message:', messageText);
        
        if (!messageText) {
            console.warn('⚠️ No message text to send');
            return;
        }
        
        if (!messages) {
            console.error('❌ Messages container not found');
            return;
        }
        
        // Benutzer-Nachricht hinzufügen
        this.addMessage(messageText, 'user');
        
        // Input leeren
        if (input) {
            input.value = '';
        }
        
        console.log('✅ Message sent successfully');
        
        // Kira-Antwort simulieren
        setTimeout(() => {
            this.addMessage('Das ist eine Test-Antwort von Kira. Die echte AI-Integration folgt bald!', 'kira');
        }, 1000);
    }

    addMessage(text, sender) {
        const messages = document.getElementById('chatMessages');
        if (!messages) {
            console.error('❌ Messages container not found');
            return;
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const isUser = sender === 'user';
        const avatar = isUser ? '👤' : '🤖';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">${text}</div>
        `;
        
        messages.appendChild(messageDiv);
        messages.scrollTop = messages.scrollHeight;
        
        console.log(`💬 Message added: ${sender} - ${text.substring(0, 50)}...`);
    }
}

// Globale Chat-Funktionen mit verbesserter Fehlerbehandlung
function sendKiraMessage() {
    console.log('📤 sendKiraMessage() called');
    
    if (!window.kiraChat) {
        console.error('❌ Kira Chat Widget ist nicht initialisiert');
        console.log('💡 Versuche Chat-Widget zu erstellen...');
        window.kiraChat = new KiraChatWidget();
    }
    
    const input = document.getElementById('kiraChatInput');
    const message = input ? input.value.trim() : '';
    
    if (message) {
        window.kiraChat.sendMessage(message);
    } else {
        console.warn('⚠️ Keine Nachricht zum Senden');
    }
}

// Stelle sicher, dass die Funktionen global verfügbar sind
window.KiraChatWidget = KiraChatWidget;
window.sendKiraMessage = sendKiraMessage;

// Auto-Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 DOM loaded, chat widget ready for initialization');
});

console.log('✅ Chat Widget JavaScript loaded successfully');
