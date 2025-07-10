/**
 * Kira Main JavaScript
 * Globale Funktionen und Initialisierung
 */

// Globale Variablen
window.kiraChat = null;
let systemStatus = null;

// Globale Funktionen für HTML onclick events
function toggleKiraChat() {
    console.log('toggleKiraChat called');
    if (!window.kiraChat) {
        window.kiraChat = new KiraChatWidget();
    }
    window.kiraChat.toggle();
}

function sendKiraMessage() {
    console.log('sendKiraMessage called');
    if (window.kiraChat) {
        const input = document.getElementById('kiraChatInput');
        const message = input ? input.value.trim() : '';
        if (message) {
            window.kiraChat.sendMessage(message);
        }
    } else {
        console.error('❌ Kira Chat Widget ist nicht initialisiert');
    }
}

function closeKiraChat() {
    console.log('closeKiraChat called');
    if (window.kiraChat) {
        window.kiraChat.close();
    }
}

function minimizeKiraChat() {
    console.log('minimizeKiraChat called');
    if (window.kiraChat) {
        window.kiraChat.minimize();
    }
}

function handleChatKeyPress(event) {
    if (event.key === 'Enter') {
        sendKiraMessage();
    }
}

function refreshSystemHealth() {
    console.log('Refreshing system health...');
    loadSystemOverview();
}

function openModule(module) {
    console.log('Opening module:', module);
    window.location.href = `/${module}`;
}

function viewLogs() {
    console.log('Opening logs...');
    // Implementierung folgt
}

function restartSystem() {
    if (confirm('System wirklich neustarten?')) {
        console.log('Restarting system...');
        // Implementierung folgt
    }
}

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('Kira Main JS loaded');
    initializeGlobalFunctions();
});

function initializeGlobalFunctions() {
    // Stelle sicher, dass alle Funktionen global verfügbar sind
    window.toggleKiraChat = toggleKiraChat;
    window.sendKiraMessage = sendKiraMessage;
    window.closeKiraChat = closeKiraChat;
    window.minimizeKiraChat = minimizeKiraChat;
    window.handleChatKeyPress = handleChatKeyPress;
    window.refreshSystemHealth = refreshSystemHealth;
    window.openModule = openModule;
    window.viewLogs = viewLogs;
    window.restartSystem = restartSystem;
    
    console.log('Global functions registered');
}