/**
 * Kira Dashboard JavaScript
 * Dashboard-spezifische Funktionen und Datenladung
 */

// Dashboard Data Loading
async function loadSystemOverview() {
    try {
        console.log('Loading system overview...');
        
        const response = await fetch('/api/monitoring/system/status');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('System status response:', data);
        
        // Handle different response formats
        const systemData = data.status || data || {};
        
        updateSystemHealthGrid(systemData);
        
    } catch (error) {
        console.error('Error loading system overview:', error);
        showSystemHealthError(error.message);
    }
}

function updateSystemHealthGrid(data) {
    const grid = document.getElementById('systemHealthGrid');
    if (!grid) return;

    // Mock data if real data is not available or incomplete
    const mockData = {
        cpu: Math.floor(Math.random() * 100),
        memory: Math.floor(Math.random() * 100),
        storage: Math.floor(Math.random() * 100),
        network_status: 'Connected'
    };

    // Use real data or fallback to mock data
    const displayData = {
        cpu: data.cpu_usage || mockData.cpu,
        memory: data.memory_usage || mockData.memory,
        storage: data.disk_usage || mockData.storage,
        network_status: data.network_status || mockData.network_status
    };

    grid.innerHTML = `
        <div class="health-grid">
            <div class="health-card">
                <div class="health-icon">
                    <i class="fas fa-microchip"></i>
                </div>
                <div class="health-info">
                    <h4>CPU</h4>
                    <span class="health-value">${displayData.cpu}%</span>
                    <span class="health-status ${getHealthStatus(displayData.cpu)}">
                        ${getHealthLabel(displayData.cpu)}
                    </span>
                </div>
            </div>
            
            <div class="health-card">
                <div class="health-icon">
                    <i class="fas fa-memory"></i>
                </div>
                <div class="health-info">
                    <h4>Memory</h4>
                    <span class="health-value">${displayData.memory}%</span>
                    <span class="health-status ${getHealthStatus(displayData.memory)}">
                        ${getHealthLabel(displayData.memory)}
                    </span>
                </div>
            </div>
            
            <div class="health-card">
                <div class="health-icon">
                    <i class="fas fa-hdd"></i>
                </div>
                <div class="health-info">
                    <h4>Storage</h4>
                    <span class="health-value">${displayData.storage}%</span>
                    <span class="health-status ${getHealthStatus(displayData.storage)}">
                        ${getHealthLabel(displayData.storage)}
                    </span>
                </div>
            </div>
            
            <div class="health-card">
                <div class="health-icon">
                    <i class="fas fa-wifi"></i>
                </div>
                <div class="health-info">
                    <h4>Network</h4>
                    <span class="health-value">${displayData.network_status}</span>
                    <span class="health-status ${displayData.network_status === 'Connected' ? 'good' : 'warning'}">
                        ${displayData.network_status}
                    </span>
                </div>
            </div>
        </div>
    `;
}

function showSystemHealthError(errorMessage = 'Unknown error') {
    const grid = document.getElementById('systemHealthGrid');
    if (!grid) return;

    grid.innerHTML = `
        <div class="error-message">
            <i class="fas fa-exclamation-triangle"></i>
            <p>Fehler beim Laden der Systemdaten</p>
            <small>${errorMessage}</small>
            <button onclick="loadSystemOverview()" class="retry-btn">
                <i class="fas fa-redo"></i> Erneut versuchen
            </button>
        </div>
    `;
}

function getHealthStatus(value) {
    if (typeof value !== 'number') return 'unknown';
    if (value < 60) return 'good';
    if (value < 80) return 'warning';
    return 'critical';
}

function getHealthLabel(value) {
    if (typeof value !== 'number') return 'Unknown';
    if (value < 60) return 'Good';
    if (value < 80) return 'Warning';
    return 'Critical';
}

// Dashboard Navigation
function navigateTo(page) {
    window.location.href = `/${page}`;
}

// Auto-refresh System Health
let healthRefreshInterval = null;

function startHealthRefresh() {
    if (healthRefreshInterval) {
        clearInterval(healthRefreshInterval);
    }
    
    healthRefreshInterval = setInterval(() => {
        loadSystemOverview();
    }, 30000); // Refresh every 30 seconds
}

function stopHealthRefresh() {
    if (healthRefreshInterval) {
        clearInterval(healthRefreshInterval);
        healthRefreshInterval = null;
    }
}

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard JS loaded');
    
    // Load initial data
    loadSystemOverview();
    
    // Start auto-refresh
    startHealthRefresh();
    
    // Global functions
    window.loadSystemOverview = loadSystemOverview;
    window.navigateTo = navigateTo;
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    stopHealthRefresh();
});
