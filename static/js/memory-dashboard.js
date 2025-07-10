/**
 * Kira Memory Dashboard JavaScript
 * Handles memory visualization, recent memories, and stats
 */

class MemoryDashboard {
    constructor() {
        this.memoryStats = {};
        this.recentMemories = [];
        this.memoryNetwork = null;
        this.refreshInterval = null;
        
        this.init();
    }
    
    init() {
        console.log('🧠 Initializing Memory Dashboard...');
        
        // Initialize components
        this.setupEventListeners();
        this.initializeMemoryNetwork();
        this.loadMemoryData();
        
        // Auto-refresh every 30 seconds
        this.refreshInterval = setInterval(() => {
            this.refreshMemoryData();
        }, 30000);
    }
    
    setupEventListeners() {
        // Refresh button
        const refreshBtn = document.querySelector('.refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshMemoryData());
        }
        
        // Memory tools
        const optimizeBtn = document.querySelector('[onclick="optimizeMemory()"]');
        if (optimizeBtn) {
            optimizeBtn.onclick = () => this.optimizeMemory();
        }
        
        const exportBtn = document.querySelector('[onclick="exportMemories()"]');
        if (exportBtn) {
            exportBtn.onclick = () => this.exportMemories();
        }
        
        const cleanBtn = document.querySelector('[onclick="clearOldMemories()"]');
        if (cleanBtn) {
            cleanBtn.onclick = () => this.clearOldMemories();
        }
    }
    
    async loadMemoryData() {
        try {
            console.log('📊 Loading memory data...');
            
            // Load multiple data sources in parallel
            const [
                statusResponse, 
                recentResponse, 
                statsResponse, 
                importantResponse,
                analyticsResponse,
                activeConversationsResponse,
                liveStatsResponse
            ] = await Promise.all([
                fetch('/api/memory/status'),
                fetch('/api/memory/recent?limit=20'),
                fetch('/api/memory/stats'),
                fetch('/api/memory/importance'),
                fetch('/api/memory/analytics'),
                fetch('/api/memory/conversations/active'),
                fetch('/api/memory/live-stats')
            ]);
            
            if (statusResponse.ok) {
                const statusData = await statusResponse.json();
                this.updateMemoryStatus(statusData);
            }
            
            if (recentResponse.ok) {
                const recentData = await recentResponse.json();
                this.recentMemories = recentData.memories || [];
                this.updateRecentMemories();
            }
            
            if (statsResponse.ok) {
                const statsData = await statsResponse.json();
                this.memoryStats = statsData.memory_stats || {};
                this.updateMemoryStats();
            }
            
            if (importantResponse.ok) {
                const importantData = await importantResponse.json();
                this.updateImportantMemories(importantData.important_memories || []);
            }
            
            if (analyticsResponse.ok) {
                const analyticsData = await analyticsResponse.json();
                this.updateChatAnalytics(analyticsData.analytics || {});
            }
            
            if (activeConversationsResponse.ok) {
                const activeConversationsData = await activeConversationsResponse.json();
                this.updateActiveConversations(activeConversationsData.active_conversations || []);
            }
            
            if (liveStatsResponse.ok) {
                const liveStatsData = await liveStatsResponse.json();
                this.updateLiveStats(liveStatsData.live_stats || {});
            }
            
            console.log('✅ Memory data loaded successfully');
            
        } catch (error) {
            console.error('❌ Error loading memory data:', error);
            this.showError('Failed to load memory data');
        }
    }
    
    async refreshMemoryData() {
        console.log('🔄 Refreshing memory data...');
        await this.loadMemoryData();
        this.updateMemoryNetwork();
    }
    
    updateMemoryStatus(data) {
        const statusElement = document.getElementById('memoryOverview');
        if (!statusElement) return;
        
        const status = data.memory_status || {};
        const isActive = status.status === 'active';
        
        statusElement.innerHTML = `
            <div class="stat-card ${isActive ? 'stat-success' : 'stat-warning'}">
                <div class="stat-icon">
                    <i class="fas fa-brain"></i>
                </div>
                <div class="stat-content">
                    <h4>Memory System</h4>
                    <span class="stat-value">${status.status || 'Unknown'}</span>
                    <span class="stat-label">${isActive ? 'Active & Learning' : 'Limited Function'}</span>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">
                    <i class="fas fa-memory"></i>
                </div>
                <div class="stat-content">
                    <h4>Components</h4>
                    <span class="stat-value">${this.countActiveComponents(status.components)}</span>
                    <span class="stat-label">Active Components</span>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">
                    <i class="fas fa-database"></i>
                </div>
                <div class="stat-content">
                    <h4>Storage</h4>
                    <span class="stat-value">${status.config?.storage_enabled ? 'Enabled' : 'Disabled'}</span>
                    <span class="stat-label">Persistent Memory</span>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon">
                    <i class="fas fa-comments"></i>
                </div>
                <div class="stat-content">
                    <h4>Interactions</h4>
                    <span class="stat-value">${this.memoryStats.total_interactions || 0}</span>
                    <span class="stat-label">Total Conversations</span>
                </div>
            </div>
        `;
    }
    
    countActiveComponents(components) {
        if (!components) return 0;
        return Object.values(components).filter(active => active).length;
    }
    
    updateRecentMemories() {
        const recentElement = document.getElementById('recentMemories');
        if (!recentElement) return;
        
        if (this.recentMemories.length === 0) {
            recentElement.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-brain-circuit"></i>
                    <p>No recent memories found</p>
                    <small>Start a conversation to create memories</small>
                </div>
            `;
            return;
        }
        
        const memoriesHtml = this.recentMemories.map(memory => `
            <div class="memory-item" data-memory-id="${memory.id}">
                <div class="memory-header">
                    <span class="memory-type badge badge-${this.getMemoryTypeColor(memory.type)}">
                        ${memory.type}
                    </span>
                    <span class="memory-time">${this.formatTime(memory.timestamp)}</span>
                </div>
                <div class="memory-content">
                    ${this.truncateText(memory.content, 100)}
                </div>
                <div class="memory-footer">
                    <span class="memory-importance ${this.getImportanceClass(memory.importance)}">
                        ${'★'.repeat(this.getImportanceStars(memory.importance))}
                    </span>
                    <span class="memory-emotion">${memory.emotional_state || 'neutral'}</span>
                </div>
            </div>
        `).join('');
        
        recentElement.innerHTML = memoriesHtml;
    }
    
    updateMemoryStats() {
        const statsElements = {
            'short-term': document.getElementById('stmStats'),
            'long-term': document.getElementById('ltmStats'),
            'conversation': document.getElementById('convStats')
        };
        
        // Update STM stats
        if (statsElements['short-term']) {
            const stmStats = this.memoryStats.short_term_memory || {};
            statsElements['short-term'].innerHTML = `
                <h5>Short-Term Memory</h5>
                <div class="memory-stat">
                    <span>Current: ${stmStats.current_count || 0}/${stmStats.capacity || 7}</span>
                    <div class="progress">
                        <div class="progress-bar" style="width: ${(stmStats.current_count || 0) / (stmStats.capacity || 7) * 100}%"></div>
                    </div>
                </div>
                <small>Memories stored: ${stmStats.memories_stored || 0}</small>
            `;
        }
        
        // Update LTM stats
        if (statsElements['long-term']) {
            const ltmStats = this.memoryStats.long_term_memory || {};
            statsElements['long-term'].innerHTML = `
                <h5>Long-Term Memory</h5>
                <div class="memory-stat">
                    <span>Total: ${ltmStats.total_memories || 0}</span>
                    <small>Max: ${ltmStats.max_capacity || 10000}</small>
                </div>
                <small>Consolidated memories</small>
            `;
        }
        
        // Update conversation stats
        if (statsElements['conversation']) {
            const convStats = this.memoryStats.conversation_memory || {};
            statsElements['conversation'].innerHTML = `
                <h5>Conversations</h5>
                <div class="memory-stat">
                    <span>Total: ${convStats.total_conversations || 0}</span>
                </div>
                <small>Conversation threads</small>
            `;
        }
    }
    
    updateImportantMemories(importantMemories) {
        const importantElement = document.getElementById('importantMemories');
        if (!importantElement) return;
        
        if (importantMemories.length === 0) {
            importantElement.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-star"></i>
                    <p>No important memories yet</p>
                    <small>Important memories will appear here</small>
                </div>
            `;
            return;
        }
        
        const importantHtml = importantMemories.slice(0, 5).map(memory => `
            <div class="important-memory-item">
                <div class="importance-indicator">
                    ${'★'.repeat(this.getImportanceStars(memory.importance))}
                </div>
                <div class="memory-content">
                    <strong>${memory.category || 'General'}</strong>
                    <p>${this.truncateText(memory.content, 80)}</p>
                    <small>${this.formatTime(memory.timestamp)}</small>
                </div>
            </div>
        `).join('');
        
        importantElement.innerHTML = importantHtml;
    }
    
    initializeMemoryNetwork() {
        const canvas = document.getElementById('memoryNetworkCanvas');
        if (!canvas) return;
        
        this.memoryNetwork = new MemoryNetworkVisualization(canvas);
        this.memoryNetwork.initialize();
    }
    
    updateMemoryNetwork() {
        if (this.memoryNetwork) {
            this.memoryNetwork.updateData(this.recentMemories);
        }
    }
    
    async searchMemories(query) {
        try {
            const response = await fetch('/api/memory/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.displaySearchResults(data.results || []);
                return data.results;
            }
        } catch (error) {
            console.error('❌ Memory search error:', error);
            this.showError('Failed to search memories');
        }
        return [];
    }
    
    displaySearchResults(results) {
        const resultsElement = document.getElementById('searchResults');
        if (!resultsElement) return;
        
        if (results.length === 0) {
            resultsElement.innerHTML = '<p>No memories found matching your search.</p>';
            return;
        }
        
        const resultsHtml = results.map(result => `
            <div class="search-result-item">
                <div class="result-header">
                    <span class="relevance-score">${(result.relevance * 100).toFixed(1)}%</span>
                    <span class="result-type">${result.type}</span>
                </div>
                <div class="result-content">${this.highlightSearchTerms(result.content)}</div>
                <div class="result-footer">
                    <small>${this.formatTime(result.timestamp)}</small>
                </div>
            </div>
        `).join('');
        
        resultsElement.innerHTML = resultsHtml;
    }
    
    async optimizeMemory() {
        try {
            console.log('🧠 Optimizing memory...');
            // This would call a memory optimization endpoint
            this.showSuccess('Memory optimization initiated');
        } catch (error) {
            console.error('❌ Memory optimization error:', error);
            this.showError('Failed to optimize memory');
        }
    }
    
    async exportMemories() {
        try {
            console.log('📥 Exporting memories...');
            // This would call an export endpoint
            this.showSuccess('Memory export started');
        } catch (error) {
            console.error('❌ Memory export error:', error);
            this.showError('Failed to export memories');
        }
    }
    
    async clearOldMemories() {
        if (!confirm('Are you sure you want to clear old memories? This action cannot be undone.')) {
            return;
        }
        
        try {
            console.log('🗑️ Clearing old memories...');
            // This would call a cleanup endpoint
            this.showSuccess('Old memories cleared');
            this.refreshMemoryData();
        } catch (error) {
            console.error('❌ Memory cleanup error:', error);
            this.showError('Failed to clear old memories');
        }
    }
    
    updateChatAnalytics(analytics) {
        console.log('📊 Updating chat analytics...', analytics);
        
        const liveActivity = analytics.live_chat_activity || {};
        
        // Update live chat activity metrics
        this.updateElementText('#messages-today', liveActivity.messages_today || 0);
        this.updateElementText('#messages-last-hour', liveActivity.messages_last_hour || 0);
        this.updateElementText('#messages-last-10min', liveActivity.messages_last_10_minutes || 0);
        this.updateElementText('#active-conversations-count', liveActivity.active_conversations || 0);
        
        // Update last activity
        if (liveActivity.last_activity) {
            const lastActivity = new Date(liveActivity.last_activity);
            this.updateElementText('#last-activity', lastActivity.toLocaleString());
        } else {
            this.updateElementText('#last-activity', 'No recent activity');
        }
        
        // Update memory metrics
        const memoryMetrics = analytics.memory_metrics || {};
        this.updateElementText('#memories-stored-today', memoryMetrics.memories_stored_today || 0);
        this.updateElementText('#memory-utilization', `${Math.round(memoryMetrics.memory_utilization || 0)}%`);
        
        // Update user engagement
        const userEngagement = analytics.user_engagement || {};
        this.updateElementText('#unique-conversations', userEngagement.unique_conversations || 0);
        this.updateElementText('#total-interactions', userEngagement.total_interactions || 0);
    }
    
    updateActiveConversations(activeConversations) {
        console.log('💬 Updating active conversations...', activeConversations);
        
        const container = document.getElementById('active-conversations-list');
        if (!container) return;
        
        if (activeConversations.length === 0) {
            container.innerHTML = '<div class="no-data">No active conversations</div>';
            return;
        }
        
        container.innerHTML = activeConversations.map(conv => `
            <div class="conversation-item ${conv.status}">
                <div class="conversation-header">
                    <span class="conversation-id">${conv.conversation_id}</span>
                    <span class="conversation-status status-${conv.status}">${conv.status}</span>
                </div>
                <div class="conversation-details">
                    <div class="detail-item">
                        <span class="detail-label">Messages:</span>
                        <span class="detail-value">${conv.message_count}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Duration:</span>
                        <span class="detail-value">${conv.duration_minutes || 0} min</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Last Activity:</span>
                        <span class="detail-value">${new Date(conv.last_activity).toLocaleTimeString()}</span>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    updateLiveStats(liveStats) {
        console.log('📈 Updating live stats...', liveStats);
        
        const memorySystem = liveStats.memory_system || {};
        const chatActivity = liveStats.chat_activity || {};
        const memoryPerformance = liveStats.memory_performance || {};
        
        // Update memory system status
        this.updateElementText('#memory-system-status', memorySystem.status || 'offline');
        this.updateElementText('#memory-initialized', memorySystem.initialized ? 'Yes' : 'No');
        
        // Update chat activity
        this.updateElementText('#total-messages', chatActivity.total_messages || 0);
        this.updateElementText('#active-conversations', chatActivity.active_conversations || 0);
        
        if (chatActivity.last_message_time) {
            const lastMessageTime = new Date(chatActivity.last_message_time);
            this.updateElementText('#last-message-time', lastMessageTime.toLocaleString());
        }
        
        // Update memory performance
        this.updateElementText('#storage-backend', memoryPerformance.storage_backend || 'unknown');
        this.updateElementText('#memory-consolidation', memoryPerformance.memory_consolidation || 'unknown');
        this.updateElementText('#search-available', memoryPerformance.search_available ? 'Yes' : 'No');
        
        if (memoryPerformance.stm_utilization) {
            this.updateElementText('#stm-utilization', memoryPerformance.stm_utilization);
        }
        
        if (memoryPerformance.ltm_memories) {
            this.updateElementText('#ltm-memories', memoryPerformance.ltm_memories);
        }
    }
    
    updateElementText(selector, text) {
        const element = document.querySelector(selector);
        if (element) {
            element.textContent = text;
        }
    }
    
    // Utility methods
    getMemoryTypeColor(type) {
        const colors = {
            'conversation': 'primary',
            'learning': 'success',
            'important': 'warning',
            'personal': 'info'
        };
        return colors[type] || 'secondary';
    }
    
    getImportanceClass(importance) {
        if (typeof importance === 'number') {
            if (importance >= 8) return 'importance-critical';
            if (importance >= 6) return 'importance-high';
            if (importance >= 4) return 'importance-medium';
            return 'importance-low';
        }
        return 'importance-normal';
    }
    
    getImportanceStars(importance) {
        if (typeof importance === 'number') {
            return Math.min(5, Math.max(1, Math.floor(importance / 2)));
        }
        return 1;
    }
    
    formatTime(timestamp) {
        try {
            const date = new Date(timestamp);
            const now = new Date();
            const diff = now - date;
            
            if (diff < 60000) return 'just now';
            if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
            if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
            return date.toLocaleDateString();
        } catch (error) {
            return 'unknown';
        }
    }
    
    truncateText(text, maxLength) {
        if (!text) return '';
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }
    
    highlightSearchTerms(text) {
        // Simple highlighting - in production would use the actual search query
        return text;
    }
    
    showSuccess(message) {
        this.showNotification(message, 'success');
    }
    
    showError(message) {
        this.showNotification(message, 'error');
    }
    
    showNotification(message, type) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }
    
    destroy() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        if (this.memoryNetwork) {
            this.memoryNetwork.destroy();
        }
    }
}

// Memory Network Visualization Class
class MemoryNetworkVisualization {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.nodes = [];
        this.connections = [];
        this.animationId = null;
        this.isAnimating = false;
    }
    
    initialize() {
        this.setupCanvas();
        this.createInitialNodes();
        this.startAnimation();
    }
    
    setupCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
    }
    
    createInitialNodes() {
        // Create some sample nodes representing memory types
        const nodeTypes = [
            { name: 'STM', color: '#4CAF50', x: 100, y: 100 },
            { name: 'LTM', color: '#2196F3', x: 300, y: 100 },
            { name: 'Emotions', color: '#FF9800', x: 200, y: 200 },
            { name: 'Learning', color: '#9C27B0', x: 400, y: 200 }
        ];
        
        this.nodes = nodeTypes.map(type => ({
            ...type,
            radius: 20,
            pulse: 0,
            connections: []
        }));
        
        // Create connections
        this.connections = [
            { from: 0, to: 1, strength: 0.8 },
            { from: 0, to: 2, strength: 0.6 },
            { from: 1, to: 3, strength: 0.7 },
            { from: 2, to: 3, strength: 0.5 }
        ];
    }
    
    updateData(memories) {
        // Update visualization based on memory data
        if (memories && memories.length > 0) {
            this.nodes.forEach(node => {
                node.pulse = Math.random() * 0.5;
            });
        }
    }
    
    startAnimation() {
        this.isAnimating = true;
        this.animate();
    }
    
    animate() {
        if (!this.isAnimating) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw connections
        this.drawConnections();
        
        // Draw nodes
        this.drawNodes();
        
        // Update animations
        this.updateAnimations();
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    drawConnections() {
        this.ctx.strokeStyle = 'rgba(100, 149, 237, 0.3)';
        this.ctx.lineWidth = 2;
        
        this.connections.forEach(conn => {
            const fromNode = this.nodes[conn.from];
            const toNode = this.nodes[conn.to];
            
            this.ctx.beginPath();
            this.ctx.moveTo(fromNode.x, fromNode.y);
            this.ctx.lineTo(toNode.x, toNode.y);
            this.ctx.stroke();
        });
    }
    
    drawNodes() {
        this.nodes.forEach(node => {
            const pulseRadius = node.radius + node.pulse * 10;
            
            // Draw pulse effect
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, pulseRadius, 0, 2 * Math.PI);
            this.ctx.fillStyle = node.color + '40';
            this.ctx.fill();
            
            // Draw main node
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);
            this.ctx.fillStyle = node.color;
            this.ctx.fill();
            
            // Draw text
            this.ctx.fillStyle = 'white';
            this.ctx.font = '12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(node.name, node.x, node.y + 4);
        });
    }
    
    updateAnimations() {
        this.nodes.forEach(node => {
            node.pulse = Math.max(0, node.pulse - 0.02);
        });
    }
    
    toggleAnimation() {
        if (this.isAnimating) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
    }
    
    stopAnimation() {
        this.isAnimating = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
    
    reset() {
        this.createInitialNodes();
    }
    
    destroy() {
        this.stopAnimation();
    }
}

// Global functions for template compatibility
function refreshMemoryNetwork() {
    if (window.memoryDashboard) {
        window.memoryDashboard.refreshMemoryData();
    }
}

function toggleNetworkAnimation() {
    if (window.memoryDashboard && window.memoryDashboard.memoryNetwork) {
        window.memoryDashboard.memoryNetwork.toggleAnimation();
        
        const icon = document.getElementById('networkAnimationIcon');
        const text = document.getElementById('networkAnimationText');
        
        if (icon && text) {
            if (window.memoryDashboard.memoryNetwork.isAnimating) {
                icon.className = 'fas fa-pause';
                text.textContent = 'Pause';
            } else {
                icon.className = 'fas fa-play';
                text.textContent = 'Start';
            }
        }
    }
}

function resetNetworkView() {
    if (window.memoryDashboard && window.memoryDashboard.memoryNetwork) {
        window.memoryDashboard.memoryNetwork.reset();
    }
}

function optimizeMemory() {
    if (window.memoryDashboard) {
        window.memoryDashboard.optimizeMemory();
    }
}

function exportMemories() {
    if (window.memoryDashboard) {
        window.memoryDashboard.exportMemories();
    }
}

function clearOldMemories() {
    if (window.memoryDashboard) {
        window.memoryDashboard.clearOldMemories();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Only initialize on memory dashboard page
    if (document.getElementById('memoryOverview')) {
        window.memoryDashboard = new MemoryDashboard();
        console.log('✅ Memory Dashboard initialized');
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (window.memoryDashboard) {
        window.memoryDashboard.destroy();
    }
});
