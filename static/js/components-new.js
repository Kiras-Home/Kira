/**
 * Kira Components JavaScript
 * Wiederverwendbare UI-Komponenten und Utilities
 */

// Modal System
class KiraModal {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            backdrop: true,
            keyboard: true,
            ...options
        };
        this.isOpen = false;
        this.createModal();
    }

    createModal() {
        const modal = document.createElement('div');
        modal.id = this.id;
        modal.className = 'kira-modal';
        modal.innerHTML = `
            <div class="kira-modal-backdrop"></div>
            <div class="kira-modal-dialog">
                <div class="kira-modal-content">
                    <div class="kira-modal-header">
                        <h5 class="kira-modal-title"></h5>
                        <button type="button" class="kira-modal-close" onclick="closeModal('${this.id}')">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="kira-modal-body"></div>
                    <div class="kira-modal-footer"></div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    show(title, content, footer = '') {
        const modal = document.getElementById(this.id);
        if (!modal) return;

        modal.querySelector('.kira-modal-title').textContent = title;
        modal.querySelector('.kira-modal-body').innerHTML = content;
        modal.querySelector('.kira-modal-footer').innerHTML = footer;
        
        modal.style.display = 'block';
        this.isOpen = true;
        
        // Add event listeners
        if (this.options.backdrop) {
            modal.querySelector('.kira-modal-backdrop').addEventListener('click', () => {
                this.hide();
            });
        }
        
        if (this.options.keyboard) {
            document.addEventListener('keydown', this.handleKeydown.bind(this));
        }
    }

    hide() {
        const modal = document.getElementById(this.id);
        if (modal) {
            modal.style.display = 'none';
            this.isOpen = false;
            document.removeEventListener('keydown', this.handleKeydown.bind(this));
        }
    }

    handleKeydown(e) {
        if (e.key === 'Escape' && this.isOpen) {
            this.hide();
        }
    }
}

// Toast Notifications
class KiraToast {
    static show(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `kira-toast kira-toast-${type}`;
        toast.innerHTML = `
            <div class="kira-toast-content">
                <i class="fas fa-${this.getIcon(type)}"></i>
                <span>${message}</span>
            </div>
            <button class="kira-toast-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Add to container or create one
        let container = document.getElementById('kira-toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'kira-toast-container';
            container.className = 'kira-toast-container';
            document.body.appendChild(container);
        }
        
        container.appendChild(toast);
        
        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.remove();
                }
            }, duration);
        }
    }

    static getIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-triangle',
            warning: 'exclamation-circle',
            info: 'info-circle'
        };
        return icons[type] || icons.info;
    }
}

// Loading Spinner
function showLoading(element, text = 'Loading...') {
    if (typeof element === 'string') {
        element = document.getElementById(element);
    }
    
    if (element) {
        element.innerHTML = `
            <div class="kira-loading">
                <div class="kira-spinner"></div>
                <span>${text}</span>
            </div>
        `;
    }
}

function hideLoading(element, content = '') {
    if (typeof element === 'string') {
        element = document.getElementById(element);
    }
    
    if (element) {
        element.innerHTML = content;
    }
}

// Global Component Functions
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
}

function showToast(message, type = 'info', duration = 3000) {
    KiraToast.show(message, type, duration);
}

// Initialize Components
document.addEventListener('DOMContentLoaded', function() {
    console.log('Components JS loaded');
    
    // Make components globally available
    window.KiraModal = KiraModal;
    window.KiraToast = KiraToast;
    window.showLoading = showLoading;
    window.hideLoading = hideLoading;
    window.closeModal = closeModal;
    window.showToast = showToast;
});
