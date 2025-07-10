/* ===== KIRA DARK MODE SYSTEM ===== */

class KiraDarkMode {
    constructor() {
        this.init();
    }

    init() {
        // Check saved preference
        const savedMode = localStorage.getItem('kira-dark-mode');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        if (savedMode === 'enabled' || (savedMode === null && prefersDark)) {
            this.enableDarkMode();
        }

        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (localStorage.getItem('kira-dark-mode') === null) {
                if (e.matches) {
                    this.enableDarkMode();
                } else {
                    this.disableDarkMode();
                }
            }
        });

        console.log('🌙 Kira Dark Mode System initialized');
    }

    enableDarkMode() {
        document.body.classList.add('dark-mode');
        localStorage.setItem('kira-dark-mode', 'enabled');
        this.updateToggleButton(true);
        console.log('🌙 Dark Mode enabled');
    }

    disableDarkMode() {
        document.body.classList.remove('dark-mode');
        localStorage.setItem('kira-dark-mode', 'disabled');
        this.updateToggleButton(false);
        console.log('☀️ Light Mode enabled');
    }

    toggle() {
        if (document.body.classList.contains('dark-mode')) {
            this.disableDarkMode();
        } else {
            this.enableDarkMode();
        }
    }

    updateToggleButton(isDark) {
        const toggleBtn = document.getElementById('darkModeToggle');
        if (toggleBtn) {
            const icon = toggleBtn.querySelector('i');
            if (icon) {
                if (isDark) {
                    icon.className = 'fas fa-sun';
                    toggleBtn.title = 'Switch to Light Mode';
                } else {
                    icon.className = 'fas fa-moon';
                    toggleBtn.title = 'Switch to Dark Mode';
                }
            }
        }
    }

    isDarkMode() {
        return document.body.classList.contains('dark-mode');
    }
}

// Global functions for HTML onclick events
let kiraDarkMode = null;

function toggleDarkMode() {
    if (kiraDarkMode) {
        kiraDarkMode.toggle();
    }
}

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    kiraDarkMode = new KiraDarkMode();
});