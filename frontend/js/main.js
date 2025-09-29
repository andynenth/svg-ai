'use strict';

/**
 * Main Application Module
 * Orchestrates all modules and handles overall application logic
 */

import UploadModule from './modules/upload.js';
import ConverterModule from './modules/converter.js';
import UIModule from './modules/ui.js';
import { SplitViewController } from './modules/splitView.js';
import appState from './modules/appState.js';
import errorHandler from './modules/errorHandler.js';
import './modules/logoClassifier.js';

class MainApplication {
    constructor() {
        this.apiBase = '';
        this.modules = {};
        this.initializeModules();
        this.setupInterModuleCommunication();
        console.log('[INIT] Application initialized with modular architecture');
    }

    initializeModules() {
        // Initialize modules
        this.modules.upload = new UploadModule(this.apiBase);
        this.modules.converter = new ConverterModule(this.apiBase);
        this.modules.ui = new UIModule();
        this.modules.splitView = new SplitViewController();

        console.log('[INIT] All modules initialized');
    }

    setupInterModuleCommunication() {
        // Upload completion handler
        document.addEventListener('uploadComplete', (e) => {
            console.log('[Main] Upload completed, file ID:', e.detail.fileId);

            // Notify converter module
            this.modules.converter.currentFileId = e.detail.fileId;

            // Trigger automatic conversion after upload
            setTimeout(() => {
                console.log('[Auto-convert] File uploaded, starting automatic conversion');
                this.modules.converter.handleConvert();
            }, 500);
        });

        // Image loaded handler
        document.addEventListener('imageLoaded', (e) => {
            console.log('[Main] Image loaded, initializing UI features');
            // UI module handles this internally
        });

        // Conversion completion handler
        document.addEventListener('conversionComplete', (e) => {
            console.log('[Main] Conversion completed');
            // Split view module listens for this event
        });

        // Download request handler
        document.addEventListener('requestSvgContent', () => {
            const svgContent = appState.currentSvgContent;
            if (svgContent) {
                this.handleDownload(svgContent);
            } else {
                this.modules.ui.showError('No SVG content available to download');
            }
        });

        // Application reset handler
        document.addEventListener('resetApplication', () => {
            console.log('[Main] Resetting application state');
            this.resetAllModules();
        });
    }

    handleDownload(svgContent) {
        const blob = new Blob([svgContent], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');

        a.href = url;
        a.download = 'converted.svg';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        // Clean up object URL
        setTimeout(() => URL.revokeObjectURL(url), 100);

        console.log('[Download] SVG file downloaded');
    }

    resetAllModules() {
        // Reset app state
        appState.reset();

        // Reset all modules to initial state
        Object.values(this.modules).forEach(module => {
            if (typeof module.reset === 'function') {
                module.reset();
            }
        });

        console.log('[Main] All modules reset');
    }

    // Global error handler
    handleGlobalError(error, context = 'Unknown') {
        errorHandler.handleError(error, context, {
            metadata: { source: 'global' }
        });
    }

    // Global success handler
    handleGlobalSuccess(message, context = 'Operation') {
        errorHandler.showUserError(message, { type: 'success', duration: 3000 });
    }
}

// Global variables for backward compatibility with existing code
let splitViewController = null;

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('[INIT] DOM loaded, initializing modular application');

    // Create main application instance
    const app = new MainApplication();

    // Set up global references for backward compatibility
    splitViewController = app.modules.splitView;

    // Expose appState properties globally for backward compatibility
    Object.defineProperty(window, 'currentFileId', {
        get: () => appState.currentFileId,
        set: (value) => { appState.currentFileId = value; }
    });

    Object.defineProperty(window, 'currentSvgContent', {
        get: () => appState.currentSvgContent,
        set: (value) => { appState.currentSvgContent = value; }
    });

    // Global error handling
    window.addEventListener('error', (e) => {
        app.handleGlobalError(e.error, 'Global');
    });

    window.addEventListener('unhandledrejection', (e) => {
        app.handleGlobalError(e.reason, 'Promise');
    });

    // Make app, state, and error handler available globally for debugging
    window.svgApp = app;
    window.appState = appState;
    window.errorHandler = errorHandler;

    console.log('[INIT] Application fully initialized');
});

// Export for potential module use
export default MainApplication;