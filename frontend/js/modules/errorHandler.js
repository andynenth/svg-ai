'use strict';

/**
 * Error Handling Module
 * Centralized error display, logging, and reporting
 */

class ErrorHandler {
    constructor() {
        this.errorContainer = null;
        this.createErrorContainer();
        this.setupGlobalErrorHandling();
    }

    createErrorContainer() {
        // Create error display container if it doesn't exist
        this.errorContainer = document.getElementById('errorContainer');

        if (!this.errorContainer) {
            this.errorContainer = document.createElement('div');
            this.errorContainer.id = 'errorContainer';
            this.errorContainer.className = 'error-container';
            document.body.appendChild(this.errorContainer);
        }
    }

    /**
     * Display error to user with user-friendly message
     * @param {string} message - User-friendly error message
     * @param {Object} options - Error options
     * @param {string} options.type - Error type ('error', 'warning', 'info')
     * @param {number} options.duration - Auto-hide duration in ms
     * @param {boolean} options.persistent - Whether error persists until manually dismissed
     */
    showUserError(message, options = {}) {
        const {
            type = 'error',
            duration = 5000,
            persistent = false
        } = options;

        // Remove alert() calls - create proper error display
        const errorElement = document.createElement('div');
        errorElement.className = `error-message error-${type}`;

        // Sanitize message to prevent XSS
        errorElement.innerHTML = DOMPurify.sanitize(`
            <div class="error-content">
                <div class="error-icon">${this.getErrorIcon(type)}</div>
                <div class="error-text">${message}</div>
                <button class="error-dismiss" aria-label="Dismiss error">×</button>
            </div>
        `);

        // Add dismiss functionality
        const dismissBtn = errorElement.querySelector('.error-dismiss');
        dismissBtn.addEventListener('click', () => {
            this.dismissError(errorElement);
        });

        // Add to container
        this.errorContainer.appendChild(errorElement);

        // Auto-dismiss after duration (unless persistent)
        if (!persistent && duration > 0) {
            setTimeout(() => {
                this.dismissError(errorElement);
            }, duration);
        }

        // Animate in
        requestAnimationFrame(() => {
            errorElement.classList.add('show');
        });

        return errorElement;
    }

    /**
     * Log technical error details for developers
     * @param {Error} error - The error object
     * @param {string} context - Context where error occurred
     * @param {Object} metadata - Additional error metadata
     */
    logError(error, context = 'Unknown', metadata = {}) {
        const errorDetails = {
            message: error.message || 'Unknown error',
            stack: error.stack,
            context,
            timestamp: new Date().toISOString(),
            url: window.location.href,
            userAgent: navigator.userAgent,
            ...metadata
        };

        // Log to console with full context
        console.group(`🚨 Error in ${context}`);
        console.error('Message:', errorDetails.message);
        console.error('Stack:', errorDetails.stack);
        console.error('Context:', errorDetails.context);
        console.error('Metadata:', metadata);
        console.groupEnd();

        // Send to error reporting service (if configured)
        this.reportError(errorDetails);

        return errorDetails;
    }

    /**
     * Handle different types of errors with appropriate user/developer feedback
     * @param {Error} error - The error object
     * @param {string} context - Context where error occurred
     * @param {Object} options - Error handling options
     */
    handleError(error, context = 'Unknown', options = {}) {
        const {
            showUser = true,
            userMessage = null,
            logToConsole = true,
            reportToService = true,
            metadata = {}
        } = options;

        // Log technical details for developers
        if (logToConsole) {
            this.logError(error, context, metadata);
        }

        // Show user-friendly message
        if (showUser) {
            const message = userMessage || this.getUserFriendlyMessage(error, context);
            this.showUserError(message, { type: 'error' });
        }

        // Report to error service
        if (reportToService) {
            this.reportError({ error, context, metadata });
        }

        return { error, context, handled: true };
    }

    /**
     * Handle specific error types with contextual messages
     */
    handleUploadError(error, options = {}) {
        const userMessage = this.getUploadErrorMessage(error);
        return this.handleError(error, 'Upload', {
            userMessage,
            metadata: { module: 'upload' },
            ...options
        });
    }

    handleConversionError(error, options = {}) {
        const userMessage = this.getConversionErrorMessage(error);
        return this.handleError(error, 'Conversion', {
            userMessage,
            metadata: { module: 'converter' },
            ...options
        });
    }

    handleUIError(error, options = {}) {
        const userMessage = 'A display error occurred. Please refresh the page.';
        return this.handleError(error, 'UI', {
            userMessage,
            metadata: { module: 'ui' },
            ...options
        });
    }

    handleSplitViewError(error, options = {}) {
        const userMessage = 'An error occurred with the image comparison view.';
        return this.handleError(error, 'SplitView', {
            userMessage,
            metadata: { module: 'splitView' },
            ...options
        });
    }

    /**
     * Generate user-friendly error messages based on error type
     */
    getUserFriendlyMessage(error, context) {
        const message = error.message?.toLowerCase() || '';

        // Network errors
        if (message.includes('network') || message.includes('fetch')) {
            return 'Network connection problem. Please check your internet connection.';
        }

        // File errors
        if (message.includes('file') || context === 'Upload') {
            return 'File upload failed. Please try again with a valid PNG or JPEG image.';
        }

        // Conversion errors
        if (context === 'Conversion') {
            return 'Image conversion failed. Please try with a different image or converter.';
        }

        // Generic fallback
        return 'An unexpected error occurred. Please try again.';
    }

    getUploadErrorMessage(error) {
        const message = error.message?.toLowerCase() || '';

        if (message.includes('type') || message.includes('format')) {
            return 'Invalid file type. Please upload a PNG or JPEG image.';
        }
        if (message.includes('size')) {
            return 'File too large. Please use an image smaller than 10MB.';
        }
        if (message.includes('network') || message.includes('fetch')) {
            return 'Upload failed due to network error. Please check your connection.';
        }

        return 'File upload failed. Please try again.';
    }

    getConversionErrorMessage(error) {
        const message = error.message?.toLowerCase() || '';

        if (message.includes('timeout')) {
            return 'Conversion is taking longer than expected. Please try with a smaller image.';
        }
        if (message.includes('format') || message.includes('invalid')) {
            return 'Image format not supported for conversion. Please try a different image.';
        }
        if (message.includes('server') || message.includes('500')) {
            return 'Server error during conversion. Please try again in a moment.';
        }

        return 'Image conversion failed. Please try again or use a different converter.';
    }

    getErrorIcon(type) {
        switch (type) {
            case 'error': return '❌';
            case 'warning': return '⚠️';
            case 'info': return 'ℹ️';
            case 'success': return '✅';
            default: return '❌';
        }
    }

    dismissError(errorElement) {
        if (errorElement && errorElement.parentNode) {
            errorElement.classList.add('hide');
            setTimeout(() => {
                if (errorElement.parentNode) {
                    errorElement.parentNode.removeChild(errorElement);
                }
            }, 300); // Match CSS transition duration
        }
    }

    clearAllErrors() {
        const errors = this.errorContainer.querySelectorAll('.error-message');
        errors.forEach(error => this.dismissError(error));
    }

    /**
     * Set up global error handling for uncaught errors
     */
    setupGlobalErrorHandling() {
        // Handle uncaught JavaScript errors
        window.addEventListener('error', (event) => {
            this.handleError(event.error || new Error(event.message), 'Global', {
                metadata: {
                    filename: event.filename,
                    lineno: event.lineno,
                    colno: event.colno
                }
            });
        });

        // Handle unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            this.handleError(
                event.reason instanceof Error ? event.reason : new Error(String(event.reason)),
                'Promise',
                { metadata: { type: 'unhandledrejection' } }
            );
        });
    }

    /**
     * Report error to external service (placeholder for future implementation)
     */
    reportError(errorDetails) {
        // Placeholder for error reporting service integration
        // Could send to services like Sentry, LogRocket, etc.
        console.debug('[ErrorReporting] Error logged:', errorDetails);
    }

    /**
     * Create error boundary for module functions
     */
    createErrorBoundary(fn, context, options = {}) {
        return async (...args) => {
            try {
                return await fn(...args);
            } catch (error) {
                this.handleError(error, context, options);
                throw error; // Re-throw so calling code can handle if needed
            }
        };
    }

    /**
     * Wrap module methods with error boundaries
     */
    wrapModule(moduleInstance, moduleName) {
        const methodNames = Object.getOwnPropertyNames(Object.getPrototypeOf(moduleInstance))
            .filter(name => name !== 'constructor' && typeof moduleInstance[name] === 'function');

        methodNames.forEach(methodName => {
            const originalMethod = moduleInstance[methodName];
            moduleInstance[methodName] = this.createErrorBoundary(
                originalMethod.bind(moduleInstance),
                `${moduleName}.${methodName}`,
                { metadata: { module: moduleName, method: methodName } }
            );
        });

        return moduleInstance;
    }
}

// Create singleton instance
const errorHandler = new ErrorHandler();

// Export singleton
export default errorHandler;