'use strict';

/**
 * Application State Management Module
 * Centralized state management to replace global variables
 */

class AppState {
    constructor() {
        this.state = {
            currentFileId: null,
            currentSvgContent: null,
            isUploading: false,
            isConverting: false,
            conversionResults: null,
            splitViewState: {
                splitPercentage: 50,
                zoomLevel: 1,
                isDragging: false
            },
            uiState: {
                activeConverter: 'smart_auto',
                showMetrics: false,
                showRoutingInfo: false
            }
        };

        this.listeners = new Map();
        this.eventTarget = new EventTarget();
    }

    // Get current state
    get(key) {
        if (key.includes('.')) {
            return this.getNestedValue(key);
        }
        return this.state[key];
    }

    // Set state value and notify listeners
    set(key, value) {
        const oldValue = this.get(key);

        if (key.includes('.')) {
            this.setNestedValue(key, value);
        } else {
            this.state[key] = value;
        }

        // Notify listeners if value changed
        if (oldValue !== value) {
            this.notifyListeners(key, value, oldValue);
            this.emitStateChange(key, value, oldValue);
        }
    }

    // Update multiple state values
    update(updates) {
        const changes = [];

        Object.entries(updates).forEach(([key, value]) => {
            const oldValue = this.get(key);

            if (key.includes('.')) {
                this.setNestedValue(key, value);
            } else {
                this.state[key] = value;
            }

            if (oldValue !== value) {
                changes.push({ key, value, oldValue });
            }
        });

        // Notify all changes
        changes.forEach(({ key, value, oldValue }) => {
            this.notifyListeners(key, value, oldValue);
            this.emitStateChange(key, value, oldValue);
        });
    }

    // Subscribe to state changes
    subscribe(key, callback) {
        if (!this.listeners.has(key)) {
            this.listeners.set(key, new Set());
        }
        this.listeners.get(key).add(callback);

        // Return unsubscribe function
        return () => {
            const keyListeners = this.listeners.get(key);
            if (keyListeners) {
                keyListeners.delete(callback);
                if (keyListeners.size === 0) {
                    this.listeners.delete(key);
                }
            }
        };
    }

    // Listen to state change events
    addEventListener(eventType, callback) {
        this.eventTarget.addEventListener(eventType, callback);
    }

    removeEventListener(eventType, callback) {
        this.eventTarget.removeEventListener(eventType, callback);
    }

    // Reset state to initial values
    reset() {
        const oldState = { ...this.state };

        this.state = {
            currentFileId: null,
            currentSvgContent: null,
            isUploading: false,
            isConverting: false,
            conversionResults: null,
            splitViewState: {
                splitPercentage: 50,
                zoomLevel: 1,
                isDragging: false
            },
            uiState: {
                activeConverter: 'smart_auto',
                showMetrics: false,
                showRoutingInfo: false
            }
        };

        // Notify reset
        this.emitStateChange('__reset__', this.state, oldState);
    }

    // Helper methods
    getNestedValue(key) {
        const keys = key.split('.');
        let value = this.state;

        for (const k of keys) {
            if (value && typeof value === 'object' && k in value) {
                value = value[k];
            } else {
                return undefined;
            }
        }

        return value;
    }

    setNestedValue(key, value) {
        const keys = key.split('.');
        let current = this.state;

        for (let i = 0; i < keys.length - 1; i++) {
            const k = keys[i];
            if (!(k in current) || typeof current[k] !== 'object') {
                current[k] = {};
            }
            current = current[k];
        }

        current[keys[keys.length - 1]] = value;
    }

    notifyListeners(key, value, oldValue) {
        const keyListeners = this.listeners.get(key);
        if (keyListeners) {
            keyListeners.forEach(callback => {
                try {
                    callback(value, oldValue, key);
                } catch (error) {
                    console.error(`Error in state listener for ${key}:`, error);
                }
            });
        }
    }

    emitStateChange(key, value, oldValue) {
        const event = new CustomEvent('stateChange', {
            detail: { key, value, oldValue }
        });
        this.eventTarget.dispatchEvent(event);

        // Also emit a specific event for the key
        const keyEvent = new CustomEvent(`stateChange:${key}`, {
            detail: { value, oldValue }
        });
        this.eventTarget.dispatchEvent(keyEvent);
    }

    // Convenience getters for commonly used state
    get currentFileId() {
        return this.state.currentFileId;
    }

    set currentFileId(value) {
        this.set('currentFileId', value);
    }

    get currentSvgContent() {
        return this.state.currentSvgContent;
    }

    set currentSvgContent(value) {
        this.set('currentSvgContent', value);
    }

    get isUploading() {
        return this.state.isUploading;
    }

    set isUploading(value) {
        this.set('isUploading', value);
    }

    get isConverting() {
        return this.state.isConverting;
    }

    set isConverting(value) {
        this.set('isConverting', value);
    }

    // Debug helper
    getFullState() {
        return JSON.parse(JSON.stringify(this.state));
    }
}

// Create singleton instance
const appState = new AppState();

// Export singleton instance
export default appState;