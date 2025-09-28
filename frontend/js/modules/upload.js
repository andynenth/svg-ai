'use strict';

/**
 * Upload Module
 * Handles file drag & drop, upload progress, and file validation
 */

import appState from './appState.js';
import errorHandler from './errorHandler.js';

class UploadModule {
    constructor(apiBase = '') {
        this.apiBase = apiBase;
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        this.dropzone = document.getElementById('dropzone');
        this.fileInput = document.getElementById('fileInput');
        this.mainContent = document.getElementById('mainContent');
    }

    setupEventListeners() {
        // Setup Click to Upload
        this.dropzone.addEventListener('click', () => {
            this.fileInput.click();
        });

        // Setup Drag and Drop
        this.dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropzone.classList.add('dragover');
        });

        this.dropzone.addEventListener('dragleave', () => {
            this.dropzone.classList.remove('dragover');
        });

        this.dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropzone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            this.handleFile(file);
        });

        // Handle File Input Change
        this.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            this.handleFile(file);
        });
    }

    async handleFile(file) {
        try {
            if (!file) return;

            if (!file.type.match('image/(png|jpeg|jpg)')) {
                errorHandler.handleUploadError(
                    new Error('Invalid file type'),
                    { userMessage: 'Please upload a PNG or JPEG image' }
                );
                return;
            }

            // Show progressive loading
            this.showImagePlaceholder(file);
            appState.isUploading = true;

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.apiBase}/api/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed with status ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            appState.currentFileId = data.file_id;
            appState.isUploading = false;

            // Display Original Image with memory-efficient object URL
            this.displayUploadedImage(file);

            // Hide drop zone and show main content
            this.showMainContent();

            // Emit upload complete event
            this.emitUploadComplete(data.file_id);

        } catch (error) {
            appState.isUploading = false;
            this.hideImagePlaceholder();
            errorHandler.handleUploadError(error, {
                metadata: { fileSize: file?.size, fileType: file?.type }
            });
        }
    }

    displayUploadedImage(file) {
        const imageElement = document.getElementById('originalImage');

        // Clean up any existing object URL to prevent memory leaks
        if (imageElement.src && imageElement.src.startsWith('blob:')) {
            URL.revokeObjectURL(imageElement.src);
        }

        // Create object URL for immediate, memory-efficient display
        const objectURL = URL.createObjectURL(file);
        imageElement.src = objectURL;

        // Store object URL for cleanup later
        imageElement.dataset.objectUrl = objectURL;

        // Progressive loading completion
        imageElement.onload = () => {
            console.log('[Progressive] Image loaded successfully');
            this.hideImagePlaceholder();

            // Remove inline opacity and add loaded class for CSS transition
            imageElement.style.opacity = '';
            imageElement.classList.add('loaded');

            // Emit image loaded event
            this.emitImageLoaded(imageElement);
        };

        // Handle image load errors
        imageElement.onerror = () => {
            this.hideImagePlaceholder();
            errorHandler.handleUploadError(
                new Error('Failed to load uploaded image'),
                { userMessage: 'The uploaded image could not be displayed. Please try again.' }
            );
        };

        // Fallback timeout to hide placeholder if something goes wrong
        setTimeout(() => {
            if (!imageElement.classList.contains('loaded')) {
                console.warn('[Progressive] Image taking too long, hiding placeholder');
                this.hideImagePlaceholder();
            }
        }, 5000); // 5 second timeout
    }

    showImagePlaceholder(file) {
        const placeholder = document.getElementById('imagePlaceholder');
        const imageInfo = document.getElementById('imageInfo');
        const imageElement = document.getElementById('originalImage');

        // Reset image state completely
        if (imageElement) {
            imageElement.classList.remove('loaded');
            imageElement.src = '';
            imageElement.style.opacity = '0';
        }

        // Ensure placeholder is visible (only if it exists)
        if (placeholder) {
            placeholder.classList.remove('hidden');
            placeholder.style.opacity = '1';
            placeholder.style.display = 'flex';
        }

        // Display file information (only if element exists)
        if (imageInfo) {
            const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
            // Sanitize user-provided filename and type to prevent XSS
            imageInfo.innerHTML = DOMPurify.sanitize(`
                <strong>${file.name}</strong><br>
                Size: ${fileSizeMB} MB<br>
                Type: ${file.type}<br>
                <em>Loading preview...</em>
            `);
        }
    }

    hideImagePlaceholder() {
        const placeholder = document.getElementById('imagePlaceholder');
        if (placeholder) {
            // Immediate hide with CSS class
            placeholder.classList.add('hidden');
            console.log('[Progressive] Placeholder hidden');
        }
    }

    showMainContent() {
        document.querySelector('.upload-section').style.display = 'none';
        this.mainContent.classList.remove('hidden');
    }

    showError(message) {
        errorHandler.showUserError(message, { type: 'error' });
    }

    getCurrentFileId() {
        return appState.currentFileId;
    }

    // Event emitters for module communication
    emitUploadComplete(fileId) {
        const event = new CustomEvent('uploadComplete', {
            detail: { fileId }
        });
        document.dispatchEvent(event);
    }

    emitImageLoaded(imageElement) {
        const event = new CustomEvent('imageLoaded', {
            detail: { imageElement }
        });
        document.dispatchEvent(event);
    }

    // Reset state for new file
    reset() {
        appState.currentFileId = null;

        // Show drop zone again and reset state
        document.querySelector('.upload-section').style.display = 'block';
        this.mainContent.classList.add('hidden');

        // Clean up image
        const imageElement = document.getElementById('originalImage');
        if (imageElement && imageElement.src && imageElement.src.startsWith('blob:')) {
            URL.revokeObjectURL(imageElement.src);
        }
    }
}

// Export for module use
export default UploadModule;