'use strict';

/**
 * UI Module
 * Handles DOM manipulation, event handling, and state management
 */

import errorHandler from './errorHandler.js';

class UIModule {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.initializeTooltips();
    }

    initializeElements() {
        this.downloadBtn = document.getElementById('downloadBtn');
        this.newFileBtn = document.getElementById('newFileBtn');
    }

    setupEventListeners() {
        // Download button
        this.downloadBtn.addEventListener('click', () => this.handleDownload());

        // New file button
        this.newFileBtn.addEventListener('click', () => this.handleNewFile());

        // Listen for conversion events
        document.addEventListener('conversionComplete', (e) => {
            this.displayOptimizedSVG(e.detail.svgContent);
        });

        // Listen for image loaded events
        document.addEventListener('imageLoaded', (e) => {
            this.adjustContainerSizing(e.detail.imageElement);
            this.initializeOriginalImageZoom();
        });

        // Memory management
        window.addEventListener('beforeunload', () => this.cleanupObjectURLs());
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.cleanupObjectURLs();
            }
        });

        // Responsive handling
        window.addEventListener('resize', () => {
            const imageElement = document.getElementById('originalImage');
            if (imageElement && imageElement.classList.contains('loaded')) {
                this.adjustContainerSizing(imageElement);
            }
        });

        // Initialize tooltips when DOM is ready
        document.addEventListener('DOMContentLoaded', () => {
            setTimeout(() => this.initializeTooltips(), 100);
        });

        // Also initialize on window load as backup
        window.addEventListener('load', () => {
            setTimeout(() => this.initializeTooltips(), 100);
        });
    }

    handleDownload() {
        // Get SVG content from converter module
        const event = new CustomEvent('requestSvgContent');
        document.dispatchEvent(event);

        // This will be handled by the main.js orchestrator
        // that has access to all modules
    }

    handleNewFile() {
        // Show drop zone again and reset state
        document.querySelector('.upload-section').style.display = 'block';
        document.getElementById('mainContent').classList.add('hidden');

        // Emit reset event for all modules
        const event = new CustomEvent('resetApplication');
        document.dispatchEvent(event);

        // Clear split view images
        const splitOriginalImg = document.getElementById('splitOriginalImage');
        const splitSvgContainer = document.getElementById('splitSvgContainer');
        if (splitOriginalImg) splitOriginalImg.style.display = 'none';
        if (splitSvgContainer) splitSvgContainer.innerHTML = '';

        // Hide metrics
        document.getElementById('metrics').classList.add('hidden');
    }

    adjustContainerSizing(imageElement) {
        const container = imageElement.closest('.image-container');
        const svgContainer = document.getElementById('svgContainer');

        // Skip if no container found (grid view removed)
        if (!container) {
            console.log('[Sizing] No container found, skipping sizing adjustment');
            return;
        }

        // Get actual image dimensions
        const naturalWidth = imageElement.naturalWidth;
        const naturalHeight = imageElement.naturalHeight;
        const aspectRatio = naturalWidth / naturalHeight;

        // Calculate optimal display dimensions
        const containerWidth = container.clientWidth - 30; // Account for padding
        const maxHeight = window.innerWidth < 768 ? 350 : 500; // Mobile vs desktop
        const minHeight = window.innerWidth < 768 ? 150 : 200;

        // Calculate height based on aspect ratio
        let optimalHeight = containerWidth / aspectRatio;

        // Clamp to min/max bounds
        optimalHeight = Math.max(minHeight, Math.min(maxHeight, optimalHeight));

        // Apply consistent height to both containers
        const heightPx = Math.round(optimalHeight) + 'px';

        // Update image container
        imageElement.style.height = heightPx;

        // Update SVG container to match
        if (svgContainer) {
            svgContainer.style.height = heightPx;
        }

        console.log('[Sizing] Applied consistent height:', heightPx, 'for aspect ratio:', aspectRatio.toFixed(2));
    }

    initializeOriginalImageZoom() {
        const controls = document.querySelector('.image-controls');
        const wrapper = document.querySelector('.image-wrapper');
        if (!controls || !wrapper) return;

        let currentZoom = 1;
        const minZoom = 0.25;
        const maxZoom = 4;
        const zoomStep = 0.25;

        controls.addEventListener('click', (e) => {
            if (!e.target.matches('.zoom-btn')) return;

            const action = e.target.getAttribute('data-action');
            const target = e.target.getAttribute('data-target');

            if (target !== 'original') return;

            switch (action) {
                case 'zoom-in':
                    currentZoom = Math.min(maxZoom, currentZoom + zoomStep);
                    break;
                case 'zoom-out':
                    currentZoom = Math.max(minZoom, currentZoom - zoomStep);
                    break;
                case 'zoom-reset':
                    currentZoom = 1;
                    break;
            }

            // Apply zoom
            const imageElement = document.getElementById('originalImage');
            if (imageElement) {
                imageElement.style.transform = `scale(${currentZoom})`;
            }
        });

        // Mouse wheel zoom
        wrapper.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -zoomStep : zoomStep;
            currentZoom = Math.max(minZoom, Math.min(maxZoom, currentZoom + delta));

            const imageElement = document.getElementById('originalImage');
            if (imageElement) {
                imageElement.style.transform = `scale(${currentZoom})`;
            }
        });
    }

    displayOptimizedSVG(svgContent) {
        const container = document.getElementById('svgContainer');

        // Clear container
        container.innerHTML = '';

        // Create wrapper for zoom functionality
        const svgWrapper = document.createElement('div');
        svgWrapper.className = 'svg-wrapper';

        // Sanitize SVG content to prevent XSS attacks
        svgWrapper.innerHTML = DOMPurify.sanitize(svgContent);

        // Get SVG element
        const svgElement = svgWrapper.querySelector('svg');
        if (!svgElement) {
            container.innerHTML = '<p class="error">Invalid SVG content</p>';
            return;
        }

        // Optimize SVG attributes for proper scaling
        this.optimizeSVGAttributes(svgElement);

        // Only add container and wrapper (no zoom controls - split view has its own)
        container.appendChild(svgWrapper);

        console.log('[SVG] Optimized SVG display complete');
    }

    optimizeSVGAttributes(svgElement) {
        // Ensure proper viewBox for scaling
        if (!svgElement.getAttribute('viewBox')) {
            const width = svgElement.getAttribute('width') || '100';
            const height = svgElement.getAttribute('height') || '100';
            svgElement.setAttribute('viewBox', `0 0 ${width} ${height}`);
        }

        // Remove fixed dimensions to allow responsive scaling
        svgElement.removeAttribute('width');
        svgElement.removeAttribute('height');

        // Set responsive attributes
        svgElement.setAttribute('width', '100%');
        svgElement.setAttribute('height', '100%');
        svgElement.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    }

    displayRoutingInfo(routingInfo) {
        // Update routing information display
        document.getElementById('routedTo').textContent = routingInfo.routed_to || '-';
        document.getElementById('imageType').textContent = routingInfo.is_colored ? 'Colored' : 'Grayscale/B&W';
        document.getElementById('routingConfidence').textContent = (routingInfo.routing_confidence * 100).toFixed(1) + '%';
        document.getElementById('uniqueColors').textContent = routingInfo.unique_colors || '-';

        // Show routing info section
        document.getElementById('routingInfo').classList.remove('hidden');

        console.log('[Frontend] Routing info displayed:', {
            routed_to: routingInfo.routed_to,
            is_colored: routingInfo.is_colored,
            confidence: routingInfo.routing_confidence,
            unique_colors: routingInfo.unique_colors
        });
    }

    // Flowbite-style Tooltip System
    initializeTooltips() {
        // Remove existing event listeners by cloning icons
        const infoIcons = document.querySelectorAll('.info-icon[data-tooltip]');
        console.log(`[Tooltips] Found ${infoIcons.length} info icons`);

        infoIcons.forEach((icon, index) => {
            // Clone to remove old event listeners
            const newIcon = icon.cloneNode(true);
            icon.parentNode.replaceChild(newIcon, icon);

            let tooltip = null;
            console.log(`[Tooltips] Setting up icon ${index + 1}:`, newIcon.getAttribute('data-tooltip'));

            // Create tooltip on mouseenter
            newIcon.addEventListener('mouseenter', () => {
                console.log('[Tooltips] Mouseenter on icon:', newIcon.getAttribute('data-tooltip'));

                // Remove any existing tooltip
                if (tooltip) {
                    tooltip.remove();
                }

                // Create new tooltip
                tooltip = document.createElement('div');
                tooltip.className = 'tooltip';
                tooltip.textContent = newIcon.getAttribute('data-tooltip');

                // Position tooltip
                document.body.appendChild(tooltip);
                const iconRect = newIcon.getBoundingClientRect();
                const tooltipRect = tooltip.getBoundingClientRect();

                // Position above the icon, centered
                let left = iconRect.left + (iconRect.width / 2) - (tooltipRect.width / 2);
                let top = iconRect.top - tooltipRect.height - 8;

                // Adjust if tooltip goes off screen
                if (left < 10) left = 10;
                if (left + tooltipRect.width > window.innerWidth - 10) {
                    left = window.innerWidth - tooltipRect.width - 10;
                }
                if (top < 10) {
                    top = iconRect.bottom + 8; // Show below instead
                }

                tooltip.style.left = left + 'px';
                tooltip.style.top = top + 'px';

                // Show tooltip with animation
                requestAnimationFrame(() => {
                    tooltip.style.opacity = '1';
                    tooltip.style.transform = 'translateY(0)';
                });

                console.log('[Tooltips] Tooltip positioned at:', left, top);
            });

            // Remove tooltip on mouseleave
            newIcon.addEventListener('mouseleave', () => {
                console.log('[Tooltips] Mouseleave on icon');
                if (tooltip) {
                    tooltip.style.opacity = '0';
                    tooltip.style.transform = 'translateY(-4px)';
                    setTimeout(() => {
                        if (tooltip && tooltip.parentNode) {
                            tooltip.remove();
                        }
                    }, 150); // Match CSS transition duration
                }
            });
        });
    }

    cleanupObjectURLs() {
        const imageElement = document.getElementById('originalImage');
        if (imageElement && imageElement.dataset.objectUrl) {
            URL.revokeObjectURL(imageElement.dataset.objectUrl);
            delete imageElement.dataset.objectUrl;
        }
    }

    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    showError(message, title = 'Error') {
        errorHandler.showUserError(message, { type: 'error' });
    }

    showSuccess(message, title = 'Success') {
        errorHandler.showUserError(message, { type: 'success', duration: 3000 });
    }
}

// Make initializeTooltips available globally for backward compatibility
window.initializeTooltips = function() {
    const uiModule = new UIModule();
    uiModule.initializeTooltips();
};

// Export for module use
export default UIModule;