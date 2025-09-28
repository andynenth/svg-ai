'use strict';

/**
 * Split View Module
 * Handles image comparison, zoom controls, and drag functionality
 */

import errorHandler from './errorHandler.js';

class SplitViewController {
    constructor() {
        this.splitContainer = document.getElementById('splitViewContainer');
        this.isActive = true; // Always active since it's the only view
        this.isDragging = false;
        this.imageSynchronizer = null;

        // Split divider drag state
        this.minPercentage = 20;
        this.maxPercentage = 80;

        // Image drag state
        this.imageDragState = {
            isDragging: false,
            startX: 0,
            startY: 0,
            initialTranslateX: 0,
            initialTranslateY: 0,
            currentElements: [], // Array to hold both synchronized images
            animationFrame: null // For smooth updates
        };

        this.init();
    }

    init() {
        if (!this.splitContainer) {
            console.log('Split view container not found');
            return;
        }

        this.setupDragHandlers();
        this.setupZoomControls();
        this.setupKeyboardHandlers();
        this.setupImageDragHandlers();

        // Initialize immediately since it's the only view
        this.loadSavedSplit();

        // Listen for conversion events
        document.addEventListener('conversionComplete', () => {
            this.updateConversion();
        });

        // Listen for upload events
        document.addEventListener('uploadComplete', () => {
            setTimeout(() => this.syncImages(), 500);
        });

        // Listen for zoom reset requests from converter controls
        document.addEventListener('resetZoomBeforeConversion', () => {
            console.log('[Zoom] Resetting zoom due to converter control change');
            this.resetZoom();
        });
    }

    syncImages() {
        console.log('[SYNC DEBUG] Starting syncImages()');

        // Hide both panels initially
        const leftPanel = document.querySelector('.split-panel.left-panel');
        const rightPanel = document.querySelector('.split-panel.right-panel');

        if (leftPanel) {
            leftPanel.style.opacity = '0';
            leftPanel.style.transition = 'none';
        }
        if (rightPanel) {
            rightPanel.style.opacity = '0';
            rightPanel.style.transition = 'none';
        }

        console.log('[SYNC DEBUG] Hid both panels');

        // Copy original image
        const originalImg = document.getElementById('originalImage');
        const splitOriginalImg = document.getElementById('splitOriginalImage');

        if (originalImg && originalImg.src && splitOriginalImg) {
            console.log('[SYNC DEBUG] Copying original image to split view');

            splitOriginalImg.onerror = () => this.showImageError('left');
            splitOriginalImg.onload = () => {
                console.log('[SYNC DEBUG] Split original image loaded');
                this.initializeImageSync();
                this.equalizeImageSizes();
            };

            splitOriginalImg.src = originalImg.src;
            splitOriginalImg.style.display = 'block';
        }

        // Copy SVG
        const originalSvg = document.getElementById('svgContainer');
        const splitSvg = document.getElementById('splitSvgContainer');

        if (originalSvg && splitSvg) {
            console.log('[SYNC DEBUG] Copying SVG to split view');

            // Get original SVG element first
            const originalSvgElement = originalSvg.querySelector('svg');
            if (!originalSvgElement) {
                console.log('[SYNC DEBUG] No SVG element found in original container');
                return;
            }

            // Clone the SVG element
            const clonedSvg = originalSvgElement.cloneNode(true);

            // Calculate initial size BEFORE inserting into DOM
            const leftWidth = splitOriginalImg ? (splitOriginalImg.naturalWidth || 512) : 512;
            const leftHeight = splitOriginalImg ? (splitOriginalImg.naturalHeight || 512) : 512;

            const viewBox = clonedSvg.getAttribute('viewBox');
            let rightWidth = leftWidth, rightHeight = leftHeight;

            if (viewBox) {
                const parts = viewBox.split(' ');
                rightWidth = parseFloat(parts[2]) || leftWidth;
                rightHeight = parseFloat(parts[3]) || leftHeight;
            }

            // Calculate container size for proper scaling
            const container = document.getElementById('splitRightViewer');
            const containerWidth = container ? container.clientWidth : 514;
            const containerHeight = container ? container.clientHeight : 406;

            // Calculate scale to fit
            const scale = Math.min(containerWidth / rightWidth, containerHeight / rightHeight);

            // Apply size to cloned SVG BEFORE insertion
            clonedSvg.style.width = `${rightWidth * scale}px`;
            clonedSvg.style.height = `${rightHeight * scale}px`;

            console.log('[SYNC DEBUG] Pre-sized SVG:', clonedSvg.style.width, 'x', clonedSvg.style.height);

            // Hide container initially
            splitSvg.style.opacity = '0';
            splitSvg.style.transition = 'none';

            // Clear and insert
            splitSvg.innerHTML = '';
            splitSvg.appendChild(clonedSvg);

            // Now do proper sizing and show both panels together
            setTimeout(() => {
                this.equalizeImageSizes();

                // Show both panels together after everything is ready
                requestAnimationFrame(() => {
                    console.log('[SYNC DEBUG] Showing both panels with fade-in');

                    const leftPanel = document.querySelector('.split-panel.left-panel');
                    const rightPanel = document.querySelector('.split-panel.right-panel');

                    // Enable transitions and show panels
                    if (leftPanel) {
                        leftPanel.style.transition = 'opacity 0.3s ease';
                        leftPanel.style.opacity = '1';
                    }
                    if (rightPanel) {
                        rightPanel.style.transition = 'opacity 0.3s ease';
                        rightPanel.style.opacity = '1';
                    }

                    // Also ensure the SVG container is visible
                    splitSvg.style.transition = 'opacity 0.3s ease';
                    splitSvg.style.opacity = '1';
                });
            }, 50);
        }
    }

    equalizeImageSizes() {
        console.log('[SIZE DEBUG] Starting equalizeImageSizes()');

        const leftImg = document.getElementById('splitOriginalImage');
        const rightSvg = document.querySelector('#splitSvgContainer svg');

        if (!leftImg || !rightSvg) {
            console.log('[SIZE DEBUG] Missing elements:', {
                leftImg: !!leftImg,
                rightSvg: !!rightSvg
            });
            return;
        }

        // Log current SVG state before sizing
        console.log('[SIZE DEBUG] SVG state before sizing:', {
            width: rightSvg.getAttribute('width'),
            height: rightSvg.getAttribute('height'),
            styleWidth: rightSvg.style.width,
            styleHeight: rightSvg.style.height,
            computedWidth: getComputedStyle(rightSvg).width,
            computedHeight: getComputedStyle(rightSvg).height,
            offsetWidth: rightSvg.offsetWidth,
            offsetHeight: rightSvg.offsetHeight
        });

        // Get natural dimensions
        const leftWidth = leftImg.naturalWidth;
        const leftHeight = leftImg.naturalHeight;
        console.log('[SIZE DEBUG] Left image natural dimensions:', leftWidth, 'x', leftHeight);

        // Get SVG viewBox dimensions
        const viewBox = rightSvg.getAttribute('viewBox');
        let rightWidth = 100, rightHeight = 100;

        if (viewBox) {
            const parts = viewBox.split(' ');
            rightWidth = parseFloat(parts[2]) || 100;
            rightHeight = parseFloat(parts[3]) || 100;
        }

        console.log('[SIZE DEBUG] SVG viewBox dimensions:', rightWidth, 'x', rightHeight);

        // Get container dimensions
        const leftContainer = document.getElementById('splitLeftViewer');
        const rightContainer = document.getElementById('splitRightViewer');

        if (!leftContainer || !rightContainer) {
            console.log('[SIZE DEBUG] Missing containers');
            return;
        }

        const containerWidth = Math.min(leftContainer.clientWidth, rightContainer.clientWidth) - 40;
        const containerHeight = Math.min(leftContainer.clientHeight, rightContainer.clientHeight) - 40;

        console.log('[SIZE DEBUG] Container dimensions:', containerWidth, 'x', containerHeight);

        // Calculate scale to fit both images at same size
        const leftScale = Math.min(containerWidth / leftWidth, containerHeight / leftHeight);
        const rightScale = Math.min(containerWidth / rightWidth, containerHeight / rightHeight);

        // Use the smaller scale for both to ensure equal visual size
        const scale = Math.min(leftScale, rightScale, 1); // Cap at 1 to prevent upscaling
        console.log('[SIZE DEBUG] Calculated scale:', scale);

        // Apply consistent sizing
        const finalWidth = leftWidth * scale;
        const finalHeight = leftHeight * scale;

        leftImg.style.width = `${finalWidth}px`;
        leftImg.style.height = `${finalHeight}px`;
        leftImg.style.objectFit = 'contain';

        rightSvg.style.width = `${finalWidth}px`;
        rightSvg.style.height = `${finalHeight}px`;

        console.log('[SIZE DEBUG] Applied dimensions:', finalWidth, 'x', finalHeight);

        // Log final SVG state after sizing
        console.log('[SIZE DEBUG] SVG state after sizing:', {
            styleWidth: rightSvg.style.width,
            styleHeight: rightSvg.style.height,
            offsetWidth: rightSvg.offsetWidth,
            offsetHeight: rightSvg.offsetHeight
        });
    }

    // Drag functionality
    setupDragHandlers() {
        const divider = document.getElementById('splitDivider');
        if (!divider) return;

        // Mouse events
        divider.addEventListener('mousedown', (e) => this.startDrag(e));
        document.addEventListener('mousemove', (e) => this.onDrag(e));
        document.addEventListener('mouseup', () => this.endDrag());

        // Touch events
        divider.addEventListener('touchstart', (e) => this.startDrag(e.touches[0]));
        document.addEventListener('touchmove', (e) => this.onDrag(e.touches[0]));
        document.addEventListener('touchend', () => this.endDrag());
    }

    startDrag(e) {
        if (!this.isActive) return;

        this.isDragging = true;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        document.getElementById('splitDivider').classList.add('dragging');

        e.preventDefault();
    }

    onDrag(e) {
        if (!this.isDragging || !this.isActive) return;

        const containerRect = this.splitContainer.getBoundingClientRect();
        const mouseX = e.clientX - containerRect.left;
        const containerWidth = containerRect.width;

        let percentage = (mouseX / containerWidth) * 100;
        percentage = Math.max(this.minPercentage, Math.min(this.maxPercentage, percentage));

        this.updateSplit(percentage);
        e.preventDefault();
    }

    endDrag() {
        if (!this.isDragging) return;

        this.isDragging = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        document.getElementById('splitDivider').classList.remove('dragging');

        // Save preference
        localStorage.setItem('splitViewColumns', this.splitContainer.style.gridTemplateColumns);
    }

    updateSplit(leftPercentage) {
        const rightPercentage = 100 - leftPercentage;
        this.splitContainer.style.gridTemplateColumns = `${leftPercentage}% 6px ${rightPercentage}%`;
    }

    loadSavedSplit() {
        const saved = localStorage.getItem('splitViewColumns');
        if (saved) {
            this.splitContainer.style.gridTemplateColumns = saved;
        }
    }

    // Zoom functionality
    setupZoomControls() {
        const zoomButtons = this.splitContainer.querySelectorAll('.zoom-btn');
        console.log(`[Zoom] Found ${zoomButtons.length} zoom buttons`);

        zoomButtons.forEach((btn, index) => {
            console.log(`[Zoom] Setting up button ${index}: ${btn.getAttribute('data-action')}`);
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const action = e.target.getAttribute('data-action');
                console.log(`[Zoom] Button clicked: ${action}`);
                this.handleZoom(action);
            });
        });

        // Mouse wheel zoom
        this.splitContainer.addEventListener('wheel', (e) => {
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                this.zoom(delta);
            }
        });
    }

    handleZoom(action) {
        console.log(`[Zoom] Handling action: ${action}`);
        switch (action) {
            case 'zoom-in':
                this.zoom(1.25);
                break;
            case 'zoom-out':
                this.zoom(0.8);
                break;
            case 'zoom-reset':
                this.resetZoom();
                break;
        }
    }

    zoom(factor) {
        // Target specific elements in split view
        const leftImg = document.getElementById('splitOriginalImage');
        const rightSvg = document.querySelector('#splitSvgContainer svg');
        const elements = [leftImg, rightSvg].filter(el => el);

        console.log(`[Zoom] Found ${elements.length} elements to zoom`);

        elements.forEach(element => {
            const currentTransform = element.style.transform || 'scale(1)';
            const currentScale = parseFloat(currentTransform.match(/scale\(([\d.]+)\)/)?.[1] || '1');
            const newScale = Math.max(0.1, Math.min(5, currentScale * factor));

            // Preserve existing translate values
            const translateMatch = currentTransform.match(/translate\(([^)]+)\)/);
            const translatePart = translateMatch ? ` translate(${translateMatch[1]})` : '';

            element.style.transform = `scale(${newScale})${translatePart}`;
            element.style.transformOrigin = 'center center';

            // Update cached scale for fast drag operations
            element._cachedScale = newScale.toString();
        });

        // Update draggable state for both viewers after zoom
        this.updateDraggableState(document.getElementById('splitLeftViewer'));
        this.updateDraggableState(document.getElementById('splitRightViewer'));

        this.updateZoomDisplay();
    }

    resetZoom() {
        const leftImg = document.getElementById('splitOriginalImage');
        const rightSvg = document.querySelector('#splitSvgContainer svg');
        const elements = [leftImg, rightSvg].filter(el => el);

        elements.forEach(element => {
            // Reset both scale and any translate transforms
            element.style.transform = 'scale(1)';
            element.style.transformOrigin = 'center center';

            // Reset cached scale for fast drag operations
            element._cachedScale = '1';
        });

        // Update draggable state and remove draggable classes when zoom is reset
        this.updateDraggableState(document.getElementById('splitLeftViewer'));
        this.updateDraggableState(document.getElementById('splitRightViewer'));

        this.updateZoomDisplay();
    }

    updateZoomDisplay() {
        const leftImg = document.getElementById('splitOriginalImage');
        if (!leftImg) return;

        const currentTransform = leftImg.style.transform || 'scale(1)';
        const currentScale = parseFloat(currentTransform.match(/scale\(([\d.]+)\)/)?.[1] || '1');
        const percentage = Math.round(currentScale * 100);

        // Update all zoom level displays in split view
        this.splitContainer.querySelectorAll('.zoom-level').forEach(display => {
            display.textContent = percentage + '%';
        });

        console.log(`[Zoom] Updated zoom display to ${percentage}%`);
    }

    // Image drag functionality
    setupImageDragHandlers() {
        console.log('[ImageDrag] Setting up image drag handlers');

        const leftViewer = document.getElementById('splitLeftViewer');
        const rightViewer = document.getElementById('splitRightViewer');

        [leftViewer, rightViewer].forEach(viewer => {
            if (viewer) {
                this.addImageDragListeners(viewer);
            }
        });
    }

    addImageDragListeners(viewer) {
        // Mouse events
        viewer.addEventListener('mousedown', (e) => this.handleImageDragStart(e, viewer));
        viewer.addEventListener('mousemove', (e) => this.handleImageDragMove(e));
        viewer.addEventListener('mouseup', (e) => this.handleImageDragEnd(e));
        viewer.addEventListener('mouseleave', (e) => this.handleImageDragEnd(e));

        // Touch events for mobile
        viewer.addEventListener('touchstart', (e) => this.handleImageDragStart(e, viewer), {passive: false});
        viewer.addEventListener('touchmove', (e) => this.handleImageDragMove(e), {passive: false});
        viewer.addEventListener('touchend', (e) => this.handleImageDragEnd(e));
        viewer.addEventListener('touchcancel', (e) => this.handleImageDragEnd(e));

        // Update draggable state when zoom changes
        viewer.addEventListener('transitionend', () => this.updateDraggableState(viewer));
    }

    handleImageDragStart(e, viewer) {
        // Only allow drag if image is larger than container (zoomed)
        if (!this.isImageDraggable(viewer)) {
            return;
        }

        // Prevent text selection and default behavior
        e.preventDefault();

        const clientX = e.type.startsWith('touch') ? e.touches[0].clientX : e.clientX;
        const clientY = e.type.startsWith('touch') ? e.touches[0].clientY : e.clientY;

        this.imageDragState.isDragging = true;
        this.imageDragState.startX = clientX;
        this.imageDragState.startY = clientY;

        // Get both image elements for synchronized dragging
        const leftImg = document.getElementById('splitOriginalImage');
        const rightSvg = document.querySelector('#splitSvgContainer svg');

        // Filter elements and validate they have proper container structure
        this.imageDragState.currentElements = [leftImg, rightSvg].filter(el => {
            if (!el) return false;
            if (!el.closest('.image-viewer')) {
                console.warn('[ImageDrag] Element missing .image-viewer container:', el);
                return false;
            }
            return true;
        });

        // Abort drag if no valid elements found
        if (this.imageDragState.currentElements.length === 0) {
            console.warn('[ImageDrag] No valid draggable elements found, aborting drag');
            this.imageDragState.isDragging = false;
            return;
        }

        // Get initial translate values from the primary element
        if (this.imageDragState.currentElements.length > 0) {
            const primaryElement = this.imageDragState.currentElements[0];
            const transform = primaryElement.style.transform || '';
            const translateMatch = transform.match(/translate\(([^)]+)\)/);

            if (translateMatch) {
                const [x, y] = translateMatch[1].split(',').map(v => parseFloat(v.trim()) || 0);
                this.imageDragState.initialTranslateX = x;
                this.imageDragState.initialTranslateY = y;
            } else {
                this.imageDragState.initialTranslateX = 0;
                this.imageDragState.initialTranslateY = 0;
            }
        }

        // Add dragging class to both viewers for visual feedback
        document.getElementById('splitLeftViewer').classList.add('dragging');
        document.getElementById('splitRightViewer').classList.add('dragging');

        viewer.style.cursor = 'move';
        console.log('[ImageDrag] Started dragging with elements:', this.imageDragState.currentElements.length);
    }

    handleImageDragMove(e) {
        if (!this.imageDragState.isDragging) return;

        e.preventDefault();

        const clientX = e.type.startsWith('touch') ? e.touches[0].clientX : e.clientX;
        const clientY = e.type.startsWith('touch') ? e.touches[0].clientY : e.clientY;

        const deltaX = clientX - this.imageDragState.startX;
        const deltaY = clientY - this.imageDragState.startY;

        const newTranslateX = this.imageDragState.initialTranslateX + deltaX;
        const newTranslateY = this.imageDragState.initialTranslateY + deltaY;

        // Cancel any pending animation frame
        if (this.imageDragState.animationFrame) {
            cancelAnimationFrame(this.imageDragState.animationFrame);
        }

        // Use requestAnimationFrame for smooth updates
        this.imageDragState.animationFrame = requestAnimationFrame(() => {
            // Validate we still have elements to drag
            if (!this.imageDragState.currentElements || this.imageDragState.currentElements.length === 0) {
                console.warn('[ImageDrag] No valid elements during drag move, ending drag');
                this.imageDragState.isDragging = false;
                return;
            }

            // Apply bounds checking using the primary element
            const boundedTranslate = this.applyDragBounds(newTranslateX, newTranslateY, this.imageDragState.currentElements[0]);

            // Apply transform to all synchronized elements instantly
            this.imageDragState.currentElements.forEach(element => {
                if (element && element.style) {
                    this.applyImageTransform(element, boundedTranslate.x, boundedTranslate.y);
                }
            });

            this.imageDragState.animationFrame = null;
        });
    }

    handleImageDragEnd(e) {
        if (!this.imageDragState.isDragging) return;

        this.imageDragState.isDragging = false;

        // Cancel any pending animation frame
        if (this.imageDragState.animationFrame) {
            cancelAnimationFrame(this.imageDragState.animationFrame);
            this.imageDragState.animationFrame = null;
        }

        // Remove dragging classes from both viewers
        document.getElementById('splitLeftViewer').classList.remove('dragging');
        document.getElementById('splitRightViewer').classList.remove('dragging');

        // Reset cursor for both viewers
        document.getElementById('splitLeftViewer').style.cursor = '';
        document.getElementById('splitRightViewer').style.cursor = '';

        console.log('[ImageDrag] Ended dragging');
    }

    applyDragBounds(translateX, translateY, element) {
        if (!element) return { x: translateX, y: translateY };

        // Get element dimensions and container dimensions
        const elementRect = element.getBoundingClientRect();
        const container = element.closest('.image-viewer');

        // If no container found, return original translation without bounds
        if (!container) {
            console.warn('[ImageDrag] No .image-viewer container found for element, skipping bounds check');
            return { x: translateX, y: translateY };
        }

        const containerRect = container.getBoundingClientRect();

        // Calculate max allowed translation to keep image within reasonable bounds
        const maxTranslateX = Math.max(0, (elementRect.width - containerRect.width) / 2);
        const maxTranslateY = Math.max(0, (elementRect.height - containerRect.height) / 2);

        // Bound the translation
        const boundedX = Math.max(-maxTranslateX, Math.min(maxTranslateX, translateX));
        const boundedY = Math.max(-maxTranslateY, Math.min(maxTranslateY, translateY));

        return { x: boundedX, y: boundedY };
    }

    applyImageTransform(element, translateX, translateY) {
        if (!element) return;

        // Get existing scale transform
        const currentTransform = element.style.transform || '';
        const scaleMatch = currentTransform.match(/scale\(([\d.]+)\)/);
        const scale = scaleMatch ? scaleMatch[1] : '1';

        // Combine scale and translate
        element.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
    }

    isImageDraggable(viewer) {
        // Simple check: if image is scaled beyond 1, it's draggable
        const image = viewer.querySelector('img, svg');
        if (!image) return false;

        const transform = image.style.transform || '';
        const scaleMatch = transform.match(/scale\(([\d.]+)\)/);
        const scale = scaleMatch ? parseFloat(scaleMatch[1]) : 1;

        return scale > 1.1; // Allow drag if zoomed in
    }

    updateDraggableState(viewer) {
        if (!viewer) return;

        // Apply transform directly without string parsing
        const isDraggable = this.isImageDraggable(viewer);
        viewer.classList.toggle('draggable', isDraggable);

        console.log(`[ImageDrag] Updated draggable state for viewer:`, isDraggable);
    }

    initializeImageSync() {
        if (!this.imageSynchronizer) {
            this.imageSynchronizer = new ImageSynchronizer(
                document.getElementById('splitLeftViewer'),
                document.getElementById('splitRightViewer')
            );
        }

        const leftImg = document.getElementById('splitOriginalImage');
        const rightContainer = document.getElementById('splitSvgContainer');

        this.imageSynchronizer.synchronizeImages(leftImg, rightContainer);
        this.updateZoomDisplay();

        // Update draggable state for both viewers when images are synchronized
        this.updateDraggableState(document.getElementById('splitLeftViewer'));
        this.updateDraggableState(document.getElementById('splitRightViewer'));
    }

    // Keyboard shortcuts
    setupKeyboardHandlers() {
        document.addEventListener('keydown', (e) => {
            if (!this.isActive) return;
            if (e.target.matches('input, textarea, select')) return;

            const step = 5;

            if (e.code === 'ArrowLeft' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.adjustSplit(-step);
            }

            if (e.code === 'ArrowRight' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.adjustSplit(step);
            }
        });
    }

    adjustSplit(delta) {
        const current = this.getCurrentSplitPercentage();
        const newPercentage = Math.max(this.minPercentage,
                                      Math.min(this.maxPercentage, current + delta));
        this.updateSplit(newPercentage);
    }

    getCurrentSplitPercentage() {
        const columns = this.splitContainer.style.gridTemplateColumns;
        const match = columns.match(/^([\d.]+)%/);
        return match ? parseFloat(match[1]) : 50;
    }

    showImageError(side) {
        const errorMessage = '<div class="image-error">Image not available</div>';
        if (side === 'left') {
            document.getElementById('splitLeftViewer').innerHTML = errorMessage;
        }
    }

    showError(message) {
        errorHandler.handleSplitViewError(new Error(message));
    }

    // Public method for integration
    updateConversion() {
        console.log('[UPDATE DEBUG] Starting updateConversion()');
        // Always sync images since split view is the only interface
        this.syncImages();
    }
}

// Image Synchronizer Class
class ImageSynchronizer {
    constructor(leftViewer, rightViewer) {
        this.leftViewer = leftViewer;
        this.rightViewer = rightViewer;
    }

    synchronizeImages(leftImg, rightContainer) {
        if (!leftImg || !rightContainer) return;

        // Simple approach: let CSS handle the scaling consistently
        leftImg.style.maxWidth = '100%';
        leftImg.style.maxHeight = '100%';
        leftImg.style.objectFit = 'contain';

        const svg = rightContainer.querySelector('svg');
        if (svg) {
            svg.style.maxWidth = '100%';
            svg.style.maxHeight = '100%';
        }
    }
}

// Export for module use
export { SplitViewController, ImageSynchronizer };