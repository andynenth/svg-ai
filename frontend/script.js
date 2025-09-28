'use strict';

// Global variables
let currentFileId = null;
let currentSvgContent = null;
const API_BASE = '';

// Get DOM Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const mainContent = document.getElementById('mainContent');
const convertBtn = document.getElementById('convertBtn');
const converterSelect = document.getElementById('converter');
const loadingDiv = document.getElementById('loading');
const metricsDiv = document.getElementById('metrics');
const downloadBtn = document.getElementById('downloadBtn');

// Parameter containers
const potraceParams = document.getElementById('potraceParams');
const vtracerParams = document.getElementById('vtracerParams');
const alphaParams = document.getElementById('alphaParams');

// Setup Click to Upload
dropzone.addEventListener('click', () => {
    fileInput.click();
});

// Setup Drag and Drop
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    handleFile(file);
});

// Handle File Input Change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFile(file);
});

// Parameter Management Functions
function showConverterParams(converter) {
    // Hide all parameter groups
    potraceParams.classList.add('hidden');
    vtracerParams.classList.add('hidden');
    alphaParams.classList.add('hidden');

    // Show the selected converter's parameters
    switch(converter) {
        case 'smart':
            // Smart Potrace uses same parameters as regular Potrace
            potraceParams.classList.remove('hidden');
            break;
        case 'potrace':
            potraceParams.classList.remove('hidden');
            break;
        case 'vtracer':
            vtracerParams.classList.remove('hidden');
            break;
        case 'alpha':
            alphaParams.classList.remove('hidden');
            break;
    }

    // Re-initialize tooltips after showing parameters
    setTimeout(initializeTooltips, 100);
}

function collectPotraceParams() {
    const params = {
        threshold: parseInt(document.getElementById('potraceThreshold').value),
        turnpolicy: document.getElementById('potraceTurnpolicy').value,
        turdsize: parseInt(document.getElementById('potraceTurdsize').value),
        alphamax: parseFloat(document.getElementById('potraceAlphamax').value) / 100, // Convert 0-134 to 0-1.34
        opttolerance: parseFloat(document.getElementById('potraceOpttolerance').value) / 100 // Convert 1-100 to 0.01-1.0
    };
    console.log('[Frontend] Collected Potrace params:', params);
    return params;
}

function collectVTracerParams() {
    return {
        threshold: 128, // VTracer uses its own threshold mapping
        colormode: document.querySelector('input[name="vtracerColormode"]:checked').value,
        color_precision: parseInt(document.getElementById('vtracerColorPrecision').value),
        layer_difference: parseInt(document.getElementById('vtracerLayerDifference').value),
        path_precision: parseInt(document.getElementById('vtracerPathPrecision').value),
        corner_threshold: parseInt(document.getElementById('vtracerCornerThreshold').value),
        length_threshold: parseFloat(document.getElementById('vtracerLengthThreshold').value),
        max_iterations: parseInt(document.getElementById('vtracerMaxIterations').value),
        splice_threshold: parseInt(document.getElementById('vtracerSpliceThreshold').value)
    };
}

function collectAlphaParams() {
    return {
        threshold: parseInt(document.getElementById('alphaThreshold').value),
        use_potrace: document.getElementById('alphaUsePotrace').checked,
        preserve_antialiasing: document.getElementById('alphaPreserveAntialiasing').checked
    };
}

// Update converter change handler
converterSelect.addEventListener('change', (e) => {
    showConverterParams(e.target.value);
});

// Set initial parameter display
showConverterParams(converterSelect.value);

// Real-time value displays for sliders
document.getElementById('potraceThreshold').addEventListener('input', (e) => {
    document.getElementById('potraceThresholdValue').textContent = e.target.value;
});

document.getElementById('potraceAlphamax').addEventListener('input', (e) => {
    const value = (parseFloat(e.target.value) / 100).toFixed(2); // Convert 0-134 to 0.00-1.34
    document.getElementById('potraceAlphamaxValue').textContent = value;
});

document.getElementById('potraceOpttolerance').addEventListener('input', (e) => {
    const value = (parseFloat(e.target.value) / 100).toFixed(2); // Convert 0-100 to 0.00-1.00
    document.getElementById('potraceOpttoleranceValue').textContent = value;
});

// VTracer slider value displays
document.getElementById('vtracerColorPrecision').addEventListener('input', (e) => {
    document.getElementById('vtracerColorPrecisionValue').textContent = e.target.value;
});

document.getElementById('vtracerLayerDifference').addEventListener('input', (e) => {
    document.getElementById('vtracerLayerDifferenceValue').textContent = e.target.value;
});

document.getElementById('vtracerPathPrecision').addEventListener('input', (e) => {
    document.getElementById('vtracerPathPrecisionValue').textContent = e.target.value;
});

document.getElementById('vtracerCornerThreshold').addEventListener('input', (e) => {
    document.getElementById('vtracerCornerThresholdValue').textContent = e.target.value;
});

document.getElementById('vtracerMaxIterations').addEventListener('input', (e) => {
    document.getElementById('vtracerMaxIterationsValue').textContent = e.target.value;
});

document.getElementById('vtracerSpliceThreshold').addEventListener('input', (e) => {
    document.getElementById('vtracerSpliceThresholdValue').textContent = e.target.value;
});

// Alpha slider value display
document.getElementById('alphaThreshold').addEventListener('input', (e) => {
    document.getElementById('alphaThresholdValue').textContent = e.target.value;
});

// Setup Convert Button
convertBtn.addEventListener('click', handleConvert);

// Setup Download Button
downloadBtn.addEventListener('click', handleDownload);

// Parameter Preset Functions
function applyPotracePreset(preset) {
    switch(preset) {
        case 'quality':
            // Quality preset: smooth, accurate settings
            document.getElementById('potraceThreshold').value = 128;
            document.getElementById('potraceTurnpolicy').value = 'white'; // Smooth corners
            document.getElementById('potraceTurdsize').value = 5; // Remove more noise
            document.getElementById('potraceAlphamax').value = 120; // More smoothness (1.2)
            document.getElementById('potraceOpttolerance').value = 10; // Higher accuracy (0.1)
            break;
        case 'fast':
            // Fast preset: basic, quick settings
            document.getElementById('potraceThreshold').value = 128;
            document.getElementById('potraceTurnpolicy').value = 'black'; // Sharp corners
            document.getElementById('potraceTurdsize').value = 1; // Minimal noise removal
            document.getElementById('potraceAlphamax').value = 100; // Default smoothness (1.0)
            document.getElementById('potraceOpttolerance').value = 20; // Default accuracy (0.2)
            break;
        case 'ultra-precise':
            // Ultra-precise preset: maximum accuracy, larger files
            document.getElementById('potraceThreshold').value = 128;
            document.getElementById('potraceTurnpolicy').value = 'white'; // Smooth corners
            document.getElementById('potraceTurdsize').value = 2; // Standard noise removal
            document.getElementById('potraceAlphamax').value = 120; // More smoothness (1.2)
            document.getElementById('potraceOpttolerance').value = 1; // Ultra precise (0.01)
            break;
        case 'ultra-fast':
            // Ultra-fast preset: aggressive optimization, much smaller files
            document.getElementById('potraceThreshold').value = 128;
            document.getElementById('potraceTurnpolicy').value = 'black'; // Sharp corners
            document.getElementById('potraceTurdsize').value = 1; // Minimal noise removal
            document.getElementById('potraceAlphamax').value = 80; // Less smoothness (0.8)
            document.getElementById('potraceOpttolerance').value = 80; // Aggressive optimization (0.8)
            break;
    }
    updatePotraceDisplayValues();
}

function applyVTracerPreset(preset) {
    switch(preset) {
        case 'quality':
            // Quality preset: high precision, more colors
            document.querySelector('input[name="vtracerColormode"][value="color"]').checked = true;
            document.getElementById('vtracerColorPrecision').value = 8; // More colors
            document.getElementById('vtracerLayerDifference').value = 8; // Gradients
            document.getElementById('vtracerPathPrecision').value = 8; // High precision
            document.getElementById('vtracerCornerThreshold').value = 30; // Gentle corners
            document.getElementById('vtracerLengthThreshold').value = 2.0; // Keep more paths
            document.getElementById('vtracerMaxIterations').value = 20; // More iterations
            document.getElementById('vtracerSpliceThreshold').value = 60; // Connect more paths
            break;
        case 'fast':
            // Fast preset: basic settings for speed
            document.querySelector('input[name="vtracerColormode"][value="color"]').checked = true;
            document.getElementById('vtracerColorPrecision').value = 4; // Fewer colors
            document.getElementById('vtracerLayerDifference').value = 16; // Less gradients
            document.getElementById('vtracerPathPrecision').value = 3; // Lower precision
            document.getElementById('vtracerCornerThreshold').value = 60; // More corners
            document.getElementById('vtracerLengthThreshold').value = 5.0; // Remove short paths
            document.getElementById('vtracerMaxIterations').value = 5; // Fewer iterations
            document.getElementById('vtracerSpliceThreshold').value = 30; // Less path joining
            break;
    }
    updateVTracerDisplayValues();
}

function applyAlphaPreset(preset) {
    switch(preset) {
        case 'quality':
            // Quality preset: preserve transparency, clean edges
            document.getElementById('alphaThreshold').value = 64; // Lower threshold for semi-transparency
            document.getElementById('alphaUsePotrace').checked = true; // Clean edges
            document.getElementById('alphaPreserveAntialiasing').checked = true; // Smooth edges
            break;
        case 'fast':
            // Fast preset: basic transparency handling
            document.getElementById('alphaThreshold').value = 128; // Standard threshold
            document.getElementById('alphaUsePotrace').checked = true; // Clean edges
            document.getElementById('alphaPreserveAntialiasing').checked = false; // No antialiasing
            break;
    }
    updateAlphaDisplayValues();
}

function resetPotraceDefaults() {
    document.getElementById('potraceThreshold').value = 128;
    document.getElementById('potraceTurnpolicy').value = 'black';
    document.getElementById('potraceTurdsize').value = 2;
    document.getElementById('potraceAlphamax').value = 100; // 1.0
    document.getElementById('potraceOpttolerance').value = 20; // 0.2
    updatePotraceDisplayValues();
}

function resetVTracerDefaults() {
    document.querySelector('input[name="vtracerColormode"][value="color"]').checked = true;
    document.getElementById('vtracerColorPrecision').value = 6;
    document.getElementById('vtracerLayerDifference').value = 16;
    document.getElementById('vtracerPathPrecision').value = 5;
    document.getElementById('vtracerCornerThreshold').value = 60;
    document.getElementById('vtracerLengthThreshold').value = 5.0;
    document.getElementById('vtracerMaxIterations').value = 10;
    document.getElementById('vtracerSpliceThreshold').value = 45;
    updateVTracerDisplayValues();
}

function resetAlphaDefaults() {
    document.getElementById('alphaThreshold').value = 128;
    document.getElementById('alphaUsePotrace').checked = true;
    document.getElementById('alphaPreserveAntialiasing').checked = false;
    updateAlphaDisplayValues();
}

// Helper functions to update display values
function updatePotraceDisplayValues() {
    document.getElementById('potraceThresholdValue').textContent = document.getElementById('potraceThreshold').value;
    document.getElementById('potraceAlphamaxValue').textContent = (parseFloat(document.getElementById('potraceAlphamax').value) / 100).toFixed(2);
    document.getElementById('potraceOpttoleranceValue').textContent = (parseFloat(document.getElementById('potraceOpttolerance').value) / 100).toFixed(2);
}

function updateVTracerDisplayValues() {
    document.getElementById('vtracerColorPrecisionValue').textContent = document.getElementById('vtracerColorPrecision').value;
    document.getElementById('vtracerLayerDifferenceValue').textContent = document.getElementById('vtracerLayerDifference').value;
    document.getElementById('vtracerPathPrecisionValue').textContent = document.getElementById('vtracerPathPrecision').value;
    document.getElementById('vtracerCornerThresholdValue').textContent = document.getElementById('vtracerCornerThreshold').value;
    document.getElementById('vtracerMaxIterationsValue').textContent = document.getElementById('vtracerMaxIterations').value;
    document.getElementById('vtracerSpliceThresholdValue').textContent = document.getElementById('vtracerSpliceThreshold').value;
}

function updateAlphaDisplayValues() {
    document.getElementById('alphaThresholdValue').textContent = document.getElementById('alphaThreshold').value;
}

// File Handler Function
async function handleFile(file) {
    if (!file) return;

    if (!file.type.match('image/(png|jpeg|jpg)')) {
        alert('Please upload a PNG or JPEG image');
        return;
    }

    // Show progressive loading
    showImagePlaceholder(file);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            hideImagePlaceholder();
            alert(data.error);
            return;
        }

        currentFileId = data.file_id;

        // Display Original Image with memory-efficient object URL
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
            adjustContainerSizing(imageElement);
            hideImagePlaceholder();

            // Remove inline opacity and add loaded class for CSS transition
            imageElement.style.opacity = '';
            imageElement.classList.add('loaded');

            // Initialize zoom functionality for original image
            initializeOriginalImageZoom();

            console.log('[Progressive] Image should now be visible with opacity:', getComputedStyle(imageElement).opacity);
        };

        // Handle image load errors
        imageElement.onerror = () => {
            console.error('[Progressive] Image failed to load');
            hideImagePlaceholder();
            const container = document.getElementById('originalImageContainer');
            container.innerHTML = '<p class="error">Failed to load image</p>';
        };

        // Fallback timeout to hide placeholder if something goes wrong
        setTimeout(() => {
            if (!imageElement.classList.contains('loaded')) {
                console.warn('[Progressive] Image taking too long, hiding placeholder');
                hideImagePlaceholder();
            }
        }, 5000); // 5 second timeout

        mainContent.classList.remove('hidden');

        // Auto-convert to SVG after successful upload
        console.log('[Auto-convert] File uploaded, starting automatic conversion');

        // First sync the original image to split view
        if (splitViewController) {
            splitViewController.syncImages();
        }

        // Then auto-convert
        setTimeout(() => {
            handleConvert();
        }, 500); // Small delay to ensure UI is ready

    } catch (error) {
        hideImagePlaceholder();
        alert('Upload failed: ' + error.message);
    }
}

// Convert Handler
async function handleConvert() {
    if (!currentFileId) {
        alert('Please upload an image first');
        return;
    }

    // Show loading
    loadingDiv.classList.remove('hidden');
    metricsDiv.classList.add('hidden');

    // Disable button and change text
    convertBtn.disabled = true;
    convertBtn.textContent = 'Converting...';

    // Collect parameters based on selected converter
    const converter = converterSelect.value;
    let requestData = {
        file_id: currentFileId,
        converter: converter
    };

    // Add converter-specific parameters
    console.log('[Frontend] Selected converter:', converter);
    switch(converter) {
        case 'smart':
            // Smart Potrace uses same parameters as regular Potrace
            const smartParams = collectPotraceParams();
            console.log('[Frontend] Smart Potrace params:', smartParams);
            Object.assign(requestData, smartParams);
            break;
        case 'potrace':
            Object.assign(requestData, collectPotraceParams());
            break;
        case 'vtracer':
            Object.assign(requestData, collectVTracerParams());
            break;
        case 'alpha':
            Object.assign(requestData, collectAlphaParams());
            break;
    }

    console.log('[Frontend] Sending conversion request:', requestData);

    try {
        const response = await fetch(`${API_BASE}/api/convert`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const result = await response.json();

        if (!result.success) {
            alert('Conversion failed: ' + result.error);
            return;
        }

        // Display Results with optimized SVG handling
        currentSvgContent = result.svg;

        // Debug SVG content
        console.log('[Frontend] SVG length:', result.svg.length);
        console.log('[Frontend] SVG preview:', result.svg.substring(0, 300));

        // Optimize and display SVG
        displayOptimizedSVG(result.svg);

        // Check what actually rendered
        const svgElement = document.querySelector('#svgContainer svg');
        if (svgElement) {
            console.log('[Frontend] SVG rendered with dimensions:', svgElement.getAttribute('width'), 'x', svgElement.getAttribute('height'));
            console.log('[Frontend] SVG viewBox:', svgElement.getAttribute('viewBox'));
        } else {
            console.log('[Frontend] ERROR: No SVG element found in container');
        }

        document.getElementById('ssimScore').textContent = (result.ssim * 100).toFixed(1) + '%';
        document.getElementById('fileSize').textContent = formatFileSize(result.size);
        document.getElementById('pathCount').textContent = result.path_count || '-';
        document.getElementById('avgPathLength').textContent = result.avg_path_length || '-';

        metricsDiv.classList.remove('hidden');

        // Update split view after conversion
        if (splitViewController) {
            console.log('[Auto-convert] Updating split view with conversion results');
            splitViewController.updateConversion();
        }

    } catch (error) {
        alert('Conversion failed: ' + error.message);
    } finally {
        loadingDiv.classList.add('hidden');
        convertBtn.disabled = false;
        convertBtn.textContent = 'Convert to SVG';
    }
}

// Download Handler
function handleDownload() {
    if (!currentSvgContent) return;

    const blob = new Blob([currentSvgContent], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'converted.svg';
    a.click();

    URL.revokeObjectURL(url);
}

// File Size Formatter
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Flowbite-style Tooltip System
function initializeTooltips() {
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

            // Position directly above the icon, centered
            const left = iconRect.left + (iconRect.width / 2) - (tooltipRect.width / 2);
            const top = iconRect.top - tooltipRect.height - 8 + window.scrollY;

            // Adjust if tooltip goes off-screen horizontally
            const adjustedLeft = Math.max(8, Math.min(left, window.innerWidth - tooltipRect.width - 8));

            tooltip.style.position = 'absolute';
            tooltip.style.left = adjustedLeft + 'px';
            tooltip.style.top = top + 'px';

            console.log(`[Tooltips] Positioned tooltip at (${adjustedLeft}, ${top})`);

            // Show tooltip with animation
            requestAnimationFrame(() => {
                tooltip.classList.add('show');
            });
        });

        // Remove tooltip on mouseleave
        newIcon.addEventListener('mouseleave', () => {
            console.log('[Tooltips] Mouseleave on icon');
            if (tooltip) {
                tooltip.classList.remove('show');
                setTimeout(() => {
                    if (tooltip && tooltip.parentNode) {
                        tooltip.remove();
                    }
                }, 150); // Match CSS transition duration
                tooltip = null;
            }
        });
    });
}

// Initialize tooltips when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('[Tooltips] DOM loaded, initializing tooltips');
    initializeTooltips();
});

// Also initialize on window load as backup
window.addEventListener('load', () => {
    console.log('[Tooltips] Window loaded, re-initializing tooltips');
    setTimeout(initializeTooltips, 100);
});

// Memory Management: Clean up object URLs
function cleanupObjectURLs() {
    const imageElement = document.getElementById('originalImage');
    if (imageElement && imageElement.dataset.objectUrl) {
        URL.revokeObjectURL(imageElement.dataset.objectUrl);
        delete imageElement.dataset.objectUrl;
    }
}

// Clean up on page unload to prevent memory leaks
window.addEventListener('beforeunload', cleanupObjectURLs);

// Clean up on visibility change (tab switch)
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        cleanupObjectURLs();
    }
});

// Dynamic Container Sizing
function adjustContainerSizing(imageElement) {
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
    svgContainer.style.height = heightPx;

    console.log(`[Sizing] Image: ${naturalWidth}x${naturalHeight}, ratio: ${aspectRatio.toFixed(2)}, container: ${heightPx}`);
}

// Re-adjust sizing on window resize
window.addEventListener('resize', () => {
    const imageElement = document.getElementById('originalImage');
    if (imageElement && imageElement.complete) {
        adjustContainerSizing(imageElement);
    }
});

// Initialize zoom for original image
function initializeOriginalImageZoom() {
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
        wrapper.style.transform = `scale(${currentZoom})`;
        wrapper.style.transformOrigin = 'center center';

        console.log('[Image] Zoom level:', currentZoom);
    });

    // Mouse wheel zoom
    wrapper.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -zoomStep : zoomStep;
        currentZoom = Math.max(minZoom, Math.min(maxZoom, currentZoom + delta));
        wrapper.style.transform = `scale(${currentZoom})`;
        wrapper.style.transformOrigin = 'center center';
    });
}

// Progressive Loading Management
function showImagePlaceholder(file) {
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
        imageInfo.innerHTML = `
            <strong>${file.name}</strong><br>
            Size: ${fileSizeMB} MB<br>
            Type: ${file.type}<br>
            <em>Loading preview...</em>
        `;
    }

    console.log('[Progressive] Showing placeholder for:', file.name);
}

function hideImagePlaceholder() {
    const placeholder = document.getElementById('imagePlaceholder');

    if (placeholder) {
        // Immediate hide with CSS class
        placeholder.classList.add('hidden');
        console.log('[Progressive] Placeholder hidden');
    } else {
        // Grid view removed, no placeholder to hide
        console.log('[Progressive] No placeholder to hide (split view only)');
    }
}

// Optimized SVG Display
function displayOptimizedSVG(svgContent) {
    const container = document.getElementById('svgContainer');

    // Clear container
    container.innerHTML = '';

    // Create wrapper for zoom functionality
    const svgWrapper = document.createElement('div');
    svgWrapper.className = 'svg-wrapper';
    svgWrapper.innerHTML = svgContent;

    // Get SVG element
    const svgElement = svgWrapper.querySelector('svg');
    if (!svgElement) {
        container.innerHTML = '<p class="error">Invalid SVG content</p>';
        return;
    }

    // Optimize SVG attributes for proper scaling
    optimizeSVGAttributes(svgElement);

    // Only add container and wrapper (no zoom controls - split view has its own)
    container.appendChild(svgWrapper);

    console.log('[SVG] Optimized SVG display complete');
}

function optimizeSVGAttributes(svgElement) {
    // Ensure proper viewBox for scaling
    if (!svgElement.getAttribute('viewBox')) {
        const width = svgElement.getAttribute('width') || '100';
        const height = svgElement.getAttribute('height') || '100';
        svgElement.setAttribute('viewBox', `0 0 ${width} ${height}`);
    }

    // Remove fixed dimensions to allow responsive scaling
    svgElement.removeAttribute('width');
    svgElement.removeAttribute('height');

    // Add responsive styling
    svgElement.style.width = '100%';
    svgElement.style.height = 'auto';
    svgElement.style.maxWidth = '100%';
    svgElement.style.maxHeight = '100%';

    console.log('[SVG] Optimized attributes for responsive scaling');
}


// Split View Implementation - Complete functionality
class SplitViewController {
    constructor() {
        this.splitContainer = document.getElementById('splitViewContainer');
        this.isActive = true; // Always active since it's the only view
        this.isDragging = false;
        this.imageSynchronizer = null;

        // Drag state
        this.minPercentage = 20;
        this.maxPercentage = 80;

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

        // Initialize immediately since it's the only view
        this.loadSavedSplit();
    }


    syncImages() {
        // Copy original image
        const originalImg = document.getElementById('originalImage');
        const splitOriginalImg = document.getElementById('splitOriginalImage');

        if (originalImg && originalImg.src && splitOriginalImg) {
            splitOriginalImg.onerror = () => this.showImageError('left');
            splitOriginalImg.onload = () => this.initializeImageSync();
            splitOriginalImg.src = originalImg.src;
            splitOriginalImg.style.display = 'block';
        }

        // Copy SVG
        const originalSvg = document.getElementById('svgContainer');
        const splitSvg = document.getElementById('splitSvgContainer');

        if (originalSvg && splitSvg) {
            splitSvg.innerHTML = originalSvg.innerHTML;
        }
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
        console.log(`[Zoom] Applying zoom factor: ${factor}`);

        // Target specific elements in split view
        const leftImg = document.getElementById('splitOriginalImage');
        const rightSvg = document.querySelector('#splitSvgContainer svg');

        const elements = [leftImg, rightSvg].filter(el => el);
        console.log(`[Zoom] Found ${elements.length} elements to zoom`);

        elements.forEach(element => {
            const currentTransform = element.style.transform || 'scale(1)';
            const currentScale = parseFloat(currentTransform.match(/scale\(([\d.]+)\)/)?.[1] || '1');
            const newScale = Math.max(0.1, Math.min(5, currentScale * factor));
            element.style.transform = `scale(${newScale})`;
            element.style.transformOrigin = 'center center';
            console.log(`[Zoom] Applied scale ${newScale} to element`);
        });

        this.updateZoomDisplay();
    }

    resetZoom() {
        console.log('[Zoom] Resetting zoom to 100%');

        const leftImg = document.getElementById('splitOriginalImage');
        const rightSvg = document.querySelector('#splitSvgContainer svg');

        const elements = [leftImg, rightSvg].filter(el => el);
        elements.forEach(element => {
            element.style.transform = 'scale(1)';
            element.style.transformOrigin = 'center center';
        });

        this.updateZoomDisplay();
    }

    updateZoomDisplay() {
        const leftImg = document.getElementById('splitOriginalImage');
        if (!leftImg) {
            console.log('[Zoom] No image found for zoom display update');
            return;
        }

        const transform = leftImg.style.transform || 'scale(1)';
        const scale = parseFloat(transform.match(/scale\(([\d.]+)\)/)?.[1] || '1');
        const percentage = Math.round(scale * 100);

        // Update all zoom level displays in split view
        this.splitContainer.querySelectorAll('.zoom-level').forEach(display => {
            display.textContent = percentage + '%';
        });

        console.log(`[Zoom] Updated zoom display to ${percentage}%`);
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

            if (e.code === 'KeyR' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.updateSplit(50);
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
        console.error(message);
        // Could show user-facing error message here
    }


    // Public method for integration
    updateConversion() {
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

// Initialize Split View Controller
let splitViewController;
document.addEventListener('DOMContentLoaded', function() {
    // Initialize immediately for auto-convert support
    splitViewController = new SplitViewController();
});

// Integration hook - add to existing convert button handler
const originalConvertBtn = document.getElementById('convertBtn');
if (originalConvertBtn) {
    originalConvertBtn.addEventListener('click', function() {
        setTimeout(() => {
            if (splitViewController) {
                splitViewController.updateConversion();
            }
        }, 1000);
    });
}