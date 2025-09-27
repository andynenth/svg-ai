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

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        currentFileId = data.file_id;

        // Display Original Image
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('originalImage').src = e.target.result;
            mainContent.classList.remove('hidden');
        };
        reader.readAsDataURL(file);

    } catch (error) {
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

        // Display Results
        currentSvgContent = result.svg;

        // Debug SVG content
        console.log('[Frontend] SVG length:', result.svg.length);
        console.log('[Frontend] SVG preview:', result.svg.substring(0, 300));
        console.log('[Frontend] SVG contains <rect>:', result.svg.includes('<rect'));
        console.log('[Frontend] SVG contains fill="white":', result.svg.includes('fill="white"'));
        console.log('[Frontend] SVG contains fill="#000000":', result.svg.includes('fill="#000000"'));

        document.getElementById('svgContainer').innerHTML = result.svg;

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