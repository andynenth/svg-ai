'use strict';

// Global variables
let currentFileId = null;
let currentSvgContent = null;
const API_BASE = '';

// Get DOM Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const mainContent = document.getElementById('mainContent');
const thresholdSlider = document.getElementById('threshold');
const thresholdValue = document.getElementById('thresholdValue');
const convertBtn = document.getElementById('convertBtn');
const converterSelect = document.getElementById('converter');
const loadingDiv = document.getElementById('loading');
const metricsDiv = document.getElementById('metrics');
const downloadBtn = document.getElementById('downloadBtn');

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

// Handle Slider Change
thresholdSlider.addEventListener('input', (e) => {
    thresholdValue.textContent = e.target.value;
});

// Setup Convert Button
convertBtn.addEventListener('click', handleConvert);

// Setup Download Button
downloadBtn.addEventListener('click', handleDownload);

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

    const requestData = {
        file_id: currentFileId,
        threshold: parseInt(thresholdSlider.value),
        converter: converterSelect.value
    };

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
        document.getElementById('svgContainer').innerHTML = result.svg;
        document.getElementById('ssimScore').textContent = (result.ssim * 100).toFixed(1) + '%';
        document.getElementById('fileSize').textContent = formatFileSize(result.size);
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