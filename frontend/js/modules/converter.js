'use strict';

/**
 * Converter Module
 * Handles parameter collection, API calls, and result processing
 */

import appState from './appState.js';
import errorHandler from './errorHandler.js';

class ConverterModule {
    constructor(apiBase = '') {
        this.apiBase = apiBase;
        this.autoConvertTimer = null;
        this.initializeElements();
        this.setupEventListeners();
        this.setupParameterEvents();
        this.setupPresetDropdowns();
    }

    initializeElements() {
        this.converterSelect = document.getElementById('converter');
        this.convertBtn = document.getElementById('convertBtn');
        this.loadingDiv = document.getElementById('loading');
        this.metricsDiv = document.getElementById('metrics');

        // Parameter containers
        this.potraceParams = document.getElementById('potraceParams');
        this.vtracerParams = document.getElementById('vtracerParams');
        this.alphaParams = document.getElementById('alphaParams');
    }

    setupEventListeners() {
        // Convert button
        this.convertBtn.addEventListener('click', () => this.handleConvert());

        // Converter selection change
        this.converterSelect.addEventListener('change', (e) => {
            this.showConverterParams(e.target.value);
            this.triggerAutoConvert();
        });

        // Listen for upload events
        document.addEventListener('uploadComplete', (e) => {
            appState.currentFileId = e.detail.fileId;
        });

        // Set initial parameter display
        this.showConverterParams(this.converterSelect.value);
    }

    setupParameterEvents() {
        // Potrace parameter events
        document.getElementById('potraceThreshold').addEventListener('input', (e) => {
            document.getElementById('potraceThresholdValue').textContent = e.target.value;
            document.getElementById('potracePreset').value = 'custom';
            this.triggerAutoConvert();
        });

        document.getElementById('potraceAlphamax').addEventListener('input', (e) => {
            const value = (parseFloat(e.target.value) / 100).toFixed(2);
            document.getElementById('potraceAlphamaxValue').textContent = value;
            document.getElementById('potracePreset').value = 'custom';
            this.triggerAutoConvert();
        });

        document.getElementById('potraceOpttolerance').addEventListener('input', (e) => {
            const value = (parseFloat(e.target.value) / 100).toFixed(2);
            document.getElementById('potraceOpttoleranceValue').textContent = value;
            document.getElementById('potracePreset').value = 'custom';
            this.triggerAutoConvert();
        });

        document.getElementById('potraceTurdsize').addEventListener('input', (e) => {
            document.getElementById('potraceTurdsizeValue').textContent = e.target.value;
            document.getElementById('potracePreset').value = 'custom';
            this.triggerAutoConvert();
        });

        document.getElementById('potraceTurnpolicy').addEventListener('change', () => {
            document.getElementById('potracePreset').value = 'custom';
            this.triggerAutoConvert();
        });

        // VTracer parameter events
        document.getElementById('vtracerColorPrecision').addEventListener('input', (e) => {
            document.getElementById('vtracerColorPrecisionValue').textContent = e.target.value;
            document.getElementById('vtracerPreset').value = 'custom';
            this.triggerAutoConvert();
        });

        document.getElementById('vtracerLayerDifference').addEventListener('input', (e) => {
            document.getElementById('vtracerLayerDifferenceValue').textContent = e.target.value;
            document.getElementById('vtracerPreset').value = 'custom';
            this.triggerAutoConvert();
        });

        document.getElementById('vtracerPathPrecision').addEventListener('input', (e) => {
            document.getElementById('vtracerPathPrecisionValue').textContent = e.target.value;
            document.getElementById('vtracerPreset').value = 'custom';
            this.triggerAutoConvert();
        });

        document.getElementById('vtracerCornerThreshold').addEventListener('input', (e) => {
            document.getElementById('vtracerCornerThresholdValue').textContent = e.target.value;
            document.getElementById('vtracerPreset').value = 'custom';
            this.triggerAutoConvert();
        });

        document.getElementById('vtracerMaxIterations').addEventListener('input', (e) => {
            document.getElementById('vtracerMaxIterationsValue').textContent = e.target.value;
            document.getElementById('vtracerPreset').value = 'custom';
            this.triggerAutoConvert();
        });

        document.getElementById('vtracerSpliceThreshold').addEventListener('input', (e) => {
            document.getElementById('vtracerSpliceThresholdValue').textContent = e.target.value;
            document.getElementById('vtracerPreset').value = 'custom';
            this.triggerAutoConvert();
        });

        document.querySelectorAll('input[name="vtracerColormode"]').forEach(radio => {
            radio.addEventListener('change', () => {
                document.getElementById('vtracerPreset').value = 'custom';
                this.triggerAutoConvert();
            });
        });

        document.getElementById('vtracerLengthThreshold').addEventListener('change', () => {
            document.getElementById('vtracerPreset').value = 'custom';
            this.triggerAutoConvert();
        });

        // Alpha parameter events
        document.getElementById('alphaThreshold').addEventListener('input', (e) => {
            document.getElementById('alphaThresholdValue').textContent = e.target.value;
            document.getElementById('alphaPreset').value = 'custom';
            this.triggerAutoConvert();
        });

        document.getElementById('alphaUsePotrace').addEventListener('change', () => {
            document.getElementById('alphaPreset').value = 'custom';
            this.triggerAutoConvert();
        });

        document.getElementById('alphaPreserveAntialiasing').addEventListener('change', () => {
            document.getElementById('alphaPreset').value = 'custom';
            this.triggerAutoConvert();
        });
    }

    setupPresetDropdowns() {
        document.getElementById('potracePreset').addEventListener('change', (e) => {
            if (e.target.value !== 'custom') {
                this.applyPotracePreset(e.target.value);
            }
        });

        document.getElementById('vtracerPreset').addEventListener('change', (e) => {
            if (e.target.value !== 'custom') {
                this.applyVTracerPreset(e.target.value);
            }
        });

        document.getElementById('alphaPreset').addEventListener('change', (e) => {
            if (e.target.value !== 'custom') {
                this.applyAlphaPreset(e.target.value);
            }
        });
    }

    showConverterParams(converter) {
        // Hide all parameter groups
        this.potraceParams.classList.add('hidden');
        this.vtracerParams.classList.add('hidden');
        this.alphaParams.classList.add('hidden');

        // Show the selected converter's parameters
        switch(converter) {
            case 'smart_auto':
                // Smart Auto doesn't need parameter configuration - it's automatic
                break;
            case 'smart':
                // Smart Potrace uses same parameters as regular Potrace
                this.potraceParams.classList.remove('hidden');
                break;
            case 'potrace':
                this.potraceParams.classList.remove('hidden');
                break;
            case 'vtracer':
                this.vtracerParams.classList.remove('hidden');
                break;
            case 'alpha':
                this.alphaParams.classList.remove('hidden');
                break;
        }

        // Re-initialize tooltips after showing parameters
        setTimeout(() => {
            if (window.initializeTooltips) {
                window.initializeTooltips();
            }
        }, 100);
    }

    triggerAutoConvert() {
        // Clear any pending conversion
        if (this.autoConvertTimer) {
            clearTimeout(this.autoConvertTimer);
        }

        // Only auto-convert if we have a file loaded
        if (!appState.currentFileId) return;

        // Debounce: wait 500ms after user stops changing settings
        this.autoConvertTimer = setTimeout(() => {
            console.log('[Auto-convert] Settings changed, updating preview');
            this.handleConvert();
        }, 500);
    }

    async handleConvert() {
        try {
            if (!appState.currentFileId) {
                errorHandler.showUserError('Please upload an image first', { type: 'warning' });
                return;
            }

            // Show loading state
            this.loadingDiv.classList.remove('hidden');
            this.metricsDiv.classList.add('hidden');
            appState.isConverting = true;

            // Disable button and change text
            this.convertBtn.disabled = true;
            this.convertBtn.textContent = 'Converting...';

            // Collect parameters based on selected converter
            const converter = this.converterSelect.value;
            let requestData = {
                file_id: appState.currentFileId,
                converter: converter
            };

            // Add converter-specific parameters
            console.log('[Frontend] Selected converter:', converter);
            switch(converter) {
                case 'smart_auto':
                    // Smart Auto automatically selects optimal parameters
                    console.log('[Frontend] Smart Auto - using automatic parameter selection');
                    break;
                case 'smart':
                    // Smart Potrace uses same parameters as regular Potrace
                    const smartParams = this.collectPotraceParams();
                    console.log('[Frontend] Smart Potrace params:', smartParams);
                    Object.assign(requestData, smartParams);
                    break;
                case 'potrace':
                    Object.assign(requestData, this.collectPotraceParams());
                    break;
                case 'vtracer':
                    Object.assign(requestData, this.collectVTracerParams());
                    break;
                case 'alpha':
                    Object.assign(requestData, this.collectAlphaParams());
                    break;
            }

            console.log('[Frontend] Sending conversion request:', requestData);

            const response = await fetch(`${this.apiBase}/api/convert`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`Conversion request failed with status ${response.status}`);
            }

            const result = await response.json();

            if (!result.success) {
                throw new Error(result.error || 'Conversion failed');
            }

            // Store result
            appState.currentSvgContent = result.svg;
            appState.conversionResults = result;

            // Display results
            this.displayConversionResults(result, converter);

            // Emit conversion complete event
            this.emitConversionComplete(result);

        } catch (error) {
            errorHandler.handleConversionError(error, {
                metadata: {
                    converter: this.converterSelect.value,
                    fileId: appState.currentFileId
                }
            });
        } finally {
            // Reset UI state
            appState.isConverting = false;
            this.loadingDiv.classList.add('hidden');
            this.convertBtn.disabled = false;
            this.convertBtn.textContent = 'Convert';
        }
    }

    displayConversionResults(result, converter) {
        // Debug SVG content
        console.log('[Frontend] SVG length:', result.svg.length);
        console.log('[Frontend] SVG preview:', result.svg.substring(0, 300));

        // Display metrics
        document.getElementById('ssimScore').textContent = (result.ssim * 100).toFixed(1) + '%';
        document.getElementById('fileSize').textContent = this.formatFileSize(result.size);
        document.getElementById('pathCount').textContent = result.path_count || '-';
        document.getElementById('avgPathLength').textContent = result.avg_path_length || '-';

        // Show routing information if using smart_auto
        if (converter === 'smart_auto' && result.routing_info) {
            console.log('[Frontend] Displaying routing info:', result.routing_info);
            this.displayRoutingInfo(result.routing_info);
        } else {
            // Hide routing info for other converters
            document.getElementById('routingInfo').classList.add('hidden');
        }

        // Always show metrics div after first conversion
        this.metricsDiv.classList.remove('hidden');
    }

    collectPotraceParams() {
        const params = {
            threshold: parseInt(document.getElementById('potraceThreshold').value),
            turnpolicy: document.getElementById('potraceTurnpolicy').value,
            turdsize: parseInt(document.getElementById('potraceTurdsize').value),
            alphamax: parseFloat(document.getElementById('potraceAlphamax').value) / 100,
            opttolerance: parseFloat(document.getElementById('potraceOpttolerance').value) / 100
        };
        console.log('[Frontend] Collected Potrace params:', params);
        return params;
    }

    collectVTracerParams() {
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

    collectAlphaParams() {
        return {
            threshold: parseInt(document.getElementById('alphaThreshold').value),
            use_potrace: document.getElementById('alphaUsePotrace').checked,
            preserve_antialiasing: document.getElementById('alphaPreserveAntialiasing').checked
        };
    }

    applyPotracePreset(preset) {
        switch(preset) {
            case 'quality':
                document.getElementById('potraceThreshold').value = 100;
                document.getElementById('potraceTurnpolicy').value = 'white';
                document.getElementById('potraceTurdsize').value = 5;
                document.getElementById('potraceAlphamax').value = 100;
                document.getElementById('potraceOpttolerance').value = 10;
                break;
            case 'fast':
                document.getElementById('potraceThreshold').value = 128;
                document.getElementById('potraceTurnpolicy').value = 'black';
                document.getElementById('potraceTurdsize').value = 1;
                document.getElementById('potraceAlphamax').value = 134;
                document.getElementById('potraceOpttolerance').value = 20;
                break;
            case 'balanced':
                document.getElementById('potraceThreshold').value = 128;
                document.getElementById('potraceTurnpolicy').value = 'white';
                document.getElementById('potraceTurdsize').value = 2;
                document.getElementById('potraceAlphamax').value = 100;
                document.getElementById('potraceOpttolerance').value = 20;
                break;
            case 'text':
                document.getElementById('potraceThreshold').value = 128;
                document.getElementById('potraceTurnpolicy').value = 'black';
                document.getElementById('potraceTurdsize').value = 1;
                document.getElementById('potraceAlphamax').value = 134;
                document.getElementById('potraceOpttolerance').value = 1;
                break;
        }
        this.updatePotraceDisplayValues();
        this.triggerAutoConvert();
    }

    applyVTracerPreset(preset) {
        switch(preset) {
            case 'quality':
                document.querySelector('input[name="vtracerColormode"][value="color"]').checked = true;
                document.getElementById('vtracerColorPrecision').value = 8;
                document.getElementById('vtracerLayerDifference').value = 8;
                document.getElementById('vtracerPathPrecision').value = 8;
                document.getElementById('vtracerCornerThreshold').value = 30;
                document.getElementById('vtracerLengthThreshold').value = 2.0;
                document.getElementById('vtracerMaxIterations').value = 20;
                document.getElementById('vtracerSpliceThreshold').value = 60;
                break;
            case 'fast':
                document.querySelector('input[name="vtracerColormode"][value="color"]').checked = true;
                document.getElementById('vtracerColorPrecision').value = 4;
                document.getElementById('vtracerLayerDifference').value = 16;
                document.getElementById('vtracerPathPrecision').value = 3;
                document.getElementById('vtracerCornerThreshold').value = 60;
                document.getElementById('vtracerLengthThreshold').value = 5.0;
                document.getElementById('vtracerMaxIterations').value = 5;
                document.getElementById('vtracerSpliceThreshold').value = 30;
                break;
        }
        this.updateVTracerDisplayValues();
        this.triggerAutoConvert();
    }

    applyAlphaPreset(preset) {
        switch(preset) {
            case 'quality':
                document.getElementById('alphaThreshold').value = 64;
                document.getElementById('alphaUsePotrace').checked = true;
                document.getElementById('alphaPreserveAntialiasing').checked = true;
                break;
            case 'fast':
                document.getElementById('alphaThreshold').value = 128;
                document.getElementById('alphaUsePotrace').checked = true;
                document.getElementById('alphaPreserveAntialiasing').checked = false;
                break;
        }
        this.updateAlphaDisplayValues();
        this.triggerAutoConvert();
    }

    updatePotraceDisplayValues() {
        document.getElementById('potraceThresholdValue').textContent = document.getElementById('potraceThreshold').value;
        document.getElementById('potraceAlphamaxValue').textContent = (parseFloat(document.getElementById('potraceAlphamax').value) / 100).toFixed(2);
        document.getElementById('potraceOpttoleranceValue').textContent = (parseFloat(document.getElementById('potraceOpttolerance').value) / 100).toFixed(2);
        document.getElementById('potraceTurdsizeValue').textContent = document.getElementById('potraceTurdsize').value;
    }

    updateVTracerDisplayValues() {
        document.getElementById('vtracerColorPrecisionValue').textContent = document.getElementById('vtracerColorPrecision').value;
        document.getElementById('vtracerLayerDifferenceValue').textContent = document.getElementById('vtracerLayerDifference').value;
        document.getElementById('vtracerPathPrecisionValue').textContent = document.getElementById('vtracerPathPrecision').value;
        document.getElementById('vtracerCornerThresholdValue').textContent = document.getElementById('vtracerCornerThreshold').value;
        document.getElementById('vtracerMaxIterationsValue').textContent = document.getElementById('vtracerMaxIterations').value;
        document.getElementById('vtracerSpliceThresholdValue').textContent = document.getElementById('vtracerSpliceThreshold').value;
    }

    updateAlphaDisplayValues() {
        document.getElementById('alphaThresholdValue').textContent = document.getElementById('alphaThreshold').value;
    }

    displayRoutingInfo(routingInfo) {
        const routingDiv = document.getElementById('routingInfo');
        routingDiv.classList.remove('hidden');
        // Implementation would depend on the routing info structure
        console.log('Routing info:', routingInfo);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showError(message) {
        errorHandler.showUserError(message, { type: 'error' });
    }

    getCurrentSvgContent() {
        return appState.currentSvgContent;
    }

    // Event emitters for module communication
    emitConversionComplete(result) {
        const event = new CustomEvent('conversionComplete', {
            detail: { result, svgContent: appState.currentSvgContent }
        });
        document.dispatchEvent(event);
    }

    // Reset state
    reset() {
        appState.currentFileId = null;
        appState.currentSvgContent = null;

        if (this.autoConvertTimer) {
            clearTimeout(this.autoConvertTimer);
            this.autoConvertTimer = null;
        }
    }
}

// Export for module use
export default ConverterModule;