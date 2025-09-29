class LogoClassifier {
    constructor() {
        this.apiBase = '/api';
        this.currentClassification = null;
    }

    async classifyLogo(file, options = {}) {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('method', options.method || 'auto');
        formData.append('include_features', options.includeFeatures || 'false');

        if (options.timeBudget) {
            formData.append('time_budget', options.timeBudget);
        }

        try {
            const response = await fetch(`${this.apiBase}/classify-logo`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Classification failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.currentClassification = result;
            return result;

        } catch (error) {
            console.error('Logo classification error:', error);
            throw error;
        }
    }

    async analyzeFeatures(file) {
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch(`${this.apiBase}/analyze-logo-features`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Feature analysis failed: ${response.statusText}`);
            }

            return await response.json();

        } catch (error) {
            console.error('Feature analysis error:', error);
            throw error;
        }
    }

    async convertWithAI(file_id, options = {}) {
        // The convert endpoint expects JSON with file_id, not FormData
        const requestData = {
            file_id: file_id,
            use_ai: true,
            ai_method: options.method || 'auto',
            // Add any VTracer parameter overrides
            ...options.parameters
        };

        try {
            const response = await fetch(`${this.apiBase}/convert`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`AI conversion failed: ${response.statusText}`);
            }

            return await response.json();

        } catch (error) {
            console.error('AI conversion error:', error);
            throw error;
        }
    }

    displayClassificationResult(result, container) {
        if (!result || !container) return;

        const logoTypeColors = {
            'simple': '#4CAF50',
            'text': '#2196F3',
            'gradient': '#FF9800',
            'complex': '#9C27B0',
            'unknown': '#757575'
        };

        const confidenceColor = result.confidence > 0.8 ? '#4CAF50' :
                               result.confidence > 0.6 ? '#FF9800' : '#F44336';

        container.innerHTML = `
            <div class="classification-result">
                <h4>Logo Classification</h4>
                <div class="logo-type" style="color: ${logoTypeColors[result.logo_type]}">
                    <strong>${result.logo_type.toUpperCase()}</strong>
                </div>
                <div class="confidence" style="color: ${confidenceColor}">
                    Confidence: ${(result.confidence * 100).toFixed(1)}%
                </div>
                <div class="method-used">
                    Method: ${result.method_used.replace('_', ' ')}
                </div>
                <div class="processing-time">
                    Time: ${(result.processing_time * 1000).toFixed(0)}ms
                </div>
                ${result.reasoning ? `<div class="reasoning">${result.reasoning}</div>` : ''}
            </div>
        `;
    }

    displayFeatures(features, container) {
        if (!features || !container) return;

        const featureDescriptions = {
            'edge_density': 'Edge Content',
            'unique_colors': 'Color Complexity',
            'entropy': 'Information Content',
            'corner_density': 'Sharp Features',
            'gradient_strength': 'Gradient Strength',
            'complexity_score': 'Overall Complexity'
        };

        let featuresHtml = '<div class="features-analysis"><h4>Image Features</h4>';

        Object.entries(features).forEach(([key, value]) => {
            const percentage = (value * 100).toFixed(1);
            const description = featureDescriptions[key] || key;

            featuresHtml += `
                <div class="feature-item">
                    <label>${description}:</label>
                    <div class="feature-bar">
                        <div class="feature-value" style="width: ${percentage}%"></div>
                    </div>
                    <span class="feature-percentage">${percentage}%</span>
                </div>
            `;
        });

        featuresHtml += '</div>';
        container.innerHTML = featuresHtml;
    }
}

// Initialize global classifier
window.logoClassifier = new LogoClassifier();