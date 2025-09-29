class ErrorHandler {
    static handleClassificationError(error, container) {
        let errorMessage = 'Classification failed. ';

        if (error.message.includes('No image file')) {
            errorMessage += 'Please select an image file.';
        } else if (error.message.includes('Invalid image')) {
            errorMessage += 'Please select a valid image file (PNG, JPG, JPEG).';
        } else if (error.message.includes('too large')) {
            errorMessage += 'Image file is too large. Please use a smaller image.';
        } else if (error.message.includes('timeout')) {
            errorMessage += 'Classification took too long. Try using a faster method.';
        } else {
            errorMessage += 'Please try again or contact support.';
        }

        container.innerHTML = `
            <div class="error-message">
                <i class="error-icon">⚠️</i>
                <span>${errorMessage}</span>
                <button onclick="this.parentElement.style.display='none'">Dismiss</button>
            </div>
        `;
    }

    static showLoadingIndicator(container, message = 'Classifying logo...') {
        container.innerHTML = `
            <div class="loading-indicator">
                <div class="spinner"></div>
                <span>${message}</span>
            </div>
        `;
    }

    static clearMessages(container) {
        container.innerHTML = '';
    }
}

// Progress indicators for long-running operations
async function classifyWithProgress(file) {
    const resultsContainer = document.getElementById('classificationResults');
    const method = document.getElementById('classificationMethod').value;

    try {
        // Show loading indicator
        ErrorHandler.showLoadingIndicator(resultsContainer,
            method === 'neural_network' ? 'Running neural network analysis...' : 'Analyzing logo...');

        // Start classification
        const result = await logoClassifier.classifyLogo(file, {
            method: method,
            includeFeatures: document.getElementById('showFeatures').checked,
            timeBudget: document.getElementById('timeBudget').value || undefined
        });

        // Display results
        logoClassifier.displayClassificationResult(result, resultsContainer);

        // Show features if requested
        if (result.features && document.getElementById('showFeatures').checked) {
            const featuresContainer = document.getElementById('featuresAnalysis');
            logoClassifier.displayFeatures(result.features, featuresContainer);
        }

        return result;

    } catch (error) {
        ErrorHandler.handleClassificationError(error, resultsContainer);
        throw error;
    }
}

// Make functions globally available
window.ErrorHandler = ErrorHandler;
window.classifyWithProgress = classifyWithProgress;

export { ErrorHandler, classifyWithProgress };