# AI Modules Integration Patterns

## Overview

This document describes common patterns for integrating AI modules with the existing SVG-AI Enhanced Conversion Pipeline.

## Core Integration Patterns

### 1. Pipeline Integration Pattern

The AI modules are designed to integrate seamlessly with the existing VTracer-based conversion pipeline.

```python
from backend.converters.vtracer_converter import VTracerConverter
from backend.ai_modules.base_ai_converter import BaseAIConverter

class AIEnhancedVTracerConverter(BaseAIConverter):
    """AI-enhanced VTracer converter"""

    def __init__(self):
        super().__init__("AI-Enhanced VTracer")
        self.vtracer_converter = VTracerConverter()
        self.feature_extractor = ImageFeatureExtractor()
        self.classifier = RuleBasedClassifier()
        self.optimizer = FeatureMappingOptimizer()
        self.predictor = QualityPredictor()

    def convert(self, image_path: str, **kwargs) -> str:
        """Convert with AI parameter optimization"""
        # Extract features
        features = self.extract_features(image_path)

        # Classify and optimize
        logo_type, confidence = self.classify_image(image_path)
        optimized_params = self.optimize_parameters(image_path, features)

        # Use optimized parameters for VTracer conversion
        return self.vtracer_converter.convert(image_path, **optimized_params)

    def extract_features(self, image_path: str):
        return self.feature_extractor.extract_features(image_path)

    def classify_image(self, image_path: str):
        features = self.extract_features(image_path)
        return self.classifier.classify(features)

    def optimize_parameters(self, image_path: str, features: dict):
        return self.optimizer.optimize(features)

    def predict_quality(self, image_path: str, parameters: dict):
        return self.predictor.predict_quality(image_path, parameters)
```

### 2. API Endpoint Integration Pattern

Integrate AI analysis with existing API endpoints.

```python
from flask import Flask, request, jsonify
from backend.api.conversion_api import existing_convert_endpoint

@app.route('/api/convert/ai-enhanced', methods=['POST'])
def ai_enhanced_convert():
    """AI-enhanced conversion endpoint"""
    try:
        # Get image file from request
        image_file = request.files.get('image')
        options = request.form.to_dict()

        # Save uploaded file temporarily
        temp_path = save_uploaded_file(image_file)

        # Create AI converter
        ai_converter = AIEnhancedVTracerConverter()

        # Perform AI analysis
        features = ai_converter.extract_features(temp_path)
        logo_type, confidence = ai_converter.classify_image(temp_path)
        optimized_params = ai_converter.optimize_parameters(temp_path, features)
        predicted_quality = ai_converter.predict_quality(temp_path, optimized_params)

        # Convert using optimized parameters
        svg_content = ai_converter.convert(temp_path, **optimized_params)

        # Return enhanced response
        return jsonify({
            'success': True,
            'svg': svg_content,
            'ai_analysis': {
                'features': features,
                'logo_type': logo_type,
                'confidence': confidence,
                'optimized_parameters': optimized_params,
                'predicted_quality': predicted_quality
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### 3. Batch Processing Integration Pattern

Integrate AI modules with existing batch processing workflows.

```python
from backend.batch_convert import BatchConverter
import concurrent.futures

class AIBatchConverter(BatchConverter):
    """AI-enhanced batch converter"""

    def __init__(self):
        super().__init__()
        self.ai_converter = AIEnhancedVTracerConverter()

    def process_batch_with_ai(self, image_paths: list, output_dir: str):
        """Process batch with AI optimization"""
        results = []

        def process_single_image(image_path):
            try:
                # AI analysis
                result = self.ai_converter.convert_with_ai_metadata(image_path)

                # Save SVG
                output_path = self._get_output_path(image_path, output_dir)
                with open(output_path, 'w') as f:
                    f.write(result['svg'])

                return {
                    'input_path': image_path,
                    'output_path': output_path,
                    'success': True,
                    'metadata': result['metadata']
                }

            except Exception as e:
                return {
                    'input_path': image_path,
                    'success': False,
                    'error': str(e)
                }

        # Process concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_single_image, path) for path in image_paths]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        return results

    def generate_batch_report(self, results: list):
        """Generate enhanced batch report with AI insights"""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        # Analyze AI insights
        logo_types = {}
        avg_quality = 0
        total_processing_time = 0

        for result in successful:
            metadata = result.get('metadata', {})
            logo_type = metadata.get('logo_type', 'unknown')
            logo_types[logo_type] = logo_types.get(logo_type, 0) + 1
            avg_quality += metadata.get('predicted_quality', 0)
            total_processing_time += metadata.get('processing_time', 0)

        if successful:
            avg_quality /= len(successful)

        report = {
            'summary': {
                'total_images': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(results) if results else 0
            },
            'ai_insights': {
                'logo_type_distribution': logo_types,
                'average_predicted_quality': avg_quality,
                'total_processing_time': total_processing_time,
                'average_processing_time': total_processing_time / len(successful) if successful else 0
            },
            'failed_conversions': [r['input_path'] for r in failed]
        }

        return report
```

### 4. Web Interface Integration Pattern

Integrate AI capabilities with the existing web interface.

```python
# Add to web_server.py

@app.route('/analyze', methods=['POST'])
def analyze_upload():
    """Analyze uploaded image with AI"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save uploaded file
        temp_path = save_uploaded_file(file)

        # AI analysis
        ai_converter = AIEnhancedVTracerConverter()
        features = ai_converter.extract_features(temp_path)
        logo_type, confidence = ai_converter.classify_image(temp_path)
        optimized_params = ai_converter.optimize_parameters(temp_path, features)
        predicted_quality = ai_converter.predict_quality(temp_path, optimized_params)

        # Clean up
        os.unlink(temp_path)

        return jsonify({
            'success': True,
            'analysis': {
                'logo_type': logo_type,
                'confidence': confidence,
                'predicted_quality': predicted_quality,
                'features': features,
                'recommended_parameters': optimized_params
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Add JavaScript for frontend integration
"""
// Add to static/js/main.js

function analyzeImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayAnalysis(data.analysis);
        } else {
            showError(data.error);
        }
    })
    .catch(error => {
        console.error('Analysis failed:', error);
        showError('Analysis failed');
    });
}

function displayAnalysis(analysis) {
    // Update UI with AI analysis results
    document.getElementById('logo-type').textContent = analysis.logo_type;
    document.getElementById('confidence').textContent = (analysis.confidence * 100).toFixed(1) + '%';
    document.getElementById('predicted-quality').textContent = analysis.predicted_quality.toFixed(3);

    // Show recommended parameters
    const paramsDiv = document.getElementById('recommended-params');
    paramsDiv.innerHTML = '';
    for (const [param, value] of Object.entries(analysis.recommended_parameters)) {
        const paramEl = document.createElement('div');
        paramEl.innerHTML = `<strong>${param}:</strong> ${value}`;
        paramsDiv.appendChild(paramEl);
    }
}
"""
```

### 5. Caching Integration Pattern

Integrate AI results with the existing caching system.

```python
from backend.utils.cache import CacheManager

class AICacheManager(CacheManager):
    """Enhanced cache manager for AI results"""

    def __init__(self):
        super().__init__()
        self.ai_cache = {}

    def get_ai_analysis(self, image_path: str):
        """Get cached AI analysis"""
        cache_key = self._get_cache_key(image_path)
        return self.ai_cache.get(cache_key)

    def cache_ai_analysis(self, image_path: str, analysis_result: dict):
        """Cache AI analysis results"""
        cache_key = self._get_cache_key(image_path)
        self.ai_cache[cache_key] = {
            'result': analysis_result,
            'timestamp': time.time()
        }

        # Limit cache size
        if len(self.ai_cache) > 1000:
            oldest_key = min(self.ai_cache.keys(),
                           key=lambda k: self.ai_cache[k]['timestamp'])
            del self.ai_cache[oldest_key]

    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key for image"""
        import hashlib
        with open(image_path, 'rb') as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()

# Usage in AI converter
class CachedAIConverter(AIEnhancedVTracerConverter):
    """AI converter with caching"""

    def __init__(self):
        super().__init__()
        self.cache_manager = AICacheManager()

    def convert_with_ai_metadata(self, image_path: str, **kwargs):
        """Convert with caching"""
        # Check cache first
        cached_result = self.cache_manager.get_ai_analysis(image_path)
        if cached_result:
            return cached_result['result']

        # Perform AI analysis
        result = super().convert_with_ai_metadata(image_path, **kwargs)

        # Cache result
        self.cache_manager.cache_ai_analysis(image_path, result)

        return result
```

### 6. Quality Monitoring Integration Pattern

Integrate AI quality predictions with the existing quality monitoring system.

```python
from backend.utils.quality_metrics import QualityCalculator

class AIQualityMonitor:
    """Monitor AI predictions vs actual quality"""

    def __init__(self):
        self.quality_calculator = QualityCalculator()
        self.prediction_history = []

    def validate_prediction(self, image_path: str, svg_content: str,
                          predicted_quality: float):
        """Validate AI quality prediction against actual metrics"""
        try:
            # Calculate actual quality
            actual_metrics = self.quality_calculator.calculate_comprehensive_metrics(
                image_path, svg_content
            )
            actual_ssim = actual_metrics.ssim

            # Record prediction accuracy
            prediction_error = abs(predicted_quality - actual_ssim)
            self.prediction_history.append({
                'image_path': image_path,
                'predicted': predicted_quality,
                'actual': actual_ssim,
                'error': prediction_error,
                'timestamp': time.time()
            })

            # Alert if prediction is significantly off
            if prediction_error > 0.2:  # 20% error threshold
                self._log_prediction_warning(image_path, predicted_quality, actual_ssim)

            return {
                'predicted_quality': predicted_quality,
                'actual_quality': actual_ssim,
                'prediction_error': prediction_error,
                'accuracy_grade': self._grade_prediction(prediction_error)
            }

        except Exception as e:
            print(f"Quality validation failed: {e}")
            return None

    def get_prediction_accuracy(self, window_size: int = 100):
        """Get recent prediction accuracy statistics"""
        recent_predictions = self.prediction_history[-window_size:]

        if not recent_predictions:
            return None

        errors = [p['error'] for p in recent_predictions]
        return {
            'sample_size': len(recent_predictions),
            'mean_error': sum(errors) / len(errors),
            'max_error': max(errors),
            'accuracy_rate': sum(1 for e in errors if e < 0.1) / len(errors),  # <10% error
            'recent_predictions': recent_predictions[-10:]  # Last 10 predictions
        }

    def _grade_prediction(self, error: float) -> str:
        """Grade prediction accuracy"""
        if error < 0.05:
            return "Excellent"
        elif error < 0.1:
            return "Good"
        elif error < 0.2:
            return "Fair"
        else:
            return "Poor"

    def _log_prediction_warning(self, image_path: str, predicted: float, actual: float):
        """Log warning for poor predictions"""
        print(f"⚠️  AI Prediction Warning:")
        print(f"   Image: {image_path}")
        print(f"   Predicted: {predicted:.3f}")
        print(f"   Actual: {actual:.3f}")
        print(f"   Error: {abs(predicted - actual):.3f}")
```

### 7. Configuration Integration Pattern

Integrate AI configuration with existing system configuration.

```python
# Add to existing config.py or create ai_config.py

class AIConfig:
    """AI-specific configuration management"""

    def __init__(self, base_config=None):
        self.base_config = base_config or {}
        self.ai_settings = self._load_ai_settings()

    def _load_ai_settings(self):
        """Load AI-specific settings"""
        return {
            'enable_ai_optimization': os.getenv('ENABLE_AI_OPTIMIZATION', 'true').lower() == 'true',
            'ai_quality_threshold': float(os.getenv('AI_QUALITY_THRESHOLD', '0.85')),
            'max_optimization_time': float(os.getenv('MAX_OPTIMIZATION_TIME', '10.0')),
            'enable_prediction_caching': os.getenv('ENABLE_PREDICTION_CACHING', 'true').lower() == 'true',
            'ai_log_level': os.getenv('AI_LOG_LEVEL', 'INFO'),
            'fallback_to_defaults': os.getenv('AI_FALLBACK_DEFAULTS', 'true').lower() == 'true'
        }

    def should_use_ai_optimization(self) -> bool:
        """Check if AI optimization should be used"""
        return self.ai_settings['enable_ai_optimization']

    def get_quality_threshold(self) -> float:
        """Get quality threshold for AI predictions"""
        return self.ai_settings['ai_quality_threshold']

    def get_optimization_timeout(self) -> float:
        """Get timeout for optimization operations"""
        return self.ai_settings['max_optimization_time']

# Usage in converters
def create_converter(config: AIConfig):
    """Factory function for creating appropriate converter"""
    if config.should_use_ai_optimization():
        return AIEnhancedVTracerConverter()
    else:
        return VTracerConverter()
```

## Best Practices for Integration

### 1. Graceful Degradation

Always implement fallback mechanisms when AI components fail:

```python
def safe_ai_conversion(image_path: str, **kwargs):
    """AI conversion with fallback to standard conversion"""
    try:
        # Try AI-enhanced conversion
        ai_converter = AIEnhancedVTracerConverter()
        return ai_converter.convert_with_ai_metadata(image_path, **kwargs)
    except Exception as e:
        # Fall back to standard conversion
        print(f"AI conversion failed: {e}. Falling back to standard conversion.")
        standard_converter = VTracerConverter()
        svg_content = standard_converter.convert(image_path, **kwargs)
        return {
            'svg': svg_content,
            'metadata': {'fallback_used': True, 'error': str(e)},
            'success': True
        }
```

### 2. Performance Monitoring

Monitor AI performance impact on the overall system:

```python
@performance_monitor.time_operation("ai_enhanced_conversion")
def monitored_ai_conversion(image_path: str):
    """AI conversion with performance monitoring"""
    converter = AIEnhancedVTracerConverter()
    return converter.convert_with_ai_metadata(image_path)

def check_ai_performance():
    """Check if AI is meeting performance targets"""
    summary = performance_monitor.get_summary("ai_enhanced_conversion")
    if summary and summary['average_duration'] > 10.0:  # 10 second threshold
        print("⚠️  AI conversion is taking too long, consider optimization")
        return False
    return True
```

### 3. Feature Flags

Use feature flags to control AI functionality:

```python
class FeatureFlags:
    """Feature flag management for AI components"""

    @staticmethod
    def is_ai_classification_enabled() -> bool:
        return os.getenv('AI_CLASSIFICATION_ENABLED', 'true').lower() == 'true'

    @staticmethod
    def is_ai_optimization_enabled() -> bool:
        return os.getenv('AI_OPTIMIZATION_ENABLED', 'true').lower() == 'true'

    @staticmethod
    def is_quality_prediction_enabled() -> bool:
        return os.getenv('QUALITY_PREDICTION_ENABLED', 'true').lower() == 'true'

# Usage in converter
def convert_with_selective_ai(image_path: str):
    """Convert using only enabled AI features"""
    converter = VTracerConverter()

    # Start with default parameters
    parameters = converter.get_default_parameters()

    if FeatureFlags.is_ai_classification_enabled():
        # Use AI classification
        features = extractor.extract_features(image_path)
        logo_type, confidence = classifier.classify(features)
        print(f"Classified as: {logo_type} ({confidence:.1%})")

    if FeatureFlags.is_ai_optimization_enabled():
        # Use AI optimization
        optimizer = FeatureMappingOptimizer()
        parameters = optimizer.optimize(features)

    # Convert with determined parameters
    svg_content = converter.convert(image_path, **parameters)

    if FeatureFlags.is_quality_prediction_enabled():
        # Predict quality
        predictor = QualityPredictor()
        predicted_quality = predictor.predict_quality(image_path, parameters)
        print(f"Predicted quality: {predicted_quality:.3f}")

    return svg_content
```

These integration patterns provide flexible ways to incorporate AI capabilities into existing workflows while maintaining system stability and performance.