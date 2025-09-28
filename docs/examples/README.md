# AI Modules Usage Examples

## Overview

This document provides practical examples for using the AI modules in the SVG-AI Enhanced Conversion Pipeline.

## Basic Usage Examples

### Example 1: Basic Feature Extraction

```python
#!/usr/bin/env python3
"""Basic feature extraction example"""

from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor

def basic_feature_extraction():
    # Initialize feature extractor
    extractor = ImageFeatureExtractor()

    # Extract features from an image
    image_path = "data/logos/simple_geometric/circle_00.png"
    features = extractor.extract_features(image_path)

    # Display features
    print("Extracted Features:")
    for feature_name, value in features.items():
        print(f"  {feature_name}: {value:.3f}")

    # Check if it's a simple logo (example criteria)
    if features['unique_colors'] <= 5 and features['complexity_score'] <= 0.3:
        print("✅ This appears to be a simple logo")
    else:
        print("ℹ️  This appears to be a complex logo")

if __name__ == "__main__":
    basic_feature_extraction()
```

### Example 2: Logo Classification

```python
#!/usr/bin/env python3
"""Logo classification example"""

from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier

def classify_logo():
    # Initialize components
    extractor = ImageFeatureExtractor()
    classifier = RuleBasedClassifier()

    # Process image
    image_path = "data/logos/text_based/company_logo.png"

    # Extract features
    features = extractor.extract_features(image_path)
    print(f"Features extracted: {len(features)} features")

    # Classify logo type
    logo_type, confidence = classifier.classify(features)

    print(f"Classification Result:")
    print(f"  Logo Type: {logo_type}")
    print(f"  Confidence: {confidence:.2%}")

    # Provide recommendations based on type
    recommendations = {
        'simple': "Use low color precision and moderate corner threshold",
        'text': "Use high corner threshold and low color precision",
        'gradient': "Use high color precision and layer difference",
        'complex': "Use balanced parameters with higher iterations"
    }

    print(f"Recommendation: {recommendations.get(logo_type, 'Use default parameters')}")

if __name__ == "__main__":
    classify_logo()
```

### Example 3: Parameter Optimization

```python
#!/usr/bin/env python3
"""Parameter optimization example"""

from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

def optimize_parameters():
    # Initialize components
    extractor = ImageFeatureExtractor()
    optimizer = FeatureMappingOptimizer()

    # Process image
    image_path = "data/logos/gradient/sunset_logo.png"

    # Extract features
    print("🔍 Extracting features...")
    features = extractor.extract_features(image_path)

    # Optimize parameters
    print("⚙️  Optimizing parameters...")
    parameters = optimizer.optimize(features)

    print("Optimized VTracer Parameters:")
    for param, value in parameters.items():
        print(f"  {param}: {value}")

    # Show parameter explanations
    explanations = {
        'color_precision': 'Number of colors in quantization',
        'corner_threshold': 'Sensitivity of corner detection',
        'length_threshold': 'Minimum path segment length',
        'splice_threshold': 'Path merging sensitivity',
        'filter_speckle': 'Size threshold for noise removal'
    }

    print("\nParameter Explanations:")
    for param, value in parameters.items():
        if param in explanations:
            print(f"  {param} ({value}): {explanations[param]}")

if __name__ == "__main__":
    optimize_parameters()
```

### Example 4: Quality Prediction

```python
#!/usr/bin/env python3
"""Quality prediction example"""

from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
from backend.ai_modules.prediction.quality_predictor import QualityPredictor

def predict_quality():
    # Initialize components
    extractor = ImageFeatureExtractor()
    optimizer = FeatureMappingOptimizer()
    predictor = QualityPredictor()

    # Process image
    image_path = "data/logos/complex/detailed_logo.png"

    # Extract features and optimize parameters
    features = extractor.extract_features(image_path)
    parameters = optimizer.optimize(features)

    # Predict quality
    predicted_quality = predictor.predict_quality(image_path, parameters)

    print(f"Quality Prediction:")
    print(f"  Predicted SSIM: {predicted_quality:.3f}")

    # Provide quality assessment
    if predicted_quality >= 0.95:
        quality_level = "Excellent"
        emoji = "🏆"
    elif predicted_quality >= 0.85:
        quality_level = "Good"
        emoji = "✅"
    elif predicted_quality >= 0.70:
        quality_level = "Acceptable"
        emoji = "⚠️"
    else:
        quality_level = "Poor"
        emoji = "❌"

    print(f"  Quality Level: {emoji} {quality_level}")

    # Suggest improvements if needed
    if predicted_quality < 0.85:
        print("\n💡 Suggestions for improvement:")
        if features.get('complexity_score', 0) > 0.7:
            print("  • Try increasing max_iterations for complex images")
        if features.get('unique_colors', 0) > 20:
            print("  • Consider increasing color_precision for colorful images")
        if features.get('edge_density', 0) > 0.5:
            print("  • Adjust corner_threshold for edge-heavy images")

if __name__ == "__main__":
    predict_quality()
```

## Advanced Usage Examples

### Example 5: Complete AI Pipeline

```python
#!/usr/bin/env python3
"""Complete AI pipeline example"""

from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
from backend.ai_modules.prediction.quality_predictor import QualityPredictor
from backend.ai_modules.utils.performance_monitor import performance_monitor
import time

class AIEnhancedConverter:
    """Example AI-enhanced converter implementation"""

    def __init__(self):
        self.extractor = ImageFeatureExtractor()
        self.classifier = RuleBasedClassifier()
        self.optimizer = FeatureMappingOptimizer()
        self.predictor = QualityPredictor()

    @performance_monitor.time_operation("complete_ai_pipeline")
    def process_image(self, image_path: str):
        """Process image through complete AI pipeline"""

        print(f"🚀 Processing: {image_path}")

        # Phase 1: Feature extraction
        print("  📊 Extracting features...")
        features = self.extractor.extract_features(image_path)

        # Phase 2: Classification
        print("  🏷️  Classifying logo type...")
        logo_type, confidence = self.classifier.classify(features)

        # Phase 3: Parameter optimization
        print("  ⚙️  Optimizing parameters...")
        parameters = self.optimizer.optimize(features)

        # Phase 4: Quality prediction
        print("  🎯 Predicting quality...")
        predicted_quality = self.predictor.predict_quality(image_path, parameters)

        # Compile results
        results = {
            'features': features,
            'classification': {
                'type': logo_type,
                'confidence': confidence
            },
            'parameters': parameters,
            'predicted_quality': predicted_quality,
            'processing_time': time.time()
        }

        return results

    def batch_process(self, image_paths: list):
        """Process multiple images"""
        results = []

        print(f"📦 Batch processing {len(image_paths)} images...")

        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}]")
            try:
                result = self.process_image(image_path)
                result['success'] = True
                results.append(result)
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'image_path': image_path
                })

        # Summary
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n📈 Batch Summary:")
        print(f"  Processed: {successful}/{len(image_paths)} images")
        print(f"  Success Rate: {successful/len(image_paths):.1%}")

        return results

def main():
    """Main example function"""
    converter = AIEnhancedConverter()

    # Single image processing
    print("=== Single Image Processing ===")
    result = converter.process_image("data/logos/simple_geometric/circle_00.png")

    print(f"\nResults:")
    print(f"  Logo Type: {result['classification']['type']} ({result['classification']['confidence']:.1%})")
    print(f"  Predicted Quality: {result['predicted_quality']:.3f}")
    print(f"  Key Parameters: color_precision={result['parameters']['color_precision']}, corner_threshold={result['parameters']['corner_threshold']}")

    # Batch processing example
    print("\n=== Batch Processing ===")
    test_images = [
        "tests/data/simple/simple_logo_0.png",
        "tests/data/text/text_logo_0.png",
        "tests/data/gradient/gradient_logo_0.png"
    ]

    batch_results = converter.batch_process(test_images)

    # Performance summary
    summary = performance_monitor.get_summary("complete_ai_pipeline")
    if summary:
        print(f"\n⏱️  Performance Summary:")
        print(f"  Average Time: {summary['average_duration']:.3f}s")
        print(f"  Memory Usage: +{summary['average_memory_delta']:.1f}MB average")

if __name__ == "__main__":
    main()
```

### Example 6: Concurrent Processing

```python
#!/usr/bin/env python3
"""Concurrent processing example"""

import concurrent.futures
import time
from typing import List, Dict, Any
from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier

def process_single_image(image_path: str) -> Dict[str, Any]:
    """Process a single image (thread-safe)"""
    # Create new instances for thread safety
    extractor = ImageFeatureExtractor()
    classifier = RuleBasedClassifier()

    try:
        start_time = time.time()

        # Extract features
        features = extractor.extract_features(image_path)

        # Classify
        logo_type, confidence = classifier.classify(features)

        processing_time = time.time() - start_time

        return {
            'image_path': image_path,
            'success': True,
            'logo_type': logo_type,
            'confidence': confidence,
            'features': features,
            'processing_time': processing_time
        }

    except Exception as e:
        return {
            'image_path': image_path,
            'success': False,
            'error': str(e)
        }

def concurrent_processing_example():
    """Example of concurrent image processing"""

    # Test images
    image_paths = [
        "tests/data/simple/simple_logo_0.png",
        "tests/data/simple/simple_logo_1.png",
        "tests/data/text/text_logo_0.png",
        "tests/data/text/text_logo_1.png",
        "tests/data/gradient/gradient_logo_0.png",
        "tests/data/complex/complex_logo_0.png"
    ]

    print(f"🔄 Processing {len(image_paths)} images concurrently...")

    # Sequential processing (for comparison)
    print("\n📏 Sequential Processing:")
    start_time = time.time()
    sequential_results = [process_single_image(path) for path in image_paths]
    sequential_time = time.time() - start_time
    print(f"  Time: {sequential_time:.2f}s")

    # Concurrent processing
    print("\n⚡ Concurrent Processing:")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_image, path) for path in image_paths]

        # Collect results as they complete
        concurrent_results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            concurrent_results.append(result)
            if result['success']:
                print(f"  ✅ {result['image_path']}: {result['logo_type']} ({result['processing_time']:.3f}s)")
            else:
                print(f"  ❌ {result['image_path']}: {result['error']}")

    concurrent_time = time.time() - start_time
    print(f"  Total Time: {concurrent_time:.2f}s")

    # Performance comparison
    speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1
    print(f"\n📊 Performance Comparison:")
    print(f"  Sequential: {sequential_time:.2f}s")
    print(f"  Concurrent: {concurrent_time:.2f}s")
    print(f"  Speedup: {speedup:.1f}x")

    # Success statistics
    successful = sum(1 for r in concurrent_results if r['success'])
    print(f"  Success Rate: {successful}/{len(image_paths)} ({successful/len(image_paths):.1%})")

if __name__ == "__main__":
    concurrent_processing_example()
```

### Example 7: Performance Monitoring

```python
#!/usr/bin/env python3
"""Performance monitoring example"""

from backend.ai_modules.utils.performance_monitor import performance_monitor, PerformanceMonitor
from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
import time
import random

# Custom performance monitor instance
custom_monitor = PerformanceMonitor()

@custom_monitor.time_operation("custom_feature_extraction")
def monitored_feature_extraction(image_path: str):
    """Feature extraction with custom monitoring"""
    extractor = ImageFeatureExtractor()

    # Simulate some processing
    time.sleep(random.uniform(0.1, 0.3))

    return extractor.extract_features(image_path)

def performance_monitoring_example():
    """Example of performance monitoring usage"""

    print("📊 Performance Monitoring Example")
    print("=" * 40)

    # Test images
    test_images = [
        "tests/data/simple/simple_logo_0.png",
        "tests/data/text/text_logo_0.png",
        "tests/data/gradient/gradient_logo_0.png"
    ]

    # Process images with monitoring
    print("\n🔍 Processing images with monitoring...")

    for i, image_path in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] Processing: {image_path}")

        # Use global monitor
        @performance_monitor.time_operation(f"image_{i}_processing")
        def process_image():
            return monitored_feature_extraction(image_path)

        try:
            features = process_image()
            print(f"  ✅ Extracted {len(features)} features")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Get performance summaries
    print("\n📈 Performance Summary (Global Monitor):")
    for operation_name in ["image_1_processing", "image_2_processing", "image_3_processing"]:
        summary = performance_monitor.get_summary(operation_name)
        if summary and summary.get('total_operations', 0) > 0:
            print(f"  {operation_name}:")
            print(f"    Duration: {summary['average_duration']:.3f}s")
            print(f"    Memory: +{summary['average_memory_delta']:.1f}MB")

    print("\n📈 Performance Summary (Custom Monitor):")
    custom_summary = custom_monitor.get_summary("custom_feature_extraction")
    if custom_summary and custom_summary.get('total_operations', 0) > 0:
        print(f"  Custom Feature Extraction:")
        print(f"    Operations: {custom_summary['total_operations']}")
        print(f"    Success Rate: {custom_summary['successful_operations']}/{custom_summary['total_operations']}")
        print(f"    Avg Duration: {custom_summary['average_duration']:.3f}s")
        print(f"    Max Duration: {custom_summary['max_duration']:.3f}s")
        print(f"    Avg Memory: +{custom_summary['average_memory_delta']:.1f}MB")
        print(f"    Max Memory: +{custom_summary['max_memory_delta']:.1f}MB")

    # Overall performance summary
    overall_summary = performance_monitor.get_summary()
    if overall_summary and overall_summary.get('total_operations', 0) > 0:
        print(f"\n🎯 Overall Performance (All Operations):")
        print(f"  Total Operations: {overall_summary['total_operations']}")
        print(f"  Success Rate: {overall_summary['successful_operations']}/{overall_summary['total_operations']}")
        print(f"  Average Duration: {overall_summary['average_duration']:.3f}s")
        print(f"  Average Memory Delta: {overall_summary['average_memory_delta']:.1f}MB")

if __name__ == "__main__":
    performance_monitoring_example()
```

## Integration Examples

### Example 8: Web API Integration

```python
#!/usr/bin/env python3
"""Example of AI integration with web API"""

from flask import Flask, request, jsonify
import tempfile
import os
from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

app = Flask(__name__)

# Initialize AI components once
extractor = ImageFeatureExtractor()
classifier = RuleBasedClassifier()
optimizer = FeatureMappingOptimizer()

@app.route('/api/ai/analyze', methods=['POST'])
def analyze_image():
    """API endpoint for AI analysis"""
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            image_file.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            # Extract features
            features = extractor.extract_features(temp_path)

            # Classify logo type
            logo_type, confidence = classifier.classify(features)

            # Optimize parameters
            parameters = optimizer.optimize(features)

            # Return results
            return jsonify({
                'success': True,
                'analysis': {
                    'features': features,
                    'classification': {
                        'type': logo_type,
                        'confidence': confidence
                    },
                    'recommended_parameters': parameters
                }
            })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Quick validation that AI modules are working
        test_features = {
            'complexity_score': 0.5,
            'unique_colors': 10,
            'edge_density': 0.3,
            'aspect_ratio': 1.0,
            'fill_ratio': 0.4,
            'entropy': 6.0,
            'corner_density': 0.02,
            'gradient_strength': 20.0
        }

        # Test classification
        logo_type, confidence = classifier.classify(test_features)

        # Test optimization
        parameters = optimizer.optimize(test_features)

        return jsonify({
            'status': 'healthy',
            'components': {
                'feature_extractor': 'ok',
                'classifier': 'ok',
                'optimizer': 'ok'
            },
            'test_results': {
                'classification': {'type': logo_type, 'confidence': confidence},
                'optimization': len(parameters) > 0
            }
        })

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting AI-enhanced API server...")
    app.run(debug=True, host='0.0.0.0', port=5001)
```

### Example 9: Testing and Validation

```python
#!/usr/bin/env python3
"""Example testing and validation script"""

import unittest
import tempfile
import numpy as np
import cv2
from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

class TestAIModulesIntegration(unittest.TestCase):
    """Integration tests for AI modules"""

    def setUp(self):
        """Set up test fixtures"""
        self.extractor = ImageFeatureExtractor()
        self.classifier = RuleBasedClassifier()
        self.optimizer = FeatureMappingOptimizer()

        # Create test image
        self.test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        self.test_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(self.test_file.name, self.test_image)

    def test_feature_extraction(self):
        """Test feature extraction"""
        features = self.extractor.extract_features(self.test_file.name)

        # Validate feature structure
        expected_features = ['complexity_score', 'unique_colors', 'edge_density',
                           'aspect_ratio', 'fill_ratio', 'entropy',
                           'corner_density', 'gradient_strength']

        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))

        # Validate feature ranges
        self.assertGreaterEqual(features['complexity_score'], 0)
        self.assertLessEqual(features['complexity_score'], 1)
        self.assertGreater(features['unique_colors'], 0)
        self.assertGreaterEqual(features['edge_density'], 0)
        self.assertLessEqual(features['edge_density'], 1)

    def test_classification(self):
        """Test logo classification"""
        features = self.extractor.extract_features(self.test_file.name)
        logo_type, confidence = self.classifier.classify(features)

        # Validate classification results
        valid_types = ['simple', 'text', 'gradient', 'complex']
        self.assertIn(logo_type, valid_types)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_optimization(self):
        """Test parameter optimization"""
        features = self.extractor.extract_features(self.test_file.name)
        parameters = self.optimizer.optimize(features)

        # Validate parameter structure
        expected_params = ['color_precision', 'corner_threshold', 'length_threshold',
                          'splice_threshold', 'filter_speckle', 'color_tolerance',
                          'layer_difference']

        for param in expected_params:
            self.assertIn(param, parameters)
            self.assertIsInstance(parameters[param], (int, float))

        # Validate parameter ranges
        self.assertGreaterEqual(parameters['color_precision'], 1)
        self.assertLessEqual(parameters['color_precision'], 16)
        self.assertGreaterEqual(parameters['corner_threshold'], 10)
        self.assertLessEqual(parameters['corner_threshold'], 100)

    def test_complete_pipeline(self):
        """Test complete AI pipeline"""
        # Extract features
        features = self.extractor.extract_features(self.test_file.name)
        self.assertIsInstance(features, dict)

        # Classify
        logo_type, confidence = self.classifier.classify(features)
        self.assertIsInstance(logo_type, str)
        self.assertIsInstance(confidence, float)

        # Optimize
        parameters = self.optimizer.optimize(features)
        self.assertIsInstance(parameters, dict)

        # Validate pipeline consistency
        self.assertGreater(len(features), 0)
        self.assertGreater(len(parameters), 0)
        self.assertIn(logo_type, ['simple', 'text', 'gradient', 'complex'])

    def tearDown(self):
        """Clean up test fixtures"""
        import os
        if os.path.exists(self.test_file.name):
            os.unlink(self.test_file.name)

def run_validation_suite():
    """Run complete validation suite"""
    print("🧪 Running AI Modules Validation Suite")
    print("=" * 45)

    # Run unit tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAIModulesIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    if result.wasSuccessful():
        print("\n✅ All validation tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return False

if __name__ == '__main__':
    success = run_validation_suite()
    exit(0 if success else 1)
```

## Best Practices

### 1. Error Handling

```python
try:
    features = extractor.extract_features(image_path)
except FileNotFoundError:
    print(f"Image file not found: {image_path}")
    # Use default features or skip image
except Exception as e:
    print(f"Feature extraction failed: {e}")
    # Log error and continue with fallback
```

### 2. Resource Management

```python
# Clear caches periodically
extractor.clear_cache()

# Use context managers for temporary files
with tempfile.NamedTemporaryFile(suffix='.png') as tmp_file:
    # Process file
    pass  # File automatically cleaned up
```

### 3. Performance Optimization

```python
# Batch processing for multiple images
features_batch = []
for image_path in image_paths[:100]:  # Process in batches
    features = extractor.extract_features(image_path)
    features_batch.append(features)

# Process batch together
results = optimizer.optimize_batch(features_batch)
```

### 4. Configuration Management

```python
from backend.ai_modules.config import MODEL_CONFIG, PERFORMANCE_TARGETS

# Use configuration for consistent behavior
if processing_time > PERFORMANCE_TARGETS['tier_1']['max_time']:
    print("Performance target exceeded, consider optimization")
```

These examples demonstrate practical usage patterns for all AI modules and provide a foundation for building AI-enhanced applications.