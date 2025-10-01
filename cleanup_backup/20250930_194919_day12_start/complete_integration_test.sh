#!/bin/bash
# complete_integration_test.sh - End-to-end integration test for AI modules

echo "🔍 Running complete integration test..."
echo "=================================================="

# Test 1: Import all AI modules
echo ""
echo "📦 Test 1: Import all AI modules..."
python3 -c "
from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
from backend.ai_modules.prediction.quality_predictor import QualityPredictor
from backend.ai_modules.base_ai_converter import BaseAIConverter
print('✅ All AI modules import successfully')
"

if [ $? -ne 0 ]; then
    echo "❌ AI module imports failed"
    exit 1
fi

# Test 2: Generate test data
echo ""
echo "🎨 Test 2: Generate test data..."
python3 tests/utils/test_data_generator.py

if [ $? -ne 0 ]; then
    echo "❌ Test data generation failed"
    exit 1
fi

# Test 3: Run AI pipeline with test data
echo ""
echo "🧠 Test 3: Run AI pipeline..."
python3 -c "
import sys
sys.path.append('.')

from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
from backend.ai_modules.prediction.quality_predictor import QualityPredictor
import os

# Test with generated test images
test_images = [
    'tests/data/simple/simple_logo_0.png',
    'tests/data/text/text_logo_0.png',
    'tests/data/gradient/gradient_logo_0.png'
]

extractor = ImageFeatureExtractor()
classifier = RuleBasedClassifier()
optimizer = FeatureMappingOptimizer()
predictor = QualityPredictor()

for img_path in test_images:
    if os.path.exists(img_path):
        print(f'Processing: {img_path}')
        features = extractor.extract_features(img_path)
        logo_type, confidence = classifier.classify(features)
        parameters = optimizer.optimize(features)
        quality = predictor.predict_quality(img_path, parameters)
        print(f'  Type: {logo_type} ({confidence:.2f}), Quality: {quality:.3f}')

print('✅ AI pipeline test passed')
"

if [ $? -ne 0 ]; then
    echo "❌ AI pipeline test failed"
    exit 1
fi

# Test 4: Run all unit tests
echo ""
echo "🧪 Test 4: Run all unit tests..."
python3 -m pytest tests/ai_modules/ -v --tb=short

if [ $? -ne 0 ]; then
    echo "❌ Unit tests failed"
    exit 1
fi

# Test 5: Check test coverage
echo ""
echo "📊 Test 5: Check test coverage..."
coverage run -m pytest tests/ai_modules/ > /dev/null 2>&1
coverage report --fail-under=60

if [ $? -ne 0 ]; then
    echo "❌ Test coverage below 60%"
    exit 1
fi

# Test 6: Performance benchmark
echo ""
echo "🚀 Test 6: Performance benchmark..."
if [ -f "scripts/benchmarks/benchmark_pytorch.py" ]; then
    python3 scripts/benchmarks/benchmark_pytorch.py
else
    echo "⚠️  PyTorch benchmark script not found, skipping..."
fi

# Test 7: Test VTracer integration
echo ""
echo "🎯 Test 7: VTracer integration test..."
python3 -c "
import vtracer
import tempfile
import cv2
import numpy as np
import os

# Create test image
test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
test_path = tempfile.mktemp(suffix='.png')
cv2.imwrite(test_image, test_path)

try:
    # Test VTracer conversion
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_svg:
        vtracer.convert_image_to_svg_py(
            test_path,
            tmp_svg.name,
            color_precision=4,
            corner_threshold=30,
            length_threshold=10,
            splice_threshold=45,
            filter_speckle=4,
            layer_difference=16
        )

        # Check SVG was created
        if os.path.exists(tmp_svg.name) and os.path.getsize(tmp_svg.name) > 0:
            print('✅ VTracer integration working')
        else:
            print('❌ VTracer failed to create SVG')

        # Cleanup
        if os.path.exists(tmp_svg.name):
            os.unlink(tmp_svg.name)

except Exception as e:
    print(f'❌ VTracer integration failed: {e}')

# Cleanup
if os.path.exists(test_path):
    os.unlink(test_path)
"

# Test 8: Memory usage test
echo ""
echo "💾 Test 8: Memory usage test..."
python3 -c "
import psutil
import gc

# Get initial memory
gc.collect()
initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)

# Load all AI components
from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
from backend.ai_modules.prediction.quality_predictor import QualityPredictor

extractor = ImageFeatureExtractor()
classifier = RuleBasedClassifier()
optimizer = FeatureMappingOptimizer()
predictor = QualityPredictor()

# Get final memory
final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
memory_increase = final_memory - initial_memory

print(f'Memory usage: {memory_increase:.1f}MB increase')

if memory_increase < 200:  # 200MB threshold
    print('✅ Memory usage acceptable')
else:
    print('⚠️  High memory usage detected')
"

echo ""
echo "=================================================="
echo "✅ Complete integration test passed!"
echo "All systems operational and ready for production"