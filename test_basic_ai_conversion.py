#!/usr/bin/env python3
"""
Test Basic AI Conversion Workflow
Tests the end-to-end AI conversion pipeline with working components
"""

import sys
import time
import tempfile
from pathlib import Path

def test_basic_workflow():
    """Test basic AI conversion workflow"""
    print("🧪 Testing Basic AI Conversion Workflow")
    print("=" * 50)

    try:
        # Step 1: Test AI Enhanced Converter
        print("\n1. Testing AI Enhanced Converter...")
        from backend.converters.ai_enhanced_converter import AIEnhancedConverter

        converter = AIEnhancedConverter()
        print("✅ AI Enhanced Converter instantiated")
        print(f"   Converter name: {converter.name}")
        print(f"   Available methods: {hasattr(converter, 'convert')}")

        # Step 2: Test Feature Extraction
        print("\n2. Testing Feature Extraction...")
        from backend.ai_modules.feature_extraction import ImageFeatureExtractor

        extractor = ImageFeatureExtractor()
        print("✅ Feature Extractor ready")

        # Step 3: Test Classification
        print("\n3. Testing Logo Classification...")
        from backend.ai_modules.classification.hybrid_classifier import HybridClassifier

        classifier = HybridClassifier()
        print("✅ Logo Classifier ready")

        # Step 4: Test Method 1 Optimizer
        print("\n4. Testing Method 1 Optimizer...")
        from backend.ai_modules.optimization.feature_mapping_optimizer import FeatureMappingOptimizer

        optimizer = FeatureMappingOptimizer()
        print("✅ Method 1 Optimizer ready")

        # Step 5: Test Basic Router
        print("\n5. Testing Basic Router...")
        from backend.ai_modules.optimization.intelligent_router import IntelligentRouter

        router = IntelligentRouter()
        print("✅ Basic Router ready")
        print(f"   Available methods: {list(router.available_methods.keys())}")

        # Step 6: Test Quality Predictor
        print("\n6. Testing Quality Predictor...")
        from backend.ai_modules.prediction.quality_predictor import QualityPredictor

        predictor = QualityPredictor()
        print("✅ Quality Predictor ready")

        print("\n🎉 All core components are functional!")
        return True

    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_sample_image():
    """Test conversion with a sample image"""
    print("\n🖼️  Testing with Sample Image")
    print("=" * 40)

    try:
        # Create a simple test PNG file
        import numpy as np
        from PIL import Image

        # Create test image directory
        test_dir = Path("/tmp/claude/test_images")
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create a simple test image (50x50 red circle)
        img_size = 50
        img_array = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Draw a simple circle
        center = img_size // 2
        radius = 20
        for y in range(img_size):
            for x in range(img_size):
                if (x - center) ** 2 + (y - center) ** 2 <= radius ** 2:
                    img_array[y, x] = [255, 0, 0]  # Red

        # Save as PNG
        test_image_path = test_dir / "test_circle.png"
        Image.fromarray(img_array).save(test_image_path)
        print(f"✅ Created test image: {test_image_path}")

        # Test feature extraction on real image
        print("\n1. Testing feature extraction on image...")
        from backend.ai_modules.feature_extraction import ImageFeatureExtractor

        extractor = ImageFeatureExtractor()
        features = extractor.extract_features(str(test_image_path))
        print("✅ Features extracted successfully")
        print(f"   Features: {list(features.keys())}")
        print(f"   Sample values: {dict(list(features.items())[:3])}")

        # Test classification on real image
        print("\n2. Testing classification on image...")
        from backend.ai_modules.classification.hybrid_classifier import HybridClassifier

        classifier = HybridClassifier()
        classification_result = classifier.classify(str(test_image_path))
        print("✅ Classification successful")
        print(f"   Logo type: {classification_result.get('logo_type', 'unknown')}")
        print(f"   Confidence: {classification_result.get('confidence', 0.0):.3f}")
        print(f"   Method used: {classification_result.get('method_used', 'unknown')}")

        # Extract for later use
        logo_type = classification_result.get('logo_type', 'unknown')
        confidence = classification_result.get('confidence', 0.0)

        # Test Method 1 optimization
        print("\n3. Testing Method 1 parameter optimization...")
        from backend.ai_modules.optimization.feature_mapping_optimizer import FeatureMappingOptimizer

        optimizer = FeatureMappingOptimizer()
        optimized_params = optimizer.optimize(features)
        print("✅ Parameter optimization successful")
        print(f"   Optimized parameters: {optimized_params}")

        # Test Basic Router decision
        print("\n4. Testing routing decision...")
        from backend.ai_modules.optimization.intelligent_router import IntelligentRouter

        router = IntelligentRouter()
        decision = router.route_optimization(
            image_path=str(test_image_path),
            features=features
        )
        print("✅ Routing decision successful")
        print(f"   Selected method: {decision.primary_method}")
        print(f"   Confidence: {decision.confidence:.3f}")
        print(f"   Reasoning: {decision.reasoning}")

        print("\n🎉 Sample image testing successful!")
        return True, test_image_path

    except Exception as e:
        print(f"❌ Sample image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_ai_enhanced_conversion():
    """Test the complete AI enhanced conversion"""
    print("\n🔄 Testing Complete AI Enhanced Conversion")
    print("=" * 50)

    try:
        # Get the test image from previous step
        success, test_image_path = test_with_sample_image()
        if not success:
            print("❌ Cannot proceed without test image")
            return False

        print("\n5. Testing complete AI conversion...")
        from backend.converters.ai_enhanced_converter import AIEnhancedConverter

        converter = AIEnhancedConverter()

        # Test conversion
        print(f"   Converting: {test_image_path}")
        start_time = time.time()

        svg_content = converter.convert(str(test_image_path))
        conversion_time = time.time() - start_time

        print("✅ AI Enhanced Conversion successful")
        print(f"   Conversion time: {conversion_time:.2f}s")
        print(f"   SVG content length: {len(svg_content)} characters")
        print(f"   SVG starts with: {svg_content[:100]}...")

        # Save result
        output_path = test_image_path.parent / "test_circle_ai_converted.svg"
        with open(output_path, 'w') as f:
            f.write(svg_content)
        print(f"   Saved to: {output_path}")

        print("\n🎉 Complete AI Enhanced Conversion SUCCESSFUL!")
        return True

    except Exception as e:
        print(f"❌ AI Enhanced Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Basic AI Conversion Workflow Tests")
    print(f"📅 Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    # Test 1: Basic component workflow
    workflow_ok = test_basic_workflow()

    # Test 2: Complete AI conversion
    if workflow_ok:
        conversion_ok = test_ai_enhanced_conversion()
    else:
        conversion_ok = False

    # Results
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    print(f"✅ Basic Workflow: {'PASSED' if workflow_ok else 'FAILED'}")
    print(f"✅ AI Conversion: {'PASSED' if conversion_ok else 'FAILED'}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")

    if workflow_ok and conversion_ok:
        print("\n🎉 ALL TESTS PASSED - Basic AI Conversion is WORKING!")
        print("\n🎯 Next Steps:")
        print("  1. Test with real logo images")
        print("  2. Measure quality improvements")
        print("  3. Test other optimization methods")
        print("  4. Integrate with API")
        return True
    else:
        print("\n⚠️  TESTS FAILED - Issues need to be resolved")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)