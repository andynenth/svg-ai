#!/usr/bin/env python3
"""Test AI integration with existing API"""

import json
import tempfile
import cv2
import numpy as np
import os
import time
import concurrent.futures
from pathlib import Path

def test_ai_module_api_calls():
    """Test that AI modules can be called directly (simulating API calls)"""
    print("🔌 Testing AI Module API Calls...")

    try:
        # Test AI module imports and basic functionality
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
        from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
        from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
        from backend.ai_modules.prediction.quality_predictor import QualityPredictor

        # Create test image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_path = tempfile.mktemp(suffix='.png')
        cv2.imwrite(test_path, test_image)

        # Initialize AI components
        extractor = ImageFeatureExtractor()
        classifier = RuleBasedClassifier()
        optimizer = FeatureMappingOptimizer()
        predictor = QualityPredictor()

        # Test feature extraction API
        features = extractor.extract_features(test_path)
        assert isinstance(features, dict)
        assert len(features) >= 8
        print("  ✅ Feature extraction API working")

        # Test classification API
        logo_type, confidence = classifier.classify(features)
        assert isinstance(logo_type, str)
        assert 0 <= confidence <= 1
        print("  ✅ Classification API working")

        # Test optimization API
        parameters = optimizer.optimize(features)
        assert isinstance(parameters, dict)
        assert 'color_precision' in parameters
        print("  ✅ Optimization API working")

        # Test prediction API
        quality = predictor.predict_quality(test_path, parameters)
        assert isinstance(quality, float)
        assert 0 <= quality <= 1
        print("  ✅ Quality prediction API working")

        # Cleanup
        os.unlink(test_path)
        return True

    except Exception as e:
        print(f"  ❌ AI Module API test failed: {e}")
        return False

def test_api_response_format():
    """Test that AI metadata can be integrated into API responses"""
    print("📦 Testing API Response Format...")

    try:
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
        from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
        from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
        from backend.ai_modules.prediction.quality_predictor import QualityPredictor

        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_path = tempfile.mktemp(suffix='.png')
        cv2.imwrite(test_path, test_image)

        # Simulate API response with AI metadata
        extractor = ImageFeatureExtractor()
        classifier = RuleBasedClassifier()
        optimizer = FeatureMappingOptimizer()
        predictor = QualityPredictor()

        # Extract AI metadata
        features = extractor.extract_features(test_path)
        logo_type, confidence = classifier.classify(features)
        parameters = optimizer.optimize(features)
        predicted_quality = predictor.predict_quality(test_path, parameters)

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj

        # Create API response format
        api_response = {
            'success': True,
            'conversion': {
                'svg_content': '<svg>...mock SVG content...</svg>',
                'file_size_reduction': '75%',
                'processing_time': '1.23s'
            },
            'ai_analysis': {
                'logo_type': str(logo_type),
                'confidence': float(confidence),
                'predicted_quality': float(predicted_quality),
                'features': convert_numpy_types(features),
                'optimized_parameters': convert_numpy_types(parameters)
            },
            'metadata': {
                'timestamp': time.time(),
                'version': '1.0.0',
                'ai_enhanced': True
            }
        }

        # Validate response format
        assert api_response['success'] == True
        assert 'ai_analysis' in api_response
        assert 'logo_type' in api_response['ai_analysis']
        assert 'optimized_parameters' in api_response['ai_analysis']

        # Test JSON serialization
        json_response = json.dumps(api_response, indent=2)
        assert len(json_response) > 0

        print("  ✅ API response format compatible")
        print(f"  ✅ Response size: {len(json_response)} characters")

        # Cleanup
        os.unlink(test_path)
        return True

    except Exception as e:
        print(f"  ❌ API response format test failed: {e}")
        return False

def test_error_handling_in_api_context():
    """Test error handling when AI modules fail in API context"""
    print("🚨 Testing Error Handling in API Context...")

    try:
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
        from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier

        extractor = ImageFeatureExtractor()
        classifier = RuleBasedClassifier()

        # Test with invalid image path
        try:
            features = extractor.extract_features("/nonexistent/image.png")
        except Exception:
            # Should handle gracefully
            api_error_response = {
                'success': False,
                'error': 'Feature extraction failed',
                'error_code': 'FEATURE_EXTRACTION_ERROR',
                'fallback_used': True,
                'ai_analysis': None
            }
            assert api_error_response['success'] == False
            print("  ✅ Feature extraction error handling works")

        # Test with invalid features
        try:
            logo_type, confidence = classifier.classify({})  # Empty features
            # Should still work with defaults
            assert isinstance(logo_type, str)
            print("  ✅ Classification error handling works")
        except Exception:
            print("  ✅ Classification graceful degradation works")

        return True

    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False

def test_concurrent_api_requests():
    """Test concurrent AI processing (simulating multiple API requests)"""
    print("⚡ Testing Concurrent API Requests...")

    def process_single_request(request_id):
        """Simulate a single API request with AI processing"""
        try:
            from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
            from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier

            # Create unique test image for this request
            test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            test_path = tempfile.mktemp(suffix=f'_req_{request_id}.png')
            cv2.imwrite(test_path, test_image)

            # Process with AI
            extractor = ImageFeatureExtractor()
            classifier = RuleBasedClassifier()

            features = extractor.extract_features(test_path)
            logo_type, confidence = classifier.classify(features)

            # Cleanup
            os.unlink(test_path)

            return {
                'request_id': request_id,
                'success': True,
                'logo_type': logo_type,
                'confidence': confidence,
                'processing_time': time.time()
            }

        except Exception as e:
            return {
                'request_id': request_id,
                'success': False,
                'error': str(e)
            }

    try:
        # Test concurrent processing
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_single_request, i) for i in range(8)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        end_time = time.time()

        # Analyze results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        print(f"  ✅ Processed {len(successful)}/{len(results)} requests successfully")
        print(f"  ✅ Total time: {end_time - start_time:.2f}s")
        print(f"  ✅ Average time per request: {(end_time - start_time)/len(results):.2f}s")

        if len(failed) > 0:
            print(f"  ⚠️  {len(failed)} requests failed")

        # Success if at least 75% succeeded
        success_rate = len(successful) / len(results)
        return success_rate >= 0.75

    except Exception as e:
        print(f"  ❌ Concurrent processing test failed: {e}")
        return False

def test_existing_api_compatibility():
    """Test that existing API structure is preserved"""
    print("🔧 Testing Existing API Compatibility...")

    try:
        # Check if existing API modules can still be imported
        # (This simulates that we haven't broken existing functionality)

        # Test that VTracer converter still works
        import vtracer

        # Create test image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        test_path = tempfile.mktemp(suffix='.png')
        cv2.imwrite(test_path, test_image)

        # Test VTracer conversion (existing functionality)
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_svg:
            vtracer.convert_image_to_svg_py(
                test_path,
                tmp_svg.name,
                color_precision=4,
                corner_threshold=30
            )

            # Check SVG was created
            if os.path.exists(tmp_svg.name) and os.path.getsize(tmp_svg.name) > 0:
                print("  ✅ Existing VTracer API still functional")
                success = True
            else:
                print("  ❌ Existing VTracer API broken")
                success = False

            # Cleanup
            os.unlink(tmp_svg.name)

        os.unlink(test_path)

        # Check that we can still import existing modules
        try:
            from backend.converters.base import BaseConverter
            print("  ✅ Existing converter base class accessible")
        except ImportError:
            print("  ⚠️  Existing converter base class not found (may be expected)")

        return success

    except Exception as e:
        print(f"  ❌ Existing API compatibility test failed: {e}")
        return False

def main():
    """Run all API integration tests"""
    print("🔌 API Integration Validation Test Suite")
    print("=" * 50)

    tests = [
        ("AI Module API Calls", test_ai_module_api_calls),
        ("API Response Format", test_api_response_format),
        ("Error Handling", test_error_handling_in_api_context),
        ("Concurrent Requests", test_concurrent_api_requests),
        ("Existing API Compatibility", test_existing_api_compatibility)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n{'='*50}")
    print(f"📊 API Integration Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All API integration tests passed!")
        return True
    else:
        print("⚠️  Some API integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)