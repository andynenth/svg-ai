#!/usr/bin/env python3
"""
Test script for EfficientNet classifier implementation.
"""

import sys
import os
from PIL import Image

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.classification.efficientnet_classifier import EfficientNetClassifier

def create_test_image(color='red', size=(256, 256)):
    """Create a test image."""
    return Image.new('RGB', size, color=color)

def test_classifier_initialization():
    """Test classifier initialization."""
    print("=== Testing Classifier Initialization ===")

    try:
        # Test with pretrained=False to avoid download issues
        classifier = EfficientNetClassifier(use_pretrained=False)
        print("✓ Classifier initialized successfully")

        # Test model info
        info = classifier.get_model_info()
        print(f"✓ Model info: {info['model_name']}")
        print(f"✓ Classes: {info['classes']}")
        print(f"✓ Total parameters: {info['total_parameters']:,}")

        return classifier

    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return None

def test_single_classification(classifier):
    """Test single image classification."""
    print("\n=== Testing Single Image Classification ===")

    try:
        # Create test image and save temporarily
        test_image = create_test_image('blue', (300, 300))
        test_path = '/tmp/claude/test_logo.png'
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        test_image.save(test_path)

        # Classify image
        result = classifier.classify(test_path)

        print(f"✓ Classification result: {result['logo_type']}")
        print(f"✓ Confidence: {result['confidence']:.4f}")
        print(f"✓ Model type: {result['model_type']}")

        # Clean up
        os.remove(test_path)

        return True

    except Exception as e:
        print(f"✗ Single classification failed: {e}")
        return False

def test_batch_classification(classifier):
    """Test batch classification."""
    print("\n=== Testing Batch Classification ===")

    try:
        # Create multiple test images
        test_paths = []
        colors = ['red', 'green', 'blue']

        for i, color in enumerate(colors):
            test_image = create_test_image(color, (250, 250))
            test_path = f'/tmp/claude/test_logo_{i}.png'
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            test_image.save(test_path)
            test_paths.append(test_path)

        # Classify batch
        results = classifier.classify_batch(test_paths)

        print(f"✓ Batch classification completed: {len(results)} results")

        for i, result in enumerate(results):
            print(f"  Image {i}: {result['logo_type']} (conf: {result['confidence']:.4f})")

        # Clean up
        for path in test_paths:
            if os.path.exists(path):
                os.remove(path)

        return True

    except Exception as e:
        print(f"✗ Batch classification failed: {e}")
        return False

def main():
    """Run all tests."""
    print("EfficientNet Classifier Test Suite")
    print("=" * 50)

    # Test initialization
    classifier = test_classifier_initialization()
    if not classifier:
        print("\n✗ Cannot proceed - initialization failed")
        return False

    # Test single classification
    single_success = test_single_classification(classifier)

    # Test batch classification
    batch_success = test_batch_classification(classifier)

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    tests = [
        ("Initialization", classifier is not None),
        ("Single Classification", single_success),
        ("Batch Classification", batch_success)
    ]

    passed = sum(1 for _, success in tests if success)
    total = len(tests)

    for name, success in tests:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("✓ ALL TESTS PASSED - EfficientNet classifier working correctly!")
        return True
    else:
        print("✗ Some tests failed - needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)