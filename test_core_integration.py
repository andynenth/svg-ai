#!/usr/bin/env python3
"""
Comprehensive Core Integration Test Suite
Tests all core conversion functionality without API dependencies.
"""

import os
import sys
import json
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

def test_import_system():
    """Test that all core modules can be imported."""
    print("=== Testing Core Import System ===")

    try:
        # Test core converters
        from backend.converters.vtracer_converter import VTracerConverter
        print("✓ VTracer converter imported successfully")

        # Test AI modules
        from backend.ai_modules.classification import ClassificationModule
        print("✓ Classification module imported successfully")

        from backend.ai_modules.utils import UnifiedUtils
        print("✓ Unified utils imported successfully")

        # Test quality metrics
        from backend.utils.quality_metrics import QualityMetrics
        print("✓ Quality metrics imported successfully")

        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_conversion():
    """Test basic VTracer conversion without AI."""
    print("\n=== Testing Basic VTracer Conversion ===")

    try:
        from backend.converters.vtracer_converter import VTracerConverter

        # Find a test image
        test_image = None
        logo_dirs = [
            'data/logos/simple_geometric',
            'data/logos/text_based',
            'data/test'
        ]

        for logo_dir in logo_dirs:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob('*.png'))
                if png_files:
                    test_image = png_files[0]
                    break

        if not test_image:
            print("✗ No test PNG images found")
            return False

        print(f"Using test image: {test_image}")

        # Initialize converter
        converter = VTracerConverter()
        print(f"✓ Converter initialized: {converter.get_name()}")

        # Test conversion
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
            svg_content = converter.convert(str(test_image))

            if svg_content and len(svg_content) > 100:  # Basic validity check
                print(f"✓ Conversion successful, SVG size: {len(svg_content)} chars")

                # Save for inspection
                tmp.write(svg_content.encode())
                print(f"✓ SVG saved to: {tmp.name}")

                return True
            else:
                print("✗ Conversion failed or produced empty/invalid SVG")
                return False

    except Exception as e:
        print(f"✗ Basic conversion failed: {e}")
        traceback.print_exc()
        return False

def test_ai_classification():
    """Test AI classification functionality."""
    print("\n=== Testing AI Classification ===")

    try:
        from backend.ai_modules.classification import ClassificationModule

        # Find a test image
        test_image = None
        logo_dirs = ['data/logos/simple_geometric', 'data/test']

        for logo_dir in logo_dirs:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob('*.png'))
                if png_files:
                    test_image = png_files[0]
                    break

        if not test_image:
            print("✗ No test images found for classification")
            return False

        print(f"Using test image: {test_image}")

        # Initialize classifier
        classifier = ClassificationModule()
        print("✓ Classification module initialized")

        # Test feature extraction
        features = classifier.feature_extractor.extract(str(test_image))
        print(f"✓ Features extracted: {list(features.keys())}")

        # Test statistical classification
        stat_class = classifier.classify_statistical(features)
        print(f"✓ Statistical classification: {stat_class}")

        # Test full classification
        result = classifier.classify(str(test_image), use_neural=False)
        print(f"✓ Full classification result: {result['final_class']}")

        return True

    except Exception as e:
        print(f"✗ AI classification failed: {e}")
        traceback.print_exc()
        return False

def test_quality_metrics():
    """Test quality metrics calculation."""
    print("\n=== Testing Quality Metrics ===")

    try:
        from backend.utils.quality_metrics import ComprehensiveMetrics
        from backend.converters.vtracer_converter import VTracerConverter

        # Find test image
        test_image = None
        logo_dirs = ['data/logos/simple_geometric', 'data/test']

        for logo_dir in logo_dirs:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob('*.png'))
                if png_files:
                    test_image = png_files[0]
                    break

        if not test_image:
            print("✗ No test images found for quality metrics")
            return False

        print(f"Using test image: {test_image}")

        # Convert image first
        converter = VTracerConverter()
        svg_content = converter.convert(str(test_image))

        if not svg_content:
            print("✗ Could not convert image for quality testing")
            return False

        # Calculate quality metrics
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as svg_tmp:
            svg_tmp.write(svg_content.encode())
            svg_tmp.flush()

            quality_metrics = ComprehensiveMetrics()
            metrics = quality_metrics.compare_images(
                original_path=str(test_image),
                svg_path=svg_tmp.name
            )

            print(f"✓ Quality metrics calculated:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

            return True

    except Exception as e:
        print(f"✗ Quality metrics failed: {e}")
        traceback.print_exc()
        return False

def test_optimization_workflow():
    """Test basic optimization workflow."""
    print("\n=== Testing Optimization Workflow ===")

    try:
        from backend.ai_modules.optimization import OptimizationEngine

        # Find test image
        test_image = None
        logo_dirs = ['data/logos/simple_geometric']

        for logo_dir in logo_dirs:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob('*.png'))
                if png_files:
                    test_image = png_files[0]
                    break

        if not test_image:
            print("✗ No test images found for optimization")
            return False

        print(f"Using test image: {test_image}")

        # Initialize optimizer
        optimizer = OptimizationEngine()
        print("✓ Optimization module initialized")

        # Get image features first (need for optimization)
        from backend.ai_modules.classification import ClassificationModule
        classifier = ClassificationModule()
        features = classifier.feature_extractor.extract(str(test_image))

        # Test optimization
        result = optimizer.optimize(str(test_image), features, use_ml=False, fine_tune=False)

        if result and isinstance(result, dict):
            print(f"✓ Optimization successful")
            print(f"  Optimized params: {result}")
            return True
        else:
            print("✗ Optimization failed or returned invalid result")
            return False

    except Exception as e:
        print(f"✗ Optimization workflow failed: {e}")
        traceback.print_exc()
        return False

def test_batch_processing():
    """Test batch processing capabilities."""
    print("\n=== Testing Batch Processing ===")

    try:
        from backend.ai_modules.utils import UnifiedUtils
        from backend.converters.vtracer_converter import VTracerConverter

        # Find multiple test images
        test_images = []
        logo_dirs = ['data/logos/simple_geometric']

        for logo_dir in logo_dirs:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob('*.png'))[:3]  # Limit to 3 for testing
                test_images.extend(png_files)

        if len(test_images) < 2:
            print("✗ Need at least 2 test images for batch processing")
            return False

        print(f"Using {len(test_images)} test images")

        # Initialize utils and converter
        utils = UnifiedUtils()
        converter = VTracerConverter()

        def convert_single(image_path):
            """Convert single image."""
            try:
                svg_content = converter.convert(str(image_path))
                return {'image': str(image_path), 'success': True, 'size': len(svg_content) if svg_content else 0}
            except Exception as e:
                return {'image': str(image_path), 'success': False, 'error': str(e)}

        # Test parallel processing
        results = utils.process_parallel(test_images, convert_single, max_workers=2)

        successful = sum(1 for r in results if r and r.get('success'))
        print(f"✓ Batch processing completed: {successful}/{len(test_images)} successful")

        return successful > 0

    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        traceback.print_exc()
        return False

def generate_test_report(results: Dict[str, bool]):
    """Generate comprehensive test report."""
    print("\n" + "="*60)
    print("COMPREHENSIVE INTEGRATION TEST REPORT")
    print("="*60)

    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests

    print(f"\nTest Summary:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {failed_tests}")
    print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    print(f"\nDetailed Results:")
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    # Overall system status
    critical_tests = ['import_system', 'basic_conversion']
    critical_passed = all(results.get(test, False) for test in critical_tests)

    print(f"\nSystem Status:")
    if critical_passed:
        print("  ✓ CORE SYSTEM FUNCTIONAL")
    else:
        print("  ✗ CORE SYSTEM HAS ISSUES")

    # Recommendations
    print(f"\nRecommendations:")
    if not results.get('import_system', False):
        print("  - Fix import path issues")
    if not results.get('basic_conversion', False):
        print("  - Check VTracer installation and configuration")
    if not results.get('ai_classification', False):
        print("  - Verify AI dependencies are installed")
    if not results.get('quality_metrics', False):
        print("  - Check quality metrics dependencies")
    if not results.get('optimization_workflow', False):
        print("  - Review optimization module configuration")
    if not results.get('batch_processing', False):
        print("  - Check parallel processing setup")

    # Save report
    report = {
        'timestamp': str(datetime.now()),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'success_rate': (passed_tests/total_tests)*100,
        'results': results,
        'system_functional': critical_passed
    }

    with open('integration_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Detailed report saved to integration_test_report.json")

    return critical_passed

def main():
    """Run complete integration test suite."""
    print("SVG-AI Comprehensive Integration Test Suite")
    print("=" * 60)

    # Define all tests
    tests = [
        ('import_system', test_import_system),
        ('basic_conversion', test_basic_conversion),
        ('ai_classification', test_ai_classification),
        ('quality_metrics', test_quality_metrics),
        ('optimization_workflow', test_optimization_workflow),
        ('batch_processing', test_batch_processing),
    ]

    results = {}

    # Run each test
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ Test {test_name} crashed: {e}")
            results[test_name] = False

    # Generate report
    system_ok = generate_test_report(results)

    return 0 if system_ok else 1

if __name__ == '__main__':
    from datetime import datetime
    exit(main())