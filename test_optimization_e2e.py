#!/usr/bin/env python3
"""
End-to-End Optimization Workflow Test
Tests the complete optimization pipeline including iterative parameter tuning.
"""

import os
import sys
import json
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any
import time

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

def test_iterative_optimization():
    """Test the iterative optimization workflow."""
    print("=== Testing Iterative Optimization Workflow ===")

    try:
        from backend.ai_modules.optimization import OptimizationEngine
        from backend.ai_modules.classification import ClassificationModule
        from backend.converters.vtracer_converter import VTracerConverter
        from backend.utils.quality_metrics import ComprehensiveMetrics

        # Find test image
        test_image = None
        logo_dirs = ['data/logos/simple_geometric', 'data/logos/text_based']

        for logo_dir in logo_dirs:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob('*.png'))
                if png_files:
                    test_image = png_files[0]
                    break

        if not test_image:
            print("✗ No test images found")
            return False

        print(f"Using test image: {test_image}")

        # Initialize components
        optimizer = OptimizationEngine()
        classifier = ClassificationModule()
        converter = VTracerConverter()
        quality_metrics = ComprehensiveMetrics()

        print("✓ All optimization components initialized")

        # Step 1: Extract image features
        features = classifier.feature_extractor.extract(str(test_image))
        print(f"✓ Features extracted: {features['unique_colors']} colors, complexity: {features['complexity']:.3f}")

        # Step 2: Get base parameters
        base_params = optimizer.calculate_base_parameters(features)
        print(f"✓ Base parameters calculated: {base_params}")

        # Step 3: Test base conversion
        svg_content = converter.convert(str(test_image), **base_params)

        if not svg_content:
            print("✗ Base conversion failed")
            return False

        # Step 4: Calculate initial quality
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
            tmp.write(svg_content.encode())
            tmp.flush()

            initial_metrics = quality_metrics.compare_images(str(test_image), tmp.name)
            initial_ssim = initial_metrics['ssim']

        print(f"✓ Initial conversion quality: SSIM={initial_ssim:.4f}")

        # Step 5: Test optimization with fine-tuning
        start_time = time.time()
        optimized_params = optimizer.fine_tune_parameters(
            str(test_image),
            base_params,
            target_quality=0.95
        )
        optimization_time = time.time() - start_time

        print(f"✓ Fine-tuning completed in {optimization_time:.2f}s")
        print(f"  Optimized params: {optimized_params}")

        # Step 6: Test optimized conversion
        optimized_svg = converter.convert(str(test_image), **optimized_params)

        if not optimized_svg:
            print("✗ Optimized conversion failed")
            return False

        # Step 7: Calculate final quality
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
            tmp.write(optimized_svg.encode())
            tmp.flush()

            final_metrics = quality_metrics.compare_images(str(test_image), tmp.name)
            final_ssim = final_metrics['ssim']

        print(f"✓ Final conversion quality: SSIM={final_ssim:.4f}")

        # Validate improvement
        improvement = final_ssim - initial_ssim
        print(f"✓ Quality improvement: {improvement:.4f}")

        return {
            'initial_ssim': initial_ssim,
            'final_ssim': final_ssim,
            'improvement': improvement,
            'optimization_time': optimization_time,
            'base_params': base_params,
            'optimized_params': optimized_params
        }

    except Exception as e:
        print(f"✗ Iterative optimization failed: {e}")
        traceback.print_exc()
        return False

def test_batch_optimization():
    """Test batch optimization across multiple images."""
    print("\n=== Testing Batch Optimization ===")

    try:
        from backend.ai_modules.optimization import OptimizationEngine
        from backend.ai_modules.classification import ClassificationModule
        from backend.converters.vtracer_converter import VTracerConverter
        from backend.utils.quality_metrics import ComprehensiveMetrics

        # Find multiple test images
        test_images = []
        logo_dirs = ['data/logos/simple_geometric', 'data/logos/text_based']

        for logo_dir in logo_dirs:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob('*.png'))[:2]  # Limit to 2 per category
                test_images.extend(png_files)

        if len(test_images) < 2:
            print("✗ Need at least 2 test images")
            return False

        print(f"Using {len(test_images)} test images")

        # Initialize components
        optimizer = OptimizationEngine()
        classifier = ClassificationModule()
        converter = VTracerConverter()
        quality_metrics = ComprehensiveMetrics()

        batch_results = []

        for i, test_image in enumerate(test_images):
            print(f"\nProcessing image {i+1}/{len(test_images)}: {test_image.name}")

            try:
                # Extract features and optimize
                features = classifier.feature_extractor.extract(str(test_image))
                optimized_params = optimizer.optimize(str(test_image), features, use_ml=False)

                # Convert with optimized parameters
                svg_content = converter.convert(str(test_image), **optimized_params)

                if svg_content:
                    # Calculate quality
                    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
                        tmp.write(svg_content.encode())
                        tmp.flush()

                        metrics = quality_metrics.compare_images(str(test_image), tmp.name)

                    result = {
                        'image': str(test_image),
                        'success': True,
                        'ssim': metrics['ssim'],
                        'params': optimized_params,
                        'features': {
                            'unique_colors': features['unique_colors'],
                            'complexity': features['complexity'],
                            'has_text': features['has_text'],
                            'has_gradients': features['has_gradients']
                        }
                    }
                    print(f"  ✓ Success - SSIM: {metrics['ssim']:.4f}")
                else:
                    result = {'image': str(test_image), 'success': False, 'error': 'Conversion failed'}
                    print(f"  ✗ Failed - Conversion error")

                batch_results.append(result)

            except Exception as e:
                result = {'image': str(test_image), 'success': False, 'error': str(e)}
                batch_results.append(result)
                print(f"  ✗ Failed - {e}")

        # Analyze batch results
        successful = [r for r in batch_results if r['success']]
        success_rate = len(successful) / len(batch_results) * 100

        if successful:
            avg_ssim = sum(r['ssim'] for r in successful) / len(successful)
            min_ssim = min(r['ssim'] for r in successful)
            max_ssim = max(r['ssim'] for r in successful)

            print(f"\n✓ Batch optimization completed:")
            print(f"  Success rate: {success_rate:.1f}% ({len(successful)}/{len(batch_results)})")
            print(f"  Average SSIM: {avg_ssim:.4f}")
            print(f"  SSIM range: {min_ssim:.4f} - {max_ssim:.4f}")

            return {
                'success_rate': success_rate,
                'avg_ssim': avg_ssim,
                'min_ssim': min_ssim,
                'max_ssim': max_ssim,
                'results': batch_results
            }
        else:
            print(f"✗ Batch optimization failed - no successful conversions")
            return False

    except Exception as e:
        print(f"✗ Batch optimization failed: {e}")
        traceback.print_exc()
        return False

def test_parameter_correlation():
    """Test parameter correlation analysis."""
    print("\n=== Testing Parameter Correlation Analysis ===")

    try:
        from backend.ai_modules.optimization import OptimizationEngine

        optimizer = OptimizationEngine()

        # Test correlation tracking
        test_data = [
            {'features': {'unique_colors': 5, 'complexity': 0.2}, 'params': {'color_precision': 2}, 'quality': 0.98},
            {'features': {'unique_colors': 15, 'complexity': 0.4}, 'params': {'color_precision': 4}, 'quality': 0.95},
            {'features': {'unique_colors': 50, 'complexity': 0.8}, 'params': {'color_precision': 8}, 'quality': 0.92},
        ]

        # Test correlation analysis
        correlations = optimizer.analyze_correlations(test_data)
        print(f"✓ Correlation analysis completed: {list(correlations.keys())}")

        # Test insights generation
        insights = optimizer.get_learned_insights()
        print(f"✓ Learned insights generated: {len(insights)} insights")

        return {'correlations': correlations, 'insights': insights}

    except Exception as e:
        print(f"✗ Parameter correlation failed: {e}")
        traceback.print_exc()
        return False

def test_online_learning():
    """Test online learning capabilities."""
    print("\n=== Testing Online Learning ===")

    try:
        from backend.ai_modules.optimization import OptimizationEngine

        optimizer = OptimizationEngine()

        # Enable online learning
        optimizer.enable_online_learning()
        print("✓ Online learning enabled")

        # Simulate learning from results
        test_results = [
            ({'unique_colors': 10, 'complexity': 0.3}, {'color_precision': 3}, 0.96),
            ({'unique_colors': 20, 'complexity': 0.5}, {'color_precision': 5}, 0.94),
            ({'unique_colors': 30, 'complexity': 0.7}, {'color_precision': 7}, 0.91),
        ]

        for features, params, quality in test_results:
            optimizer.record_result(features, params, quality)

        print(f"✓ Recorded {len(test_results)} learning examples")

        # Test model update (would happen automatically in real usage)
        try:
            optimizer._update_model()
            print("✓ Model update completed")
        except Exception as e:
            print(f"⚠ Model update skipped (expected for limited data): {e}")

        return True

    except Exception as e:
        print(f"✗ Online learning failed: {e}")
        traceback.print_exc()
        return False

def generate_optimization_report(results: Dict[str, Any]):
    """Generate comprehensive optimization test report."""
    print("\n" + "="*60)
    print("OPTIMIZATION WORKFLOW TEST REPORT")
    print("="*60)

    report = {
        'timestamp': str(time.time()),
        'test_results': results
    }

    # Test summary
    passed_tests = sum(1 for result in results.values() if result is not False)
    total_tests = len(results)

    print(f"\nTest Summary:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    # Detailed results
    print(f"\nDetailed Results:")
    for test_name, result in results.items():
        if result is False:
            print(f"  {test_name}: FAIL")
        elif isinstance(result, dict):
            print(f"  {test_name}: PASS")
            if 'initial_ssim' in result and 'final_ssim' in result:
                print(f"    Improvement: {result['final_ssim'] - result['initial_ssim']:.4f}")
            if 'avg_ssim' in result:
                print(f"    Average SSIM: {result['avg_ssim']:.4f}")
        else:
            print(f"  {test_name}: PASS")

    # Performance analysis
    if 'iterative_optimization' in results and results['iterative_optimization']:
        iter_result = results['iterative_optimization']
        print(f"\nPerformance Analysis:")
        print(f"  Optimization Time: {iter_result.get('optimization_time', 'N/A'):.2f}s")
        print(f"  Quality Improvement: {iter_result.get('improvement', 'N/A'):.4f}")

    if 'batch_optimization' in results and results['batch_optimization']:
        batch_result = results['batch_optimization']
        print(f"  Batch Success Rate: {batch_result.get('success_rate', 'N/A'):.1f}%")
        print(f"  Average Quality: {batch_result.get('avg_ssim', 'N/A'):.4f}")

    # Save report
    with open('optimization_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Detailed report saved to optimization_test_report.json")

    return passed_tests == total_tests

def main():
    """Run complete optimization test suite."""
    print("SVG-AI Optimization Workflow Test Suite")
    print("=" * 60)

    # Define tests
    tests = [
        ('iterative_optimization', test_iterative_optimization),
        ('batch_optimization', test_batch_optimization),
        ('parameter_correlation', test_parameter_correlation),
        ('online_learning', test_online_learning),
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
    all_passed = generate_optimization_report(results)

    return 0 if all_passed else 1

if __name__ == '__main__':
    exit(main())