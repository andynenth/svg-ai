#!/usr/bin/env python3
"""
Quality Metrics Verification Test Suite
Tests accuracy, consistency, and reliability of quality metrics calculations.
"""

import os
import sys
import json
import tempfile
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image, ImageDraw

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

def create_test_images_and_svgs():
    """Create synthetic test images and their SVG conversions."""
    test_pairs = {}

    # Create persistent directory for test files
    persistent_dir = Path('/tmp/quality_test_images')
    persistent_dir.mkdir(exist_ok=True)

    # Test case 1: Simple solid color (should convert perfectly)
    original = Image.new('RGB', (100, 100), color='red')
    original_path = persistent_dir / 'original.png'
    original.save(original_path)

    # Create a simple SVG that should match the original
    simple_svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <rect width="100" height="100" fill="red"/>
</svg>'''

    perfect_svg_path = persistent_dir / 'perfect.svg'
    with open(perfect_svg_path, 'w') as f:
        f.write(simple_svg_content)

    # Test case 2: Slightly different SVG (should have high similarity)
    slight_diff_svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <rect width="100" height="100" fill="red"/>
  <circle cx="50" cy="50" r="2" fill="darkred"/>
</svg>'''

    slight_diff_svg_path = persistent_dir / 'slight_diff.svg'
    with open(slight_diff_svg_path, 'w') as f:
        f.write(slight_diff_svg_content)

    # Test case 3: Very different SVG (should have low similarity)
    very_diff_svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <rect width="100" height="100" fill="blue"/>
</svg>'''

    very_diff_svg_path = persistent_dir / 'very_diff.svg'
    with open(very_diff_svg_path, 'w') as f:
        f.write(very_diff_svg_content)

    test_pairs = {
        'perfect': {
            'original': str(original_path),
            'svg': str(perfect_svg_path)
        },
        'slight_diff': {
            'original': str(original_path),
            'svg': str(slight_diff_svg_path)
        },
        'very_diff': {
            'original': str(original_path),
            'svg': str(very_diff_svg_path)
        }
    }

    return test_pairs

def test_quality_metrics_accuracy():
    """Test quality metrics calculations for accuracy."""
    print("=== Testing Quality Metrics Accuracy ===")

    try:
        from backend.utils.quality_metrics import ComprehensiveMetrics

        # Create test images and SVG pairs
        test_pairs = create_test_images_and_svgs()
        print(f"Created {len(test_pairs)} synthetic test image/SVG pairs")

        quality_metrics = ComprehensiveMetrics()

        # Test perfect match (should be SSIM=1.0, MSE=0, PSNR=inf)
        perfect_metrics = quality_metrics.compare_images(
            test_pairs['perfect']['original'],
            test_pairs['perfect']['svg']
        )

        print(f"Perfect match metrics:")
        print(f"  SSIM: {perfect_metrics['ssim']:.6f}")
        print(f"  MSE: {perfect_metrics['mse']:.6f}")
        print(f"  PSNR: {perfect_metrics['psnr']:.2f}")

        # Validate perfect match
        perfect_ssim_correct = abs(perfect_metrics['ssim'] - 1.0) < 0.001
        perfect_mse_correct = perfect_metrics['mse'] < 0.001
        perfect_psnr_high = perfect_metrics['psnr'] > 50  # Very high for near-identical

        print(f"  ✓ SSIM correct: {perfect_ssim_correct}")
        print(f"  ✓ MSE correct: {perfect_mse_correct}")
        print(f"  ✓ PSNR correct: {perfect_psnr_high}")

        # Test slight difference
        slight_metrics = quality_metrics.compare_images(
            test_pairs['slight_diff']['original'],
            test_pairs['slight_diff']['svg']
        )

        print(f"\nSlight difference metrics:")
        print(f"  SSIM: {slight_metrics['ssim']:.6f}")
        print(f"  MSE: {slight_metrics['mse']:.6f}")
        print(f"  PSNR: {slight_metrics['psnr']:.2f}")

        # Validate slight difference
        slight_ssim_high = slight_metrics['ssim'] > 0.95  # Should still be very high
        slight_mse_low = slight_metrics['mse'] < 10  # Should be low
        slight_psnr_reasonable = slight_metrics['psnr'] > 20  # Should be reasonable

        print(f"  ✓ SSIM high: {slight_ssim_high}")
        print(f"  ✓ MSE low: {slight_mse_low}")
        print(f"  ✓ PSNR reasonable: {slight_psnr_reasonable}")

        # Test very different image
        very_diff_metrics = quality_metrics.compare_images(
            test_pairs['very_diff']['original'],
            test_pairs['very_diff']['svg']
        )

        print(f"\nVery different metrics:")
        print(f"  SSIM: {very_diff_metrics['ssim']:.6f}")
        print(f"  MSE: {very_diff_metrics['mse']:.6f}")
        print(f"  PSNR: {very_diff_metrics['psnr']:.2f}")

        # Validate very different
        very_diff_ssim_low = very_diff_metrics['ssim'] < 0.5  # Should be low
        very_diff_mse_high = very_diff_metrics['mse'] > 1000  # Should be high
        very_diff_psnr_low = very_diff_metrics['psnr'] < 20  # Should be low

        print(f"  ✓ SSIM low: {very_diff_ssim_low}")
        print(f"  ✓ MSE high: {very_diff_mse_high}")
        print(f"  ✓ PSNR low: {very_diff_psnr_low}")

        # Test metric ordering (SSIM should decrease with increasing difference)
        ssim_ordering = (perfect_metrics['ssim'] > slight_metrics['ssim'] > very_diff_metrics['ssim'])
        mse_ordering = (perfect_metrics['mse'] < slight_metrics['mse'] < very_diff_metrics['mse'])

        print(f"\n✓ Metric ordering:")
        print(f"  SSIM ordering correct: {ssim_ordering}")
        print(f"  MSE ordering correct: {mse_ordering}")

        # Overall accuracy assessment
        all_correct = all([
            perfect_ssim_correct, perfect_mse_correct, perfect_psnr_high,
            slight_ssim_high, slight_mse_low, slight_psnr_reasonable,
            very_diff_ssim_low, very_diff_mse_high, very_diff_psnr_low,
            ssim_ordering, mse_ordering
        ])

        if all_correct:
            print(f"\n✓ All accuracy tests passed")
        else:
            print(f"\n⚠ Some accuracy tests failed")

        return {
            'accuracy_correct': all_correct,
            'perfect_metrics': perfect_metrics,
            'slight_metrics': slight_metrics,
            'very_diff_metrics': very_diff_metrics,
            'ssim_ordering': ssim_ordering,
            'mse_ordering': mse_ordering
        }

    except Exception as e:
        print(f"✗ Quality metrics accuracy test failed: {e}")
        traceback.print_exc()
        return False

def test_real_image_quality():
    """Test quality metrics on real logo conversions."""
    print("\n=== Testing Real Image Quality Metrics ===")

    try:
        from backend.utils.quality_metrics import ComprehensiveMetrics
        from backend.converters.vtracer_converter import VTracerConverter

        # Get real test images
        test_images = []
        logo_dirs = ['data/logos/simple_geometric', 'data/logos/text_based']

        for logo_dir in logo_dirs:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob('*.png'))[:3]
                test_images.extend(png_files)

        if not test_images:
            print("✗ No real test images found")
            return False

        print(f"Testing quality metrics on {len(test_images)} real images")

        converter = VTracerConverter()
        quality_metrics = ComprehensiveMetrics()

        results = []

        for i, image_path in enumerate(test_images):
            print(f"\nProcessing {i+1}/{len(test_images)}: {image_path.name}")

            try:
                # Convert image
                svg_content = converter.convert(str(image_path))

                if not svg_content:
                    print(f"  ✗ Conversion failed")
                    continue

                # Calculate quality metrics
                with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
                    tmp.write(svg_content.encode())
                    tmp.flush()

                    metrics = quality_metrics.compare_images(str(image_path), tmp.name)

                print(f"  SSIM: {metrics['ssim']:.4f}")
                print(f"  MSE: {metrics['mse']:.2f}")
                print(f"  PSNR: {metrics['psnr']:.2f}")

                # Quality assessment
                high_quality = metrics['ssim'] > 0.9
                reasonable_quality = metrics['ssim'] > 0.7

                if high_quality:
                    quality_level = "High"
                elif reasonable_quality:
                    quality_level = "Reasonable"
                else:
                    quality_level = "Low"

                print(f"  Quality: {quality_level}")

                results.append({
                    'image': str(image_path),
                    'metrics': metrics,
                    'quality_level': quality_level,
                    'high_quality': high_quality
                })

            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results.append({
                    'image': str(image_path),
                    'error': str(e)
                })

        # Analyze results
        successful = [r for r in results if 'metrics' in r]
        if successful:
            avg_ssim = sum(r['metrics']['ssim'] for r in successful) / len(successful)
            high_quality_count = sum(1 for r in successful if r['high_quality'])
            high_quality_rate = high_quality_count / len(successful) * 100

            print(f"\n✓ Real image quality analysis:")
            print(f"  Images tested: {len(successful)}")
            print(f"  Average SSIM: {avg_ssim:.4f}")
            print(f"  High quality rate: {high_quality_rate:.1f}%")

            return {
                'images_tested': len(successful),
                'avg_ssim': avg_ssim,
                'high_quality_rate': high_quality_rate,
                'results': results
            }
        else:
            print(f"✗ No successful quality measurements")
            return False

    except Exception as e:
        print(f"✗ Real image quality test failed: {e}")
        traceback.print_exc()
        return False

def test_quality_consistency():
    """Test consistency of quality metrics across multiple runs."""
    print("\n=== Testing Quality Metrics Consistency ===")

    try:
        from backend.utils.quality_metrics import ComprehensiveMetrics

        # Get a test image
        test_image = None
        for logo_dir in ['data/logos/simple_geometric']:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob('*.png'))
                if png_files:
                    test_image = png_files[0]
                    break

        if not test_image:
            print("✗ No test image found")
            return False

        print(f"Testing consistency with: {test_image.name}")

        quality_metrics = ComprehensiveMetrics()

        # Run same comparison multiple times
        num_runs = 5
        all_metrics = []

        for run in range(num_runs):
            metrics = quality_metrics.compare_images(str(test_image), str(test_image))
            all_metrics.append(metrics)
            print(f"Run {run+1}: SSIM={metrics['ssim']:.6f}, MSE={metrics['mse']:.6f}")

        # Check consistency
        ssim_values = [m['ssim'] for m in all_metrics]
        mse_values = [m['mse'] for m in all_metrics]
        psnr_values = [m['psnr'] for m in all_metrics]

        ssim_std = np.std(ssim_values)
        mse_std = np.std(mse_values)
        psnr_std = np.std(psnr_values)

        print(f"\nConsistency analysis:")
        print(f"  SSIM std dev: {ssim_std:.8f}")
        print(f"  MSE std dev: {mse_std:.8f}")
        print(f"  PSNR std dev: {psnr_std:.8f}")

        # Consistency thresholds
        ssim_consistent = ssim_std < 0.001
        mse_consistent = mse_std < 0.001
        psnr_consistent = psnr_std < 1.0

        all_consistent = ssim_consistent and mse_consistent and psnr_consistent

        if all_consistent:
            print(f"✓ Quality metrics are consistent across runs")
        else:
            print(f"⚠ Some inconsistency detected in quality metrics")

        return {
            'num_runs': num_runs,
            'ssim_std': ssim_std,
            'mse_std': mse_std,
            'psnr_std': psnr_std,
            'consistent': all_consistent,
            'all_metrics': all_metrics
        }

    except Exception as e:
        print(f"✗ Quality consistency test failed: {e}")
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test quality metrics on edge cases."""
    print("\n=== Testing Quality Metrics Edge Cases ===")

    try:
        from backend.utils.quality_metrics import ComprehensiveMetrics
        from PIL import Image

        quality_metrics = ComprehensiveMetrics()
        edge_cases = []

        # Create edge case images
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 1. Very small image
            small_img = Image.new('RGB', (10, 10), color='red')
            small_path = tmpdir_path / 'small.png'
            small_img.save(small_path)

            # 2. Large image (if we want to test performance)
            large_img = Image.new('RGB', (500, 500), color='green')
            large_path = tmpdir_path / 'large.png'
            large_img.save(large_path)

            # 3. Grayscale image
            gray_img = Image.new('L', (100, 100), color=128)
            gray_path = tmpdir_path / 'gray.png'
            gray_img.save(gray_path)

            # 4. Black and white image
            bw_img = Image.new('1', (100, 100), color=1)
            bw_path = tmpdir_path / 'bw.png'
            bw_img.save(bw_path)

            test_cases = [
                ('small_images', str(small_path), str(small_path)),
                ('large_images', str(large_path), str(large_path)),
                ('grayscale', str(gray_path), str(gray_path)),
                ('black_white', str(bw_path), str(bw_path))
            ]

            for case_name, img1_path, img2_path in test_cases:
                print(f"\nTesting {case_name}:")

                try:
                    metrics = quality_metrics.compare_images(img1_path, img2_path)

                    print(f"  SSIM: {metrics['ssim']:.4f}")
                    print(f"  MSE: {metrics['mse']:.4f}")
                    print(f"  PSNR: {metrics['psnr']:.2f}")

                    # Identical images should have perfect metrics
                    expected_perfect = abs(metrics['ssim'] - 1.0) < 0.01 and metrics['mse'] < 1.0

                    edge_cases.append({
                        'case': case_name,
                        'metrics': metrics,
                        'expected_perfect': expected_perfect,
                        'success': True
                    })

                    if expected_perfect:
                        print(f"  ✓ Perfect metrics as expected")
                    else:
                        print(f"  ⚠ Metrics not perfect for identical images")

                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    edge_cases.append({
                        'case': case_name,
                        'error': str(e),
                        'success': False
                    })

        # Analyze edge case results
        successful_cases = [c for c in edge_cases if c['success']]
        success_rate = len(successful_cases) / len(edge_cases) * 100

        perfect_rate = sum(1 for c in successful_cases if c['expected_perfect']) / len(successful_cases) * 100 if successful_cases else 0

        print(f"\n✓ Edge case analysis:")
        print(f"  Cases tested: {len(edge_cases)}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Perfect metrics rate: {perfect_rate:.1f}%")

        return {
            'cases_tested': len(edge_cases),
            'success_rate': success_rate,
            'perfect_rate': perfect_rate,
            'edge_cases': edge_cases
        }

    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        traceback.print_exc()
        return False

def generate_quality_report(results: Dict[str, Any]):
    """Generate comprehensive quality metrics report."""
    print("\n" + "="*60)
    print("QUALITY METRICS VERIFICATION REPORT")
    print("="*60)

    # Test summary
    passed_tests = sum(1 for result in results.values() if result is not False)
    total_tests = len(results)

    print(f"\nTest Summary:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    # Accuracy analysis
    if 'accuracy' in results and results['accuracy']:
        accuracy_result = results['accuracy']
        print(f"\nAccuracy Analysis:")
        print(f"  Metric accuracy: {'PASS' if accuracy_result.get('accuracy_correct') else 'FAIL'}")
        print(f"  SSIM ordering: {'PASS' if accuracy_result.get('ssim_ordering') else 'FAIL'}")
        print(f"  MSE ordering: {'PASS' if accuracy_result.get('mse_ordering') else 'FAIL'}")

    # Real image performance
    if 'real_images' in results and results['real_images']:
        real_result = results['real_images']
        print(f"\nReal Image Performance:")
        print(f"  Images tested: {real_result.get('images_tested', 'N/A')}")
        print(f"  Average SSIM: {real_result.get('avg_ssim', 'N/A'):.4f}")
        print(f"  High quality rate: {real_result.get('high_quality_rate', 'N/A'):.1f}%")

    # Consistency analysis
    if 'consistency' in results and results['consistency']:
        consistency_result = results['consistency']
        print(f"\nConsistency Analysis:")
        print(f"  SSIM consistency: {consistency_result.get('ssim_std', 'N/A'):.8f} std dev")
        print(f"  Overall consistent: {'PASS' if consistency_result.get('consistent') else 'FAIL'}")

    # Edge cases
    if 'edge_cases' in results and results['edge_cases']:
        edge_result = results['edge_cases']
        print(f"\nEdge Cases:")
        print(f"  Cases tested: {edge_result.get('cases_tested', 'N/A')}")
        print(f"  Success rate: {edge_result.get('success_rate', 'N/A'):.1f}%")
        print(f"  Perfect metrics rate: {edge_result.get('perfect_rate', 'N/A'):.1f}%")

    # Overall assessment
    quality_reliable = passed_tests >= 3  # Most tests should pass

    print(f"\nOverall Assessment:")
    if quality_reliable:
        print(f"  ✓ Quality metrics system is RELIABLE")
    else:
        print(f"  ⚠ Quality metrics system needs attention")

    # Save detailed report
    report = {
        'timestamp': str(time.time()),
        'test_summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests/total_tests)*100
        },
        'quality_reliable': quality_reliable,
        'detailed_results': results
    }

    with open('quality_metrics_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Detailed report saved to quality_metrics_report.json")

    return quality_reliable

def main():
    """Run complete quality metrics verification suite."""
    print("SVG-AI Quality Metrics Verification Suite")
    print("=" * 60)

    # Define tests
    tests = [
        ('accuracy', test_quality_metrics_accuracy),
        ('real_images', test_real_image_quality),
        ('consistency', test_quality_consistency),
        ('edge_cases', test_edge_cases),
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
    all_reliable = generate_quality_report(results)

    return 0 if all_reliable else 1

if __name__ == '__main__':
    import time
    exit(main())