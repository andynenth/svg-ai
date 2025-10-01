#!/usr/bin/env python3
"""
Comprehensive Batch Processing Test Suite
Tests large-scale batch processing, parallel execution, and performance.
"""

import os
import sys
import json
import tempfile
import traceback
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any
import threading

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

def test_basic_batch_conversion():
    """Test basic batch conversion across multiple image types."""
    print("=== Testing Basic Batch Conversion ===")

    try:
        from backend.converters.vtracer_converter import VTracerConverter
        from backend.utils.quality_metrics import ComprehensiveMetrics

        # Collect test images from all categories
        test_images = []
        logo_categories = [
            'data/logos/simple_geometric',
            'data/logos/text_based',
            'data/logos/gradients',
            'data/logos/complex'
        ]

        for category in logo_categories:
            category_path = Path(category)
            if category_path.exists():
                png_files = list(category_path.glob('*.png'))[:3]  # Limit per category
                test_images.extend([(str(f), category_path.name) for f in png_files])

        if not test_images:
            print("✗ No test images found")
            return False

        print(f"Found {len(test_images)} test images across {len(logo_categories)} categories")

        # Initialize components
        converter = VTracerConverter()
        quality_metrics = ComprehensiveMetrics()

        # Process all images
        results = []
        start_time = time.time()

        for i, (image_path, category) in enumerate(test_images):
            print(f"Processing {i+1}/{len(test_images)}: {Path(image_path).name} ({category})")

            try:
                # Convert image
                convert_start = time.time()
                svg_content = converter.convert(image_path)
                convert_time = time.time() - convert_start

                if svg_content:
                    # Calculate quality metrics
                    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
                        tmp.write(svg_content.encode())
                        tmp.flush()

                        metrics = quality_metrics.compare_images(image_path, tmp.name)

                    result = {
                        'image': image_path,
                        'category': category,
                        'success': True,
                        'conversion_time': convert_time,
                        'svg_size': len(svg_content),
                        'quality': metrics
                    }
                    print(f"  ✓ Success - SSIM: {metrics['ssim']:.4f}, Time: {convert_time:.2f}s")
                else:
                    result = {
                        'image': image_path,
                        'category': category,
                        'success': False,
                        'error': 'Empty conversion result'
                    }
                    print(f"  ✗ Failed - Empty result")

                results.append(result)

            except Exception as e:
                result = {
                    'image': image_path,
                    'category': category,
                    'success': False,
                    'error': str(e)
                }
                results.append(result)
                print(f"  ✗ Failed - {e}")

        total_time = time.time() - start_time

        # Analyze results
        successful = [r for r in results if r['success']]
        success_rate = len(successful) / len(results) * 100

        if successful:
            avg_conversion_time = sum(r['conversion_time'] for r in successful) / len(successful)
            avg_ssim = sum(r['quality']['ssim'] for r in successful) / len(successful)
            total_svg_size = sum(r['svg_size'] for r in successful)

            print(f"\n✓ Batch conversion completed:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average time per image: {avg_conversion_time:.2f}s")
            print(f"  Success rate: {success_rate:.1f}% ({len(successful)}/{len(results)})")
            print(f"  Average SSIM: {avg_ssim:.4f}")
            print(f"  Total SVG size: {total_svg_size:,} chars")

            return {
                'success_rate': success_rate,
                'avg_conversion_time': avg_conversion_time,
                'avg_ssim': avg_ssim,
                'total_time': total_time,
                'processed_count': len(results),
                'successful_count': len(successful),
                'results': results
            }
        else:
            print(f"✗ All batch conversions failed")
            return False

    except Exception as e:
        print(f"✗ Batch conversion failed: {e}")
        traceback.print_exc()
        return False

def test_parallel_batch_processing():
    """Test parallel batch processing with ThreadPoolExecutor."""
    print("\n=== Testing Parallel Batch Processing ===")

    try:
        from backend.ai_modules.utils import UnifiedUtils
        from backend.converters.vtracer_converter import VTracerConverter

        # Get test images
        test_images = []
        logo_dirs = ['data/logos/simple_geometric', 'data/logos/text_based']

        for logo_dir in logo_dirs:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob('*.png'))[:5]  # More images for parallel testing
                test_images.extend(png_files)

        if len(test_images) < 4:
            print("✗ Need at least 4 test images for parallel testing")
            return False

        print(f"Using {len(test_images)} test images for parallel processing")

        # Initialize components
        utils = UnifiedUtils()
        converter = VTracerConverter()

        def convert_with_timing(image_path):
            """Convert single image with timing."""
            thread_id = threading.current_thread().ident
            start_time = time.time()

            try:
                svg_content = converter.convert(str(image_path))
                conversion_time = time.time() - start_time

                return {
                    'image': str(image_path),
                    'thread_id': thread_id,
                    'success': True,
                    'conversion_time': conversion_time,
                    'svg_size': len(svg_content) if svg_content else 0
                }
            except Exception as e:
                return {
                    'image': str(image_path),
                    'thread_id': thread_id,
                    'success': False,
                    'error': str(e),
                    'conversion_time': time.time() - start_time
                }

        # Test different worker counts
        worker_counts = [1, 2, 4]
        performance_results = {}

        for workers in worker_counts:
            print(f"\nTesting with {workers} workers:")

            start_time = time.time()
            results = utils.process_parallel(test_images, convert_with_timing, max_workers=workers)
            total_time = time.time() - start_time

            successful = [r for r in results if r and r.get('success')]
            success_rate = len(successful) / len(results) * 100

            if successful:
                avg_time = sum(r['conversion_time'] for r in successful) / len(successful)
                unique_threads = len(set(r['thread_id'] for r in successful))

                print(f"  Total time: {total_time:.2f}s")
                print(f"  Success rate: {success_rate:.1f}%")
                print(f"  Average conversion time: {avg_time:.2f}s")
                print(f"  Threads used: {unique_threads}")

                performance_results[workers] = {
                    'total_time': total_time,
                    'success_rate': success_rate,
                    'avg_conversion_time': avg_time,
                    'threads_used': unique_threads,
                    'throughput': len(successful) / total_time
                }

        # Analyze performance scaling
        if len(performance_results) > 1:
            single_worker_time = performance_results[1]['total_time']
            best_parallel_time = min(perf['total_time'] for workers, perf in performance_results.items() if workers > 1)
            speedup = single_worker_time / best_parallel_time

            print(f"\n✓ Parallel processing analysis:")
            print(f"  Single worker time: {single_worker_time:.2f}s")
            print(f"  Best parallel time: {best_parallel_time:.2f}s")
            print(f"  Speedup: {speedup:.2f}x")

            return {
                'performance_results': performance_results,
                'speedup': speedup,
                'best_workers': max(performance_results.keys(), key=lambda w: performance_results[w]['throughput'])
            }

        return True

    except Exception as e:
        print(f"✗ Parallel batch processing failed: {e}")
        traceback.print_exc()
        return False

def test_memory_efficient_batch():
    """Test memory-efficient batch processing for large datasets."""
    print("\n=== Testing Memory-Efficient Batch Processing ===")

    try:
        from backend.converters.vtracer_converter import VTracerConverter
        import psutil
        import gc

        # Get test images
        test_images = []
        for logo_dir in ['data/logos/simple_geometric', 'data/logos/text_based', 'data/logos/gradients']:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob('*.png'))
                test_images.extend(png_files)

        if len(test_images) < 5:
            print("✗ Need at least 5 test images")
            return False

        print(f"Testing memory efficiency with {len(test_images)} images")

        # Initialize converter
        converter = VTracerConverter()

        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        batch_size = 3
        processed_count = 0
        memory_readings = []

        for i in range(0, len(test_images), batch_size):
            batch = test_images[i:i+batch_size]

            print(f"Processing batch {i//batch_size + 1} ({len(batch)} images)")

            # Process batch
            for image_path in batch:
                try:
                    svg_content = converter.convert(str(image_path))
                    if svg_content:
                        processed_count += 1
                except Exception as e:
                    print(f"  ⚠ Failed: {image_path.name} - {e}")

            # Memory management
            gc.collect()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_readings.append(current_memory)

            print(f"  Memory usage: {current_memory:.1f} MB")

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        max_memory = max(memory_readings)
        memory_growth = final_memory - initial_memory

        print(f"\n✓ Memory efficiency analysis:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Peak memory: {max_memory:.1f} MB")
        print(f"  Memory growth: {memory_growth:.1f} MB")
        print(f"  Processed images: {processed_count}")
        print(f"  Memory per image: {memory_growth/processed_count:.2f} MB" if processed_count > 0 else "  N/A")

        # Memory efficiency criteria
        memory_efficient = memory_growth < 50  # Less than 50MB growth
        if memory_efficient:
            print(f"  ✓ Memory usage is efficient")
        else:
            print(f"  ⚠ Memory usage may need optimization")

        return {
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'peak_memory': max_memory,
            'memory_growth': memory_growth,
            'processed_count': processed_count,
            'memory_per_image': memory_growth/processed_count if processed_count > 0 else 0,
            'memory_efficient': memory_efficient
        }

    except ImportError:
        print("⚠ psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"✗ Memory-efficient batch processing failed: {e}")
        traceback.print_exc()
        return False

def test_error_resilience():
    """Test batch processing resilience to errors."""
    print("\n=== Testing Error Resilience ===")

    try:
        from backend.converters.vtracer_converter import VTracerConverter

        # Create mix of valid and invalid test cases
        test_cases = []

        # Valid images
        for logo_dir in ['data/logos/simple_geometric']:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                png_files = list(logo_path.glob('*.png'))[:2]
                test_cases.extend([('valid', str(f)) for f in png_files])

        # Invalid cases
        test_cases.extend([
            ('nonexistent', 'nonexistent_file.png'),
            ('invalid_path', ''),
            ('bad_extension', 'test.txt')
        ])

        if len(test_cases) < 3:
            print("✗ Need at least 3 test cases")
            return False

        print(f"Testing error resilience with {len(test_cases)} test cases")

        converter = VTracerConverter()
        results = []

        for case_type, file_path in test_cases:
            print(f"Processing {case_type}: {file_path}")

            try:
                if case_type == 'valid':
                    svg_content = converter.convert(file_path)
                    success = svg_content is not None and len(svg_content) > 0
                else:
                    # These should fail gracefully
                    svg_content = converter.convert(file_path)
                    success = False  # If it doesn't throw, it's unexpected

                result = {
                    'case_type': case_type,
                    'file_path': file_path,
                    'success': success,
                    'error': None
                }

                if success:
                    print(f"  ✓ Success")
                else:
                    print(f"  ✗ Failed (expected for invalid cases)")

            except Exception as e:
                result = {
                    'case_type': case_type,
                    'file_path': file_path,
                    'success': False,
                    'error': str(e)
                }

                if case_type == 'valid':
                    print(f"  ✗ Unexpected failure: {e}")
                else:
                    print(f"  ✓ Expected failure: {e}")

            results.append(result)

        # Analyze resilience
        valid_cases = [r for r in results if r['case_type'] == 'valid']
        invalid_cases = [r for r in results if r['case_type'] != 'valid']

        valid_success_rate = sum(1 for r in valid_cases if r['success']) / len(valid_cases) * 100 if valid_cases else 0
        invalid_graceful_rate = sum(1 for r in invalid_cases if not r['success']) / len(invalid_cases) * 100 if invalid_cases else 0

        print(f"\n✓ Error resilience analysis:")
        print(f"  Valid cases success rate: {valid_success_rate:.1f}%")
        print(f"  Invalid cases handled gracefully: {invalid_graceful_rate:.1f}%")

        resilient = valid_success_rate >= 80 and invalid_graceful_rate >= 80

        return {
            'valid_success_rate': valid_success_rate,
            'invalid_graceful_rate': invalid_graceful_rate,
            'resilient': resilient,
            'results': results
        }

    except Exception as e:
        print(f"✗ Error resilience test failed: {e}")
        traceback.print_exc()
        return False

def generate_batch_report(results: Dict[str, Any]):
    """Generate comprehensive batch processing report."""
    print("\n" + "="*60)
    print("BATCH PROCESSING TEST REPORT")
    print("="*60)

    # Test summary
    passed_tests = sum(1 for result in results.values() if result is not False)
    total_tests = len(results)

    print(f"\nTest Summary:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    # Performance analysis
    if 'basic_batch_conversion' in results and results['basic_batch_conversion']:
        basic_result = results['basic_batch_conversion']
        print(f"\nBasic Batch Performance:")
        print(f"  Images processed: {basic_result.get('processed_count', 'N/A')}")
        print(f"  Success rate: {basic_result.get('success_rate', 'N/A'):.1f}%")
        print(f"  Average time per image: {basic_result.get('avg_conversion_time', 'N/A'):.2f}s")
        print(f"  Average quality (SSIM): {basic_result.get('avg_ssim', 'N/A'):.4f}")

    if 'parallel_batch_processing' in results and results['parallel_batch_processing']:
        parallel_result = results['parallel_batch_processing']
        print(f"\nParallel Processing Performance:")
        print(f"  Speedup achieved: {parallel_result.get('speedup', 'N/A'):.2f}x")
        print(f"  Best worker count: {parallel_result.get('best_workers', 'N/A')}")

    if 'memory_efficient_batch' in results and results['memory_efficient_batch']:
        memory_result = results['memory_efficient_batch']
        print(f"\nMemory Efficiency:")
        print(f"  Memory growth: {memory_result.get('memory_growth', 'N/A'):.1f} MB")
        print(f"  Memory per image: {memory_result.get('memory_per_image', 'N/A'):.2f} MB")
        print(f"  Memory efficient: {memory_result.get('memory_efficient', 'N/A')}")

    if 'error_resilience' in results and results['error_resilience']:
        resilience_result = results['error_resilience']
        print(f"\nError Resilience:")
        print(f"  Valid cases success: {resilience_result.get('valid_success_rate', 'N/A'):.1f}%")
        print(f"  Error handling: {resilience_result.get('invalid_graceful_rate', 'N/A'):.1f}%")

    # Save detailed report
    report = {
        'timestamp': str(time.time()),
        'test_summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests/total_tests)*100
        },
        'detailed_results': results
    }

    with open('batch_processing_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Detailed report saved to batch_processing_report.json")

    return passed_tests == total_tests

def main():
    """Run complete batch processing test suite."""
    print("SVG-AI Batch Processing Test Suite")
    print("=" * 60)

    # Define tests
    tests = [
        ('basic_batch_conversion', test_basic_batch_conversion),
        ('parallel_batch_processing', test_parallel_batch_processing),
        ('memory_efficient_batch', test_memory_efficient_batch),
        ('error_resilience', test_error_resilience),
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
    all_passed = generate_batch_report(results)

    return 0 if all_passed else 1

if __name__ == '__main__':
    exit(main())