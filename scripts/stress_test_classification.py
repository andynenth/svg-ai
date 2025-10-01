#!/usr/bin/env python3
"""
Comprehensive Stress Testing for Classification System (Day 7)
Tests concurrent processing, memory usage, error handling, and performance consistency
"""

import sys
import os
import time
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ai_modules.classification.hybrid_classifier import HybridClassifier

def stress_test_classification():
    """Comprehensive stress testing of classification system"""
    print("=" * 70)
    print("COMPREHENSIVE STRESS TESTING - CLASSIFICATION SYSTEM")
    print("=" * 70)

    hybrid = HybridClassifier()
    results = {
        'concurrent_test': {},
        'memory_stress_test': {},
        'long_running_test': {},
        'error_handling_test': {},
        'performance_consistency_test': {}
    }

    # Test images (using available test images)
    test_images = [
        'test-data/circle_00.png',
        'test-data/text_tech_00.png',
        'test-data/gradient_radial_00.png'
    ]

    # Check which images exist
    existing_images = []
    for img_path in test_images:
        if os.path.exists(img_path):
            existing_images.append(img_path)

    if not existing_images:
        print("❌ Error: No test images found for stress testing")
        return results

    print(f"Using {len(existing_images)} test images:")
    for img in existing_images:
        print(f"  - {img}")

    # Test 1: Concurrent Classification
    print("\n" + "=" * 70)
    print("TEST 1: CONCURRENT CLASSIFICATION")
    print("=" * 70)

    def classify_concurrent(image_path):
        """Concurrent classification function"""
        try:
            return hybrid.classify_safe(image_path)
        except Exception as e:
            return {'error': True, 'error_message': str(e)}

    # Create concurrent test workload
    concurrent_test_images = existing_images * 20  # 60 total classifications
    max_workers = 10

    print(f"Running {len(concurrent_test_images)} concurrent classifications with {max_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        start_time = time.time()
        futures = [executor.submit(classify_concurrent, img) for img in concurrent_test_images]
        concurrent_results = []

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=30)  # 30 second timeout
                concurrent_results.append(result)
            except Exception as e:
                concurrent_results.append({'error': True, 'error_message': f'Future failed: {str(e)}'})

        concurrent_time = time.time() - start_time

    # Analyze concurrent test results
    successful_results = [r for r in concurrent_results if not r.get('error', False)]
    failed_results = [r for r in concurrent_results if r.get('error', False)]

    results['concurrent_test'] = {
        'total_images': len(concurrent_test_images),
        'successful_classifications': len(successful_results),
        'failed_classifications': len(failed_results),
        'success_rate': len(successful_results) / len(concurrent_test_images) * 100,
        'total_time': concurrent_time,
        'average_time_per_image': concurrent_time / len(concurrent_test_images),
        'throughput': len(concurrent_test_images) / concurrent_time
    }

    print(f"✅ Concurrent test completed:")
    print(f"   Success rate: {results['concurrent_test']['success_rate']:.1f}%")
    print(f"   Throughput: {results['concurrent_test']['throughput']:.1f} images/second")
    print(f"   Average time per image: {results['concurrent_test']['average_time_per_image']:.3f}s")

    # Test 2: Memory Stress Test
    print("\n" + "=" * 70)
    print("TEST 2: MEMORY STRESS TEST")
    print("=" * 70)

    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        print("⚠️  Warning: psutil not available, installing...")
        os.system("pip install psutil")
        try:
            import psutil
            PSUTIL_AVAILABLE = True
        except ImportError:
            PSUTIL_AVAILABLE = False
            print("❌ Could not install psutil, skipping memory test")

    if PSUTIL_AVAILABLE:
        memory_usage = []
        print("Running 100 iterations to test for memory leaks...")

        initial_memory = psutil.virtual_memory().used
        peak_memory = initial_memory

        for i in range(100):
            if i % 20 == 0:
                print(f"  Iteration {i}/100...")

            mem_before = psutil.virtual_memory().used
            result = hybrid.classify_safe(existing_images[i % len(existing_images)])
            mem_after = psutil.virtual_memory().used

            memory_delta = mem_after - mem_before
            memory_usage.append(memory_delta)

            if mem_after > peak_memory:
                peak_memory = mem_after

            # Trigger garbage collection every 25 iterations
            if i % 25 == 0:
                import gc
                gc.collect()

        final_memory = psutil.virtual_memory().used
        total_memory_growth = final_memory - initial_memory

        results['memory_stress_test'] = {
            'iterations': 100,
            'initial_memory_mb': initial_memory / (1024 * 1024),
            'final_memory_mb': final_memory / (1024 * 1024),
            'peak_memory_mb': peak_memory / (1024 * 1024),
            'total_memory_growth_mb': total_memory_growth / (1024 * 1024),
            'average_memory_delta_mb': sum(memory_usage) / len(memory_usage) / (1024 * 1024),
            'max_memory_delta_mb': max(memory_usage) / (1024 * 1024),
            'memory_leak_detected': total_memory_growth > 100 * 1024 * 1024  # 100MB threshold
        }

        print(f"✅ Memory stress test completed:")
        print(f"   Total memory growth: {results['memory_stress_test']['total_memory_growth_mb']:.1f}MB")
        print(f"   Peak memory: {results['memory_stress_test']['peak_memory_mb']:.1f}MB")
        print(f"   Memory leak detected: {results['memory_stress_test']['memory_leak_detected']}")

    # Test 3: Long Running Test
    print("\n" + "=" * 70)
    print("TEST 3: LONG RUNNING PERFORMANCE TEST")
    print("=" * 70)

    print("Running sustained performance test for 2 minutes...")
    end_time = time.time() + 120  # 2 minutes
    long_run_results = []
    iteration_count = 0

    while time.time() < end_time:
        start_iter = time.time()
        image_path = existing_images[iteration_count % len(existing_images)]
        result = hybrid.classify_safe(image_path)
        end_iter = time.time()

        long_run_results.append({
            'iteration': iteration_count,
            'processing_time': end_iter - start_iter,
            'success': not result.get('error', False),
            'memory_usage': psutil.virtual_memory().used / (1024 * 1024) if PSUTIL_AVAILABLE else 0
        })

        iteration_count += 1

        # Brief pause to avoid overwhelming the system
        time.sleep(0.1)

    # Analyze long running results
    successful_long_runs = [r for r in long_run_results if r['success']]
    processing_times = [r['processing_time'] for r in successful_long_runs]

    results['long_running_test'] = {
        'duration_seconds': 120,
        'total_iterations': len(long_run_results),
        'successful_iterations': len(successful_long_runs),
        'success_rate': len(successful_long_runs) / len(long_run_results) * 100 if long_run_results else 0,
        'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
        'throughput': len(long_run_results) / 120
    }

    print(f"✅ Long running test completed:")
    print(f"   Iterations: {results['long_running_test']['total_iterations']}")
    print(f"   Success rate: {results['long_running_test']['success_rate']:.1f}%")
    print(f"   Throughput: {results['long_running_test']['throughput']:.1f} images/second")

    # Test 4: Error Handling Test
    print("\n" + "=" * 70)
    print("TEST 4: ERROR HANDLING TEST")
    print("=" * 70)

    # Create test error cases
    error_cases = [
        ('nonexistent_file.png', 'File not found'),
        ('/tmp/empty_file.png', 'Empty file'),  # We'll create this
        ('/tmp/tiny_file.png', 'Too small'),    # We'll create this
    ]

    # Create test error files
    try:
        # Empty file
        with open('/tmp/empty_file.png', 'w') as f:
            pass

        # Tiny file
        with open('/tmp/tiny_file.png', 'wb') as f:
            f.write(b'tiny')

        print("Created test error files")
    except Exception as e:
        print(f"Warning: Could not create test error files: {e}")

    error_handling_results = []
    for error_case, description in error_cases:
        print(f"Testing error case: {description}")
        try:
            result = hybrid.classify_safe(error_case)
            error_handling_results.append({
                'case': error_case,
                'description': description,
                'handled_gracefully': result.get('error', False),
                'error_type': result.get('error_type', 'none'),
                'error_message': result.get('error_message', '')
            })
            print(f"  ✅ Handled gracefully: {result.get('error_type', 'unknown')}")
        except Exception as e:
            error_handling_results.append({
                'case': error_case,
                'description': description,
                'handled_gracefully': False,
                'exception': str(e)
            })
            print(f"  ❌ Unhandled exception: {str(e)}")

    results['error_handling_test'] = error_handling_results

    # Clean up test files
    for file_path in ['/tmp/empty_file.png', '/tmp/tiny_file.png']:
        try:
            os.remove(file_path)
        except:
            pass

    # Test 5: Performance Consistency Test
    print("\n" + "=" * 70)
    print("TEST 5: PERFORMANCE CONSISTENCY TEST")
    print("=" * 70)

    print("Testing performance consistency across multiple runs...")
    consistency_results = []

    for run in range(10):
        run_times = []
        for image_path in existing_images:
            start_time = time.time()
            result = hybrid.classify_safe(image_path)
            end_time = time.time()

            if not result.get('error', False):
                run_times.append(end_time - start_time)

        if run_times:
            avg_time = sum(run_times) / len(run_times)
            consistency_results.append(avg_time)
            print(f"  Run {run+1}: {avg_time:.3f}s average")

    if consistency_results:
        import statistics
        mean_time = statistics.mean(consistency_results)
        stdev_time = statistics.stdev(consistency_results) if len(consistency_results) > 1 else 0
        cv = stdev_time / mean_time if mean_time > 0 else 0  # Coefficient of variation

        results['performance_consistency_test'] = {
            'runs': len(consistency_results),
            'mean_processing_time': mean_time,
            'std_dev_processing_time': stdev_time,
            'coefficient_of_variation': cv,
            'min_time': min(consistency_results),
            'max_time': max(consistency_results),
            'consistent_performance': cv < 0.2  # CV < 20% is considered consistent
        }

        print(f"✅ Performance consistency test completed:")
        print(f"   Mean time: {mean_time:.3f}s")
        print(f"   Std deviation: {stdev_time:.3f}s")
        print(f"   Coefficient of variation: {cv:.3f}")
        print(f"   Performance consistent: {results['performance_consistency_test']['consistent_performance']}")

    return results

def print_stress_test_summary(results: Dict[str, Any]):
    """Print comprehensive stress test summary"""
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)

    # Overall assessment
    tests_passed = 0
    total_tests = 0

    # Concurrent test
    if 'concurrent_test' in results:
        total_tests += 1
        concurrent_success = results['concurrent_test'].get('success_rate', 0) >= 95
        status = "✅ PASS" if concurrent_success else "❌ FAIL"
        print(f"Concurrent Processing: {status}")
        print(f"  Success rate: {results['concurrent_test'].get('success_rate', 0):.1f}%")
        if concurrent_success:
            tests_passed += 1

    # Memory test
    if 'memory_stress_test' in results:
        total_tests += 1
        memory_ok = not results['memory_stress_test'].get('memory_leak_detected', True)
        status = "✅ PASS" if memory_ok else "❌ FAIL"
        print(f"Memory Management: {status}")
        print(f"  Memory growth: {results['memory_stress_test'].get('total_memory_growth_mb', 0):.1f}MB")
        if memory_ok:
            tests_passed += 1

    # Long running test
    if 'long_running_test' in results:
        total_tests += 1
        long_run_ok = results['long_running_test'].get('success_rate', 0) >= 95
        status = "✅ PASS" if long_run_ok else "❌ FAIL"
        print(f"Long Running Performance: {status}")
        print(f"  Success rate: {results['long_running_test'].get('success_rate', 0):.1f}%")
        if long_run_ok:
            tests_passed += 1

    # Error handling
    if 'error_handling_test' in results:
        total_tests += 1
        error_cases = results['error_handling_test']
        handled_gracefully = sum(1 for case in error_cases if case.get('handled_gracefully', False))
        error_ok = handled_gracefully == len(error_cases)
        status = "✅ PASS" if error_ok else "❌ FAIL"
        print(f"Error Handling: {status}")
        print(f"  Handled gracefully: {handled_gracefully}/{len(error_cases)}")
        if error_ok:
            tests_passed += 1

    # Performance consistency
    if 'performance_consistency_test' in results:
        total_tests += 1
        consistency_ok = results['performance_consistency_test'].get('consistent_performance', False)
        status = "✅ PASS" if consistency_ok else "❌ FAIL"
        print(f"Performance Consistency: {status}")
        cv = results['performance_consistency_test'].get('coefficient_of_variation', 1.0)
        print(f"  Coefficient of variation: {cv:.3f}")
        if consistency_ok:
            tests_passed += 1

    print("\n" + "=" * 70)
    print(f"OVERALL RESULTS: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 ALL STRESS TESTS PASSED - SYSTEM IS ROBUST")
    elif tests_passed >= total_tests * 0.8:
        print("⚠️  MOST TESTS PASSED - MINOR ISSUES DETECTED")
    else:
        print("❌ MULTIPLE FAILURES - SYSTEM NEEDS OPTIMIZATION")

    print("=" * 70)

if __name__ == "__main__":
    try:
        results = stress_test_classification()
        print_stress_test_summary(results)

        # Save results to file
        import json
        results_file = 'scripts/stress_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")

    except Exception as e:
        print(f"❌ Stress testing failed: {e}")
        import traceback
        traceback.print_exc()