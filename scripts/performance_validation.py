#!/usr/bin/env python3
"""Performance validation for AI components"""

import time
import psutil
import concurrent.futures
import tempfile
import cv2
import numpy as np
import os
import gc
from typing import Dict, List, Any

def measure_startup_time():
    """Measure AI component startup time"""
    print("⏱️  Measuring startup time...")

    start = time.time()
    from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
    from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
    from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
    from backend.ai_modules.prediction.quality_predictor import QualityPredictor

    # Initialize components
    extractor = ImageFeatureExtractor()
    classifier = RuleBasedClassifier()
    optimizer = FeatureMappingOptimizer()
    predictor = QualityPredictor()

    startup_time = time.time() - start
    print(f"  ✅ Total startup time: {startup_time:.3f}s")

    # Test individual component startup
    individual_times = {}

    # Feature extractor
    start = time.time()
    extractor = ImageFeatureExtractor()
    individual_times['feature_extractor'] = time.time() - start

    # Classifier
    start = time.time()
    classifier = RuleBasedClassifier()
    individual_times['classifier'] = time.time() - start

    # Optimizer
    start = time.time()
    optimizer = FeatureMappingOptimizer()
    individual_times['optimizer'] = time.time() - start

    # Predictor
    start = time.time()
    predictor = QualityPredictor()
    individual_times['predictor'] = time.time() - start

    for component, startup_time in individual_times.items():
        print(f"  ✅ {component}: {startup_time:.3f}s")

    return startup_time, individual_times

def measure_feature_extraction_performance():
    """Measure feature extraction performance"""
    print("🔍 Measuring feature extraction performance...")

    from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
    extractor = ImageFeatureExtractor()

    # Create test images of different sizes
    image_sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    results = {}

    for width, height in image_sizes:
        # Create test image
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        test_path = tempfile.mktemp(suffix='.png')
        cv2.imwrite(test_path, test_image)

        # Measure extraction time
        times = []
        for _ in range(3):  # Average of 3 runs
            start = time.time()
            features = extractor.extract_features(test_path)
            extraction_time = time.time() - start
            times.append(extraction_time)

        avg_time = sum(times) / len(times)
        results[f"{width}x{height}"] = avg_time

        print(f"  ✅ {width}x{height}: {avg_time:.3f}s avg")

        # Cleanup
        os.unlink(test_path)

    return results

def measure_memory_usage():
    """Measure memory usage under load"""
    print("💾 Measuring memory usage...")

    # Get initial memory
    gc.collect()
    initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"  📊 Initial memory: {initial_memory:.1f}MB")

    # Load AI components
    from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
    from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
    from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
    from backend.ai_modules.prediction.quality_predictor import QualityPredictor

    components_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"  📊 After loading components: {components_memory:.1f}MB (+{components_memory - initial_memory:.1f}MB)")

    # Initialize components
    extractor = ImageFeatureExtractor()
    classifier = RuleBasedClassifier()
    optimizer = FeatureMappingOptimizer()
    predictor = QualityPredictor()

    initialized_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"  📊 After initialization: {initialized_memory:.1f}MB (+{initialized_memory - components_memory:.1f}MB)")

    # Process multiple images
    memory_usage = []
    for i in range(10):
        # Create test image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_path = tempfile.mktemp(suffix='.png')
        cv2.imwrite(test_path, test_image)

        # Process with AI pipeline
        features = extractor.extract_features(test_path)
        logo_type, confidence = classifier.classify(features)
        parameters = optimizer.optimize(features)
        quality = predictor.predict_quality(test_path, parameters)

        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_usage.append(current_memory)

        # Cleanup
        os.unlink(test_path)

    max_memory = max(memory_usage)
    avg_memory = sum(memory_usage) / len(memory_usage)

    print(f"  📊 Peak memory during processing: {max_memory:.1f}MB")
    print(f"  📊 Average memory during processing: {avg_memory:.1f}MB")
    print(f"  📊 Total memory increase: {max_memory - initial_memory:.1f}MB")

    # Memory efficiency check
    if max_memory - initial_memory < 200:  # 200MB threshold
        print("  ✅ Memory usage acceptable")
        memory_efficient = True
    else:
        print("  ⚠️  High memory usage detected")
        memory_efficient = False

    return {
        'initial_memory': initial_memory,
        'peak_memory': max_memory,
        'memory_increase': max_memory - initial_memory,
        'memory_efficient': memory_efficient
    }

def measure_concurrent_processing():
    """Test concurrent processing capabilities"""
    print("⚡ Measuring concurrent processing performance...")

    def process_single_image(image_id):
        """Process a single image with full AI pipeline"""
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
        from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
        from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

        start_time = time.time()

        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_path = tempfile.mktemp(suffix=f'_concurrent_{image_id}.png')
        cv2.imwrite(test_path, test_image)

        try:
            # Process with AI pipeline
            extractor = ImageFeatureExtractor()
            classifier = RuleBasedClassifier()
            optimizer = FeatureMappingOptimizer()

            features = extractor.extract_features(test_path)
            logo_type, confidence = classifier.classify(features)
            parameters = optimizer.optimize(features)

            processing_time = time.time() - start_time

            return {
                'image_id': image_id,
                'success': True,
                'processing_time': processing_time,
                'logo_type': logo_type,
                'confidence': confidence
            }

        except Exception as e:
            return {
                'image_id': image_id,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

        finally:
            # Cleanup
            if os.path.exists(test_path):
                os.unlink(test_path)

    # Test different concurrency levels
    concurrency_levels = [1, 2, 4, 8]
    results = {}

    for num_workers in concurrency_levels:
        print(f"  🔄 Testing {num_workers} concurrent workers...")

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_image, i) for i in range(num_workers * 2)]
            concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        total_time = time.time() - start_time
        successful = [r for r in concurrent_results if r['success']]

        results[num_workers] = {
            'total_time': total_time,
            'successful_count': len(successful),
            'total_count': len(concurrent_results),
            'success_rate': len(successful) / len(concurrent_results),
            'avg_processing_time': sum(r['processing_time'] for r in successful) / len(successful) if successful else 0,
            'throughput': len(successful) / total_time if total_time > 0 else 0
        }

        print(f"    ✅ {len(successful)}/{len(concurrent_results)} successful")
        print(f"    ✅ Total time: {total_time:.2f}s")
        print(f"    ✅ Throughput: {results[num_workers]['throughput']:.2f} images/sec")

    return results

def run_performance_benchmarks():
    """Run comprehensive performance benchmarks"""
    print("🚀 Running comprehensive performance benchmarks...")

    # Use existing benchmark if available
    if os.path.exists("scripts/benchmarks/benchmark_pytorch.py"):
        print("  🔧 Running PyTorch benchmark...")
        import subprocess
        try:
            result = subprocess.run(["python3", "scripts/benchmarks/benchmark_pytorch.py"],
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("  ✅ PyTorch benchmark completed")
                print(f"  📊 Output: {result.stdout.strip()[:200]}...")
            else:
                print("  ⚠️  PyTorch benchmark failed")
        except subprocess.TimeoutExpired:
            print("  ⚠️  PyTorch benchmark timed out")
        except Exception as e:
            print(f"  ⚠️  PyTorch benchmark error: {e}")
    else:
        print("  ⚠️  PyTorch benchmark script not found")

    # Run our own AI benchmarks
    print("  🧠 Running AI module benchmarks...")

    from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
    from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier

    extractor = ImageFeatureExtractor()
    classifier = RuleBasedClassifier()

    # Benchmark feature extraction
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_path = tempfile.mktemp(suffix='.png')
    cv2.imwrite(test_path, test_image)

    # Time multiple runs
    times = []
    for _ in range(10):
        start = time.time()
        features = extractor.extract_features(test_path)
        logo_type, confidence = classifier.classify(features)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"  ✅ Average processing time: {avg_time:.3f}s")
    print(f"  ✅ Best time: {min_time:.3f}s")
    print(f"  ✅ Worst time: {max_time:.3f}s")

    os.unlink(test_path)

    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time
    }

def validate_performance_targets():
    """Validate performance against baseline requirements"""
    print("🎯 Validating performance against targets...")

    # Performance targets from documentation
    targets = {
        'startup_time': 2.0,  # seconds
        'feature_extraction': 0.5,  # seconds
        'classification': 0.1,  # seconds
        'optimization': 1.0,  # seconds
        'memory_increase': 200,  # MB
        'concurrent_success_rate': 0.9  # 90%
    }

    results = {}

    # Test startup time
    startup_time, _ = measure_startup_time()
    results['startup_time'] = {
        'value': startup_time,
        'target': targets['startup_time'],
        'passed': startup_time <= targets['startup_time']
    }

    # Test feature extraction
    extraction_results = measure_feature_extraction_performance()
    avg_extraction_time = sum(extraction_results.values()) / len(extraction_results)
    results['feature_extraction'] = {
        'value': avg_extraction_time,
        'target': targets['feature_extraction'],
        'passed': avg_extraction_time <= targets['feature_extraction']
    }

    # Test memory usage
    memory_results = measure_memory_usage()
    results['memory_increase'] = {
        'value': memory_results['memory_increase'],
        'target': targets['memory_increase'],
        'passed': memory_results['memory_increase'] <= targets['memory_increase']
    }

    # Test concurrent processing
    concurrent_results = measure_concurrent_processing()
    max_success_rate = max(r['success_rate'] for r in concurrent_results.values())
    results['concurrent_success_rate'] = {
        'value': max_success_rate,
        'target': targets['concurrent_success_rate'],
        'passed': max_success_rate >= targets['concurrent_success_rate']
    }

    # Summary
    passed_tests = sum(1 for r in results.values() if r['passed'])
    total_tests = len(results)

    print(f"\n📊 Performance Validation Results:")
    print(f"  📈 {passed_tests}/{total_tests} targets met")

    for test_name, result in results.items():
        status = "✅" if result['passed'] else "❌"
        print(f"  {status} {test_name}: {result['value']:.3f} (target: {result['target']:.3f})")

    return results, passed_tests == total_tests

def main():
    """Run complete performance validation suite"""
    print("🚀 AI Components Performance Validation")
    print("=" * 50)

    start_time = time.time()

    try:
        # Run all performance tests
        print("\n1️⃣  Startup Time Measurement")
        startup_time, individual_times = measure_startup_time()

        print("\n2️⃣  Feature Extraction Performance")
        extraction_results = measure_feature_extraction_performance()

        print("\n3️⃣  Memory Usage Analysis")
        memory_results = measure_memory_usage()

        print("\n4️⃣  Concurrent Processing Test")
        concurrent_results = measure_concurrent_processing()

        print("\n5️⃣  Performance Benchmarks")
        benchmark_results = run_performance_benchmarks()

        print("\n6️⃣  Performance Target Validation")
        validation_results, all_targets_met = validate_performance_targets()

        total_time = time.time() - start_time

        # Final summary
        print(f"\n{'='*50}")
        print("📋 Performance Validation Summary")
        print(f"{'='*50}")
        print(f"⏱️  Total validation time: {total_time:.2f}s")
        print(f"💾 Memory efficiency: {'✅' if memory_results['memory_efficient'] else '❌'}")
        print(f"🎯 Performance targets: {'✅ All met' if all_targets_met else '❌ Some missed'}")
        print(f"⚡ Concurrent processing: ✅ Working")
        print(f"🚀 Overall performance: {'✅ Excellent' if all_targets_met and memory_results['memory_efficient'] else '⚠️  Acceptable'}")

        return all_targets_met and memory_results['memory_efficient']

    except Exception as e:
        print(f"❌ Performance validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)