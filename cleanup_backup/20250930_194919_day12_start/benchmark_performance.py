#!/usr/bin/env python3
"""Performance Benchmarking Suite for AI Components"""

import time
import tempfile
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_modules.utils.performance_monitor import performance_monitor
from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
from backend.ai_modules.prediction.quality_predictor import QualityPredictor

def create_test_image(size=(512, 512)):
    """Create a test image for benchmarking"""
    # Create a simple geometric image for testing
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    # Add some geometric shapes
    cv2.circle(image, (size[0]//4, size[1]//4), 50, (255, 0, 0), -1)
    cv2.rectangle(image, (size[0]//2, size[1]//2), (size[0]//2 + 100, size[1]//2 + 100), (0, 255, 0), -1)
    cv2.ellipse(image, (3*size[0]//4, 3*size[1]//4), (60, 40), 0, 0, 360, (0, 0, 255), -1)

    return image

def benchmark_feature_extraction(iterations=10):
    """Benchmark feature extraction performance"""
    print("🔍 Benchmarking Feature Extraction...")

    extractor = ImageFeatureExtractor()

    # Create test image
    test_image = create_test_image()
    test_path = tempfile.mktemp(suffix='.png')
    cv2.imwrite(test_path, test_image)

    try:
        # Benchmark with performance monitor
        results = performance_monitor.benchmark_operation(
            extractor.extract_features,
            test_path,
            iterations=iterations
        )

        # Calculate statistics
        durations = [r['duration'] for r in results if r['success']]
        memory_deltas = [r['memory_delta'] for r in results if r['success']]

        print(f"  Iterations: {len(durations)}")
        print(f"  Average time: {np.mean(durations):.3f}s")
        print(f"  Min time: {np.min(durations):.3f}s")
        print(f"  Max time: {np.max(durations):.3f}s")
        print(f"  Average memory delta: {np.mean(memory_deltas):.1f}MB")
        print(f"  Max memory delta: {np.max(memory_deltas):.1f}MB")

        return np.mean(durations)

    finally:
        # Clean up
        import os
        if os.path.exists(test_path):
            os.unlink(test_path)

def benchmark_classification(iterations=10):
    """Benchmark classification performance"""
    print("\n🏷️  Benchmarking Classification...")

    classifier = RuleBasedClassifier()

    # Create test features
    test_features = {
        'complexity_score': 0.4,
        'unique_colors': 12,
        'edge_density': 0.2,
        'aspect_ratio': 1.2,
        'fill_ratio': 0.4,
        'entropy': 6.2,
        'corner_density': 0.018,
        'gradient_strength': 28.0
    }

    # Benchmark with performance monitor
    results = performance_monitor.benchmark_operation(
        classifier.classify,
        test_features,
        iterations=iterations
    )

    # Calculate statistics
    durations = [r['duration'] for r in results if r['success']]
    memory_deltas = [r['memory_delta'] for r in results if r['success']]

    print(f"  Iterations: {len(durations)}")
    print(f"  Average time: {np.mean(durations)*1000:.2f}ms")
    print(f"  Min time: {np.min(durations)*1000:.2f}ms")
    print(f"  Max time: {np.max(durations)*1000:.2f}ms")
    print(f"  Average memory delta: {np.mean(memory_deltas):.3f}MB")

    return np.mean(durations)

def benchmark_optimization(iterations=10):
    """Benchmark optimization performance"""
    print("\n⚙️  Benchmarking Optimization...")

    optimizer = FeatureMappingOptimizer()

    # Create test features
    test_features = {
        'complexity_score': 0.4,
        'unique_colors': 12,
        'edge_density': 0.2,
        'aspect_ratio': 1.2,
        'fill_ratio': 0.4
    }

    # Benchmark with performance monitor
    results = performance_monitor.benchmark_operation(
        optimizer.optimize,
        test_features,
        'simple',
        iterations=iterations
    )

    # Calculate statistics
    durations = [r['duration'] for r in results if r['success']]
    memory_deltas = [r['memory_delta'] for r in results if r['success']]

    print(f"  Iterations: {len(durations)}")
    print(f"  Average time: {np.mean(durations)*1000:.2f}ms")
    print(f"  Min time: {np.min(durations)*1000:.2f}ms")
    print(f"  Max time: {np.max(durations)*1000:.2f}ms")
    print(f"  Average memory delta: {np.mean(memory_deltas):.3f}MB")

    return np.mean(durations)

def benchmark_prediction(iterations=10):
    """Benchmark quality prediction performance"""
    print("\n🎯 Benchmarking Quality Prediction...")

    predictor = QualityPredictor()

    # Create test features and parameters
    test_features = {
        'complexity_score': 0.4,
        'unique_colors': 12,
        'edge_density': 0.2,
        'aspect_ratio': 1.2,
        'fill_ratio': 0.4,
        'entropy': 6.2,
        'corner_density': 0.018,
        'gradient_strength': 28.0
    }

    test_parameters = {
        'color_precision': 5,
        'corner_threshold': 50,
        'path_precision': 15,
        'layer_difference': 5,
        'splice_threshold': 60,
        'filter_speckle': 4,
        'segment_length': 10,
        'max_iterations': 10
    }

    # Benchmark with performance monitor
    results = performance_monitor.benchmark_operation(
        predictor.predict_quality,
        test_features,
        test_parameters,
        iterations=iterations
    )

    # Calculate statistics
    durations = [r['duration'] for r in results if r['success']]
    memory_deltas = [r['memory_delta'] for r in results if r['success']]

    print(f"  Iterations: {len(durations)}")
    print(f"  Average time: {np.mean(durations)*1000:.2f}ms")
    print(f"  Min time: {np.min(durations)*1000:.2f}ms")
    print(f"  Max time: {np.max(durations)*1000:.2f}ms")
    print(f"  Average memory delta: {np.mean(memory_deltas):.3f}MB")

    return np.mean(durations)

def benchmark_complete_pipeline(iterations=5):
    """Benchmark complete AI pipeline"""
    print("\n🚀 Benchmarking Complete AI Pipeline...")

    # Create components
    extractor = ImageFeatureExtractor()
    classifier = RuleBasedClassifier()
    optimizer = FeatureMappingOptimizer()
    predictor = QualityPredictor()

    # Create test image
    test_image = create_test_image()
    test_path = tempfile.mktemp(suffix='.png')
    cv2.imwrite(test_path, test_image)

    def complete_pipeline():
        # Extract features
        features = extractor.extract_features(test_path)

        # Classify
        logo_type, confidence = classifier.classify(features)

        # Optimize parameters
        parameters = optimizer.optimize(features, logo_type)

        # Predict quality
        quality = predictor.predict_quality(features, parameters)

        return {
            'features': features,
            'logo_type': logo_type,
            'confidence': confidence,
            'parameters': parameters,
            'quality': quality
        }

    try:
        # Benchmark with performance monitor
        results = performance_monitor.benchmark_operation(
            complete_pipeline,
            iterations=iterations
        )

        # Calculate statistics
        durations = [r['duration'] for r in results if r['success']]
        memory_deltas = [r['memory_delta'] for r in results if r['success']]

        print(f"  Iterations: {len(durations)}")
        print(f"  Average time: {np.mean(durations):.3f}s")
        print(f"  Min time: {np.min(durations):.3f}s")
        print(f"  Max time: {np.max(durations):.3f}s")
        print(f"  Average memory delta: {np.mean(memory_deltas):.1f}MB")
        print(f"  Max memory delta: {np.max(memory_deltas):.1f}MB")

        return np.mean(durations)

    finally:
        # Clean up
        import os
        if os.path.exists(test_path):
            os.unlink(test_path)

def check_performance_targets():
    """Check if benchmarks meet performance targets"""
    print("\n📊 Performance Target Analysis...")

    # Define performance targets from config
    targets = {
        'benchmark_extract_features': {'max_duration': 0.1, 'max_memory': 50},
        'benchmark_classify': {'max_duration': 0.001, 'max_memory': 1},
        'benchmark_optimize': {'max_duration': 0.01, 'max_memory': 5},
        'benchmark_predict_quality': {'max_duration': 0.01, 'max_memory': 10},
        'benchmark_complete_pipeline': {'max_duration': 1.0, 'max_memory': 100}
    }

    results = performance_monitor.check_performance_targets(targets)

    for operation, meets_target in results.items():
        status = "✅ PASS" if meets_target else "❌ FAIL"
        print(f"  {operation}: {status}")

    return results

def main():
    parser = argparse.ArgumentParser(description='AI Performance Benchmarking Suite')
    parser.add_argument('--iterations', type=int, default=10, help='Number of benchmark iterations')
    parser.add_argument('--pipeline-iterations', type=int, default=5, help='Pipeline benchmark iterations')
    parser.add_argument('--report', help='Save detailed report to file')

    args = parser.parse_args()

    print("🏃‍♂️ AI Performance Benchmarking Suite")
    print("=" * 50)

    # Reset previous metrics
    performance_monitor.reset_metrics()

    # Run benchmarks
    feature_time = benchmark_feature_extraction(args.iterations)
    classify_time = benchmark_classification(args.iterations)
    optimize_time = benchmark_optimization(args.iterations)
    predict_time = benchmark_prediction(args.iterations)
    pipeline_time = benchmark_complete_pipeline(args.pipeline_iterations)

    # Check performance targets
    target_results = check_performance_targets()

    # Summary
    print("\n📈 Performance Summary:")
    print(f"  Feature Extraction: {feature_time:.3f}s avg")
    print(f"  Classification: {classify_time*1000:.2f}ms avg")
    print(f"  Optimization: {optimize_time*1000:.2f}ms avg")
    print(f"  Quality Prediction: {predict_time*1000:.2f}ms avg")
    print(f"  Complete Pipeline: {pipeline_time:.3f}s avg")

    # Overall performance grade
    passed_targets = sum(target_results.values())
    total_targets = len(target_results)
    grade = "A" if passed_targets == total_targets else "B" if passed_targets >= total_targets * 0.8 else "C"

    print(f"\n🎖️  Performance Grade: {grade} ({passed_targets}/{total_targets} targets met)")

    # Save detailed report if requested
    if args.report:
        report = performance_monitor.get_performance_report()
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"\n📋 Detailed report saved to: {args.report}")

if __name__ == "__main__":
    main()