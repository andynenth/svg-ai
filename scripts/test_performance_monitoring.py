#!/usr/bin/env python3
"""Test Performance Monitoring on AI Modules"""

import tempfile
import cv2
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_modules.utils.performance_monitor import (
    performance_monitor,
    monitor_performance,
    monitor_feature_extraction,
    monitor_classification,
    monitor_optimization,
    monitor_prediction
)
from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
from backend.ai_modules.prediction.quality_predictor import QualityPredictor

def test_decorator_functionality():
    """Test that performance monitoring decorators work correctly"""
    print("🧪 Testing Performance Monitor Decorators...")

    # Test generic monitor decorator
    @monitor_performance("test_operation")
    def test_function(x, y):
        import time
        time.sleep(0.01)  # Simulate work
        return x + y

    result = test_function(5, 3)
    assert result == 8
    print("  ✅ Generic monitor decorator works")

    # Test specific decorators
    @monitor_feature_extraction
    def mock_feature_extraction():
        return {'test_feature': 0.5}

    @monitor_classification
    def mock_classification():
        return 'simple', 0.8

    @monitor_optimization
    def mock_optimization():
        return {'color_precision': 4}

    @monitor_prediction
    def mock_prediction():
        return 0.85

    # Execute monitored functions
    mock_feature_extraction()
    mock_classification()
    mock_optimization()
    mock_prediction()

    print("  ✅ All specific decorators work")

def test_metrics_collection():
    """Test that metrics are collected correctly"""
    print("\n📊 Testing Metrics Collection...")

    # Reset metrics
    performance_monitor.reset_metrics()

    # Create test image and run operations
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_path = tempfile.mktemp(suffix='.png')
    cv2.imwrite(test_path, test_image)

    try:
        # Run monitored operations
        extractor = ImageFeatureExtractor()
        classifier = RuleBasedClassifier()
        optimizer = FeatureMappingOptimizer()
        predictor = QualityPredictor()

        # Extract features with monitoring
        @monitor_feature_extraction
        def extract_features():
            return extractor.extract_features(test_path)

        # Classify with monitoring
        @monitor_classification
        def classify_features():
            features = extract_features()
            return classifier.classify(features)

        # Optimize with monitoring
        @monitor_optimization
        def optimize_params():
            features = extract_features()
            logo_type, _ = classify_features()
            return optimizer.optimize(features, logo_type)

        # Predict with monitoring
        @monitor_prediction
        def predict_quality():
            features = extract_features()
            params = optimize_params()
            return predictor.predict_quality(features, params)

        # Execute pipeline
        features = extract_features()
        logo_type, confidence = classify_features()
        params = optimize_params()
        quality = predict_quality()

        print(f"  ✅ Pipeline executed: {logo_type} logo, quality: {quality:.3f}")

        # Check metrics were collected
        summary = performance_monitor.get_summary()
        print(f"  ✅ Metrics collected: {summary['total_operations']} operations")

        # Check specific operation metrics
        for operation in ['feature_extraction', 'classification', 'optimization', 'prediction']:
            op_summary = performance_monitor.get_summary(operation)
            if op_summary:
                print(f"    {operation}: {op_summary['average_duration']:.3f}s avg")

    finally:
        import os
        if os.path.exists(test_path):
            os.unlink(test_path)

def test_performance_reporting():
    """Test performance reporting functionality"""
    print("\n📋 Testing Performance Reporting...")

    # Generate a performance report
    report = performance_monitor.get_performance_report()
    print("  ✅ Performance report generated")
    print(f"  Report length: {len(report)} characters")

    # Test detailed metrics
    detailed = performance_monitor.get_detailed_metrics()
    print(f"  ✅ Detailed metrics: {len(detailed)} operations tracked")

    # Test performance targets
    targets = {
        'feature_extraction': {'max_duration': 0.2, 'max_memory': 100},
        'classification': {'max_duration': 0.01, 'max_memory': 10},
        'optimization': {'max_duration': 0.05, 'max_memory': 20},
        'prediction': {'max_duration': 0.05, 'max_memory': 50}
    }

    target_results = performance_monitor.check_performance_targets(targets)
    passed = sum(target_results.values())
    total = len(target_results)
    print(f"  ✅ Performance targets: {passed}/{total} met")

def test_memory_tracking():
    """Test memory usage tracking"""
    print("\n💾 Testing Memory Tracking...")

    @monitor_performance("memory_test")
    def memory_intensive_operation():
        # Create some data to use memory
        data = np.random.rand(1000, 1000)
        return np.sum(data)

    result = memory_intensive_operation()
    print(f"  ✅ Memory intensive operation completed: {result:.2f}")

    # Check memory metrics
    memory_summary = performance_monitor.get_summary("memory_test")
    if memory_summary:
        print(f"  Memory delta: {memory_summary['average_memory_delta']:.1f}MB")

def test_error_handling():
    """Test error handling in performance monitoring"""
    print("\n🚨 Testing Error Handling...")

    @monitor_performance("error_test")
    def failing_operation():
        raise ValueError("Test error")

    try:
        failing_operation()
    except ValueError:
        pass  # Expected

    # Check that error was recorded
    error_summary = performance_monitor.get_summary("error_test")
    if error_summary:
        failed_ops = error_summary['total_operations'] - error_summary['successful_operations']
        print(f"  ✅ Error handling: {failed_ops} failed operations recorded")

def main():
    """Run all performance monitoring tests"""
    print("🏃‍♂️ Performance Monitoring Test Suite")
    print("=" * 50)

    test_decorator_functionality()
    test_metrics_collection()
    test_performance_reporting()
    test_memory_tracking()
    test_error_handling()

    print("\n🎉 All performance monitoring tests completed!")

    # Print final report
    print("\n" + "=" * 50)
    print("📊 FINAL PERFORMANCE REPORT")
    print("=" * 50)
    print(performance_monitor.get_performance_report())

if __name__ == "__main__":
    main()