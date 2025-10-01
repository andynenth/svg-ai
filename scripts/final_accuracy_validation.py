#!/usr/bin/env python3
"""
Final Accuracy Validation for Day 7 Testing
Measures classification accuracy on available test dataset
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ai_modules.classification.hybrid_classifier import HybridClassifier

def load_test_dataset() -> List[Tuple[str, str]]:
    """Load available test images with their expected labels"""
    test_cases = []

    # Map of available test images to expected labels
    test_mapping = {
        'test-data/circle_00.png': 'simple',
        'test-data/text_tech_00.png': 'text',
        'test-data/gradient_radial_00.png': 'gradient',

        # Check for additional test images in data/logos structure
        'data/logos/simple/simple_geometric_00.png': 'simple',
        'data/logos/text/text_based_00.png': 'text',
        'data/logos/gradient/gradient_00.png': 'gradient',
        'data/logos/complex/complex_multi_01.png': 'complex',
        'data/logos/complex/complex_multi_02.png': 'complex',
        'data/logos/complex/complex_multi_03.png': 'complex',
    }

    for image_path, expected_label in test_mapping.items():
        if os.path.exists(image_path):
            test_cases.append((image_path, expected_label))

    print(f"Found {len(test_cases)} test images:")
    for img_path, label in test_cases:
        print(f"  {img_path} -> {label}")

    return test_cases

def validate_classification_accuracy():
    """Comprehensive accuracy validation"""
    print("=" * 70)
    print("FINAL CLASSIFICATION ACCURACY VALIDATION")
    print("=" * 70)

    # Load test dataset
    test_dataset = load_test_dataset()

    if not test_dataset:
        print("❌ No test images found for validation")
        return None

    # Initialize classifier
    classifier = HybridClassifier()

    # Run accuracy tests
    results = {
        'test_cases': [],
        'overall_accuracy': 0.0,
        'per_category_accuracy': {},
        'method_usage': {'rule_based': 0, 'neural_network': 0, 'ensemble': 0},
        'confidence_analysis': {'high': 0, 'medium': 0, 'low': 0},
        'processing_times': [],
        'total_time': 0.0
    }

    correct_predictions = 0
    category_stats = {}

    start_time = time.time()

    print("\nRunning classification tests...")
    print("-" * 70)

    for i, (image_path, true_label) in enumerate(test_dataset):
        print(f"Test {i+1}/{len(test_dataset)}: {os.path.basename(image_path)}")

        # Classify image
        result = classifier.classify_safe(image_path)

        # Extract results
        predicted_label = result.get('logo_type', 'unknown')
        confidence = result.get('confidence', 0.0)
        method_used = result.get('method_used', 'unknown')
        processing_time = result.get('processing_time', 0.0)
        is_correct = predicted_label == true_label

        if is_correct:
            correct_predictions += 1

        # Update category statistics
        if true_label not in category_stats:
            category_stats[true_label] = {'total': 0, 'correct': 0}
        category_stats[true_label]['total'] += 1
        if is_correct:
            category_stats[true_label]['correct'] += 1

        # Update method usage
        if 'rule_based' in method_used:
            results['method_usage']['rule_based'] += 1
        elif 'neural' in method_used:
            results['method_usage']['neural_network'] += 1
        elif 'ensemble' in method_used:
            results['method_usage']['ensemble'] += 1

        # Update confidence analysis
        if confidence >= 0.8:
            results['confidence_analysis']['high'] += 1
        elif confidence >= 0.6:
            results['confidence_analysis']['medium'] += 1
        else:
            results['confidence_analysis']['low'] += 1

        # Store detailed result
        test_case_result = {
            'image': image_path,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'method_used': method_used,
            'correct': is_correct,
            'processing_time': processing_time
        }
        results['test_cases'].append(test_case_result)
        results['processing_times'].append(processing_time)

        # Print result
        status = "✅" if is_correct else "❌"
        print(f"  {status} Predicted: {predicted_label} (confidence: {confidence:.3f}, "
              f"method: {method_used}, time: {processing_time:.3f}s)")

    total_time = time.time() - start_time

    # Calculate final metrics
    results['overall_accuracy'] = correct_predictions / len(test_dataset)
    results['total_time'] = total_time
    results['average_time'] = sum(results['processing_times']) / len(results['processing_times'])

    # Calculate per-category accuracy
    for category, stats in category_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        results['per_category_accuracy'][category] = accuracy

    return results

def print_validation_summary(results: Dict):
    """Print comprehensive validation summary"""
    if not results:
        return

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    # Overall accuracy
    overall_acc = results['overall_accuracy'] * 100
    print(f"Overall Accuracy: {overall_acc:.1f}%")

    if overall_acc >= 92:
        print("✅ EXCELLENT - Exceeds 92% target")
    elif overall_acc >= 85:
        print("✅ GOOD - Above 85% threshold")
    elif overall_acc >= 75:
        print("⚠️  ACCEPTABLE - Above 75% threshold")
    else:
        print("❌ NEEDS IMPROVEMENT - Below 75%")

    # Per-category accuracy
    print(f"\nPer-Category Accuracy:")
    for category, accuracy in results['per_category_accuracy'].items():
        acc_pct = accuracy * 100
        status = "✅" if acc_pct >= 85 else "⚠️" if acc_pct >= 70 else "❌"
        print(f"  {status} {category}: {acc_pct:.1f}%")

    # Method usage
    print(f"\nMethod Usage Distribution:")
    total_tests = len(results['test_cases'])
    for method, count in results['method_usage'].items():
        percentage = (count / total_tests) * 100 if total_tests > 0 else 0
        print(f"  {method}: {count}/{total_tests} ({percentage:.1f}%)")

    # Confidence analysis
    print(f"\nConfidence Distribution:")
    for level, count in results['confidence_analysis'].items():
        percentage = (count / total_tests) * 100 if total_tests > 0 else 0
        print(f"  {level} confidence (≥0.8/≥0.6): {count}/{total_tests} ({percentage:.1f}%)")

    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  Total time: {results['total_time']:.3f}s")
    print(f"  Average time per image: {results['average_time']:.3f}s")
    print(f"  Throughput: {len(results['test_cases']) / results['total_time']:.1f} images/second")

    # Performance targets validation
    print(f"\nTarget Validation:")
    avg_time = results['average_time']
    if avg_time < 2.0:
        print(f"✅ Processing time target met: {avg_time:.3f}s < 2.0s")
    else:
        print(f"❌ Processing time target missed: {avg_time:.3f}s > 2.0s")

    # High confidence accuracy
    high_conf_cases = [case for case in results['test_cases'] if case['confidence'] >= 0.8]
    if high_conf_cases:
        high_conf_correct = sum(1 for case in high_conf_cases if case['correct'])
        high_conf_accuracy = high_conf_correct / len(high_conf_cases)
        print(f"High confidence accuracy: {high_conf_accuracy*100:.1f}% ({high_conf_correct}/{len(high_conf_cases)})")

        if high_conf_accuracy >= 0.95:
            print("✅ High confidence accuracy target met")
        else:
            print("❌ High confidence accuracy target missed")

def cross_validate_with_repeated_tests():
    """Cross-validation with repeated tests for reliability"""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION WITH REPEATED TESTS")
    print("=" * 70)

    test_dataset = load_test_dataset()
    if len(test_dataset) < 3:
        print("Insufficient test data for cross-validation")
        return

    classifier = HybridClassifier()

    # Run each test case multiple times
    repeated_results = {}

    for image_path, true_label in test_dataset[:3]:  # Test first 3 images multiple times
        print(f"\nRepeated testing: {os.path.basename(image_path)}")
        results_for_image = []

        for run in range(3):
            result = classifier.classify_safe(image_path)
            predicted_label = result.get('logo_type', 'unknown')
            confidence = result.get('confidence', 0.0)
            is_correct = predicted_label == true_label

            results_for_image.append({
                'predicted_label': predicted_label,
                'confidence': confidence,
                'correct': is_correct,
                'run': run + 1
            })

            print(f"  Run {run + 1}: {predicted_label} ({confidence:.3f}) {'✅' if is_correct else '❌'}")

        # Analyze consistency
        predictions = [r['predicted_label'] for r in results_for_image]
        unique_predictions = set(predictions)
        consistency = len(unique_predictions) == 1

        repeated_results[image_path] = {
            'true_label': true_label,
            'results': results_for_image,
            'consistent': consistency,
            'most_common_prediction': max(set(predictions), key=predictions.count)
        }

        print(f"  Consistency: {'✅ Consistent' if consistency else '❌ Inconsistent'}")

    # Summary
    consistent_count = sum(1 for r in repeated_results.values() if r['consistent'])
    consistency_rate = consistent_count / len(repeated_results)

    print(f"\nCross-validation summary:")
    print(f"  Consistent predictions: {consistent_count}/{len(repeated_results)} ({consistency_rate*100:.1f}%)")

    if consistency_rate >= 0.8:
        print("✅ Good prediction consistency")
    else:
        print("⚠️  Prediction consistency needs improvement")

    return repeated_results

if __name__ == "__main__":
    try:
        # Run accuracy validation
        results = validate_classification_accuracy()

        if results:
            print_validation_summary(results)

            # Cross-validation
            cross_validate_with_repeated_tests()

            # Save results
            results_file = 'scripts/final_accuracy_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {results_file}")

    except Exception as e:
        print(f"❌ Accuracy validation failed: {e}")
        import traceback
        traceback.print_exc()