#!/usr/bin/env python3
"""
Hybrid Classification System Performance Testing
Tests accuracy, performance, and routing efficiency as specified in Day 6 plan
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import concurrent.futures
from collections import defaultdict

from backend.ai_modules.classification.hybrid_classifier import HybridClassifier


class HybridPerformanceTester:
    """Comprehensive testing suite for hybrid classification system"""

    def __init__(self, test_data_dir: str = "data/logos"):
        self.test_data_dir = Path(test_data_dir)
        self.hybrid_classifier = HybridClassifier()
        self.test_results = {}
        self.performance_metrics = {}

        # Expected labels for test images (you can update these based on your test data)
        self.test_labels = {
            'circle_00.png': 'simple',
            'text_tech_00.png': 'text',
            'gradient_radial_00.png': 'gradient'
        }

    def load_test_dataset(self) -> List[Tuple[str, str]]:
        """Load test dataset with image paths and true labels"""
        test_images = []

        # Look for test images in test-data directory first
        test_data_path = Path("test-data")
        if test_data_path.exists():
            for image_file in test_data_path.glob("*.png"):
                image_path = str(image_file)
                # Infer label from filename or use mapping
                if image_file.name in self.test_labels:
                    true_label = self.test_labels[image_file.name]
                elif 'circle' in image_file.name or 'simple' in image_file.name:
                    true_label = 'simple'
                elif 'text' in image_file.name:
                    true_label = 'text'
                elif 'gradient' in image_file.name:
                    true_label = 'gradient'
                else:
                    true_label = 'complex'  # default

                test_images.append((image_path, true_label))

        # Look in main data directory if it exists
        if self.test_data_dir.exists():
            for category_dir in self.test_data_dir.iterdir():
                if category_dir.is_dir() and category_dir.name in ['simple', 'text', 'gradient', 'complex']:
                    for image_file in category_dir.glob("*.png"):
                        test_images.append((str(image_file), category_dir.name))

        return test_images

    def test_hybrid_performance(self) -> Dict:
        """
        Test hybrid system performance as specified in Day 6 plan
        Tests accuracy, processing time, and routing decisions
        """
        print("🧪 TESTING HYBRID CLASSIFICATION SYSTEM")
        print("=" * 60)

        test_images = self.load_test_dataset()
        if not test_images:
            print("⚠️  No test images found. Creating synthetic test...")
            return self._run_synthetic_test()

        print(f"Found {len(test_images)} test images")

        results = {
            'total_tests': len(test_images),
            'correct_predictions': 0,
            'method_usage': {'rule_based': 0, 'neural_network': 0, 'ensemble': 0},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'processing_times': [],
            'accuracy_by_method': {},
            'routing_efficiency': {},
            'detailed_results': []
        }

        print("\nRunning individual tests...")
        for i, (image_path, true_label) in enumerate(test_images):
            try:
                start_time = time.time()
                result = self.hybrid_classifier.classify(image_path)
                processing_time = time.time() - start_time

                # Analyze result
                predicted_label = result['logo_type']
                confidence = result['confidence']
                method_used = result['method_used']
                is_correct = predicted_label == true_label

                # Update statistics
                if is_correct:
                    results['correct_predictions'] += 1

                # Method usage tracking
                base_method = self._extract_base_method(method_used)
                results['method_usage'][base_method] += 1

                # Confidence distribution
                if confidence >= 0.8:
                    results['confidence_distribution']['high'] += 1
                elif confidence >= 0.6:
                    results['confidence_distribution']['medium'] += 1
                else:
                    results['confidence_distribution']['low'] += 1

                # Processing time
                results['processing_times'].append(processing_time)

                # Detailed result
                detailed_result = {
                    'image': image_path,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'method_used': method_used,
                    'correct': is_correct,
                    'processing_time': processing_time,
                    'reasoning': result.get('reasoning', '')
                }
                results['detailed_results'].append(detailed_result)

                # Update calibration feedback
                self.hybrid_classifier.update_calibration_feedback(result, true_label)

                print(f"  [{i+1}/{len(test_images)}] {Path(image_path).name}: {predicted_label} "
                      f"({'✓' if is_correct else '✗'}, {confidence:.3f}, {processing_time:.3f}s)")

            except Exception as e:
                print(f"  [{i+1}/{len(test_images)}] {Path(image_path).name}: ERROR - {e}")

        # Calculate final metrics
        self._calculate_performance_metrics(results)

        self.test_results = results
        return results

    def _extract_base_method(self, method_used: str) -> str:
        """Extract base method from detailed method name"""
        if 'rule_based' in method_used or method_used == 'rule_based':
            return 'rule_based'
        elif 'neural_network' in method_used or method_used == 'neural_network':
            return 'neural_network'
        elif 'ensemble' in method_used:
            return 'ensemble'
        else:
            return 'rule_based'  # default

    def _calculate_performance_metrics(self, results: Dict):
        """Calculate comprehensive performance metrics"""
        total_tests = results['total_tests']
        if total_tests == 0:
            return

        # Overall accuracy
        results['accuracy'] = results['correct_predictions'] / total_tests
        results['average_confidence'] = np.mean([r['confidence'] for r in results['detailed_results']])
        results['average_processing_time'] = np.mean(results['processing_times'])
        results['median_processing_time'] = np.median(results['processing_times'])
        results['max_processing_time'] = np.max(results['processing_times'])

        # Method-specific accuracy
        method_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        for result in results['detailed_results']:
            method = self._extract_base_method(result['method_used'])
            method_accuracy[method]['total'] += 1
            if result['correct']:
                method_accuracy[method]['correct'] += 1

        results['accuracy_by_method'] = {}
        for method, stats in method_accuracy.items():
            if stats['total'] > 0:
                results['accuracy_by_method'][method] = stats['correct'] / stats['total']

        # High confidence accuracy
        high_conf_results = [r for r in results['detailed_results'] if r['confidence'] >= 0.8]
        if high_conf_results:
            high_conf_correct = sum(1 for r in high_conf_results if r['correct'])
            results['high_confidence_accuracy'] = high_conf_correct / len(high_conf_results)
        else:
            results['high_confidence_accuracy'] = 0.0

        # Routing efficiency analysis
        results['routing_efficiency'] = self._analyze_routing_efficiency(results['detailed_results'])

    def _analyze_routing_efficiency(self, detailed_results: List[Dict]) -> Dict:
        """Analyze routing decision efficiency"""
        routing_analysis = {
            'total_decisions': len(detailed_results),
            'fast_decisions': 0,  # <0.5s
            'medium_decisions': 0,  # 0.5-2s
            'slow_decisions': 0,  # >2s
            'optimal_decisions': 0,
            'routing_accuracy': {}
        }

        for result in detailed_results:
            processing_time = result['processing_time']
            confidence = result['confidence']
            is_correct = result['correct']

            # Categorize by processing time
            if processing_time < 0.5:
                routing_analysis['fast_decisions'] += 1
            elif processing_time < 2.0:
                routing_analysis['medium_decisions'] += 1
            else:
                routing_analysis['slow_decisions'] += 1

            # Optimal decision heuristic: high confidence + correct + fast
            if confidence >= 0.8 and is_correct and processing_time < 1.0:
                routing_analysis['optimal_decisions'] += 1

        # Calculate routing efficiency percentage
        total = routing_analysis['total_decisions']
        if total > 0:
            routing_analysis['fast_percentage'] = (routing_analysis['fast_decisions'] / total) * 100
            routing_analysis['optimal_percentage'] = (routing_analysis['optimal_decisions'] / total) * 100

        return routing_analysis

    def _run_synthetic_test(self) -> Dict:
        """Run synthetic test with available images"""
        print("Running synthetic test with available images...")

        # Use available test images
        test_images = [
            ("test-data/circle_00.png", "simple"),
            ("test-data/text_tech_00.png", "text"),
            ("test-data/gradient_radial_00.png", "gradient")
        ]

        # Filter to only existing images
        existing_images = []
        for image_path, label in test_images:
            if os.path.exists(image_path):
                existing_images.append((image_path, label))

        if not existing_images:
            return {
                'error': 'No test images available',
                'total_tests': 0,
                'accuracy': 0.0
            }

        # Run tests on existing images multiple times to simulate larger dataset
        synthetic_tests = []
        for _ in range(5):  # Repeat each test 5 times
            synthetic_tests.extend(existing_images)

        results = {
            'total_tests': len(synthetic_tests),
            'correct_predictions': 0,
            'method_usage': {'rule_based': 0, 'neural_network': 0, 'ensemble': 0},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'processing_times': [],
            'detailed_results': []
        }

        for i, (image_path, true_label) in enumerate(synthetic_tests):
            try:
                result = self.hybrid_classifier.classify(image_path)
                processing_time = result['processing_time']

                is_correct = result['logo_type'] == true_label
                if is_correct:
                    results['correct_predictions'] += 1

                base_method = self._extract_base_method(result['method_used'])
                results['method_usage'][base_method] += 1

                confidence = result['confidence']
                if confidence >= 0.8:
                    results['confidence_distribution']['high'] += 1
                elif confidence >= 0.6:
                    results['confidence_distribution']['medium'] += 1
                else:
                    results['confidence_distribution']['low'] += 1

                results['processing_times'].append(processing_time)

            except Exception as e:
                print(f"Synthetic test error: {e}")

        self._calculate_performance_metrics(results)
        return results

    def test_time_budget_constraints(self) -> Dict:
        """Test time budget constraint functionality"""
        print("\n⏰ TESTING TIME BUDGET CONSTRAINTS")
        print("-" * 40)

        if not os.path.exists("test-data/circle_00.png"):
            return {'error': 'No test image available'}

        test_image = "test-data/circle_00.png"
        time_budgets = [0.1, 0.5, 1.0, 2.0, 5.0, None]
        budget_results = {}

        for budget in time_budgets:
            try:
                start_time = time.time()
                result = self.hybrid_classifier.classify(test_image, time_budget=budget)
                actual_time = time.time() - start_time

                budget_results[str(budget)] = {
                    'time_budget': budget,
                    'actual_time': actual_time,
                    'method_used': result['method_used'],
                    'confidence': result['confidence'],
                    'within_budget': budget is None or actual_time <= (budget + 0.5)  # Allow 0.5s tolerance
                }

                budget_str = f"{budget}s" if budget else "None"
                status = "✓" if budget_results[str(budget)]['within_budget'] else "✗"
                print(f"  Budget {budget_str:>5}: {actual_time:.3f}s ({result['method_used']}) {status}")

            except Exception as e:
                print(f"  Budget {budget}: ERROR - {e}")

        return budget_results

    def test_concurrent_performance(self, num_concurrent: int = 5) -> Dict:
        """Test concurrent classification performance"""
        print(f"\n🔀 TESTING CONCURRENT PERFORMANCE ({num_concurrent} simultaneous)")
        print("-" * 50)

        if not os.path.exists("test-data/circle_00.png"):
            return {'error': 'No test image available'}

        test_image = "test-data/circle_00.png"
        concurrent_results = []

        def classify_image():
            start_time = time.time()
            result = self.hybrid_classifier.classify(test_image)
            end_time = time.time()
            return {
                'processing_time': end_time - start_time,
                'method_used': result['method_used'],
                'confidence': result['confidence']
            }

        # Run concurrent tests
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(classify_image) for _ in range(num_concurrent)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    concurrent_results.append(result)
                except Exception as e:
                    print(f"Concurrent test error: {e}")

        total_time = time.time() - start_time

        if concurrent_results:
            avg_time = np.mean([r['processing_time'] for r in concurrent_results])
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Average per request: {avg_time:.3f}s")
            print(f"  Throughput: {len(concurrent_results)/total_time:.1f} requests/second")

        return {
            'num_concurrent': num_concurrent,
            'total_time': total_time,
            'results': concurrent_results,
            'average_processing_time': np.mean([r['processing_time'] for r in concurrent_results]) if concurrent_results else 0,
            'throughput': len(concurrent_results) / total_time if total_time > 0 else 0
        }

    def validate_success_criteria(self) -> Dict:
        """Validate against Day 6 success criteria"""
        print("\n📊 VALIDATING SUCCESS CRITERIA")
        print("-" * 40)

        if not self.test_results:
            print("⚠️  No test results available. Running basic test...")
            self.test_hybrid_performance()

        criteria = {
            'hybrid_accuracy_95': {
                'target': 0.95,
                'actual': self.test_results.get('accuracy', 0.0),
                'status': 'PASS' if self.test_results.get('accuracy', 0.0) >= 0.95 else 'FAIL'
            },
            'avg_processing_time_2s': {
                'target': 2.0,
                'actual': self.test_results.get('average_processing_time', 0.0),
                'status': 'PASS' if self.test_results.get('average_processing_time', 0.0) <= 2.0 else 'FAIL'
            },
            'high_confidence_accuracy_95': {
                'target': 0.95,
                'actual': self.test_results.get('high_confidence_accuracy', 0.0),
                'status': 'PASS' if self.test_results.get('high_confidence_accuracy', 0.0) >= 0.95 else 'FAIL'
            },
            'routing_efficiency_90': {
                'target': 0.90,
                'actual': self.test_results.get('routing_efficiency', {}).get('optimal_percentage', 0) / 100,
                'status': 'PASS' if self.test_results.get('routing_efficiency', {}).get('optimal_percentage', 0) >= 90 else 'FAIL'
            }
        }

        for criterion, result in criteria.items():
            status_symbol = "✅" if result['status'] == 'PASS' else "❌"
            print(f"  {status_symbol} {criterion}: {result['actual']:.3f} (target: {result['target']})")

        # Overall assessment
        passed_criteria = sum(1 for c in criteria.values() if c['status'] == 'PASS')
        total_criteria = len(criteria)

        overall_status = "EXCELLENT" if passed_criteria == total_criteria else \
                        "GOOD" if passed_criteria >= 3 else \
                        "FAIR" if passed_criteria >= 2 else "POOR"

        print(f"\n📈 OVERALL ASSESSMENT: {overall_status} ({passed_criteria}/{total_criteria} criteria passed)")

        return {
            'criteria': criteria,
            'passed': passed_criteria,
            'total': total_criteria,
            'overall_status': overall_status
        }

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("🎯 HYBRID CLASSIFIER PERFORMANCE REPORT")
        report.append("=" * 50)

        if not self.test_results:
            report.append("No test results available.")
            return "\n".join(report)

        # Overall Performance
        report.append(f"\n📊 OVERALL PERFORMANCE")
        report.append(f"Total tests: {self.test_results['total_tests']}")
        report.append(f"Accuracy: {self.test_results.get('accuracy', 0)*100:.1f}%")
        report.append(f"Average confidence: {self.test_results.get('average_confidence', 0):.3f}")
        report.append(f"Average processing time: {self.test_results.get('average_processing_time', 0):.3f}s")

        # Method Usage
        report.append(f"\n🔀 METHOD USAGE")
        for method, count in self.test_results['method_usage'].items():
            percentage = (count / self.test_results['total_tests']) * 100
            report.append(f"{method}: {count} ({percentage:.1f}%)")

        # Method Accuracy
        if 'accuracy_by_method' in self.test_results:
            report.append(f"\n🎯 METHOD ACCURACY")
            for method, accuracy in self.test_results['accuracy_by_method'].items():
                report.append(f"{method}: {accuracy*100:.1f}%")

        # Performance Stats
        performance_stats = self.hybrid_classifier.get_performance_stats()
        if 'cache_hit_rate' in performance_stats:
            report.append(f"\n⚡ CACHE PERFORMANCE")
            report.append(f"Cache hit rate: {performance_stats['cache_hit_rate']:.1f}%")

        return "\n".join(report)


def main():
    """Main testing function"""
    tester = HybridPerformanceTester()

    # Run comprehensive testing
    print("Starting comprehensive hybrid classifier testing...\n")

    # 1. Basic performance test
    results = tester.test_hybrid_performance()

    # 2. Time budget test
    budget_results = tester.test_time_budget_constraints()

    # 3. Concurrent performance test
    concurrent_results = tester.test_concurrent_performance()

    # 4. Success criteria validation
    criteria_results = tester.validate_success_criteria()

    # 5. Generate report
    print("\n" + "=" * 60)
    print(tester.generate_performance_report())

    # Save results
    all_results = {
        'performance_test': results,
        'time_budget_test': budget_results,
        'concurrent_test': concurrent_results,
        'success_criteria': criteria_results,
        'timestamp': time.time()
    }

    with open('hybrid_performance_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: hybrid_performance_results.json")
    print("✅ Testing completed successfully!")


if __name__ == "__main__":
    main()