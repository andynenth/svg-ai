#!/usr/bin/env python3
"""
Neural Network Performance Testing Script

Tests inference speed, memory usage, and integration for the trained EfficientNet model.
"""

import torch
import time
import psutil
import os
import sys
import json
from pathlib import Path
import tracemalloc
from PIL import Image

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ai_modules.classification.efficientnet_classifier import EfficientNetClassifier

class PerformanceTester:
    """Tests neural network performance and integration."""

    def __init__(self, model_path: str = 'backend/ai_modules/models/trained/efficientnet_logo_classifier.pth'):
        self.model_path = model_path
        self.test_results = {}

    def create_test_images(self, count: int = 10) -> list:
        """Create test images for performance testing."""
        test_dir = Path('/tmp/claude/nn_performance_test')
        test_dir.mkdir(exist_ok=True, parents=True)

        test_images = []
        colors = ['red', 'green', 'blue', 'yellow', 'purple']

        for i in range(count):
            color = colors[i % len(colors)]
            size = (256 + i * 10, 256 + i * 10)  # Varying sizes

            # Create test image
            image = Image.new('RGB', size, color=color)
            image_path = test_dir / f'test_image_{i:03d}.png'
            image.save(image_path)
            test_images.append(str(image_path))

        return test_images

    def test_model_loading(self):
        """Test model loading time and memory usage."""
        print("=== Testing Model Loading ===")

        # Monitor memory before loading
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        try:
            # Test with trained weights
            classifier = EfficientNetClassifier(
                model_path=self.model_path if os.path.exists(self.model_path) else None,
                use_pretrained=False
            )
            loading_time = time.time() - start_time

            # Monitor memory after loading
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before

            print(f"✓ Model loaded in {loading_time:.3f}s")
            print(f"✓ Memory usage: {memory_used:.1f} MB")

            # Get model info
            model_info = classifier.get_model_info()
            print(f"✓ Model parameters: {model_info['total_parameters']:,}")

            self.test_results['model_loading'] = {
                'loading_time': loading_time,
                'memory_used_mb': memory_used,
                'total_parameters': model_info['total_parameters'],
                'success': True
            }

            return classifier

        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            self.test_results['model_loading'] = {
                'success': False,
                'error': str(e)
            }
            return None

    def test_single_inference_speed(self, classifier, test_images):
        """Test single image inference speed."""
        print("\n=== Testing Single Inference Speed ===")

        if not classifier or not test_images:
            print("✗ Cannot test - classifier or test images missing")
            return

        inference_times = []
        target_time = 5.0  # 5 seconds as per requirements

        for i, image_path in enumerate(test_images[:5]):  # Test first 5 images
            try:
                start_time = time.time()
                result = classifier.classify(image_path)
                inference_time = time.time() - start_time

                inference_times.append(inference_time)

                print(f"  Image {i+1}: {inference_time:.3f}s -> {result['logo_type']} ({result['confidence']:.3f})")

            except Exception as e:
                print(f"  Image {i+1}: ERROR - {e}")

        if inference_times:
            avg_time = sum(inference_times) / len(inference_times)
            max_time = max(inference_times)
            min_time = min(inference_times)

            print(f"\n✓ Average inference time: {avg_time:.3f}s")
            print(f"✓ Min time: {min_time:.3f}s, Max time: {max_time:.3f}s")

            # Check target compliance
            if avg_time < target_time:
                print(f"✓ Meets target (<{target_time}s)")
                target_met = True
            else:
                print(f"⚠ Exceeds target (>{target_time}s)")
                target_met = False

            self.test_results['single_inference'] = {
                'average_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'target_time': target_time,
                'target_met': target_met,
                'test_count': len(inference_times)
            }

    def test_batch_processing(self, classifier, test_images):
        """Test batch processing capabilities."""
        print("\n=== Testing Batch Processing ===")

        if not classifier or not test_images:
            print("✗ Cannot test - classifier or test images missing")
            return

        batch_sizes = [1, 3, 5, 8]
        batch_results = {}

        for batch_size in batch_sizes:
            if batch_size > len(test_images):
                continue

            try:
                test_batch = test_images[:batch_size]

                start_time = time.time()
                results = classifier.classify_batch(test_batch)
                batch_time = time.time() - start_time

                avg_time_per_image = batch_time / batch_size

                print(f"  Batch size {batch_size}: {batch_time:.3f}s total, {avg_time_per_image:.3f}s per image")

                batch_results[batch_size] = {
                    'total_time': batch_time,
                    'time_per_image': avg_time_per_image,
                    'results_count': len(results)
                }

            except Exception as e:
                print(f"  Batch size {batch_size}: ERROR - {e}")
                batch_results[batch_size] = {'error': str(e)}

        self.test_results['batch_processing'] = batch_results

    def test_memory_usage_during_inference(self, classifier, test_images):
        """Test memory usage during inference."""
        print("\n=== Testing Memory Usage During Inference ===")

        if not classifier or not test_images:
            print("✗ Cannot test - classifier or test images missing")
            return

        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_measurements = []

        try:
            # Process multiple images and monitor memory
            for i, image_path in enumerate(test_images):
                result = classifier.classify(image_path)

                # Measure memory every few images
                if i % 2 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_measurements.append(current_memory)

            final_memory = process.memory_info().rss / 1024 / 1024
            max_memory = max(memory_measurements) if memory_measurements else final_memory
            memory_increase = max_memory - initial_memory

            print(f"✓ Initial memory: {initial_memory:.1f} MB")
            print(f"✓ Peak memory: {max_memory:.1f} MB")
            print(f"✓ Memory increase: {memory_increase:.1f} MB")

            # Check if memory usage is reasonable (target: <200MB)
            if max_memory < 200:
                print("✓ Memory usage within target (<200MB)")
                memory_ok = True
            else:
                print(f"⚠ High memory usage: {max_memory:.1f} MB")
                memory_ok = False

            self.test_results['memory_usage'] = {
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': max_memory,
                'memory_increase_mb': memory_increase,
                'memory_target_met': memory_ok,
                'target_mb': 200
            }

        except Exception as e:
            print(f"✗ Memory test failed: {e}")
            self.test_results['memory_usage'] = {'error': str(e)}

        finally:
            tracemalloc.stop()

    def test_output_format_consistency(self, classifier, test_images):
        """Test output format consistency."""
        print("\n=== Testing Output Format Consistency ===")

        if not classifier or not test_images:
            print("✗ Cannot test - classifier or test images missing")
            return

        expected_keys = ['logo_type', 'confidence', 'all_probabilities', 'model_type', 'device']
        format_issues = []

        try:
            # Test single classification format
            single_result = classifier.classify(test_images[0])

            # Check required keys
            for key in expected_keys:
                if key not in single_result:
                    format_issues.append(f"Missing key in single result: {key}")

            # Check data types
            if not isinstance(single_result.get('logo_type'), str):
                format_issues.append("logo_type is not string")

            if not isinstance(single_result.get('confidence'), (int, float)):
                format_issues.append("confidence is not numeric")

            if not isinstance(single_result.get('all_probabilities'), dict):
                format_issues.append("all_probabilities is not dict")

            # Test batch classification format
            batch_results = classifier.classify_batch(test_images[:3])

            if not isinstance(batch_results, list):
                format_issues.append("Batch results is not a list")
            elif len(batch_results) != 3:
                format_issues.append(f"Batch results count mismatch: expected 3, got {len(batch_results)}")
            else:
                # Check each batch result has consistent format
                for i, result in enumerate(batch_results):
                    for key in expected_keys:
                        if key not in result:
                            format_issues.append(f"Missing key in batch result {i}: {key}")

            if format_issues:
                print("✗ Format consistency issues found:")
                for issue in format_issues:
                    print(f"  - {issue}")
            else:
                print("✓ All output formats consistent")

            self.test_results['output_format'] = {
                'consistent': len(format_issues) == 0,
                'issues': format_issues
            }

        except Exception as e:
            print(f"✗ Output format test failed: {e}")
            self.test_results['output_format'] = {'error': str(e)}

    def test_integration_with_existing_pipeline(self, classifier):
        """Test integration with existing classification pipeline."""
        print("\n=== Testing Integration ===")

        if not classifier:
            print("✗ Cannot test - classifier missing")
            return

        try:
            # Test model info retrieval
            model_info = classifier.get_model_info()
            print(f"✓ Model info: {model_info['model_name']}")
            print(f"✓ Classes: {model_info['classes']}")

            # Test error handling
            try:
                result = classifier.classify('/nonexistent/path.png')
                if 'error' in result or result['logo_type'] == 'unknown':
                    print("✓ Error handling works correctly")
                    error_handling_ok = True
                else:
                    print("⚠ Error handling may not be working")
                    error_handling_ok = False
            except:
                print("⚠ Error handling throws exceptions")
                error_handling_ok = False

            self.test_results['integration'] = {
                'model_info_available': True,
                'error_handling_ok': error_handling_ok
            }

        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            self.test_results['integration'] = {'error': str(e)}

    def cleanup_test_images(self, test_images):
        """Clean up test images."""
        for image_path in test_images:
            try:
                os.remove(image_path)
            except:
                pass

        try:
            os.rmdir(os.path.dirname(test_images[0]))
        except:
            pass

    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': self.model_path,
            'test_results': self.test_results,
            'summary': self._generate_summary()
        }

        report_path = 'neural_network_performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Performance report saved: {report_path}")
        return report

    def _generate_summary(self):
        """Generate test summary."""
        summary = {
            'tests_run': len(self.test_results),
            'tests_passed': 0,
            'tests_failed': 0,
            'performance_targets_met': {},
            'overall_status': 'unknown'
        }

        for test_name, result in self.test_results.items():
            if isinstance(result, dict) and 'error' not in result:
                summary['tests_passed'] += 1
            else:
                summary['tests_failed'] += 1

        # Check specific performance targets
        if 'single_inference' in self.test_results:
            inference_result = self.test_results['single_inference']
            summary['performance_targets_met']['inference_speed'] = inference_result.get('target_met', False)

        if 'memory_usage' in self.test_results:
            memory_result = self.test_results['memory_usage']
            summary['performance_targets_met']['memory_usage'] = memory_result.get('memory_target_met', False)

        # Overall status
        if summary['tests_failed'] == 0:
            summary['overall_status'] = 'excellent'
        elif summary['tests_passed'] > summary['tests_failed']:
            summary['overall_status'] = 'good'
        else:
            summary['overall_status'] = 'needs_attention'

        return summary

    def run_all_tests(self):
        """Run all performance tests."""
        print("Neural Network Performance Testing")
        print("=" * 50)

        # Create test images
        print("Creating test images...")
        test_images = self.create_test_images(10)

        try:
            # Test 1: Model Loading
            classifier = self.test_model_loading()

            if classifier:
                # Test 2: Single Inference Speed
                self.test_single_inference_speed(classifier, test_images)

                # Test 3: Batch Processing
                self.test_batch_processing(classifier, test_images)

                # Test 4: Memory Usage
                self.test_memory_usage_during_inference(classifier, test_images)

                # Test 5: Output Format
                self.test_output_format_consistency(classifier, test_images)

                # Test 6: Integration
                self.test_integration_with_existing_pipeline(classifier)

            # Generate report
            report = self.generate_performance_report()

            # Summary
            print("\n" + "=" * 50)
            print("PERFORMANCE TEST SUMMARY")
            print("=" * 50)

            summary = report['summary']
            print(f"Tests run: {summary['tests_run']}")
            print(f"Tests passed: {summary['tests_passed']}")
            print(f"Tests failed: {summary['tests_failed']}")
            print(f"Overall status: {summary['overall_status'].upper()}")

            # Performance targets
            targets_met = summary['performance_targets_met']
            print(f"\nPerformance targets:")
            for target, met in targets_met.items():
                status = "✓ MET" if met else "✗ NOT MET"
                print(f"  {target}: {status}")

            return summary['overall_status'] in ['excellent', 'good']

        finally:
            # Cleanup
            self.cleanup_test_images(test_images)

def main():
    """Main function."""
    tester = PerformanceTester()
    success = tester.run_all_tests()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)