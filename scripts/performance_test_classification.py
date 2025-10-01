#!/usr/bin/env python3
"""
Day 3: Performance Testing for Classification System

Comprehensive performance testing including:
- Single image classification timing
- Concurrent classification testing (10+ simultaneous)
- Memory usage monitoring under load
- Different image size testing
- Validation of <0.5s processing time requirement
"""

import time
import gc
import sys
import os
import json
import threading
import concurrent.futures
import psutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2
from statistics import mean, median, stdev

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.ai_modules.feature_pipeline import FeaturePipeline
from backend.ai_modules.feature_extraction import ImageFeatureExtractor
from backend.ai_modules.rule_based_classifier import RuleBasedClassifier


class PerformanceTestClassification:
    """Comprehensive performance testing for classification system"""

    def __init__(self):
        self.pipeline = FeaturePipeline(cache_enabled=False)
        self.extractor = ImageFeatureExtractor(cache_enabled=False)
        self.classifier = RuleBasedClassifier()
        self.results = {
            'single_image_performance': {},
            'concurrent_performance': {},
            'memory_usage': {},
            'image_size_performance': {},
            'stress_test_results': {},
            'summary': {}
        }
        self.test_images = self._find_test_images()

    def _find_test_images(self) -> List[Dict]:
        """Find test images for performance testing"""
        test_images = []

        # Look for test images in data/logos directory
        logos_dir = Path("data/logos")
        if logos_dir.exists():
            for category_dir in logos_dir.iterdir():
                if category_dir.is_dir():
                    for img_file in list(category_dir.glob("*.png"))[:3]:  # 3 per category
                        test_images.append({
                            'path': str(img_file),
                            'category': category_dir.name,
                            'filename': img_file.name,
                            'size': self._get_image_size(str(img_file))
                        })

        # If no test images found, create synthetic ones
        if not test_images:
            test_images = self._create_synthetic_test_images()

        return test_images[:15]  # Limit to 15 images for performance testing

    def _get_image_size(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions"""
        try:
            img = cv2.imread(image_path)
            if img is not None:
                return img.shape[1], img.shape[0]  # width, height
        except Exception:
            pass
        return (0, 0)

    def _create_synthetic_test_images(self) -> List[Dict]:
        """Create synthetic test images for performance testing"""
        test_images = []
        temp_dir = Path("temp_perf_test_images")
        temp_dir.mkdir(exist_ok=True)

        # Create images of different sizes and types
        test_configs = [
            {'name': 'small_simple', 'size': (50, 50), 'type': 'simple'},
            {'name': 'medium_simple', 'size': (200, 200), 'type': 'simple'},
            {'name': 'large_simple', 'size': (500, 500), 'type': 'simple'},
            {'name': 'small_complex', 'size': (50, 50), 'type': 'complex'},
            {'name': 'medium_complex', 'size': (200, 200), 'type': 'complex'},
            {'name': 'large_complex', 'size': (500, 500), 'type': 'complex'},
        ]

        for config in test_configs:
            img = self._create_synthetic_image(config['size'], config['type'])
            img_path = temp_dir / f"{config['name']}.png"
            cv2.imwrite(str(img_path), img)

            test_images.append({
                'path': str(img_path),
                'category': config['type'],
                'filename': img_path.name,
                'size': config['size']
            })

        return test_images

    def _create_synthetic_image(self, size: Tuple[int, int], image_type: str) -> np.ndarray:
        """Create synthetic test image of specified type"""
        width, height = size
        img = np.zeros((height, width, 3), dtype=np.uint8)

        if image_type == 'simple':
            # Simple geometric shape
            center = (width // 2, height // 2)
            radius = min(width, height) // 4
            cv2.circle(img, center, radius, (255, 255, 255), -1)

        elif image_type == 'complex':
            # Complex pattern with many features
            # Add random rectangles
            for _ in range(10):
                x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
                x2, y2 = np.random.randint(x1, width), np.random.randint(y1, height)
                color = tuple(np.random.randint(0, 256, 3).tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            # Add random lines
            for _ in range(20):
                x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
                x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
                color = tuple(np.random.randint(0, 256, 3).tolist())
                cv2.line(img, (x1, y1), (x2, y2), color, 2)

        return img

    def test_single_image_performance(self) -> Dict[str, Any]:
        """Test classification time for single images"""
        print("🔍 Testing single image performance...")

        single_times = []
        feature_times = []
        classification_times = []
        detailed_results = []

        for img_info in self.test_images[:10]:  # Test 10 images
            # Test complete pipeline
            start_time = time.perf_counter()
            result = self.pipeline.process_image(img_info['path'])
            total_time = time.perf_counter() - start_time

            single_times.append(total_time)

            # Test individual components
            feature_start = time.perf_counter()
            features = self.extractor.extract_features(img_info['path'])
            feature_time = time.perf_counter() - feature_start
            feature_times.append(feature_time)

            classification_start = time.perf_counter()
            classification = self.classifier.classify(features)
            classification_time = time.perf_counter() - classification_start
            classification_times.append(classification_time)

            detailed_results.append({
                'filename': img_info['filename'],
                'total_time': total_time,
                'feature_time': feature_time,
                'classification_time': classification_time,
                'classification': classification.get('logo_type', 'unknown'),
                'confidence': classification.get('confidence', 0.0),
                'image_size': img_info['size']
            })

            print(f"  {img_info['filename']}: {total_time:.4f}s "
                  f"(features: {feature_time:.4f}s, classification: {classification_time:.4f}s)")

        # Calculate statistics
        results = {
            'total_processing': {
                'mean': mean(single_times),
                'median': median(single_times),
                'min': min(single_times),
                'max': max(single_times),
                'std': stdev(single_times) if len(single_times) > 1 else 0,
                'all_under_target': all(t < 0.5 for t in single_times),
                'target_compliance_rate': sum(1 for t in single_times if t < 0.5) / len(single_times)
            },
            'feature_extraction': {
                'mean': mean(feature_times),
                'median': median(feature_times),
                'min': min(feature_times),
                'max': max(feature_times)
            },
            'classification': {
                'mean': mean(classification_times),
                'median': median(classification_times),
                'min': min(classification_times),
                'max': max(classification_times)
            },
            'detailed_results': detailed_results
        }

        self.results['single_image_performance'] = results

        print(f"✅ Single image performance complete:")
        print(f"   Average processing time: {results['total_processing']['mean']:.4f}s")
        print(f"   Target compliance (<0.5s): {results['total_processing']['target_compliance_rate']:.1%}")

        return results

    def test_concurrent_classification(self, num_workers: int = 10) -> Dict[str, Any]:
        """Test concurrent classification with multiple simultaneous requests"""
        print(f"🔍 Testing concurrent classification with {num_workers} workers...")

        def classify_image(image_path: str) -> Dict[str, Any]:
            """Classify a single image and return timing info"""
            start_time = time.perf_counter()

            try:
                result = self.pipeline.process_image(image_path)
                end_time = time.perf_counter()

                # Handle different result formats
                classification = result.get('classification', {})
                if isinstance(classification, dict):
                    logo_type = classification.get('logo_type', 'unknown')
                    confidence = classification.get('confidence', 0.0)
                else:
                    # Fallback for unexpected format
                    logo_type = 'unknown'
                    confidence = 0.0

                return {
                    'processing_time': end_time - start_time,
                    'classification': logo_type,
                    'confidence': confidence,
                    'thread_id': threading.current_thread().ident
                }
            except Exception as e:
                end_time = time.perf_counter()
                return {
                    'processing_time': end_time - start_time,
                    'classification': 'error',
                    'confidence': 0.0,
                    'thread_id': threading.current_thread().ident,
                    'error': str(e)
                }

        # Prepare test images (repeat if needed)
        test_paths = [img['path'] for img in self.test_images]
        while len(test_paths) < num_workers:
            test_paths.extend([img['path'] for img in self.test_images])
        test_paths = test_paths[:num_workers]

        # Test concurrent processing
        start_time = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(classify_image, path) for path in test_paths]
            concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        end_time = time.perf_counter()
        total_concurrent_time = end_time - start_time

        # Analyze results
        processing_times = [r['processing_time'] for r in concurrent_results]
        thread_ids = set(r['thread_id'] for r in concurrent_results)

        results = {
            'num_workers': num_workers,
            'total_concurrent_time': total_concurrent_time,
            'individual_processing_times': {
                'mean': mean(processing_times),
                'median': median(processing_times),
                'min': min(processing_times),
                'max': max(processing_times),
                'std': stdev(processing_times) if len(processing_times) > 1 else 0
            },
            'concurrent_efficiency': {
                'theoretical_sequential_time': sum(processing_times),
                'actual_concurrent_time': total_concurrent_time,
                'speedup_factor': sum(processing_times) / total_concurrent_time,
                'efficiency': (sum(processing_times) / total_concurrent_time) / num_workers
            },
            'thread_utilization': {
                'unique_threads_used': len(thread_ids),
                'expected_threads': min(num_workers, os.cpu_count() or 1)
            },
            'target_compliance': {
                'all_under_target': all(t < 0.5 for t in processing_times),
                'compliance_rate': sum(1 for t in processing_times if t < 0.5) / len(processing_times)
            },
            'detailed_results': concurrent_results
        }

        self.results['concurrent_performance'] = results

        print(f"✅ Concurrent performance complete:")
        print(f"   Total concurrent time: {total_concurrent_time:.4f}s")
        print(f"   Average individual time: {results['individual_processing_times']['mean']:.4f}s")
        print(f"   Speedup factor: {results['concurrent_efficiency']['speedup_factor']:.2f}x")
        print(f"   Target compliance: {results['target_compliance']['compliance_rate']:.1%}")

        return results

    def test_memory_usage_under_load(self) -> Dict[str, Any]:
        """Monitor memory usage under various load conditions"""
        print("🔍 Testing memory usage under load...")

        process = psutil.Process(os.getpid())

        # Baseline memory
        gc.collect()  # Force garbage collection
        baseline_memory = process.memory_info().rss

        memory_measurements = []

        # Test memory usage during single image processing
        for i, img_info in enumerate(self.test_images[:10]):
            before_memory = process.memory_info().rss

            result = self.pipeline.process_image(img_info['path'])

            after_memory = process.memory_info().rss
            memory_increase = after_memory - before_memory

            # Handle classification result safely
            classification = result.get('classification', {})
            logo_type = classification.get('logo_type', 'unknown') if isinstance(classification, dict) else 'unknown'

            memory_measurements.append({
                'iteration': i,
                'before_memory_mb': before_memory / 1024 / 1024,
                'after_memory_mb': after_memory / 1024 / 1024,
                'memory_increase_mb': memory_increase / 1024 / 1024,
                'classification': logo_type
            })

            if i % 3 == 0:  # Garbage collect every 3 iterations
                gc.collect()

        # Test memory usage during concurrent processing
        before_concurrent = process.memory_info().rss

        def classify_for_memory_test(image_path):
            return self.pipeline.process_image(image_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(classify_for_memory_test, img['path'])
                      for img in self.test_images[:5]]
            concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        after_concurrent = process.memory_info().rss
        concurrent_memory_increase = after_concurrent - before_concurrent

        # Final memory measurement
        gc.collect()
        final_memory = process.memory_info().rss

        results = {
            'baseline_memory_mb': baseline_memory / 1024 / 1024,
            'final_memory_mb': final_memory / 1024 / 1024,
            'total_memory_increase_mb': (final_memory - baseline_memory) / 1024 / 1024,
            'single_image_processing': {
                'measurements': memory_measurements,
                'avg_memory_increase_mb': mean([m['memory_increase_mb'] for m in memory_measurements]),
                'max_memory_increase_mb': max([m['memory_increase_mb'] for m in memory_measurements]),
                'memory_stable': abs(memory_measurements[-1]['after_memory_mb'] -
                                   memory_measurements[0]['before_memory_mb']) < 50  # Within 50MB
            },
            'concurrent_processing': {
                'memory_increase_mb': concurrent_memory_increase / 1024 / 1024,
                'memory_efficient': concurrent_memory_increase < 100 * 1024 * 1024  # Less than 100MB
            },
            'memory_leak_assessment': {
                'potential_leak': (final_memory - baseline_memory) > 100 * 1024 * 1024,  # >100MB
                'memory_growth_rate_mb_per_operation': ((final_memory - baseline_memory) / 1024 / 1024) / len(memory_measurements)
            }
        }

        self.results['memory_usage'] = results

        print(f"✅ Memory usage test complete:")
        print(f"   Baseline memory: {results['baseline_memory_mb']:.1f}MB")
        print(f"   Final memory: {results['final_memory_mb']:.1f}MB")
        print(f"   Total increase: {results['total_memory_increase_mb']:.1f}MB")
        print(f"   Memory stable: {results['single_image_processing']['memory_stable']}")

        return results

    def test_different_image_sizes(self) -> Dict[str, Any]:
        """Test performance with different image sizes"""
        print("🔍 Testing performance with different image sizes...")

        size_categories = {
            'small': [],    # < 100x100
            'medium': [],   # 100x100 to 400x400
            'large': []     # > 400x400
        }

        # Categorize images by size
        for img_info in self.test_images:
            width, height = img_info['size']
            area = width * height

            if area < 10000:  # < 100x100
                size_categories['small'].append(img_info)
            elif area < 160000:  # < 400x400
                size_categories['medium'].append(img_info)
            else:
                size_categories['large'].append(img_info)

        results = {}

        for category, images in size_categories.items():
            if not images:
                continue

            processing_times = []
            for img_info in images[:5]:  # Test up to 5 images per category
                start_time = time.perf_counter()
                result = self.pipeline.process_image(img_info['path'])
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)

            if processing_times:
                results[category] = {
                    'sample_count': len(processing_times),
                    'avg_processing_time': mean(processing_times),
                    'min_processing_time': min(processing_times),
                    'max_processing_time': max(processing_times),
                    'target_compliance': all(t < 0.5 for t in processing_times),
                    'processing_times': processing_times
                }

                print(f"   {category.capitalize()} images: {results[category]['avg_processing_time']:.4f}s avg")

        self.results['image_size_performance'] = results

        print(f"✅ Image size performance test complete")

        return results

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance tests"""
        print("🚀 Starting comprehensive performance testing...")
        print(f"📊 Found {len(self.test_images)} test images")

        # Run all test categories
        self.test_single_image_performance()
        self.test_concurrent_classification()
        self.test_memory_usage_under_load()
        self.test_different_image_sizes()

        # Generate summary
        self._generate_summary()

        return self.results

    def _generate_summary(self):
        """Generate performance test summary"""
        summary = {
            'overall_performance': 'PASS',
            'issues_found': [],
            'recommendations': []
        }

        # Check single image performance
        single_perf = self.results.get('single_image_performance', {})
        if single_perf:
            avg_time = single_perf['total_processing'].get('mean', 1.0)
            compliance = single_perf['total_processing'].get('target_compliance_rate', 0.0)

            if avg_time >= 0.5:
                summary['issues_found'].append(f"Average processing time ({avg_time:.3f}s) exceeds 0.5s target")
                summary['overall_performance'] = 'FAIL'

            if compliance < 0.95:
                summary['issues_found'].append(f"Target compliance rate ({compliance:.1%}) below 95%")
                if summary['overall_performance'] != 'FAIL':
                    summary['overall_performance'] = 'WARNING'

        # Check memory usage
        memory_results = self.results.get('memory_usage', {})
        if memory_results:
            if memory_results.get('memory_leak_assessment', {}).get('potential_leak', False):
                summary['issues_found'].append("Potential memory leak detected")
                summary['overall_performance'] = 'FAIL'

        # Check concurrent performance
        concurrent_results = self.results.get('concurrent_performance', {})
        if concurrent_results:
            compliance = concurrent_results.get('target_compliance', {}).get('compliance_rate', 0.0)
            if compliance < 0.95:
                summary['issues_found'].append(f"Concurrent target compliance ({compliance:.1%}) below 95%")
                if summary['overall_performance'] != 'FAIL':
                    summary['overall_performance'] = 'WARNING'

        # Generate recommendations
        if single_perf and single_perf['total_processing'].get('mean', 0) > 0.1:
            summary['recommendations'].append("Consider optimizing feature extraction for better performance")

        if memory_results and memory_results.get('total_memory_increase_mb', 0) > 50:
            summary['recommendations'].append("Monitor memory usage in production; consider implementing memory optimization")

        summary['test_completion_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

        self.results['summary'] = summary

        print(f"\n📋 Performance Test Summary:")
        print(f"   Overall Performance: {summary['overall_performance']}")
        if summary['issues_found']:
            print(f"   Issues Found: {len(summary['issues_found'])}")
            for issue in summary['issues_found']:
                print(f"     - {issue}")
        if summary['recommendations']:
            print(f"   Recommendations: {len(summary['recommendations'])}")
            for rec in summary['recommendations']:
                print(f"     - {rec}")

    def save_results(self, output_file: str = "performance_test_results.json"):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Results saved to: {output_file}")

    def cleanup(self):
        """Clean up temporary files"""
        temp_dir = Path("temp_perf_test_images")
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            print("🧹 Cleaned up temporary test images")


def main():
    """Main function to run performance tests"""
    print("🔬 Day 3: Classification Performance Testing")
    print("=" * 50)

    tester = PerformanceTestClassification()

    try:
        # Run all tests
        results = tester.run_all_tests()

        # Save results
        tester.save_results()

        # Print final status
        overall_status = results['summary']['overall_performance']
        print(f"\n🎯 Final Status: {overall_status}")

        if overall_status == 'PASS':
            print("✅ All performance tests passed! System ready for production.")
        elif overall_status == 'WARNING':
            print("⚠️  Performance tests completed with warnings. Review recommendations.")
        else:
            print("❌ Performance tests failed. Address issues before production deployment.")

        return overall_status == 'PASS'

    except Exception as e:
        print(f"❌ Performance testing failed with error: {e}")
        return False

    finally:
        tester.cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)