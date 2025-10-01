#!/usr/bin/env python3
"""Performance benchmarking for feature extraction"""

import time
import psutil
import numpy as np
import cv2
import json
import tempfile
import os
from typing import Dict, List
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.ai_modules.feature_extraction import ImageFeatureExtractor


class PerformanceBenchmark:
    """Benchmark feature extraction performance"""

    def __init__(self):
        self.extractor = ImageFeatureExtractor()
        self.results = {}
        self.test_images = []

    def setup_test_images(self) -> List[str]:
        """Create test image dataset with known characteristics"""
        test_images = []

        # Create temporary directory for test images
        temp_dir = tempfile.mkdtemp(prefix="benchmark_")

        # Test Image 1: Simple geometric (100x100)
        simple_image = np.zeros((100, 100, 3), dtype=np.uint8)
        simple_image[25:75, 25:75] = [255, 255, 255]  # White square
        simple_path = os.path.join(temp_dir, "simple_100x100.png")
        cv2.imwrite(simple_path, simple_image)
        test_images.append(simple_path)

        # Test Image 2: Complex pattern (100x100)
        complex_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        complex_path = os.path.join(temp_dir, "complex_100x100.png")
        cv2.imwrite(complex_path, complex_image)
        test_images.append(complex_path)

        # Test Image 3: Medium size (256x256)
        medium_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        medium_path = os.path.join(temp_dir, "medium_256x256.png")
        cv2.imwrite(medium_path, medium_image)
        test_images.append(medium_path)

        # Test Image 4: Large size (512x512)
        large_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        large_path = os.path.join(temp_dir, "large_512x512.png")
        cv2.imwrite(large_path, large_image)
        test_images.append(large_path)

        # Test Image 5: Very large (1024x1024)
        xlarge_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        xlarge_path = os.path.join(temp_dir, "xlarge_1024x1024.png")
        cv2.imwrite(xlarge_path, xlarge_image)
        test_images.append(xlarge_path)

        # Add real test images if available
        real_logo_dirs = [
            'data/logos/simple_geometric',
            'data/logos/text_based',
            'data/logos/gradients',
            'data/logos/complex'
        ]

        for logo_dir in real_logo_dirs:
            logo_path = Path(logo_dir)
            if logo_path.exists():
                for img_file in logo_path.glob('*.png'):
                    test_images.append(str(img_file))
                    if len(test_images) >= 10:  # Limit for benchmark speed
                        break

        self.test_images = test_images
        return test_images

    def benchmark_edge_density(self, test_images: List[str]) -> Dict:
        """Benchmark edge density calculation"""
        print("🔍 Benchmarking edge density calculation...")

        times = []
        memory_usage = []
        results = []

        for image_path in test_images:
            if not Path(image_path).exists():
                continue

            # Memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Timing edge density calculation
            start_time = time.perf_counter()
            result = self.extractor._calculate_edge_density(image)
            end_time = time.perf_counter()

            # Memory after
            mem_after = process.memory_info().rss
            memory_used = mem_after - mem_before

            processing_time = end_time - start_time
            times.append(processing_time)
            memory_usage.append(memory_used)
            results.append(result)

            print(f"  {Path(image_path).name}: {result:.4f} in {processing_time:.4f}s "
                  f"({memory_used / 1024 / 1024:.1f}MB)")

        if not times:
            return {'error': 'No valid images processed'}

        return {
            'method': 'edge_density',
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'avg_memory': np.mean(memory_usage),
            'max_memory': np.max(memory_usage),
            'success_rate': len(times) / len(test_images),
            'sample_results': results[:5],  # First 5 results for validation
            'target_met': np.mean(times) < 0.1  # <0.1s target
        }

    def benchmark_complete_extraction(self, test_images: List[str]) -> Dict:
        """Benchmark complete feature extraction pipeline"""
        print("🚀 Benchmarking complete feature extraction...")

        times = []
        memory_usage = []
        extraction_results = []

        for image_path in test_images:
            if not Path(image_path).exists():
                continue

            # Memory before
            process = psutil.Process()
            mem_before = process.memory_info().rss

            # Timing complete extraction
            start_time = time.perf_counter()
            try:
                result = self.extractor.extract_features(image_path)
                extraction_results.append(result)
                success = True
            except Exception as e:
                print(f"  ❌ Failed {Path(image_path).name}: {e}")
                result = None
                success = False

            end_time = time.perf_counter()

            # Memory after
            mem_after = process.memory_info().rss
            memory_used = mem_after - mem_before

            if success:
                processing_time = end_time - start_time
                times.append(processing_time)
                memory_usage.append(memory_used)

                print(f"  ✅ {Path(image_path).name}: {processing_time:.4f}s "
                      f"({memory_used / 1024 / 1024:.1f}MB)")

        if not times:
            return {'error': 'No successful extractions'}

        return {
            'method': 'complete_extraction',
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'avg_memory': np.mean(memory_usage),
            'max_memory': np.max(memory_usage),
            'success_rate': len(times) / len(test_images),
            'sample_results': extraction_results[:3],  # First 3 complete results
            'target_met': np.mean(times) < 0.5  # <0.5s target for complete extraction
        }

    def benchmark_memory_usage(self, test_images: List[str]) -> Dict:
        """Benchmark memory usage patterns"""
        print("💾 Benchmarking memory usage...")

        memory_stats = []

        for image_path in test_images[:5]:  # Limit to 5 images for memory test
            if not Path(image_path).exists():
                continue

            process = psutil.Process()

            # Baseline memory
            baseline_memory = process.memory_info().rss

            # Load image and extract features
            try:
                image = cv2.imread(image_path)
                image_size = image.nbytes if image is not None else 0

                # Memory during processing
                start_memory = process.memory_info().rss
                result = self.extractor.extract_features(image_path)
                peak_memory = process.memory_info().rss

                # Calculate memory usage
                memory_increase = peak_memory - baseline_memory
                memory_per_pixel = memory_increase / (image.shape[0] * image.shape[1]) if image is not None else 0

                memory_stats.append({
                    'image_path': Path(image_path).name,
                    'image_size_bytes': image_size,
                    'memory_increase_bytes': memory_increase,
                    'memory_increase_mb': memory_increase / 1024 / 1024,
                    'memory_per_pixel': memory_per_pixel
                })

                print(f"  {Path(image_path).name}: "
                      f"+{memory_increase / 1024 / 1024:.1f}MB "
                      f"({memory_per_pixel:.2e} bytes/pixel)")

            except Exception as e:
                print(f"  ❌ Memory test failed for {Path(image_path).name}: {e}")

        if not memory_stats:
            return {'error': 'No memory stats collected'}

        avg_memory_mb = np.mean([stat['memory_increase_mb'] for stat in memory_stats])

        return {
            'method': 'memory_usage',
            'stats': memory_stats,
            'avg_memory_increase_mb': avg_memory_mb,
            'max_memory_increase_mb': max([stat['memory_increase_mb'] for stat in memory_stats]),
            'target_met': avg_memory_mb < 50  # <50MB target
        }

    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        print("🏁 Starting comprehensive feature extraction benchmark...")
        print("=" * 60)

        # Setup test images
        test_images = self.setup_test_images()
        print(f"📋 Created {len(test_images)} test images")

        # Run benchmarks
        results = {}

        # Edge density benchmark
        results['edge_density'] = self.benchmark_edge_density(test_images)

        # Complete extraction benchmark
        results['complete_extraction'] = self.benchmark_complete_extraction(test_images)

        # Memory usage benchmark
        results['memory_usage'] = self.benchmark_memory_usage(test_images)

        # Generate summary
        results['summary'] = self.generate_summary(results)

        # Cleanup test images
        self.cleanup_test_images(test_images)

        return results

    def generate_summary(self, results: Dict) -> Dict:
        """Generate benchmark summary"""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': 3,
            'passed_tests': 0,
            'performance_score': 0.0,
            'recommendations': []
        }

        # Check each benchmark result
        for test_name, test_result in results.items():
            if test_name == 'summary':
                continue

            if isinstance(test_result, dict) and test_result.get('target_met', False):
                summary['passed_tests'] += 1

        # Calculate performance score
        summary['performance_score'] = summary['passed_tests'] / summary['total_tests']

        # Generate recommendations
        if results.get('edge_density', {}).get('avg_time', 1.0) > 0.05:
            summary['recommendations'].append("Consider optimizing edge density calculation")

        if results.get('complete_extraction', {}).get('avg_time', 1.0) > 0.3:
            summary['recommendations'].append("Complete extraction pipeline could be optimized")

        if results.get('memory_usage', {}).get('avg_memory_increase_mb', 100) > 30:
            summary['recommendations'].append("Memory usage could be reduced")

        return summary

    def cleanup_test_images(self, test_images: List[str]):
        """Clean up temporary test images"""
        for image_path in test_images:
            path = Path(image_path)
            if path.exists() and 'benchmark_' in str(path.parent):
                try:
                    path.unlink()
                except Exception:
                    pass

        # Remove temporary directories
        for image_path in test_images:
            parent_dir = Path(image_path).parent
            if 'benchmark_' in str(parent_dir):
                try:
                    parent_dir.rmdir()
                    break
                except Exception:
                    pass

    def save_results(self, results: Dict, output_file: str = None):
        """Save benchmark results to JSON file"""
        if output_file is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_file = f"benchmark_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Benchmark results saved to: {output_file}")

    def print_results(self, results: Dict):
        """Print formatted benchmark results"""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS")
        print("=" * 60)

        # Summary
        summary = results.get('summary', {})
        print(f"🎯 Performance Score: {summary.get('performance_score', 0):.1%}")
        print(f"✅ Tests Passed: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}")

        # Individual results
        for test_name, test_result in results.items():
            if test_name == 'summary' or not isinstance(test_result, dict):
                continue

            print(f"\n📈 {test_name.upper().replace('_', ' ')}")
            print("-" * 40)

            if 'avg_time' in test_result:
                target_met = "✅" if test_result.get('target_met', False) else "❌"
                print(f"  Average time: {test_result['avg_time']:.4f}s {target_met}")
                print(f"  Min time: {test_result['min_time']:.4f}s")
                print(f"  Max time: {test_result['max_time']:.4f}s")

            if 'avg_memory_increase_mb' in test_result:
                target_met = "✅" if test_result.get('target_met', False) else "❌"
                print(f"  Average memory: {test_result['avg_memory_increase_mb']:.1f}MB {target_met}")

            if 'success_rate' in test_result:
                print(f"  Success rate: {test_result['success_rate']:.1%}")

        # Recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print("\n" + "=" * 60)


def main():
    """Main benchmark execution"""
    benchmark = PerformanceBenchmark()

    try:
        # Run full benchmark
        results = benchmark.run_full_benchmark()

        # Print results
        benchmark.print_results(results)

        # Save results
        benchmark.save_results(results)

    except KeyboardInterrupt:
        print("\n⏹️ Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()