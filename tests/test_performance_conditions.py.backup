#!/usr/bin/env python3
"""
Performance Testing Under Various Conditions

This module tests the PNG to SVG conversion system under different performance
conditions including high load, various image sizes, concurrent processing,
and resource constraints.
"""

import pytest
import time
import threading
import concurrent.futures
from pathlib import Path
import tempfile
import os
import sys
from typing import List, Dict, Any
from PIL import Image
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


class PerformanceTestRunner:
    """Comprehensive performance testing under various conditions."""

    def __init__(self):
        self.performance_results = []
        self.load_test_results = []
        self.memory_usage_results = []

    def log_performance_result(self, test_name: str, metrics: Dict[str, Any]):
        """Log performance test result."""
        self.performance_results.append({
            'test_name': test_name,
            'timestamp': time.time(),
            **metrics
        })

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_results:
            return {'no_results': True}

        conversion_times = [r.get('avg_conversion_time', 0) for r in self.performance_results if 'avg_conversion_time' in r]

        return {
            'total_performance_tests': len(self.performance_results),
            'overall_avg_conversion_time': statistics.mean(conversion_times) if conversion_times else 0,
            'fastest_conversion': min(conversion_times) if conversion_times else 0,
            'slowest_conversion': max(conversion_times) if conversion_times else 0,
            'conversion_time_std': statistics.stdev(conversion_times) if len(conversion_times) > 1 else 0,
            'detailed_results': self.performance_results
        }


# Global performance test runner
perf_runner = PerformanceTestRunner()


class TestImageSizePerformance:
    """Test performance with different image sizes."""

    def create_test_image(self, width: int, height: int, complexity: str = 'simple') -> str:
        """Create test image with specified dimensions and complexity."""
        if complexity == 'simple':
            # Simple solid color rectangle
            img = Image.new('RGB', (width, height), color='blue')
        elif complexity == 'gradient':
            # Gradient image
            img = Image.new('RGB', (width, height))
            pixels = img.load()
            for x in range(width):
                for y in range(height):
                    r = int(255 * x / width)
                    g = int(255 * y / height)
                    b = 128
                    pixels[x, y] = (r, g, b)
        else:  # complex
            # Complex pattern
            img = Image.new('RGB', (width, height))
            pixels = img.load()
            for x in range(width):
                for y in range(height):
                    r = (x * y) % 256
                    g = (x + y) % 256
                    b = (x ^ y) % 256
                    pixels[x, y] = (r, g, b)

        # Save to temporary file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.png')
        os.close(tmp_fd)
        img.save(tmp_path, 'PNG')
        return tmp_path

    def test_small_image_performance(self):
        """Test performance with small images (100x100)."""
        start_time = time.time()

        try:
            from backend.converters.vtracer_converter import VTracerConverter

            converter = VTracerConverter()

            # Test different complexities
            complexities = ['simple', 'gradient', 'complex']
            results = {}

            for complexity in complexities:
                image_path = self.create_test_image(100, 100, complexity)

                try:
                    # Run multiple conversions
                    conversion_times = []
                    for _ in range(5):
                        conv_start = time.time()
                        result = converter.convert_with_metrics(image_path)
                        conv_time = time.time() - conv_start

                        if result['success']:
                            conversion_times.append(conv_time)

                    if conversion_times:
                        results[complexity] = {
                            'avg_time': statistics.mean(conversion_times),
                            'min_time': min(conversion_times),
                            'max_time': max(conversion_times),
                            'std_time': statistics.stdev(conversion_times) if len(conversion_times) > 1 else 0,
                            'successful_conversions': len(conversion_times)
                        }

                finally:
                    os.unlink(image_path)

            metrics = {
                'image_size': '100x100',
                'complexity_results': results,
                'avg_conversion_time': statistics.mean([r['avg_time'] for r in results.values()]) if results else 0,
                'test_duration': time.time() - start_time
            }

            perf_runner.log_performance_result('small_image_performance', metrics)

        except Exception as e:
            perf_runner.log_performance_result('small_image_performance', {
                'error': str(e),
                'test_duration': time.time() - start_time
            })
            raise

    def test_large_image_performance(self):
        """Test performance with large images (1000x1000)."""
        start_time = time.time()

        try:
            from backend.converters.vtracer_converter import VTracerConverter

            converter = VTracerConverter()

            # Test simple large image
            image_path = self.create_test_image(1000, 1000, 'simple')

            try:
                # Single conversion test (large images take longer)
                conv_start = time.time()
                result = converter.convert_with_metrics(image_path)
                conv_time = time.time() - conv_start

                metrics = {
                    'image_size': '1000x1000',
                    'conversion_success': result['success'],
                    'conversion_time': conv_time,
                    'avg_conversion_time': conv_time,
                    'svg_size': len(result.get('svg', '')) if result.get('svg') else 0,
                    'test_duration': time.time() - start_time
                }

                perf_runner.log_performance_result('large_image_performance', metrics)

            finally:
                os.unlink(image_path)

        except Exception as e:
            perf_runner.log_performance_result('large_image_performance', {
                'error': str(e),
                'test_duration': time.time() - start_time
            })
            raise


class TestConcurrentProcessing:
    """Test performance under concurrent processing load."""

    def test_concurrent_conversions(self, temp_png_file):
        """Test multiple concurrent conversions."""
        start_time = time.time()

        try:
            from backend.converters.vtracer_converter import VTracerConverter

            # Test with multiple threads
            num_threads = 4
            conversions_per_thread = 3

            def worker_conversion():
                """Worker function for concurrent conversion."""
                converter = VTracerConverter()
                worker_results = []

                for _ in range(conversions_per_thread):
                    conv_start = time.time()
                    result = converter.convert_with_metrics(temp_png_file)
                    conv_time = time.time() - conv_start

                    worker_results.append({
                        'success': result['success'],
                        'time': conv_time,
                        'svg_size': len(result.get('svg', '')) if result.get('svg') else 0
                    })

                return worker_results

            # Run concurrent conversions
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker_conversion) for _ in range(num_threads)]
                all_results = []

                for future in concurrent.futures.as_completed(futures):
                    worker_results = future.result()
                    all_results.extend(worker_results)

            # Analyze results
            successful_conversions = sum(1 for r in all_results if r['success'])
            conversion_times = [r['time'] for r in all_results if r['success']]

            metrics = {
                'num_threads': num_threads,
                'conversions_per_thread': conversions_per_thread,
                'total_conversions': len(all_results),
                'successful_conversions': successful_conversions,
                'success_rate': (successful_conversions / len(all_results)) * 100 if all_results else 0,
                'avg_conversion_time': statistics.mean(conversion_times) if conversion_times else 0,
                'min_conversion_time': min(conversion_times) if conversion_times else 0,
                'max_conversion_time': max(conversion_times) if conversion_times else 0,
                'total_test_duration': time.time() - start_time
            }

            perf_runner.log_performance_result('concurrent_conversions', metrics)

            # Validate performance expectations
            assert metrics['success_rate'] >= 80.0, f"Concurrent conversion success rate too low: {metrics['success_rate']:.1f}%"

        except Exception as e:
            perf_runner.log_performance_result('concurrent_conversions', {
                'error': str(e),
                'test_duration': time.time() - start_time
            })
            raise

    def test_ai_concurrent_processing(self, temp_png_file):
        """Test AI-enhanced conversions under concurrent load."""
        start_time = time.time()

        try:
            from backend.converters.ai_enhanced_converter import AIEnhancedSVGConverter

            num_threads = 2  # Fewer threads for AI processing
            conversions_per_thread = 2

            def ai_worker_conversion():
                """Worker function for concurrent AI conversion."""
                converter = AIEnhancedSVGConverter(enable_ai=True, ai_timeout=10.0)
                worker_results = []

                for _ in range(conversions_per_thread):
                    conv_start = time.time()
                    result = converter.convert_with_ai_analysis(temp_png_file)
                    conv_time = time.time() - conv_start

                    worker_results.append({
                        'success': result['success'],
                        'ai_enhanced': result.get('ai_enhanced', False),
                        'total_time': conv_time,
                        'ai_time': result.get('ai_analysis_time', 0),
                        'conversion_time': result.get('conversion_time', 0),
                        'logo_type': result.get('classification', {}).get('logo_type', 'unknown')
                    })

                return worker_results

            # Run concurrent AI conversions
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(ai_worker_conversion) for _ in range(num_threads)]
                all_results = []

                for future in concurrent.futures.as_completed(futures):
                    worker_results = future.result()
                    all_results.extend(worker_results)

            # Analyze AI concurrent results
            successful_conversions = sum(1 for r in all_results if r['success'])
            ai_enhanced_conversions = sum(1 for r in all_results if r.get('ai_enhanced', False))
            total_times = [r['total_time'] for r in all_results if r['success']]
            ai_times = [r['ai_time'] for r in all_results if r.get('ai_enhanced', False)]

            metrics = {
                'num_threads': num_threads,
                'conversions_per_thread': conversions_per_thread,
                'total_conversions': len(all_results),
                'successful_conversions': successful_conversions,
                'ai_enhanced_conversions': ai_enhanced_conversions,
                'ai_enhancement_rate': (ai_enhanced_conversions / successful_conversions) * 100 if successful_conversions > 0 else 0,
                'avg_total_time': statistics.mean(total_times) if total_times else 0,
                'avg_ai_time': statistics.mean(ai_times) if ai_times else 0,
                'total_test_duration': time.time() - start_time
            }

            perf_runner.log_performance_result('ai_concurrent_processing', metrics)

        except ImportError:
            pytest.skip("AI modules not available for concurrent AI testing")
        except Exception as e:
            perf_runner.log_performance_result('ai_concurrent_processing', {
                'error': str(e),
                'test_duration': time.time() - start_time
            })
            raise


class TestResourceConstraints:
    """Test performance under resource constraints."""

    def test_memory_efficiency(self, temp_png_file):
        """Test memory usage during conversions."""
        start_time = time.time()

        try:
            import psutil
            import os

            from backend.converters.vtracer_converter import VTracerConverter

            converter = VTracerConverter()
            process = psutil.Process(os.getpid())

            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

            memory_measurements = []

            # Run multiple conversions and measure memory
            for i in range(10):
                conv_start = time.time()
                result = converter.convert_with_metrics(temp_png_file)
                conv_time = time.time() - conv_start

                # Measure memory after conversion
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_measurements.append({
                    'iteration': i,
                    'memory_mb': current_memory,
                    'memory_delta': current_memory - baseline_memory,
                    'conversion_time': conv_time,
                    'success': result['success']
                })

            # Analyze memory usage
            max_memory = max(m['memory_mb'] for m in memory_measurements)
            max_delta = max(m['memory_delta'] for m in memory_measurements)
            avg_delta = statistics.mean(m['memory_delta'] for m in memory_measurements)

            metrics = {
                'baseline_memory_mb': baseline_memory,
                'max_memory_mb': max_memory,
                'max_memory_delta_mb': max_delta,
                'avg_memory_delta_mb': avg_delta,
                'memory_measurements': memory_measurements,
                'test_duration': time.time() - start_time
            }

            perf_runner.log_performance_result('memory_efficiency', metrics)

            # Memory should not grow excessively
            assert max_delta < 500, f"Memory usage grew too much: {max_delta:.1f}MB"

        except ImportError:
            pytest.skip("psutil not available for memory testing")
        except Exception as e:
            perf_runner.log_performance_result('memory_efficiency', {
                'error': str(e),
                'test_duration': time.time() - start_time
            })
            raise

    def test_timeout_handling(self, temp_png_file):
        """Test behavior under timeout constraints."""
        start_time = time.time()

        try:
            from backend.converters.ai_enhanced_converter import AIEnhancedSVGConverter

            # Test with very short AI timeout
            converter = AIEnhancedSVGConverter(enable_ai=True, ai_timeout=0.1)  # 100ms timeout

            result = converter.convert_with_ai_analysis(temp_png_file)

            # Should handle timeout gracefully
            metrics = {
                'timeout_setting': 0.1,
                'conversion_success': result['success'],
                'ai_enhanced': result.get('ai_enhanced', False),
                'total_time': result.get('total_time', 0),
                'ai_time': result.get('ai_analysis_time', 0),
                'fallback_used': not result.get('ai_enhanced', False),
                'test_duration': time.time() - start_time
            }

            perf_runner.log_performance_result('timeout_handling', metrics)

            # Should succeed even with timeout
            assert result['success'], "Conversion should succeed even with AI timeout"

        except ImportError:
            pytest.skip("AI modules not available for timeout testing")
        except Exception as e:
            perf_runner.log_performance_result('timeout_handling', {
                'error': str(e),
                'test_duration': time.time() - start_time
            })
            raise


def test_performance_summary():
    """Generate comprehensive performance test summary."""
    summary = perf_runner.get_performance_summary()

    print("\n" + "="*80)
    print("PERFORMANCE TESTING UNDER VARIOUS CONDITIONS - SUMMARY")
    print("="*80)

    if summary.get('no_results'):
        print("No performance results available.")
        return

    print(f"Total Performance Tests: {summary['total_performance_tests']}")
    print(f"Overall Average Conversion Time: {summary['overall_avg_conversion_time']:.3f}s")
    print(f"Fastest Conversion: {summary['fastest_conversion']:.3f}s")
    print(f"Slowest Conversion: {summary['slowest_conversion']:.3f}s")
    print(f"Conversion Time Std Dev: {summary['conversion_time_std']:.3f}s")

    print("\nDetailed Performance Results:")
    for result in summary['detailed_results']:
        print(f"\n  {result['test_name']}:")
        for key, value in result.items():
            if key not in ['test_name', 'timestamp']:
                if isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
                elif isinstance(value, dict):
                    print(f"    {key}: (complex data)")
                else:
                    print(f"    {key}: {value}")

    print("\n" + "="*80)

    # Performance validation
    assert summary['overall_avg_conversion_time'] < 10.0, f"Overall performance too slow: {summary['overall_avg_conversion_time']:.3f}s"


if __name__ == "__main__":
    # Run with pytest when called directly
    pytest.main([__file__, '-v', '--tb=short'])