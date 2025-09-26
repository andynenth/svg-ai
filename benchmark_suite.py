#!/usr/bin/env python3
"""
Comprehensive performance benchmark suite for SVG conversion pipeline.

This script tests conversion speed, quality, and memory usage across
different configurations and optimizations.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
import time
import psutil
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from converters.vtracer_converter import VTracerConverter
from utils.image_loader import QualityMetricsWrapper
from utils.ai_detector import create_detector
from utils.optimized_detector import OptimizedDetector
from utils.parameter_cache import ParameterCache
from batch_optimize_parallel import ParallelOptimizer


class BenchmarkSuite:
    """Comprehensive benchmark suite for SVG conversion."""

    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark suite.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.converter = VTracerConverter()
        self.metrics = QualityMetricsWrapper()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def benchmark_single_conversion(self, png_path: str, params: Dict) -> Dict:
        """
        Benchmark single file conversion.

        Args:
            png_path: Path to PNG file
            params: VTracer parameters

        Returns:
            Benchmark results
        """
        png_path = Path(png_path)
        svg_path = self.output_dir / f"test_{png_path.stem}.svg"

        # Memory before
        mem_before = self.get_memory_usage()

        # Conversion timing
        start_time = time.time()
        result = self.converter.convert_with_params(
            str(png_path),
            str(svg_path),
            **params
        )
        conversion_time = time.time() - start_time

        # Memory after
        mem_after = self.get_memory_usage()
        mem_used = mem_after - mem_before

        if result['success']:
            # Quality measurement timing
            quality_start = time.time()
            ssim = self.metrics.calculate_ssim_from_paths(
                str(png_path),
                str(svg_path)
            )
            quality_time = time.time() - quality_start

            # File sizes
            png_size = png_path.stat().st_size
            svg_size = svg_path.stat().st_size

            # Clean up
            svg_path.unlink(missing_ok=True)

            return {
                'success': True,
                'conversion_time': conversion_time,
                'quality_time': quality_time,
                'total_time': conversion_time + quality_time,
                'memory_used': mem_used,
                'ssim': ssim,
                'png_size': png_size,
                'svg_size': svg_size,
                'size_ratio': svg_size / png_size
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error')
            }

    def benchmark_detection_methods(self, test_images: List[str]) -> Dict:
        """
        Benchmark different detection methods.

        Args:
            test_images: List of test image paths

        Returns:
            Detection benchmark results
        """
        print("\n📊 Detection Method Benchmark")
        print("-" * 40)

        results = {}

        # Standard detector
        print("Testing standard detector...")
        std_detector = create_detector()

        start = time.time()
        for img in test_images:
            _ = std_detector.detect_logo_type(img)
        std_time = time.time() - start

        results['standard'] = {
            'total_time': std_time,
            'per_image': std_time / len(test_images),
            'images_per_second': len(test_images) / std_time
        }

        # Optimized detector
        print("Testing optimized detector...")
        opt_detector = OptimizedDetector()
        opt_detector.warmup()

        start = time.time()
        _ = opt_detector.detect_batch(test_images)
        opt_time = time.time() - start

        results['optimized'] = {
            'total_time': opt_time,
            'per_image': opt_time / len(test_images),
            'images_per_second': len(test_images) / opt_time,
            'speedup': std_time / opt_time
        }

        # With cache (second run)
        print("Testing with cache...")
        start = time.time()
        _ = opt_detector.detect_batch(test_images)
        cache_time = time.time() - start

        results['cached'] = {
            'total_time': cache_time,
            'per_image': cache_time / len(test_images),
            'images_per_second': len(test_images) / cache_time,
            'speedup': std_time / cache_time
        }

        return results

    def benchmark_parallel_scaling(self, test_dir: str) -> Dict:
        """
        Benchmark parallel processing scaling.

        Args:
            test_dir: Directory with test images

        Returns:
            Parallel scaling results
        """
        print("\n📊 Parallel Processing Benchmark")
        print("-" * 40)

        test_images = list(Path(test_dir).glob("**/*.png"))[:12]
        test_paths = [str(p) for p in test_images]

        if not test_paths:
            return {}

        results = {}
        worker_counts = [1, 2, 4, 8]

        for workers in worker_counts:
            if workers > os.cpu_count():
                continue

            print(f"Testing with {workers} workers...")

            optimizer = ParallelOptimizer(num_workers=workers)

            start = time.time()
            batch_results, stats = optimizer.optimize_batch(
                test_paths,
                str(self.output_dir),
                use_cache=False,
                show_progress=False
            )
            elapsed = time.time() - start

            results[workers] = {
                'workers': workers,
                'total_time': elapsed,
                'files_per_second': len(test_paths) / elapsed,
                'successful': stats['successful'],
                'avg_ssim': stats.get('avg_ssim', 0)
            }

        # Calculate scaling efficiency
        if 1 in results:
            baseline = results[1]['total_time']
            for workers, data in results.items():
                data['speedup'] = baseline / data['total_time']
                data['efficiency'] = data['speedup'] / workers * 100

        return results

    def benchmark_cache_impact(self, test_images: List[str]) -> Dict:
        """
        Benchmark cache impact on performance.

        Args:
            test_images: List of test image paths

        Returns:
            Cache impact results
        """
        print("\n📊 Cache Impact Benchmark")
        print("-" * 40)

        cache = ParameterCache()
        cache.clear_cache()

        results = {
            'cold_cache': {},
            'warm_cache': {},
            'cache_stats': {}
        }

        # Cold cache run
        print("Testing with cold cache...")
        start = time.time()

        for img_path in test_images:
            # Simulate parameter lookup
            from learn_parameters import ImageFeatureExtractor
            extractor = ImageFeatureExtractor()
            features = extractor.extract_features(img_path)
            _ = cache.get_best_parameters(img_path, features)

        cold_time = time.time() - start
        results['cold_cache'] = {
            'total_time': cold_time,
            'per_image': cold_time / len(test_images)
        }

        # Add entries to cache
        for img_path in test_images:
            features = extractor.extract_features(img_path)
            params = {'color_precision': 4, 'layer_difference': 8}
            metrics = {'ssim': 0.95, 'size_ratio': 0.5}
            cache.add_entry(img_path, features, params, metrics)

        # Warm cache run
        print("Testing with warm cache...")
        start = time.time()

        cache_hits = 0
        for img_path in test_images:
            features = extractor.extract_features(img_path)
            params = cache.get_best_parameters(img_path, features)
            if params:
                cache_hits += 1

        warm_time = time.time() - start
        results['warm_cache'] = {
            'total_time': warm_time,
            'per_image': warm_time / len(test_images),
            'speedup': cold_time / warm_time if warm_time > 0 else 0
        }

        # Cache statistics
        stats = cache.get_statistics()
        results['cache_stats'] = stats
        results['cache_stats']['hit_rate'] = cache_hits / len(test_images) * 100

        return results

    def run_full_benchmark(self, test_dir: str = "data/logos") -> Dict:
        """
        Run complete benchmark suite.

        Args:
            test_dir: Directory with test images

        Returns:
            Complete benchmark results
        """
        print("="*60)
        print("COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("="*60)
        print(f"Test directory: {test_dir}")
        print(f"CPUs available: {os.cpu_count()}")
        print(f"Memory available: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")

        # Prepare test data
        test_images = list(Path(test_dir).glob("**/*.png"))[:20]
        test_paths = [str(p) for p in test_images]

        if not test_paths:
            print("No test images found")
            return {}

        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_directory': str(test_dir),
                'num_images': len(test_paths),
                'cpu_count': os.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024
            },
            'benchmarks': {}
        }

        # Single conversion benchmark
        print("\n📊 Single Conversion Benchmark")
        print("-" * 40)

        single_results = []
        test_params = {
            'color_precision': 4,
            'layer_difference': 8,
            'corner_threshold': 40,
            'length_threshold': 5.0,
            'max_iterations': 10,
            'splice_threshold': 45,
            'path_precision': 6
        }

        for img_path in test_paths[:5]:
            result = self.benchmark_single_conversion(img_path, test_params)
            if result['success']:
                single_results.append(result)
                print(f"  {Path(img_path).name}: {result['conversion_time']:.3f}s, SSIM={result['ssim']:.3f}")

        if single_results:
            results['benchmarks']['single_conversion'] = {
                'avg_conversion_time': np.mean([r['conversion_time'] for r in single_results]),
                'avg_quality_time': np.mean([r['quality_time'] for r in single_results]),
                'avg_memory_mb': np.mean([r['memory_used'] for r in single_results]),
                'avg_ssim': np.mean([r['ssim'] for r in single_results])
            }

        # Detection benchmarks
        results['benchmarks']['detection'] = self.benchmark_detection_methods(test_paths[:10])

        # Parallel scaling
        results['benchmarks']['parallel'] = self.benchmark_parallel_scaling(test_dir)

        # Cache impact
        results['benchmarks']['cache'] = self.benchmark_cache_impact(test_paths[:10])

        return results

    def generate_report(self, results: Dict, output_file: str = None):
        """
        Generate benchmark report.

        Args:
            results: Benchmark results
            output_file: Output file path
        """
        if output_file is None:
            output_file = self.output_dir / "benchmark_report.json"

        # Save JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✅ Benchmark report saved to {output_file}")

        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)

        if 'single_conversion' in results.get('benchmarks', {}):
            conv = results['benchmarks']['single_conversion']
            print(f"\n📌 Single Conversion:")
            print(f"  Avg time: {conv['avg_conversion_time']:.3f}s")
            print(f"  Avg SSIM: {conv['avg_ssim']:.3f}")
            print(f"  Avg memory: {conv['avg_memory_mb']:.1f} MB")

        if 'detection' in results.get('benchmarks', {}):
            det = results['benchmarks']['detection']
            if 'optimized' in det:
                print(f"\n📌 Detection Optimization:")
                print(f"  Speedup: {det['optimized']['speedup']:.2f}x")
                print(f"  Cached speedup: {det.get('cached', {}).get('speedup', 0):.2f}x")

        if 'parallel' in results.get('benchmarks', {}):
            par = results['benchmarks']['parallel']
            max_workers = max(par.keys()) if par else 1
            if max_workers in par:
                print(f"\n📌 Parallel Processing ({max_workers} workers):")
                print(f"  Speedup: {par[max_workers].get('speedup', 1):.2f}x")
                print(f"  Efficiency: {par[max_workers].get('efficiency', 0):.1f}%")

        if 'cache' in results.get('benchmarks', {}):
            cache = results['benchmarks']['cache']
            print(f"\n📌 Cache Impact:")
            print(f"  Hit rate: {cache['cache_stats'].get('hit_rate', 0):.1f}%")
            print(f"  Speedup: {cache['warm_cache'].get('speedup', 0):.2f}x")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance benchmark suite")
    parser.add_argument('--test-dir', default='data/logos', help='Test directory')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory')
    parser.add_argument('--full', action='store_true', help='Run full benchmark')

    args = parser.parse_args()

    # Create benchmark suite
    suite = BenchmarkSuite(output_dir=args.output_dir)

    if args.full:
        # Run full benchmark
        results = suite.run_full_benchmark(args.test_dir)
        suite.generate_report(results)
    else:
        # Quick test
        test_images = list(Path(args.test_dir).glob("**/*.png"))[:5]
        test_paths = [str(p) for p in test_images]

        if test_paths:
            print("Running quick benchmark...")
            results = {
                'benchmarks': {
                    'detection': suite.benchmark_detection_methods(test_paths)
                }
            }
            suite.generate_report(results, "quick_benchmark.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())