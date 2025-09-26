#!/usr/bin/env python3
"""
Parallel batch optimization for PNG to SVG conversion.

This script processes multiple images concurrently for maximum performance.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from converters.vtracer_converter import VTracerConverter
from utils.image_loader import QualityMetricsWrapper
from utils.parameter_cache import ParameterCache
from learn_parameters import ImageFeatureExtractor, ParameterLearner


def process_single_image(png_path: str, output_dir: str,
                        target_ssim: float = 0.95,
                        use_cache: bool = True) -> Dict:
    """
    Process a single image (worker function for parallel processing).

    Args:
        png_path: Path to PNG file
        output_dir: Output directory
        target_ssim: Target SSIM quality
        use_cache: Whether to use parameter cache

    Returns:
        Processing results
    """
    # Import in worker to avoid serialization issues
    from utils.ai_detector import create_detector

    png_path = Path(png_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    start_time = time.time()

    # Initialize components
    converter = VTracerConverter()
    metrics_wrapper = QualityMetricsWrapper()
    detector = create_detector()

    # Check cache
    params = None
    cache_hit = False

    if use_cache:
        cache = ParameterCache()
        extractor = ImageFeatureExtractor()

        # Extract features
        features = extractor.extract_features(str(png_path))

        # Check cache
        params = cache.get_best_parameters(str(png_path), features)
        if params:
            cache_hit = True

    # If no cache hit, detect type and use defaults
    if params is None:
        logo_type, confidence, _ = detector.detect_logo_type(str(png_path))

        # Type-specific defaults
        type_params = {
            'simple': {
                'color_precision': 3,
                'layer_difference': 4,
                'corner_threshold': 30,
                'length_threshold': 4.0,
                'max_iterations': 10,
                'splice_threshold': 45,
                'path_precision': 8
            },
            'text': {
                'color_precision': 2,
                'layer_difference': 4,
                'corner_threshold': 20,
                'length_threshold': 3.0,
                'max_iterations': 10,
                'splice_threshold': 30,
                'path_precision': 10
            },
            'gradient': {
                'color_precision': 8,
                'layer_difference': 8,
                'corner_threshold': 50,
                'length_threshold': 5.0,
                'max_iterations': 15,
                'splice_threshold': 60,
                'path_precision': 6
            },
            'complex': {
                'color_precision': 10,
                'layer_difference': 10,
                'corner_threshold': 60,
                'length_threshold': 5.0,
                'max_iterations': 20,
                'splice_threshold': 70,
                'path_precision': 5
            }
        }

        params = type_params.get(logo_type, type_params['complex'])

    # Convert with parameters
    svg_path = output_dir / f"{png_path.stem}.svg"

    try:
        result = converter.convert_with_params(
            str(png_path),
            str(svg_path),
            **params
        )

        if not result['success']:
            raise Exception(f"Conversion failed: {result.get('error', 'Unknown error')}")

        # Calculate quality
        ssim = metrics_wrapper.calculate_ssim_from_paths(
            str(png_path),
            str(svg_path)
        )

        # File sizes
        png_size = png_path.stat().st_size
        svg_size = svg_path.stat().st_size
        size_reduction = (1 - svg_size / png_size) * 100

        processing_time = time.time() - start_time

        # Add to cache if successful and not from cache
        if use_cache and not cache_hit and ssim >= target_ssim:
            cache = ParameterCache()
            extractor = ImageFeatureExtractor()
            features = extractor.extract_features(str(png_path))

            cache.add_entry(
                str(png_path),
                features,
                params,
                {'ssim': ssim, 'size_ratio': svg_size / png_size}
            )

        return {
            'success': True,
            'png_path': str(png_path),
            'svg_path': str(svg_path),
            'ssim': ssim,
            'png_size': png_size,
            'svg_size': svg_size,
            'size_reduction': size_reduction,
            'processing_time': processing_time,
            'cache_hit': cache_hit,
            'parameters': params
        }

    except Exception as e:
        return {
            'success': False,
            'png_path': str(png_path),
            'error': str(e),
            'processing_time': time.time() - start_time
        }


class ParallelOptimizer:
    """Optimize multiple images in parallel."""

    def __init__(self, num_workers: Optional[int] = None):
        """
        Initialize parallel optimizer.

        Args:
            num_workers: Number of worker processes (None for auto)
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.cache = ParameterCache()

    def optimize_batch(self, png_files: List[str], output_dir: str,
                      target_ssim: float = 0.95,
                      use_cache: bool = True,
                      show_progress: bool = True) -> Tuple[List[Dict], Dict]:
        """
        Optimize a batch of images in parallel.

        Args:
            png_files: List of PNG file paths
            output_dir: Output directory
            target_ssim: Target SSIM quality
            use_cache: Whether to use parameter cache
            show_progress: Show progress updates

        Returns:
            Tuple of (results list, statistics dict)
        """
        print(f"Processing {len(png_files)} files with {self.num_workers} workers...")
        start_time = time.time()

        # Create worker function with fixed parameters
        worker_func = partial(
            process_single_image,
            output_dir=output_dir,
            target_ssim=target_ssim,
            use_cache=use_cache
        )

        results = []
        completed = 0

        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(worker_func, f): f for f in png_files}

            # Process as completed
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    if show_progress:
                        if result['success']:
                            status = f"✅ SSIM={result['ssim']:.3f}"
                            if result.get('cache_hit'):
                                status += " (cached)"
                        else:
                            status = f"❌ {result.get('error', 'Failed')}"

                        print(f"  [{completed}/{len(png_files)}] {Path(futures[future]).name}: {status}")

                except Exception as e:
                    completed += 1
                    print(f"  [{completed}/{len(png_files)}] Error: {e}")
                    results.append({
                        'success': False,
                        'png_path': futures[future],
                        'error': str(e)
                    })

        # Calculate statistics
        total_time = time.time() - start_time
        successful = [r for r in results if r['success']]

        stats = {
            'total_files': len(png_files),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'total_time': total_time,
            'avg_time_per_file': total_time / len(png_files) if png_files else 0,
            'speedup': len(png_files) / total_time if total_time > 0 else 0,
            'cache_hits': sum(1 for r in successful if r.get('cache_hit', False))
        }

        if successful:
            stats['avg_ssim'] = np.mean([r['ssim'] for r in successful])
            stats['avg_size_reduction'] = np.mean([r['size_reduction'] for r in successful])

            # Time comparison
            sequential_estimate = sum(r['processing_time'] for r in successful)
            stats['sequential_estimate'] = sequential_estimate
            stats['parallel_speedup'] = sequential_estimate / total_time if total_time > 0 else 1

        return results, stats

    def benchmark_parallel_performance(self, test_dir: str,
                                      worker_counts: List[int] = [1, 2, 4, 8]) -> Dict:
        """
        Benchmark parallel processing with different worker counts.

        Args:
            test_dir: Directory with test images
            worker_counts: List of worker counts to test

        Returns:
            Benchmark results
        """
        test_dir = Path(test_dir)
        png_files = list(test_dir.glob("**/*.png"))[:20]  # Limit for benchmarking

        if not png_files:
            print(f"No PNG files found in {test_dir}")
            return {}

        print(f"\n{'='*60}")
        print(f"PARALLEL PROCESSING BENCHMARK")
        print(f"{'='*60}")
        print(f"Testing with {len(png_files)} files")

        benchmark_results = {}
        output_dir = "benchmark_output"

        for num_workers in worker_counts:
            if num_workers > mp.cpu_count():
                continue

            print(f"\n📊 Testing with {num_workers} worker(s)...")

            # Clear cache for fair comparison
            self.cache.clear_cache()

            # Update worker count
            self.num_workers = num_workers

            # Run optimization
            results, stats = self.optimize_batch(
                [str(f) for f in png_files],
                output_dir,
                use_cache=False,  # No cache for fair comparison
                show_progress=False
            )

            benchmark_results[num_workers] = stats

            print(f"  Time: {stats['total_time']:.2f}s")
            print(f"  Speed: {stats['speedup']:.2f} files/sec")
            if 'avg_ssim' in stats:
                print(f"  Avg SSIM: {stats['avg_ssim']:.3f}")

        # Print comparison
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")

        print("\n| Workers | Time (s) | Files/sec | Speedup |")
        print("|---------|----------|-----------|---------|")

        baseline_time = benchmark_results.get(1, {}).get('total_time', 1)

        for workers, stats in sorted(benchmark_results.items()):
            speedup = baseline_time / stats['total_time'] if stats.get('total_time') else 0
            print(f"| {workers:7} | {stats['total_time']:8.2f} | {stats['speedup']:9.2f} | {speedup:7.2f}x |")

        return benchmark_results


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Parallel batch optimization")
    parser.add_argument('directory', nargs='?', help='Directory with PNG files')
    parser.add_argument('--output', default='optimized_output', help='Output directory')
    parser.add_argument('--workers', type=int, help='Number of workers (default: auto)')
    parser.add_argument('--target-ssim', type=float, default=0.95, help='Target SSIM')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--report', help='Save JSON report')

    args = parser.parse_args()

    # Create optimizer
    optimizer = ParallelOptimizer(num_workers=args.workers)

    if args.benchmark:
        # Run benchmark
        test_dir = args.directory or "data/logos"
        results = optimizer.benchmark_parallel_performance(test_dir)

        if args.report:
            with open(args.report, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Benchmark report saved to {args.report}")

    elif args.directory:
        # Process directory
        png_files = list(Path(args.directory).glob("**/*.png"))

        if not png_files:
            print(f"No PNG files found in {args.directory}")
            return 1

        # Optimize batch
        results, stats = optimizer.optimize_batch(
            [str(f) for f in png_files],
            args.output,
            target_ssim=args.target_ssim,
            use_cache=not args.no_cache
        )

        # Print summary
        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Processed: {stats['total_files']} files")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Total time: {stats['total_time']:.2f}s")
        print(f"Average time per file: {stats['avg_time_per_file']:.2f}s")

        if 'avg_ssim' in stats:
            print(f"Average SSIM: {stats['avg_ssim']:.3f}")
            print(f"Average size reduction: {stats['avg_size_reduction']:.1f}%")

        if stats.get('cache_hits'):
            print(f"Cache hits: {stats['cache_hits']} ({stats['cache_hits']/stats['successful']*100:.1f}%)")

        if 'parallel_speedup' in stats:
            print(f"Parallel speedup: {stats['parallel_speedup']:.2f}x")

        # Save report if requested
        if args.report:
            report_data = {
                'statistics': stats,
                'results': results
            }
            with open(args.report, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"\n✅ Report saved to {args.report}")

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())