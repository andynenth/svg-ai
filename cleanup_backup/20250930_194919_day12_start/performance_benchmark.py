#!/usr/bin/env python3
"""
Performance Benchmarking Suite - Task 2 Implementation
Comprehensive benchmark testing for AI pipeline performance validation.
"""

import sys
import time
import json
import logging
import argparse
import tracemalloc
import statistics
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import psutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import pipeline and components
try:
    from backend.ai_modules.pipeline.unified_ai_pipeline import UnifiedAIPipeline, PipelineResult
    from backend.converters.ai_enhanced_converter import AIEnhancedConverter
    from backend.converters.vtracer_converter import VTracerConverter
except ImportError as e:
    print(f"Warning: Failed to import required modules: {e}")
    print("Some benchmarks may not be available")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    test_name: str
    tier: Optional[int]
    processing_time: float
    memory_current_mb: float
    memory_peak_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    quality_score: Optional[float] = None
    output_size_bytes: Optional[int] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for AI pipeline validation.
    """

    def __init__(self, output_file: str = "benchmarks.json"):
        """
        Initialize performance benchmark suite.

        Args:
            output_file: Output file for benchmark results
        """
        self.output_file = output_file
        self.results: List[BenchmarkMetrics] = []

        # Performance targets (as specified in DAY10_VALIDATION.md)
        self.metrics = {
            'tier1': {'target': 2.0, 'results': []},
            'tier2': {'target': 5.0, 'results': []},
            'tier3': {'target': 15.0, 'results': []},
        }

        # Initialize pipeline
        self.pipeline = None
        self.baseline_converter = None
        self._initialize_components()

        # Load test images
        self.test_images = self._load_test_images()

        logger.info(f"Performance benchmark initialized with {len(self.test_images)} test images")

    def _initialize_components(self):
        """Initialize pipeline and converter components."""
        try:
            self.pipeline = UnifiedAIPipeline(
                enable_caching=True,
                enable_fallbacks=True,
                performance_mode="balanced"
            )
            logger.info("✓ Unified AI Pipeline initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize AI pipeline: {e}")
            self.pipeline = None

        try:
            self.baseline_converter = VTracerConverter()
            logger.info("✓ Baseline converter initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize baseline converter: {e}")
            self.baseline_converter = None

    def _load_test_images(self) -> List[str]:
        """Load test images for benchmarking."""
        test_images = []
        base_path = Path("data/logos")

        if not base_path.exists():
            logger.warning(f"Test data path {base_path} not found")
            return []

        categories = ["simple_geometric", "text_based", "gradients", "complex", "abstract"]

        for category in categories:
            category_path = base_path / category
            if category_path.exists():
                # Get 3 images per category for performance testing
                category_images = list(category_path.glob("*.png"))
                # Filter out processed images
                category_images = [
                    str(img) for img in category_images
                    if "optimized" not in str(img) and ".cache" not in str(img)
                ][:3]  # Take first 3
                test_images.extend(category_images)

        logger.info(f"Loaded {len(test_images)} test images for benchmarking")
        return test_images

    def benchmark_processing_times(self):
        """Measure processing times for each tier."""
        logger.info("Starting processing time benchmarks...")

        if not self.pipeline:
            logger.error("Pipeline not available for processing time benchmarks")
            return

        if not self.test_images:
            logger.error("No test images available for benchmarking")
            return

        for tier in [1, 2, 3]:
            logger.info(f"Benchmarking Tier {tier} processing times...")

            tier_results = []

            for i, image_path in enumerate(self.test_images[:10]):  # Test 10 images per tier
                logger.info(f"  Testing image {i+1}/10: {Path(image_path).name}")

                try:
                    # Measure memory before
                    memory_before = self._get_memory_usage()
                    cpu_before = psutil.cpu_percent()

                    # Process with specific tier
                    start = time.perf_counter()
                    result = self._process_with_tier(image_path, tier)
                    duration = time.perf_counter() - start

                    # Measure memory after
                    memory_after = self._get_memory_usage()
                    cpu_after = psutil.cpu_percent()

                    # Calculate metrics
                    memory_delta = memory_after - memory_before
                    cpu_usage = (cpu_before + cpu_after) / 2

                    # Get output size if successful
                    output_size = len(result.svg_content) if result and result.success and result.svg_content else 0

                    # Record metrics
                    metrics = BenchmarkMetrics(
                        test_name=f"tier{tier}_processing",
                        tier=tier,
                        processing_time=duration,
                        memory_current_mb=memory_after,
                        memory_peak_mb=memory_delta,
                        cpu_usage_percent=cpu_usage,
                        success=result.success if result else False,
                        error_message=result.error_message if result and not result.success else None,
                        quality_score=result.quality_score if result else None,
                        output_size_bytes=output_size
                    )

                    self.results.append(metrics)
                    self.metrics[f'tier{tier}']['results'].append(duration)
                    tier_results.append(duration)

                    logger.info(f"    Processed in {duration:.3f}s (target: {self.metrics[f'tier{tier}']['target']}s)")

                except Exception as e:
                    logger.error(f"    Failed to process {Path(image_path).name}: {e}")

                    # Record failure
                    metrics = BenchmarkMetrics(
                        test_name=f"tier{tier}_processing",
                        tier=tier,
                        processing_time=0.0,
                        memory_current_mb=self._get_memory_usage(),
                        memory_peak_mb=0.0,
                        cpu_usage_percent=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    self.results.append(metrics)

            # Calculate tier statistics
            if tier_results:
                avg_time = statistics.mean(tier_results)
                p95_time = statistics.quantiles(tier_results, n=20)[18] if len(tier_results) >= 5 else max(tier_results)
                target = self.metrics[f'tier{tier}']['target']

                logger.info(f"Tier {tier} Results:")
                logger.info(f"  Average time: {avg_time:.3f}s")
                logger.info(f"  95th percentile: {p95_time:.3f}s")
                logger.info(f"  Target: {target}s")
                logger.info(f"  Meets target: {'✓' if p95_time < target else '✗'}")

    def _process_with_tier(self, image_path: str, tier: int) -> Optional[PipelineResult]:
        """Process image with specific tier."""
        try:
            # Configure time constraint based on tier
            time_constraints = {1: 2.0, 2: 5.0, 3: 15.0}
            time_constraint = time_constraints.get(tier, 30.0)

            # For now, process with pipeline (tier routing happens internally)
            result = self.pipeline.process(
                image_path=image_path,
                target_quality=0.85,
                time_constraint=time_constraint
            )

            return result

        except Exception as e:
            logger.error(f"Tier {tier} processing failed for {image_path}: {e}")
            return None

    def benchmark_memory_usage(self):
        """Track memory consumption during processing."""
        logger.info("Starting memory usage benchmarks...")

        if not self.pipeline or not self.test_images:
            logger.error("Pipeline or test images not available for memory benchmarks")
            return

        # Start memory tracking
        tracemalloc.start()

        try:
            # Process batch of images
            test_batch = self.test_images[:10]
            logger.info(f"Processing batch of {len(test_batch)} images for memory analysis...")

            memory_samples = []
            initial_memory = self._get_memory_usage()

            for i, image_path in enumerate(test_batch):
                logger.info(f"  Processing image {i+1}/{len(test_batch)}: {Path(image_path).name}")

                try:
                    # Measure memory before processing
                    memory_before = self._get_memory_usage()

                    # Process image
                    result = self.pipeline.process(image_path)

                    # Measure memory after processing
                    memory_after = self._get_memory_usage()
                    memory_delta = memory_after - memory_before

                    memory_samples.append({
                        'image': Path(image_path).name,
                        'memory_before_mb': memory_before,
                        'memory_after_mb': memory_after,
                        'memory_delta_mb': memory_delta,
                        'success': result.success if result else False
                    })

                    logger.info(f"    Memory delta: {memory_delta:+.1f} MB")

                except Exception as e:
                    logger.error(f"    Memory test failed for {Path(image_path).name}: {e}")

            # Get final memory statistics
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            final_memory = self._get_memory_usage()
            total_delta = final_memory - initial_memory

            memory_metrics = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'total_delta_mb': total_delta,
                'tracemalloc_current_mb': current / 1024 / 1024,
                'tracemalloc_peak_mb': peak / 1024 / 1024,
                'samples': memory_samples,
                'target_peak_mb': 500.0,  # Target from DAY10_VALIDATION.md
                'meets_target': peak / 1024 / 1024 < 500.0
            }

            # Record overall memory metrics
            overall_metrics = BenchmarkMetrics(
                test_name="memory_usage",
                tier=None,
                processing_time=0.0,
                memory_current_mb=memory_metrics['tracemalloc_current_mb'],
                memory_peak_mb=memory_metrics['tracemalloc_peak_mb'],
                cpu_usage_percent=0.0,
                success=memory_metrics['meets_target']
            )
            self.results.append(overall_metrics)

            logger.info(f"Memory Usage Results:")
            logger.info(f"  Peak memory: {memory_metrics['tracemalloc_peak_mb']:.1f} MB")
            logger.info(f"  Target: 500 MB")
            logger.info(f"  Meets target: {'✓' if memory_metrics['meets_target'] else '✗'}")

            return memory_metrics

        except Exception as e:
            logger.error(f"Memory benchmark failed: {e}")
            tracemalloc.stop()
            return None

    def benchmark_concurrent_processing(self):
        """Test concurrent request handling."""
        logger.info("Starting concurrent processing benchmarks...")

        if not self.pipeline or not self.test_images:
            logger.error("Pipeline or test images not available for concurrent benchmarks")
            return

        # Test with different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        test_images_subset = self.test_images[:20]  # Use 20 images for concurrent testing

        for max_workers in concurrency_levels:
            logger.info(f"Testing concurrent processing with {max_workers} workers...")

            try:
                start_time = time.perf_counter()
                memory_before = self._get_memory_usage()

                # Run concurrent processing
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    futures = []
                    for image_path in test_images_subset:
                        future = executor.submit(self._process_concurrent, image_path)
                        futures.append((future, image_path))

                    # Collect results
                    successful = 0
                    failed = 0
                    processing_times = []

                    for future, image_path in futures:
                        try:
                            result, duration = future.result(timeout=60)  # 60 second timeout
                            if result and result.success:
                                successful += 1
                                processing_times.append(duration)
                            else:
                                failed += 1
                        except Exception as e:
                            logger.warning(f"Concurrent processing failed for {Path(image_path).name}: {e}")
                            failed += 1

                total_time = time.perf_counter() - start_time
                memory_after = self._get_memory_usage()

                # Calculate metrics
                if processing_times:
                    avg_individual_time = statistics.mean(processing_times)
                    throughput = successful / total_time  # images per second
                else:
                    avg_individual_time = 0
                    throughput = 0

                concurrent_metrics = BenchmarkMetrics(
                    test_name=f"concurrent_{max_workers}_workers",
                    tier=None,
                    processing_time=total_time,
                    memory_current_mb=memory_after,
                    memory_peak_mb=memory_after - memory_before,
                    cpu_usage_percent=psutil.cpu_percent(),
                    success=successful > 0,
                    quality_score=throughput,  # Use quality_score field for throughput
                    output_size_bytes=successful
                )
                self.results.append(concurrent_metrics)

                logger.info(f"  Concurrency {max_workers} Results:")
                logger.info(f"    Total time: {total_time:.3f}s")
                logger.info(f"    Successful: {successful}/{len(test_images_subset)}")
                logger.info(f"    Failed: {failed}")
                logger.info(f"    Throughput: {throughput:.2f} images/second")
                logger.info(f"    Avg individual time: {avg_individual_time:.3f}s")

            except Exception as e:
                logger.error(f"Concurrent benchmark failed for {max_workers} workers: {e}")

    def _process_concurrent(self, image_path: str) -> tuple:
        """Process image for concurrent testing."""
        start_time = time.perf_counter()
        try:
            result = self.pipeline.process(image_path)
            duration = time.perf_counter() - start_time
            return result, duration
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Concurrent processing error for {image_path}: {e}")
            return None, duration

    def profile_bottlenecks(self):
        """Profile and identify performance bottlenecks."""
        logger.info("Starting bottleneck profiling...")

        if not self.pipeline or not self.test_images:
            logger.error("Pipeline or test images not available for profiling")
            return

        # Profile with a representative image
        test_image = self.test_images[0]
        logger.info(f"Profiling with image: {Path(test_image).name}")

        try:
            import cProfile
            import pstats
            import io

            # Profile the processing
            profiler = cProfile.Profile()
            profiler.enable()

            start_time = time.perf_counter()
            result = self.pipeline.process(test_image)
            total_time = time.perf_counter() - start_time

            profiler.disable()

            # Analyze profiling results
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions

            profiling_output = s.getvalue()

            profiling_metrics = BenchmarkMetrics(
                test_name="bottleneck_profiling",
                tier=None,
                processing_time=total_time,
                memory_current_mb=self._get_memory_usage(),
                memory_peak_mb=0.0,
                cpu_usage_percent=0.0,
                success=result.success if result else False
            )
            self.results.append(profiling_metrics)

            logger.info(f"Profiling completed in {total_time:.3f}s")
            logger.info("Top performance bottlenecks:")

            # Extract key insights from profiling
            lines = profiling_output.split('\n')[:25]  # First 25 lines
            for line in lines:
                if line.strip() and 'function calls' not in line:
                    logger.info(f"  {line}")

            return profiling_output

        except ImportError:
            logger.warning("cProfile not available for bottleneck analysis")
            return None
        except Exception as e:
            logger.error(f"Profiling failed: {e}")
            return None

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        logger.info("Generating performance report...")

        # Calculate summary statistics
        tier_summaries = {}
        for tier in [1, 2, 3]:
            tier_results = self.metrics[f'tier{tier}']['results']
            target = self.metrics[f'tier{tier}']['target']

            if tier_results:
                avg_time = statistics.mean(tier_results)
                p95_time = statistics.quantiles(tier_results, n=20)[18] if len(tier_results) >= 5 else max(tier_results)
                meets_target = p95_time < target

                tier_summaries[f'tier{tier}'] = {
                    'count': len(tier_results),
                    'average_time': avg_time,
                    'p95_time': p95_time,
                    'target_time': target,
                    'meets_target': meets_target,
                    'all_times': tier_results
                }
            else:
                tier_summaries[f'tier{tier}'] = {
                    'count': 0,
                    'average_time': 0,
                    'p95_time': 0,
                    'target_time': target,
                    'meets_target': False,
                    'all_times': []
                }

        # Overall success metrics
        successful_tests = sum(1 for r in self.results if r.success)
        total_tests = len(self.results)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0

        # Memory analysis
        memory_tests = [r for r in self.results if r.test_name == "memory_usage"]
        peak_memory = memory_tests[0].memory_peak_mb if memory_tests else 0
        memory_target_met = peak_memory < 500.0

        # Performance targets assessment
        targets_met = {
            'tier1_performance': tier_summaries['tier1']['meets_target'],
            'tier2_performance': tier_summaries['tier2']['meets_target'],
            'tier3_performance': tier_summaries['tier3']['meets_target'],
            'memory_usage': memory_target_met
        }

        all_targets_met = all(targets_met.values())

        report = {
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate,
                'all_targets_met': all_targets_met,
                'timestamp': datetime.now().isoformat()
            },
            'tier_performance': tier_summaries,
            'targets_assessment': targets_met,
            'memory_analysis': {
                'peak_memory_mb': peak_memory,
                'target_memory_mb': 500.0,
                'meets_target': memory_target_met
            },
            'detailed_results': [asdict(r) for r in self.results],
            'recommendations': self._generate_recommendations(targets_met, tier_summaries)
        }

        return report

    def _generate_recommendations(self, targets_met: Dict[str, bool], tier_summaries: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        # Tier-specific recommendations
        for tier in [1, 2, 3]:
            if not targets_met.get(f'tier{tier}_performance', False):
                tier_data = tier_summaries[f'tier{tier}']
                if tier_data['count'] > 0:
                    recommendations.append(
                        f"Tier {tier} performance below target: {tier_data['p95_time']:.2f}s > {tier_data['target_time']}s. "
                        f"Consider optimizing tier {tier} processing pipeline."
                    )

        # Memory recommendations
        if not targets_met.get('memory_usage', False):
            recommendations.append(
                "Memory usage exceeds 500MB target. Consider implementing more aggressive caching "
                "or reducing memory footprint of AI models."
            )

        # General recommendations
        if targets_met.get('tier1_performance', False):
            recommendations.append("✓ Tier 1 performance meets targets - suitable for real-time processing")

        if all(targets_met.values()):
            recommendations.append("✓ All performance targets met - system ready for production deployment")
        else:
            recommendations.append("⚠ Some performance targets not met - optimization required before production")

        return recommendations

    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to file."""
        if filename is None:
            filename = self.output_file

        report = self.generate_performance_report()

        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Benchmark results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results to {filename}: {e}")

    def print_summary(self):
        """Print human-readable summary of benchmark results."""
        report = self.generate_performance_report()

        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*80)

        # Summary
        summary = report['summary']
        print(f"\n📊 TEST SUMMARY:")
        print(f"   • Total tests: {summary['total_tests']}")
        print(f"   • Successful: {summary['successful_tests']}")
        print(f"   • Success rate: {summary['success_rate']:.1%}")
        print(f"   • All targets met: {'✅' if summary['all_targets_met'] else '❌'}")

        # Tier performance
        print(f"\n⚡ TIER PERFORMANCE:")
        for tier in [1, 2, 3]:
            tier_data = report['tier_performance'][f'tier{tier}']
            status = "✅" if tier_data['meets_target'] else "❌"
            print(f"   • Tier {tier}: {status}")
            if tier_data['count'] > 0:
                print(f"     95th percentile: {tier_data['p95_time']:.3f}s (target: {tier_data['target_time']}s)")
                print(f"     Average: {tier_data['average_time']:.3f}s")
                print(f"     Tests: {tier_data['count']}")

        # Memory analysis
        memory = report['memory_analysis']
        status = "✅" if memory['meets_target'] else "❌"
        print(f"\n💾 MEMORY USAGE: {status}")
        print(f"   • Peak memory: {memory['peak_memory_mb']:.1f} MB")
        print(f"   • Target: {memory['target_memory_mb']} MB")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   • {rec}")

        print("\n" + "="*80)


def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description="Performance Benchmark Suite")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--output", default="benchmarks.json", help="Output file for results")
    parser.add_argument("--memory-only", action="store_true", help="Run only memory benchmarks")
    parser.add_argument("--concurrent-only", action="store_true", help="Run only concurrent benchmarks")
    parser.add_argument("--profile", action="store_true", help="Include performance profiling")

    args = parser.parse_args()

    try:
        benchmark = PerformanceBenchmark(output_file=args.output)

        if args.memory_only:
            benchmark.benchmark_memory_usage()
        elif args.concurrent_only:
            benchmark.benchmark_concurrent_processing()
        elif args.full:
            benchmark.benchmark_processing_times()
            benchmark.benchmark_memory_usage()
            benchmark.benchmark_concurrent_processing()
            if args.profile:
                benchmark.profile_bottlenecks()
        else:
            # Default: run processing time benchmarks
            benchmark.benchmark_processing_times()

        # Generate and save results
        benchmark.save_results()
        benchmark.print_summary()

        # Exit with appropriate code
        report = benchmark.generate_performance_report()
        if report['summary']['all_targets_met']:
            logger.info("🎉 All performance targets met!")
            return 0
        else:
            logger.warning("⚠️ Some performance targets not met")
            return 1

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 2


if __name__ == "__main__":
    exit(main())