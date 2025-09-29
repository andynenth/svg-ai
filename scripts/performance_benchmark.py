#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Suite for Week 2 Implementation

This script runs comprehensive performance benchmarks across all system components
and generates detailed performance reports for production readiness validation.
"""

import os
import sys
import time
import json
import statistics
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkResult:
    """Structured benchmark result data."""
    test_name: str
    success: bool
    duration: float
    memory_delta_mb: float
    cpu_usage_percent: float
    output_size: int
    quality_score: Optional[float]
    error_message: Optional[str]
    timestamp: float
    metadata: Dict[str, Any]


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmarking suite."""

    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results: List[BenchmarkResult] = []
        self.system_info = self._collect_system_info()
        self.start_time = time.time()

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmark context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat()
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024**2)

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)

    def create_test_images(self) -> List[str]:
        """Create standardized test images for benchmarking."""
        test_images = []

        # Test image specifications
        test_specs = [
            {'name': 'simple_small', 'size': (100, 100), 'type': 'simple'},
            {'name': 'simple_medium', 'size': (500, 500), 'type': 'simple'},
            {'name': 'simple_large', 'size': (1000, 1000), 'type': 'simple'},
            {'name': 'complex_small', 'size': (100, 100), 'type': 'complex'},
            {'name': 'complex_medium', 'size': (500, 500), 'type': 'complex'},
            {'name': 'text_medium', 'size': (400, 200), 'type': 'text'},
            {'name': 'gradient_medium', 'size': (300, 300), 'type': 'gradient'}
        ]

        for spec in test_specs:
            image_path = self._create_test_image(spec)
            test_images.append(image_path)

        return test_images

    def _create_test_image(self, spec: Dict[str, Any]) -> str:
        """Create a single test image based on specification."""
        width, height = spec['size']
        img_type = spec['type']

        if img_type == 'simple':
            # Simple geometric shape
            img = Image.new('RGB', (width, height), color='white')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            margin = min(width, height) // 10
            draw.ellipse([margin, margin, width-margin, height-margin], fill='blue')

        elif img_type == 'complex':
            # Complex pattern
            img = Image.new('RGB', (width, height))
            pixels = img.load()
            for x in range(width):
                for y in range(height):
                    r = (x * y) % 256
                    g = (x + y * 2) % 256
                    b = (x * 2 + y) % 256
                    pixels[x, y] = (r, g, b)

        elif img_type == 'text':
            # Text-like rectangles
            img = Image.new('RGB', (width, height), color='white')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            for i in range(5):
                x = 20 + i * 60
                draw.rectangle([x, height//2-20, x+40, height//2+20], fill='black')

        elif img_type == 'gradient':
            # Gradient image
            img = Image.new('RGB', (width, height))
            pixels = img.load()
            for x in range(width):
                for y in range(height):
                    r = int(255 * x / width)
                    g = int(255 * y / height)
                    b = 128
                    pixels[x, y] = (r, g, b)

        # Save image
        temp_file = tempfile.NamedTemporaryFile(suffix=f'_{spec["name"]}.png', delete=False)
        img.save(temp_file.name, 'PNG')
        return temp_file.name

    def benchmark_basic_conversion(self, test_images: List[str]) -> Dict[str, Any]:
        """Benchmark basic VTracer conversion performance."""
        print("Running basic conversion benchmarks...")

        try:
            from backend.converters.vtracer_converter import VTracerConverter
            converter = VTracerConverter()

            conversion_results = []

            for image_path in test_images:
                # Pre-benchmark state
                memory_before = self._get_memory_usage()

                start_time = time.time()
                cpu_before = self._get_cpu_usage()

                try:
                    result = converter.convert_with_metrics(image_path)

                    end_time = time.time()
                    memory_after = self._get_memory_usage()
                    cpu_after = self._get_cpu_usage()

                    benchmark_result = BenchmarkResult(
                        test_name=f"basic_conversion_{Path(image_path).stem}",
                        success=result['success'],
                        duration=end_time - start_time,
                        memory_delta_mb=memory_after - memory_before,
                        cpu_usage_percent=max(cpu_before, cpu_after),
                        output_size=len(result.get('svg', '')),
                        quality_score=None,
                        error_message=None,
                        timestamp=start_time,
                        metadata={
                            'converter': 'VTracerConverter',
                            'image_path': Path(image_path).name,
                            'conversion_time': result.get('time', 0)
                        }
                    )

                except Exception as e:
                    benchmark_result = BenchmarkResult(
                        test_name=f"basic_conversion_{Path(image_path).stem}",
                        success=False,
                        duration=time.time() - start_time,
                        memory_delta_mb=0,
                        cpu_usage_percent=0,
                        output_size=0,
                        quality_score=None,
                        error_message=str(e),
                        timestamp=start_time,
                        metadata={'converter': 'VTracerConverter', 'image_path': Path(image_path).name}
                    )

                conversion_results.append(benchmark_result)
                self.results.append(benchmark_result)

        except ImportError as e:
            print(f"VTracer converter not available: {e}")
            return {'error': 'VTracer converter not available'}

        # Analyze results
        successful_conversions = [r for r in conversion_results if r.success]

        if not successful_conversions:
            return {'error': 'No successful conversions'}

        analysis = {
            'total_tests': len(conversion_results),
            'successful_tests': len(successful_conversions),
            'success_rate': len(successful_conversions) / len(conversion_results) * 100,
            'avg_duration': statistics.mean([r.duration for r in successful_conversions]),
            'min_duration': min([r.duration for r in successful_conversions]),
            'max_duration': max([r.duration for r in successful_conversions]),
            'avg_memory_delta': statistics.mean([r.memory_delta_mb for r in successful_conversions]),
            'avg_output_size': statistics.mean([r.output_size for r in successful_conversions])
        }

        print(f"Basic conversion benchmark complete: {analysis['success_rate']:.1f}% success rate")
        return analysis

    def benchmark_ai_enhanced_conversion(self, test_images: List[str]) -> Dict[str, Any]:
        """Benchmark AI-enhanced conversion performance."""
        print("Running AI-enhanced conversion benchmarks...")

        try:
            from backend.converters.ai_enhanced_converter import AIEnhancedSVGConverter
            converter = AIEnhancedSVGConverter(enable_ai=True, ai_timeout=10.0)

            ai_results = []

            for image_path in test_images:
                memory_before = self._get_memory_usage()
                start_time = time.time()

                try:
                    result = converter.convert_with_ai_analysis(image_path)

                    end_time = time.time()
                    memory_after = self._get_memory_usage()

                    benchmark_result = BenchmarkResult(
                        test_name=f"ai_enhanced_{Path(image_path).stem}",
                        success=result['success'],
                        duration=end_time - start_time,
                        memory_delta_mb=memory_after - memory_before,
                        cpu_usage_percent=self._get_cpu_usage(),
                        output_size=len(result.get('svg', '')),
                        quality_score=result.get('classification', {}).get('confidence', 0),
                        error_message=None,
                        timestamp=start_time,
                        metadata={
                            'converter': 'AIEnhancedSVGConverter',
                            'ai_enhanced': result.get('ai_enhanced', False),
                            'logo_type': result.get('classification', {}).get('logo_type', 'unknown'),
                            'confidence': result.get('classification', {}).get('confidence', 0),
                            'ai_analysis_time': result.get('ai_analysis_time', 0),
                            'conversion_time': result.get('conversion_time', 0)
                        }
                    )

                except Exception as e:
                    benchmark_result = BenchmarkResult(
                        test_name=f"ai_enhanced_{Path(image_path).stem}",
                        success=False,
                        duration=time.time() - start_time,
                        memory_delta_mb=0,
                        cpu_usage_percent=0,
                        output_size=0,
                        quality_score=None,
                        error_message=str(e),
                        timestamp=start_time,
                        metadata={'converter': 'AIEnhancedSVGConverter'}
                    )

                ai_results.append(benchmark_result)
                self.results.append(benchmark_result)

        except ImportError as e:
            print(f"AI-enhanced converter not available: {e}")
            return {'error': 'AI-enhanced converter not available', 'ai_available': False}

        # Analyze AI-enhanced results
        successful_ai = [r for r in ai_results if r.success]
        ai_enhanced = [r for r in ai_results if r.metadata.get('ai_enhanced', False)]

        if not successful_ai:
            return {'error': 'No successful AI conversions', 'ai_available': True}

        analysis = {
            'ai_available': True,
            'total_tests': len(ai_results),
            'successful_tests': len(successful_ai),
            'ai_enhanced_conversions': len(ai_enhanced),
            'ai_enhancement_rate': len(ai_enhanced) / len(successful_ai) * 100 if successful_ai else 0,
            'avg_duration': statistics.mean([r.duration for r in successful_ai]),
            'avg_ai_analysis_time': statistics.mean([r.metadata.get('ai_analysis_time', 0) for r in ai_enhanced]) if ai_enhanced else 0,
            'avg_confidence': statistics.mean([r.quality_score for r in successful_ai if r.quality_score]),
            'logo_type_distribution': self._analyze_logo_types(ai_enhanced)
        }

        print(f"AI-enhanced benchmark complete: {analysis['ai_enhancement_rate']:.1f}% AI enhancement rate")
        return analysis

    def benchmark_cache_performance(self, test_images: List[str]) -> Dict[str, Any]:
        """Benchmark caching system performance."""
        print("Running cache performance benchmarks...")

        try:
            from backend.ai_modules.advanced_cache import MultiLevelCache
            cache = MultiLevelCache()

            cache_results = []

            # Test cache performance with feature extraction
            try:
                from backend.ai_modules.cached_components import CachedFeatureExtractor
                extractor = CachedFeatureExtractor(cache=cache)

                for image_path in test_images:
                    # First extraction (cache miss)
                    start_time = time.time()
                    first_result = extractor.extract_features(image_path)
                    first_time = time.time() - start_time

                    # Second extraction (cache hit)
                    start_time = time.time()
                    second_result = extractor.extract_features(image_path)
                    second_time = time.time() - start_time

                    speedup = first_time / second_time if second_time > 0 else float('inf')

                    cache_result = BenchmarkResult(
                        test_name=f"cache_performance_{Path(image_path).stem}",
                        success=first_result == second_result,
                        duration=second_time,
                        memory_delta_mb=0,
                        cpu_usage_percent=0,
                        output_size=len(str(second_result)),
                        quality_score=speedup,
                        error_message=None,
                        timestamp=start_time,
                        metadata={
                            'first_extraction_time': first_time,
                            'cached_extraction_time': second_time,
                            'speedup_ratio': speedup,
                            'cache_hit': True,
                            'results_match': first_result == second_result
                        }
                    )

                    cache_results.append(cache_result)
                    self.results.append(cache_result)

            except ImportError:
                return {'error': 'Cached components not available'}

        except ImportError:
            return {'error': 'Cache system not available'}

        # Analyze cache performance
        successful_cache = [r for r in cache_results if r.success]

        if not successful_cache:
            return {'error': 'No successful cache tests'}

        analysis = {
            'cache_available': True,
            'total_tests': len(cache_results),
            'successful_tests': len(successful_cache),
            'avg_speedup': statistics.mean([r.quality_score for r in successful_cache]),
            'min_speedup': min([r.quality_score for r in successful_cache]),
            'max_speedup': max([r.quality_score for r in successful_cache]),
            'cache_effectiveness': sum(1 for r in successful_cache if r.quality_score > 2.0) / len(successful_cache) * 100
        }

        print(f"Cache benchmark complete: {analysis['avg_speedup']:.1f}x average speedup")
        return analysis

    def benchmark_concurrent_performance(self, test_images: List[str], max_workers: int = 5) -> Dict[str, Any]:
        """Benchmark concurrent processing performance."""
        print(f"Running concurrent performance benchmarks with {max_workers} workers...")

        try:
            from backend.converters.vtracer_converter import VTracerConverter

            def convert_single_image(image_path):
                converter = VTracerConverter()
                start_time = time.time()
                memory_before = self._get_memory_usage()

                try:
                    result = converter.convert_with_metrics(image_path)
                    duration = time.time() - start_time
                    memory_after = self._get_memory_usage()

                    return {
                        'success': result['success'],
                        'duration': duration,
                        'memory_delta': memory_after - memory_before,
                        'output_size': len(result.get('svg', '')),
                        'image_path': image_path
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'duration': time.time() - start_time,
                        'error': str(e),
                        'image_path': image_path
                    }

            # Sequential processing baseline
            sequential_start = time.time()
            sequential_results = []

            for image_path in test_images:
                result = convert_single_image(image_path)
                sequential_results.append(result)

            sequential_time = time.time() - sequential_start

            # Concurrent processing
            concurrent_start = time.time()
            concurrent_results = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(convert_single_image, img): img for img in test_images}

                for future in as_completed(futures):
                    result = future.result()
                    concurrent_results.append(result)

            concurrent_time = time.time() - concurrent_start

            # Analyze concurrent performance
            sequential_successful = sum(1 for r in sequential_results if r['success'])
            concurrent_successful = sum(1 for r in concurrent_results if r['success'])

            analysis = {
                'sequential_time': sequential_time,
                'concurrent_time': concurrent_time,
                'speedup_ratio': sequential_time / concurrent_time if concurrent_time > 0 else 0,
                'sequential_success_rate': sequential_successful / len(sequential_results) * 100,
                'concurrent_success_rate': concurrent_successful / len(concurrent_results) * 100,
                'max_workers': max_workers,
                'throughput_improvement': (len(test_images) / concurrent_time) / (len(test_images) / sequential_time) if sequential_time > 0 else 0
            }

            print(f"Concurrent benchmark complete: {analysis['speedup_ratio']:.2f}x speedup with {max_workers} workers")
            return analysis

        except ImportError:
            return {'error': 'VTracer converter not available for concurrent testing'}

    def _analyze_logo_types(self, results: List[BenchmarkResult]) -> Dict[str, int]:
        """Analyze distribution of logo types in AI-enhanced results."""
        logo_types = {}
        for result in results:
            logo_type = result.metadata.get('logo_type', 'unknown')
            logo_types[logo_type] = logo_types.get(logo_type, 0) + 1
        return logo_types

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        print("Generating comprehensive performance report...")

        # Create test images
        test_images = self.create_test_images()

        # Run all benchmarks
        basic_results = self.benchmark_basic_conversion(test_images)
        ai_results = self.benchmark_ai_enhanced_conversion(test_images)
        cache_results = self.benchmark_cache_performance(test_images)
        concurrent_results = self.benchmark_concurrent_performance(test_images)

        # Overall analysis
        total_duration = time.time() - self.start_time

        report = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'total_duration': total_duration,
                'system_info': self.system_info,
                'test_images_count': len(test_images)
            },
            'results': {
                'basic_conversion': basic_results,
                'ai_enhanced_conversion': ai_results,
                'cache_performance': cache_results,
                'concurrent_performance': concurrent_results
            },
            'overall_statistics': self._calculate_overall_statistics(),
            'recommendations': self._generate_recommendations(),
            'detailed_results': [asdict(r) for r in self.results]
        }

        # Save report
        report_file = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Performance report saved to: {report_file}")

        # Cleanup test images
        for image_path in test_images:
            try:
                os.unlink(image_path)
            except:
                pass

        return report

    def _calculate_overall_statistics(self) -> Dict[str, Any]:
        """Calculate overall performance statistics."""
        if not self.results:
            return {'error': 'No results available'}

        successful_results = [r for r in self.results if r.success]

        if not successful_results:
            return {'error': 'No successful tests'}

        return {
            'total_tests': len(self.results),
            'successful_tests': len(successful_results),
            'overall_success_rate': len(successful_results) / len(self.results) * 100,
            'avg_duration': statistics.mean([r.duration for r in successful_results]),
            'median_duration': statistics.median([r.duration for r in successful_results]),
            'min_duration': min([r.duration for r in successful_results]),
            'max_duration': max([r.duration for r in successful_results]),
            'avg_memory_usage': statistics.mean([r.memory_delta_mb for r in successful_results]),
            'total_output_size': sum([r.output_size for r in successful_results])
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        successful_results = [r for r in self.results if r.success]

        if not successful_results:
            return ["No successful tests to analyze"]

        # Analyze performance patterns
        avg_duration = statistics.mean([r.duration for r in successful_results])
        max_duration = max([r.duration for r in successful_results])
        avg_memory = statistics.mean([r.memory_delta_mb for r in successful_results])

        # Duration recommendations
        if avg_duration > 2.0:
            recommendations.append("Average conversion time is slow (>2s). Consider optimizing parameters or upgrading hardware.")
        elif avg_duration < 0.5:
            recommendations.append("Excellent conversion speed achieved (<0.5s average).")

        if max_duration > 10.0:
            recommendations.append("Some conversions are very slow (>10s). Consider timeout limits and parameter optimization.")

        # Memory recommendations
        if avg_memory > 100:
            recommendations.append("High memory usage detected (>100MB per conversion). Consider memory optimization.")
        elif avg_memory < 20:
            recommendations.append("Good memory efficiency achieved (<20MB per conversion).")

        # AI-specific recommendations
        ai_results = [r for r in self.results if 'ai_enhanced' in r.metadata]
        if ai_results:
            ai_enhanced_count = sum(1 for r in ai_results if r.metadata.get('ai_enhanced', False))
            ai_rate = ai_enhanced_count / len(ai_results) * 100

            if ai_rate > 80:
                recommendations.append("Excellent AI enhancement rate (>80%). AI system is working well.")
            elif ai_rate < 50:
                recommendations.append("Low AI enhancement rate (<50%). Check AI module configuration.")

        # Cache recommendations
        cache_results = [r for r in self.results if 'speedup_ratio' in r.metadata]
        if cache_results:
            avg_speedup = statistics.mean([r.metadata['speedup_ratio'] for r in cache_results])

            if avg_speedup > 5.0:
                recommendations.append("Excellent cache performance (>5x speedup). Cache is highly effective.")
            elif avg_speedup < 2.0:
                recommendations.append("Cache performance could be improved (<2x speedup). Check cache configuration.")

        return recommendations

    def print_summary_report(self, report: Dict[str, Any]):
        """Print a summary of the performance report."""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)

        # System info
        system_info = report['benchmark_info']['system_info']
        print(f"System: {system_info['cpu_count']} CPUs, {system_info['memory_total_gb']:.1f}GB RAM")
        print(f"Test Duration: {report['benchmark_info']['total_duration']:.1f}s")
        print(f"Test Images: {report['benchmark_info']['test_images_count']}")

        # Overall statistics
        stats = report['overall_statistics']
        if 'error' not in stats:
            print(f"\nOverall Results:")
            print(f"  Success Rate: {stats['overall_success_rate']:.1f}%")
            print(f"  Average Duration: {stats['avg_duration']:.3f}s")
            print(f"  Memory Usage: {stats['avg_memory_usage']:.1f}MB")

        # Component results
        print(f"\nComponent Performance:")

        basic = report['results']['basic_conversion']
        if 'error' not in basic:
            print(f"  Basic Conversion: {basic['success_rate']:.1f}% success, {basic['avg_duration']:.3f}s avg")

        ai = report['results']['ai_enhanced_conversion']
        if 'error' not in ai and ai.get('ai_available'):
            print(f"  AI Enhanced: {ai['ai_enhancement_rate']:.1f}% AI rate, {ai['avg_duration']:.3f}s avg")

        cache = report['results']['cache_performance']
        if 'error' not in cache and cache.get('cache_available'):
            print(f"  Cache System: {cache['avg_speedup']:.1f}x speedup, {cache['cache_effectiveness']:.1f}% effective")

        concurrent = report['results']['concurrent_performance']
        if 'error' not in concurrent:
            print(f"  Concurrent Processing: {concurrent['speedup_ratio']:.1f}x speedup with {concurrent['max_workers']} workers")

        # Recommendations
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")

        print("="*80)


def main():
    """Run comprehensive performance benchmarks."""
    print("Starting comprehensive performance benchmark suite...")

    # Initialize benchmark suite
    suite = PerformanceBenchmarkSuite()

    # Generate comprehensive report
    report = suite.generate_comprehensive_report()

    # Print summary
    suite.print_summary_report(report)

    print(f"\nBenchmark complete! Full report available in: {suite.output_dir}")


if __name__ == "__main__":
    main()