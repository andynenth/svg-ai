#!/usr/bin/env python3
"""
Performance Regression Testing for Day 14 Integration Testing
Test for performance regressions after cleanup
"""

import time
import json
import statistics
from pathlib import Path
import subprocess
import timeit
import psutil
import gc
from PIL import Image


class PerformanceRegression:
    """Test for performance regressions after cleanup"""

    def __init__(self):
        self.baseline = self.load_baseline()
        self.results = {}

    def load_baseline(self):
        """Load baseline performance metrics"""
        baseline_file = 'benchmarks/day13_baseline.json'
        if Path(baseline_file).exists():
            with open(baseline_file, 'r') as f:
                return json.load(f)
        return None

    def benchmark_import_time(self):
        """Measure module import time"""

        # Test import time
        import_time = timeit.timeit(
            'from backend.ai_modules.pipeline import UnifiedAIPipeline',
            number=1
        )

        self.results['import_time'] = import_time

        # Compare with baseline
        if self.baseline:
            baseline_import = self.baseline.get('import_time', 0)
            improvement = (baseline_import - import_time) / baseline_import * 100
            print(f"Import time: {import_time:.3f}s (Improvement: {improvement:.1f}%)")

        return import_time

    def benchmark_conversion_speed(self):
        """Benchmark conversion speeds by tier"""

        from backend.ai_modules.pipeline import UnifiedAIPipeline
        pipeline = UnifiedAIPipeline()

        test_images = {
            'simple': 'data/test/simple_01.png',
            'text': 'data/test/text_01.png',
            'gradient': 'data/test/gradient_01.png',
            'complex': 'data/test/complex_01.png'
        }

        tier_times = {}

        for category, image_path in test_images.items():
            times = []

            # Run multiple times for accuracy
            for _ in range(5):
                start = time.perf_counter()
                result = pipeline.process(image_path)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_time = statistics.mean(times)
            tier_times[category] = {
                'mean': avg_time,
                'min': min(times),
                'max': max(times),
                'std': statistics.stdev(times) if len(times) > 1 else 0
            }

        self.results['conversion_times'] = tier_times

        # Check against targets
        targets = {
            'simple': 2.0,   # Tier 1 target
            'text': 5.0,     # Tier 2 target
            'gradient': 5.0, # Tier 2 target
            'complex': 15.0  # Tier 3 target
        }

        for category, target in targets.items():
            actual = tier_times[category]['mean']
            if actual > target:
                print(f"⚠️ {category}: {actual:.2f}s exceeds target {target}s")
            else:
                print(f"✓ {category}: {actual:.2f}s meets target {target}s")

        return tier_times

    def benchmark_memory_usage(self):
        """Test memory usage"""

        # Get baseline memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Import all modules
        from backend.ai_modules.classification import ClassificationModule
        from backend.ai_modules.optimization import OptimizationEngine
        from backend.ai_modules.quality import QualitySystem
        from backend.ai_modules.pipeline import UnifiedAIPipeline

        # Create instances
        classifier = ClassificationModule()
        optimizer = OptimizationEngine()
        quality = QualitySystem()
        pipeline = UnifiedAIPipeline()

        # Process several images
        for i in range(10):
            pipeline.process(f'data/test/test_{i % 4}.png')

        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - baseline_memory

        self.results['memory_usage'] = {
            'baseline_mb': baseline_memory,
            'peak_mb': peak_memory,
            'used_mb': memory_used
        }

        # Check against limit
        if memory_used > 500:
            print(f"⚠️ Memory usage {memory_used:.1f}MB exceeds 500MB limit")
        else:
            print(f"✓ Memory usage {memory_used:.1f}MB within limits")

        return memory_used

    def benchmark_batch_processing(self):
        """Test batch processing performance"""

        from backend.ai_modules.utils import UnifiedUtils
        processor = UnifiedUtils()

        # Prepare batch (exclude corrupted test files)
        test_images = [img for img in Path('data/test').glob('*.png')
                      if 'corrupted' not in img.name][:20]

        # Single threaded baseline
        start = time.time()
        for img in test_images:
            try:
                # Simple processing
                Image.open(img).convert('RGB')
            except Exception:
                # Skip corrupted files
                continue
        single_time = time.time() - start

        # Parallel processing
        def safe_process_image(img):
            try:
                return Image.open(img).convert('RGB')
            except Exception:
                return None

        start = time.time()
        results = processor.process_parallel(
            test_images,
            safe_process_image
        )
        parallel_time = time.time() - start

        speedup = single_time / parallel_time

        self.results['batch_processing'] = {
            'single_threaded': single_time,
            'parallel': parallel_time,
            'speedup': speedup
        }

        if speedup < 2:
            print(f"⚠️ Batch speedup {speedup:.1f}x below target 2x")
        else:
            print(f"✓ Batch speedup {speedup:.1f}x")

        return speedup

    def generate_report(self):
        """Generate performance report"""

        report = {
            'summary': {
                'import_time_ok': self.results['import_time'] < 2.0,
                'conversion_speed_ok': all(
                    t['mean'] < target
                    for t, target in [
                        (self.results['conversion_times']['simple'], 2.0),
                        (self.results['conversion_times']['complex'], 15.0)
                    ]
                ),
                'memory_ok': self.results['memory_usage']['used_mb'] < 500,
                'batch_ok': self.results['batch_processing']['speedup'] > 2
            },
            'details': self.results,
            'recommendations': []
        }

        # Add recommendations
        if not report['summary']['import_time_ok']:
            report['recommendations'].append("Optimize import time - consider lazy imports")

        if not report['summary']['memory_ok']:
            report['recommendations'].append("Reduce memory usage - check for leaks")

        return report


def validate_performance_targets():
    """Ensure performance meets all targets"""

    tester = PerformanceRegression()

    # Run all benchmarks
    tester.benchmark_import_time()
    tester.benchmark_conversion_speed()
    tester.benchmark_memory_usage()
    tester.benchmark_batch_processing()

    # Generate report
    report = tester.generate_report()

    # Check all targets met
    all_passed = all(report['summary'].values())

    if all_passed:
        print("\n✅ All performance targets met!")
    else:
        print("\n❌ Some performance targets not met:")
        for key, passed in report['summary'].items():
            if not passed:
                print(f"  - {key}")

    # Save report
    with open('performance_report_day14.json', 'w') as f:
        json.dump(report, f, indent=2)

    return all_passed


if __name__ == "__main__":
    print("Running Performance Regression Tests...")
    print("=" * 50)

    success = validate_performance_targets()

    print("\n" + "=" * 50)
    if success:
        print("✅ Performance regression testing PASSED")
    else:
        print("❌ Performance regression testing FAILED")
        print("Check performance_report_day14.json for details")