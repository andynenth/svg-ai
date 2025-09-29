#!/usr/bin/env python3
"""
Performance Profiling and Optimization System

Provides comprehensive profiling tools for:
- Feature extraction performance analysis
- Memory usage optimization
- Image loading and preprocessing optimization
- Parallel processing implementation
- Performance regression testing
"""

import cProfile
import gc
import io
import os
import psutil
import pstats
import sys
import time
import threading
import tracemalloc
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
import json
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data"""
    execution_time: float
    memory_peak: int  # bytes
    memory_current: int  # bytes
    cpu_percent: float
    function_name: str
    timestamp: float
    input_size: Optional[int] = None
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_time': self.execution_time,
            'memory_peak': self.memory_peak,
            'memory_current': self.memory_current,
            'cpu_percent': self.cpu_percent,
            'function_name': self.function_name,
            'timestamp': self.timestamp,
            'input_size': self.input_size,
            'cache_hit': self.cache_hit
        }


class PerformanceProfiler:
    """Advanced performance profiler for AI pipeline components"""

    def __init__(self, enable_memory_tracking: bool = True):
        self.enable_memory_tracking = enable_memory_tracking
        self.metrics_history = deque(maxlen=1000)
        self.function_stats = defaultdict(list)
        self.process = psutil.Process()
        self.lock = threading.Lock()

        if enable_memory_tracking:
            tracemalloc.start()

    @contextmanager
    def profile_block(self, block_name: str, input_size: Optional[int] = None):
        """Context manager for profiling code blocks"""
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss if not self.enable_memory_tracking else None
        start_cpu = self.process.cpu_percent()

        if self.enable_memory_tracking:
            snapshot_start = tracemalloc.take_snapshot()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            current_memory = self.process.memory_info().rss
            cpu_percent = self.process.cpu_percent()

            peak_memory = current_memory
            if self.enable_memory_tracking:
                snapshot_end = tracemalloc.take_snapshot()
                top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
                if top_stats:
                    peak_memory = max(stat.size for stat in top_stats)

            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_peak=peak_memory,
                memory_current=current_memory,
                cpu_percent=cpu_percent,
                function_name=block_name,
                timestamp=time.time(),
                input_size=input_size
            )

            with self.lock:
                self.metrics_history.append(metrics)
                self.function_stats[block_name].append(metrics)

    def profile_function(self, func: Callable) -> Callable:
        """Decorator for profiling function performance"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            input_size = None

            # Try to determine input size from common patterns
            if args:
                first_arg = args[0]
                if isinstance(first_arg, str) and os.path.exists(first_arg):
                    try:
                        input_size = os.path.getsize(first_arg)
                    except OSError:
                        pass
                elif isinstance(first_arg, np.ndarray):
                    input_size = first_arg.nbytes

            with self.profile_block(func.__name__, input_size):
                return func(*args, **kwargs)

        return wrapper

    def get_function_stats(self, function_name: str) -> Dict[str, Any]:
        """Get aggregated statistics for a specific function"""
        with self.lock:
            metrics_list = self.function_stats.get(function_name, [])

            if not metrics_list:
                return {}

            execution_times = [m.execution_time for m in metrics_list]
            memory_peaks = [m.memory_peak for m in metrics_list]

            return {
                'call_count': len(metrics_list),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'total_execution_time': sum(execution_times),
                'avg_memory_peak': sum(memory_peaks) / len(memory_peaks),
                'max_memory_peak': max(memory_peaks),
                'last_call': metrics_list[-1].timestamp
            }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self.lock:
            total_functions = len(self.function_stats)
            total_calls = sum(len(stats) for stats in self.function_stats.values())

            function_reports = {}
            for func_name in self.function_stats.keys():
                function_reports[func_name] = self.get_function_stats(func_name)

            # Find performance bottlenecks
            bottlenecks = sorted(
                function_reports.items(),
                key=lambda x: x[1].get('total_execution_time', 0),
                reverse=True
            )[:5]

            return {
                'summary': {
                    'total_functions_profiled': total_functions,
                    'total_function_calls': total_calls,
                    'profiling_enabled': True,
                    'memory_tracking_enabled': self.enable_memory_tracking
                },
                'function_stats': function_reports,
                'top_bottlenecks': [
                    {
                        'function': name,
                        'total_time': stats['total_execution_time'],
                        'avg_time': stats['avg_execution_time'],
                        'call_count': stats['call_count']
                    }
                    for name, stats in bottlenecks if stats
                ],
                'recommendations': self._generate_optimization_recommendations(function_reports)
            }

    def _generate_optimization_recommendations(self, function_stats: Dict) -> List[str]:
        """Generate optimization recommendations based on profiling data"""
        recommendations = []

        for func_name, stats in function_stats.items():
            if not stats:
                continue

            avg_time = stats.get('avg_execution_time', 0)
            max_time = stats.get('max_execution_time', 0)
            call_count = stats.get('call_count', 0)

            # High average execution time
            if avg_time > 1.0:  # > 1 second
                recommendations.append(
                    f"🔴 {func_name}: High average execution time ({avg_time:.2f}s) - consider optimization"
                )

            # High variance in execution time
            if max_time > avg_time * 3:  # Max is 3x average
                recommendations.append(
                    f"🟡 {func_name}: High execution time variance - investigate input size impact"
                )

            # Frequently called slow functions
            if call_count > 10 and avg_time > 0.1:
                recommendations.append(
                    f"🟡 {func_name}: Frequently called ({call_count} times) - consider caching"
                )

            # Memory intensive functions
            max_memory = stats.get('max_memory_peak', 0)
            if max_memory > 100 * 1024 * 1024:  # > 100MB
                recommendations.append(
                    f"🔴 {func_name}: High memory usage ({max_memory / (1024*1024):.1f}MB) - optimize memory usage"
                )

        if not recommendations:
            recommendations.append("✅ No significant performance issues detected")

        return recommendations


class ImageLoadingOptimizer:
    """Optimized image loading and preprocessing"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.load_cache = {}  # Simple in-memory cache for recently loaded images
        self.max_cache_size = 50

    @staticmethod
    def get_optimal_image_size(image_path: str, target_max_dimension: int = 1024) -> Tuple[int, int]:
        """Determine optimal image size for processing"""
        try:
            # Read image dimensions without loading full image
            with open(image_path, 'rb') as f:
                # For PNG files, dimensions are at bytes 16-20 and 20-24
                if image_path.lower().endswith('.png'):
                    f.seek(16)
                    width = int.from_bytes(f.read(4), 'big')
                    height = int.from_bytes(f.read(4), 'big')
                else:
                    # Fallback to cv2 for other formats
                    img = cv2.imread(image_path)
                    if img is not None:
                        height, width = img.shape[:2]
                    else:
                        return target_max_dimension, target_max_dimension

            # Calculate optimal size maintaining aspect ratio
            if max(width, height) <= target_max_dimension:
                return width, height

            if width > height:
                new_width = target_max_dimension
                new_height = int(height * target_max_dimension / width)
            else:
                new_height = target_max_dimension
                new_width = int(width * target_max_dimension / height)

            return new_width, new_height

        except Exception as e:
            logger.warning(f"Error determining optimal size for {image_path}: {e}")
            return target_max_dimension, target_max_dimension

    def load_image_optimized(self, image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Load image with optimizations"""
        cache_key = f"{image_path}:{target_size}"

        # Check cache first
        if cache_key in self.load_cache:
            logger.debug(f"Image cache hit: {Path(image_path).name}")
            return self.load_cache[cache_key].copy()

        with self.profiler.profile_block("load_image_optimized", os.path.getsize(image_path)):
            # Determine optimal size if not specified
            if target_size is None:
                target_size = self.get_optimal_image_size(image_path)

            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Resize if needed
            current_size = (img.shape[1], img.shape[0])
            if current_size != target_size:
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

            # Cache the result
            if len(self.load_cache) >= self.max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.load_cache))
                del self.load_cache[oldest_key]

            self.load_cache[cache_key] = img.copy()

            logger.debug(f"Loaded and cached image: {Path(image_path).name} -> {img.shape}")
            return img

    def preprocess_batch(self, image_paths: List[str], target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """Optimized batch preprocessing"""
        with self.profiler.profile_block("preprocess_batch", len(image_paths)):
            results = []

            for image_path in image_paths:
                try:
                    img = self.load_image_optimized(image_path, target_size)
                    results.append(img)
                except Exception as e:
                    logger.error(f"Error preprocessing {image_path}: {e}")
                    results.append(None)

            return results

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get image loading optimization statistics"""
        return {
            'cache_size': len(self.load_cache),
            'max_cache_size': self.max_cache_size,
            'profiling_stats': self.profiler.get_performance_report()
        }


class MemoryOptimizer:
    """Memory usage optimization and garbage collection management"""

    def __init__(self):
        self.memory_snapshots = deque(maxlen=100)
        self.gc_stats = defaultdict(int)
        self.process = psutil.Process()

    def take_memory_snapshot(self, label: str = ""):
        """Take memory usage snapshot"""
        memory_info = self.process.memory_info()
        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'rss': memory_info.rss,  # Resident Set Size
            'vms': memory_info.vms,  # Virtual Memory Size
            'percent': self.process.memory_percent(),
            'available': psutil.virtual_memory().available
        }
        self.memory_snapshots.append(snapshot)
        return snapshot

    def optimize_memory(self, aggressive: bool = False):
        """Perform memory optimization"""
        before_snapshot = self.take_memory_snapshot("before_optimization")

        # Force garbage collection
        collected = gc.collect()
        self.gc_stats['manual_collections'] += 1
        self.gc_stats['objects_collected'] += collected

        if aggressive:
            # More aggressive optimization
            for generation in range(3):
                collected += gc.collect(generation)

            # Clear various caches
            sys.intern.__class__.clear()  # Clear string intern cache

        after_snapshot = self.take_memory_snapshot("after_optimization")

        memory_freed = before_snapshot['rss'] - after_snapshot['rss']

        logger.info(f"Memory optimization: freed {memory_freed / (1024*1024):.1f}MB, "
                   f"collected {collected} objects")

        return {
            'memory_freed_bytes': memory_freed,
            'objects_collected': collected,
            'before_memory_mb': before_snapshot['rss'] / (1024*1024),
            'after_memory_mb': after_snapshot['rss'] / (1024*1024)
        }

    def get_memory_trends(self, minutes: int = 30) -> Dict[str, Any]:
        """Analyze memory usage trends"""
        cutoff_time = time.time() - (minutes * 60)
        recent_snapshots = [s for s in self.memory_snapshots if s['timestamp'] >= cutoff_time]

        if len(recent_snapshots) < 2:
            return {'insufficient_data': True}

        rss_values = [s['rss'] for s in recent_snapshots]

        return {
            'sample_count': len(recent_snapshots),
            'current_memory_mb': rss_values[-1] / (1024*1024),
            'peak_memory_mb': max(rss_values) / (1024*1024),
            'min_memory_mb': min(rss_values) / (1024*1024),
            'avg_memory_mb': sum(rss_values) / len(rss_values) / (1024*1024),
            'memory_growth_mb': (rss_values[-1] - rss_values[0]) / (1024*1024),
            'gc_stats': dict(self.gc_stats)
        }

    @contextmanager
    def memory_limit_context(self, max_memory_mb: int = 512):
        """Context manager that monitors memory usage and optimizes if needed"""
        initial_memory = self.process.memory_info().rss / (1024*1024)

        try:
            yield
        finally:
            current_memory = self.process.memory_info().rss / (1024*1024)

            if current_memory > max_memory_mb:
                logger.warning(f"Memory limit exceeded: {current_memory:.1f}MB > {max_memory_mb}MB")
                self.optimize_memory(aggressive=True)


class ParallelProcessor:
    """Parallel processing implementation for batch operations"""

    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        self.max_workers = max_workers or min(32, os.cpu_count() + 4)
        self.use_processes = use_processes
        self.profiler = PerformanceProfiler()

    def process_batch_parallel(self, func: Callable, items: List[Any],
                             chunk_size: Optional[int] = None) -> List[Any]:
        """Process items in parallel using thread or process pool"""
        if chunk_size is None:
            chunk_size = max(1, len(items) // self.max_workers)

        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        with self.profiler.profile_block(f"parallel_batch_{func.__name__}", len(items)):
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(func, item): i
                    for i, item in enumerate(items)
                }

                # Collect results in order
                results = [None] * len(items)
                completed_count = 0

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                        completed_count += 1

                        if completed_count % max(1, len(items) // 10) == 0:
                            logger.debug(f"Batch progress: {completed_count}/{len(items)}")

                    except Exception as e:
                        logger.error(f"Error processing item {index}: {e}")
                        results[index] = None

                return results

    def benchmark_parallel_vs_sequential(self, func: Callable, items: List[Any]) -> Dict[str, Any]:
        """Benchmark parallel vs sequential processing"""
        # Sequential benchmark
        start_time = time.perf_counter()
        sequential_results = [func(item) for item in items]
        sequential_time = time.perf_counter() - start_time

        # Parallel benchmark
        start_time = time.perf_counter()
        parallel_results = self.process_batch_parallel(func, items)
        parallel_time = time.perf_counter() - start_time

        speedup = sequential_time / parallel_time if parallel_time > 0 else 0

        return {
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'efficiency': speedup / self.max_workers,
            'parallel_faster': parallel_time < sequential_time,
            'item_count': len(items),
            'workers_used': self.max_workers
        }


class PerformanceRegressionTester:
    """Performance regression testing system"""

    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.baseline_data = self._load_baseline()
        self.profiler = PerformanceProfiler()

    def _load_baseline(self) -> Dict[str, Any]:
        """Load performance baseline data"""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading baseline data: {e}")
        return {}

    def save_baseline(self, performance_data: Dict[str, Any]):
        """Save current performance as baseline"""
        with open(self.baseline_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
        logger.info(f"Saved performance baseline to {self.baseline_file}")

    def run_performance_test(self, test_name: str, test_func: Callable,
                           test_args: List[Any], iterations: int = 5) -> Dict[str, Any]:
        """Run performance test with multiple iterations"""
        execution_times = []
        memory_peaks = []

        for i in range(iterations):
            with self.profiler.profile_block(f"{test_name}_iteration_{i}"):
                test_func(*test_args)

            # Get metrics from last execution
            if self.profiler.metrics_history:
                last_metric = self.profiler.metrics_history[-1]
                execution_times.append(last_metric.execution_time)
                memory_peaks.append(last_metric.memory_peak)

        if not execution_times:
            return {'error': 'No performance data collected'}

        return {
            'test_name': test_name,
            'iterations': iterations,
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'execution_time_std': np.std(execution_times),
            'avg_memory_peak': sum(memory_peaks) / len(memory_peaks),
            'max_memory_peak': max(memory_peaks),
            'timestamp': time.time()
        }

    def compare_with_baseline(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current performance with baseline"""
        test_name = current_results['test_name']
        baseline = self.baseline_data.get(test_name, {})

        if not baseline:
            return {'status': 'no_baseline', 'message': 'No baseline data available'}

        current_time = current_results['avg_execution_time']
        baseline_time = baseline.get('avg_execution_time', 0)

        current_memory = current_results['avg_memory_peak']
        baseline_memory = baseline.get('avg_memory_peak', 0)

        time_regression = (current_time - baseline_time) / baseline_time if baseline_time > 0 else 0
        memory_regression = (current_memory - baseline_memory) / baseline_memory if baseline_memory > 0 else 0

        # Determine regression status
        status = 'pass'
        issues = []

        if time_regression > 0.1:  # 10% slower
            status = 'regression'
            issues.append(f"Execution time regression: {time_regression:.1%}")

        if memory_regression > 0.2:  # 20% more memory
            status = 'regression'
            issues.append(f"Memory usage regression: {memory_regression:.1%}")

        return {
            'status': status,
            'time_regression_percent': time_regression * 100,
            'memory_regression_percent': memory_regression * 100,
            'current_time': current_time,
            'baseline_time': baseline_time,
            'current_memory_mb': current_memory / (1024*1024),
            'baseline_memory_mb': baseline_memory / (1024*1024),
            'issues': issues
        }


# Global profiler instance
_global_profiler = None


def get_global_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_performance(func: Callable) -> Callable:
    """Decorator for performance profiling"""
    return get_global_profiler().profile_function(func)


def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary"""
    profiler = get_global_profiler()
    return profiler.get_performance_report()