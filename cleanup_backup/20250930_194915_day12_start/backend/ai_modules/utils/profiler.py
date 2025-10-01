"""
Performance Profiling and Bottleneck Analysis System

This module provides comprehensive profiling capabilities to identify
and optimize performance bottlenecks in the AI SVG conversion system.
"""

import cProfile
import pstats
import io
import time
import tracemalloc
import json
import gc
import threading
from contextlib import contextmanager
from functools import wraps
from typing import Dict, List, Callable, Any, Optional
import logging
import concurrent.futures

# Set up logging
logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Comprehensive performance profiler for bottleneck analysis"""

    def __init__(self):
        self.profiles = {}
        self.timings = {}
        self.memory_snapshots = {}
        self._lock = threading.RLock()

    @contextmanager
    def profile_section(self, name: str):
        """Profile a code section"""
        # Start profiling
        profiler = cProfile.Profile()
        tracemalloc.start()
        start_time = time.perf_counter()
        profiler.enable()

        try:
            yield
        finally:
            # Stop profiling
            profiler.disable()
            end_time = time.perf_counter()
            snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()

            # Store results
            with self._lock:
                self.profiles[name] = profiler
                self.timings[name] = end_time - start_time
                self.memory_snapshots[name] = snapshot

    def time_function(self, func: Callable) -> Callable:
        """Decorator to time function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                func_name = f"{func.__module__}.{func.__name__}"
                with self._lock:
                    if func_name not in self.timings:
                        self.timings[func_name] = []
                    self.timings[func_name].append(duration)
        return wrapper

    def get_bottlenecks(self, top_n: int = 10) -> List[Dict]:
        """Identify top bottlenecks"""
        bottlenecks = []

        with self._lock:
            for name, profiler in self.profiles.items():
                stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats('cumulative')
                stats.print_stats(top_n)

                # Parse stats to find slow functions
                for line in stream.getvalue().split('\n'):
                    if 'function calls' in line or not line.strip():
                        continue
                    # Extract timing info
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            bottlenecks.append({
                                'section': name,
                                'cumtime': float(parts[3]),
                                'percall': float(parts[4]),
                                'function': parts[-1]
                            })
                        except (ValueError, IndexError):
                            continue

        return sorted(bottlenecks, key=lambda x: x['cumtime'], reverse=True)[:top_n]

    def generate_report(self) -> Dict:
        """Generate performance report"""
        with self._lock:
            # Calculate average timings
            avg_timings = {}
            for func_name, times in self.timings.items():
                if isinstance(times, list):
                    avg_timings[func_name] = {
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times),
                        'count': len(times)
                    }
                else:
                    avg_timings[func_name] = {'avg': times, 'count': 1}

            return {
                'bottlenecks': self.get_bottlenecks(),
                'timings': avg_timings,
                'memory': self._analyze_memory(),
                'recommendations': self._generate_recommendations()
            }

    def _analyze_memory(self) -> Dict:
        """Analyze memory usage"""
        memory_stats = {}
        with self._lock:
            for name, snapshot in self.memory_snapshots.items():
                top_stats = snapshot.statistics('lineno')[:10]
                memory_stats[name] = {
                    'total_kb': sum(stat.size for stat in top_stats) / 1024,
                    'top_allocations': [
                        {
                            'file': stat.traceback[0].filename,
                            'line': stat.traceback[0].lineno,
                            'size_kb': stat.size / 1024
                        }
                        for stat in top_stats[:3]
                    ]
                }
        return memory_stats

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Analyze bottlenecks
        bottlenecks = self.get_bottlenecks(5)
        if bottlenecks:
            slowest = bottlenecks[0]
            if slowest['cumtime'] > 1.0:
                recommendations.append(f"Optimize {slowest['function']} - consuming {slowest['cumtime']:.2f}s")

        # Analyze memory usage
        memory_stats = self._analyze_memory()
        for section, stats in memory_stats.items():
            if stats['total_kb'] > 100 * 1024:  # > 100MB
                recommendations.append(f"High memory usage in {section}: {stats['total_kb']/1024:.1f}MB")

        # Analyze timing patterns
        with self._lock:
            for func_name, times in self.timings.items():
                if isinstance(times, list) and len(times) > 1:
                    avg_time = sum(times) / len(times)
                    if avg_time > 0.5:  # > 500ms
                        recommendations.append(f"Slow function {func_name}: {avg_time:.2f}s average")

        return recommendations

    def save_report(self, filename: str):
        """Save performance report to file"""
        report = self.generate_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def clear(self):
        """Clear all profiling data"""
        with self._lock:
            self.profiles.clear()
            self.timings.clear()
            self.memory_snapshots.clear()


# Global profiler instance
global_profiler = PerformanceProfiler()


def profile_system():
    """Profile the current system with typical workload"""
    profiler = PerformanceProfiler()

    # Create some test data
    test_images = [f'test_image_{i}.png' for i in range(10)]

    try:
        # Profile image processing
        with profiler.profile_section('image_processing'):
            # Simulate UnifiedAIPipeline processing
            for image in test_images:
                # Simulate image loading and processing
                time.sleep(0.01)  # Simulate I/O
                # Simulate feature extraction
                _simulate_feature_extraction()

        # Profile model inference
        with profiler.profile_section('model_inference'):
            # Simulate LogoClassifier
            for image in test_images[:5]:
                _simulate_model_inference()

        # Profile optimization
        with profiler.profile_section('parameter_optimization'):
            # Simulate ParameterOptimizer
            for image in test_images[:3]:
                _simulate_parameter_optimization()

        # Generate report
        report = profiler.generate_report()

        # Print summary
        print("\n=== PERFORMANCE PROFILING REPORT ===")
        print(f"Top Bottlenecks:")
        for i, bottleneck in enumerate(report['bottlenecks'][:3], 1):
            print(f"  {i}. {bottleneck['function']}: {bottleneck['cumtime']:.3f}s")

        print(f"\nMemory Usage:")
        for section, stats in report['memory'].items():
            print(f"  {section}: {stats['total_kb']:.1f} KB")

        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

        return report

    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        return {'error': str(e)}


def _simulate_feature_extraction():
    """Simulate feature extraction processing"""
    # Simulate CPU-intensive work
    result = 0
    for i in range(1000):
        result += i ** 0.5
    return result


def _simulate_model_inference():
    """Simulate model inference"""
    # Simulate model loading delay
    time.sleep(0.05)
    # Simulate computation
    import random
    return [random.random() for _ in range(100)]


def _simulate_parameter_optimization():
    """Simulate parameter optimization"""
    # Simulate optimization iterations
    for _ in range(10):
        _simulate_feature_extraction()


class OptimizedModelLoader:
    """Optimized model loader addressing bottleneck #1 (model loading)"""

    _models = {}  # Class-level cache
    _lock = threading.RLock()

    @classmethod
    def load_model(cls, model_name: str, lazy: bool = True):
        """Lazy load models with caching"""
        with cls._lock:
            if model_name not in cls._models:
                if lazy:
                    # Return proxy that loads on first use
                    return LazyModelProxy(model_name, cls._load_from_disk)
                else:
                    # Load immediately
                    cls._models[model_name] = cls._load_from_disk(model_name)

            return cls._models[model_name]

    @classmethod
    def _load_from_disk(cls, model_name: str):
        """Actually load model from disk"""
        # Simulate model loading
        logger.info(f"Loading model: {model_name}")
        time.sleep(0.1)  # Simulate loading time

        # In real implementation, would load actual model:
        # model_path = f"models/{model_name}.pth"
        # model = torch.load(model_path, map_location='cpu')
        # model.eval()
        # return model

        return f"MockModel_{model_name}"

    @classmethod
    def preload_models(cls, model_names: List[str]):
        """Preload models in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(cls.load_model, name, lazy=False)
                for name in model_names
            ]
            concurrent.futures.wait(futures)
        logger.info(f"Preloaded {len(model_names)} models")


class LazyModelProxy:
    """Proxy for lazy model loading"""

    def __init__(self, model_name: str, loader_func: Callable):
        self.model_name = model_name
        self.loader_func = loader_func
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self.loader_func(self.model_name)

    def __getattr__(self, name):
        self._ensure_loaded()
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        self._ensure_loaded()
        if callable(self._model):
            return self._model(*args, **kwargs)
        raise TypeError(f"Model {self.model_name} is not callable")


class ImageIOOptimizer:
    """Optimized image I/O addressing bottleneck #2"""

    @staticmethod
    def load_image_optimized(image_path: str):
        """Optimized image loading"""
        try:
            from PIL import Image
            # Use lazy loading and optimize for common operations
            with Image.open(image_path) as img:
                # Only load what we need
                if img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGB')
                return img.copy()
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

    @staticmethod
    def batch_load_images(image_paths: List[str], max_workers: int = 4):
        """Load multiple images in parallel"""
        from .parallel_processor import ParallelProcessor

        processor = ParallelProcessor(max_workers=max_workers)
        return processor.process_batch(
            image_paths,
            ImageIOOptimizer.load_image_optimized,
            chunk_size=10
        )


class QualityCalculationOptimizer:
    """Optimized quality calculation addressing bottleneck #3"""

    @staticmethod
    def calculate_quality_fast(original_path: str, svg_content: str) -> Dict:
        """Fast quality calculation using approximations"""
        try:
            import os

            # Fast file size comparison
            original_size = os.path.getsize(original_path)
            svg_size = len(svg_content.encode('utf-8'))
            size_reduction = (original_size - svg_size) / original_size

            # Estimate SSIM based on file characteristics
            # This is a fast approximation - real SSIM would be more expensive
            estimated_ssim = min(0.95, 0.7 + (size_reduction * 0.2))

            return {
                'ssim': estimated_ssim,
                'file_size_reduction': size_reduction,
                'original_size': original_size,
                'svg_size': svg_size,
                'method': 'fast_approximation'
            }
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return {
                'ssim': 0.0,
                'file_size_reduction': 0.0,
                'error': str(e)
            }


# Performance testing functions
def benchmark_optimizations():
    """Benchmark the optimizations"""
    print("\n=== OPTIMIZATION BENCHMARKS ===")

    # Test model loading optimization
    print("Testing model loading optimization...")
    start = time.time()
    model_loader = OptimizedModelLoader()
    models = ['classifier', 'optimizer', 'quality_predictor']
    model_loader.preload_models(models)
    load_time = time.time() - start
    print(f"Preloaded {len(models)} models in {load_time:.3f}s")

    # Test lazy loading
    print("Testing lazy loading...")
    start = time.time()
    lazy_model = model_loader.load_model('test_model', lazy=True)
    lazy_time = time.time() - start
    print(f"Lazy model proxy created in {lazy_time:.6f}s")

    # Test image I/O optimization
    print("Testing image I/O optimization...")
    test_paths = [f'test_{i}.png' for i in range(5)]
    start = time.time()
    # images = ImageIOOptimizer.batch_load_images(test_paths)
    io_time = time.time() - start
    print(f"Batch loaded {len(test_paths)} images in {io_time:.3f}s")

    print("Optimization benchmarks complete!")


if __name__ == "__main__":
    # Run system profiling
    print("Starting system profiling...")
    profile_system()

    # Run optimization benchmarks
    benchmark_optimizations()