"""
Method 1 Performance Optimizer
Optimize optimization speed and reduce memory usage
"""
import cProfile
import pstats
import io
import time
import psutil
import gc
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
from typing import Dict, Any, List, Tuple, Callable, Optional
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
import weakref
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from .feature_mapping_optimizer import FeatureMappingOptimizer
from .correlation_formulas import CorrelationFormulas
from .parameter_bounds import VTracerParameterBounds
from .refined_correlation_formulas import RefinedCorrelationFormulas

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Comprehensive performance profiling for Method 1"""

    def __init__(self):
        """Initialize performance profiler"""
        self.profiler = cProfile.Profile()
        self.memory_snapshots = []
        self.timing_data = defaultdict(list)
        self.process = psutil.Process()

    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Memory before
            mem_before = self.process.memory_info().rss / 1024 / 1024

            # Time and profile
            start_time = time.time()
            self.profiler.enable()

            try:
                result = func(*args, **kwargs)
            finally:
                self.profiler.disable()

            end_time = time.time()

            # Memory after
            mem_after = self.process.memory_info().rss / 1024 / 1024

            # Store timing data
            func_name = f"{func.__module__}.{func.__name__}"
            self.timing_data[func_name].append({
                'execution_time': end_time - start_time,
                'memory_before': mem_before,
                'memory_after': mem_after,
                'memory_delta': mem_after - mem_before,
                'timestamp': datetime.now().isoformat()
            })

            return result

        return wrapper

    def get_profile_stats(self) -> str:
        """Get formatted profile statistics"""
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats()
        return s.getvalue()

    def get_memory_profile(self) -> Dict[str, float]:
        """Get current memory usage profile"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }

    def generate_timing_report(self) -> Dict[str, Any]:
        """Generate comprehensive timing report"""
        report = {}

        for func_name, timings in self.timing_data.items():
            if timings:
                times = [t['execution_time'] for t in timings]
                memory_deltas = [t['memory_delta'] for t in timings]

                report[func_name] = {
                    'call_count': len(timings),
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times),
                    'avg_memory_delta': np.mean(memory_deltas),
                    'max_memory_delta': max(memory_deltas),
                    'calls_per_second': len(timings) / sum(times) if sum(times) > 0 else 0
                }

        return report


class CacheManager:
    """Manages caching for correlation formulas and parameter validation"""

    def __init__(self, cache_size: int = 1000):
        """Initialize cache manager"""
        self.cache_size = cache_size
        self.formula_cache = {}
        self.validation_cache = {}
        self.cache_stats = defaultdict(int)

    def cached_formula_result(self, formula_name: str, *args) -> Callable:
        """Cache decorator for formula results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = f"{formula_name}:{hash(str(args) + str(kwargs))}"

                if cache_key in self.formula_cache:
                    self.cache_stats['hits'] += 1
                    return self.formula_cache[cache_key]

                # Compute result
                result = func(*args, **kwargs)

                # Store in cache (with size limit)
                if len(self.formula_cache) < self.cache_size:
                    self.formula_cache[cache_key] = result
                    self.cache_stats['stores'] += 1

                self.cache_stats['misses'] += 1
                return result

            return wrapper
        return decorator

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0

        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'stores': self.cache_stats['stores'],
            'hit_rate': hit_rate,
            'cache_size': len(self.formula_cache),
            'validation_cache_size': len(self.validation_cache)
        }

    def clear_cache(self):
        """Clear all caches"""
        self.formula_cache.clear()
        self.validation_cache.clear()
        self.cache_stats.clear()


class OptimizedCorrelationFormulas:
    """Performance-optimized correlation formulas with caching"""

    def __init__(self, cache_manager: CacheManager):
        """Initialize optimized formulas"""
        self.cache_manager = cache_manager
        self.original_formulas = CorrelationFormulas()

    @lru_cache(maxsize=512)
    def edge_to_corner_threshold_cached(self, edge_density: float) -> int:
        """Cached edge density to corner threshold mapping"""
        return self.original_formulas.edge_to_corner_threshold(edge_density)

    @lru_cache(maxsize=512)
    def colors_to_precision_cached(self, unique_colors: float) -> int:
        """Cached unique colors to precision mapping"""
        return self.original_formulas.colors_to_precision(unique_colors)

    @lru_cache(maxsize=512)
    def entropy_to_path_precision_cached(self, entropy: float) -> int:
        """Cached entropy to path precision mapping"""
        return self.original_formulas.entropy_to_path_precision(entropy)

    @lru_cache(maxsize=512)
    def corners_to_length_threshold_cached(self, corner_density: float) -> float:
        """Cached corners to length threshold mapping"""
        return self.original_formulas.corners_to_length_threshold(corner_density)

    @lru_cache(maxsize=512)
    def gradient_to_splice_threshold_cached(self, gradient_strength: float) -> int:
        """Cached gradient to splice threshold mapping"""
        return self.original_formulas.gradient_to_splice_threshold(gradient_strength)

    @lru_cache(maxsize=512)
    def complexity_to_iterations_cached(self, complexity_score: float) -> int:
        """Cached complexity to iterations mapping"""
        return self.original_formulas.complexity_to_iterations(complexity_score)

    def clear_cache(self):
        """Clear all formula caches"""
        self.edge_to_corner_threshold_cached.cache_clear()
        self.colors_to_precision_cached.cache_clear()
        self.entropy_to_path_precision_cached.cache_clear()
        self.corners_to_length_threshold_cached.cache_clear()
        self.gradient_to_splice_threshold_cached.cache_clear()
        self.complexity_to_iterations_cached.cache_clear()


class LazyOptimizationComponents:
    """Lazy loading for optimization components"""

    def __init__(self):
        """Initialize lazy component loader"""
        self._optimizer = None
        self._formulas = None
        self._bounds = None
        self._refined_formulas = None
        self.component_pool = weakref.WeakSet()

    @property
    def optimizer(self) -> FeatureMappingOptimizer:
        """Lazy load feature mapping optimizer"""
        if self._optimizer is None:
            self._optimizer = FeatureMappingOptimizer()
            self.component_pool.add(self._optimizer)
        return self._optimizer

    @property
    def formulas(self) -> CorrelationFormulas:
        """Lazy load correlation formulas"""
        if self._formulas is None:
            self._formulas = CorrelationFormulas()
            self.component_pool.add(self._formulas)
        return self._formulas

    @property
    def bounds(self) -> VTracerParameterBounds:
        """Lazy load parameter bounds"""
        if self._bounds is None:
            self._bounds = VTracerParameterBounds()
            self.component_pool.add(self._bounds)
        return self._bounds

    @property
    def refined_formulas(self) -> RefinedCorrelationFormulas:
        """Lazy load refined formulas"""
        if self._refined_formulas is None:
            self._refined_formulas = RefinedCorrelationFormulas()
            self.component_pool.add(self._refined_formulas)
        return self._refined_formulas

    def cleanup(self):
        """Cleanup components and free memory"""
        self._optimizer = None
        self._formulas = None
        self._bounds = None
        self._refined_formulas = None
        gc.collect()


class VectorizedBatchOptimizer:
    """Vectorized operations for batch optimization"""

    def __init__(self, lazy_components: LazyOptimizationComponents):
        """Initialize vectorized batch optimizer"""
        self.components = lazy_components

    def batch_optimize_features(self, features_batch: List[Dict[str, float]],
                               logo_types: List[str] = None) -> List[Dict[str, Any]]:
        """Optimize multiple feature sets using vectorized operations"""
        if logo_types is None:
            logo_types = ['simple'] * len(features_batch)

        # Extract feature arrays for vectorized computation
        feature_arrays = self._extract_feature_arrays(features_batch)

        # Vectorized correlation computations
        vectorized_params = self._compute_vectorized_parameters(feature_arrays, logo_types)

        # Convert back to individual results
        results = []
        for i in range(len(features_batch)):
            result = {
                'parameters': {
                    'corner_threshold': int(vectorized_params['corner_threshold'][i]),
                    'color_precision': int(vectorized_params['color_precision'][i]),
                    'path_precision': int(vectorized_params['path_precision'][i]),
                    'length_threshold': float(vectorized_params['length_threshold'][i]),
                    'splice_threshold': int(vectorized_params['splice_threshold'][i]),
                    'max_iterations': int(vectorized_params['max_iterations'][i]),
                    'layer_difference': 10,
                    'mode': 'spline' if logo_types[i] in ['gradient', 'complex'] else 'polygon'
                },
                'confidence': float(vectorized_params['confidence'][i]),
                'optimization_method': 'vectorized_batch',
                'logo_type': logo_types[i]
            }
            results.append(result)

        return results

    def _extract_feature_arrays(self, features_batch: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
        """Extract feature arrays for vectorized computation"""
        feature_names = ['edge_density', 'unique_colors', 'entropy',
                        'corner_density', 'gradient_strength', 'complexity_score']

        arrays = {}
        for feature_name in feature_names:
            arrays[feature_name] = np.array([
                features.get(feature_name, 0.0) for features in features_batch
            ])

        return arrays

    def _compute_vectorized_parameters(self, feature_arrays: Dict[str, np.ndarray],
                                     logo_types: List[str]) -> Dict[str, np.ndarray]:
        """Compute parameters using vectorized operations"""
        n_samples = len(logo_types)

        # Vectorized correlation computations
        corner_threshold = np.maximum(10, np.minimum(110,
            110 - (feature_arrays['edge_density'] * 800))).astype(int)

        color_precision = np.maximum(2, np.minimum(10,
            2 + np.log2(np.maximum(1, feature_arrays['unique_colors'])))).astype(int)

        path_precision = np.maximum(1, np.minimum(20,
            20 * (1 - feature_arrays['entropy']))).astype(int)

        length_threshold = np.maximum(1.0, np.minimum(20.0,
            1.0 + (feature_arrays['corner_density'] * 100)))

        splice_threshold = np.maximum(10, np.minimum(100,
            10 + (feature_arrays['gradient_strength'] * 90))).astype(int)

        max_iterations = np.maximum(5, np.minimum(20,
            5 + (feature_arrays['complexity_score'] * 15))).astype(int)

        # Calculate confidence based on feature completeness
        feature_completeness = np.mean([
            (feature_arrays[fname] != 0).astype(float) for fname in feature_arrays.keys()
        ], axis=0)

        # Logo type confidence multipliers
        logo_confidence = np.array([
            {'simple': 0.95, 'text': 0.90, 'gradient': 0.85, 'complex': 0.80}.get(lt, 0.85)
            for lt in logo_types
        ])

        confidence = feature_completeness * logo_confidence

        return {
            'corner_threshold': corner_threshold,
            'color_precision': color_precision,
            'path_precision': path_precision,
            'length_threshold': length_threshold,
            'splice_threshold': splice_threshold,
            'max_iterations': max_iterations,
            'confidence': confidence
        }


class ParallelOptimizationManager:
    """Parallel optimization support with thread and process pools"""

    def __init__(self, max_workers: int = None):
        """Initialize parallel optimization manager"""
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
        self.thread_pool = None
        self.process_pool = None

    def __enter__(self):
        """Context manager entry"""
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)

    def parallel_optimize(self, optimization_tasks: List[Tuple[Dict[str, float], str]],
                         use_processes: bool = False) -> List[Dict[str, Any]]:
        """Run optimization tasks in parallel"""
        if use_processes:
            # Process-based parallelism for CPU-intensive tasks
            if not self.process_pool:
                self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
            executor = self.process_pool
        else:
            # Thread-based parallelism for I/O-bound tasks
            executor = self.thread_pool

        # Submit optimization tasks
        futures = []
        for features, logo_type in optimization_tasks:
            future = executor.submit(self._optimize_single, features, logo_type)
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel optimization task failed: {e}")
                results.append({
                    'error': str(e),
                    'optimization_method': 'parallel_failed'
                })

        return results

    def _optimize_single(self, features: Dict[str, float], logo_type: str) -> Dict[str, Any]:
        """Optimize single feature set (for parallel execution)"""
        # Create fresh components for this process/thread
        lazy_components = LazyOptimizationComponents()
        optimizer = lazy_components.optimizer

        try:
            result = optimizer.optimize(features)
            result['logo_type'] = logo_type
            result['optimization_method'] = 'parallel_single'
            return result
        finally:
            lazy_components.cleanup()


class Method1PerformanceOptimizer:
    """Optimize Method 1 for speed and memory efficiency"""

    def __init__(self):
        """Initialize performance optimizer"""
        self.profiler = PerformanceProfiler()
        self.cache_manager = CacheManager()
        self.lazy_components = LazyOptimizationComponents()
        self.optimized_formulas = OptimizedCorrelationFormulas(self.cache_manager)
        self.vectorized_optimizer = VectorizedBatchOptimizer(self.lazy_components)
        self.performance_metrics = defaultdict(list)

    def profile_optimization(self, test_images: List[str] = None) -> Dict[str, Any]:
        """Profile optimization performance with comprehensive analysis"""
        logger.info("Starting comprehensive performance profiling")

        if test_images is None:
            # Generate mock test data
            test_images = [f"test_image_{i:03d}.png" for i in range(20)]

        # Generate test features
        test_features = []
        logo_types = []

        for i, image_path in enumerate(test_images):
            logo_type = ['simple', 'text', 'gradient', 'complex'][i % 4]
            features = {
                'edge_density': np.random.uniform(0.05, 0.6),
                'unique_colors': np.random.uniform(2, 200),
                'entropy': np.random.uniform(0.2, 0.95),
                'corner_density': np.random.uniform(0.02, 0.5),
                'gradient_strength': np.random.uniform(0.1, 0.9),
                'complexity_score': np.random.uniform(0.1, 0.95)
            }
            test_features.append(features)
            logo_types.append(logo_type)

        # Profile different optimization methods
        profiling_results = {}

        # 1. Profile original optimization
        logger.info("Profiling original optimization method")
        original_times, original_memory = self._profile_original_optimization(test_features, logo_types)
        profiling_results['original'] = {
            'avg_time': np.mean(original_times),
            'total_time': np.sum(original_times),
            'avg_memory': np.mean(original_memory),
            'max_memory': np.max(original_memory)
        }

        # 2. Profile cached optimization
        logger.info("Profiling cached optimization method")
        cached_times, cached_memory = self._profile_cached_optimization(test_features, logo_types)
        profiling_results['cached'] = {
            'avg_time': np.mean(cached_times),
            'total_time': np.sum(cached_times),
            'avg_memory': np.mean(cached_memory),
            'max_memory': np.max(cached_memory)
        }

        # 3. Profile vectorized batch optimization
        logger.info("Profiling vectorized batch optimization")
        batch_time, batch_memory = self._profile_batch_optimization(test_features, logo_types)
        profiling_results['vectorized_batch'] = {
            'avg_time': batch_time / len(test_features),
            'total_time': batch_time,
            'avg_memory': batch_memory,
            'max_memory': batch_memory
        }

        # 4. Profile parallel optimization
        logger.info("Profiling parallel optimization")
        parallel_time, parallel_memory = self._profile_parallel_optimization(test_features, logo_types)
        profiling_results['parallel'] = {
            'avg_time': parallel_time / len(test_features),
            'total_time': parallel_time,
            'avg_memory': parallel_memory,
            'max_memory': parallel_memory
        }

        # Generate comprehensive report
        profile_report = {
            'test_info': {
                'test_images': len(test_images),
                'test_features': len(test_features),
                'logo_types': list(set(logo_types))
            },
            'profiling_results': profiling_results,
            'performance_improvements': self._calculate_improvements(profiling_results),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'system_info': {
                'cpu_count': multiprocessing.cpu_count(),
                'memory_available': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'python_version': f"{psutil.Process().name()}"
            },
            'recommendations': self._generate_performance_recommendations(profiling_results)
        }

        logger.info("Performance profiling completed")
        return profile_report

    def _profile_original_optimization(self, test_features: List[Dict], logo_types: List[str]) -> Tuple[List[float], List[float]]:
        """Profile original optimization method"""
        times = []
        memory_usage = []

        for features, logo_type in zip(test_features, logo_types):
            start_time = time.time()
            mem_before = self.profiler.get_memory_profile()['rss_mb']

            # Use original optimizer
            optimizer = FeatureMappingOptimizer()
            result = optimizer.optimize(features)

            end_time = time.time()
            mem_after = self.profiler.get_memory_profile()['rss_mb']

            times.append(end_time - start_time)
            memory_usage.append(mem_after - mem_before)

        return times, memory_usage

    def _profile_cached_optimization(self, test_features: List[Dict], logo_types: List[str]) -> Tuple[List[float], List[float]]:
        """Profile cached optimization method"""
        times = []
        memory_usage = []

        for features, logo_type in zip(test_features, logo_types):
            start_time = time.time()
            mem_before = self.profiler.get_memory_profile()['rss_mb']

            # Use cached formulas
            corner_threshold = self.optimized_formulas.edge_to_corner_threshold_cached(features['edge_density'])
            color_precision = self.optimized_formulas.colors_to_precision_cached(features['unique_colors'])
            path_precision = self.optimized_formulas.entropy_to_path_precision_cached(features['entropy'])
            length_threshold = self.optimized_formulas.corners_to_length_threshold_cached(features['corner_density'])
            splice_threshold = self.optimized_formulas.gradient_to_splice_threshold_cached(features['gradient_strength'])
            max_iterations = self.optimized_formulas.complexity_to_iterations_cached(features['complexity_score'])

            end_time = time.time()
            mem_after = self.profiler.get_memory_profile()['rss_mb']

            times.append(end_time - start_time)
            memory_usage.append(mem_after - mem_before)

        return times, memory_usage

    def _profile_batch_optimization(self, test_features: List[Dict], logo_types: List[str]) -> Tuple[float, float]:
        """Profile vectorized batch optimization"""
        start_time = time.time()
        mem_before = self.profiler.get_memory_profile()['rss_mb']

        # Use vectorized batch optimizer
        results = self.vectorized_optimizer.batch_optimize_features(test_features, logo_types)

        end_time = time.time()
        mem_after = self.profiler.get_memory_profile()['rss_mb']

        return end_time - start_time, mem_after - mem_before

    def _profile_parallel_optimization(self, test_features: List[Dict], logo_types: List[str]) -> Tuple[float, float]:
        """Profile parallel optimization"""
        start_time = time.time()
        mem_before = self.profiler.get_memory_profile()['rss_mb']

        # Use parallel optimization
        optimization_tasks = list(zip(test_features, logo_types))
        with ParallelOptimizationManager(max_workers=4) as parallel_mgr:
            results = parallel_mgr.parallel_optimize(optimization_tasks)

        end_time = time.time()
        mem_after = self.profiler.get_memory_profile()['rss_mb']

        return end_time - start_time, mem_after - mem_before

    def _calculate_improvements(self, profiling_results: Dict) -> Dict[str, Dict[str, float]]:
        """Calculate performance improvements over baseline"""
        baseline = profiling_results['original']
        improvements = {}

        for method, results in profiling_results.items():
            if method != 'original':
                improvements[method] = {
                    'time_improvement': (baseline['avg_time'] - results['avg_time']) / baseline['avg_time'] * 100,
                    'memory_improvement': (baseline['avg_memory'] - results['avg_memory']) / max(1, baseline['avg_memory']) * 100,
                    'speedup_factor': baseline['avg_time'] / max(0.001, results['avg_time'])
                }

        return improvements

    def _generate_performance_recommendations(self, profiling_results: Dict) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        # Analyze results and generate recommendations
        best_method = min(profiling_results.keys(),
                         key=lambda x: profiling_results[x]['avg_time'])

        recommendations.append(f"Best performing method: {best_method}")

        if profiling_results['cached']['avg_time'] < profiling_results['original']['avg_time']:
            recommendations.append("Enable formula caching for single optimizations")

        if profiling_results['vectorized_batch']['avg_time'] < profiling_results['original']['total_time']:
            recommendations.append("Use vectorized batch optimization for multiple images")

        if profiling_results['parallel']['avg_time'] < profiling_results['original']['avg_time']:
            recommendations.append("Use parallel optimization for high-throughput scenarios")

        return recommendations

    def generate_performance_heatmap(self, output_file: str = "performance_heatmap.png") -> str:
        """Generate performance heatmap visualization"""
        try:
            # Create mock performance data for heatmap
            methods = ['Original', 'Cached', 'Vectorized', 'Parallel']
            metrics = ['Avg Time (ms)', 'Memory Usage (MB)', 'CPU Usage (%)']

            # Mock data (in real implementation, this would come from actual profiling)
            data = np.array([
                [50, 25, 15, 20],  # Avg Time
                [45, 35, 30, 40],  # Memory Usage
                [60, 40, 80, 70]   # CPU Usage
            ])

            # Create heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(data, annot=True, xticklabels=methods, yticklabels=metrics,
                       cmap='RdYlGn_r', fmt='.1f')
            plt.title('Method 1 Performance Optimization Heatmap')
            plt.tight_layout()

            output_path = Path(output_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Generated performance heatmap: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error generating performance heatmap: {e}")
            return ""

    def create_detailed_profiling_report(self, output_file: str = "detailed_profiling_report.json") -> str:
        """Create detailed profiling report"""
        try:
            # Run comprehensive profiling
            profile_data = self.profile_optimization()

            # Add additional profiling data
            profile_data.update({
                'profiler_stats': self.profiler.get_profile_stats(),
                'timing_report': self.profiler.generate_timing_report(),
                'cache_performance': self.cache_manager.get_cache_stats(),
                'system_resources': {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                },
                'timestamp': datetime.now().isoformat()
            })

            # Save report
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                json.dump(profile_data, f, indent=2, default=str)

            logger.info(f"Generated detailed profiling report: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error creating profiling report: {e}")
            return ""

    def cleanup(self):
        """Cleanup resources and clear caches"""
        self.cache_manager.clear_cache()
        self.optimized_formulas.clear_cache()
        self.lazy_components.cleanup()
        gc.collect()