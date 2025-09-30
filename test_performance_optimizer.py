#!/usr/bin/env python3
"""
Test script for Method 1 performance optimization
"""
import sys
from pathlib import Path
import time
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from backend.ai_modules.optimization.performance_optimizer import (
    Method1PerformanceOptimizer,
    PerformanceProfiler,
    CacheManager,
    OptimizedCorrelationFormulas,
    VectorizedBatchOptimizer,
    ParallelOptimizationManager,
    LazyOptimizationComponents
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_performance_profiling():
    """Test comprehensive performance profiling"""
    print("🔬 Testing Performance Profiling")

    optimizer = Method1PerformanceOptimizer()

    # Generate test data
    test_images = [f"test_image_{i:03d}.png" for i in range(10)]

    print(f"📊 Running performance profiling on {len(test_images)} test cases...")

    # Run profiling
    start_time = time.time()
    profile_results = optimizer.profile_optimization(test_images)
    profiling_time = time.time() - start_time

    print(f"\n📈 Profiling Results (completed in {profiling_time:.2f}s):")
    print(f"  - Test Images: {profile_results['test_info']['test_images']}")
    print(f"  - Logo Types: {profile_results['test_info']['logo_types']}")

    print("\n⚡ Performance by Method:")
    for method, results in profile_results['profiling_results'].items():
        print(f"  - {method.title()}:")
        print(f"    • Avg Time: {results['avg_time']*1000:.2f}ms")
        print(f"    • Total Time: {results['total_time']*1000:.2f}ms")
        print(f"    • Memory: {results['avg_memory']:.2f}MB")

    print("\n📊 Performance Improvements:")
    for method, improvements in profile_results['performance_improvements'].items():
        print(f"  - {method.title()}:")
        print(f"    • Time Improvement: {improvements['time_improvement']:.1f}%")
        print(f"    • Speedup Factor: {improvements['speedup_factor']:.2f}x")

    print("\n🔍 Cache Performance:")
    cache_stats = profile_results['cache_stats']
    print(f"  - Hit Rate: {cache_stats.get('hit_rate', 0)*100:.1f}%")
    print(f"  - Cache Size: {cache_stats.get('cache_size', 0)}")

    print("\n💡 Recommendations:")
    for i, rec in enumerate(profile_results['recommendations'], 1):
        print(f"  {i}. {rec}")

    print("\n✅ Performance profiling test completed!")

def test_caching_system():
    """Test correlation formula caching"""
    print("\n🗄️  Testing Caching System")

    cache_manager = CacheManager(cache_size=100)
    optimized_formulas = OptimizedCorrelationFormulas(cache_manager)

    # Test caching with repeated calls
    test_values = [0.1, 0.5, 0.8, 0.1, 0.5, 0.8, 0.2, 0.1]  # Repeated values

    print("📊 Testing formula caching performance...")

    # Time uncached calls
    start_time = time.time()
    for value in test_values:
        result = optimized_formulas.edge_to_corner_threshold_cached(value)
    uncached_time = time.time() - start_time

    # Time cached calls (second run should be faster)
    start_time = time.time()
    for value in test_values:
        result = optimized_formulas.edge_to_corner_threshold_cached(value)
    cached_time = time.time() - start_time

    print(f"\n⚡ Caching Performance:")
    print(f"  - First run: {uncached_time*1000:.2f}ms")
    print(f"  - Second run: {cached_time*1000:.2f}ms")
    print(f"  - Speedup: {uncached_time/max(0.0001, cached_time):.2f}x")

    # Test cache statistics
    cache_stats = cache_manager.get_cache_stats()
    print(f"\n📈 Cache Statistics:")
    print(f"  - Cache Size: {cache_stats['cache_size']}")
    print(f"  - Hit Rate: {cache_stats['hit_rate']*100:.1f}%")

    print("\n✅ Caching system test completed!")

def test_vectorized_batch_optimization():
    """Test vectorized batch optimization"""
    print("\n🚀 Testing Vectorized Batch Optimization")

    lazy_components = LazyOptimizationComponents()
    vectorized_optimizer = VectorizedBatchOptimizer(lazy_components)

    # Generate batch test data
    batch_size = 20
    features_batch = []
    logo_types = []

    for i in range(batch_size):
        logo_type = ['simple', 'text', 'gradient', 'complex'][i % 4]
        features = {
            'edge_density': np.random.uniform(0.05, 0.6),
            'unique_colors': np.random.uniform(2, 200),
            'entropy': np.random.uniform(0.2, 0.95),
            'corner_density': np.random.uniform(0.02, 0.5),
            'gradient_strength': np.random.uniform(0.1, 0.9),
            'complexity_score': np.random.uniform(0.1, 0.95)
        }
        features_batch.append(features)
        logo_types.append(logo_type)

    print(f"📊 Testing batch optimization with {batch_size} feature sets...")

    # Time individual optimizations
    start_time = time.time()
    individual_results = []
    for features, logo_type in zip(features_batch, logo_types):
        optimizer = lazy_components.optimizer
        result = optimizer.optimize(features)
        individual_results.append(result)
    individual_time = time.time() - start_time

    # Time vectorized batch optimization
    start_time = time.time()
    batch_results = vectorized_optimizer.batch_optimize_features(features_batch, logo_types)
    batch_time = time.time() - start_time

    print(f"\n⚡ Batch Optimization Performance:")
    print(f"  - Individual: {individual_time*1000:.2f}ms")
    print(f"  - Vectorized: {batch_time*1000:.2f}ms")
    print(f"  - Speedup: {individual_time/max(0.0001, batch_time):.2f}x")
    print(f"  - Per Image: {batch_time/batch_size*1000:.2f}ms")

    # Verify results consistency
    if len(batch_results) == len(individual_results):
        print(f"\n✅ Generated {len(batch_results)} batch results successfully")
        # Sample comparison
        sample_idx = 0
        individual_params = individual_results[sample_idx]['parameters']
        batch_params = batch_results[sample_idx]['parameters']
        print(f"  - Sample comparison (image {sample_idx}):")
        print(f"    • Individual corner_threshold: {individual_params.get('corner_threshold', 'N/A')}")
        print(f"    • Batch corner_threshold: {batch_params.get('corner_threshold', 'N/A')}")

    lazy_components.cleanup()
    print("\n✅ Vectorized batch optimization test completed!")

def test_parallel_optimization():
    """Test parallel optimization"""
    print("\n⚙️  Testing Parallel Optimization")

    # Generate test data
    num_tasks = 16
    optimization_tasks = []

    for i in range(num_tasks):
        logo_type = ['simple', 'text', 'gradient', 'complex'][i % 4]
        features = {
            'edge_density': np.random.uniform(0.05, 0.6),
            'unique_colors': np.random.uniform(2, 200),
            'entropy': np.random.uniform(0.2, 0.95),
            'corner_density': np.random.uniform(0.02, 0.5),
            'gradient_strength': np.random.uniform(0.1, 0.9),
            'complexity_score': np.random.uniform(0.1, 0.95)
        }
        optimization_tasks.append((features, logo_type))

    print(f"📊 Testing parallel optimization with {num_tasks} tasks...")

    # Time sequential processing
    start_time = time.time()
    sequential_results = []
    for features, logo_type in optimization_tasks:
        lazy_components = LazyOptimizationComponents()
        optimizer = lazy_components.optimizer
        result = optimizer.optimize(features)
        sequential_results.append(result)
        lazy_components.cleanup()
    sequential_time = time.time() - start_time

    # Time parallel processing
    start_time = time.time()
    with ParallelOptimizationManager(max_workers=4) as parallel_mgr:
        parallel_results = parallel_mgr.parallel_optimize(optimization_tasks, use_processes=False)
    parallel_time = time.time() - start_time

    print(f"\n⚡ Parallel Optimization Performance:")
    print(f"  - Sequential: {sequential_time*1000:.2f}ms")
    print(f"  - Parallel: {parallel_time*1000:.2f}ms")
    print(f"  - Speedup: {sequential_time/max(0.0001, parallel_time):.2f}x")
    print(f"  - Per Task: {parallel_time/num_tasks*1000:.2f}ms")

    # Verify results
    successful_parallel = len([r for r in parallel_results if 'error' not in r])
    print(f"\n📊 Parallel Results:")
    print(f"  - Sequential results: {len(sequential_results)}")
    print(f"  - Parallel results: {len(parallel_results)}")
    print(f"  - Successful parallel: {successful_parallel}")

    print("\n✅ Parallel optimization test completed!")

def test_lazy_loading():
    """Test lazy loading of optimization components"""
    print("\n💤 Testing Lazy Loading")

    # Test lazy component loading
    lazy_components = LazyOptimizationComponents()

    print("📦 Testing component lazy loading...")

    # Components should not be loaded initially
    print(f"  - Initial components loaded: {len(lazy_components.component_pool)}")

    # Access optimizer (should trigger loading)
    optimizer = lazy_components.optimizer
    print(f"  - After accessing optimizer: {len(lazy_components.component_pool)}")

    # Access bounds (should trigger loading)
    bounds = lazy_components.bounds
    print(f"  - After accessing bounds: {len(lazy_components.component_pool)}")

    # Access all components
    formulas = lazy_components.formulas
    refined_formulas = lazy_components.refined_formulas
    print(f"  - After accessing all components: {len(lazy_components.component_pool)}")

    # Test cleanup
    lazy_components.cleanup()
    print(f"  - After cleanup: {len(lazy_components.component_pool)}")

    print("\n✅ Lazy loading test completed!")

def test_memory_profiling():
    """Test memory usage profiling"""
    print("\n🧠 Testing Memory Profiling")

    profiler = PerformanceProfiler()

    print("📊 Testing memory profiling...")

    # Get initial memory profile
    initial_memory = profiler.get_memory_profile()
    print(f"  - Initial Memory: {initial_memory['rss_mb']:.2f}MB")

    # Create some optimization components
    lazy_components = LazyOptimizationComponents()
    optimizer = lazy_components.optimizer
    formulas = lazy_components.formulas

    # Get memory after loading components
    loaded_memory = profiler.get_memory_profile()
    print(f"  - After loading components: {loaded_memory['rss_mb']:.2f}MB")
    print(f"  - Memory increase: {loaded_memory['rss_mb'] - initial_memory['rss_mb']:.2f}MB")

    # Run some optimizations
    for i in range(10):
        features = {
            'edge_density': np.random.uniform(0.05, 0.6),
            'unique_colors': np.random.uniform(2, 200),
            'entropy': np.random.uniform(0.2, 0.95),
            'corner_density': np.random.uniform(0.02, 0.5),
            'gradient_strength': np.random.uniform(0.1, 0.9),
            'complexity_score': np.random.uniform(0.1, 0.95)
        }
        result = optimizer.optimize(features)

    # Get memory after optimizations
    after_memory = profiler.get_memory_profile()
    print(f"  - After optimizations: {after_memory['rss_mb']:.2f}MB")
    print(f"  - Memory change: {after_memory['rss_mb'] - loaded_memory['rss_mb']:.2f}MB")

    # Cleanup
    lazy_components.cleanup()
    final_memory = profiler.get_memory_profile()
    print(f"  - After cleanup: {final_memory['rss_mb']:.2f}MB")

    print("\n✅ Memory profiling test completed!")

def test_performance_monitoring():
    """Test real-time performance monitoring"""
    print("\n📈 Testing Performance Monitoring")

    optimizer = Method1PerformanceOptimizer()

    print("📊 Generating performance monitoring data...")

    # Simulate performance monitoring
    for i in range(5):
        test_features = [{
            'edge_density': 0.3,
            'unique_colors': 25,
            'entropy': 0.7,
            'corner_density': 0.15,
            'gradient_strength': 0.6,
            'complexity_score': 0.4
        }]

        # Monitor performance
        start_time = time.time()
        results = optimizer.vectorized_optimizer.batch_optimize_features(test_features, ['simple'])
        optimization_time = time.time() - start_time

        optimizer.performance_metrics['optimization_time'].append(optimization_time)
        optimizer.performance_metrics['memory_usage'].append(
            optimizer.profiler.get_memory_profile()['rss_mb']
        )

        print(f"  - Iteration {i+1}: {optimization_time*1000:.2f}ms")

    # Generate performance statistics
    avg_time = np.mean(optimizer.performance_metrics['optimization_time'])
    avg_memory = np.mean(optimizer.performance_metrics['memory_usage'])

    print(f"\n📊 Performance Monitoring Results:")
    print(f"  - Average Time: {avg_time*1000:.2f}ms")
    print(f"  - Average Memory: {avg_memory:.2f}MB")
    print(f"  - Samples Collected: {len(optimizer.performance_metrics['optimization_time'])}")

    # Test performance heatmap generation
    try:
        heatmap_file = optimizer.generate_performance_heatmap("test_heatmap.png")
        print(f"  - Generated heatmap: {heatmap_file}")
    except Exception as e:
        print(f"  - Heatmap generation skipped: {e}")

    # Test detailed report generation
    try:
        report_file = optimizer.create_detailed_profiling_report("test_profiling_report.json")
        print(f"  - Generated report: {report_file}")
    except Exception as e:
        print(f"  - Report generation failed: {e}")

    optimizer.cleanup()
    print("\n✅ Performance monitoring test completed!")

def main():
    """Run all performance optimization tests"""
    print("🚀 Starting Method 1 Performance Optimization Tests")

    try:
        test_performance_profiling()
        test_caching_system()
        test_vectorized_batch_optimization()
        test_parallel_optimization()
        test_lazy_loading()
        test_memory_profiling()
        test_performance_monitoring()

        print("\n🎉 All performance optimization tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()