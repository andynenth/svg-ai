# AI Modules Performance Guide

## Overview

This guide provides detailed information about optimizing performance when using AI modules in the SVG-AI Enhanced Conversion Pipeline.

## Performance Targets

### Response Time Targets

| Operation | Tier 1 (Fast) | Tier 2 (Balanced) | Tier 3 (Quality) |
|-----------|----------------|-------------------|-------------------|
| Feature Extraction | <0.5s | <2s | <5s |
| Classification | <0.1s | <0.5s | <1s |
| Parameter Optimization | <1s | <5s | <15s |
| Quality Prediction | <0.1s | <0.5s | <1s |
| Complete Pipeline | <2s | <8s | <20s |

### Memory Usage Targets

| Component | Startup | Per Image | Peak Usage |
|-----------|---------|-----------|------------|
| Feature Extractor | <10MB | <5MB | <50MB |
| Classifier | <5MB | <1MB | <20MB |
| Optimizer | <20MB | <10MB | <100MB |
| Quality Predictor | <15MB | <2MB | <50MB |
| Complete Pipeline | <50MB | <15MB | <200MB |

## Performance Monitoring

### 1. Built-in Performance Monitoring

All AI modules include built-in performance monitoring:

```python
from backend.ai_modules.utils.performance_monitor import performance_monitor

# Use decorator for automatic monitoring
@performance_monitor.time_operation("custom_operation")
def my_operation():
    # Your code here
    pass

# Get performance summary
summary = performance_monitor.get_summary("custom_operation")
print(f"Average time: {summary['average_duration']:.3f}s")
print(f"Memory delta: {summary['average_memory_delta']:.1f}MB")
```

### 2. Custom Performance Monitoring

```python
import time
import psutil
from contextlib import contextmanager

@contextmanager
def monitor_performance(operation_name: str):
    """Context manager for performance monitoring"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        duration = end_time - start_time
        memory_delta = end_memory - start_memory

        print(f"{operation_name}:")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Memory: +{memory_delta:.1f}MB")

# Usage
with monitor_performance("Feature Extraction"):
    features = extractor.extract_features(image_path)
```

### 3. Batch Performance Monitoring

```python
def monitor_batch_performance(operation_func, data_list: list, batch_name: str):
    """Monitor performance across batch operations"""
    times = []
    memory_deltas = []

    for i, data in enumerate(data_list):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        try:
            result = operation_func(data)
            success = True
        except Exception as e:
            print(f"Batch item {i} failed: {e}")
            success = False

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        if success:
            times.append(end_time - start_time)
            memory_deltas.append(end_memory - start_memory)

    # Calculate statistics
    if times:
        print(f"\n📊 {batch_name} Performance Summary:")
        print(f"  Processed: {len(times)}/{len(data_list)} items")
        print(f"  Average time: {sum(times)/len(times):.3f}s")
        print(f"  Max time: {max(times):.3f}s")
        print(f"  Average memory: +{sum(memory_deltas)/len(memory_deltas):.1f}MB")
        print(f"  Total time: {sum(times):.1f}s")
```

## Optimization Strategies

### 1. Caching Strategies

#### Feature Caching
```python
from functools import lru_cache
import hashlib

class OptimizedFeatureExtractor:
    """Feature extractor with advanced caching"""

    def __init__(self, cache_size: int = 128):
        self.cache_size = cache_size
        self.memory_cache = {}
        self.disk_cache_dir = Path("data/cache/features")
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_image_hash(self, image_path: str) -> str:
        """Get hash of image for caching"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def extract_features(self, image_path: str) -> dict:
        """Extract features with multi-level caching"""
        image_hash = self._get_image_hash(image_path)

        # Check memory cache
        if image_hash in self.memory_cache:
            return self.memory_cache[image_hash]

        # Check disk cache
        cache_file = self.disk_cache_dir / f"{image_hash}.json"
        if cache_file.exists():
            import json
            with open(cache_file, 'r') as f:
                features = json.load(f)
                self.memory_cache[image_hash] = features
                return features

        # Extract features
        features = self._extract_features_internal(image_path)

        # Cache results
        self._cache_features(image_hash, features)
        return features

    def _cache_features(self, image_hash: str, features: dict):
        """Cache features in memory and disk"""
        # Memory cache
        self.memory_cache[image_hash] = features

        # Limit memory cache size
        if len(self.memory_cache) > self.cache_size:
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]

        # Disk cache
        cache_file = self.disk_cache_dir / f"{image_hash}.json"
        import json
        with open(cache_file, 'w') as f:
            json.dump(features, f)
```

#### Model Caching
```python
class ModelCache:
    """Cache for AI models to avoid repeated loading"""

    def __init__(self):
        self._models = {}

    def get_model(self, model_name: str, model_loader_func):
        """Get model from cache or load it"""
        if model_name not in self._models:
            print(f"Loading model: {model_name}")
            self._models[model_name] = model_loader_func()
        return self._models[model_name]

    def clear_model(self, model_name: str):
        """Remove model from cache"""
        if model_name in self._models:
            del self._models[model_name]

    def clear_all(self):
        """Clear all cached models"""
        self._models.clear()

# Global model cache
model_cache = ModelCache()

# Usage
def get_quality_predictor():
    return model_cache.get_model('quality_predictor', lambda: QualityPredictor())
```

### 2. Parallel Processing

#### Thread-based Parallelism
```python
import concurrent.futures
from typing import List, Callable, Any

def parallel_process(
    items: List[Any],
    process_func: Callable,
    max_workers: int = 4,
    timeout: float = 30.0
) -> List[Any]:
    """Process items in parallel using threads"""

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(process_func, item): item
            for item in items
        }

        # Collect results
        for future in concurrent.futures.as_completed(future_to_item, timeout=timeout):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append({'item': item, 'result': result, 'success': True})
            except Exception as e:
                results.append({'item': item, 'error': str(e), 'success': False})

    return results

# Usage example
def process_images_parallel(image_paths: List[str]):
    """Process multiple images in parallel"""

    def process_single_image(image_path: str):
        extractor = ImageFeatureExtractor()
        classifier = RuleBasedClassifier()

        features = extractor.extract_features(image_path)
        logo_type, confidence = classifier.classify(features)

        return {
            'image_path': image_path,
            'features': features,
            'logo_type': logo_type,
            'confidence': confidence
        }

    results = parallel_process(image_paths, process_single_image, max_workers=4)
    successful_results = [r['result'] for r in results if r['success']]

    return successful_results
```

#### Process-based Parallelism
```python
import multiprocessing as mp
from multiprocessing import Pool

def process_image_chunk(image_paths_chunk: List[str]) -> List[dict]:
    """Process a chunk of images in a separate process"""
    # Import modules inside process to avoid pickling issues
    from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
    from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier

    extractor = ImageFeatureExtractor()
    classifier = RuleBasedClassifier()

    results = []
    for image_path in image_paths_chunk:
        try:
            features = extractor.extract_features(image_path)
            logo_type, confidence = classifier.classify(features)

            results.append({
                'image_path': image_path,
                'logo_type': logo_type,
                'confidence': confidence,
                'success': True
            })
        except Exception as e:
            results.append({
                'image_path': image_path,
                'error': str(e),
                'success': False
            })

    return results

def process_images_multiprocess(image_paths: List[str], num_processes: int = None):
    """Process images using multiple processes"""
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 4)

    # Split images into chunks
    chunk_size = max(1, len(image_paths) // num_processes)
    chunks = [
        image_paths[i:i + chunk_size]
        for i in range(0, len(image_paths), chunk_size)
    ]

    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_image_chunk, chunks)

    # Flatten results
    all_results = []
    for chunk_result in chunk_results:
        all_results.extend(chunk_result)

    return all_results
```

### 3. Memory Optimization

#### Memory-Efficient Batch Processing
```python
def memory_efficient_batch_processing(
    image_paths: List[str],
    batch_size: int = 10,
    memory_threshold: float = 85.0  # Percent
):
    """Process images in batches with memory monitoring"""

    def check_memory():
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < memory_threshold

    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]

        # Check memory before processing batch
        if not check_memory():
            print("⚠️  High memory usage, forcing garbage collection...")
            import gc
            gc.collect()
            time.sleep(1)  # Give time for cleanup

        print(f"Processing batch {i//batch_size + 1}/{len(image_paths)//batch_size + 1}")

        batch_results = []
        for image_path in batch:
            try:
                # Process single image
                result = process_single_image_minimal(image_path)
                batch_results.append(result)

                # Check memory after each image if needed
                if not check_memory():
                    print(f"Memory threshold exceeded, stopping batch early")
                    break

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                batch_results.append({'error': str(e), 'path': image_path})

        results.extend(batch_results)

        # Force cleanup between batches
        import gc
        gc.collect()

    return results

def process_single_image_minimal(image_path: str):
    """Memory-efficient single image processing"""
    # Use local imports to minimize memory footprint
    from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor

    extractor = ImageFeatureExtractor()
    try:
        features = extractor.extract_features(image_path)
        return {'features': features, 'success': True, 'path': image_path}
    finally:
        # Explicit cleanup
        del extractor
```

#### Smart Caching with Memory Limits
```python
class MemoryAwareLRUCache:
    """LRU cache that respects memory limits"""

    def __init__(self, max_memory_mb: int = 100):
        self.max_memory_mb = max_memory_mb
        self.cache = {}
        self.access_order = []

    def get(self, key: str):
        """Get item from cache"""
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any):
        """Put item in cache with memory management"""
        # Add to cache
        self.cache[key] = value
        if key not in self.access_order:
            self.access_order.append(key)

        # Check memory usage
        self._enforce_memory_limit()

    def _enforce_memory_limit(self):
        """Remove items if memory limit exceeded"""
        current_memory = self._estimate_memory_usage()

        while current_memory > self.max_memory_mb and self.access_order:
            # Remove least recently used item
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            current_memory = self._estimate_memory_usage()

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        import sys
        total_size = 0
        for value in self.cache.values():
            total_size += sys.getsizeof(value)
        return total_size / (1024 * 1024)  # Convert to MB
```

### 4. Algorithmic Optimizations

#### Progressive Feature Extraction
```python
class ProgressiveFeatureExtractor:
    """Extract features progressively based on time constraints"""

    def __init__(self):
        self.quick_features = ['unique_colors', 'aspect_ratio']
        self.medium_features = ['entropy', 'fill_ratio']
        self.slow_features = ['edge_density', 'corner_density', 'gradient_strength', 'complexity_score']

    def extract_features_progressive(self, image_path: str, time_limit: float = 1.0):
        """Extract features within time limit"""
        start_time = time.time()
        features = {}

        # Always extract quick features
        features.update(self._extract_quick_features(image_path))

        if time.time() - start_time < time_limit * 0.3:
            # Extract medium features if we have time
            features.update(self._extract_medium_features(image_path))

        if time.time() - start_time < time_limit * 0.7:
            # Extract slow features if we still have time
            features.update(self._extract_slow_features(image_path))

        # Fill in missing features with defaults
        self._fill_missing_features(features)

        return features

    def _extract_quick_features(self, image_path: str) -> dict:
        """Extract features that are fast to compute"""
        # Implement quick feature extraction
        return {'unique_colors': 16, 'aspect_ratio': 1.0}

    def _extract_medium_features(self, image_path: str) -> dict:
        """Extract features with medium computation time"""
        # Implement medium feature extraction
        return {'entropy': 6.0, 'fill_ratio': 0.4}

    def _extract_slow_features(self, image_path: str) -> dict:
        """Extract features that are slow to compute"""
        # Implement slow feature extraction
        return {
            'edge_density': 0.1,
            'corner_density': 0.02,
            'gradient_strength': 20.0,
            'complexity_score': 0.5
        }

    def _fill_missing_features(self, features: dict):
        """Fill missing features with defaults"""
        defaults = {
            'complexity_score': 0.5,
            'unique_colors': 16,
            'edge_density': 0.1,
            'aspect_ratio': 1.0,
            'fill_ratio': 0.4,
            'entropy': 6.0,
            'corner_density': 0.02,
            'gradient_strength': 20.0
        }

        for feature, default_value in defaults.items():
            if feature not in features:
                features[feature] = default_value
```

#### Adaptive Optimization
```python
class AdaptivePerformanceOptimizer:
    """Optimizer that adapts based on performance constraints"""

    def __init__(self):
        self.performance_history = []

    def optimize_with_constraints(self, features: dict, time_limit: float = 5.0):
        """Optimize parameters within time and quality constraints"""
        start_time = time.time()

        # Choose optimization strategy based on time limit
        if time_limit < 1.0:
            return self._quick_optimize(features)
        elif time_limit < 5.0:
            return self._balanced_optimize(features)
        else:
            return self._thorough_optimize(features, start_time, time_limit)

    def _quick_optimize(self, features: dict) -> dict:
        """Quick optimization using lookup tables"""
        # Use pre-computed lookup table based on features
        complexity = features.get('complexity_score', 0.5)

        if complexity < 0.3:
            return {'color_precision': 3, 'corner_threshold': 30}
        elif complexity < 0.7:
            return {'color_precision': 6, 'corner_threshold': 45}
        else:
            return {'color_precision': 10, 'corner_threshold': 60}

    def _balanced_optimize(self, features: dict) -> dict:
        """Balanced optimization using heuristics"""
        # Use feature mapping with some optimization
        from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
        optimizer = FeatureMappingOptimizer()
        return optimizer.optimize(features)

    def _thorough_optimize(self, features: dict, start_time: float, time_limit: float) -> dict:
        """Thorough optimization using multiple strategies"""
        best_params = self._balanced_optimize(features)
        best_score = self._evaluate_parameters(features, best_params)

        strategies = ['genetic', 'grid_search', 'random_search']

        for strategy in strategies:
            if time.time() - start_time > time_limit * 0.8:
                break

            try:
                params = self._optimize_with_strategy(features, strategy)
                score = self._evaluate_parameters(features, params)

                if score > best_score:
                    best_params = params
                    best_score = score

            except Exception as e:
                print(f"Strategy {strategy} failed: {e}")

        return best_params

    def _evaluate_parameters(self, features: dict, parameters: dict) -> float:
        """Evaluate parameter quality (mock implementation)"""
        # This would use actual quality prediction in real implementation
        return 0.8  # Mock score
```

## Performance Benchmarking

### Comprehensive Benchmark Suite
```python
#!/usr/bin/env python3
"""Comprehensive AI Modules Performance Benchmark"""

import time
import psutil
import statistics
from typing import List, Dict, Any

class AIPerformanceBenchmark:
    """Comprehensive benchmark suite for AI modules"""

    def __init__(self):
        self.results = {}

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks"""
        print("🚀 Running AI Modules Performance Benchmark")
        print("=" * 50)

        benchmarks = [
            ('Feature Extraction', self.benchmark_feature_extraction),
            ('Classification', self.benchmark_classification),
            ('Optimization', self.benchmark_optimization),
            ('Quality Prediction', self.benchmark_quality_prediction),
            ('Complete Pipeline', self.benchmark_complete_pipeline),
            ('Memory Usage', self.benchmark_memory_usage),
            ('Concurrent Processing', self.benchmark_concurrent_processing)
        ]

        for name, benchmark_func in benchmarks:
            print(f"\n📊 {name}:")
            try:
                result = benchmark_func()
                self.results[name] = result
                self._print_benchmark_result(result)
            except Exception as e:
                print(f"  ❌ Benchmark failed: {e}")
                self.results[name] = {'error': str(e)}

        self._print_summary()
        return self.results

    def benchmark_feature_extraction(self) -> Dict[str, Any]:
        """Benchmark feature extraction performance"""
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor

        extractor = ImageFeatureExtractor()
        test_images = self._get_test_images()

        times = []
        for image_path in test_images:
            start = time.time()
            features = extractor.extract_features(image_path)
            end = time.time()
            times.append(end - start)

        return {
            'average_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'max_time': max(times),
            'min_time': min(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
            'samples': len(times)
        }

    def benchmark_complete_pipeline(self) -> Dict[str, Any]:
        """Benchmark complete AI pipeline"""
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
        from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
        from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

        extractor = ImageFeatureExtractor()
        classifier = RuleBasedClassifier()
        optimizer = FeatureMappingOptimizer()

        test_images = self._get_test_images()
        times = []

        for image_path in test_images:
            start = time.time()

            # Complete pipeline
            features = extractor.extract_features(image_path)
            logo_type, confidence = classifier.classify(features)
            parameters = optimizer.optimize(features)

            end = time.time()
            times.append(end - start)

        return {
            'average_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'max_time': max(times),
            'throughput': len(test_images) / sum(times),  # images per second
            'samples': len(times)
        }

    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage"""
        import gc

        # Clear memory first
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        # Load AI components
        from backend.ai_modules.classification.feature_extractor import ImageFeatureExtractor
        from backend.ai_modules.classification.rule_based_classifier import RuleBasedClassifier
        from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

        extractor = ImageFeatureExtractor()
        classifier = RuleBasedClassifier()
        optimizer = FeatureMappingOptimizer()

        loaded_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        # Process test images
        test_images = self._get_test_images()
        for image_path in test_images:
            features = extractor.extract_features(image_path)
            logo_type, confidence = classifier.classify(features)
            parameters = optimizer.optimize(features)

        peak_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        return {
            'baseline_memory_mb': baseline_memory,
            'loaded_memory_mb': loaded_memory,
            'peak_memory_mb': peak_memory,
            'loading_overhead_mb': loaded_memory - baseline_memory,
            'processing_overhead_mb': peak_memory - loaded_memory,
            'total_overhead_mb': peak_memory - baseline_memory
        }

    def _get_test_images(self) -> List[str]:
        """Get list of test images"""
        # Return test images or create them if needed
        test_images = [
            "tests/data/simple/simple_logo_0.png",
            "tests/data/text/text_logo_0.png",
            "tests/data/gradient/gradient_logo_0.png"
        ]

        # Filter to existing images
        import os
        return [img for img in test_images if os.path.exists(img)]

    def _print_benchmark_result(self, result: Dict[str, Any]):
        """Print benchmark result"""
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
            return

        if 'average_time' in result:
            print(f"  ⏱️  Average: {result['average_time']:.3f}s")
            print(f"  📊 Median: {result['median_time']:.3f}s")
            print(f"  🔺 Max: {result['max_time']:.3f}s")

        if 'throughput' in result:
            print(f"  🚀 Throughput: {result['throughput']:.1f} images/sec")

        if 'total_overhead_mb' in result:
            print(f"  💾 Memory Overhead: {result['total_overhead_mb']:.1f}MB")
            print(f"  📈 Peak Memory: {result['peak_memory_mb']:.1f}MB")

    def _print_summary(self):
        """Print benchmark summary"""
        print("\n📋 Performance Summary:")
        print("=" * 30)

        # Overall performance grade
        pipeline_result = self.results.get('Complete Pipeline', {})
        if 'average_time' in pipeline_result:
            avg_time = pipeline_result['average_time']
            if avg_time < 2.0:
                grade = "🏆 Excellent"
            elif avg_time < 5.0:
                grade = "✅ Good"
            elif avg_time < 10.0:
                grade = "⚠️  Fair"
            else:
                grade = "❌ Poor"

            print(f"Overall Performance: {grade}")
            print(f"Pipeline Time: {avg_time:.2f}s per image")

        # Memory efficiency
        memory_result = self.results.get('Memory Usage', {})
        if 'total_overhead_mb' in memory_result:
            overhead = memory_result['total_overhead_mb']
            if overhead < 50:
                memory_grade = "🏆 Excellent"
            elif overhead < 100:
                memory_grade = "✅ Good"
            elif overhead < 200:
                memory_grade = "⚠️  Fair"
            else:
                memory_grade = "❌ Poor"

            print(f"Memory Efficiency: {memory_grade}")
            print(f"Memory Overhead: {overhead:.1f}MB")

if __name__ == "__main__":
    benchmark = AIPerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
```

This performance guide provides comprehensive strategies for optimizing AI module performance across all dimensions: speed, memory usage, and resource efficiency.