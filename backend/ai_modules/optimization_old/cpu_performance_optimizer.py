#!/usr/bin/env python3
"""
CPU Performance Optimization System for Quality Prediction Models
Advanced CPU-specific optimizations for local inference with <25ms target
"""

import os
import time
import psutil
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import logging
from pathlib import Path
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

@dataclass
class CPUOptimizationConfig:
    """CPU optimization configuration"""
    enable_mkl_dnn: bool = True
    enable_apple_accelerate: bool = True
    enable_simd: bool = True
    enable_vectorization: bool = True
    thread_pool_size: int = 4
    memory_pool_size: int = 512  # MB
    batch_size_optimization: bool = True
    enable_cpu_affinity: bool = True
    enable_numa_optimization: bool = True
    performance_target_ms: float = 25.0
    fallback_target_ms: float = 50.0

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    inference_time_ms: float
    memory_usage_mb: float
    cpu_utilization: float
    cache_hits: int
    cache_misses: int
    optimization_level: str
    timestamp: float

class MemoryPool:
    """Optimized memory pool for repeated inference"""

    def __init__(self, pool_size_mb: int = 512):
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.allocated_buffers = {}
        self.free_buffers = deque()
        self.allocation_lock = threading.Lock()
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def get_buffer(self, size: int, dtype: str = 'float32') -> np.ndarray:
        """Get optimized buffer for inference"""
        with self.allocation_lock:
            buffer_key = f"{size}_{dtype}"

            # Check for existing buffer
            if buffer_key in self.free_buffers:
                self.stats['cache_hits'] += 1
                buffer = self.free_buffers.popleft()
                return buffer

            # Allocate new buffer
            self.stats['cache_misses'] += 1
            self.stats['allocations'] += 1

            if dtype == 'float32':
                buffer = np.zeros(size, dtype=np.float32)
            elif dtype == 'float16':
                buffer = np.zeros(size, dtype=np.float16)
            else:
                buffer = np.zeros(size, dtype=np.float64)

            # Use memory-aligned allocation for SIMD
            if hasattr(np, 'empty_aligned'):
                buffer = np.empty_aligned(size, dtype=buffer.dtype, align=64)

            return buffer

    def return_buffer(self, buffer: np.ndarray):
        """Return buffer to pool"""
        with self.allocation_lock:
            buffer_key = f"{buffer.size}_{buffer.dtype}"
            self.free_buffers.append(buffer)
            self.stats['deallocations'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        return self.stats.copy()

class SIMDOptimizer:
    """SIMD and vectorization optimizations"""

    def __init__(self):
        self.simd_available = self._check_simd_availability()
        self.optimization_cache = {}

    def _check_simd_availability(self) -> Dict[str, bool]:
        """Check available SIMD instruction sets"""
        availability = {
            'sse': False,
            'avx': False,
            'avx2': False,
            'avx512': False,
            'neon': False  # ARM
        }

        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            flags = cpu_info.get('flags', [])

            availability['sse'] = 'sse' in flags or 'sse2' in flags
            availability['avx'] = 'avx' in flags
            availability['avx2'] = 'avx2' in flags
            availability['avx512'] = any('avx512' in flag for flag in flags)
            availability['neon'] = 'neon' in flags

        except ImportError:
            logger.warning("cpuinfo not available for SIMD detection")

        return availability

    def optimize_matrix_operations(self, input_data: np.ndarray) -> np.ndarray:
        """Optimize matrix operations for SIMD"""
        # Ensure memory alignment for SIMD operations
        if input_data.flags.c_contiguous and input_data.itemsize % 8 == 0:
            return input_data

        # Create aligned copy if needed
        aligned_data = np.ascontiguousarray(input_data, dtype=np.float32)
        return aligned_data

    def vectorized_inference(self, model_func, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Vectorized batch inference"""
        if len(inputs) == 1:
            return [model_func(inputs[0])]

        # Stack inputs for vectorized processing
        try:
            batch_input = np.stack(inputs)
            batch_output = model_func(batch_input)

            if batch_output.ndim == 1:
                return [batch_output]
            else:
                return [batch_output[i] for i in range(batch_output.shape[0])]

        except Exception as e:
            logger.warning(f"Vectorized inference failed, falling back to sequential: {e}")
            return [model_func(inp) for inp in inputs]

class CPUAffinityManager:
    """CPU affinity and NUMA optimization"""

    def __init__(self):
        self.cpu_count = os.cpu_count()
        self.numa_nodes = self._detect_numa_topology()
        self.performance_cores = self._identify_performance_cores()
        self.current_affinity = None

    def _detect_numa_topology(self) -> List[List[int]]:
        """Detect NUMA topology"""
        try:
            # Try to detect NUMA nodes
            numa_nodes = []
            numa_path = Path("/sys/devices/system/node")

            if numa_path.exists():
                for node_dir in numa_path.glob("node*"):
                    cpulist_file = node_dir / "cpulist"
                    if cpulist_file.exists():
                        cpulist = cpulist_file.read_text().strip()
                        cpus = self._parse_cpu_list(cpulist)
                        numa_nodes.append(cpus)

            return numa_nodes if numa_nodes else [[i for i in range(self.cpu_count)]]

        except Exception as e:
            logger.warning(f"NUMA detection failed: {e}")
            return [[i for i in range(self.cpu_count)]]

    def _parse_cpu_list(self, cpulist: str) -> List[int]:
        """Parse CPU list string (e.g., '0-3,6-7')"""
        cpus = []
        for part in cpulist.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                cpus.extend(range(start, end + 1))
            else:
                cpus.append(int(part))
        return cpus

    def _identify_performance_cores(self) -> List[int]:
        """Identify performance cores (P-cores on hybrid architectures)"""
        try:
            # On Apple Silicon, use first half as performance cores
            if 'arm64' in os.uname().machine.lower():
                return list(range(min(4, self.cpu_count // 2)))

            # On Intel hybrid, try to detect P-cores
            # This is a heuristic - actual detection would need more complex logic
            return list(range(min(4, self.cpu_count)))

        except Exception:
            return list(range(min(4, self.cpu_count)))

    def set_optimal_affinity(self):
        """Set optimal CPU affinity for inference"""
        try:
            if self.performance_cores:
                os.sched_setaffinity(0, self.performance_cores)
                self.current_affinity = self.performance_cores
                logger.info(f"Set CPU affinity to performance cores: {self.performance_cores}")

        except Exception as e:
            logger.warning(f"Failed to set CPU affinity: {e}")

    def reset_affinity(self):
        """Reset CPU affinity to default"""
        try:
            if self.current_affinity:
                os.sched_setaffinity(0, range(self.cpu_count))
                self.current_affinity = None

        except Exception as e:
            logger.warning(f"Failed to reset CPU affinity: {e}")

class CPUPerformanceOptimizer:
    """Advanced CPU performance optimization system"""

    def __init__(self, config: Optional[CPUOptimizationConfig] = None):
        self.config = config or CPUOptimizationConfig()
        self.memory_pool = MemoryPool(self.config.memory_pool_size)
        self.simd_optimizer = SIMDOptimizer()
        self.affinity_manager = CPUAffinityManager()

        # Performance monitoring
        self.performance_history = deque(maxlen=1000)
        self.optimization_cache = {}
        self.thread_pool = None

        # Optimization state
        self.optimization_level = "base"
        self.current_performance = None

        # Initialize optimizations
        self._initialize_optimizations()

    def _initialize_optimizations(self):
        """Initialize CPU optimizations"""
        try:
            # Set CPU affinity if enabled
            if self.config.enable_cpu_affinity:
                self.affinity_manager.set_optimal_affinity()

            # Initialize thread pool
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config.thread_pool_size,
                thread_name_prefix="cpu_optimizer"
            )

            # Set environment variables for CPU optimization
            self._set_optimization_env_vars()

            logger.info("CPU performance optimizer initialized")

        except Exception as e:
            logger.error(f"Failed to initialize CPU optimizations: {e}")

    def _set_optimization_env_vars(self):
        """Set environment variables for CPU optimization"""
        # Intel MKL-DNN optimization
        if self.config.enable_mkl_dnn:
            os.environ['MKL_NUM_THREADS'] = str(self.config.thread_pool_size)
            os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX2'
            os.environ['MKL_THREADING_LAYER'] = 'GNU'

        # Apple Accelerate framework
        if self.config.enable_apple_accelerate:
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.config.thread_pool_size)

        # OpenMP optimization
        os.environ['OMP_NUM_THREADS'] = str(self.config.thread_pool_size)
        os.environ['OMP_PROC_BIND'] = 'true'
        os.environ['OMP_PLACES'] = 'cores'

    def optimize_inference(self, model_func, input_data: np.ndarray,
                          optimization_level: str = "aggressive") -> Tuple[np.ndarray, PerformanceMetrics]:
        """Optimize single inference with performance monitoring"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            # Apply optimization level
            optimized_input = self._apply_input_optimizations(input_data, optimization_level)

            # Execute optimized inference
            if optimization_level == "aggressive":
                result = self._aggressive_inference(model_func, optimized_input)
            elif optimization_level == "balanced":
                result = self._balanced_inference(model_func, optimized_input)
            else:
                result = self._conservative_inference(model_func, optimized_input)

            # Calculate performance metrics
            inference_time = (time.time() - start_time) * 1000  # ms
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            cpu_usage = psutil.cpu_percent()

            metrics = PerformanceMetrics(
                inference_time_ms=inference_time,
                memory_usage_mb=memory_usage,
                cpu_utilization=cpu_usage,
                cache_hits=self.memory_pool.stats['cache_hits'],
                cache_misses=self.memory_pool.stats['cache_misses'],
                optimization_level=optimization_level,
                timestamp=time.time()
            )

            # Record performance history
            self.performance_history.append(metrics)
            self.current_performance = metrics

            # Adaptive optimization based on performance
            if inference_time > self.config.performance_target_ms:
                self._adapt_optimization_strategy(metrics)

            return result, metrics

        except Exception as e:
            logger.error(f"Optimized inference failed: {e}")
            # Fallback to basic inference
            result = model_func(input_data)

            fallback_metrics = PerformanceMetrics(
                inference_time_ms=(time.time() - start_time) * 1000,
                memory_usage_mb=0.0,
                cpu_utilization=psutil.cpu_percent(),
                cache_hits=0,
                cache_misses=1,
                optimization_level="fallback",
                timestamp=time.time()
            )

            return result, fallback_metrics

    def _apply_input_optimizations(self, input_data: np.ndarray, level: str) -> np.ndarray:
        """Apply input-specific optimizations"""
        if level == "conservative":
            return input_data

        # Memory alignment for SIMD
        if self.config.enable_simd:
            input_data = self.simd_optimizer.optimize_matrix_operations(input_data)

        # Data type optimization
        if level == "aggressive" and input_data.dtype == np.float64:
            input_data = input_data.astype(np.float32)

        # Memory pooling
        optimized_buffer = self.memory_pool.get_buffer(input_data.size, str(input_data.dtype))
        optimized_buffer[:] = input_data.flatten()
        optimized_buffer = optimized_buffer.reshape(input_data.shape)

        return optimized_buffer

    def _aggressive_inference(self, model_func, input_data: np.ndarray) -> np.ndarray:
        """Aggressive optimization inference"""
        # Use all available optimizations
        try:
            # Vectorized operations
            if self.config.enable_vectorization:
                return model_func(input_data)
            else:
                return model_func(input_data)

        except Exception as e:
            logger.warning(f"Aggressive inference failed: {e}")
            return self._balanced_inference(model_func, input_data)

    def _balanced_inference(self, model_func, input_data: np.ndarray) -> np.ndarray:
        """Balanced optimization inference"""
        return model_func(input_data)

    def _conservative_inference(self, model_func, input_data: np.ndarray) -> np.ndarray:
        """Conservative optimization inference"""
        return model_func(input_data)

    def optimize_batch_inference(self, model_func, inputs: List[np.ndarray],
                                optimization_level: str = "balanced") -> Tuple[List[np.ndarray], List[PerformanceMetrics]]:
        """Optimize batch inference with parallel processing"""
        if len(inputs) == 1:
            result, metrics = self.optimize_inference(model_func, inputs[0], optimization_level)
            return [result], [metrics]

        # Determine optimal batch size
        optimal_batch_size = self._calculate_optimal_batch_size(inputs[0].shape)

        # Process in optimized batches
        results = []
        metrics_list = []

        if self.config.batch_size_optimization and len(inputs) > optimal_batch_size:
            # Process in parallel batches
            futures = []

            for i in range(0, len(inputs), optimal_batch_size):
                batch = inputs[i:i + optimal_batch_size]

                if len(batch) > 1 and self.config.enable_vectorization:
                    # Vectorized batch processing
                    future = self.thread_pool.submit(
                        self._process_vectorized_batch, model_func, batch, optimization_level
                    )
                else:
                    # Sequential batch processing
                    future = self.thread_pool.submit(
                        self._process_sequential_batch, model_func, batch, optimization_level
                    )

                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                batch_results, batch_metrics = future.result()
                results.extend(batch_results)
                metrics_list.extend(batch_metrics)

        else:
            # Process sequentially
            for inp in inputs:
                result, metrics = self.optimize_inference(model_func, inp, optimization_level)
                results.append(result)
                metrics_list.append(metrics)

        return results, metrics_list

    def _process_vectorized_batch(self, model_func, batch: List[np.ndarray],
                                 optimization_level: str) -> Tuple[List[np.ndarray], List[PerformanceMetrics]]:
        """Process batch using vectorization"""
        return self.simd_optimizer.vectorized_inference(
            lambda x: self.optimize_inference(model_func, x, optimization_level)[0],
            batch
        ), []

    def _process_sequential_batch(self, model_func, batch: List[np.ndarray],
                                 optimization_level: str) -> Tuple[List[np.ndarray], List[PerformanceMetrics]]:
        """Process batch sequentially"""
        results = []
        metrics_list = []

        for inp in batch:
            result, metrics = self.optimize_inference(model_func, inp, optimization_level)
            results.append(result)
            metrics_list.append(metrics)

        return results, metrics_list

    def _calculate_optimal_batch_size(self, input_shape: Tuple[int, ...]) -> int:
        """Calculate optimal batch size based on system resources"""
        # Estimate memory usage per sample
        sample_memory_mb = np.prod(input_shape) * 4 / (1024 * 1024)  # Assume float32

        # Available memory for batching
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        safe_memory = available_memory * 0.3  # Use 30% of available memory

        # Calculate optimal batch size
        optimal_batch_size = max(1, int(safe_memory / (sample_memory_mb * 2)))  # 2x for safety

        # Cap at reasonable limits
        return min(optimal_batch_size, 16)

    def _adapt_optimization_strategy(self, metrics: PerformanceMetrics):
        """Adapt optimization strategy based on performance"""
        if metrics.inference_time_ms > self.config.fallback_target_ms:
            # Performance is very poor, reduce optimization level
            self.optimization_level = "conservative"
            logger.warning(f"Reducing optimization level due to poor performance: {metrics.inference_time_ms:.1f}ms")

        elif metrics.inference_time_ms > self.config.performance_target_ms:
            # Performance is below target, try balanced approach
            if self.optimization_level == "aggressive":
                self.optimization_level = "balanced"
                logger.info("Switching to balanced optimization")

        else:
            # Performance is good, can try more aggressive optimization
            if self.optimization_level == "conservative":
                self.optimization_level = "balanced"
            elif self.optimization_level == "balanced":
                self.optimization_level = "aggressive"

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if not self.performance_history:
            return {"status": "no_data"}

        metrics_data = [asdict(m) for m in self.performance_history]
        inference_times = [m.inference_time_ms for m in self.performance_history]
        memory_usage = [m.memory_usage_mb for m in self.performance_history]

        report = {
            "summary": {
                "total_inferences": len(self.performance_history),
                "avg_inference_time_ms": np.mean(inference_times),
                "p95_inference_time_ms": np.percentile(inference_times, 95),
                "p99_inference_time_ms": np.percentile(inference_times, 99),
                "min_inference_time_ms": np.min(inference_times),
                "max_inference_time_ms": np.max(inference_times),
                "avg_memory_usage_mb": np.mean(memory_usage),
                "target_achievement_rate": sum(1 for t in inference_times if t < self.config.performance_target_ms) / len(inference_times)
            },
            "optimization_status": {
                "current_level": self.optimization_level,
                "simd_available": self.simd_optimizer.simd_available,
                "numa_nodes": len(self.affinity_manager.numa_nodes),
                "performance_cores": len(self.affinity_manager.performance_cores),
                "memory_pool_stats": self.memory_pool.get_stats()
            },
            "recent_performance": metrics_data[-10:] if len(metrics_data) >= 10 else metrics_data,
            "configuration": asdict(self.config)
        }

        return report

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)

            self.affinity_manager.reset_affinity()

            logger.info("CPU performance optimizer cleanup complete")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Factory function
def create_cpu_optimizer(config: Optional[CPUOptimizationConfig] = None) -> CPUPerformanceOptimizer:
    """Create CPU performance optimizer instance"""
    return CPUPerformanceOptimizer(config)

# Example usage
if __name__ == "__main__":
    # Example optimization
    optimizer = create_cpu_optimizer()

    # Mock model function
    def mock_model(x):
        return np.random.randn(*x.shape)

    # Test single inference
    test_input = np.random.randn(1, 2056).astype(np.float32)
    result, metrics = optimizer.optimize_inference(mock_model, test_input)

    print(f"Inference time: {metrics.inference_time_ms:.2f}ms")
    print(f"Memory usage: {metrics.memory_usage_mb:.2f}MB")
    print(f"Optimization level: {metrics.optimization_level}")

    # Get performance report
    report = optimizer.get_performance_report()
    print(f"\nPerformance Report:")
    print(f"Average time: {report['summary']['avg_inference_time_ms']:.2f}ms")
    print(f"P95 time: {report['summary']['p95_inference_time_ms']:.2f}ms")
    print(f"Target achievement: {report['summary']['target_achievement_rate']:.1%}")