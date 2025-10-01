#!/usr/bin/env python3
"""
Performance Testing Framework for Quality Prediction Integration
Comprehensive testing suite for validating <25ms inference performance and production readiness
"""

import os
import time
import threading
import multiprocessing as mp
import numpy as np
import psutil
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns

from .unified_prediction_api import UnifiedPredictionAPI, UnifiedPredictionConfig, PredictionMethod
from .quality_prediction_integration import QualityPredictionIntegrator, QualityPredictionConfig
from .cpu_performance_optimizer import CPUPerformanceOptimizer, CPUOptimizationConfig

logger = logging.getLogger(__name__)

@dataclass
class PerformanceTestConfig:
    """Configuration for performance testing"""
    target_inference_ms: float = 25.0
    fallback_target_ms: float = 50.0
    test_duration_seconds: int = 60
    warmup_iterations: int = 10
    test_iterations: int = 1000
    stress_test_duration: int = 300  # 5 minutes
    concurrent_threads: int = 4
    memory_limit_mb: int = 1024
    enable_profiling: bool = True
    save_detailed_results: bool = True
    test_batch_sizes: List[int] = None
    enable_stress_testing: bool = True

    def __post_init__(self):
        if self.test_batch_sizes is None:
            self.test_batch_sizes = [1, 2, 4, 8, 16]

@dataclass
class PerformanceTestResult:
    """Individual performance test result"""
    test_name: str
    method_used: str
    inference_time_ms: float
    memory_usage_mb: float
    cpu_utilization: float
    quality_score: float
    success: bool
    error: Optional[str]
    timestamp: float
    batch_size: int = 1
    thread_id: Optional[str] = None

@dataclass
class PerformanceBenchmark:
    """Performance benchmark results"""
    test_config: PerformanceTestConfig
    total_tests: int
    successful_tests: int
    failed_tests: int
    avg_inference_time_ms: float
    p50_inference_time_ms: float
    p95_inference_time_ms: float
    p99_inference_time_ms: float
    max_inference_time_ms: float
    min_inference_time_ms: float
    target_achievement_rate: float
    avg_memory_usage_mb: float
    avg_cpu_utilization: float
    avg_quality_score: float
    throughput_per_second: float
    detailed_results: List[PerformanceTestResult]
    system_info: Dict[str, Any]
    test_duration_seconds: float
    timestamp: float

class MockModelInterface:
    """Mock model interface for testing without actual models"""

    def __init__(self, target_time_ms: float = 20.0, variability: float = 0.2):
        self.target_time_ms = target_time_ms
        self.variability = variability
        self.call_count = 0

    def predict_quality(self, image_path: str, vtracer_params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock prediction with realistic timing"""
        self.call_count += 1

        # Simulate realistic inference time with variability
        base_time = self.target_time_ms / 1000.0
        actual_time = base_time * (1.0 + np.random.normal(0, self.variability))
        actual_time = max(0.001, actual_time)  # Minimum 1ms

        time.sleep(actual_time)

        return {
            'quality_score': np.random.uniform(0.7, 0.98),
            'confidence': np.random.uniform(0.8, 0.99),
            'inference_time_ms': actual_time * 1000,
            'model_version': 'mock-1.0',
            'device_used': 'cpu',
            'optimization_level': 'aggressive',
            'cache_hit': False,
            'fallback_used': False,
            'timestamp': time.time()
        }

class SystemMonitor:
    """System resource monitoring during tests"""

    def __init__(self, monitor_interval: float = 0.1):
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.monitor_thread = None
        self.resource_history = deque(maxlen=10000)

    def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()

                # Get current process metrics
                process = psutil.Process()
                process_memory = process.memory_info().rss / 1024 / 1024  # MB

                resource_data = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'process_memory_mb': process_memory,
                    'cpu_count': psutil.cpu_count(),
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
                }

                self.resource_history.append(resource_data)

            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")

            time.sleep(self.monitor_interval)

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary"""
        if not self.resource_history:
            return {}

        cpu_values = [r['cpu_percent'] for r in self.resource_history]
        memory_values = [r['process_memory_mb'] for r in self.resource_history]

        return {
            'cpu_utilization': {
                'avg': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'p95': np.percentile(cpu_values, 95)
            },
            'memory_usage_mb': {
                'avg': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'p95': np.percentile(memory_values, 95)
            },
            'sample_count': len(self.resource_history)
        }

class PerformanceTestSuite:
    """Comprehensive performance testing suite"""

    def __init__(self, config: Optional[PerformanceTestConfig] = None):
        self.config = config or PerformanceTestConfig()
        self.system_monitor = SystemMonitor()
        self.test_results = []
        self.test_start_time = None

        # Initialize prediction API (with mock if models not available)
        self.prediction_api = None
        self.mock_interface = None
        self._initialize_test_environment()

    def _initialize_test_environment(self):
        """Initialize the test environment"""
        try:
            # Try to initialize real prediction API
            api_config = UnifiedPredictionConfig(
                performance_target_ms=self.config.target_inference_ms
            )
            self.prediction_api = UnifiedPredictionAPI(api_config)
            logger.info("Initialized real prediction API for testing")

        except Exception as e:
            logger.warning(f"Failed to initialize real API, using mock interface: {e}")
            self.mock_interface = MockModelInterface(
                target_time_ms=self.config.target_inference_ms
            )

    def run_comprehensive_benchmark(self) -> PerformanceBenchmark:
        """Run comprehensive performance benchmark"""
        logger.info("Starting comprehensive performance benchmark")
        self.test_start_time = time.time()
        self.system_monitor.start_monitoring()

        try:
            # Run all test suites
            test_suites = [
                ("Single Inference", self._test_single_inference),
                ("Batch Processing", self._test_batch_processing),
                ("Concurrent Load", self._test_concurrent_load),
                ("Memory Pressure", self._test_memory_pressure),
                ("Sustained Load", self._test_sustained_load),
                ("Cold Start", self._test_cold_start_performance)
            ]

            all_results = []

            for suite_name, test_func in test_suites:
                logger.info(f"Running test suite: {suite_name}")
                try:
                    suite_results = test_func()
                    all_results.extend(suite_results)
                    logger.info(f"Completed {suite_name}: {len(suite_results)} tests")
                except Exception as e:
                    logger.error(f"Test suite {suite_name} failed: {e}")

            # Generate benchmark report
            benchmark = self._generate_benchmark_report(all_results)

            return benchmark

        finally:
            self.system_monitor.stop_monitoring()

    def _test_single_inference(self) -> List[PerformanceTestResult]:
        """Test single inference performance"""
        results = []

        # Warmup
        self._run_warmup()

        # Test different scenarios
        test_scenarios = [
            ("simple_logo", self._create_simple_params()),
            ("complex_logo", self._create_complex_params()),
            ("text_logo", self._create_text_params()),
            ("gradient_logo", self._create_gradient_params())
        ]

        for scenario_name, params in test_scenarios:
            for i in range(self.config.test_iterations // len(test_scenarios)):
                result = self._run_single_test(
                    f"single_{scenario_name}_{i}",
                    "mock_image.png",
                    params
                )
                results.append(result)

        return results

    def _test_batch_processing(self) -> List[PerformanceTestResult]:
        """Test batch processing performance"""
        results = []

        for batch_size in self.config.test_batch_sizes:
            batch_params = [self._create_simple_params() for _ in range(batch_size)]
            batch_images = [f"mock_image_{i}.png" for i in range(batch_size)]

            for i in range(max(1, self.config.test_iterations // (len(self.config.test_batch_sizes) * 10))):
                result = self._run_batch_test(
                    f"batch_{batch_size}_{i}",
                    batch_images,
                    batch_params
                )
                results.extend(result)

        return results

    def _test_concurrent_load(self) -> List[PerformanceTestResult]:
        """Test concurrent processing performance"""
        results = []
        results_lock = threading.Lock()

        def worker_thread(thread_id: str, test_count: int):
            thread_results = []
            for i in range(test_count):
                result = self._run_single_test(
                    f"concurrent_{thread_id}_{i}",
                    "mock_image.png",
                    self._create_simple_params()
                )
                result.thread_id = thread_id
                thread_results.append(result)

            with results_lock:
                results.extend(thread_results)

        # Run concurrent threads
        tests_per_thread = self.config.test_iterations // self.config.concurrent_threads
        threads = []

        for t in range(self.config.concurrent_threads):
            thread = threading.Thread(
                target=worker_thread,
                args=(f"thread_{t}", tests_per_thread)
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        return results

    def _test_memory_pressure(self) -> List[PerformanceTestResult]:
        """Test performance under memory pressure"""
        results = []

        # Create memory pressure
        memory_hogs = []
        try:
            # Allocate memory to create pressure
            available_memory = psutil.virtual_memory().available
            pressure_size = min(
                self.config.memory_limit_mb * 1024 * 1024,
                int(available_memory * 0.3)  # Use 30% of available memory
            )

            chunk_size = 10 * 1024 * 1024  # 10MB chunks
            chunks_needed = pressure_size // chunk_size

            for _ in range(chunks_needed):
                chunk = np.random.bytes(chunk_size)
                memory_hogs.append(chunk)

            logger.info(f"Created memory pressure: {len(memory_hogs) * chunk_size / 1024 / 1024:.1f} MB")

            # Run tests under memory pressure
            for i in range(self.config.test_iterations // 4):  # Fewer iterations due to memory pressure
                result = self._run_single_test(
                    f"memory_pressure_{i}",
                    "mock_image.png",
                    self._create_simple_params()
                )
                results.append(result)

        finally:
            # Clean up memory
            del memory_hogs

        return results

    def _test_sustained_load(self) -> List[PerformanceTestResult]:
        """Test sustained load performance"""
        results = []
        start_time = time.time()
        test_count = 0

        logger.info(f"Starting sustained load test for {self.config.stress_test_duration} seconds")

        while time.time() - start_time < self.config.stress_test_duration:
            result = self._run_single_test(
                f"sustained_{test_count}",
                "mock_image.png",
                self._create_simple_params()
            )
            results.append(result)
            test_count += 1

            # Brief pause to prevent overwhelming the system
            time.sleep(0.001)

        logger.info(f"Completed sustained load test: {test_count} tests in {time.time() - start_time:.1f}s")
        return results

    def _test_cold_start_performance(self) -> List[PerformanceTestResult]:
        """Test cold start performance"""
        results = []

        # Test cold start by creating new API instances
        for i in range(5):  # Test 5 cold starts
            try:
                # Create new API instance
                if self.prediction_api:
                    cold_api = UnifiedPredictionAPI(UnifiedPredictionConfig(
                        performance_target_ms=self.config.target_inference_ms
                    ))
                    start_time = time.time()

                    # First prediction (cold start)
                    # This would use the new API instance
                    result = self._run_single_test(
                        f"cold_start_{i}",
                        "mock_image.png",
                        self._create_simple_params(),
                        api_instance=cold_api
                    )
                    result.test_name = f"cold_start_first_{i}"
                    results.append(result)

                    # Second prediction (warmed up)
                    result = self._run_single_test(
                        f"cold_start_second_{i}",
                        "mock_image.png",
                        self._create_simple_params(),
                        api_instance=cold_api
                    )
                    result.test_name = f"cold_start_second_{i}"
                    results.append(result)

                    # Cleanup
                    cold_api.cleanup()

                else:
                    # Mock cold start
                    mock = MockModelInterface(target_time_ms=self.config.target_inference_ms * 2)  # Slower cold start
                    result_data = mock.predict_quality("mock_image.png", self._create_simple_params())

                    result = PerformanceTestResult(
                        test_name=f"cold_start_mock_{i}",
                        method_used="mock",
                        inference_time_ms=result_data['inference_time_ms'],
                        memory_usage_mb=50.0,  # Estimated
                        cpu_utilization=psutil.cpu_percent(),
                        quality_score=result_data['quality_score'],
                        success=True,
                        error=None,
                        timestamp=time.time()
                    )
                    results.append(result)

            except Exception as e:
                logger.error(f"Cold start test {i} failed: {e}")

        return results

    def _run_warmup(self):
        """Run warmup iterations"""
        logger.info(f"Running {self.config.warmup_iterations} warmup iterations")

        for i in range(self.config.warmup_iterations):
            try:
                self._run_single_test(
                    f"warmup_{i}",
                    "mock_image.png",
                    self._create_simple_params()
                )
            except Exception as e:
                logger.warning(f"Warmup iteration {i} failed: {e}")

    def _run_single_test(self, test_name: str, image_path: str, vtracer_params: Dict[str, Any],
                        api_instance: Optional[UnifiedPredictionAPI] = None) -> PerformanceTestResult:
        """Run a single performance test"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            api = api_instance or self.prediction_api

            if api:
                # Use real API
                result = api.predict_quality(image_path, vtracer_params)

                inference_time = result.inference_time_ms
                quality_score = result.quality_score
                method_used = result.method_used
                success = not result.fallback_used
                error = None

            else:
                # Use mock interface
                result_data = self.mock_interface.predict_quality(image_path, vtracer_params)

                inference_time = result_data['inference_time_ms']
                quality_score = result_data['quality_score']
                method_used = "mock"
                success = True
                error = None

            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory

            return PerformanceTestResult(
                test_name=test_name,
                method_used=method_used,
                inference_time_ms=inference_time,
                memory_usage_mb=memory_usage,
                cpu_utilization=psutil.cpu_percent(),
                quality_score=quality_score,
                success=success,
                error=error,
                timestamp=time.time()
            )

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory

            return PerformanceTestResult(
                test_name=test_name,
                method_used="error",
                inference_time_ms=total_time,
                memory_usage_mb=memory_usage,
                cpu_utilization=psutil.cpu_percent(),
                quality_score=0.0,
                success=False,
                error=str(e),
                timestamp=time.time()
            )

    def _run_batch_test(self, test_name: str, image_paths: List[str],
                       vtracer_params_list: List[Dict[str, Any]]) -> List[PerformanceTestResult]:
        """Run batch test"""
        batch_start_time = time.time()
        results = []

        try:
            if self.prediction_api:
                # Use real API batch processing
                batch_results = self.prediction_api.predict_quality_batch(
                    image_paths, vtracer_params_list
                )

                for i, result in enumerate(batch_results):
                    test_result = PerformanceTestResult(
                        test_name=f"{test_name}_item_{i}",
                        method_used=result.method_used,
                        inference_time_ms=result.inference_time_ms,
                        memory_usage_mb=0.0,  # Batch doesn't track individual memory
                        cpu_utilization=psutil.cpu_percent(),
                        quality_score=result.quality_score,
                        success=not result.fallback_used,
                        error=None,
                        timestamp=result.timestamp,
                        batch_size=len(image_paths)
                    )
                    results.append(test_result)

            else:
                # Use mock interface
                for i, (image_path, params) in enumerate(zip(image_paths, vtracer_params_list)):
                    result_data = self.mock_interface.predict_quality(image_path, params)

                    test_result = PerformanceTestResult(
                        test_name=f"{test_name}_item_{i}",
                        method_used="mock",
                        inference_time_ms=result_data['inference_time_ms'],
                        memory_usage_mb=5.0,  # Estimated
                        cpu_utilization=psutil.cpu_percent(),
                        quality_score=result_data['quality_score'],
                        success=True,
                        error=None,
                        timestamp=time.time(),
                        batch_size=len(image_paths)
                    )
                    results.append(test_result)

        except Exception as e:
            # Create error results for each item in batch
            for i in range(len(image_paths)):
                error_result = PerformanceTestResult(
                    test_name=f"{test_name}_item_{i}_error",
                    method_used="error",
                    inference_time_ms=(time.time() - batch_start_time) * 1000 / len(image_paths),
                    memory_usage_mb=0.0,
                    cpu_utilization=psutil.cpu_percent(),
                    quality_score=0.0,
                    success=False,
                    error=str(e),
                    timestamp=time.time(),
                    batch_size=len(image_paths)
                )
                results.append(error_result)

        return results

    def _generate_benchmark_report(self, results: List[PerformanceTestResult]) -> PerformanceBenchmark:
        """Generate comprehensive benchmark report"""
        if not results:
            raise ValueError("No test results to analyze")

        # Filter successful results for performance metrics
        successful_results = [r for r in results if r.success]
        inference_times = [r.inference_time_ms for r in successful_results]
        memory_usage = [r.memory_usage_mb for r in successful_results]
        cpu_utilization = [r.cpu_utilization for r in successful_results]
        quality_scores = [r.quality_score for r in successful_results if r.quality_score > 0]

        # Calculate performance metrics
        target_achievement_rate = 0.0
        if inference_times:
            target_achievement_rate = sum(
                1 for t in inference_times if t <= self.config.target_inference_ms
            ) / len(inference_times)

        # System information
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'platform': os.uname()._asdict() if hasattr(os, 'uname') else {},
            'python_version': os.sys.version,
            'test_environment': 'mock' if self.mock_interface else 'real'
        }

        # Add resource monitoring summary
        system_info['resource_monitoring'] = self.system_monitor.get_resource_summary()

        # Calculate throughput
        test_duration = time.time() - self.test_start_time if self.test_start_time else 1.0
        throughput = len(successful_results) / test_duration

        benchmark = PerformanceBenchmark(
            test_config=self.config,
            total_tests=len(results),
            successful_tests=len(successful_results),
            failed_tests=len(results) - len(successful_results),
            avg_inference_time_ms=statistics.mean(inference_times) if inference_times else 0.0,
            p50_inference_time_ms=np.percentile(inference_times, 50) if inference_times else 0.0,
            p95_inference_time_ms=np.percentile(inference_times, 95) if inference_times else 0.0,
            p99_inference_time_ms=np.percentile(inference_times, 99) if inference_times else 0.0,
            max_inference_time_ms=max(inference_times) if inference_times else 0.0,
            min_inference_time_ms=min(inference_times) if inference_times else 0.0,
            target_achievement_rate=target_achievement_rate,
            avg_memory_usage_mb=statistics.mean(memory_usage) if memory_usage else 0.0,
            avg_cpu_utilization=statistics.mean(cpu_utilization) if cpu_utilization else 0.0,
            avg_quality_score=statistics.mean(quality_scores) if quality_scores else 0.0,
            throughput_per_second=throughput,
            detailed_results=results if self.config.save_detailed_results else [],
            system_info=system_info,
            test_duration_seconds=test_duration,
            timestamp=time.time()
        )

        return benchmark

    def _create_simple_params(self) -> Dict[str, Any]:
        """Create simple logo parameters"""
        return {
            'color_precision': 3.0,
            'corner_threshold': 30.0,
            'path_precision': 5.0,
            'layer_difference': 5.0,
            'filter_speckle': 2.0,
            'splice_threshold': 45.0,
            'mode': 0.0,
            'hierarchical': 1.0
        }

    def _create_complex_params(self) -> Dict[str, Any]:
        """Create complex logo parameters"""
        return {
            'color_precision': 8.0,
            'corner_threshold': 20.0,
            'path_precision': 15.0,
            'layer_difference': 10.0,
            'filter_speckle': 1.0,
            'splice_threshold': 30.0,
            'mode': 0.0,
            'hierarchical': 1.0
        }

    def _create_text_params(self) -> Dict[str, Any]:
        """Create text logo parameters"""
        return {
            'color_precision': 2.0,
            'corner_threshold': 20.0,
            'path_precision': 10.0,
            'layer_difference': 16.0,
            'filter_speckle': 4.0,
            'splice_threshold': 60.0,
            'mode': 0.0,
            'hierarchical': 0.0
        }

    def _create_gradient_params(self) -> Dict[str, Any]:
        """Create gradient logo parameters"""
        return {
            'color_precision': 10.0,
            'corner_threshold': 40.0,
            'path_precision': 8.0,
            'layer_difference': 8.0,
            'filter_speckle': 2.0,
            'splice_threshold': 35.0,
            'mode': 1.0,
            'hierarchical': 1.0
        }

    def save_benchmark_report(self, benchmark: PerformanceBenchmark, output_path: str):
        """Save benchmark report to file"""
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Convert to JSON-serializable format
            report_data = asdict(benchmark)

            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            logger.info(f"Benchmark report saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save benchmark report: {e}")

    def generate_performance_plots(self, benchmark: PerformanceBenchmark, output_dir: str):
        """Generate performance visualization plots"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if not benchmark.detailed_results:
                logger.warning("No detailed results available for plotting")
                return

            successful_results = [r for r in benchmark.detailed_results if r.success]
            if not successful_results:
                logger.warning("No successful results for plotting")
                return

            # Inference time distribution
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            inference_times = [r.inference_time_ms for r in successful_results]
            plt.hist(inference_times, bins=50, alpha=0.7, color='blue')
            plt.axvline(benchmark.avg_inference_time_ms, color='red', linestyle='--', label=f'Mean: {benchmark.avg_inference_time_ms:.1f}ms')
            plt.axvline(self.config.target_inference_ms, color='green', linestyle='--', label=f'Target: {self.config.target_inference_ms}ms')
            plt.xlabel('Inference Time (ms)')
            plt.ylabel('Frequency')
            plt.title('Inference Time Distribution')
            plt.legend()

            # Performance over time
            plt.subplot(2, 2, 2)
            timestamps = [r.timestamp for r in successful_results]
            start_time = min(timestamps)
            relative_times = [(t - start_time) / 60 for t in timestamps]  # Minutes
            plt.scatter(relative_times, inference_times, alpha=0.6, s=10)
            plt.xlabel('Time (minutes)')
            plt.ylabel('Inference Time (ms)')
            plt.title('Performance Over Time')
            plt.axhline(self.config.target_inference_ms, color='red', linestyle='--', alpha=0.7)

            # Method comparison
            plt.subplot(2, 2, 3)
            method_times = defaultdict(list)
            for r in successful_results:
                method_times[r.method_used].append(r.inference_time_ms)

            methods = list(method_times.keys())
            avg_times = [statistics.mean(method_times[m]) for m in methods]
            plt.bar(methods, avg_times, alpha=0.7)
            plt.xlabel('Method')
            plt.ylabel('Average Inference Time (ms)')
            plt.title('Performance by Method')
            plt.xticks(rotation=45)

            # Memory usage
            plt.subplot(2, 2, 4)
            memory_usage = [r.memory_usage_mb for r in successful_results]
            plt.plot(relative_times, memory_usage, alpha=0.7, linewidth=1)
            plt.xlabel('Time (minutes)')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage Over Time')

            plt.tight_layout()
            plt.savefig(output_path / 'performance_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Performance plots saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to generate performance plots: {e}")

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.prediction_api:
                self.prediction_api.cleanup()

            self.system_monitor.stop_monitoring()

            logger.info("Performance testing framework cleanup complete")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Factory function
def create_performance_test_suite(config: Optional[PerformanceTestConfig] = None) -> PerformanceTestSuite:
    """Create performance test suite instance"""
    return PerformanceTestSuite(config)

# Command-line interface for running tests
def run_performance_benchmark():
    """Run performance benchmark from command line"""
    import argparse

    parser = argparse.ArgumentParser(description='Run quality prediction performance benchmark')
    parser.add_argument('--target-ms', type=float, default=25.0, help='Target inference time in ms')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of test iterations')
    parser.add_argument('--threads', type=int, default=4, help='Number of concurrent threads')
    parser.add_argument('--output', type=str, default='performance_results', help='Output directory')
    parser.add_argument('--stress-duration', type=int, default=60, help='Stress test duration in seconds')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create test configuration
    config = PerformanceTestConfig(
        target_inference_ms=args.target_ms,
        test_iterations=args.iterations,
        concurrent_threads=args.threads,
        stress_test_duration=args.stress_duration
    )

    # Run benchmark
    test_suite = create_performance_test_suite(config)

    try:
        logger.info("Starting comprehensive performance benchmark")
        benchmark = test_suite.run_comprehensive_benchmark()

        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / 'benchmark_report.json'
        test_suite.save_benchmark_report(benchmark, str(report_path))

        # Generate plots
        test_suite.generate_performance_plots(benchmark, str(output_dir))

        # Print summary
        print(f"\n{'='*60}")
        print(f"PERFORMANCE BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {benchmark.total_tests}")
        print(f"Successful: {benchmark.successful_tests}")
        print(f"Failed: {benchmark.failed_tests}")
        print(f"Success Rate: {benchmark.successful_tests/benchmark.total_tests:.1%}")
        print(f"\nPerformance Metrics:")
        print(f"Average Inference Time: {benchmark.avg_inference_time_ms:.2f}ms")
        print(f"P95 Inference Time: {benchmark.p95_inference_time_ms:.2f}ms")
        print(f"P99 Inference Time: {benchmark.p99_inference_time_ms:.2f}ms")
        print(f"Target Achievement Rate: {benchmark.target_achievement_rate:.1%}")
        print(f"Throughput: {benchmark.throughput_per_second:.1f} predictions/sec")
        print(f"\nTarget Status: {'✅ ACHIEVED' if benchmark.target_achievement_rate > 0.9 else '❌ NOT ACHIEVED'}")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

    finally:
        test_suite.cleanup()

    return 0

if __name__ == "__main__":
    exit(run_performance_benchmark())