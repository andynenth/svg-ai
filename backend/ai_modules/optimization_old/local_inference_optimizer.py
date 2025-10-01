"""
Local CPU/MPS Inference Performance Testing & Optimization
Implements Task 12.2.2: Local CPU/MPS Inference Performance Testing

Comprehensive testing and optimization for:
- CPU inference optimization (single/multi-thread)
- Apple Silicon MPS acceleration
- Memory usage optimization
- Batch processing optimization
- Real-world deployment scenarios
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import threading
import multiprocessing
import platform
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')


@dataclass
class LocalInferenceConfig:
    """Configuration for local inference testing"""
    # Performance targets
    target_inference_time_ms: float = 50.0
    target_throughput_samples_per_sec: float = 20.0
    max_memory_usage_mb: float = 500.0

    # Test parameters
    test_samples: int = 1000
    warmup_iterations: int = 20
    timing_iterations: int = 100
    batch_sizes: List[int] = None

    # Threading configuration
    test_single_thread: bool = True
    test_multi_thread: bool = True
    max_threads: int = None

    # Device configuration
    test_cpu: bool = True
    test_mps: bool = True
    cpu_optimization: bool = True
    mps_optimization: bool = True

    # Memory optimization
    enable_memory_profiling: bool = True
    test_memory_pressure: bool = True

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32, 64]
        if self.max_threads is None:
            self.max_threads = min(8, multiprocessing.cpu_count())


@dataclass
class InferenceResult:
    """Result of inference performance test"""
    device: str
    model_format: str
    test_type: str

    # Performance metrics
    inference_time_ms: float
    throughput_samples_per_sec: float
    memory_usage_mb: float
    memory_peak_mb: float

    # Batch performance
    batch_performance: Dict[int, float] = None

    # Threading performance
    thread_performance: Dict[int, float] = None

    # Optimization results
    optimization_applied: bool = False
    optimization_speedup: float = 1.0

    # Test status
    passed: bool = False
    error_message: Optional[str] = None


class CPUOptimizer:
    """CPU-specific optimization for local inference"""

    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.cpu_count_logical = psutil.cpu_count(logical=True)

        print(f"💻 CPU Optimizer initialized")
        print(f"   Physical cores: {self.cpu_count}")
        print(f"   Logical cores: {self.cpu_count_logical}")

    def optimize_for_cpu(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for CPU inference"""
        print(f"🔧 Applying CPU optimizations...")

        # Set optimal thread count
        optimal_threads = min(4, self.cpu_count)  # Usually 4 threads is optimal for inference
        torch.set_num_threads(optimal_threads)

        # Optimize model for inference
        model.eval()

        # Apply graph optimization if TorchScript
        if hasattr(model, '_c'):  # TorchScript model
            try:
                model = torch.jit.optimize_for_inference(model)
                print(f"   ✅ TorchScript optimization applied")
            except Exception as e:
                print(f"   ⚠️ TorchScript optimization failed: {e}")

        # Set CPU-optimized settings
        torch.backends.mkldnn.enabled = True  # Enable Intel MKL-DNN if available

        print(f"   ✅ CPU optimization complete (threads: {optimal_threads})")
        return model

    def test_threading_performance(self, model: torch.nn.Module,
                                 test_input: torch.Tensor,
                                 max_threads: int = 8) -> Dict[int, float]:
        """Test performance across different thread counts"""
        print(f"🧵 Testing threading performance...")

        thread_performance = {}
        original_threads = torch.get_num_threads()

        for num_threads in range(1, max_threads + 1):
            torch.set_num_threads(num_threads)

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(test_input)

            # Time inference
            start_time = time.time()
            with torch.no_grad():
                for _ in range(20):
                    _ = model(test_input)

            total_time = time.time() - start_time
            inference_time_ms = (total_time / 20) * 1000
            thread_performance[num_threads] = inference_time_ms

            print(f"   Threads {num_threads}: {inference_time_ms:.1f}ms")

        # Restore original thread count
        torch.set_num_threads(original_threads)

        # Find optimal thread count
        optimal_threads = min(thread_performance, key=thread_performance.get)
        print(f"   🎯 Optimal thread count: {optimal_threads} ({thread_performance[optimal_threads]:.1f}ms)")

        return thread_performance


class MPSOptimizer:
    """Apple Silicon MPS optimization for local inference"""

    def __init__(self):
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self.device = torch.device('mps' if self.mps_available else 'cpu')

        print(f"🍎 MPS Optimizer initialized")
        print(f"   MPS available: {self.mps_available}")

    def optimize_for_mps(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for Apple Silicon MPS"""
        if not self.mps_available:
            print(f"   ⚠️ MPS not available, skipping optimization")
            return model

        print(f"🔧 Applying MPS optimizations...")

        try:
            # Move model to MPS
            model = model.to(self.device)
            model.eval()

            # MPS-specific optimizations
            if hasattr(torch.backends.mps, 'allow_tf32'):
                torch.backends.mps.allow_tf32 = True

            print(f"   ✅ MPS optimization complete")
            return model

        except Exception as e:
            print(f"   ❌ MPS optimization failed: {e}")
            return model.cpu()

    def test_mps_vs_cpu(self, model: torch.nn.Module,
                       test_input: torch.Tensor) -> Dict[str, float]:
        """Compare MPS vs CPU performance"""
        if not self.mps_available:
            return {"cpu": 0.0, "mps": float('inf')}

        print(f"⚖️ Comparing MPS vs CPU performance...")

        results = {}

        # Test CPU
        cpu_model = model.cpu()
        cpu_input = test_input.cpu()

        # CPU warmup
        with torch.no_grad():
            for _ in range(10):
                _ = cpu_model(cpu_input)

        # CPU timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = cpu_model(cpu_input)

        cpu_time = ((time.time() - start_time) / 50) * 1000
        results["cpu"] = cpu_time

        # Test MPS
        try:
            mps_model = model.to(self.device)
            mps_input = test_input.to(self.device)

            # MPS warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = mps_model(mps_input)

            # MPS timing
            start_time = time.time()
            with torch.no_grad():
                for _ in range(50):
                    output = mps_model(mps_input)
                    # Ensure computation is complete
                    if hasattr(output, 'cpu'):
                        _ = output.cpu()

            mps_time = ((time.time() - start_time) / 50) * 1000
            results["mps"] = mps_time

            speedup = cpu_time / mps_time if mps_time > 0 else 1.0
            print(f"   CPU: {cpu_time:.1f}ms")
            print(f"   MPS: {mps_time:.1f}ms")
            print(f"   🚀 MPS speedup: {speedup:.1f}x")

        except Exception as e:
            print(f"   ❌ MPS testing failed: {e}")
            results["mps"] = float('inf')

        return results


class MemoryProfiler:
    """Memory usage profiling and optimization"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.get_memory_usage()

        print(f"💾 Memory Profiler initialized")
        print(f"   Baseline memory: {self.baseline_memory:.1f}MB")

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / (1024 * 1024)

    def profile_inference(self, model: torch.nn.Module,
                         test_input: torch.Tensor,
                         iterations: int = 100) -> Dict[str, float]:
        """Profile memory usage during inference"""
        print(f"📊 Profiling memory usage...")

        memory_measurements = []
        peak_memory = self.baseline_memory

        # Initial measurement
        memory_measurements.append(self.get_memory_usage())

        # Run inference with memory monitoring
        model.eval()
        with torch.no_grad():
            for i in range(iterations):
                _ = model(test_input)

                # Measure memory every 10 iterations
                if i % 10 == 0:
                    current_memory = self.get_memory_usage()
                    memory_measurements.append(current_memory)
                    peak_memory = max(peak_memory, current_memory)

        # Final measurement
        final_memory = self.get_memory_usage()
        memory_measurements.append(final_memory)

        memory_stats = {
            "baseline_mb": self.baseline_memory,
            "peak_mb": peak_memory,
            "final_mb": final_memory,
            "increase_mb": final_memory - self.baseline_memory,
            "peak_increase_mb": peak_memory - self.baseline_memory
        }

        print(f"   Peak memory: {peak_memory:.1f}MB (+{memory_stats['peak_increase_mb']:.1f}MB)")
        print(f"   Final memory: {final_memory:.1f}MB (+{memory_stats['increase_mb']:.1f}MB)")

        return memory_stats

    def test_memory_pressure(self, model: torch.nn.Module,
                           test_input: torch.Tensor) -> Dict[str, Any]:
        """Test performance under memory pressure"""
        print(f"🔥 Testing under memory pressure...")

        # Create memory pressure
        memory_hogs = []
        try:
            # Allocate memory to create pressure
            for _ in range(5):
                memory_hog = torch.randn(1000, 1000, 100)  # ~400MB each
                memory_hogs.append(memory_hog)

            pressure_memory = self.get_memory_usage()

            # Test inference under pressure
            start_time = time.time()
            with torch.no_grad():
                for _ in range(20):
                    _ = model(test_input)

            inference_time = ((time.time() - start_time) / 20) * 1000

            result = {
                "memory_pressure_mb": pressure_memory,
                "inference_time_ms": inference_time,
                "performance_degradation": True if inference_time > 100 else False
            }

            print(f"   Under pressure: {inference_time:.1f}ms @ {pressure_memory:.1f}MB")

        except Exception as e:
            print(f"   ❌ Memory pressure test failed: {e}")
            result = {
                "memory_pressure_mb": 0,
                "inference_time_ms": float('inf'),
                "performance_degradation": True,
                "error": str(e)
            }

        finally:
            # Clean up memory hogs
            del memory_hogs

        return result


class LocalInferenceOptimizer:
    """Complete local inference optimization and testing system"""

    def __init__(self, config: LocalInferenceConfig = None):
        self.config = config or LocalInferenceConfig()

        # Initialize optimizers
        self.cpu_optimizer = CPUOptimizer()
        self.mps_optimizer = MPSOptimizer()
        self.memory_profiler = MemoryProfiler()

        # Results storage
        self.test_results = {}

        print(f"🚀 Local Inference Optimizer initialized")

    def optimize_and_test_model(self, model: torch.nn.Module,
                               model_format: str) -> Dict[str, InferenceResult]:
        """Comprehensive optimization and testing"""

        print(f"\n🔍 Testing {model_format} model...")
        results = {}

        # Create test input
        test_input = torch.randn(1, 2056)

        # Test CPU performance
        if self.config.test_cpu:
            print(f"\n💻 Testing CPU performance...")
            cpu_result = self._test_cpu_performance(model, test_input, model_format)
            results['cpu'] = cpu_result

        # Test MPS performance
        if self.config.test_mps and self.mps_optimizer.mps_available:
            print(f"\n🍎 Testing MPS performance...")
            mps_result = self._test_mps_performance(model, test_input, model_format)
            results['mps'] = mps_result

        # Compare devices
        if 'cpu' in results and 'mps' in results:
            self._compare_device_performance(results['cpu'], results['mps'])

        return results

    def _test_cpu_performance(self, model: torch.nn.Module,
                            test_input: torch.Tensor,
                            model_format: str) -> InferenceResult:
        """Test CPU performance with optimizations"""

        try:
            # Apply CPU optimizations
            optimized_model = self.cpu_optimizer.optimize_for_cpu(model.cpu())
            cpu_input = test_input.cpu()

            # Basic performance test
            inference_time, throughput = self._measure_inference_performance(
                optimized_model, cpu_input
            )

            # Memory profiling
            memory_stats = self.memory_profiler.profile_inference(
                optimized_model, cpu_input, self.config.timing_iterations
            )

            # Batch performance testing
            batch_performance = self._test_batch_performance(optimized_model, 'cpu')

            # Threading performance
            thread_performance = self.cpu_optimizer.test_threading_performance(
                optimized_model, cpu_input, self.config.max_threads
            )

            # Memory pressure test
            memory_pressure_result = self.memory_profiler.test_memory_pressure(
                optimized_model, cpu_input
            )

            # Check if targets are met
            passed = (
                inference_time < self.config.target_inference_time_ms and
                throughput > self.config.target_throughput_samples_per_sec and
                memory_stats['peak_mb'] < self.config.max_memory_usage_mb
            )

            result = InferenceResult(
                device='cpu',
                model_format=model_format,
                test_type='optimized',
                inference_time_ms=inference_time,
                throughput_samples_per_sec=throughput,
                memory_usage_mb=memory_stats['final_mb'],
                memory_peak_mb=memory_stats['peak_mb'],
                batch_performance=batch_performance,
                thread_performance=thread_performance,
                optimization_applied=True,
                optimization_speedup=1.0,  # Baseline
                passed=passed
            )

            print(f"   ✅ CPU test complete: {inference_time:.1f}ms, {throughput:.1f} samples/sec")
            return result

        except Exception as e:
            print(f"   ❌ CPU test failed: {e}")
            return InferenceResult(
                device='cpu',
                model_format=model_format,
                test_type='failed',
                inference_time_ms=float('inf'),
                throughput_samples_per_sec=0.0,
                memory_usage_mb=0.0,
                memory_peak_mb=0.0,
                passed=False,
                error_message=str(e)
            )

    def _test_mps_performance(self, model: torch.nn.Module,
                            test_input: torch.Tensor,
                            model_format: str) -> InferenceResult:
        """Test Apple Silicon MPS performance"""

        try:
            # Apply MPS optimizations
            optimized_model = self.mps_optimizer.optimize_for_mps(model)
            mps_input = test_input.to(self.mps_optimizer.device)

            # Basic performance test
            inference_time, throughput = self._measure_inference_performance(
                optimized_model, mps_input
            )

            # Memory profiling (approximate for MPS)
            memory_stats = self.memory_profiler.profile_inference(
                optimized_model, mps_input, self.config.timing_iterations
            )

            # Batch performance testing
            batch_performance = self._test_batch_performance(optimized_model, 'mps')

            # MPS vs CPU comparison
            device_comparison = self.mps_optimizer.test_mps_vs_cpu(model, test_input)
            speedup = device_comparison.get('cpu', 1.0) / device_comparison.get('mps', 1.0)

            # Check if targets are met
            passed = (
                inference_time < self.config.target_inference_time_ms and
                throughput > self.config.target_throughput_samples_per_sec
            )

            result = InferenceResult(
                device='mps',
                model_format=model_format,
                test_type='optimized',
                inference_time_ms=inference_time,
                throughput_samples_per_sec=throughput,
                memory_usage_mb=memory_stats['final_mb'],
                memory_peak_mb=memory_stats['peak_mb'],
                batch_performance=batch_performance,
                optimization_applied=True,
                optimization_speedup=speedup,
                passed=passed
            )

            print(f"   ✅ MPS test complete: {inference_time:.1f}ms, {throughput:.1f} samples/sec")
            return result

        except Exception as e:
            print(f"   ❌ MPS test failed: {e}")
            return InferenceResult(
                device='mps',
                model_format=model_format,
                test_type='failed',
                inference_time_ms=float('inf'),
                throughput_samples_per_sec=0.0,
                memory_usage_mb=0.0,
                memory_peak_mb=0.0,
                passed=False,
                error_message=str(e)
            )

    def _measure_inference_performance(self, model: torch.nn.Module,
                                     test_input: torch.Tensor) -> Tuple[float, float]:
        """Measure basic inference performance"""

        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = model(test_input)

        # Timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(self.config.timing_iterations):
                output = model(test_input)
                # Ensure computation is complete
                if hasattr(output, 'cpu'):
                    _ = output.cpu()

        total_time = time.time() - start_time
        inference_time_ms = (total_time / self.config.timing_iterations) * 1000
        throughput = self.config.timing_iterations / total_time

        return inference_time_ms, throughput

    def _test_batch_performance(self, model: torch.nn.Module, device: str) -> Dict[int, float]:
        """Test performance across batch sizes"""

        batch_performance = {}
        device_obj = torch.device(device)

        for batch_size in self.config.batch_sizes:
            try:
                batch_input = torch.randn(batch_size, 2056).to(device_obj)

                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(batch_input)

                # Timing
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(10):
                        output = model(batch_input)
                        if hasattr(output, 'cpu'):
                            _ = output.cpu()

                total_time = time.time() - start_time
                inference_time_ms = (total_time / 10) * 1000
                batch_performance[batch_size] = inference_time_ms

            except Exception as e:
                print(f"     ⚠️ Batch size {batch_size} failed: {e}")
                batch_performance[batch_size] = float('inf')

        return batch_performance

    def _compare_device_performance(self, cpu_result: InferenceResult,
                                  mps_result: InferenceResult):
        """Compare CPU vs MPS performance"""
        print(f"\n⚖️ CPU vs MPS Comparison:")
        print(f"   CPU: {cpu_result.inference_time_ms:.1f}ms, {cpu_result.throughput_samples_per_sec:.1f} samples/sec")
        print(f"   MPS: {mps_result.inference_time_ms:.1f}ms, {mps_result.throughput_samples_per_sec:.1f} samples/sec")

        if mps_result.inference_time_ms > 0:
            speedup = cpu_result.inference_time_ms / mps_result.inference_time_ms
            print(f"   🚀 MPS speedup: {speedup:.1f}x")
        else:
            print(f"   ❌ MPS comparison failed")

    def generate_optimization_report(self, all_results: Dict[str, Dict[str, InferenceResult]]) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""

        report = {
            "optimization_timestamp": time.time(),
            "system_info": {
                "platform": platform.system(),
                "cpu_count": self.cpu_optimizer.cpu_count,
                "mps_available": self.mps_optimizer.mps_available
            },
            "config": asdict(self.config),
            "results": {},
            "summary": {
                "models_tested": len(all_results),
                "cpu_tests_passed": 0,
                "mps_tests_passed": 0,
                "deployment_ready": []
            },
            "recommendations": []
        }

        # Process results
        for model_format, format_results in all_results.items():
            report["results"][model_format] = {}

            for device, result in format_results.items():
                report["results"][model_format][device] = asdict(result)

                # Count passes
                if result.passed:
                    if device == 'cpu':
                        report["summary"]["cpu_tests_passed"] += 1
                    elif device == 'mps':
                        report["summary"]["mps_tests_passed"] += 1

                    # Check deployment readiness
                    if result.inference_time_ms < self.config.target_inference_time_ms:
                        report["summary"]["deployment_ready"].append(f"{model_format}/{device}")

        # Generate recommendations
        report["recommendations"] = self._generate_optimization_recommendations(all_results)

        # Save report
        report_path = Path("local_inference_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Local inference report saved: {report_path}")
        self._print_optimization_summary(report)

        return report

    def _generate_optimization_recommendations(self, results: Dict[str, Dict[str, InferenceResult]]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Analyze performance patterns
        fastest_times = []
        for format_results in results.values():
            for result in format_results.values():
                if result.passed:
                    fastest_times.append(result.inference_time_ms)

        if fastest_times:
            best_time = min(fastest_times)
            if best_time < 25:
                recommendations.append("Excellent inference performance achieved. Ready for production deployment.")
            elif best_time < 50:
                recommendations.append("Good inference performance. Consider batch optimization for higher throughput.")
            else:
                recommendations.append("Performance targets met but consider further optimization for production.")

        # Device-specific recommendations
        mps_available = any(
            'mps' in format_results for format_results in results.values()
        )
        if mps_available:
            recommendations.append("Apple Silicon MPS acceleration available. Recommend CoreML for optimal performance.")
        else:
            recommendations.append("Consider CPU threading optimization and TorchScript compilation for best performance.")

        return recommendations

    def _print_optimization_summary(self, report: Dict[str, Any]):
        """Print optimization summary"""
        print(f"\n📋 Local Inference Optimization Summary")
        print(f"=" * 60)

        summary = report["summary"]
        print(f"🧪 Models tested: {summary['models_tested']}")
        print(f"💻 CPU tests passed: {summary['cpu_tests_passed']}")
        print(f"🍎 MPS tests passed: {summary['mps_tests_passed']}")
        print(f"🚀 Deployment ready: {len(summary['deployment_ready'])}")

        if summary['deployment_ready']:
            print(f"   Ready for deployment: {', '.join(summary['deployment_ready'])}")

        # Performance summary
        print(f"\n⚡ Performance Summary:")
        for model_format, format_results in report["results"].items():
            for device, result in format_results.items():
                status = "✅" if result.get('passed', False) else "❌"
                inference_time = result.get('inference_time_ms', 0)
                throughput = result.get('throughput_samples_per_sec', 0)
                print(f"   {status} {model_format}/{device}: {inference_time:.1f}ms, {throughput:.1f} samples/sec")

        # Recommendations
        if report["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Local Inference Optimizer")

    # Create config
    config = LocalInferenceConfig(
        target_inference_time_ms=50.0,
        test_samples=100
    )

    # Create optimizer
    optimizer = LocalInferenceOptimizer(config)

    print("✅ Local inference optimizer test completed")