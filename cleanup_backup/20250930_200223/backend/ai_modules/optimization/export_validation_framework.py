"""
Export Validation Framework for SVG Quality Prediction Models
Implements Task 12.2.2: Cross-Platform Export Validation & Performance Testing

Validates exported models across formats with comprehensive testing:
- Accuracy preservation validation
- Cross-platform compatibility testing
- Local CPU/MPS inference performance testing
- Model size and optimization validation
"""

import torch
import numpy as np
import json
import time
import platform
import psutil
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import warnings
from collections import defaultdict

# Import export pipeline
from .model_export_pipeline import ExportResult, ExportConfig, load_trained_model

warnings.filterwarnings('ignore')


@dataclass
class ValidationConfig:
    """Configuration for export validation"""
    # Accuracy validation settings
    accuracy_tolerance: float = 0.001  # Maximum prediction difference tolerance
    test_samples: int = 100  # Number of test samples
    correlation_threshold: float = 0.99  # Minimum correlation with original model

    # Performance validation settings
    max_inference_time_ms: float = 50.0  # Maximum inference time
    min_throughput_samples_per_sec: float = 20.0  # Minimum throughput
    warmup_iterations: int = 10  # Warmup before timing
    timing_iterations: int = 50  # Iterations for timing

    # System validation settings
    max_memory_usage_mb: float = 500.0  # Maximum memory usage
    test_batch_sizes: List[int] = None  # Batch sizes to test

    # Platform-specific settings
    test_cpu: bool = True
    test_mps: bool = True  # Apple Silicon
    test_gpu: bool = False  # Skip GPU testing for local deployment

    def __post_init__(self):
        if self.test_batch_sizes is None:
            self.test_batch_sizes = [1, 4, 8, 16, 32]


@dataclass
class ValidationResult:
    """Result of validation test"""
    format: str
    platform: str
    device: str

    # Accuracy metrics
    accuracy_preserved: bool
    max_prediction_diff: float
    correlation_with_original: float

    # Performance metrics
    inference_time_ms: float
    throughput_samples_per_sec: float
    memory_usage_mb: float

    # Test status
    passed: bool
    error_message: Optional[str] = None

    # Batch performance
    batch_performance: Dict[int, float] = None  # batch_size -> inference_time_ms


class SystemProfiler:
    """Profile system capabilities for validation"""

    def __init__(self):
        self.system_info = self._gather_system_info()
        print(f"💻 System Profiler initialized")
        self._print_system_info()

    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information"""
        info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }

        # GPU information
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return info

    def _print_system_info(self):
        """Print system information"""
        print(f"   Platform: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"   CPU: {self.system_info['cpu_count']} cores ({self.system_info['cpu_count_logical']} logical)")
        print(f"   Memory: {self.system_info['memory_total_gb']:.1f}GB")
        print(f"   PyTorch: {self.system_info['pytorch_version']}")

        if self.system_info['cuda_available']:
            print(f"   GPU: {self.system_info.get('gpu_name', 'Unknown')} ({self.system_info.get('gpu_memory_gb', 0):.1f}GB)")

        if self.system_info['mps_available']:
            print(f"   MPS: Apple Silicon GPU acceleration available")


class AccuracyValidator:
    """Validate model accuracy preservation across export formats"""

    def __init__(self, original_model: torch.nn.Module, config: ValidationConfig):
        self.original_model = original_model
        self.config = config

        # Generate test samples
        self.test_samples = self._generate_test_samples()

        # Get original predictions for comparison
        self.original_predictions = self._get_original_predictions()

        print(f"🎯 Accuracy Validator initialized with {len(self.test_samples)} test samples")

    def _generate_test_samples(self) -> torch.Tensor:
        """Generate diverse test samples for validation"""
        samples = []

        # Random samples across the input range
        for _ in range(self.config.test_samples):
            # Image features (2048 dims) - ResNet output range
            image_features = torch.randn(2048) * 0.5  # Typical ResNet output scale

            # VTracer parameters (8 dims) - normalized [0,1]
            vtracer_params = torch.rand(8)

            # Combine features
            sample = torch.cat([image_features, vtracer_params])
            samples.append(sample)

        return torch.stack(samples)

    def _get_original_predictions(self) -> torch.Tensor:
        """Get predictions from original model"""
        self.original_model.eval()

        with torch.no_grad():
            predictions = self.original_model(self.test_samples)

        return predictions.squeeze()

    def validate_exported_model(self, model_path: str, format: str, device: str = 'cpu') -> Dict[str, Any]:
        """Validate accuracy of exported model"""
        try:
            # Load exported model based on format
            if format.startswith('TorchScript'):
                model = torch.jit.load(model_path, map_location=device)
            elif format == 'ONNX':
                return self._validate_onnx_accuracy(model_path)
            elif format == 'CoreML':
                return self._validate_coreml_accuracy(model_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            model.eval()

            # Get predictions from exported model
            test_samples_device = self.test_samples.to(device)

            with torch.no_grad():
                exported_predictions = model(test_samples_device)

            if exported_predictions.dim() > 1:
                exported_predictions = exported_predictions.squeeze()

            # Move to CPU for comparison
            exported_predictions = exported_predictions.cpu()

            # Calculate accuracy metrics
            prediction_diffs = torch.abs(exported_predictions - self.original_predictions)
            max_diff = torch.max(prediction_diffs).item()

            # Calculate correlation
            correlation = torch.corrcoef(torch.stack([
                exported_predictions, self.original_predictions
            ]))[0, 1].item()

            # Check if accuracy is preserved
            accuracy_preserved = (
                max_diff < self.config.accuracy_tolerance and
                correlation > self.config.correlation_threshold
            )

            return {
                'accuracy_preserved': accuracy_preserved,
                'max_prediction_diff': max_diff,
                'correlation_with_original': correlation,
                'mean_absolute_error': torch.mean(prediction_diffs).item(),
                'success': True
            }

        except Exception as e:
            return {
                'accuracy_preserved': False,
                'max_prediction_diff': float('inf'),
                'correlation_with_original': 0.0,
                'mean_absolute_error': float('inf'),
                'success': False,
                'error': str(e)
            }

    def _validate_onnx_accuracy(self, model_path: str) -> Dict[str, Any]:
        """Validate ONNX model accuracy"""
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name

            # Run inference
            predictions = []
            for sample in self.test_samples:
                pred = session.run(None, {input_name: sample.unsqueeze(0).numpy()})[0]
                predictions.append(pred[0])

            exported_predictions = torch.tensor(predictions).squeeze()

            # Calculate metrics
            prediction_diffs = torch.abs(exported_predictions - self.original_predictions)
            max_diff = torch.max(prediction_diffs).item()

            correlation = torch.corrcoef(torch.stack([
                exported_predictions, self.original_predictions
            ]))[0, 1].item()

            accuracy_preserved = (
                max_diff < self.config.accuracy_tolerance and
                correlation > self.config.correlation_threshold
            )

            return {
                'accuracy_preserved': accuracy_preserved,
                'max_prediction_diff': max_diff,
                'correlation_with_original': correlation,
                'mean_absolute_error': torch.mean(prediction_diffs).item(),
                'success': True
            }

        except ImportError:
            return {
                'accuracy_preserved': False,
                'max_prediction_diff': float('inf'),
                'correlation_with_original': 0.0,
                'success': False,
                'error': 'ONNX Runtime not available'
            }
        except Exception as e:
            return {
                'accuracy_preserved': False,
                'max_prediction_diff': float('inf'),
                'correlation_with_original': 0.0,
                'success': False,
                'error': str(e)
            }

    def _validate_coreml_accuracy(self, model_path: str) -> Dict[str, Any]:
        """Validate CoreML model accuracy"""
        try:
            if platform.system() != "Darwin":
                return {
                    'accuracy_preserved': False,
                    'max_prediction_diff': float('inf'),
                    'correlation_with_original': 0.0,
                    'success': False,
                    'error': 'CoreML only available on macOS'
                }

            import coremltools as ct

            model = ct.models.MLModel(model_path)

            # Run inference
            predictions = []
            for sample in self.test_samples:
                input_dict = {"input_features": sample.numpy()}
                pred = model.predict(input_dict)
                # CoreML output names vary, take first output
                pred_value = list(pred.values())[0]
                if hasattr(pred_value, 'item'):
                    predictions.append(pred_value.item())
                else:
                    predictions.append(float(pred_value))

            exported_predictions = torch.tensor(predictions)

            # Calculate metrics
            prediction_diffs = torch.abs(exported_predictions - self.original_predictions)
            max_diff = torch.max(prediction_diffs).item()

            correlation = torch.corrcoef(torch.stack([
                exported_predictions, self.original_predictions
            ]))[0, 1].item()

            accuracy_preserved = (
                max_diff < self.config.accuracy_tolerance and
                correlation > self.config.correlation_threshold
            )

            return {
                'accuracy_preserved': accuracy_preserved,
                'max_prediction_diff': max_diff,
                'correlation_with_original': correlation,
                'mean_absolute_error': torch.mean(prediction_diffs).item(),
                'success': True
            }

        except ImportError:
            return {
                'accuracy_preserved': False,
                'max_prediction_diff': float('inf'),
                'correlation_with_original': 0.0,
                'success': False,
                'error': 'CoreML tools not available'
            }
        except Exception as e:
            return {
                'accuracy_preserved': False,
                'max_prediction_diff': float('inf'),
                'correlation_with_original': 0.0,
                'success': False,
                'error': str(e)
            }


class PerformanceValidator:
    """Validate model performance across devices and batch sizes"""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.available_devices = self._get_available_devices()

        print(f"⚡ Performance Validator initialized")
        print(f"   Available devices: {self.available_devices}")

    def _get_available_devices(self) -> List[str]:
        """Get available devices for testing"""
        devices = ['cpu']

        if self.config.test_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps')

        if self.config.test_gpu and torch.cuda.is_available():
            devices.append('cuda')

        return devices

    def validate_performance(self, model_path: str, format: str) -> Dict[str, ValidationResult]:
        """Validate performance across all available devices"""
        results = {}

        for device in self.available_devices:
            print(f"   Testing {format} on {device}...")

            try:
                result = self._validate_device_performance(model_path, format, device)
                results[device] = result

                # Print quick summary
                status = "✅" if result.passed else "❌"
                print(f"     {status} {device}: {result.inference_time_ms:.1f}ms, {result.throughput_samples_per_sec:.1f} samples/sec")

            except Exception as e:
                print(f"     ❌ {device}: Failed - {e}")
                results[device] = ValidationResult(
                    format=format,
                    platform=platform.system(),
                    device=device,
                    accuracy_preserved=False,
                    max_prediction_diff=float('inf'),
                    correlation_with_original=0.0,
                    inference_time_ms=float('inf'),
                    throughput_samples_per_sec=0.0,
                    memory_usage_mb=0.0,
                    passed=False,
                    error_message=str(e)
                )

        return results

    def _validate_device_performance(self, model_path: str, format: str, device: str) -> ValidationResult:
        """Validate performance on specific device"""

        # Load model based on format
        if format.startswith('TorchScript'):
            model = torch.jit.load(model_path, map_location=device)
            return self._test_torchscript_performance(model, format, device)
        elif format == 'ONNX':
            return self._test_onnx_performance(model_path, format, device)
        elif format == 'CoreML':
            return self._test_coreml_performance(model_path, format, device)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _test_torchscript_performance(self, model, format: str, device: str) -> ValidationResult:
        """Test TorchScript model performance"""
        model.eval()

        # Create test input
        test_input = torch.randn(1, 2056).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = model(test_input)

        # Measure memory before timing
        if device == 'cuda':
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated() / (1024**2)
        else:
            memory_before = 0

        # Time inference
        start_time = time.time()

        with torch.no_grad():
            for _ in range(self.config.timing_iterations):
                output = model(test_input)
                if device == 'cuda':
                    torch.cuda.synchronize()

        total_time = time.time() - start_time

        # Calculate metrics
        inference_time_ms = (total_time / self.config.timing_iterations) * 1000
        throughput = self.config.timing_iterations / total_time

        # Measure memory after
        if device == 'cuda':
            memory_after = torch.cuda.memory_allocated() / (1024**2)
            memory_usage = memory_after - memory_before
        else:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024**2)

        # Test batch performance
        batch_performance = self._test_batch_performance(model, device)

        # Check if performance meets targets
        passed = (
            inference_time_ms < self.config.max_inference_time_ms and
            throughput > self.config.min_throughput_samples_per_sec and
            memory_usage < self.config.max_memory_usage_mb
        )

        return ValidationResult(
            format=format,
            platform=platform.system(),
            device=device,
            accuracy_preserved=True,  # Assumed for TorchScript
            max_prediction_diff=0.0,
            correlation_with_original=1.0,
            inference_time_ms=inference_time_ms,
            throughput_samples_per_sec=throughput,
            memory_usage_mb=memory_usage,
            passed=passed,
            batch_performance=batch_performance
        )

    def _test_onnx_performance(self, model_path: str, format: str, device: str) -> ValidationResult:
        """Test ONNX model performance"""
        try:
            import onnxruntime as ort

            # Set providers based on device
            if device == 'cuda':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            session = ort.InferenceSession(model_path, providers=providers)
            input_name = session.get_inputs()[0].name

            # Create test input
            test_input = np.random.randn(1, 2056).astype(np.float32)

            # Warmup
            for _ in range(self.config.warmup_iterations):
                _ = session.run(None, {input_name: test_input})

            # Time inference
            start_time = time.time()
            for _ in range(self.config.timing_iterations):
                _ = session.run(None, {input_name: test_input})

            total_time = time.time() - start_time

            # Calculate metrics
            inference_time_ms = (total_time / self.config.timing_iterations) * 1000
            throughput = self.config.timing_iterations / total_time

            # Memory usage (rough estimate)
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024**2)

            # Test batch performance
            batch_performance = self._test_onnx_batch_performance(session, input_name)

            passed = (
                inference_time_ms < self.config.max_inference_time_ms and
                throughput > self.config.min_throughput_samples_per_sec
            )

            return ValidationResult(
                format=format,
                platform=platform.system(),
                device=device,
                accuracy_preserved=True,  # Validated separately
                max_prediction_diff=0.0,
                correlation_with_original=1.0,
                inference_time_ms=inference_time_ms,
                throughput_samples_per_sec=throughput,
                memory_usage_mb=memory_usage,
                passed=passed,
                batch_performance=batch_performance
            )

        except ImportError:
            raise RuntimeError("ONNX Runtime not available")
        except Exception as e:
            raise RuntimeError(f"ONNX performance test failed: {e}")

    def _test_coreml_performance(self, model_path: str, format: str, device: str) -> ValidationResult:
        """Test CoreML model performance"""
        if platform.system() != "Darwin":
            raise RuntimeError("CoreML only available on macOS")

        try:
            import coremltools as ct

            model = ct.models.MLModel(model_path)

            # Create test input
            test_input = {"input_features": np.random.randn(2056).astype(np.float32)}

            # Warmup
            for _ in range(self.config.warmup_iterations):
                _ = model.predict(test_input)

            # Time inference
            start_time = time.time()
            for _ in range(self.config.timing_iterations):
                _ = model.predict(test_input)

            total_time = time.time() - start_time

            # Calculate metrics
            inference_time_ms = (total_time / self.config.timing_iterations) * 1000
            throughput = self.config.timing_iterations / total_time

            # Memory usage
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024**2)

            passed = (
                inference_time_ms < self.config.max_inference_time_ms and
                throughput > self.config.min_throughput_samples_per_sec
            )

            return ValidationResult(
                format=format,
                platform=platform.system(),
                device=device,
                accuracy_preserved=True,
                max_prediction_diff=0.0,
                correlation_with_original=1.0,
                inference_time_ms=inference_time_ms,
                throughput_samples_per_sec=throughput,
                memory_usage_mb=memory_usage,
                passed=passed
            )

        except ImportError:
            raise RuntimeError("CoreML tools not available")
        except Exception as e:
            raise RuntimeError(f"CoreML performance test failed: {e}")

    def _test_batch_performance(self, model, device: str) -> Dict[int, float]:
        """Test performance across different batch sizes"""
        batch_performance = {}

        for batch_size in self.config.test_batch_sizes:
            try:
                test_input = torch.randn(batch_size, 2056).to(device)

                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(test_input)

                # Time inference
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(10):
                        output = model(test_input)
                        if device == 'cuda':
                            torch.cuda.synchronize()

                total_time = time.time() - start_time
                inference_time_ms = (total_time / 10) * 1000

                batch_performance[batch_size] = inference_time_ms

            except Exception as e:
                batch_performance[batch_size] = float('inf')

        return batch_performance

    def _test_onnx_batch_performance(self, session, input_name: str) -> Dict[int, float]:
        """Test ONNX performance across batch sizes"""
        batch_performance = {}

        for batch_size in self.config.test_batch_sizes:
            try:
                test_input = np.random.randn(batch_size, 2056).astype(np.float32)

                # Warmup
                for _ in range(5):
                    _ = session.run(None, {input_name: test_input})

                # Time inference
                start_time = time.time()
                for _ in range(10):
                    _ = session.run(None, {input_name: test_input})

                total_time = time.time() - start_time
                inference_time_ms = (total_time / 10) * 1000

                batch_performance[batch_size] = inference_time_ms

            except Exception as e:
                batch_performance[batch_size] = float('inf')

        return batch_performance


class ExportValidationFramework:
    """Complete export validation framework"""

    def __init__(self, validation_config: ValidationConfig = None):
        self.config = validation_config or ValidationConfig()
        self.system_profiler = SystemProfiler()
        self.validation_results = {}

        print(f"🔍 Export Validation Framework initialized")

    def validate_all_exports(self, export_results: Dict[str, ExportResult],
                           original_model: torch.nn.Module) -> Dict[str, Any]:
        """Validate all exported models comprehensively"""

        print(f"\n🧪 Starting comprehensive export validation...")
        print(f"   Testing {len(export_results)} exported models")

        # Initialize validators
        accuracy_validator = AccuracyValidator(original_model, self.config)
        performance_validator = PerformanceValidator(self.config)

        all_results = {}

        for format_name, export_result in export_results.items():
            if not export_result.success:
                print(f"   ⏭️ Skipping {format_name} (export failed)")
                continue

            print(f"\n🔬 Validating {format_name}...")

            # Accuracy validation
            accuracy_results = accuracy_validator.validate_exported_model(
                export_result.file_path, export_result.format
            )

            # Performance validation
            performance_results = performance_validator.validate_performance(
                export_result.file_path, export_result.format
            )

            # Combine results
            all_results[format_name] = {
                'export_info': asdict(export_result),
                'accuracy_validation': accuracy_results,
                'performance_validation': {
                    device: asdict(result) for device, result in performance_results.items()
                }
            }

        # Generate comprehensive report
        validation_report = self._generate_validation_report(all_results)

        return validation_report

    def _generate_validation_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""

        report = {
            "validation_timestamp": time.time(),
            "system_info": self.system_profiler.system_info,
            "validation_config": asdict(self.config),
            "results": results,
            "summary": {
                "total_models_tested": len(results),
                "accuracy_passed": 0,
                "performance_passed": 0,
                "deployment_ready": 0
            },
            "recommendations": []
        }

        # Analyze results
        for format_name, format_results in results.items():
            accuracy = format_results.get('accuracy_validation', {})
            performance = format_results.get('performance_validation', {})

            # Count passes
            if accuracy.get('accuracy_preserved', False):
                report["summary"]["accuracy_passed"] += 1

            # Check if any device passed performance
            any_device_passed = any(
                device_result.get('passed', False)
                for device_result in performance.values()
            )
            if any_device_passed:
                report["summary"]["performance_passed"] += 1

            # Check deployment readiness
            if accuracy.get('accuracy_preserved', False) and any_device_passed:
                report["summary"]["deployment_ready"] += 1

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(results)

        # Save report
        report_path = Path("validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Validation report saved: {report_path}")
        self._print_validation_summary(report)

        return report

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        accuracy_issues = []
        performance_issues = []

        for format_name, format_results in results.items():
            accuracy = format_results.get('accuracy_validation', {})
            performance = format_results.get('performance_validation', {})

            # Check accuracy issues
            if not accuracy.get('accuracy_preserved', False):
                accuracy_issues.append(format_name)

            # Check performance issues
            slow_devices = []
            for device, device_result in performance.items():
                if not device_result.get('passed', False):
                    slow_devices.append(f"{format_name}/{device}")

            if slow_devices:
                performance_issues.extend(slow_devices)

        # Generate specific recommendations
        if accuracy_issues:
            recommendations.append(f"Accuracy issues detected in: {', '.join(accuracy_issues)}. Consider adjusting export parameters or model architecture.")

        if performance_issues:
            recommendations.append(f"Performance issues detected in: {', '.join(performance_issues)}. Consider model optimization or quantization.")

        # Platform-specific recommendations
        if self.system_profiler.system_info['mps_available']:
            recommendations.append("Apple Silicon MPS available - CoreML export recommended for optimal performance.")

        if self.system_profiler.system_info['cuda_available']:
            recommendations.append("CUDA available - ONNX with GPU provider recommended for high-throughput scenarios.")

        return recommendations

    def _print_validation_summary(self, report: Dict[str, Any]):
        """Print validation summary"""
        print(f"\n📋 Validation Summary")
        print(f"=" * 50)

        summary = report["summary"]
        print(f"✅ Models tested: {summary['total_models_tested']}")
        print(f"🎯 Accuracy passed: {summary['accuracy_passed']}/{summary['total_models_tested']}")
        print(f"⚡ Performance passed: {summary['performance_passed']}/{summary['total_models_tested']}")
        print(f"🚀 Deployment ready: {summary['deployment_ready']}/{summary['total_models_tested']}")

        # Performance details
        print(f"\n⚡ Performance Details:")
        for format_name, format_results in report["results"].items():
            performance = format_results.get('performance_validation', {})
            for device, device_result in performance.items():
                status = "✅" if device_result.get('passed', False) else "❌"
                inference_time = device_result.get('inference_time_ms', 0)
                throughput = device_result.get('throughput_samples_per_sec', 0)
                print(f"   {status} {format_name}/{device}: {inference_time:.1f}ms, {throughput:.1f} samples/sec")

        # Recommendations
        if report["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"   • {rec}")


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Export Validation Framework")

    # Create validation config
    val_config = ValidationConfig(
        max_inference_time_ms=50.0,
        test_samples=50
    )

    # Create framework
    framework = ExportValidationFramework(val_config)

    print("✅ Validation framework test completed")