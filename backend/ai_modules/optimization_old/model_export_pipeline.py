"""
Multi-Format Model Export Pipeline for SVG Quality Prediction
Implements Task 12.2.1: Multi-Format Model Export Pipeline

Supports export to:
- TorchScript (cross-platform PyTorch deployment)
- ONNX (broad framework compatibility)
- CoreML (Apple Silicon optimization)
"""

import torch
import torch.jit
import numpy as np
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import tempfile
import platform

# Import model architecture
from .gpu_model_architecture import QualityPredictorGPU, ColabTrainingConfig

warnings.filterwarnings('ignore')


@dataclass
class ExportConfig:
    """Configuration for model export pipeline"""
    export_dir: str = "exported_models"
    model_name: str = "svg_quality_predictor"

    # Export format settings
    export_torchscript: bool = True
    export_onnx: bool = True
    export_coreml: bool = True

    # Optimization settings
    optimize_for_mobile: bool = False
    quantize_model: bool = False

    # Target constraints
    max_model_size_mb: float = 100.0
    target_inference_time_ms: float = 50.0

    # ONNX specific
    onnx_opset_version: int = 11
    onnx_optimize: bool = True

    # CoreML specific
    coreml_minimum_deployment_target: str = "13.0"
    coreml_compute_units: str = "ALL"  # ALL, CPU_ONLY, CPU_AND_GPU


@dataclass
class ExportResult:
    """Result of model export operation"""
    format: str
    file_path: str
    file_size_mb: float
    export_time_ms: float
    success: bool
    error_message: Optional[str] = None
    validation_passed: bool = False
    inference_time_ms: Optional[float] = None


class ModelExportPipeline:
    """Complete model export pipeline with multi-format support"""

    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
        self.export_results: List[ExportResult] = []

        # Create export directory
        self.export_dir = Path(self.config.export_dir)
        self.export_dir.mkdir(exist_ok=True)

        print(f"🚀 Model Export Pipeline initialized")
        print(f"   Export directory: {self.export_dir}")
        print(f"   Target formats: {self._get_enabled_formats()}")

    def _get_enabled_formats(self) -> List[str]:
        """Get list of enabled export formats"""
        formats = []
        if self.config.export_torchscript:
            formats.append("TorchScript")
        if self.config.export_onnx:
            formats.append("ONNX")
        if self.config.export_coreml:
            formats.append("CoreML")
        return formats

    def export_all_formats(self, model: QualityPredictorGPU,
                          sample_input: Optional[torch.Tensor] = None) -> Dict[str, ExportResult]:
        """Export model to all enabled formats"""

        print(f"\n🔄 Starting multi-format model export...")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Prepare model for export
        model.eval()
        model = model.cpu()  # Ensure CPU for export

        # Create sample input if not provided
        if sample_input is None:
            sample_input = torch.randn(1, 2056)

        results = {}

        # Export to TorchScript
        if self.config.export_torchscript:
            print("\n📦 Exporting to TorchScript...")
            results['torchscript_traced'] = self._export_torchscript_traced(model, sample_input)
            results['torchscript_scripted'] = self._export_torchscript_scripted(model)

        # Export to ONNX
        if self.config.export_onnx:
            print("\n📦 Exporting to ONNX...")
            results['onnx'] = self._export_onnx(model, sample_input)

        # Export to CoreML
        if self.config.export_coreml:
            print("\n📦 Exporting to CoreML...")
            results['coreml'] = self._export_coreml(model, sample_input)

        # Store results
        self.export_results = list(results.values())

        # Generate summary report
        self._generate_export_report(results)

        return results

    def _export_torchscript_traced(self, model: torch.nn.Module,
                                  sample_input: torch.Tensor) -> ExportResult:
        """Export model to TorchScript using tracing"""
        start_time = time.time()
        export_path = self.export_dir / f"{self.config.model_name}_traced.pt"

        try:
            # Trace the model
            traced_model = torch.jit.trace(model, sample_input)

            # Optimize if requested
            if self.config.optimize_for_mobile:
                traced_model = torch.jit.optimize_for_inference(traced_model)

            # Save traced model
            torch.jit.save(traced_model, str(export_path))

            # Calculate metrics
            export_time = (time.time() - start_time) * 1000
            file_size = os.path.getsize(export_path) / (1024 * 1024)

            # Test inference
            inference_time = self._test_torchscript_inference(traced_model, sample_input)

            result = ExportResult(
                format="TorchScript_Traced",
                file_path=str(export_path),
                file_size_mb=file_size,
                export_time_ms=export_time,
                success=True,
                inference_time_ms=inference_time
            )

            print(f"   ✅ TorchScript traced: {file_size:.1f}MB, {export_time:.1f}ms")

        except Exception as e:
            result = ExportResult(
                format="TorchScript_Traced",
                file_path=str(export_path),
                file_size_mb=0.0,
                export_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e)
            )
            print(f"   ❌ TorchScript traced failed: {e}")

        return result

    def _export_torchscript_scripted(self, model: torch.nn.Module) -> ExportResult:
        """Export model to TorchScript using scripting"""
        start_time = time.time()
        export_path = self.export_dir / f"{self.config.model_name}_scripted.pt"

        try:
            # Script the model
            scripted_model = torch.jit.script(model)

            # Optimize if requested
            if self.config.optimize_for_mobile:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)

            # Save scripted model
            torch.jit.save(scripted_model, str(export_path))

            # Calculate metrics
            export_time = (time.time() - start_time) * 1000
            file_size = os.path.getsize(export_path) / (1024 * 1024)

            # Test inference
            sample_input = torch.randn(1, 2056)
            inference_time = self._test_torchscript_inference(scripted_model, sample_input)

            result = ExportResult(
                format="TorchScript_Scripted",
                file_path=str(export_path),
                file_size_mb=file_size,
                export_time_ms=export_time,
                success=True,
                inference_time_ms=inference_time
            )

            print(f"   ✅ TorchScript scripted: {file_size:.1f}MB, {export_time:.1f}ms")

        except Exception as e:
            result = ExportResult(
                format="TorchScript_Scripted",
                file_path=str(export_path),
                file_size_mb=0.0,
                export_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e)
            )
            print(f"   ❌ TorchScript scripted failed: {e}")

        return result

    def _export_onnx(self, model: torch.nn.Module, sample_input: torch.Tensor) -> ExportResult:
        """Export model to ONNX format"""
        start_time = time.time()
        export_path = self.export_dir / f"{self.config.model_name}.onnx"

        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                sample_input,
                str(export_path),
                export_params=True,
                opset_version=self.config.onnx_opset_version,
                do_constant_folding=True,
                input_names=['input_features'],
                output_names=['quality_prediction'],
                dynamic_axes={
                    'input_features': {0: 'batch_size'},
                    'quality_prediction': {0: 'batch_size'}
                }
            )

            # Calculate metrics
            export_time = (time.time() - start_time) * 1000
            file_size = os.path.getsize(export_path) / (1024 * 1024)

            # Test inference if ONNX Runtime available
            inference_time = self._test_onnx_inference(str(export_path), sample_input)

            result = ExportResult(
                format="ONNX",
                file_path=str(export_path),
                file_size_mb=file_size,
                export_time_ms=export_time,
                success=True,
                inference_time_ms=inference_time
            )

            print(f"   ✅ ONNX export: {file_size:.1f}MB, {export_time:.1f}ms")

        except Exception as e:
            result = ExportResult(
                format="ONNX",
                file_path=str(export_path),
                file_size_mb=0.0,
                export_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e)
            )
            print(f"   ❌ ONNX export failed: {e}")

        return result

    def _export_coreml(self, model: torch.nn.Module, sample_input: torch.Tensor) -> ExportResult:
        """Export model to CoreML format"""
        start_time = time.time()
        export_path = self.export_dir / f"{self.config.model_name}.mlmodel"

        try:
            # Check if CoreML tools are available
            try:
                import coremltools as ct
            except ImportError:
                raise ImportError("coremltools not available. Install with: pip install coremltools")

            # Convert to CoreML
            traced_model = torch.jit.trace(model, sample_input)

            # Create CoreML model
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input_features", shape=sample_input.shape)],
                minimum_deployment_target=ct.target.iOS13 if self.config.coreml_minimum_deployment_target == "13.0" else ct.target.iOS14,
                compute_units=getattr(ct.ComputeUnit, self.config.coreml_compute_units, ct.ComputeUnit.ALL)
            )

            # Set metadata
            coreml_model.short_description = "SVG Quality Predictor"
            coreml_model.input_description["input_features"] = "Combined image features and VTracer parameters"
            coreml_model.output_description["var_8"] = "Predicted SSIM quality score"

            # Save CoreML model
            coreml_model.save(str(export_path))

            # Calculate metrics
            export_time = (time.time() - start_time) * 1000
            file_size = os.path.getsize(export_path) / (1024 * 1024)

            # Test inference
            inference_time = self._test_coreml_inference(coreml_model, sample_input)

            result = ExportResult(
                format="CoreML",
                file_path=str(export_path),
                file_size_mb=file_size,
                export_time_ms=export_time,
                success=True,
                inference_time_ms=inference_time
            )

            print(f"   ✅ CoreML export: {file_size:.1f}MB, {export_time:.1f}ms")

        except Exception as e:
            result = ExportResult(
                format="CoreML",
                file_path=str(export_path),
                file_size_mb=0.0,
                export_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e)
            )
            print(f"   ❌ CoreML export failed: {e}")

        return result

    def _test_torchscript_inference(self, model, sample_input: torch.Tensor) -> Optional[float]:
        """Test TorchScript model inference speed"""
        try:
            model.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(sample_input)

            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(sample_input)

            inference_time = ((time.time() - start_time) / 10) * 1000
            return inference_time

        except Exception as e:
            print(f"   ⚠️ TorchScript inference test failed: {e}")
            return None

    def _test_onnx_inference(self, model_path: str, sample_input: torch.Tensor) -> Optional[float]:
        """Test ONNX model inference speed"""
        try:
            import onnxruntime as ort

            # Create inference session
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name

            sample_np = sample_input.numpy()

            # Warmup
            for _ in range(5):
                _ = session.run(None, {input_name: sample_np})

            # Measure inference time
            start_time = time.time()
            for _ in range(10):
                _ = session.run(None, {input_name: sample_np})

            inference_time = ((time.time() - start_time) / 10) * 1000
            return inference_time

        except ImportError:
            print(f"   ⚠️ ONNX Runtime not available for inference testing")
            return None
        except Exception as e:
            print(f"   ⚠️ ONNX inference test failed: {e}")
            return None

    def _test_coreml_inference(self, model, sample_input: torch.Tensor) -> Optional[float]:
        """Test CoreML model inference speed"""
        try:
            # Only test on macOS
            if platform.system() != "Darwin":
                print(f"   ⚠️ CoreML inference testing only available on macOS")
                return None

            sample_dict = {"input_features": sample_input.numpy()}

            # Warmup
            for _ in range(5):
                _ = model.predict(sample_dict)

            # Measure inference time
            start_time = time.time()
            for _ in range(10):
                _ = model.predict(sample_dict)

            inference_time = ((time.time() - start_time) / 10) * 1000
            return inference_time

        except Exception as e:
            print(f"   ⚠️ CoreML inference test failed: {e}")
            return None

    def _generate_export_report(self, results: Dict[str, ExportResult]):
        """Generate comprehensive export report"""
        report = {
            "export_timestamp": time.time(),
            "export_config": asdict(self.config),
            "system_info": {
                "platform": platform.system(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__
            },
            "export_results": {},
            "performance_summary": {
                "successful_exports": 0,
                "failed_exports": 0,
                "total_size_mb": 0.0,
                "fastest_inference_ms": None,
                "slowest_inference_ms": None
            }
        }

        inference_times = []

        for format_name, result in results.items():
            report["export_results"][format_name] = asdict(result)

            if result.success:
                report["performance_summary"]["successful_exports"] += 1
                report["performance_summary"]["total_size_mb"] += result.file_size_mb

                if result.inference_time_ms:
                    inference_times.append(result.inference_time_ms)
            else:
                report["performance_summary"]["failed_exports"] += 1

        # Calculate inference time statistics
        if inference_times:
            report["performance_summary"]["fastest_inference_ms"] = min(inference_times)
            report["performance_summary"]["slowest_inference_ms"] = max(inference_times)
            report["performance_summary"]["average_inference_ms"] = sum(inference_times) / len(inference_times)

        # Performance target analysis
        report["target_analysis"] = self._analyze_performance_targets(results)

        # Save report
        report_path = self.export_dir / f"export_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Export report saved: {report_path}")
        self._print_export_summary(report)

        return report

    def _analyze_performance_targets(self, results: Dict[str, ExportResult]) -> Dict[str, Any]:
        """Analyze performance against targets"""
        analysis = {
            "size_target_met": True,
            "inference_target_met": True,
            "size_violations": [],
            "inference_violations": [],
            "recommended_actions": []
        }

        for format_name, result in results.items():
            if not result.success:
                continue

            # Check size target
            if result.file_size_mb > self.config.max_model_size_mb:
                analysis["size_target_met"] = False
                analysis["size_violations"].append({
                    "format": format_name,
                    "actual_mb": result.file_size_mb,
                    "target_mb": self.config.max_model_size_mb,
                    "excess_mb": result.file_size_mb - self.config.max_model_size_mb
                })

            # Check inference time target
            if result.inference_time_ms and result.inference_time_ms > self.config.target_inference_time_ms:
                analysis["inference_target_met"] = False
                analysis["inference_violations"].append({
                    "format": format_name,
                    "actual_ms": result.inference_time_ms,
                    "target_ms": self.config.target_inference_time_ms,
                    "excess_ms": result.inference_time_ms - self.config.target_inference_time_ms
                })

        # Generate recommendations
        if analysis["size_violations"]:
            analysis["recommended_actions"].append("Consider model quantization to reduce size")
            analysis["recommended_actions"].append("Enable mobile optimization for TorchScript")

        if analysis["inference_violations"]:
            analysis["recommended_actions"].append("Optimize model architecture for faster inference")
            analysis["recommended_actions"].append("Consider using mobile-optimized export options")

        return analysis

    def _print_export_summary(self, report: Dict[str, Any]):
        """Print export summary to console"""
        print(f"\n📋 Export Summary")
        print(f"=" * 50)

        perf = report["performance_summary"]
        print(f"✅ Successful exports: {perf['successful_exports']}")
        print(f"❌ Failed exports: {perf['failed_exports']}")
        print(f"💾 Total model size: {perf['total_size_mb']:.1f}MB")

        if perf.get('average_inference_ms'):
            print(f"⚡ Average inference: {perf['average_inference_ms']:.1f}ms")

        # Target analysis
        targets = report["target_analysis"]
        size_status = "✅" if targets["size_target_met"] else "❌"
        inference_status = "✅" if targets["inference_target_met"] else "❌"

        print(f"\n🎯 Performance Targets:")
        print(f"   {size_status} Size limit ({self.config.max_model_size_mb}MB): {'PASSED' if targets['size_target_met'] else 'FAILED'}")
        print(f"   {inference_status} Inference time ({self.config.target_inference_time_ms}ms): {'PASSED' if targets['inference_target_met'] else 'FAILED'}")

        # Recommendations
        if targets["recommended_actions"]:
            print(f"\n💡 Recommendations:")
            for action in targets["recommended_actions"]:
                print(f"   • {action}")


def load_trained_model(checkpoint_path: str) -> Tuple[QualityPredictorGPU, Dict[str, Any]]:
    """Load trained model from checkpoint"""
    print(f"📂 Loading trained model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract configuration
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = ColabTrainingConfig(**config_dict)
    else:
        # Fallback configuration
        config = ColabTrainingConfig(device='cpu')

    # Create and load model
    model = QualityPredictorGPU(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Extract metadata
    metadata = {
        'training_correlation': checkpoint.get('val_correlation', 0.0),
        'training_loss': checkpoint.get('val_loss', 0.0),
        'epoch': checkpoint.get('epoch', 0),
        'config': config_dict if 'config' in checkpoint else {}
    }

    print(f"✅ Model loaded:")
    print(f"   Training correlation: {metadata.get('training_correlation', 0.0):.4f}")
    print(f"   Training epoch: {metadata.get('epoch', 0)}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, metadata


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Model Export Pipeline")

    # Create export configuration
    export_config = ExportConfig(
        export_dir="test_exports",
        target_inference_time_ms=50.0,
        max_model_size_mb=100.0
    )

    # Create pipeline
    pipeline = ModelExportPipeline(export_config)

    # Create dummy model for testing
    config = ColabTrainingConfig(device='cpu')
    model = QualityPredictorGPU(config)

    # Test export
    results = pipeline.export_all_formats(model)

    print(f"\n✅ Export pipeline test completed")
    print(f"   Exported {len(results)} formats")