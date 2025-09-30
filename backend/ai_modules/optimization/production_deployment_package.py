"""
Production Deployment Package Creation System
Implements Task 12.2.3: Production Deployment Package Creation

Creates complete deployment packages with:
- All validated model formats
- Local integration interfaces
- Performance optimization utilities
- Deployment configuration
- Documentation and examples
"""

import torch
import numpy as np
import json
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import time
import platform
import os
import warnings

# Import related modules
from .model_export_pipeline import ExportResult, ExportConfig, load_trained_model
from .export_validation_framework import ValidationResult, ExportValidationFramework
from .local_inference_optimizer import LocalInferenceOptimizer, InferenceResult

warnings.filterwarnings('ignore')


@dataclass
class DeploymentConfig:
    """Configuration for deployment package creation"""
    package_name: str = "svg_quality_predictor_deployment"
    version: str = "1.0.0"
    target_platforms: List[str] = None

    # Package contents
    include_all_formats: bool = True
    include_validation_results: bool = True
    include_optimization_results: bool = True
    include_examples: bool = True
    include_documentation: bool = True

    # Optimization settings
    optimize_for_size: bool = True
    optimize_for_speed: bool = True
    include_quantized_models: bool = False

    # Documentation settings
    generate_api_docs: bool = True
    include_performance_benchmarks: bool = True
    include_integration_examples: bool = True

    def __post_init__(self):
        if self.target_platforms is None:
            self.target_platforms = ["linux", "macos", "windows"]


@dataclass
class PackageManifest:
    """Manifest describing deployment package contents"""
    package_name: str
    version: str
    creation_timestamp: float
    target_platforms: List[str]

    # Model information
    models: Dict[str, Any]
    validation_results: Dict[str, Any]
    performance_benchmarks: Dict[str, Any]

    # Package structure
    directories: Dict[str, str]
    files: Dict[str, str]

    # Requirements
    python_requirements: List[str]
    system_requirements: Dict[str, Any]

    # Integration information
    api_interfaces: List[str]
    example_scripts: List[str]


class ModelPackager:
    """Package models with optimization and validation results"""

    def __init__(self, deployment_config: Optional[DeploymentConfig] = None):
        self.config = deployment_config or DeploymentConfig(
            package_name="ai_conversion_models",
            version="1.0.0"
        )
        self.package_dir = Path(self.config.package_name)

        print(f"📦 Model Packager initialized")
        print(f"   Package: {self.config.package_name} v{self.config.version}")

    def create_model_package(self, export_results: Dict[str, ExportResult],
                           validation_results: Dict[str, Any],
                           optimization_results: Dict[str, Any]) -> Path:
        """Create complete model deployment package"""

        print(f"\n🏗️ Creating deployment package...")

        # Create package directory structure
        self._create_package_structure()

        # Package models
        self._package_models(export_results)

        # Package validation results
        if self.config.include_validation_results:
            self._package_validation_results(validation_results)

        # Package optimization results
        if self.config.include_optimization_results:
            self._package_optimization_results(optimization_results)

        # Create API interfaces
        self._create_api_interfaces()

        # Generate documentation
        if self.config.include_documentation:
            self._generate_documentation()

        # Create examples
        if self.config.include_examples:
            self._create_examples()

        # Generate requirements
        self._generate_requirements()

        # Create manifest
        manifest = self._create_manifest(export_results, validation_results, optimization_results)

        # Package everything
        package_path = self._create_final_package()

        print(f"✅ Deployment package created: {package_path}")
        return package_path

    def _create_package_structure(self):
        """Create deployment package directory structure"""
        # Clean and create main directory
        if self.package_dir.exists():
            shutil.rmtree(self.package_dir)

        self.package_dir.mkdir(parents=True)

        # Create subdirectories
        subdirs = [
            "models",
            "validation",
            "optimization",
            "api",
            "examples",
            "docs",
            "utils",
            "tests",
            "config"
        ]

        for subdir in subdirs:
            (self.package_dir / subdir).mkdir()

        print(f"   📁 Package structure created")

    def _package_models(self, export_results: Dict[str, ExportResult]):
        """Package all exported models"""
        print(f"   📋 Packaging models...")

        models_dir = self.package_dir / "models"

        model_info = {}

        for format_name, result in export_results.items():
            if not result.success:
                continue

            # Copy model file
            src_path = Path(result.file_path)
            if src_path.exists():
                dst_path = models_dir / src_path.name
                shutil.copy2(src_path, dst_path)

                model_info[format_name] = {
                    "file_path": str(dst_path.relative_to(self.package_dir)),
                    "format": result.format,
                    "file_size_mb": result.file_size_mb,
                    "export_time_ms": result.export_time_ms,
                    "inference_time_ms": result.inference_time_ms,
                    "validated": result.validation_passed
                }

                print(f"     ✅ {format_name}: {result.file_size_mb:.1f}MB")

        # Save model metadata
        with open(models_dir / "models_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)

    def _package_validation_results(self, validation_results: Dict[str, Any]):
        """Package validation results and reports"""
        print(f"   🔍 Packaging validation results...")

        validation_dir = self.package_dir / "validation"

        # Save complete validation results
        with open(validation_dir / "validation_results.json", 'w') as f:
            json.dump(validation_results, f, indent=2)

        # Create validation summary
        summary = self._create_validation_summary(validation_results)
        with open(validation_dir / "validation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

    def _package_optimization_results(self, optimization_results: Dict[str, Any]):
        """Package optimization results and benchmarks"""
        print(f"   ⚡ Packaging optimization results...")

        optimization_dir = self.package_dir / "optimization"

        # Save complete optimization results
        with open(optimization_dir / "optimization_results.json", 'w') as f:
            json.dump(optimization_results, f, indent=2)

        # Create performance benchmarks
        benchmarks = self._create_performance_benchmarks(optimization_results)
        with open(optimization_dir / "performance_benchmarks.json", 'w') as f:
            json.dump(benchmarks, f, indent=2)

    def _create_api_interfaces(self):
        """Create API interface classes for easy integration"""
        print(f"   🔌 Creating API interfaces...")

        api_dir = self.package_dir / "api"

        # Create base predictor interface
        self._create_base_predictor_interface(api_dir)

        # Create format-specific interfaces
        self._create_torchscript_interface(api_dir)
        self._create_onnx_interface(api_dir)
        self._create_coreml_interface(api_dir)

        # Create high-level predictor
        self._create_unified_predictor(api_dir)

    def _create_base_predictor_interface(self, api_dir: Path):
        """Create base predictor interface"""
        interface_code = '''"""
Base Predictor Interface for SVG Quality Prediction Models
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union


class BaseQualityPredictor(ABC):
    """Abstract base class for quality predictors"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = 'cpu'
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the model from file"""
        pass

    @abstractmethod
    def predict_quality(self, image_features: np.ndarray,
                       vtracer_params: Dict[str, float]) -> float:
        """Predict SSIM quality for given features and parameters"""
        pass

    def prepare_input(self, image_features: np.ndarray,
                     vtracer_params: Dict[str, float]) -> np.ndarray:
        """Prepare combined input features"""
        # Normalize VTracer parameters
        normalized_params = [
            vtracer_params.get('color_precision', 6.0) / 10.0,
            vtracer_params.get('corner_threshold', 60.0) / 100.0,
            vtracer_params.get('length_threshold', 4.0) / 10.0,
            vtracer_params.get('max_iterations', 10) / 20.0,
            vtracer_params.get('splice_threshold', 45.0) / 100.0,
            vtracer_params.get('path_precision', 8) / 16.0,
            vtracer_params.get('layer_difference', 16.0) / 32.0,
            vtracer_params.get('mode', 0) / 1.0
        ]

        # Combine features
        combined_features = np.concatenate([image_features, normalized_params])
        return combined_features.astype(np.float32)

    def predict_batch(self, features_list: List[np.ndarray],
                     params_list: List[Dict[str, float]]) -> List[float]:
        """Predict quality for batch of inputs"""
        predictions = []
        for features, params in zip(features_list, params_list):
            pred = self.predict_quality(features, params)
            predictions.append(pred)
        return predictions
'''

        with open(api_dir / "base_predictor.py", 'w') as f:
            f.write(interface_code)

    def _create_torchscript_interface(self, api_dir: Path):
        """Create TorchScript predictor interface"""
        interface_code = '''"""
TorchScript Quality Predictor Interface
"""

import torch
import numpy as np
from typing import Dict
from .base_predictor import BaseQualityPredictor


class TorchScriptPredictor(BaseQualityPredictor):
    """TorchScript model predictor"""

    def load_model(self):
        """Load TorchScript model"""
        try:
            self.model = torch.jit.load(self.model_path, map_location='cpu')
            self.model.eval()
            print(f"✅ TorchScript model loaded: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load TorchScript model: {e}")

    def predict_quality(self, image_features: np.ndarray,
                       vtracer_params: Dict[str, float]) -> float:
        """Predict SSIM quality using TorchScript model"""
        # Prepare input
        combined_features = self.prepare_input(image_features, vtracer_params)
        input_tensor = torch.FloatTensor(combined_features).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            return float(output.squeeze().item())

    def set_device(self, device: str):
        """Set device for inference"""
        self.device = device
        if device != 'cpu':
            self.model = self.model.to(device)
'''

        with open(api_dir / "torchscript_predictor.py", 'w') as f:
            f.write(interface_code)

    def _create_onnx_interface(self, api_dir: Path):
        """Create ONNX predictor interface"""
        interface_code = '''"""
ONNX Quality Predictor Interface
"""

import numpy as np
from typing import Dict
from .base_predictor import BaseQualityPredictor


class ONNXPredictor(BaseQualityPredictor):
    """ONNX model predictor"""

    def load_model(self):
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            print(f"✅ ONNX model loaded: {self.model_path}")
        except ImportError:
            raise RuntimeError("ONNX Runtime not available. Install with: pip install onnxruntime")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def predict_quality(self, image_features: np.ndarray,
                       vtracer_params: Dict[str, float]) -> float:
        """Predict SSIM quality using ONNX model"""
        # Prepare input
        combined_features = self.prepare_input(image_features, vtracer_params)
        input_data = combined_features.reshape(1, -1)

        # Predict
        outputs = self.session.run(None, {self.input_name: input_data})
        return float(outputs[0][0])

    def set_providers(self, providers: list):
        """Set ONNX execution providers"""
        # Recreate session with new providers
        import onnxruntime as ort
        self.session = ort.InferenceSession(self.model_path, providers=providers)
'''

        with open(api_dir / "onnx_predictor.py", 'w') as f:
            f.write(interface_code)

    def _create_coreml_interface(self, api_dir: Path):
        """Create CoreML predictor interface"""
        interface_code = '''"""
CoreML Quality Predictor Interface
"""

import numpy as np
import platform
from typing import Dict
from .base_predictor import BaseQualityPredictor


class CoreMLPredictor(BaseQualityPredictor):
    """CoreML model predictor (macOS only)"""

    def load_model(self):
        """Load CoreML model"""
        if platform.system() != "Darwin":
            raise RuntimeError("CoreML is only available on macOS")

        try:
            import coremltools as ct
            self.model = ct.models.MLModel(self.model_path)
            print(f"✅ CoreML model loaded: {self.model_path}")
        except ImportError:
            raise RuntimeError("CoreML tools not available. Install with: pip install coremltools")
        except Exception as e:
            raise RuntimeError(f"Failed to load CoreML model: {e}")

    def predict_quality(self, image_features: np.ndarray,
                       vtracer_params: Dict[str, float]) -> float:
        """Predict SSIM quality using CoreML model"""
        # Prepare input
        combined_features = self.prepare_input(image_features, vtracer_params)
        input_dict = {"input_features": combined_features}

        # Predict
        output = self.model.predict(input_dict)
        # CoreML output names vary, take first output value
        prediction = list(output.values())[0]
        if hasattr(prediction, 'item'):
            return float(prediction.item())
        else:
            return float(prediction)
'''

        with open(api_dir / "coreml_predictor.py", 'w') as f:
            f.write(interface_code)

    def _create_unified_predictor(self, api_dir: Path):
        """Create unified predictor that auto-selects best model"""
        interface_code = '''"""
Unified Quality Predictor
Auto-selects best available model format for optimal performance
"""

import platform
import os
from pathlib import Path
from typing import Dict, Optional
from .torchscript_predictor import TorchScriptPredictor
from .onnx_predictor import ONNXPredictor
from .coreml_predictor import CoreMLPredictor


class UnifiedQualityPredictor:
    """Unified predictor that auto-selects optimal model format"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.predictor = None
        self.format_used = None

        self._select_optimal_predictor()

    def _select_optimal_predictor(self):
        """Select optimal predictor based on platform and available models"""

        # Priority order based on platform
        if platform.system() == "Darwin":  # macOS
            priority = ["coreml", "torchscript_traced", "onnx"]
        else:
            priority = ["torchscript_traced", "onnx", "coreml"]

        for format_name in priority:
            try:
                model_path = self._find_model_file(format_name)
                if model_path:
                    predictor = self._create_predictor(format_name, model_path)
                    if predictor:
                        self.predictor = predictor
                        self.format_used = format_name
                        print(f"🚀 Using {format_name} predictor: {model_path}")
                        return
            except Exception as e:
                print(f"⚠️ Failed to load {format_name}: {e}")
                continue

        raise RuntimeError("No compatible model format found")

    def _find_model_file(self, format_name: str) -> Optional[str]:
        """Find model file for given format"""
        extensions = {
            "torchscript_traced": [".pt"],
            "torchscript_scripted": [".pt"],
            "onnx": [".onnx"],
            "coreml": [".mlmodel"]
        }

        for ext in extensions.get(format_name, []):
            candidates = list(self.models_dir.glob(f"*{ext}"))
            if candidates:
                return str(candidates[0])

        return None

    def _create_predictor(self, format_name: str, model_path: str):
        """Create predictor for given format"""
        try:
            if format_name.startswith("torchscript"):
                return TorchScriptPredictor(model_path)
            elif format_name == "onnx":
                return ONNXPredictor(model_path)
            elif format_name == "coreml":
                return CoreMLPredictor(model_path)
        except Exception:
            return None

    def predict_quality(self, image_features, vtracer_params: Dict[str, float]) -> float:
        """Predict quality using selected model"""
        if not self.predictor:
            raise RuntimeError("No predictor available")

        return self.predictor.predict_quality(image_features, vtracer_params)

    def predict_batch(self, features_list, params_list) -> list:
        """Predict quality for batch of inputs"""
        if not self.predictor:
            raise RuntimeError("No predictor available")

        return self.predictor.predict_batch(features_list, params_list)

    def get_model_info(self) -> Dict[str, str]:
        """Get information about the selected model"""
        return {
            "format": self.format_used,
            "model_path": self.predictor.model_path if self.predictor else None,
            "platform": platform.system()
        }
'''

        with open(api_dir / "unified_predictor.py", 'w') as f:
            f.write(interface_code)

        # Create __init__.py for API package
        init_code = '''"""
SVG Quality Predictor API Package
"""

from .unified_predictor import UnifiedQualityPredictor
from .torchscript_predictor import TorchScriptPredictor
from .onnx_predictor import ONNXPredictor
from .coreml_predictor import CoreMLPredictor

__all__ = [
    'UnifiedQualityPredictor',
    'TorchScriptPredictor',
    'ONNXPredictor',
    'CoreMLPredictor'
]
'''

        with open(api_dir / "__init__.py", 'w') as f:
            f.write(init_code)

    def _create_examples(self):
        """Create usage examples"""
        print(f"   📚 Creating examples...")

        examples_dir = self.package_dir / "examples"

        # Basic usage example
        self._create_basic_example(examples_dir)

        # Batch processing example
        self._create_batch_example(examples_dir)

        # Integration example
        self._create_integration_example(examples_dir)

        # Performance testing example
        self._create_performance_example(examples_dir)

    def _create_basic_example(self, examples_dir: Path):
        """Create basic usage example"""
        example_code = '''"""
Basic SVG Quality Predictor Usage Example
"""

import numpy as np
import sys
from pathlib import Path

# Add API to path
sys.path.append(str(Path(__file__).parent.parent / "api"))

from unified_predictor import UnifiedQualityPredictor


def main():
    """Basic usage demonstration"""

    # Initialize predictor (auto-selects best model)
    predictor = UnifiedQualityPredictor("../models")

    print(f"Using model: {predictor.get_model_info()}")

    # Example image features (2048 dims from ResNet-50)
    image_features = np.random.randn(2048) * 0.5

    # Example VTracer parameters
    vtracer_params = {
        'color_precision': 6.0,
        'corner_threshold': 60.0,
        'length_threshold': 4.0,
        'max_iterations': 10,
        'splice_threshold': 45.0,
        'path_precision': 8,
        'layer_difference': 16.0,
        'mode': 0
    }

    # Predict quality
    predicted_ssim = predictor.predict_quality(image_features, vtracer_params)

    print(f"Predicted SSIM quality: {predicted_ssim:.4f}")

    # Test with different parameters
    optimized_params = vtracer_params.copy()
    optimized_params['color_precision'] = 4.0
    optimized_params['corner_threshold'] = 30.0

    optimized_ssim = predictor.predict_quality(image_features, optimized_params)

    print(f"Optimized SSIM quality: {optimized_ssim:.4f}")
    print(f"Improvement: {optimized_ssim - predicted_ssim:.4f}")


if __name__ == "__main__":
    main()
'''

        with open(examples_dir / "basic_usage.py", 'w') as f:
            f.write(example_code)

    def _create_batch_example(self, examples_dir: Path):
        """Create batch processing example"""
        example_code = '''"""
Batch Processing Example
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add API to path
sys.path.append(str(Path(__file__).parent.parent / "api"))

from unified_predictor import UnifiedQualityPredictor


def main():
    """Batch processing demonstration"""

    # Initialize predictor
    predictor = UnifiedQualityPredictor("../models")

    print(f"Using model: {predictor.get_model_info()}")

    # Generate batch of test data
    batch_size = 100

    features_list = []
    params_list = []

    for i in range(batch_size):
        # Random image features
        features = np.random.randn(2048) * 0.5
        features_list.append(features)

        # Random parameters within valid ranges
        params = {
            'color_precision': np.random.uniform(2, 10),
            'corner_threshold': np.random.uniform(10, 100),
            'length_threshold': np.random.uniform(1, 10),
            'max_iterations': int(np.random.uniform(5, 20)),
            'splice_threshold': np.random.uniform(20, 80),
            'path_precision': int(np.random.uniform(4, 16)),
            'layer_difference': np.random.uniform(8, 32),
            'mode': int(np.random.uniform(0, 2))
        }
        params_list.append(params)

    # Time batch prediction
    start_time = time.time()
    predictions = predictor.predict_batch(features_list, params_list)
    batch_time = time.time() - start_time

    # Calculate statistics
    predictions = np.array(predictions)

    print(f"\\nBatch Results:")
    print(f"  Samples processed: {batch_size}")
    print(f"  Total time: {batch_time:.3f}s")
    print(f"  Time per sample: {(batch_time / batch_size) * 1000:.1f}ms")
    print(f"  Throughput: {batch_size / batch_time:.1f} samples/sec")

    print(f"\\nPrediction Statistics:")
    print(f"  Mean SSIM: {predictions.mean():.4f}")
    print(f"  Std SSIM: {predictions.std():.4f}")
    print(f"  Min SSIM: {predictions.min():.4f}")
    print(f"  Max SSIM: {predictions.max():.4f}")


if __name__ == "__main__":
    main()
'''

        with open(examples_dir / "batch_processing.py", 'w') as f:
            f.write(example_code)

    def _create_integration_example(self, examples_dir: Path):
        """Create integration example with existing optimizer"""
        example_code = '''"""
Integration Example with SVG Optimization System
"""

import numpy as np
import sys
from pathlib import Path

# Add API to path
sys.path.append(str(Path(__file__).parent.parent / "api"))

from unified_predictor import UnifiedQualityPredictor


class QualityGuidedOptimizer:
    """Example optimizer that uses quality predictions to guide parameter search"""

    def __init__(self, predictor):
        self.predictor = predictor

    def optimize_parameters(self, image_features, initial_params, target_ssim=0.9):
        """Optimize VTracer parameters to achieve target SSIM"""

        best_params = initial_params.copy()
        best_ssim = self.predictor.predict_quality(image_features, best_params)

        print(f"Initial SSIM: {best_ssim:.4f}")

        # Simple grid search optimization
        param_ranges = {
            'color_precision': [2, 4, 6, 8, 10],
            'corner_threshold': [20, 40, 60, 80],
            'path_precision': [4, 8, 12, 16]
        }

        iterations = 0
        for color_prec in param_ranges['color_precision']:
            for corner_thresh in param_ranges['corner_threshold']:
                for path_prec in param_ranges['path_precision']:

                    test_params = best_params.copy()
                    test_params.update({
                        'color_precision': color_prec,
                        'corner_threshold': corner_thresh,
                        'path_precision': path_prec
                    })

                    predicted_ssim = self.predictor.predict_quality(image_features, test_params)
                    iterations += 1

                    if predicted_ssim > best_ssim:
                        best_ssim = predicted_ssim
                        best_params = test_params.copy()
                        print(f"  Iteration {iterations}: New best SSIM {best_ssim:.4f}")

                        if best_ssim >= target_ssim:
                            print(f"  Target SSIM {target_ssim:.4f} achieved!")
                            break

                if best_ssim >= target_ssim:
                    break
            if best_ssim >= target_ssim:
                break

        print(f"\\nOptimization complete:")
        print(f"  Iterations: {iterations}")
        print(f"  Final SSIM: {best_ssim:.4f}")
        print(f"  Improvement: {best_ssim - self.predictor.predict_quality(image_features, initial_params):.4f}")

        return best_params, best_ssim


def main():
    """Integration demonstration"""

    # Initialize predictor
    predictor = UnifiedQualityPredictor("../models")
    optimizer = QualityGuidedOptimizer(predictor)

    print(f"Using model: {predictor.get_model_info()}")

    # Example scenario: optimize parameters for a complex logo
    image_features = np.random.randn(2048) * 0.5

    initial_params = {
        'color_precision': 6.0,
        'corner_threshold': 60.0,
        'length_threshold': 4.0,
        'max_iterations': 10,
        'splice_threshold': 45.0,
        'path_precision': 8,
        'layer_difference': 16.0,
        'mode': 0
    }

    # Optimize for target SSIM
    optimized_params, final_ssim = optimizer.optimize_parameters(
        image_features, initial_params, target_ssim=0.95
    )

    print(f"\\nOptimized Parameters:")
    for key, value in optimized_params.items():
        if value != initial_params[key]:
            print(f"  {key}: {initial_params[key]} → {value}")


if __name__ == "__main__":
    main()
'''

        with open(examples_dir / "integration_example.py", 'w') as f:
            f.write(example_code)

    def _create_performance_example(self, examples_dir: Path):
        """Create performance testing example"""
        example_code = '''"""
Performance Testing Example
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add API to path
sys.path.append(str(Path(__file__).parent.parent / "api"))

from torchscript_predictor import TorchScriptPredictor
from onnx_predictor import ONNXPredictor
from coreml_predictor import CoreMLPredictor


def benchmark_predictor(predictor, name, iterations=1000):
    """Benchmark a specific predictor"""

    print(f"\\n🔬 Benchmarking {name}...")

    # Generate test data
    image_features = np.random.randn(2048) * 0.5
    vtracer_params = {
        'color_precision': 6.0,
        'corner_threshold': 60.0,
        'length_threshold': 4.0,
        'max_iterations': 10,
        'splice_threshold': 45.0,
        'path_precision': 8,
        'layer_difference': 16.0,
        'mode': 0
    }

    # Warmup
    for _ in range(10):
        _ = predictor.predict_quality(image_features, vtracer_params)

    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        _ = predictor.predict_quality(image_features, vtracer_params)

    total_time = time.time() - start_time

    print(f"  Total time: {total_time:.3f}s")
    print(f"  Time per prediction: {(total_time / iterations) * 1000:.2f}ms")
    print(f"  Throughput: {iterations / total_time:.1f} predictions/sec")

    return total_time / iterations


def main():
    """Performance comparison of different model formats"""

    models_dir = Path("../models")

    predictors = {}

    # Try to load different format predictors
    formats = [
        ("TorchScript", TorchScriptPredictor, "*.pt"),
        ("ONNX", ONNXPredictor, "*.onnx"),
        ("CoreML", CoreMLPredictor, "*.mlmodel")
    ]

    for name, predictor_class, pattern in formats:
        try:
            model_files = list(models_dir.glob(pattern))
            if model_files:
                predictor = predictor_class(str(model_files[0]))
                predictors[name] = predictor
                print(f"✅ Loaded {name}: {model_files[0].name}")
            else:
                print(f"⚠️ No {name} model found")
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")

    if not predictors:
        print("No predictors available for benchmarking")
        return

    # Benchmark all available predictors
    results = {}
    for name, predictor in predictors.items():
        try:
            inference_time = benchmark_predictor(predictor, name)
            results[name] = inference_time
        except Exception as e:
            print(f"❌ {name} benchmark failed: {e}")

    # Summary
    if results:
        print(f"\\n📊 Performance Summary:")
        sorted_results = sorted(results.items(), key=lambda x: x[1])

        fastest_time = sorted_results[0][1]

        for name, time_per_pred in sorted_results:
            speedup = fastest_time / time_per_pred
            print(f"  {name}: {time_per_pred * 1000:.2f}ms ({speedup:.1f}x)")


if __name__ == "__main__":
    main()
'''

        with open(examples_dir / "performance_test.py", 'w') as f:
            f.write(example_code)

    def _generate_documentation(self):
        """Generate comprehensive documentation"""
        print(f"   📖 Generating documentation...")

        docs_dir = self.package_dir / "docs"

        # Create main README
        self._create_main_readme(docs_dir)

        # Create API documentation
        self._create_api_docs(docs_dir)

        # Create deployment guide
        self._create_deployment_guide(docs_dir)

        # Create performance guide
        self._create_performance_guide(docs_dir)

    def _create_main_readme(self, docs_dir: Path):
        """Create main README file"""
        readme_content = f'''# SVG Quality Predictor Deployment Package

Version: {self.config.version}
Created: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This package contains pre-trained neural network models for predicting SVG quality (SSIM) given image features and VTracer parameters. The models enable intelligent parameter optimization for SVG conversion.

## Package Contents

```
{self.config.package_name}/
├── models/           # Exported model files (TorchScript, ONNX, CoreML)
├── api/             # Python API interfaces
├── validation/      # Validation results and reports
├── optimization/    # Performance optimization results
├── examples/        # Usage examples
├── docs/           # Documentation
├── utils/          # Utility scripts
├── tests/          # Test scripts
└── config/         # Configuration files
```

## Quick Start

```python
from api import UnifiedQualityPredictor
import numpy as np

# Initialize predictor (auto-selects best model)
predictor = UnifiedQualityPredictor("models")

# Example usage
image_features = np.random.randn(2048)  # ResNet-50 features
vtracer_params = {{
    'color_precision': 6.0,
    'corner_threshold': 60.0,
    'length_threshold': 4.0,
    'max_iterations': 10,
    'splice_threshold': 45.0,
    'path_precision': 8,
    'layer_difference': 16.0,
    'mode': 0
}}

predicted_ssim = predictor.predict_quality(image_features, vtracer_params)
print(f"Predicted SSIM: {{predicted_ssim:.4f}}")
```

## Model Formats

- **TorchScript**: Cross-platform PyTorch deployment
- **ONNX**: Broad framework compatibility
- **CoreML**: Apple Silicon optimization (macOS only)

The API automatically selects the optimal format for your platform.

## Requirements

### Python Dependencies
```
torch>=1.9.0
numpy>=1.20.0
onnxruntime>=1.8.0  # For ONNX models
coremltools>=5.0.0  # For CoreML models (macOS only)
```

### System Requirements
- Python 3.8+
- CPU: Any modern processor
- Memory: 512MB available RAM
- Storage: 500MB available space

## Performance

- **Inference Time**: <50ms per prediction (CPU)
- **Throughput**: >20 predictions/second
- **Memory Usage**: <500MB
- **Model Size**: <100MB per format

## Examples

See the `examples/` directory for:
- `basic_usage.py`: Simple prediction example
- `batch_processing.py`: Batch inference example
- `integration_example.py`: Integration with optimization systems
- `performance_test.py`: Performance benchmarking

## Documentation

- `api_reference.md`: Complete API documentation
- `deployment_guide.md`: Deployment instructions
- `performance_guide.md`: Performance optimization tips

## License

This package is provided for research and development purposes.

## Support

For questions or issues, please refer to the documentation or check the validation reports in the `validation/` directory.
'''

        with open(docs_dir / "README.md", 'w') as f:
            f.write(readme_content)

    def _create_api_docs(self, docs_dir: Path):
        """Create API documentation"""
        api_docs = '''# API Reference

## UnifiedQualityPredictor

The main entry point for quality prediction. Automatically selects the optimal model format.

### Constructor

```python
UnifiedQualityPredictor(models_dir="models")
```

**Parameters:**
- `models_dir` (str): Directory containing model files

### Methods

#### predict_quality(image_features, vtracer_params)

Predict SSIM quality for given features and parameters.

**Parameters:**
- `image_features` (np.ndarray): ResNet-50 features (2048 dims)
- `vtracer_params` (dict): VTracer parameters

**Returns:**
- `float`: Predicted SSIM value [0, 1]

#### predict_batch(features_list, params_list)

Predict quality for batch of inputs.

**Parameters:**
- `features_list` (list): List of feature arrays
- `params_list` (list): List of parameter dictionaries

**Returns:**
- `list`: List of predicted SSIM values

## Format-Specific Predictors

### TorchScriptPredictor

Direct interface to TorchScript models.

```python
predictor = TorchScriptPredictor("models/model.pt")
prediction = predictor.predict_quality(features, params)
```

### ONNXPredictor

Direct interface to ONNX models.

```python
predictor = ONNXPredictor("models/model.onnx")
prediction = predictor.predict_quality(features, params)
```

### CoreMLPredictor

Direct interface to CoreML models (macOS only).

```python
predictor = CoreMLPredictor("models/model.mlmodel")
prediction = predictor.predict_quality(features, params)
```

## Parameter Format

VTracer parameters should be provided as a dictionary:

```python
vtracer_params = {
    'color_precision': 6.0,      # [2-10] Color quantization
    'corner_threshold': 60.0,    # [10-100] Corner detection
    'length_threshold': 4.0,     # [1-10] Path length threshold
    'max_iterations': 10,        # [5-20] Maximum iterations
    'splice_threshold': 45.0,    # [20-80] Path splicing
    'path_precision': 8,         # [4-16] Path precision
    'layer_difference': 16.0,    # [8-32] Layer separation
    'mode': 0                    # [0,1] 0=spline, 1=polygon
}
```

## Error Handling

All predictors raise appropriate exceptions for:
- Missing model files
- Invalid input dimensions
- Runtime errors

```python
try:
    predictor = UnifiedQualityPredictor("models")
    result = predictor.predict_quality(features, params)
except RuntimeError as e:
    print(f"Prediction failed: {e}")
```
'''

        with open(docs_dir / "api_reference.md", 'w') as f:
            f.write(api_docs)

    def _create_deployment_guide(self, docs_dir: Path):
        """Create deployment guide"""
        deployment_guide = '''# Deployment Guide

## Installation

1. Extract the deployment package
2. Install Python dependencies:
   ```bash
   pip install torch numpy
   pip install onnxruntime  # For ONNX support
   pip install coremltools  # For CoreML support (macOS)
   ```

## Platform-Specific Setup

### Linux/Windows
- TorchScript and ONNX models supported
- No additional setup required

### macOS
- All model formats supported
- CoreML provides optimal performance on Apple Silicon

### Embedded/Edge Devices
- Use ONNX models with ONNX Runtime
- Consider quantized models for memory-constrained environments

## Integration Patterns

### Direct Integration
```python
from api import UnifiedQualityPredictor

predictor = UnifiedQualityPredictor()
quality = predictor.predict_quality(features, params)
```

### Service Integration
Create a prediction service:

```python
from flask import Flask, request, jsonify
from api import UnifiedQualityPredictor

app = Flask(__name__)
predictor = UnifiedQualityPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features']
    params = data['params']

    quality = predictor.predict_quality(features, params)
    return jsonify({'quality': quality})
```

### Batch Processing
For high-throughput scenarios:

```python
# Process in batches for efficiency
batch_size = 32
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    predictions = predictor.predict_batch(
        [item['features'] for item in batch],
        [item['params'] for item in batch]
    )
```

## Performance Optimization

### Model Selection
- **TorchScript**: Best for PyTorch environments
- **ONNX**: Best for cross-platform deployment
- **CoreML**: Best for Apple Silicon

### Threading
TorchScript models support multi-threading:
```python
import torch
torch.set_num_threads(4)  # Optimize for your CPU
```

### Batch Processing
Batch multiple predictions for better throughput:
```python
# Faster than individual predictions
predictions = predictor.predict_batch(features_list, params_list)
```

### Memory Management
For long-running services:
```python
import gc
import torch

# Periodic cleanup
if iteration % 1000 == 0:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## Monitoring

### Performance Monitoring
```python
import time

start_time = time.time()
prediction = predictor.predict_quality(features, params)
inference_time = (time.time() - start_time) * 1000

if inference_time > 100:  # 100ms threshold
    logger.warning(f"Slow inference: {inference_time:.1f}ms")
```

### Error Monitoring
```python
try:
    prediction = predictor.predict_quality(features, params)
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    # Fallback logic
```

## Troubleshooting

### Common Issues

**"No compatible model format found"**
- Ensure model files are in the models directory
- Check file permissions
- Verify dependencies are installed

**"ONNX Runtime not available"**
- Install: `pip install onnxruntime`

**"CoreML tools not available"**
- Install: `pip install coremltools` (macOS only)

**Slow inference times**
- Check CPU thread count settings
- Consider using ONNX with GPU providers
- Use batch processing for multiple predictions

### Performance Benchmarking
Run the performance test:
```bash
python examples/performance_test.py
```

This will compare all available model formats and show performance characteristics.
'''

        with open(docs_dir / "deployment_guide.md", 'w') as f:
            f.write(deployment_guide)

    def _create_performance_guide(self, docs_dir: Path):
        """Create performance optimization guide"""
        performance_guide = '''# Performance Optimization Guide

## Target Performance

- **Inference Time**: <50ms per prediction
- **Throughput**: >20 predictions/second
- **Memory Usage**: <500MB
- **Startup Time**: <5 seconds

## Platform Optimization

### CPU Optimization

#### Thread Configuration
```python
import torch
# Set optimal thread count (usually 4 for inference)
torch.set_num_threads(4)
```

#### BLAS Optimization
Enable optimized BLAS libraries:
```bash
# Intel MKL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# OpenBLAS
export OPENBLAS_NUM_THREADS=4
```

### Apple Silicon (MPS)

#### CoreML Optimization
CoreML provides best performance on Apple Silicon:
```python
# Automatically selected on macOS
predictor = UnifiedQualityPredictor()
```

#### MPS Backend (PyTorch)
For TorchScript models:
```python
if torch.backends.mps.is_available():
    predictor.set_device('mps')
```

### GPU Acceleration

#### ONNX with GPU
```python
from api import ONNXPredictor

predictor = ONNXPredictor("models/model.onnx")
predictor.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])
```

## Memory Optimization

### Model Loading
Load models on demand:
```python
class LazyPredictor:
    def __init__(self):
        self._predictor = None

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = UnifiedQualityPredictor()
        return self._predictor
```

### Batch Size Tuning
Find optimal batch size:
```python
def find_optimal_batch_size(predictor, max_batch=64):
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        try:
            features = [np.random.randn(2048) for _ in range(batch_size)]
            params = [default_params for _ in range(batch_size)]

            start_time = time.time()
            predictor.predict_batch(features, params)
            time_per_sample = (time.time() - start_time) / batch_size

            print(f"Batch {batch_size}: {time_per_sample*1000:.1f}ms/sample")

        except RuntimeError:
            break
```

## Deployment Patterns

### High-Throughput Service
```python
from concurrent.futures import ThreadPoolExecutor
import queue

class PredictionService:
    def __init__(self, num_workers=4):
        self.predictors = [
            UnifiedQualityPredictor()
            for _ in range(num_workers)
        ]
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def predict_async(self, features, params):
        return self.executor.submit(
            self._predict_worker, features, params
        )

    def _predict_worker(self, features, params):
        # Round-robin worker selection
        worker_id = threading.current_thread().ident % len(self.predictors)
        return self.predictors[worker_id].predict_quality(features, params)
```

### Caching Layer
```python
from functools import lru_cache
import hashlib

class CachedPredictor:
    def __init__(self):
        self.predictor = UnifiedQualityPredictor()

    @lru_cache(maxsize=1000)
    def predict_quality_cached(self, features_hash, params_hash):
        # Note: Actual implementation needs proper hashing
        return self.predictor.predict_quality(features, params)

    def predict_quality(self, features, params):
        features_hash = hashlib.md5(features.tobytes()).hexdigest()
        params_hash = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()

        return self.predict_quality_cached(features_hash, params_hash)
```

## Benchmarking

### Latency Testing
```python
def benchmark_latency(predictor, iterations=1000):
    features = np.random.randn(2048)
    params = default_params

    # Warmup
    for _ in range(10):
        predictor.predict_quality(features, params)

    # Measure
    latencies = []
    for _ in range(iterations):
        start = time.time()
        predictor.predict_quality(features, params)
        latencies.append((time.time() - start) * 1000)

    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99)
    }
```

### Throughput Testing
```python
def benchmark_throughput(predictor, duration=60):
    features = np.random.randn(2048)
    params = default_params

    start_time = time.time()
    count = 0

    while time.time() - start_time < duration:
        predictor.predict_quality(features, params)
        count += 1

    throughput = count / duration
    return throughput
```

## Production Considerations

### Health Checks
```python
def health_check(predictor):
    try:
        test_features = np.random.randn(2048)
        test_params = default_params

        start_time = time.time()
        prediction = predictor.predict_quality(test_features, test_params)
        inference_time = (time.time() - start_time) * 1000

        return {
            'healthy': 0 <= prediction <= 1 and inference_time < 100,
            'inference_time_ms': inference_time,
            'prediction': prediction
        }
    except Exception as e:
        return {'healthy': False, 'error': str(e)}
```

### Monitoring Metrics
- Inference time percentiles (p50, p95, p99)
- Throughput (predictions/second)
- Error rate
- Memory usage
- CPU utilization

### Scaling Guidelines
- **Single instance**: Up to 100 predictions/second
- **Multi-instance**: Linear scaling with CPU cores
- **Load balancing**: Round-robin across instances
- **Auto-scaling**: Based on queue depth or latency
'''

        with open(docs_dir / "performance_guide.md", 'w') as f:
            f.write(performance_guide)

    def _generate_requirements(self):
        """Generate requirements and setup files"""
        print(f"   📋 Generating requirements...")

        # Python requirements
        requirements = [
            "torch>=1.9.0",
            "numpy>=1.20.0",
            "onnxruntime>=1.8.0",
            "coremltools>=5.0.0; sys_platform == 'darwin'"
        ]

        with open(self.package_dir / "requirements.txt", 'w') as f:
            f.write('\n'.join(requirements))

        # Setup script
        setup_py = f'''"""
Setup script for SVG Quality Predictor deployment package
"""

from setuptools import setup, find_packages

setup(
    name="{self.config.package_name}",
    version="{self.config.version}",
    description="SVG Quality Predictor - Neural network models for SVG quality prediction",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "onnxruntime>=1.8.0"
    ],
    extras_require={{
        "coreml": ["coremltools>=5.0.0"],
        "all": ["coremltools>=5.0.0"]
    }},
    package_data={{
        "": ["*.pt", "*.onnx", "*.mlmodel", "*.json", "*.md"]
    }},
    include_package_data=True
)
'''

        with open(self.package_dir / "setup.py", 'w') as f:
            f.write(setup_py)

    def _create_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation summary"""
        summary = {
            "validation_passed": validation_results.get("summary", {}).get("deployment_ready", 0) > 0,
            "accuracy_validation": {},
            "performance_validation": {},
            "platform_compatibility": {}
        }

        # Process results
        for format_name, format_results in validation_results.get("results", {}).items():
            # Accuracy summary
            accuracy = format_results.get("accuracy_validation", {})
            summary["accuracy_validation"][format_name] = {
                "passed": accuracy.get("accuracy_preserved", False),
                "correlation": accuracy.get("correlation_with_original", 0.0),
                "max_difference": accuracy.get("max_prediction_diff", 0.0)
            }

            # Performance summary
            performance = format_results.get("performance_validation", {})
            summary["performance_validation"][format_name] = {}

            for device, device_result in performance.items():
                summary["performance_validation"][format_name][device] = {
                    "passed": device_result.get("passed", False),
                    "inference_time_ms": device_result.get("inference_time_ms", 0),
                    "throughput": device_result.get("throughput_samples_per_sec", 0)
                }

        return summary

    def _create_performance_benchmarks(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance benchmarks summary"""
        benchmarks = {
            "cpu_performance": {},
            "mps_performance": {},
            "batch_performance": {},
            "recommendations": []
        }

        # Process optimization results
        for format_name, format_results in optimization_results.items():
            for device, device_result in format_results.items():
                if isinstance(device_result, dict):
                    perf_data = {
                        "inference_time_ms": device_result.get("inference_time_ms", 0),
                        "throughput": device_result.get("throughput_samples_per_sec", 0),
                        "memory_usage_mb": device_result.get("memory_usage_mb", 0),
                        "passed": device_result.get("passed", False)
                    }

                    if device == "cpu":
                        benchmarks["cpu_performance"][format_name] = perf_data
                    elif device == "mps":
                        benchmarks["mps_performance"][format_name] = perf_data

                    # Batch performance
                    if "batch_performance" in device_result:
                        benchmarks["batch_performance"][f"{format_name}_{device}"] = device_result["batch_performance"]

        return benchmarks

    def _create_manifest(self, export_results: Dict[str, ExportResult],
                        validation_results: Dict[str, Any],
                        optimization_results: Dict[str, Any]) -> PackageManifest:
        """Create package manifest"""

        # Model information
        models = {}
        for format_name, result in export_results.items():
            if result.success:
                models[format_name] = {
                    "format": result.format,
                    "file_path": str(Path(result.file_path).name),
                    "file_size_mb": result.file_size_mb,
                    "validated": result.validation_passed
                }

        # Directory structure
        directories = {
            "models": "Exported model files",
            "api": "Python API interfaces",
            "validation": "Validation results and reports",
            "optimization": "Performance optimization results",
            "examples": "Usage examples",
            "docs": "Documentation",
            "utils": "Utility scripts",
            "tests": "Test scripts",
            "config": "Configuration files"
        }

        # Files
        files = {
            "requirements.txt": "Python dependencies",
            "setup.py": "Package installation script",
            "README.md": "Main documentation"
        }

        # Python requirements
        python_requirements = [
            "torch>=1.9.0",
            "numpy>=1.20.0",
            "onnxruntime>=1.8.0",
            "coremltools>=5.0.0 (macOS only)"
        ]

        # System requirements
        system_requirements = {
            "python_version": ">=3.8",
            "memory_mb": 512,
            "storage_mb": 500,
            "platforms": self.config.target_platforms
        }

        manifest = PackageManifest(
            package_name=self.config.package_name,
            version=self.config.version,
            creation_timestamp=time.time(),
            target_platforms=self.config.target_platforms,
            models=models,
            validation_results=self._create_validation_summary(validation_results),
            performance_benchmarks=self._create_performance_benchmarks(optimization_results),
            directories=directories,
            files=files,
            python_requirements=python_requirements,
            system_requirements=system_requirements,
            api_interfaces=["UnifiedQualityPredictor", "TorchScriptPredictor", "ONNXPredictor", "CoreMLPredictor"],
            example_scripts=["basic_usage.py", "batch_processing.py", "integration_example.py", "performance_test.py"]
        )

        # Save manifest
        with open(self.package_dir / "manifest.json", 'w') as f:
            json.dump(asdict(manifest), f, indent=2)

        return manifest

    def _create_final_package(self) -> Path:
        """Create final packaged archive"""
        print(f"   📦 Creating final package archive...")

        # Create zip archive
        archive_path = Path(f"{self.config.package_name}_v{self.config.version}.zip")

        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.package_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.package_dir.parent)
                    zipf.write(file_path, arcname)

        # Calculate archive size
        archive_size = archive_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Package created: {archive_path} ({archive_size:.1f}MB)")

        return archive_path


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Production Deployment Package")

    # Create deployment config
    config = DeploymentConfig(
        package_name="svg_quality_predictor_test",
        version="1.0.0"
    )

    # Create packager
    packager = ModelPackager(config)

    print("✅ Deployment package test completed")