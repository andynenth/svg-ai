#!/usr/bin/env python3
"""
Mock utilities for comprehensive testing without external dependencies.
Provides mock patterns for AI/ML modules, file operations, and external services.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from typing import Dict, Any, Optional, List
import tempfile
import os


class MockExternalDependencies:
    """Context manager for mocking external dependencies"""

    def __init__(self, mock_opencv=True, mock_torch=True, mock_sklearn=True, mock_vtracer=True):
        self.mock_opencv = mock_opencv
        self.mock_torch = mock_torch
        self.mock_sklearn = mock_sklearn
        self.mock_vtracer = mock_vtracer
        self.patches = []

    def __enter__(self):
        # Mock OpenCV
        if self.mock_opencv:
            cv2_mock = MagicMock()
            cv2_mock.imread.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2_mock.imwrite.return_value = True
            cv2_mock.cvtColor.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            cv2_mock.Sobel.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            cv2_mock.Laplacian.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

            cv2_patch = patch('cv2', cv2_mock)
            self.patches.append(cv2_patch)
            cv2_patch.start()

        # Mock PyTorch
        if self.mock_torch:
            torch_mock = MagicMock()
            torch_mock.load.return_value = MagicMock()
            torch_mock.tensor.return_value = MagicMock()

            torch_patch = patch('torch', torch_mock)
            self.patches.append(torch_patch)
            torch_patch.start()

        # Mock scikit-learn
        if self.mock_sklearn:
            sklearn_patch = patch('sklearn.cluster.KMeans')
            kmeans_mock = sklearn_patch.start()
            kmeans_mock.return_value.fit.return_value = MagicMock()
            kmeans_mock.return_value.n_clusters_ = 5
            self.patches.append(sklearn_patch)

        # Mock VTracer
        if self.mock_vtracer:
            vtracer_mock = MagicMock()
            vtracer_mock.convert_image_to_svg_py.return_value = None

            vtracer_patch = patch('vtracer', vtracer_mock)
            self.patches.append(vtracer_patch)
            vtracer_patch.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for patch_obj in self.patches:
            patch_obj.stop()


class MockImageGenerator:
    """Generate mock images for testing"""

    @staticmethod
    def create_simple_logo(width=100, height=100):
        """Create a simple geometric logo image"""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        # Add a simple rectangle
        image[20:80, 20:80] = [255, 0, 0]  # Red rectangle
        return image

    @staticmethod
    def create_text_logo(width=200, height=100):
        """Create a text-like logo image"""
        image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
        # Add some black stripes to simulate text
        image[30:70, 20:180:10] = [0, 0, 0]  # Vertical black stripes
        return image

    @staticmethod
    def create_gradient_logo(width=100, height=100):
        """Create a gradient logo image"""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                intensity = int(255 * (i + j) / (height + width))
                image[i, j] = [intensity, intensity//2, intensity//3]
        return image

    @staticmethod
    def create_complex_logo(width=150, height=150):
        """Create a complex logo with multiple elements"""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        # Add multiple colored shapes
        image[20:50, 20:50] = [255, 100, 100]  # Red square
        image[70:100, 70:100] = [100, 255, 100]  # Green square
        image[100:130, 20:50] = [100, 100, 255]  # Blue square
        return image


class MockFileOperations:
    """Mock file operations for testing"""

    @staticmethod
    def mock_image_file_read(image_array: np.ndarray):
        """Mock reading an image file"""
        return patch('backend.ai_modules.feature_extraction.cv2.imread', return_value=image_array)

    @staticmethod
    def mock_svg_file_write(svg_content: str = '<svg>mock content</svg>'):
        """Mock writing SVG file"""
        return mock_open(read_data=svg_content)

    @staticmethod
    def mock_json_file_operations(json_data: Dict[str, Any]):
        """Mock JSON file read/write operations"""
        import json
        return mock_open(read_data=json.dumps(json_data))

    @staticmethod
    def create_temp_image_file(image_array: Optional[np.ndarray] = None) -> str:
        """Create a temporary image file for testing"""
        if image_array is None:
            image_array = MockImageGenerator.create_simple_logo()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Write minimal PNG header + data
            tmp.write(b'\x89PNG\r\n\x1a\n' + image_array.tobytes())
            return tmp.name


class MockAIModules:
    """Mock AI modules for testing without model dependencies"""

    @staticmethod
    def mock_feature_extractor():
        """Mock ImageFeatureExtractor"""
        mock_extractor = MagicMock()
        mock_extractor.extract_features.return_value = {
            "edge_density": 0.4,
            "unique_colors": 0.3,
            "entropy": 0.6,
            "corner_density": 0.2,
            "gradient_strength": 0.3,
            "complexity_score": 0.5
        }
        return mock_extractor

    @staticmethod
    def mock_classification_module():
        """Mock ClassificationModule"""
        mock_classifier = MagicMock()
        mock_classifier.classify_image.return_value = {
            "logo_type": "simple",
            "confidence": 0.85,
            "features": {
                "geometry_score": 0.8,
                "text_score": 0.2,
                "complexity_score": 0.3
            }
        }
        return mock_classifier

    @staticmethod
    def mock_optimization_engine():
        """Mock OptimizationEngine"""
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = {
            "parameters": {
                "colormode": "color",
                "color_precision": 6,
                "layer_difference": 16,
                "path_precision": 5,
                "corner_threshold": 60,
                "length_threshold": 5.0,
                "max_iterations": 10,
                "splice_threshold": 45
            },
            "confidence": 0.8,
            "method": "mock_optimization"
        }
        return mock_optimizer

    @staticmethod
    def mock_quality_system():
        """Mock QualitySystem"""
        mock_quality = MagicMock()
        mock_quality.calculate_comprehensive_metrics.return_value = {
            "ssim": 0.85,
            "mse": 100.0,
            "psnr": 25.0,
            "perceptual_loss": 0.1,
            "overall_score": 0.8
        }
        mock_quality.calculate_metrics.return_value = {
            "ssim": 0.85,
            "quality_score": 0.8
        }
        return mock_quality


class MockErrorConditions:
    """Mock various error conditions for robust testing"""

    @staticmethod
    def mock_file_not_found():
        """Mock file not found errors"""
        return patch('builtins.open', side_effect=FileNotFoundError("File not found"))

    @staticmethod
    def mock_permission_error():
        """Mock permission errors"""
        return patch('builtins.open', side_effect=PermissionError("Permission denied"))

    @staticmethod
    def mock_opencv_import_error():
        """Mock OpenCV import error"""
        return patch('backend.ai_modules.feature_extraction.cv2', side_effect=ImportError("OpenCV not available"))

    @staticmethod
    def mock_memory_error():
        """Mock memory error for large operations"""
        return patch('numpy.array', side_effect=MemoryError("Out of memory"))

    @staticmethod
    def mock_network_error():
        """Mock network-related errors"""
        return patch('requests.get', side_effect=ConnectionError("Network error"))

    @staticmethod
    def mock_model_loading_error():
        """Mock model loading errors"""
        return patch('torch.load', side_effect=RuntimeError("Model loading failed"))


class MockPerformanceConditions:
    """Mock performance-related conditions"""

    @staticmethod
    def mock_slow_operation(delay: float = 1.0):
        """Mock slow operations"""
        import time
        def slow_function(*args, **kwargs):
            time.sleep(delay)
            return MagicMock()
        return slow_function

    @staticmethod
    def mock_large_dataset():
        """Mock large dataset operations"""
        large_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        return large_array


# Pytest fixtures for common mock scenarios
@pytest.fixture
def mock_external_deps():
    """Fixture for mocking all external dependencies"""
    with MockExternalDependencies() as deps:
        yield deps


@pytest.fixture
def mock_image_simple():
    """Fixture for simple mock image"""
    return MockImageGenerator.create_simple_logo()


@pytest.fixture
def mock_image_complex():
    """Fixture for complex mock image"""
    return MockImageGenerator.create_complex_logo()


@pytest.fixture
def mock_feature_extractor():
    """Fixture for mock feature extractor"""
    return MockAIModules.mock_feature_extractor()


@pytest.fixture
def mock_classifier():
    """Fixture for mock classifier"""
    return MockAIModules.mock_classification_module()


@pytest.fixture
def mock_optimizer():
    """Fixture for mock optimizer"""
    return MockAIModules.mock_optimization_engine()


@pytest.fixture
def mock_quality_system():
    """Fixture for mock quality system"""
    return MockAIModules.mock_quality_system()


@pytest.fixture
def temp_image_file():
    """Fixture for temporary image file"""
    filepath = MockFileOperations.create_temp_image_file()
    yield filepath
    try:
        os.unlink(filepath)
    except FileNotFoundError:
        pass


# Decorators for common mock patterns
def mock_heavy_dependencies(func):
    """Decorator to mock heavy external dependencies"""
    def wrapper(*args, **kwargs):
        with MockExternalDependencies():
            return func(*args, **kwargs)
    return wrapper


def mock_file_operations(svg_content='<svg>test</svg>'):
    """Decorator to mock file operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with patch('builtins.open', MockFileOperations.mock_svg_file_write(svg_content)):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def mock_ai_models(func):
    """Decorator to mock AI model operations"""
    def wrapper(*args, **kwargs):
        with patch('backend.ai_modules.feature_extraction.ImageFeatureExtractor', MockAIModules.mock_feature_extractor), \
             patch('backend.ai_modules.classification.ClassificationModule', MockAIModules.mock_classification_module), \
             patch('backend.ai_modules.optimization.OptimizationEngine', MockAIModules.mock_optimization_engine):
            return func(*args, **kwargs)
    return wrapper


# Utility functions for test assertions
def assert_mock_called_with_image(mock_func, expected_shape=None):
    """Assert mock function was called with image-like data"""
    assert mock_func.called
    if expected_shape:
        args, kwargs = mock_func.call_args
        if args:
            image_arg = args[0]
            if hasattr(image_arg, 'shape'):
                assert image_arg.shape == expected_shape


def assert_valid_svg_content(svg_content: str):
    """Assert SVG content is valid"""
    assert isinstance(svg_content, str)
    assert len(svg_content) > 0
    assert '<svg' in svg_content.lower()
    assert '</svg>' in svg_content.lower()


def assert_valid_features(features: Dict[str, float]):
    """Assert feature dictionary is valid"""
    assert isinstance(features, dict)
    expected_features = ['edge_density', 'unique_colors', 'entropy', 'complexity_score']
    for feature in expected_features:
        assert feature in features
        assert isinstance(features[feature], (int, float))
        assert 0.0 <= features[feature] <= 1.0


def assert_valid_optimization_result(optimization_result: Dict[str, Any]):
    """Assert optimization result is valid"""
    assert isinstance(optimization_result, dict)
    assert 'parameters' in optimization_result
    assert 'confidence' in optimization_result
    assert isinstance(optimization_result['parameters'], dict)
    assert 0.0 <= optimization_result['confidence'] <= 1.0


# Mock data generators
class MockDataGenerator:
    """Generate mock data for various test scenarios"""

    @staticmethod
    def generate_features_batch(count: int = 10) -> List[Dict[str, float]]:
        """Generate batch of mock features"""
        features_list = []
        for i in range(count):
            features = {
                "edge_density": np.random.uniform(0, 1),
                "unique_colors": np.random.uniform(0, 1),
                "entropy": np.random.uniform(0, 1),
                "corner_density": np.random.uniform(0, 1),
                "gradient_strength": np.random.uniform(0, 1),
                "complexity_score": np.random.uniform(0, 1)
            }
            features_list.append(features)
        return features_list

    @staticmethod
    def generate_optimization_results(count: int = 10) -> List[Dict[str, Any]]:
        """Generate batch of mock optimization results"""
        results = []
        for i in range(count):
            result = {
                "parameters": {
                    "color_precision": np.random.randint(1, 10),
                    "corner_threshold": np.random.randint(10, 100),
                    "layer_difference": np.random.randint(1, 30)
                },
                "confidence": np.random.uniform(0, 1),
                "method": f"mock_method_{i}"
            }
            results.append(result)
        return results


# Context managers for specific mock scenarios
class MockOpenCVUnavailable:
    """Context manager for testing without OpenCV"""

    def __enter__(self):
        self.patch = patch('backend.ai_modules.feature_extraction.OPENCV_AVAILABLE', False)
        self.patch.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patch.stop()


class MockLowMemoryCondition:
    """Context manager for testing low memory conditions"""

    def __enter__(self):
        # Mock numpy operations to raise memory errors
        self.patches = [
            patch('numpy.zeros', side_effect=MemoryError("Insufficient memory")),
            patch('numpy.array', side_effect=MemoryError("Insufficient memory"))
        ]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self.patches:
            p.stop()


# Helper functions for mock validation
def validate_mock_setup():
    """Validate that mock setup is working correctly"""
    try:
        with MockExternalDependencies():
            import cv2
            assert hasattr(cv2, 'imread')
            assert callable(cv2.imread)
        return True
    except Exception:
        return False


def cleanup_mock_files():
    """Clean up any temporary files created during mocking"""
    import glob
    temp_files = glob.glob('/tmp/mock_test_*')
    for file in temp_files:
        try:
            os.unlink(file)
        except OSError:
            pass