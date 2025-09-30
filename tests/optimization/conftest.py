"""Pytest fixtures for optimization module testing"""
import pytest
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


@pytest.fixture
def test_data_dir():
    """Path to optimization test data directory"""
    return Path("data/optimization_test")


@pytest.fixture
def test_images():
    """Dictionary of test images by category"""
    test_dir = Path("data/optimization_test")
    return {
        "simple": list((test_dir / "simple").glob("*.png")),
        "text": list((test_dir / "text").glob("*.png")),
        "gradient": list((test_dir / "gradient").glob("*.png")),
        "complex": list((test_dir / "complex").glob("*.png"))
    }


@pytest.fixture
def sample_vtracer_params():
    """Sample VTracer parameters for testing"""
    return {
        'color_precision': 6,
        'layer_difference': 10,
        'corner_threshold': 60,
        'length_threshold': 5.0,
        'max_iterations': 10,
        'splice_threshold': 45,
        'path_precision': 8,
        'mode': 'spline'
    }


@pytest.fixture
def invalid_vtracer_params():
    """Invalid VTracer parameters for testing validation"""
    return {
        'color_precision': 15,  # Out of range
        'layer_difference': -5,  # Negative value
        'corner_threshold': 'invalid',  # Wrong type
        'length_threshold': 0,  # Below minimum
        'max_iterations': 100,  # Above maximum
        'splice_threshold': 150,  # Out of range
        'path_precision': 'high',  # Wrong type
        'mode': 'unknown'  # Invalid option
    }


@pytest.fixture
def ground_truth_params():
    """Load ground truth parameters for test images"""
    ground_truth_file = Path("tests/optimization/fixtures/ground_truth_params.json")
    if ground_truth_file.exists():
        with open(ground_truth_file, 'r') as f:
            return json.load(f)
    return {}


@pytest.fixture
def test_config():
    """Load test configuration"""
    config_file = Path("tests/optimization/test_config.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return {
        "timeout": 30,
        "max_parallel": 4,
        "quality_threshold": 0.85,
        "enable_caching": True,
        "benchmark_iterations": 3
    }


@pytest.fixture
def mock_feature_vector():
    """Mock feature vector for testing"""
    return np.array([
        0.15,  # edge_density
        32,    # unique_colors
        0.75,  # entropy
        0.08,  # corner_density
        0.6,   # gradient_strength
        0.7,   # complexity_score
        0.3,   # symmetry
        0.5    # texture_complexity
    ])


@pytest.fixture
def optimization_result_template():
    """Template for optimization results"""
    return {
        "image_path": "",
        "logo_type": "",
        "parameters": {},
        "quality_metrics": {
            "ssim": 0.0,
            "mse": 0.0,
            "psnr": 0.0
        },
        "performance": {
            "conversion_time": 0.0,
            "file_size_reduction": 0.0,
            "memory_usage": 0.0
        },
        "success": False,
        "error_message": None,
        "timestamp": ""
    }


@pytest.fixture
def performance_benchmark():
    """Performance benchmark expectations"""
    return {
        "parameter_validation_ms": 10,
        "single_conversion_timeout": 30,
        "batch_conversion_parallel": 10,
        "memory_limit_mb": 500,
        "quality_target": {
            "simple": 0.95,
            "text": 0.90,
            "gradient": 0.85,
            "complex": 0.80
        }
    }