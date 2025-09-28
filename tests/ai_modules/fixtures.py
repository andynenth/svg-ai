"""Test fixtures for AI modules"""

import tempfile
import numpy as np
import cv2
import pytest
from pathlib import Path

@pytest.fixture
def test_image_simple():
    """Create a simple test image"""
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    # Add a simple circle
    cv2.circle(image, (128, 128), 50, (255, 0, 0), -1)
    return image

@pytest.fixture
def test_image_complex():
    """Create a complex test image with multiple shapes"""
    image = np.zeros((512, 512, 3), dtype=np.uint8)

    # Add multiple shapes
    cv2.circle(image, (100, 100), 30, (255, 0, 0), -1)
    cv2.rectangle(image, (200, 200), (300, 300), (0, 255, 0), -1)
    cv2.ellipse(image, (400, 400), (50, 30), 45, 0, 360, (0, 0, 255), -1)

    # Add some noise
    noise = np.random.randint(0, 50, (512, 512, 3), dtype=np.uint8)
    image = cv2.add(image, noise)

    return image

@pytest.fixture
def test_image_gradient():
    """Create a gradient test image"""
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    # Create gradient
    for i in range(256):
        for j in range(256):
            image[i, j] = [i, j, (i + j) // 2]

    return image

@pytest.fixture
def test_image_text():
    """Create a text-like test image"""
    image = np.zeros((400, 200, 3), dtype=np.uint8)

    # Add text-like rectangles
    cv2.rectangle(image, (20, 80), (60, 120), (255, 255, 255), -1)
    cv2.rectangle(image, (80, 80), (120, 120), (255, 255, 255), -1)
    cv2.rectangle(image, (140, 80), (180, 120), (255, 255, 255), -1)

    return image

@pytest.fixture
def temp_image_file(test_image_simple):
    """Create a temporary image file"""
    temp_file = tempfile.mktemp(suffix='.png')
    cv2.imwrite(temp_file, test_image_simple)
    yield temp_file
    # Cleanup
    import os
    if os.path.exists(temp_file):
        os.unlink(temp_file)

@pytest.fixture
def test_features_simple():
    """Simple logo features"""
    return {
        'complexity_score': 0.2,
        'unique_colors': 4,
        'edge_density': 0.1,
        'aspect_ratio': 1.0,
        'fill_ratio': 0.3,
        'entropy': 5.0,
        'corner_density': 0.008,
        'gradient_strength': 15.0
    }

@pytest.fixture
def test_features_complex():
    """Complex logo features"""
    return {
        'complexity_score': 0.8,
        'unique_colors': 25,
        'edge_density': 0.3,
        'aspect_ratio': 1.3,
        'fill_ratio': 0.7,
        'entropy': 7.5,
        'corner_density': 0.035,
        'gradient_strength': 45.0
    }

@pytest.fixture
def test_features_gradient():
    """Gradient logo features"""
    return {
        'complexity_score': 0.6,
        'unique_colors': 40,
        'edge_density': 0.1,
        'aspect_ratio': 1.2,
        'fill_ratio': 0.6,
        'entropy': 7.0,
        'corner_density': 0.015,
        'gradient_strength': 20.0
    }

@pytest.fixture
def test_features_text():
    """Text logo features"""
    return {
        'complexity_score': 0.5,
        'unique_colors': 6,
        'edge_density': 0.4,
        'aspect_ratio': 3.0,
        'fill_ratio': 0.2,
        'entropy': 5.5,
        'corner_density': 0.025,
        'gradient_strength': 35.0
    }

@pytest.fixture
def test_parameters():
    """Standard VTracer parameters"""
    return {
        'color_precision': 5,
        'corner_threshold': 50,
        'path_precision': 15,
        'layer_difference': 5,
        'splice_threshold': 60,
        'filter_speckle': 4,
        'segment_length': 10,
        'max_iterations': 10
    }

@pytest.fixture
def test_training_data():
    """Sample training data for AI models"""
    features_list = [
        {
            'complexity_score': 0.3,
            'unique_colors': 10,
            'edge_density': 0.1,
            'aspect_ratio': 1.0,
            'fill_ratio': 0.3,
            'entropy': 6.0,
            'corner_density': 0.01,
            'gradient_strength': 20.0
        },
        {
            'complexity_score': 0.6,
            'unique_colors': 20,
            'edge_density': 0.25,
            'aspect_ratio': 1.2,
            'fill_ratio': 0.5,
            'entropy': 6.5,
            'corner_density': 0.02,
            'gradient_strength': 30.0
        }
    ]

    parameters_list = [
        {
            'color_precision': 4,
            'corner_threshold': 40,
            'path_precision': 12,
            'layer_difference': 4,
            'splice_threshold': 50,
            'filter_speckle': 3,
            'segment_length': 8,
            'max_iterations': 8
        },
        {
            'color_precision': 6,
            'corner_threshold': 55,
            'path_precision': 18,
            'layer_difference': 6,
            'splice_threshold': 65,
            'filter_speckle': 5,
            'segment_length': 12,
            'max_iterations': 12
        }
    ]

    qualities_list = [0.85, 0.92]

    return features_list, parameters_list, qualities_list

# Parametrized test scenarios
LOGO_TYPE_SCENARIOS = [
    ('simple', {
        'complexity_score': 0.2, 'unique_colors': 4, 'edge_density': 0.1,
        'aspect_ratio': 1.0, 'fill_ratio': 0.3, 'entropy': 5.0,
        'corner_density': 0.008, 'gradient_strength': 15.0
    }),
    ('text', {
        'complexity_score': 0.5, 'unique_colors': 6, 'edge_density': 0.4,
        'aspect_ratio': 3.0, 'fill_ratio': 0.2, 'entropy': 5.5,
        'corner_density': 0.025, 'gradient_strength': 35.0
    }),
    ('gradient', {
        'complexity_score': 0.6, 'unique_colors': 40, 'edge_density': 0.1,
        'aspect_ratio': 1.2, 'fill_ratio': 0.6, 'entropy': 7.0,
        'corner_density': 0.015, 'gradient_strength': 20.0
    }),
    ('complex', {
        'complexity_score': 0.8, 'unique_colors': 25, 'edge_density': 0.3,
        'aspect_ratio': 1.3, 'fill_ratio': 0.7, 'entropy': 7.5,
        'corner_density': 0.035, 'gradient_strength': 45.0
    })
]

OPTIMIZATION_SCENARIOS = [
    ('feature_mapping', 'simple'),
    ('genetic_algorithm', 'complex'),
    ('grid_search', 'text'),
    ('random_search', 'gradient')
]

IMAGE_SIZE_SCENARIOS = [
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024)
]