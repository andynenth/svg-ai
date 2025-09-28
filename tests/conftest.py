"""
Shared test fixtures for SVG-AI converter tests.

This module provides pytest fixtures that can be used across
unit tests and integration tests.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from PIL import Image
import io



@pytest.fixture
def sample_png_bytes():
    """Create a simple PNG image as bytes for testing."""
    # Create a simple 100x100 red square PNG
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def sample_jpeg_bytes():
    """Create a simple JPEG image as bytes for testing."""
    # Create a simple 100x100 blue square JPEG
    img = Image.new('RGB', (100, 100), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def sample_png_with_transparency():
    """Create a PNG with transparency for testing alpha-aware converters."""
    # Create a 100x100 image with transparency
    img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))  # Semi-transparent red
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def temp_png_file(sample_png_bytes):
    """Create a temporary PNG file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp.write(sample_png_bytes)
        tmp.flush()
        yield tmp.name
    # Cleanup
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def temp_jpeg_file(sample_jpeg_bytes):
    """Create a temporary JPEG file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(sample_jpeg_bytes)
        tmp.flush()
        yield tmp.name
    # Cleanup
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def temp_transparent_png_file(sample_png_with_transparency):
    """Create a temporary PNG file with transparency for testing."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp.write(sample_png_with_transparency)
        tmp.flush()
        yield tmp.name
    # Cleanup
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def sample_svg_content():
    """Return a simple SVG content string for testing."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
    <rect x="10" y="10" width="80" height="80" fill="red"/>
</svg>'''


@pytest.fixture
def flask_app():
    """Create a Flask app instance for testing API endpoints."""
    from backend.app import app
    app.config['TESTING'] = True
    return app


@pytest.fixture
def flask_client(flask_app):
    """Create a Flask test client."""
    return flask_app.test_client()


@pytest.fixture
def converter_params():
    """Return default converter parameters for testing."""
    return {
        'potrace': {
            'threshold': 128,
            'turnpolicy': 'minority',
            'turdsize': 2,
            'alphamax': 1.0,
            'opttolerance': 0.2
        },
        'vtracer': {
            'colormode': 'color',
            'color_precision': 6,
            'layer_difference': 16,
            'path_precision': 5,
            'corner_threshold': 60,
            'length_threshold': 5.0,
            'max_iterations': 10,
            'splice_threshold': 45
        },
        'smart_auto': {
            # Smart auto uses automatic parameter selection
        }
    }


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent.parent / "data" / "logos"


@pytest.fixture
def mock_upload_folder(tmp_path):
    """Create a temporary upload folder for testing."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return str(upload_dir)