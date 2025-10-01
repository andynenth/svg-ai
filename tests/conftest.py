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


# Shared fixtures for new test structure

@pytest.fixture
def test_images_collection():
    """Load test images from all categories for integration testing."""
    test_dir = Path('data/test')
    images = {
        'simple': [],
        'text': [],
        'gradient': [],
        'complex': []
    }

    # Try to find categorized images
    for category in images.keys():
        category_dir = test_dir / category
        if category_dir.exists():
            images[category].extend(list(category_dir.glob('*.png')))

    # If no categorized images, try to find any test images
    if all(len(imgs) == 0 for imgs in images.values()) and test_dir.exists():
        # Put all images in 'simple' category as fallback
        images['simple'] = list(test_dir.glob('*.png'))[:4]

    # Filter to only existing files
    for category in images:
        images[category] = [img for img in images[category] if img.exists()]

    return images


@pytest.fixture
def ai_system_components():
    """Setup AI system components for testing."""
    components = {}

    try:
        from backend.ai_modules.classification import ClassificationModule
        components['classifier'] = ClassificationModule()
    except ImportError:
        components['classifier'] = None

    try:
        from backend.ai_modules.optimization import OptimizationEngine
        components['optimizer'] = OptimizationEngine()
    except ImportError:
        components['optimizer'] = None

    try:
        from backend.ai_modules.quality import QualitySystem
        components['quality'] = QualitySystem()
    except ImportError:
        components['quality'] = None

    try:
        from backend.ai_modules.pipeline.unified_ai_pipeline import UnifiedAIPipeline
        components['pipeline'] = UnifiedAIPipeline()
    except ImportError:
        components['pipeline'] = None

    try:
        from backend.converters.ai_enhanced_converter import AIEnhancedConverter
        components['converter'] = AIEnhancedConverter()
    except ImportError:
        components['converter'] = None

    return components


@pytest.fixture
def performance_benchmark():
    """Create a performance benchmark utility for testing."""
    class SimpleBenchmark:
        def __init__(self):
            self.results = []
            self.start_time = None

        def start(self):
            import time
            self.start_time = time.time()

        def stop(self, operation: str) -> float:
            import time
            if self.start_time is None:
                raise RuntimeError("Benchmark not started")
            elapsed = time.time() - self.start_time
            self.results.append({"operation": operation, "duration": elapsed})
            self.start_time = None
            return elapsed

        def get_average_time(self) -> float:
            if not self.results:
                return 0.0
            return sum(r["duration"] for r in self.results) / len(self.results)

    return SimpleBenchmark()


@pytest.fixture
def test_result_comparator():
    """Create a test result comparator utility."""
    class TestComparator:
        @staticmethod
        def compare_quality_metrics(metrics1: dict, metrics2: dict, tolerance: float = 0.01) -> bool:
            """Compare quality metrics within tolerance"""
            for key in ["ssim", "mse", "psnr"]:
                if key in metrics1 and key in metrics2:
                    if abs(metrics1[key] - metrics2[key]) > tolerance:
                        return False
            return True

        @staticmethod
        def compare_parameters(params1: dict, params2: dict) -> dict:
            """Compare two parameter sets and return differences"""
            differences = {}
            all_keys = set(params1.keys()) | set(params2.keys())

            for key in all_keys:
                val1 = params1.get(key)
                val2 = params2.get(key)

                if val1 != val2:
                    differences[key] = {
                        "value1": val1,
                        "value2": val2,
                        "difference": abs(val1 - val2) if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else None
                    }

            return differences

    return TestComparator()


@pytest.fixture
def test_image_generator():
    """Generate test images of various types and complexities."""
    class TestImageGenerator:
        @staticmethod
        def create_simple_image(width: int = 100, height: int = 100, color: str = 'red') -> Image.Image:
            """Create a simple solid color image"""
            return Image.new('RGB', (width, height), color=color)

        @staticmethod
        def create_gradient_image(width: int = 100, height: int = 100) -> Image.Image:
            """Create a gradient image"""
            img = Image.new('RGB', (width, height))
            pixels = img.load()
            for x in range(width):
                for y in range(height):
                    r = int(255 * x / width)
                    g = int(255 * y / height)
                    b = 128
                    pixels[x, y] = (r, g, b)
            return img

        @staticmethod
        def create_complex_image(width: int = 100, height: int = 100) -> Image.Image:
            """Create a complex pattern image"""
            img = Image.new('RGB', (width, height))
            pixels = img.load()
            for x in range(width):
                for y in range(height):
                    r = (x * y) % 256
                    g = (x + y) % 256
                    b = (x ^ y) % 256
                    pixels[x, y] = (r, g, b)
            return img

        @staticmethod
        def create_transparent_image(width: int = 100, height: int = 100, alpha: int = 128) -> Image.Image:
            """Create an image with transparency"""
            return Image.new('RGBA', (width, height), color=(255, 0, 0, alpha))

    return TestImageGenerator()