"""
Tests for converter modules.
"""

import pytest
import os
from pathlib import Path
import numpy as np
from PIL import Image

from converters.vtracer_converter import VTracerConverter
from converters.base import BaseConverter


# Fixtures
@pytest.fixture
def simple_test_image(tmp_path):
    """Create a simple test image."""
    img = Image.new('RGB', (100, 100), color='white')
    # Draw a red square
    pixels = img.load()
    for i in range(30, 70):
        for j in range(30, 70):
            pixels[i, j] = (255, 0, 0)

    img_path = tmp_path / "test_image.png"
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def complex_test_image(tmp_path):
    """Create a more complex test image."""
    img = Image.new('RGB', (200, 200), color='white')
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    # Draw multiple shapes
    draw.ellipse([20, 20, 80, 80], fill='red')
    draw.rectangle([120, 20, 180, 80], fill='blue')
    draw.polygon([(100, 120), (60, 180), (140, 180)], fill='green')

    img_path = tmp_path / "complex_test.png"
    img.save(img_path)
    return str(img_path)


class TestVTracerConverter:
    """Tests for VTracer converter."""

    def test_initialization(self):
        """Test converter initialization."""
        converter = VTracerConverter(color_precision=6)
        assert converter.color_precision == 6
        assert converter.layer_difference == 16
        assert isinstance(converter, BaseConverter)

    def test_basic_conversion(self, simple_test_image):
        """Test basic PNG to SVG conversion."""
        converter = VTracerConverter()
        svg = converter.convert(simple_test_image)

        assert svg is not None
        assert len(svg) > 0
        assert '<svg' in svg
        assert '</svg>' in svg

    def test_conversion_with_params(self, simple_test_image):
        """Test conversion with different parameters."""
        converter = VTracerConverter(color_precision=4, layer_difference=32)
        svg = converter.convert(simple_test_image)

        assert svg is not None
        # Lower precision should generally result in smaller SVG
        assert len(svg) > 0

    def test_logo_optimization(self, simple_test_image):
        """Test logo-optimized conversion."""
        converter = VTracerConverter()
        svg = converter.optimize_for_logos(simple_test_image)

        assert svg is not None
        assert '<svg' in svg

    def test_get_name(self):
        """Test converter name generation."""
        converter = VTracerConverter(color_precision=8, layer_difference=24)
        name = converter.get_name()

        assert 'VTracer' in name
        assert '8' in name  # color_precision
        assert '24' in name  # layer_difference

    def test_conversion_with_metrics(self, simple_test_image):
        """Test conversion with metrics calculation."""
        converter = VTracerConverter()
        result = converter.convert_with_metrics(simple_test_image)

        assert result['success'] == True
        assert result['svg'] is not None
        assert result['time'] > 0
        assert result['converter'] == converter.get_name()

    def test_file_not_found(self):
        """Test handling of missing file."""
        converter = VTracerConverter()

        with pytest.raises(FileNotFoundError):
            converter.convert("nonexistent.png")

    def test_stats_tracking(self, simple_test_image):
        """Test statistics tracking."""
        converter = VTracerConverter()

        # Perform conversions
        converter.convert_with_metrics(simple_test_image)
        converter.convert_with_metrics(simple_test_image)

        stats = converter.get_stats()
        assert stats['total_conversions'] == 2
        assert stats['total_failures'] == 0
        assert stats['average_time'] > 0

    @pytest.mark.parametrize("color_precision", [1, 4, 6, 8, 10])
    def test_different_color_precisions(self, simple_test_image, color_precision):
        """Test conversion with different color precisions."""
        converter = VTracerConverter(color_precision=color_precision)
        svg = converter.convert(simple_test_image)

        assert svg is not None
        assert len(svg) > 0

    def test_complex_image_conversion(self, complex_test_image):
        """Test conversion of complex multi-shape image."""
        converter = VTracerConverter(color_precision=8)
        svg = converter.convert(complex_test_image)

        assert svg is not None
        # Complex image should have multiple paths
        assert svg.count('<path') >= 3  # At least 3 shapes


class TestBaseConverter:
    """Tests for base converter functionality."""

    def test_abstract_methods(self):
        """Test that base class requires implementation."""
        with pytest.raises(TypeError):
            # Should not be able to instantiate abstract class
            BaseConverter()

    def test_stats_initialization(self):
        """Test stats initialization in derived class."""
        converter = VTracerConverter()
        stats = converter.get_stats()

        assert stats['total_conversions'] == 0
        assert stats['total_failures'] == 0
        assert stats['success_rate'] == 1.0  # No failures yet