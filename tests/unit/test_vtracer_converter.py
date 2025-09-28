#!/usr/bin/env python3
"""
Unit tests for VTracerConverter class.

Tests VTracer-based PNG to SVG conversion functionality.
"""

import pytest
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from backend.converters.base import BaseConverter
from backend.converters.vtracer_converter import VTracerConverter


class TestVTracerConverter:
    """Test cases for VTracerConverter class."""

    def test_vtracer_converter_initialization(self):
        """Test that VTracer converter can be initialized with default parameters."""
        converter = VTracerConverter()
        assert converter.get_name().startswith("VTracer")
        assert converter.colormode == 'color'
        assert converter.color_precision == 6
        assert converter.layer_difference == 16

    def test_vtracer_converter_custom_parameters(self):
        """Test VTracer converter initialization with custom parameters."""
        converter = VTracerConverter(
            colormode='binary',
            color_precision=4,
            layer_difference=32,
            corner_threshold=30
        )
        assert converter.colormode == 'binary'
        assert converter.color_precision == 4
        assert converter.layer_difference == 32
        assert converter.corner_threshold == 30

    def test_vtracer_converter_inheritance(self):
        """Test that VTracer converter properly inherits from BaseConverter."""
        converter = VTracerConverter()
        assert isinstance(converter, BaseConverter)

    def test_get_name_method(self):
        """Test that get_name method returns descriptive string."""
        converter = VTracerConverter(color_precision=8, layer_difference=24)
        name = converter.get_name()
        assert isinstance(name, str)
        assert "VTracer" in name
        assert "color_precision=8" in name
        assert "layer_diff=24" in name

    @patch('converters.vtracer_converter.vtracer')
    @patch('converters.vtracer_converter.os.path.exists')
    def test_convert_nonexistent_file(self, mock_exists, mock_vtracer):
        """Test conversion with nonexistent file raises FileNotFoundError."""
        mock_exists.return_value = False
        converter = VTracerConverter()

        with pytest.raises(FileNotFoundError) as exc_info:
            converter.convert("nonexistent.png")

        assert "Image not found" in str(exc_info.value)

    @patch('converters.vtracer_converter.vtracer')
    @patch('converters.vtracer_converter.os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='<svg>test</svg>')
    @patch('converters.vtracer_converter.tempfile.NamedTemporaryFile')
    @patch('converters.vtracer_converter.os_cleanup.unlink')
    def test_convert_successful(self, mock_unlink, mock_tempfile, mock_file, mock_exists, mock_vtracer):
        """Test successful conversion process."""
        # Setup mocks
        mock_exists.return_value = True
        mock_temp = MagicMock()
        mock_temp.name = '/tmp/test.svg'
        mock_tempfile.return_value.__enter__.return_value = mock_temp
        mock_tempfile.return_value.__exit__.return_value = None

        converter = VTracerConverter()
        result = converter.convert("test.png")

        # Verify result
        assert result == '<svg>test</svg>'
        mock_vtracer.convert_image_to_svg_py.assert_called_once()
        mock_unlink.assert_called_once_with('/tmp/test.svg')

    @patch('converters.vtracer_converter.vtracer')
    @patch('converters.vtracer_converter.os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='<svg width="100" height="100">test</svg>')
    @patch('converters.vtracer_converter.tempfile.NamedTemporaryFile')
    @patch('converters.vtracer_converter.os_cleanup.unlink')
    def test_convert_adds_viewbox(self, mock_unlink, mock_tempfile, mock_file, mock_exists, mock_vtracer):
        """Test that convert adds viewBox when missing."""
        # Setup mocks
        mock_exists.return_value = True
        mock_temp = MagicMock()
        mock_temp.name = '/tmp/test.svg'
        mock_tempfile.return_value.__enter__.return_value = mock_temp
        mock_tempfile.return_value.__exit__.return_value = None

        converter = VTracerConverter()
        result = converter.convert("test.png")

        # Should add viewBox
        assert 'viewBox="0 0 100 100"' in result

    @patch('converters.vtracer_converter.vtracer')
    @patch('converters.vtracer_converter.os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='<svg>test</svg>')
    @patch('converters.vtracer_converter.tempfile.NamedTemporaryFile')
    @patch('converters.vtracer_converter.os_cleanup.unlink')
    def test_convert_with_threshold_low(self, mock_unlink, mock_tempfile, mock_file, mock_exists, mock_vtracer):
        """Test threshold mapping for low values (binary mode)."""
        # Setup mocks
        mock_exists.return_value = True
        mock_temp = MagicMock()
        mock_temp.name = '/tmp/test.svg'
        mock_tempfile.return_value.__enter__.return_value = mock_temp
        mock_tempfile.return_value.__exit__.return_value = None

        converter = VTracerConverter()
        converter.convert("test.png", threshold=64)

        # Check that vtracer was called with binary mode
        call_args = mock_vtracer.convert_image_to_svg_py.call_args
        assert call_args[1]['colormode'] == 'binary'
        assert call_args[1]['color_precision'] >= 1

    @patch('converters.vtracer_converter.vtracer')
    @patch('converters.vtracer_converter.os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='<svg>test</svg>')
    @patch('converters.vtracer_converter.tempfile.NamedTemporaryFile')
    @patch('converters.vtracer_converter.os_cleanup.unlink')
    def test_convert_with_threshold_high(self, mock_unlink, mock_tempfile, mock_file, mock_exists, mock_vtracer):
        """Test threshold mapping for high values (color mode)."""
        # Setup mocks
        mock_exists.return_value = True
        mock_temp = MagicMock()
        mock_temp.name = '/tmp/test.svg'
        mock_tempfile.return_value.__enter__.return_value = mock_temp
        mock_tempfile.return_value.__exit__.return_value = None

        converter = VTracerConverter()
        converter.convert("test.png", threshold=200)

        # Check that vtracer was called with color mode
        call_args = mock_vtracer.convert_image_to_svg_py.call_args
        assert call_args[1]['colormode'] == 'color'
        assert call_args[1]['color_precision'] >= 3

    @patch('converters.vtracer_converter.vtracer')
    @patch('converters.vtracer_converter.os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='<svg>test</svg>')
    @patch('converters.vtracer_converter.tempfile.NamedTemporaryFile')
    @patch('converters.vtracer_converter.os_cleanup.unlink')
    def test_convert_with_custom_params(self, mock_unlink, mock_tempfile, mock_file, mock_exists, mock_vtracer):
        """Test conversion with custom parameters."""
        # Setup mocks
        mock_exists.return_value = True
        mock_temp = MagicMock()
        mock_temp.name = '/tmp/test.svg'
        mock_tempfile.return_value.__enter__.return_value = mock_temp
        mock_tempfile.return_value.__exit__.return_value = None

        converter = VTracerConverter()
        converter.convert(
            "test.png",
            colormode='binary',
            color_precision=8,
            corner_threshold=30
        )

        # Check custom parameters were passed
        call_args = mock_vtracer.convert_image_to_svg_py.call_args
        assert call_args[1]['colormode'] == 'binary'
        assert call_args[1]['color_precision'] == 8
        assert call_args[1]['corner_threshold'] == 30

    @patch('converters.vtracer_converter.vtracer')
    def test_convert_with_params_success(self, mock_vtracer):
        """Test convert_with_params method with successful conversion."""
        converter = VTracerConverter()

        result = converter.convert_with_params(
            "input.png",
            "output.svg",
            colormode='binary',
            color_precision=4
        )

        assert result['success'] == True
        assert 'conversion_time' in result
        mock_vtracer.convert_image_to_svg_py.assert_called_once()

    @patch('converters.vtracer_converter.vtracer')
    def test_convert_with_params_error(self, mock_vtracer):
        """Test convert_with_params method with conversion error."""
        mock_vtracer.convert_image_to_svg_py.side_effect = Exception("VTracer error")
        converter = VTracerConverter()

        result = converter.convert_with_params("input.png", "output.svg")

        assert result['success'] == False
        assert 'error' in result
        assert "VTracer error" in result['error']

    @patch('converters.vtracer_converter.vtracer')
    @patch('builtins.open', new_callable=mock_open, read_data='<svg>optimized</svg>')
    @patch('converters.vtracer_converter.tempfile.NamedTemporaryFile')
    @patch('converters.vtracer_converter.os_cleanup.unlink')
    def test_optimize_for_logos(self, mock_unlink, mock_tempfile, mock_file, mock_vtracer):
        """Test optimize_for_logos method."""
        # Setup mocks
        mock_temp = MagicMock()
        mock_temp.name = '/tmp/logo.svg'
        mock_tempfile.return_value.__enter__.return_value = mock_temp
        mock_tempfile.return_value.__exit__.return_value = None

        converter = VTracerConverter()
        result = converter.optimize_for_logos("logo.png")

        assert result == '<svg>optimized</svg>'

        # Check that optimization parameters were used
        call_args = mock_vtracer.convert_image_to_svg_py.call_args
        assert call_args[1]['color_precision'] == 4
        assert call_args[1]['layer_difference'] == 32
        assert call_args[1]['path_precision'] == 6
        assert call_args[1]['corner_threshold'] == 45

    @patch('converters.vtracer_converter.vtracer')
    @patch('converters.vtracer_converter.os.path.exists')
    def test_convert_vtracer_exception(self, mock_exists, mock_vtracer):
        """Test that VTracer exceptions are properly raised."""
        mock_exists.return_value = True
        mock_vtracer.convert_image_to_svg_py.side_effect = RuntimeError("VTracer internal error")

        converter = VTracerConverter()

        with pytest.raises(RuntimeError) as exc_info:
            converter.convert("test.png")

        assert "VTracer internal error" in str(exc_info.value)

    def test_parameter_validation_ranges(self):
        """Test that converter accepts valid parameter ranges."""
        # Test valid parameters
        converter = VTracerConverter(
            color_precision=1,  # minimum
            layer_difference=0,  # minimum
            path_precision=10,  # maximum
            corner_threshold=180,  # maximum
            length_threshold=0.1,  # small value
            max_iterations=1,  # minimum
            splice_threshold=180  # maximum
        )
        assert converter.color_precision == 1
        assert converter.layer_difference == 0
        assert converter.path_precision == 10

    def test_edge_case_empty_kwargs(self):
        """Test conversion with empty kwargs uses default parameters."""
        converter = VTracerConverter(color_precision=8)

        # Mock to check what parameters are actually passed
        with patch('converters.vtracer_converter.vtracer') as mock_vtracer, \
             patch('converters.vtracer_converter.os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data='<svg>test</svg>')), \
             patch('converters.vtracer_converter.tempfile.NamedTemporaryFile'), \
             patch('converters.vtracer_converter.os_cleanup.unlink'):

            mock_temp = MagicMock()
            mock_temp.name = '/tmp/test.svg'

            converter.convert("test.png", **{})  # Empty kwargs

            call_args = mock_vtracer.convert_image_to_svg_py.call_args
            assert call_args[1]['color_precision'] == 8  # Should use instance default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])