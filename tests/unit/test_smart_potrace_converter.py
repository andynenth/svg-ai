#!/usr/bin/env python3
"""
Unit tests for SmartPotraceConverter class.

Tests smart potrace-based PNG to SVG conversion functionality
with automatic transparency detection.
"""

import pytest
import sys
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from PIL import Image

from backend.converters.base import BaseConverter
from backend.converters.smart_potrace_converter import SmartPotraceConverter


class TestSmartPotraceConverter:
    """Test cases for SmartPotraceConverter class."""

    @patch('converters.smart_potrace_converter.subprocess.run')
    def test_smart_potrace_initialization_with_potrace(self, mock_subprocess):
        """Test that Smart Potrace converter initializes when potrace is found."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "potrace 1.16"

        converter = SmartPotraceConverter()
        assert converter.get_name() == "Smart Potrace"
        assert converter.potrace_cmd is not None
        assert converter.turnpolicy == "minority"
        assert converter.turdsize == 2

    @patch('converters.smart_potrace_converter.subprocess.run')
    def test_smart_potrace_initialization_without_potrace(self, mock_subprocess):
        """Test initialization when potrace is not found."""
        mock_subprocess.side_effect = FileNotFoundError()

        converter = SmartPotraceConverter()
        assert converter.potrace_cmd is None

    def test_smart_potrace_inheritance(self):
        """Test that Smart Potrace converter properly inherits from BaseConverter."""
        converter = SmartPotraceConverter()
        assert isinstance(converter, BaseConverter)

    @patch('converters.smart_potrace_converter.subprocess.run')
    def test_find_potrace_success(self, mock_subprocess):
        """Test _find_potrace method when potrace is found."""
        mock_subprocess.return_value.returncode = 0

        converter = SmartPotraceConverter()
        potrace_cmd = converter._find_potrace()
        assert potrace_cmd in ['potrace', '/usr/local/bin/potrace', '/opt/homebrew/bin/potrace', '/usr/bin/potrace']

    @patch('converters.smart_potrace_converter.subprocess.run')
    def test_find_potrace_not_found(self, mock_subprocess):
        """Test _find_potrace method when potrace is not found."""
        mock_subprocess.side_effect = FileNotFoundError()

        converter = SmartPotraceConverter()
        potrace_cmd = converter._find_potrace()
        assert potrace_cmd is None

    def test_has_significant_transparency_with_rgba(self):
        """Test transparency detection with RGBA image."""
        converter = SmartPotraceConverter()

        # Create test image with significant transparency
        img_data = np.ones((100, 100, 4), dtype=np.uint8) * 255
        img_data[:, :, 3] = 100  # Semi-transparent alpha
        img = Image.fromarray(img_data, 'RGBA')

        assert converter._has_significant_transparency(img) == True

    def test_has_significant_transparency_with_opaque_rgba(self):
        """Test transparency detection with opaque RGBA image."""
        converter = SmartPotraceConverter()

        # Create opaque RGBA image
        img_data = np.ones((100, 100, 4), dtype=np.uint8) * 255
        img = Image.fromarray(img_data, 'RGBA')

        assert converter._has_significant_transparency(img) == False

    def test_has_significant_transparency_with_rgb(self):
        """Test transparency detection with RGB image (no alpha)."""
        converter = SmartPotraceConverter()

        # Create RGB image (no alpha channel)
        img_data = np.ones((100, 100, 3), dtype=np.uint8) * 255
        img = Image.fromarray(img_data, 'RGB')

        assert converter._has_significant_transparency(img) == False

    def test_has_significant_transparency_with_minor_transparency(self):
        """Test transparency detection with minor transparency (< 5%)."""
        converter = SmartPotraceConverter()

        # Create image with minimal transparency
        img_data = np.ones((100, 100, 4), dtype=np.uint8) * 255
        img_data[0:2, 0:2, 3] = 100  # Only 4 pixels transparent (0.04%)
        img = Image.fromarray(img_data, 'RGBA')

        assert converter._has_significant_transparency(img) == False

    def test_convert_no_potrace(self):
        """Test convert method when potrace is not available."""
        converter = SmartPotraceConverter()
        converter.potrace_cmd = None

        with pytest.raises(Exception) as exc_info:
            converter.convert("test.png")

        assert "Potrace not found" in str(exc_info.value)

    @patch('converters.smart_potrace_converter.Image.open')
    @patch('converters.smart_potrace_converter.subprocess.run')
    def test_convert_transparent_image(self, mock_subprocess, mock_image_open):
        """Test convert method with transparent image (uses alpha mode)."""
        # Setup transparent image
        img_data = np.ones((10, 10, 4), dtype=np.uint8)
        img_data[:, :, :3] = [255, 0, 0]  # Red color
        img_data[:, :, 3] = 100  # Semi-transparent
        mock_img = Image.fromarray(img_data, 'RGBA')
        mock_image_open.return_value = mock_img

        # Setup subprocess mock
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""

        # Setup converter
        converter = SmartPotraceConverter()
        converter.potrace_cmd = "potrace"

        with patch('builtins.open', mock_open(read_data='<svg fill="#000000">test</svg>')) as mock_file, \
             patch('converters.smart_potrace_converter.tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('converters.smart_potrace_converter.os.unlink'):

            mock_temp = MagicMock()
            mock_temp.name = '/tmp/test.pbm'
            mock_tempfile.return_value.__enter__.return_value = mock_temp

            result = converter.convert("test.png", threshold=150)

            assert isinstance(result, str)
            assert '<svg' in result

    @patch('converters.smart_potrace_converter.Image.open')
    @patch('converters.smart_potrace_converter.subprocess.run')
    def test_convert_opaque_image(self, mock_subprocess, mock_image_open):
        """Test convert method with opaque image (uses standard mode)."""
        # Setup opaque RGB image
        img_data = np.ones((10, 10, 3), dtype=np.uint8) * 128
        mock_img = Image.fromarray(img_data, 'RGB')
        mock_image_open.return_value = mock_img

        # Setup subprocess mock
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""

        # Setup converter
        converter = SmartPotraceConverter()
        converter.potrace_cmd = "potrace"

        with patch('builtins.open', mock_open(read_data='<svg fill="#000000">test</svg>')) as mock_file, \
             patch('converters.smart_potrace_converter.tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('converters.smart_potrace_converter.os.unlink'):

            mock_temp = MagicMock()
            mock_temp.name = '/tmp/test.pbm'
            mock_tempfile.return_value.__enter__.return_value = mock_temp

            result = converter.convert("test.png", threshold=100)

            assert isinstance(result, str)
            assert '<svg' in result

    @patch('converters.smart_potrace_converter.subprocess.run')
    def test_run_potrace_with_custom_parameters(self, mock_subprocess):
        """Test _run_potrace method with custom parameters."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""

        converter = SmartPotraceConverter()
        converter.potrace_cmd = "potrace"

        # Create binary test image
        img_binary = Image.new('1', (10, 10), 0)
        rgb_color = [255, 0, 0]  # Red

        with patch('builtins.open', mock_open(read_data='<svg fill="#000000">test</svg>')) as mock_file, \
             patch('converters.smart_potrace_converter.tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('converters.smart_potrace_converter.os.unlink'):

            mock_temp = MagicMock()
            mock_temp.name = '/tmp/test.pbm'
            mock_tempfile.return_value.__enter__.return_value = mock_temp

            result = converter._run_potrace(
                img_binary,
                rgb_color,
                turnpolicy="majority",
                turdsize=5,
                alphamax=1.5,
                opttolerance=0.1
            )

            # Check that custom parameters were passed to subprocess
            call_args = mock_subprocess.call_args[0][0]
            assert "-z" in call_args and "majority" in call_args
            assert "-t" in call_args and "5" in call_args
            assert "-a" in call_args and "1.5" in call_args
            assert "-O" in call_args and "0.1" in call_args

    @patch('converters.smart_potrace_converter.subprocess.run')
    def test_run_potrace_color_replacement(self, mock_subprocess):
        """Test that _run_potrace correctly replaces colors in SVG."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""

        converter = SmartPotraceConverter()
        converter.potrace_cmd = "potrace"

        img_binary = Image.new('1', (10, 10), 0)
        rgb_color = [255, 0, 0]  # Red

        with patch('builtins.open', mock_open(read_data='<svg fill="#000000">test</svg>')) as mock_file, \
             patch('converters.smart_potrace_converter.tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('converters.smart_potrace_converter.os.unlink'):

            mock_temp = MagicMock()
            mock_temp.name = '/tmp/test.pbm'
            mock_tempfile.return_value.__enter__.return_value = mock_temp

            result = converter._run_potrace(img_binary, rgb_color)

            # Should replace black with red
            assert 'fill="#ff0000"' in result or 'fill="#FF0000"' in result

    @patch('converters.smart_potrace_converter.subprocess.run')
    def test_run_potrace_error_handling(self, mock_subprocess):
        """Test _run_potrace error handling when potrace fails."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Potrace error message"

        converter = SmartPotraceConverter()
        converter.potrace_cmd = "potrace"

        img_binary = Image.new('1', (10, 10), 0)
        rgb_color = [0, 0, 0]

        with patch('converters.smart_potrace_converter.tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('converters.smart_potrace_converter.os.unlink'):

            mock_temp = MagicMock()
            mock_temp.name = '/tmp/test.pbm'
            mock_tempfile.return_value.__enter__.return_value = mock_temp

            with pytest.raises(Exception) as exc_info:
                converter._run_potrace(img_binary, rgb_color)

            assert "Potrace failed" in str(exc_info.value)

    @patch('converters.smart_potrace_converter.Image.open')
    @patch('converters.smart_potrace_converter.subprocess.run')
    def test_convert_with_params_success(self, mock_subprocess, mock_image_open):
        """Test convert_with_params method with successful conversion."""
        # Setup opaque image
        img_data = np.ones((10, 10, 3), dtype=np.uint8) * 128
        mock_img = Image.fromarray(img_data, 'RGB')
        mock_image_open.return_value = mock_img

        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""

        converter = SmartPotraceConverter()
        converter.potrace_cmd = "potrace"

        with patch('builtins.open', mock_open(read_data='<svg>test content</svg>')) as mock_file, \
             patch('converters.smart_potrace_converter.tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('converters.smart_potrace_converter.os.unlink'):

            mock_temp = MagicMock()
            mock_temp.name = '/tmp/test.pbm'
            mock_tempfile.return_value.__enter__.return_value = mock_temp

            result = converter.convert_with_params("input.png", "output.svg", threshold=128)

            assert result['success'] == True
            assert 'svg_size' in result
            assert result['output_path'] == "output.svg"

    @patch('converters.smart_potrace_converter.Image.open')
    def test_convert_with_params_error(self, mock_image_open):
        """Test convert_with_params method with conversion error."""
        mock_image_open.side_effect = Exception("Image load error")

        converter = SmartPotraceConverter()

        result = converter.convert_with_params("input.png", "output.svg")

        assert result['success'] == False
        assert 'error' in result
        assert "Image load error" in result['error']

    def test_convert_with_alpha_mode_selection(self):
        """Test that alpha mode correctly processes alpha channel."""
        converter = SmartPotraceConverter()

        # Create RGBA image with defined alpha pattern
        img_data = np.ones((10, 10, 4), dtype=np.uint8) * 255
        img_data[:, :, :3] = [200, 100, 50]  # Orange color
        img_data[2:8, 2:8, 3] = 200  # Center area with high alpha
        img_data[:2, :, 3] = 50      # Top rows with low alpha
        img = Image.fromarray(img_data, 'RGBA')

        with patch('converters.smart_potrace_converter.subprocess.run') as mock_subprocess, \
             patch('builtins.open', mock_open(read_data='<svg fill="#000000">test</svg>')) as mock_file, \
             patch('converters.smart_potrace_converter.tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('converters.smart_potrace_converter.os.unlink'):

            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stderr = ""
            converter.potrace_cmd = "potrace"

            mock_temp = MagicMock()
            mock_temp.name = '/tmp/test.pbm'
            mock_tempfile.return_value.__enter__.return_value = mock_temp

            result = converter._convert_with_alpha(img, threshold=150)

            assert isinstance(result, str)
            # Should extract orange-ish color from high-alpha pixels
            call_args = mock_subprocess.call_args[0][0]
            assert "potrace" in call_args[0]

    def test_convert_standard_mode_rgba_compositing(self):
        """Test that standard mode correctly composites RGBA on white background."""
        converter = SmartPotraceConverter()

        # Create RGBA image
        img_data = np.ones((10, 10, 4), dtype=np.uint8)
        img_data[:, :, :3] = [255, 0, 0]  # Red
        img_data[:, :, 3] = 128           # Semi-transparent
        img = Image.fromarray(img_data, 'RGBA')

        with patch('converters.smart_potrace_converter.subprocess.run') as mock_subprocess, \
             patch('builtins.open', mock_open(read_data='<svg fill="#000000">test</svg>')) as mock_file, \
             patch('converters.smart_potrace_converter.tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('converters.smart_potrace_converter.os.unlink'):

            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stderr = ""
            converter.potrace_cmd = "potrace"

            mock_temp = MagicMock()
            mock_temp.name = '/tmp/test.pbm'
            mock_tempfile.return_value.__enter__.return_value = mock_temp

            result = converter._convert_standard(img, threshold=128)

            assert isinstance(result, str)

    def test_convert_standard_mode_palette(self):
        """Test that standard mode correctly handles palette images."""
        converter = SmartPotraceConverter()

        # Create palette image
        img = Image.new('P', (10, 10))
        img.putpalette([i for i in range(768)])  # Create simple palette

        with patch('converters.smart_potrace_converter.subprocess.run') as mock_subprocess, \
             patch('builtins.open', mock_open(read_data='<svg fill="#000000">test</svg>')) as mock_file, \
             patch('converters.smart_potrace_converter.tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('converters.smart_potrace_converter.os.unlink'):

            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stderr = ""
            converter.potrace_cmd = "potrace"

            mock_temp = MagicMock()
            mock_temp.name = '/tmp/test.pbm'
            mock_tempfile.return_value.__enter__.return_value = mock_temp

            result = converter._convert_standard(img, threshold=128)

            assert isinstance(result, str)

    def test_opttolerance_minimum_value(self):
        """Test that opttolerance has minimum value of 0.001."""
        converter = SmartPotraceConverter()
        converter.potrace_cmd = "potrace"

        img_binary = Image.new('1', (10, 10), 0)
        rgb_color = [0, 0, 0]

        with patch('converters.smart_potrace_converter.subprocess.run') as mock_subprocess, \
             patch('builtins.open', mock_open(read_data='<svg>test</svg>')) as mock_file, \
             patch('converters.smart_potrace_converter.tempfile.NamedTemporaryFile') as mock_tempfile, \
             patch('converters.smart_potrace_converter.os.unlink'):

            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stderr = ""

            mock_temp = MagicMock()
            mock_temp.name = '/tmp/test.pbm'
            mock_tempfile.return_value.__enter__.return_value = mock_temp

            converter._run_potrace(img_binary, rgb_color, opttolerance=0.0)

            # Should use 0.001 instead of 0.0
            call_args = mock_subprocess.call_args[0][0]
            assert "-O" in call_args
            opt_index = call_args.index("-O")
            assert float(call_args[opt_index + 1]) == 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])