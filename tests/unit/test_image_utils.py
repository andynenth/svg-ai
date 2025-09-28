#!/usr/bin/env python3
"""
Unit tests for ImageUtils class.

Tests all image processing utilities that consolidate duplicate functionality
across converters.
"""

import pytest
import sys
import numpy as np
import tempfile
import os
from pathlib import Path
from PIL import Image

from backend.utils.image_utils import ImageUtils


class TestImageUtils:
    """Test cases for ImageUtils class."""

    @pytest.fixture
    def sample_rgb_image(self):
        """Create a sample RGB image."""
        img = Image.new('RGB', (100, 100), color='red')
        return img

    @pytest.fixture
    def sample_rgba_image(self):
        """Create a sample RGBA image with transparency."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))  # Semi-transparent red
        return img

    @pytest.fixture
    def sample_grayscale_image(self):
        """Create a sample grayscale image."""
        img = Image.new('L', (100, 100), color=128)
        return img

    @pytest.fixture
    def sample_palette_image(self):
        """Create a sample palette image."""
        img = Image.new('P', (100, 100))
        img.putpalette([i % 256 for i in range(768)])  # Simple palette with valid range
        return img

    @pytest.fixture
    def temp_image_file(self, sample_rgb_image):
        """Create a temporary image file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            sample_rgb_image.save(tmp.name, 'PNG')
            yield tmp.name
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    def test_convert_to_rgba_from_file(self, temp_image_file):
        """Test converting image file to RGBA."""
        result = ImageUtils.convert_to_rgba(temp_image_file)

        assert result.mode == 'RGBA'
        assert result.size == (100, 100)
        assert isinstance(result, Image.Image)

    def test_convert_to_rgba_file_not_found(self):
        """Test convert_to_rgba with non-existent file."""
        with pytest.raises(FileNotFoundError):
            ImageUtils.convert_to_rgba("nonexistent_file.png")

    def test_convert_to_rgba_already_rgba(self, sample_rgba_image, temp_image_file):
        """Test convert_to_rgba with already RGBA image."""
        # Save RGBA image to file
        sample_rgba_image.save(temp_image_file, 'PNG')

        result = ImageUtils.convert_to_rgba(temp_image_file)
        assert result.mode == 'RGBA'

    def test_composite_on_background_rgba(self, sample_rgba_image):
        """Test compositing RGBA image on white background."""
        result = ImageUtils.composite_on_background(sample_rgba_image)

        assert result.mode == 'RGB'
        assert result.size == sample_rgba_image.size
        # Should have composited semi-transparent red on white
        pixel = result.getpixel((50, 50))
        assert pixel[0] > 128  # Should be lighter red due to transparency

    def test_composite_on_background_custom_color(self, sample_rgba_image):
        """Test compositing with custom background color."""
        blue_bg = (0, 0, 255)
        result = ImageUtils.composite_on_background(sample_rgba_image, blue_bg)

        assert result.mode == 'RGB'
        # Should blend red and blue
        pixel = result.getpixel((50, 50))
        assert pixel[2] > 0  # Should have some blue component

    def test_composite_on_background_rgb_passthrough(self, sample_rgb_image):
        """Test that RGB images pass through unchanged."""
        result = ImageUtils.composite_on_background(sample_rgb_image)

        assert result.mode == 'RGB'
        assert result.size == sample_rgb_image.size

    def test_composite_on_background_palette(self, sample_palette_image):
        """Test compositing palette image."""
        result = ImageUtils.composite_on_background(sample_palette_image)

        assert result.mode == 'RGB'
        assert result.size == sample_palette_image.size

    def test_composite_on_background_grayscale(self, sample_grayscale_image):
        """Test compositing grayscale image."""
        result = ImageUtils.composite_on_background(sample_grayscale_image)

        assert result.mode == 'RGB'
        assert result.size == sample_grayscale_image.size

    def test_composite_on_background_invalid_color(self, sample_rgba_image):
        """Test composite with invalid background color."""
        with pytest.raises(ValueError):
            ImageUtils.composite_on_background(sample_rgba_image, (255, 255))  # Only 2 values

        with pytest.raises(ValueError):
            ImageUtils.composite_on_background(sample_rgba_image, (300, 0, 0))  # Out of range

    def test_convert_to_grayscale_rgb(self, sample_rgb_image):
        """Test converting RGB image to grayscale."""
        result = ImageUtils.convert_to_grayscale(sample_rgb_image)

        assert result.mode == 'L'
        assert result.size == sample_rgb_image.size

    def test_convert_to_grayscale_rgba(self, sample_rgba_image):
        """Test converting RGBA image to grayscale."""
        result = ImageUtils.convert_to_grayscale(sample_rgba_image)

        assert result.mode == 'L'
        assert result.size == sample_rgba_image.size

    def test_convert_to_grayscale_already_grayscale(self, sample_grayscale_image):
        """Test converting already grayscale image."""
        result = ImageUtils.convert_to_grayscale(sample_grayscale_image)

        assert result.mode == 'L'
        assert result.size == sample_grayscale_image.size

    def test_apply_alpha_threshold_rgba(self, sample_rgba_image):
        """Test applying alpha threshold to RGBA image."""
        result = ImageUtils.apply_alpha_threshold(sample_rgba_image, threshold=100)

        assert result.mode == 'RGBA'
        assert result.size == sample_rgba_image.size

        # Check that alpha values are binary
        alpha_array = np.array(result.split()[3])
        unique_values = np.unique(alpha_array)
        assert len(unique_values) <= 2  # Should only have 0 and/or 255

    def test_apply_alpha_threshold_rgb_conversion(self, sample_rgb_image):
        """Test applying alpha threshold to RGB image (should convert to RGBA)."""
        result = ImageUtils.apply_alpha_threshold(sample_rgb_image, threshold=128)

        assert result.mode == 'RGBA'
        assert result.size == sample_rgb_image.size

    def test_apply_alpha_threshold_invalid_threshold(self, sample_rgba_image):
        """Test alpha threshold with invalid threshold values."""
        with pytest.raises(ValueError):
            ImageUtils.apply_alpha_threshold(sample_rgba_image, threshold=-1)

        with pytest.raises(ValueError):
            ImageUtils.apply_alpha_threshold(sample_rgba_image, threshold=256)

    def test_get_image_mode_info_rgb(self, sample_rgb_image):
        """Test getting mode info for RGB image."""
        info = ImageUtils.get_image_mode_info(sample_rgb_image)

        assert info['mode'] == 'RGB'
        assert info['size'] == (100, 100)
        assert info['has_alpha'] == False
        assert info['is_grayscale'] == False
        assert info['is_palette'] == False
        assert info['bands'] == 3

    def test_get_image_mode_info_rgba(self, sample_rgba_image):
        """Test getting mode info for RGBA image."""
        info = ImageUtils.get_image_mode_info(sample_rgba_image)

        assert info['mode'] == 'RGBA'
        assert info['has_alpha'] == True
        assert info['bands'] == 4
        assert 'alpha_stats' in info
        assert 'min' in info['alpha_stats']
        assert 'max' in info['alpha_stats']

    def test_get_image_mode_info_grayscale(self, sample_grayscale_image):
        """Test getting mode info for grayscale image."""
        info = ImageUtils.get_image_mode_info(sample_grayscale_image)

        assert info['mode'] == 'L'
        assert info['is_grayscale'] == True
        assert info['has_alpha'] == False

    def test_get_image_mode_info_palette(self, sample_palette_image):
        """Test getting mode info for palette image."""
        info = ImageUtils.get_image_mode_info(sample_palette_image)

        assert info['mode'] == 'P'
        assert info['is_palette'] == True

    def test_create_binary_mask_grayscale(self, sample_grayscale_image):
        """Test creating binary mask from grayscale image."""
        result = ImageUtils.create_binary_mask(sample_grayscale_image, threshold=100)

        assert result.mode == '1'
        assert result.size == sample_grayscale_image.size

    def test_create_binary_mask_with_invert(self, sample_grayscale_image):
        """Test creating inverted binary mask."""
        normal = ImageUtils.create_binary_mask(sample_grayscale_image, threshold=100, invert=False)
        inverted = ImageUtils.create_binary_mask(sample_grayscale_image, threshold=100, invert=True)

        # Check that they are different
        normal_array = np.array(normal)
        inverted_array = np.array(inverted)
        assert not np.array_equal(normal_array, inverted_array)

    def test_create_binary_mask_rgba(self, sample_rgba_image):
        """Test creating binary mask from RGBA image."""
        result = ImageUtils.create_binary_mask(sample_rgba_image, threshold=128)

        assert result.mode == '1'
        assert result.size == sample_rgba_image.size

    def test_safe_image_load_success(self, temp_image_file):
        """Test safe image loading with valid file."""
        result = ImageUtils.safe_image_load(temp_image_file)

        assert result is not None
        assert result.mode == 'RGBA'

    def test_safe_image_load_failure(self):
        """Test safe image loading with invalid file."""
        result = ImageUtils.safe_image_load("nonexistent_file.png")

        assert result is None

    def test_validate_image_for_conversion_valid(self, sample_rgb_image):
        """Test image validation with valid image."""
        is_valid, message = ImageUtils.validate_image_for_conversion(sample_rgb_image)

        assert is_valid == True
        assert "valid" in message.lower()

    def test_validate_image_for_conversion_zero_dimensions(self):
        """Test image validation with zero dimensions."""
        img = Image.new('RGB', (0, 100), color='red')
        is_valid, message = ImageUtils.validate_image_for_conversion(img)

        assert is_valid == False
        assert "zero dimensions" in message

    def test_validate_image_for_conversion_too_large(self):
        """Test image validation with oversized image."""
        img = Image.new('RGB', (15000, 15000), color='red')
        is_valid, message = ImageUtils.validate_image_for_conversion(img)

        assert is_valid == False
        assert "too large" in message

    def test_validate_image_for_conversion_completely_transparent(self):
        """Test image validation with completely transparent image."""
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 0))  # Transparent
        is_valid, message = ImageUtils.validate_image_for_conversion(img)

        assert is_valid == False
        assert "completely transparent" in message

    def test_validate_image_for_conversion_very_small(self):
        """Test image validation with very small image."""
        img = Image.new('RGB', (5, 5), color='red')
        is_valid, message = ImageUtils.validate_image_for_conversion(img)

        # Should be valid but with warning
        assert is_valid == True
        assert "very small" in message.lower()

    def test_edge_case_1x1_pixel(self):
        """Test processing 1x1 pixel image."""
        img = Image.new('RGB', (1, 1), color='red')

        # Should handle conversion operations
        rgba_result = ImageUtils.composite_on_background(img)
        grayscale_result = ImageUtils.convert_to_grayscale(img)
        binary_result = ImageUtils.create_binary_mask(img)

        assert rgba_result.size == (1, 1)
        assert grayscale_result.size == (1, 1)
        assert binary_result.size == (1, 1)

    def test_complex_alpha_patterns(self):
        """Test with complex alpha transparency patterns."""
        # Create image with gradient alpha
        img = Image.new('RGBA', (100, 100))
        pixels = []
        for y in range(100):
            for x in range(100):
                alpha = int(255 * (x / 100))  # Gradient from transparent to opaque
                pixels.append((255, 0, 0, alpha))
        img.putdata(pixels)

        # Test threshold operation
        thresholded = ImageUtils.apply_alpha_threshold(img, threshold=128)

        # Check that left side is transparent, right side is opaque
        left_pixel = thresholded.getpixel((10, 50))
        right_pixel = thresholded.getpixel((90, 50))

        assert left_pixel[3] == 0    # Should be transparent
        assert right_pixel[3] == 255  # Should be opaque

    def test_mode_conversions_chain(self, sample_rgb_image):
        """Test chaining multiple mode conversions."""
        # Convert through multiple formats
        rgba_img = sample_rgb_image.convert('RGBA')
        composited = ImageUtils.composite_on_background(rgba_img)
        grayscale = ImageUtils.convert_to_grayscale(composited)
        binary = ImageUtils.create_binary_mask(grayscale)

        assert composited.mode == 'RGB'
        assert grayscale.mode == 'L'
        assert binary.mode == '1'

        # All should maintain original size
        original_size = sample_rgb_image.size
        assert composited.size == original_size
        assert grayscale.size == original_size
        assert binary.size == original_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])