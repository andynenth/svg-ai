#!/usr/bin/env python3
"""
Comprehensive tests for ImageFeatureExtractor module
Tests core AI feature extraction functionality
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Handle OpenCV import carefully
try:
    import cv2
    OPENCV_AVAILABLE = True
except (ImportError, AttributeError):
    OPENCV_AVAILABLE = False
    cv2 = None

from backend.ai_modules.feature_extraction import ImageFeatureExtractor


class TestImageFeatureExtractor:
    """Test suite for ImageFeatureExtractor class"""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance for testing"""
        return ImageFeatureExtractor(cache_enabled=False, log_level="ERROR")

    @pytest.fixture
    def sample_image_path(self):
        """Create a simple test image and return its path"""
        if not OPENCV_AVAILABLE:
            pytest.skip("OpenCV not available")

        # Create a 100x100 test image with some patterns
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Add some colored rectangles for feature variation
        image[20:40, 20:40] = [255, 0, 0]  # Red rectangle
        image[60:80, 60:80] = [0, 255, 0]  # Green rectangle
        image[20:40, 60:80] = [0, 0, 255]  # Blue rectangle

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            tmp_path = tmp.name

        yield tmp_path

        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    @pytest.fixture
    def sample_image_array(self):
        """Create a sample image array for testing"""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Add patterns for edge detection
        image[20:40, 20:40] = [255, 255, 255]  # White square
        image[60:80, 60:80] = [128, 128, 128]  # Gray square

        return image

    def test_init_default(self):
        """Test default initialization"""
        extractor = ImageFeatureExtractor()
        assert extractor.cache_enabled is True
        assert extractor.cache == {}
        assert extractor.logger is not None

    def test_init_custom(self):
        """Test custom initialization"""
        extractor = ImageFeatureExtractor(cache_enabled=False, log_level="DEBUG")
        assert extractor.cache_enabled is False
        assert extractor.cache == {}
        assert extractor.logger.level == 10  # DEBUG level

    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
    def test_extract_features_success(self, extractor, sample_image_path):
        """Test successful feature extraction"""
        features = extractor.extract_features(sample_image_path)

        # Check that all expected features are present
        expected_features = [
            'edge_density', 'unique_colors', 'entropy',
            'corner_density', 'gradient_strength', 'complexity_score'
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert 0 <= features[feature] <= 1  # Features should be normalized

    def test_extract_features_invalid_path(self, extractor):
        """Test feature extraction with invalid path"""
        with pytest.raises(ValueError, match="Image path must be a non-empty string"):
            extractor.extract_features("")

        with pytest.raises(ValueError, match="Image path must be a non-empty string"):
            extractor.extract_features(None)

    def test_extract_features_nonexistent_file(self, extractor):
        """Test feature extraction with non-existent file"""
        with pytest.raises(FileNotFoundError):
            extractor.extract_features("/nonexistent/path/image.png")

    def test_load_and_validate_image_success(self, extractor, sample_image_path):
        """Test successful image loading"""
        image = extractor._load_and_validate_image(sample_image_path)

        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3  # Should be color image
        assert image.shape[2] == 3  # RGB channels

    def test_load_and_validate_image_invalid_file(self, extractor):
        """Test image loading with invalid file"""
        # Create a text file instead of image
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"This is not an image")
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="Invalid image format"):
                extractor._load_and_validate_image(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_calculate_edge_density(self, extractor, sample_image_array):
        """Test edge density calculation"""
        edge_density = extractor._calculate_edge_density(sample_image_array)

        assert isinstance(edge_density, float)
        assert 0 <= edge_density <= 1

    def test_calculate_edge_density_solid_color(self, extractor):
        """Test edge density on solid color image (should be low)"""
        # Create solid color image (no edges)
        solid_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        edge_density = extractor._calculate_edge_density(solid_image)

        assert edge_density < 0.1  # Very low edge density expected

    def test_count_unique_colors(self, extractor, sample_image_array):
        """Test unique color counting"""
        color_score = extractor._count_unique_colors(sample_image_array)

        assert isinstance(color_score, float)
        assert 0 <= color_score <= 1

    def test_count_unique_colors_simple(self, extractor):
        """Test unique color counting on simple image"""
        # Create image with exactly 2 colors
        simple_image = np.zeros((50, 50, 3), dtype=np.uint8)
        simple_image[25:, :] = [255, 255, 255]  # Half white, half black

        color_score = extractor._count_unique_colors(simple_image)
        assert color_score < 0.5  # Should be low for simple image

    def test_calculate_entropy(self, extractor, sample_image_array):
        """Test entropy calculation"""
        entropy = extractor._calculate_entropy(sample_image_array)

        assert isinstance(entropy, float)
        assert 0 <= entropy <= 1

    def test_calculate_entropy_uniform(self, extractor):
        """Test entropy on uniform image (should be low)"""
        uniform_image = np.full((50, 50, 3), 128, dtype=np.uint8)
        entropy = extractor._calculate_entropy(uniform_image)

        assert entropy < 0.3  # Low entropy expected for uniform image

    def test_sobel_edge_density(self, extractor):
        """Test Sobel edge detection"""
        # Create image with clear edges
        edge_image = np.zeros((100, 100), dtype=np.uint8)
        edge_image[40:60, :] = 255  # Horizontal stripe

        edge_density = extractor._sobel_edge_density(edge_image)

        assert isinstance(edge_density, float)
        assert edge_density > 0  # Should detect edges

    def test_laplacian_edge_density(self, extractor):
        """Test Laplacian edge detection"""
        # Create image with clear edges
        edge_image = np.zeros((100, 100), dtype=np.uint8)
        edge_image[40:60, 40:60] = 255  # Square

        edge_density = extractor._laplacian_edge_density(edge_image)

        assert isinstance(edge_density, float)
        assert edge_density > 0  # Should detect edges

    def test_quantize_colors(self, extractor, sample_image_array):
        """Test color quantization"""
        quantized = extractor._quantize_colors(sample_image_array, levels=8)

        assert isinstance(quantized, np.ndarray)
        assert quantized.shape == sample_image_array.shape

        # Check that values are quantized (limited set of values)
        unique_values = np.unique(quantized)
        assert len(unique_values) <= 8 * 3  # 8 levels per channel max

    def test_quantized_color_count(self, extractor, sample_image_array):
        """Test quantized color counting"""
        color_count = extractor._quantized_color_count(sample_image_array)

        assert isinstance(color_count, int)
        assert color_count >= 1  # At least one color

    def test_fast_quantized_color_count(self, extractor, sample_image_array):
        """Test fast quantized color counting"""
        color_count = extractor._fast_quantized_color_count(sample_image_array)

        assert isinstance(color_count, int)
        assert color_count >= 1  # At least one color

    def test_hsv_color_analysis(self, extractor, sample_image_array):
        """Test HSV color analysis"""
        color_count = extractor._hsv_color_analysis(sample_image_array)

        assert isinstance(color_count, int)
        assert color_count >= 1

    def test_perceptual_color_clustering(self, extractor, sample_image_array):
        """Test perceptual color clustering"""
        cluster_count = extractor._perceptual_color_clustering(sample_image_array, max_clusters=8)

        assert isinstance(cluster_count, int)
        assert 1 <= cluster_count <= 8

    def test_calculate_histogram_entropy(self, extractor):
        """Test histogram entropy calculation"""
        # Create grayscale image
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        entropy = extractor._calculate_histogram_entropy(gray_image)

        assert isinstance(entropy, float)
        assert entropy >= 0

    def test_caching_functionality(self):
        """Test feature caching when enabled"""
        extractor = ImageFeatureExtractor(cache_enabled=True, log_level="ERROR")

        # Mock the actual computation to verify caching
        with patch.object(extractor, '_load_and_validate_image') as mock_load:
            mock_image = np.zeros((50, 50, 3), dtype=np.uint8)
            mock_load.return_value = mock_image

            # Create a temporary image file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, mock_image)
                tmp_path = tmp.name

            try:
                # First call should load and cache
                features1 = extractor.extract_features(tmp_path)

                # Second call should use cache
                features2 = extractor.extract_features(tmp_path)

                # Results should be identical
                assert features1 == features2

                # Cache should contain the result
                assert len(extractor.cache) > 0

            finally:
                os.unlink(tmp_path)

    def test_error_handling_corrupted_image(self, extractor):
        """Test error handling for corrupted image data"""
        # Create a file that looks like an image but isn't
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b'\x89PNG\r\n\x1a\n' + b'corrupted_data')
            tmp_path = tmp.name

        try:
            with pytest.raises(ValueError, match="Invalid image format"):
                extractor.extract_features(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_feature_normalization(self, extractor, sample_image_path):
        """Test that all features are properly normalized to [0, 1]"""
        features = extractor.extract_features(sample_image_path)

        for feature_name, value in features.items():
            assert 0 <= value <= 1, f"Feature {feature_name} = {value} is not normalized"

    def test_feature_consistency(self, extractor, sample_image_path):
        """Test that features are consistent across multiple extractions"""
        features1 = extractor.extract_features(sample_image_path)
        features2 = extractor.extract_features(sample_image_path)

        # Results should be identical for same image
        for feature_name in features1:
            assert abs(features1[feature_name] - features2[feature_name]) < 1e-6

    @patch('cv2.imread')
    def test_opencv_error_handling(self, mock_imread, extractor):
        """Test handling of OpenCV errors"""
        mock_imread.return_value = None  # Simulate OpenCV failure

        with pytest.raises(ValueError, match="Invalid image format"):
            extractor._load_and_validate_image("test_path.png")