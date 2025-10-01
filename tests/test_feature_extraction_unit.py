#!/usr/bin/env python3
"""
Unit tests for ImageFeatureExtractor module using mocks
Tests core functionality without requiring OpenCV installation
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open

from backend.ai_modules.feature_extraction import ImageFeatureExtractor


class TestImageFeatureExtractorUnit:
    """Unit test suite using mocks for ImageFeatureExtractor"""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance for testing"""
        return ImageFeatureExtractor(cache_enabled=False, log_level="ERROR")

    @pytest.fixture
    def mock_image(self):
        """Create mock image array"""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_init_default(self):
        """Test default initialization"""
        extractor = ImageFeatureExtractor()
        assert extractor.cache_enabled is True
        assert extractor.cache == {}
        assert extractor.logger is not None

    def test_init_custom(self):
        """Test custom initialization with different parameters"""
        extractor = ImageFeatureExtractor(cache_enabled=False, log_level="DEBUG")
        assert extractor.cache_enabled is False
        assert extractor.cache == {}
        assert extractor.logger.level == 10  # DEBUG level

    def test_extract_features_invalid_input(self, extractor):
        """Test feature extraction with invalid inputs"""
        with pytest.raises(ValueError, match="Image path must be a non-empty string"):
            extractor.extract_features("")

        with pytest.raises(ValueError, match="Image path must be a non-empty string"):
            extractor.extract_features(None)

    def test_extract_features_nonexistent_file(self, extractor):
        """Test feature extraction with non-existent file"""
        with pytest.raises(FileNotFoundError):
            extractor.extract_features("/nonexistent/path/image.png")

    @patch('backend.ai_modules.feature_extraction.Path.exists')
    @patch('backend.ai_modules.feature_extraction.cv2.imread')
    def test_extract_features_success_mocked(self, mock_imread, mock_exists, extractor):
        """Test successful feature extraction using mocks"""
        # Mock file exists
        mock_exists.return_value = True

        # Mock successful image loading
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        # Mock the individual feature extraction methods
        with patch.object(extractor, '_calculate_edge_density', return_value=0.5), \
             patch.object(extractor, '_count_unique_colors', return_value=0.3), \
             patch.object(extractor, '_calculate_entropy', return_value=0.7), \
             patch.object(extractor, '_calculate_corner_density', return_value=0.4), \
             patch.object(extractor, '_calculate_gradient_strength', return_value=0.6), \
             patch.object(extractor, '_calculate_complexity_score', return_value=0.8):

            features = extractor.extract_features("test_image.png")

            # Verify all expected features are present
            expected_features = ['edge_density', 'unique_colors', 'entropy',
                               'corner_density', 'gradient_strength', 'complexity_score']

            for feature in expected_features:
                assert feature in features
                assert isinstance(features[feature], (int, float))
                assert 0 <= features[feature] <= 1

    @patch('backend.ai_modules.feature_extraction.cv2.imread')
    def test_load_and_validate_image_success(self, mock_imread, extractor):
        """Test successful image loading with mock"""
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        result = extractor._load_and_validate_image("test_image.png")

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, mock_image)

    @patch('backend.ai_modules.feature_extraction.cv2.imread')
    def test_load_and_validate_image_failure(self, mock_imread, extractor):
        """Test image loading failure"""
        mock_imread.return_value = None  # Simulates load failure

        with pytest.raises(ValueError, match="Invalid image format"):
            extractor._load_and_validate_image("invalid_image.png")

    def test_calculate_edge_density_basic(self, extractor, mock_image):
        """Test basic edge density calculation"""
        with patch.object(extractor, '_sobel_edge_density', return_value=0.3), \
             patch.object(extractor, '_laplacian_edge_density', return_value=0.4):

            result = extractor._calculate_edge_density(mock_image)
            assert isinstance(result, float)
            assert 0 <= result <= 1

    def test_sobel_edge_density_calculation(self, extractor):
        """Test Sobel edge density calculation"""
        # Create test image with clear edge
        test_image = np.zeros((50, 50), dtype=np.uint8)
        test_image[20:30, :] = 255  # Horizontal stripe

        result = extractor._sobel_edge_density(test_image)

        assert isinstance(result, float)
        assert result >= 0

    def test_laplacian_edge_density_calculation(self, extractor):
        """Test Laplacian edge density calculation"""
        # Create test image with clear edges
        test_image = np.zeros((50, 50), dtype=np.uint8)
        test_image[20:30, 20:30] = 255  # Square

        result = extractor._laplacian_edge_density(test_image)

        assert isinstance(result, float)
        assert result >= 0

    def test_count_unique_colors_methods(self, extractor, mock_image):
        """Test unique color counting methods"""
        with patch.object(extractor, '_quantized_color_count', return_value=10), \
             patch.object(extractor, '_fast_quantized_color_count', return_value=8), \
             patch.object(extractor, '_hsv_color_analysis', return_value=12), \
             patch.object(extractor, '_perceptual_color_clustering', return_value=6):

            result = extractor._count_unique_colors(mock_image)
            assert isinstance(result, float)
            assert 0 <= result <= 1

    def test_quantize_colors(self, extractor):
        """Test color quantization"""
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        result = extractor._quantize_colors(test_image, levels=8)

        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape

    def test_quantized_color_count(self, extractor):
        """Test quantized color counting"""
        # Create simple test image
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image[25:, :] = [255, 255, 255]  # Half white, half black

        result = extractor._quantized_color_count(test_image)

        assert isinstance(result, int)
        assert result >= 1

    def test_fast_quantized_color_count(self, extractor):
        """Test fast quantized color counting"""
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        result = extractor._fast_quantized_color_count(test_image)

        assert isinstance(result, int)
        assert result >= 1

    def test_hsv_color_analysis(self, extractor):
        """Test HSV color analysis"""
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        result = extractor._hsv_color_analysis(test_image)

        assert isinstance(result, int)
        assert result >= 1

    def test_perceptual_color_clustering(self, extractor):
        """Test perceptual color clustering"""
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        result = extractor._perceptual_color_clustering(test_image, max_clusters=5)

        assert isinstance(result, int)
        assert 1 <= result <= 5

    def test_calculate_entropy_methods(self, extractor, mock_image):
        """Test entropy calculation methods"""
        with patch.object(extractor, '_calculate_histogram_entropy', return_value=0.8):
            result = extractor._calculate_entropy(mock_image)

            assert isinstance(result, float)
            assert 0 <= result <= 1

    def test_calculate_histogram_entropy(self, extractor):
        """Test histogram entropy calculation"""
        # Create test grayscale image
        test_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

        result = extractor._calculate_histogram_entropy(test_image)

        assert isinstance(result, float)
        assert result >= 0

    def test_caching_enabled(self):
        """Test caching functionality"""
        extractor = ImageFeatureExtractor(cache_enabled=True, log_level="ERROR")

        # Mock all dependencies
        with patch.object(extractor, '_load_and_validate_image') as mock_load, \
             patch.object(extractor, '_calculate_edge_density', return_value=0.5), \
             patch.object(extractor, '_count_unique_colors', return_value=0.3), \
             patch.object(extractor, '_calculate_entropy', return_value=0.7), \
             patch.object(extractor, '_calculate_corner_density', return_value=0.4), \
             patch.object(extractor, '_calculate_gradient_strength', return_value=0.6), \
             patch.object(extractor, '_calculate_complexity_score', return_value=0.8), \
             patch('backend.ai_modules.feature_extraction.Path.exists', return_value=True):

            mock_image = np.zeros((50, 50, 3), dtype=np.uint8)
            mock_load.return_value = mock_image

            # First call
            features1 = extractor.extract_features("test_image.png")

            # Second call should use cache
            features2 = extractor.extract_features("test_image.png")

            # Results should be identical
            assert features1 == features2

            # Verify caching worked (load should only be called once)
            assert mock_load.call_count == 1

    def test_error_handling_edge_cases(self, extractor):
        """Test error handling for various edge cases"""
        # Test with very small image
        tiny_image = np.zeros((1, 1, 3), dtype=np.uint8)

        # These should not crash
        edge_density = extractor._calculate_edge_density(tiny_image)
        assert isinstance(edge_density, float)

        color_score = extractor._count_unique_colors(tiny_image)
        assert isinstance(color_score, float)

        entropy = extractor._calculate_entropy(tiny_image)
        assert isinstance(entropy, float)

    def test_normalization_functions(self, extractor):
        """Test that helper functions return normalized values"""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Test edge density normalization
        edge_density = extractor._calculate_edge_density(test_image)
        assert 0 <= edge_density <= 1

        # Test color count normalization
        color_score = extractor._count_unique_colors(test_image)
        assert 0 <= color_score <= 1

        # Test entropy normalization
        entropy = extractor._calculate_entropy(test_image)
        assert 0 <= entropy <= 1