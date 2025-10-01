#!/usr/bin/env python3
"""
Comprehensive tests for QualityMetrics module
Tests image quality assessment functionality
"""

import pytest
import numpy as np
import tempfile
import os
from PIL import Image
from unittest.mock import patch, MagicMock

from backend.utils.quality_metrics import QualityMetrics


class TestQualityMetrics:
    """Test suite for QualityMetrics class"""

    @pytest.fixture
    def sample_images(self):
        """Create sample test images"""
        # Create two similar 100x100 RGB images
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()

        # Make img2 slightly different
        img2[50:60, 50:60] = [255, 0, 0]  # Add red square

        return img1, img2

    @pytest.fixture
    def identical_images(self):
        """Create identical test images"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return img, img.copy()

    @pytest.fixture
    def rgba_images(self):
        """Create RGBA test images"""
        img1 = np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8)
        img2 = img1.copy()
        img2[25:, :, :3] = [255, 0, 0]  # Add red area

        return img1, img2

    def test_calculate_mse_identical_images(self, identical_images):
        """Test MSE calculation with identical images"""
        img1, img2 = identical_images
        mse = QualityMetrics.calculate_mse(img1, img2)

        assert mse == 0.0  # Identical images should have MSE = 0

    def test_calculate_mse_different_images(self, sample_images):
        """Test MSE calculation with different images"""
        img1, img2 = sample_images
        mse = QualityMetrics.calculate_mse(img1, img2)

        assert mse > 0  # Different images should have MSE > 0
        assert isinstance(mse, float)

    def test_calculate_mse_different_shapes(self):
        """Test MSE calculation with different image shapes"""
        img1 = np.zeros((50, 50, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Images must have the same dimensions"):
            QualityMetrics.calculate_mse(img1, img2)

    def test_calculate_mse_rgba_images(self, rgba_images):
        """Test MSE calculation with RGBA images"""
        img1, img2 = rgba_images
        mse = QualityMetrics.calculate_mse(img1, img2)

        assert isinstance(mse, float)
        assert mse >= 0

    def test_calculate_psnr_identical_images(self, identical_images):
        """Test PSNR calculation with identical images"""
        img1, img2 = identical_images
        psnr = QualityMetrics.calculate_psnr(img1, img2)

        assert psnr == float('inf')  # Identical images should have infinite PSNR

    def test_calculate_psnr_different_images(self, sample_images):
        """Test PSNR calculation with different images"""
        img1, img2 = sample_images
        psnr = QualityMetrics.calculate_psnr(img1, img2)

        assert isinstance(psnr, float)
        assert psnr > 0
        assert not np.isinf(psnr)

    def test_calculate_perceptual_loss(self, sample_images):
        """Test perceptual loss calculation"""
        img1, img2 = sample_images
        loss = QualityMetrics.calculate_perceptual_loss(img1, img2)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_calculate_perceptual_loss_identical(self, identical_images):
        """Test perceptual loss with identical images"""
        img1, img2 = identical_images
        loss = QualityMetrics.calculate_perceptual_loss(img1, img2)

        assert loss == 0.0  # Identical images should have zero perceptual loss

    def test_calculate_unified_score(self):
        """Test unified score calculation"""
        ssim = 0.8
        psnr = 25.0
        perceptual = 0.1
        mse = 100.0

        score = QualityMetrics.calculate_unified_score(ssim, psnr, perceptual, mse)

        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_calculate_unified_score_perfect(self):
        """Test unified score with perfect metrics"""
        ssim = 1.0
        psnr = float('inf')
        perceptual = 0.0
        mse = 0.0

        score = QualityMetrics.calculate_unified_score(ssim, psnr, perceptual, mse)

        assert score == 1.0  # Perfect metrics should give score = 1

    def test_calculate_ssim_identical_images(self, identical_images):
        """Test SSIM calculation with identical images"""
        img1, img2 = identical_images
        ssim = QualityMetrics.calculate_ssim(img1, img2)

        assert abs(ssim - 1.0) < 1e-6  # SSIM should be 1.0 for identical images

    def test_calculate_ssim_different_images(self, sample_images):
        """Test SSIM calculation with different images"""
        img1, img2 = sample_images
        ssim = QualityMetrics.calculate_ssim(img1, img2)

        assert isinstance(ssim, float)
        assert 0 <= ssim <= 1

    def test_calculate_ssim_with_custom_params(self, sample_images):
        """Test SSIM calculation with custom parameters"""
        img1, img2 = sample_images
        ssim = QualityMetrics.calculate_ssim(img1, img2, k1=0.1, k2=0.2, win_size=7)

        assert isinstance(ssim, float)
        assert 0 <= ssim <= 1

    def test_window_mean(self):
        """Test window mean calculation"""
        img = np.ones((10, 10), dtype=np.float64) * 100
        mean = QualityMetrics._window_mean(img, win_size=3)

        assert isinstance(mean, np.ndarray)
        assert mean.shape == img.shape
        assert np.allclose(mean, 100)  # Should be close to 100 for uniform image

    def test_calculate_edge_similarity(self, sample_images):
        """Test edge similarity calculation"""
        img1, img2 = sample_images
        similarity = QualityMetrics.calculate_edge_similarity(img1, img2)

        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1

    def test_calculate_edge_similarity_identical(self, identical_images):
        """Test edge similarity with identical images"""
        img1, img2 = identical_images
        similarity = QualityMetrics.calculate_edge_similarity(img1, img2)

        assert abs(similarity - 1.0) < 1e-6  # Should be 1.0 for identical images

    def test_sobel_edges(self):
        """Test Sobel edge detection"""
        # Create image with clear edge
        img = np.zeros((50, 50), dtype=np.uint8)
        img[20:30, :] = 255  # Horizontal stripe

        edges = QualityMetrics._sobel_edges(img)

        assert isinstance(edges, np.ndarray)
        assert edges.shape == img.shape
        assert edges.max() > 0  # Should detect edges

    def test_calculate_color_accuracy(self, sample_images):
        """Test color accuracy calculation"""
        img1, img2 = sample_images
        accuracy = QualityMetrics.calculate_color_accuracy(img1, img2)

        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_calculate_color_accuracy_identical(self, identical_images):
        """Test color accuracy with identical images"""
        img1, img2 = identical_images
        accuracy = QualityMetrics.calculate_color_accuracy(img1, img2)

        assert abs(accuracy - 1.0) < 1e-6  # Should be 1.0 for identical images

    @patch('cairosvg.svg2png')
    def test_render_to_array(self, mock_svg2png):
        """Test SVG rendering to array with mock"""
        # Mock the cairo SVG rendering
        mock_png_data = b'\x89PNG\r\n\x1a\n' + b'mock_png_data'
        mock_svg2png.return_value = mock_png_data

        # Mock PIL Image.open to return a test image
        with patch('PIL.Image.open') as mock_open:
            mock_image = Image.new('RGB', (100, 100), color='red')
            mock_open.return_value = mock_image

            svg_content = '<svg><rect width="100" height="100" fill="red"/></svg>'
            result = QualityMetrics.render_to_array(svg_content, 100, 100)

            assert isinstance(result, np.ndarray)
            assert result.shape[2] == 3  # RGB

    @patch('cairosvg.svg2png')
    def test_svg_to_png_success(self, mock_svg2png):
        """Test successful SVG to PNG conversion"""
        # Create a mock PNG image
        test_image = Image.new('RGB', (256, 256), color='blue')
        png_buffer = io.BytesIO()
        test_image.save(png_buffer, format='PNG')
        mock_png_data = png_buffer.getvalue()

        mock_svg2png.return_value = mock_png_data

        svg_content = '<svg><rect width="256" height="256" fill="blue"/></svg>'
        result = QualityMetrics.svg_to_png(svg_content)

        assert isinstance(result, np.ndarray)
        assert result.shape == (256, 256, 3)

    @patch('cairosvg.svg2png')
    def test_svg_to_png_failure(self, mock_svg2png):
        """Test SVG to PNG conversion failure"""
        mock_svg2png.side_effect = Exception("Conversion failed")

        svg_content = '<svg><rect width="100" height="100" fill="red"/></svg>'
        result = QualityMetrics.svg_to_png(svg_content)

        assert result is None

    def test_comprehensive_metrics_init(self):
        """Test ComprehensiveMetrics initialization"""
        from backend.utils.quality_metrics import ComprehensiveMetrics
        metrics = ComprehensiveMetrics()

        assert metrics is not None

    def test_comprehensive_metrics_compare_images(self):
        """Test ComprehensiveMetrics compare_images method"""
        from backend.utils.quality_metrics import ComprehensiveMetrics

        # Create temporary image files
        img = Image.new('RGB', (100, 100), color='red')

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp1:
            img.save(tmp1.name)
            original_path = tmp1.name

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp2:
            img.save(tmp2.name)
            svg_path = tmp2.name

        try:
            metrics = ComprehensiveMetrics()
            result = metrics.compare_images(original_path, svg_path)

            assert isinstance(result, dict)
            assert 'ssim' in result
            assert 'mse' in result
            assert 'psnr' in result
            assert all(isinstance(v, (int, float)) for v in result.values())

        finally:
            os.unlink(original_path)
            os.unlink(svg_path)

    @patch('backend.utils.quality_metrics.QualityMetrics.svg_to_png')
    def test_comprehensive_metrics_evaluate(self, mock_svg_to_png):
        """Test ComprehensiveMetrics evaluate method"""
        from backend.utils.quality_metrics import ComprehensiveMetrics

        # Create temporary PNG file
        img = Image.new('RGB', (100, 100), color='green')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            png_path = tmp.name

        # Mock SVG to PNG conversion
        mock_svg_to_png.return_value = np.array(img)

        try:
            metrics = ComprehensiveMetrics()
            svg_content = '<svg><rect width="100" height="100" fill="green"/></svg>'

            result = metrics.evaluate(png_path, svg_content)

            assert isinstance(result, dict)
            assert 'overall_score' in result
            assert 'detailed_metrics' in result
            assert isinstance(result['overall_score'], (int, float))

        finally:
            os.unlink(png_path)

    def test_analyze_svg_complexity(self):
        """Test SVG complexity analysis"""
        from backend.utils.quality_metrics import ComprehensiveMetrics

        metrics = ComprehensiveMetrics()

        # Simple SVG
        simple_svg = '<svg><rect width="100" height="100" fill="red"/></svg>'
        complexity = metrics._analyze_svg_complexity(simple_svg)

        assert isinstance(complexity, dict)
        assert 'path_count' in complexity
        assert 'complexity_score' in complexity

        # Complex SVG
        complex_svg = '''<svg>
            <path d="M10,10 L50,50 C75,25 100,50 125,25"/>
            <circle cx="50" cy="50" r="20"/>
            <polygon points="10,10 20,20 30,10"/>
        </svg>'''

        complexity_complex = metrics._analyze_svg_complexity(complex_svg)
        assert complexity_complex['path_count'] > complexity['path_count']

    def test_grayscale_conversion(self, sample_images):
        """Test grayscale image handling"""
        img1, img2 = sample_images

        # Convert to grayscale
        gray1 = np.mean(img1, axis=2).astype(np.uint8)
        gray2 = np.mean(img2, axis=2).astype(np.uint8)

        # Should work with grayscale images
        mse = QualityMetrics.calculate_mse(gray1, gray2)
        assert isinstance(mse, float)
        assert mse >= 0

    def test_edge_cases_small_images(self):
        """Test edge cases with very small images"""
        # 1x1 images
        img1 = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Red pixel
        img2 = np.array([[[0, 255, 0]]], dtype=np.uint8)  # Green pixel

        mse = QualityMetrics.calculate_mse(img1, img2)
        assert mse > 0

        psnr = QualityMetrics.calculate_psnr(img1, img2)
        assert isinstance(psnr, float)

        # Should not crash with small windows
        ssim = QualityMetrics.calculate_ssim(img1, img2, win_size=1)
        assert isinstance(ssim, float)

    def test_edge_cases_extreme_values(self):
        """Test edge cases with extreme pixel values"""
        # All black vs all white
        black = np.zeros((50, 50, 3), dtype=np.uint8)
        white = np.full((50, 50, 3), 255, dtype=np.uint8)

        mse = QualityMetrics.calculate_mse(black, white)
        assert mse == 255**2  # Maximum possible MSE

        psnr = QualityMetrics.calculate_psnr(black, white)
        assert isinstance(psnr, float)
        assert psnr > 0

    def test_error_handling_invalid_inputs(self):
        """Test error handling with invalid inputs"""
        # Test with None inputs
        with pytest.raises((ValueError, TypeError)):
            QualityMetrics.calculate_mse(None, None)

        # Test with mismatched data types
        img1 = np.ones((50, 50, 3), dtype=np.uint8)
        img2 = np.ones((50, 50, 3), dtype=np.float64)

        # Should handle different dtypes gracefully
        mse = QualityMetrics.calculate_mse(img1, img2)
        assert isinstance(mse, float)

    def test_memory_efficiency_large_images(self):
        """Test memory efficiency with larger images"""
        # Create moderately large images
        img1 = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        img2 = img1 + np.random.randint(-10, 10, img1.shape, dtype=np.int16)
        img2 = np.clip(img2, 0, 255).astype(np.uint8)

        # Should complete without memory issues
        mse = QualityMetrics.calculate_mse(img1, img2)
        assert isinstance(mse, float)

        psnr = QualityMetrics.calculate_psnr(img1, img2)
        assert isinstance(psnr, float)

        # SSIM might be slower but should still work
        ssim = QualityMetrics.calculate_ssim(img1, img2)
        assert isinstance(ssim, float)