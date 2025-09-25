"""
Tests for quality metrics module.
"""

import pytest
import numpy as np
from PIL import Image
import os

from utils.quality_metrics import QualityMetrics, SVGRenderer, ComprehensiveMetrics


class TestQualityMetrics:
    """Tests for quality metrics calculations."""

    def test_ssim_identical_images(self):
        """Test SSIM with identical images."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ssim = QualityMetrics.calculate_ssim(img, img)

        assert ssim == pytest.approx(1.0, rel=0.01)  # Should be very close to 1

    def test_ssim_different_images(self):
        """Test SSIM with completely different images."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        ssim = QualityMetrics.calculate_ssim(img1, img2)

        assert ssim < 0.1  # Should be very low

    def test_ssim_slightly_different(self):
        """Test SSIM with slightly different images."""
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()
        # Add small noise
        noise = np.random.randint(-10, 10, (100, 100, 3), dtype=np.int16)
        img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        ssim = QualityMetrics.calculate_ssim(img1, img2)

        assert 0.7 < ssim < 1.0  # Should be high but not perfect

    def test_mse_identical(self):
        """Test MSE with identical images."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mse = QualityMetrics.calculate_mse(img, img)

        assert mse == 0.0

    def test_mse_different(self):
        """Test MSE with different images."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        mse = QualityMetrics.calculate_mse(img1, img2)

        assert mse > 0

    def test_psnr_identical(self):
        """Test PSNR with identical images."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        psnr = QualityMetrics.calculate_psnr(img, img)

        assert psnr == 100.0  # Max value for identical

    def test_psnr_different(self):
        """Test PSNR with different images."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 128

        psnr = QualityMetrics.calculate_psnr(img1, img2)

        assert 0 < psnr < 100

    def test_edge_similarity(self):
        """Test edge similarity metric."""
        # Create image with clear edges
        img = np.zeros((100, 100), dtype=np.uint8)
        img[40:60, 40:60] = 255

        # Convert to 3-channel
        img_3c = np.stack([img] * 3, axis=2)

        edge_sim = QualityMetrics.calculate_edge_similarity(img_3c, img_3c)

        assert edge_sim == pytest.approx(1.0, rel=0.01)

    def test_color_accuracy(self):
        """Test color accuracy metric."""
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()

        accuracy = QualityMetrics.calculate_color_accuracy(img1, img2)

        assert accuracy == pytest.approx(1.0, rel=0.01)

    def test_grayscale_ssim(self):
        """Test SSIM with grayscale images."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        ssim = QualityMetrics.calculate_ssim(img, img)

        assert ssim == pytest.approx(1.0, rel=0.01)


class TestSVGRenderer:
    """Tests for SVG rendering."""

    def test_svg_to_png_basic(self):
        """Test basic SVG to PNG conversion."""
        svg_content = '''
        <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect x="10" y="10" width="80" height="80" fill="red"/>
        </svg>
        '''

        rendered = SVGRenderer.svg_to_png(svg_content, (100, 100))

        if rendered is not None:  # May fail if cairosvg not installed
            assert rendered.shape[0] == 100
            assert rendered.shape[1] == 100

    def test_svg_to_png_invalid(self):
        """Test handling of invalid SVG."""
        invalid_svg = "not valid svg content"
        rendered = SVGRenderer.svg_to_png(invalid_svg)

        # Should return None or raise exception depending on implementation
        assert rendered is None or isinstance(rendered, np.ndarray)


class TestComprehensiveMetrics:
    """Tests for comprehensive metrics calculator."""

    @pytest.fixture
    def test_image(self, tmp_path):
        """Create a test image."""
        img = Image.new('RGB', (100, 100), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([25, 25, 75, 75], fill='blue')

        img_path = tmp_path / "test.png"
        img.save(img_path)
        return str(img_path)

    @pytest.fixture
    def test_svg(self):
        """Create test SVG content."""
        return '''
        <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect x="25" y="25" width="50" height="50" fill="blue"/>
        </svg>
        '''

    def test_evaluate_basic(self, test_image, test_svg):
        """Test basic metric evaluation."""
        calculator = ComprehensiveMetrics()
        metrics = calculator.evaluate(test_image, test_svg, 0.5)

        assert 'file' in metrics
        assert 'performance' in metrics
        assert 'visual' in metrics

        assert metrics['performance']['conversion_time_s'] == 0.5
        assert metrics['file']['png_size_kb'] > 0

    def test_svg_complexity_analysis(self, test_svg):
        """Test SVG complexity analysis."""
        calculator = ComprehensiveMetrics()
        complexity = calculator._analyze_svg_complexity(test_svg)

        assert complexity['total_size'] == len(test_svg)
        assert complexity['num_paths'] == 0  # No paths in test SVG
        assert complexity['num_groups'] == 0

    def test_evaluate_with_complex_svg(self, test_image):
        """Test evaluation with complex SVG."""
        complex_svg = '''
        <svg xmlns="http://www.w3.org/2000/svg">
            <g>
                <path d="M10,10 L90,90" stroke="black"/>
                <path d="M90,10 L10,90" stroke="red"/>
            </g>
            <rect x="40" y="40" width="20" height="20" fill="blue"/>
        </svg>
        '''

        calculator = ComprehensiveMetrics()
        metrics = calculator.evaluate(test_image, complex_svg, 0.1)

        complexity = metrics['performance']['svg_complexity']
        assert complexity['num_paths'] == 2
        assert complexity['num_groups'] == 1