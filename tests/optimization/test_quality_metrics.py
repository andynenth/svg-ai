"""Unit tests for quality metrics system"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from backend.ai_modules.optimization.quality_metrics import OptimizationQualityMetrics
from backend.ai_modules.optimization.parameter_bounds import VTracerParameterBounds


class TestOptimizationQualityMetrics:
    """Test suite for optimization quality metrics"""

    def setup_method(self):
        """Setup for each test"""
        self.metrics = OptimizationQualityMetrics()

    def teardown_method(self):
        """Cleanup after each test"""
        self.metrics.cleanup()

    def test_initialization(self):
        """Test metrics system initialization"""
        metrics = OptimizationQualityMetrics()
        assert metrics.temp_dir is not None
        assert metrics.harness is not None
        assert metrics.bounds is not None
        metrics.cleanup()

    def test_average_metrics(self):
        """Test metrics averaging"""
        results = [
            {
                "success": True,
                "metrics": {"ssim": 0.9, "mse": 100, "psnr": 28},
                "performance": {
                    "conversion_time": 1.0,
                    "file_size_reduction": 0.5,
                    "svg_size_bytes": 5000
                }
            },
            {
                "success": True,
                "metrics": {"ssim": 0.95, "mse": 80, "psnr": 30},
                "performance": {
                    "conversion_time": 1.2,
                    "file_size_reduction": 0.6,
                    "svg_size_bytes": 4500
                }
            }
        ]

        avg = self.metrics._average_metrics(results)

        assert avg["success_rate"] == 1.0
        assert 0.92 < avg["ssim"] < 0.93
        assert avg["mse"] == 90
        assert avg["psnr"] == 29
        assert avg["conversion_time"] == 1.1
        assert "ssim_std" in avg

    def test_average_metrics_with_failures(self):
        """Test averaging with some failed results"""
        results = [
            {"success": False, "error": "Failed"},
            {
                "success": True,
                "metrics": {"ssim": 0.9, "mse": 100, "psnr": 28},
                "performance": {
                    "conversion_time": 1.0,
                    "file_size_reduction": 0.5,
                    "svg_size_bytes": 5000
                }
            }
        ]

        avg = self.metrics._average_metrics(results)

        assert avg["success_rate"] == 0.5
        assert avg["ssim"] == 0.9

    def test_calculate_improvements(self):
        """Test improvement calculation"""
        default_metrics = {
            "ssim": 0.8,
            "mse": 150,
            "psnr": 25,
            "svg_size_bytes": 10000,
            "conversion_time": 2.0
        }

        optimized_metrics = {
            "ssim": 0.9,
            "mse": 100,
            "psnr": 28,
            "svg_size_bytes": 8000,
            "conversion_time": 1.5
        }

        improvements = self.metrics._calculate_improvements(
            default_metrics, optimized_metrics
        )

        assert abs(improvements["ssim_improvement"] - 12.5) < 0.01  # (0.9-0.8)/0.8 * 100
        assert abs(improvements["ssim_absolute"] - 0.1) < 0.001
        assert abs(improvements["file_size_improvement"] - 20.0) < 0.01  # (10000-8000)/10000 * 100
        assert abs(improvements["speed_improvement"] - 25.0) < 0.01  # (2.0-1.5)/2.0 * 100
        assert improvements["mse_improvement"] > 0
        assert improvements["psnr_improvement"] > 0

    def test_calculate_improvements_edge_cases(self):
        """Test improvement calculation with edge cases"""
        # Zero baseline
        default_metrics = {"ssim": 0.0, "svg_size_bytes": 0}
        optimized_metrics = {"ssim": 0.5, "svg_size_bytes": 1000}

        improvements = self.metrics._calculate_improvements(
            default_metrics, optimized_metrics
        )

        assert improvements["ssim_improvement"] == 100.0

    def test_statistical_significance(self):
        """Test statistical significance testing"""
        default_results = [
            {"success": True, "metrics": {"ssim": 0.80},
             "performance": {"conversion_time": 2.0}},
            {"success": True, "metrics": {"ssim": 0.82},
             "performance": {"conversion_time": 2.1}},
            {"success": True, "metrics": {"ssim": 0.81},
             "performance": {"conversion_time": 1.9}}
        ]

        optimized_results = [
            {"success": True, "metrics": {"ssim": 0.90},
             "performance": {"conversion_time": 1.5}},
            {"success": True, "metrics": {"ssim": 0.92},
             "performance": {"conversion_time": 1.4}},
            {"success": True, "metrics": {"ssim": 0.91},
             "performance": {"conversion_time": 1.6}}
        ]

        significance = self.metrics._test_significance(
            default_results, optimized_results
        )

        assert "ssim_t_statistic" in significance
        assert "ssim_p_value" in significance
        assert "ssim_significant" in significance
        assert significance["ssim_significant"] == True  # Should be significant improvement
        assert "ssim_improvement_ci" in significance
        assert len(significance["ssim_improvement_ci"]) == 2

    def test_statistical_significance_insufficient_data(self):
        """Test significance testing with insufficient data"""
        default_results = [
            {"success": True, "metrics": {"ssim": 0.80},
             "performance": {"conversion_time": 2.0}}
        ]

        optimized_results = [
            {"success": True, "metrics": {"ssim": 0.90},
             "performance": {"conversion_time": 1.5}}
        ]

        significance = self.metrics._test_significance(
            default_results, optimized_results
        )

        # Should not have statistical tests with only 1 sample
        assert "ssim_t_statistic" not in significance

    @patch('cv2.imread')
    @patch('cairosvg.svg2png')
    def test_edge_preservation_measurement(self, mock_svg2png, mock_imread):
        """Test edge preservation measurement"""
        # Create mock image data
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mock_imread.return_value = test_image
        mock_svg2png.return_value = test_image.tobytes()

        score = self.metrics._measure_edge_preservation("test.png", "test.svg")

        assert 0 <= score <= 1
        mock_imread.assert_called_once()
        mock_svg2png.assert_called_once()

    @patch('PIL.Image.open')
    @patch('cairosvg.svg2png')
    def test_color_accuracy_measurement(self, mock_svg2png, mock_image_open):
        """Test color accuracy measurement"""
        # Create mock image
        mock_img = MagicMock()
        mock_img.width = 100
        mock_img.height = 100
        mock_img.convert.return_value = mock_img
        mock_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_img.__array__ = lambda: mock_array

        mock_image_open.return_value = mock_img
        mock_svg2png.return_value = b'fake_png_data'

        score = self.metrics._measure_color_accuracy("test.png", "test.svg")

        assert 0 <= score <= 1

    @patch('cv2.imread')
    @patch('cairosvg.svg2png')
    def test_shape_fidelity_measurement(self, mock_svg2png, mock_imread):
        """Test shape fidelity measurement"""
        # Create mock image with simple shape
        test_image = np.zeros((100, 100), dtype=np.uint8)
        test_image[25:75, 25:75] = 255  # White square

        mock_imread.return_value = test_image
        mock_svg2png.return_value = test_image.tobytes()

        score = self.metrics._measure_shape_fidelity("test.png", "test.svg")

        assert 0 <= score <= 1

    @patch.object(OptimizationQualityMetrics, '_assess_visual_quality')
    @patch('backend.ai_modules.optimization.quality_metrics.VTracerTestHarness')
    def test_measure_improvement(self, mock_harness_class, mock_visual):
        """Test full improvement measurement"""
        # Setup mock harness
        mock_harness = MagicMock()
        mock_harness_class.return_value = mock_harness

        # Mock test results
        default_result = {
            "success": True,
            "metrics": {"ssim": 0.8, "mse": 150, "psnr": 25},
            "performance": {
                "conversion_time": 2.0,
                "file_size_reduction": 0.3,
                "svg_size_bytes": 10000
            },
            "svg_path": "default.svg"
        }

        optimized_result = {
            "success": True,
            "metrics": {"ssim": 0.9, "mse": 100, "psnr": 28},
            "performance": {
                "conversion_time": 1.5,
                "file_size_reduction": 0.4,
                "svg_size_bytes": 8000
            },
            "svg_path": "optimized.svg"
        }

        mock_harness.test_parameters.side_effect = [
            default_result, optimized_result,
            default_result, optimized_result,
            default_result, optimized_result
        ]

        mock_visual.return_value = {
            "edge_preservation_default": 0.8,
            "edge_preservation_optimized": 0.9,
            "color_accuracy_default": 0.85,
            "color_accuracy_optimized": 0.92
        }

        # Create new metrics instance with mocked harness
        metrics = OptimizationQualityMetrics()
        metrics.harness = mock_harness

        default_params = VTracerParameterBounds.get_default_parameters()
        optimized_params = default_params.copy()
        optimized_params["color_precision"] = 8

        result = metrics.measure_improvement(
            "test.png", default_params, optimized_params, runs=3
        )

        assert "default_metrics" in result
        assert "optimized_metrics" in result
        assert "improvements" in result
        assert "statistical_significance" in result
        assert "visual_quality" in result

        assert abs(result["improvements"]["ssim_improvement"] - 12.5) < 0.01

        metrics.cleanup()

    def test_generate_json_report(self):
        """Test JSON report generation"""
        test_data = {
            "image_path": "test.png",
            "improvements": {
                "ssim_improvement": 10.0,
                "file_size_improvement": 20.0
            }
        }

        report = self.metrics.generate_quality_report(test_data, "json")
        assert isinstance(report, str)

        import json
        parsed = json.loads(report)
        assert parsed["image_path"] == "test.png"
        assert parsed["improvements"]["ssim_improvement"] == 10.0

    def test_generate_html_report(self):
        """Test HTML report generation"""
        test_data = {
            "image_path": "test.png",
            "improvements": {
                "ssim_improvement": 10.0,
                "file_size_improvement": 20.0
            },
            "default_metrics": {
                "ssim": 0.8,
                "conversion_time": 2.0
            },
            "optimized_metrics": {
                "ssim": 0.88,
                "conversion_time": 1.5
            },
            "statistical_significance": {
                "ssim_significant": True,
                "ssim_p_value": 0.01
            }
        }

        report = self.metrics.generate_quality_report(test_data, "html")

        assert isinstance(report, str)
        assert "<html>" in report
        assert "test.png" in report
        assert "10.00%" in report or "10.0%" in report
        assert "Statistical Significance" in report

    def test_cleanup(self):
        """Test cleanup of temporary files"""
        metrics = OptimizationQualityMetrics()
        temp_dir = metrics.temp_dir

        assert Path(temp_dir).exists()

        metrics.cleanup()

        assert not Path(temp_dir).exists()