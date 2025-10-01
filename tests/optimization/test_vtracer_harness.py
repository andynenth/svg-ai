"""Unit tests for VTracer Test Harness"""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from backend.ai_modules.optimization import OptimizationEngine
from backend.ai_modules.optimization import OptimizationEngine


class TestVTracerTestHarness:
    """Test suite for VTracerTestHarness"""

    def setup_method(self):
        """Setup for each test"""
        self.harness = OptimizationEngine()(timeout=10)

    def test_initialization(self):
        """Test harness initialization"""
        harness = OptimizationEngine()(timeout=15)
        assert harness.timeout == 15
        assert harness.results_cache == {}
        assert harness.validator is not None
        assert harness.executor is not None

    def test_cache_key_generation(self, sample_vtracer_params):
        """Test cache key generation is consistent"""
        image_path = "test.png"

        key1 = self.harness._generate_cache_key(image_path, sample_vtracer_params)
        key2 = self.harness._generate_cache_key(image_path, sample_vtracer_params)

        assert key1 == key2
        assert len(key1) == 32  # MD5 hash length

    def test_cache_key_different_params(self, sample_vtracer_params):
        """Test cache keys differ for different parameters"""
        image_path = "test.png"

        key1 = self.harness._generate_cache_key(image_path, sample_vtracer_params)

        modified_params = sample_vtracer_params.copy()
        modified_params['color_precision'] = 8

        key2 = self.harness._generate_cache_key(image_path, modified_params)

        assert key1 != key2

    @patch('backend.ai_modules.optimization.vtracer_test.vtracer')
    def test_successful_conversion(self, mock_vtracer, sample_vtracer_params, test_data_dir):
        """Test successful parameter testing"""
        # Setup mock
        mock_vtracer.convert_image_to_svg_py.return_value = None

        # Use first test image
        test_images = list((test_data_dir / "simple").glob("*.png"))
        if not test_images:
            pytest.skip("No test images available")

        image_path = str(test_images[0])

        # Mock quality metrics calculation
        with patch.object(self.harness, '_calculate_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "ssim": 0.95,
                "mse": 100.0,
                "psnr": 28.0
            }

            result = self.harness.test_parameters(image_path, sample_vtracer_params)

        assert result["success"] is True
        assert result["error"] is None
        assert result["metrics"]["ssim"] == 0.95
        assert "performance" in result
        assert "conversion_time" in result["performance"]

    def test_invalid_parameters(self, invalid_vtracer_params):
        """Test handling of invalid parameters"""
        result = self.harness.test_parameters("test.png", invalid_vtracer_params)

        assert result["success"] is False
        assert result["error"] is not None
        assert "Invalid parameters" in result["error"]

    @patch('backend.ai_modules.optimization.vtracer_test.vtracer')
    def test_conversion_timeout(self, mock_vtracer, sample_vtracer_params):
        """Test timeout handling"""
        import time

        # Make conversion take too long
        def slow_conversion(*args, **kwargs):
            time.sleep(2)

        mock_vtracer.convert_image_to_svg_py.side_effect = slow_conversion

        # Use very short timeout
        harness = OptimizationEngine()(timeout=0.1)
        result = harness.test_parameters("test.png", sample_vtracer_params)

        assert result["success"] is False
        assert "timeout" in result["error"].lower()

    @patch('backend.ai_modules.optimization.vtracer_test.vtracer')
    def test_conversion_error_handling(self, mock_vtracer, sample_vtracer_params):
        """Test error handling during conversion"""
        # Make conversion raise an error
        mock_vtracer.convert_image_to_svg_py.side_effect = RuntimeError("VTracer error")

        result = self.harness.test_parameters("test.png", sample_vtracer_params)

        assert result["success"] is False
        assert result["error"] is not None
        assert "VTracer error" in result["error"]

    def test_cache_functionality(self, sample_vtracer_params):
        """Test result caching"""
        image_path = "test.png"

        # Mock the actual conversion
        with patch.object(self.harness, '_run_conversion_with_timeout') as mock_convert:
            mock_convert.return_value = {
                "success": True,
                "metrics": {"ssim": 0.9},
                "performance": {"conversion_time": 1.0},
                "svg_path": "test.svg"
            }

            # First call - should run conversion
            result1 = self.harness.test_parameters(image_path, sample_vtracer_params)
            assert mock_convert.call_count == 1

            # Second call - should use cache
            result2 = self.harness.test_parameters(image_path, sample_vtracer_params)
            assert mock_convert.call_count == 1  # Not called again

            assert result1 == result2

    def test_clear_cache(self, sample_vtracer_params):
        """Test cache clearing"""
        # Add something to cache
        cache_key = self.harness._generate_cache_key("test.png", sample_vtracer_params)
        self.harness.results_cache[cache_key] = {"test": "data"}

        assert len(self.harness.results_cache) == 1

        self.harness.clear_cache()

        assert len(self.harness.results_cache) == 0

    def test_cache_stats(self, sample_vtracer_params):
        """Test cache statistics"""
        # Add items to cache
        for i in range(3):
            cache_key = self.harness._generate_cache_key(f"test{i}.png", sample_vtracer_params)
            self.harness.results_cache[cache_key] = {"test": i}

        stats = self.harness.get_cache_stats()

        assert stats["cache_size"] == 3
        assert len(stats["cache_entries"]) == 3

    @patch('backend.ai_modules.optimization.vtracer_test.vtracer')
    def test_batch_testing(self, mock_vtracer, sample_vtracer_params):
        """Test batch parameter testing"""
        mock_vtracer.convert_image_to_svg_py.return_value = None

        image_paths = ["test1.png", "test2.png"]
        param_sets = [
            sample_vtracer_params,
            {**sample_vtracer_params, "color_precision": 8}
        ]

        with patch.object(self.harness, '_calculate_quality_metrics') as mock_metrics:
            mock_metrics.return_value = {"ssim": 0.9, "mse": 100.0, "psnr": 28.0}

            results = self.harness.batch_test(image_paths, param_sets, parallel=2)

        assert len(results) == 4  # 2 images x 2 param sets
        assert all("image_path" in r for r in results)
        assert all("parameters" in r for r in results)

    def test_save_results(self, tmp_path):
        """Test saving results to file"""
        results = [
            {"test": 1, "success": True},
            {"test": 2, "success": False}
        ]

        output_file = tmp_path / "test_results.json"
        self.harness.save_results(results, str(output_file))

        assert output_file.exists()

        with open(output_file) as f:
            loaded_results = json.load(f)

        assert len(loaded_results) == 2
        assert loaded_results[0]["test"] == 1

    def test_analyze_results_successful(self):
        """Test analysis of successful results"""
        results = [
            {
                "success": True,
                "metrics": {"ssim": 0.9, "mse": 100, "psnr": 28},
                "performance": {"conversion_time": 1.0, "file_size_reduction": 0.5}
            },
            {
                "success": True,
                "metrics": {"ssim": 0.95, "mse": 80, "psnr": 30},
                "performance": {"conversion_time": 1.5, "file_size_reduction": 0.6}
            }
        ]

        analysis = self.harness.analyze_results(results)

        assert analysis["total_tests"] == 2
        assert analysis["successful"] == 2
        assert analysis["failed"] == 0
        assert analysis["success_rate"] == 1.0
        assert 0.92 < analysis["quality"]["avg_ssim"] < 0.93
        assert analysis["performance"]["avg_conversion_time"] == 1.25

    def test_analyze_results_with_failures(self):
        """Test analysis with failed results"""
        results = [
            {"success": True, "metrics": {"ssim": 0.9}, "performance": {"conversion_time": 1.0, "file_size_reduction": 0.5}},
            {"success": False, "error": "Timeout: conversion failed"},
            {"success": False, "error": "Invalid: parameters wrong"}
        ]

        analysis = self.harness.analyze_results(results)

        assert analysis["total_tests"] == 3
        assert analysis["successful"] == 1
        assert analysis["failed"] == 2
        assert analysis["success_rate"] == 1/3
        assert "error_summary" in analysis
        assert "Timeout" in analysis["error_summary"]
        assert "Invalid" in analysis["error_summary"]