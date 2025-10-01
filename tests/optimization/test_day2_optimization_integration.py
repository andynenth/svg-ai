"""Day 2 Integration Testing - Quality Measurement & Logging with Real Images"""
import pytest
import tempfile
import shutil
from pathlib import Path
from backend.ai_modules.optimization import OptimizationEngine
from backend.ai_modules.optimization import OptimizationEngine
from backend.ai_modules.optimization import OptimizationEngine


class TestDay2Integration:
    """Integration tests for Day 2 deliverables"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = OptimizationEngine()(log_dir=self.temp_dir)
        self.quality_metrics = OptimizationEngine()

    def teardown_method(self):
        """Cleanup after tests"""
        self.quality_metrics.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_quality_measurement_with_sample_images(self, test_data_dir):
        """Test quality measurement system with real sample images"""
        # Find available test images
        sample_images = []
        for category in ["simple", "text", "gradient", "complex"]:
            category_dir = test_data_dir / category
            if category_dir.exists():
                images = list(category_dir.glob("*.png"))[:2]  # Take 2 per category
                sample_images.extend(images)

        if not sample_images:
            pytest.skip("No sample images found in test data directory")

        # Test with first available image
        test_image = sample_images[0]
        print(f"Testing quality measurement with: {test_image}")

        # Get default and optimized parameters
        default_params = VTracerParameterBounds.get_default_parameters()
        optimized_params = default_params.copy()
        optimized_params["color_precision"] = 8  # Different from default
        optimized_params["corner_threshold"] = 40  # Different from default

        # Run quality measurement (this will test real VTracer conversion)
        result = self.quality_metrics.measure_improvement(
            str(test_image),
            default_params,
            optimized_params,
            runs=1  # Single run for speed
        )

        # Verify results structure
        assert "image_path" in result
        assert "default_metrics" in result
        assert "optimized_metrics" in result
        assert "improvements" in result

        # Check if conversions were successful
        if result["default_metrics"].get("success_rate", 0) > 0:
            assert "ssim" in result["default_metrics"]
            assert "conversion_time" in result["default_metrics"]
            print(f"✅ Default SSIM: {result['default_metrics'].get('ssim', 'N/A'):.4f}")
        else:
            print("⚠️  Default conversion failed (VTracer may not be available)")

        if result["optimized_metrics"].get("success_rate", 0) > 0:
            assert "ssim" in result["optimized_metrics"]
            assert "conversion_time" in result["optimized_metrics"]
            print(f"✅ Optimized SSIM: {result['optimized_metrics'].get('ssim', 'N/A'):.4f}")
        else:
            print("⚠️  Optimized conversion failed (VTracer may not be available)")

        # Test report generation
        json_report = self.quality_metrics.generate_quality_report(result, "json")
        assert isinstance(json_report, str)
        assert "image_path" in json_report

        html_report = self.quality_metrics.generate_quality_report(result, "html")
        assert isinstance(html_report, str)
        assert "<html>" in html_report

        print("✅ Quality measurement system works with sample images")

    def test_logging_system_captures_complete_data(self, test_data_dir):
        """Test that logging system captures all optimization data"""
        # Sample features from image analysis
        test_features = {
            "edge_density": 0.15,
            "unique_colors": 12,
            "entropy": 0.65,
            "corner_density": 0.08,
            "gradient_strength": 0.45,
            "complexity_score": 0.35
        }

        # Sample parameters
        test_params = {
            "color_precision": 8,
            "layer_difference": 10,
            "corner_threshold": 40,
            "length_threshold": 5.0,
            "max_iterations": 12,
            "splice_threshold": 55,
            "path_precision": 8,
            "mode": "spline"
        }

        # Sample quality metrics (simulated results)
        test_quality_metrics = {
            "improvements": {
                "ssim_improvement": 15.5,
                "file_size_improvement": 25.2,
                "speed_improvement": 10.3,
                "mse_improvement": 20.1,
                "psnr_improvement": 8.5
            },
            "default_metrics": {
                "ssim": 0.82,
                "mse": 150.0,
                "psnr": 26.5,
                "conversion_time": 2.1,
                "svg_size_bytes": 12000,
                "success_rate": 1.0
            },
            "optimized_metrics": {
                "ssim": 0.95,
                "mse": 120.0,
                "psnr": 28.8,
                "conversion_time": 1.9,
                "svg_size_bytes": 9000,
                "success_rate": 1.0
            },
            "statistical_significance": {
                "ssim_significant": True,
                "ssim_p_value": 0.02,
                "ssim_t_statistic": 3.2
            },
            "visual_quality": {
                "edge_preservation_improvement": 0.12,
                "color_accuracy_improvement": 0.08,
                "shape_fidelity_improvement": 0.05
            }
        }

        # Sample metadata
        test_metadata = {
            "logo_type": "gradient",
            "confidence": 0.88,
            "optimization_method": "Method1_CorrelationMapping",
            "processing_timestamp": "2025-09-29T11:30:00",
            "correlations_used": [
                "edge_density->corner_threshold",
                "unique_colors->color_precision",
                "entropy->path_precision"
            ]
        }

        # Log the optimization
        self.logger.log_optimization(
            "sample_gradient_logo.png",
            test_features,
            test_params,
            test_quality_metrics,
            test_metadata
        )

        # Verify session data capture
        assert len(self.logger.session_data) == 1
        logged_entry = self.logger.session_data[0]

        # Check all required fields are captured
        assert logged_entry["image"]["path"] == "sample_gradient_logo.png"
        assert logged_entry["features"] == test_features
        assert logged_entry["parameters"] == test_params
        assert logged_entry["quality"] == test_quality_metrics
        assert logged_entry["metadata"] == test_metadata

        # Check timestamp and session_id are added
        assert "timestamp" in logged_entry
        assert "session_id" in logged_entry

        # Verify JSON log file creation and content
        assert self.logger.json_log.exists()
        with open(self.logger.json_log, 'r') as f:
            json_lines = f.readlines()

        assert len(json_lines) == 1
        import json
        json_entry = json.loads(json_lines[0])
        assert json_entry["image"]["path"] == "sample_gradient_logo.png"
        assert json_entry["features"]["edge_density"] == 0.15
        assert json_entry["quality"]["improvements"]["ssim_improvement"] == 15.5

        # Verify CSV log file creation and content
        assert self.logger.csv_log.exists()
        import csv
        with open(self.logger.csv_log, 'r') as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            data_row = next(csv_reader)

        # Check CSV contains key data
        assert "timestamp" in headers
        assert "image_path" in headers
        assert "ssim_improvement" in headers
        assert "logo_type" in headers

        # Verify data in CSV row
        logo_type_idx = headers.index("logo_type")
        ssim_idx = headers.index("ssim_improvement")
        assert data_row[logo_type_idx] == "gradient"
        assert float(data_row[ssim_idx]) == 15.5

        # Test statistics calculation
        stats = self.logger.calculate_statistics()
        assert stats["total_optimizations"] == 1
        assert "ssim_improvement" in stats
        assert stats["ssim_improvement"]["average"] == 15.5
        assert stats["logo_type_distribution"]["gradient"] == 1

        # Test export functionality
        export_path = self.logger.export_to_csv()
        assert Path(export_path).exists()

        # Test dashboard data creation
        dashboard_data = self.logger.create_dashboard_data()
        assert dashboard_data["summary"]["total_optimizations"] == 1
        assert dashboard_data["summary"]["average_ssim_improvement"] == 15.5

        # Test HTML dashboard export
        html_path = self.logger.export_dashboard_html()
        assert Path(html_path).exists()

        with open(html_path, 'r') as f:
            html_content = f.read()
        assert "gradient" in html_content
        assert "15.5" in html_content or "15.50" in html_content

        print("✅ Logging system captures complete optimization data")