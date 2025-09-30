"""Unit tests for validation pipeline"""
import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from backend.ai_modules.optimization.validation_pipeline import (
    Method1ValidationPipeline,
    ValidationResult
)


class TestMethod1ValidationPipeline:
    """Test suite for validation pipeline"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = Method1ValidationPipeline(enable_quality_measurement=False)

    def teardown_method(self):
        """Cleanup after each test"""
        if hasattr(self.pipeline, 'quality_metrics') and self.pipeline.quality_metrics:
            self.pipeline.quality_metrics.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test pipeline initialization"""
        pipeline = Method1ValidationPipeline(enable_quality_measurement=True)

        assert pipeline.optimizer is not None
        assert pipeline.feature_extractor is not None
        assert pipeline.quality_metrics is not None
        assert pipeline.bounds is not None
        assert pipeline.enable_quality_measurement is True
        assert pipeline.results == []

        # Test without quality measurement
        pipeline_no_quality = Method1ValidationPipeline(enable_quality_measurement=False)
        assert pipeline_no_quality.quality_metrics is None
        assert pipeline_no_quality.enable_quality_measurement is False

    def test_validate_single_image_success(self):
        """Test successful single image validation"""
        # Mock feature extraction
        mock_features = {
            "edge_density": 0.15,
            "unique_colors": 12,
            "entropy": 0.65,
            "corner_density": 0.08,
            "gradient_strength": 0.45,
            "complexity_score": 0.35
        }

        self.pipeline.feature_extractor.extract_features = Mock(return_value=mock_features)

        # Mock optimization result
        mock_optimization = {
            "parameters": {
                "color_precision": 6,
                "layer_difference": 10,
                "corner_threshold": 50,
                "length_threshold": 5.0,
                "max_iterations": 12,
                "splice_threshold": 55,
                "path_precision": 8,
                "mode": "spline"
            },
            "confidence": 0.85,
            "metadata": {"optimization_method": "Method1"}
        }

        self.pipeline.optimizer.optimize = Mock(return_value=mock_optimization)

        # Mock parameter validation
        self.pipeline.bounds.validate_parameters = Mock(return_value={"valid": True, "errors": []})

        result = self.pipeline._validate_single_image("test_image.png", "simple")

        assert result.success is True
        assert result.image_path == "test_image.png"
        assert result.logo_type == "simple"
        assert result.features == mock_features
        assert result.optimized_params == mock_optimization["parameters"]
        assert result.confidence == 0.85
        assert result.error_message == ""
        assert result.processing_time > 0

    def test_validate_single_image_feature_extraction_failure(self):
        """Test validation with feature extraction failure"""
        self.pipeline.feature_extractor.extract_features = Mock(return_value=None)

        result = self.pipeline._validate_single_image("test_image.png", "simple")

        assert result.success is False
        assert result.error_message == "Feature extraction failed"
        assert result.features == {}
        assert result.optimized_params == {}

    def test_validate_single_image_optimization_failure(self):
        """Test validation with optimization failure"""
        mock_features = {"edge_density": 0.15}
        self.pipeline.feature_extractor.extract_features = Mock(return_value=mock_features)
        self.pipeline.optimizer.optimize = Mock(return_value=None)

        result = self.pipeline._validate_single_image("test_image.png", "simple")

        assert result.success is False
        assert result.error_message == "Parameter optimization failed"
        assert result.features == mock_features
        assert result.optimized_params == {}

    def test_validate_single_image_invalid_parameters(self):
        """Test validation with invalid parameters"""
        mock_features = {"edge_density": 0.15}
        mock_optimization = {
            "parameters": {"color_precision": 999},  # Invalid value
            "confidence": 0.5
        }

        self.pipeline.feature_extractor.extract_features = Mock(return_value=mock_features)
        self.pipeline.optimizer.optimize = Mock(return_value=mock_optimization)
        self.pipeline.bounds.validate_parameters = Mock(return_value={
            "valid": False,
            "errors": ["color_precision out of range"]
        })

        result = self.pipeline._validate_single_image("test_image.png", "simple")

        assert result.success is False
        assert "Invalid parameters" in result.error_message
        assert result.optimized_params == {"color_precision": 999}

    def test_validate_dataset_no_images(self):
        """Test validation with no images in dataset"""
        # Create empty dataset directory
        empty_dataset = Path(self.temp_dir) / "empty_dataset"
        empty_dataset.mkdir()

        result = self.pipeline.validate_dataset(str(empty_dataset))

        assert "error" in result
        assert result["error"] == "No images found in dataset"

    def test_validate_dataset_with_mock_images(self):
        """Test validation with mock dataset"""
        # Setup mock dataset structure
        dataset_path = Path(self.temp_dir) / "test_dataset"
        dataset_path.mkdir()

        # Create category directories with actual mock image files
        categories_images = {
            "simple": ["image1.png", "image2.png"],
            "text": ["image3.png"],
            "gradient": ["image4.png"],
            "complex": []
        }

        for category, images in categories_images.items():
            category_dir = dataset_path / category
            category_dir.mkdir()
            # Create actual mock image files
            for image in images:
                (category_dir / image).touch()

        # Mock the _validate_single_image method to return results based on calls
        call_count = 0
        def mock_validate_single(image_path, logo_type):
            nonlocal call_count
            results = [
                ValidationResult(
                    image_path=str(image_path),
                    features={"edge_density": 0.1},
                    optimized_params={"color_precision": 6},
                    quality_improvement=15.0,
                    processing_time=0.05,
                    success=True,
                    logo_type=logo_type,
                    confidence=0.9
                ),
                ValidationResult(
                    image_path=str(image_path),
                    features={"edge_density": 0.12},
                    optimized_params={"color_precision": 6},
                    quality_improvement=18.0,
                    processing_time=0.04,
                    success=True,
                    logo_type=logo_type,
                    confidence=0.85
                ),
                ValidationResult(
                    image_path=str(image_path),
                    features={"edge_density": 0.25},
                    optimized_params={"color_precision": 4},
                    quality_improvement=12.0,
                    processing_time=0.06,
                    success=True,
                    logo_type=logo_type,
                    confidence=0.8
                ),
                ValidationResult(
                    image_path=str(image_path),
                    features={"edge_density": 0.2},
                    optimized_params={"color_precision": 8},
                    quality_improvement=0.0,
                    processing_time=0.08,
                    success=False,
                    error_message="VTracer conversion failed",
                    logo_type=logo_type
                )
            ]
            result = results[call_count % len(results)]
            call_count += 1
            return result

        self.pipeline._validate_single_image = Mock(side_effect=mock_validate_single)

        # Run validation
        result = self.pipeline.validate_dataset(str(dataset_path), max_images_per_category=5)

        # Verify results - should have 4 images (2 simple + 1 text + 1 gradient + 0 complex)
        assert "validation_summary" in result
        assert result["validation_summary"]["total_images"] == 4
        assert result["validation_summary"]["successful_optimizations"] == 3
        assert result["validation_summary"]["overall_success_rate"] == 75.0

        # Check category analysis
        assert "category_analysis" in result
        assert result["category_analysis"]["simple"]["total_images"] == 2
        assert result["category_analysis"]["text"]["total_images"] == 1
        assert result["category_analysis"]["gradient"]["total_images"] == 1
        assert result["category_analysis"]["complex"]["total_images"] == 0

    def test_generate_statistical_analysis_no_results(self):
        """Test statistical analysis with no results"""
        analysis = self.pipeline._generate_statistical_analysis()

        assert "error" in analysis
        assert analysis["error"] == "No validation results to analyze"

    def test_generate_statistical_analysis_with_results(self):
        """Test statistical analysis with sample results"""
        # Add sample results
        self.pipeline.results = [
            ValidationResult(
                image_path="simple1.png",
                features={"edge_density": 0.1},
                optimized_params={"color_precision": 6},
                quality_improvement=15.0,
                processing_time=0.05,
                success=True,
                logo_type="simple"
            ),
            ValidationResult(
                image_path="simple2.png",
                features={"edge_density": 0.12},
                optimized_params={"color_precision": 6},
                quality_improvement=20.0,
                processing_time=0.04,
                success=True,
                logo_type="simple"
            ),
            ValidationResult(
                image_path="complex1.png",
                features={"edge_density": 0.4},
                optimized_params={},
                quality_improvement=0.0,
                processing_time=0.08,
                success=False,
                error_message="Optimization failed",
                logo_type="complex"
            )
        ]

        analysis = self.pipeline._generate_statistical_analysis()

        # Verify overall metrics
        assert analysis["validation_summary"]["total_images"] == 3
        assert analysis["validation_summary"]["successful_optimizations"] == 2
        assert abs(analysis["validation_summary"]["overall_success_rate"] - 66.67) < 0.1  # ~66.67%

        # Verify quality analysis
        assert analysis["quality_analysis"]["improvements_measured"] == 2
        assert analysis["quality_analysis"]["mean_improvement"] == 17.5
        assert analysis["quality_analysis"]["min_improvement"] == 15.0
        assert analysis["quality_analysis"]["max_improvement"] == 20.0

        # Verify category analysis
        assert analysis["category_analysis"]["simple"]["success_rate"] == 100.0
        assert analysis["category_analysis"]["simple"]["total_images"] == 2
        assert analysis["category_analysis"]["complex"]["success_rate"] == 0.0
        assert analysis["category_analysis"]["complex"]["total_images"] == 1

        # Verify failure analysis
        assert analysis["failure_analysis"]["total_failures"] == 1
        assert abs(analysis["failure_analysis"]["failure_rate"] - 33.33) < 0.1  # ~33.33%
        assert "Optimization failed" in analysis["failure_analysis"]["error_types"]

    def test_check_success_criteria(self):
        """Test success criteria checking"""
        category_analysis = {
            "simple": {"success_rate": 96.0},
            "text": {"success_rate": 92.0},
            "gradient": {"success_rate": 88.0},
            "complex": {"success_rate": 82.0}
        }

        result = self.pipeline._check_success_criteria(category_analysis, 85.0)

        assert result["criteria_met"]["overall_success_rate_80_percent"] is True
        assert result["criteria_met"]["simple_success_rate_95_percent"] is True
        assert result["criteria_met"]["text_success_rate_90_percent"] is True
        assert result["criteria_met"]["gradient_success_rate_85_percent"] is True
        assert result["criteria_met"]["complex_success_rate_80_percent"] is True
        assert result["met_criteria"] == 5
        assert result["success_percentage"] == 100.0
        assert result["overall_success"] is True

    def test_analyze_feature_correlations(self):
        """Test feature correlation analysis"""
        # Add sample results with features and quality improvements
        self.pipeline.results = [
            ValidationResult(
                image_path="img1.png",
                features={"edge_density": 0.1, "unique_colors": 5},
                optimized_params={},
                quality_improvement=10.0,
                processing_time=0.05,
                success=True,
                logo_type="simple"
            ),
            ValidationResult(
                image_path="img2.png",
                features={"edge_density": 0.2, "unique_colors": 10},
                optimized_params={},
                quality_improvement=20.0,
                processing_time=0.05,
                success=True,
                logo_type="simple"
            ),
            ValidationResult(
                image_path="img3.png",
                features={"edge_density": 0.3, "unique_colors": 15},
                optimized_params={},
                quality_improvement=30.0,
                processing_time=0.05,
                success=True,
                logo_type="simple"
            )
        ]

        correlations = self.pipeline._analyze_feature_correlations()

        # Should find positive correlation between features and quality improvement
        assert "edge_density" in correlations
        assert "unique_colors" in correlations
        assert correlations["edge_density"]["correlation"] > 0.9  # Perfect positive correlation
        assert correlations["edge_density"]["sample_size"] == 3

    def test_export_results(self):
        """Test results export functionality"""
        # Add sample result
        self.pipeline.results = [
            ValidationResult(
                image_path="test.png",
                features={"edge_density": 0.1},
                optimized_params={"color_precision": 6},
                quality_improvement=15.0,
                processing_time=0.05,
                success=True,
                logo_type="simple",
                confidence=0.9
            )
        ]

        export_paths = self.pipeline.export_results(self.temp_dir)

        # Check files were created
        assert Path(export_paths["json"]).exists()
        assert Path(export_paths["csv"]).exists()
        assert Path(export_paths["html"]).exists()

        # Verify JSON content
        with open(export_paths["json"], 'r') as f:
            json_data = json.load(f)

        assert "results" in json_data
        assert "analysis" in json_data
        assert len(json_data["results"]) == 1
        assert json_data["results"][0]["image_path"] == "test.png"

        # Verify CSV content
        with open(export_paths["csv"], 'r') as f:
            csv_content = f.read()

        assert "image_path" in csv_content
        assert "test.png" in csv_content
        assert "simple" in csv_content

        # Verify HTML content
        with open(export_paths["html"], 'r') as f:
            html_content = f.read()

        assert "<!DOCTYPE html>" in html_content
        assert "Method 1 Validation Report" in html_content
        assert "15.0" in html_content

    def test_get_summary(self):
        """Test summary generation"""
        # Test with no results
        summary = self.pipeline.get_summary()
        assert summary["message"] == "No validation results available"

        # Test with results
        self.pipeline.results = [
            ValidationResult(
                image_path="test1.png",
                features={},
                optimized_params={},
                quality_improvement=15.0,
                processing_time=0.05,
                success=True,
                logo_type="simple"
            ),
            ValidationResult(
                image_path="test2.png",
                features={},
                optimized_params={},
                quality_improvement=0.0,
                processing_time=0.08,
                success=False,
                error_message="Failed",
                logo_type="complex"
            )
        ]

        summary = self.pipeline.get_summary()

        assert summary["total_images"] == 2
        assert summary["successful_optimizations"] == 1
        assert summary["success_rate"] == 50.0
        assert summary["quality_measurement_enabled"] is False
        assert summary["average_processing_time"] == 0.065
        assert summary["results_available"] == 2

    def test_validation_with_quality_measurement_enabled(self):
        """Test validation pipeline with quality measurement enabled"""
        pipeline_with_quality = Method1ValidationPipeline(enable_quality_measurement=True)

        # Mock all dependencies
        pipeline_with_quality.feature_extractor.extract_features = Mock(return_value={"edge_density": 0.1})
        pipeline_with_quality.optimizer.optimize = Mock(return_value={
            "parameters": {"color_precision": 6},
            "confidence": 0.8
        })
        pipeline_with_quality.bounds.validate_parameters = Mock(return_value={"valid": True, "errors": []})
        pipeline_with_quality.bounds.get_default_parameters = Mock(return_value={"color_precision": 5})

        # Mock quality measurement
        mock_quality_result = {
            "improvements": {
                "ssim_improvement": 12.5,
                "file_size_improvement": 20.0
            },
            "default_metrics": {"ssim": 0.8},
            "optimized_metrics": {"ssim": 0.9, "conversion_time": 1.5}
        }
        pipeline_with_quality.quality_metrics.measure_improvement = Mock(return_value=mock_quality_result)

        result = pipeline_with_quality._validate_single_image("test.png", "simple")

        assert result.success is True
        assert result.quality_improvement == 12.5
        assert result.default_ssim == 0.8
        assert result.optimized_ssim == 0.9
        assert result.file_size_improvement == 20.0
        assert result.conversion_time == 1.5

        # Cleanup
        pipeline_with_quality.cleanup()

    def test_cleanup(self):
        """Test resource cleanup"""
        pipeline_with_quality = Method1ValidationPipeline(enable_quality_measurement=True)
        pipeline_with_quality.quality_metrics = Mock()

        pipeline_with_quality.cleanup()

        pipeline_with_quality.quality_metrics.cleanup.assert_called_once()

        # Test cleanup with no quality metrics
        pipeline_no_quality = Method1ValidationPipeline(enable_quality_measurement=False)
        pipeline_no_quality.cleanup()  # Should not raise error