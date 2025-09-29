#!/usr/bin/env python3
"""
Unit tests for Quality Validation System

Tests quality metrics calculation, SSIM measurement, quality-based recommendations,
and parameter optimization feedback.
"""

import unittest
import tempfile
import cv2
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.ai_modules.quality_validator import (
    QualityValidator, QualityLevel, QualityMetrics, QualityReport
)


class TestQualityValidator(unittest.TestCase):
    """Test suite for QualityValidator class"""

    def setUp(self):
        """Set up test environment"""
        self.validator = QualityValidator(quality_threshold=0.85)
        self.test_image_path = self._create_test_image()
        self.test_svg_content = self._create_test_svg()

    def tearDown(self):
        """Clean up test environment"""
        if Path(self.test_image_path).exists():
            Path(self.test_image_path).unlink()

    def _create_test_image(self) -> str:
        """Create a test image for quality validation"""
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 100, 100), -1)  # Blue rectangle
        cv2.circle(test_image, (100, 100), 30, (100, 255, 100), -1)  # Green circle

        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        cv2.imwrite(temp_path, test_image)
        return temp_path

    def _create_test_svg(self) -> str:
        """Create test SVG content"""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <rect x="50" y="50" width="100" height="100" fill="rgb(255,100,100)"/>
    <circle cx="100" cy="100" r="30" fill="rgb(100,255,100)"/>
</svg>'''

    def test_validator_initialization(self):
        """Test quality validator initialization"""
        self.assertIsInstance(self.validator, QualityValidator)
        self.assertEqual(self.validator.quality_threshold, 0.85)
        self.assertEqual(self.validator.stats['total_validations'], 0)
        self.assertIsInstance(self.validator.validation_history, list)

    def test_quality_level_determination(self):
        """Test quality level determination based on SSIM scores"""
        # Test excellent quality
        excellent_level = self.validator._determine_quality_level(0.96)
        self.assertEqual(excellent_level, QualityLevel.EXCELLENT)

        # Test good quality
        good_level = self.validator._determine_quality_level(0.88)
        self.assertEqual(good_level, QualityLevel.GOOD)

        # Test acceptable quality
        acceptable_level = self.validator._determine_quality_level(0.75)
        self.assertEqual(acceptable_level, QualityLevel.ACCEPTABLE)

        # Test poor quality
        poor_level = self.validator._determine_quality_level(0.65)
        self.assertEqual(poor_level, QualityLevel.POOR)

        # Test edge cases
        self.assertEqual(self.validator._determine_quality_level(0.95), QualityLevel.EXCELLENT)  # 0.95 >= 0.95 threshold
        self.assertEqual(self.validator._determine_quality_level(0.85), QualityLevel.GOOD)
        self.assertEqual(self.validator._determine_quality_level(0.70), QualityLevel.ACCEPTABLE)

    def test_basic_ssim_approximation(self):
        """Test basic SSIM approximation calculation"""
        # Test identical images
        img1 = np.ones((100, 100), dtype=np.uint8) * 128
        img2 = np.ones((100, 100), dtype=np.uint8) * 128
        ssim_identical = self.validator._basic_ssim_approximation(img1, img2)
        self.assertAlmostEqual(ssim_identical, 1.0, places=2)

        # Test completely different images
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img2 = np.ones((100, 100), dtype=np.uint8) * 255
        ssim_different = self.validator._basic_ssim_approximation(img1, img2)
        self.assertLess(ssim_different, 0.5)

        # Test similar images
        img1 = np.ones((100, 100), dtype=np.uint8) * 128
        img2 = np.ones((100, 100), dtype=np.uint8) * 130  # Slightly different
        ssim_similar = self.validator._basic_ssim_approximation(img1, img2)
        self.assertGreater(ssim_similar, 0.8)

    def test_quality_threshold_setting(self):
        """Test quality threshold setting and validation"""
        # Test valid threshold
        self.validator.set_quality_threshold(0.9)
        self.assertEqual(self.validator.quality_threshold, 0.9)

        # Test edge cases
        self.validator.set_quality_threshold(0.0)
        self.assertEqual(self.validator.quality_threshold, 0.0)

        self.validator.set_quality_threshold(1.0)
        self.assertEqual(self.validator.quality_threshold, 1.0)

        # Test invalid thresholds
        with self.assertRaises(ValueError):
            self.validator.set_quality_threshold(-0.1)

        with self.assertRaises(ValueError):
            self.validator.set_quality_threshold(1.1)

    def test_quality_metrics_structure(self):
        """Test QualityMetrics dataclass structure"""
        metrics = QualityMetrics(
            ssim_score=0.85,
            mse_score=100.0,
            psnr_score=28.0,
            structural_similarity_index=0.83,
            quality_level=QualityLevel.GOOD,
            file_size_ratio=0.6,
            conversion_time=1.2,
            quality_analysis_time=0.5
        )

        self.assertEqual(metrics.ssim_score, 0.85)
        self.assertEqual(metrics.quality_level, QualityLevel.GOOD)
        self.assertEqual(metrics.file_size_ratio, 0.6)

    def test_quality_report_structure(self):
        """Test QualityReport dataclass structure"""
        metrics = QualityMetrics(
            ssim_score=0.85, mse_score=100.0, psnr_score=28.0,
            structural_similarity_index=0.83, quality_level=QualityLevel.GOOD,
            file_size_ratio=0.6, conversion_time=1.2, quality_analysis_time=0.5
        )

        report = QualityReport(
            original_image_path="test.png",
            svg_content="<svg></svg>",
            metrics=metrics,
            quality_passed=True,
            quality_threshold=0.85,
            recommendations=["Test recommendation"],
            parameter_suggestions={'color_precision': 7},
            validation_time=1.0
        )

        self.assertEqual(report.original_image_path, "test.png")
        self.assertTrue(report.quality_passed)
        self.assertIn("Test recommendation", report.recommendations)

    def test_parameter_suggestions_poor_quality(self):
        """Test parameter suggestions for poor quality conversions"""
        # Create metrics indicating poor quality
        poor_metrics = QualityMetrics(
            ssim_score=0.6, mse_score=500.0, psnr_score=20.0,
            structural_similarity_index=0.58, quality_level=QualityLevel.POOR,
            file_size_ratio=1.0, conversion_time=2.0, quality_analysis_time=0.3
        )

        current_params = {
            'color_precision': 4,
            'layer_difference': 20,
            'path_precision': 3,
            'max_iterations': 8
        }

        suggestions = self.validator._generate_parameter_suggestions(
            poor_metrics, current_params, None
        )

        # Should suggest quality improvements
        self.assertGreater(suggestions.get('color_precision', 0), current_params['color_precision'])
        self.assertLess(suggestions.get('layer_difference', 100), current_params['layer_difference'])
        self.assertGreater(suggestions.get('path_precision', 0), current_params['path_precision'])
        self.assertGreater(suggestions.get('max_iterations', 0), current_params['max_iterations'])

    def test_parameter_suggestions_large_file_size(self):
        """Test parameter suggestions for large file sizes"""
        # Create metrics indicating large file size
        large_file_metrics = QualityMetrics(
            ssim_score=0.92, mse_score=50.0, psnr_score=35.0,
            structural_similarity_index=0.90, quality_level=QualityLevel.EXCELLENT,
            file_size_ratio=2.5, conversion_time=1.0, quality_analysis_time=0.2
        )

        current_params = {
            'color_precision': 8,
            'layer_difference': 8
        }

        suggestions = self.validator._generate_parameter_suggestions(
            large_file_metrics, current_params, None
        )

        # Should suggest file size reduction
        self.assertGreater(suggestions.get('layer_difference', 0), current_params['layer_difference'])
        # May suggest reducing color precision since quality is excellent
        if 'color_precision' in suggestions:
            self.assertLess(suggestions['color_precision'], current_params['color_precision'])

    def test_recommendations_generation(self):
        """Test quality improvement recommendations generation"""
        # Test poor quality recommendations
        poor_metrics = QualityMetrics(
            ssim_score=0.6, mse_score=500.0, psnr_score=20.0,
            structural_similarity_index=0.58, quality_level=QualityLevel.POOR,
            file_size_ratio=1.0, conversion_time=2.0, quality_analysis_time=0.3
        )

        recommendations = self.validator._generate_recommendations(
            poor_metrics, {'color_precision': 3}, None
        )

        self.assertGreater(len(recommendations), 0)
        # Should mention quality improvement
        quality_mentioned = any('quality' in rec.lower() or 'precision' in rec.lower()
                              for rec in recommendations)
        self.assertTrue(quality_mentioned)

        # Test good quality recommendations
        good_metrics = QualityMetrics(
            ssim_score=0.95, mse_score=25.0, psnr_score=40.0,
            structural_similarity_index=0.93, quality_level=QualityLevel.EXCELLENT,
            file_size_ratio=0.4, conversion_time=1.0, quality_analysis_time=0.2
        )

        recommendations = self.validator._generate_recommendations(
            good_metrics, {'color_precision': 6}, None
        )

        # Should indicate good quality
        self.assertGreater(len(recommendations), 0)

    def test_recommendations_with_features(self):
        """Test recommendations generation with image features"""
        metrics = QualityMetrics(
            ssim_score=0.78, mse_score=200.0, psnr_score=25.0,
            structural_similarity_index=0.76, quality_level=QualityLevel.ACCEPTABLE,
            file_size_ratio=1.2, conversion_time=1.5, quality_analysis_time=0.3
        )

        # Features indicating high edge density
        features = {
            'edge_density': 0.8,
            'unique_colors': 0.5,
            'complexity_score': 0.3
        }

        recommendations = self.validator._generate_recommendations(
            metrics, {'corner_threshold': 60}, features
        )

        # Should mention edge-related recommendations
        edge_mentioned = any('edge' in rec.lower() or 'corner' in rec.lower()
                           for rec in recommendations)
        self.assertTrue(edge_mentioned)

    def test_optimization_feedback(self):
        """Test optimization feedback generation"""
        metrics = QualityMetrics(
            ssim_score=0.78, mse_score=200.0, psnr_score=25.0,
            structural_similarity_index=0.76, quality_level=QualityLevel.ACCEPTABLE,
            file_size_ratio=1.2, conversion_time=1.5, quality_analysis_time=0.3
        )

        report = QualityReport(
            original_image_path="test.png",
            svg_content="<svg></svg>",
            metrics=metrics,
            quality_passed=False,
            quality_threshold=0.85,
            recommendations=["Improve quality"],
            parameter_suggestions={'color_precision': 7},
            validation_time=1.0
        )

        feedback = self.validator.get_optimization_feedback(report)

        # Verify feedback structure
        self.assertIn('quality_score', feedback)
        self.assertIn('quality_level', feedback)
        self.assertIn('quality_passed', feedback)
        self.assertIn('suggested_parameters', feedback)
        self.assertIn('improvement_potential', feedback)
        self.assertIn('optimization_priority', feedback)

        # Verify values
        self.assertEqual(feedback['quality_score'], 0.78)
        self.assertEqual(feedback['quality_level'], 'acceptable')
        self.assertFalse(feedback['quality_passed'])
        self.assertAlmostEqual(feedback['improvement_potential'], 0.22, places=2)

    def test_statistics_tracking(self):
        """Test quality validation statistics tracking"""
        initial_stats = self.validator.get_quality_stats()
        self.assertEqual(initial_stats['total_validations'], 0)

        # Create mock reports for statistics testing
        good_metrics = QualityMetrics(
            ssim_score=0.88, mse_score=100.0, psnr_score=30.0,
            structural_similarity_index=0.86, quality_level=QualityLevel.GOOD,
            file_size_ratio=0.8, conversion_time=1.0, quality_analysis_time=0.2
        )

        poor_metrics = QualityMetrics(
            ssim_score=0.65, mse_score=300.0, psnr_score=22.0,
            structural_similarity_index=0.63, quality_level=QualityLevel.POOR,
            file_size_ratio=1.5, conversion_time=2.0, quality_analysis_time=0.4
        )

        # Create mock reports
        good_report = QualityReport(
            original_image_path="good.png", svg_content="<svg></svg>",
            metrics=good_metrics, quality_passed=True, quality_threshold=0.85,
            recommendations=[], parameter_suggestions={}, validation_time=1.0
        )

        poor_report = QualityReport(
            original_image_path="poor.png", svg_content="<svg></svg>",
            metrics=poor_metrics, quality_passed=False, quality_threshold=0.85,
            recommendations=[], parameter_suggestions={}, validation_time=1.5
        )

        # Update statistics
        self.validator._update_validation_stats(good_report)
        self.validator._update_validation_stats(poor_report)

        # Check updated statistics
        final_stats = self.validator.get_quality_stats()
        self.assertEqual(final_stats['total_validations'], 2)
        self.assertEqual(final_stats['quality_passed'], 1)
        self.assertEqual(final_stats['quality_failed'], 1)
        self.assertEqual(final_stats['pass_rate'], 50.0)
        self.assertEqual(final_stats['by_quality_level']['good'], 1)
        self.assertEqual(final_stats['by_quality_level']['poor'], 1)

        # Check average SSIM
        expected_avg_ssim = (0.88 + 0.65) / 2
        self.assertAlmostEqual(final_stats['average_ssim'], expected_avg_ssim, places=3)

    def test_fallback_report_creation(self):
        """Test fallback report creation when analysis fails"""
        error_message = "Test error"
        fallback_report = self.validator._create_fallback_report(
            "test.png", "<svg></svg>", {'color_precision': 6}, error_message
        )

        # Verify fallback report structure
        self.assertEqual(fallback_report.original_image_path, "test.png")
        self.assertEqual(fallback_report.metrics.quality_level, QualityLevel.POOR)
        self.assertFalse(fallback_report.quality_passed)
        self.assertIn(error_message, fallback_report.recommendations[0])

    def test_input_validation(self):
        """Test input validation for quality validation"""
        # Test with nonexistent file
        with self.assertRaises(FileNotFoundError):
            self.validator.validate_conversion(
                "nonexistent.png", self.test_svg_content
            )

        # Test with invalid SVG content
        with self.assertRaises(ValueError):
            self.validator.validate_conversion(
                self.test_image_path, ""
            )

        with self.assertRaises(ValueError):
            self.validator.validate_conversion(
                self.test_image_path, None
            )

    def test_enhanced_ssim_calculation(self):
        """Test enhanced SSIM calculation with color channels"""
        # Create test color images
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()  # Identical images

        enhanced_ssim = self.validator._calculate_enhanced_ssim(img1, img2)
        self.assertAlmostEqual(enhanced_ssim, 1.0, places=2)

        # Test with different images
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        enhanced_ssim_diff = self.validator._calculate_enhanced_ssim(img1, img2)
        self.assertLess(enhanced_ssim_diff, 0.8)

    def test_feature_based_suggestions(self):
        """Test parameter suggestions based on image features"""
        metrics = QualityMetrics(
            ssim_score=0.78, mse_score=200.0, psnr_score=25.0,
            structural_similarity_index=0.76, quality_level=QualityLevel.ACCEPTABLE,
            file_size_ratio=1.2, conversion_time=1.5, quality_analysis_time=0.3
        )

        # Features with high gradient strength
        gradient_features = {
            'gradient_strength': 0.8,
            'edge_density': 0.3,
            'complexity_score': 0.4
        }

        current_params = {'color_precision': 5, 'layer_difference': 18}

        suggestions = self.validator._generate_parameter_suggestions(
            metrics, current_params, gradient_features
        )

        # Should optimize for gradients
        if 'color_precision' in suggestions:
            self.assertGreaterEqual(suggestions['color_precision'], current_params['color_precision'])
        if 'layer_difference' in suggestions:
            self.assertLessEqual(suggestions['layer_difference'], current_params['layer_difference'])

    def test_validation_history_management(self):
        """Test validation history management and limits"""
        # Create many validation reports to test history limit
        for i in range(55):  # More than the 50 limit
            metrics = QualityMetrics(
                ssim_score=0.8, mse_score=100.0, psnr_score=30.0,
                structural_similarity_index=0.78, quality_level=QualityLevel.GOOD,
                file_size_ratio=0.8, conversion_time=1.0, quality_analysis_time=0.2
            )

            report = QualityReport(
                original_image_path=f"test_{i}.png", svg_content="<svg></svg>",
                metrics=metrics, quality_passed=True, quality_threshold=0.85,
                recommendations=[], parameter_suggestions={}, validation_time=1.0
            )

            self.validator._update_validation_stats(report)

        # Should keep only last 50 reports
        self.assertEqual(len(self.validator.validation_history), 50)
        self.assertEqual(self.validator.stats['total_validations'], 55)


if __name__ == '__main__':
    unittest.main()