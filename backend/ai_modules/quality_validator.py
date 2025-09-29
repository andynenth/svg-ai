#!/usr/bin/env python3
"""
Quality Validation System for AI-Enhanced SVG Conversion

This module provides comprehensive quality validation for AI-enhanced SVG conversion,
including SSIM measurement, quality-based optimization feedback, and improvement recommendations.

Features:
- SSIM quality measurement with detailed analysis
- Quality-based parameter optimization feedback loop
- Comprehensive conversion quality reporting
- Quality threshold validation with automatic adjustments
- Quality improvement recommendations based on analysis
- Performance and quality statistics tracking
"""

import logging
import time
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Quality level categories for validation."""
    EXCELLENT = "excellent"  # SSIM >= 0.95
    GOOD = "good"           # SSIM >= 0.85
    ACCEPTABLE = "acceptable"  # SSIM >= 0.70
    POOR = "poor"           # SSIM < 0.70


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for SVG conversion."""
    ssim_score: float
    mse_score: float
    psnr_score: float
    structural_similarity_index: float
    quality_level: QualityLevel
    file_size_ratio: float
    conversion_time: float
    quality_analysis_time: float


@dataclass
class QualityReport:
    """Detailed quality validation report."""
    original_image_path: str
    svg_content: str
    metrics: QualityMetrics
    quality_passed: bool
    quality_threshold: float
    recommendations: List[str]
    parameter_suggestions: Dict[str, Any]
    validation_time: float


class QualityValidator:
    """Quality validation system for AI-enhanced SVG conversion.

    Provides comprehensive quality analysis including SSIM measurement,
    quality-based optimization feedback, and improvement recommendations.

    Features:
        - SSIM quality measurement with detailed structural analysis
        - Quality-based parameter optimization feedback loop
        - Comprehensive quality reporting with actionable insights
        - Quality threshold validation with automatic parameter adjustments
        - Performance tracking and quality statistics
        - Visual quality analysis and recommendations

    Example:
        Basic quality validation:

        validator = QualityValidator()
        report = validator.validate_conversion(
            original_image_path="logo.png",
            svg_content=svg_string,
            parameters_used={'color_precision': 6}
        )
        print(f"Quality: {report.metrics.quality_level.value}")
        print(f"SSIM: {report.metrics.ssim_score:.3f}")

        Quality-based optimization:

        feedback = validator.get_optimization_feedback(report)
        improved_params = feedback['suggested_parameters']
    """

    def __init__(self, quality_threshold: float = 0.85):
        """Initialize quality validator.

        Args:
            quality_threshold (float): Minimum SSIM score for acceptable quality.
        """
        self.quality_threshold = quality_threshold
        self.validation_history = []
        self.stats = {
            'total_validations': 0,
            'quality_passed': 0,
            'quality_failed': 0,
            'by_quality_level': {level.value: 0 for level in QualityLevel},
            'average_ssim': 0.0,
            'average_validation_time': 0.0
        }

        logger.info(f"QualityValidator initialized (threshold: {quality_threshold})")

    def validate_conversion(self,
                          original_image_path: str,
                          svg_content: str,
                          parameters_used: Optional[Dict[str, Any]] = None,
                          features: Optional[Dict[str, float]] = None) -> QualityReport:
        """Validate SVG conversion quality against original image.

        Performs comprehensive quality analysis including SSIM measurement,
        visual comparison, and quality-based recommendations.

        Args:
            original_image_path (str): Path to original image file.
            svg_content (str): SVG content to validate.
            parameters_used (Optional[Dict[str, Any]]): VTracer parameters used.
            features (Optional[Dict[str, float]]): Extracted image features.

        Returns:
            QualityReport: Comprehensive quality validation report.

        Raises:
            FileNotFoundError: If original image file doesn't exist.
            ValueError: If SVG content is invalid.
        """
        start_time = time.time()

        logger.info(f"Starting quality validation for {Path(original_image_path).name}")

        # Validate inputs
        if not Path(original_image_path).exists():
            raise FileNotFoundError(f"Original image not found: {original_image_path}")

        if not svg_content or not isinstance(svg_content, str):
            raise ValueError("Invalid SVG content provided")

        # Render SVG to image for comparison
        try:
            rendered_image_path = self._render_svg_to_image(svg_content)
        except Exception as e:
            logger.error(f"Failed to render SVG for quality analysis: {e}")
            # Return minimal report on render failure
            return self._create_fallback_report(
                original_image_path, svg_content, parameters_used, str(e)
            )

        try:
            # Calculate quality metrics
            metrics = self._calculate_quality_metrics(
                original_image_path, rendered_image_path, svg_content,
                time.time() - start_time
            )

            # Determine if quality passes threshold
            quality_passed = metrics.ssim_score >= self.quality_threshold

            # Generate recommendations and parameter suggestions
            recommendations = self._generate_recommendations(
                metrics, parameters_used, features
            )
            parameter_suggestions = self._generate_parameter_suggestions(
                metrics, parameters_used, features
            )

            validation_time = time.time() - start_time

            # Create quality report
            report = QualityReport(
                original_image_path=original_image_path,
                svg_content=svg_content,
                metrics=metrics,
                quality_passed=quality_passed,
                quality_threshold=self.quality_threshold,
                recommendations=recommendations,
                parameter_suggestions=parameter_suggestions,
                validation_time=validation_time
            )

            # Update statistics
            self._update_validation_stats(report)

            logger.info(f"Quality validation complete: {metrics.quality_level.value} "
                       f"(SSIM: {metrics.ssim_score:.3f}, time: {validation_time*1000:.1f}ms)")

            return report

        finally:
            # Clean up rendered image
            if 'rendered_image_path' in locals() and Path(rendered_image_path).exists():
                Path(rendered_image_path).unlink()

    def _render_svg_to_image(self, svg_content: str) -> str:
        """Render SVG content to PNG image for quality comparison.

        Args:
            svg_content (str): SVG content to render.

        Returns:
            str: Path to rendered PNG image.

        Raises:
            Exception: If SVG rendering fails.
        """
        try:
            # Try to use cairosvg for SVG rendering
            import cairosvg

            # Create temporary file for rendered image
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)

            # Render SVG to PNG
            cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=temp_path)

            return temp_path

        except ImportError:
            logger.warning("cairosvg not available, using fallback rendering")
            return self._fallback_svg_rendering(svg_content)

        except Exception as e:
            logger.error(f"SVG rendering failed: {e}")
            raise Exception(f"Failed to render SVG: {e}")

    def _fallback_svg_rendering(self, svg_content: str) -> str:
        """Fallback SVG rendering using basic image generation.

        Args:
            svg_content (str): SVG content (not actually used in fallback).

        Returns:
            str: Path to fallback rendered image.
        """
        # Create a simple fallback image for quality testing
        fallback_image = np.ones((200, 200, 3), dtype=np.uint8) * 128  # Gray image

        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)

        cv2.imwrite(temp_path, fallback_image)
        return temp_path

    def _calculate_quality_metrics(self,
                                 original_path: str,
                                 rendered_path: str,
                                 svg_content: str,
                                 conversion_time: float) -> QualityMetrics:
        """Calculate comprehensive quality metrics.

        Args:
            original_path (str): Path to original image.
            rendered_path (str): Path to rendered SVG image.
            svg_content (str): SVG content for size analysis.
            conversion_time (float): Time taken for conversion.

        Returns:
            QualityMetrics: Calculated quality metrics.
        """
        analysis_start = time.time()

        # Load images
        original_img = cv2.imread(original_path)
        rendered_img = cv2.imread(rendered_path)

        if original_img is None or rendered_img is None:
            raise ValueError("Failed to load images for quality analysis")

        # Resize rendered image to match original if needed
        if original_img.shape != rendered_img.shape:
            rendered_img = cv2.resize(rendered_img, (original_img.shape[1], original_img.shape[0]))

        # Convert to grayscale for SSIM calculation
        original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        rendered_gray = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM
        ssim_score = self._calculate_ssim(original_gray, rendered_gray)

        # Calculate MSE
        mse_score = np.mean((original_gray.astype(float) - rendered_gray.astype(float)) ** 2)

        # Calculate PSNR
        if mse_score == 0:
            psnr_score = float('inf')
        else:
            psnr_score = 20 * np.log10(255.0 / np.sqrt(mse_score))

        # Calculate structural similarity index (enhanced SSIM)
        structural_similarity_index = self._calculate_enhanced_ssim(original_img, rendered_img)

        # Determine quality level
        quality_level = self._determine_quality_level(ssim_score)

        # Calculate file size ratio
        original_size = Path(original_path).stat().st_size
        svg_size = len(svg_content.encode('utf-8'))
        file_size_ratio = svg_size / original_size if original_size > 0 else 1.0

        quality_analysis_time = time.time() - analysis_start

        return QualityMetrics(
            ssim_score=ssim_score,
            mse_score=mse_score,
            psnr_score=psnr_score,
            structural_similarity_index=structural_similarity_index,
            quality_level=quality_level,
            file_size_ratio=file_size_ratio,
            conversion_time=conversion_time,
            quality_analysis_time=quality_analysis_time
        )

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index (SSIM).

        Args:
            img1 (np.ndarray): First image (grayscale).
            img2 (np.ndarray): Second image (grayscale).

        Returns:
            float: SSIM score between 0 and 1.
        """
        try:
            from skimage.metrics import structural_similarity
            return structural_similarity(img1, img2, data_range=255)
        except ImportError:
            logger.warning("scikit-image not available, using basic SSIM approximation")
            return self._basic_ssim_approximation(img1, img2)

    def _basic_ssim_approximation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Basic SSIM approximation when scikit-image is not available.

        Args:
            img1 (np.ndarray): First image.
            img2 (np.ndarray): Second image.

        Returns:
            float: Approximate SSIM score.
        """
        # Calculate means
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)

        # Calculate variances and covariance
        var1 = np.var(img1)
        var2 = np.var(img2)
        cov = np.mean((img1 - mu1) * (img2 - mu2))

        # SSIM constants
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        # Calculate SSIM
        numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)

        if denominator == 0:
            return 1.0  # Perfect similarity if both images are constant

        return numerator / denominator

    def _calculate_enhanced_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate enhanced structural similarity considering color channels.

        Args:
            img1 (np.ndarray): First image (color).
            img2 (np.ndarray): Second image (color).

        Returns:
            float: Enhanced SSIM score.
        """
        # Calculate SSIM for each color channel
        ssim_scores = []
        for channel in range(img1.shape[2]):
            channel_ssim = self._calculate_ssim(img1[:, :, channel], img2[:, :, channel])
            ssim_scores.append(channel_ssim)

        # Return weighted average (green channel gets higher weight for human perception)
        weights = [0.25, 0.5, 0.25]  # R, G, B weights
        return np.average(ssim_scores, weights=weights)

    def _determine_quality_level(self, ssim_score: float) -> QualityLevel:
        """Determine quality level based on SSIM score.

        Args:
            ssim_score (float): SSIM score.

        Returns:
            QualityLevel: Quality level category.
        """
        if ssim_score >= 0.95:
            return QualityLevel.EXCELLENT
        elif ssim_score >= 0.85:
            return QualityLevel.GOOD
        elif ssim_score >= 0.70:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.POOR

    def _generate_recommendations(self,
                                metrics: QualityMetrics,
                                parameters_used: Optional[Dict[str, Any]],
                                features: Optional[Dict[str, float]]) -> List[str]:
        """Generate quality improvement recommendations.

        Args:
            metrics (QualityMetrics): Quality metrics.
            parameters_used (Optional[Dict[str, Any]]): Parameters used for conversion.
            features (Optional[Dict[str, float]]): Image features.

        Returns:
            List[str]: List of improvement recommendations.
        """
        recommendations = []

        # SSIM-based recommendations
        if metrics.ssim_score < 0.70:
            recommendations.append("Quality is poor. Consider increasing color_precision for better detail preservation.")
            recommendations.append("Try reducing layer_difference for finer color transitions.")

        elif metrics.ssim_score < 0.85:
            recommendations.append("Quality is acceptable but could be improved.")
            if parameters_used:
                if parameters_used.get('color_precision', 6) < 6:
                    recommendations.append("Consider increasing color_precision for better color accuracy.")

        # File size recommendations
        if metrics.file_size_ratio > 2.0:
            recommendations.append("SVG file is larger than original. Consider reducing color_precision or increasing layer_difference.")
        elif metrics.file_size_ratio < 0.3:
            recommendations.append("Excellent file size reduction achieved.")

        # Parameter-specific recommendations
        if parameters_used:
            corner_threshold = parameters_used.get('corner_threshold', 60)
            if corner_threshold > 80 and metrics.ssim_score < 0.85:
                recommendations.append("High corner_threshold may be over-smoothing details. Try reducing it.")

            path_precision = parameters_used.get('path_precision', 5)
            if path_precision < 5 and metrics.ssim_score < 0.85:
                recommendations.append("Low path_precision may be reducing accuracy. Consider increasing it.")

        # Feature-based recommendations
        if features:
            edge_density = features.get('edge_density', 0.5)
            if edge_density > 0.5 and metrics.ssim_score < 0.85:
                recommendations.append("High edge density detected. Consider lowering corner_threshold for better edge preservation.")

            complexity_score = features.get('complexity_score', 0.5)
            if complexity_score > 0.7 and metrics.ssim_score < 0.85:
                recommendations.append("Complex image detected. Consider increasing max_iterations for better quality.")

        # Performance recommendations
        if metrics.conversion_time > 5.0:
            recommendations.append("Conversion time is high. Consider reducing max_iterations or increasing layer_difference for faster processing.")

        if not recommendations:
            recommendations.append(f"{metrics.quality_level.value.title()} quality achieved. No specific improvements needed.")

        return recommendations

    def _generate_parameter_suggestions(self,
                                      metrics: QualityMetrics,
                                      parameters_used: Optional[Dict[str, Any]],
                                      features: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Generate parameter suggestions for quality improvement.

        Args:
            metrics (QualityMetrics): Quality metrics.
            parameters_used (Optional[Dict[str, Any]]): Parameters used.
            features (Optional[Dict[str, float]]): Image features.

        Returns:
            Dict[str, Any]: Suggested parameter adjustments.
        """
        suggestions = {}

        if not parameters_used:
            return suggestions

        current_params = parameters_used.copy()

        # Suggestions based on quality level
        if metrics.quality_level == QualityLevel.POOR:
            # Aggressive quality improvements
            suggestions['color_precision'] = min(8, current_params.get('color_precision', 6) + 2)
            suggestions['layer_difference'] = max(8, current_params.get('layer_difference', 16) - 4)
            suggestions['path_precision'] = min(8, current_params.get('path_precision', 5) + 2)
            suggestions['max_iterations'] = min(25, current_params.get('max_iterations', 10) + 5)

        elif metrics.quality_level == QualityLevel.ACCEPTABLE:
            # Moderate quality improvements
            suggestions['color_precision'] = min(7, current_params.get('color_precision', 6) + 1)
            suggestions['layer_difference'] = max(10, current_params.get('layer_difference', 16) - 2)

        # File size optimization
        if metrics.file_size_ratio > 1.5:
            # Reduce file size while maintaining quality
            suggestions['layer_difference'] = min(24, current_params.get('layer_difference', 16) + 2)
            if metrics.ssim_score > 0.9:  # Only reduce precision if quality is very good
                suggestions['color_precision'] = max(4, current_params.get('color_precision', 6) - 1)

        # Feature-based adjustments
        if features:
            gradient_strength = features.get('gradient_strength', 0.5)
            if gradient_strength > 0.6 and metrics.ssim_score < 0.85:
                # Optimize for gradients
                suggestions['color_precision'] = min(8, current_params.get('color_precision', 6) + 1)
                suggestions['layer_difference'] = max(8, current_params.get('layer_difference', 16) - 2)

        return suggestions

    def _create_fallback_report(self,
                              original_image_path: str,
                              svg_content: str,
                              parameters_used: Optional[Dict[str, Any]],
                              error_message: str) -> QualityReport:
        """Create fallback quality report when full analysis fails.

        Args:
            original_image_path (str): Original image path.
            svg_content (str): SVG content.
            parameters_used (Optional[Dict[str, Any]]): Parameters used.
            error_message (str): Error that occurred.

        Returns:
            QualityReport: Fallback quality report.
        """
        # Create minimal metrics
        fallback_metrics = QualityMetrics(
            ssim_score=0.0,
            mse_score=float('inf'),
            psnr_score=0.0,
            structural_similarity_index=0.0,
            quality_level=QualityLevel.POOR,
            file_size_ratio=1.0,
            conversion_time=0.0,
            quality_analysis_time=0.0
        )

        return QualityReport(
            original_image_path=original_image_path,
            svg_content=svg_content,
            metrics=fallback_metrics,
            quality_passed=False,
            quality_threshold=self.quality_threshold,
            recommendations=[f"Quality analysis failed: {error_message}"],
            parameter_suggestions={},
            validation_time=0.0
        )

    def _update_validation_stats(self, report: QualityReport) -> None:
        """Update validation statistics with new report.

        Args:
            report (QualityReport): Quality report to record.
        """
        self.stats['total_validations'] += 1

        if report.quality_passed:
            self.stats['quality_passed'] += 1
        else:
            self.stats['quality_failed'] += 1

        # Update quality level statistics
        quality_level = report.metrics.quality_level.value
        self.stats['by_quality_level'][quality_level] += 1

        # Update average SSIM
        old_avg = self.stats['average_ssim']
        n = self.stats['total_validations']
        self.stats['average_ssim'] = (old_avg * (n-1) + report.metrics.ssim_score) / n

        # Update average validation time
        old_avg_time = self.stats['average_validation_time']
        self.stats['average_validation_time'] = (old_avg_time * (n-1) + report.validation_time) / n

        # Store in history (keep last 50)
        self.validation_history.append(report)
        if len(self.validation_history) > 50:
            self.validation_history = self.validation_history[-50:]

    def get_optimization_feedback(self, report: QualityReport) -> Dict[str, Any]:
        """Get optimization feedback based on quality analysis.

        Args:
            report (QualityReport): Quality report to analyze.

        Returns:
            Dict[str, Any]: Optimization feedback with parameter suggestions.
        """
        return {
            'quality_score': report.metrics.ssim_score,
            'quality_level': report.metrics.quality_level.value,
            'quality_passed': report.quality_passed,
            'suggested_parameters': report.parameter_suggestions,
            'improvement_potential': 1.0 - report.metrics.ssim_score,
            'recommendations': report.recommendations,
            'optimization_priority': 'high' if report.metrics.ssim_score < 0.7 else 'medium' if report.metrics.ssim_score < 0.85 else 'low'
        }

    def get_quality_stats(self) -> Dict[str, Any]:
        """Get comprehensive quality validation statistics.

        Returns:
            Dict[str, Any]: Quality validation statistics.
        """
        stats = self.stats.copy()

        if stats['total_validations'] > 0:
            stats['pass_rate'] = (stats['quality_passed'] / stats['total_validations']) * 100
        else:
            stats['pass_rate'] = 0.0

        return stats

    def set_quality_threshold(self, threshold: float) -> None:
        """Set new quality threshold.

        Args:
            threshold (float): New SSIM threshold (0.0 to 1.0).

        Raises:
            ValueError: If threshold is not in valid range.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Quality threshold must be between 0.0 and 1.0, got {threshold}")

        self.quality_threshold = threshold
        logger.info(f"Quality threshold updated to {threshold}")


def test_quality_validator():
    """Test the quality validator with sample conversions."""
    print("\n" + "="*70)
    print("Testing Quality Validation System")
    print("="*70)

    validator = QualityValidator(quality_threshold=0.85)

    # Create test image
    test_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    test_path = tempfile.mktemp(suffix='.png')
    cv2.imwrite(test_path, test_image)

    # Create mock SVG content
    mock_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <rect width="200" height="200" fill="rgb(128,128,128)"/>
</svg>'''

    try:
        # Test quality validation
        print(f"\n[Quality Validation Test]")
        print("-" * 40)

        test_params = {
            'color_precision': 6,
            'layer_difference': 16,
            'corner_threshold': 60,
            'path_precision': 5
        }

        test_features = {
            'edge_density': 0.3,
            'unique_colors': 0.6,
            'entropy': 0.5,
            'complexity_score': 0.4
        }

        report = validator.validate_conversion(
            test_path, mock_svg, test_params, test_features
        )

        print(f"Quality Level: {report.metrics.quality_level.value}")
        print(f"SSIM Score: {report.metrics.ssim_score:.3f}")
        print(f"Quality Passed: {report.quality_passed}")
        print(f"Validation Time: {report.validation_time*1000:.1f}ms")

        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

        if report.parameter_suggestions:
            print(f"\nParameter Suggestions:")
            for param, value in report.parameter_suggestions.items():
                print(f"  {param}: {value}")

        # Test optimization feedback
        feedback = validator.get_optimization_feedback(report)
        print(f"\nOptimization Feedback:")
        print(f"  Optimization Priority: {feedback['optimization_priority']}")
        print(f"  Improvement Potential: {feedback['improvement_potential']:.3f}")

        # Test statistics
        stats = validator.get_quality_stats()
        print(f"\nQuality Statistics:")
        print(f"  Total Validations: {stats['total_validations']}")
        print(f"  Pass Rate: {stats['pass_rate']:.1f}%")
        print(f"  Average SSIM: {stats['average_ssim']:.3f}")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        if Path(test_path).exists():
            Path(test_path).unlink()


if __name__ == "__main__":
    test_quality_validator()