"""Comprehensive quality validation for all optimization methods"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import cv2
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import json
import logging
import time
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import os
import hashlib
from PIL import Image
import io
import cairosvg
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from .quality_metrics import OptimizationQualityMetrics
from .vtracer_test import VTracerTestHarness
from .parameter_bounds import VTracerParameterBounds

logger = logging.getLogger(__name__)


@dataclass
class QualityValidationResult:
    """Structure for quality validation results"""
    method: str
    image_path: str
    ssim_improvement: float
    visual_quality_score: float
    file_size_reduction: float
    processing_time: float
    success: bool
    validation_notes: List[str]

    # Additional detailed metrics
    default_ssim: float = 0.0
    optimized_ssim: float = 0.0
    absolute_ssim_improvement: float = 0.0
    edge_preservation_score: float = 0.0
    color_accuracy_score: float = 0.0
    shape_fidelity_score: float = 0.0
    perceptual_hash_similarity: float = 0.0
    compression_efficiency: float = 0.0
    consistency_score: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    statistical_significance: bool = False
    p_value: float = 1.0
    regression_detected: bool = False
    quality_grade: str = "F"
    validation_timestamp: str = ""


class ComprehensiveQualityValidator:
    """Comprehensive quality validation for all optimization methods"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Quality thresholds for each method
        self.quality_thresholds = {
            'method1': {'ssim_min': 0.15, 'time_max': 0.1},
            'method2': {'ssim_min': 0.25, 'time_max': 5.0},
            'method3': {'ssim_min': 0.35, 'time_max': 30.0}
        }

        # Validation criteria
        self.validation_criteria = {
            'minimum_improvement': 0.10,    # At least 10% improvement
            'maximum_degradation': -0.05,   # No more than 5% quality loss
            'file_size_limit': 5.0,         # Max 5MB SVG files
            'processing_timeout': 60.0      # Max 60s processing time
        }

        # Initialize quality measurement infrastructure
        self.quality_metrics = OptimizationQualityMetrics()
        self.harness = VTracerTestHarness(timeout=30)
        self.bounds = VTracerParameterBounds()

        # Historical quality data for regression detection
        self.quality_history = {}

        # Batch validation results
        self.batch_results = []

        # Dashboard data
        self.dashboard_data = {
            'real_time_metrics': [],
            'quality_trends': [],
            'method_performance': {},
            'alerts': []
        }

    def validate_optimization_quality(self,
                                    method: str,
                                    image_path: str,
                                    optimization_result: Dict[str, Any],
                                    runs: int = 3) -> QualityValidationResult:
        """Comprehensively validate optimization quality"""

        validation_notes = []
        success = True
        start_time = time.time()

        try:
            # Validate SSIM improvement
            ssim_validation = self._validate_ssim_improvement(
                method, image_path, optimization_result, runs
            )

            # Validate visual quality
            visual_quality = self._validate_visual_quality(
                image_path, optimization_result
            )

            # Validate file size
            file_size_validation = self._validate_file_size(
                image_path, optimization_result
            )

            # Validate processing time
            processing_time = optimization_result.get('processing_time', 0.0)
            time_validation = self._validate_processing_time(method, processing_time)

            # Apply method-specific validation
            method_validation = self._validate_method_specific(
                method, ssim_validation['improvement'], processing_time
            )

            # Statistical significance testing
            statistical_analysis = self._perform_statistical_validation(
                image_path, optimization_result, runs
            )

            # Quality consistency validation
            consistency_analysis = self._validate_quality_consistency(
                image_path, optimization_result, runs
            )

            # Regression detection
            regression_analysis = self._detect_quality_regression(
                method, image_path, ssim_validation['improvement']
            )

            # Combine all validation results
            ssim_improvement = ssim_validation['improvement']
            visual_quality_score = visual_quality['overall_score']
            file_size_reduction = file_size_validation['reduction_percentage']

            # Determine overall success
            if not method_validation['success']:
                success = False
                validation_notes.extend(method_validation['notes'])

            if ssim_improvement < self.validation_criteria['minimum_improvement']:
                if ssim_improvement < self.validation_criteria['maximum_degradation']:
                    success = False
                    validation_notes.append(f"Quality degradation detected: {ssim_improvement:.1f}%")

            if processing_time > self.validation_criteria['processing_timeout']:
                success = False
                validation_notes.append(f"Processing timeout: {processing_time:.1f}s")

            # Calculate quality grade
            quality_grade = self._calculate_quality_grade(
                ssim_improvement, visual_quality_score, processing_time, method
            )

            # Create comprehensive result
            result = QualityValidationResult(
                method=method,
                image_path=image_path,
                ssim_improvement=ssim_improvement,
                visual_quality_score=visual_quality_score,
                file_size_reduction=file_size_reduction,
                processing_time=processing_time,
                success=success,
                validation_notes=validation_notes,
                default_ssim=ssim_validation.get('default_ssim', 0.0),
                optimized_ssim=ssim_validation.get('optimized_ssim', 0.0),
                absolute_ssim_improvement=ssim_validation.get('absolute_improvement', 0.0),
                edge_preservation_score=visual_quality.get('edge_preservation', 0.0),
                color_accuracy_score=visual_quality.get('color_accuracy', 0.0),
                shape_fidelity_score=visual_quality.get('shape_fidelity', 0.0),
                perceptual_hash_similarity=visual_quality.get('perceptual_similarity', 0.0),
                compression_efficiency=file_size_validation.get('efficiency_score', 0.0),
                consistency_score=consistency_analysis.get('consistency_score', 0.0),
                confidence_interval=statistical_analysis.get('confidence_interval', (0.0, 0.0)),
                statistical_significance=statistical_analysis.get('significant', False),
                p_value=statistical_analysis.get('p_value', 1.0),
                regression_detected=regression_analysis.get('regression_detected', False),
                quality_grade=quality_grade,
                validation_timestamp=datetime.now().isoformat()
            )

            # Update dashboard data
            self._update_dashboard_data(result)

            # Store for regression detection
            self._store_quality_history(method, image_path, ssim_improvement)

        except Exception as e:
            self.logger.error(f"Quality validation failed: {e}", exc_info=True)
            success = False
            validation_notes.append(f"Validation error: {str(e)}")

            result = QualityValidationResult(
                method=method,
                image_path=image_path,
                ssim_improvement=0.0,
                visual_quality_score=0.0,
                file_size_reduction=0.0,
                processing_time=time.time() - start_time,
                success=success,
                validation_notes=validation_notes,
                validation_timestamp=datetime.now().isoformat()
            )

        return result

    def _validate_ssim_improvement(self,
                                  method: str,
                                  image_path: str,
                                  optimization_result: Dict[str, Any],
                                  runs: int = 3) -> Dict[str, Any]:
        """Validate SSIM improvement with comprehensive analysis"""

        ssim_validation = {
            'improvement': 0.0,
            'default_ssim': 0.0,
            'optimized_ssim': 0.0,
            'absolute_improvement': 0.0,
            'measurements': [],
            'edge_cases_handled': True,
            'rendering_successful': True
        }

        try:
            # Get optimized parameters
            optimized_params = optimization_result.get('parameters', {})
            if not optimized_params:
                optimized_params = optimization_result.get('optimized_parameters', {})

            if not optimized_params:
                self.logger.warning(f"No optimized parameters found for {image_path}")
                return ssim_validation

            # Get default parameters for comparison
            default_params = self.bounds.get_default_parameters()

            # Perform multiple measurements for statistical validity
            default_ssims = []
            optimized_ssims = []

            for run in range(runs):
                try:
                    # Test default parameters
                    default_result = self.harness.test_parameters(image_path, default_params)
                    if default_result.get('success', False):
                        default_ssims.append(default_result['metrics']['ssim'])

                    # Test optimized parameters
                    optimized_result = self.harness.test_parameters(image_path, optimized_params)
                    if optimized_result.get('success', False):
                        optimized_ssims.append(optimized_result['metrics']['ssim'])

                    # Clear cache between runs
                    self.harness.clear_cache()

                except Exception as e:
                    self.logger.warning(f"SSIM measurement run {run+1} failed: {e}")
                    continue

            # Calculate average values if we have successful measurements
            if default_ssims and optimized_ssims:
                default_ssim = statistics.mean(default_ssims)
                optimized_ssim = statistics.mean(optimized_ssims)

                # Calculate improvement percentage
                if default_ssim > 0:
                    improvement = ((optimized_ssim - default_ssim) / default_ssim) * 100
                else:
                    improvement = 100.0 if optimized_ssim > 0 else 0.0

                ssim_validation.update({
                    'improvement': improvement,
                    'default_ssim': default_ssim,
                    'optimized_ssim': optimized_ssim,
                    'absolute_improvement': optimized_ssim - default_ssim,
                    'measurements': {
                        'default_ssims': default_ssims,
                        'optimized_ssims': optimized_ssims,
                        'default_std': statistics.stdev(default_ssims) if len(default_ssims) > 1 else 0.0,
                        'optimized_std': statistics.stdev(optimized_ssims) if len(optimized_ssims) > 1 else 0.0
                    }
                })

                # Handle edge cases
                if default_ssim < 0.1:
                    ssim_validation['edge_cases_handled'] = False
                    self.logger.warning(f"Very low default SSIM ({default_ssim:.3f}) for {image_path}")

                if optimized_ssim > 1.0:
                    ssim_validation['edge_cases_handled'] = False
                    self.logger.warning(f"Invalid optimized SSIM ({optimized_ssim:.3f}) for {image_path}")

            else:
                ssim_validation['rendering_successful'] = False
                self.logger.error(f"Failed to measure SSIM for {image_path}")

        except Exception as e:
            self.logger.error(f"SSIM validation failed for {image_path}: {e}")
            ssim_validation['rendering_successful'] = False

        return ssim_validation

    def _validate_visual_quality(self,
                               image_path: str,
                               optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced visual quality metrics validation"""

        visual_quality = {
            'overall_score': 0.0,
            'edge_preservation': 0.0,
            'color_accuracy': 0.0,
            'shape_fidelity': 0.0,
            'perceptual_similarity': 0.0,
            'structural_similarity': 0.0,
            'detail_preservation': 0.0,
            'artifact_detection': 0.0
        }

        try:
            # Get SVG path from optimization result
            svg_path = optimization_result.get('svg_path', '')
            if not svg_path or not os.path.exists(svg_path):
                self.logger.warning(f"SVG path not found: {svg_path}")
                return visual_quality

            # Load original image
            original = cv2.imread(image_path)
            if original is None:
                self.logger.error(f"Could not load original image: {image_path}")
                return visual_quality

            # Render SVG to same dimensions as original
            try:
                svg_png_data = cairosvg.svg2png(
                    url=svg_path,
                    output_width=original.shape[1],
                    output_height=original.shape[0]
                )
                svg_image = Image.open(io.BytesIO(svg_png_data)).convert('RGB')
                svg_array = np.array(svg_image)
            except Exception as e:
                self.logger.error(f"Failed to render SVG {svg_path}: {e}")
                return visual_quality

            # Edge preservation metric
            edge_preservation = self._measure_edge_preservation_advanced(original, svg_array)
            visual_quality['edge_preservation'] = edge_preservation

            # Color accuracy using multiple color spaces
            color_accuracy = self._measure_color_accuracy_advanced(original, svg_array)
            visual_quality['color_accuracy'] = color_accuracy

            # Shape fidelity using multiple methods
            shape_fidelity = self._measure_shape_fidelity_advanced(original, svg_array)
            visual_quality['shape_fidelity'] = shape_fidelity

            # Perceptual hash similarity
            perceptual_similarity = self._measure_perceptual_similarity(original, svg_array)
            visual_quality['perceptual_similarity'] = perceptual_similarity

            # Structural similarity beyond basic SSIM
            structural_similarity = self._measure_structural_similarity_advanced(original, svg_array)
            visual_quality['structural_similarity'] = structural_similarity

            # Detail preservation analysis
            detail_preservation = self._measure_detail_preservation(original, svg_array)
            visual_quality['detail_preservation'] = detail_preservation

            # Visual artifact detection
            artifact_score = self._detect_visual_artifacts(svg_array)
            visual_quality['artifact_detection'] = artifact_score

            # Calculate overall score (weighted combination)
            weights = {
                'edge_preservation': 0.20,
                'color_accuracy': 0.20,
                'shape_fidelity': 0.15,
                'perceptual_similarity': 0.15,
                'structural_similarity': 0.15,
                'detail_preservation': 0.10,
                'artifact_detection': 0.05
            }

            overall_score = sum(
                visual_quality[metric] * weight
                for metric, weight in weights.items()
            )
            visual_quality['overall_score'] = overall_score

        except Exception as e:
            self.logger.error(f"Visual quality validation failed: {e}")

        return visual_quality

    def _validate_file_size(self,
                          image_path: str,
                          optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """File size optimization validation"""

        file_size_validation = {
            'reduction_percentage': 0.0,
            'original_size_bytes': 0,
            'svg_size_bytes': 0,
            'compression_ratio': 0.0,
            'efficiency_score': 0.0,
            'size_appropriate': True
        }

        try:
            # Get original file size
            original_size = os.path.getsize(image_path)
            file_size_validation['original_size_bytes'] = original_size

            # Get SVG file size
            svg_path = optimization_result.get('svg_path', '')
            if svg_path and os.path.exists(svg_path):
                svg_size = os.path.getsize(svg_path)
                file_size_validation['svg_size_bytes'] = svg_size

                # Calculate reduction percentage
                if original_size > 0:
                    reduction = ((original_size - svg_size) / original_size) * 100
                    file_size_validation['reduction_percentage'] = reduction

                    # Compression ratio
                    compression_ratio = original_size / svg_size if svg_size > 0 else 0
                    file_size_validation['compression_ratio'] = compression_ratio

                    # Efficiency score (considering quality and size)
                    quality_score = optimization_result.get('quality_improvement', 0) / 100.0
                    size_score = min(reduction / 50.0, 1.0)  # Normalize to 50% reduction as excellent
                    efficiency_score = (quality_score * 0.7) + (size_score * 0.3)
                    file_size_validation['efficiency_score'] = efficiency_score

                # Check if size is appropriate (not too large)
                max_size_mb = self.validation_criteria['file_size_limit']
                if svg_size > (max_size_mb * 1024 * 1024):
                    file_size_validation['size_appropriate'] = False

            else:
                self.logger.warning(f"SVG file not found: {svg_path}")

        except Exception as e:
            self.logger.error(f"File size validation failed: {e}")

        return file_size_validation

    def _validate_processing_time(self, method: str, processing_time: float) -> Dict[str, Any]:
        """Processing time validation"""

        time_validation = {
            'within_threshold': False,
            'threshold': 0.0,
            'performance_grade': 'F'
        }

        try:
            threshold = self.quality_thresholds.get(method, {}).get('time_max', 60.0)
            time_validation['threshold'] = threshold
            time_validation['within_threshold'] = processing_time <= threshold

            # Performance grade based on time
            if processing_time <= threshold * 0.5:
                time_validation['performance_grade'] = 'A'
            elif processing_time <= threshold * 0.75:
                time_validation['performance_grade'] = 'B'
            elif processing_time <= threshold:
                time_validation['performance_grade'] = 'C'
            elif processing_time <= threshold * 1.5:
                time_validation['performance_grade'] = 'D'
            else:
                time_validation['performance_grade'] = 'F'

        except Exception as e:
            self.logger.error(f"Processing time validation failed: {e}")

        return time_validation

    def _validate_method_specific(self,
                                method: str,
                                ssim_improvement: float,
                                processing_time: float) -> Dict[str, Any]:
        """Apply method-specific validation criteria"""

        validation = {
            'success': True,
            'notes': []
        }

        try:
            thresholds = self.quality_thresholds.get(method, {})

            # Check SSIM improvement threshold
            min_ssim = thresholds.get('ssim_min', 0.10) * 100  # Convert to percentage
            if ssim_improvement < min_ssim:
                validation['success'] = False
                validation['notes'].append(
                    f"{method} SSIM improvement ({ssim_improvement:.1f}%) below threshold ({min_ssim:.1f}%)"
                )

            # Check processing time threshold
            max_time = thresholds.get('time_max', 60.0)
            if processing_time > max_time:
                validation['success'] = False
                validation['notes'].append(
                    f"{method} processing time ({processing_time:.1f}s) exceeds threshold ({max_time:.1f}s)"
                )

            # Method-specific additional checks
            if method == 'method1':
                # Method 1 should be very fast
                if processing_time > 0.2:
                    validation['notes'].append("Method 1 processing slower than expected")
            elif method == 'method2':
                # Method 2 should show learning progress
                if ssim_improvement < 20.0:  # 20% minimum
                    validation['notes'].append("Method 2 (PPO) learning may need improvement")
            elif method == 'method3':
                # Method 3 should handle complex cases well
                if ssim_improvement < 30.0:  # 30% minimum
                    validation['notes'].append("Method 3 spatial optimization underperforming")

        except Exception as e:
            self.logger.error(f"Method-specific validation failed: {e}")
            validation['success'] = False
            validation['notes'].append(f"Validation error: {str(e)}")

        return validation

    def _perform_statistical_validation(self,
                                      image_path: str,
                                      optimization_result: Dict[str, Any],
                                      runs: int = 3) -> Dict[str, Any]:
        """Statistical quality validation with confidence intervals"""

        statistical_analysis = {
            'significant': False,
            'p_value': 1.0,
            'confidence_interval': (0.0, 0.0),
            'effect_size': 0.0,
            'power_analysis': 0.0
        }

        try:
            if runs < 2:
                return statistical_analysis

            # Get measurement data
            optimized_params = optimization_result.get('parameters', {})
            default_params = self.bounds.get_default_parameters()

            # Collect multiple measurements
            default_measurements = []
            optimized_measurements = []

            for _ in range(runs):
                try:
                    # Default measurement
                    default_result = self.harness.test_parameters(image_path, default_params)
                    if default_result.get('success', False):
                        default_measurements.append(default_result['metrics']['ssim'])

                    # Optimized measurement
                    optimized_result = self.harness.test_parameters(image_path, optimized_params)
                    if optimized_result.get('success', False):
                        optimized_measurements.append(optimized_result['metrics']['ssim'])

                    self.harness.clear_cache()
                except:
                    continue

            # Statistical testing if we have enough data
            if len(default_measurements) >= 2 and len(optimized_measurements) >= 2:
                # Paired t-test
                t_stat, p_value = stats.ttest_ind(optimized_measurements, default_measurements)
                statistical_analysis['p_value'] = p_value
                statistical_analysis['significant'] = p_value < 0.05

                # Confidence interval for the difference
                diff_mean = np.mean(optimized_measurements) - np.mean(default_measurements)
                diff_se = np.sqrt(
                    np.var(optimized_measurements) / len(optimized_measurements) +
                    np.var(default_measurements) / len(default_measurements)
                )
                ci_lower = diff_mean - 1.96 * diff_se
                ci_upper = diff_mean + 1.96 * diff_se
                statistical_analysis['confidence_interval'] = (ci_lower, ci_upper)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    ((len(optimized_measurements) - 1) * np.var(optimized_measurements) +
                     (len(default_measurements) - 1) * np.var(default_measurements)) /
                    (len(optimized_measurements) + len(default_measurements) - 2)
                )
                if pooled_std > 0:
                    effect_size = diff_mean / pooled_std
                    statistical_analysis['effect_size'] = effect_size

        except Exception as e:
            self.logger.error(f"Statistical validation failed: {e}")

        return statistical_analysis

    def _validate_quality_consistency(self,
                                    image_path: str,
                                    optimization_result: Dict[str, Any],
                                    runs: int = 3) -> Dict[str, Any]:
        """Validate quality improvement consistency across multiple runs"""

        consistency_analysis = {
            'consistency_score': 0.0,
            'variance': 0.0,
            'coefficient_of_variation': 0.0,
            'stable': True
        }

        try:
            optimized_params = optimization_result.get('parameters', {})
            improvements = []

            # Collect multiple measurements
            for _ in range(runs):
                try:
                    result = self.harness.test_parameters(image_path, optimized_params)
                    if result.get('success', False):
                        improvements.append(result['metrics']['ssim'])
                    self.harness.clear_cache()
                except:
                    continue

            if len(improvements) >= 2:
                # Calculate consistency metrics
                mean_improvement = np.mean(improvements)
                variance = np.var(improvements)
                std_dev = np.std(improvements)

                consistency_analysis['variance'] = variance

                # Coefficient of variation (relative standard deviation)
                if mean_improvement > 0:
                    cv = std_dev / mean_improvement
                    consistency_analysis['coefficient_of_variation'] = cv

                    # Consistency score (lower CV = higher consistency)
                    consistency_score = max(0, 1.0 - cv * 2)  # Scale CV to 0-1
                    consistency_analysis['consistency_score'] = consistency_score

                    # Consider stable if CV < 0.1 (10% relative variation)
                    consistency_analysis['stable'] = cv < 0.1

        except Exception as e:
            self.logger.error(f"Quality consistency validation failed: {e}")

        return consistency_analysis

    def _detect_quality_regression(self,
                                 method: str,
                                 image_path: str,
                                 current_improvement: float) -> Dict[str, Any]:
        """Detect quality regression compared to historical performance"""

        regression_analysis = {
            'regression_detected': False,
            'historical_mean': 0.0,
            'deviation_from_mean': 0.0,
            'trend_direction': 'stable'
        }

        try:
            # Create key for this method-image combination
            key = f"{method}:{os.path.basename(image_path)}"

            # Get historical data
            if key in self.quality_history:
                historical_improvements = self.quality_history[key]

                if len(historical_improvements) >= 3:  # Need enough history
                    historical_mean = np.mean(historical_improvements)
                    historical_std = np.std(historical_improvements)

                    regression_analysis['historical_mean'] = historical_mean
                    deviation = current_improvement - historical_mean
                    regression_analysis['deviation_from_mean'] = deviation

                    # Detect regression (current performance significantly lower)
                    if deviation < -2 * historical_std:  # 2 standard deviations below
                        regression_analysis['regression_detected'] = True

                    # Trend analysis
                    if len(historical_improvements) >= 5:
                        recent_trend = np.polyfit(
                            range(len(historical_improvements)),
                            historical_improvements,
                            1
                        )[0]

                        if recent_trend > 0.1:
                            regression_analysis['trend_direction'] = 'improving'
                        elif recent_trend < -0.1:
                            regression_analysis['trend_direction'] = 'declining'

        except Exception as e:
            self.logger.error(f"Quality regression detection failed: {e}")

        return regression_analysis

    def _store_quality_history(self, method: str, image_path: str, improvement: float):
        """Store quality improvement for regression detection"""
        try:
            key = f"{method}:{os.path.basename(image_path)}"

            if key not in self.quality_history:
                self.quality_history[key] = []

            self.quality_history[key].append(improvement)

            # Keep only last 20 measurements
            if len(self.quality_history[key]) > 20:
                self.quality_history[key] = self.quality_history[key][-20:]

        except Exception as e:
            self.logger.error(f"Failed to store quality history: {e}")

    def _calculate_quality_grade(self,
                               ssim_improvement: float,
                               visual_quality_score: float,
                               processing_time: float,
                               method: str) -> str:
        """Calculate overall quality grade"""

        try:
            # Get method thresholds
            thresholds = self.quality_thresholds.get(method, {})
            min_ssim = thresholds.get('ssim_min', 0.10) * 100
            max_time = thresholds.get('time_max', 60.0)

            # Score components (0-100 scale)
            ssim_score = min(100, max(0, (ssim_improvement / min_ssim) * 100))
            visual_score = visual_quality_score * 100
            time_score = max(0, 100 - (processing_time / max_time) * 100)

            # Weighted overall score
            overall_score = (ssim_score * 0.5) + (visual_score * 0.3) + (time_score * 0.2)

            # Convert to letter grade
            if overall_score >= 90:
                return 'A'
            elif overall_score >= 80:
                return 'B'
            elif overall_score >= 70:
                return 'C'
            elif overall_score >= 60:
                return 'D'
            else:
                return 'F'

        except Exception as e:
            self.logger.error(f"Quality grade calculation failed: {e}")
            return 'F'

    def _update_dashboard_data(self, result: QualityValidationResult):
        """Update real-time dashboard data"""
        try:
            # Add to real-time metrics
            self.dashboard_data['real_time_metrics'].append({
                'timestamp': result.validation_timestamp,
                'method': result.method,
                'image': os.path.basename(result.image_path),
                'ssim_improvement': result.ssim_improvement,
                'visual_quality': result.visual_quality_score,
                'processing_time': result.processing_time,
                'success': result.success,
                'grade': result.quality_grade
            })

            # Keep only last 100 real-time entries
            if len(self.dashboard_data['real_time_metrics']) > 100:
                self.dashboard_data['real_time_metrics'] = self.dashboard_data['real_time_metrics'][-100:]

            # Update method performance tracking
            method = result.method
            if method not in self.dashboard_data['method_performance']:
                self.dashboard_data['method_performance'][method] = {
                    'total_validations': 0,
                    'success_count': 0,
                    'average_improvement': 0.0,
                    'average_time': 0.0,
                    'quality_trend': []
                }

            perf = self.dashboard_data['method_performance'][method]
            perf['total_validations'] += 1
            if result.success:
                perf['success_count'] += 1

            # Update running averages
            total = perf['total_validations']
            perf['average_improvement'] = (
                (perf['average_improvement'] * (total - 1) + result.ssim_improvement) / total
            )
            perf['average_time'] = (
                (perf['average_time'] * (total - 1) + result.processing_time) / total
            )

            # Add to quality trend
            perf['quality_trend'].append({
                'timestamp': result.validation_timestamp,
                'improvement': result.ssim_improvement
            })
            if len(perf['quality_trend']) > 50:
                perf['quality_trend'] = perf['quality_trend'][-50:]

            # Generate alerts for significant issues
            if result.regression_detected:
                self.dashboard_data['alerts'].append({
                    'timestamp': result.validation_timestamp,
                    'type': 'regression',
                    'method': result.method,
                    'message': f"Quality regression detected for {result.method}",
                    'severity': 'high'
                })

            if not result.success and result.ssim_improvement < -5.0:
                self.dashboard_data['alerts'].append({
                    'timestamp': result.validation_timestamp,
                    'type': 'quality_degradation',
                    'method': result.method,
                    'message': f"Significant quality degradation: {result.ssim_improvement:.1f}%",
                    'severity': 'critical'
                })

            # Keep only last 20 alerts
            if len(self.dashboard_data['alerts']) > 20:
                self.dashboard_data['alerts'] = self.dashboard_data['alerts'][-20:]

        except Exception as e:
            self.logger.error(f"Dashboard data update failed: {e}")

    # Advanced visual quality measurement methods

    def _measure_edge_preservation_advanced(self, original: np.ndarray, svg_array: np.ndarray) -> float:
        """Advanced edge preservation measurement with multiple methods"""
        try:
            # Convert to grayscale
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            svg_gray = cv2.cvtColor(svg_array, cv2.COLOR_RGB2GRAY)

            # Multiple edge detection methods
            edge_scores = []

            # Canny edge detection
            orig_canny = cv2.Canny(orig_gray, 50, 150)
            svg_canny = cv2.Canny(svg_gray, 50, 150)
            canny_overlap = np.logical_and(orig_canny > 0, svg_canny > 0).sum()
            orig_edge_pixels = (orig_canny > 0).sum()
            if orig_edge_pixels > 0:
                edge_scores.append(canny_overlap / orig_edge_pixels)

            # Sobel edge detection
            orig_sobel = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 1, ksize=3)
            svg_sobel = cv2.Sobel(svg_gray, cv2.CV_64F, 1, 1, ksize=3)
            sobel_corr = np.corrcoef(orig_sobel.flatten(), svg_sobel.flatten())[0, 1]
            if not np.isnan(sobel_corr):
                edge_scores.append(max(0, sobel_corr))

            # Laplacian edge detection
            orig_lap = cv2.Laplacian(orig_gray, cv2.CV_64F)
            svg_lap = cv2.Laplacian(svg_gray, cv2.CV_64F)
            lap_corr = np.corrcoef(orig_lap.flatten(), svg_lap.flatten())[0, 1]
            if not np.isnan(lap_corr):
                edge_scores.append(max(0, lap_corr))

            return np.mean(edge_scores) if edge_scores else 0.0

        except Exception as e:
            self.logger.warning(f"Advanced edge preservation measurement failed: {e}")
            return 0.0

    def _measure_color_accuracy_advanced(self, original: np.ndarray, svg_array: np.ndarray) -> float:
        """Advanced color accuracy measurement using multiple color spaces"""
        try:
            # Resize to same dimensions if needed
            if original.shape != svg_array.shape:
                svg_array = cv2.resize(svg_array, (original.shape[1], original.shape[0]))

            color_scores = []

            # RGB space comparison
            rgb_mse = np.mean((original - svg_array) ** 2)
            rgb_score = 1.0 / (1.0 + rgb_mse / 1000.0)  # Normalize MSE
            color_scores.append(rgb_score)

            # LAB space comparison (perceptually uniform)
            orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
            svg_lab = cv2.cvtColor(svg_array, cv2.COLOR_RGB2LAB)
            lab_mse = np.mean((orig_lab - svg_lab) ** 2)
            lab_score = 1.0 / (1.0 + lab_mse / 1000.0)
            color_scores.append(lab_score)

            # HSV space comparison
            orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            svg_hsv = cv2.cvtColor(svg_array, cv2.COLOR_RGB2HSV)
            hsv_mse = np.mean((orig_hsv - svg_hsv) ** 2)
            hsv_score = 1.0 / (1.0 + hsv_mse / 1000.0)
            color_scores.append(hsv_score)

            # Color histogram comparison
            orig_hist = cv2.calcHist([original], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            svg_hist = cv2.calcHist([svg_array], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist_corr = cv2.compareHist(orig_hist, svg_hist, cv2.HISTCMP_CORREL)
            color_scores.append(max(0, hist_corr))

            return np.mean(color_scores)

        except Exception as e:
            self.logger.warning(f"Advanced color accuracy measurement failed: {e}")
            return 0.0

    def _measure_shape_fidelity_advanced(self, original: np.ndarray, svg_array: np.ndarray) -> float:
        """Advanced shape fidelity measurement using multiple methods"""
        try:
            # Convert to grayscale and threshold
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            svg_gray = cv2.cvtColor(svg_array, cv2.COLOR_RGB2GRAY)

            _, orig_binary = cv2.threshold(orig_gray, 127, 255, cv2.THRESH_BINARY)
            _, svg_binary = cv2.threshold(svg_gray, 127, 255, cv2.THRESH_BINARY)

            shape_scores = []

            # Hu moments comparison
            orig_moments = cv2.moments(orig_binary)
            svg_moments = cv2.moments(svg_binary)

            if orig_moments['m00'] > 0 and svg_moments['m00'] > 0:
                orig_hu = cv2.HuMoments(orig_moments).flatten()
                svg_hu = cv2.HuMoments(svg_moments).flatten()

                # Log transform for scale invariance
                orig_hu_log = -np.sign(orig_hu) * np.log10(np.abs(orig_hu) + 1e-10)
                svg_hu_log = -np.sign(svg_hu) * np.log10(np.abs(svg_hu) + 1e-10)

                hu_distance = np.linalg.norm(orig_hu_log - svg_hu_log)
                hu_score = 1.0 / (1.0 + hu_distance)
                shape_scores.append(hu_score)

            # Contour matching
            orig_contours, _ = cv2.findContours(orig_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            svg_contours, _ = cv2.findContours(svg_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if orig_contours and svg_contours:
                # Use largest contour from each
                orig_contour = max(orig_contours, key=cv2.contourArea)
                svg_contour = max(svg_contours, key=cv2.contourArea)

                # Shape context matching (simplified)
                contour_match = cv2.matchShapes(orig_contour, svg_contour, cv2.CONTOURS_MATCH_I1, 0)
                contour_score = 1.0 / (1.0 + contour_match)
                shape_scores.append(contour_score)

            # Binary image similarity
            binary_overlap = np.logical_and(orig_binary > 0, svg_binary > 0).sum()
            binary_union = np.logical_or(orig_binary > 0, svg_binary > 0).sum()
            if binary_union > 0:
                jaccard_score = binary_overlap / binary_union
                shape_scores.append(jaccard_score)

            return np.mean(shape_scores) if shape_scores else 0.0

        except Exception as e:
            self.logger.warning(f"Advanced shape fidelity measurement failed: {e}")
            return 0.0

    def _measure_perceptual_similarity(self, original: np.ndarray, svg_array: np.ndarray) -> float:
        """Perceptual hash similarity measurement"""
        try:
            # Simple perceptual hash implementation
            def perceptual_hash(image, hash_size=8):
                # Resize and convert to grayscale
                resized = cv2.resize(image, (hash_size + 1, hash_size))
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

                # Compute the DCT
                dct = cv2.dct(np.float32(gray))

                # Get the top-left hash_size x hash_size region
                dct_low = dct[:hash_size, :hash_size]

                # Compute hash based on median
                median = np.median(dct_low)
                return (dct_low > median).flatten()

            # Compute hashes
            orig_hash = perceptual_hash(original)
            svg_hash = perceptual_hash(svg_array)

            # Hamming distance
            hamming_distance = np.sum(orig_hash != svg_hash)
            max_distance = len(orig_hash)

            # Convert to similarity score
            similarity = 1.0 - (hamming_distance / max_distance)
            return similarity

        except Exception as e:
            self.logger.warning(f"Perceptual similarity measurement failed: {e}")
            return 0.0

    def _measure_structural_similarity_advanced(self, original: np.ndarray, svg_array: np.ndarray) -> float:
        """Advanced structural similarity beyond basic SSIM"""
        try:
            # Convert to grayscale
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            svg_gray = cv2.cvtColor(svg_array, cv2.COLOR_RGB2GRAY)

            # Resize if needed
            if orig_gray.shape != svg_gray.shape:
                svg_gray = cv2.resize(svg_gray, (orig_gray.shape[1], orig_gray.shape[0]))

            # Multi-scale SSIM
            ssim_scores = []
            scales = [(1.0, 1.0), (0.5, 0.5), (0.25, 0.25)]

            for scale in scales:
                if scale != (1.0, 1.0):
                    h, w = orig_gray.shape
                    new_h, new_w = int(h * scale[0]), int(w * scale[1])
                    if new_h > 0 and new_w > 0:
                        orig_scaled = cv2.resize(orig_gray, (new_w, new_h))
                        svg_scaled = cv2.resize(svg_gray, (new_w, new_h))
                    else:
                        continue
                else:
                    orig_scaled = orig_gray
                    svg_scaled = svg_gray

                if orig_scaled.shape[0] >= 7 and orig_scaled.shape[1] >= 7:  # Minimum size for SSIM
                    score = ssim(orig_scaled, svg_scaled, data_range=255)
                    ssim_scores.append(score)

            return np.mean(ssim_scores) if ssim_scores else 0.0

        except Exception as e:
            self.logger.warning(f"Advanced structural similarity measurement failed: {e}")
            return 0.0

    def _measure_detail_preservation(self, original: np.ndarray, svg_array: np.ndarray) -> float:
        """Detail preservation analysis using high-frequency components"""
        try:
            # Convert to grayscale
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            svg_gray = cv2.cvtColor(svg_array, cv2.COLOR_RGB2GRAY)

            # High-pass filter to isolate details
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            orig_details = cv2.filter2D(orig_gray, -1, kernel)
            svg_details = cv2.filter2D(svg_gray, -1, kernel)

            # Correlation between detail maps
            orig_flat = orig_details.flatten()
            svg_flat = svg_details.flatten()

            if len(orig_flat) > 1 and len(svg_flat) > 1:
                correlation = np.corrcoef(orig_flat, svg_flat)[0, 1]
                return max(0.0, correlation) if not np.isnan(correlation) else 0.0

            return 0.0

        except Exception as e:
            self.logger.warning(f"Detail preservation measurement failed: {e}")
            return 0.0

    def _detect_visual_artifacts(self, svg_array: np.ndarray) -> float:
        """Detect visual artifacts in the SVG rendering"""
        try:
            artifact_score = 1.0  # Start with perfect score

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(svg_array, cv2.COLOR_RGB2GRAY)

            # Check for aliasing (jagged edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            total_pixels = gray.shape[0] * gray.shape[1]

            if total_pixels > 0:
                edge_ratio = edge_pixels / total_pixels
                # Penalize excessive edges (possible aliasing)
                if edge_ratio > 0.3:
                    artifact_score -= 0.2

            # Check for color banding (false contours)
            # Look for repeated intensity levels that shouldn't be there
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_peaks = []
            for i in range(1, 255):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 100:
                    hist_peaks.append(i)

            # Too many peaks might indicate banding
            if len(hist_peaks) > 20:
                artifact_score -= 0.1

            # Check for noise (high-frequency variations)
            noise_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            noise_map = cv2.filter2D(gray, -1, noise_kernel)
            noise_level = np.std(noise_map)

            # Penalize excessive noise
            if noise_level > 30:
                artifact_score -= 0.1

            return max(0.0, artifact_score)

        except Exception as e:
            self.logger.warning(f"Visual artifact detection failed: {e}")
            return 1.0  # Assume no artifacts if detection fails

    # Batch validation methods

    def validate_batch(self,
                      image_paths: List[str],
                      optimization_results: List[Dict[str, Any]],
                      methods: List[str]) -> Dict[str, Any]:
        """Validate optimization quality for batch processing"""

        batch_results = []
        start_time = time.time()

        try:
            for i, (image_path, result, method) in enumerate(zip(image_paths, optimization_results, methods)):
                self.logger.info(f"Validating batch item {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

                validation_result = self.validate_optimization_quality(method, image_path, result)
                batch_results.append(validation_result)

            # Generate batch analysis
            batch_analysis = self._analyze_batch_results(batch_results)
            batch_analysis['total_processing_time'] = time.time() - start_time
            batch_analysis['timestamp'] = datetime.now().isoformat()

            # Store batch results
            self.batch_results.extend(batch_results)

            return batch_analysis

        except Exception as e:
            self.logger.error(f"Batch validation failed: {e}")
            return {'error': str(e), 'partial_results': batch_results}

    def _analyze_batch_results(self, results: List[QualityValidationResult]) -> Dict[str, Any]:
        """Analyze batch validation results"""

        analysis = {
            'summary': {
                'total_images': len(results),
                'successful_validations': sum(1 for r in results if r.success),
                'failed_validations': sum(1 for r in results if not r.success),
                'success_rate': 0.0
            },
            'quality_metrics': {
                'mean_ssim_improvement': 0.0,
                'median_ssim_improvement': 0.0,
                'mean_visual_quality': 0.0,
                'mean_processing_time': 0.0
            },
            'method_performance': {},
            'quality_distribution': {},
            'consistency_analysis': {}
        }

        if not results:
            return analysis

        try:
            # Summary statistics
            successful_results = [r for r in results if r.success]
            analysis['summary']['success_rate'] = len(successful_results) / len(results) * 100

            # Quality metrics
            if successful_results:
                ssim_improvements = [r.ssim_improvement for r in successful_results]
                visual_qualities = [r.visual_quality_score for r in successful_results]
                processing_times = [r.processing_time for r in results]

                analysis['quality_metrics'] = {
                    'mean_ssim_improvement': np.mean(ssim_improvements),
                    'median_ssim_improvement': np.median(ssim_improvements),
                    'std_ssim_improvement': np.std(ssim_improvements),
                    'mean_visual_quality': np.mean(visual_qualities),
                    'mean_processing_time': np.mean(processing_times),
                    'total_processing_time': np.sum(processing_times)
                }

            # Method performance breakdown
            methods = set(r.method for r in results)
            for method in methods:
                method_results = [r for r in results if r.method == method]
                method_successful = [r for r in method_results if r.success]

                if method_results:
                    analysis['method_performance'][method] = {
                        'total': len(method_results),
                        'successful': len(method_successful),
                        'success_rate': len(method_successful) / len(method_results) * 100,
                        'mean_improvement': np.mean([r.ssim_improvement for r in method_successful]) if method_successful else 0.0,
                        'mean_processing_time': np.mean([r.processing_time for r in method_results])
                    }

            # Quality grade distribution
            grades = [r.quality_grade for r in results]
            analysis['quality_distribution'] = {
                'A': grades.count('A'),
                'B': grades.count('B'),
                'C': grades.count('C'),
                'D': grades.count('D'),
                'F': grades.count('F')
            }

            # Consistency analysis across batch
            if len(successful_results) > 1:
                improvements = [r.ssim_improvement for r in successful_results]
                analysis['consistency_analysis'] = {
                    'coefficient_of_variation': np.std(improvements) / np.mean(improvements) if np.mean(improvements) > 0 else 0,
                    'range': max(improvements) - min(improvements),
                    'outliers_detected': len([x for x in improvements if abs(x - np.mean(improvements)) > 2 * np.std(improvements)])
                }

        except Exception as e:
            self.logger.error(f"Batch analysis failed: {e}")
            analysis['error'] = str(e)

        return analysis

    # Quality validation automation

    def setup_automated_validation(self,
                                 enable_ci_cd: bool = True,
                                 enable_regression_monitoring: bool = True,
                                 enable_alerts: bool = True) -> Dict[str, Any]:
        """Setup automated quality validation systems"""

        automation_config = {
            'ci_cd_integration': enable_ci_cd,
            'regression_monitoring': enable_regression_monitoring,
            'alert_system': enable_alerts,
            'monitoring_intervals': {
                'real_time': 60,  # seconds
                'batch_summary': 3600,  # 1 hour
                'trend_analysis': 86400  # 24 hours
            }
        }

        if enable_alerts:
            self._setup_quality_alerts()

        return automation_config

    def _setup_quality_alerts(self):
        """Setup quality monitoring alerts"""

        # Define alert thresholds
        self.alert_thresholds = {
            'regression_threshold': -10.0,  # % decline in quality
            'success_rate_threshold': 80.0,  # minimum success rate
            'processing_time_spike': 2.0,  # multiplier for processing time spike
            'consistency_threshold': 0.3  # maximum coefficient of variation
        }

    def generate_quality_dashboard(self) -> str:
        """Generate HTML quality validation dashboard"""

        try:
            dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quality Validation Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .dashboard-header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .alert {{ padding: 10px; margin: 5px 0; border-radius: 4px; }}
        .alert-critical {{ background: #e74c3c; color: white; }}
        .alert-high {{ background: #f39c12; color: white; }}
        .alert-medium {{ background: #f1c40f; color: black; }}
        .status-indicator {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; }}
        .status-good {{ background: #27ae60; }}
        .status-warning {{ background: #f39c12; }}
        .status-error {{ background: #e74c3c; }}
        .refresh-button {{ background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>🧠 ULTRATHINK Quality Validation Dashboard</h1>
        <p>Real-time quality monitoring for all optimization methods</p>
        <button class="refresh-button" onclick="refreshDashboard()">🔄 Refresh</button>
        <span style="float: right;">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{len(self.dashboard_data['real_time_metrics'])}</div>
            <div class="metric-label">Total Validations</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len([m for m in self.dashboard_data['real_time_metrics'] if m['success']])}</div>
            <div class="metric-label">Successful Validations</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(self.dashboard_data['method_performance'])}</div>
            <div class="metric-label">Active Methods</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(self.dashboard_data['alerts'])}</div>
            <div class="metric-label">Active Alerts</div>
        </div>
    </div>

    <div class="chart-container">
        <h3>Real-time Quality Trends</h3>
        <div id="qualityTrendChart" style="height: 400px;"></div>
    </div>

    <div class="chart-container">
        <h3>Method Performance Comparison</h3>
        <div id="methodComparisonChart" style="height: 400px;"></div>
    </div>

    <div class="chart-container">
        <h3>Recent Quality Validations</h3>
        <div id="recentValidationsTable"></div>
    </div>

    <div class="chart-container">
        <h3>System Alerts</h3>
        <div id="alertsContainer">
"""

            # Add alerts
            for alert in self.dashboard_data['alerts'][-10:]:  # Last 10 alerts
                severity_class = f"alert-{alert['severity']}"
                dashboard_html += f"""
            <div class="alert {severity_class}">
                <strong>{alert['type'].replace('_', ' ').title()}:</strong> {alert['message']}
                <small style="float: right;">{alert['timestamp']}</small>
            </div>"""

            dashboard_html += """
        </div>
    </div>

    <script>
        // Quality trend chart
        function updateQualityTrendChart() {
            var data = """ + json.dumps(self.dashboard_data['real_time_metrics'][-50:]) + """;

            var trace = {
                x: data.map(d => d.timestamp),
                y: data.map(d => d.ssim_improvement),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'SSIM Improvement',
                line: {color: '#3498db'}
            };

            var layout = {
                title: 'Quality Improvement Trend',
                xaxis: {title: 'Time'},
                yaxis: {title: 'SSIM Improvement (%)'},
                showlegend: true
            };

            Plotly.newPlot('qualityTrendChart', [trace], layout);
        }

        // Method comparison chart
        function updateMethodComparisonChart() {
            var methodData = """ + json.dumps(self.dashboard_data['method_performance']) + """;

            var methods = Object.keys(methodData);
            var improvements = methods.map(m => methodData[m].average_improvement);
            var times = methods.map(m => methodData[m].average_time);

            var trace1 = {
                x: methods,
                y: improvements,
                type: 'bar',
                name: 'Avg Improvement (%)',
                yaxis: 'y',
                marker: {color: '#27ae60'}
            };

            var trace2 = {
                x: methods,
                y: times,
                type: 'bar',
                name: 'Avg Time (s)',
                yaxis: 'y2',
                marker: {color: '#e74c3c'}
            };

            var layout = {
                title: 'Method Performance Comparison',
                xaxis: {title: 'Optimization Method'},
                yaxis: {title: 'SSIM Improvement (%)', side: 'left'},
                yaxis2: {title: 'Processing Time (s)', side: 'right', overlaying: 'y'},
                showlegend: true
            };

            Plotly.newPlot('methodComparisonChart', [trace1, trace2], layout);
        }

        // Recent validations table
        function updateRecentValidationsTable() {
            var data = """ + json.dumps(self.dashboard_data['real_time_metrics'][-20:]) + """;

            var tableHTML = `
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background: #f8f9fa;">
                        <th style="padding: 10px; border: 1px solid #dee2e6;">Image</th>
                        <th style="padding: 10px; border: 1px solid #dee2e6;">Method</th>
                        <th style="padding: 10px; border: 1px solid #dee2e6;">Improvement</th>
                        <th style="padding: 10px; border: 1px solid #dee2e6;">Time</th>
                        <th style="padding: 10px; border: 1px solid #dee2e6;">Grade</th>
                        <th style="padding: 10px; border: 1px solid #dee2e6;">Status</th>
                    </tr>`;

            data.reverse().forEach(function(item) {
                var statusClass = item.success ? 'status-good' : 'status-error';
                var statusText = item.success ? 'Success' : 'Failed';

                tableHTML += `
                    <tr>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">${item.image}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">${item.method}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">${item.ssim_improvement.toFixed(1)}%</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">${item.processing_time.toFixed(3)}s</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">${item.grade}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">
                            <span class="status-indicator ${statusClass}"></span>${statusText}
                        </td>
                    </tr>`;
            });

            tableHTML += '</table>';
            document.getElementById('recentValidationsTable').innerHTML = tableHTML;
        }

        // Refresh dashboard
        function refreshDashboard() {
            updateQualityTrendChart();
            updateMethodComparisonChart();
            updateRecentValidationsTable();
        }

        // Initialize dashboard
        window.onload = function() {
            refreshDashboard();
            // Auto-refresh every 30 seconds
            setInterval(refreshDashboard, 30000);
        };
    </script>
</body>
</html>"""

            return dashboard_html

        except Exception as e:
            self.logger.error(f"Dashboard generation failed: {e}")
            return f"<html><body><h1>Dashboard Error</h1><p>{str(e)}</p></body></html>"

    def export_validation_report(self,
                                output_dir: str = "quality_validation_reports",
                                include_visualizations: bool = True) -> Dict[str, str]:
        """Export comprehensive quality validation report"""

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Export batch results as JSON
            json_path = output_dir / f"quality_validation_{timestamp}.json"
            report_data = {
                'batch_results': [asdict(result) for result in self.batch_results],
                'dashboard_data': self.dashboard_data,
                'quality_history': self.quality_history,
                'validation_config': {
                    'quality_thresholds': self.quality_thresholds,
                    'validation_criteria': self.validation_criteria
                },
                'export_timestamp': datetime.now().isoformat()
            }

            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            # Export dashboard HTML
            html_path = output_dir / f"quality_dashboard_{timestamp}.html"
            dashboard_html = self.generate_quality_dashboard()

            with open(html_path, 'w') as f:
                f.write(dashboard_html)

            # Export CSV summary
            csv_path = output_dir / f"quality_summary_{timestamp}.csv"

            import csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'image_path', 'method', 'ssim_improvement', 'visual_quality_score',
                    'file_size_reduction', 'processing_time', 'success', 'quality_grade',
                    'validation_timestamp'
                ])

                for result in self.batch_results:
                    writer.writerow([
                        os.path.basename(result.image_path),
                        result.method,
                        f"{result.ssim_improvement:.2f}",
                        f"{result.visual_quality_score:.3f}",
                        f"{result.file_size_reduction:.2f}",
                        f"{result.processing_time:.3f}",
                        result.success,
                        result.quality_grade,
                        result.validation_timestamp
                    ])

            return {
                'json_report': str(json_path),
                'html_dashboard': str(html_path),
                'csv_summary': str(csv_path)
            }

        except Exception as e:
            self.logger.error(f"Report export failed: {e}")
            return {'error': str(e)}

    def get_validation_api_status(self) -> Dict[str, Any]:
        """Get current validation system status for API integration"""

        try:
            total_validations = len(self.batch_results)
            successful_validations = sum(1 for r in self.batch_results if r.success)

            recent_results = self.batch_results[-100:] if self.batch_results else []
            recent_success_rate = (
                sum(1 for r in recent_results if r.success) / len(recent_results) * 100
                if recent_results else 0.0
            )

            return {
                'status': 'operational',
                'total_validations': total_validations,
                'successful_validations': successful_validations,
                'overall_success_rate': (successful_validations / total_validations * 100) if total_validations > 0 else 0.0,
                'recent_success_rate': recent_success_rate,
                'active_methods': len(self.dashboard_data['method_performance']),
                'active_alerts': len(self.dashboard_data['alerts']),
                'last_validation': self.batch_results[-1].validation_timestamp if self.batch_results else None,
                'system_health': 'good' if recent_success_rate >= 80 else 'warning' if recent_success_rate >= 60 else 'critical'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'system_health': 'critical'
            }

    def cleanup(self):
        """Cleanup validation resources"""
        try:
            if hasattr(self, 'quality_metrics'):
                self.quality_metrics.cleanup()
            if hasattr(self, 'harness'):
                # VTracerTestHarness cleanup if available
                pass
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")