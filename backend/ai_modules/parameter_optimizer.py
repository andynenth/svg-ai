#!/usr/bin/env python3
"""
VTracer Parameter Optimization Engine

This module provides intelligent parameter optimization for VTracer based on AI classification
and feature analysis. It maps logo types to optimal parameter sets and provides confidence-based
adjustments with comprehensive validation.

Features:
- Logo type to parameter mapping (simple, text, gradient, complex)
- Confidence-based parameter adjustment
- Feature-driven fine-tuning
- Parameter validation and bounds checking
- Optimization history tracking
- Performance analytics
"""

import logging
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LogoType(Enum):
    """Enumeration of supported logo types for parameter optimization."""
    SIMPLE = "simple"
    TEXT = "text"
    GRADIENT = "gradient"
    COMPLEX = "complex"


@dataclass
class ParameterBounds:
    """Parameter bounds for VTracer parameter validation."""
    colormode_options: List[str]
    color_precision_range: Tuple[int, int]
    layer_difference_range: Tuple[int, int]
    path_precision_range: Tuple[int, int]
    corner_threshold_range: Tuple[int, int]
    length_threshold_range: Tuple[float, float]
    max_iterations_range: Tuple[int, int]
    splice_threshold_range: Tuple[int, int]


@dataclass
class OptimizationResult:
    """Result of parameter optimization with metadata."""
    parameters: Dict[str, Any]
    logo_type: str
    confidence: float
    optimization_method: str
    validation_passed: bool
    adjustments_applied: List[str]
    optimization_time: float


class VTracerParameterOptimizer:
    """Intelligent VTracer parameter optimization based on AI analysis.

    Provides logo type-specific parameter optimization with confidence adjustments,
    feature-based fine-tuning, and comprehensive parameter validation.

    Features:
        - 4 logo type parameter sets (simple, text, gradient, complex)
        - Confidence-based parameter adjustment (high/medium/low confidence)
        - Feature-driven fine-tuning using 6 extracted features
        - Parameter bounds validation with automatic correction
        - Optimization history tracking for analytics
        - Performance monitoring and statistics

    Example:
        Basic parameter optimization:

        optimizer = VTracerParameterOptimizer()
        result = optimizer.optimize_parameters(
            classification={'logo_type': 'simple', 'confidence': 0.85},
            features={'edge_density': 0.1, 'unique_colors': 0.3}
        )
        print(f"Optimized parameters: {result.parameters}")

        Advanced optimization with custom base parameters:

        custom_base = {'color_precision': 5, 'layer_difference': 20}
        result = optimizer.optimize_parameters(
            classification={'logo_type': 'gradient', 'confidence': 0.6},
            features=features,
            base_parameters=custom_base
        )
    """

    def __init__(self):
        """Initialize the parameter optimizer with bounds and default configurations."""
        self.bounds = self._define_parameter_bounds()
        self.optimization_history = []
        self.stats = {
            'total_optimizations': 0,
            'by_logo_type': {},
            'by_confidence_level': {'high': 0, 'medium': 0, 'low': 0},
            'average_optimization_time': 0.0,
            'validation_failures': 0
        }

        logger.info("VTracerParameterOptimizer initialized")

    def _define_parameter_bounds(self) -> ParameterBounds:
        """Define valid parameter bounds for VTracer based on documentation and testing.

        Returns:
            ParameterBounds: Complete parameter bounds specification.
        """
        return ParameterBounds(
            colormode_options=['color', 'binary'],
            color_precision_range=(1, 10),
            layer_difference_range=(1, 32),
            path_precision_range=(0, 10),
            corner_threshold_range=(0, 180),
            length_threshold_range=(0.1, 10.0),
            max_iterations_range=(1, 50),
            splice_threshold_range=(0, 180)
        )

    def optimize_parameters(self,
                          classification: Dict[str, Any],
                          features: Dict[str, float],
                          base_parameters: Optional[Dict[str, Any]] = None,
                          user_overrides: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Optimize VTracer parameters based on AI classification and features.

        Performs comprehensive parameter optimization using logo type mapping,
        confidence adjustments, and feature-based fine-tuning.

        Args:
            classification (Dict[str, Any]): Logo classification results:
                - logo_type (str): One of 'simple', 'text', 'gradient', 'complex'
                - confidence (float): Classification confidence (0.0 to 1.0)
            features (Dict[str, float]): Extracted features (all normalized to [0,1]):
                - edge_density (float): Edge density measure
                - unique_colors (float): Normalized unique color count
                - entropy (float): Shannon entropy measure
                - corner_density (float): Corner density measure
                - gradient_strength (float): Gradient strength measure
                - complexity_score (float): Overall complexity score
            base_parameters (Optional[Dict[str, Any]]): Custom base parameters to start from.
                If None, uses logo type defaults.
            user_overrides (Optional[Dict[str, Any]]): User parameters that override optimization.

        Returns:
            OptimizationResult: Comprehensive optimization result with metadata.

        Raises:
            ValueError: If logo type is not supported or parameters are invalid.
            KeyError: If required classification or feature keys are missing.
        """
        start_time = time.time()

        # Validate inputs
        self._validate_optimization_inputs(classification, features)

        logo_type = classification['logo_type']
        confidence = classification['confidence']

        logger.info(f"Optimizing parameters for {logo_type} logo (confidence: {confidence:.2%})")

        # Step 1: Get base parameters for logo type
        if base_parameters is None:
            base_params = self._get_base_parameters_for_type(logo_type)
            optimization_method = f"logo_type_{logo_type}"
        else:
            base_params = base_parameters.copy()
            optimization_method = f"custom_base_{logo_type}"

        adjustments_applied = []

        # Step 2: Apply confidence-based adjustments
        confidence_adjusted = self._apply_confidence_adjustments(base_params, confidence)
        if confidence_adjusted != base_params:
            adjustments_applied.append(f"confidence_{self._get_confidence_level(confidence)}")

        # Step 3: Apply feature-based fine-tuning
        feature_tuned = self._apply_feature_adjustments(confidence_adjusted, features)
        if feature_tuned != confidence_adjusted:
            adjustments_applied.append("feature_tuning")

        # Step 4: Apply user overrides
        final_params = feature_tuned.copy()
        if user_overrides:
            final_params.update(user_overrides)
            adjustments_applied.append("user_overrides")

        # Step 5: Validate and correct parameters
        validated_params, validation_passed = self._validate_and_correct_parameters(final_params)
        if not validation_passed:
            adjustments_applied.append("bounds_correction")
            self.stats['validation_failures'] += 1

        optimization_time = time.time() - start_time

        # Create optimization result
        result = OptimizationResult(
            parameters=validated_params,
            logo_type=logo_type,
            confidence=confidence,
            optimization_method=optimization_method,
            validation_passed=validation_passed,
            adjustments_applied=adjustments_applied,
            optimization_time=optimization_time
        )

        # Update statistics
        self._update_optimization_stats(result)

        logger.info(f"Parameter optimization complete ({optimization_time*1000:.1f}ms, "
                   f"adjustments: {', '.join(adjustments_applied) if adjustments_applied else 'none'})")

        return result

    def _validate_optimization_inputs(self, classification: Dict[str, Any], features: Dict[str, float]) -> None:
        """Validate optimization inputs for correctness and completeness.

        Args:
            classification (Dict[str, Any]): Classification results to validate.
            features (Dict[str, float]): Features to validate.

        Raises:
            ValueError: If inputs are invalid.
            KeyError: If required keys are missing.
        """
        # Validate classification
        if 'logo_type' not in classification:
            raise KeyError("Classification must contain 'logo_type' key")

        if 'confidence' not in classification:
            raise KeyError("Classification must contain 'confidence' key")

        logo_type = classification['logo_type']
        if logo_type not in [lt.value for lt in LogoType]:
            raise ValueError(f"Unsupported logo type: {logo_type}. "
                           f"Supported types: {[lt.value for lt in LogoType]}")

        confidence = classification['confidence']
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {confidence}")

        # Validate features (all should be in [0,1] range)
        required_features = ['edge_density', 'unique_colors', 'entropy',
                           'corner_density', 'gradient_strength', 'complexity_score']

        for feature_name in required_features:
            if feature_name not in features:
                logger.warning(f"Missing feature '{feature_name}', using default value 0.5")
                features[feature_name] = 0.5
            else:
                value = features[feature_name]
                if not 0.0 <= value <= 1.0:
                    logger.warning(f"Feature '{feature_name}' value {value} outside [0,1] range, clipping")
                    features[feature_name] = max(0.0, min(1.0, value))

    def _get_base_parameters_for_type(self, logo_type: str) -> Dict[str, Any]:
        """Get optimized base parameters for specific logo type.

        Parameter sets are researched and tested for optimal results:
        - Simple: Clean parameters for geometric shapes
        - Text: High precision for readable text
        - Gradient: Maximum precision for smooth transitions
        - Complex: Balanced parameters for detail preservation

        Args:
            logo_type (str): Logo type ('simple', 'text', 'gradient', 'complex').

        Returns:
            Dict[str, Any]: Base VTracer parameters optimized for the logo type.
        """
        parameter_sets = {
            LogoType.SIMPLE.value: {
                'colormode': 'color',
                'color_precision': 3,          # Fewer colors for clean output
                'layer_difference': 32,        # High separation for distinct regions
                'path_precision': 6,           # High precision for sharp edges
                'corner_threshold': 30,        # Sharp corners for geometric shapes
                'length_threshold': 3.0,       # Keep small geometric details
                'max_iterations': 10,          # Standard iterations
                'splice_threshold': 45         # Standard splicing
            },
            LogoType.TEXT.value: {
                'colormode': 'color',
                'color_precision': 2,          # Minimal colors for text clarity
                'layer_difference': 24,        # Good separation for legibility
                'path_precision': 8,           # Maximum precision for text quality
                'corner_threshold': 20,        # Very sharp corners for letter forms
                'length_threshold': 2.0,       # Preserve fine text details
                'max_iterations': 12,          # Extra iterations for quality
                'splice_threshold': 40         # Conservative splicing for text
            },
            LogoType.GRADIENT.value: {
                'colormode': 'color',
                'color_precision': 8,          # High precision for color transitions
                'layer_difference': 8,         # Fine layers for smooth gradients
                'path_precision': 6,           # Good precision for smooth curves
                'corner_threshold': 60,        # Higher threshold for smooth curves
                'length_threshold': 4.0,       # Balance detail vs smoothness
                'max_iterations': 15,          # More iterations for smoothness
                'splice_threshold': 60         # Aggressive splicing for smooth paths
            },
            LogoType.COMPLEX.value: {
                'colormode': 'color',
                'color_precision': 6,          # Balanced color handling
                'layer_difference': 16,        # Medium separation for detail
                'path_precision': 5,           # Standard precision
                'corner_threshold': 45,        # Balanced corner detection
                'length_threshold': 5.0,       # Standard detail preservation
                'max_iterations': 20,          # More iterations for complexity
                'splice_threshold': 50         # Balanced splicing
            }
        }

        return parameter_sets.get(logo_type, parameter_sets[LogoType.COMPLEX.value])

    def _apply_confidence_adjustments(self, params: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Apply confidence-based parameter adjustments.

        Lower confidence indicates uncertain classification, so we use more conservative
        parameters that work well across multiple logo types.

        Args:
            params (Dict[str, Any]): Base parameters to adjust.
            confidence (float): Classification confidence (0.0 to 1.0).

        Returns:
            Dict[str, Any]: Confidence-adjusted parameters.
        """
        adjusted_params = params.copy()
        confidence_level = self._get_confidence_level(confidence)

        if confidence_level == 'low':
            # Low confidence (<0.6) - use conservative parameters
            logger.info(f"Low confidence ({confidence:.2%}), applying conservative adjustments")

            # More conservative color precision (middle range)
            adjusted_params['color_precision'] = 5

            # Standard layer difference that works for most types
            adjusted_params['layer_difference'] = 16

            # Moderate corner threshold
            adjusted_params['corner_threshold'] = 50

            # Standard iterations
            adjusted_params['max_iterations'] = 12

        elif confidence_level == 'medium':
            # Medium confidence (0.6-0.8) - moderate adjustments toward middle ground
            logger.info(f"Medium confidence ({confidence:.2%}), applying moderate adjustments")

            # Moderate adjustments toward conservative values
            if params['color_precision'] <= 2:
                adjusted_params['color_precision'] = 3
            elif params['color_precision'] >= 8:
                adjusted_params['color_precision'] = 6

            # Adjust extreme layer differences
            if params['layer_difference'] <= 8:
                adjusted_params['layer_difference'] = 12
            elif params['layer_difference'] >= 30:
                adjusted_params['layer_difference'] = 24

        # High confidence (>=0.8) - use parameters as-is
        # No adjustments needed for high confidence

        return adjusted_params

    def _get_confidence_level(self, confidence: float) -> str:
        """Categorize confidence level for parameter adjustments.

        Args:
            confidence (float): Confidence value (0.0 to 1.0).

        Returns:
            str: Confidence level ('low', 'medium', 'high').
        """
        if confidence < 0.6:
            return 'low'
        elif confidence < 0.8:
            return 'medium'
        else:
            return 'high'

    def _apply_feature_adjustments(self, params: Dict[str, Any], features: Dict[str, float]) -> Dict[str, Any]:
        """Apply feature-based fine-tuning to parameters.

        Uses individual feature values to fine-tune parameters beyond logo type classification:
        - Edge density → corner threshold adjustments
        - Color complexity → color precision adjustments
        - Entropy/complexity → iteration and layer adjustments
        - Gradient strength → smoothness optimizations

        Args:
            params (Dict[str, Any]): Base parameters to fine-tune.
            features (Dict[str, float]): Extracted features (all in [0,1] range).

        Returns:
            Dict[str, Any]: Feature-tuned parameters.
        """
        tuned_params = params.copy()

        # Edge density adjustments for corner detection
        edge_density = features.get('edge_density', 0.5)
        if 'corner_threshold' in tuned_params:
            if edge_density > 0.3:  # High edge density - many edges
                # Lower corner threshold for better edge detection
                current_corner = tuned_params['corner_threshold']
                tuned_params['corner_threshold'] = max(20, current_corner - 10)
                logger.debug(f"High edge density ({edge_density:.3f}), lowered corner threshold")
            elif edge_density < 0.1:  # Very low edge density - smooth shapes
                # Higher corner threshold for smoother curves
                current_corner = tuned_params['corner_threshold']
                tuned_params['corner_threshold'] = min(80, current_corner + 15)
                logger.debug(f"Low edge density ({edge_density:.3f}), raised corner threshold")

        # Color complexity adjustments
        unique_colors = features.get('unique_colors', 0.5)
        if 'color_precision' in tuned_params:
            if unique_colors > 0.7:  # Many colors - preserve color detail
                # Increase color precision to preserve detail
                current_precision = tuned_params['color_precision']
                tuned_params['color_precision'] = min(8, current_precision + 1)
                logger.debug(f"High color complexity ({unique_colors:.3f}), increased color precision")
            elif unique_colors < 0.2:  # Few colors - simplify
                # Decrease color precision for cleaner output
                current_precision = tuned_params['color_precision']
                tuned_params['color_precision'] = max(2, current_precision - 1)
                logger.debug(f"Low color complexity ({unique_colors:.3f}), decreased color precision")

        # Overall complexity adjustments
        complexity_score = features.get('complexity_score', 0.5)
        if complexity_score > 0.7:  # High complexity - need more processing
            # More iterations and finer layer separation
            if 'max_iterations' in tuned_params:
                current_iterations = tuned_params['max_iterations']
                tuned_params['max_iterations'] = min(25, current_iterations + 5)

            if 'layer_difference' in tuned_params:
                current_layer_diff = tuned_params['layer_difference']
                tuned_params['layer_difference'] = max(8, current_layer_diff - 4)

            logger.debug(f"High complexity ({complexity_score:.3f}), increased iterations and layer fineness")

        # Gradient strength adjustments for smoothness
        gradient_strength = features.get('gradient_strength', 0.5)
        if gradient_strength > 0.6:  # Strong gradients - optimize for smoothness
            # Optimize parameters for smooth gradients
            if 'color_precision' in tuned_params:
                tuned_params['color_precision'] = min(8, max(6, tuned_params['color_precision']))
            if 'layer_difference' in tuned_params:
                tuned_params['layer_difference'] = min(12, tuned_params['layer_difference'])
            if 'corner_threshold' in tuned_params:
                tuned_params['corner_threshold'] = max(50, tuned_params['corner_threshold'])
            logger.debug(f"Strong gradients ({gradient_strength:.3f}), optimized for smoothness")

        # Entropy adjustments for detail preservation
        entropy = features.get('entropy', 0.5)
        if entropy > 0.8:  # Very high entropy - lots of detail/noise
            # Conservative settings to handle detail without over-processing
            if 'length_threshold' in tuned_params:
                tuned_params['length_threshold'] = max(3.0, tuned_params['length_threshold'])
            logger.debug(f"High entropy ({entropy:.3f}), conservative length threshold")

        return tuned_params

    def _validate_and_correct_parameters(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Validate parameters against bounds and correct invalid values.

        Args:
            params (Dict[str, Any]): Parameters to validate.

        Returns:
            Tuple[Dict[str, Any], bool]: (corrected_parameters, validation_passed)
        """
        corrected_params = params.copy()
        validation_passed = True

        # Validate colormode
        if 'colormode' in corrected_params:
            if corrected_params['colormode'] not in self.bounds.colormode_options:
                logger.warning(f"Invalid colormode '{corrected_params['colormode']}', "
                             f"correcting to 'color'")
                corrected_params['colormode'] = 'color'
                validation_passed = False

        # Validate numeric parameters with bounds checking
        numeric_validations = [
            ('color_precision', self.bounds.color_precision_range, int),
            ('layer_difference', self.bounds.layer_difference_range, int),
            ('path_precision', self.bounds.path_precision_range, int),
            ('corner_threshold', self.bounds.corner_threshold_range, int),
            ('length_threshold', self.bounds.length_threshold_range, float),
            ('max_iterations', self.bounds.max_iterations_range, int),
            ('splice_threshold', self.bounds.splice_threshold_range, int),
        ]

        for param_name, (min_val, max_val), param_type in numeric_validations:
            if param_name in corrected_params:
                value = corrected_params[param_name]

                # Type correction
                try:
                    value = param_type(value)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid type for {param_name}: {value}, using default")
                    # Use middle of range as default
                    value = param_type((min_val + max_val) / 2)
                    validation_passed = False

                # Bounds checking
                if value < min_val:
                    logger.warning(f"{param_name} value {value} below minimum {min_val}, correcting")
                    corrected_params[param_name] = min_val
                    validation_passed = False
                elif value > max_val:
                    logger.warning(f"{param_name} value {value} above maximum {max_val}, correcting")
                    corrected_params[param_name] = max_val
                    validation_passed = False
                else:
                    corrected_params[param_name] = value

        return corrected_params, validation_passed

    def _update_optimization_stats(self, result: OptimizationResult) -> None:
        """Update optimization statistics with new result.

        Args:
            result (OptimizationResult): Optimization result to record.
        """
        self.stats['total_optimizations'] += 1

        # Track by logo type
        logo_type = result.logo_type
        if logo_type not in self.stats['by_logo_type']:
            self.stats['by_logo_type'][logo_type] = 0
        self.stats['by_logo_type'][logo_type] += 1

        # Track by confidence level
        confidence_level = self._get_confidence_level(result.confidence)
        self.stats['by_confidence_level'][confidence_level] += 1

        # Update average optimization time
        old_avg = self.stats['average_optimization_time']
        n = self.stats['total_optimizations']
        self.stats['average_optimization_time'] = (old_avg * (n-1) + result.optimization_time) / n

        # Store in history (keep last 100)
        self.optimization_history.append(result)
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]

    def get_parameter_recommendations(self, logo_type: str) -> Dict[str, Any]:
        """Get recommended parameters for a specific logo type without features.

        Args:
            logo_type (str): Logo type to get recommendations for.

        Returns:
            Dict[str, Any]: Recommended parameters for the logo type.

        Raises:
            ValueError: If logo type is not supported.
        """
        if logo_type not in [lt.value for lt in LogoType]:
            raise ValueError(f"Unsupported logo type: {logo_type}")

        return self._get_base_parameters_for_type(logo_type)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics.

        Returns:
            Dict[str, Any]: Optimization statistics and analytics.
        """
        return self.stats.copy()

    def get_parameter_bounds(self) -> ParameterBounds:
        """Get parameter bounds specification.

        Returns:
            ParameterBounds: Complete parameter bounds.
        """
        return self.bounds


def test_parameter_optimizer():
    """Test the parameter optimizer with various scenarios."""
    print("\n" + "="*70)
    print("Testing VTracer Parameter Optimizer")
    print("="*70)

    optimizer = VTracerParameterOptimizer()

    # Test cases for different logo types and confidence levels
    test_cases = [
        # Simple logo, high confidence
        {
            'name': 'Simple Logo (High Confidence)',
            'classification': {'logo_type': 'simple', 'confidence': 0.9},
            'features': {'edge_density': 0.1, 'unique_colors': 0.2, 'entropy': 0.3,
                        'corner_density': 0.2, 'gradient_strength': 0.1, 'complexity_score': 0.2}
        },
        # Text logo, medium confidence
        {
            'name': 'Text Logo (Medium Confidence)',
            'classification': {'logo_type': 'text', 'confidence': 0.7},
            'features': {'edge_density': 0.3, 'unique_colors': 0.15, 'entropy': 0.4,
                        'corner_density': 0.4, 'gradient_strength': 0.05, 'complexity_score': 0.3}
        },
        # Gradient logo, low confidence
        {
            'name': 'Gradient Logo (Low Confidence)',
            'classification': {'logo_type': 'gradient', 'confidence': 0.5},
            'features': {'edge_density': 0.05, 'unique_colors': 0.8, 'entropy': 0.6,
                        'corner_density': 0.1, 'gradient_strength': 0.9, 'complexity_score': 0.7}
        },
        # Complex logo with extreme features
        {
            'name': 'Complex Logo (Extreme Features)',
            'classification': {'logo_type': 'complex', 'confidence': 0.85},
            'features': {'edge_density': 0.9, 'unique_colors': 0.95, 'entropy': 0.9,
                        'corner_density': 0.8, 'gradient_strength': 0.3, 'complexity_score': 0.95}
        }
    ]

    for test_case in test_cases:
        print(f"\n[{test_case['name']}]")
        print("-" * 50)

        try:
            result = optimizer.optimize_parameters(
                test_case['classification'],
                test_case['features']
            )

            print(f"Logo Type: {result.logo_type}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"Optimization Method: {result.optimization_method}")
            print(f"Validation Passed: {result.validation_passed}")
            print(f"Adjustments Applied: {', '.join(result.adjustments_applied) if result.adjustments_applied else 'None'}")
            print(f"Optimization Time: {result.optimization_time*1000:.2f}ms")

            print(f"Optimized Parameters:")
            for param, value in result.parameters.items():
                print(f"  {param}: {value}")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Print optimization statistics
    print("\n" + "="*70)
    print("Optimization Statistics")
    print("="*70)

    stats = optimizer.get_optimization_stats()
    print(f"Total Optimizations: {stats['total_optimizations']}")
    print(f"Average Time: {stats['average_optimization_time']*1000:.2f}ms")
    print(f"Validation Failures: {stats['validation_failures']}")

    print("\nBy Logo Type:")
    for logo_type, count in stats['by_logo_type'].items():
        print(f"  {logo_type}: {count}")

    print("\nBy Confidence Level:")
    for level, count in stats['by_confidence_level'].items():
        print(f"  {level}: {count}")


if __name__ == "__main__":
    test_parameter_optimizer()