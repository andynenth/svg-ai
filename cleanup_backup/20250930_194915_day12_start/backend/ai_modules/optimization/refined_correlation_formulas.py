"""
Refined Correlation Formulas for Method 1 Parameter Optimization
Enhanced formulas based on Day 3 validation results and analysis
"""
import numpy as np
import math
from typing import Dict, Any, Tuple, Optional
from scipy import stats
import logging

from .correlation_formulas import CorrelationFormulas

logger = logging.getLogger(__name__)


class RefinedCorrelationFormulas:
    """Enhanced correlation formulas with improved accuracy"""

    def __init__(self):
        """Initialize refined correlation formulas"""
        self.original_formulas = CorrelationFormulas()

        # Confidence interval parameters (derived from analysis)
        self.confidence_intervals = {
            'edge_to_corner_threshold': {'std': 5.2, 'confidence': 0.95},
            'colors_to_precision': {'std': 0.8, 'confidence': 0.95},
            'entropy_to_path_precision': {'std': 1.5, 'confidence': 0.95},
            'corners_to_length_threshold': {'std': 2.1, 'confidence': 0.95},
            'gradient_to_splice_threshold': {'std': 8.3, 'confidence': 0.95},
            'complexity_to_iterations': {'std': 1.4, 'confidence': 0.95}
        }

        # Logo type specific scaling factors (from optimal ranges analysis)
        self.logo_type_adjustments = {
            'simple': {
                'corner_threshold_factor': 1.0,
                'color_precision_bonus': 0,
                'path_precision_factor': 1.1,
                'length_threshold_factor': 0.8,
                'splice_threshold_factor': 0.9,
                'iterations_factor': 0.8
            },
            'text': {
                'corner_threshold_factor': 0.9,
                'color_precision_bonus': 0,
                'path_precision_factor': 1.2,
                'length_threshold_factor': 1.2,
                'splice_threshold_factor': 1.1,
                'iterations_factor': 1.0
            },
            'gradient': {
                'corner_threshold_factor': 0.8,
                'color_precision_bonus': 2,
                'path_precision_factor': 0.8,
                'length_threshold_factor': 1.0,
                'splice_threshold_factor': 1.4,
                'iterations_factor': 1.2
            },
            'complex': {
                'corner_threshold_factor': 0.7,
                'color_precision_bonus': 1,
                'path_precision_factor': 0.9,
                'length_threshold_factor': 1.1,
                'splice_threshold_factor': 1.3,
                'iterations_factor': 1.4
            }
        }

    def edge_to_corner_threshold_refined(self, edge_density: float, logo_type: str = 'simple') -> int:
        """
        Refined edge density to corner threshold mapping

        Original: 110 - (edge_density * 800)
        Refined: Improved with logo type adjustments and better boundary handling
        """
        try:
            # Enhanced formula with logo type specific adjustments
            type_factor = self.logo_type_adjustments.get(logo_type, {}).get('corner_threshold_factor', 1.0)

            # Alternative formula as specified in document
            base_value = max(10, min(110, 80 - (edge_density * 600)))

            # Apply logo type adjustment
            refined_value = base_value * type_factor

            # Ensure within bounds
            result = max(10, min(110, int(refined_value)))

            return result

        except Exception as e:
            logger.warning(f"Error in refined edge_to_corner_threshold: {e}")
            return self.original_formulas.edge_to_corner_threshold(edge_density)

    def colors_to_precision_refined(self, unique_colors: float, logo_type: str = 'simple') -> int:
        """
        Improved colors to precision mapping with gradient-specific scaling

        Enhanced for gradient logos with logarithmic vs linear mapping
        """
        try:
            # Get logo type adjustments
            color_bonus = self.logo_type_adjustments.get(logo_type, {}).get('color_precision_bonus', 0)

            if logo_type == 'gradient':
                # Enhanced logarithmic scaling for gradients
                if unique_colors >= 50:
                    # Use enhanced logarithmic formula for high color counts
                    base_precision = max(6, min(10, int(6 + np.log10(unique_colors / 10))))
                else:
                    # Standard formula for lower color counts
                    base_precision = max(2, min(10, int(2 + np.log2(max(1, unique_colors)))))
            else:
                # Standard formula for other logo types
                base_precision = max(2, min(10, int(2 + np.log2(max(1, unique_colors)))))

            # Apply bonus
            result = max(2, min(10, base_precision + color_bonus))

            return result

        except Exception as e:
            logger.warning(f"Error in refined colors_to_precision: {e}")
            return self.original_formulas.colors_to_precision(unique_colors)

    def entropy_to_path_precision_refined(self, entropy: float, logo_type: str = 'simple') -> int:
        """
        Optimized entropy to path precision for text logos

        Enhanced with text detection bonus factor
        """
        try:
            # Get logo type factor
            type_factor = self.logo_type_adjustments.get(logo_type, {}).get('path_precision_factor', 1.0)

            # Base formula with enhancement
            base_precision = max(1, min(20, int(20 * (1 - entropy))))

            if logo_type == 'text':
                # Text detection bonus - higher precision for text elements
                text_bonus = max(0, min(5, int(entropy * 10)))  # Higher entropy gives more precision for text
                base_precision += text_bonus

            # Apply type factor
            refined_precision = base_precision * type_factor

            # Ensure within bounds
            result = max(1, min(20, int(refined_precision)))

            return result

        except Exception as e:
            logger.warning(f"Error in refined entropy_to_path_precision: {e}")
            return self.original_formulas.entropy_to_path_precision(entropy)

    def corners_to_length_threshold_refined(self, corner_density: float, logo_type: str = 'simple') -> float:
        """
        Enhanced corners to length threshold with logo type adjustments
        """
        try:
            # Get logo type factor
            type_factor = self.logo_type_adjustments.get(logo_type, {}).get('length_threshold_factor', 1.0)

            # Enhanced base formula
            base_threshold = max(1.0, min(20.0, 1.0 + (corner_density * 100)))

            # Apply type factor
            refined_threshold = base_threshold * type_factor

            # Ensure within bounds
            result = max(1.0, min(20.0, refined_threshold))

            return result

        except Exception as e:
            logger.warning(f"Error in refined corners_to_length_threshold: {e}")
            return self.original_formulas.corners_to_length_threshold(corner_density)

    def gradient_to_splice_threshold_refined(self, gradient_strength: float, logo_type: str = 'simple') -> int:
        """
        Enhanced gradient to splice threshold mapping
        """
        try:
            # Get logo type factor
            type_factor = self.logo_type_adjustments.get(logo_type, {}).get('splice_threshold_factor', 1.0)

            # Enhanced base formula
            base_threshold = max(10, min(100, int(10 + (gradient_strength * 90))))

            # Apply type factor
            refined_threshold = base_threshold * type_factor

            # Ensure within bounds
            result = max(10, min(100, int(refined_threshold)))

            return result

        except Exception as e:
            logger.warning(f"Error in refined gradient_to_splice_threshold: {e}")
            return self.original_formulas.gradient_to_splice_threshold(gradient_strength)

    def complexity_to_iterations_refined(self, complexity_score: float, logo_type: str = 'simple') -> int:
        """
        Enhanced complexity to iterations with tiered complexity scoring

        Uses tiered complexity scoring and diminishing returns scaling
        """
        try:
            # Get logo type factor
            type_factor = self.logo_type_adjustments.get(logo_type, {}).get('iterations_factor', 1.0)

            # Tiered complexity scoring
            if complexity_score >= 0.8:
                # Very complex - use enhanced scaling
                base_iterations = max(15, min(20, int(15 + (complexity_score - 0.8) * 25)))
            elif complexity_score >= 0.6:
                # Moderately complex
                base_iterations = max(10, min(15, int(10 + (complexity_score - 0.6) * 25)))
            elif complexity_score >= 0.3:
                # Somewhat complex
                base_iterations = max(7, min(10, int(7 + (complexity_score - 0.3) * 10)))
            else:
                # Simple
                base_iterations = max(5, min(7, int(5 + complexity_score * 6)))

            # Apply diminishing returns scaling
            if base_iterations > 15:
                # Apply diminishing returns for very high iteration counts
                excess = base_iterations - 15
                base_iterations = 15 + int(excess * 0.7)

            # Apply type factor
            refined_iterations = base_iterations * type_factor

            # Ensure within bounds
            result = max(5, min(20, int(refined_iterations)))

            return result

        except Exception as e:
            logger.warning(f"Error in refined complexity_to_iterations: {e}")
            return self.original_formulas.complexity_to_iterations(complexity_score)

    def get_confidence_interval(self, formula_name: str, predicted_value: float) -> Tuple[float, float]:
        """Calculate confidence interval for formula prediction"""
        try:
            if formula_name not in self.confidence_intervals:
                return predicted_value, predicted_value

            ci_params = self.confidence_intervals[formula_name]
            std = ci_params['std']
            confidence = ci_params['confidence']

            # Calculate confidence interval
            z_score = stats.norm.ppf((1 + confidence) / 2)
            margin = z_score * std

            lower_bound = max(0, predicted_value - margin)
            upper_bound = predicted_value + margin

            return lower_bound, upper_bound

        except Exception as e:
            logger.warning(f"Error calculating confidence interval for {formula_name}: {e}")
            return predicted_value, predicted_value

    def optimize_parameters_with_refinements(self, features: Dict[str, float], logo_type: str = 'simple') -> Dict[str, Any]:
        """
        Optimize parameters using refined correlation formulas

        Args:
            features: Image feature dictionary
            logo_type: Type of logo for specific adjustments

        Returns:
            Optimized parameters with confidence intervals
        """
        try:
            # Apply refined correlation formulas
            corner_threshold = self.edge_to_corner_threshold_refined(
                features.get('edge_density', 0.1), logo_type
            )

            color_precision = self.colors_to_precision_refined(
                features.get('unique_colors', 10), logo_type
            )

            path_precision = self.entropy_to_path_precision_refined(
                features.get('entropy', 0.5), logo_type
            )

            length_threshold = self.corners_to_length_threshold_refined(
                features.get('corner_density', 0.1), logo_type
            )

            splice_threshold = self.gradient_to_splice_threshold_refined(
                features.get('gradient_strength', 0.3), logo_type
            )

            max_iterations = self.complexity_to_iterations_refined(
                features.get('complexity_score', 0.3), logo_type
            )

            # Set other parameters based on logo type
            layer_difference = 10  # Default

            # Mode selection based on complexity and logo type
            if logo_type in ['gradient', 'complex'] or features.get('complexity_score', 0) > 0.6:
                mode = 'spline'
            else:
                mode = 'polygon'

            # Build parameter set
            parameters = {
                'color_precision': color_precision,
                'layer_difference': layer_difference,
                'corner_threshold': corner_threshold,
                'length_threshold': length_threshold,
                'max_iterations': max_iterations,
                'splice_threshold': splice_threshold,
                'path_precision': path_precision,
                'mode': mode
            }

            # Calculate confidence intervals
            confidence_intervals = {}
            for param, value in parameters.items():
                if param != 'mode':
                    formula_name = None
                    if param == 'corner_threshold':
                        formula_name = 'edge_to_corner_threshold'
                    elif param == 'color_precision':
                        formula_name = 'colors_to_precision'
                    elif param == 'path_precision':
                        formula_name = 'entropy_to_path_precision'
                    elif param == 'length_threshold':
                        formula_name = 'corners_to_length_threshold'
                    elif param == 'splice_threshold':
                        formula_name = 'gradient_to_splice_threshold'
                    elif param == 'max_iterations':
                        formula_name = 'complexity_to_iterations'

                    if formula_name:
                        lower, upper = self.get_confidence_interval(formula_name, value)
                        confidence_intervals[param] = {'lower': lower, 'upper': upper}

            # Calculate overall confidence based on feature quality
            feature_completeness = len([v for v in features.values() if v is not None and not np.isnan(v)]) / 6
            logo_type_confidence = {
                'simple': 0.95,
                'text': 0.90,
                'gradient': 0.85,
                'complex': 0.80
            }.get(logo_type, 0.85)

            overall_confidence = feature_completeness * logo_type_confidence

            return {
                'parameters': parameters,
                'confidence_intervals': confidence_intervals,
                'overall_confidence': overall_confidence,
                'logo_type': logo_type,
                'refinement_method': 'enhanced_correlations',
                'formula_version': '2.0'
            }

        except Exception as e:
            logger.error(f"Error in refined parameter optimization: {e}")
            # Fallback to original formulas
            from .feature_mapping_optimizer import FeatureMappingOptimizer
            fallback_optimizer = FeatureMappingOptimizer()
            return fallback_optimizer.optimize(features)


class FormulaABTester:
    """A/B testing framework for formula comparison"""

    def __init__(self):
        """Initialize A/B testing framework"""
        self.original_formulas = CorrelationFormulas()
        self.refined_formulas = RefinedCorrelationFormulas()
        self.test_results = []

    def compare_formulas(self, features_list: list, logo_types: list = None) -> Dict[str, Any]:
        """
        Compare original vs refined formulas on test data

        Args:
            features_list: List of feature dictionaries
            logo_types: List of corresponding logo types

        Returns:
            Comparison results
        """
        if logo_types is None:
            logo_types = ['simple'] * len(features_list)

        original_results = []
        refined_results = []

        for i, features in enumerate(features_list):
            logo_type = logo_types[i] if i < len(logo_types) else 'simple'

            # Test original formulas
            orig_corner = self.original_formulas.edge_to_corner_threshold(features.get('edge_density', 0.1))
            orig_color = self.original_formulas.colors_to_precision(features.get('unique_colors', 10))
            orig_path = self.original_formulas.entropy_to_path_precision(features.get('entropy', 0.5))
            orig_length = self.original_formulas.corners_to_length_threshold(features.get('corner_density', 0.1))
            orig_splice = self.original_formulas.gradient_to_splice_threshold(features.get('gradient_strength', 0.3))
            orig_iterations = self.original_formulas.complexity_to_iterations(features.get('complexity_score', 0.3))

            # Test refined formulas
            ref_corner = self.refined_formulas.edge_to_corner_threshold_refined(features.get('edge_density', 0.1), logo_type)
            ref_color = self.refined_formulas.colors_to_precision_refined(features.get('unique_colors', 10), logo_type)
            ref_path = self.refined_formulas.entropy_to_path_precision_refined(features.get('entropy', 0.5), logo_type)
            ref_length = self.refined_formulas.corners_to_length_threshold_refined(features.get('corner_density', 0.1), logo_type)
            ref_splice = self.refined_formulas.gradient_to_splice_threshold_refined(features.get('gradient_strength', 0.3), logo_type)
            ref_iterations = self.refined_formulas.complexity_to_iterations_refined(features.get('complexity_score', 0.3), logo_type)

            original_results.append({
                'corner_threshold': orig_corner,
                'color_precision': orig_color,
                'path_precision': orig_path,
                'length_threshold': orig_length,
                'splice_threshold': orig_splice,
                'max_iterations': orig_iterations
            })

            refined_results.append({
                'corner_threshold': ref_corner,
                'color_precision': ref_color,
                'path_precision': ref_path,
                'length_threshold': ref_length,
                'splice_threshold': ref_splice,
                'max_iterations': ref_iterations
            })

        # Calculate differences
        differences = []
        for orig, refined in zip(original_results, refined_results):
            diff = {}
            for param in orig.keys():
                diff[param] = refined[param] - orig[param]
            differences.append(diff)

        # Calculate statistics
        avg_differences = {}
        for param in original_results[0].keys():
            param_diffs = [d[param] for d in differences]
            avg_differences[param] = {
                'mean_difference': np.mean(param_diffs),
                'std_difference': np.std(param_diffs),
                'improvement_rate': len([d for d in param_diffs if abs(d) > 0]) / len(param_diffs)
            }

        return {
            'original_results': original_results,
            'refined_results': refined_results,
            'differences': differences,
            'average_differences': avg_differences,
            'test_count': len(features_list)
        }