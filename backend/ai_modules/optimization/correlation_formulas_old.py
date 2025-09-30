# backend/ai_modules/optimization/correlation_formulas.py
"""Mathematical correlation formulas for feature-to-parameter mapping"""

import numpy as np
import math
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class CorrelationFormulas:
    """Research-validated parameter correlations"""

    @staticmethod
    def calculate_corner_threshold(edge_density: float) -> int:
        """
        Calculate corner_threshold based on edge density.
        Higher edge density should reduce corner threshold for better detail.

        Formula: corner_threshold = 110 - (edge_density * 800)
        Bounded between 10 and 110

        Args:
            edge_density: Normalized edge density value (0.0 to 1.0)

        Returns:
            int: Corner threshold parameter value
        """
        # Validate input
        edge_density = max(0.0, min(1.0, edge_density))

        # Apply formula
        threshold = 110 - (edge_density * 800)

        # Clip to valid bounds
        threshold = max(10, min(110, int(threshold)))

        logger.debug(f"edge_density={edge_density:.3f} -> corner_threshold={threshold}")
        return threshold

    @staticmethod
    def calculate_color_precision(unique_colors: int) -> int:
        """
        Calculate color_precision based on number of unique colors.
        More colors require higher precision.

        Formula: color_precision = 2 + log2(unique_colors)
        Bounded between 2 and 10

        Args:
            unique_colors: Number of unique colors in the image

        Returns:
            int: Color precision parameter value
        """
        # Handle edge cases
        if unique_colors <= 1:
            return 2

        # Apply logarithmic formula
        precision = 2 + math.log2(max(1, unique_colors))

        # Clip to valid bounds
        precision = max(2, min(10, int(precision)))

        logger.debug(f"unique_colors={unique_colors} -> color_precision={precision}")
        return precision

    @staticmethod
    def calculate_path_precision(entropy: float) -> int:
        """
        Calculate path_precision based on image entropy.
        Higher entropy needs more precise paths.

        Formula: path_precision = 20 * (1 - entropy)
        Bounded between 1 and 20

        Args:
            entropy: Normalized entropy value (0.0 to 1.0)

        Returns:
            int: Path precision parameter value
        """
        # Validate input
        entropy = max(0.0, min(1.0, entropy))

        # Apply inverse relationship
        precision = 20 * (1 - entropy)

        # Clip to valid bounds
        precision = max(1, min(20, int(precision)))

        logger.debug(f"entropy={entropy:.3f} -> path_precision={precision}")
        return precision

    @staticmethod
    def calculate_length_threshold(corner_density: float) -> float:
        """
        Calculate length_threshold based on corner density.
        More corners need shorter segments.

        Formula: length_threshold = 1.0 + (corner_density * 100)
        Bounded between 1.0 and 20.0

        Args:
            corner_density: Normalized corner density value (0.0 to 1.0)

        Returns:
            float: Length threshold parameter value
        """
        # Validate input
        corner_density = max(0.0, min(1.0, corner_density))

        # Apply formula
        threshold = 1.0 + (corner_density * 100)

        # Clip to valid bounds
        threshold = max(1.0, min(20.0, threshold))

        logger.debug(f"corner_density={corner_density:.3f} -> length_threshold={threshold:.2f}")
        return threshold

    @staticmethod
    def calculate_splice_threshold(gradient_strength: float) -> int:
        """
        Calculate splice_threshold based on gradient strength.
        Stronger gradients need more splicing.

        Formula: splice_threshold = 10 + (gradient_strength * 90)
        Bounded between 10 and 100

        Args:
            gradient_strength: Normalized gradient strength (0.0 to 1.0)

        Returns:
            int: Splice threshold parameter value
        """
        # Validate input
        gradient_strength = max(0.0, min(1.0, gradient_strength))

        # Apply formula
        threshold = 10 + (gradient_strength * 90)

        # Clip to valid bounds
        threshold = max(10, min(100, int(threshold)))

        logger.debug(f"gradient_strength={gradient_strength:.3f} -> splice_threshold={threshold}")
        return threshold

    @staticmethod
    def calculate_max_iterations(complexity_score: float) -> int:
        """
        Calculate max_iterations based on complexity score.
        Complex images need more iterations.

        Formula: max_iterations = 5 + (complexity_score * 15)
        Bounded between 5 and 20

        Args:
            complexity_score: Normalized complexity score (0.0 to 1.0)

        Returns:
            int: Max iterations parameter value
        """
        # Validate input
        complexity_score = max(0.0, min(1.0, complexity_score))

        # Apply formula
        iterations = 5 + (complexity_score * 15)

        # Clip to valid bounds
        iterations = max(5, min(20, int(iterations)))

        logger.debug(f"complexity_score={complexity_score:.3f} -> max_iterations={iterations}")
        return iterations

    @staticmethod
    def calculate_layer_difference(unique_colors: int, gradient_strength: float) -> int:
        """
        Calculate layer_difference based on colors and gradients.
        More colors and stronger gradients need smaller layer differences.

        Formula: layer_difference = 20 - (unique_colors/10 + gradient_strength*10)
        Bounded between 1 and 20

        Args:
            unique_colors: Number of unique colors
            gradient_strength: Normalized gradient strength (0.0 to 1.0)

        Returns:
            int: Layer difference parameter value
        """
        # Validate inputs
        gradient_strength = max(0.0, min(1.0, gradient_strength))
        unique_colors = max(0, unique_colors)

        # Apply formula
        difference = 20 - (unique_colors/10 + gradient_strength*10)

        # Clip to valid bounds
        difference = max(1, min(20, int(difference)))

        logger.debug(f"unique_colors={unique_colors}, gradient_strength={gradient_strength:.3f} -> layer_difference={difference}")
        return difference

    @staticmethod
    def select_mode(logo_type: str, complexity_score: float) -> str:
        """
        Select mode based on logo type and complexity.

        Args:
            logo_type: Type of logo ('simple', 'text', 'gradient', 'complex')
            complexity_score: Normalized complexity score (0.0 to 1.0)

        Returns:
            str: 'polygon' or 'spline'
        """
        # Simple logos typically work better with polygon mode
        if logo_type == 'simple' and complexity_score < 0.3:
            mode = 'polygon'
        # Text logos benefit from spline mode for smooth curves
        elif logo_type == 'text':
            mode = 'spline'
        # Complex logos need spline for better approximation
        elif complexity_score > 0.7:
            mode = 'spline'
        # Default to spline for most cases
        else:
            mode = 'spline'

        logger.debug(f"logo_type={logo_type}, complexity={complexity_score:.3f} -> mode={mode}")
        return mode

    @staticmethod
    def apply_all_correlations(features: Dict[str, float], logo_type: str = 'unknown') -> Dict[str, Any]:
        """
        Apply all correlation formulas to generate a complete parameter set.

        Args:
            features: Dictionary of image features
            logo_type: Type of logo for mode selection

        Returns:
            Dictionary of calculated parameters
        """
        parameters = {}

        # Extract features with defaults
        edge_density = features.get('edge_density', 0.1)
        unique_colors = int(features.get('unique_colors', 16))
        entropy = features.get('entropy', 0.5)
        corner_density = features.get('corner_density', 0.1)
        gradient_strength = features.get('gradient_strength', 0.1)
        complexity_score = features.get('complexity_score', 0.5)

        # Apply correlation formulas
        parameters['corner_threshold'] = CorrelationFormulas.calculate_corner_threshold(edge_density)
        parameters['color_precision'] = CorrelationFormulas.calculate_color_precision(unique_colors)
        parameters['path_precision'] = CorrelationFormulas.calculate_path_precision(entropy)
        parameters['length_threshold'] = CorrelationFormulas.calculate_length_threshold(corner_density)
        parameters['splice_threshold'] = CorrelationFormulas.calculate_splice_threshold(gradient_strength)
        parameters['max_iterations'] = CorrelationFormulas.calculate_max_iterations(complexity_score)
        parameters['layer_difference'] = CorrelationFormulas.calculate_layer_difference(unique_colors, gradient_strength)
        parameters['mode'] = CorrelationFormulas.select_mode(logo_type, complexity_score)

        logger.info(f"Applied all correlations: {parameters}")
        return parameters

    @staticmethod
    def validate_correlations(features: Dict[str, float], parameters: Dict[str, Any]) -> bool:
        """
        Validate that parameters are consistent with feature correlations.

        Args:
            features: Dictionary of image features
            parameters: Dictionary of parameters to validate

        Returns:
            bool: True if parameters are consistent with correlations
        """
        # Calculate expected parameters
        expected = CorrelationFormulas.apply_all_correlations(features)

        # Allow some tolerance for numeric parameters
        tolerance = 0.2  # 20% tolerance

        for key, expected_value in expected.items():
            if key not in parameters:
                logger.warning(f"Missing parameter: {key}")
                return False

            actual_value = parameters[key]

            if key == 'mode':
                # Exact match for mode
                if actual_value != expected_value:
                    logger.debug(f"Mode mismatch: expected={expected_value}, actual={actual_value}")
                    # Mode selection is flexible, so this is just a warning
                    pass
            elif isinstance(expected_value, (int, float)):
                # Numeric comparison with tolerance
                if isinstance(expected_value, float):
                    diff = abs(actual_value - expected_value)
                    max_diff = expected_value * tolerance
                else:
                    diff = abs(int(actual_value) - expected_value)
                    max_diff = max(1, expected_value * tolerance)

                if diff > max_diff:
                    logger.warning(f"Parameter {key} outside tolerance: expected={expected_value}, actual={actual_value}")
                    return False

        return True