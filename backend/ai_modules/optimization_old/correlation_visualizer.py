# backend/ai_modules/optimization/correlation_visualizer.py
"""Generate correlation matrix visualizations for feature-parameter relationships"""

import numpy as np
import logging
from typing import Dict, List, Tuple
import json

logger = logging.getLogger(__name__)


class CorrelationVisualizer:
    """Visualize correlations between features and parameters"""

    # Correlation data based on research
    CORRELATION_MATRIX = {
        'edge_density': {
            'color_precision': -0.2,
            'layer_difference': 0.1,
            'corner_threshold': -0.87,
            'length_threshold': 0.3,
            'max_iterations': 0.2,
            'splice_threshold': 0.1,
            'path_precision': -0.1
        },
        'unique_colors': {
            'color_precision': 0.92,
            'layer_difference': -0.6,
            'corner_threshold': 0.1,
            'length_threshold': 0.0,
            'max_iterations': 0.3,
            'splice_threshold': 0.2,
            'path_precision': 0.1
        },
        'entropy': {
            'color_precision': -0.1,
            'layer_difference': 0.2,
            'corner_threshold': 0.1,
            'length_threshold': 0.1,
            'max_iterations': 0.4,
            'splice_threshold': 0.3,
            'path_precision': -0.79
        },
        'corner_density': {
            'color_precision': 0.1,
            'layer_difference': 0.0,
            'corner_threshold': -0.3,
            'length_threshold': 0.81,
            'max_iterations': 0.2,
            'splice_threshold': 0.2,
            'path_precision': 0.3
        },
        'gradient_strength': {
            'color_precision': 0.3,
            'layer_difference': -0.4,
            'corner_threshold': 0.0,
            'length_threshold': 0.1,
            'max_iterations': 0.3,
            'splice_threshold': 0.88,
            'path_precision': 0.2
        },
        'complexity_score': {
            'color_precision': 0.4,
            'layer_difference': 0.1,
            'corner_threshold': -0.2,
            'length_threshold': 0.3,
            'max_iterations': 0.85,
            'splice_threshold': 0.4,
            'path_precision': -0.3
        }
    }

    @classmethod
    def get_correlation_matrix(cls) -> np.ndarray:
        """
        Get correlation matrix as numpy array.

        Returns:
            numpy array of shape (6, 7) with correlation values
        """
        features = list(cls.CORRELATION_MATRIX.keys())
        parameters = list(next(iter(cls.CORRELATION_MATRIX.values())).keys())

        matrix = np.zeros((len(features), len(parameters)))
        for i, feature in enumerate(features):
            for j, param in enumerate(parameters):
                matrix[i, j] = cls.CORRELATION_MATRIX[feature][param]

        return matrix

    @classmethod
    def get_strongest_correlations(cls, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Get feature-parameter pairs with strongest correlations.

        Args:
            threshold: Minimum absolute correlation value

        Returns:
            List of (feature, parameter, correlation) tuples
        """
        strong_correlations = []

        for feature, params in cls.CORRELATION_MATRIX.items():
            for param, correlation in params.items():
                if abs(correlation) >= threshold:
                    strong_correlations.append((feature, param, correlation))

        # Sort by absolute correlation strength
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        return strong_correlations

    @classmethod
    def describe_correlation(cls, value: float) -> str:
        """
        Describe correlation strength in words.

        Args:
            value: Correlation coefficient

        Returns:
            String description of correlation strength
        """
        abs_value = abs(value)
        if abs_value > 0.8:
            strength = "Strong"
        elif abs_value > 0.6:
            strength = "Moderate"
        elif abs_value > 0.4:
            strength = "Weak"
        else:
            strength = "Negligible"

        if value < 0:
            return f"{strength} negative"
        elif value > 0:
            return f"{strength} positive"
        else:
            return "No correlation"

    @classmethod
    def format_text_matrix(cls) -> str:
        """
        Format correlation matrix as text table.

        Returns:
            Formatted text representation of correlation matrix
        """
        features = list(cls.CORRELATION_MATRIX.keys())
        parameters = list(next(iter(cls.CORRELATION_MATRIX.values())).keys())

        # Create header
        param_abbrev = {
            'color_precision': 'CP',
            'layer_difference': 'LD',
            'corner_threshold': 'CT',
            'length_threshold': 'LT',
            'max_iterations': 'MI',
            'splice_threshold': 'ST',
            'path_precision': 'PP'
        }

        lines = []
        lines.append("Feature/Parameter Correlation Matrix:")
        lines.append("")

        # Header row
        header = "                    " + "  ".join(f"{param_abbrev[p]:>4}" for p in parameters)
        lines.append(header)

        # Data rows
        for feature in features:
            row_values = []
            for param in parameters:
                value = cls.CORRELATION_MATRIX[feature][param]
                row_values.append(f"{value:>4.2f}")
            row = f"{feature:<18} " + " ".join(row_values)
            lines.append(row)

        lines.append("")
        lines.append("Legend:")
        for param, abbrev in param_abbrev.items():
            lines.append(f"{abbrev} = {param}")

        return "\n".join(lines)

    @classmethod
    def get_feature_importance(cls, parameter: str) -> Dict[str, float]:
        """
        Get feature importance for a specific parameter.

        Args:
            parameter: Target parameter name

        Returns:
            Dictionary of feature importances
        """
        importance = {}
        for feature, params in cls.CORRELATION_MATRIX.items():
            if parameter in params:
                importance[feature] = abs(params[parameter])

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    @classmethod
    def validate_correlation_consistency(cls) -> Dict[str, List[str]]:
        """
        Check for potential inconsistencies in correlations.

        Returns:
            Dictionary of warnings about correlation patterns
        """
        warnings = []

        # Check for features with no strong correlations
        for feature in cls.CORRELATION_MATRIX:
            correlations = cls.CORRELATION_MATRIX[feature].values()
            if all(abs(c) < 0.4 for c in correlations):
                warnings.append(f"Feature '{feature}' has no strong correlations")

        # Check for parameters with multiple strong correlations
        parameters = list(next(iter(cls.CORRELATION_MATRIX.values())).keys())
        for param in parameters:
            strong_features = []
            for feature in cls.CORRELATION_MATRIX:
                if abs(cls.CORRELATION_MATRIX[feature][param]) > 0.7:
                    strong_features.append(feature)

            if len(strong_features) > 2:
                warnings.append(f"Parameter '{param}' has multiple strong correlations: {strong_features}")

        return {"warnings": warnings} if warnings else {"status": "All correlations are consistent"}

    @classmethod
    def export_correlation_data(cls, filepath: str):
        """
        Export correlation data to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        export_data = {
            "correlation_matrix": cls.CORRELATION_MATRIX,
            "strongest_correlations": [
                {"feature": f, "parameter": p, "correlation": c}
                for f, p, c in cls.get_strongest_correlations()
            ],
            "validation": cls.validate_correlation_consistency()
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported correlation data to {filepath}")


if __name__ == "__main__":
    # Generate and display correlation analysis
    visualizer = CorrelationVisualizer()

    print("=" * 60)
    print("VTracer Parameter Correlation Analysis")
    print("=" * 60)
    print()

    # Show correlation matrix
    print(visualizer.format_text_matrix())
    print()

    # Show strongest correlations
    print("Strongest Correlations (|r| >= 0.7):")
    print("-" * 40)
    for feature, param, corr in visualizer.get_strongest_correlations():
        description = visualizer.describe_correlation(corr)
        print(f"{feature} → {param}: {corr:.2f} ({description})")
    print()

    # Show parameter importance
    print("Feature Importance by Parameter:")
    print("-" * 40)
    for param in ['corner_threshold', 'color_precision', 'path_precision']:
        importance = visualizer.get_feature_importance(param)
        top_feature = list(importance.keys())[0]
        top_value = importance[top_feature]
        print(f"{param}: {top_feature} (importance: {top_value:.2f})")
    print()

    # Validate consistency
    validation = visualizer.validate_correlation_consistency()
    print("Correlation Consistency Check:")
    print("-" * 40)
    if "warnings" in validation:
        for warning in validation["warnings"]:
            print(f"⚠️  {warning}")
    else:
        print("✓ All correlations are consistent")
    print()

    print("=" * 60)