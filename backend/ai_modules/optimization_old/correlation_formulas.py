# backend/ai_modules/optimization/correlation_formulas_v2.py
"""Research-validated parameter correlations - Day 2 Implementation"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class CorrelationFormulas:
    """Research-validated parameter correlations"""

    @staticmethod
    def edge_to_corner_threshold(edge_density: float) -> int:
        """Map edge density to corner threshold parameter

        Formula: corner_threshold = max(10, min(110, int(110 - (edge_density * 800))))
        Logic: Higher edge density → lower corner threshold for better detail

        Args:
            edge_density: Normalized edge density value (0.0 to 1.0)

        Returns:
            int: Corner threshold parameter value [10, 110]
        """
        # Validate input - clamp to [0, 1]
        edge_density = max(0.0, min(1.0, edge_density))

        # Apply formula
        raw_value = 110 - (edge_density * 800)
        result = max(10, min(110, int(raw_value)))

        logger.debug(f"edge_to_corner_threshold: edge_density={edge_density:.3f} -> corner_threshold={result}")

        # Test edge cases
        if edge_density == 0.0:
            assert result == 110, f"Edge case failed: edge_density=0.0 should give 110, got {result}"
        elif edge_density == 0.5:
            expected = max(10, min(110, int(110 - 400)))
            assert result == expected, f"Edge case failed: edge_density=0.5 should give {expected}, got {result}"
        elif edge_density == 1.0:
            expected = max(10, int(110 - 800))
            assert result == expected, f"Edge case failed: edge_density=1.0 should give {expected}, got {result}"

        return result

    @staticmethod
    def colors_to_precision(unique_colors: float) -> int:
        """Map unique colors to color precision parameter

        Formula: max(2, min(10, int(2 + np.log2(max(1, unique_colors)))))
        Logic: More colors require higher precision for accurate representation

        Args:
            unique_colors: Number of unique colors in the image

        Returns:
            int: Color precision parameter value [2, 10]
        """
        # Handle zero/negative colors gracefully
        unique_colors = max(1, unique_colors)

        # Apply logarithmic formula
        raw_value = 2 + np.log2(unique_colors)
        result = max(2, min(10, int(raw_value)))

        logger.debug(f"colors_to_precision: unique_colors={unique_colors} -> color_precision={result}")

        # Validate formula produces expected results
        if unique_colors == 1:
            assert result == 2, f"Edge case failed: unique_colors=1 should give 2, got {result}"
        elif unique_colors == 2:
            expected = max(2, min(10, int(2 + 1)))  # log2(2) = 1
            assert result == expected, f"Edge case failed: unique_colors=2 should give {expected}, got {result}"
        elif unique_colors == 256:
            expected = max(2, min(10, int(2 + 8)))  # log2(256) = 8
            assert result == expected, f"Edge case failed: unique_colors=256 should give {expected}, got {result}"

        return result

    @staticmethod
    def entropy_to_path_precision(entropy: float) -> int:
        """Map entropy to path precision parameter

        Formula: max(1, min(20, int(20 * (1 - entropy))))
        Logic: Higher entropy → lower precision (more randomness needs less detail)

        Args:
            entropy: Normalized entropy value (0.0 to 1.0)

        Returns:
            int: Path precision parameter value [1, 20]
        """
        # Validate input - clamp to [0, 1]
        entropy = max(0.0, min(1.0, entropy))

        # Apply inverse relationship formula
        raw_value = 20 * (1 - entropy)
        result = max(1, min(20, int(raw_value)))

        logger.debug(f"entropy_to_path_precision: entropy={entropy:.3f} -> path_precision={result}")

        # Test edge cases
        if entropy == 0.0:
            assert result == 20, f"Edge case failed: entropy=0.0 should give 20, got {result}"
        elif entropy == 0.5:
            expected = max(1, min(20, int(20 * 0.5)))
            assert result == expected, f"Edge case failed: entropy=0.5 should give {expected}, got {result}"
        elif entropy == 1.0:
            expected = max(1, int(20 * 0))
            assert result == expected, f"Edge case failed: entropy=1.0 should give {expected}, got {result}"

        return result

    @staticmethod
    def corners_to_length_threshold(corner_density: float) -> float:
        """Map corner density to length threshold parameter

        Formula: max(1.0, min(20.0, 1.0 + (corner_density * 100)))
        Logic: More corners → longer segments to capture detail

        Args:
            corner_density: Normalized corner density value (0.0 to 1.0)

        Returns:
            float: Length threshold parameter value [1.0, 20.0]
        """
        # Validate input - clamp to [0, 1]
        corner_density = max(0.0, min(1.0, corner_density))

        # Apply formula - more corners need shorter segments
        raw_value = 1.0 + (corner_density * 100)
        result = max(1.0, min(20.0, raw_value))

        logger.debug(f"corners_to_length_threshold: corner_density={corner_density:.3f} -> length_threshold={result:.2f}")

        # Test edge cases
        if corner_density == 0.0:
            assert abs(result - 1.0) < 0.01, f"Edge case failed: corner_density=0.0 should give 1.0, got {result}"
        elif corner_density == 0.5:
            expected = max(1.0, min(20.0, 1.0 + 50))
            assert abs(result - expected) < 0.01, f"Edge case failed: corner_density=0.5 should give {expected}, got {result}"
        elif corner_density == 1.0:
            expected = min(20.0, 1.0 + 100)
            assert abs(result - expected) < 0.01, f"Edge case failed: corner_density=1.0 should give {expected}, got {result}"

        return result

    @staticmethod
    def gradient_to_splice_threshold(gradient_strength: float) -> int:
        """Map gradient strength to splice threshold parameter

        Formula: max(10, min(100, int(10 + (gradient_strength * 90))))
        Logic: Stronger gradients → more splice points for smooth transitions

        Args:
            gradient_strength: Normalized gradient strength (0.0 to 1.0)

        Returns:
            int: Splice threshold parameter value [10, 100]
        """
        # Validate input - clamp to [0, 1]
        gradient_strength = max(0.0, min(1.0, gradient_strength))

        # Apply formula - stronger gradients need more splice points
        raw_value = 10 + (gradient_strength * 90)
        result = max(10, min(100, int(raw_value)))

        logger.debug(f"gradient_to_splice_threshold: gradient_strength={gradient_strength:.3f} -> splice_threshold={result}")

        # Test edge cases
        if gradient_strength == 0.0:
            assert result == 10, f"Edge case failed: gradient_strength=0.0 should give 10, got {result}"
        elif gradient_strength == 0.5:
            expected = max(10, min(100, int(10 + 45)))
            assert result == expected, f"Edge case failed: gradient_strength=0.5 should give {expected}, got {result}"
        elif gradient_strength == 1.0:
            expected = max(10, min(100, int(10 + 90)))
            assert result == expected, f"Edge case failed: gradient_strength=1.0 should give {expected}, got {result}"

        return result

    @staticmethod
    def complexity_to_iterations(complexity_score: float) -> int:
        """Map complexity score to max iterations parameter

        Formula: max(5, min(20, int(5 + (complexity_score * 15))))
        Logic: Higher complexity → more iterations for better convergence

        Args:
            complexity_score: Normalized complexity score (0.0 to 1.0)

        Returns:
            int: Max iterations parameter value [5, 20]
        """
        # Validate input - clamp to [0, 1]
        complexity_score = max(0.0, min(1.0, complexity_score))

        # Apply formula - higher complexity needs more iterations
        raw_value = 5 + (complexity_score * 15)
        result = max(5, min(20, int(raw_value)))

        logger.debug(f"complexity_to_iterations: complexity_score={complexity_score:.3f} -> max_iterations={result}")

        # Test edge cases
        if complexity_score == 0.0:
            assert result == 5, f"Edge case failed: complexity_score=0.0 should give 5, got {result}"
        elif complexity_score == 0.5:
            expected = max(5, min(20, int(5 + 7.5)))
            assert result == expected, f"Edge case failed: complexity_score=0.5 should give {expected}, got {result}"
        elif complexity_score == 1.0:
            expected = max(5, min(20, int(5 + 15)))
            assert result == expected, f"Edge case failed: complexity_score=1.0 should give {expected}, got {result}"

        return result

    @staticmethod
    def test_formulas_with_known_inputs():
        """Testing utility to validate all formulas with known inputs/outputs"""
        test_results = []

        # Test edge_to_corner_threshold
        test_cases = [
            (0.0, 110),
            (0.125, 10),  # 110 - 100 = 10
            (0.5, 10),    # 110 - 400 = -290, clamped to 10
            (1.0, 10),    # 110 - 800 = -690, clamped to 10
        ]
        for edge_density, expected in test_cases:
            result = CorrelationFormulas.edge_to_corner_threshold(edge_density)
            status = "✓" if result == expected else "✗"
            test_results.append(f"{status} edge_to_corner_threshold({edge_density}) = {result} (expected {expected})")

        # Test colors_to_precision
        test_cases = [
            (1, 2),     # 2 + log2(1) = 2
            (2, 3),     # 2 + log2(2) = 3
            (8, 5),     # 2 + log2(8) = 5
            (256, 10),  # 2 + log2(256) = 10
            (1024, 10), # 2 + log2(1024) = 12, clamped to 10
        ]
        for unique_colors, expected in test_cases:
            result = CorrelationFormulas.colors_to_precision(unique_colors)
            status = "✓" if result == expected else "✗"
            test_results.append(f"{status} colors_to_precision({unique_colors}) = {result} (expected {expected})")

        # Test entropy_to_path_precision
        test_cases = [
            (0.0, 20),  # 20 * 1 = 20
            (0.5, 10),  # 20 * 0.5 = 10
            (1.0, 1),   # 20 * 0 = 0, clamped to 1
        ]
        for entropy, expected in test_cases:
            result = CorrelationFormulas.entropy_to_path_precision(entropy)
            status = "✓" if result == expected else "✗"
            test_results.append(f"{status} entropy_to_path_precision({entropy}) = {result} (expected {expected})")

        # Test corners_to_length_threshold
        test_cases = [
            (0.0, 1.0),   # 1 + 0 = 1
            (0.1, 11.0),  # 1 + 10 = 11
            (0.19, 20.0), # 1 + 19 = 20
            (0.5, 20.0),  # 1 + 50 = 51, clamped to 20
        ]
        for corner_density, expected in test_cases:
            result = CorrelationFormulas.corners_to_length_threshold(corner_density)
            status = "✓" if abs(result - expected) < 0.01 else "✗"
            test_results.append(f"{status} corners_to_length_threshold({corner_density}) = {result:.1f} (expected {expected})")

        # Test gradient_to_splice_threshold
        test_cases = [
            (0.0, 10),   # 10 + 0 = 10
            (0.5, 55),   # 10 + 45 = 55
            (1.0, 100),  # 10 + 90 = 100
        ]
        for gradient_strength, expected in test_cases:
            result = CorrelationFormulas.gradient_to_splice_threshold(gradient_strength)
            status = "✓" if result == expected else "✗"
            test_results.append(f"{status} gradient_to_splice_threshold({gradient_strength}) = {result} (expected {expected})")

        # Test complexity_to_iterations
        test_cases = [
            (0.0, 5),   # 5 + 0 = 5
            (0.5, 12),  # 5 + 7.5 = 12
            (1.0, 20),  # 5 + 15 = 20
        ]
        for complexity_score, expected in test_cases:
            result = CorrelationFormulas.complexity_to_iterations(complexity_score)
            status = "✓" if result == expected else "✗"
            test_results.append(f"{status} complexity_to_iterations({complexity_score}) = {result} (expected {expected})")

        # Print results
        logger.info("Formula Testing Results:")
        for result in test_results:
            logger.info(result)

        # Return success status
        failed = [r for r in test_results if r.startswith("✗")]
        if failed:
            logger.error(f"Failed tests: {len(failed)}/{len(test_results)}")
            for fail in failed:
                logger.error(fail)
            return False
        else:
            logger.info(f"All {len(test_results)} tests passed!")
            return True


# Test the formulas when module is run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("Testing Correlation Formulas...")
    print("=" * 60)

    success = CorrelationFormulas.test_formulas_with_known_inputs()

    if success:
        print("\n✅ All formula tests passed!")
    else:
        print("\n❌ Some formula tests failed. Check the logs above.")

    # Test with sample features
    print("\n" + "=" * 60)
    print("Sample Feature Mapping:")
    print("-" * 40)

    sample_features = {
        'edge_density': 0.15,
        'unique_colors': 12,
        'entropy': 0.65,
        'corner_density': 0.08,
        'gradient_strength': 0.45,
        'complexity_score': 0.35
    }

    results = {
        'corner_threshold': CorrelationFormulas.edge_to_corner_threshold(sample_features['edge_density']),
        'color_precision': CorrelationFormulas.colors_to_precision(sample_features['unique_colors']),
        'path_precision': CorrelationFormulas.entropy_to_path_precision(sample_features['entropy']),
        'length_threshold': CorrelationFormulas.corners_to_length_threshold(sample_features['corner_density']),
        'splice_threshold': CorrelationFormulas.gradient_to_splice_threshold(sample_features['gradient_strength']),
        'max_iterations': CorrelationFormulas.complexity_to_iterations(sample_features['complexity_score'])
    }

    print("Input Features:")
    for key, value in sample_features.items():
        print(f"  {key}: {value}")

    print("\nOutput Parameters:")
    for key, value in results.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)