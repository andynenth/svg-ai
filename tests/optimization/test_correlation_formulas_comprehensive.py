# tests/optimization/test_correlation_formulas_comprehensive.py
"""Comprehensive unit tests for correlation formulas - Day 3"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.ai_modules.optimization import OptimizationEngine
from backend.ai_modules.optimization import OptimizationEngine


class TestCorrelationFormulas:
    """Comprehensive test suite for correlation formula accuracy"""

    def setup_method(self):
        """Setup test fixtures"""
        self.bounds = OptimizationEngine()

    def test_edge_to_corner_threshold_boundary_values(self):
        """Test edge density to corner threshold mapping with boundary values"""
        # Test exact boundary conditions as specified
        assert CorrelationFormulas.edge_to_corner_threshold(0.0) == 110
        assert CorrelationFormulas.edge_to_corner_threshold(0.125) == 10  # 110 - (0.125 * 800) = 10
        assert CorrelationFormulas.edge_to_corner_threshold(0.0625) == 60  # 110 - (0.0625 * 800) = 60
        assert CorrelationFormulas.edge_to_corner_threshold(1.0) == 10  # Clamped to minimum

    def test_edge_to_corner_threshold_edge_cases(self):
        """Test edge cases for edge_to_corner_threshold"""
        # Test negative values (should be clamped)
        assert CorrelationFormulas.edge_to_corner_threshold(-0.5) == 110

        # Test values > 1.0 (should be clamped)
        assert CorrelationFormulas.edge_to_corner_threshold(2.0) == 10

        # Test very small positive values
        assert CorrelationFormulas.edge_to_corner_threshold(0.001) == 109  # 110 - 0.8 = 109

        # Test intermediate values
        assert CorrelationFormulas.edge_to_corner_threshold(0.1) == 30  # 110 - 80 = 30

    def test_edge_to_corner_threshold_bounds_compliance(self):
        """Verify all edge_to_corner_threshold outputs are within bounds"""
        test_values = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.75, 1.0]
        bounds_info = self.bounds.get_parameter_info('corner_threshold')
        min_val = bounds_info['corner_threshold']['min']
        max_val = bounds_info['corner_threshold']['max']

        for edge_density in test_values:
            result = CorrelationFormulas.edge_to_corner_threshold(edge_density)
            assert min_val <= result <= max_val, f"Result {result} for edge_density {edge_density} is outside bounds [{min_val}, {max_val}]"

    def test_colors_to_precision_known_values(self):
        """Test unique colors to precision mapping with known values"""
        # Test exact values as specified
        assert CorrelationFormulas.colors_to_precision(2) == 3  # 2 + log2(2) = 3
        assert CorrelationFormulas.colors_to_precision(16) == 6  # 2 + log2(16) = 6
        assert CorrelationFormulas.colors_to_precision(256) == 10  # 2 + log2(256) = 10

        # Additional test cases
        assert CorrelationFormulas.colors_to_precision(1) == 2  # 2 + log2(1) = 2
        assert CorrelationFormulas.colors_to_precision(4) == 4  # 2 + log2(4) = 4
        assert CorrelationFormulas.colors_to_precision(8) == 5  # 2 + log2(8) = 5

    def test_colors_to_precision_edge_cases(self):
        """Test edge cases for colors_to_precision"""
        # Test zero and negative values (should be handled gracefully)
        assert CorrelationFormulas.colors_to_precision(0) == 2  # Should default to minimum
        assert CorrelationFormulas.colors_to_precision(-5) == 2  # Should default to minimum

        # Test very large values (should be clamped to maximum)
        assert CorrelationFormulas.colors_to_precision(10000) == 10  # Should clamp to max

        # Test floating point values
        assert CorrelationFormulas.colors_to_precision(3.5) >= 3  # Should handle floats

    def test_colors_to_precision_bounds_compliance(self):
        """Verify all colors_to_precision outputs are within bounds"""
        test_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        bounds_info = self.bounds.get_parameter_info('color_precision')
        min_val = bounds_info['color_precision']['min']
        max_val = bounds_info['color_precision']['max']

        for unique_colors in test_values:
            result = CorrelationFormulas.colors_to_precision(unique_colors)
            assert min_val <= result <= max_val, f"Result {result} for unique_colors {unique_colors} is outside bounds [{min_val}, {max_val}]"

    def test_entropy_to_path_precision_mapping(self):
        """Test entropy to path precision mapping"""
        # Test exact values as specified
        assert CorrelationFormulas.entropy_to_path_precision(0.0) == 20  # 20 * (1 - 0) = 20
        assert CorrelationFormulas.entropy_to_path_precision(1.0) == 1   # 20 * (1 - 1) = 0, clamped to 1
        assert CorrelationFormulas.entropy_to_path_precision(0.5) == 10  # 20 * (1 - 0.5) = 10

        # Additional test cases
        assert CorrelationFormulas.entropy_to_path_precision(0.25) == 15  # 20 * 0.75 = 15
        assert CorrelationFormulas.entropy_to_path_precision(0.75) == 5   # 20 * 0.25 = 5

    def test_entropy_to_path_precision_edge_cases(self):
        """Test edge cases for entropy_to_path_precision"""
        # Test negative values (should be clamped)
        assert CorrelationFormulas.entropy_to_path_precision(-0.5) == 20

        # Test values > 1.0 (should be clamped)
        assert CorrelationFormulas.entropy_to_path_precision(1.5) == 1

        # Test very small values
        assert CorrelationFormulas.entropy_to_path_precision(0.001) == 19  # 20 * 0.999 ≈ 19

    def test_entropy_to_path_precision_bounds_compliance(self):
        """Verify all entropy_to_path_precision outputs are within bounds"""
        test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bounds_info = self.bounds.get_parameter_info('path_precision')
        min_val = bounds_info['path_precision']['min']
        max_val = bounds_info['path_precision']['max']

        for entropy in test_values:
            result = CorrelationFormulas.entropy_to_path_precision(entropy)
            assert min_val <= result <= max_val, f"Result {result} for entropy {entropy} is outside bounds [{min_val}, {max_val}]"

    def test_corners_to_length_threshold_conversion(self):
        """Test corner density to length threshold conversion"""
        # Test exact values as specified
        assert abs(CorrelationFormulas.corners_to_length_threshold(0.0) - 1.0) < 0.01  # 1.0 + (0 * 100) = 1.0
        assert abs(CorrelationFormulas.corners_to_length_threshold(0.19) - 20.0) < 0.01  # 1.0 + (0.19 * 100) = 20.0

        # Additional test cases
        assert abs(CorrelationFormulas.corners_to_length_threshold(0.1) - 11.0) < 0.01  # 1.0 + 10 = 11.0
        assert abs(CorrelationFormulas.corners_to_length_threshold(0.05) - 6.0) < 0.01  # 1.0 + 5 = 6.0

    def test_corners_to_length_threshold_edge_cases(self):
        """Test edge cases for corners_to_length_threshold"""
        # Test negative values (should be clamped)
        assert abs(CorrelationFormulas.corners_to_length_threshold(-0.1) - 1.0) < 0.01

        # Test values that would exceed maximum (should be clamped)
        assert abs(CorrelationFormulas.corners_to_length_threshold(0.5) - 20.0) < 0.01  # Should clamp to 20.0
        assert abs(CorrelationFormulas.corners_to_length_threshold(1.0) - 20.0) < 0.01  # Should clamp to 20.0

    def test_corners_to_length_threshold_bounds_compliance(self):
        """Verify all corners_to_length_threshold outputs are within bounds"""
        test_values = [0.0, 0.01, 0.05, 0.1, 0.15, 0.19, 0.2, 0.5, 1.0]
        bounds_info = self.bounds.get_parameter_info('length_threshold')
        min_val = bounds_info['length_threshold']['min']
        max_val = bounds_info['length_threshold']['max']

        for corner_density in test_values:
            result = CorrelationFormulas.corners_to_length_threshold(corner_density)
            assert min_val <= result <= max_val, f"Result {result} for corner_density {corner_density} is outside bounds [{min_val}, {max_val}]"

    def test_gradient_to_splice_threshold_mapping(self):
        """Test gradient strength to splice threshold mapping"""
        # Test exact values as specified
        assert CorrelationFormulas.gradient_to_splice_threshold(0.0) == 10   # 10 + (0 * 90) = 10
        assert CorrelationFormulas.gradient_to_splice_threshold(1.0) == 100  # 10 + (1 * 90) = 100

        # Additional test cases
        assert CorrelationFormulas.gradient_to_splice_threshold(0.5) == 55   # 10 + 45 = 55
        assert CorrelationFormulas.gradient_to_splice_threshold(0.25) == 32  # 10 + 22.5 = 32 (rounded)
        assert CorrelationFormulas.gradient_to_splice_threshold(0.75) == 77  # 10 + 67.5 = 77 (rounded)

    def test_gradient_to_splice_threshold_edge_cases(self):
        """Test edge cases for gradient_to_splice_threshold"""
        # Test negative values (should be clamped)
        assert CorrelationFormulas.gradient_to_splice_threshold(-0.5) == 10

        # Test values > 1.0 (should be clamped)
        assert CorrelationFormulas.gradient_to_splice_threshold(2.0) == 100

    def test_gradient_to_splice_threshold_bounds_compliance(self):
        """Verify all gradient_to_splice_threshold outputs are within bounds"""
        test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bounds_info = self.bounds.get_parameter_info('splice_threshold')
        min_val = bounds_info['splice_threshold']['min']
        max_val = bounds_info['splice_threshold']['max']

        for gradient_strength in test_values:
            result = CorrelationFormulas.gradient_to_splice_threshold(gradient_strength)
            assert min_val <= result <= max_val, f"Result {result} for gradient_strength {gradient_strength} is outside bounds [{min_val}, {max_val}]"

    def test_complexity_to_iterations_conversion(self):
        """Test complexity score to max iterations conversion"""
        # Test exact values as specified
        assert CorrelationFormulas.complexity_to_iterations(0.0) == 5   # 5 + (0 * 15) = 5
        assert CorrelationFormulas.complexity_to_iterations(1.0) == 20  # 5 + (1 * 15) = 20

        # Additional test cases
        assert CorrelationFormulas.complexity_to_iterations(0.5) == 12  # 5 + 7.5 = 12 (rounded)
        assert CorrelationFormulas.complexity_to_iterations(0.33) == 9  # 5 + 4.95 = 9 (rounded)
        assert CorrelationFormulas.complexity_to_iterations(0.67) == 15  # 5 + 10.05 = 15 (rounded)

    def test_complexity_to_iterations_edge_cases(self):
        """Test edge cases for complexity_to_iterations"""
        # Test negative values (should be clamped)
        assert CorrelationFormulas.complexity_to_iterations(-0.5) == 5

        # Test values > 1.0 (should be clamped)
        assert CorrelationFormulas.complexity_to_iterations(2.0) == 20

    def test_complexity_to_iterations_bounds_compliance(self):
        """Verify all complexity_to_iterations outputs are within bounds"""
        test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bounds_info = self.bounds.get_parameter_info('max_iterations')
        min_val = bounds_info['max_iterations']['min']
        max_val = bounds_info['max_iterations']['max']

        for complexity_score in test_values:
            result = CorrelationFormulas.complexity_to_iterations(complexity_score)
            assert min_val <= result <= max_val, f"Result {result} for complexity_score {complexity_score} is outside bounds [{min_val}, {max_val}]"

    def test_all_formulas_with_invalid_inputs(self):
        """Test all formulas with various invalid inputs"""
        invalid_inputs = [None, "string", float('inf'), float('-inf'), float('nan')]

        for invalid_input in invalid_inputs:
            try:
                # These should either handle gracefully or raise appropriate exceptions
                result = CorrelationFormulas.edge_to_corner_threshold(invalid_input)
                # If no exception, verify result is within bounds
                assert 10 <= result <= 110
            except (TypeError, ValueError, OverflowError):
                # Expected behavior for invalid inputs
                pass

            try:
                result = CorrelationFormulas.colors_to_precision(invalid_input)
                assert 2 <= result <= 10
            except (TypeError, ValueError, OverflowError):
                pass

            try:
                result = CorrelationFormulas.entropy_to_path_precision(invalid_input)
                assert 1 <= result <= 20
            except (TypeError, ValueError, OverflowError):
                pass

            try:
                result = CorrelationFormulas.corners_to_length_threshold(invalid_input)
                assert 1.0 <= result <= 20.0
            except (TypeError, ValueError, OverflowError):
                pass

            try:
                result = CorrelationFormulas.gradient_to_splice_threshold(invalid_input)
                assert 10 <= result <= 100
            except (TypeError, ValueError, OverflowError):
                pass

            try:
                result = CorrelationFormulas.complexity_to_iterations(invalid_input)
                assert 5 <= result <= 20
            except (TypeError, ValueError, OverflowError):
                pass

    def test_formula_consistency(self):
        """Test that formulas are mathematically consistent"""
        # Test that increasing input generally leads to expected output changes

        # edge_to_corner_threshold: higher edge density should give lower threshold
        result1 = CorrelationFormulas.edge_to_corner_threshold(0.0)
        result2 = CorrelationFormulas.edge_to_corner_threshold(0.1)
        assert result1 > result2, "Higher edge density should give lower corner threshold"

        # colors_to_precision: more colors should give higher precision
        result1 = CorrelationFormulas.colors_to_precision(2)
        result2 = CorrelationFormulas.colors_to_precision(16)
        assert result1 < result2, "More colors should give higher precision"

        # entropy_to_path_precision: higher entropy should give lower precision
        result1 = CorrelationFormulas.entropy_to_path_precision(0.0)
        result2 = CorrelationFormulas.entropy_to_path_precision(0.5)
        assert result1 > result2, "Higher entropy should give lower path precision"

        # corners_to_length_threshold: more corners should give longer threshold
        result1 = CorrelationFormulas.corners_to_length_threshold(0.0)
        result2 = CorrelationFormulas.corners_to_length_threshold(0.1)
        assert result1 < result2, "More corners should give longer length threshold"

        # gradient_to_splice_threshold: stronger gradients should give higher threshold
        result1 = CorrelationFormulas.gradient_to_splice_threshold(0.0)
        result2 = CorrelationFormulas.gradient_to_splice_threshold(0.5)
        assert result1 < result2, "Stronger gradients should give higher splice threshold"

        # complexity_to_iterations: higher complexity should give more iterations
        result1 = CorrelationFormulas.complexity_to_iterations(0.0)
        result2 = CorrelationFormulas.complexity_to_iterations(0.5)
        assert result1 < result2, "Higher complexity should give more iterations"

    def test_formula_precision(self):
        """Test that formulas maintain precision across different input ranges"""
        # Test multiple values in each range to ensure precision is maintained

        edge_densities = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        for i in range(len(edge_densities) - 1):
            result1 = CorrelationFormulas.edge_to_corner_threshold(edge_densities[i])
            result2 = CorrelationFormulas.edge_to_corner_threshold(edge_densities[i + 1])
            # Results should be different for different inputs (unless at bounds)
            if edge_densities[i] < 0.125 and edge_densities[i + 1] < 0.125:
                assert result1 != result2 or (result1 == 10 and result2 == 10), f"Results should differ for inputs {edge_densities[i]} and {edge_densities[i + 1]}"

    def test_mathematical_correctness(self):
        """Verify mathematical correctness of formula implementations"""

        # Test specific mathematical relationships

        # For colors_to_precision with powers of 2
        for power in range(1, 9):  # 2^1 to 2^8
            colors = 2 ** power
            expected = 2 + power
            if expected <= 10:  # Within bounds
                result = CorrelationFormulas.colors_to_precision(colors)
                assert result == expected, f"For {colors} colors (2^{power}), expected {expected}, got {result}"

        # For entropy_to_path_precision with specific fractions
        test_cases = [(0.0, 20), (0.25, 15), (0.5, 10), (0.75, 5), (1.0, 1)]
        for entropy, expected in test_cases:
            result = CorrelationFormulas.entropy_to_path_precision(entropy)
            assert result == expected, f"For entropy {entropy}, expected {expected}, got {result}"

        # For gradient_to_splice_threshold with specific values
        test_cases = [(0.0, 10), (0.1, 19), (0.5, 55), (0.9, 91), (1.0, 100)]
        for gradient, expected in test_cases:
            result = CorrelationFormulas.gradient_to_splice_threshold(gradient)
            assert result == expected, f"For gradient {gradient}, expected {expected}, got {result}"