#!/usr/bin/env python3
"""
Unit tests for VTracerParameterOptimizer

Tests parameter optimization logic, bounds validation, confidence adjustments,
and feature-based fine-tuning.
"""

import unittest
import sys
import os
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.ai_modules.parameter_optimizer import (
    VTracerParameterOptimizer, LogoType, ParameterBounds, OptimizationResult
)


class TestVTracerParameterOptimizer(unittest.TestCase):
    """Test suite for VTracerParameterOptimizer class"""

    def setUp(self):
        """Set up test environment"""
        self.optimizer = VTracerParameterOptimizer()

        # Standard test features (all in [0,1] range)
        self.test_features = {
            'edge_density': 0.5,
            'unique_colors': 0.5,
            'entropy': 0.5,
            'corner_density': 0.5,
            'gradient_strength': 0.5,
            'complexity_score': 0.5
        }

    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        self.assertIsInstance(self.optimizer, VTracerParameterOptimizer)
        self.assertIsInstance(self.optimizer.bounds, ParameterBounds)
        self.assertEqual(self.optimizer.stats['total_optimizations'], 0)
        self.assertIsInstance(self.optimizer.optimization_history, list)

    def test_parameter_bounds_definition(self):
        """Test parameter bounds are correctly defined"""
        bounds = self.optimizer.bounds

        # Test colormode options
        self.assertIn('color', bounds.colormode_options)
        self.assertIn('binary', bounds.colormode_options)

        # Test numeric ranges
        self.assertEqual(bounds.color_precision_range, (1, 10))
        self.assertEqual(bounds.layer_difference_range, (1, 32))
        self.assertEqual(bounds.path_precision_range, (0, 10))
        self.assertEqual(bounds.corner_threshold_range, (0, 180))
        self.assertEqual(bounds.length_threshold_range, (0.1, 10.0))
        self.assertEqual(bounds.max_iterations_range, (1, 50))
        self.assertEqual(bounds.splice_threshold_range, (0, 180))

    def test_logo_type_base_parameters(self):
        """Test base parameters for each logo type"""
        for logo_type in LogoType:
            params = self.optimizer._get_base_parameters_for_type(logo_type.value)

            # Verify all required parameters are present
            required_params = [
                'colormode', 'color_precision', 'layer_difference', 'path_precision',
                'corner_threshold', 'length_threshold', 'max_iterations', 'splice_threshold'
            ]

            for param in required_params:
                self.assertIn(param, params, f"Missing parameter {param} for {logo_type.value}")

            # Verify parameter types and basic bounds
            self.assertIn(params['colormode'], ['color', 'binary'])
            self.assertIsInstance(params['color_precision'], int)
            self.assertTrue(1 <= params['color_precision'] <= 10)
            self.assertIsInstance(params['layer_difference'], int)
            self.assertTrue(1 <= params['layer_difference'] <= 32)

    def test_simple_logo_parameters(self):
        """Test specific parameters for simple logos"""
        params = self.optimizer._get_base_parameters_for_type('simple')

        # Simple logos should have fewer colors and sharp corners
        self.assertEqual(params['colormode'], 'color')
        self.assertEqual(params['color_precision'], 3)  # Fewer colors
        self.assertEqual(params['corner_threshold'], 30)  # Sharp corners
        self.assertEqual(params['layer_difference'], 32)  # High separation

    def test_text_logo_parameters(self):
        """Test specific parameters for text logos"""
        params = self.optimizer._get_base_parameters_for_type('text')

        # Text logos should have minimal colors and very sharp corners
        self.assertEqual(params['color_precision'], 2)  # Minimal colors
        self.assertEqual(params['corner_threshold'], 20)  # Very sharp corners
        self.assertEqual(params['path_precision'], 8)  # High precision

    def test_gradient_logo_parameters(self):
        """Test specific parameters for gradient logos"""
        params = self.optimizer._get_base_parameters_for_type('gradient')

        # Gradient logos should have high precision and smooth curves
        self.assertEqual(params['color_precision'], 8)  # High precision
        self.assertEqual(params['layer_difference'], 8)  # Fine layers
        self.assertEqual(params['corner_threshold'], 60)  # Smoother curves

    def test_complex_logo_parameters(self):
        """Test specific parameters for complex logos"""
        params = self.optimizer._get_base_parameters_for_type('complex')

        # Complex logos should have balanced parameters with more iterations
        self.assertEqual(params['color_precision'], 6)  # Balanced
        self.assertEqual(params['max_iterations'], 20)  # More iterations
        self.assertEqual(params['corner_threshold'], 45)  # Balanced

    def test_confidence_adjustments_high(self):
        """Test confidence adjustments for high confidence (>=0.8)"""
        base_params = {'color_precision': 3, 'layer_difference': 32, 'corner_threshold': 30}

        # High confidence should not change parameters
        adjusted = self.optimizer._apply_confidence_adjustments(base_params, 0.9)
        self.assertEqual(adjusted, base_params)

    def test_confidence_adjustments_medium(self):
        """Test confidence adjustments for medium confidence (0.6-0.8)"""
        # Test extreme low color precision adjustment
        base_params = {'color_precision': 2, 'layer_difference': 8, 'corner_threshold': 30}
        adjusted = self.optimizer._apply_confidence_adjustments(base_params, 0.7)
        self.assertEqual(adjusted['color_precision'], 3)  # Should be increased from 2

        # Test extreme high color precision adjustment
        base_params = {'color_precision': 8, 'layer_difference': 30, 'corner_threshold': 30}
        adjusted = self.optimizer._apply_confidence_adjustments(base_params, 0.7)
        self.assertEqual(adjusted['color_precision'], 6)  # Should be decreased from 8

    def test_confidence_adjustments_low(self):
        """Test confidence adjustments for low confidence (<0.6)"""
        base_params = {'color_precision': 2, 'layer_difference': 8, 'corner_threshold': 20}

        # Low confidence should use conservative parameters
        adjusted = self.optimizer._apply_confidence_adjustments(base_params, 0.4)
        self.assertEqual(adjusted['color_precision'], 5)
        self.assertEqual(adjusted['layer_difference'], 16)
        self.assertEqual(adjusted['corner_threshold'], 50)

    def test_feature_adjustments_edge_density(self):
        """Test feature adjustments based on edge density"""
        base_params = {'corner_threshold': 50}

        # High edge density should lower corner threshold
        features = self.test_features.copy()
        features['edge_density'] = 0.8
        adjusted = self.optimizer._apply_feature_adjustments(base_params, features)
        self.assertLess(adjusted['corner_threshold'], base_params['corner_threshold'])

        # Low edge density should raise corner threshold
        features['edge_density'] = 0.05
        adjusted = self.optimizer._apply_feature_adjustments(base_params, features)
        self.assertGreater(adjusted['corner_threshold'], base_params['corner_threshold'])

    def test_feature_adjustments_color_complexity(self):
        """Test feature adjustments based on color complexity"""
        base_params = {'color_precision': 5}

        # High color complexity should increase precision
        features = self.test_features.copy()
        features['unique_colors'] = 0.9
        adjusted = self.optimizer._apply_feature_adjustments(base_params, features)
        self.assertGreater(adjusted['color_precision'], base_params['color_precision'])

        # Low color complexity should decrease precision
        features['unique_colors'] = 0.1
        adjusted = self.optimizer._apply_feature_adjustments(base_params, features)
        self.assertLess(adjusted['color_precision'], base_params['color_precision'])

    def test_feature_adjustments_complexity_score(self):
        """Test feature adjustments based on overall complexity"""
        base_params = {'max_iterations': 10, 'layer_difference': 16}

        # High complexity should increase iterations and decrease layer difference
        features = self.test_features.copy()
        features['complexity_score'] = 0.9
        adjusted = self.optimizer._apply_feature_adjustments(base_params, features)
        self.assertGreater(adjusted['max_iterations'], base_params['max_iterations'])
        self.assertLess(adjusted['layer_difference'], base_params['layer_difference'])

    def test_feature_adjustments_gradient_strength(self):
        """Test feature adjustments based on gradient strength"""
        base_params = {'color_precision': 4, 'layer_difference': 20, 'corner_threshold': 30}

        # Strong gradients should optimize for smoothness
        features = self.test_features.copy()
        features['gradient_strength'] = 0.8
        adjusted = self.optimizer._apply_feature_adjustments(base_params, features)
        self.assertGreaterEqual(adjusted['color_precision'], 6)  # Should be at least 6
        self.assertLessEqual(adjusted['layer_difference'], 12)  # Should be at most 12
        self.assertGreaterEqual(adjusted['corner_threshold'], 50)  # Should be at least 50

    def test_parameter_validation_colormode(self):
        """Test parameter validation for colormode"""
        # Valid colormode
        params = {'colormode': 'color'}
        validated, passed = self.optimizer._validate_and_correct_parameters(params)
        self.assertTrue(passed)
        self.assertEqual(validated['colormode'], 'color')

        # Invalid colormode
        params = {'colormode': 'invalid'}
        validated, passed = self.optimizer._validate_and_correct_parameters(params)
        self.assertFalse(passed)
        self.assertEqual(validated['colormode'], 'color')

    def test_parameter_validation_numeric_bounds(self):
        """Test parameter validation for numeric bounds"""
        # Test color_precision bounds
        params = {'color_precision': 15}  # Above max (10)
        validated, passed = self.optimizer._validate_and_correct_parameters(params)
        self.assertFalse(passed)
        self.assertEqual(validated['color_precision'], 10)

        params = {'color_precision': 0}  # Below min (1)
        validated, passed = self.optimizer._validate_and_correct_parameters(params)
        self.assertFalse(passed)
        self.assertEqual(validated['color_precision'], 1)

        # Test layer_difference bounds
        params = {'layer_difference': 50}  # Above max (32)
        validated, passed = self.optimizer._validate_and_correct_parameters(params)
        self.assertFalse(passed)
        self.assertEqual(validated['layer_difference'], 32)

    def test_parameter_validation_type_correction(self):
        """Test parameter validation with type correction"""
        # Test string to int conversion
        params = {'color_precision': '5'}
        validated, passed = self.optimizer._validate_and_correct_parameters(params)
        self.assertTrue(passed)
        self.assertEqual(validated['color_precision'], 5)
        self.assertIsInstance(validated['color_precision'], int)

        # Test invalid type that can't be converted
        params = {'color_precision': 'invalid'}
        validated, passed = self.optimizer._validate_and_correct_parameters(params)
        self.assertFalse(passed)
        # Should use middle of range as default
        self.assertEqual(validated['color_precision'], 5)  # (1+10)/2 = 5.5 -> 5

    def test_optimization_input_validation(self):
        """Test optimization input validation"""
        # Valid inputs
        classification = {'logo_type': 'simple', 'confidence': 0.8}
        features = self.test_features.copy()

        # Should not raise any exceptions
        self.optimizer._validate_optimization_inputs(classification, features)

        # Missing logo_type
        with self.assertRaises(KeyError):
            self.optimizer._validate_optimization_inputs({}, features)

        # Invalid logo_type
        with self.assertRaises(ValueError):
            self.optimizer._validate_optimization_inputs(
                {'logo_type': 'invalid', 'confidence': 0.8}, features
            )

        # Invalid confidence
        with self.assertRaises(ValueError):
            self.optimizer._validate_optimization_inputs(
                {'logo_type': 'simple', 'confidence': 1.5}, features
            )

    def test_complete_optimization_workflow(self):
        """Test complete optimization workflow"""
        classification = {'logo_type': 'simple', 'confidence': 0.85}
        features = {
            'edge_density': 0.2,
            'unique_colors': 0.3,
            'entropy': 0.4,
            'corner_density': 0.25,
            'gradient_strength': 0.1,
            'complexity_score': 0.3
        }

        result = self.optimizer.optimize_parameters(classification, features)

        # Verify result structure
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.logo_type, 'simple')
        self.assertEqual(result.confidence, 0.85)
        self.assertIsInstance(result.parameters, dict)
        self.assertTrue(result.validation_passed)
        self.assertIsInstance(result.adjustments_applied, list)
        self.assertGreater(result.optimization_time, 0)

    def test_optimization_with_user_overrides(self):
        """Test optimization with user parameter overrides"""
        classification = {'logo_type': 'simple', 'confidence': 0.9}
        features = self.test_features.copy()
        user_overrides = {'color_precision': 7, 'corner_threshold': 25}

        result = self.optimizer.optimize_parameters(
            classification, features, user_overrides=user_overrides
        )

        # User overrides should be preserved
        self.assertEqual(result.parameters['color_precision'], 7)
        self.assertEqual(result.parameters['corner_threshold'], 25)
        self.assertIn('user_overrides', result.adjustments_applied)

    def test_optimization_with_custom_base_parameters(self):
        """Test optimization with custom base parameters"""
        classification = {'logo_type': 'simple', 'confidence': 0.9}
        features = self.test_features.copy()
        custom_base = {
            'colormode': 'color',
            'color_precision': 5,
            'layer_difference': 20,
            'path_precision': 7,
            'corner_threshold': 40,
            'length_threshold': 4.0,
            'max_iterations': 15,
            'splice_threshold': 50
        }

        result = self.optimizer.optimize_parameters(
            classification, features, base_parameters=custom_base
        )

        # Should start from custom base instead of logo type defaults
        self.assertTrue(result.optimization_method.startswith('custom_base'))

    def test_optimization_statistics_tracking(self):
        """Test optimization statistics tracking"""
        initial_count = self.optimizer.stats['total_optimizations']

        # Perform several optimizations
        for logo_type in ['simple', 'text', 'gradient']:
            classification = {'logo_type': logo_type, 'confidence': 0.8}
            self.optimizer.optimize_parameters(classification, self.test_features)

        # Verify statistics updated
        self.assertEqual(
            self.optimizer.stats['total_optimizations'],
            initial_count + 3
        )
        self.assertGreater(self.optimizer.stats['average_optimization_time'], 0)

        # Check logo type breakdown
        stats = self.optimizer.get_optimization_stats()
        self.assertIn('simple', stats['by_logo_type'])
        self.assertIn('text', stats['by_logo_type'])
        self.assertIn('gradient', stats['by_logo_type'])

    def test_confidence_level_categorization(self):
        """Test confidence level categorization"""
        self.assertEqual(self.optimizer._get_confidence_level(0.5), 'low')
        self.assertEqual(self.optimizer._get_confidence_level(0.7), 'medium')
        self.assertEqual(self.optimizer._get_confidence_level(0.9), 'high')

        # Edge cases
        self.assertEqual(self.optimizer._get_confidence_level(0.6), 'medium')
        self.assertEqual(self.optimizer._get_confidence_level(0.8), 'high')

    def test_parameter_recommendations(self):
        """Test parameter recommendations for logo types"""
        for logo_type in ['simple', 'text', 'gradient', 'complex']:
            params = self.optimizer.get_parameter_recommendations(logo_type)
            self.assertIsInstance(params, dict)

            # Verify all required parameters present
            required_params = [
                'colormode', 'color_precision', 'layer_difference', 'path_precision',
                'corner_threshold', 'length_threshold', 'max_iterations', 'splice_threshold'
            ]
            for param in required_params:
                self.assertIn(param, params)

        # Test invalid logo type
        with self.assertRaises(ValueError):
            self.optimizer.get_parameter_recommendations('invalid')

    def test_optimization_history_tracking(self):
        """Test optimization history tracking"""
        initial_history_length = len(self.optimizer.optimization_history)

        # Perform optimization
        classification = {'logo_type': 'simple', 'confidence': 0.8}
        result = self.optimizer.optimize_parameters(classification, self.test_features)

        # Verify history updated
        self.assertEqual(
            len(self.optimizer.optimization_history),
            initial_history_length + 1
        )
        self.assertEqual(self.optimizer.optimization_history[-1], result)

    def test_missing_features_handling(self):
        """Test handling of missing features"""
        classification = {'logo_type': 'simple', 'confidence': 0.8}
        incomplete_features = {'edge_density': 0.3, 'unique_colors': 0.4}  # Missing some features

        # Should not raise exception, should use defaults
        result = self.optimizer.optimize_parameters(classification, incomplete_features)
        self.assertIsInstance(result, OptimizationResult)

    def test_feature_value_clipping(self):
        """Test feature value clipping to [0,1] range"""
        classification = {'logo_type': 'simple', 'confidence': 0.8}
        invalid_features = {
            'edge_density': 1.5,  # Above 1.0
            'unique_colors': -0.1,  # Below 0.0
            'entropy': 0.5,
            'corner_density': 0.5,
            'gradient_strength': 0.5,
            'complexity_score': 0.5
        }

        # Should clip values and proceed without error
        result = self.optimizer.optimize_parameters(classification, invalid_features)
        self.assertIsInstance(result, OptimizationResult)


class TestParameterBounds(unittest.TestCase):
    """Test suite for ParameterBounds dataclass"""

    def test_parameter_bounds_structure(self):
        """Test ParameterBounds dataclass structure"""
        bounds = ParameterBounds(
            colormode_options=['color', 'binary'],
            color_precision_range=(1, 10),
            layer_difference_range=(1, 32),
            path_precision_range=(0, 10),
            corner_threshold_range=(0, 180),
            length_threshold_range=(0.1, 10.0),
            max_iterations_range=(1, 50),
            splice_threshold_range=(0, 180)
        )

        self.assertEqual(bounds.colormode_options, ['color', 'binary'])
        self.assertEqual(bounds.color_precision_range, (1, 10))
        self.assertEqual(bounds.layer_difference_range, (1, 32))


class TestOptimizationResult(unittest.TestCase):
    """Test suite for OptimizationResult dataclass"""

    def test_optimization_result_structure(self):
        """Test OptimizationResult dataclass structure"""
        result = OptimizationResult(
            parameters={'color_precision': 5},
            logo_type='simple',
            confidence=0.8,
            optimization_method='logo_type_simple',
            validation_passed=True,
            adjustments_applied=['confidence_high'],
            optimization_time=0.001
        )

        self.assertEqual(result.parameters, {'color_precision': 5})
        self.assertEqual(result.logo_type, 'simple')
        self.assertEqual(result.confidence, 0.8)
        self.assertTrue(result.validation_passed)
        self.assertIn('confidence_high', result.adjustments_applied)


if __name__ == '__main__':
    unittest.main()