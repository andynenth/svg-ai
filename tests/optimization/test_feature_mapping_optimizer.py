# tests/optimization/test_feature_mapping_optimizer.py
"""Unit tests for Feature Mapping Optimizer - Day 2"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.ai_modules.optimization import OptimizationEngine
from backend.ai_modules.optimization import OptimizationEngine


class TestFeatureMappingOptimizer:
    """Test suite for Feature Mapping Optimizer"""

    def setup_method(self):
        """Setup test fixtures"""
        self.optimizer = OptimizationEngine()
        self.bounds = OptimizationEngine()

    def test_complete_feature_optimization(self):
        """Test optimization with all required features"""
        features = {
            'edge_density': 0.15,
            'unique_colors': 12,
            'entropy': 0.65,
            'corner_density': 0.08,
            'gradient_strength': 0.45,
            'complexity_score': 0.35
        }

        result = self.optimizer.optimize(features)

        # Check structure
        assert 'parameters' in result
        assert 'confidence' in result
        assert 'metadata' in result

        # Check all parameters are present
        params = result['parameters']
        required_params = [
            'corner_threshold', 'color_precision', 'path_precision',
            'length_threshold', 'max_iterations', 'splice_threshold',
            'layer_difference', 'mode'
        ]
        for param in required_params:
            assert param in params

        # Check confidence is high for complete features
        assert result['confidence'] >= 0.8
        assert result['confidence'] <= 1.0

        # Verify parameters are within bounds
        is_valid, errors = self.bounds.validate_parameter_set(params)
        assert is_valid, f"Parameter validation failed: {errors}"

    def test_partial_feature_optimization(self):
        """Test optimization with missing features"""
        features = {
            'edge_density': 0.25,
            'unique_colors': 64,
            'complexity_score': 0.7
        }

        result = self.optimizer.optimize(features)

        # Should still return valid parameters
        assert 'parameters' in result
        params = result['parameters']

        # Check that defaults are used for missing correlations
        assert params['path_precision'] == self.bounds.get_default_parameters()['path_precision']
        assert params['length_threshold'] == self.bounds.get_default_parameters()['length_threshold']
        assert params['splice_threshold'] == self.bounds.get_default_parameters()['splice_threshold']

        # Confidence should be lower
        assert result['confidence'] < 1.0

    def test_correlation_formulas_applied(self):
        """Test that correlation formulas are correctly applied"""
        features = {
            'edge_density': 0.0,  # Should give corner_threshold = 110
            'unique_colors': 2,  # Should give color_precision = 3
            'entropy': 1.0,  # Should give path_precision = 1
            'corner_density': 0.0,  # Should give length_threshold = 1.0
            'gradient_strength': 0.0,  # Should give splice_threshold = 10
            'complexity_score': 0.0  # Should give max_iterations = 5
        }

        result = self.optimizer.optimize(features)
        params = result['parameters']

        assert params['corner_threshold'] == 110
        assert params['color_precision'] == 3
        assert params['path_precision'] == 1
        assert params['length_threshold'] == 1.0
        assert params['splice_threshold'] == 10
        assert params['max_iterations'] == 5

    def test_mode_selection_logic(self):
        """Test mode selection based on complexity"""
        # Low complexity should give polygon
        features_simple = {'complexity_score': 0.2}
        result_simple = self.optimizer.optimize(features_simple)
        assert result_simple['parameters']['mode'] == 'polygon'

        # High complexity should give spline
        features_complex = {'complexity_score': 0.8}
        result_complex = self.optimizer.optimize(features_complex)
        assert result_complex['parameters']['mode'] == 'spline'

    def test_caching_functionality(self):
        """Test that caching works correctly"""
        features = {
            'edge_density': 0.5,
            'unique_colors': 50,
            'entropy': 0.5,
            'corner_density': 0.5,
            'gradient_strength': 0.5,
            'complexity_score': 0.5
        }

        # First call - should not be cached
        result1 = self.optimizer.optimize(features)
        assert result1['metadata']['cache_hit'] == False

        # Second call with same features - should be cached
        result2 = self.optimizer.optimize(features)
        assert result2['metadata']['cache_hit'] == True

        # Parameters should be identical
        assert result1['parameters'] == result2['parameters']
        assert result1['confidence'] == result2['confidence']