# tests/optimization/test_feature_mapping_optimizer.py
"""Unit tests for Feature Mapping Optimizer - Day 2"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.ai_modules.optimization.feature_mapping_optimizer import FeatureMappingOptimizer
from backend.ai_modules.optimization.parameter_bounds import VTracerParameterBounds


class TestFeatureMappingOptimizer:
    """Test suite for Feature Mapping Optimizer"""

    def setup_method(self):
        """Setup test fixtures"""
        self.optimizer = FeatureMappingOptimizer()
        self.bounds = VTracerParameterBounds()

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
