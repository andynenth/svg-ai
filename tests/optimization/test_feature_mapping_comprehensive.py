# tests/optimization/test_feature_mapping_comprehensive.py
"""Comprehensive unit tests for Feature Mapping Optimizer - Day 3"""

import pytest
import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.ai_modules.optimization.feature_mapping_optimizer import FeatureMappingOptimizer
from backend.ai_modules.optimization.parameter_bounds import VTracerParameterBounds


class TestFeatureMappingOptimizer:
    """Comprehensive test suite for Feature Mapping Optimizer"""

    def setup_method(self):
        """Setup test fixtures"""
        self.optimizer = FeatureMappingOptimizer()
        self.bounds = VTracerParameterBounds()

    def test_optimize_with_known_features(self):
        """Test FeatureMappingOptimizer.optimize() with known features"""
        # Test case 1: Complete feature set with known expected outputs
        features = {
            'edge_density': 0.0,      # Should give corner_threshold = 110
            'unique_colors': 2,       # Should give color_precision = 3
            'entropy': 1.0,           # Should give path_precision = 1
            'corner_density': 0.0,    # Should give length_threshold = 1.0
            'gradient_strength': 0.0, # Should give splice_threshold = 10
            'complexity_score': 0.0   # Should give max_iterations = 5
        }

        result = self.optimizer.optimize(features)

        # Verify structure
        assert 'parameters' in result
        assert 'confidence' in result
        assert 'metadata' in result

        # Verify known parameter values
        params = result['parameters']
        assert params['corner_threshold'] == 110
        assert params['color_precision'] == 3
        assert params['path_precision'] == 1
        assert params['length_threshold'] == 1.0
        assert params['splice_threshold'] == 10
        assert params['max_iterations'] == 5

        # Verify defaults are set
        assert params['layer_difference'] == 10
        assert params['mode'] in ['polygon', 'spline']

        # Test case 2: High values
        features_high = {
            'edge_density': 1.0,      # Should give corner_threshold = 10
            'unique_colors': 256,     # Should give color_precision = 10
            'entropy': 0.0,           # Should give path_precision = 20
            'corner_density': 0.19,   # Should give length_threshold = 20.0
            'gradient_strength': 1.0, # Should give splice_threshold = 100
            'complexity_score': 1.0   # Should give max_iterations = 20
        }

        result_high = self.optimizer.optimize(features_high)
        params_high = result_high['parameters']

        assert params_high['corner_threshold'] == 10
        assert params_high['color_precision'] == 10
        assert params_high['path_precision'] == 20
        assert abs(params_high['length_threshold'] - 20.0) < 0.01
        assert params_high['splice_threshold'] == 100
        assert params_high['max_iterations'] == 20

    def test_complete_parameter_set_generation(self):
        """Verify complete parameter set generation (all 8 parameters)"""
        features = {
            'edge_density': 0.15,
            'unique_colors': 12,
            'entropy': 0.65,
            'corner_density': 0.08,
            'gradient_strength': 0.45,
            'complexity_score': 0.35
        }

        result = self.optimizer.optimize(features)
        params = result['parameters']

        # Check all 8 required parameters are present
        required_params = [
            'corner_threshold', 'color_precision', 'path_precision',
            'length_threshold', 'max_iterations', 'splice_threshold',
            'layer_difference', 'mode'
        ]

        for param in required_params:
            assert param in params, f"Missing parameter: {param}"

        # Verify parameter types
        assert isinstance(params['corner_threshold'], int)
        assert isinstance(params['color_precision'], int)
        assert isinstance(params['path_precision'], int)
        assert isinstance(params['length_threshold'], (int, float))
        assert isinstance(params['max_iterations'], int)
        assert isinstance(params['splice_threshold'], int)
        assert isinstance(params['layer_difference'], int)
        assert isinstance(params['mode'], str)

        # Verify bounds compliance
        is_valid, errors = self.bounds.validate_parameter_set(params)
        assert is_valid, f"Generated parameters are invalid: {errors}"

    def test_confidence_calculation_with_various_feature_sets(self):
        """Test confidence calculation with various feature sets"""

        # Test case 1: Complete, well-distributed features (should have high confidence)
        complete_features = {
            'edge_density': 0.2,
            'unique_colors': 20,
            'entropy': 0.5,
            'corner_density': 0.1,
            'gradient_strength': 0.3,
            'complexity_score': 0.4
        }
        result_complete = self.optimizer.optimize(complete_features)
        confidence_complete = result_complete['confidence']
        assert confidence_complete >= 0.8, f"Complete features should have high confidence, got {confidence_complete}"

        # Test case 2: Incomplete features (should have lower confidence)
        incomplete_features = {
            'edge_density': 0.2,
            'unique_colors': 20
        }
        result_incomplete = self.optimizer.optimize(incomplete_features)
        confidence_incomplete = result_incomplete['confidence']
        assert confidence_incomplete < confidence_complete, "Incomplete features should have lower confidence"

        # Test case 3: Extreme values (should penalize confidence)
        extreme_features = {
            'edge_density': 0.99,
            'unique_colors': 1000,
            'entropy': 0.01,
            'corner_density': 0.99,
            'gradient_strength': 0.99,
            'complexity_score': 0.99
        }
        result_extreme = self.optimizer.optimize(extreme_features)
        confidence_extreme = result_extreme['confidence']
        assert confidence_extreme < confidence_complete, "Extreme features should have lower confidence"

        # Test case 4: All same values (poor distribution should lower confidence)
        uniform_features = {
            'edge_density': 0.5,
            'unique_colors': 128,  # Normalized to ~0.5
            'entropy': 0.5,
            'corner_density': 0.5,
            'gradient_strength': 0.5,
            'complexity_score': 0.5
        }
        result_uniform = self.optimizer.optimize(uniform_features)
        confidence_uniform = result_uniform['confidence']
        assert confidence_uniform < confidence_complete, "Uniform features should have lower confidence"

        # Verify confidence is always in valid range
        all_results = [result_complete, result_incomplete, result_extreme, result_uniform]
        for result in all_results:
            assert 0.0 <= result['confidence'] <= 1.0, f"Confidence {result['confidence']} is outside [0, 1] range"

    def test_optimization_metadata_generation(self):
        """Test optimization metadata generation"""
        features = {
            'edge_density': 0.15,
            'unique_colors': 12,
            'entropy': 0.65,
            'corner_density': 0.08,
            'gradient_strength': 0.45,
            'complexity_score': 0.35
        }

        result = self.optimizer.optimize(features)
        metadata = result['metadata']

        # Check required metadata fields
        required_fields = [
            'timestamp', 'optimization_number', 'correlation_log',
            'optimization_time_seconds', 'cache_hit', 'confidence_explanation',
            'parameter_explanations', 'correlations_used'
        ]

        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

        # Verify metadata content quality
        assert isinstance(metadata['timestamp'], str)
        assert isinstance(metadata['optimization_number'], int)
        assert isinstance(metadata['correlation_log'], list)
        assert len(metadata['correlation_log']) > 0
        assert isinstance(metadata['optimization_time_seconds'], (int, float))
        assert isinstance(metadata['cache_hit'], bool)
        assert isinstance(metadata['confidence_explanation'], str)
        assert isinstance(metadata['parameter_explanations'], dict)
        assert isinstance(metadata['correlations_used'], list)

        # Verify correlation log entries
        for log_entry in metadata['correlation_log']:
            assert isinstance(log_entry, str)
            assert '→' in log_entry or 'default' in log_entry.lower() or 'mode:' in log_entry.lower()

        # Verify parameter explanations
        for param, explanation in metadata['parameter_explanations'].items():
            assert isinstance(explanation, str)
            assert len(explanation) > 10, f"Explanation for {param} is too short: {explanation}"

    def test_caching_functionality_with_repeated_features(self):
        """Test caching functionality with repeated features"""
        features = {
            'edge_density': 0.25,
            'unique_colors': 50,
            'entropy': 0.6,
            'corner_density': 0.12,
            'gradient_strength': 0.7,
            'complexity_score': 0.8
        }

        # First optimization - should not be cached
        result1 = self.optimizer.optimize(features)
        assert result1['metadata']['cache_hit'] == False

        # Second optimization with same features - should be cached
        result2 = self.optimizer.optimize(features)
        assert result2['metadata']['cache_hit'] == True

        # Results should be identical (except for cache_hit flag)
        assert result1['parameters'] == result2['parameters']
        assert result1['confidence'] == result2['confidence']

        # Third optimization with slightly different features - should not be cached
        features_different = features.copy()
        features_different['edge_density'] = 0.26
        result3 = self.optimizer.optimize(features_different)
        assert result3['metadata']['cache_hit'] == False

        # Test cache size limit
        original_limit = self.optimizer.max_cache_size
        self.optimizer.max_cache_size = 2

        # Add more entries than cache limit
        for i in range(5):
            test_features = {'complexity_score': i * 0.1}
            self.optimizer.optimize(test_features)

        assert len(self.optimizer.cache) <= 2

        # Restore original limit
        self.optimizer.max_cache_size = original_limit

    def test_error_handling_with_invalid_features(self):
        """Test error handling with invalid features"""

        # Test with None as features
        result_none = self.optimizer.optimize(None)
        assert result_none['confidence'] == 0.0
        assert 'error' in result_none['metadata']
        assert result_none['parameters'] == self.bounds.get_default_parameters()

        # Test with string instead of dict
        result_string = self.optimizer.optimize("invalid")
        assert result_string['confidence'] == 0.0
        assert 'error' in result_string['metadata']

        # Test with empty dict
        result_empty = self.optimizer.optimize({})
        assert 'parameters' in result_empty
        assert result_empty['confidence'] >= 0.0  # Should still work with defaults

        # Test with features containing invalid values
        invalid_features = {
            'edge_density': "not_a_number",
            'unique_colors': None,
            'entropy': float('inf'),
            'corner_density': -999,
            'gradient_strength': 999,
            'complexity_score': "invalid"
        }
        result_invalid = self.optimizer.optimize(invalid_features)
        # Should handle gracefully and return valid parameters
        is_valid, errors = self.bounds.validate_parameter_set(result_invalid['parameters'])
        assert is_valid, f"Invalid features resulted in invalid parameters: {errors}"

    def test_parameter_explanation_generation(self):
        """Test parameter explanation generation"""

        # Test with features that should generate specific explanations
        features = {
            'edge_density': 0.05,     # Low - should result in "smooth" or "high" explanation
            'unique_colors': 2,       # Very low - should result in "simple" explanation
            'entropy': 0.9,           # High - should result in "random" or "low precision" explanation
            'corner_density': 0.02,   # Low - should result in "minimal corners" explanation
            'gradient_strength': 0.8, # High - should result in "strong gradients" explanation
            'complexity_score': 0.1   # Low - should result in "simple image" explanation
        }

        result = self.optimizer.optimize(features)
        explanations = result['metadata']['parameter_explanations']

        # Verify explanations exist and are meaningful
        assert 'corner_threshold' in explanations
        assert 'color_precision' in explanations
        assert 'path_precision' in explanations
        assert 'length_threshold' in explanations
        assert 'splice_threshold' in explanations
        assert 'max_iterations' in explanations

        # Check that explanations reflect the input characteristics
        corner_explanation = explanations['corner_threshold'].lower()
        assert 'smooth' in corner_explanation or 'high' in corner_explanation

        color_explanation = explanations['color_precision'].lower()
        assert 'simple' in color_explanation or 'low' in color_explanation

        path_explanation = explanations['path_precision'].lower()
        assert 'random' in path_explanation or 'low' in path_explanation

        splice_explanation = explanations['splice_threshold'].lower()
        assert 'strong' in splice_explanation or 'many' in splice_explanation

        iteration_explanation = explanations['max_iterations'].lower()
        assert 'simple' in iteration_explanation or 'few' in iteration_explanation

    def test_parameters_pass_bounds_validation(self):
        """Verify all parameters pass bounds validation"""

        # Test with multiple different feature combinations
        test_cases = [
            # Minimum values
            {
                'edge_density': 0.0,
                'unique_colors': 1,
                'entropy': 0.0,
                'corner_density': 0.0,
                'gradient_strength': 0.0,
                'complexity_score': 0.0
            },
            # Maximum values
            {
                'edge_density': 1.0,
                'unique_colors': 1000,
                'entropy': 1.0,
                'corner_density': 1.0,
                'gradient_strength': 1.0,
                'complexity_score': 1.0
            },
            # Random combinations
            {
                'edge_density': 0.33,
                'unique_colors': 77,
                'entropy': 0.66,
                'corner_density': 0.22,
                'gradient_strength': 0.88,
                'complexity_score': 0.44
            },
            # Partial features
            {
                'edge_density': 0.5,
                'complexity_score': 0.7
            }
        ]

        for i, features in enumerate(test_cases):
            result = self.optimizer.optimize(features)
            params = result['parameters']

            is_valid, errors = self.bounds.validate_parameter_set(params)
            assert is_valid, f"Test case {i} produced invalid parameters: {errors}"

    def test_with_extreme_feature_values(self):
        """Test with extreme feature values"""

        # Test with values way outside normal range
        extreme_features = {
            'edge_density': 10.0,     # Way above 1.0
            'unique_colors': -50,     # Negative
            'entropy': -5.0,          # Negative
            'corner_density': 999,    # Way above 1.0
            'gradient_strength': -10, # Negative
            'complexity_score': 100   # Way above 1.0
        }

        result = self.optimizer.optimize(extreme_features)
        params = result['parameters']

        # Should still produce valid parameters due to clamping/bounds checking
        is_valid, errors = self.bounds.validate_parameter_set(params)
        assert is_valid, f"Extreme features produced invalid parameters: {errors}"

        # Confidence should be penalized
        assert result['confidence'] < 0.8, f"Extreme features should have low confidence, got {result['confidence']}"

    def test_with_real_image_features(self):
        """Test with real image features from test dataset"""

        # Simulate realistic features from different logo types
        real_feature_sets = {
            'simple_circle': {
                'edge_density': 0.05,
                'unique_colors': 3,
                'entropy': 0.2,
                'corner_density': 0.02,
                'gradient_strength': 0.0,
                'complexity_score': 0.1
            },
            'text_logo': {
                'edge_density': 0.2,
                'unique_colors': 2,
                'entropy': 0.4,
                'corner_density': 0.15,
                'gradient_strength': 0.0,
                'complexity_score': 0.3
            },
            'gradient_logo': {
                'edge_density': 0.1,
                'unique_colors': 128,
                'entropy': 0.7,
                'corner_density': 0.05,
                'gradient_strength': 0.9,
                'complexity_score': 0.6
            },
            'complex_illustration': {
                'edge_density': 0.35,
                'unique_colors': 256,
                'entropy': 0.85,
                'corner_density': 0.4,
                'gradient_strength': 0.6,
                'complexity_score': 0.9
            }
        }

        for logo_type, features in real_feature_sets.items():
            result = self.optimizer.optimize(features)
            params = result['parameters']

            # Verify valid parameters
            is_valid, errors = self.bounds.validate_parameter_set(params)
            assert is_valid, f"{logo_type} features produced invalid parameters: {errors}"

            # Verify reasonable confidence for realistic features
            assert result['confidence'] >= 0.5, f"{logo_type} should have reasonable confidence, got {result['confidence']}"

            # Verify mode selection makes sense
            if features['complexity_score'] < 0.3:
                assert params['mode'] == 'polygon', f"Simple {logo_type} should use polygon mode"
            else:
                assert params['mode'] == 'spline', f"Complex {logo_type} should use spline mode"

    def test_optimization_consistency(self):
        """Test that optimization is consistent and deterministic"""

        features = {
            'edge_density': 0.25,
            'unique_colors': 16,
            'entropy': 0.55,
            'corner_density': 0.1,
            'gradient_strength': 0.4,
            'complexity_score': 0.6
        }

        # Clear cache to ensure fresh optimizations
        self.optimizer.cache.clear()

        # Run optimization multiple times
        results = []
        for _ in range(3):
            self.optimizer.cache.clear()  # Clear cache between runs
            result = self.optimizer.optimize(features)
            results.append(result)

        # All results should have identical parameters and confidence
        for i in range(1, len(results)):
            assert results[i]['parameters'] == results[0]['parameters'], f"Run {i} parameters differ from run 0"
            assert results[i]['confidence'] == results[0]['confidence'], f"Run {i} confidence differs from run 0"

    def test_performance_requirements(self):
        """Test that optimization meets performance requirements"""
        import time

        features = {
            'edge_density': 0.15,
            'unique_colors': 12,
            'entropy': 0.65,
            'corner_density': 0.08,
            'gradient_strength': 0.45,
            'complexity_score': 0.35
        }

        # Clear cache to ensure full optimization
        self.optimizer.cache.clear()

        # Measure optimization time
        start_time = time.time()
        result = self.optimizer.optimize(features)
        end_time = time.time()

        optimization_time = end_time - start_time

        # Should complete in <0.1s as per Day 3 requirements
        assert optimization_time < 0.1, f"Optimization took {optimization_time:.3f}s, should be <0.1s"

        # Verify the reported time in metadata is also reasonable
        reported_time = result['metadata']['optimization_time_seconds']
        assert reported_time < 0.1, f"Reported optimization time {reported_time:.3f}s should be <0.1s"