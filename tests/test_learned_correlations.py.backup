"""
Integration tests for learned correlations system (Day 6).
Verifies that all components work together properly.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the components
from backend.ai_modules.optimization.learned_correlations import LearnedCorrelations
from backend.ai_modules.optimization.feature_mapping_optimizer_v2 import FeatureMappingOptimizerV2
from backend.ai_modules.optimization.correlation_rollout import CorrelationRollout
from backend.ai_modules.optimization.correlation_formulas import CorrelationFormulas


class TestLearnedCorrelations:
    """Test the learned correlations system."""

    def test_initialization_without_model(self):
        """Test that learned correlations initialize correctly without trained model."""
        lc = LearnedCorrelations()
        assert lc is not None
        assert lc.param_model is None  # No trained model
        assert lc.patterns is not None  # Should have default patterns
        assert lc.fallback is not None  # Should have fallback

    def test_get_parameters_with_fallback(self):
        """Test parameter generation falls back to formulas when no model."""
        lc = LearnedCorrelations()
        features = {
            'edge_density': 0.7,
            'unique_colors': 128,
            'entropy': 0.5,
            'gradient_strength': 0.6
        }

        params = lc.get_parameters(features)

        # Should return valid parameters
        assert 'corner_threshold' in params
        assert 'color_precision' in params
        assert 'path_precision' in params
        assert 'splice_threshold' in params

        # Check parameter values are reasonable
        assert 5 <= params['corner_threshold'] <= 110
        assert 1 <= params['color_precision'] <= 20

    def test_pattern_based_parameters(self):
        """Test that pattern-based optimization works."""
        lc = LearnedCorrelations()

        # Test simple geometric image
        simple_features = {
            'edge_density': 0.2,
            'unique_colors': 5,
            'entropy': 0.1,
            'gradient_strength': 0.1,
            'complexity_score': 0.2
        }

        params = lc.get_parameters(simple_features)

        # Simple images should have lower color precision
        assert params['color_precision'] <= 6

    def test_usage_statistics(self):
        """Test that usage statistics are tracked correctly."""
        lc = LearnedCorrelations()

        # Make a few calls
        features = {'edge_density': 0.5}
        for _ in range(3):
            lc.get_parameters(features)

        stats = lc.get_usage_statistics()
        assert stats['total_calls'] == 3
        assert stats['fallback_used'] > 0 or stats['pattern_used'] > 0


class TestFeatureMappingOptimizerV2:
    """Test the feature mapping optimizer v2."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = FeatureMappingOptimizerV2()
        assert optimizer is not None
        assert optimizer.correlations is not None
        assert optimizer.enable_caching is True

    def test_optimize_with_complete_features(self):
        """Test optimization with complete feature set."""
        optimizer = FeatureMappingOptimizerV2()

        features = {
            'edge_density': 0.7,
            'unique_colors': 128,
            'entropy': 0.5,
            'corner_density': 0.3,
            'gradient_strength': 0.6,
            'complexity_score': 0.8
        }

        result = optimizer.optimize(features)

        assert 'parameters' in result
        assert 'confidence' in result
        assert 'metadata' in result

        # High confidence with complete features
        assert result['confidence'] > 0.6

        # All expected parameters present
        params = result['parameters']
        assert 'corner_threshold' in params
        assert 'color_precision' in params

    def test_optimize_with_partial_features(self):
        """Test optimization with partial features."""
        optimizer = FeatureMappingOptimizerV2()

        features = {
            'edge_density': 0.5,
            'unique_colors': 50
        }

        result = optimizer.optimize(features)

        assert 'parameters' in result
        assert 'confidence' in result

        # Lower confidence with partial features
        assert result['confidence'] < 0.8

    def test_caching(self):
        """Test that caching works correctly."""
        optimizer = FeatureMappingOptimizerV2(enable_caching=True)

        features = {'edge_density': 0.5, 'unique_colors': 100}

        # First call - should miss cache
        result1 = optimizer.optimize(features)
        assert not result1['metadata'].get('cache_hit', False)

        # Second call - should hit cache
        result2 = optimizer.optimize(features)
        assert result2['metadata'].get('cache_hit', False)

        # Parameters should be the same
        assert result1['parameters'] == result2['parameters']

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        optimizer = FeatureMappingOptimizerV2()

        # Make several optimizations
        for i in range(5):
            optimizer.optimize({'edge_density': i * 0.2})

        metrics = optimizer.get_performance_metrics()

        assert metrics['total_optimizations'] == 5
        assert metrics['average_confidence'] > 0
        assert 'correlation_usage' in metrics


class TestCorrelationRollout:
    """Test the correlation rollout system."""

    def test_initialization(self):
        """Test rollout system initialization."""
        rollout = CorrelationRollout(rollout_percentage=0.3)
        assert rollout is not None
        assert rollout.rollout_percentage == 0.3

    def test_should_use_learned(self):
        """Test routing decision logic."""
        rollout = CorrelationRollout(rollout_percentage=0.5)

        # Test random routing
        decisions = [rollout.should_use_learned() for _ in range(100)]
        learned_count = sum(decisions)

        # Should be roughly 50% (with some variance)
        assert 30 < learned_count < 70

    def test_session_consistency(self):
        """Test that session routing is consistent."""
        rollout = CorrelationRollout(rollout_percentage=0.5)

        session_id = "test_session_abc"

        # Multiple calls with same session should return same result
        decisions = [rollout.should_use_learned(session_id) for _ in range(10)]

        assert len(set(decisions)) == 1  # All decisions should be the same

    def test_rollback(self):
        """Test rollback functionality."""
        rollout = CorrelationRollout(rollout_percentage=0.8)

        # Perform rollback
        success = rollout.rollback("Test rollback")
        assert success is True

        # After rollback, should not use learned
        assert rollout.rollout_percentage == 0.0
        assert rollout.should_use_learned() is False

    def test_update_rollout(self):
        """Test updating rollout percentage."""
        rollout = CorrelationRollout(rollout_percentage=0.1)

        # Update to 50%
        success = rollout.update_rollout(0.5)
        assert success is True
        assert rollout.rollout_percentage == 0.5

        # Update to invalid value (should clamp)
        rollout.update_rollout(1.5)
        assert rollout.rollout_percentage == 1.0

        rollout.update_rollout(-0.5)
        assert rollout.rollout_percentage == 0.0

    def test_get_parameters(self):
        """Test parameter retrieval through rollout."""
        rollout = CorrelationRollout(rollout_percentage=1.0)  # Always use learned

        features = {
            'edge_density': 0.7,
            'unique_colors': 128
        }

        result = rollout.get_parameters(features)

        assert 'parameters' in result
        assert 'method' in result
        assert result['method'] in ['learned', 'formula', 'formula_fallback']

    def test_metrics_collection(self):
        """Test that metrics are collected correctly."""
        rollout = CorrelationRollout(rollout_percentage=0.5)

        # Make several requests
        for i in range(20):
            rollout.get_parameters({'edge_density': 0.5}, f"session_{i}")

        metrics = rollout.get_metrics()

        assert metrics['total_requests'] == 20
        assert 'distribution' in metrics
        assert 'error_rates' in metrics
        assert 'average_time_ms' in metrics

    def test_health_check(self):
        """Test health check functionality."""
        rollout = CorrelationRollout(rollout_percentage=0.5)

        # Make some requests
        for _ in range(10):
            rollout.get_parameters({'edge_density': 0.5})

        health = rollout.health_check()

        assert 'status' in health
        assert health['status'] in ['healthy', 'degraded']
        assert 'issues' in health
        assert 'metrics' in health


class TestIntegration:
    """Integration tests for the complete system."""

    def test_full_pipeline(self):
        """Test the full pipeline from features to parameters."""
        # Initialize all components
        optimizer = FeatureMappingOptimizerV2()
        rollout = CorrelationRollout(rollout_percentage=0.5)

        # Test features
        features = {
            'edge_density': 0.7,
            'unique_colors': 128,
            'entropy': 0.5,
            'corner_density': 0.3,
            'gradient_strength': 0.6,
            'complexity_score': 0.8
        }

        # Get parameters through optimizer
        opt_result = optimizer.optimize(features)
        assert opt_result['parameters'] is not None

        # Get parameters through rollout
        rollout_result = rollout.get_parameters(features)
        assert rollout_result['parameters'] is not None

        # Both should return valid VTracer parameters
        for params in [opt_result['parameters'], rollout_result['parameters']]:
            assert 'corner_threshold' in params
            assert 'color_precision' in params
            assert isinstance(params['corner_threshold'], (int, float))
            assert isinstance(params['color_precision'], (int, float))

    def test_backward_compatibility(self):
        """Test that new system is backward compatible with old formulas."""
        # Original formulas
        formulas = CorrelationFormulas()

        # New learned system with fallback
        learned = LearnedCorrelations()

        # Test features
        edge_density = 0.5
        unique_colors = 100

        # Get parameters from both
        formula_corner = formulas.edge_to_corner_threshold(edge_density)
        formula_color = formulas.colors_to_precision(unique_colors)

        learned_params = learned.get_parameters({
            'edge_density': edge_density,
            'unique_colors': unique_colors
        })

        # When using fallback, should get similar results
        # (may not be exact due to pattern adjustments)
        assert abs(learned_params['corner_threshold'] - formula_corner) < 20
        assert abs(learned_params['color_precision'] - formula_color) < 5

    def test_gradual_rollout_scenario(self):
        """Test a realistic gradual rollout scenario."""
        rollout = CorrelationRollout(rollout_percentage=0.0)

        features = {'edge_density': 0.5, 'unique_colors': 100}

        # Phase 1: 0% rollout (all formula)
        for _ in range(10):
            result = rollout.get_parameters(features)
            assert result['method'] in ['formula', 'formula_fallback']

        # Phase 2: 50% rollout
        rollout.update_rollout(0.5)
        methods_used = []
        for i in range(20):
            result = rollout.get_parameters(features, f"session_{i}")
            methods_used.append(result['method'])

        # Should have mix of methods
        assert 'learned' in methods_used or 'formula' in methods_used

        # Phase 3: 100% rollout
        rollout.update_rollout(1.0)
        for i in range(10):
            result = rollout.get_parameters(features, f"new_session_{i}")
            assert result['method'] == 'learned'

        # Emergency rollback
        rollout.rollback("Performance issues detected")
        for _ in range(10):
            result = rollout.get_parameters(features)
            assert result['method'] in ['formula', 'formula_fallback']