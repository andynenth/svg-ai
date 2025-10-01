"""
Feature Mapping Optimizer V2 - Task 3 Implementation
Uses learned correlations instead of hardcoded formulas.
"""

from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import logging
import json
import hashlib
import numpy as np

# Import learned correlations (new)
from backend.ai_modules.optimization.learned_correlations import LearnedCorrelations

# Import parameter bounds
try:
    from backend.ai_modules.optimization.parameter_bounds import VTracerParameterBounds
except ImportError:
    # Fallback parameter bounds implementation
    class VTracerParameterBounds:
        @staticmethod
        def get_default_parameters():
            return {
                'color_precision': 4,
                'corner_threshold': 30,
                'path_precision': 8,
                'splice_threshold': 45,
                'max_iterations': 10,
                'length_threshold': 5.0
            }


class FeatureMappingOptimizerV2:
    """
    Map image features to optimal VTracer parameters using learned correlations.
    Drop-in replacement for original FeatureMappingOptimizer.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 patterns_path: Optional[str] = None,
                 enable_caching: bool = True,
                 enable_blending: bool = True):
        """
        Initialize Feature Mapping Optimizer V2.

        Args:
            model_path: Path to trained model
            patterns_path: Path to success patterns
            enable_caching: Whether to cache results
            enable_blending: Whether to blend learned and formula results
        """
        # Initialize learned correlations (new approach)
        self.correlations = LearnedCorrelations(
            model_path=model_path,
            patterns_path=patterns_path,
            enable_fallback=True
        )

        # Keep bounds for validation
        self.bounds = VTracerParameterBounds()
        self.logger = logging.getLogger(__name__)

        # Enhanced caching with confidence tracking
        self.enable_caching = enable_caching
        self.cache = {}
        self.max_cache_size = 200
        self.cache_hits = 0
        self.cache_misses = 0

        # Blending configuration
        self.enable_blending = enable_blending

        # Performance monitoring
        self.optimization_count = 0
        self.total_processing_time = 0
        self.confidence_history = []

    def optimize(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate optimized parameters from image features.

        Args:
            features: Dictionary containing image features

        Returns:
            Dictionary containing:
                - parameters: Optimized VTracer parameters
                - confidence: Confidence score (0-1)
                - metadata: Additional optimization information
        """
        self.optimization_count += 1
        start_time = datetime.now()

        # Input validation
        if not isinstance(features, dict):
            self.logger.error(f"Invalid features input: {type(features)}")
            return {
                'parameters': self.bounds.get_default_parameters(),
                'confidence': 0.0,
                'metadata': {
                    'error': f'Invalid features input: expected dict, got {type(features)}',
                    'timestamp': datetime.now().isoformat(),
                    'method': 'error'
                }
            }

        # Check cache
        if self.enable_caching:
            cache_key = self._get_cache_key(features)
            if cache_key in self.cache:
                self.cache_hits += 1
                self.logger.debug(f"Cache hit for features hash {cache_key[:8]}")
                cached_result = self.cache[cache_key].copy()
                cached_result['metadata']['cache_hit'] = True
                return cached_result
            else:
                self.cache_misses += 1

        try:
            # Get parameters using learned correlations
            parameters = self.correlations.get_parameters(features)

            # Calculate confidence based on source
            confidence = self._calculate_confidence(features, parameters)

            # Optionally blend with formula results for medium confidence
            if self.enable_blending and 0.3 < confidence < 0.7:
                parameters = self._blend_parameters(features, parameters, confidence)
                method_used = 'blended'
            else:
                method_used = self._determine_method_used()

            # Validate and bound parameters
            parameters = self._validate_parameters(parameters)

            # Create result
            processing_time = (datetime.now() - start_time).total_seconds()
            self.total_processing_time += processing_time

            result = {
                'parameters': parameters,
                'confidence': confidence,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'optimization_id': self.optimization_count,
                    'processing_time_ms': processing_time * 1000,
                    'method_used': method_used,
                    'cache_hit': False,
                    'features_used': list(features.keys()),
                    'model_available': self.correlations.param_model is not None
                }
            }

            # Update confidence history
            self.confidence_history.append(confidence)
            if len(self.confidence_history) > 100:
                self.confidence_history.pop(0)

            # Cache result
            if self.enable_caching:
                self._cache_result(cache_key, result)

            return result

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return {
                'parameters': self.bounds.get_default_parameters(),
                'confidence': 0.0,
                'metadata': {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'method': 'fallback'
                }
            }

    def _calculate_confidence(self, features: Dict[str, float], parameters: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the optimization.

        Uses multiple factors:
        - Feature completeness
        - Model/pattern availability
        - Historical performance
        - Parameter validity
        """
        confidence_factors = []

        # Feature completeness (how many expected features are present)
        expected_features = [
            'edge_density', 'unique_colors', 'entropy',
            'corner_density', 'gradient_strength', 'complexity_score'
        ]
        feature_coverage = sum(1 for f in expected_features if f in features) / len(expected_features)
        confidence_factors.append(feature_coverage)

        # Method source confidence
        stats = self.correlations.get_usage_statistics()
        if stats['model_used'] > 0:
            confidence_factors.append(0.9)  # High confidence for model
        elif stats['pattern_used'] > 0:
            confidence_factors.append(0.7)  # Medium confidence for patterns
        else:
            confidence_factors.append(0.5)  # Lower confidence for fallback

        # Parameter validity (are all parameters within expected ranges?)
        param_validity = 1.0
        for param_name, param_value in parameters.items():
            if param_name == 'corner_threshold' and not (5 <= param_value <= 110):
                param_validity *= 0.9
            elif param_name == 'color_precision' and not (1 <= param_value <= 20):
                param_validity *= 0.9
        confidence_factors.append(param_validity)

        # Historical performance (if we have history)
        if len(self.confidence_history) > 10:
            avg_historical = np.mean(self.confidence_history[-10:])
            confidence_factors.append(avg_historical)

        # Calculate weighted average
        confidence = np.mean(confidence_factors)

        return float(min(1.0, max(0.0, confidence)))

    def _blend_parameters(self,
                         features: Dict[str, float],
                         learned_params: Dict[str, Any],
                         confidence: float) -> Dict[str, Any]:
        """
        Blend learned parameters with formula-based parameters.

        Higher confidence → more weight to learned parameters
        Lower confidence → more weight to formula parameters
        """
        try:
            # Get formula-based parameters using fallback
            fallback_params = self.correlations.fallback.edge_to_corner_threshold(features.get('edge_density', 0.5))

            # Create full fallback parameter set
            formula_params = {
                'corner_threshold': self.correlations.fallback.edge_to_corner_threshold(
                    features.get('edge_density', 0.5)),
                'color_precision': self.correlations.fallback.colors_to_precision(
                    features.get('unique_colors', 128)),
                'path_precision': self.correlations.fallback.entropy_to_path_precision(
                    features.get('entropy', 0.5)),
                'splice_threshold': self.correlations.fallback.gradient_to_splice_threshold(
                    features.get('gradient_strength', 0.5)),
                'max_iterations': self.correlations.fallback.complexity_to_iterations(
                    features.get('complexity_score', 0.5))
            }

            # Blend based on confidence
            blended = {}
            for param_name in learned_params:
                if param_name in formula_params:
                    learned_value = learned_params[param_name]
                    formula_value = formula_params[param_name]

                    # Weighted average based on confidence
                    if isinstance(learned_value, (int, float)) and isinstance(formula_value, (int, float)):
                        blended_value = (confidence * learned_value) + ((1 - confidence) * formula_value)

                        # Preserve type
                        if isinstance(learned_value, int):
                            blended[param_name] = int(round(blended_value))
                        else:
                            blended[param_name] = float(blended_value)
                    else:
                        # For non-numeric, use learned if confidence > 0.5
                        blended[param_name] = learned_value if confidence > 0.5 else formula_value
                else:
                    blended[param_name] = learned_params[param_name]

            return blended

        except Exception as e:
            self.logger.warning(f"Blending failed: {e}, using learned parameters")
            return learned_params

    def _determine_method_used(self) -> str:
        """Determine which method was used for the latest optimization."""
        stats = self.correlations.get_usage_statistics()

        # Check most recent call
        if stats['total_calls'] > 0:
            if stats['model_used'] > 0:
                return 'learned_model'
            elif stats['pattern_used'] > 0:
                return 'pattern_based'
            elif stats['fallback_used'] > 0:
                return 'formula_fallback'

        return 'unknown'

    def _validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and ensure parameters are within bounds."""
        return self.correlations.validate_parameters(parameters)

    def _get_cache_key(self, features: Dict[str, float]) -> str:
        """Generate cache key from features."""
        # Sort features for consistent hashing
        sorted_features = sorted(features.items())
        feature_str = json.dumps(sorted_features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache optimization result with size management."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = result

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0.0
        avg_processing_time = (
            self.total_processing_time / self.optimization_count * 1000
            if self.optimization_count > 0 else 0
        )

        # Get correlation usage stats
        correlation_stats = self.correlations.get_usage_statistics()

        return {
            'total_optimizations': self.optimization_count,
            'average_confidence': float(avg_confidence),
            'average_processing_time_ms': avg_processing_time,
            'cache_hit_rate': (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0
            ),
            'cache_size': len(self.cache),
            'correlation_usage': correlation_stats,
            'confidence_trend': 'improving' if (
                len(self.confidence_history) > 20 and
                np.mean(self.confidence_history[-10:]) > np.mean(self.confidence_history[-20:-10])
            ) else 'stable'
        }

    def reset_cache(self):
        """Clear the parameter cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("Cache cleared")

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        metrics = self.get_performance_metrics()
        return (f"FeatureMappingOptimizerV2(optimizations={self.optimization_count}, "
                f"avg_confidence={metrics['average_confidence']:.3f}, "
                f"cache_hit_rate={metrics['cache_hit_rate']:.2%})")


def test_feature_mapping_optimizer_v2():
    """Test the Feature Mapping Optimizer V2."""
    print("Testing Feature Mapping Optimizer V2...")

    # Initialize optimizer
    optimizer = FeatureMappingOptimizerV2()
    print(f"✓ Optimizer initialized: {optimizer}")

    # Test with different confidence scenarios
    test_cases = [
        # High confidence case (complete features)
        {
            'edge_density': 0.7,
            'unique_colors': 128,
            'entropy': 0.5,
            'corner_density': 0.3,
            'gradient_strength': 0.6,
            'complexity_score': 0.8
        },
        # Medium confidence case (partial features)
        {
            'edge_density': 0.4,
            'unique_colors': 32,
            'complexity_score': 0.5
        },
        # Low confidence case (minimal features)
        {
            'edge_density': 0.5
        }
    ]

    for i, features in enumerate(test_cases):
        result = optimizer.optimize(features)
        print(f"\n✓ Test case {i+1} ({len(features)} features):")
        print(f"  Parameters: {list(result['parameters'].keys())}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Method: {result['metadata']['method_used']}")
        print(f"  Processing time: {result['metadata']['processing_time_ms']:.2f}ms")

    # Test caching
    cached_result = optimizer.optimize(test_cases[0])
    print(f"\n✓ Cache test: hit={cached_result['metadata'].get('cache_hit', False)}")

    # Show performance metrics
    metrics = optimizer.get_performance_metrics()
    print(f"\n✓ Performance metrics:")
    print(f"  Total optimizations: {metrics['total_optimizations']}")
    print(f"  Average confidence: {metrics['average_confidence']:.3f}")
    print(f"  Cache hit rate: {metrics['cache_hit_rate']:.2%}")
    print(f"  Correlation usage: {metrics['correlation_usage']}")

    return optimizer


if __name__ == "__main__":
    test_feature_mapping_optimizer_v2()