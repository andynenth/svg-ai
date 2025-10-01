# backend/ai_modules/optimization/feature_mapping_optimizer.py
"""Feature mapping optimizer using correlation formulas - Day 2 Implementation"""

from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import logging
import json
import hashlib

try:
    from .correlation_formulas import CorrelationFormulas
    from .parameter_bounds import VTracerParameterBounds
except ImportError:
    # For standalone testing
    from correlation_formulas import CorrelationFormulas
    from parameter_bounds import VTracerParameterBounds

logger = logging.getLogger(__name__)


class FeatureMappingOptimizer:
    """Map image features to optimal VTracer parameters using correlations"""

    def __init__(self):
        self.formulas = CorrelationFormulas()
        self.bounds = VTracerParameterBounds()
        self.logger = logging.getLogger(__name__)

        # Basic caching for repeated feature sets
        self.cache = {}
        self.max_cache_size = 100

        # Optimization counter for metadata
        self.optimization_count = 0

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

        # Handle invalid inputs early
        if not isinstance(features, dict):
            self.logger.error(f"Invalid features input: {type(features)}")
            return {
                'parameters': self.bounds.get_default_parameters(),
                'confidence': 0.0,
                'metadata': {
                    'error': f'Invalid features input: expected dict, got {type(features)}',
                    'timestamp': datetime.now().isoformat()
                }
            }

        # Check cache first
        cache_key = self._get_cache_key(features)
        if cache_key in self.cache:
            self.logger.debug(f"Cache hit for features hash {cache_key[:8]}")
            cached_result = self.cache[cache_key].copy()
            cached_result['metadata']['cache_hit'] = True
            return cached_result

        try:
            # Apply all 6 correlation formulas in sequence
            parameters = {}
            correlation_log = []

            # 1. edge_density → corner_threshold
            if 'edge_density' in features:
                value = self.formulas.edge_to_corner_threshold(features['edge_density'])
                parameters['corner_threshold'] = value
                correlation_log.append(f"edge_density({features['edge_density']:.3f}) → corner_threshold({value})")
            else:
                parameters['corner_threshold'] = self.bounds.get_default_parameters()['corner_threshold']
                correlation_log.append("corner_threshold: using default (edge_density missing)")

            # 2. unique_colors → color_precision
            if 'unique_colors' in features:
                value = self.formulas.colors_to_precision(features['unique_colors'])
                parameters['color_precision'] = value
                correlation_log.append(f"unique_colors({features['unique_colors']}) → color_precision({value})")
            else:
                parameters['color_precision'] = self.bounds.get_default_parameters()['color_precision']
                correlation_log.append("color_precision: using default (unique_colors missing)")

            # 3. entropy → path_precision
            if 'entropy' in features:
                value = self.formulas.entropy_to_path_precision(features['entropy'])
                parameters['path_precision'] = value
                correlation_log.append(f"entropy({features['entropy']:.3f}) → path_precision({value})")
            else:
                parameters['path_precision'] = self.bounds.get_default_parameters()['path_precision']
                correlation_log.append("path_precision: using default (entropy missing)")

            # 4. corner_density → length_threshold
            if 'corner_density' in features:
                value = self.formulas.corners_to_length_threshold(features['corner_density'])
                parameters['length_threshold'] = value
                correlation_log.append(f"corner_density({features['corner_density']:.3f}) → length_threshold({value:.2f})")
            else:
                parameters['length_threshold'] = self.bounds.get_default_parameters()['length_threshold']
                correlation_log.append("length_threshold: using default (corner_density missing)")

            # 5. gradient_strength → splice_threshold
            if 'gradient_strength' in features:
                value = self.formulas.gradient_to_splice_threshold(features['gradient_strength'])
                parameters['splice_threshold'] = value
                correlation_log.append(f"gradient_strength({features['gradient_strength']:.3f}) → splice_threshold({value})")
            else:
                parameters['splice_threshold'] = self.bounds.get_default_parameters()['splice_threshold']
                correlation_log.append("splice_threshold: using default (gradient_strength missing)")

            # 6. complexity_score → max_iterations
            if 'complexity_score' in features:
                value = self.formulas.complexity_to_iterations(features['complexity_score'])
                parameters['max_iterations'] = value
                correlation_log.append(f"complexity_score({features['complexity_score']:.3f}) → max_iterations({value})")
            else:
                parameters['max_iterations'] = self.bounds.get_default_parameters()['max_iterations']
                correlation_log.append("max_iterations: using default (complexity_score missing)")

            # Set default values for non-correlated parameters
            # layer_difference: use default (10)
            parameters['layer_difference'] = self.bounds.get_default_parameters()['layer_difference']
            correlation_log.append(f"layer_difference: using default ({parameters['layer_difference']})")

            # mode: choose based on complexity ('spline' for complex, 'polygon' for simple)
            complexity = features.get('complexity_score', 0.5)
            if complexity < 0.3:
                parameters['mode'] = 'polygon'
                correlation_log.append(f"mode: 'polygon' (low complexity: {complexity:.2f})")
            else:
                parameters['mode'] = 'spline'
                correlation_log.append(f"mode: 'spline' (complexity: {complexity:.2f})")

            # Validate all parameters are within bounds
            is_valid, errors = self.bounds.validate_parameter_set(parameters)
            if not is_valid:
                self.logger.warning(f"Parameter validation errors: {errors}")
                # Apply bounds clipping to fix issues
                for param_name in parameters:
                    if param_name != 'mode':
                        parameters[param_name] = self.bounds.clip_to_bounds(param_name, parameters[param_name])

            # Calculate confidence score
            confidence = self.calculate_confidence(features)

            # Generate optimization metadata
            optimization_time = (datetime.now() - start_time).total_seconds()
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'optimization_number': self.optimization_count,
                'correlation_log': correlation_log,
                'optimization_time_seconds': optimization_time,
                'cache_hit': False,
                'confidence_explanation': self._explain_confidence(features, confidence),
                'parameter_explanations': self._explain_parameters(parameters, features),
                'correlations_used': [
                    'edge_density → corner_threshold',
                    'unique_colors → color_precision',
                    'entropy → path_precision',
                    'corner_density → length_threshold',
                    'gradient_strength → splice_threshold',
                    'complexity_score → max_iterations'
                ]
            }

            result = {
                'parameters': parameters,
                'confidence': confidence,
                'metadata': metadata
            }

            # Cache the result
            self._update_cache(cache_key, result)

            self.logger.info(f"Optimization complete: confidence={confidence:.2f}, time={optimization_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            # Return defaults on error
            return {
                'parameters': self.bounds.get_default_parameters(),
                'confidence': 0.0,
                'metadata': {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }

    def calculate_confidence(self, features: Dict[str, float]) -> float:
        """
        Calculate confidence score for optimization.

        Args:
            features: Dictionary of image features

        Returns:
            Confidence score between 0 and 1
        """
        confidence_factors = []

        # Check feature completeness (all 6 key features present)
        required_features = [
            'edge_density', 'unique_colors', 'entropy',
            'corner_density', 'gradient_strength', 'complexity_score'
        ]
        features_present = sum(1 for f in required_features if f in features)
        completeness_score = features_present / len(required_features)
        confidence_factors.append(('completeness', completeness_score, 0.3))

        # Check for extreme values (penalize outliers)
        extreme_penalty = 0.0
        for feature, value in features.items():
            if feature in ['edge_density', 'entropy', 'corner_density', 'gradient_strength', 'complexity_score']:
                # These should be in [0, 1]
                if value < 0.0 or value > 1.0:
                    extreme_penalty += 0.1
                elif value < 0.05 or value > 0.95:
                    extreme_penalty += 0.05
            elif feature == 'unique_colors':
                # Check for reasonable color count
                if value < 1 or value > 1000:
                    extreme_penalty += 0.1

        extreme_score = max(0.0, 1.0 - extreme_penalty)
        confidence_factors.append(('no_extremes', extreme_score, 0.3))

        # Check for well-distributed features (not all the same)
        if len(features) > 1:
            normalized_values = []
            for feature, value in features.items():
                if feature in ['edge_density', 'entropy', 'corner_density', 'gradient_strength', 'complexity_score']:
                    normalized_values.append(value)
                elif feature == 'unique_colors':
                    # Normalize color count to [0, 1] range (log scale)
                    import math
                    normalized = min(1.0, math.log(max(1, value)) / math.log(256))
                    normalized_values.append(normalized)

            if normalized_values:
                # Calculate standard deviation as measure of distribution
                mean = sum(normalized_values) / len(normalized_values)
                variance = sum((x - mean) ** 2 for x in normalized_values) / len(normalized_values)
                std_dev = variance ** 0.5
                # Good distribution has std_dev around 0.2-0.4
                if 0.15 <= std_dev <= 0.45:
                    distribution_score = 1.0
                elif std_dev < 0.15:
                    distribution_score = std_dev / 0.15
                else:
                    distribution_score = max(0.3, 1.0 - (std_dev - 0.45) * 2)
            else:
                distribution_score = 0.5
        else:
            distribution_score = 0.3

        confidence_factors.append(('distribution', distribution_score, 0.4))

        # Calculate weighted confidence
        total_confidence = 0.0
        total_weight = 0.0
        for name, score, weight in confidence_factors:
            total_confidence += score * weight
            total_weight += weight
            self.logger.debug(f"Confidence factor '{name}': {score:.2f} (weight: {weight})")

        if total_weight > 0:
            final_confidence = total_confidence / total_weight
        else:
            final_confidence = 0.5

        # Ensure confidence is in [0, 1]
        final_confidence = max(0.0, min(1.0, final_confidence))

        return final_confidence

    def _explain_confidence(self, features: Dict[str, float], confidence: float) -> str:
        """Generate human-readable explanation for confidence score."""
        explanations = []

        # Feature completeness
        required = ['edge_density', 'unique_colors', 'entropy', 'corner_density', 'gradient_strength', 'complexity_score']
        missing = [f for f in required if f not in features]
        if missing:
            explanations.append(f"Missing features: {', '.join(missing)}")
        else:
            explanations.append("All required features present")

        # Confidence level interpretation
        if confidence >= 0.8:
            explanations.append("High confidence: Well-balanced features with good coverage")
        elif confidence >= 0.6:
            explanations.append("Moderate confidence: Most features within expected ranges")
        elif confidence >= 0.4:
            explanations.append("Low confidence: Some features missing or extreme")
        else:
            explanations.append("Very low confidence: Many features missing or invalid")

        return ". ".join(explanations)

    def _explain_parameters(self, parameters: Dict[str, Any], features: Dict[str, float]) -> Dict[str, str]:
        """Generate human-readable explanations for each parameter choice."""
        explanations = {}

        # Corner threshold
        if 'edge_density' in features:
            edge_val = features['edge_density']
            if edge_val < 0.1:
                explanations['corner_threshold'] = f"High value ({parameters.get('corner_threshold')}) for smooth edges (density={edge_val:.2f})"
            elif edge_val > 0.3:
                explanations['corner_threshold'] = f"Low value ({parameters.get('corner_threshold')}) for detailed edges (density={edge_val:.2f})"
            else:
                explanations['corner_threshold'] = f"Moderate value ({parameters.get('corner_threshold')}) for balanced edge detail"

        # Color precision
        if 'unique_colors' in features:
            colors = features['unique_colors']
            if colors <= 4:
                explanations['color_precision'] = f"Low precision ({parameters.get('color_precision')}) for simple palette ({colors} colors)"
            elif colors >= 100:
                explanations['color_precision'] = f"High precision ({parameters.get('color_precision')}) for complex palette ({colors} colors)"
            else:
                explanations['color_precision'] = f"Moderate precision ({parameters.get('color_precision')}) for {colors} colors"

        # Path precision
        if 'entropy' in features:
            entropy = features['entropy']
            if entropy < 0.3:
                explanations['path_precision'] = f"High precision ({parameters.get('path_precision')}) for organized patterns (entropy={entropy:.2f})"
            elif entropy > 0.7:
                explanations['path_precision'] = f"Low precision ({parameters.get('path_precision')}) for random patterns (entropy={entropy:.2f})"
            else:
                explanations['path_precision'] = f"Balanced precision ({parameters.get('path_precision')}) for moderate complexity"

        # Length threshold
        if 'corner_density' in features:
            corners = features['corner_density']
            if corners < 0.05:
                explanations['length_threshold'] = f"Short segments ({parameters.get('length_threshold'):.1f}) for minimal corners"
            elif corners > 0.15:
                explanations['length_threshold'] = f"Long segments ({parameters.get('length_threshold'):.1f}) for many corners"
            else:
                explanations['length_threshold'] = f"Moderate segments ({parameters.get('length_threshold'):.1f}) for balanced corners"

        # Splice threshold
        if 'gradient_strength' in features:
            gradient = features['gradient_strength']
            if gradient < 0.2:
                explanations['splice_threshold'] = f"Few splice points ({parameters.get('splice_threshold')}) for weak gradients"
            elif gradient > 0.6:
                explanations['splice_threshold'] = f"Many splice points ({parameters.get('splice_threshold')}) for strong gradients"
            else:
                explanations['splice_threshold'] = f"Moderate splicing ({parameters.get('splice_threshold')}) for medium gradients"

        # Max iterations
        if 'complexity_score' in features:
            complexity = features['complexity_score']
            if complexity < 0.3:
                explanations['max_iterations'] = f"Few iterations ({parameters.get('max_iterations')}) for simple image"
            elif complexity > 0.7:
                explanations['max_iterations'] = f"Many iterations ({parameters.get('max_iterations')}) for complex image"
            else:
                explanations['max_iterations'] = f"Moderate iterations ({parameters.get('max_iterations')}) for medium complexity"

        # Mode
        explanations['mode'] = f"Using '{parameters.get('mode')}' mode based on complexity"

        # Layer difference
        explanations['layer_difference'] = f"Default value ({parameters.get('layer_difference')}) - not correlation-based"

        return explanations

    def _get_cache_key(self, features: Dict[str, float]) -> str:
        """Generate cache key from features."""
        # Sort features for consistent hashing
        sorted_features = sorted(features.items())
        feature_str = json.dumps(sorted_features)
        return hashlib.md5(feature_str.encode()).hexdigest()

    def _update_cache(self, key: str, result: Dict[str, Any]):
        """Update cache with size limit."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        # Store a copy to avoid mutations
        self.cache[key] = {
            'parameters': result['parameters'].copy(),
            'confidence': result['confidence'],
            'metadata': result['metadata'].copy()
        }


# Test the optimizer when module is run directly
if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing Feature Mapping Optimizer...")
    print("=" * 60)

    # Create optimizer
    optimizer = FeatureMappingOptimizer()

    # Test Case 1: Complete features
    print("\nTest Case 1: Complete features")
    print("-" * 40)
    features1 = {
        'edge_density': 0.15,
        'unique_colors': 12,
        'entropy': 0.65,
        'corner_density': 0.08,
        'gradient_strength': 0.45,
        'complexity_score': 0.35
    }

    result1 = optimizer.optimize(features1)
    print(f"Confidence: {result1['confidence']:.2f}")
    print(f"Parameters: {json.dumps(result1['parameters'], indent=2)}")
    print(f"Confidence Explanation: {result1['metadata']['confidence_explanation']}")

    # Test Case 2: Missing features
    print("\nTest Case 2: Partial features")
    print("-" * 40)
    features2 = {
        'edge_density': 0.25,
        'unique_colors': 64,
        'complexity_score': 0.7
    }

    result2 = optimizer.optimize(features2)
    print(f"Confidence: {result2['confidence']:.2f}")
    print(f"Parameters: {json.dumps(result2['parameters'], indent=2)}")
    print(f"Confidence Explanation: {result2['metadata']['confidence_explanation']}")

    # Test Case 3: Extreme values
    print("\nTest Case 3: Extreme values")
    print("-" * 40)
    features3 = {
        'edge_density': 0.95,
        'unique_colors': 1000,
        'entropy': 0.01,
        'corner_density': 0.9,
        'gradient_strength': 0.99,
        'complexity_score': 0.99
    }

    result3 = optimizer.optimize(features3)
    print(f"Confidence: {result3['confidence']:.2f}")
    print(f"Parameters: {json.dumps(result3['parameters'], indent=2)}")

    # Test Case 4: Cache hit
    print("\nTest Case 4: Cache hit test")
    print("-" * 40)
    result4 = optimizer.optimize(features1)  # Same as test 1
    print(f"Cache hit: {result4['metadata']['cache_hit']}")

    # Display parameter explanations
    print("\nParameter Explanations for Test Case 1:")
    print("-" * 40)
    for param, explanation in result1['metadata']['parameter_explanations'].items():
        print(f"{param}: {explanation}")

    print("\n" + "=" * 60)
    print("✅ Feature Mapping Optimizer tests complete!")