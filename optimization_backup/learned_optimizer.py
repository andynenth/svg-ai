"""
Learned Optimizer - DAY3 Task 5

Integration wrapper that uses statistical models for parameter optimization and quality prediction.
Provides a drop-in replacement for existing optimizers with fallback capabilities.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.ai_modules.optimization.statistical_parameter_predictor import StatisticalParameterPredictor
    from backend.ai_modules.prediction.statistical_quality_predictor import StatisticalQualityPredictor
except ImportError as e:
    print(f"Warning: Could not import statistical models: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearnedOptimizer:
    """
    Learned optimizer that uses statistical models for parameter optimization.

    Provides a drop-in replacement for existing optimizers with improved accuracy
    and confidence scoring based on machine learning models.
    """

    def __init__(self,
                 param_predictor_path: Optional[str] = None,
                 quality_predictor_path: Optional[str] = None,
                 enable_fallback: bool = True):
        """
        Initialize the learned optimizer.

        Args:
            param_predictor_path: Path to parameter predictor model
            quality_predictor_path: Path to quality predictor model
            enable_fallback: Whether to enable fallback to correlation formulas
        """
        self.param_predictor = None
        self.quality_predictor = None
        self.enable_fallback = enable_fallback

        # Performance tracking
        self.optimization_count = 0
        self.fallback_count = 0
        self.total_optimization_time = 0.0

        # Load models
        self._load_models(param_predictor_path, quality_predictor_path)

    def _load_models(self, param_predictor_path: Optional[str] = None,
                    quality_predictor_path: Optional[str] = None) -> None:
        """
        Load statistical models with error handling.

        Args:
            param_predictor_path: Path to parameter predictor model
            quality_predictor_path: Path to quality predictor model
        """
        logger.info("Loading learned optimization models")

        # Load parameter predictor
        try:
            self.param_predictor = StatisticalParameterPredictor(param_predictor_path)
            if not self.param_predictor.load_model():
                logger.warning("Failed to load parameter predictor, fallback enabled")
                self.param_predictor = None
        except Exception as e:
            logger.warning(f"Parameter predictor initialization failed: {e}")
            self.param_predictor = None

        # Load quality predictor
        try:
            self.quality_predictor = StatisticalQualityPredictor(quality_predictor_path)
            if not self.quality_predictor.load_model():
                logger.warning("Failed to load quality predictor, fallback enabled")
                self.quality_predictor = None
        except Exception as e:
            logger.warning(f"Quality predictor initialization failed: {e}")
            self.quality_predictor = None

        # Check model status
        models_loaded = sum([
            self.param_predictor is not None,
            self.quality_predictor is not None
        ])

        logger.info(f"Loaded {models_loaded}/2 statistical models")

        if models_loaded == 0 and not self.enable_fallback:
            logger.error("No models loaded and fallback disabled")

    def optimize(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize VTracer parameters based on image features.

        Args:
            features: Dictionary of image features

        Returns:
            Optimization result with parameters, quality prediction, and confidence
        """
        start_time = time.time()
        self.optimization_count += 1

        try:
            # Attempt statistical model optimization
            if self.param_predictor is not None:
                result = self._optimize_with_models(features)
                if result['success']:
                    optimization_time = time.time() - start_time
                    self.total_optimization_time += optimization_time
                    result['optimization_time'] = optimization_time
                    result['method'] = 'statistical_models'
                    return result

            # Fallback to correlation formulas if models fail
            if self.enable_fallback:
                logger.info("Falling back to correlation formulas")
                self.fallback_count += 1
                result = self._optimize_with_fallback(features)
                optimization_time = time.time() - start_time
                self.total_optimization_time += optimization_time
                result['optimization_time'] = optimization_time
                result['method'] = 'correlation_fallback'
                return result
            else:
                return self._create_error_result("Statistical models failed and fallback disabled")

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self._create_error_result(f"Optimization error: {e}")

    def _optimize_with_models(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize using statistical models.

        Args:
            features: Image features

        Returns:
            Optimization result
        """
        try:
            # Predict optimal parameters
            param_result = self.param_predictor.predict_parameters(features)

            if not param_result['success']:
                return self._create_error_result(f"Parameter prediction failed: {param_result.get('error')}")

            predicted_params = param_result['parameters']
            param_confidence = param_result['confidence']

            # Predict expected quality if quality predictor is available
            predicted_quality = None
            quality_confidence = 0.5

            if self.quality_predictor is not None:
                quality_result = self.quality_predictor.predict_quality(features, predicted_params)

                if quality_result['success']:
                    predicted_quality = quality_result['predicted_ssim']
                    quality_confidence = quality_result['confidence']

            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(param_confidence, quality_confidence)

            # Convert parameters to VTracer format
            vtracer_params = self._convert_to_vtracer_format(predicted_params)

            return {
                'parameters': vtracer_params,
                'predicted_quality': predicted_quality,
                'confidence': overall_confidence,
                'parameter_confidence': param_confidence,
                'quality_confidence': quality_confidence,
                'success': True,
                'models_used': {
                    'parameter_predictor': True,
                    'quality_predictor': self.quality_predictor is not None
                }
            }

        except Exception as e:
            logger.error(f"Model-based optimization failed: {e}")
            return self._create_error_result(f"Model optimization error: {e}")

    def _optimize_with_fallback(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize using fallback correlation formulas.

        Args:
            features: Image features

        Returns:
            Optimization result
        """
        try:
            # Simple correlation-based parameter optimization
            # This is a simplified version of what CorrelationFormulas might do

            # Extract relevant features with defaults
            complexity = features.get('complexity_score', 0.5)
            edge_density = features.get('edge_density', 0.5)
            unique_colors = features.get('unique_colors', 0.5)
            entropy = features.get('entropy', 0.5)
            gradient_strength = features.get('gradient_strength', 0.5)

            # Simple heuristic parameter selection
            params = {}

            # Color precision: more colors need higher precision
            params['color_precision'] = int(2 + unique_colors * 6)

            # Corner threshold: high edge density needs lower threshold
            params['corner_threshold'] = int(60 - edge_density * 40)

            # Max iterations: complex images need more iterations
            params['max_iterations'] = int(10 + complexity * 15)

            # Path precision: moderate values work best
            params['path_precision'] = int(3 + entropy * 4)

            # Layer difference: higher for gradients
            params['layer_difference'] = int(8 + gradient_strength * 12)

            # Length threshold: moderate values
            params['length_threshold'] = 3.0 + complexity * 2.0

            # Splice threshold: varies with complexity
            params['splice_threshold'] = int(45 + complexity * 30)

            # Colormode: use color mode for images with high color diversity
            params['colormode'] = 'color' if unique_colors > 0.5 else 'binary'

            # Predict quality using simple heuristic
            predicted_quality = 0.8 - (complexity * 0.3) - (edge_density * 0.2) + (unique_colors * 0.1)
            predicted_quality = max(0.1, min(1.0, predicted_quality))

            # Lower confidence for fallback methods
            fallback_confidence = 0.4  # Lower confidence for fallback

            return {
                'parameters': params,
                'predicted_quality': predicted_quality,
                'confidence': fallback_confidence,
                'parameter_confidence': {'overall': fallback_confidence},
                'quality_confidence': fallback_confidence,
                'success': True,
                'models_used': {
                    'parameter_predictor': False,
                    'quality_predictor': False
                }
            }

        except Exception as e:
            logger.error(f"Fallback optimization failed: {e}")
            return self._create_error_result(f"Fallback optimization error: {e}")

    def _convert_to_vtracer_format(self, predicted_params: Dict[str, float]) -> Dict[str, Any]:
        """
        Convert predicted parameters to VTracer format.

        Args:
            predicted_params: Raw parameter predictions

        Returns:
            VTracer-formatted parameters
        """
        vtracer_params = {}

        # Convert numerical parameters
        vtracer_params['color_precision'] = int(predicted_params.get('color_precision', 5))
        vtracer_params['corner_threshold'] = int(predicted_params.get('corner_threshold', 40))
        vtracer_params['max_iterations'] = int(predicted_params.get('max_iterations', 15))
        vtracer_params['path_precision'] = int(predicted_params.get('path_precision', 5))
        vtracer_params['layer_difference'] = int(predicted_params.get('layer_difference', 12))
        vtracer_params['length_threshold'] = float(predicted_params.get('length_threshold', 4.0))
        vtracer_params['splice_threshold'] = int(predicted_params.get('splice_threshold', 60))

        # Convert colormode
        colormode_value = predicted_params.get('colormode', 0.5)
        vtracer_params['colormode'] = 'color' if colormode_value > 0.5 else 'binary'

        return vtracer_params

    def _calculate_overall_confidence(self, param_confidence: Dict[str, float],
                                    quality_confidence: float) -> float:
        """
        Calculate overall optimization confidence.

        Args:
            param_confidence: Parameter prediction confidence scores
            quality_confidence: Quality prediction confidence

        Returns:
            Overall confidence score
        """
        try:
            # Get overall parameter confidence
            param_overall = param_confidence.get('overall', 0.5)

            # Weight parameter and quality confidence
            # Parameter prediction is more important for optimization
            overall_confidence = (param_overall * 0.7) + (quality_confidence * 0.3)

            return float(np.clip(overall_confidence, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        Create standardized error result.

        Args:
            error_message: Error description

        Returns:
            Error result dictionary
        """
        return {
            'parameters': {
                'color_precision': 5,
                'corner_threshold': 40,
                'max_iterations': 15,
                'path_precision': 5,
                'layer_difference': 12,
                'length_threshold': 4.0,
                'splice_threshold': 60,
                'colormode': 'color'
            },
            'predicted_quality': 0.5,
            'confidence': 0.0,
            'parameter_confidence': {'overall': 0.0},
            'quality_confidence': 0.0,
            'error': error_message,
            'success': False,
            'models_used': {
                'parameter_predictor': False,
                'quality_predictor': False
            }
        }

    def batch_optimize(self, features_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Optimize parameters for multiple images.

        Args:
            features_list: List of feature dictionaries

        Returns:
            List of optimization results
        """
        logger.info(f"Starting batch optimization for {len(features_list)} images")

        results = []
        for i, features in enumerate(features_list):
            try:
                result = self.optimize(features)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Batch optimization failed for item {i}: {e}")
                error_result = self._create_error_result(f"Batch item {i} error: {e}")
                error_result['batch_index'] = i
                results.append(error_result)

        successful_count = sum(1 for r in results if r.get('success', False))
        logger.info(f"Batch optimization completed: {successful_count}/{len(features_list)} successful")

        return results

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization performance statistics.

        Returns:
            Performance statistics
        """
        fallback_rate = (self.fallback_count / self.optimization_count) if self.optimization_count > 0 else 0
        avg_time = (self.total_optimization_time / self.optimization_count) if self.optimization_count > 0 else 0

        return {
            'total_optimizations': self.optimization_count,
            'fallback_count': self.fallback_count,
            'fallback_rate': fallback_rate,
            'avg_optimization_time': avg_time,
            'models_available': {
                'parameter_predictor': self.param_predictor is not None,
                'quality_predictor': self.quality_predictor is not None
            },
            'fallback_enabled': self.enable_fallback
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.

        Returns:
            Model information
        """
        info = {
            'optimizer_name': 'Learned Optimizer',
            'version': '1.0.0',
            'models_loaded': 0,
            'fallback_enabled': self.enable_fallback
        }

        if self.param_predictor is not None:
            param_info = self.param_predictor.get_model_info()
            info['parameter_predictor'] = param_info
            info['models_loaded'] += 1

        if self.quality_predictor is not None:
            quality_info = self.quality_predictor.get_model_info()
            info['quality_predictor'] = quality_info
            info['models_loaded'] += 1

        # Add performance stats
        info['performance_stats'] = self.get_optimization_stats()

        return info

    def is_ready(self) -> bool:
        """
        Check if optimizer is ready for use.

        Returns:
            True if ready (either models loaded or fallback enabled)
        """
        models_ready = (self.param_predictor is not None) or (self.quality_predictor is not None)
        return models_ready or self.enable_fallback

    def reload_models(self) -> bool:
        """
        Reload statistical models.

        Returns:
            True if at least one model loaded successfully
        """
        logger.info("Reloading statistical models")
        self._load_models()
        return (self.param_predictor is not None) or (self.quality_predictor is not None)


def main():
    """Test the learned optimizer."""
    print("Testing Learned Optimizer")
    print("=" * 50)

    # Initialize optimizer
    optimizer = LearnedOptimizer()

    # Test features
    test_features = {
        'edge_density': 0.5,
        'unique_colors': 0.3,
        'entropy': 0.7,
        'complexity_score': 0.4,
        'gradient_strength': 0.2,
        'image_size': 256,
        'aspect_ratio': 1.0
    }

    # Test optimization
    print("Testing single optimization...")
    result = optimizer.optimize(test_features)

    if result['success']:
        print("✓ Optimization successful")
        print(f"  Method: {result.get('method', 'unknown')}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Predicted quality: {result.get('predicted_quality', 'N/A')}")
        print("  Parameters:")
        for param, value in result['parameters'].items():
            print(f"    {param}: {value}")
    else:
        print(f"✗ Optimization failed: {result.get('error')}")

    # Test batch optimization
    print("\nTesting batch optimization...")
    batch_features = [test_features] * 3
    batch_results = optimizer.batch_optimize(batch_features)
    successful = sum(1 for r in batch_results if r.get('success', False))
    print(f"✓ Batch optimization: {successful}/{len(batch_results)} successful")

    # Show optimizer info
    print("\nOptimizer Information:")
    info = optimizer.get_model_info()
    print(f"  Models loaded: {info['models_loaded']}/2")
    print(f"  Fallback enabled: {info['fallback_enabled']}")

    stats = optimizer.get_optimization_stats()
    print(f"  Total optimizations: {stats['total_optimizations']}")
    print(f"  Fallback rate: {stats['fallback_rate']:.1%}")


if __name__ == "__main__":
    main()