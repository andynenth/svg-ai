#!/usr/bin/env python3
"""
Unified Prediction API for Quality Prediction Integration
Provides a seamless interface between quality prediction models and existing optimization methods
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import threading
from enum import Enum
import numpy as np

from .quality_prediction_integration import QualityPredictionIntegrator, QualityPredictionConfig, PredictionResult
from .intelligent_router import IntelligentRouter, RoutingDecision
from .feature_mapping import FeatureMappingOptimizer
from .base_optimizer import BaseOptimizer
from ..feature_extraction import ImageFeatureExtractor

logger = logging.getLogger(__name__)

class PredictionMethod(Enum):
    """Available prediction methods"""
    QUALITY_PREDICTION = "quality_prediction"
    FEATURE_MAPPING = "feature_mapping"
    REGRESSION = "regression"
    PPO = "ppo"
    PERFORMANCE = "performance"
    HYBRID = "hybrid"

@dataclass
class UnifiedPredictionConfig:
    """Configuration for unified prediction API"""
    enable_quality_prediction: bool = True
    enable_intelligent_routing: bool = True
    default_method: PredictionMethod = PredictionMethod.QUALITY_PREDICTION
    fallback_method: PredictionMethod = PredictionMethod.FEATURE_MAPPING
    performance_target_ms: float = 25.0
    quality_threshold: float = 0.85
    enable_hybrid_mode: bool = True
    confidence_threshold: float = 0.8
    max_prediction_time_ms: float = 100.0

@dataclass
class UnifiedPredictionResult:
    """Unified prediction result"""
    quality_score: float
    parameters: Dict[str, Any]
    method_used: str
    confidence: float
    inference_time_ms: float
    routing_decision: Optional[RoutingDecision]
    fallback_used: bool
    performance_metrics: Dict[str, Any]
    timestamp: float

class MethodRegistry:
    """Registry for different prediction/optimization methods"""

    def __init__(self):
        self.methods = {}
        self.method_configs = {}
        self.method_performance = {}

    def register_method(self, method_name: str, method_instance: Any, config: Optional[Dict] = None):
        """Register a prediction/optimization method"""
        self.methods[method_name] = method_instance
        self.method_configs[method_name] = config or {}
        self.method_performance[method_name] = {
            'total_calls': 0,
            'successful_calls': 0,
            'avg_time_ms': 0.0,
            'avg_quality': 0.0
        }

        logger.info(f"Registered method: {method_name}")

    def get_method(self, method_name: str) -> Optional[Any]:
        """Get method instance by name"""
        return self.methods.get(method_name)

    def get_available_methods(self) -> List[str]:
        """Get list of available method names"""
        return list(self.methods.keys())

    def update_performance(self, method_name: str, execution_time: float,
                          quality_score: float, success: bool):
        """Update performance metrics for a method"""
        if method_name not in self.method_performance:
            return

        perf = self.method_performance[method_name]
        perf['total_calls'] += 1

        if success:
            perf['successful_calls'] += 1

            # Update running averages
            n = perf['successful_calls']
            perf['avg_time_ms'] = ((n - 1) * perf['avg_time_ms'] + execution_time) / n
            perf['avg_quality'] = ((n - 1) * perf['avg_quality'] + quality_score) / n

    def get_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all methods"""
        return self.method_performance.copy()

class HybridPredictor:
    """Hybrid predictor that combines multiple methods"""

    def __init__(self, registry: MethodRegistry, config: UnifiedPredictionConfig):
        self.registry = registry
        self.config = config
        self.prediction_history = []

    def predict_hybrid(self, image_path: str, vtracer_params: Dict[str, Any],
                      method_weights: Optional[Dict[str, float]] = None) -> UnifiedPredictionResult:
        """
        Hybrid prediction combining multiple methods

        Args:
            image_path: Path to image
            vtracer_params: VTracer parameters
            method_weights: Optional weights for different methods

        Returns:
            Combined prediction result
        """
        start_time = time.time()
        predictions = {}
        execution_times = {}

        # Default weights
        if method_weights is None:
            method_weights = {
                'quality_prediction': 0.6,
                'feature_mapping': 0.3,
                'regression': 0.1
            }

        # Get predictions from available methods
        for method_name, weight in method_weights.items():
            if weight <= 0:
                continue

            method = self.registry.get_method(method_name)
            if method is None:
                continue

            try:
                method_start = time.time()

                if method_name == 'quality_prediction':
                    # Quality prediction method
                    result = method.predict_quality(image_path, vtracer_params)
                    predictions[method_name] = {
                        'quality_score': result.quality_score,
                        'confidence': result.confidence,
                        'weight': weight
                    }
                else:
                    # Other optimization methods - predict quality indirectly
                    # This would need to be implemented based on the specific method
                    predictions[method_name] = {
                        'quality_score': 0.85,  # Placeholder
                        'confidence': 0.7,
                        'weight': weight
                    }

                execution_times[method_name] = (time.time() - method_start) * 1000

            except Exception as e:
                logger.warning(f"Method {method_name} failed in hybrid prediction: {e}")

        # Combine predictions
        if not predictions:
            raise RuntimeError("No methods succeeded in hybrid prediction")

        combined_quality = self._combine_predictions(predictions)
        total_time = (time.time() - start_time) * 1000

        # Calculate overall confidence
        weighted_confidence = sum(
            pred['confidence'] * pred['weight']
            for pred in predictions.values()
        ) / sum(pred['weight'] for pred in predictions.values())

        return UnifiedPredictionResult(
            quality_score=combined_quality,
            parameters=vtracer_params,  # Would be optimized parameters in full implementation
            method_used="hybrid",
            confidence=weighted_confidence,
            inference_time_ms=total_time,
            routing_decision=None,
            fallback_used=False,
            performance_metrics={
                'method_predictions': predictions,
                'execution_times': execution_times,
                'combination_strategy': 'weighted_average'
            },
            timestamp=time.time()
        )

    def _combine_predictions(self, predictions: Dict[str, Dict[str, Any]]) -> float:
        """Combine multiple predictions using weighted average"""
        total_weight = sum(pred['weight'] for pred in predictions.values())
        if total_weight == 0:
            return 0.85  # Default

        weighted_sum = sum(
            pred['quality_score'] * pred['weight']
            for pred in predictions.values()
        )

        return weighted_sum / total_weight

class UnifiedPredictionAPI:
    """Main unified prediction API"""

    def __init__(self, config: Optional[UnifiedPredictionConfig] = None):
        self.config = config or UnifiedPredictionConfig()
        self.method_registry = MethodRegistry()
        self.feature_extractor = ImageFeatureExtractor()

        # Core components
        self.quality_integrator = None
        self.intelligent_router = None
        self.hybrid_predictor = None

        # Performance monitoring
        self.api_calls = 0
        self.successful_calls = 0
        self.performance_history = []

        # Thread safety
        self.lock = threading.RLock()

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all prediction components"""
        try:
            # Initialize quality prediction integrator
            if self.config.enable_quality_prediction:
                quality_config = QualityPredictionConfig(
                    performance_target_ms=self.config.performance_target_ms
                )
                self.quality_integrator = QualityPredictionIntegrator(quality_config)
                self.method_registry.register_method(
                    'quality_prediction',
                    self.quality_integrator
                )

            # Initialize traditional optimizers
            feature_mapping = FeatureMappingOptimizer()
            self.method_registry.register_method('feature_mapping', feature_mapping)

            # Initialize intelligent router
            if self.config.enable_intelligent_routing:
                self.intelligent_router = IntelligentRouter()

            # Initialize hybrid predictor
            if self.config.enable_hybrid_mode:
                self.hybrid_predictor = HybridPredictor(self.method_registry, self.config)

            logger.info("Unified prediction API initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize unified prediction API: {e}")
            raise

    def predict_quality(self, image_path: str, vtracer_params: Dict[str, Any],
                       method: Optional[PredictionMethod] = None,
                       quality_target: float = 0.85) -> UnifiedPredictionResult:
        """
        Main prediction interface

        Args:
            image_path: Path to the image
            vtracer_params: VTracer parameters
            method: Prediction method to use (None for automatic selection)
            quality_target: Target quality score

        Returns:
            Unified prediction result
        """
        start_time = time.time()
        self.api_calls += 1

        with self.lock:
            try:
                # Route to optimal method if no specific method requested
                if method is None:
                    method = self._route_to_optimal_method(image_path, vtracer_params, quality_target)

                # Execute prediction based on selected method
                if method == PredictionMethod.QUALITY_PREDICTION:
                    result = self._predict_with_quality_model(image_path, vtracer_params)

                elif method == PredictionMethod.HYBRID:
                    result = self._predict_with_hybrid(image_path, vtracer_params)

                elif method == PredictionMethod.FEATURE_MAPPING:
                    result = self._predict_with_feature_mapping(image_path, vtracer_params)

                else:
                    # Fallback to quality prediction or feature mapping
                    try:
                        result = self._predict_with_quality_model(image_path, vtracer_params)
                    except Exception:
                        result = self._predict_with_feature_mapping(image_path, vtracer_params)

                # Update performance tracking
                self.successful_calls += 1
                self._update_method_performance(result)

                # Record API performance
                total_time = (time.time() - start_time) * 1000
                self.performance_history.append({
                    'total_time_ms': total_time,
                    'method_used': result.method_used,
                    'quality_score': result.quality_score,
                    'success': True,
                    'timestamp': time.time()
                })

                logger.debug(f"Prediction completed: method={result.method_used}, "
                           f"quality={result.quality_score:.3f}, time={total_time:.1f}ms")

                return result

            except Exception as e:
                logger.error(f"Prediction failed: {e}")

                # Create fallback result
                fallback_result = UnifiedPredictionResult(
                    quality_score=0.85,
                    parameters=vtracer_params,
                    method_used="fallback",
                    confidence=0.5,
                    inference_time_ms=(time.time() - start_time) * 1000,
                    routing_decision=None,
                    fallback_used=True,
                    performance_metrics={'error': str(e)},
                    timestamp=time.time()
                )

                return fallback_result

    def predict_quality_batch(self, image_paths: List[str],
                            vtracer_params_list: List[Dict[str, Any]],
                            method: Optional[PredictionMethod] = None) -> List[UnifiedPredictionResult]:
        """Batch prediction interface"""
        if len(image_paths) != len(vtracer_params_list):
            raise ValueError("Image paths and parameters list length mismatch")

        # Single item optimization
        if len(image_paths) == 1:
            return [self.predict_quality(image_paths[0], vtracer_params_list[0], method)]

        results = []

        # Determine optimal method for batch
        if method is None:
            # Use first image to determine method
            method = self._route_to_optimal_method(
                image_paths[0], vtracer_params_list[0], self.config.quality_threshold
            )

        # Execute batch prediction
        try:
            if method == PredictionMethod.QUALITY_PREDICTION and self.quality_integrator:
                batch_results = self.quality_integrator.predict_quality_batch(
                    image_paths, vtracer_params_list
                )

                # Convert to unified results
                for i, pred_result in enumerate(batch_results):
                    unified_result = UnifiedPredictionResult(
                        quality_score=pred_result.quality_score,
                        parameters=vtracer_params_list[i],
                        method_used="quality_prediction",
                        confidence=pred_result.confidence,
                        inference_time_ms=pred_result.inference_time_ms,
                        routing_decision=None,
                        fallback_used=pred_result.fallback_used,
                        performance_metrics=asdict(pred_result),
                        timestamp=pred_result.timestamp
                    )
                    results.append(unified_result)

            else:
                # Sequential processing for other methods
                for image_path, params in zip(image_paths, vtracer_params_list):
                    result = self.predict_quality(image_path, params, method)
                    results.append(result)

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # Fallback to sequential processing
            for image_path, params in zip(image_paths, vtracer_params_list):
                try:
                    result = self.predict_quality(image_path, params, method)
                    results.append(result)
                except Exception as seq_e:
                    logger.error(f"Sequential fallback failed for {image_path}: {seq_e}")
                    fallback_result = self._create_fallback_result(params)
                    results.append(fallback_result)

        return results

    def _route_to_optimal_method(self, image_path: str, vtracer_params: Dict[str, Any],
                               quality_target: float) -> PredictionMethod:
        """Route to optimal prediction method"""
        try:
            if self.intelligent_router:
                # Use intelligent routing
                routing_decision = self.intelligent_router.route_optimization(
                    image_path, quality_target=quality_target
                )

                # Map routing decision to prediction method
                method_mapping = {
                    'feature_mapping': PredictionMethod.FEATURE_MAPPING,
                    'regression': PredictionMethod.REGRESSION,
                    'ppo': PredictionMethod.PPO,
                    'performance': PredictionMethod.PERFORMANCE
                }

                return method_mapping.get(
                    routing_decision.primary_method,
                    PredictionMethod.QUALITY_PREDICTION
                )

            # Fallback to simple heuristics
            if self.config.enable_quality_prediction and self.quality_integrator:
                return PredictionMethod.QUALITY_PREDICTION
            else:
                return PredictionMethod.FEATURE_MAPPING

        except Exception as e:
            logger.warning(f"Routing failed, using default method: {e}")
            return self.config.default_method

    def _predict_with_quality_model(self, image_path: str,
                                  vtracer_params: Dict[str, Any]) -> UnifiedPredictionResult:
        """Predict using quality prediction model"""
        if not self.quality_integrator:
            raise RuntimeError("Quality prediction integrator not available")

        pred_result = self.quality_integrator.predict_quality(image_path, vtracer_params)

        return UnifiedPredictionResult(
            quality_score=pred_result.quality_score,
            parameters=vtracer_params,
            method_used="quality_prediction",
            confidence=pred_result.confidence,
            inference_time_ms=pred_result.inference_time_ms,
            routing_decision=None,
            fallback_used=pred_result.fallback_used,
            performance_metrics=asdict(pred_result),
            timestamp=pred_result.timestamp
        )

    def _predict_with_hybrid(self, image_path: str,
                           vtracer_params: Dict[str, Any]) -> UnifiedPredictionResult:
        """Predict using hybrid approach"""
        if not self.hybrid_predictor:
            raise RuntimeError("Hybrid predictor not available")

        return self.hybrid_predictor.predict_hybrid(image_path, vtracer_params)

    def _predict_with_feature_mapping(self, image_path: str,
                                    vtracer_params: Dict[str, Any]) -> UnifiedPredictionResult:
        """Predict using feature mapping optimizer"""
        feature_mapping = self.method_registry.get_method('feature_mapping')
        if not feature_mapping:
            raise RuntimeError("Feature mapping optimizer not available")

        start_time = time.time()

        try:
            # Extract features
            features = self.feature_extractor.extract_features(image_path)

            # Infer logo type
            logo_type = self._infer_logo_type(features)

            # Get optimized parameters (this doesn't directly predict quality)
            optimization_result = feature_mapping.optimize(features, logo_type)

            # Estimate quality based on optimization confidence
            estimated_quality = 0.85  # Placeholder - would need actual quality estimation

            inference_time = (time.time() - start_time) * 1000

            return UnifiedPredictionResult(
                quality_score=estimated_quality,
                parameters=optimization_result.get('parameters', vtracer_params),
                method_used="feature_mapping",
                confidence=0.8,
                inference_time_ms=inference_time,
                routing_decision=None,
                fallback_used=False,
                performance_metrics={
                    'optimization_result': optimization_result,
                    'logo_type': logo_type,
                    'features': features
                },
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"Feature mapping prediction failed: {e}")
            raise

    def _infer_logo_type(self, features: Dict[str, float]) -> str:
        """Infer logo type from features"""
        complexity = features.get('complexity_score', 0.5)
        unique_colors = features.get('unique_colors', 16)
        edge_density = features.get('edge_density', 0.3)

        if complexity < 0.3 and edge_density < 0.2:
            return "simple"
        elif edge_density > 0.6 and unique_colors < 10:
            return "text"
        elif unique_colors > 20:
            return "gradient"
        else:
            return "complex"

    def _update_method_performance(self, result: UnifiedPredictionResult):
        """Update performance tracking for the used method"""
        self.method_registry.update_performance(
            result.method_used,
            result.inference_time_ms,
            result.quality_score,
            not result.fallback_used
        )

    def _create_fallback_result(self, vtracer_params: Dict[str, Any]) -> UnifiedPredictionResult:
        """Create fallback result"""
        return UnifiedPredictionResult(
            quality_score=0.85,
            parameters=vtracer_params,
            method_used="fallback",
            confidence=0.5,
            inference_time_ms=1.0,
            routing_decision=None,
            fallback_used=True,
            performance_metrics={},
            timestamp=time.time()
        )

    def get_api_status(self) -> Dict[str, Any]:
        """Get API status and performance information"""
        success_rate = self.successful_calls / max(self.api_calls, 1)

        # Calculate performance statistics
        if self.performance_history:
            recent_history = self.performance_history[-100:]  # Last 100 calls
            avg_time = np.mean([h['total_time_ms'] for h in recent_history])
            success_rate_recent = np.mean([h['success'] for h in recent_history])
        else:
            avg_time = 0.0
            success_rate_recent = 0.0

        status = {
            "api_statistics": {
                "total_calls": self.api_calls,
                "successful_calls": self.successful_calls,
                "success_rate": success_rate,
                "recent_success_rate": success_rate_recent,
                "avg_response_time_ms": avg_time
            },
            "available_methods": self.method_registry.get_available_methods(),
            "method_performance": self.method_registry.get_performance_stats(),
            "configuration": asdict(self.config),
            "component_status": {
                "quality_integrator": self.quality_integrator is not None,
                "intelligent_router": self.intelligent_router is not None,
                "hybrid_predictor": self.hybrid_predictor is not None
            }
        }

        # Add quality integrator status if available
        if self.quality_integrator:
            status["quality_prediction"] = self.quality_integrator.get_performance_info()

        return status

    def optimize_parameters(self, image_path: str, target_quality: float = 0.9) -> Dict[str, Any]:
        """Optimize VTracer parameters to achieve target quality"""
        # This would implement parameter optimization using quality prediction
        # For now, return a basic implementation

        try:
            # Start with default parameters
            base_params = {
                'color_precision': 6,
                'corner_threshold': 60,
                'path_precision': 5,
                'layer_difference': 16,
                'filter_speckle': 2,
                'splice_threshold': 45,
                'mode': 0,
                'hierarchical': 1
            }

            # Predict quality with base parameters
            result = self.predict_quality(image_path, base_params)

            # Simple optimization loop (placeholder)
            best_params = base_params.copy()
            best_quality = result.quality_score

            # Try variations (simplified approach)
            if best_quality < target_quality:
                variations = [
                    {'color_precision': 8, 'path_precision': 8},
                    {'corner_threshold': 40, 'splice_threshold': 30},
                    {'layer_difference': 10, 'filter_speckle': 4}
                ]

                for variation in variations:
                    test_params = base_params.copy()
                    test_params.update(variation)

                    test_result = self.predict_quality(image_path, test_params)
                    if test_result.quality_score > best_quality:
                        best_quality = test_result.quality_score
                        best_params = test_params

                        if best_quality >= target_quality:
                            break

            return {
                'optimized_parameters': best_params,
                'predicted_quality': best_quality,
                'target_achieved': best_quality >= target_quality,
                'optimization_steps': 1,
                'method_used': 'quality_prediction_optimization'
            }

        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return {
                'optimized_parameters': base_params,
                'predicted_quality': 0.85,
                'target_achieved': False,
                'error': str(e),
                'method_used': 'fallback'
            }

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.quality_integrator:
                self.quality_integrator.cleanup()

            if self.intelligent_router:
                self.intelligent_router.shutdown()

            logger.info("Unified prediction API cleanup complete")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Factory function
def create_unified_prediction_api(config: Optional[UnifiedPredictionConfig] = None) -> UnifiedPredictionAPI:
    """Create unified prediction API instance"""
    return UnifiedPredictionAPI(config)

# Example usage and integration
if __name__ == "__main__":
    # Example usage
    config = UnifiedPredictionConfig(
        enable_quality_prediction=True,
        enable_intelligent_routing=True,
        enable_hybrid_mode=True,
        performance_target_ms=25.0
    )

    api = create_unified_prediction_api(config)

    # Test parameters
    test_params = {
        'color_precision': 3.0,
        'corner_threshold': 30.0,
        'path_precision': 8.0,
        'layer_difference': 5.0,
        'filter_speckle': 2.0,
        'splice_threshold': 45.0,
        'mode': 0.0,
        'hierarchical': 1.0
    }

    try:
        # Test prediction (would need actual image)
        # result = api.predict_quality("test_image.png", test_params)
        # print(f"Quality prediction: {result.quality_score:.3f}")
        # print(f"Method used: {result.method_used}")
        # print(f"Inference time: {result.inference_time_ms:.1f}ms")

        # Get API status
        status = api.get_api_status()
        print(f"API Status:")
        print(f"Available methods: {status['available_methods']}")
        print(f"Component status: {status['component_status']}")

    except Exception as e:
        print(f"Test failed: {e}")

    finally:
        api.cleanup()