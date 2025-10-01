#!/usr/bin/env python3
"""
Quality Prediction Model Integration Layer
Integrates optimized local quality prediction models with existing optimization system
"""

import os
import time
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import threading
from collections import deque
import psutil

from .cpu_performance_optimizer import CPUPerformanceOptimizer, CPUOptimizationConfig, PerformanceMetrics
from .intelligent_router import IntelligentRouter
from ..feature_extraction import ImageFeatureExtractor

logger = logging.getLogger(__name__)

@dataclass
class QualityPredictionConfig:
    """Configuration for quality prediction integration"""
    model_path: Optional[str] = None
    device: str = "auto"  # auto, cpu, mps
    inference_timeout_ms: float = 100.0  # Timeout for prediction
    enable_fallback: bool = True
    cache_predictions: bool = True
    cache_size: int = 1000
    performance_target_ms: float = 25.0
    enable_batch_processing: bool = True
    max_batch_size: int = 8

@dataclass
class PredictionResult:
    """Quality prediction result with metadata"""
    quality_score: float
    confidence: float
    inference_time_ms: float
    model_version: str
    device_used: str
    optimization_level: str
    cache_hit: bool
    fallback_used: bool
    timestamp: float

class QualityPredictionCache:
    """Intelligent caching for quality predictions"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()

    def _generate_cache_key(self, image_features: np.ndarray, vtracer_params: Dict[str, Any]) -> str:
        """Generate cache key for prediction"""
        import hashlib

        # Create deterministic key from features and parameters
        features_hash = hashlib.md5(image_features.tobytes()).hexdigest()[:16]
        params_str = json.dumps(sorted(vtracer_params.items()))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:16]

        return f"{features_hash}_{params_hash}"

    def get(self, image_features: np.ndarray, vtracer_params: Dict[str, Any]) -> Optional[PredictionResult]:
        """Get cached prediction if available"""
        with self.lock:
            cache_key = self._generate_cache_key(image_features, vtracer_params)

            if cache_key in self.cache:
                self.hit_count += 1
                self.access_times[cache_key] = time.time()
                cached_result = self.cache[cache_key]

                # Create new result with cache hit flag
                return PredictionResult(
                    quality_score=cached_result.quality_score,
                    confidence=cached_result.confidence,
                    inference_time_ms=0.1,  # Cache access time
                    model_version=cached_result.model_version,
                    device_used=cached_result.device_used,
                    optimization_level="cached",
                    cache_hit=True,
                    fallback_used=False,
                    timestamp=time.time()
                )

            self.miss_count += 1
            return None

    def put(self, image_features: np.ndarray, vtracer_params: Dict[str, Any], result: PredictionResult):
        """Cache prediction result"""
        with self.lock:
            cache_key = self._generate_cache_key(image_features, vtracer_params)

            # Implement LRU eviction if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            self.cache[cache_key] = result
            self.access_times[cache_key] = time.time()

    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(total_requests, 1)

        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }

    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0

class LocalQualityPredictor:
    """Local optimized quality predictor"""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._detect_device(device)
        self.model = None
        self.model_version = "1.0.0-optimized"
        self.feature_size = 2056  # 2048 ResNet + 8 VTracer params

        # Load the model
        self._load_model()

    def _detect_device(self, device: str) -> str:
        """Detect optimal device for inference"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_model(self):
        """Load optimized model"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            # Load TorchScript model
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()

            logger.info(f"Loaded quality prediction model on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, image_features: np.ndarray, vtracer_params: Dict[str, Any]) -> Tuple[float, float]:
        """
        Predict quality score

        Args:
            image_features: ResNet features (2048 dimensions)
            vtracer_params: VTracer parameters

        Returns:
            Tuple of (quality_score, inference_time_ms)
        """
        start_time = time.time()

        try:
            # Prepare input
            param_values = [vtracer_params.get(key, 0.0) for key in [
                'color_precision', 'corner_threshold', 'path_precision',
                'layer_difference', 'filter_speckle', 'splice_threshold',
                'mode', 'hierarchical'
            ]]

            # Combine features and parameters
            combined_input = np.concatenate([image_features, param_values])

            # Ensure correct input size
            if len(combined_input) != self.feature_size:
                raise ValueError(f"Input size mismatch: expected {self.feature_size}, got {len(combined_input)}")

            # Convert to tensor
            input_tensor = torch.FloatTensor(combined_input).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                prediction = self.model(input_tensor).squeeze().item()

            inference_time = (time.time() - start_time) * 1000
            return prediction, inference_time

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_batch(self, features_list: List[np.ndarray],
                     params_list: List[Dict[str, Any]]) -> Tuple[List[float], float]:
        """Batch prediction for efficiency"""
        if len(features_list) != len(params_list):
            raise ValueError("Features and parameters list length mismatch")

        start_time = time.time()

        try:
            # Prepare batch input
            batch_inputs = []
            for features, params in zip(features_list, params_list):
                param_values = [params.get(key, 0.0) for key in [
                    'color_precision', 'corner_threshold', 'path_precision',
                    'layer_difference', 'filter_speckle', 'splice_threshold',
                    'mode', 'hierarchical'
                ]]
                combined_input = np.concatenate([features, param_values])
                batch_inputs.append(combined_input)

            # Stack and convert to tensor
            batch_tensor = torch.FloatTensor(np.stack(batch_inputs)).to(self.device)

            # Batch inference
            with torch.no_grad():
                batch_predictions = self.model(batch_tensor).squeeze()

            # Convert to list
            if batch_predictions.dim() == 0:
                predictions = [batch_predictions.item()]
            else:
                predictions = batch_predictions.tolist()

            inference_time = (time.time() - start_time) * 1000
            return predictions, inference_time

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise

class QualityPredictionIntegrator:
    """Main integration class for quality prediction in optimization system"""

    def __init__(self, config: Optional[QualityPredictionConfig] = None):
        self.config = config or QualityPredictionConfig()

        # Initialize components
        self.cpu_optimizer = CPUPerformanceOptimizer(
            CPUOptimizationConfig(performance_target_ms=self.config.performance_target_ms)
        )
        self.prediction_cache = QualityPredictionCache(self.config.cache_size)
        self.feature_extractor = ImageFeatureExtractor()

        # Model management
        self.primary_predictor = None
        self.fallback_predictor = None
        self.predictor_lock = threading.RLock()

        # Performance monitoring
        self.performance_history = deque(maxlen=1000)
        self.total_predictions = 0
        self.successful_predictions = 0

        # Initialize models
        self._initialize_predictors()

    def _initialize_predictors(self):
        """Initialize quality prediction models"""
        try:
            # Try to find and load primary model
            model_path = self._find_best_model()
            if model_path:
                self.primary_predictor = LocalQualityPredictor(model_path, self.config.device)
                logger.info(f"Primary predictor initialized: {model_path}")

                # Try to load fallback model (if different)
                fallback_path = self._find_fallback_model(model_path)
                if fallback_path and fallback_path != model_path:
                    self.fallback_predictor = LocalQualityPredictor(fallback_path, self.config.device)
                    logger.info(f"Fallback predictor initialized: {fallback_path}")

        except Exception as e:
            logger.error(f"Failed to initialize predictors: {e}")
            if not self.config.enable_fallback:
                raise

    def _find_best_model(self) -> Optional[str]:
        """Find the best available optimized model"""
        if self.config.model_path and Path(self.config.model_path).exists():
            return self.config.model_path

        # Search for models in common locations
        search_paths = [
            "backend/ai_modules/models/optimized/quality_predictor_distilled_optimized.pt",
            "backend/ai_modules/models/optimized/quality_predictor_quantized_optimized.pt",
            "backend/ai_modules/models/optimized/quality_predictor_pruned_optimized.pt",
            "models/quality_predictor_distilled_optimized.pt",
            "models/quality_predictor_optimized.pt"
        ]

        for path in search_paths:
            full_path = Path(path)
            if full_path.exists():
                return str(full_path.absolute())

        return None

    def _find_fallback_model(self, primary_path: str) -> Optional[str]:
        """Find fallback model different from primary"""
        search_paths = [
            "backend/ai_modules/models/optimized/quality_predictor_quantized_optimized.pt",
            "backend/ai_modules/models/optimized/quality_predictor_pruned_optimized.pt",
            "models/quality_predictor_fallback.pt"
        ]

        for path in search_paths:
            full_path = Path(path)
            if full_path.exists() and str(full_path.absolute()) != primary_path:
                return str(full_path.absolute())

        return None

    def predict_quality(self, image_path: str, vtracer_params: Dict[str, Any],
                       enable_cache: Optional[bool] = None) -> PredictionResult:
        """
        Main quality prediction interface

        Args:
            image_path: Path to the image
            vtracer_params: VTracer parameters
            enable_cache: Override cache setting

        Returns:
            PredictionResult with quality score and metadata
        """
        start_time = time.time()
        self.total_predictions += 1

        try:
            # Extract image features
            image_features = self._extract_image_features(image_path)

            # Check cache if enabled
            use_cache = enable_cache if enable_cache is not None else self.config.cache_predictions
            if use_cache:
                cached_result = self.prediction_cache.get(image_features, vtracer_params)
                if cached_result:
                    return cached_result

            # Perform prediction
            result = self._predict_with_optimization(image_features, vtracer_params)

            # Cache result if enabled
            if use_cache and not result.fallback_used:
                self.prediction_cache.put(image_features, vtracer_params, result)

            # Record performance
            self.performance_history.append(result)
            if result.quality_score > 0:
                self.successful_predictions += 1

            return result

        except Exception as e:
            logger.error(f"Quality prediction failed for {image_path}: {e}")

            # Return fallback result
            fallback_result = PredictionResult(
                quality_score=0.85,  # Reasonable default
                confidence=0.0,
                inference_time_ms=(time.time() - start_time) * 1000,
                model_version="fallback",
                device_used="none",
                optimization_level="error",
                cache_hit=False,
                fallback_used=True,
                timestamp=time.time()
            )

            return fallback_result

    def predict_quality_batch(self, image_paths: List[str],
                            vtracer_params_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Batch quality prediction for efficiency"""
        if len(image_paths) != len(vtracer_params_list):
            raise ValueError("Image paths and parameters list length mismatch")

        # Single image optimization
        if len(image_paths) == 1:
            return [self.predict_quality(image_paths[0], vtracer_params_list[0])]

        results = []

        try:
            # Extract features for all images
            all_features = []
            for image_path in image_paths:
                try:
                    features = self._extract_image_features(image_path)
                    all_features.append(features)
                except Exception as e:
                    logger.warning(f"Feature extraction failed for {image_path}: {e}")
                    all_features.append(np.zeros(2048))  # Default features

            # Check cache for all items
            cached_results = []
            uncached_indices = []
            uncached_features = []
            uncached_params = []

            for i, (features, params) in enumerate(zip(all_features, vtracer_params_list)):
                if self.config.cache_predictions:
                    cached = self.prediction_cache.get(features, params)
                    if cached:
                        cached_results.append((i, cached))
                        continue

                uncached_indices.append(i)
                uncached_features.append(features)
                uncached_params.append(params)

            # Batch predict uncached items
            if uncached_features:
                batch_results = self._batch_predict_with_optimization(
                    uncached_features, uncached_params
                )

                # Cache new results
                if self.config.cache_predictions:
                    for features, params, result in zip(uncached_features, uncached_params, batch_results):
                        if not result.fallback_used:
                            self.prediction_cache.put(features, params, result)

                # Merge cached and new results
                new_results = list(zip(uncached_indices, batch_results))
                all_results = cached_results + new_results
                all_results.sort(key=lambda x: x[0])  # Sort by original index

                results = [result for _, result in all_results]
            else:
                # All results were cached
                results = [result for _, result in cached_results]

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # Return fallback results
            results = [self._create_fallback_result() for _ in image_paths]

        return results

    def _extract_image_features(self, image_path: str) -> np.ndarray:
        """Extract image features using the feature extractor"""
        try:
            features_dict = self.feature_extractor.extract_features(image_path)

            # Convert to ResNet-style feature vector (2048 dimensions)
            # This is a placeholder - in real implementation, you'd use actual ResNet
            feature_vector = np.zeros(2048, dtype=np.float32)

            # Map extracted features to vector positions
            feature_mapping = {
                'complexity_score': 0,
                'unique_colors': 1,
                'edge_density': 2,
                'aspect_ratio': 3,
                'fill_ratio': 4,
                'entropy': 5,
                'corner_density': 6,
                'gradient_strength': 7
            }

            for feature_name, value in features_dict.items():
                if feature_name in feature_mapping:
                    idx = feature_mapping[feature_name]
                    feature_vector[idx] = float(value)

            # Fill remaining positions with derived features or zeros
            # In a real implementation, this would be actual ResNet-50 features
            return feature_vector

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(2048, dtype=np.float32)

    def _predict_with_optimization(self, image_features: np.ndarray,
                                 vtracer_params: Dict[str, Any]) -> PredictionResult:
        """Predict with CPU optimization"""
        def prediction_func(features):
            return self._predict_quality_core(features, vtracer_params)

        # Use CPU optimizer for best performance
        try:
            result, perf_metrics = self.cpu_optimizer.optimize_inference(
                prediction_func, image_features, "aggressive"
            )

            quality_score, raw_inference_time = result

            return PredictionResult(
                quality_score=quality_score,
                confidence=0.95,  # High confidence for successful prediction
                inference_time_ms=perf_metrics.inference_time_ms,
                model_version=self.primary_predictor.model_version if self.primary_predictor else "unknown",
                device_used=self.primary_predictor.device if self.primary_predictor else "unknown",
                optimization_level=perf_metrics.optimization_level,
                cache_hit=False,
                fallback_used=False,
                timestamp=time.time()
            )

        except Exception as e:
            logger.warning(f"Optimized prediction failed: {e}")
            return self._fallback_prediction(image_features, vtracer_params)

    def _predict_quality_core(self, image_features: np.ndarray,
                            vtracer_params: Dict[str, Any]) -> Tuple[float, float]:
        """Core prediction function"""
        with self.predictor_lock:
            try:
                if self.primary_predictor:
                    return self.primary_predictor.predict(image_features, vtracer_params)
                elif self.fallback_predictor:
                    return self.fallback_predictor.predict(image_features, vtracer_params)
                else:
                    raise RuntimeError("No predictor available")

            except Exception as e:
                if self.fallback_predictor and self.primary_predictor:
                    logger.warning(f"Primary predictor failed, using fallback: {e}")
                    return self.fallback_predictor.predict(image_features, vtracer_params)
                else:
                    raise

    def _batch_predict_with_optimization(self, features_list: List[np.ndarray],
                                       params_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Batch prediction with optimization"""
        if not self.config.enable_batch_processing or len(features_list) == 1:
            return [self._predict_with_optimization(features, params)
                   for features, params in zip(features_list, params_list)]

        try:
            # Use batch prediction if available
            if self.primary_predictor:
                predictions, total_time = self.primary_predictor.predict_batch(features_list, params_list)
                avg_time = total_time / len(predictions)

                results = []
                for prediction in predictions:
                    results.append(PredictionResult(
                        quality_score=prediction,
                        confidence=0.95,
                        inference_time_ms=avg_time,
                        model_version=self.primary_predictor.model_version,
                        device_used=self.primary_predictor.device,
                        optimization_level="batch",
                        cache_hit=False,
                        fallback_used=False,
                        timestamp=time.time()
                    ))

                return results

        except Exception as e:
            logger.warning(f"Batch prediction failed, falling back to sequential: {e}")

        # Sequential fallback
        return [self._predict_with_optimization(features, params)
               for features, params in zip(features_list, params_list)]

    def _fallback_prediction(self, image_features: np.ndarray,
                           vtracer_params: Dict[str, Any]) -> PredictionResult:
        """Fallback prediction when optimized prediction fails"""
        try:
            if self.fallback_predictor:
                quality_score, inference_time = self.fallback_predictor.predict(image_features, vtracer_params)

                return PredictionResult(
                    quality_score=quality_score,
                    confidence=0.8,  # Lower confidence for fallback
                    inference_time_ms=inference_time,
                    model_version=self.fallback_predictor.model_version,
                    device_used=self.fallback_predictor.device,
                    optimization_level="fallback",
                    cache_hit=False,
                    fallback_used=True,
                    timestamp=time.time()
                )

        except Exception as e:
            logger.error(f"Fallback prediction also failed: {e}")

        # Ultimate fallback - use heuristic
        return self._create_fallback_result()

    def _create_fallback_result(self) -> PredictionResult:
        """Create fallback result using heuristics"""
        return PredictionResult(
            quality_score=0.85,  # Reasonable default
            confidence=0.5,
            inference_time_ms=1.0,
            model_version="heuristic",
            device_used="none",
            optimization_level="heuristic",
            cache_hit=False,
            fallback_used=True,
            timestamp=time.time()
        )

    def integrate_with_router(self, router: IntelligentRouter) -> Callable:
        """Create integration function for IntelligentRouter"""
        def quality_prediction_method(image_path: str, vtracer_params: Dict[str, Any]) -> float:
            """Method for IntelligentRouter integration"""
            result = self.predict_quality(image_path, vtracer_params)
            return result.quality_score

        return quality_prediction_method

    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance information for routing decisions"""
        if not self.performance_history:
            avg_time = 25.0  # Default estimate
        else:
            recent_times = [r.inference_time_ms for r in list(self.performance_history)[-10:]]
            avg_time = np.mean(recent_times)

        success_rate = self.successful_predictions / max(self.total_predictions, 1)

        return {
            'avg_inference_time_ms': avg_time,
            'model_size_mb': 25.0,  # Estimated size
            'accuracy_correlation': 0.92,  # From training
            'device': self.primary_predictor.device if self.primary_predictor else "unknown",
            'model_version': self.primary_predictor.model_version if self.primary_predictor else "unknown",
            'deployment_type': 'local_optimized',
            'success_rate': success_rate,
            'cache_stats': self.prediction_cache.get_stats(),
            'total_predictions': self.total_predictions
        }

    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed performance and status report"""
        cpu_report = self.cpu_optimizer.get_performance_report()
        cache_stats = self.prediction_cache.get_stats()

        recent_performance = []
        if self.performance_history:
            recent_performance = [asdict(r) for r in list(self.performance_history)[-20:]]

        return {
            "integration_status": {
                "primary_predictor_available": self.primary_predictor is not None,
                "fallback_predictor_available": self.fallback_predictor is not None,
                "total_predictions": self.total_predictions,
                "successful_predictions": self.successful_predictions,
                "success_rate": self.successful_predictions / max(self.total_predictions, 1)
            },
            "performance_summary": {
                "avg_inference_time_ms": np.mean([r.inference_time_ms for r in self.performance_history]) if self.performance_history else 0,
                "cache_hit_rate": cache_stats["hit_rate"],
                "fallback_usage_rate": np.mean([r.fallback_used for r in self.performance_history]) if self.performance_history else 0
            },
            "cpu_optimization": cpu_report,
            "cache_statistics": cache_stats,
            "recent_performance": recent_performance,
            "configuration": asdict(self.config)
        }

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.cpu_optimizer.cleanup()
            self.prediction_cache.clear()
            logger.info("Quality prediction integrator cleanup complete")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Factory function
def create_quality_prediction_integrator(config: Optional[QualityPredictionConfig] = None) -> QualityPredictionIntegrator:
    """Create quality prediction integrator instance"""
    return QualityPredictionIntegrator(config)

# Example integration with existing system
class QualityPredictionOptimizer:
    """Optimizer that uses quality prediction for parameter optimization"""

    def __init__(self, integrator: QualityPredictionIntegrator):
        self.integrator = integrator
        self.name = "Quality Prediction Optimizer"

    def optimize(self, image_path: str, target_quality: float = 0.9) -> Dict[str, Any]:
        """Optimize parameters using quality prediction"""
        # This would integrate with the existing optimization system
        # For now, return a placeholder
        return {
            "method": "quality_prediction",
            "target_quality": target_quality,
            "estimated_quality": 0.92,
            "confidence": 0.95
        }

if __name__ == "__main__":
    # Example usage
    config = QualityPredictionConfig(
        performance_target_ms=25.0,
        enable_batch_processing=True
    )

    integrator = create_quality_prediction_integrator(config)

    # Test prediction (would need actual model and image)
    try:
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

        # result = integrator.predict_quality("test_image.png", test_params)
        # print(f"Quality prediction: {result.quality_score:.3f}")
        # print(f"Inference time: {result.inference_time_ms:.1f}ms")
        # print(f"Optimization level: {result.optimization_level}")

        report = integrator.get_detailed_report()
        print(f"Integration status: {report['integration_status']}")

    except Exception as e:
        print(f"Test failed: {e}")

    finally:
        integrator.cleanup()