# backend/ai_modules/optimization/adaptive_optimizer.py
"""
Method 3: Adaptive Spatial Optimization System

Implements intelligent method selection and adaptive optimization using spatial analysis.
Routes images to optimal optimization methods based on complexity analysis.
"""

import numpy as np
import time
import logging
import os
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path

from .regional_optimizer import RegionalParameterOptimizer
from .feature_mapping import FeatureMappingOptimizer
from ..feature_extraction import ImageFeatureExtractor

try:
    from .ppo_optimizer import PPOVTracerOptimizer
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    logging.getLogger(__name__).warning("PPO optimizer not available - falling back to Method 1")

try:
    from ..classification.hybrid_classifier import HybridClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    logging.getLogger(__name__).warning("Hybrid classifier not available - using basic classification")


class AdaptiveOptimizer:
    """Method 3: Adaptive spatial optimization system"""

    def __init__(self):
        """Initialize adaptive optimization components"""

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AdaptiveOptimizer")

        # Initialize optimization components
        try:
            self.regional_optimizer = RegionalParameterOptimizer()
            self.logger.info("RegionalParameterOptimizer initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize RegionalParameterOptimizer: {e}")
            raise

        try:
            self.method1_optimizer = FeatureMappingOptimizer()
            self.logger.info("Method 1 (FeatureMappingOptimizer) initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize FeatureMappingOptimizer: {e}")
            raise

        # Initialize Method 2 (PPO) if available
        self.method2_optimizer = None
        if PPO_AVAILABLE:
            try:
                self.method2_optimizer = PPOVTracerOptimizer()
                self.logger.info("Method 2 (PPOVTracerOptimizer) initialized")
            except Exception as e:
                self.logger.warning(f"PPO optimizer initialization failed: {e}")
                self.method2_optimizer = None

        # Analysis components
        try:
            self.feature_extractor = ImageFeatureExtractor()
            self.logger.info("ImageFeatureExtractor initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize ImageFeatureExtractor: {e}")
            raise

        # Initialize classifier if available
        self.classifier = None
        if CLASSIFIER_AVAILABLE:
            try:
                self.classifier = HybridClassifier()
                self.logger.info("HybridClassifier initialized")
            except Exception as e:
                self.logger.warning(f"HybridClassifier initialization failed: {e}")
                self.classifier = None

        # Performance tracking
        self.optimization_history = []
        self.performance_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'adaptive_optimizations': 0,
            'method1_optimizations': 0,
            'method2_optimizations': 0,
            'average_improvement': 0.0,
            'average_processing_time': 0.0,
            'quality_improvements': []
        }

        # Caching system for optimization results
        self.optimization_cache = {}
        self.cache_max_size = 100

        self.logger.info("AdaptiveOptimizer initialization complete")

    def optimize(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """
        Adaptive optimization using spatial analysis

        Args:
            image_path: Path to the image to optimize
            **kwargs: Additional optimization parameters

        Returns:
            Dict containing optimization results with success status, method used,
            quality improvement, processing time, and optimization details
        """

        start_time = time.time()
        self.logger.info(f"Starting adaptive optimization for {image_path}")

        try:
            # Check cache first
            cache_key = self._generate_cache_key(image_path, kwargs)
            if cache_key in self.optimization_cache:
                self.logger.debug(f"Cache hit for {image_path}")
                cached_result = self.optimization_cache[cache_key].copy()
                cached_result['cache_hit'] = True
                return cached_result

            # Extract global features and classify
            self.logger.debug("Extracting features and classifying image")
            features = self.feature_extractor.extract_features(image_path)

            # Classify logo type
            if self.classifier:
                classification_result = self.classifier.classify(image_path)
                # Handle classifier returning dict or string
                if isinstance(classification_result, dict):
                    logo_type = classification_result.get('logo_type', 'unknown')
                else:
                    logo_type = classification_result
            else:
                # Basic classification fallback
                logo_type = self._basic_classify(features)

            self.logger.info(f"Image classified as: {logo_type}, complexity: {features.get('complexity_score', 0.5):.3f}")

            # Determine if adaptive optimization is beneficial
            if self._should_use_adaptive_optimization(features, logo_type):
                self.logger.info("Using adaptive regional optimization")
                result = self._adaptive_regional_optimization(image_path, features)
                result['method_used'] = 'adaptive_regional'
            else:
                # Fall back to Method 1 or 2
                self.logger.info("Using fallback optimization")
                result = self._fallback_optimization(image_path, features, logo_type)

            # Track performance
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            self._update_performance_stats(result, processing_time)

            # Cache result if successful
            if result.get('success', False):
                self._cache_result(cache_key, result)

            # Add metadata
            result['features'] = features
            result['logo_type'] = logo_type
            result['optimization_timestamp'] = time.time()

            self.logger.info(f"Adaptive optimization completed: {result.get('method_used', 'unknown')} "
                           f"in {processing_time:.2f}s, improvement: {result.get('quality_improvement', 0):.1%}")

            return result

        except Exception as e:
            self.logger.error(f"Adaptive optimization failed: {e}")
            processing_time = time.time() - start_time
            return self._emergency_fallback(image_path, processing_time, str(e))

    def _should_use_adaptive_optimization(self, features: Dict[str, float], logo_type: str) -> bool:
        """
        Determine if adaptive optimization is beneficial based on image complexity

        Complex images (>0.7 complexity) → Adaptive regional optimization
        Medium complexity (0.4-0.7) → Method 2 (RL) if available
        Simple images (<0.4) → Method 1 (correlation mapping)
        """

        try:
            complexity = features.get('complexity_score', 0.5)
            edge_density = features.get('edge_density', 0.1)
            unique_colors = features.get('unique_colors', 8)

            # High complexity images benefit from adaptive regional optimization
            if complexity > 0.7:
                self.logger.debug(f"High complexity ({complexity:.3f}) - using adaptive optimization")
                return True

            # Images with high edge density and many colors also benefit
            if edge_density > 0.3 and unique_colors > 15:
                self.logger.debug(f"High edge density ({edge_density:.3f}) and colors ({unique_colors}) - using adaptive optimization")
                return True

            # Complex logo types that typically have spatial variations
            if logo_type in ['complex', 'mixed', 'detailed']:
                self.logger.debug(f"Complex logo type ({logo_type}) - using adaptive optimization")
                return True

            # Images larger than a certain size may benefit from regional optimization
            image_area = features.get('width', 100) * features.get('height', 100)
            if image_area > 250000:  # Large images (>500x500)
                self.logger.debug(f"Large image ({image_area} pixels) - using adaptive optimization")
                return True

            self.logger.debug(f"Standard optimization suitable (complexity: {complexity:.3f})")
            return False

        except Exception as e:
            self.logger.warning(f"Error in adaptive decision logic: {e}")
            return False

    def _adaptive_regional_optimization(self, image_path: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Core adaptive regional optimization method"""

        try:
            self.logger.debug("Starting adaptive regional optimization")

            # Use RegionalParameterOptimizer for spatial analysis and optimization
            regional_result = self.regional_optimizer.optimize_regional_parameters(
                image_path, features
            )

            # Extract key results
            regional_parameters = regional_result.get('regional_parameters', {})
            parameter_maps = regional_result.get('parameter_maps', {})
            regions = regional_result.get('regions', [])
            complexity_analysis = regional_result.get('complexity_analysis', {})
            metadata = regional_result.get('optimization_metadata', {})

            # Calculate quality improvement estimate
            quality_improvement = self._estimate_quality_improvement(
                complexity_analysis, metadata, len(regions)
            )

            # Validate results
            success = (
                len(regional_parameters) > 0 and
                len(parameter_maps) > 0 and
                len(regions) > 0 and
                metadata.get('optimization_success', False)
            )

            if success:
                self.logger.info(f"Adaptive optimization successful: {len(regions)} regions, "
                               f"estimated improvement: {quality_improvement:.1%}")
            else:
                self.logger.warning("Adaptive optimization failed validation checks")

            return {
                'success': success,
                'quality_improvement': quality_improvement,
                'optimized_parameters': self._extract_best_parameters(regional_parameters),
                'regional_parameters': regional_parameters,
                'parameter_maps': parameter_maps,
                'regions': regions,
                'complexity_analysis': complexity_analysis,
                'metadata': metadata,
                'confidence': metadata.get('overall_confidence', 0.5)
            }

        except Exception as e:
            self.logger.error(f"Adaptive regional optimization failed: {e}")
            return {
                'success': False,
                'quality_improvement': 0.0,
                'error': str(e)
            }

    def _fallback_optimization(self, image_path: str, features: Dict[str, float], logo_type: str) -> Dict[str, Any]:
        """Fallback to Method 1 or 2 based on complexity and availability"""

        try:
            complexity = features.get('complexity_score', 0.5)

            # Medium complexity (0.4-0.7) → Method 2 (RL) if available
            if 0.4 <= complexity <= 0.7 and self.method2_optimizer is not None:
                self.logger.info("Using Method 2 (PPO) for medium complexity image")
                try:
                    # Use PPO optimizer
                    ppo_result = self.method2_optimizer.optimize(image_path, **{'logo_type': logo_type})
                    if ppo_result.get('success', False):
                        ppo_result['method_used'] = 'method2_ppo'
                        return ppo_result
                    else:
                        self.logger.warning("Method 2 failed, falling back to Method 1")
                except Exception as e:
                    self.logger.warning(f"Method 2 error: {e}, falling back to Method 1")

            # Simple images (<0.4) or fallback → Method 1 (correlation mapping)
            self.logger.info("Using Method 1 (correlation mapping)")
            method1_result = self.method1_optimizer._optimize_impl(features, logo_type)

            # Convert Method 1 result to expected format
            if isinstance(method1_result, dict):
                # Extract optimization parameters
                optimized_params = {k: v for k, v in method1_result.items()
                                  if k in ['color_precision', 'corner_threshold', 'path_precision',
                                          'layer_difference', 'max_iterations', 'splice_threshold', 'length_threshold']}

                # Estimate quality improvement based on logo type and complexity
                quality_improvement = self._estimate_method1_improvement(complexity, logo_type)

                return {
                    'success': True,
                    'method_used': 'method1_correlation',
                    'quality_improvement': quality_improvement,
                    'optimized_parameters': optimized_params,
                    'confidence': 0.8 if complexity < 0.4 else 0.6
                }
            else:
                self.logger.error("Method 1 returned invalid result format")
                return {
                    'success': False,
                    'method_used': 'method1_correlation',
                    'quality_improvement': 0.0,
                    'error': 'Invalid result format from Method 1'
                }

        except Exception as e:
            self.logger.error(f"Fallback optimization failed: {e}")
            return {
                'success': False,
                'quality_improvement': 0.0,
                'error': str(e)
            }

    def _emergency_fallback(self, image_path: str, processing_time: float = 0.0, error_msg: str = "") -> Dict[str, Any]:
        """Emergency fallback with default parameters"""

        self.logger.warning(f"Emergency fallback activated for {image_path}")

        # Use basic default parameters
        default_params = {
            'color_precision': 6,
            'corner_threshold': 60,
            'path_precision': 8,
            'layer_difference': 10,
            'max_iterations': 10,
            'splice_threshold': 45,
            'length_threshold': 3.5
        }

        return {
            'success': False,
            'method_used': 'emergency_fallback',
            'quality_improvement': 0.0,
            'processing_time': processing_time,
            'optimized_parameters': default_params,
            'confidence': 0.1,
            'error': error_msg,
            'emergency_fallback': True
        }

    def _update_performance_stats(self, result: Dict[str, Any], processing_time: float):
        """Update performance tracking statistics"""

        try:
            self.performance_stats['total_optimizations'] += 1

            if result.get('success', False):
                self.performance_stats['successful_optimizations'] += 1

                # Track method usage
                method = result.get('method_used', 'unknown')
                if method == 'adaptive_regional':
                    self.performance_stats['adaptive_optimizations'] += 1
                elif method == 'method1_correlation':
                    self.performance_stats['method1_optimizations'] += 1
                elif method == 'method2_ppo':
                    self.performance_stats['method2_optimizations'] += 1

                # Track quality improvements
                quality_improvement = result.get('quality_improvement', 0.0)
                self.performance_stats['quality_improvements'].append(quality_improvement)

                # Update average improvement
                improvements = self.performance_stats['quality_improvements']
                if improvements:
                    self.performance_stats['average_improvement'] = np.mean(improvements)

            # Update average processing time
            self.optimization_history.append(processing_time)
            if self.optimization_history:
                self.performance_stats['average_processing_time'] = np.mean(self.optimization_history[-100:])  # Last 100

            # Add to optimization history
            history_entry = {
                'timestamp': time.time(),
                'method': result.get('method_used', 'unknown'),
                'success': result.get('success', False),
                'processing_time': processing_time,
                'quality_improvement': result.get('quality_improvement', 0.0)
            }
            self.optimization_history.append(history_entry)

            # Limit history size
            if len(self.optimization_history) > 1000:
                self.optimization_history = self.optimization_history[-1000:]

        except Exception as e:
            self.logger.warning(f"Error updating performance stats: {e}")

    # Helper methods

    def _basic_classify(self, features: Dict[str, float]) -> str:
        """Basic logo classification fallback when HybridClassifier is not available"""

        try:
            complexity = features.get('complexity_score', 0.5)
            unique_colors = features.get('unique_colors', 8)
            edge_density = features.get('edge_density', 0.1)

            if complexity > 0.7 or (edge_density > 0.3 and unique_colors > 15):
                return 'complex'
            elif complexity < 0.3 and unique_colors <= 5:
                return 'simple'
            elif edge_density < 0.1 and unique_colors > 10:
                return 'gradient'
            else:
                return 'text'

        except Exception as e:
            self.logger.warning(f"Error in basic classification: {e}")
            return 'unknown'

    def _generate_cache_key(self, image_path: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for optimization results"""

        try:
            # Use file path and modification time for cache key
            file_path = Path(image_path)
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                key_data = f"{image_path}_{mtime}_{sorted(kwargs.items())}"
            else:
                key_data = f"{image_path}_{sorted(kwargs.items())}"

            return hashlib.md5(key_data.encode()).hexdigest()

        except Exception as e:
            self.logger.warning(f"Error generating cache key: {e}")
            return f"cache_error_{time.time()}"

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache optimization result"""

        try:
            # Manage cache size
            if len(self.optimization_cache) >= self.cache_max_size:
                # Remove oldest entries
                oldest_keys = list(self.optimization_cache.keys())[:10]
                for key in oldest_keys:
                    del self.optimization_cache[key]

            # Cache result (without heavy data like parameter maps)
            cached_result = result.copy()
            if 'parameter_maps' in cached_result:
                # Store only metadata about parameter maps, not the full arrays
                cached_result['parameter_maps_info'] = {
                    'shape': str(list(cached_result['parameter_maps'].values())[0].shape) if cached_result['parameter_maps'] else 'none',
                    'parameters': list(cached_result['parameter_maps'].keys())
                }
                del cached_result['parameter_maps']

            self.optimization_cache[cache_key] = cached_result

        except Exception as e:
            self.logger.warning(f"Error caching result: {e}")

    def _estimate_quality_improvement(self, complexity_analysis: Dict[str, Any],
                                    metadata: Dict[str, Any], num_regions: int) -> float:
        """Estimate quality improvement for adaptive optimization"""

        try:
            base_improvement = 0.35  # Target 35% improvement

            # Adjust based on complexity
            overall_complexity = complexity_analysis.get('overall_complexity', 0.5)
            if overall_complexity > 0.7:
                improvement_factor = 1.2  # 20% bonus for high complexity
            elif overall_complexity < 0.3:
                improvement_factor = 0.8  # 20% reduction for low complexity
            else:
                improvement_factor = 1.0

            # Adjust based on confidence
            confidence = metadata.get('overall_confidence', 0.5)
            confidence_factor = 0.5 + confidence  # 0.5-1.5 range

            # Adjust based on number of regions (more regions = better optimization potential)
            region_factor = min(1.3, 1.0 + (num_regions - 1) * 0.1)  # Up to 30% bonus

            estimated_improvement = base_improvement * improvement_factor * confidence_factor * region_factor

            # Clamp to reasonable range
            return max(0.1, min(0.8, estimated_improvement))

        except Exception as e:
            self.logger.warning(f"Error estimating quality improvement: {e}")
            return 0.35

    def _estimate_method1_improvement(self, complexity: float, logo_type: str) -> float:
        """Estimate quality improvement for Method 1 optimization"""

        try:
            base_improvement = 0.25  # Base 25% improvement for Method 1

            # Adjust based on logo type suitability
            if logo_type in ['simple', 'geometric']:
                type_factor = 1.2  # Method 1 works well for simple logos
            elif logo_type in ['text']:
                type_factor = 1.1  # Good for text
            elif logo_type in ['gradient']:
                type_factor = 0.9  # Less optimal for gradients
            else:
                type_factor = 0.8  # Not ideal for complex logos

            # Adjust based on complexity
            if complexity < 0.3:
                complexity_factor = 1.2  # Works well for simple images
            elif complexity > 0.6:
                complexity_factor = 0.7  # Limited effectiveness for complex images
            else:
                complexity_factor = 1.0

            estimated_improvement = base_improvement * type_factor * complexity_factor

            # Clamp to reasonable range
            return max(0.05, min(0.6, estimated_improvement))

        except Exception as e:
            self.logger.warning(f"Error estimating Method 1 improvement: {e}")
            return 0.25

    def _extract_best_parameters(self, regional_parameters: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract the best parameters from regional optimization results"""

        try:
            if not regional_parameters:
                return {}

            # Find region with highest confidence
            best_region_id = None
            best_confidence = 0.0

            for region_id, params in regional_parameters.items():
                confidence = params.get('confidence', 0.0)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_region_id = region_id

            if best_region_id is not None:
                best_params = regional_parameters[best_region_id].copy()
                # Remove confidence from parameters
                if 'confidence' in best_params:
                    del best_params['confidence']
                return best_params
            else:
                return {}

        except Exception as e:
            self.logger.warning(f"Error extracting best parameters: {e}")
            return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of adaptive optimizer performance"""

        return {
            'total_optimizations': self.performance_stats['total_optimizations'],
            'success_rate': self.performance_stats['successful_optimizations'] / max(1, self.performance_stats['total_optimizations']),
            'average_improvement': self.performance_stats['average_improvement'],
            'average_processing_time': self.performance_stats['average_processing_time'],
            'method_distribution': {
                'adaptive': self.performance_stats['adaptive_optimizations'],
                'method1': self.performance_stats['method1_optimizations'],
                'method2': self.performance_stats['method2_optimizations']
            },
            'cache_size': len(self.optimization_cache)
        }