#!/usr/bin/env python3
"""
AI-Enhanced Converter - Method 1 Integration with BaseConverter
Integrates Method 1 parameter optimization with existing converter system
"""

import time
import tempfile
import hashlib
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import vtracer

from .base import BaseConverter
from ..ai_modules.optimization.feature_mapping import FeatureMappingOptimizer
from ..ai_modules.feature_extraction import ImageFeatureExtractor
from ..ai_modules.optimization.error_handler import OptimizationErrorHandler
from ..utils.quality_metrics import ComprehensiveMetrics
from ..ai_modules.optimization.performance_optimizer import Method1PerformanceOptimizer


class AIEnhancedConverter(BaseConverter):
    """AI-enhanced converter using Method 1 parameter optimization"""

    def __init__(self):
        super().__init__("AI-Enhanced Converter (Method 1)")

        # Core AI components
        self.optimizer = FeatureMappingOptimizer()
        self.feature_extractor = ImageFeatureExtractor()
        self.error_handler = OptimizationErrorHandler()

        # Metrics and performance monitoring
        self.quality_metrics = ComprehensiveMetrics()
        self.performance_optimizer = Method1PerformanceOptimizer()

        # Caching system
        self.optimization_cache = {}
        self.feature_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Configuration
        self.config = {
            "enable_ai_optimization": True,
            "enable_caching": True,
            "cache_max_size": 1000,
            "similarity_threshold": 0.95,
            "quality_target": 0.85,
            "speed_priority": "balanced"  # fast, balanced, quality
        }

        # Conversion tracking
        self.conversion_metadata = []
        self.optimization_history = []

        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("AI-Enhanced Converter initialized with Method 1 optimization")

    def convert(self, image_path: str, **kwargs) -> str:
        """Convert image using AI-optimized parameters"""
        try:
            # Start timing
            start_time = time.time()

            # Validate input
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Extract or retrieve cached features
            features = self._get_features_with_cache(image_path)

            # Get optimization result with caching
            optimization_result = self._get_optimization_with_cache(features, image_path)

            # Apply optimized parameters for conversion
            svg_content = self._convert_with_optimized_params(
                image_path,
                optimization_result['parameters'],
                **kwargs
            )

            # Track conversion results
            conversion_time = time.time() - start_time
            self._track_conversion_result(
                image_path, features, optimization_result,
                conversion_time, svg_content, **kwargs
            )

            self.logger.info(f"AI-enhanced conversion completed in {conversion_time:.3f}s")
            return svg_content

        except Exception as e:
            # Use error handler for graceful failure handling
            return self._handle_conversion_error(e, image_path, **kwargs)

    def convert_with_ai_tier(self, image_path: str, tier: int = 1, include_metadata: bool = True, **kwargs) -> Dict[str, Any]:
        """Convert image using specified AI tier with comprehensive metadata"""
        try:
            start_time = time.time()

            # Validate input
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Extract features
            features = self._get_features_with_cache(image_path)

            # Get tier-specific optimization
            optimization_result = self._get_tier_optimization(features, image_path, tier)

            # Perform conversion
            svg_content = self._convert_with_optimized_params(
                image_path,
                optimization_result['parameters'],
                **kwargs
            )

            # Calculate quality metrics if requested
            predicted_quality = None
            actual_quality = None

            if include_metadata:
                try:
                    # Try to predict quality using the quality predictor
                    if hasattr(self, 'quality_predictor'):
                        predicted_quality = self.quality_predictor.predict_quality(
                            image_path, optimization_result['parameters']
                        )

                    # Calculate actual quality using metrics
                    quality_result = self.quality_metrics.calculate_comprehensive_metrics(
                        image_path, svg_content
                    )
                    actual_quality = quality_result.get('ssim', 0.0)

                except Exception as e:
                    self.logger.warning(f"Quality calculation failed: {e}")

            conversion_time = time.time() - start_time

            # Build comprehensive result
            result = {
                'success': True,
                'svg': svg_content,
                'tier_used': tier,
                'processing_time': conversion_time,
                'predicted_quality': predicted_quality,
                'actual_quality': actual_quality
            }

            if include_metadata:
                result['metadata'] = {
                    'features': features,
                    'optimization': optimization_result,
                    'parameters_used': optimization_result['parameters'],
                    'tier': tier,
                    'method': f"AI-Enhanced Tier {tier}",
                    'confidence': optimization_result.get('confidence', 0.0)
                }

            # Track results
            self._track_conversion_result(
                image_path, features, optimization_result,
                conversion_time, svg_content, **kwargs
            )

            self.logger.info(f"AI Tier {tier} conversion completed in {conversion_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"AI Tier {tier} conversion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'tier_used': tier,
                'processing_time': time.time() - start_time,
                'svg': None
            }

    def _get_tier_optimization(self, features: Dict[str, float], image_path: str, tier: int) -> Dict[str, Any]:
        """Get optimization result for specific tier"""
        try:
            if tier == 1:
                # Fast optimization using Method 1
                return self._get_optimization_with_cache(features, image_path)
            elif tier == 2:
                # Enhanced optimization with quality prediction
                base_result = self._get_optimization_with_cache(features, image_path)
                return self._enhance_with_quality_prediction(base_result, features, image_path)
            elif tier == 3:
                # Full optimization with all methods
                return self._get_full_optimization(features, image_path)
            else:
                # Default to tier 1
                return self._get_optimization_with_cache(features, image_path)

        except Exception as e:
            self.logger.warning(f"Tier {tier} optimization failed: {e}, using default")
            return {"parameters": self._get_default_vtracer_params(), "confidence": 0.0, "method": "fallback"}

    def _enhance_with_quality_prediction(self, base_result: Dict[str, Any], features: Dict[str, float], image_path: str) -> Dict[str, Any]:
        """Enhance Method 1 result with quality prediction guidance"""
        try:
            # Start with base parameters
            enhanced_params = base_result['parameters'].copy()

            # Adjust parameters based on complexity
            complexity = features.get('complexity_score', 0.5)

            if complexity > 0.6:
                # High complexity - increase quality-focused parameters
                enhanced_params['max_iterations'] = min(enhanced_params.get('max_iterations', 10) + 5, 30)
                enhanced_params['path_precision'] = min(enhanced_params.get('path_precision', 3) + 1, 8)
            elif complexity < 0.4:
                # Low complexity - optimize for speed
                enhanced_params['max_iterations'] = max(enhanced_params.get('max_iterations', 10) - 3, 5)

            return {
                'parameters': enhanced_params,
                'confidence': min(base_result.get('confidence', 0.0) + 0.1, 1.0),
                'method': 'Method 1 + Quality Prediction'
            }

        except Exception as e:
            self.logger.warning(f"Quality prediction enhancement failed: {e}")
            return base_result

    def _get_full_optimization(self, features: Dict[str, float], image_path: str) -> Dict[str, Any]:
        """Get full optimization using all available methods"""
        try:
            # Start with enhanced Method 1+2
            base_result = self._get_optimization_with_cache(features, image_path)
            enhanced_result = self._enhance_with_quality_prediction(base_result, features, image_path)

            # Apply Method 3 enhancements
            full_params = enhanced_result['parameters'].copy()

            complexity = features.get('complexity_score', 0.5)
            edge_density = features.get('edge_density', 0.3)

            # Full optimization adjustments
            full_params.update({
                'max_iterations': min(full_params.get('max_iterations', 10) + 10, 50),
                'path_precision': min(full_params.get('path_precision', 3) + 2, 10),
                'corner_threshold': max(full_params.get('corner_threshold', 60) - 5, 10),
                'length_threshold': full_params.get('length_threshold', 5.0) * 0.8
            })

            # Edge-specific adjustments
            if edge_density > 0.5:
                full_params['splice_threshold'] = min(full_params.get('splice_threshold', 45) + 10, 90)

            return {
                'parameters': full_params,
                'confidence': min(enhanced_result.get('confidence', 0.0) + 0.2, 1.0),
                'method': 'Full Optimization (Method 1+2+3)'
            }

        except Exception as e:
            self.logger.warning(f"Full optimization failed: {e}")
            return self._get_optimization_with_cache(features, image_path)

    def _get_features_with_cache(self, image_path: str) -> Dict[str, float]:
        """Extract features with caching support"""
        # Generate cache key based on file path and modification time
        path_obj = Path(image_path)
        cache_key = self._generate_cache_key(
            image_path,
            str(path_obj.stat().st_mtime)
        )

        # Check feature cache
        if self.config["enable_caching"] and cache_key in self.feature_cache:
            self.cache_hits += 1
            self.logger.debug(f"Feature cache hit for {image_path}")
            return self.feature_cache[cache_key]

        # Extract features
        try:
            features = self.feature_extractor.extract_features(image_path)

            # Cache features if enabled
            if self.config["enable_caching"]:
                self._manage_feature_cache(cache_key, features)
                self.cache_misses += 1

            return features

        except Exception as e:
            error = self.error_handler.detect_error(e, {"operation": "feature_extraction", "image_path": image_path})
            recovery_result = self.error_handler.attempt_recovery(error)

            if recovery_result.get("success"):
                # Use default/fallback features
                return self._get_default_features()
            else:
                raise

    def _get_optimization_with_cache(self, features: Dict[str, float], image_path: str) -> Dict[str, Any]:
        """Get optimization result with similarity-based caching"""
        if not self.config["enable_ai_optimization"]:
            return {"parameters": self._get_default_vtracer_params(), "confidence": 0.0, "method": "default"}

        # Check for similar cached optimizations
        if self.config["enable_caching"]:
            cached_result = self._find_similar_optimization(features)
            if cached_result:
                self.cache_hits += 1
                self.logger.debug("Similar optimization found in cache")
                return cached_result

        # Perform optimization
        try:
            # Infer logo type from features
            logo_type = self._infer_logo_type(features)

            # Optimize parameters
            optimization_result = self.optimizer.optimize(features, logo_type)

            # Validate and enhance result
            if not optimization_result.get('parameters'):
                optimization_result = {"parameters": self._get_default_vtracer_params(), "confidence": 0.0}

            # Add metadata
            optimization_result.update({
                "logo_type": logo_type,
                "method": "method_1_correlation",
                "timestamp": time.time()
            })

            # Cache optimization result
            if self.config["enable_caching"]:
                self._cache_optimization_result(features, optimization_result)
                self.cache_misses += 1

            return optimization_result

        except Exception as e:
            error = self.error_handler.detect_error(e, {"operation": "parameter_optimization", "image_path": image_path})
            recovery_result = self.error_handler.attempt_recovery(error)

            if recovery_result.get("success") and "fallback_parameters" in recovery_result:
                return {
                    "parameters": recovery_result["fallback_parameters"],
                    "confidence": 0.5,
                    "method": "error_recovery",
                    "recovery_message": recovery_result.get("message")
                }
            else:
                # Final fallback
                return {
                    "parameters": self._get_default_vtracer_params(),
                    "confidence": 0.0,
                    "method": "final_fallback"
                }

    def _convert_with_optimized_params(self, image_path: str, parameters: Dict[str, Any], **kwargs) -> str:
        """Apply optimized parameters for VTracer conversion"""
        try:
            # Validate parameters
            validated_params = self._validate_vtracer_parameters(parameters)

            # Override any user-provided parameters
            final_params = {**validated_params, **kwargs}

            # Log optimization decision
            self.logger.info(f"Using optimized parameters: {validated_params}")

            # Convert with VTracer using temporary file (required by VTracer 0.6.11+)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp:
                try:
                    vtracer.convert_image_to_svg_py(
                        image_path,
                        tmp.name,
                        **final_params
                    )

                    # Read SVG content
                    with open(tmp.name, 'r') as f:
                        svg_content = f.read()

                    # Validate SVG content
                    if not svg_content or not svg_content.strip():
                        raise ValueError("VTracer produced empty SVG content")

                    return svg_content

                finally:
                    # Cleanup temporary file
                    try:
                        Path(tmp.name).unlink()
                    except:
                        pass

        except Exception as e:
            error = self.error_handler.detect_error(e, {"operation": "vtracer_conversion", "image_path": image_path, "parameters": parameters})
            recovery_result = self.error_handler.attempt_recovery(error)

            if recovery_result.get("success") and "fallback_parameters" in recovery_result:
                # Retry with fallback parameters
                return self._convert_with_optimized_params(image_path, recovery_result["fallback_parameters"], **kwargs)
            else:
                raise RuntimeError(f"VTracer conversion failed: {e}")

    def _handle_conversion_error(self, exception: Exception, image_path: str, **kwargs) -> str:
        """Handle conversion errors gracefully"""
        try:
            error = self.error_handler.detect_error(exception, {"operation": "conversion", "image_path": image_path})
            recovery_result = self.error_handler.attempt_recovery(error)

            if recovery_result.get("success"):
                if "fallback_parameters" in recovery_result:
                    # Retry with fallback parameters
                    return self._convert_with_optimized_params(image_path, recovery_result["fallback_parameters"], **kwargs)
                elif recovery_result.get("message"):
                    self.logger.warning(f"Conversion recovery: {recovery_result['message']}")

            # Final fallback - try with absolute minimal parameters
            try:
                minimal_params = {
                    "colormode": "color",
                    "color_precision": 4,
                    "corner_threshold": 60,
                    "length_threshold": 5.0,
                    "max_iterations": 10,
                    "splice_threshold": 45,
                    "path_precision": 5,
                    "layer_difference": 16
                }

                return self._convert_with_optimized_params(image_path, minimal_params, **kwargs)

            except Exception as final_error:
                # Absolute final fallback
                self.logger.error(f"All conversion attempts failed for {image_path}: {final_error}")
                raise RuntimeError(f"AI-enhanced conversion failed: {exception}")

        except Exception as handler_error:
            self.logger.error(f"Error handler failed: {handler_error}")
            raise RuntimeError(f"Conversion and error handling failed: {exception}")

    def _track_conversion_result(self, image_path: str, features: Dict[str, float],
                               optimization_result: Dict[str, Any], conversion_time: float,
                               svg_content: str, **kwargs):
        """Track conversion results and optimization effectiveness"""
        try:
            # Calculate quality metrics if requested
            quality_metrics = {}
            if kwargs.get("calculate_quality", False):
                # Save SVG temporarily for quality calculation
                with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp:
                    tmp.write(svg_content)
                    tmp.flush()

                    try:
                        quality_metrics = self.quality_metrics.compare_images(image_path, tmp.name)
                    except Exception as e:
                        self.logger.warning(f"Quality metrics calculation failed: {e}")
                        quality_metrics = {"error": str(e)}
                    finally:
                        Path(tmp.name).unlink()

            # Create conversion metadata
            metadata = {
                "timestamp": time.time(),
                "image_path": image_path,
                "features": features,
                "optimization_result": optimization_result,
                "conversion_time": conversion_time,
                "svg_size": len(svg_content),
                "quality_metrics": quality_metrics,
                "cache_stats": {
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses)
                }
            }

            # Store metadata
            self.conversion_metadata.append(metadata)

            # Limit metadata history size
            if len(self.conversion_metadata) > 1000:
                self.conversion_metadata = self.conversion_metadata[-500:]

            # Log optimization effectiveness
            confidence = optimization_result.get("confidence", 0)
            method = optimization_result.get("method", "unknown")
            self.logger.info(f"Conversion tracked: method={method}, confidence={confidence:.3f}, time={conversion_time:.3f}s")

        except Exception as e:
            self.logger.warning(f"Failed to track conversion result: {e}")

    def _find_similar_optimization(self, features: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Find cached optimization for similar features"""
        threshold = self.config["similarity_threshold"]

        for cached_features, cached_result in self.optimization_cache.items():
            similarity = self._calculate_feature_similarity(features, cached_features)
            if similarity >= threshold:
                return cached_result.copy()

        return None

    def _calculate_feature_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate similarity between two feature sets"""
        try:
            # Ensure same keys
            common_keys = set(features1.keys()) & set(features2.keys())
            if not common_keys:
                return 0.0

            # Calculate Euclidean distance in normalized space
            differences = []
            for key in common_keys:
                diff = abs(features1[key] - features2[key])
                differences.append(diff)

            # Convert distance to similarity (0-1)
            avg_difference = sum(differences) / len(differences)
            similarity = max(0.0, 1.0 - avg_difference)

            return similarity

        except Exception:
            return 0.0

    def _cache_optimization_result(self, features: Dict[str, float], result: Dict[str, Any]):
        """Cache optimization result with feature similarity"""
        # Convert features to hashable key
        feature_key = tuple(sorted(features.items()))

        # Manage cache size
        if len(self.optimization_cache) >= self.config["cache_max_size"]:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.optimization_cache.keys())[:100]
            for key in oldest_keys:
                del self.optimization_cache[key]

        # Cache result
        self.optimization_cache[feature_key] = result.copy()

    def _manage_feature_cache(self, cache_key: str, features: Dict[str, float]):
        """Manage feature cache size and storage"""
        # Manage cache size
        if len(self.feature_cache) >= self.config["cache_max_size"]:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.feature_cache.keys())[:100]
            for key in oldest_keys:
                del self.feature_cache[key]

        # Cache features
        self.feature_cache[cache_key] = features.copy()

    def _generate_cache_key(self, *components) -> str:
        """Generate cache key from components"""
        key_string = "|".join(str(c) for c in components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _infer_logo_type(self, features: Dict[str, float]) -> str:
        """Infer logo type from features"""
        edge_density = features.get("edge_density", 0.5)
        unique_colors = features.get("unique_colors", 0.5)
        entropy = features.get("entropy", 0.5)
        complexity = features.get("complexity_score", 0.5)

        # Simple heuristic classification
        if complexity < 0.3 and edge_density < 0.2:
            return "simple"
        elif entropy > 0.8 and unique_colors < 0.2:
            return "text"
        elif unique_colors > 0.6 or features.get("gradient_strength", 0) > 0.5:
            return "gradient"
        else:
            return "complex"

    def _validate_vtracer_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize VTracer parameters"""
        # Default safe parameters
        safe_params = {
            "colormode": "color",
            "color_precision": 6,
            "layer_difference": 16,
            "path_precision": 5,
            "corner_threshold": 60,
            "length_threshold": 5.0,
            "max_iterations": 10,
            "splice_threshold": 45
        }

        # Parameter bounds
        bounds = {
            "color_precision": (1, 10),
            "layer_difference": (1, 30),
            "path_precision": (1, 20),
            "corner_threshold": (10, 110),
            "length_threshold": (1.0, 20.0),
            "max_iterations": (5, 30),
            "splice_threshold": (10, 100)
        }

        # Validate and clamp parameters
        for param, value in parameters.items():
            if param in bounds:
                min_val, max_val = bounds[param]
                if isinstance(value, (int, float)):
                    safe_params[param] = max(min_val, min(max_val, value))
                else:
                    self.logger.warning(f"Invalid parameter type for {param}: {type(value)}")
            elif param == "colormode" and value in ["color", "binary"]:
                safe_params[param] = value

        return safe_params

    def _get_default_vtracer_params(self) -> Dict[str, Any]:
        """Get default VTracer parameters"""
        return {
            "colormode": "color",
            "color_precision": 6,
            "layer_difference": 16,
            "path_precision": 5,
            "corner_threshold": 60,
            "length_threshold": 5.0,
            "max_iterations": 10,
            "splice_threshold": 45
        }

    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values for fallback"""
        return {
            "edge_density": 0.3,
            "unique_colors": 0.4,
            "entropy": 0.6,
            "corner_density": 0.2,
            "gradient_strength": 0.3,
            "complexity_score": 0.5
        }

    def get_name(self) -> str:
        """Get the human-readable name of this converter"""
        return self.name

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization and caching statistics"""
        total_conversions = self.cache_hits + self.cache_misses

        return {
            "total_conversions": total_conversions,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(1, total_conversions),
            "optimization_cache_size": len(self.optimization_cache),
            "feature_cache_size": len(self.feature_cache),
            "config": self.config.copy(),
            "recent_conversions": len(self.conversion_metadata)
        }

    def configure(self, **config_updates):
        """Update converter configuration"""
        for key, value in config_updates.items():
            if key in self.config:
                self.config[key] = value
                self.logger.info(f"Configuration updated: {key} = {value}")
            else:
                self.logger.warning(f"Unknown configuration key: {key}")

    def clear_cache(self):
        """Clear all caches"""
        self.optimization_cache.clear()
        self.feature_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info("All caches cleared")

    def export_conversion_history(self) -> List[Dict[str, Any]]:
        """Export conversion history for analysis"""
        return self.conversion_metadata.copy()

    # Batch processing support
    def batch_convert(self, image_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Convert multiple images with batch optimization"""
        results = []

        # Extract features for all images in parallel if possible
        all_features = []
        for image_path in image_paths:
            try:
                features = self._get_features_with_cache(image_path)
                all_features.append(features)
            except Exception as e:
                all_features.append(self._get_default_features())
                self.logger.warning(f"Feature extraction failed for {image_path}: {e}")

        # Batch optimize parameters
        batch_optimizations = []
        for features, image_path in zip(all_features, image_paths):
            try:
                optimization = self._get_optimization_with_cache(features, image_path)
                batch_optimizations.append(optimization)
            except Exception as e:
                batch_optimizations.append({
                    "parameters": self._get_default_vtracer_params(),
                    "confidence": 0.0,
                    "method": "batch_fallback"
                })
                self.logger.warning(f"Optimization failed for {image_path}: {e}")

        # Convert images with optimized parameters
        for image_path, optimization in zip(image_paths, batch_optimizations):
            try:
                start_time = time.time()
                svg_content = self._convert_with_optimized_params(
                    image_path, optimization["parameters"], **kwargs
                )
                conversion_time = time.time() - start_time

                results.append({
                    "image_path": image_path,
                    "svg_content": svg_content,
                    "optimization": optimization,
                    "conversion_time": conversion_time,
                    "success": True
                })

            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "svg_content": None,
                    "optimization": optimization,
                    "conversion_time": 0.0,
                    "success": False,
                    "error": str(e)
                })

        return results

    def convert_with_quality_validation(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Convert with automatic quality validation"""
        # Perform conversion
        svg_content = self.convert(image_path, **kwargs)

        # Calculate quality metrics
        quality_metrics = {}
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp:
                tmp.write(svg_content)
                tmp.flush()
                quality_metrics = self.quality_metrics.compare_images(image_path, tmp.name)
                Path(tmp.name).unlink()
        except Exception as e:
            quality_metrics = {"error": str(e)}

        # Get the last conversion metadata
        last_conversion = self.conversion_metadata[-1] if self.conversion_metadata else {}

        return {
            "svg_content": svg_content,
            "quality_metrics": quality_metrics,
            "optimization_metadata": last_conversion,
            "meets_quality_target": quality_metrics.get("ssim", 0) >= self.config["quality_target"]
        }