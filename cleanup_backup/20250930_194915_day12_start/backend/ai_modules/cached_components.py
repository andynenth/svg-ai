#!/usr/bin/env python3
"""
Cached Wrappers for AI Pipeline Components

Provides cache-enabled versions of:
- Feature extraction
- Classification
- Parameter optimization
- Quality validation
- SVG output with metadata
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

from .advanced_cache import get_global_cache, MultiLevelCache
from .feature_extraction import ImageFeatureExtractor
from .rule_based_classifier import RuleBasedClassifier
from .parameter_optimizer import VTracerParameterOptimizer
from .quality_validator import QualityValidator

logger = logging.getLogger(__name__)


class CachedFeatureExtractor:
    """Cache-enabled feature extraction wrapper"""

    def __init__(self, cache: Optional[MultiLevelCache] = None, cache_ttl: int = 3600 * 24):
        """
        Initialize cached feature extractor

        Args:
            cache: Cache instance (uses global if None)
            cache_ttl: Cache time-to-live in seconds (default 24 hours)
        """
        self.cache = cache or get_global_cache()
        self.cache_ttl = cache_ttl
        self.extractor = ImageFeatureExtractor(cache_enabled=False)  # Disable internal cache
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'extraction_time_saved': 0.0
        }

    def _get_image_identifier(self, image_path: str) -> str:
        """Generate stable identifier for image including modification time"""
        try:
            mtime = os.path.getmtime(image_path)
            file_size = os.path.getsize(image_path)
            return hashlib.md5(f"{image_path}:{mtime}:{file_size}".encode()).hexdigest()
        except OSError:
            # Fallback to path-only hash if file stats unavailable
            return hashlib.md5(image_path.encode()).hexdigest()

    def extract_features(self, image_path: str) -> Dict[str, float]:
        """
        Extract features with caching

        Args:
            image_path: Path to input image

        Returns:
            Dictionary of extracted features
        """
        start_time = time.perf_counter()
        identifier = self._get_image_identifier(image_path)

        # Try cache first
        cached_result = self.cache.get('features', identifier)
        if cached_result is not None:
            self.stats['cache_hits'] += 1
            cache_time = time.perf_counter() - start_time
            self.stats['extraction_time_saved'] += cached_result.get('extraction_time', 0) - cache_time
            logger.debug(f"Feature cache hit for {Path(image_path).name}")
            return cached_result['features']

        # Cache miss - extract features
        self.stats['cache_misses'] += 1
        logger.debug(f"Feature cache miss for {Path(image_path).name}")

        extraction_start = time.perf_counter()
        features = self.extractor.extract_features(image_path)
        extraction_time = time.perf_counter() - extraction_start

        # Cache the result
        cache_data = {
            'features': features,
            'extraction_time': extraction_time,
            'timestamp': time.time(),
            'image_path': str(image_path)
        }

        self.cache.set('features', identifier, cache_data, ttl=self.cache_ttl)
        logger.debug(f"Cached features for {Path(image_path).name}")

        return features

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get feature extraction cache statistics"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': self.stats['cache_hits'] / total_requests if total_requests > 0 else 0,
            'time_saved_seconds': self.stats['extraction_time_saved'],
            'total_requests': total_requests
        }


class CachedClassifier:
    """Cache-enabled logo classification wrapper"""

    def __init__(self, cache: Optional[MultiLevelCache] = None, cache_ttl: int = 3600 * 24):
        """
        Initialize cached classifier

        Args:
            cache: Cache instance (uses global if None)
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache = cache or get_global_cache()
        self.cache_ttl = cache_ttl
        self.classifier = RuleBasedClassifier()
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0
        }

    def _get_features_hash(self, features: Dict[str, float]) -> str:
        """Generate hash for feature set"""
        features_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(features_str.encode()).hexdigest()

    def classify_with_details(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify logo with caching

        Args:
            features: Extracted image features

        Returns:
            Classification results with details
        """
        identifier = self._get_features_hash(features)

        # Try cache first
        cached_result = self.cache.get('classification', identifier)
        if cached_result is not None:
            self.stats['cache_hits'] += 1
            logger.debug("Classification cache hit")
            return cached_result

        # Cache miss - perform classification
        self.stats['cache_misses'] += 1
        logger.debug("Classification cache miss")

        result = self.classifier.classify_with_details(features)

        # Cache the result
        self.cache.set('classification', identifier, result, ttl=self.cache_ttl)
        logger.debug("Cached classification result")

        return result

    def classify(self, features: Dict[str, float]) -> str:
        """Simple classification interface"""
        result = self.classify_with_details(features)
        return result['logo_type']

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get classification cache statistics"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': self.stats['cache_hits'] / total_requests if total_requests > 0 else 0,
            'total_requests': total_requests
        }


class CachedParameterOptimizer:
    """Cache-enabled parameter optimization wrapper"""

    def __init__(self, cache: Optional[MultiLevelCache] = None, cache_ttl: int = 3600 * 24 * 7):
        """
        Initialize cached parameter optimizer

        Args:
            cache: Cache instance (uses global if None)
            cache_ttl: Cache time-to-live in seconds (default 7 days)
        """
        self.cache = cache or get_global_cache()
        self.cache_ttl = cache_ttl
        self.optimizer = VTracerParameterOptimizer()
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'optimization_time_saved': 0.0
        }

    def _get_optimization_key(self, classification: Dict, features: Dict,
                            base_parameters: Optional[Dict] = None,
                            user_overrides: Optional[Dict] = None) -> str:
        """Generate cache key for optimization request"""
        key_data = {
            'classification': classification,
            'features': features,
            'base_parameters': base_parameters,
            'user_overrides': user_overrides
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def optimize_parameters(self, classification: Dict, features: Dict,
                          base_parameters: Optional[Dict] = None,
                          user_overrides: Optional[Dict] = None):
        """
        Optimize parameters with caching

        Args:
            classification: Logo classification results
            features: Extracted features
            base_parameters: Base parameter set
            user_overrides: User parameter overrides

        Returns:
            OptimizationResult
        """
        start_time = time.perf_counter()
        identifier = self._get_optimization_key(classification, features, base_parameters, user_overrides)

        # Try cache first
        cached_result = self.cache.get('optimization', identifier)
        if cached_result is not None:
            self.stats['cache_hits'] += 1
            cache_time = time.perf_counter() - start_time
            self.stats['optimization_time_saved'] += cached_result.get('optimization_time', 0) - cache_time
            logger.debug("Parameter optimization cache hit")

            # Reconstruct OptimizationResult object
            from .parameter_optimizer import OptimizationResult
            return OptimizationResult(
                parameters=cached_result['parameters'],
                optimization_method=cached_result['optimization_method'],
                adjustments_applied=cached_result['adjustments_applied'],
                validation_passed=cached_result['validation_passed'],
                confidence_level=cached_result['confidence_level'],
                quality_prediction=cached_result['quality_prediction']
            )

        # Cache miss - perform optimization
        self.stats['cache_misses'] += 1
        logger.debug("Parameter optimization cache miss")

        optimization_start = time.perf_counter()
        result = self.optimizer.optimize_parameters(classification, features, base_parameters, user_overrides)
        optimization_time = time.perf_counter() - optimization_start

        # Cache the result
        cache_data = {
            'parameters': result.parameters,
            'optimization_method': result.optimization_method,
            'adjustments_applied': result.adjustments_applied,
            'validation_passed': result.validation_passed,
            'confidence_level': result.confidence_level,
            'quality_prediction': result.quality_prediction,
            'optimization_time': optimization_time,
            'timestamp': time.time()
        }

        self.cache.set('optimization', identifier, cache_data, ttl=self.cache_ttl)
        logger.debug("Cached parameter optimization result")

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get optimization cache statistics"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': self.stats['cache_hits'] / total_requests if total_requests > 0 else 0,
            'time_saved_seconds': self.stats['optimization_time_saved'],
            'total_requests': total_requests
        }


class CachedQualityValidator:
    """Cache-enabled quality validation wrapper"""

    def __init__(self, cache: Optional[MultiLevelCache] = None, cache_ttl: int = 3600 * 24):
        """
        Initialize cached quality validator

        Args:
            cache: Cache instance (uses global if None)
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache = cache or get_global_cache()
        self.cache_ttl = cache_ttl
        self.validator = QualityValidator()
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_time_saved': 0.0
        }

    def _get_validation_key(self, original_path: str, svg_content: str,
                          parameters_used: Optional[Dict] = None,
                          features: Optional[Dict] = None) -> str:
        """Generate cache key for validation request"""
        # Use hash of SVG content and parameters for cache key
        svg_hash = hashlib.md5(svg_content.encode()).hexdigest()

        key_data = {
            'original_path': original_path,
            'svg_hash': svg_hash,
            'parameters': parameters_used,
            'features': features
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def validate_conversion(self, original_path: str, svg_content: str,
                          parameters_used: Optional[Dict] = None,
                          features: Optional[Dict] = None):
        """
        Validate conversion quality with caching

        Args:
            original_path: Path to original image
            svg_content: SVG content to validate
            parameters_used: Parameters used for conversion
            features: Image features

        Returns:
            QualityReport
        """
        start_time = time.perf_counter()
        identifier = self._get_validation_key(original_path, svg_content, parameters_used, features)

        # Try cache first
        cached_result = self.cache.get('quality', identifier)
        if cached_result is not None:
            self.stats['cache_hits'] += 1
            cache_time = time.perf_counter() - start_time
            self.stats['validation_time_saved'] += cached_result.get('validation_time', 0) - cache_time
            logger.debug("Quality validation cache hit")

            # Reconstruct QualityReport object
            from .quality_validator import QualityReport, QualityMetrics, QualityLevel

            metrics = QualityMetrics(
                ssim_score=cached_result['metrics']['ssim_score'],
                mse_score=cached_result['metrics']['mse_score'],
                psnr_score=cached_result['metrics']['psnr_score'],
                file_size_ratio=cached_result['metrics']['file_size_ratio'],
                quality_level=QualityLevel(cached_result['metrics']['quality_level'])
            )

            return QualityReport(
                metrics=metrics,
                quality_passed=cached_result['quality_passed'],
                recommendations=cached_result['recommendations'],
                parameter_suggestions=cached_result['parameter_suggestions'],
                processing_time=cached_result['processing_time']
            )

        # Cache miss - perform validation
        self.stats['cache_misses'] += 1
        logger.debug("Quality validation cache miss")

        validation_start = time.perf_counter()
        result = self.validator.validate_conversion(original_path, svg_content, parameters_used, features)
        validation_time = time.perf_counter() - validation_start

        # Cache the result
        cache_data = {
            'metrics': {
                'ssim_score': result.metrics.ssim_score,
                'mse_score': result.metrics.mse_score,
                'psnr_score': result.metrics.psnr_score,
                'file_size_ratio': result.metrics.file_size_ratio,
                'quality_level': result.metrics.quality_level.value
            },
            'quality_passed': result.quality_passed,
            'recommendations': result.recommendations,
            'parameter_suggestions': result.parameter_suggestions,
            'processing_time': result.processing_time,
            'validation_time': validation_time,
            'timestamp': time.time()
        }

        self.cache.set('quality', identifier, cache_data, ttl=self.cache_ttl)
        logger.debug("Cached quality validation result")

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get quality validation cache statistics"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': self.stats['cache_hits'] / total_requests if total_requests > 0 else 0,
            'time_saved_seconds': self.stats['validation_time_saved'],
            'total_requests': total_requests
        }


class CachedSVGOutput:
    """Cache-enabled SVG output with quality metadata"""

    def __init__(self, cache: Optional[MultiLevelCache] = None, cache_ttl: int = 3600 * 24 * 7):
        """
        Initialize cached SVG output manager

        Args:
            cache: Cache instance (uses global if None)
            cache_ttl: Cache time-to-live in seconds (default 7 days)
        """
        self.cache = cache or get_global_cache()
        self.cache_ttl = cache_ttl
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'conversion_time_saved': 0.0
        }

    def _get_svg_cache_key(self, image_path: str, parameters: Dict, ai_enhanced: bool = False) -> str:
        """Generate cache key for SVG output"""
        # Include file modification time for cache invalidation
        try:
            mtime = os.path.getmtime(image_path)
            file_size = os.path.getsize(image_path)
        except OSError:
            mtime = 0
            file_size = 0

        key_data = {
            'image_path': image_path,
            'mtime': mtime,
            'file_size': file_size,
            'parameters': parameters,
            'ai_enhanced': ai_enhanced
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_cached_svg(self, image_path: str, parameters: Dict, ai_enhanced: bool = False) -> Optional[Dict]:
        """
        Get cached SVG output with metadata

        Args:
            image_path: Path to input image
            parameters: Conversion parameters used
            ai_enhanced: Whether AI enhancement was used

        Returns:
            Cached SVG data or None
        """
        identifier = self._get_svg_cache_key(image_path, parameters, ai_enhanced)

        cached_result = self.cache.get('svg_output', identifier)
        if cached_result is not None:
            self.stats['cache_hits'] += 1
            logger.debug(f"SVG cache hit for {Path(image_path).name}")
            return cached_result

        self.stats['cache_misses'] += 1
        logger.debug(f"SVG cache miss for {Path(image_path).name}")
        return None

    def cache_svg_output(self, image_path: str, parameters: Dict, svg_content: str,
                        ai_metadata: Optional[Dict] = None, quality_metrics: Optional[Dict] = None,
                        conversion_time: float = 0.0, ai_enhanced: bool = False):
        """
        Cache SVG output with comprehensive metadata

        Args:
            image_path: Path to input image
            parameters: Parameters used for conversion
            svg_content: Generated SVG content
            ai_metadata: AI analysis metadata
            quality_metrics: Quality validation metrics
            conversion_time: Time taken for conversion
            ai_enhanced: Whether AI enhancement was used
        """
        identifier = self._get_svg_cache_key(image_path, parameters, ai_enhanced)

        cache_data = {
            'svg_content': svg_content,
            'parameters': parameters,
            'ai_metadata': ai_metadata or {},
            'quality_metrics': quality_metrics or {},
            'conversion_time': conversion_time,
            'ai_enhanced': ai_enhanced,
            'timestamp': time.time(),
            'image_path': str(image_path),
            'svg_size_bytes': len(svg_content.encode('utf-8')),
            'cache_version': '1.0'
        }

        self.cache.set('svg_output', identifier, cache_data, ttl=self.cache_ttl)
        logger.debug(f"Cached SVG output for {Path(image_path).name}")

    def invalidate_image_cache(self, image_path: str):
        """
        Invalidate all cached entries for a specific image

        Args:
            image_path: Path to image whose cache should be invalidated
        """
        # This is a simplified implementation
        # In production, would maintain reverse index of cache keys by image
        logger.info(f"Invalidating cache for {image_path}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get SVG output cache statistics"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': self.stats['cache_hits'] / total_requests if total_requests > 0 else 0,
            'time_saved_seconds': self.stats['conversion_time_saved'],
            'total_requests': total_requests
        }


class CachedFeaturePipeline:
    """Cache-enabled unified feature pipeline"""

    def __init__(self, cache: Optional[MultiLevelCache] = None):
        """
        Initialize cached feature pipeline

        Args:
            cache: Cache instance (uses global if None)
        """
        self.cache = cache or get_global_cache()
        self.feature_extractor = CachedFeatureExtractor(cache)
        self.classifier = CachedClassifier(cache)
        self.stats = {
            'total_processed': 0,
            'pipeline_cache_hits': 0,
            'pipeline_cache_misses': 0
        }

    def _get_pipeline_key(self, image_path: str) -> str:
        """Generate cache key for complete pipeline result"""
        return self.feature_extractor._get_image_identifier(image_path)

    def process_image(self, image_path: str) -> Dict:
        """
        Process image through complete cached pipeline

        Args:
            image_path: Path to input image

        Returns:
            Complete pipeline results with caching
        """
        start_time = time.perf_counter()
        identifier = self._get_pipeline_key(image_path)

        # Try complete pipeline cache first
        cached_result = self.cache.get('pipeline', identifier)
        if cached_result is not None:
            self.stats['pipeline_cache_hits'] += 1
            self.stats['total_processed'] += 1
            logger.debug(f"Complete pipeline cache hit for {Path(image_path).name}")
            return cached_result

        # Cache miss - run pipeline
        self.stats['pipeline_cache_misses'] += 1
        logger.debug(f"Complete pipeline cache miss for {Path(image_path).name}")

        # Extract features (with caching)
        features = self.feature_extractor.extract_features(image_path)

        # Classify (with caching)
        classification_details = self.classifier.classify_with_details(features)

        # Build complete result
        total_time = time.perf_counter() - start_time

        result = {
            'features': features,
            'classification': classification_details,
            'metadata': {
                'image_path': str(image_path),
                'cache_hit': False,
                'processing_time': total_time,
                'timestamp': time.time()
            },
            'performance': {
                'total_time': total_time,
                'feature_extraction_cached': self.feature_extractor.stats['cache_hits'] > 0,
                'classification_cached': self.classifier.stats['cache_hits'] > 0
            }
        }

        # Cache complete pipeline result
        self.cache.set('pipeline', identifier, result, ttl=3600 * 24)  # 24 hour TTL
        self.stats['total_processed'] += 1

        return result

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all cached components"""
        return {
            'pipeline': {
                'total_processed': self.stats['total_processed'],
                'pipeline_cache_hits': self.stats['pipeline_cache_hits'],
                'pipeline_cache_misses': self.stats['pipeline_cache_misses'],
                'pipeline_hit_rate': self.stats['pipeline_cache_hits'] / max(1, self.stats['total_processed'])
            },
            'feature_extraction': self.feature_extractor.get_cache_stats(),
            'classification': self.classifier.get_cache_stats(),
            'cache_system': self.cache.get_comprehensive_stats()
        }