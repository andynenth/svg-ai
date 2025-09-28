# backend/ai_modules/classification/base_feature_extractor.py
"""Base class for feature extraction"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
import time

logger = logging.getLogger(__name__)

class BaseFeatureExtractor(ABC):
    """Base class for image feature extraction"""

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.feature_cache = {}
        self.extraction_stats = {
            'total_extractions': 0,
            'cache_hits': 0,
            'average_time': 0.0
        }

    @abstractmethod
    def _extract_features_impl(self, image_path: str) -> Dict[str, float]:
        """Implement actual feature extraction logic"""
        pass

    def extract_features(self, image_path: str) -> Dict[str, float]:
        """Extract features with caching and error handling"""
        start_time = time.time()

        # Check cache first
        if self.cache_enabled and image_path in self.feature_cache:
            self.extraction_stats['cache_hits'] += 1
            logger.debug(f"Cache hit for {image_path}")
            return self.feature_cache[image_path]

        try:
            # Extract features
            features = self._extract_features_impl(image_path)

            # Cache results
            if self.cache_enabled:
                self.feature_cache[image_path] = features

            # Update stats
            extraction_time = time.time() - start_time
            self.extraction_stats['total_extractions'] += 1
            self.extraction_stats['average_time'] = (
                (self.extraction_stats['average_time'] * (self.extraction_stats['total_extractions'] - 1) + extraction_time)
                / self.extraction_stats['total_extractions']
            )

            logger.debug(f"Features extracted for {image_path} in {extraction_time:.3f}s")
            return features

        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path}: {e}")
            # Return default features to allow pipeline to continue
            return self._get_default_features()

    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when extraction fails"""
        return {
            'edge_density': 0.1,
            'unique_colors': 16,
            'entropy': 6.0,
            'corner_density': 0.01,
            'gradient_strength': 25.0,
            'complexity_score': 0.5
        }

    def clear_cache(self):
        """Clear feature cache"""
        self.feature_cache.clear()
        logger.info("Feature cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        cache_hit_rate = (
            self.extraction_stats['cache_hits'] / max(1, self.extraction_stats['total_extractions'])
        )
        return {
            **self.extraction_stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.feature_cache)
        }