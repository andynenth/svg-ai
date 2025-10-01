#!/usr/bin/env python3
"""
Quality Prediction Cache System
High-performance caching system for ML-based quality predictions with <25ms inference targets
Task 14.1.2: Quality Prediction Integration
"""

import time
import json
import hashlib
import pickle
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import OrderedDict, defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Quality prediction cache entry"""
    prediction_data: Dict[str, Any]
    timestamp: float
    access_count: int = 1
    last_accessed: float = None
    cache_hit_savings_ms: float = 0.0

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp


@dataclass
class CacheMetrics:
    """Performance metrics for the cache system"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    avg_hit_time_ms: float = 0.0
    avg_miss_time_ms: float = 0.0
    total_time_saved_ms: float = 0.0
    memory_usage_mb: float = 0.0
    evictions: int = 0


class QualityPredictionCache:
    """High-performance LRU cache for quality predictions with performance optimization"""

    def __init__(self, max_size: int = 5000, ttl_seconds: int = 1800,
                 enable_persistence: bool = True, cache_dir: str = "/tmp/claude/prediction_cache"):
        """
        Initialize quality prediction cache

        Args:
            max_size: Maximum number of cache entries
            ttl_seconds: Time-to-live for cache entries (30 minutes default)
            enable_persistence: Enable disk persistence for cache
            cache_dir: Directory for cache persistence
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_persistence = enable_persistence
        self.cache_dir = Path(cache_dir)

        # Thread-safe cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Performance metrics
        self.metrics = CacheMetrics()

        # Cache optimization settings
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        self.background_optimization = True

        # Cache warming settings
        self.warm_cache_on_init = True
        self.common_scenarios_cache = {}

        # Initialize cache system
        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize the cache system"""
        try:
            if self.enable_persistence:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._load_persistent_cache()

            if self.warm_cache_on_init:
                self._warm_common_scenarios()

            logger.info(f"Quality prediction cache initialized - Size: {len(self._cache)}, "
                       f"Max: {self.max_size}, TTL: {self.ttl_seconds}s")

        except Exception as e:
            logger.error(f"Cache initialization failed: {e}")

    def get_prediction(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction with performance tracking

        Args:
            cache_key: Unique cache key for the prediction

        Returns:
            Cached prediction data or None if not found/expired
        """
        start_time = time.time()

        with self._lock:
            try:
                self.metrics.total_requests += 1

                # Check if key exists
                if cache_key not in self._cache:
                    self.metrics.cache_misses += 1
                    self._update_cache_metrics(start_time, hit=False)
                    return None

                entry = self._cache[cache_key]

                # Check TTL
                if time.time() - entry.timestamp > self.ttl_seconds:
                    # Expired - remove and return None
                    del self._cache[cache_key]
                    self.metrics.cache_misses += 1
                    self._update_cache_metrics(start_time, hit=False)
                    return None

                # Cache hit - update access info and move to end (LRU)
                entry.access_count += 1
                entry.last_accessed = time.time()
                entry.cache_hit_savings_ms = (time.time() - start_time) * 1000

                # Move to end for LRU
                self._cache.move_to_end(cache_key)

                self.metrics.cache_hits += 1
                self._update_cache_metrics(start_time, hit=True, entry=entry)

                logger.debug(f"Cache HIT for key {cache_key[:8]}... (age: {time.time() - entry.timestamp:.1f}s)")
                return entry.prediction_data.copy()

            except Exception as e:
                logger.error(f"Cache get error: {e}")
                self.metrics.cache_misses += 1
                return None

    def set_prediction(self, cache_key: str, prediction_data: Dict[str, Any],
                      inference_time_ms: float = 0.0) -> bool:
        """
        Cache prediction data with LRU eviction

        Args:
            cache_key: Unique cache key
            prediction_data: Prediction result to cache
            inference_time_ms: Original inference time for savings calculation

        Returns:
            True if successfully cached
        """
        with self._lock:
            try:
                current_time = time.time()

                # Create cache entry
                entry = CacheEntry(
                    prediction_data=prediction_data.copy(),
                    timestamp=current_time,
                    cache_hit_savings_ms=inference_time_ms
                )

                # Check if we need to evict entries
                if len(self._cache) >= self.max_size:
                    self._evict_lru_entries(1)

                # Add to cache
                self._cache[cache_key] = entry

                logger.debug(f"Cache SET for key {cache_key[:8]}... "
                           f"(size: {len(self._cache)}/{self.max_size})")

                # Trigger background optimization if needed
                if self.background_optimization and time.time() - self.last_cleanup > self.cleanup_interval:
                    self._background_optimization()

                return True

            except Exception as e:
                logger.error(f"Cache set error: {e}")
                return False

    def generate_cache_key(self, image_features: Dict[str, Any], method: str,
                          vtracer_params: Dict[str, Any], quality_target: float = 0.85) -> str:
        """
        Generate deterministic cache key for quality prediction

        Args:
            image_features: Image feature vector
            method: Optimization method name
            vtracer_params: VTracer parameters
            quality_target: Target quality threshold

        Returns:
            Unique cache key string
        """
        try:
            # Create standardized key components
            key_components = {
                'method': method,
                'quality_target': round(quality_target, 2),
                'image_features': {
                    'complexity': round(image_features.get('complexity_score', 0.5), 3),
                    'colors': image_features.get('unique_colors', 16),
                    'edges': round(image_features.get('edge_density', 0.3), 3),
                    'aspect': round(image_features.get('aspect_ratio', 1.0), 2),
                    'size_class': self._classify_size(image_features.get('file_size', 10000)),
                    'text_prob': round(image_features.get('text_probability', 0.3), 2)
                },
                'vtracer_params': {
                    'color_precision': vtracer_params.get('color_precision', 6),
                    'corner_threshold': vtracer_params.get('corner_threshold', 60),
                    'layer_difference': vtracer_params.get('layer_difference', 16),
                    'path_precision': vtracer_params.get('path_precision', 8)
                }
            }

            # Generate deterministic hash
            key_string = json.dumps(key_components, sort_keys=True)
            cache_key = hashlib.sha256(key_string.encode()).hexdigest()[:24]

            return cache_key

        except Exception as e:
            logger.error(f"Cache key generation failed: {e}")
            # Fallback key
            return hashlib.md5(f"{method}_{time.time()}".encode()).hexdigest()[:16]

    def _classify_size(self, file_size: int) -> str:
        """Classify file size into buckets for cache key"""
        if file_size < 5000:
            return 'small'
        elif file_size < 25000:
            return 'medium'
        elif file_size < 100000:
            return 'large'
        else:
            return 'xlarge'

    def _evict_lru_entries(self, count: int = 1):
        """Evict least recently used entries"""
        try:
            for _ in range(min(count, len(self._cache))):
                if self._cache:
                    # Remove least recently used (first item in OrderedDict)
                    evicted_key, evicted_entry = self._cache.popitem(last=False)
                    self.metrics.evictions += 1
                    logger.debug(f"Evicted LRU entry {evicted_key[:8]}... "
                               f"(age: {time.time() - evicted_entry.timestamp:.1f}s)")

        except Exception as e:
            logger.error(f"LRU eviction failed: {e}")

    def _update_cache_metrics(self, start_time: float, hit: bool, entry: Optional[CacheEntry] = None):
        """Update cache performance metrics"""
        try:
            elapsed_ms = (time.time() - start_time) * 1000

            if hit:
                self.metrics.avg_hit_time_ms = (
                    (self.metrics.avg_hit_time_ms * (self.metrics.cache_hits - 1) + elapsed_ms) /
                    self.metrics.cache_hits if self.metrics.cache_hits > 0 else elapsed_ms
                )
                if entry:
                    self.metrics.total_time_saved_ms += entry.cache_hit_savings_ms
            else:
                self.metrics.avg_miss_time_ms = (
                    (self.metrics.avg_miss_time_ms * (self.metrics.cache_misses - 1) + elapsed_ms) /
                    self.metrics.cache_misses if self.metrics.cache_misses > 0 else elapsed_ms
                )

            # Update hit rate
            self.metrics.cache_hit_rate = (
                self.metrics.cache_hits / self.metrics.total_requests
                if self.metrics.total_requests > 0 else 0.0
            )

        except Exception as e:
            logger.warning(f"Metrics update failed: {e}")

    def _background_optimization(self):
        """Perform background cache optimization"""
        try:
            self.last_cleanup = time.time()

            # 1. Remove expired entries
            expired_count = self._cleanup_expired_entries()

            # 2. Update memory usage estimate
            self._update_memory_usage()

            # 3. Optimize cache if memory usage is high
            if self.metrics.memory_usage_mb > 50:  # 50MB threshold
                self._optimize_memory_usage()

            # 4. Persist cache if enabled
            if self.enable_persistence and len(self._cache) > 0:
                self._persist_cache_background()

            if expired_count > 0:
                logger.debug(f"Background optimization: removed {expired_count} expired entries, "
                           f"memory: {self.metrics.memory_usage_mb:.1f}MB")

        except Exception as e:
            logger.error(f"Background optimization failed: {e}")

    def _cleanup_expired_entries(self) -> int:
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []

        for key, entry in list(self._cache.items()):
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def _update_memory_usage(self):
        """Estimate memory usage of cache"""
        try:
            # Rough estimate: each entry ~1KB average
            estimated_mb = len(self._cache) * 0.001
            self.metrics.memory_usage_mb = estimated_mb
        except Exception:
            self.metrics.memory_usage_mb = 0.0

    def _optimize_memory_usage(self):
        """Optimize memory usage by removing less valuable entries"""
        try:
            if len(self._cache) < 100:
                return

            # Calculate value score for each entry (access_count / age)
            current_time = time.time()
            entry_scores = []

            for key, entry in self._cache.items():
                age_hours = (current_time - entry.timestamp) / 3600
                score = entry.access_count / max(age_hours, 0.1)  # Avoid division by zero
                entry_scores.append((key, score))

            # Sort by score (ascending) and remove bottom 20%
            entry_scores.sort(key=lambda x: x[1])
            remove_count = max(1, len(entry_scores) // 5)

            for key, _ in entry_scores[:remove_count]:
                if key in self._cache:
                    del self._cache[key]
                    self.metrics.evictions += 1

            logger.debug(f"Memory optimization: removed {remove_count} low-value entries")

        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")

    def _warm_common_scenarios(self):
        """Pre-populate cache with common prediction scenarios"""
        try:
            common_scenarios = [
                # Simple geometric logos
                {
                    'features': {
                        'complexity_score': 0.2, 'unique_colors': 3, 'edge_density': 0.2,
                        'aspect_ratio': 1.0, 'file_size': 5000, 'text_probability': 0.1
                    },
                    'methods': ['feature_mapping', 'performance'],
                    'vtracer_params': {'color_precision': 3, 'corner_threshold': 30, 'path_precision': 8}
                },
                # Text-based logos
                {
                    'features': {
                        'complexity_score': 0.4, 'unique_colors': 2, 'edge_density': 0.7,
                        'aspect_ratio': 2.0, 'file_size': 8000, 'text_probability': 0.8
                    },
                    'methods': ['regression', 'feature_mapping'],
                    'vtracer_params': {'color_precision': 2, 'corner_threshold': 20, 'path_precision': 10}
                },
                # Complex gradient logos
                {
                    'features': {
                        'complexity_score': 0.8, 'unique_colors': 20, 'edge_density': 0.6,
                        'aspect_ratio': 1.2, 'file_size': 40000, 'text_probability': 0.2
                    },
                    'methods': ['ppo', 'regression'],
                    'vtracer_params': {'color_precision': 8, 'corner_threshold': 50, 'layer_difference': 8}
                }
            ]

            warmed_count = 0
            for scenario in common_scenarios:
                for method in scenario['methods']:
                    try:
                        # Generate cache key
                        cache_key = self.generate_cache_key(
                            scenario['features'], method, scenario['vtracer_params']
                        )

                        # Create synthetic prediction (would be replaced by actual ML prediction)
                        synthetic_prediction = self._generate_synthetic_prediction(
                            scenario['features'], method
                        )

                        # Cache the prediction
                        self.set_prediction(cache_key, synthetic_prediction, 25.0)
                        warmed_count += 1

                    except Exception as e:
                        logger.warning(f"Cache warming failed for {method}: {e}")

            if warmed_count > 0:
                logger.info(f"Cache warmed with {warmed_count} common scenarios")

        except Exception as e:
            logger.error(f"Cache warming failed: {e}")

    def _generate_synthetic_prediction(self, features: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Generate synthetic prediction for cache warming"""

        complexity = features.get('complexity_score', 0.5)

        # Method-specific quality estimates
        base_qualities = {
            'feature_mapping': 0.85 + (1 - complexity) * 0.1,
            'regression': 0.88 + features.get('text_probability', 0.3) * 0.05,
            'ppo': 0.92 - complexity * 0.05,
            'performance': 0.82
        }

        return {
            'predicted_quality': base_qualities.get(method, 0.85),
            'prediction_time_ms': 25.0,
            'estimated_time_seconds': {'feature_mapping': 0.15, 'regression': 0.30,
                                     'ppo': 0.60, 'performance': 0.05}.get(method, 0.20),
            'confidence': 0.8,
            'method_params': {},
            'ml_predicted': False,
            'synthetic': True
        }

    def _persist_cache_background(self):
        """Persist cache to disk in background (non-blocking)"""
        try:
            # Only persist most valuable entries to limit file size
            current_time = time.time()
            valuable_entries = {}

            for key, entry in list(self._cache.items()):
                # Only persist recent, frequently accessed entries
                age_hours = (current_time - entry.timestamp) / 3600
                if age_hours < 24 and entry.access_count > 1:  # Less than 24h old, accessed multiple times
                    valuable_entries[key] = entry

            if valuable_entries:
                cache_file = self.cache_dir / 'quality_predictions.cache'

                # Use pickle for efficient serialization
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'entries': valuable_entries,
                        'metrics': asdict(self.metrics),
                        'saved_at': current_time
                    }, f)

                logger.debug(f"Persisted {len(valuable_entries)} cache entries to {cache_file}")

        except Exception as e:
            logger.warning(f"Cache persistence failed: {e}")

    def _load_persistent_cache(self):
        """Load cache from disk"""
        try:
            cache_file = self.cache_dir / 'quality_predictions.cache'

            if not cache_file.exists():
                logger.debug("No persistent cache file found")
                return

            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            entries = cache_data.get('entries', {})
            saved_at = cache_data.get('saved_at', 0)

            # Only load recent entries (last 24 hours)
            current_time = time.time()
            loaded_count = 0

            for key, entry in entries.items():
                # Check if entry is still valid
                age_hours = (current_time - entry.timestamp) / 3600
                if age_hours < 24:  # Less than 24 hours old
                    self._cache[key] = entry
                    loaded_count += 1

            if loaded_count > 0:
                logger.info(f"Loaded {loaded_count} cache entries from persistent storage")

        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._lock:
            current_time = time.time()

            # Calculate age distribution
            age_distribution = {'<1h': 0, '1-6h': 0, '6-24h': 0, '>24h': 0}
            access_distribution = {'1': 0, '2-5': 0, '6-10': 0, '>10': 0}

            for entry in self._cache.values():
                age_hours = (current_time - entry.timestamp) / 3600
                if age_hours < 1:
                    age_distribution['<1h'] += 1
                elif age_hours < 6:
                    age_distribution['1-6h'] += 1
                elif age_hours < 24:
                    age_distribution['6-24h'] += 1
                else:
                    age_distribution['>24h'] += 1

                if entry.access_count == 1:
                    access_distribution['1'] += 1
                elif entry.access_count <= 5:
                    access_distribution['2-5'] += 1
                elif entry.access_count <= 10:
                    access_distribution['6-10'] += 1
                else:
                    access_distribution['>10'] += 1

            return {
                'cache_performance': {
                    'hit_rate': self.metrics.cache_hit_rate,
                    'total_requests': self.metrics.total_requests,
                    'cache_hits': self.metrics.cache_hits,
                    'cache_misses': self.metrics.cache_misses,
                    'avg_hit_time_ms': self.metrics.avg_hit_time_ms,
                    'avg_miss_time_ms': self.metrics.avg_miss_time_ms,
                    'total_time_saved_ms': self.metrics.total_time_saved_ms
                },
                'cache_state': {
                    'current_size': len(self._cache),
                    'max_size': self.max_size,
                    'utilization': len(self._cache) / self.max_size,
                    'memory_usage_mb': self.metrics.memory_usage_mb,
                    'evictions': self.metrics.evictions
                },
                'cache_distribution': {
                    'age_distribution': age_distribution,
                    'access_distribution': access_distribution
                },
                'configuration': {
                    'ttl_seconds': self.ttl_seconds,
                    'max_size': self.max_size,
                    'persistence_enabled': self.enable_persistence,
                    'background_optimization': self.background_optimization
                }
            }

    def clear_cache(self, preserve_frequent: bool = False):
        """Clear cache with option to preserve frequently used entries"""
        with self._lock:
            if preserve_frequent:
                # Keep entries accessed more than 5 times
                frequent_entries = {
                    k: v for k, v in self._cache.items()
                    if v.access_count > 5
                }
                cleared_count = len(self._cache) - len(frequent_entries)
                self._cache.clear()
                self._cache.update(frequent_entries)
                logger.info(f"Cleared {cleared_count} cache entries, preserved {len(frequent_entries)} frequent entries")
            else:
                cleared_count = len(self._cache)
                self._cache.clear()
                logger.info(f"Cleared all {cleared_count} cache entries")

    def shutdown(self):
        """Gracefully shutdown cache system"""
        logger.info("Shutting down quality prediction cache...")

        try:
            if self.enable_persistence and len(self._cache) > 0:
                self._persist_cache_background()

            # Clear cache
            self._cache.clear()

            logger.info("Quality prediction cache shutdown complete")

        except Exception as e:
            logger.error(f"Cache shutdown error: {e}")


# Factory function
def create_quality_prediction_cache(max_size: int = 5000, ttl_seconds: int = 1800,
                                  enable_persistence: bool = True) -> QualityPredictionCache:
    """Create optimized quality prediction cache"""
    return QualityPredictionCache(max_size=max_size, ttl_seconds=ttl_seconds,
                                enable_persistence=enable_persistence)


if __name__ == "__main__":
    # Test the cache system
    cache = create_quality_prediction_cache()

    # Test cache operations
    test_features = {
        'complexity_score': 0.4,
        'unique_colors': 8,
        'edge_density': 0.3,
        'aspect_ratio': 1.2,
        'file_size': 15000,
        'text_probability': 0.5
    }

    test_params = {
        'color_precision': 4,
        'corner_threshold': 40,
        'path_precision': 8
    }

    # Generate cache key
    cache_key = cache.generate_cache_key(test_features, 'regression', test_params, 0.9)
    print(f"Generated cache key: {cache_key}")

    # Test cache miss
    result = cache.get_prediction(cache_key)
    print(f"Cache miss result: {result}")

    # Test cache set
    test_prediction = {
        'predicted_quality': 0.887,
        'prediction_time_ms': 23.5,
        'confidence': 0.82
    }

    success = cache.set_prediction(cache_key, test_prediction, 23.5)
    print(f"Cache set success: {success}")

    # Test cache hit
    result = cache.get_prediction(cache_key)
    print(f"Cache hit result: {result}")

    # Display cache stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {json.dumps(stats, indent=2)}")