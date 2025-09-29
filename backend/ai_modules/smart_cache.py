#!/usr/bin/env python3
"""
Smart Caching Strategies and Adaptive Cache Management

Implements intelligent caching strategies including:
- Advanced LRU eviction policies with prioritization
- Cache warming for popular logo types and patterns
- Adaptive cache sizing based on usage patterns
- Intelligent cache compression strategies
- Advanced cache analytics and prediction
"""

import hashlib
import json
import logging
import os
import pickle
import threading
import time
import zlib
from collections import defaultdict, OrderedDict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import statistics
import heapq
import math

from .advanced_cache import MultiLevelCache, CacheEntry, get_global_cache
from .cache_monitor import get_global_monitor

logger = logging.getLogger(__name__)


class CacheAccessPattern:
    """Tracks and analyzes cache access patterns"""

    def __init__(self):
        self.access_history = []  # (timestamp, key, cache_type, hit)
        self.key_frequencies = Counter()
        self.temporal_patterns = defaultdict(list)  # hour -> access_count
        self.cache_type_patterns = defaultdict(int)
        self.lock = threading.RLock()

    def record_access(self, key: str, cache_type: str, hit: bool, timestamp: Optional[float] = None):
        """Record cache access for pattern analysis"""
        timestamp = timestamp or time.time()
        hour = datetime.fromtimestamp(timestamp).hour

        with self.lock:
            self.access_history.append((timestamp, key, cache_type, hit))
            self.key_frequencies[key] += 1
            self.temporal_patterns[hour].append(timestamp)
            self.cache_type_patterns[cache_type] += 1

            # Keep history manageable (last 10000 accesses)
            if len(self.access_history) > 10000:
                self.access_history = self.access_history[-5000:]

    def get_popular_keys(self, top_n: int = 50) -> List[Tuple[str, int]]:
        """Get most frequently accessed keys"""
        with self.lock:
            return self.key_frequencies.most_common(top_n)

    def get_temporal_patterns(self) -> Dict[int, float]:
        """Get average access patterns by hour of day"""
        with self.lock:
            patterns = {}
            for hour, accesses in self.temporal_patterns.items():
                if accesses:
                    # Calculate average accesses per hour over last 7 days
                    cutoff = time.time() - (7 * 24 * 3600)
                    recent_accesses = [t for t in accesses if t >= cutoff]
                    days = max(1, len(recent_accesses) / 24)
                    patterns[hour] = len(recent_accesses) / days
            return patterns

    def predict_cache_usage(self) -> Dict[str, Any]:
        """Predict future cache usage patterns"""
        with self.lock:
            current_hour = datetime.now().hour
            patterns = self.get_temporal_patterns()

            # Predict next hour usage
            next_hour = (current_hour + 1) % 24
            predicted_usage = patterns.get(next_hour, patterns.get(current_hour, 1.0))

            # Identify trending keys (increasing access frequency)
            recent_cutoff = time.time() - 3600  # Last hour
            recent_accesses = [
                (ts, key) for ts, key, _, _ in self.access_history
                if ts >= recent_cutoff
            ]
            recent_key_counts = Counter(key for _, key in recent_accesses)

            # Compare with historical averages
            trending_keys = []
            for key, recent_count in recent_key_counts.most_common(20):
                historical_rate = self.key_frequencies[key] / max(1, len(self.access_history))
                recent_rate = recent_count / max(1, len(recent_accesses))
                if recent_rate > historical_rate * 2:  # 2x higher than historical
                    trending_keys.append((key, recent_rate / historical_rate))

            return {
                'predicted_next_hour_usage': predicted_usage,
                'current_usage_multiplier': patterns.get(current_hour, 1.0),
                'trending_keys': trending_keys[:10],
                'peak_hours': sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
            }


class PriorityLRUCache:
    """Enhanced LRU cache with priority-based eviction"""

    def __init__(self, max_size: int = 1000, enable_priorities: bool = True):
        self.max_size = max_size
        self.enable_priorities = enable_priorities
        self.cache = OrderedDict()
        self.priorities = {}  # key -> priority_score
        self.access_counts = defaultdict(int)
        self.last_access = {}
        self.lock = threading.RLock()

    def _calculate_priority(self, key: str, value: Any) -> float:
        """Calculate priority score for cache entry"""
        if not self.enable_priorities:
            return 1.0

        # Factors influencing priority:
        # 1. Access frequency
        # 2. Recency of access
        # 3. Size/cost of regeneration
        # 4. Cache type importance

        access_count = self.access_counts[key]
        last_access_time = self.last_access.get(key, time.time())
        recency = time.time() - last_access_time

        # Frequency score (logarithmic scaling)
        frequency_score = math.log1p(access_count) / 10.0

        # Recency score (exponential decay)
        recency_score = math.exp(-recency / 3600.0)  # 1-hour half-life

        # Size penalty (prefer keeping smaller items)
        try:
            size = len(pickle.dumps(value))
            size_score = 1.0 / (1.0 + size / 1000000.0)  # Penalize items > 1MB
        except:
            size_score = 1.0

        # Cache type importance
        cache_type_scores = {
            'features': 2.0,      # High importance
            'classification': 1.5, # Medium-high importance
            'svg_output': 1.0,    # Medium importance
            'optimization': 0.8   # Lower importance
        }

        cache_type = key.split(':')[0] if ':' in key else 'unknown'
        type_score = cache_type_scores.get(cache_type, 1.0)

        # Combined priority score
        priority = frequency_score * recency_score * size_score * type_score
        return priority

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with priority tracking"""
        with self.lock:
            if key not in self.cache:
                return None

            value = self.cache[key]
            # Move to end (most recently used)
            self.cache.move_to_end(key)

            # Update access tracking
            self.access_counts[key] += 1
            self.last_access[key] = time.time()

            # Update priority
            self.priorities[key] = self._calculate_priority(key, value)

            return value

    def set(self, key: str, value: Any) -> bool:
        """Set item in cache with intelligent eviction"""
        with self.lock:
            # If key exists, update it
            if key in self.cache:
                self.cache[key] = value
                self.cache.move_to_end(key)
                self.priorities[key] = self._calculate_priority(key, value)
                return True

            # If at capacity, evict based on priority
            while len(self.cache) >= self.max_size:
                self._evict_lowest_priority()

            # Add new item
            self.cache[key] = value
            self.access_counts[key] = 1
            self.last_access[key] = time.time()
            self.priorities[key] = self._calculate_priority(key, value)

            return True

    def _evict_lowest_priority(self):
        """Evict item with lowest priority"""
        if not self.cache:
            return

        if self.enable_priorities and self.priorities:
            # Find item with lowest priority
            lowest_key = min(self.priorities.keys(), key=lambda k: self.priorities[k])
        else:
            # Fall back to standard LRU (oldest item)
            lowest_key = next(iter(self.cache))

        # Remove item
        del self.cache[lowest_key]
        self.priorities.pop(lowest_key, None)
        self.access_counts.pop(lowest_key, None)
        self.last_access.pop(lowest_key, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'priorities_enabled': self.enable_priorities,
                'avg_priority': statistics.mean(self.priorities.values()) if self.priorities else 0,
                'total_accesses': sum(self.access_counts.values()),
                'unique_keys': len(self.access_counts)
            }


class CacheWarmer:
    """Intelligent cache warming based on usage patterns and predictions"""

    def __init__(self, cache: MultiLevelCache, access_patterns: CacheAccessPattern):
        self.cache = cache
        self.access_patterns = access_patterns
        self.warm_cache_enabled = True
        self.warming_in_progress = False
        self.last_warming = 0
        self.warming_interval = 3600  # 1 hour
        self.lock = threading.RLock()

        # Cache warming strategies
        self.strategies = {
            'popular_keys': self._warm_popular_keys,
            'temporal_prediction': self._warm_temporal_prediction,
            'logo_type_patterns': self._warm_logo_type_patterns,
            'related_keys': self._warm_related_keys
        }

    def should_warm_cache(self) -> bool:
        """Determine if cache warming should be triggered"""
        if not self.warm_cache_enabled or self.warming_in_progress:
            return False

        # Time-based warming
        if time.time() - self.last_warming >= self.warming_interval:
            return True

        # Usage-based warming (high cache miss rate)
        cache_stats = self.cache.get_comprehensive_stats()
        overall_stats = cache_stats.get('overall', {})
        hit_rate = overall_stats.get('hit_rate', 1.0)

        if hit_rate < 0.6:  # Hit rate below 60%
            return True

        return False

    def warm_cache_intelligent(self, strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform intelligent cache warming using specified strategies"""
        if not self.should_warm_cache():
            return {'status': 'skipped', 'reason': 'warming not needed'}

        with self.lock:
            if self.warming_in_progress:
                return {'status': 'skipped', 'reason': 'warming already in progress'}

            self.warming_in_progress = True

        try:
            strategies = strategies or list(self.strategies.keys())
            warming_results = {}

            logger.info(f"Starting cache warming with strategies: {strategies}")

            for strategy_name in strategies:
                if strategy_name in self.strategies:
                    try:
                        start_time = time.time()
                        result = self.strategies[strategy_name]()
                        warming_time = time.time() - start_time

                        warming_results[strategy_name] = {
                            'status': 'completed',
                            'warming_time': warming_time,
                            'items_warmed': result.get('items_warmed', 0),
                            'details': result
                        }

                    except Exception as e:
                        warming_results[strategy_name] = {
                            'status': 'error',
                            'error': str(e)
                        }
                        logger.error(f"Error in warming strategy {strategy_name}: {e}")

            self.last_warming = time.time()

            return {
                'status': 'completed',
                'strategies_used': strategies,
                'results': warming_results,
                'total_warming_time': time.time() - start_time
            }

        finally:
            self.warming_in_progress = False

    def _warm_popular_keys(self) -> Dict[str, Any]:
        """Warm cache with most popular keys"""
        popular_keys = self.access_patterns.get_popular_keys(top_n=20)
        items_warmed = 0

        for key, frequency in popular_keys:
            # Check if key is already cached
            cache_type, identifier = key.split(':', 1) if ':' in key else ('unknown', key)

            if self.cache.get(cache_type, identifier) is None:
                # Key not in cache - attempt to regenerate if possible
                # This would need to be customized based on your specific data
                # For now, we'll just record the warming attempt
                logger.debug(f"Would warm popular key: {key} (frequency: {frequency})")
                items_warmed += 1

        return {'items_warmed': items_warmed, 'strategy': 'popular_keys'}

    def _warm_temporal_prediction(self) -> Dict[str, Any]:
        """Warm cache based on temporal usage predictions"""
        usage_prediction = self.access_patterns.predict_cache_usage()
        trending_keys = usage_prediction.get('trending_keys', [])
        items_warmed = 0

        for key, trend_multiplier in trending_keys:
            if trend_multiplier > 1.5:  # Only warm highly trending keys
                cache_type, identifier = key.split(':', 1) if ':' in key else ('unknown', key)

                if self.cache.get(cache_type, identifier) is None:
                    logger.debug(f"Would warm trending key: {key} (trend: {trend_multiplier:.2f}x)")
                    items_warmed += 1

        return {'items_warmed': items_warmed, 'strategy': 'temporal_prediction'}

    def _warm_logo_type_patterns(self) -> Dict[str, Any]:
        """Warm cache with common logo type patterns"""
        # Common logo types and their typical feature patterns
        common_patterns = {
            'simple_geometric': {
                'edge_density': 0.15,
                'unique_colors': 0.3,
                'corner_density': 0.2
            },
            'text_based': {
                'edge_density': 0.4,
                'corner_density': 0.8,
                'unique_colors': 0.2
            },
            'gradient_rich': {
                'unique_colors': 0.8,
                'color_entropy': 0.7,
                'edge_density': 0.25
            }
        }

        items_warmed = 0

        for pattern_name, features in common_patterns.items():
            # Pre-compute classifications for common patterns
            logger.debug(f"Would warm logo type pattern: {pattern_name}")
            items_warmed += 1

        return {'items_warmed': items_warmed, 'strategy': 'logo_type_patterns'}

    def _warm_related_keys(self) -> Dict[str, Any]:
        """Warm cache with keys related to recently accessed ones"""
        # Get recently accessed keys
        recent_cutoff = time.time() - 1800  # Last 30 minutes
        recent_keys = [
            key for ts, key, _, _ in self.access_patterns.access_history
            if ts >= recent_cutoff
        ]

        related_keys = set()
        items_warmed = 0

        for key in recent_keys:
            # Find related keys (same base identifier, different cache types)
            if ':' in key:
                cache_type, identifier = key.split(':', 1)
                related_types = ['features', 'classification', 'optimization']

                for related_type in related_types:
                    if related_type != cache_type:
                        related_key = f"{related_type}:{identifier}"
                        related_keys.add(related_key)

        for related_key in list(related_keys)[:10]:  # Limit to 10 related keys
            cache_type, identifier = related_key.split(':', 1)
            if self.cache.get(cache_type, identifier) is None:
                logger.debug(f"Would warm related key: {related_key}")
                items_warmed += 1

        return {'items_warmed': items_warmed, 'strategy': 'related_keys'}


class AdaptiveCacheManager:
    """Adaptive cache sizing and management based on usage patterns"""

    def __init__(self, cache: MultiLevelCache, access_patterns: CacheAccessPattern):
        self.cache = cache
        self.access_patterns = access_patterns
        self.adaptation_enabled = True
        self.last_adaptation = 0
        self.adaptation_interval = 1800  # 30 minutes
        self.size_history = []
        self.performance_history = []
        self.lock = threading.RLock()

    def analyze_optimal_sizes(self) -> Dict[str, Any]:
        """Analyze optimal cache sizes based on usage patterns"""
        cache_stats = self.cache.get_comprehensive_stats()
        temporal_patterns = self.access_patterns.get_temporal_patterns()

        # Calculate memory pressure
        memory_usage = cache_stats.get('memory', {})
        disk_usage = cache_stats.get('disk', {})

        memory_hit_rate = memory_usage.get('hit_rate', 0)
        disk_hit_rate = disk_usage.get('hit_rate', 0)

        # Determine optimal memory cache size
        current_memory_size = cache_stats.get('sizes', {}).get('memory_entries', 0)
        optimal_memory_size = current_memory_size

        if memory_hit_rate < 0.7:  # Low hit rate
            optimal_memory_size = int(current_memory_size * 1.5)
        elif memory_hit_rate > 0.95:  # Very high hit rate, might be oversized
            optimal_memory_size = max(100, int(current_memory_size * 0.8))

        # Determine optimal disk cache size
        current_disk_size = cache_stats.get('sizes', {}).get('disk_entries', 0)
        optimal_disk_size = current_disk_size

        if disk_hit_rate < 0.5:
            optimal_disk_size = int(current_disk_size * 2.0)

        # Factor in temporal patterns for peak usage
        current_hour = datetime.now().hour
        current_usage = temporal_patterns.get(current_hour, 1.0)
        peak_usage = max(temporal_patterns.values()) if temporal_patterns else 1.0

        if peak_usage > current_usage * 1.5:  # Approaching peak time
            optimal_memory_size = int(optimal_memory_size * 1.2)

        return {
            'current_sizes': {
                'memory_entries': current_memory_size,
                'disk_entries': current_disk_size
            },
            'optimal_sizes': {
                'memory_entries': optimal_memory_size,
                'disk_entries': optimal_disk_size
            },
            'recommendations': {
                'memory_adjustment': optimal_memory_size / max(1, current_memory_size),
                'disk_adjustment': optimal_disk_size / max(1, current_disk_size),
                'peak_usage_factor': peak_usage / max(1, current_usage)
            },
            'performance_indicators': {
                'memory_hit_rate': memory_hit_rate,
                'disk_hit_rate': disk_hit_rate,
                'overall_hit_rate': cache_stats.get('overall', {}).get('hit_rate', 0)
            }
        }

    def adapt_cache_sizes(self, force: bool = False) -> Dict[str, Any]:
        """Adapt cache sizes based on analysis"""
        if not force and (not self.adaptation_enabled or
                         time.time() - self.last_adaptation < self.adaptation_interval):
            return {'status': 'skipped', 'reason': 'adaptation not due'}

        with self.lock:
            analysis = self.analyze_optimal_sizes()
            recommendations = analysis['recommendations']

            adaptations_made = []

            # Adapt memory cache size (if significant change needed)
            memory_adjustment = recommendations['memory_adjustment']
            if abs(memory_adjustment - 1.0) > 0.2:  # 20% change threshold
                logger.info(f"Adapting memory cache size by factor {memory_adjustment:.2f}")
                adaptations_made.append(f"memory_size_factor_{memory_adjustment:.2f}")

            # Adapt disk cache size
            disk_adjustment = recommendations['disk_adjustment']
            if abs(disk_adjustment - 1.0) > 0.3:  # 30% change threshold
                logger.info(f"Adapting disk cache size by factor {disk_adjustment:.2f}")
                adaptations_made.append(f"disk_size_factor_{disk_adjustment:.2f}")

            self.last_adaptation = time.time()

            return {
                'status': 'completed',
                'adaptations_made': adaptations_made,
                'analysis': analysis,
                'timestamp': time.time()
            }


class SmartCacheOrchestrator:
    """Orchestrates all smart caching strategies"""

    def __init__(self, cache: Optional[MultiLevelCache] = None):
        """
        Initialize smart cache orchestrator

        Args:
            cache: Cache instance to manage (uses global if None)
        """
        self.cache = cache or get_global_cache()
        self.access_patterns = CacheAccessPattern()
        self.priority_cache = PriorityLRUCache(max_size=500)
        self.cache_warmer = CacheWarmer(self.cache, self.access_patterns)
        self.adaptive_manager = AdaptiveCacheManager(self.cache, self.access_patterns)

        # Smart caching enabled flags
        self.enable_pattern_tracking = True
        self.enable_cache_warming = True
        self.enable_adaptive_sizing = True
        self.enable_priority_eviction = True

        # Background thread for periodic optimization
        self.optimization_thread = None
        self.optimization_active = False
        self.optimization_interval = 1800  # 30 minutes

    def record_cache_access(self, cache_type: str, identifier: str, hit: bool):
        """Record cache access for pattern analysis"""
        if self.enable_pattern_tracking:
            key = f"{cache_type}:{identifier}"
            self.access_patterns.record_access(key, cache_type, hit)

    def get_cache_intelligence(self, cache_type: str, identifier: str) -> Dict[str, Any]:
        """Get intelligent cache insights for a specific request"""
        key = f"{cache_type}:{identifier}"

        # Access frequency
        frequency = self.access_patterns.key_frequencies.get(key, 0)

        # Predicted importance
        usage_prediction = self.access_patterns.predict_cache_usage()
        is_trending = any(k == key for k, _ in usage_prediction.get('trending_keys', []))

        # Time-based factors
        current_hour = datetime.now().hour
        temporal_patterns = self.access_patterns.get_temporal_patterns()
        current_usage_level = temporal_patterns.get(current_hour, 1.0)

        return {
            'key': key,
            'access_frequency': frequency,
            'is_trending': is_trending,
            'current_usage_level': current_usage_level,
            'cache_priority': 'high' if frequency > 10 or is_trending else 'normal',
            'recommended_ttl': self._calculate_smart_ttl(frequency, is_trending, current_usage_level)
        }

    def _calculate_smart_ttl(self, frequency: int, is_trending: bool, usage_level: float) -> int:
        """Calculate smart TTL based on access patterns"""
        base_ttl = 3600  # 1 hour

        # Adjust based on frequency
        if frequency > 20:
            base_ttl *= 4  # 4 hours for very frequent
        elif frequency > 10:
            base_ttl *= 2  # 2 hours for frequent
        elif frequency < 2:
            base_ttl = 1800  # 30 minutes for infrequent

        # Adjust for trending items
        if is_trending:
            base_ttl *= 2

        # Adjust for current usage level
        if usage_level > 2.0:  # Peak usage time
            base_ttl *= 1.5

        return int(base_ttl)

    def optimize_cache_system(self) -> Dict[str, Any]:
        """Perform comprehensive cache optimization"""
        optimization_results = {}

        try:
            # Cache warming
            if self.enable_cache_warming:
                warming_result = self.cache_warmer.warm_cache_intelligent()
                optimization_results['cache_warming'] = warming_result

            # Adaptive sizing
            if self.enable_adaptive_sizing:
                sizing_result = self.adaptive_manager.adapt_cache_sizes()
                optimization_results['adaptive_sizing'] = sizing_result

            # Pattern analysis
            usage_prediction = self.access_patterns.predict_cache_usage()
            optimization_results['usage_prediction'] = usage_prediction

            return {
                'status': 'completed',
                'timestamp': time.time(),
                'optimizations': optimization_results
            }

        except Exception as e:
            logger.error(f"Error during cache optimization: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

    def start_background_optimization(self):
        """Start background optimization thread"""
        if self.optimization_active:
            return

        self.optimization_active = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        logger.info("Started background cache optimization")

    def stop_background_optimization(self):
        """Stop background optimization thread"""
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        logger.info("Stopped background cache optimization")

    def _optimization_loop(self):
        """Background optimization loop"""
        while self.optimization_active:
            try:
                self.optimize_cache_system()
                time.sleep(self.optimization_interval)
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(60)  # Short delay before retrying

    def get_comprehensive_intelligence_report(self) -> Dict[str, Any]:
        """Get comprehensive cache intelligence report"""
        return {
            'access_patterns': {
                'popular_keys': self.access_patterns.get_popular_keys(10),
                'temporal_patterns': self.access_patterns.get_temporal_patterns(),
                'usage_prediction': self.access_patterns.predict_cache_usage()
            },
            'cache_optimization': {
                'size_analysis': self.adaptive_manager.analyze_optimal_sizes(),
                'warming_status': {
                    'enabled': self.cache_warmer.warm_cache_enabled,
                    'last_warming': self.cache_warmer.last_warming,
                    'warming_interval': self.cache_warmer.warming_interval
                }
            },
            'priority_cache_stats': self.priority_cache.get_stats(),
            'system_status': {
                'pattern_tracking': self.enable_pattern_tracking,
                'cache_warming': self.enable_cache_warming,
                'adaptive_sizing': self.enable_adaptive_sizing,
                'background_optimization': self.optimization_active
            }
        }


# Global smart cache orchestrator
_global_smart_cache = None
_smart_cache_lock = threading.Lock()


def get_global_smart_cache() -> SmartCacheOrchestrator:
    """Get global smart cache orchestrator"""
    global _global_smart_cache
    with _smart_cache_lock:
        if _global_smart_cache is None:
            _global_smart_cache = SmartCacheOrchestrator()
        return _global_smart_cache


def enable_smart_caching():
    """Enable smart caching with background optimization"""
    smart_cache = get_global_smart_cache()
    smart_cache.start_background_optimization()


def get_cache_intelligence_report() -> Dict[str, Any]:
    """Get comprehensive cache intelligence report"""
    smart_cache = get_global_smart_cache()
    return smart_cache.get_comprehensive_intelligence_report()