"""
Multi-Level Caching System for AI SVG Converter

Cache Architecture:
- L1: Memory Cache (TTL + LRU) - Fastest access, limited size
- L2: Disk Cache - Persistent storage, larger capacity
- L3: Redis Cache - Distributed caching, optional

Cache Key Structure:
- Format: {operation}:{hash(inputs)}
- Examples:
  - conversion:md5(image_path + sorted(params))
  - classification:md5(image_path)
  - optimization:md5(features + target_quality)

Eviction Policies:
- Memory: TTL + LRU combination
- Disk: Size-based LRU with 10% batch eviction
- Redis: TTL with Redis native eviction

Cache Invalidation Strategy:
- Time-based: TTL expiration
- Size-based: LRU eviction when limits reached
- Manual: Explicit cache clearing for updated models
"""

from functools import lru_cache
from cachetools import TTLCache, LRUCache
import hashlib
import pickle
import os
import json
import shutil
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union


class MemoryCache:
    """L1 Memory Cache with TTL and LRU eviction"""

    def __init__(self, max_size=1000, ttl=3600):
        self.cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self._lock = threading.RLock()

    def generate_key(self, image_path, params):
        """Generate deterministic cache key"""
        key_data = f"{image_path}:{sorted(params.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key):
        with self._lock:
            if key in self.cache:
                self.stats['hits'] += 1
                return self.cache[key]
            self.stats['misses'] += 1
            return None

    def set(self, key, value):
        with self._lock:
            if len(self.cache) >= self.cache.maxsize:
                self.stats['evictions'] += 1
            self.cache[key] = value

    def get_stats(self):
        with self._lock:
            hit_rate = self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses'])
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'size': len(self.cache)
            }

    def clear(self):
        """Clear all cached items"""
        with self._lock:
            self.cache.clear()

    def warm_cache(self, items: Dict[str, Any]):
        """Pre-populate cache with items"""
        with self._lock:
            for key, value in items.items():
                self.cache[key] = value


class DiskCache:
    """L2 Disk Cache with persistent storage"""

    def __init__(self, cache_dir='cache/disk', max_size_gb=10):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.index_file = self.cache_dir / 'index.json'
        self.index = self._load_index()
        self._lock = threading.RLock()

    def _load_index(self):
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f)

    def get(self, key):
        with self._lock:
            if key in self.index:
                cache_file = self.cache_dir / f"{key}.pkl"
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
            return None

    def set(self, key, value):
        with self._lock:
            # Check size constraints
            current_size = self._get_cache_size()
            if current_size > self.max_size_bytes:
                self._evict_oldest()

            cache_file = self.cache_dir / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)

            self.index[key] = {
                'timestamp': time.time(),
                'size': cache_file.stat().st_size
            }
            self._save_index()

    def _get_cache_size(self):
        return sum(f.stat().st_size for f in self.cache_dir.glob('*.pkl'))

    def _evict_oldest(self):
        # Remove 10% of oldest entries
        sorted_items = sorted(self.index.items(), key=lambda x: x[1]['timestamp'])
        to_remove = len(sorted_items) // 10
        for key, _ in sorted_items[:to_remove]:
            cache_file = self.cache_dir / f"{key}.pkl"
            cache_file.unlink(missing_ok=True)
            del self.index[key]

    def clear(self):
        """Clear all cached items"""
        with self._lock:
            for cache_file in self.cache_dir.glob('*.pkl'):
                cache_file.unlink()
            self.index.clear()
            self._save_index()


try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisCache:
    """L3 Redis Cache for distributed caching"""

    def __init__(self, host='localhost', port=6379, db=0, ttl=7200):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            connection_pool=redis.ConnectionPool(
                host=host, port=port, db=db, max_connections=20
            )
        )
        self.ttl = ttl
        self._lock = threading.RLock()

    def get(self, key):
        try:
            with self._lock:
                data = self.client.get(key)
                if data:
                    return pickle.loads(data)
        except redis.RedisError:
            pass  # Fallback to other caches
        return None

    def set(self, key, value):
        try:
            with self._lock:
                self.client.setex(
                    key,
                    self.ttl,
                    pickle.dumps(value)
                )
        except redis.RedisError:
            pass  # Continue without Redis

    def clear(self):
        """Clear all cached items"""
        try:
            with self._lock:
                self.client.flushdb()
        except redis.RedisError:
            pass


class MultiLevelCache:
    """Unified multi-level cache manager"""

    def __init__(self,
                 memory_config: Optional[Dict] = None,
                 disk_config: Optional[Dict] = None,
                 redis_config: Optional[Dict] = None):

        # Initialize L1 Memory Cache
        memory_cfg = memory_config or {'max_size': 1000, 'ttl': 3600}
        self.memory_cache = MemoryCache(**memory_cfg)

        # Initialize L2 Disk Cache
        disk_cfg = disk_config or {'cache_dir': 'cache/disk', 'max_size_gb': 10}
        self.disk_cache = DiskCache(**disk_cfg)

        # Initialize L3 Redis Cache (optional)
        self.redis_cache = None
        if redis_config and REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache(**redis_config)
            except Exception as e:
                print(f"Redis cache initialization failed: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, checking L1 -> L2 -> L3"""

        # Try L1 Memory Cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Try L2 Disk Cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to L1
            self.memory_cache.set(key, value)
            return value

        # Try L3 Redis Cache
        if self.redis_cache:
            value = self.redis_cache.get(key)
            if value is not None:
                # Promote to L1 and L2
                self.memory_cache.set(key, value)
                self.disk_cache.set(key, value)
                return value

        return None

    def set(self, key: str, value: Any) -> None:
        """Set item in all available cache levels"""

        # Set in L1 Memory Cache
        self.memory_cache.set(key, value)

        # Set in L2 Disk Cache
        self.disk_cache.set(key, value)

        # Set in L3 Redis Cache if available
        if self.redis_cache:
            self.redis_cache.set(key, value)

    def clear(self) -> None:
        """Clear all cache levels"""
        self.memory_cache.clear()
        self.disk_cache.clear()
        if self.redis_cache:
            self.redis_cache.clear()

    def get_stats(self) -> Dict:
        """Get statistics from all cache levels"""
        stats = {
            'memory': self.memory_cache.get_stats(),
            'disk': {
                'size': len(self.disk_cache.index),
                'size_bytes': self.disk_cache._get_cache_size()
            }
        }

        if self.redis_cache:
            try:
                info = self.redis_cache.client.info('memory')
                stats['redis'] = {
                    'used_memory': info.get('used_memory', 0),
                    'used_memory_human': info.get('used_memory_human', '0B')
                }
            except:
                stats['redis'] = {'status': 'unavailable'}

        return stats