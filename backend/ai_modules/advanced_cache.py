#!/usr/bin/env python3
"""
Advanced Multi-Level Cache Architecture for AI-Enhanced SVG Conversion

Implements comprehensive caching for:
- Feature extraction results (expensive CV operations)
- Classification results and confidence scores
- SVG output with quality metadata
- Parameter optimization results
- Quality validation results

Cache Hierarchy:
- L1: Memory cache (fastest access)
- L2: Disk cache (persistent storage)
- L3: Distributed cache (production scaling)
"""

import hashlib
import json
import pickle
import sqlite3
import time
import threading
import gzip
import lz4.frame
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import os
import tempfile
import fcntl
import platform

logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a single cache entry with metadata"""

    def __init__(self, key: str, data: Any, cache_type: str, ttl: Optional[int] = None):
        self.key = key
        self.data = data
        self.cache_type = cache_type
        self.created_at = time.time()
        self.accessed_at = time.time()
        self.access_count = 1
        self.size_bytes = self._calculate_size(data)
        self.ttl = ttl

    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes"""
        try:
            if isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, dict):
                return len(json.dumps(data).encode('utf-8'))
            else:
                return len(pickle.dumps(data))
        except Exception:
            return 0

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self):
        """Update access time and count"""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheStats:
    """Tracks cache performance statistics"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0
        self.total_size = 0
        self.operation_times = []
        self.start_time = time.time()
        self.lock = threading.Lock()

    def record_hit(self, operation_time: float = 0):
        with self.lock:
            self.hits += 1
            if operation_time > 0:
                self.operation_times.append(operation_time)

    def record_miss(self, operation_time: float = 0):
        with self.lock:
            self.misses += 1
            if operation_time > 0:
                self.operation_times.append(operation_time)

    def record_eviction(self):
        with self.lock:
            self.evictions += 1

    def record_invalidation(self):
        with self.lock:
            self.invalidations += 1

    def get_hit_rate(self) -> float:
        with self.lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0

    def get_average_operation_time(self) -> float:
        with self.lock:
            return sum(self.operation_times) / len(self.operation_times) if self.operation_times else 0.0

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.hits + self.misses
            uptime = time.time() - self.start_time
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': self.get_hit_rate(),
                'evictions': self.evictions,
                'invalidations': self.invalidations,
                'total_requests': total_requests,
                'average_operation_time_ms': self.get_average_operation_time() * 1000,
                'uptime_seconds': uptime,
                'requests_per_second': total_requests / uptime if uptime > 0 else 0
            }


class BaseCache(ABC):
    """Abstract base class for cache implementations"""

    def __init__(self, name: str):
        self.name = name
        self.stats = CacheStats()

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries"""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get number of entries in cache"""
        pass


class MemoryCache(BaseCache):
    """High-performance in-memory LRU cache"""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 256, name: str = "memory"):
        super().__init__(name)
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def _evict_lru(self):
        """Evict least recently used items"""
        while len(self.cache) >= self.max_size or self._get_memory_usage() > self.max_memory_bytes:
            if not self.cache:
                break
            key = next(iter(self.cache))
            del self.cache[key]
            self.stats.record_eviction()

    def _get_memory_usage(self) -> int:
        """Calculate current memory usage"""
        return sum(entry.size_bytes for entry in self.cache.values())

    def get(self, key: str) -> Optional[Any]:
        start_time = time.perf_counter()
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    del self.cache[key]
                    self.stats.record_miss(time.perf_counter() - start_time)
                    return None

                # Move to end (most recently used)
                entry.touch()
                self.cache.move_to_end(key)
                self.stats.record_hit(time.perf_counter() - start_time)
                return entry.data

            self.stats.record_miss(time.perf_counter() - start_time)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self.lock:
            try:
                self._evict_lru()
                entry = CacheEntry(key, value, "memory", ttl)
                self.cache[key] = entry
                return True
            except Exception as e:
                logger.error(f"Error setting cache entry {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.record_invalidation()
                return True
            return False

    def clear(self) -> bool:
        with self.lock:
            self.cache.clear()
            return True

    def size(self) -> int:
        with self.lock:
            return len(self.cache)


class DiskCache(BaseCache):
    """Persistent disk-based cache with compression"""

    def __init__(self, cache_dir: str = ".ai_cache", max_size_gb: int = 5,
                 compression: str = "lz4", name: str = "disk"):
        super().__init__(name)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.compression = compression
        self.index_file = self.cache_dir / "cache_index.db"
        self.lock = threading.RLock()
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for cache index"""
        with sqlite3.connect(self.index_file) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    cache_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    size_bytes INTEGER NOT NULL,
                    ttl INTEGER,
                    metadata TEXT
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cache_type ON cache_entries(cache_type)')

    def _get_filename(self, key: str) -> str:
        """Generate filename for cache entry"""
        hash_obj = hashlib.sha256(key.encode())
        return f"{hash_obj.hexdigest()}.cache"

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data based on selected compression method"""
        if self.compression == "gzip":
            return gzip.compress(data)
        elif self.compression == "lz4":
            return lz4.frame.compress(data)
        else:
            return data

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data based on selected compression method"""
        if self.compression == "gzip":
            return gzip.decompress(data)
        elif self.compression == "lz4":
            return lz4.frame.decompress(data)
        else:
            return data

    def _evict_lru(self):
        """Evict least recently used files to stay under size limit"""
        while self._get_total_size() > self.max_size_bytes:
            with sqlite3.connect(self.index_file) as conn:
                cursor = conn.execute(
                    'SELECT key, filename FROM cache_entries ORDER BY accessed_at ASC LIMIT 1'
                )
                row = cursor.fetchone()
                if not row:
                    break

                key, filename = row
                self.delete(key)

    def _get_total_size(self) -> int:
        """Get total size of cache in bytes"""
        with sqlite3.connect(self.index_file) as conn:
            cursor = conn.execute('SELECT SUM(size_bytes) FROM cache_entries')
            result = cursor.fetchone()
            return result[0] or 0

    def get(self, key: str) -> Optional[Any]:
        start_time = time.perf_counter()
        with self.lock:
            try:
                with sqlite3.connect(self.index_file) as conn:
                    cursor = conn.execute(
                        'SELECT filename, ttl, created_at FROM cache_entries WHERE key = ?',
                        (key,)
                    )
                    row = cursor.fetchone()
                    if not row:
                        self.stats.record_miss(time.perf_counter() - start_time)
                        return None

                    filename, ttl, created_at = row

                    # Check TTL
                    if ttl and time.time() - created_at > ttl:
                        self.delete(key)
                        self.stats.record_miss(time.perf_counter() - start_time)
                        return None

                    # Read and decompress data
                    file_path = self.cache_dir / filename
                    if not file_path.exists():
                        self.delete(key)
                        self.stats.record_miss(time.perf_counter() - start_time)
                        return None

                    with open(file_path, 'rb') as f:
                        compressed_data = f.read()

                    data_bytes = self._decompress_data(compressed_data)
                    data = pickle.loads(data_bytes)

                    # Update access time
                    conn.execute(
                        'UPDATE cache_entries SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?',
                        (time.time(), key)
                    )

                    self.stats.record_hit(time.perf_counter() - start_time)
                    return data

            except Exception as e:
                logger.error(f"Error getting cache entry {key}: {e}")
                self.stats.record_miss(time.perf_counter() - start_time)
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self.lock:
            try:
                # Serialize and compress data
                data_bytes = pickle.dumps(value)
                compressed_data = self._compress_data(data_bytes)

                # Generate filename and save data
                filename = self._get_filename(key)
                file_path = self.cache_dir / filename

                with open(file_path, 'wb') as f:
                    f.write(compressed_data)

                # Update database
                with sqlite3.connect(self.index_file) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO cache_entries
                        (key, filename, cache_type, created_at, accessed_at, size_bytes, ttl)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (key, filename, "disk", time.time(), time.time(), len(compressed_data), ttl))

                # Check if eviction needed
                self._evict_lru()

                return True

            except Exception as e:
                logger.error(f"Error setting cache entry {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        with self.lock:
            try:
                with sqlite3.connect(self.index_file) as conn:
                    cursor = conn.execute('SELECT filename FROM cache_entries WHERE key = ?', (key,))
                    row = cursor.fetchone()
                    if row:
                        filename = row[0]
                        file_path = self.cache_dir / filename
                        if file_path.exists():
                            file_path.unlink()

                        conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                        self.stats.record_invalidation()
                        return True

                return False

            except Exception as e:
                logger.error(f"Error deleting cache entry {key}: {e}")
                return False

    def clear(self) -> bool:
        with self.lock:
            try:
                # Remove all cache files
                for file_path in self.cache_dir.glob("*.cache"):
                    file_path.unlink()

                # Clear database
                with sqlite3.connect(self.index_file) as conn:
                    conn.execute('DELETE FROM cache_entries')

                return True

            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                return False

    def size(self) -> int:
        with sqlite3.connect(self.index_file) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM cache_entries')
            return cursor.fetchone()[0]


class DistributedCache(BaseCache):
    """Distributed cache coordination for production scaling"""

    def __init__(self, nodes: List[str] = None, sync_interval: int = 300, name: str = "distributed"):
        super().__init__(name)
        self.nodes = nodes or []
        self.sync_interval = sync_interval
        self.local_cache = MemoryCache(max_size=500, name="distributed_local")
        self.coordination_file = Path(".cache_coordination.json")
        self.last_sync = 0
        self.lock = threading.RLock()

    def _should_sync(self) -> bool:
        """Check if synchronization is needed"""
        return time.time() - self.last_sync > self.sync_interval

    def _sync_with_nodes(self):
        """Synchronize cache state with other nodes"""
        # This is a simplified implementation
        # In production, this would use Redis, Hazelcast, or similar
        if not self.coordination_file.exists():
            return

        try:
            with open(self.coordination_file, 'r') as f:
                coordination_data = json.load(f)

            # Process invalidation messages from other nodes
            invalidations = coordination_data.get('invalidations', [])
            for key in invalidations:
                self.local_cache.delete(key)

            self.last_sync = time.time()

        except Exception as e:
            logger.error(f"Error syncing with nodes: {e}")

    def get(self, key: str) -> Optional[Any]:
        if self._should_sync():
            self._sync_with_nodes()
        return self.local_cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        return self.local_cache.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        # Broadcast invalidation to other nodes
        self._broadcast_invalidation(key)
        return self.local_cache.delete(key)

    def _broadcast_invalidation(self, key: str):
        """Broadcast key invalidation to other nodes"""
        try:
            if self.coordination_file.exists():
                with open(self.coordination_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {'invalidations': []}

            data['invalidations'].append(key)
            data['timestamp'] = time.time()

            with open(self.coordination_file, 'w') as f:
                json.dump(data, f)

        except Exception as e:
            logger.error(f"Error broadcasting invalidation: {e}")

    def clear(self) -> bool:
        return self.local_cache.clear()

    def size(self) -> int:
        return self.local_cache.size()


class MultiLevelCache:
    """Orchestrates multi-level cache hierarchy"""

    def __init__(self,
                 memory_config: Optional[Dict] = None,
                 disk_config: Optional[Dict] = None,
                 distributed_config: Optional[Dict] = None):
        """
        Initialize multi-level cache

        Args:
            memory_config: Configuration for memory cache layer
            disk_config: Configuration for disk cache layer
            distributed_config: Configuration for distributed cache layer
        """
        # Initialize cache layers
        self.memory_cache = MemoryCache(**(memory_config or {}))
        self.disk_cache = DiskCache(**(disk_config or {}))
        self.distributed_cache = DistributedCache(**(distributed_config or {})) if distributed_config else None

        # Overall statistics
        self.stats = CacheStats()
        self.lock = threading.RLock()

        logger.info("Initialized multi-level cache system")

    def _generate_cache_key(self, cache_type: str, identifier: str, params: Optional[Dict] = None) -> str:
        """Generate standardized cache key"""
        key_parts = [cache_type, identifier]
        if params:
            params_str = json.dumps(params, sort_keys=True)
            key_parts.append(hashlib.md5(params_str.encode()).hexdigest())
        return ":".join(key_parts)

    def get(self, cache_type: str, identifier: str, params: Optional[Dict] = None) -> Optional[Any]:
        """
        Get value from cache hierarchy (L1 -> L2 -> L3)

        Args:
            cache_type: Type of cache entry (features, classification, svg, etc.)
            identifier: Unique identifier for the entry
            params: Optional parameters for cache key generation

        Returns:
            Cached value or None
        """
        start_time = time.perf_counter()
        key = self._generate_cache_key(cache_type, identifier, params)

        with self.lock:
            # Try L1 (Memory)
            result = self.memory_cache.get(key)
            if result is not None:
                self.stats.record_hit(time.perf_counter() - start_time)
                logger.debug(f"Cache L1 hit: {cache_type}:{identifier}")
                return result

            # Try L2 (Disk)
            result = self.disk_cache.get(key)
            if result is not None:
                # Promote to L1
                self.memory_cache.set(key, result)
                self.stats.record_hit(time.perf_counter() - start_time)
                logger.debug(f"Cache L2 hit: {cache_type}:{identifier}")
                return result

            # Try L3 (Distributed) if available
            if self.distributed_cache:
                result = self.distributed_cache.get(key)
                if result is not None:
                    # Promote to L1 and L2
                    self.memory_cache.set(key, result)
                    self.disk_cache.set(key, result)
                    self.stats.record_hit(time.perf_counter() - start_time)
                    logger.debug(f"Cache L3 hit: {cache_type}:{identifier}")
                    return result

            self.stats.record_miss(time.perf_counter() - start_time)
            logger.debug(f"Cache miss: {cache_type}:{identifier}")
            return None

    def set(self, cache_type: str, identifier: str, value: Any,
            params: Optional[Dict] = None, ttl: Optional[int] = None) -> bool:
        """
        Set value in all cache levels

        Args:
            cache_type: Type of cache entry
            identifier: Unique identifier
            value: Value to cache
            params: Optional parameters
            ttl: Time to live in seconds

        Returns:
            Success status
        """
        key = self._generate_cache_key(cache_type, identifier, params)

        with self.lock:
            success = True

            # Set in all levels
            if not self.memory_cache.set(key, value, ttl):
                success = False

            if not self.disk_cache.set(key, value, ttl):
                success = False

            if self.distributed_cache and not self.distributed_cache.set(key, value, ttl):
                success = False

            if success:
                logger.debug(f"Cached: {cache_type}:{identifier}")
            else:
                logger.error(f"Failed to cache: {cache_type}:{identifier}")

            return success

    def invalidate(self, cache_type: str, identifier: str, params: Optional[Dict] = None) -> bool:
        """
        Invalidate entry from all cache levels

        Args:
            cache_type: Type of cache entry
            identifier: Unique identifier
            params: Optional parameters

        Returns:
            Success status
        """
        key = self._generate_cache_key(cache_type, identifier, params)

        with self.lock:
            success = True

            if not self.memory_cache.delete(key):
                success = False

            if not self.disk_cache.delete(key):
                success = False

            if self.distributed_cache and not self.distributed_cache.delete(key):
                success = False

            self.stats.record_invalidation()
            logger.debug(f"Invalidated: {cache_type}:{identifier}")
            return success

    def clear_by_type(self, cache_type: str) -> int:
        """
        Clear all entries of a specific type

        Args:
            cache_type: Type to clear

        Returns:
            Number of entries cleared
        """
        # This is a simplified implementation
        # In production, would iterate through keys by type
        logger.info(f"Clearing cache type: {cache_type}")
        return 0

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all cache levels"""
        return {
            'overall': self.stats.get_summary(),
            'memory': self.memory_cache.stats.get_summary(),
            'disk': self.disk_cache.stats.get_summary(),
            'distributed': self.distributed_cache.stats.get_summary() if self.distributed_cache else None,
            'sizes': {
                'memory_entries': self.memory_cache.size(),
                'disk_entries': self.disk_cache.size(),
                'distributed_entries': self.distributed_cache.size() if self.distributed_cache else 0
            }
        }


# Singleton instance for global use
_global_cache = None
_cache_lock = threading.Lock()


def get_global_cache() -> MultiLevelCache:
    """Get or create global cache instance"""
    global _global_cache
    with _cache_lock:
        if _global_cache is None:
            _global_cache = MultiLevelCache()
        return _global_cache


def configure_global_cache(memory_config: Optional[Dict] = None,
                          disk_config: Optional[Dict] = None,
                          distributed_config: Optional[Dict] = None):
    """Configure global cache with custom settings"""
    global _global_cache
    with _cache_lock:
        _global_cache = MultiLevelCache(memory_config, disk_config, distributed_config)
        logger.info("Configured global cache with custom settings")


# Cache invalidation strategies
class CacheInvalidationStrategy:
    """Manages cache invalidation based on various strategies"""

    def __init__(self, cache: MultiLevelCache):
        self.cache = cache

    def invalidate_by_image_modification(self, image_path: str):
        """Invalidate cache entries when source image is modified"""
        # Get file modification time
        try:
            mtime = os.path.getmtime(image_path)
            file_hash = hashlib.md5(f"{image_path}:{mtime}".encode()).hexdigest()

            # Invalidate related entries
            cache_types = ['features', 'classification', 'svg_output', 'quality']
            for cache_type in cache_types:
                self.cache.invalidate(cache_type, file_hash)

        except Exception as e:
            logger.error(f"Error invalidating cache for {image_path}: {e}")

    def invalidate_by_parameter_change(self, params: Dict):
        """Invalidate cache entries when parameters change"""
        # This would invalidate SVG outputs that depend on changed parameters
        params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        self.cache.invalidate('parameters', params_hash)

    def cleanup_expired_entries(self):
        """Clean up expired cache entries"""
        # This would be run periodically
        logger.info("Cleaning up expired cache entries")