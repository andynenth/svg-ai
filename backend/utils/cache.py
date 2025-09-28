"""
Caching system for PNG to SVG conversions.
"""

import hashlib
import json
import pickle
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class ConversionCache:
    """File-based cache for conversion results."""

    def __init__(self, cache_dir: str = 'cache', max_age_days: int = 7):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cache files
            max_age_days: Maximum age of cache entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_age = timedelta(days=max_age_days)
        self.index_file = self.cache_dir / 'index.json'
        self.index = self._load_index()
        self.lock = threading.Lock()

    def _load_index(self) -> Dict[str, Any]:
        """Load cache index."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
                logger.error(f"Failed to load cache index from {self.index_file}: {e}")
                logger.info("Cache index will be recreated from scratch")
                return {}
        return {}

    def _save_index(self):
        """Save cache index."""
        with self.lock:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2, default=str)

    def _get_cache_key(self, image_path: str, converter_name: str,
                      converter_params: Dict[str, Any]) -> str:
        """Generate unique cache key."""
        # Create hash from file content and converter settings
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        params_str = json.dumps(converter_params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()

        return f"{file_hash}_{converter_name}_{params_hash}"

    def get(self, image_path: str, converter_name: str,
           converter_params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get cached conversion result.

        Args:
            image_path: Path to input image
            converter_name: Name of converter
            converter_params: Converter parameters

        Returns:
            Cached SVG content or None
        """
        if converter_params is None:
            converter_params = {}

        cache_key = self._get_cache_key(image_path, converter_name, converter_params)

        if cache_key in self.index:
            entry = self.index[cache_key]
            # Check age
            created = datetime.fromisoformat(entry['created'])
            if datetime.now() - created < self.max_age:
                cache_file = self.cache_dir / entry['file']
                if cache_file.exists():
                    with open(cache_file, 'r') as f:
                        return f.read()

        return None

    def set(self, image_path: str, converter_name: str, svg_content: str,
           converter_params: Optional[Dict[str, Any]] = None):
        """
        Cache conversion result.

        Args:
            image_path: Path to input image
            converter_name: Name of converter
            svg_content: SVG content to cache
            converter_params: Converter parameters
        """
        if converter_params is None:
            converter_params = {}

        cache_key = self._get_cache_key(image_path, converter_name, converter_params)
        cache_file = self.cache_dir / f"{cache_key}.svg"

        # Save SVG content
        with open(cache_file, 'w') as f:
            f.write(svg_content)

        # Update index
        self.index[cache_key] = {
            'file': cache_file.name,
            'image_path': image_path,
            'converter': converter_name,
            'params': converter_params,
            'created': datetime.now().isoformat(),
            'size': len(svg_content)
        }
        self._save_index()

    def clear_old_entries(self):
        """Remove cache entries older than max_age."""
        now = datetime.now()
        to_remove = []

        for key, entry in self.index.items():
            created = datetime.fromisoformat(entry['created'])
            if now - created > self.max_age:
                to_remove.append(key)
                # Remove file
                cache_file = self.cache_dir / entry['file']
                if cache_file.exists():
                    cache_file.unlink()

        # Update index
        for key in to_remove:
            del self.index[key]

        if to_remove:
            self._save_index()
            print(f"🧹 Cleared {len(to_remove)} old cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = 0
        for entry in self.index.values():
            total_size += entry.get('size', 0)

        return {
            'entries': len(self.index),
            'total_size_kb': total_size / 1024,
            'cache_dir': str(self.cache_dir)
        }


class MemoryCache:
    """In-memory LRU cache for fast access."""

    def __init__(self, max_size: int = 100):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries
        """
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, image_path: str, converter_name: str) -> str:
        """Create cache key."""
        # Use file modification time for cache invalidation
        mtime = os.path.getmtime(image_path)
        return f"{image_path}_{converter_name}_{mtime}"

    def get(self, image_path: str, converter_name: str) -> Optional[str]:
        """Get from memory cache."""
        key = self._make_key(image_path, converter_name)

        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                return self.cache[key]

            self.misses += 1
            return None

    def set(self, image_path: str, converter_name: str, svg_content: str):
        """Add to memory cache."""
        key = self._make_key(image_path, converter_name)

        with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]

            # Add new entry
            self.cache[key] = svg_content
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

    def clear(self):
        """Clear memory cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.hits / max(1, self.hits + self.misses)
        return {
            'entries': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'memory_usage_kb': sum(len(v) for v in self.cache.values()) / 1024
        }


class HybridCache:
    """Combined memory and disk cache."""

    def __init__(self, cache_dir: str = 'cache',
                memory_size: int = 100,
                disk_max_age_days: int = 7):
        """
        Initialize hybrid cache.

        Args:
            cache_dir: Directory for disk cache
            memory_size: Max entries in memory
            disk_max_age_days: Max age for disk cache
        """
        self.memory_cache = MemoryCache(memory_size)
        self.disk_cache = ConversionCache(cache_dir, disk_max_age_days)

    def get(self, image_path: str, converter_name: str,
           converter_params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get from cache (memory first, then disk)."""
        # Try memory cache first
        result = self.memory_cache.get(image_path, converter_name)
        if result:
            return result

        # Try disk cache
        result = self.disk_cache.get(image_path, converter_name, converter_params)
        if result:
            # Add to memory cache for faster future access
            self.memory_cache.set(image_path, converter_name, result)

        return result

    def set(self, image_path: str, converter_name: str, svg_content: str,
           converter_params: Optional[Dict[str, Any]] = None):
        """Save to both memory and disk cache."""
        self.memory_cache.set(image_path, converter_name, svg_content)
        self.disk_cache.set(image_path, converter_name, svg_content, converter_params)

    def clear(self):
        """Clear all caches."""
        self.memory_cache.clear()
        self.disk_cache.clear_old_entries()

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            'memory': self.memory_cache.get_stats(),
            'disk': self.disk_cache.get_stats()
        }