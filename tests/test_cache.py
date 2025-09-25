"""
Tests for caching system.
"""

import pytest
import os
import time
from pathlib import Path
import tempfile

from utils.cache import ConversionCache, MemoryCache, HybridCache


class TestConversionCache:
    """Tests for file-based cache."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create cache instance with temp directory."""
        return ConversionCache(cache_dir=str(tmp_path / "cache"))

    @pytest.fixture
    def test_image(self, tmp_path):
        """Create a test image file."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"fake png content")
        return str(img_path)

    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.cache_dir.exists()
        assert cache.index_file.exists() or len(cache.index) == 0

    def test_cache_set_and_get(self, cache, test_image):
        """Test setting and getting cache entries."""
        svg_content = "<svg>test</svg>"
        converter_name = "TestConverter"
        params = {"precision": 6}

        # Set cache entry
        cache.set(test_image, converter_name, svg_content, params)

        # Get cache entry
        result = cache.get(test_image, converter_name, params)
        assert result == svg_content

    def test_cache_miss(self, cache, test_image):
        """Test cache miss behavior."""
        result = cache.get(test_image, "NonExistent", {})
        assert result is None

    def test_cache_key_generation(self, cache, test_image):
        """Test cache key uniqueness."""
        key1 = cache._get_cache_key(test_image, "Conv1", {"p": 1})
        key2 = cache._get_cache_key(test_image, "Conv1", {"p": 2})
        key3 = cache._get_cache_key(test_image, "Conv2", {"p": 1})

        # Different parameters should give different keys
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_age_expiry(self, cache, test_image):
        """Test cache entry expiration."""
        from datetime import datetime, timedelta

        svg_content = "<svg>old</svg>"
        cache.set(test_image, "TestConv", svg_content)

        # Manually set old timestamp
        key = cache._get_cache_key(test_image, "TestConv", {})
        old_time = datetime.now() - timedelta(days=10)
        cache.index[key]['created'] = old_time.isoformat()

        # Should return None due to age
        result = cache.get(test_image, "TestConv", {})
        assert result is None

    def test_clear_old_entries(self, cache, test_image):
        """Test clearing old cache entries."""
        from datetime import datetime, timedelta

        # Add entries with different ages
        cache.set(test_image, "Conv1", "<svg>1</svg>")

        # Make one old
        key = cache._get_cache_key(test_image, "Conv1", {})
        old_time = datetime.now() - timedelta(days=10)
        cache.index[key]['created'] = old_time.isoformat()

        # Add a new one
        cache.set(test_image, "Conv2", "<svg>2</svg>")

        # Clear old entries
        cache.clear_old_entries()

        # Old should be gone, new should remain
        assert cache.get(test_image, "Conv1", {}) is None
        assert cache.get(test_image, "Conv2", {}) == "<svg>2</svg>"

    def test_cache_stats(self, cache, test_image):
        """Test cache statistics."""
        cache.set(test_image, "Conv1", "<svg>test</svg>")
        cache.set(test_image, "Conv2", "<svg>another test</svg>")

        stats = cache.get_stats()
        assert stats['entries'] == 2
        assert stats['total_size_kb'] > 0


class TestMemoryCache:
    """Tests for in-memory cache."""

    @pytest.fixture
    def cache(self):
        """Create memory cache instance."""
        return MemoryCache(max_size=3)

    @pytest.fixture
    def test_image(self, tmp_path):
        """Create test image."""
        img_path = tmp_path / "test.png"
        img_path.write_text("test")
        return str(img_path)

    def test_memory_cache_basic(self, cache, test_image):
        """Test basic memory cache operations."""
        svg = "<svg>content</svg>"
        converter = "TestConv"

        # Set and get
        cache.set(test_image, converter, svg)
        result = cache.get(test_image, converter)
        assert result == svg

    def test_memory_cache_lru(self, cache, tmp_path):
        """Test LRU eviction."""
        # Create multiple test files
        files = []
        for i in range(4):
            f = tmp_path / f"test{i}.png"
            f.write_text(f"test{i}")
            files.append(str(f))

        # Add 4 items to cache with max_size=3
        for i, f in enumerate(files):
            cache.set(f, "Conv", f"<svg>{i}</svg>")

        # First should be evicted
        assert cache.get(files[0], "Conv") is None
        # Others should be present
        assert cache.get(files[1], "Conv") == "<svg>1</svg>"
        assert cache.get(files[2], "Conv") == "<svg>2</svg>"
        assert cache.get(files[3], "Conv") == "<svg>3</svg>"

    def test_memory_cache_stats(self, cache, test_image):
        """Test cache statistics."""
        cache.set(test_image, "Conv", "<svg>test</svg>")

        # Get hit
        cache.get(test_image, "Conv")
        # Get miss
        cache.get(test_image, "OtherConv")

        stats = cache.get_stats()
        assert stats['entries'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5

    def test_memory_cache_clear(self, cache, test_image):
        """Test cache clearing."""
        cache.set(test_image, "Conv", "<svg>test</svg>")
        assert len(cache.cache) == 1

        cache.clear()
        assert len(cache.cache) == 0
        assert cache.get(test_image, "Conv") is None


class TestHybridCache:
    """Tests for hybrid cache."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create hybrid cache instance."""
        return HybridCache(
            cache_dir=str(tmp_path / "cache"),
            memory_size=5,
            disk_max_age_days=7
        )

    @pytest.fixture
    def test_image(self, tmp_path):
        """Create test image."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"test content")
        return str(img_path)

    def test_hybrid_cache_memory_first(self, cache, test_image):
        """Test that memory cache is checked first."""
        svg = "<svg>test</svg>"
        converter = "TestConv"

        # Set in cache (goes to both memory and disk)
        cache.set(test_image, converter, svg)

        # Should get from memory (faster)
        result = cache.get(test_image, converter)
        assert result == svg

        # Check memory hit
        assert cache.memory_cache.hits == 1

    def test_hybrid_cache_disk_fallback(self, cache, test_image):
        """Test disk cache fallback when not in memory."""
        svg = "<svg>test</svg>"
        converter = "TestConv"

        # Set in disk only
        cache.disk_cache.set(test_image, converter, svg)

        # Get should check memory (miss) then disk (hit)
        result = cache.get(test_image, converter)
        assert result == svg

        # Should now be in memory cache too
        assert cache.memory_cache.get(test_image, converter) == svg

    def test_hybrid_cache_stats(self, cache, test_image):
        """Test combined statistics."""
        cache.set(test_image, "Conv", "<svg>test</svg>")

        stats = cache.get_stats()
        assert 'memory' in stats
        assert 'disk' in stats
        assert stats['memory']['entries'] == 1
        assert stats['disk']['entries'] == 1