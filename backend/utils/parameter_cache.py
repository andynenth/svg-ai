#!/usr/bin/env python3
"""
Parameter caching system for optimized conversions.

This module caches successful parameter combinations and provides
similarity-based lookup for fast parameter selection.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class ParameterCache:
    """Cache successful parameter combinations for reuse."""

    def __init__(self, cache_dir: str = ".parameter_cache"):
        """
        Initialize the parameter cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.cache_file = self.cache_dir / "parameter_cache.json"
        self.feature_index_file = self.cache_dir / "feature_index.pkl"

        # Load existing cache
        self.cache = self._load_cache()
        self.feature_index = self._load_feature_index()

        # In-memory LRU cache for fast lookups
        self.memory_cache = {}
        self.max_memory_entries = 100

    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
                logger.warning(f"Failed to load parameter cache from {self.cache_file}: {e}")
                logger.info("Parameter cache will be recreated from scratch")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def _load_feature_index(self) -> Dict[str, Any]:
        """Load feature index from disk."""
        if self.feature_index_file.exists():
            try:
                with open(self.feature_index_file, 'rb') as f:
                    return pickle.load(f)
            except (FileNotFoundError, pickle.PickleError, PermissionError) as e:
                logger.warning(f"Failed to load feature index from {self.feature_index_file}: {e}")
                logger.info("Feature index will be rebuilt as parameters are cached")
                return {}
        return {}

    def _save_feature_index(self):
        """Save feature index to disk."""
        with open(self.feature_index_file, 'wb') as f:
            pickle.dump(self.feature_index, f)

    def _compute_image_hash(self, image_path: str) -> str:
        """
        Compute hash of image file.

        Args:
            image_path: Path to image

        Returns:
            Hash string
        """
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _extract_cache_features(self, image_features: Dict) -> np.ndarray:
        """
        Extract features for cache similarity matching.

        Args:
            image_features: Dictionary of image features

        Returns:
            Feature vector
        """
        # Key features for similarity
        feature_names = [
            'width', 'height', 'unique_colors', 'color_complexity',
            'mean_r', 'mean_g', 'mean_b',
            'std_r', 'std_g', 'std_b',
            'transparency_ratio', 'mean_edge_strength',
            'type_simple', 'type_text', 'type_gradient', 'type_complex'
        ]

        features = []
        for name in feature_names:
            value = image_features.get(name, 0)
            features.append(value)

        return np.array(features)

    def add_entry(self, image_path: str, image_features: Dict,
                  parameters: Dict, quality_metrics: Dict):
        """
        Add successful parameter combination to cache.

        Args:
            image_path: Path to image
            image_features: Extracted image features
            parameters: VTracer parameters used
            quality_metrics: Quality metrics achieved (SSIM, size ratio)
        """
        # Compute image hash
        image_hash = self._compute_image_hash(image_path)

        # Create cache entry
        entry = {
            'image_path': str(image_path),
            'image_hash': image_hash,
            'timestamp': datetime.now().isoformat(),
            'features': image_features,
            'parameters': parameters,
            'metrics': quality_metrics
        }

        # Add to cache
        cache_key = f"{image_hash}_{quality_metrics.get('ssim', 0):.3f}"
        self.cache[cache_key] = entry

        # Add to feature index for similarity search
        feature_vector = self._extract_cache_features(image_features)
        self.feature_index[cache_key] = feature_vector

        # Save to disk
        self._save_cache()
        self._save_feature_index()

        print(f"  ✅ Cached parameters for {Path(image_path).name}")

    def find_similar(self, image_features: Dict, similarity_threshold: float = 0.9,
                    max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar images in cache.

        Args:
            image_features: Features of query image
            similarity_threshold: Minimum similarity (0-1)
            max_results: Maximum number of results

        Returns:
            List of similar cache entries
        """
        if not self.feature_index:
            return []

        # Extract query features
        query_features = self._extract_cache_features(image_features)

        # Normalize for cosine similarity
        query_norm = query_features / (np.linalg.norm(query_features) + 1e-8)

        similarities = []

        for cache_key, cached_features in self.feature_index.items():
            # Normalize cached features
            cached_norm = cached_features / (np.linalg.norm(cached_features) + 1e-8)

            # Cosine similarity
            similarity = np.dot(query_norm, cached_norm)

            if similarity >= similarity_threshold:
                similarities.append((cache_key, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        results = []
        for cache_key, similarity in similarities[:max_results]:
            if cache_key in self.cache:
                entry = self.cache[cache_key].copy()
                entry['similarity'] = similarity
                results.append(entry)

        return results

    def get_exact_match(self, image_path: str) -> Optional[Dict]:
        """
        Get exact match from cache.

        Args:
            image_path: Path to image

        Returns:
            Cache entry if found
        """
        # Check memory cache first
        if image_path in self.memory_cache:
            return self.memory_cache[image_path]

        # Compute hash
        image_hash = self._compute_image_hash(image_path)

        # Look for exact match
        for cache_key, entry in self.cache.items():
            if entry.get('image_hash') == image_hash:
                # Add to memory cache
                self.memory_cache[image_path] = entry

                # Limit memory cache size
                if len(self.memory_cache) > self.max_memory_entries:
                    # Remove oldest entry (simple FIFO)
                    oldest = list(self.memory_cache.keys())[0]
                    del self.memory_cache[oldest]

                return entry

        return None

    def get_best_parameters(self, image_path: str, image_features: Dict) -> Optional[Dict]:
        """
        Get best parameters from cache for an image.

        Args:
            image_path: Path to image
            image_features: Extracted image features

        Returns:
            Best parameters if found
        """
        # Try exact match first
        exact = self.get_exact_match(image_path)
        if exact:
            print(f"  📎 Cache hit: exact match (SSIM: {exact['metrics']['ssim']:.3f})")
            return exact['parameters']

        # Try similarity search
        similar = self.find_similar(image_features, similarity_threshold=0.85)

        if similar:
            # Use parameters from most similar image with best SSIM
            best = max(similar, key=lambda x: x['metrics']['ssim'])
            print(f"  📎 Cache hit: similar image (similarity: {best['similarity']:.2f}, "
                  f"SSIM: {best['metrics']['ssim']:.3f})")
            return best['parameters']

        print(f"  ❌ Cache miss: no similar images found")
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {
                'total_entries': 0,
                'unique_images': 0,
                'avg_ssim': 0,
                'cache_size_kb': 0
            }

        # Calculate statistics
        unique_hashes = set(entry['image_hash'] for entry in self.cache.values())
        ssim_values = [entry['metrics']['ssim'] for entry in self.cache.values()]

        # Calculate cache size
        cache_size = 0
        if self.cache_file.exists():
            cache_size = self.cache_file.stat().st_size / 1024

        if self.feature_index_file.exists():
            cache_size += self.feature_index_file.stat().st_size / 1024

        return {
            'total_entries': len(self.cache),
            'unique_images': len(unique_hashes),
            'avg_ssim': np.mean(ssim_values) if ssim_values else 0,
            'cache_size_kb': cache_size,
            'memory_entries': len(self.memory_cache)
        }

    def clear_cache(self):
        """Clear all cache data."""
        self.cache = {}
        self.feature_index = {}
        self.memory_cache = {}

        # Remove cache files
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.feature_index_file.exists():
            self.feature_index_file.unlink()

        print("✅ Cache cleared")


def test_cache():
    """Test the caching system."""
    cache = ParameterCache()

    # Test data
    test_features = {
        'width': 256,
        'height': 256,
        'unique_colors': 100,
        'color_complexity': 0.5,
        'mean_r': 128,
        'mean_g': 128,
        'mean_b': 128,
        'std_r': 50,
        'std_g': 50,
        'std_b': 50,
        'transparency_ratio': 0.1,
        'mean_edge_strength': 0.3,
        'type_simple': 1,
        'type_text': 0,
        'type_gradient': 0,
        'type_complex': 0
    }

    test_params = {
        'color_precision': 4,
        'layer_difference': 8,
        'corner_threshold': 30,
        'length_threshold': 5.0,
        'max_iterations': 10,
        'splice_threshold': 45,
        'path_precision': 6
    }

    test_metrics = {
        'ssim': 0.98,
        'size_ratio': 0.5
    }

    # Test adding entry
    test_image = "data/logos/simple_geometric/circle_00.png"
    if Path(test_image).exists():
        cache.add_entry(test_image, test_features, test_params, test_metrics)

        # Test exact match
        exact = cache.get_exact_match(test_image)
        if exact:
            print(f"✅ Exact match found: SSIM={exact['metrics']['ssim']}")

        # Test similarity search
        similar_features = test_features.copy()
        similar_features['unique_colors'] = 95  # Slightly different
        similar = cache.find_similar(similar_features)
        if similar:
            print(f"✅ Found {len(similar)} similar entries")

    # Print statistics
    stats = cache.get_statistics()
    print("\nCache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_cache()