# backend/ai_modules/management/model_cache.py
import time
import logging
from typing import Dict, Any, Callable, Optional

class ModelCache:
    def __init__(self, max_memory_mb: int = 400):
        self.cache = {}
        self.access_times = {}
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0

    def get_model(self, model_name: str, loader_func: Optional[Callable] = None):
        """Get model from cache or load if needed"""
        if model_name in self.cache:
            self.access_times[model_name] = time.time()
            logging.debug(f"📋 Cache hit: {model_name}")
            return self.cache[model_name]

        if loader_func is None:
            return None

        # Check memory before loading
        if self._estimate_memory_after_load() > self.max_memory_mb:
            self._evict_least_used()

        # Load model
        model = loader_func()
        if model is not None:
            self.cache[model_name] = model
            self.access_times[model_name] = time.time()
            self._update_memory_usage()
            logging.info(f"📥 Cached: {model_name}")

        return model

    def _evict_least_used(self):
        """Remove least recently used model"""
        if not self.cache:
            return

        lru_model = min(self.access_times.items(), key=lambda x: x[1])[0]
        self.remove_model(lru_model)

    def remove_model(self, model_name: str):
        """Remove model from cache"""
        if model_name in self.cache:
            del self.cache[model_name]
            del self.access_times[model_name]
            self._update_memory_usage()
            logging.info(f"🗑️ Evicted: {model_name}")

    def clear_cache(self):
        """Clear all cached models"""
        self.cache.clear()
        self.access_times.clear()
        self.current_memory_mb = 0
        logging.info("🧹 Cache cleared")

    def _estimate_memory_after_load(self) -> float:
        """Estimate memory usage after loading a new model"""
        # Conservative estimate: assume 100MB per new model
        return self.current_memory_mb + 100

    def _update_memory_usage(self):
        """Update current memory usage estimate"""
        try:
            import psutil
            process = psutil.Process()
            self.current_memory_mb = process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logging.warning(f"Memory usage update failed: {e}")
            # Fallback: estimate based on number of cached models
            self.current_memory_mb = len(self.cache) * 50

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cached_models': list(self.cache.keys()),
            'cache_size': len(self.cache),
            'current_memory_mb': self.current_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'memory_utilization': self.current_memory_mb / self.max_memory_mb if self.max_memory_mb > 0 else 0
        }