"""
Lazy Loading System for AI Models

This module provides lazy loading capabilities for machine learning models
to reduce memory usage and startup time. Models are loaded only when needed
and can be automatically unloaded to manage memory.
"""

import weakref
import time
import gc
import threading
import sys
from typing import Any, Callable, Dict, Optional
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)


class LazyModelProxy:
    """Proxy that loads model on first access"""

    def __init__(self, loader_func: Callable, *args, **kwargs):
        self._loader_func = loader_func
        self._args = args
        self._kwargs = kwargs
        self._model = None
        self._loading = False
        self._lock = threading.Lock()
        self._load_time = None
        self._access_count = 0

    def _ensure_loaded(self):
        """Load model if not already loaded"""
        if self._model is None and not self._loading:
            with self._lock:
                if self._model is None:
                    self._loading = True
                    try:
                        logger.info(f"Loading model via proxy...")
                        start_time = time.time()
                        self._model = self._loader_func(*self._args, **self._kwargs)
                        self._load_time = time.time() - start_time
                        logger.info(f"Model loaded in {self._load_time:.3f} seconds")
                    except Exception as e:
                        logger.error(f"Model loading failed: {e}")
                        raise
                    finally:
                        self._loading = False

        self._access_count += 1

    def __getattr__(self, name):
        """Forward attribute access to loaded model"""
        self._ensure_loaded()
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        """Forward calls to loaded model"""
        self._ensure_loaded()
        return self._model(*args, **kwargs)

    def is_loaded(self) -> bool:
        """Check if model is currently loaded"""
        return self._model is not None

    def unload(self):
        """Manually unload the model"""
        with self._lock:
            if self._model is not None:
                self._model = None
                gc.collect()
                logger.info("Model unloaded manually")

    def get_stats(self) -> Dict:
        """Get statistics about this proxy"""
        return {
            'loaded': self.is_loaded(),
            'load_time': self._load_time,
            'access_count': self._access_count,
            'memory_size_mb': self._get_memory_size() if self.is_loaded() else 0
        }

    def _get_memory_size(self) -> float:
        """Estimate memory size of loaded model"""
        if self._model is not None:
            return sys.getsizeof(self._model) / (1024 * 1024)
        return 0


class LazyModelManager:
    """Manager for lazy-loaded models with automatic memory management"""

    def __init__(self):
        self.models = {}
        self.load_times = {}
        self.last_used = {}
        self.memory_limit_mb = 500
        self._lock = threading.RLock()

    def register_model(self, name: str, loader_func: Callable, *args, **kwargs):
        """Register a model for lazy loading"""
        with self._lock:
            self.models[name] = LazyModelProxy(loader_func, *args, **kwargs)
            self.last_used[name] = time.time()
            logger.info(f"Registered model: {name}")

    def get_model(self, name: str) -> Any:
        """Get a model (loading if necessary)"""
        with self._lock:
            if name in self.models:
                self.last_used[name] = time.time()
                return self.models[name]
            raise KeyError(f"Model {name} not registered")

    def unload_model(self, name: str):
        """Manually unload a model to free memory"""
        with self._lock:
            if name in self.models:
                self.models[name].unload()
                gc.collect()
                logger.info(f"Unloaded model: {name}")

    def auto_unload_unused(self, max_age_seconds: int = 300):
        """Unload models not used recently"""
        current_time = time.time()
        unloaded_count = 0

        with self._lock:
            for name, last_used in self.last_used.items():
                if current_time - last_used > max_age_seconds:
                    if self.models[name].is_loaded():
                        self.unload_model(name)
                        unloaded_count += 1

        if unloaded_count > 0:
            logger.info(f"Auto-unloaded {unloaded_count} unused models")

    def get_memory_usage(self) -> Dict:
        """Get memory usage of loaded models"""
        usage = {}
        total_mb = 0

        with self._lock:
            for name, proxy in self.models.items():
                if proxy.is_loaded():
                    size_mb = proxy._get_memory_size()
                    usage[name] = size_mb
                    total_mb += size_mb

        return {
            'models': usage,
            'total_mb': total_mb,
            'limit_mb': self.memory_limit_mb,
            'usage_percent': (total_mb / self.memory_limit_mb) * 100 if self.memory_limit_mb > 0 else 0
        }

    def enforce_memory_limit(self):
        """Enforce memory limit by unloading least recently used models"""
        memory_stats = self.get_memory_usage()

        if memory_stats['total_mb'] > self.memory_limit_mb:
            logger.warning(f"Memory limit exceeded: {memory_stats['total_mb']:.1f}MB > {self.memory_limit_mb}MB")

            # Get loaded models sorted by last used time
            loaded_models = []
            with self._lock:
                for name, proxy in self.models.items():
                    if proxy.is_loaded():
                        loaded_models.append((name, self.last_used[name]))

            # Sort by last used (oldest first)
            loaded_models.sort(key=lambda x: x[1])

            # Unload models until under limit
            for name, _ in loaded_models:
                self.unload_model(name)
                memory_stats = self.get_memory_usage()
                if memory_stats['total_mb'] <= self.memory_limit_mb:
                    break

    def preload_models(self, model_names: list, max_workers: int = 4):
        """Preload specified models in parallel"""
        import concurrent.futures

        def load_single(name):
            try:
                model = self.get_model(name)
                model._ensure_loaded()  # Force loading
                return name, True
            except Exception as e:
                logger.error(f"Failed to preload {name}: {e}")
                return name, False

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_single, name) for name in model_names]

            for future in concurrent.futures.as_completed(futures):
                name, success = future.result()
                results[name] = success

        successful = sum(results.values())
        logger.info(f"Preloaded {successful}/{len(model_names)} models")
        return results

    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        with self._lock:
            stats = {
                'total_models': len(self.models),
                'loaded_models': sum(1 for proxy in self.models.values() if proxy.is_loaded()),
                'memory_usage': self.get_memory_usage(),
                'model_details': {}
            }

            for name, proxy in self.models.items():
                stats['model_details'][name] = {
                    **proxy.get_stats(),
                    'last_used': self.last_used.get(name, 0)
                }

        return stats

    def clear_all(self):
        """Unload all models and clear registry"""
        with self._lock:
            for name in list(self.models.keys()):
                self.unload_model(name)
            self.models.clear()
            self.last_used.clear()
            logger.info("Cleared all models from manager")


# Global model manager instance
global_model_manager = LazyModelManager()


def register_model(name: str, loader_func: Callable, *args, **kwargs):
    """Convenience function to register a model globally"""
    global_model_manager.register_model(name, loader_func, *args, **kwargs)


def get_model(name: str):
    """Convenience function to get a model globally"""
    return global_model_manager.get_model(name)


# Example loader functions for different model types
def load_pytorch_model(model_path: str, device: str = 'cpu'):
    """Load PyTorch model"""
    try:
        import torch
        logger.info(f"Loading PyTorch model from {model_path}")
        model = torch.load(model_path, map_location=device)
        if hasattr(model, 'eval'):
            model.eval()
        return model
    except ImportError:
        logger.error("PyTorch not available")
        raise
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")
        raise


def load_sklearn_model(model_path: str):
    """Load scikit-learn model"""
    try:
        import pickle
        logger.info(f"Loading sklearn model from {model_path}")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load sklearn model: {e}")
        raise


def load_xgboost_model(model_path: str):
    """Load XGBoost model"""
    try:
        import xgboost as xgb
        logger.info(f"Loading XGBoost model from {model_path}")
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    except ImportError:
        logger.error("XGBoost not available")
        raise
    except Exception as e:
        logger.error(f"Failed to load XGBoost model: {e}")
        raise


# Integration example for existing codebase
def setup_model_integration():
    """Set up model integration for existing codebase"""

    # Example of how to integrate with existing code
    model_manager = LazyModelManager()

    # Register models that would previously be loaded at startup
    model_manager.register_model(
        'classifier',
        load_pytorch_model,
        'models/efficientnet_classifier.pth'
    )

    model_manager.register_model(
        'optimizer',
        load_xgboost_model,
        'models/parameter_optimizer.xgb'
    )

    model_manager.register_model(
        'quality_predictor',
        load_sklearn_model,
        'models/quality_predictor.pkl'
    )

    # Set memory limit (500MB)
    model_manager.memory_limit_mb = 500

    logger.info("Model integration setup complete")
    return model_manager


def update_existing_code_example():
    """Example of how to update existing code to use lazy loading"""

    # Before: Direct model loading
    # classifier = EfficientNetClassifier()
    # optimizer = XGBoostOptimizer()

    # After: Lazy loading
    model_manager = setup_model_integration()

    # Models are loaded only when first accessed
    classifier = model_manager.get_model('classifier')
    optimizer = model_manager.get_model('optimizer')

    # Use models normally - they load automatically on first access
    # result = classifier.predict(image)
    # params = optimizer.optimize(features)

    return model_manager


# Background thread for automatic memory management
class ModelMemoryManager:
    """Background memory manager for models"""

    def __init__(self, model_manager: LazyModelManager, check_interval: int = 60):
        self.model_manager = model_manager
        self.check_interval = check_interval
        self.running = False
        self.thread = None

    def start(self):
        """Start background memory management"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._management_loop, daemon=True)
            self.thread.start()
            logger.info("Model memory manager started")

    def stop(self):
        """Stop background memory management"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Model memory manager stopped")

    def _management_loop(self):
        """Main management loop"""
        while self.running:
            try:
                # Auto-unload unused models (older than 5 minutes)
                self.model_manager.auto_unload_unused(300)

                # Enforce memory limits
                self.model_manager.enforce_memory_limit()

                # Log memory stats
                stats = self.model_manager.get_memory_usage()
                if stats['total_mb'] > 0:
                    logger.debug(f"Model memory usage: {stats['total_mb']:.1f}MB ({stats['usage_percent']:.1f}%)")

            except Exception as e:
                logger.error(f"Model memory management error: {e}")

            time.sleep(self.check_interval)


# Usage example and testing
def test_lazy_loading():
    """Test lazy loading functionality"""
    print("\n=== Testing Lazy Loading ===")

    manager = LazyModelManager()

    # Register mock models
    def mock_loader(name):
        time.sleep(0.1)  # Simulate loading time
        return f"MockModel_{name}"

    manager.register_model('test_model_1', mock_loader, 'model1')
    manager.register_model('test_model_2', mock_loader, 'model2')

    print(f"Registered models: {list(manager.models.keys())}")

    # Test lazy loading
    print("Accessing model 1...")
    start = time.time()
    model1 = manager.get_model('test_model_1')
    # Force loading
    model1._ensure_loaded()
    load_time = time.time() - start
    print(f"Model 1 loaded in {load_time:.3f}s")

    # Test memory stats
    stats = manager.get_stats()
    print(f"Memory stats: {stats['memory_usage']}")

    # Test auto-unloading
    print("Testing auto-unload...")
    manager.auto_unload_unused(0)  # Unload all
    stats = manager.get_stats()
    print(f"After unload - loaded models: {stats['loaded_models']}")

    print("Lazy loading test complete!")


if __name__ == "__main__":
    test_lazy_loading()