#!/usr/bin/env python3
"""
Async Model Manager for Performance Optimization

Provides background model preloading to eliminate latency spikes
during first-time model initialization.
"""

import asyncio
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List, Tuple
from datetime import datetime

# Import performance monitoring
from backend.utils.performance_monitor import monitor_model_loading

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Async model manager for preloading and caching AI models.

    Eliminates latency spikes by preloading models in background
    and providing cached access to loaded models.
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._loading_tasks: Dict[str, asyncio.Task] = {}
        self._load_lock = threading.Lock()
        self._initialization_status: Dict[str, str] = {}
        self._load_times: Dict[str, float] = {}

        # Model configurations
        self._model_configs = {
            'classification': {
                'loader': self._load_classification_model,
                'path': 'backend/ai_modules/models/classification_model.pth',
                'required': False,  # Has fallback
                'timeout': 30.0
            },
            'optimization': {
                'loader': self._load_optimization_model,
                'path': 'backend/ai_modules/models/optimization_model.xgb',
                'required': False,  # Has fallback
                'timeout': 10.0
            },
            'quality_prediction': {
                'loader': self._load_quality_model,
                'path': 'backend/ai_modules/models/quality_predictor.pth',
                'required': False,  # Has fallback
                'timeout': 30.0
            }
        }

    @monitor_model_loading()
    async def preload_models(self, model_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Preload models in background for faster access.

        Args:
            model_names: List of model names to preload. If None, preloads all models.

        Returns:
            Dictionary mapping model names to success status
        """
        if model_names is None:
            model_names = list(self._model_configs.keys())

        logger.info(f"Starting background preload of {len(model_names)} models...")

        # Create async loading tasks
        tasks = []
        for name in model_names:
            if name in self._model_configs:
                task = asyncio.create_task(self._async_load(name))
                self._loading_tasks[name] = task
                tasks.append(task)
            else:
                logger.warning(f"Unknown model: {name}")

        # Wait for all models to load (with exceptions)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        load_status = {}
        for i, (name, result) in enumerate(zip(model_names, results)):
            if name in self._model_configs:
                if isinstance(result, Exception):
                    logger.error(f"Model {name} failed to load: {result}")
                    load_status[name] = False
                    self._initialization_status[name] = f"failed: {result}"
                else:
                    load_status[name] = result
                    status = "loaded" if result else "failed"
                    self._initialization_status[name] = status

        # Log summary
        successful = sum(1 for success in load_status.values() if success)
        total = len(load_status)
        logger.info(f"Preloading completed: {successful}/{total} models loaded successfully")

        return load_status

    async def _async_load(self, name: str) -> bool:
        """Load a single model asynchronously."""
        config = self._model_configs[name]
        loader = config['loader']
        timeout = config['timeout']

        start_time = asyncio.get_event_loop().time()

        try:
            # Run the synchronous loader in a thread pool
            model = await asyncio.wait_for(
                asyncio.to_thread(loader),
                timeout=timeout
            )

            # Store the loaded model
            with self._load_lock:
                self._models[name] = model
                load_time = asyncio.get_event_loop().time() - start_time
                self._load_times[name] = load_time

            logger.info(f"✓ {name} model loaded successfully ({load_time:.2f}s)")
            return True

        except asyncio.TimeoutError:
            logger.error(f"✗ {name} model loading timed out after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"✗ {name} model loading failed: {e}")
            return False

    def _load_classification_model(self) -> Any:
        """Load the classification model (neural network)."""
        try:
            from backend.ai_modules.classification import ClassificationModule

            classifier = ClassificationModule()

            # Try to load pre-trained model if it exists
            model_path = self._model_configs['classification']['path']
            if Path(model_path).exists():
                classifier.load_neural_model(model_path)
                logger.debug(f"Loaded classification model from {model_path}")
            else:
                # Load default pre-trained model (EfficientNet)
                classifier.load_neural_model("")  # Empty path triggers default model
                logger.debug("Loaded default classification model (EfficientNet)")

            return classifier

        except Exception as e:
            logger.error(f"Classification model loading failed: {e}")
            raise

    def _load_optimization_model(self) -> Any:
        """Load the optimization model (XGBoost)."""
        try:
            from backend.ai_modules.optimization import OptimizationEngine

            optimizer = OptimizationEngine()

            # Try to load pre-trained XGBoost model if it exists
            model_path = self._model_configs['optimization']['path']
            if Path(model_path).exists():
                optimizer.load_model(model_path)
                logger.debug(f"Loaded optimization model from {model_path}")
            else:
                logger.debug("Optimization model file not found, using formula-based optimization")

            return optimizer

        except Exception as e:
            logger.error(f"Optimization model loading failed: {e}")
            raise

    def _load_quality_model(self) -> Any:
        """Load the quality prediction model."""
        try:
            from backend.ai_modules.prediction.quality_predictor import QualityPredictor

            model_path = self._model_configs['quality_prediction']['path']
            predictor = QualityPredictor(model_path if Path(model_path).exists() else None)

            # Initialize the model
            predictor._load_model()

            logger.debug("Loaded quality prediction model")
            return predictor

        except Exception as e:
            logger.error(f"Quality prediction model loading failed: {e}")
            raise

    @lru_cache(maxsize=8)
    def get_cached_model(self, model_type: str) -> Optional[Any]:
        """
        Get cached model, avoiding reloading.

        Args:
            model_type: Type of model ('classification', 'optimization', 'quality_prediction')

        Returns:
            Loaded model instance or None if not available
        """
        with self._load_lock:
            if model_type in self._models:
                return self._models[model_type]

        # Model not preloaded, try to load synchronously
        logger.warning(f"Model {model_type} not preloaded, loading synchronously...")

        if model_type not in self._model_configs:
            logger.error(f"Unknown model type: {model_type}")
            return None

        try:
            config = self._model_configs[model_type]
            model = config['loader']()

            with self._load_lock:
                self._models[model_type] = model

            return model

        except Exception as e:
            logger.error(f"Synchronous loading of {model_type} failed: {e}")
            return None

    def is_model_loaded(self, model_type: str) -> bool:
        """Check if a model is loaded and ready."""
        with self._load_lock:
            return model_type in self._models and self._models[model_type] is not None

    def get_loading_status(self) -> Dict[str, Any]:
        """Get detailed loading status for all models."""
        with self._load_lock:
            status = {
                'models_loaded': len(self._models),
                'total_models': len(self._model_configs),
                'individual_status': {},
                'load_times': self._load_times.copy(),
                'initialization_status': self._initialization_status.copy()
            }

            for name in self._model_configs:
                status['individual_status'][name] = {
                    'loaded': name in self._models and self._models[name] is not None,
                    'required': self._model_configs[name]['required'],
                    'load_time': self._load_times.get(name),
                    'status': self._initialization_status.get(name, 'not_loaded')
                }

        return status

    def clear_cache(self, model_type: Optional[str] = None):
        """Clear model cache."""
        with self._load_lock:
            if model_type:
                if model_type in self._models:
                    del self._models[model_type]
                    logger.info(f"Cleared cache for {model_type}")
            else:
                self._models.clear()
                logger.info("Cleared all model cache")

        # Clear LRU cache
        self.get_cached_model.cache_clear()

    def start_background_preloading(self, model_names: Optional[List[str]] = None) -> asyncio.Task:
        """
        Start background preloading as a fire-and-forget task.

        Args:
            model_names: Models to preload. If None, preloads all.

        Returns:
            Async task for monitoring (optional)
        """
        async def preload_wrapper():
            try:
                await self.preload_models(model_names)
            except Exception as e:
                logger.error(f"Background preloading failed: {e}")

        task = asyncio.create_task(preload_wrapper())
        logger.info("Started background model preloading")
        return task

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on model loading system."""
        status = self.get_loading_status()

        health = {
            'overall_status': 'healthy',
            'issues': [],
            'warnings': [],
            'model_status': status
        }

        # Check for critical failures
        required_models = [name for name, config in self._model_configs.items()
                         if config['required']]

        for model_name in required_models:
            if not self.is_model_loaded(model_name):
                health['issues'].append(f"Required model {model_name} not loaded")
                health['overall_status'] = 'degraded'

        # Check load times
        for model_name, load_time in self._load_times.items():
            config = self._model_configs[model_name]
            if load_time > config['timeout'] * 0.8:  # 80% of timeout
                health['warnings'].append(
                    f"Model {model_name} slow to load: {load_time:.2f}s"
                )

        return health

    def __repr__(self) -> str:
        """String representation."""
        status = self.get_loading_status()
        loaded = status['models_loaded']
        total = status['total_models']
        return f"ModelManager(loaded={loaded}/{total})"


# Global model manager instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    return model_manager


# Convenience functions for easy access
@lru_cache(maxsize=3)
def get_cached_model(model_type: str) -> Optional[Any]:
    """Get cached model with global manager."""
    return model_manager.get_cached_model(model_type)


async def preload_all_models() -> Dict[str, bool]:
    """Preload all models using global manager."""
    return await model_manager.preload_models()


def start_background_loading() -> asyncio.Task:
    """Start background preloading with global manager."""
    return model_manager.start_background_preloading()