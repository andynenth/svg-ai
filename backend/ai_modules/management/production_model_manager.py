# backend/ai_modules/management/production_model_manager.py
import os
import threading
import logging
from pathlib import Path
from typing import Dict, Any

# Import performance monitoring
from backend.utils.performance_monitor import monitor_model_loading

class ProductionModelManager:
    def __init__(self, model_dir: str = None):
        # Configuration-driven model location with fallback
        if model_dir is None:
            # Check environment variable first
            model_dir = os.environ.get('MODEL_DIR', None)
            if model_dir is None:
                # Check if production models directory exists
                if Path('models/production').exists():
                    model_dir = 'models/production'
                elif Path('models').exists():
                    model_dir = 'models'
                else:
                    # Fallback to legacy path for compatibility
                    model_dir = "backend/ai_modules/models/exported"

        self.model_dir = Path(model_dir)
        self.models = {}
        self.model_metadata = {}
        self.loading_lock = threading.Lock()
        self.models_found = False  # Track if any models were successfully loaded

    def load_models(self) -> Dict[str, Any]:
        """Public facade method to load all models"""
        with self.loading_lock:
            return self._load_all_exported_models()

    @monitor_model_loading()
    def _load_all_exported_models(self) -> Dict[str, Any]:
        """Load all exported models with error handling"""
        models = {}

        # Quality Predictor (TorchScript)
        try:
            import torch
            models['quality_predictor'] = torch.jit.load(
                str(self.model_dir / 'quality_predictor.torchscript')
            )
            models['quality_predictor'].eval()
            logging.info("✅ Quality predictor loaded")
        except Exception as e:
            logging.warning(f"⚠️ Quality predictor unavailable: {e}")
            models['quality_predictor'] = None

        # Logo Classifier (ONNX)
        try:
            import onnxruntime as ort
            models['logo_classifier'] = ort.InferenceSession(
                str(self.model_dir / 'logo_classifier.onnx')
            )
            logging.info("✅ Logo classifier loaded")
        except Exception as e:
            logging.warning(f"⚠️ Logo classifier unavailable: {e}")
            models['logo_classifier'] = None

        # Correlation Models (Pickle)
        try:
            import joblib
            models['correlation_models'] = joblib.load(
                str(self.model_dir / 'correlation_models.pkl')
            )
            logging.info("✅ Correlation models loaded")
        except Exception as e:
            logging.warning(f"⚠️ Correlation models unavailable: {e}")
            models['correlation_models'] = None

        # Persist loaded models to instance
        self.models = models

        # Update models_found flag based on whether any models loaded
        self.models_found = any(model is not None for model in models.values())

        if not self.models_found:
            logging.warning(f"⚠️ No models found in {self.model_dir}")
            logging.info(f"💡 To enable AI features, export models to: {self.model_dir.absolute()}")

        return models

    @monitor_model_loading()
    def _optimize_for_production(self):
        """Optimize models for production inference"""
        for model_name, model in self.models.items():
            if model is None:
                continue

            if hasattr(model, 'eval'):
                model.eval()  # Set to evaluation mode

            # Warmup models with dummy input
            self._warmup_model(model_name, model)

            # Track memory usage
            self._track_model_memory(model_name)

    @monitor_model_loading()
    def _warmup_model(self, model_name: str, model):
        """Warmup model with dummy inference"""
        try:
            import torch
            import numpy as np

            if model_name == 'quality_predictor':
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_params = torch.randn(1, 8)
                with torch.no_grad():
                    _ = model(dummy_input, dummy_params)
            elif model_name == 'logo_classifier':
                dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
                _ = model.run(None, {model.get_inputs()[0].name: dummy_input})

            logging.info(f"✅ {model_name} warmed up")
        except Exception as e:
            logging.warning(f"⚠️ {model_name} warmup failed: {e}")

    def _track_model_memory(self, model_name: str):
        """Track memory usage for a specific model"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logging.info(f"📊 {model_name} memory usage: {memory_mb:.1f}MB")
        except Exception as e:
            logging.warning(f"⚠️ Memory tracking failed for {model_name}: {e}")