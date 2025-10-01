# config/ai_production.py
"""AI-specific production configuration extending Day 5 base config"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# Import base config from Day 5
from .environments import ProductionConfig


class AIProductionConfig(ProductionConfig):
    """AI-enhanced production configuration"""

    # AI Model Configuration
    MODEL_DIR = os.environ.get('MODEL_DIR', 'models/')
    CLASSIFIER_MODEL = os.environ.get('CLASSIFIER_MODEL', 'classifier.pth')
    OPTIMIZER_MODEL = os.environ.get('OPTIMIZER_MODEL', 'optimizer.xgb')

    # Model Loading Settings
    MODEL_LAZY_LOADING = os.environ.get('MODEL_LAZY_LOADING', 'true').lower() == 'true'
    MODEL_CACHE_SIZE = int(os.environ.get('MODEL_CACHE_SIZE', '3'))
    MODEL_TIMEOUT = int(os.environ.get('MODEL_TIMEOUT', '30'))

    # Quality Tracking Database
    QUALITY_DB_URL = os.environ.get('QUALITY_DB_URL', 'sqlite:///quality_tracking.db')
    QUALITY_TRACKING_ENABLED = os.environ.get('QUALITY_TRACKING_ENABLED', 'true').lower() == 'true'

    # AI Performance Settings
    AI_BATCH_SIZE = int(os.environ.get('AI_BATCH_SIZE', '20'))
    AI_MAX_WORKERS = int(os.environ.get('AI_MAX_WORKERS', '4'))
    AI_INFERENCE_TIMEOUT = int(os.environ.get('AI_INFERENCE_TIMEOUT', '10'))

    # Quality Targets (AI-specific)
    TARGET_QUALITY_SIMPLE = float(os.environ.get('TARGET_QUALITY_SIMPLE', '0.95'))
    TARGET_QUALITY_TEXT = float(os.environ.get('TARGET_QUALITY_TEXT', '0.90'))
    TARGET_QUALITY_GRADIENT = float(os.environ.get('TARGET_QUALITY_GRADIENT', '0.85'))
    TARGET_QUALITY_COMPLEX = float(os.environ.get('TARGET_QUALITY_COMPLEX', '0.75'))

    # AI Monitoring
    AI_METRICS_ENABLED = os.environ.get('AI_METRICS_ENABLED', 'true').lower() == 'true'
    MODEL_PERFORMANCE_TRACKING = os.environ.get('MODEL_PERFORMANCE_TRACKING', 'true').lower() == 'true'

    # Continuous Learning
    ONLINE_LEARNING_ENABLED = os.environ.get('ONLINE_LEARNING_ENABLED', 'false').lower() == 'true'
    LEARNING_RATE_DECAY = float(os.environ.get('LEARNING_RATE_DECAY', '0.95'))

    # A/B Testing Configuration
    AB_TESTING_ENABLED = os.environ.get('AB_TESTING_ENABLED', 'false').lower() == 'true'
    AB_TEST_TRAFFIC_SPLIT = float(os.environ.get('AB_TEST_TRAFFIC_SPLIT', '0.1'))

    @classmethod
    def validate_model_paths(cls):
        """Ensure all model files exist"""
        model_dir = Path(cls.MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)

        required_models = [cls.CLASSIFIER_MODEL, cls.OPTIMIZER_MODEL]
        missing_models = []

        for model in required_models:
            if not (model_dir / model).exists():
                missing_models.append(model)

        if missing_models:
            raise FileNotFoundError(f"Missing AI models: {missing_models}")

        return True

    @classmethod
    def get_ai_config(cls) -> Dict[str, Any]:
        """Get complete AI configuration"""
        return {
            'model_dir': cls.MODEL_DIR,
            'models': {
                'classifier': cls.CLASSIFIER_MODEL,
                'optimizer': cls.OPTIMIZER_MODEL
            },
            'quality_targets': {
                'simple': cls.TARGET_QUALITY_SIMPLE,
                'text': cls.TARGET_QUALITY_TEXT,
                'gradient': cls.TARGET_QUALITY_GRADIENT,
                'complex': cls.TARGET_QUALITY_COMPLEX
            },
            'performance': {
                'batch_size': cls.AI_BATCH_SIZE,
                'max_workers': cls.AI_MAX_WORKERS,
                'timeout': cls.AI_INFERENCE_TIMEOUT
            }
        }