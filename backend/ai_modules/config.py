# backend/ai_modules/config.py
"""Configuration for AI modules"""

import os
from pathlib import Path

# Base paths
AI_MODULES_PATH = Path(__file__).parent
MODELS_PATH = AI_MODULES_PATH / "models"
PRETRAINED_PATH = MODELS_PATH / "pretrained"
TRAINED_PATH = MODELS_PATH / "trained"
CACHE_PATH = MODELS_PATH / "cache"

# Model configurations
MODEL_CONFIG = {
    'efficientnet_b0': {
        'path': PRETRAINED_PATH / 'efficientnet_b0.pth',
        'input_size': (224, 224),
        'num_classes': 4  # simple, text, gradient, complex
    },
    'resnet50': {
        'path': PRETRAINED_PATH / 'resnet50_features.pth',
        'input_size': (224, 224),
        'feature_dim': 2048
    },
    'quality_predictor': {
        'path': TRAINED_PATH / 'quality_predictor.pth',
        'input_dim': 2056,  # 2048 image + 8 params
        'hidden_dims': [512, 256, 128]
    }
}

# Performance targets
PERFORMANCE_TARGETS = {
    'tier_1': {
        'max_time': 1.0,
        'target_quality': 0.85
    },
    'tier_2': {
        'max_time': 15.0,
        'target_quality': 0.90
    },
    'tier_3': {
        'max_time': 60.0,
        'target_quality': 0.95
    }
}

# Feature extraction parameters
FEATURE_CONFIG = {
    'edge_detection': {
        'canny_low': 50,
        'canny_high': 150
    },
    'corner_detection': {
        'max_corners': 100,
        'quality_level': 0.3,
        'min_distance': 7
    },
    'color_analysis': {
        'kmeans_clusters': 16,
        'sample_rate': 0.1
    }
}

# VTracer parameter ranges for optimization
VTRACER_PARAM_RANGES = {
    'color_precision': (1, 10),
    'corner_threshold': (10, 100),
    'path_precision': (5, 50),
    'layer_difference': (1, 10),
    'splice_threshold': (20, 100),
    'filter_speckle': (1, 50),
    'segment_length': (5, 50),
    'max_iterations': (5, 30)
}

# Default VTracer parameters by logo type
DEFAULT_VTRACER_PARAMS = {
    'simple': {
        'color_precision': 3,
        'corner_threshold': 30,
        'path_precision': 10,
        'layer_difference': 5,
        'splice_threshold': 60,
        'filter_speckle': 4,
        'segment_length': 10,
        'max_iterations': 10
    },
    'text': {
        'color_precision': 2,
        'corner_threshold': 20,
        'path_precision': 8,
        'layer_difference': 3,
        'splice_threshold': 45,
        'filter_speckle': 2,
        'segment_length': 8,
        'max_iterations': 8
    },
    'gradient': {
        'color_precision': 8,
        'corner_threshold': 40,
        'path_precision': 15,
        'layer_difference': 8,
        'splice_threshold': 70,
        'filter_speckle': 6,
        'segment_length': 12,
        'max_iterations': 15
    },
    'complex': {
        'color_precision': 6,
        'corner_threshold': 50,
        'path_precision': 20,
        'layer_difference': 6,
        'splice_threshold': 80,
        'filter_speckle': 8,
        'segment_length': 15,
        'max_iterations': 20
    }
}

# Reinforcement Learning configuration
RL_CONFIG = {
    'environment': {
        'observation_space_dim': 2056,  # features + current params
        'action_space_dim': 8,  # VTracer parameters
        'reward_threshold': 0.85,
        'max_episode_steps': 50
    },
    'agent': {
        'algorithm': 'PPO',
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95
    },
    'training': {
        'total_timesteps': 100000,
        'eval_freq': 5000,
        'save_freq': 10000
    }
}

# Genetic Algorithm configuration
GA_CONFIG = {
    'population_size': 50,
    'generations': 100,
    'crossover_prob': 0.7,
    'mutation_prob': 0.2,
    'tournament_size': 3,
    'elite_size': 5,
    'convergence_threshold': 0.001,
    'max_stagnant_generations': 20
}

# Environment variables and runtime configuration
def get_env_config():
    """Get configuration from environment variables"""
    return {
        'debug': os.getenv('AI_DEBUG', 'false').lower() == 'true',
        'cache_enabled': os.getenv('AI_CACHE_ENABLED', 'true').lower() == 'true',
        'max_workers': int(os.getenv('AI_MAX_WORKERS', '4')),
        'gpu_enabled': os.getenv('AI_GPU_ENABLED', 'false').lower() == 'true',
        'log_level': os.getenv('AI_LOG_LEVEL', 'INFO'),
        'model_cache_size': int(os.getenv('AI_MODEL_CACHE_SIZE', '5'))
    }

# Validation functions
def validate_config():
    """Validate configuration values"""
    errors = []

    # Check that required directories exist or can be created
    for path_name, path in [
        ('MODELS_PATH', MODELS_PATH),
        ('PRETRAINED_PATH', PRETRAINED_PATH),
        ('TRAINED_PATH', TRAINED_PATH),
        ('CACHE_PATH', CACHE_PATH)
    ]:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create {path_name} at {path}: {e}")

    # Validate performance targets
    for tier, config in PERFORMANCE_TARGETS.items():
        if config['max_time'] <= 0:
            errors.append(f"Invalid max_time for {tier}: {config['max_time']}")
        if not 0 <= config['target_quality'] <= 1:
            errors.append(f"Invalid target_quality for {tier}: {config['target_quality']}")

    # Validate VTracer parameter ranges
    for param, (min_val, max_val) in VTRACER_PARAM_RANGES.items():
        if min_val >= max_val:
            errors.append(f"Invalid range for {param}: ({min_val}, {max_val})")

    return errors

def get_config_summary():
    """Get a summary of current configuration"""
    env_config = get_env_config()
    validation_errors = validate_config()

    return {
        'ai_modules_path': str(AI_MODULES_PATH),
        'models_path': str(MODELS_PATH),
        'environment': env_config,
        'model_configs': len(MODEL_CONFIG),
        'performance_tiers': len(PERFORMANCE_TARGETS),
        'vtracer_param_count': len(VTRACER_PARAM_RANGES),
        'logo_types': len(DEFAULT_VTRACER_PARAMS),
        'validation_errors': validation_errors,
        'config_valid': len(validation_errors) == 0
    }

if __name__ == "__main__":
    # Display configuration summary when run directly
    import json
    summary = get_config_summary()
    print("AI Modules Configuration Summary")
    print("=" * 40)
    print(json.dumps(summary, indent=2))