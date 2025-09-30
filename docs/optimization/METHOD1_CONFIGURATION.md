# Method 1 Configuration Guide

*Generated on 2025-09-29 11:44:02*

## Overview

This guide covers configuration options for Method 1 Parameter Optimization Engine, including performance tuning, error handling, logging, and deployment settings.

## Basic Configuration

### Environment Variables

```bash
# Core settings
export OPTIMIZATION_LOG_LEVEL=INFO
export OPTIMIZATION_LOG_DIR=logs/optimization
export OPTIMIZATION_CACHE_SIZE=1000
export OPTIMIZATION_ENABLE_PROFILING=false

# Performance settings
export OPTIMIZATION_MAX_WORKERS=4
export OPTIMIZATION_BATCH_SIZE=10
export OPTIMIZATION_TIMEOUT=30

# Quality measurement settings
export QUALITY_MEASUREMENT_ENABLED=true
export QUALITY_MEASUREMENT_RUNS=3
export QUALITY_MEASUREMENT_TIMEOUT=10

# Error handling settings
export ERROR_NOTIFICATION_ENABLED=false
export ERROR_RECOVERY_ENABLED=true
export CIRCUIT_BREAKER_THRESHOLD=5
```

### Configuration File

Create `config/optimization.json`:

```json
{
    "optimization": {
        "cache_size": 1000,
        "enable_profiling": false,
        "default_timeout": 30,
        "max_retries": 3,
        "batch_size": 10
    },
    "parameter_bounds": {
        "color_precision": {
            "min": 2,
            "max": 10,
            "default": 6
        },
        "corner_threshold": {
            "min": 10,
            "max": 110,
            "default": 50
        }
    },
    "quality_measurement": {
        "enabled": true,
        "default_runs": 3,
        "timeout": 10,
        "ssim_threshold": 0.8
    },
    "logging": {
        "level": "INFO",
        "directory": "logs/optimization",
        "max_file_size_mb": 100,
        "max_files": 10
    },
    "error_handling": {
        "notification_enabled": false,
        "recovery_enabled": true,
        "circuit_breaker": {
            "vtracer_threshold": 3,
            "vtracer_timeout": 30
        }
    }
}
```

## Loading Configuration

### Python Configuration

```python
import json
from pathlib import Path

class OptimizationConfig:
    def __init__(self, config_path: str = "config/optimization.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        else:
            return self._get_default_config()

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default

# Usage
config = OptimizationConfig()
cache_size = config.get('optimization.cache_size', 1000)
log_level = config.get('logging.level', 'INFO')
```

## Performance Configuration

### Caching Settings

```python
from backend.ai_modules.optimization.feature_mapping import FeatureMappingOptimizer

# Configure optimizer with caching
optimizer = FeatureMappingOptimizer(
    cache_size=1000,           # Number of cached results
    enable_caching=True,       # Enable feature/parameter caching
    cache_timeout=3600,        # Cache timeout in seconds
    memory_limit_mb=100        # Maximum cache memory usage
)
```

### Memory Management

```python
import psutil
import gc

class MemoryManager:
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent

    def check_memory_usage(self) -> bool:
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < self.max_memory_percent

    def cleanup_if_needed(self):
        if not self.check_memory_usage():
            gc.collect()
            print("Memory cleanup performed")

# Usage
memory_manager = MemoryManager(max_memory_percent=75.0)
```

## Error Handling Configuration

### Circuit Breaker Settings

```python
from backend.ai_modules.optimization.error_handler import CircuitBreaker

# Configure circuit breakers
vtracer_breaker = CircuitBreaker(
    failure_threshold=3,       # Open after 3 failures
    recovery_timeout=30,       # Try to recover after 30 seconds
)
```

### Retry Configuration

```python
from backend.ai_modules.optimization.error_handler import RetryConfig

# Configure retry strategies
retry_config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0
)
```

## Deployment Configuration

### Production Settings

```python
PRODUCTION_CONFIG = {
    "optimization": {
        "cache_size": 5000,
        "enable_profiling": False,
        "default_timeout": 60,
        "max_retries": 2,
        "batch_size": 20
    },
    "logging": {
        "level": "WARNING",   # Reduced logging in production
        "directory": "/var/log/optimization",
        "max_file_size_mb": 500,
        "max_files": 20
    }
}
```

This configuration guide provides comprehensive settings for deploying and tuning Method 1 Parameter Optimization Engine in various environments.
