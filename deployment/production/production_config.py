#!/usr/bin/env python3
"""
Production Configuration for 4-Tier SVG-AI System
Comprehensive production deployment configuration management
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

# Production configuration
@dataclass
class ProductionConfig:
    """Complete production configuration for 4-tier system"""

    # Environment settings
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "INFO"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    request_timeout: int = 300  # 5 minutes

    # Database Configuration
    database_url: str = "postgresql://svgai:${DB_PASSWORD}@postgres-service:5432/svgai_prod"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_timeout: int = 30

    # Redis Configuration
    redis_url: str = "redis://:${REDIS_PASSWORD}@redis-service:6379/0"
    redis_pool_size: int = 20
    redis_timeout: int = 5

    # 4-Tier System Configuration
    tier_system_config: Dict[str, Any] = None

    # Security Configuration
    api_keys: List[str] = None
    cors_origins: List[str] = None
    rate_limiting: Dict[str, Any] = None

    # Monitoring and Logging
    monitoring_config: Dict[str, Any] = None
    logging_config: Dict[str, Any] = None

    # Resource Limits
    resource_limits: Dict[str, Any] = None

    # Performance Optimization
    performance_config: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default configurations"""
        if self.tier_system_config is None:
            self.tier_system_config = self._get_default_tier_config()

        if self.api_keys is None:
            self.api_keys = [
                "tier4-prod-key-${PRODUCTION_API_KEY}",
                "tier4-admin-key-${ADMIN_API_KEY}",
                "tier4-monitor-key-${MONITORING_API_KEY}"
            ]

        if self.cors_origins is None:
            self.cors_origins = [
                "https://svg-ai.production.com",
                "https://api.svg-ai.production.com",
                "https://admin.svg-ai.production.com"
            ]

        if self.rate_limiting is None:
            self.rate_limiting = {
                "requests_per_minute": 200,
                "requests_per_hour": 2000,
                "burst_limit": 50,
                "batch_requests_per_hour": 100
            }

        if self.monitoring_config is None:
            self.monitoring_config = self._get_default_monitoring_config()

        if self.logging_config is None:
            self.logging_config = self._get_default_logging_config()

        if self.resource_limits is None:
            self.resource_limits = self._get_default_resource_limits()

        if self.performance_config is None:
            self.performance_config = self._get_default_performance_config()

    def _get_default_tier_config(self) -> Dict[str, Any]:
        """Default 4-tier system configuration for production"""
        return {
            "max_concurrent_requests": 50,
            "enable_async_processing": True,
            "enable_caching": True,
            "cache_ttl": 7200,  # 2 hours
            "production_mode": True,
            "tier_timeouts": {
                "classification": 15.0,
                "routing": 10.0,
                "optimization": 180.0,  # 3 minutes
                "prediction": 30.0
            },
            "quality_targets": {
                "simple": 0.98,
                "text": 0.95,
                "gradient": 0.92,
                "complex": 0.88
            },
            "fallback_settings": {
                "enable_fallback": True,
                "fallback_timeout": 60.0,
                "max_fallback_attempts": 3
            },
            "optimization_methods": {
                "feature_mapping": {"enabled": True, "priority": 1},
                "regression": {"enabled": True, "priority": 2},
                "ppo": {"enabled": True, "priority": 3},
                "performance": {"enabled": True, "priority": 4}
            }
        }

    def _get_default_monitoring_config(self) -> Dict[str, Any]:
        """Default monitoring configuration"""
        return {
            "enable_metrics": True,
            "enable_tracing": True,
            "enable_profiling": False,  # Disabled in production
            "metrics_port": 9090,
            "health_check_interval": 30,
            "performance_sampling_rate": 0.1,
            "error_sampling_rate": 1.0,
            "alerting": {
                "enable_alerts": True,
                "error_threshold": 0.05,  # 5% error rate
                "latency_threshold": 30.0,  # 30 seconds
                "memory_threshold": 0.85,  # 85% memory usage
                "cpu_threshold": 0.80  # 80% CPU usage
            }
        }

    def _get_default_logging_config(self) -> Dict[str, Any]:
        """Default logging configuration"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                },
                "detailed": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",
                    "formatter": "standard"
                },
                "file": {
                    "level": "INFO",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "/app/logs/svg-ai-production.log",
                    "maxBytes": 104857600,  # 100MB
                    "backupCount": 10,
                    "formatter": "json"
                },
                "error_file": {
                    "level": "ERROR",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "/app/logs/svg-ai-errors.log",
                    "maxBytes": 52428800,  # 50MB
                    "backupCount": 5,
                    "formatter": "detailed"
                }
            },
            "loggers": {
                "svg_ai": {
                    "level": "INFO",
                    "handlers": ["console", "file", "error_file"],
                    "propagate": False
                },
                "tier4_system": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False
                },
                "optimization": {
                    "level": "INFO",
                    "handlers": ["file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "WARNING",
                "handlers": ["console", "file"]
            }
        }

    def _get_default_resource_limits(self) -> Dict[str, Any]:
        """Default resource limits"""
        return {
            "memory": {
                "api_service": "2Gi",
                "worker_service": "4Gi",
                "database": "1Gi",
                "redis": "512Mi"
            },
            "cpu": {
                "api_service": "2000m",
                "worker_service": "4000m",
                "database": "1000m",
                "redis": "500m"
            },
            "storage": {
                "database": "50Gi",
                "cache": "20Gi",
                "logs": "10Gi",
                "models": "5Gi"
            },
            "network": {
                "max_connections": 1000,
                "connection_timeout": 30,
                "read_timeout": 60,
                "write_timeout": 60
            }
        }

    def _get_default_performance_config(self) -> Dict[str, Any]:
        """Default performance optimization configuration"""
        return {
            "caching": {
                "enable_api_cache": True,
                "enable_result_cache": True,
                "enable_model_cache": True,
                "cache_strategies": {
                    "api_responses": "LRU",
                    "optimization_results": "TTL",
                    "model_predictions": "LFU"
                }
            },
            "optimization": {
                "enable_async_processing": True,
                "enable_batch_processing": True,
                "enable_connection_pooling": True,
                "worker_processes": 4,
                "thread_pool_size": 20
            },
            "compression": {
                "enable_response_compression": True,
                "compression_level": 6,
                "compression_threshold": 1024
            },
            "preloading": {
                "preload_models": True,
                "preload_optimizers": True,
                "warmup_requests": 10
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def save_to_file(self, file_path: str):
        """Save configuration to file"""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, file_path: str) -> 'ProductionConfig':
        """Load configuration from file"""
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)

    @classmethod
    def from_env(cls) -> 'ProductionConfig':
        """Load configuration from environment variables"""
        return cls(
            environment=os.getenv("ENVIRONMENT", "production"),
            debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            api_workers=int(os.getenv("API_WORKERS", "4")),
            database_url=os.getenv("DATABASE_URL", "postgresql://svgai:password@localhost:5432/svgai_prod"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0")
        )


class ProductionConfigManager:
    """Manages production configuration for the 4-tier system"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_path = config_path or "/app/config/production_config.json"
        self.config: Optional[ProductionConfig] = None
        self._load_config()

    def _load_config(self):
        """Load production configuration"""
        try:
            if Path(self.config_path).exists():
                self.config = ProductionConfig.from_file(self.config_path)
                logging.info(f"Loaded production config from {self.config_path}")
            else:
                self.config = ProductionConfig.from_env()
                logging.info("Loaded production config from environment variables")

                # Save default config to file
                self.save_config()

        except Exception as e:
            logging.error(f"Failed to load production config: {e}")
            self.config = ProductionConfig()  # Use defaults

    def save_config(self):
        """Save current configuration to file"""
        if self.config:
            self.config.save_to_file(self.config_path)
            logging.info(f"Saved production config to {self.config_path}")

    def get_config(self) -> ProductionConfig:
        """Get current configuration"""
        return self.config or ProductionConfig()

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        if self.config:
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            self.save_config()

    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }

        if not self.config:
            validation_result["valid"] = False
            validation_result["errors"].append("No configuration loaded")
            return validation_result

        # Validate database configuration
        if not self.config.database_url:
            validation_result["errors"].append("Database URL not configured")
            validation_result["valid"] = False

        # Validate Redis configuration
        if not self.config.redis_url:
            validation_result["errors"].append("Redis URL not configured")
            validation_result["valid"] = False

        # Validate API keys
        if not self.config.api_keys or len(self.config.api_keys) == 0:
            validation_result["warnings"].append("No API keys configured")

        # Validate resource limits
        if self.config.tier_system_config["max_concurrent_requests"] > 100:
            validation_result["warnings"].append("High concurrent request limit may impact performance")

        # Validate timeout settings
        total_timeout = sum(self.config.tier_system_config["tier_timeouts"].values())
        if total_timeout > 300:  # 5 minutes
            validation_result["recommendations"].append("Consider reducing tier timeout values for better responsiveness")

        return validation_result

    def get_docker_env_vars(self) -> Dict[str, str]:
        """Get environment variables for Docker deployment"""
        return {
            "ENVIRONMENT": self.config.environment,
            "DEBUG_MODE": str(self.config.debug_mode).lower(),
            "LOG_LEVEL": self.config.log_level,
            "API_HOST": self.config.api_host,
            "API_PORT": str(self.config.api_port),
            "API_WORKERS": str(self.config.api_workers),
            "DATABASE_URL": self.config.database_url,
            "REDIS_URL": self.config.redis_url,
            "MAX_REQUEST_SIZE": str(self.config.max_request_size),
            "REQUEST_TIMEOUT": str(self.config.request_timeout)
        }

    def get_kubernetes_config(self) -> Dict[str, Any]:
        """Get configuration for Kubernetes deployment"""
        return {
            "replicas": {
                "api": 3,
                "worker": 2,
                "database": 1,
                "redis": 1
            },
            "resources": {
                "api": {
                    "requests": {
                        "memory": "1Gi",
                        "cpu": "1000m"
                    },
                    "limits": {
                        "memory": self.config.resource_limits["memory"]["api_service"],
                        "cpu": self.config.resource_limits["cpu"]["api_service"]
                    }
                },
                "worker": {
                    "requests": {
                        "memory": "2Gi",
                        "cpu": "2000m"
                    },
                    "limits": {
                        "memory": self.config.resource_limits["memory"]["worker_service"],
                        "cpu": self.config.resource_limits["cpu"]["worker_service"]
                    }
                }
            },
            "autoscaling": {
                "api": {
                    "min_replicas": 2,
                    "max_replicas": 10,
                    "target_cpu_percentage": 70,
                    "target_memory_percentage": 80
                },
                "worker": {
                    "min_replicas": 1,
                    "max_replicas": 5,
                    "target_cpu_percentage": 80,
                    "target_memory_percentage": 85
                }
            }
        }


# Global configuration instance
config_manager = ProductionConfigManager()

def get_production_config() -> ProductionConfig:
    """Get global production configuration"""
    return config_manager.get_config()

def validate_production_setup() -> Dict[str, Any]:
    """Validate complete production setup"""
    validation_result = config_manager.validate_config()

    # Additional production-specific validations
    config = config_manager.get_config()

    # Check security configuration
    if config.debug_mode:
        validation_result["errors"].append("Debug mode enabled in production")
        validation_result["valid"] = False

    # Check CORS origins
    if not config.cors_origins or "localhost" in str(config.cors_origins):
        validation_result["warnings"].append("CORS origins may include development URLs")

    # Check rate limiting
    if config.rate_limiting["requests_per_minute"] > 500:
        validation_result["warnings"].append("High rate limits may allow abuse")

    return validation_result


if __name__ == "__main__":
    # Example usage
    config_manager = ProductionConfigManager()
    config = config_manager.get_config()

    print("Production Configuration:")
    print(f"Environment: {config.environment}")
    print(f"API Port: {config.api_port}")
    print(f"Max Concurrent Requests: {config.tier_system_config['max_concurrent_requests']}")

    # Validate configuration
    validation = validate_production_setup()
    print(f"\nConfiguration Valid: {validation['valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")