"""
Pipeline Configuration System - Task 3 Implementation
Flexible configuration management for the AI pipeline with hot-reloading and validation.
"""

import json
import os
import threading
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import copy

# Optional imports
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ComponentConfig:
    """Configuration for a single component."""
    primary: str
    fallback: Optional[str] = None
    confidence_threshold: float = 0.7
    cache_enabled: bool = True
    timeout_seconds: float = 30.0
    retry_attempts: int = 2
    enabled: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TierConfig:
    """Configuration for a processing tier."""
    max_time: float
    methods: List[str]
    quality_threshold: float = 0.8
    description: str = ""
    enabled: bool = True


@dataclass
class PipelineConfiguration:
    """Complete pipeline configuration."""

    # Component configurations
    classifier: ComponentConfig = field(default_factory=lambda: ComponentConfig(
        primary="statistical_classifier",
        fallback="rule_based_classifier",
        confidence_threshold=0.7
    ))

    optimizer: ComponentConfig = field(default_factory=lambda: ComponentConfig(
        primary="learned_optimizer",
        fallback="correlation_formulas",
        confidence_threshold=0.6
    ))

    quality_predictor: ComponentConfig = field(default_factory=lambda: ComponentConfig(
        primary="quality_prediction_integrator",
        fallback=None,
        confidence_threshold=0.8
    ))

    router: ComponentConfig = field(default_factory=lambda: ComponentConfig(
        primary="intelligent_router",
        fallback="default_router",
        confidence_threshold=0.7
    ))

    converter: ComponentConfig = field(default_factory=lambda: ComponentConfig(
        primary="ai_enhanced_converter",
        fallback="vtracer_converter",
        confidence_threshold=0.9
    ))

    feature_extractor: ComponentConfig = field(default_factory=lambda: ComponentConfig(
        primary="image_feature_extractor",
        fallback=None,
        confidence_threshold=0.95
    ))

    # Tier configurations
    tiers: Dict[str, TierConfig] = field(default_factory=lambda: {
        "tier1": TierConfig(
            max_time=2.0,
            methods=["statistical", "formulas"],
            quality_threshold=0.75,
            description="Fast processing with basic optimization"
        ),
        "tier2": TierConfig(
            max_time=5.0,
            methods=["statistical", "learned", "regression"],
            quality_threshold=0.85,
            description="Balanced processing with moderate optimization"
        ),
        "tier3": TierConfig(
            max_time=15.0,
            methods=["all"],
            quality_threshold=0.95,
            description="Comprehensive processing with full optimization"
        )
    })

    # Global pipeline settings
    global_settings: Dict[str, Any] = field(default_factory=lambda: {
        "enable_caching": True,
        "enable_fallbacks": True,
        "performance_mode": "balanced",  # fast, balanced, quality
        "max_concurrent_jobs": 4,
        "default_target_quality": 0.85,
        "default_time_constraint": 30.0,
        "enable_monitoring": True,
        "log_level": "INFO",
        "enable_profiling": False
    })

    # Environment-specific settings
    environment: str = "development"
    version: str = "1.0.0"
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass


class PipelineConfigManager:
    """
    Configuration manager for the AI pipeline with hot-reloading and validation.
    """

    def __init__(self,
                 config_path: Optional[str] = None,
                 enable_hot_reload: bool = True,
                 auto_save: bool = True):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
            enable_hot_reload: Whether to enable hot-reloading
            auto_save: Whether to auto-save changes
        """
        self.config_path = config_path or self._get_default_config_path()
        self.enable_hot_reload = enable_hot_reload
        self.auto_save = auto_save

        # Configuration state
        self._config = PipelineConfiguration()
        self._config_lock = threading.RLock()
        self._observers = []
        self._change_callbacks = []

        # Hot-reload setup
        self._file_observer = None
        self._last_modified = 0

        # Load initial configuration
        self._load_configuration()

        # Setup hot-reloading if enabled
        if self.enable_hot_reload:
            self._setup_hot_reload()

        logger.info(f"PipelineConfigManager initialized (path={self.config_path}, "
                   f"hot_reload={enable_hot_reload})")

    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        config_dir = Path(__file__).parent.parent.parent.parent / "config"
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / "pipeline_config.json")

    def _load_configuration(self):
        """Load configuration from file."""
        with self._config_lock:
            if Path(self.config_path).exists():
                try:
                    config_data = self._load_config_file()
                    self._config = self._merge_config(config_data)
                    logger.info(f"Configuration loaded from {self.config_path}")
                except Exception as e:
                    logger.error(f"Failed to load configuration: {e}")
                    logger.info("Using default configuration")
            else:
                logger.info(f"Configuration file not found, using defaults: {self.config_path}")
                # Save default configuration
                if self.auto_save:
                    self.save_configuration()

    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration from file (JSON or YAML)."""
        file_path = Path(self.config_path)

        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                if YAML_AVAILABLE:
                    return yaml.safe_load(f) or {}
                else:
                    raise ImportError("YAML support requires 'pyyaml' package")
            else:
                return json.load(f)

    def _merge_config(self, config_data: Dict[str, Any]) -> PipelineConfiguration:
        """Merge loaded configuration with defaults."""
        # Start with default configuration
        config = PipelineConfiguration()

        # Update with loaded data
        if 'classifier' in config_data:
            config.classifier = ComponentConfig(**config_data['classifier'])
        if 'optimizer' in config_data:
            config.optimizer = ComponentConfig(**config_data['optimizer'])
        if 'quality_predictor' in config_data:
            config.quality_predictor = ComponentConfig(**config_data['quality_predictor'])
        if 'router' in config_data:
            config.router = ComponentConfig(**config_data['router'])
        if 'converter' in config_data:
            config.converter = ComponentConfig(**config_data['converter'])
        if 'feature_extractor' in config_data:
            config.feature_extractor = ComponentConfig(**config_data['feature_extractor'])

        if 'tiers' in config_data:
            config.tiers = {
                name: TierConfig(**tier_data)
                for name, tier_data in config_data['tiers'].items()
            }

        if 'global_settings' in config_data:
            config.global_settings.update(config_data['global_settings'])

        # Update metadata
        config.environment = config_data.get('environment', config.environment)
        config.version = config_data.get('version', config.version)
        config.last_updated = datetime.now().isoformat()

        return config

    def _setup_hot_reload(self):
        """Setup file system watcher for hot-reloading."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available, hot-reload disabled")
            return

        try:
            class ConfigFileHandler(FileSystemEventHandler):
                def __init__(self, config_manager):
                    self.config_manager = config_manager

                def on_modified(self, event):
                    if event.src_path == self.config_manager.config_path:
                        # Debounce file changes
                        current_time = time.time()
                        if current_time - self.config_manager._last_modified > 1.0:
                            self.config_manager._last_modified = current_time
                            self.config_manager._reload_configuration()

            self._file_observer = Observer()
            handler = ConfigFileHandler(self)

            config_dir = str(Path(self.config_path).parent)
            self._file_observer.schedule(handler, config_dir, recursive=False)
            self._file_observer.start()

            logger.info("Hot-reload enabled for configuration file")

        except Exception as e:
            logger.error(f"Failed to setup hot-reload: {e}")

    def _reload_configuration(self):
        """Reload configuration from file."""
        logger.info("Reloading configuration due to file change")
        old_config = copy.deepcopy(self._config)

        try:
            self._load_configuration()

            # Validate new configuration
            self.validate_configuration()

            # Notify observers of changes
            self._notify_change_callbacks(old_config, self._config)

            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            # Restore old configuration
            with self._config_lock:
                self._config = old_config

    def get_configuration(self) -> PipelineConfiguration:
        """Get current configuration (thread-safe copy)."""
        with self._config_lock:
            return copy.deepcopy(self._config)

    def update_configuration(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates

        Returns:
            True if update successful, False otherwise
        """
        with self._config_lock:
            old_config = copy.deepcopy(self._config)

            try:
                # Apply updates
                updated_data = asdict(self._config)
                self._deep_update(updated_data, updates)

                # Create new configuration
                new_config = self._merge_config(updated_data)

                # Validate new configuration
                self._validate_config_object(new_config)

                # Apply changes
                self._config = new_config
                self._config.last_updated = datetime.now().isoformat()

                # Auto-save if enabled
                if self.auto_save:
                    self.save_configuration()

                # Notify observers
                self._notify_change_callbacks(old_config, self._config)

                logger.info("Configuration updated successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to update configuration: {e}")
                # Restore old configuration
                self._config = old_config
                return False

    def _deep_update(self, base_dict: Dict[str, Any], updates: Dict[str, Any]):
        """Deep update dictionary with nested values."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def save_configuration(self, path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.

        Args:
            path: Optional path to save to (defaults to current config_path)

        Returns:
            True if save successful, False otherwise
        """
        save_path = path or self.config_path

        try:
            with self._config_lock:
                config_dict = asdict(self._config)

            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            # Save based on file extension
            file_path = Path(save_path)
            with open(file_path, 'w') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    if YAML_AVAILABLE:
                        yaml.safe_dump(config_dict, f, indent=2, default_flow_style=False)
                    else:
                        raise ImportError("YAML support requires 'pyyaml' package")
                else:
                    json.dump(config_dict, f, indent=2, default=str)

            logger.info(f"Configuration saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def validate_configuration(self) -> bool:
        """
        Validate current configuration.

        Returns:
            True if valid, raises ConfigValidationError if invalid
        """
        with self._config_lock:
            return self._validate_config_object(self._config)

    def _validate_config_object(self, config: PipelineConfiguration) -> bool:
        """Validate a configuration object."""
        errors = []

        # Validate component configurations
        components = ['classifier', 'optimizer', 'quality_predictor', 'router', 'converter', 'feature_extractor']
        for component_name in components:
            component_config = getattr(config, component_name)
            component_errors = self._validate_component_config(component_name, component_config)
            errors.extend(component_errors)

        # Validate tier configurations
        for tier_name, tier_config in config.tiers.items():
            tier_errors = self._validate_tier_config(tier_name, tier_config)
            errors.extend(tier_errors)

        # Validate global settings
        global_errors = self._validate_global_settings(config.global_settings)
        errors.extend(global_errors)

        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")

        return True

    def _validate_component_config(self, name: str, config: ComponentConfig) -> List[str]:
        """Validate a component configuration."""
        errors = []

        if not config.primary:
            errors.append(f"{name}.primary cannot be empty")

        if not 0.0 <= config.confidence_threshold <= 1.0:
            errors.append(f"{name}.confidence_threshold must be between 0.0 and 1.0")

        if config.timeout_seconds <= 0:
            errors.append(f"{name}.timeout_seconds must be positive")

        if config.retry_attempts < 0:
            errors.append(f"{name}.retry_attempts must be non-negative")

        return errors

    def _validate_tier_config(self, name: str, config: TierConfig) -> List[str]:
        """Validate a tier configuration."""
        errors = []

        if config.max_time <= 0:
            errors.append(f"tier {name}.max_time must be positive")

        if not config.methods:
            errors.append(f"tier {name}.methods cannot be empty")

        if not 0.0 <= config.quality_threshold <= 1.0:
            errors.append(f"tier {name}.quality_threshold must be between 0.0 and 1.0")

        return errors

    def _validate_global_settings(self, settings: Dict[str, Any]) -> List[str]:
        """Validate global settings."""
        errors = []

        if 'max_concurrent_jobs' in settings and settings['max_concurrent_jobs'] <= 0:
            errors.append("global_settings.max_concurrent_jobs must be positive")

        if 'default_target_quality' in settings:
            quality = settings['default_target_quality']
            if not 0.0 <= quality <= 1.0:
                errors.append("global_settings.default_target_quality must be between 0.0 and 1.0")

        if 'performance_mode' in settings:
            valid_modes = ['fast', 'balanced', 'quality']
            if settings['performance_mode'] not in valid_modes:
                errors.append(f"global_settings.performance_mode must be one of {valid_modes}")

        return errors

    def get_component_config(self, component_name: str) -> ComponentConfig:
        """Get configuration for a specific component."""
        with self._config_lock:
            if hasattr(self._config, component_name):
                return copy.deepcopy(getattr(self._config, component_name))
            else:
                raise ValueError(f"Unknown component: {component_name}")

    def get_tier_config(self, tier_name: str) -> TierConfig:
        """Get configuration for a specific tier."""
        with self._config_lock:
            if tier_name in self._config.tiers:
                return copy.deepcopy(self._config.tiers[tier_name])
            else:
                raise ValueError(f"Unknown tier: {tier_name}")

    def get_global_setting(self, setting_name: str, default: Any = None) -> Any:
        """Get a global setting value."""
        with self._config_lock:
            return self._config.global_settings.get(setting_name, default)

    def add_change_callback(self, callback):
        """
        Add callback to be called when configuration changes.

        Args:
            callback: Function that takes (old_config, new_config) parameters
        """
        self._change_callbacks.append(callback)

    def remove_change_callback(self, callback):
        """Remove a change callback."""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)

    def _notify_change_callbacks(self, old_config: PipelineConfiguration, new_config: PipelineConfiguration):
        """Notify all change callbacks of configuration change."""
        for callback in self._change_callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                logger.error(f"Error in change callback: {e}")

    def get_environment_config_path(self, environment: str) -> str:
        """Get configuration path for specific environment."""
        base_path = Path(self.config_path)
        env_path = base_path.parent / f"pipeline_config_{environment}.json"
        return str(env_path)

    def switch_environment(self, environment: str) -> bool:
        """
        Switch to a different environment configuration.

        Args:
            environment: Environment name (e.g., 'development', 'production')

        Returns:
            True if switch successful, False otherwise
        """
        env_config_path = self.get_environment_config_path(environment)

        if Path(env_config_path).exists():
            old_path = self.config_path
            self.config_path = env_config_path

            try:
                self._load_configuration()
                logger.info(f"Switched to {environment} environment")
                return True
            except Exception as e:
                logger.error(f"Failed to switch environment: {e}")
                self.config_path = old_path
                return False
        else:
            logger.warning(f"Environment config not found: {env_config_path}")
            return False

    def create_environment_config(self, environment: str, base_updates: Dict[str, Any] = None) -> bool:
        """
        Create configuration for a new environment.

        Args:
            environment: Environment name
            base_updates: Optional updates to apply to base configuration

        Returns:
            True if creation successful, False otherwise
        """
        env_config_path = self.get_environment_config_path(environment)

        # Start with current configuration
        with self._config_lock:
            env_config = copy.deepcopy(self._config)

        # Apply environment-specific updates
        env_config.environment = environment

        if base_updates:
            env_config_dict = asdict(env_config)
            self._deep_update(env_config_dict, base_updates)
            env_config = self._merge_config(env_config_dict)

        # Save environment configuration
        try:
            env_config_dict = asdict(env_config)
            with open(env_config_path, 'w') as f:
                json.dump(env_config_dict, f, indent=2, default=str)

            logger.info(f"Created {environment} environment configuration")
            return True
        except Exception as e:
            logger.error(f"Failed to create environment config: {e}")
            return False

    def cleanup(self):
        """Cleanup resources (file watchers, etc.)."""
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()

        logger.info("Configuration manager cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def test_pipeline_config():
    """Test the pipeline configuration system."""
    print("Testing Pipeline Configuration System...")

    # Test default configuration
    print("\n1. Testing default configuration:")
    config_manager = PipelineConfigManager(
        config_path="/tmp/claude/test_pipeline_config.json",
        enable_hot_reload=False
    )

    config = config_manager.get_configuration()
    print(f"   ✓ Default config loaded: {config.environment}")
    print(f"   ✓ Classifier primary: {config.classifier.primary}")
    print(f"   ✓ Tiers available: {list(config.tiers.keys())}")

    # Test configuration validation
    print("\n2. Testing configuration validation:")
    try:
        is_valid = config_manager.validate_configuration()
        print(f"   ✓ Configuration valid: {is_valid}")
    except ConfigValidationError as e:
        print(f"   ✗ Configuration invalid: {e}")

    # Test configuration updates
    print("\n3. Testing configuration updates:")
    updates = {
        'global_settings': {
            'performance_mode': 'quality',
            'default_target_quality': 0.95
        },
        'classifier': {
            'confidence_threshold': 0.8
        }
    }

    success = config_manager.update_configuration(updates)
    print(f"   ✓ Configuration updated: {success}")

    if success:
        updated_config = config_manager.get_configuration()
        print(f"   ✓ Performance mode: {updated_config.global_settings['performance_mode']}")
        print(f"   ✓ Classifier threshold: {updated_config.classifier.confidence_threshold}")

    # Test component and tier access
    print("\n4. Testing component and tier access:")
    classifier_config = config_manager.get_component_config('classifier')
    print(f"   ✓ Classifier config: {classifier_config.primary}")

    tier2_config = config_manager.get_tier_config('tier2')
    print(f"   ✓ Tier 2 config: {tier2_config.max_time}s, methods={tier2_config.methods}")

    # Test global settings
    print("\n5. Testing global settings:")
    performance_mode = config_manager.get_global_setting('performance_mode')
    max_jobs = config_manager.get_global_setting('max_concurrent_jobs', 1)
    print(f"   ✓ Performance mode: {performance_mode}")
    print(f"   ✓ Max concurrent jobs: {max_jobs}")

    # Test save/load
    print("\n6. Testing save/load:")
    save_success = config_manager.save_configuration()
    print(f"   ✓ Configuration saved: {save_success}")

    # Test environment configuration
    print("\n7. Testing environment configuration:")
    env_updates = {
        'global_settings': {
            'performance_mode': 'fast',
            'enable_profiling': True
        }
    }

    env_created = config_manager.create_environment_config('test', env_updates)
    print(f"   ✓ Test environment created: {env_created}")

    # Cleanup
    config_manager.cleanup()

    print("\n✓ Pipeline configuration tests completed")

    return config_manager


if __name__ == "__main__":
    test_pipeline_config()