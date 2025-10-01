"""
Routing Configuration & Tuning - Task 5 Implementation
Configurable routing system with hot reload and A/B testing support.
"""

import logging
import json
import yaml
import threading
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Configuration source types."""
    FILE = "file"
    DICT = "dict"
    DEFAULT = "default"


@dataclass
class RoutingConfig:
    """Routing configuration container."""
    # Complexity weights for scoring
    complexity_weights: Dict[str, float]

    # Tier thresholds
    tier_thresholds: Dict[str, float]

    # Quality boost settings
    quality_boost: Dict[str, int]

    # Time constraint settings
    time_constraints: Dict[str, Any]

    # Load balancing settings
    load_balancing: Dict[str, Any]

    # A/B testing configuration
    ab_testing: Dict[str, Any]

    # Feature flags
    features: Dict[str, bool]

    # Version and metadata
    version: str
    name: str
    description: str


class RoutingConfigManager:
    """
    Manages routing configuration with support for hot reload,
    validation, and A/B testing.
    """

    DEFAULT_CONFIG = {
        "version": "1.0.0",
        "name": "default",
        "description": "Default routing configuration",
        "complexity_weights": {
            "spatial": 0.3,
            "color": 0.2,
            "edge": 0.3,
            "gradient": 0.15,
            "texture": 0.05
        },
        "tier_thresholds": {
            "tier1_max": 0.3,
            "tier2_max": 0.7
        },
        "quality_boost": {
            "high_quality": 1,
            "medium_quality": 0
        },
        "time_constraints": {
            "strict": True,
            "tier1_max_time": 2.0,
            "tier2_max_time": 5.0,
            "tier3_max_time": 15.0,
            "buffer_factor": 1.2
        },
        "load_balancing": {
            "enabled": True,
            "overload_threshold": 0.9,
            "high_load_threshold": 0.7,
            "downgrade_on_overload": True,
            "queue_priority": True
        },
        "ab_testing": {
            "enabled": False,
            "test_percentage": 0.0,
            "test_config": None
        },
        "features": {
            "ml_prediction": True,
            "adaptive_routing": True,
            "analytics": True,
            "caching": True,
            "parallel_processing": True
        }
    }

    def __init__(self,
                 config_path: Optional[str] = None,
                 auto_reload: bool = True,
                 reload_interval: int = 60):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
            auto_reload: Enable automatic configuration reload
            reload_interval: Reload check interval in seconds
        """
        self.config_path = Path(config_path) if config_path else None
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval

        # Current configurations
        self.primary_config = None
        self.test_config = None

        # Config source tracking
        self.config_source = ConfigSource.DEFAULT
        self.config_hash = None

        # A/B testing state
        self.ab_test_active = False
        self.ab_test_assignments = {}  # request_id -> config_name

        # Thread safety
        self._lock = threading.RLock()

        # Load initial configuration
        self.load_configuration()

        # Start reload thread if enabled
        self._stop_reload = False
        self._reload_thread = None
        if self.auto_reload and self.config_path:
            self._start_reload_thread()

        logger.info(f"RoutingConfigManager initialized (source={self.config_source.value})")

    def load_configuration(self, config_dict: Optional[Dict] = None) -> bool:
        """
        Load configuration from file or dictionary.

        Args:
            config_dict: Optional configuration dictionary

        Returns:
            Success flag
        """
        with self._lock:
            try:
                # Load from dict if provided
                if config_dict:
                    config_data = config_dict
                    self.config_source = ConfigSource.DICT
                # Load from file if path exists
                elif self.config_path and self.config_path.exists():
                    config_data = self._load_from_file(self.config_path)
                    self.config_source = ConfigSource.FILE
                # Use default
                else:
                    config_data = self.DEFAULT_CONFIG.copy()
                    self.config_source = ConfigSource.DEFAULT

                # Validate configuration
                validated_config = self._validate_config(config_data)

                # Create config object
                new_config = RoutingConfig(**validated_config)

                # Check if config changed
                new_hash = self._calculate_config_hash(validated_config)
                if new_hash != self.config_hash:
                    self.primary_config = new_config
                    self.config_hash = new_hash

                    # Handle A/B test configuration
                    if validated_config['ab_testing']['enabled']:
                        self._setup_ab_testing(validated_config['ab_testing'])

                    logger.info(f"Loaded configuration: {new_config.name} (v{new_config.version})")
                    return True

                return False  # No changes

            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                # Fall back to default if not already using it
                if self.primary_config is None:
                    self.primary_config = RoutingConfig(**self.DEFAULT_CONFIG)
                    self.config_source = ConfigSource.DEFAULT
                return False

    def _load_from_file(self, path: Path) -> Dict:
        """Load configuration from file."""
        if path.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    def _validate_config(self, config: Dict) -> Dict:
        """
        Validate and merge configuration with defaults.

        Args:
            config: Configuration dictionary

        Returns:
            Validated configuration
        """
        # Start with defaults
        validated = self.DEFAULT_CONFIG.copy()

        # Deep merge provided config
        self._deep_merge(validated, config)

        # Validate specific constraints
        self._validate_constraints(validated)

        return validated

    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge update into base."""
        for key, value in update.items():
            if key in base:
                if isinstance(base[key], dict) and isinstance(value, dict):
                    self._deep_merge(base[key], value)
                else:
                    base[key] = value
            else:
                base[key] = value

    def _validate_constraints(self, config: Dict):
        """Validate configuration constraints."""
        # Complexity weights should sum to 1
        weights = config['complexity_weights']
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            # Normalize weights
            for key in weights:
                weights[key] /= weight_sum

        # Tier thresholds should be in order
        if config['tier_thresholds']['tier1_max'] >= config['tier_thresholds']['tier2_max']:
            raise ValueError("tier1_max must be less than tier2_max")

        # Time constraints should be positive
        for key, value in config['time_constraints'].items():
            if key.endswith('_time') and value <= 0:
                raise ValueError(f"{key} must be positive")

        # Load thresholds should be between 0 and 1
        if not 0 < config['load_balancing']['high_load_threshold'] < 1:
            raise ValueError("high_load_threshold must be between 0 and 1")
        if not 0 < config['load_balancing']['overload_threshold'] < 1:
            raise ValueError("overload_threshold must be between 0 and 1")

        # A/B test percentage should be between 0 and 100
        if not 0 <= config['ab_testing']['test_percentage'] <= 100:
            raise ValueError("test_percentage must be between 0 and 100")

    def _calculate_config_hash(self, config: Dict) -> str:
        """Calculate hash of configuration for change detection."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _setup_ab_testing(self, ab_config: Dict):
        """Setup A/B testing configuration."""
        if ab_config.get('test_config'):
            try:
                # Load test configuration
                test_config_data = ab_config['test_config']
                validated_test = self._validate_config(test_config_data)
                self.test_config = RoutingConfig(**validated_test)
                self.ab_test_active = True
                logger.info(f"A/B testing enabled with {ab_config['test_percentage']}% traffic")
            except Exception as e:
                logger.error(f"Failed to setup A/B testing: {e}")
                self.ab_test_active = False

    def get_config(self, request_id: Optional[str] = None) -> RoutingConfig:
        """
        Get configuration for a request.

        Args:
            request_id: Optional request ID for A/B testing

        Returns:
            Routing configuration
        """
        with self._lock:
            # Check A/B testing
            if self.ab_test_active and request_id:
                config_name = self._get_ab_assignment(request_id)
                if config_name == 'test':
                    return self.test_config

            return self.primary_config

    def _get_ab_assignment(self, request_id: str) -> str:
        """Get A/B test assignment for request."""
        # Check existing assignment
        if request_id in self.ab_test_assignments:
            return self.ab_test_assignments[request_id]

        # Make new assignment
        test_percentage = self.primary_config.ab_testing['test_percentage']
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest()[:8], 16)
        is_test = (hash_value % 100) < test_percentage

        assignment = 'test' if is_test else 'primary'
        self.ab_test_assignments[request_id] = assignment

        # Clean old assignments periodically
        if len(self.ab_test_assignments) > 10000:
            # Keep only last 5000
            items = list(self.ab_test_assignments.items())
            self.ab_test_assignments = dict(items[-5000:])

        return assignment

    def get_complexity_weights(self) -> Dict[str, float]:
        """Get complexity weights."""
        return self.primary_config.complexity_weights.copy()

    def get_tier_threshold(self, complexity: float) -> int:
        """
        Get tier based on complexity threshold.

        Args:
            complexity: Complexity score

        Returns:
            Recommended tier
        """
        if complexity <= self.primary_config.tier_thresholds['tier1_max']:
            return 1
        elif complexity <= self.primary_config.tier_thresholds['tier2_max']:
            return 2
        else:
            return 3

    def should_boost_quality(self, target_quality: float) -> int:
        """
        Check if quality boost should be applied.

        Args:
            target_quality: Target quality score

        Returns:
            Tier boost amount
        """
        if target_quality > 0.95:
            return self.primary_config.quality_boost['high_quality']
        elif target_quality > 0.85:
            return self.primary_config.quality_boost['medium_quality']
        return 0

    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if feature is enabled.

        Args:
            feature: Feature name

        Returns:
            Enabled flag
        """
        return self.primary_config.features.get(feature, False)

    def _start_reload_thread(self):
        """Start configuration reload thread."""
        self._reload_thread = threading.Thread(
            target=self._reload_worker,
            daemon=True
        )
        self._reload_thread.start()

    def _reload_worker(self):
        """Worker thread for configuration reload."""
        while not self._stop_reload:
            try:
                time.sleep(self.reload_interval)

                # Check if file changed
                if self.config_path and self.config_path.exists():
                    current_mtime = self.config_path.stat().st_mtime

                    # Reload if changed
                    if self.load_configuration():
                        logger.info("Configuration reloaded due to file change")

            except Exception as e:
                logger.error(f"Error in reload worker: {e}")

    def reload(self) -> bool:
        """
        Manually reload configuration.

        Returns:
            Success flag
        """
        return self.load_configuration()

    def export_current_config(self, path: str, format: str = 'yaml'):
        """
        Export current configuration to file.

        Args:
            path: Export file path
            format: Export format (yaml or json)
        """
        with self._lock:
            config_dict = asdict(self.primary_config)

            output_path = Path(path)
            if format == 'yaml':
                with open(output_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            elif format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Exported configuration to {output_path}")

    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information."""
        with self._lock:
            return {
                'name': self.primary_config.name,
                'version': self.primary_config.version,
                'source': self.config_source.value,
                'ab_test_active': self.ab_test_active,
                'features': self.primary_config.features,
                'reload_enabled': self.auto_reload
            }

    def shutdown(self):
        """Shutdown configuration manager."""
        self._stop_reload = True
        if self._reload_thread and self._reload_thread.is_alive():
            self._reload_thread.join(timeout=2)
        logger.info("RoutingConfigManager shutdown complete")


def create_sample_configs():
    """Create sample configuration files for testing."""
    # Aggressive configuration - faster but lower quality
    aggressive_config = {
        "version": "1.0.0",
        "name": "aggressive",
        "description": "Aggressive routing for speed",
        "complexity_weights": {
            "spatial": 0.4,
            "color": 0.1,
            "edge": 0.3,
            "gradient": 0.1,
            "texture": 0.1
        },
        "tier_thresholds": {
            "tier1_max": 0.5,
            "tier2_max": 0.85
        },
        "quality_boost": {
            "high_quality": 0,
            "medium_quality": 0
        },
        "time_constraints": {
            "strict": True,
            "tier1_max_time": 1.5,
            "tier2_max_time": 3.0,
            "tier3_max_time": 8.0,
            "buffer_factor": 1.0
        }
    }

    # Conservative configuration - higher quality but slower
    conservative_config = {
        "version": "1.0.0",
        "name": "conservative",
        "description": "Conservative routing for quality",
        "complexity_weights": {
            "spatial": 0.25,
            "color": 0.25,
            "edge": 0.25,
            "gradient": 0.2,
            "texture": 0.05
        },
        "tier_thresholds": {
            "tier1_max": 0.2,
            "tier2_max": 0.5
        },
        "quality_boost": {
            "high_quality": 2,
            "medium_quality": 1
        },
        "time_constraints": {
            "strict": False,
            "tier1_max_time": 3.0,
            "tier2_max_time": 8.0,
            "tier3_max_time": 20.0,
            "buffer_factor": 1.5
        }
    }

    # A/B test configuration
    ab_test_config = {
        "version": "1.0.0",
        "name": "ab_test",
        "description": "A/B testing configuration",
        "ab_testing": {
            "enabled": True,
            "test_percentage": 20,
            "test_config": conservative_config
        }
    }

    return {
        'aggressive': aggressive_config,
        'conservative': conservative_config,
        'ab_test': ab_test_config
    }


def test_routing_config():
    """Test the routing configuration system."""
    print("Testing Routing Configuration...")

    # Test 1: Default configuration
    print("\n✓ Testing default configuration:")
    manager = RoutingConfigManager()
    assert manager.primary_config is not None
    print(f"  Config name: {manager.primary_config.name}")
    print(f"  Version: {manager.primary_config.version}")
    print(f"  Source: {manager.config_source.value}")

    # Test 2: Load from dictionary
    print("\n✓ Testing dictionary configuration:")
    configs = create_sample_configs()
    manager.load_configuration(configs['aggressive'])
    assert manager.primary_config.name == 'aggressive'
    print(f"  Loaded: {manager.primary_config.name}")

    # Test 3: Complexity weights
    print("\n✓ Testing complexity weights:")
    weights = manager.get_complexity_weights()
    weight_sum = sum(weights.values())
    print(f"  Weights sum: {weight_sum:.3f}")
    assert abs(weight_sum - 1.0) < 0.01, "Weights should sum to 1"

    # Test 4: Tier thresholds
    print("\n✓ Testing tier thresholds:")
    assert manager.get_tier_threshold(0.1) == 1
    assert manager.get_tier_threshold(0.6) == 2
    assert manager.get_tier_threshold(0.9) == 3
    print("  Thresholds working correctly")

    # Test 5: Quality boost
    print("\n✓ Testing quality boost:")
    boost_high = manager.should_boost_quality(0.96)
    boost_med = manager.should_boost_quality(0.86)
    boost_low = manager.should_boost_quality(0.75)
    print(f"  High quality boost: {boost_high}")
    print(f"  Medium quality boost: {boost_med}")
    print(f"  Low quality boost: {boost_low}")

    # Test 6: Feature flags
    print("\n✓ Testing feature flags:")
    ml_enabled = manager.is_feature_enabled('ml_prediction')
    fake_feature = manager.is_feature_enabled('nonexistent')
    print(f"  ML prediction: {ml_enabled}")
    print(f"  Nonexistent feature: {fake_feature}")

    # Test 7: A/B testing
    print("\n✓ Testing A/B testing:")
    manager.load_configuration(configs['ab_test'])

    # Simulate requests
    test_count = 0
    control_count = 0
    for i in range(100):
        request_id = f"req_{i}"
        config = manager.get_config(request_id)
        if config.name == 'conservative':
            test_count += 1
        else:
            control_count += 1

    print(f"  Test group: {test_count}%")
    print(f"  Control group: {control_count}%")
    assert 10 < test_count < 30, "A/B split should be around 20%"

    # Test 8: Export configuration
    print("\n✓ Testing configuration export:")
    export_path = "/tmp/routing_config_test.yaml"
    manager.export_current_config(export_path)
    assert Path(export_path).exists()
    print(f"  Exported to {export_path}")

    # Test 9: Configuration info
    print("\n✓ Testing configuration info:")
    info = manager.get_config_info()
    print(f"  Name: {info['name']}")
    print(f"  A/B test active: {info['ab_test_active']}")
    print(f"  Features: {len(info['features'])} enabled")

    # Cleanup
    manager.shutdown()

    print("\n✅ All routing configuration tests passed!")
    return manager


if __name__ == "__main__":
    test_routing_config()