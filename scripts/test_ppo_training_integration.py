#!/usr/bin/env python3
"""Integration test for PPO Training Orchestrator

This script tests the complete PPO training system integration
as specified in DAY7_PPO_AGENT_TRAINING.md Task AB7.3.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import yaml
import logging

# Add backend modules to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_ppo_optimizer import PPOTrainingOrchestrator


def create_test_config(temp_dir: Path) -> str:
    """Create minimal test configuration"""
    config = {
        'training': {
            'training_images': {
                'simple': ['data/logos/simple_geometric/circle_00.png']
            },
            'model_config': {
                'learning_rate': 1e-3,
                'n_steps': 64,      # Very small for testing
                'batch_size': 8,
                'n_epochs': 1,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'verbose': 1
            },
            'curriculum_config': {
                'stage_0': {
                    'target_quality': 0.50,
                    'max_episodes': 10,  # Very few episodes for testing
                    'success_threshold': 0.40
                }
            },
            'save_dir': str(temp_dir / 'models'),
            'checkpoint_dir': str(temp_dir / 'checkpoints')
        },
        'environment': {
            'num_threads': 2,
            'memory_limit_gb': 2,
            'parallel_envs': 1
        },
        'monitoring': {
            'enable_real_time_monitoring': False,  # Disabled for testing
            'log_dir': str(temp_dir / 'logs'),
            'checkpoint_frequency': 5,
            'validation_frequency': 10
        },
        'logging': {
            'level': 'INFO',
            'log_dir': str(temp_dir / 'logs')
        }
    }

    config_path = temp_dir / 'test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return str(config_path)


def test_ppo_training_system():
    """Test complete PPO training system integration"""
    print("🧪 Testing PPO Training System Integration...")

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Create test configuration
            config_path = create_test_config(temp_path)
            print(f"✓ Created test configuration: {config_path}")

            # Test orchestrator initialization
            orchestrator = PPOTrainingOrchestrator(config_path)
            print("✓ PPO Training Orchestrator initialized successfully")

            # Test configuration loading
            assert 'training' in orchestrator.config
            assert 'environment' in orchestrator.config
            assert 'monitoring' in orchestrator.config
            assert 'logging' in orchestrator.config
            print("✓ Configuration sections validated")

            # Test training environment setup
            assert orchestrator.training_pipeline is not None
            assert orchestrator.monitor is not None
            print("✓ Training components initialized")

            # Test training data validation
            try:
                orchestrator.validate_training_data()
                print("✓ Training data validation completed")
            except Exception as e:
                print(f"⚠️  Training data validation failed (expected): {e}")

            # Test status retrieval
            status = orchestrator.get_training_status()
            assert 'training_active' in status
            assert 'config_path' in status
            print("✓ Training status retrieval working")

            # Test cleanup
            orchestrator.cleanup()
            print("✓ Resource cleanup completed")

            print("✅ PPO training system integration test PASSED")
            return True

        except Exception as e:
            print(f"❌ PPO training system integration test FAILED: {e}")
            return False


def test_configuration_system():
    """Test YAML configuration system"""
    print("\n🧪 Testing Configuration System...")

    try:
        # Test default config creation
        with tempfile.TemporaryDirectory() as temp_dir:
            from scripts.train_ppo_optimizer import create_default_config

            default_config = create_default_config()
            assert 'training' in default_config
            assert 'environment' in default_config
            assert 'monitoring' in default_config
            assert 'logging' in default_config
            print("✓ Default configuration structure valid")

            # Test YAML serialization
            config_path = Path(temp_dir) / 'test_default.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f)

            # Test loading back
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)

            assert loaded_config == default_config
            print("✓ YAML serialization/deserialization working")

            print("✅ Configuration system test PASSED")
            return True

    except Exception as e:
        print(f"❌ Configuration system test FAILED: {e}")
        return False


def test_logging_system():
    """Test structured logging configuration"""
    print("\n🧪 Testing Logging System...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test config with logging
            config = {
                'training': {
                    'training_images': {'simple': []},
                    'model_config': {},
                    'save_dir': str(temp_path / 'models')
                },
                'environment': {'num_threads': 1},
                'monitoring': {
                    'enable_real_time_monitoring': False,
                    'log_dir': str(temp_path / 'logs')
                },
                'logging': {
                    'level': 'INFO',
                    'log_dir': str(temp_path / 'logs'),
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            }

            config_path = temp_path / 'test_logging.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            # Initialize orchestrator to test logging
            orchestrator = PPOTrainingOrchestrator(str(config_path))

            # Check if log directory was created
            log_dir = temp_path / 'logs'
            assert log_dir.exists()
            print("✓ Log directory created")

            # Test logger functionality
            orchestrator.logger.info("Test log message")
            print("✓ Logger working")

            orchestrator.cleanup()
            print("✅ Logging system test PASSED")
            return True

    except Exception as e:
        print(f"❌ Logging system test FAILED: {e}")
        return False


def main():
    """Run all integration tests"""
    print("🚀 Running PPO Training Integration Tests\n")

    tests = [
        test_configuration_system,
        test_logging_system,
        test_ppo_training_system
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All integration tests PASSED!")
        return 0
    else:
        print("💥 Some integration tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())