#!/usr/bin/env python3
"""Demonstration of PPO Training Orchestrator

This script demonstrates the PPO Training Orchestrator functionality
with a minimal training run for testing purposes.
"""

import sys
import tempfile
import time
import os
from pathlib import Path
import yaml

# Add backend modules to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_ppo_optimizer import PPOTrainingOrchestrator


def create_demo_config(temp_dir: Path) -> str:
    """Create demonstration configuration"""
    config = {
        'training': {
            'training_images': {
                'simple': [
                    'data/logos/simple_geometric/circle_00.png',
                    'data/logos/simple_geometric/rectangle_00.png'
                ]
            },
            'model_config': {
                'learning_rate': 3e-4,
                'n_steps': 128,      # Small for demo
                'batch_size': 16,
                'n_epochs': 2,
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
                    'target_quality': 0.60,
                    'max_episodes': 50,   # Small number for demo
                    'success_threshold': 0.50
                }
            },
            'save_dir': str(temp_dir / 'demo_models'),
            'checkpoint_dir': str(temp_dir / 'demo_checkpoints')
        },
        'environment': {
            'num_threads': 2,
            'memory_limit_gb': 4,
            'parallel_envs': 1,  # Single environment for demo
            'cuda_visible_devices': ''  # CPU only for demo
        },
        'monitoring': {
            'enable_real_time_monitoring': True,
            'log_dir': str(temp_dir / 'demo_logs'),
            'checkpoint_frequency': 10,
            'validation_frequency': 25,
            'monitoring_frequency': 5
        },
        'logging': {
            'level': 'INFO',
            'log_dir': str(temp_dir / 'demo_logs'),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'experiment': {
            'name': 'ppo_training_demo',
            'description': 'Demonstration of PPO training orchestrator'
        }
    }

    config_path = temp_dir / 'demo_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=2)

    return str(config_path)


def demonstrate_orchestrator_features():
    """Demonstrate key features of the PPO Training Orchestrator"""
    print("🎬 PPO Training Orchestrator Demonstration")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print(f"📁 Using temporary directory: {temp_path}")

        # Create demo configuration
        config_path = create_demo_config(temp_path)
        print(f"📄 Created demo configuration: {config_path}")

        # Display configuration
        with open(config_path, 'r') as f:
            config_content = f.read()
        print("\n📋 Configuration Preview:")
        print("-" * 30)
        for i, line in enumerate(config_content.split('\n')[:20], 1):
            print(f"{i:2d}: {line}")
        if len(config_content.split('\n')) > 20:
            print("    ... (truncated)")

        # Initialize orchestrator
        print("\n🚀 Initializing PPO Training Orchestrator...")
        orchestrator = PPOTrainingOrchestrator(config_path)

        # Show status
        status = orchestrator.get_training_status()
        print(f"✅ Orchestrator initialized successfully!")
        print(f"   Config: {status['config_path']}")
        print(f"   Save Dir: {status['save_dir']}")

        # Demonstrate configuration validation
        print("\n🔍 Validating configuration...")
        try:
            required_sections = ['training', 'environment', 'monitoring', 'logging']
            for section in required_sections:
                assert section in orchestrator.config
                print(f"   ✓ {section} section present")
            print("✅ Configuration validation passed!")
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return

        # Demonstrate environment setup
        print("\n🔧 Environment setup demonstration...")
        print(f"   - Log directory: {orchestrator.config['logging']['log_dir']}")
        print(f"   - Model save directory: {orchestrator.config['training']['save_dir']}")
        print(f"   - CPU threads: {orchestrator.config['environment']['num_threads']}")
        print(f"   - Memory limit: {orchestrator.config['environment']['memory_limit_gb']} GB")

        # Demonstrate training data validation
        print("\n📊 Training data validation...")
        try:
            orchestrator.validate_training_data()
            print("✅ Training data validation completed (files may not exist - that's okay for demo)")
        except Exception as e:
            print(f"⚠️  Training data validation warning: {e}")

        # Show training images configuration
        training_images = orchestrator.config['training']['training_images']
        print("\n📸 Training images configuration:")
        for category, images in training_images.items():
            print(f"   - {category}: {len(images)} images")
            for img in images:
                print(f"     • {img}")

        # Show model configuration
        print("\n🧠 Model configuration:")
        model_config = orchestrator.config['training']['model_config']
        for key, value in model_config.items():
            if key != 'policy_kwargs':
                print(f"   - {key}: {value}")

        # Show curriculum configuration
        print("\n📚 Curriculum configuration:")
        curriculum_config = orchestrator.config['training'].get('curriculum_config', {})
        for stage, settings in curriculum_config.items():
            print(f"   - {stage}:")
            for key, value in settings.items():
                print(f"     • {key}: {value}")

        # Demonstrate monitoring configuration
        print("\n📈 Monitoring configuration:")
        monitoring = orchestrator.config['monitoring']
        for key, value in monitoring.items():
            print(f"   - {key}: {value}")

        # Show directory structure created
        print("\n📁 Directory structure created:")
        for root, dirs, files in os.walk(temp_path):
            level = root.replace(str(temp_path), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

        # Demonstrate cleanup
        print("\n🧹 Cleaning up resources...")
        orchestrator.cleanup()
        print("✅ Cleanup completed successfully!")

        print("\n🎉 Demonstration completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ YAML configuration loading and validation")
        print("✓ Environment setup and resource allocation")
        print("✓ Training pipeline initialization")
        print("✓ Monitoring system setup")
        print("✓ Structured logging configuration")
        print("✓ Directory structure creation")
        print("✓ Resource cleanup")


if __name__ == "__main__":
    import os
    demonstrate_orchestrator_features()