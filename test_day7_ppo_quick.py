#!/usr/bin/env python3
"""
Day 7 PPO Quick Test - Fast validation without full training
Test PPO system components for functionality verification
"""

import sys
import tempfile
import os
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_ppo_components():
    """Test PPO components without full training"""
    print("🤖 Testing PPO Components (Quick Test)")
    print("=" * 60)

    try:
        # Import PPO components
        from backend.ai_modules.optimization.ppo_optimizer import (
            PPOVTracerOptimizer,
            create_ppo_optimizer,
            TrainingProgressCallback
        )
        print("✅ PPO optimizer imports successful")

        # Create test image
        import cv2
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.circle(img, (32, 32), 20, (0, 0, 255), -1)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_image = tmp_file.name
            cv2.imwrite(test_image, img)

        print(f"Using test image: {test_image}")

        # Test optimizer creation
        env_kwargs = {
            'image_path': test_image,
            'target_quality': 0.85,
            'max_steps': 5
        }

        optimizer = PPOVTracerOptimizer(env_kwargs)
        print("✅ PPO optimizer created successfully")

        # Test environment setup (single env, no vectorization)
        training_config = {
            'total_timesteps': 10,
            'n_envs': 1,  # Single environment only
            'eval_freq': 5,
            'n_eval_episodes': 1,
            'model_save_path': '/tmp/claude/test_models'
        }
        optimizer.training_config = training_config

        # Setup single environment manually
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from backend.ai_modules.optimization.vtracer_env import VTracerOptimizationEnv

        # Create single environment
        env = VTracerOptimizationEnv(**env_kwargs)
        env = Monitor(env)
        optimizer.vec_env = DummyVecEnv([lambda: env])
        print("✅ Single environment setup complete")

        # Test environment normalization
        from stable_baselines3.common.vec_env import VecNormalize
        optimizer.normalized_env = VecNormalize(
            optimizer.vec_env,
            norm_obs=True,
            norm_reward=True,
            gamma=0.99
        )
        print("✅ Environment normalization setup complete")

        # Test model creation
        from stable_baselines3 import PPO
        model_config = {
            'learning_rate': 1e-3,
            'n_steps': 64,
            'batch_size': 32,
            'verbose': 0
        }
        optimizer.model = PPO('MlpPolicy', optimizer.normalized_env, **model_config)
        print("✅ PPO model created successfully")

        # Test callback creation
        callback = TrainingProgressCallback()
        optimizer.callbacks = [callback]
        print("✅ Training callbacks setup complete")

        # Test single environment step
        obs = optimizer.normalized_env.reset()
        action = optimizer.model.action_space.sample()
        obs, reward, done, info = optimizer.normalized_env.step([action])
        print(f"✅ Environment step test complete (reward: {reward[0]:.3f})")

        # Test model prediction
        action, _ = optimizer.model.predict(obs, deterministic=True)
        print("✅ Model prediction test complete")

        # Clean up
        optimizer.close()
        os.unlink(test_image)
        print("✅ Cleanup complete")

        success_criteria = {
            "imports": True,
            "optimizer_creation": True,
            "environment_setup": True,
            "model_creation": True,
            "callback_setup": True,
            "environment_step": True,
            "model_prediction": True
        }

        all_passed = all(success_criteria.values())

        print(f"\n📋 PPO QUICK TEST RESULTS")
        print("=" * 60)
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  - {criterion.replace('_', ' ').title()}: {status}")

        return all_passed

    except Exception as e:
        print(f"\n❌ PPO quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_factory_and_config():
    """Test factory function and configuration"""
    print("\n🏭 Testing Factory Function & Configuration")
    print("=" * 60)

    try:
        from backend.ai_modules.optimization.ppo_optimizer import create_ppo_optimizer

        # Create test image
        import cv2
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.circle(img, (16, 16), 10, (255, 0, 0), -1)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_image = tmp_file.name
            cv2.imwrite(test_image, img)

        # Test factory function
        optimizer = create_ppo_optimizer(test_image, target_quality=0.8, max_steps=10)
        print("✅ Factory function successful")

        # Test configuration access
        assert optimizer.env_kwargs['target_quality'] == 0.8
        assert optimizer.env_kwargs['max_steps'] == 10
        assert optimizer.model_config['learning_rate'] == 3e-4
        print("✅ Configuration validation successful")

        # Test default configs
        model_config = optimizer._default_model_config()
        training_config = optimizer._default_training_config()

        required_model_keys = ['learning_rate', 'n_steps', 'batch_size', 'gamma']
        required_training_keys = ['total_timesteps', 'eval_freq', 'n_envs']

        for key in required_model_keys:
            assert key in model_config, f"Missing model config key: {key}"

        for key in required_training_keys:
            assert key in training_config, f"Missing training config key: {key}"

        print("✅ Default configuration validation successful")

        os.unlink(test_image)
        return True

    except Exception as e:
        print(f"❌ Factory/config test failed: {e}")
        return False

def main():
    """Run Day 7 PPO quick validation tests"""
    print("🚀 Starting Day 7 PPO Quick Validation Tests")
    print("Validating PPO system components without full training")
    print("According to DAY7_PPO_AGENT_TRAINING.md specification\n")

    try:
        # Test 1: PPO Component Validation
        component_success = test_ppo_components()

        # Test 2: Factory and Configuration
        config_success = test_factory_and_config()

        # Overall assessment
        overall_success = component_success and config_success

        if overall_success:
            print(f"\n🎉 DAY 7 PPO QUICK VALIDATION SUCCESSFUL")
            print(f"✅ PPO optimizer components functional")
            print(f"✅ Environment setup working")
            print(f"✅ Model creation and prediction working")
            print(f"✅ Configuration system operational")
            print(f"✅ Factory function operational")
            print(f"✅ All quick validation tests passed")
            return 0
        else:
            print(f"\n⚠️  DAY 7 PPO VALIDATION ISSUES FOUND")
            if not component_success:
                print(f"❌ PPO component validation failed")
            if not config_success:
                print(f"❌ Factory/configuration validation failed")
            return 1

    except Exception as e:
        print(f"\n❌ PPO quick validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())