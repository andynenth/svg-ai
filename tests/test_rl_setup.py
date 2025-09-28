#!/usr/bin/env python3
"""RL Setup Verification"""

import gymnasium as gym
from stable_baselines3 import PPO

def test_rl_setup():
    """Test basic RL functionality"""
    print("🤖 Testing RL setup...")

    # Create simple environment
    env = gym.make('CartPole-v1')
    print("✅ Gymnasium environment created")

    # Create PPO agent
    model = PPO('MlpPolicy', env, verbose=0)
    print("✅ PPO agent created")

    # Test environment reset
    obs, info = env.reset()
    print(f"✅ Environment reset: obs shape {obs.shape}")

    # Test basic training step (without actual training)
    print("🎯 Testing RL components...")

    # Test action sampling
    action = env.action_space.sample()
    print(f"✅ Action sampled: {action}")

    # Test environment step
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Environment step: reward={reward}, done={terminated or truncated}")

    # Test model prediction (without training)
    action, _states = model.predict(obs, deterministic=True)
    print(f"✅ Model prediction: action={action}")

    # Clean up
    env.close()
    print("✅ Environment closed successfully")

    return True

if __name__ == "__main__":
    try:
        success = test_rl_setup()
        if success:
            print("🎉 RL setup test completed successfully!")
        else:
            print("❌ RL setup test failed")
            exit(1)
    except Exception as e:
        print(f"❌ RL setup test error: {e}")
        exit(1)