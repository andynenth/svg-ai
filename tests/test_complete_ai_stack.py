#!/usr/bin/env python3
"""Complete AI Stack Verification"""

import sys
import time
import psutil
import torch
import sklearn
import stable_baselines3
import gymnasium
import deap
import cv2
import numpy as np

def test_complete_ai_stack():
    """Test all AI components together"""
    print("🔬 Testing complete AI stack...")

    # Memory usage before
    memory_before = psutil.virtual_memory().used / (1024**3)
    print(f"Memory before loading: {memory_before:.2f} GB")

    # Load all major components
    model = torch.nn.Linear(10, 1)
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor()
    from stable_baselines3 import PPO
    import gymnasium as gym
    env = gym.make('CartPole-v1')

    # Test DEAP
    from deap import base, creator, tools
    if hasattr(creator, 'FitnessMax'):
        del creator.FitnessMax
    if hasattr(creator, 'Individual'):
        del creator.Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Memory usage after
    memory_after = psutil.virtual_memory().used / (1024**3)
    print(f"Memory after loading: {memory_after:.2f} GB")
    print(f"Memory increase: {memory_after - memory_before:.2f} GB")

    print("✅ All AI components loaded successfully")

    # Test integration
    print("\n🔄 Testing AI component integration...")

    # Test PyTorch + scikit-learn
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    rf.fit(X, y)
    predictions = rf.predict(X[:10])
    torch_tensor = torch.from_numpy(predictions.astype(np.float32))
    print(f"✅ sklearn→PyTorch: {torch_tensor.shape}")

    # Test RL environment
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ RL environment: obs {obs.shape}, reward {reward}")

    # Test GA components
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, np.random.randn, 5)
    individual = toolbox.individual()
    print(f"✅ GA individual: {len(individual)} parameters")

    # Clean up
    env.close()

    return True

def main():
    """Run complete AI stack verification"""
    print("🧪 Complete AI Stack Integration Test")
    print("=" * 50)

    try:
        # Test package imports
        print("📦 Testing package imports...")
        packages = {
            'torch': torch,
            'sklearn': sklearn,
            'stable_baselines3': stable_baselines3,
            'gymnasium': gymnasium,
            'deap': deap,
            'cv2': cv2,
            'numpy': np
        }

        for name, package in packages.items():
            try:
                if hasattr(package, '__version__'):
                    version = package.__version__
                else:
                    version = "unknown"
                print(f"  {name}: ✅ {version}")
            except Exception as e:
                print(f"  {name}: ❌ {e}")

        # Test transformers separately (known to have issues)
        try:
            import transformers
            print(f"  transformers: ✅ {transformers.__version__}")
        except Exception as e:
            print(f"  transformers: ❌ {str(e)[:50]}... (expected)")

        print()

        # Test complete stack
        success = test_complete_ai_stack()

        print(f"\n📊 Final Status")
        print("=" * 30)

        if success:
            print("🎉 Core AI stack working!")
            print("✅ PyTorch, scikit-learn, RL, GA all functional")
            print("⚠️  Transformers optional (dependency issues)")
            return True
        else:
            print("❌ AI stack has critical issues")
            return False

    except Exception as e:
        print(f"❌ Critical error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)