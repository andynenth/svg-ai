#!/usr/bin/env python3
"""AI Environment Verification Script"""

import sys
import time
import importlib
import traceback

def test_imports():
    """Test all required AI package imports"""
    print("🔍 Testing AI package imports...")

    required_packages = [
        'torch', 'torchvision', 'sklearn', 'stable_baselines3',
        'gymnasium', 'deap', 'cv2', 'numpy', 'PIL', 'transformers'
    ]

    results = {}
    for package in required_packages:
        try:
            importlib.import_module(package)
            results[package] = "✅ SUCCESS"
            print(f"  {package}: ✅")
        except ImportError as e:
            results[package] = f"❌ FAILED: {e}"
            print(f"  {package}: ❌ {e}")

    return results

def test_pytorch_performance():
    """Test PyTorch CPU performance"""
    print("\n🚀 Testing PyTorch performance...")

    try:
        import torch

        # Test basic tensor operations
        start = time.time()
        x = torch.randn(1000, 1000)
        y = torch.mm(x, x.t())
        pytorch_time = time.time() - start

        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  Matrix multiplication (1000x1000): {pytorch_time:.3f}s")

        if pytorch_time < 1.0:
            print(f"  ✅ Performance good ({pytorch_time:.3f}s < 1.0s)")
            return True
        else:
            print(f"  ⚠️ Performance slow ({pytorch_time:.3f}s >= 1.0s)")
            return False

    except Exception as e:
        print(f"  ❌ PyTorch test failed: {e}")
        return False

def test_rl_components():
    """Test reinforcement learning components"""
    print("\n🤖 Testing RL components...")

    try:
        import gymnasium as gym
        from stable_baselines3 import PPO

        # Create simple environment
        env = gym.make('CartPole-v1')
        print(f"  Gymnasium version: {gym.__version__}")
        print("  ✅ Environment creation successful")

        # Test PPO agent creation
        model = PPO('MlpPolicy', env, verbose=0)
        print("  ✅ PPO agent creation successful")

        # Test environment reset
        obs, info = env.reset()
        print(f"  ✅ Environment reset successful: obs shape {obs.shape}")

        env.close()
        return True

    except Exception as e:
        print(f"  ❌ RL components test failed: {e}")
        return False

def test_genetic_algorithms():
    """Test genetic algorithm components"""
    print("\n🧬 Testing genetic algorithm components...")

    try:
        from deap import base, creator, tools, algorithms

        print(f"  DEAP version: {getattr(deap, '__version__', 'Unknown')}")
        print("  ✅ DEAP modules available")

        # Test basic GA setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        print("  ✅ Basic GA setup successful")

        return True

    except Exception as e:
        print(f"  ❌ Genetic algorithm test failed: {e}")
        return False

def test_transformers():
    """Test transformers components"""
    print("\n🔤 Testing transformers components...")

    try:
        import transformers
        import tokenizers

        print(f"  Transformers version: {transformers.__version__}")
        print(f"  Tokenizers version: {tokenizers.__version__}")
        print("  ✅ Transformers components available")

        return True

    except Exception as e:
        print(f"  ❌ Transformers test failed: {e}")
        return False

def check_memory_usage():
    """Check memory usage"""
    print("\n💾 Checking memory usage...")

    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)

        print(f"  Total memory: {memory_gb:.1f} GB")
        print(f"  Available memory: {memory_available_gb:.1f} GB")
        print(f"  Memory usage: {memory.percent:.1f}%")

        if memory_gb >= 4.0:
            print("  ✅ Memory sufficient for AI processing")
            return True
        else:
            print("  ⚠️ Memory may be insufficient (<4GB)")
            return False

    except ImportError:
        print("  ⚠️ psutil not available, skipping memory check")
        return True
    except Exception as e:
        print(f"  ❌ Memory check failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🤖 AI Environment Verification")
    print("=" * 40)

    # Test imports
    import_results = test_imports()

    # Count successful imports
    successful_imports = sum(1 for result in import_results.values() if "SUCCESS" in result)
    total_imports = len(import_results)

    # Performance tests
    pytorch_ok = test_pytorch_performance()
    rl_ok = test_rl_components()
    ga_ok = test_genetic_algorithms()
    transformers_ok = test_transformers()
    memory_ok = check_memory_usage()

    # Summary
    print("\n📊 Verification Summary")
    print("=" * 40)
    print(f"Package imports: {successful_imports}/{total_imports} successful")
    print(f"PyTorch performance: {'✅ Good' if pytorch_ok else '❌ Poor'}")
    print(f"RL components: {'✅ Working' if rl_ok else '❌ Failed'}")
    print(f"Genetic algorithms: {'✅ Working' if ga_ok else '❌ Failed'}")
    print(f"Transformers: {'✅ Working' if transformers_ok else '❌ Failed'}")
    print(f"Memory: {'✅ Sufficient' if memory_ok else '❌ Insufficient'}")

    # Overall result
    all_critical_working = (
        successful_imports >= 8 and  # Most packages working
        pytorch_ok and               # PyTorch essential
        rl_ok and                   # RL components essential
        ga_ok                       # GA components essential
    )

    print("\n🎯 Overall Result")
    print("=" * 40)
    if all_critical_working:
        print("🎉 AI environment setup SUCCESSFUL!")
        print("Ready to begin AI pipeline development.")
        return 0
    else:
        print("❌ AI environment setup INCOMPLETE!")
        print("Please resolve issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())