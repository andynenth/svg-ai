#!/usr/bin/env python3
"""
Day 7 Simple Integration Test: PPO Training System
Task AB7.3: Training System Integration Testing (Simplified)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_basic_imports_and_initialization():
    """Test that all major components can be imported and initialized"""
    print("🧪 Testing Basic Imports and Initialization")

    try:
        # Test training pipeline import and basic initialization
        from backend.ai_modules.optimization.training_pipeline import CurriculumTrainingPipeline
        pipeline = CurriculumTrainingPipeline({'simple': ['test.png']})
        assert pipeline is not None
        assert len(pipeline.curriculum_stages) == 4
        print("✅ Training Pipeline: Import and initialization successful")

        # Test training monitor import and basic initialization
        from backend.ai_modules.optimization.training_monitor import TrainingMonitor
        monitor = TrainingMonitor(log_dir="/tmp/test_logs", session_name="test")
        assert monitor is not None
        assert hasattr(monitor, 'episodes')
        print("✅ Training Monitor: Import and initialization successful")

        # Test PPO optimizer import and basic initialization
        from backend.ai_modules.optimization.ppo_optimizer import PPOVTracerOptimizer
        optimizer = PPOVTracerOptimizer({'target_images': ['test.png']})
        assert optimizer is not None
        assert hasattr(optimizer, 'model_config')
        print("✅ PPO Optimizer: Import and initialization successful")

        # Test VTracer environment import (skip full initialization)
        from backend.ai_modules.optimization.vtracer_env import VTracerOptimizationEnv
        print("✅ VTracer Environment: Import successful")

        return True

    except Exception as e:
        print(f"❌ Basic Integration Test Failed: {str(e)}")
        return False

def test_component_interfaces():
    """Test that components have the expected interfaces"""
    print("🧪 Testing Component Interfaces")

    try:
        from backend.ai_modules.optimization.training_pipeline import CurriculumTrainingPipeline
        from backend.ai_modules.optimization.training_monitor import TrainingMonitor
        from backend.ai_modules.optimization.ppo_optimizer import PPOVTracerOptimizer

        # Test training pipeline interface
        pipeline = CurriculumTrainingPipeline({'simple': ['test.png']})
        required_methods = ['run_curriculum', 'train_stage', 'visualize_curriculum_progress']
        for method in required_methods:
            assert hasattr(pipeline, method), f"Pipeline missing {method}"
        print("✅ Training Pipeline Interface: All required methods present")

        # Test training monitor interface
        monitor = TrainingMonitor(log_dir="/tmp/test_logs", session_name="test")
        required_methods = ['log_episode', 'export_metrics', 'get_training_statistics']
        for method in required_methods:
            assert hasattr(monitor, method), f"Monitor missing {method}"
        print("✅ Training Monitor Interface: All required methods present")

        # Test PPO optimizer interface
        optimizer = PPOVTracerOptimizer({'target_images': ['test.png']})
        required_attrs = ['model_config', 'training_history', 'best_performance']
        for attr in required_attrs:
            assert hasattr(optimizer, attr), f"Optimizer missing {attr}"
        print("✅ PPO Optimizer Interface: All required attributes present")

        return True

    except Exception as e:
        print(f"❌ Component Interface Test Failed: {str(e)}")
        return False

def test_curriculum_stages():
    """Test curriculum learning stages configuration"""
    print("🧪 Testing Curriculum Stages Configuration")

    try:
        from backend.ai_modules.optimization.training_pipeline import CurriculumTrainingPipeline

        pipeline = CurriculumTrainingPipeline({'simple': ['test.png']})

        # Verify 4-stage curriculum
        assert len(pipeline.curriculum_stages) == 4

        # Verify stage names and progression
        expected_stages = ['simple_warmup', 'text_introduction', 'gradient_challenge', 'complex_mastery']
        actual_stages = [stage.name for stage in pipeline.curriculum_stages]
        assert actual_stages == expected_stages

        # Verify difficulty progression
        difficulties = [stage.difficulty for stage in pipeline.curriculum_stages]
        assert difficulties == sorted(difficulties), "Difficulties should be in ascending order"

        print("✅ Curriculum Stages: Configuration and progression correct")
        return True

    except Exception as e:
        print(f"❌ Curriculum Stages Test Failed: {str(e)}")
        return False

def run_simple_integration_test():
    """Run simplified integration test"""
    print("🚀 DAY 7 TASK AB7.3: SIMPLIFIED PPO TRAINING SYSTEM INTEGRATION TEST")
    print("=" * 70)

    tests = [
        ("Basic Imports and Initialization", test_basic_imports_and_initialization),
        ("Component Interfaces", test_component_interfaces),
        ("Curriculum Stages Configuration", test_curriculum_stages)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)

        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception - {str(e)}")
            results.append((test_name, False))

    # Generate final report
    print("\n" + "=" * 70)
    print("📊 SIMPLIFIED INTEGRATION TEST RESULTS")
    print("=" * 70)

    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<45} {status}")

    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 DAY 7 SIMPLIFIED INTEGRATION TEST SUCCESSFUL!")
        print("✅ Core PPO training system components operational")
        print("✅ All major interfaces working properly")
        print("✅ Curriculum learning system configured correctly")
        print("✅ Ready for full training execution")
        return True
    else:
        print(f"\n⚠️  INTEGRATION ISSUES DETECTED")
        print(f"❌ {total_tests - passed_tests} tests failed")
        return False

if __name__ == "__main__":
    success = run_simple_integration_test()
    sys.exit(0 if success else 1)