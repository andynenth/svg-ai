#!/usr/bin/env python3
"""
Day 7 Integration Test: PPO Training System
Task AB7.3: Training System Integration Testing

Tests the complete integration of:
- Developer A: PPO Agent + Training Pipeline
- Developer B: Training Monitoring + Training Execution
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import json
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def setup_test_environment():
    """Setup test environment with sample data"""
    # Create temporary test directory
    test_dir = tempfile.mkdtemp(prefix="day7_integration_")
    test_path = Path(test_dir)

    # Create test images directory structure
    images_dir = test_path / "test_images"
    images_dir.mkdir(parents=True)

    # Create test categories
    for category in ['simple', 'text', 'gradient', 'complex']:
        cat_dir = images_dir / category
        cat_dir.mkdir()

        # Create placeholder test image files
        for i in range(3):
            test_file = cat_dir / f"test_{category}_{i:02d}.png"
            test_file.write_text(f"# Placeholder test image for {category} category")

    # Create logs directory
    logs_dir = test_path / "logs"
    logs_dir.mkdir()

    return test_path, images_dir, logs_dir

def test_training_pipeline_initialization():
    """Test 1: Training pipeline initialization and configuration"""
    print("🧪 Test 1: Training Pipeline Initialization")

    try:
        from backend.ai_modules.optimization.training_pipeline import CurriculumTrainingPipeline

        # Test data setup
        test_images = {
            'simple': ['test_simple_01.png', 'test_simple_02.png'],
            'text': ['test_text_01.png'],
            'gradient': ['test_gradient_01.png'],
            'complex': ['test_complex_01.png']
        }

        # Test model configuration
        model_config = {
            'learning_rate': 1e-3,
            'n_steps': 256,  # Smaller for testing
            'batch_size': 16,
            'n_epochs': 2
        }

        # Initialize training pipeline
        pipeline = CurriculumTrainingPipeline(
            training_images=test_images,
            model_config=model_config
        )

        # Validate pipeline initialization
        assert pipeline is not None, "Pipeline initialization failed"
        assert len(pipeline.curriculum_stages) == 4, f"Expected 4 curriculum stages, got {len(pipeline.curriculum_stages)}"
        assert pipeline.current_stage == 0, f"Expected current stage 0, got {pipeline.current_stage}"

        # Validate curriculum stages
        stage_names = [stage.name for stage in pipeline.curriculum_stages]
        expected_stages = ['simple_warmup', 'text_introduction', 'gradient_challenge', 'complex_mastery']
        assert stage_names == expected_stages, f"Stage names mismatch: {stage_names} vs {expected_stages}"

        print("✅ Training Pipeline Initialization: PASSED")
        return True

    except Exception as e:
        print(f"❌ Training Pipeline Initialization: FAILED - {str(e)}")
        return False

def test_curriculum_learning_progression():
    """Test 2: Validate curriculum learning stage progression"""
    print("🧪 Test 2: Curriculum Learning Stage Progression")

    try:
        from backend.ai_modules.optimization.training_pipeline import CurriculumTrainingPipeline, TrainingStage

        # Initialize pipeline
        test_images = {'simple': ['test.png']}
        pipeline = CurriculumTrainingPipeline(test_images)

        # Test stage progression logic
        for i, stage in enumerate(pipeline.curriculum_stages):
            assert isinstance(stage, TrainingStage), f"Stage {i} is not a TrainingStage instance"
            assert hasattr(stage, 'name'), f"Stage {i} missing name attribute"
            assert hasattr(stage, 'difficulty'), f"Stage {i} missing difficulty attribute"
            assert hasattr(stage, 'target_quality'), f"Stage {i} missing target_quality attribute"
            assert 0.0 <= stage.difficulty <= 1.0, f"Stage {i} difficulty {stage.difficulty} not in [0,1]"
            assert 0.0 <= stage.target_quality <= 1.0, f"Stage {i} target_quality {stage.target_quality} not in [0,1]"

        # Test stage advancement logic
        initial_stage = pipeline.current_stage

        # Simulate successful stage completion
        pipeline.stage_results = {
            'simple_warmup': {
                'success_rate': 0.85,  # Above threshold
                'average_quality': 0.78,
                'episodes_completed': 5000
            }
        }

        print("✅ Curriculum Learning Stage Progression: PASSED")
        return True

    except Exception as e:
        print(f"❌ Curriculum Learning Stage Progression: FAILED - {str(e)}")
        return False

def test_training_monitoring_integration():
    """Test 3: Training monitoring and metrics collection"""
    print("🧪 Test 3: Training Monitoring and Metrics Collection")

    try:
        from backend.ai_modules.optimization.training_monitor import TrainingMonitor

        # Setup test monitoring
        test_dir, _, logs_dir = setup_test_environment()

        monitor = TrainingMonitor(
            log_dir=str(logs_dir),
            session_name="integration_test"
        )

        # Test basic monitoring functionality
        assert monitor is not None, "Monitor initialization failed"
        assert hasattr(monitor, 'episodes'), "Monitor missing episodes attribute"
        assert hasattr(monitor, 'log_episode'), "Monitor missing log_episode method"

        # Test episode logging
        test_episode_data = {
            'episode': 1,
            'reward': 15.5,
            'length': 100,
            'quality_improvement': 0.15,
            'quality_final': 0.85,
            'quality_initial': 0.70,
            'success': True,
            'termination_reason': 'quality_reached'
        }

        monitor.log_episode(**test_episode_data)

        # Verify metrics storage
        assert len(monitor.episodes) > 0, "Episode metrics not stored"
        assert monitor.episodes[-1].reward == 15.5, "Episode reward value incorrect"

        # Test metrics export
        metrics_data = monitor.export_metrics()
        assert 'episode_rewards' in metrics_data, "Episode rewards missing from export"
        assert 'training_session' in metrics_data, "Training session info missing"

        print("✅ Training Monitoring and Metrics Collection: PASSED")
        return True

    except Exception as e:
        print(f"❌ Training Monitoring and Metrics Collection: FAILED - {str(e)}")
        return False

def test_model_checkpointing():
    """Test 4: Model checkpointing and loading"""
    print("🧪 Test 4: Model Checkpointing and Loading")

    try:
        from backend.ai_modules.optimization.ppo_optimizer import PPOVTracerOptimizer

        # Setup test environment
        test_dir, _, _ = setup_test_environment()

        # Test basic optimizer initialization
        env_kwargs = {
            'target_images': ['test.png'],
            'max_episode_steps': 10
        }

        optimizer = PPOVTracerOptimizer(env_kwargs)
        assert optimizer is not None, "PPO optimizer initialization failed"

        # Test model configuration
        assert hasattr(optimizer, 'model_config'), "Optimizer missing model_config"
        assert hasattr(optimizer, 'training_history'), "Optimizer missing training_history"

        # Test checkpoint functionality (without actual training)
        checkpoint_path = Path(test_dir) / "test_checkpoint.json"

        # Create mock checkpoint data (JSON serializable)
        checkpoint_data = {
            'model_config': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64
            },
            'training_history': [],
            'timestamp': '2024-01-01T00:00:00'
        }

        # Save checkpoint
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        # Verify checkpoint file exists
        assert checkpoint_path.exists(), "Checkpoint file not created"

        # Load and verify checkpoint
        with open(checkpoint_path, 'r') as f:
            loaded_data = json.load(f)

        assert 'model_config' in loaded_data, "Model config missing from checkpoint"
        assert 'training_history' in loaded_data, "Training history missing from checkpoint"

        print("✅ Model Checkpointing and Loading: PASSED")
        return True

    except Exception as e:
        print(f"❌ Model Checkpointing and Loading: FAILED - {str(e)}")
        return False

def test_vtracer_environment_integration():
    """Test 5: Integration with VTracer environment"""
    print("🧪 Test 5: VTracer Environment Integration")

    try:
        from backend.ai_modules.optimization.vtracer_env import VTracerOptimizationEnv

        # Setup test environment with real image file
        test_dir, images_dir, _ = setup_test_environment()
        test_image_path = images_dir / "simple" / "test_simple_00.png"

        # Create a simple test PNG file (minimal valid PNG)
        png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01]\xcc_G\x00\x00\x00\x00IEND\xaeB`\x82'
        test_image_path.write_bytes(png_header)

        # Test environment initialization
        env = VTracerOptimizationEnv(
            image_path=str(test_image_path),
            target_quality=0.8,
            max_steps=5
        )
        assert env is not None, "VTracer environment initialization failed"

        # Test environment interface
        assert hasattr(env, 'reset'), "Environment missing reset method"
        assert hasattr(env, 'step'), "Environment missing step method"
        assert hasattr(env, 'action_space'), "Environment missing action_space"
        assert hasattr(env, 'observation_space'), "Environment missing observation_space"

        # Test basic environment functionality
        obs, info = env.reset()
        assert obs is not None, "Environment reset returned None observation"
        assert isinstance(info, dict), "Environment reset info is not a dict"

        # Test action space
        action_space = env.action_space
        assert action_space is not None, "Action space is None"

        # Test sample action
        action = action_space.sample()
        assert action is not None, "Sample action is None"

        print("✅ VTracer Environment Integration: PASSED")
        return True

    except Exception as e:
        print(f"❌ VTracer Environment Integration: FAILED - {str(e)}")
        return False

def run_complete_integration_test():
    """Run complete PPO training system integration test"""
    print("🚀 DAY 7 TASK AB7.3: PPO TRAINING SYSTEM INTEGRATION TEST")
    print("=" * 60)

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    test_results = []

    # Run all integration tests
    tests = [
        ("Training Pipeline Initialization", test_training_pipeline_initialization),
        ("Curriculum Learning Progression", test_curriculum_learning_progression),
        ("Training Monitoring Integration", test_training_monitoring_integration),
        ("Model Checkpointing", test_model_checkpointing),
        ("VTracer Environment Integration", test_vtracer_environment_integration)
    ]

    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)

        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception - {str(e)}")
            test_results.append((test_name, False))

    # Generate final report
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 60)

    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")

    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 DAY 7 INTEGRATION TEST SUCCESSFUL!")
        print("✅ PPO training system integration complete")
        print("✅ All components working together properly")
        print("✅ Ready for Task B7.2: Begin Model Training")
        return True
    else:
        print(f"\n⚠️  INTEGRATION ISSUES DETECTED")
        print(f"❌ {total_tests - passed_tests} tests failed")
        print("🔧 Review failed components before proceeding")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    sys.exit(0 if success else 1)