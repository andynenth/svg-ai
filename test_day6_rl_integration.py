#!/usr/bin/env python3
"""
Day 6 RL Environment Integration Test
Test complete RL environment with real VTracer integration as specified in DAY6_RL_ENVIRONMENT_SETUP.md
"""

import sys
import time
from pathlib import Path
import tempfile
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_complete_rl_environment():
    """Test complete RL environment with dummy agent - Day 6 Integration Test"""
    print("🤖 Testing Complete RL Environment Integration")
    print("=" * 60)

    try:
        # Import real components
        from backend.ai_modules.optimization.vtracer_env import VTracerOptimizationEnv

        # Find a test image
        test_image_paths = [
            "tests/fixtures/images/simple_geometric/red_circle.png",
            "tests/data/simple/simple_logo_0.png",
            "data/logos/simple_geometric/circle_00.png"
        ]

        test_image = None
        for image_path in test_image_paths:
            if os.path.exists(image_path):
                test_image = image_path
                break

        if test_image is None:
            # Create a simple test image
            import numpy as np
            import cv2
            print("Creating temporary test image...")

            # Create a simple red circle
            img = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.circle(img, (64, 64), 30, (0, 0, 255), -1)

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                test_image = tmp_file.name
                cv2.imwrite(test_image, img)

        print(f"Using test image: {test_image}")

        # Create environment
        print("Creating VTracer RL environment...")
        env = VTracerOptimizationEnv(test_image, target_quality=0.85, max_steps=10)
        print(f"✅ Environment created successfully")

        # Verify environment interface
        print("Verifying environment interface...")
        assert hasattr(env, 'action_space'), "Missing action_space"
        assert hasattr(env, 'observation_space'), "Missing observation_space"
        assert hasattr(env, 'reset'), "Missing reset method"
        assert hasattr(env, 'step'), "Missing step method"
        print(f"✅ Environment interface verified")

        # Test action and observation spaces
        print("Testing action and observation spaces...")
        assert env.action_space.shape == (7,), f"Expected action space (7,), got {env.action_space.shape}"
        assert env.observation_space.shape == (15,), f"Expected observation space (15,), got {env.observation_space.shape}"
        print(f"✅ Spaces verified: Action {env.action_space.shape}, Observation {env.observation_space.shape}")

        # Test episode execution
        print("Testing episode execution...")
        obs, info = env.reset()
        print(f"Reset completed - observation shape: {obs.shape}")

        total_reward = 0
        episode_length = 0
        episode_history = []

        print("Running test episode...")
        for step in range(10):  # Short test episode
            # Random action
            action = env.action_space.sample()
            print(f"  Step {step + 1}: Action sampled")

            # Take step
            start_time = time.time()
            obs, reward, done, truncated, info = env.step(action)
            step_time = time.time() - start_time

            total_reward += reward
            episode_length += 1

            step_info = {
                'step': step + 1,
                'reward': reward,
                'quality': info.get('quality', 0.0),
                'best_quality': info.get('best_quality', 0.0),
                'processing_time': info.get('processing_time', 0.0),
                'step_time': step_time,
                'success': info.get('success', False),
                'done': done,
                'truncated': truncated
            }
            episode_history.append(step_info)

            print(f"    Reward: {reward:.3f}, Quality: {info.get('quality', 0.0):.3f}, "
                  f"Processing time: {step_time:.3f}s")

            if done or truncated:
                print(f"    Episode terminated: done={done}, truncated={truncated}")
                break

        # Validate episode results
        assert episode_length > 0, "Episode should have at least one step"
        assert len(obs) == env.observation_space.shape[0], f"Observation length mismatch: {len(obs)} != {env.observation_space.shape[0]}"
        assert isinstance(total_reward, (int, float)), f"Total reward should be numeric, got {type(total_reward)}"

        print(f"✅ Episode execution successful")

        # Test environment state consistency
        print("Testing environment state consistency...")
        summary = env.get_episode_summary()
        assert summary['steps'] == episode_length, "Episode step count mismatch"
        assert 'baseline_quality' in summary, "Missing baseline quality"
        assert 'best_quality' in summary, "Missing best quality"
        print(f"✅ Environment state consistent")

        # Test reward calculation accuracy
        print("Testing reward calculation...")
        reward_stats = env.reward_function.get_reward_statistics()
        if 'error' not in reward_stats:
            print(f"  Reward statistics: {reward_stats['total_entries']} entries")
            print(f"  Average reward: {reward_stats['reward_mean']:.3f}")
            print(f"✅ Reward function operational")
        else:
            print(f"⚠️  Reward statistics not available: {reward_stats['error']}")

        # Test action-parameter mapping accuracy
        print("Testing action-parameter mapping...")
        test_action = env.action_space.sample()
        mapped_params = env._denormalize_action(test_action)

        expected_params = ['color_precision', 'layer_difference', 'corner_threshold',
                          'length_threshold', 'max_iterations', 'splice_threshold',
                          'path_precision', 'mode']

        for param in expected_params:
            assert param in mapped_params, f"Missing parameter: {param}"

        # Test reverse mapping
        normalized_action = env._normalize_parameters(mapped_params)
        assert normalized_action.shape == (7,), f"Normalized action shape mismatch: {normalized_action.shape}"
        print(f"✅ Action-parameter mapping verified")

        # Environment cleanup
        env.close()
        print(f"✅ Environment closed successfully")

        # Clean up temporary files
        if test_image and test_image.startswith('/tmp'):
            try:
                os.unlink(test_image)
                print(f"✅ Temporary test image cleaned up")
            except:
                pass

        # Final validation
        success_criteria = {
            "environment_creation": True,
            "episode_execution": episode_length > 0,
            "observation_validity": len(obs) == env.observation_space.shape[0],
            "reward_calculation": isinstance(total_reward, (int, float)),
            "state_consistency": summary['steps'] == episode_length,
            "parameter_mapping": len(mapped_params) >= 7
        }

        all_passed = all(success_criteria.values())

        print(f"\n📋 INTEGRATION TEST RESULTS")
        print("=" * 60)
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  - {criterion.replace('_', ' ').title()}: {status}")

        print(f"\nEpisode Summary:")
        print(f"  - Episode length: {episode_length}")
        print(f"  - Total reward: {total_reward:.3f}")
        print(f"  - Best quality: {summary.get('best_quality', 0.0):.3f}")
        print(f"  - Target reached: {summary.get('target_reached', False)}")

        if all_passed:
            print(f"\n✅ RL ENVIRONMENT VALIDATION SUCCESSFUL")
            return True
        else:
            print(f"\n❌ RL ENVIRONMENT VALIDATION FAILED")
            return False

    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reward_function_integration():
    """Test reward function integration separately"""
    print("\n🎯 Testing Reward Function Integration")
    print("=" * 60)

    try:
        from backend.ai_modules.optimization.reward_functions import MultiObjectiveRewardFunction, ConversionResult

        # Create reward function
        reward_fn = MultiObjectiveRewardFunction(target_quality=0.85)
        print("✅ Reward function created")

        # Create test conversion results
        baseline_result = ConversionResult(
            quality_score=0.75,
            processing_time=0.1,
            file_size=10.0,
            success=True,
            svg_path=""
        )

        current_result = ConversionResult(
            quality_score=0.85,
            processing_time=0.08,
            file_size=8.0,
            success=True,
            svg_path=""
        )

        # Calculate reward
        reward, components = reward_fn.calculate_reward(current_result, baseline_result, 5, 50)

        print(f"Reward calculation result: {reward:.3f}")
        print(f"Components: {components}")

        # Test reward function configuration
        reward_fn.configure(quality_weight=0.8, speed_weight=0.15, size_weight=0.05)
        print("✅ Reward function configuration successful")

        # Test reward statistics
        stats = reward_fn.get_reward_statistics()
        print(f"Reward statistics available: {'error' not in stats}")

        return True

    except Exception as e:
        print(f"❌ Reward function test failed: {e}")
        return False

def main():
    """Run Day 6 RL integration tests"""
    print("🚀 Starting Day 6 RL Environment Integration Tests")
    print("Testing complete RL environment functionality")
    print("According to DAY6_RL_ENVIRONMENT_SETUP.md specification\n")

    try:
        # Test 1: Complete RL Environment
        env_success = test_complete_rl_environment()

        # Test 2: Reward Function Integration
        reward_success = test_reward_function_integration()

        # Overall assessment
        overall_success = env_success and reward_success

        if overall_success:
            print(f"\n🎉 DAY 6 RL INTEGRATION TESTS SUCCESSFUL")
            print(f"✅ VTracer RL environment fully functional")
            print(f"✅ Multi-objective reward function operational")
            print(f"✅ Action-parameter mapping working correctly")
            print(f"✅ All integration tests passed")
            return 0
        else:
            print(f"\n⚠️  DAY 6 INTEGRATION ISSUES FOUND")
            if not env_success:
                print(f"❌ RL environment integration failed")
            if not reward_success:
                print(f"❌ Reward function integration failed")
            return 1

    except Exception as e:
        print(f"\n❌ Integration test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())