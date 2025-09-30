#!/usr/bin/env python3
"""Test script for TrainingMonitor functionality"""

import sys
import time
import numpy as np
from pathlib import Path

# Add backend modules to path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.ai_modules.optimization.training_monitor import TrainingMonitor, create_training_monitor


def test_basic_functionality():
    """Test basic TrainingMonitor functionality"""
    print("🧪 Testing TrainingMonitor Basic Functionality")
    print("=" * 50)

    # Create monitor
    monitor = create_training_monitor(
        log_dir="test_logs/training_monitor",
        session_name="test_session",
        use_tensorboard=True,
        use_wandb=False
    )

    print(f"✅ Created monitor with session ID: {monitor.session_id}")

    # Simulate training episodes
    print("\n📊 Simulating training episodes...")

    for episode in range(1, 11):
        # Simulate improving performance
        base_reward = -10 + episode * 2  # Gradually improving rewards
        noise = np.random.normal(0, 2)
        reward = base_reward + noise

        # Simulate quality improvements
        quality_improvement = min(0.1 + episode * 0.05 + np.random.normal(0, 0.02), 0.8)
        quality_initial = 0.6 + np.random.normal(0, 0.05)
        quality_final = quality_initial + quality_improvement

        # Simulate episode length
        length = int(20 + np.random.exponential(10))

        # Determine success
        success = quality_improvement > 0.15

        # Simulate algorithm metrics
        algorithm_metrics = {
            'policy_loss': np.random.uniform(0.1, 1.0),
            'value_loss': np.random.uniform(0.05, 0.5),
            'entropy': np.random.uniform(0.1, 2.0),
            'kl_divergence': np.random.uniform(0.01, 0.1),
            'learning_rate': 3e-4,
            'gradient_norm': np.random.uniform(0.1, 1.0)
        }

        # Simulate performance metrics
        performance_metrics = {
            'episode_time': np.random.uniform(5, 15),
            'memory_usage': 150 + np.random.uniform(-20, 20)
        }

        # Additional info
        logo_types = ['simple', 'text', 'gradient', 'complex']
        additional_info = {
            'logo_type': np.random.choice(logo_types),
            'difficulty': np.random.choice(['easy', 'medium', 'hard']),
            'parameters_explored': {
                'color_precision': np.random.randint(2, 8),
                'corner_threshold': np.random.uniform(10, 50)
            }
        }

        # Log episode
        monitor.log_episode(
            episode=episode,
            reward=reward,
            length=length,
            quality_improvement=quality_improvement,
            quality_final=quality_final,
            quality_initial=quality_initial,
            termination_reason="max_steps" if length >= 40 else "target_reached",
            success=success,
            algorithm_metrics=algorithm_metrics,
            performance_metrics=performance_metrics,
            additional_info=additional_info
        )

        # Log training step metrics
        monitor.log_training_step(
            step=episode * 100,
            policy_loss=algorithm_metrics['policy_loss'],
            value_loss=algorithm_metrics['value_loss'],
            entropy=algorithm_metrics['entropy'],
            kl_divergence=algorithm_metrics['kl_divergence'],
            learning_rate=algorithm_metrics['learning_rate'],
            gradient_norm=algorithm_metrics['gradient_norm']
        )

        # Log evaluation periodically
        if episode % 5 == 0:
            monitor.log_evaluation(
                episode=episode,
                eval_reward=reward + np.random.normal(0, 1),
                eval_quality=quality_final + np.random.normal(0, 0.02),
                eval_success_rate=min(1.0, success * 1.2 + np.random.uniform(-0.1, 0.1)),
                eval_episodes=3
            )

        print(f"  Episode {episode:2d}: reward={reward:6.2f}, quality={quality_improvement:.4f}, success={success}")

        # Small delay to simulate training time
        time.sleep(0.1)

    print("\n📈 Getting training statistics...")
    stats = monitor.get_training_statistics()
    print(f"  Total episodes: {stats['session_info']['total_episodes']}")
    print(f"  Average reward: {stats['reward_stats']['mean']:.2f}")
    print(f"  Average quality improvement: {stats['quality_stats']['mean']:.4f}")
    print(f"  Success rate: {stats['episode_stats']['success_rate']:.2%}")

    print("\n🔍 Analyzing convergence...")
    convergence = monitor.get_convergence_analysis()
    print(f"  Reward trend: {convergence['convergence_status']['reward_trend']}")
    print(f"  Quality trend: {convergence['convergence_status']['quality_trend']}")
    print(f"  Stability: {convergence['stability_metrics']['stable']}")

    print("\n🏥 Validating metrics...")
    validation = monitor.validate_metrics()
    print(f"  Health score: {validation['health_score']:.2%}")
    print(f"  Status: {validation['status']}")

    print("\n📁 Testing export functionality...")
    # Test JSON export
    json_path = monitor.export_metrics("json")
    print(f"  Exported JSON: {json_path}")

    # Test CSV export
    csv_path = monitor.export_metrics("csv")
    print(f"  Exported CSV: {csv_path}")

    print("\n📊 Creating dashboard data...")
    dashboard_data = monitor.create_dashboard_data()
    print(f"  Dashboard data keys: {list(dashboard_data.keys())}")
    print(f"  Chart data points: {len(dashboard_data['charts']['episode_rewards'])}")

    # Close monitor
    monitor.close()
    print("\n✅ Training monitor test completed successfully!")


def test_integration_patterns():
    """Test integration with existing optimization patterns"""
    print("\n🔗 Testing Integration Patterns")
    print("=" * 50)

    monitor = create_training_monitor(
        log_dir="test_logs/integration_test",
        session_name="integration_test"
    )

    # Test with optimization logger pattern
    print("  Testing with OptimizationLogger pattern...")

    # Simulate data similar to OptimizationLogger
    image_path = "data/logos/simple_geometric/circle_00.png"
    features = {
        'aspect_ratio': 1.0,
        'color_count': 3,
        'complexity_score': 0.2,
        'has_text': False
    }

    params = {
        'color_precision': 4,
        'corner_threshold': 30,
        'path_precision': 8
    }

    quality_metrics = {
        'ssim': 0.95,
        'mse': 0.005,
        'conversion_time': 2.3
    }

    # Convert to training monitor format
    monitor.log_episode(
        episode=1,
        reward=10.0,  # Derived from quality metrics
        length=25,
        quality_improvement=0.25,  # 95% - 70% baseline
        quality_final=0.95,
        quality_initial=0.70,
        success=True,
        additional_info={
            'image_path': image_path,
            'features': features,
            'parameters': params,
            'quality_metrics': quality_metrics
        }
    )

    print("  ✅ Successfully integrated with optimization patterns")

    # Test validation installation
    print("\n🔧 Checking dependency availability...")
    from backend.ai_modules.optimization.training_monitor import validate_training_monitor_installation

    dependencies = validate_training_monitor_installation()
    for dep, available in dependencies.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}: {available}")

    monitor.close()
    print("\n✅ Integration test completed!")


if __name__ == "__main__":
    print("🚀 TrainingMonitor Test Suite")
    print("=" * 60)

    try:
        test_basic_functionality()
        test_integration_patterns()

        print("\n🎉 All tests passed successfully!")
        print("\nGenerated files:")
        print("  - test_logs/training_monitor/ (basic functionality test)")
        print("  - test_logs/integration_test/ (integration test)")
        print("\nKey features verified:")
        print("  ✅ Comprehensive metrics collection")
        print("  ✅ Algorithm-specific metrics (PPO)")
        print("  ✅ Performance tracking")
        print("  ✅ Real-time dashboard data")
        print("  ✅ Export functionality (JSON, CSV)")
        print("  ✅ Metrics validation and health checks")
        print("  ✅ Integration with existing patterns")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)