#!/usr/bin/env python3
"""Simple test script for TrainingMonitor functionality"""

import sys
import numpy as np
from pathlib import Path

# Add backend modules to path
sys.path.append(str(Path(__file__).parent / "backend"))

def test_training_monitor():
    """Test basic TrainingMonitor functionality"""
    print("🧪 Testing TrainingMonitor Basic Functionality")

    try:
        from backend.ai_modules.optimization.training_monitor import create_training_monitor

        # Create monitor
        monitor = create_training_monitor(
            log_dir="test_logs/simple_test",
            session_name="simple_test",
            use_tensorboard=False,  # Disable to avoid issues
            use_wandb=False
        )

        print(f"✅ Created monitor with session ID: {monitor.session_id}")

        # Log a few test episodes
        for episode in range(1, 4):
            monitor.log_episode(
                episode=episode,
                reward=float(episode * 2),
                length=20 + episode,
                quality_improvement=0.1 + episode * 0.05,
                quality_final=0.7 + episode * 0.05,
                quality_initial=0.6,
                success=True,
                algorithm_metrics={
                    'policy_loss': 0.5,
                    'value_loss': 0.3,
                    'entropy': 1.2
                }
            )

        # Get statistics
        stats = monitor.get_training_statistics()
        print(f"✅ Statistics: {stats['session_info']['total_episodes']} episodes")

        # Export data
        json_path = monitor.export_metrics("json", include_raw_data=False)
        print(f"✅ Exported to: {json_path}")

        # Validate
        validation = monitor.validate_metrics()
        print(f"✅ Health score: {validation['health_score']:.2%}")

        monitor.close()
        print("✅ Test completed successfully!")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_training_monitor():
        print("\n🎉 TrainingMonitor implementation verified!")
    else:
        print("\n❌ TrainingMonitor test failed!")
        sys.exit(1)