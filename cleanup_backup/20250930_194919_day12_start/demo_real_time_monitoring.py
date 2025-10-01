#!/usr/bin/env python3
"""
Demo script for real-time monitoring infrastructure
Demonstrates Task B7.1 Component 3 - Real-time Monitoring Integration

This script shows how to use the real-time monitoring system with PPO training.
"""

import asyncio
import sys
import time
import logging
from pathlib import Path

# Add the backend modules to Python path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_modules.optimization.real_time_monitor import create_real_time_monitor
from backend.ai_modules.optimization.ppo_optimizer import create_ppo_optimizer
from backend.ai_modules.optimization.training_pipeline import create_curriculum_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_real_time_monitor():
    """Demo the real-time monitoring system"""
    logger.info("🚀 Starting Real-time Monitoring Demo")

    # Create real-time monitor
    monitor = create_real_time_monitor(
        websocket_port=8765,
        update_interval=1.0,
        save_dir="logs/demo_monitoring"
    )

    try:
        # Start monitoring
        logger.info("Starting monitoring infrastructure...")
        await monitor.start_monitoring()

        logger.info("✅ Monitoring started!")
        logger.info("🌐 WebSocket server running on ws://localhost:8765")
        logger.info("📊 Open monitoring_dashboard.html in your browser to view live data")

        # Simulate some training episodes
        await simulate_training_episodes(monitor)

        # Keep monitoring running
        logger.info("📡 Monitoring will continue for 60 seconds...")
        logger.info("💡 You can connect to the dashboard to see live updates")
        await asyncio.sleep(60)

    finally:
        # Stop monitoring
        logger.info("Stopping monitoring...")
        await monitor.stop_monitoring()
        logger.info("✅ Demo completed")


async def simulate_training_episodes(monitor):
    """Simulate training episodes for demonstration"""
    import random
    import numpy as np

    logger.info("🎯 Simulating training episodes...")

    base_reward = 0.0
    base_quality = 0.7

    for episode in range(1, 21):
        # Simulate improving performance over time
        reward_trend = episode * 0.05
        quality_trend = episode * 0.01

        # Add some randomness
        reward = base_reward + reward_trend + random.gauss(0, 0.1)
        quality = min(0.99, base_quality + quality_trend + random.gauss(0, 0.02))

        # Create synthetic training metrics
        from backend.ai_modules.optimization.real_time_monitor import TrainingMetrics
        metrics = TrainingMetrics(
            timestamp=time.time(),
            episode=episode,
            reward=reward,
            episode_length=random.randint(10, 50),
            quality=quality,
            success=quality > 0.8,
            policy_loss=random.uniform(0.01, 0.1),
            value_loss=random.uniform(0.01, 0.1),
            entropy=random.uniform(0.1, 0.5),
            fps=random.uniform(50, 100),
            ssim_improvement=quality - 0.7,
            parameters_used={
                'color_precision': random.randint(2, 8),
                'corner_threshold': random.randint(20, 60)
            }
        )

        # Send to monitor
        monitor.on_episode_complete(metrics)

        # Wait a bit between episodes
        await asyncio.sleep(0.5)

    logger.info("✅ Simulation completed")


async def demo_ppo_integration():
    """Demo PPO integration with real-time monitoring"""
    logger.info("🤖 Starting PPO Integration Demo")

    # Note: This requires actual training images to work
    # For demo purposes, we'll show the setup
    try:
        # Example training images (these would need to exist)
        training_images = {
            'simple': ['data/logos/simple_geometric/circle_00.png'],
            'text': ['data/logos/text_based/text_00.png']
        }

        # Create curriculum pipeline with monitoring
        pipeline = create_curriculum_pipeline(
            training_images=training_images,
            save_dir="models/demo_curriculum"
        )

        logger.info("✅ Pipeline created with real-time monitoring")
        logger.info("📊 Monitoring WebSocket on port 8766")

        # Start pipeline monitoring
        await pipeline.start_monitoring()

        logger.info("🎯 Pipeline monitoring started")
        logger.info("💡 In real training, this would stream live data")

        # Simulate some time
        await asyncio.sleep(10)

        # Stop monitoring
        await pipeline.stop_monitoring()

        # Close pipeline
        pipeline.close()

        logger.info("✅ PPO integration demo completed")

    except Exception as e:
        logger.warning(f"PPO integration demo skipped: {e}")
        logger.info("💡 This demo requires actual training images to run fully")


def main():
    """Main demo function"""
    print("=" * 60)
    print("🚀 Real-time Monitoring Infrastructure Demo")
    print("   Task B7.1 Component 3 Implementation")
    print("=" * 60)
    print()

    print("This demo shows the real-time monitoring system for PPO training:")
    print("✅ Real-time metrics streaming")
    print("✅ WebSocket server for live dashboard")
    print("✅ Training monitoring callbacks")
    print("✅ System health monitoring")
    print("✅ Alerting system")
    print("✅ Live dashboard with charts")
    print()

    print("Demo components:")
    print("1. Real-time monitor with WebSocket server")
    print("2. Simulated training episodes")
    print("3. Live dashboard (monitoring_dashboard.html)")
    print("4. PPO integration example")
    print()

    choice = input("Which demo would you like to run?\n"
                  "1. Real-time monitor with simulated data\n"
                  "2. PPO integration demo\n"
                  "3. Both\n"
                  "Choice (1-3): ").strip()

    if choice == "1":
        asyncio.run(demo_real_time_monitor())
    elif choice == "2":
        asyncio.run(demo_ppo_integration())
    elif choice == "3":
        async def run_both():
            await demo_real_time_monitor()
            await asyncio.sleep(2)
            await demo_ppo_integration()
        asyncio.run(run_both())
    else:
        print("Invalid choice. Running default demo...")
        asyncio.run(demo_real_time_monitor())


if __name__ == "__main__":
    main()