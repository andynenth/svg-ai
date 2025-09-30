#!/usr/bin/env python3
"""
VTracer PPO Agent Training Script
Simple script to train and test PPO agent for VTracer parameter optimization
"""

import os
import sys
import argparse
import json
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from backend.ai_modules.optimization.agent_interface import VTracerAgentInterface, train_vtracer_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train VTracer PPO Agent")

    parser.add_argument('--image', type=str, required=True,
                       help='Path to training image')
    parser.add_argument('--model-dir', type=str, default='models/vtracer_ppo',
                       help='Directory to save trained model')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Number of training timesteps')
    parser.add_argument('--test-images', type=str, nargs='*',
                       help='Test images for evaluation')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced timesteps')
    parser.add_argument('--config', type=str,
                       help='Configuration file path')

    args = parser.parse_args()

    # Validate training image
    if not os.path.exists(args.image):
        logger.error(f"Training image not found: {args.image}")
        return 1

    # Adjust timesteps for quick test
    if args.quick_test:
        args.timesteps = min(args.timesteps, 5000)
        logger.info("Quick test mode: reducing timesteps to 5000")

    try:
        logger.info("🚀 Starting VTracer PPO Agent Training")
        logger.info(f"Training image: {args.image}")
        logger.info(f"Model directory: {args.model_dir}")
        logger.info(f"Training timesteps: {args.timesteps}")

        # Initialize agent interface
        agent = VTracerAgentInterface(
            model_save_dir=args.model_dir,
            config_file=args.config
        )

        # Train the agent
        logger.info("Starting training...")
        training_results = agent.train_agent(
            training_image=args.image,
            training_timesteps=args.timesteps
        )

        # Print training results
        logger.info("✅ Training completed successfully!")
        logger.info(f"Best quality achieved: {training_results.get('best_quality', 0.0):.4f}")
        logger.info(f"Training time: {training_results.get('training_time', 0.0):.2f} seconds")
        logger.info(f"Total timesteps: {training_results.get('total_timesteps', 0)}")

        # Save training report
        report_path = Path(args.model_dir) / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        logger.info(f"Training report saved: {report_path}")

        # Test optimization on the training image
        logger.info("Testing optimization on training image...")
        optimization_result = agent.optimize_image(args.image, max_episodes=5)
        logger.info(f"Optimization test - Best quality: {optimization_result['best_quality']:.4f}")

        # Test on additional images if provided
        if args.test_images:
            logger.info(f"Testing on {len(args.test_images)} additional images...")

            # Filter existing test images
            valid_test_images = [img for img in args.test_images if os.path.exists(img)]
            if not valid_test_images:
                logger.warning("No valid test images found")
            else:
                evaluation_results = agent.evaluate_performance(valid_test_images, episodes_per_image=3)

                logger.info("📊 Evaluation Results:")
                logger.info(f"Average quality: {evaluation_results['average_quality']:.4f}")
                logger.info(f"Target reached rate: {evaluation_results['target_reached_rate']:.2%}")
                logger.info(f"Quality range: {evaluation_results['min_quality']:.4f} - {evaluation_results['max_quality']:.4f}")

                # Save evaluation report
                eval_report_path = Path(args.model_dir) / "evaluation_report.json"
                with open(eval_report_path, 'w') as f:
                    json.dump(evaluation_results, f, indent=2)
                logger.info(f"Evaluation report saved: {eval_report_path}")

        # Clean up
        agent.close()

        logger.info("🎉 VTracer agent training and testing completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def quick_demo():
    """Quick demonstration with minimal training"""
    logger.info("🎮 Running Quick VTracer Agent Demo")

    # Find a test image
    test_images = [
        "tests/fixtures/images/simple_geometric/red_circle.png",
        "data/logos/simple_geometric/circle_00.png"
    ]

    training_image = None
    for img in test_images:
        if os.path.exists(img):
            training_image = img
            break

    if not training_image:
        logger.error("No test image found for demo")
        return 1

    try:
        # Quick training
        agent = train_vtracer_agent(
            training_image=training_image,
            save_dir="models/demo_ppo",
            timesteps=1000  # Very quick training for demo
        )

        # Quick optimization test
        result = agent.optimize_image(training_image, max_episodes=3)

        logger.info(f"Demo completed! Best quality: {result['best_quality']:.4f}")
        agent.close()

        return 0

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments, run quick demo
        exit(quick_demo())
    else:
        exit(main())