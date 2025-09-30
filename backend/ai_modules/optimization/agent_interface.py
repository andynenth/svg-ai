# backend/ai_modules/optimization/agent_interface.py
"""Agent-Environment Interface for VTracer Parameter Optimization"""

import logging
import time
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json

from .ppo_optimizer import PPOVTracerOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VTracerAgentInterface:
    """
    High-level interface for VTracer parameter optimization using RL agents
    Provides simple API for training and optimization
    """

    def __init__(self,
                 model_save_dir: str = "models/vtracer_ppo",
                 config_file: Optional[str] = None):
        """
        Initialize VTracer Agent Interface

        Args:
            model_save_dir: Directory to save trained models
            config_file: Optional configuration file path
        """
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config(config_file)

        # Initialize optimizer
        self.optimizer = None
        self.is_trained = False

        logger.info("VTracer Agent Interface initialized")

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "model": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "verbose": 1
            },
            "training": {
                "total_timesteps": 100000,
                "eval_freq": 10000,
                "n_eval_episodes": 5,
                "deterministic_eval": True,
                "n_envs": 4,
                "checkpoint_freq": 25000,
                "target_quality": 0.85
            },
            "environment": {
                "max_steps": 50,
                "target_quality": 0.85
            }
        }

        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                for section in default_config:
                    if section in loaded_config:
                        default_config[section].update(loaded_config[section])
                logger.info(f"Configuration loaded from: {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
                logger.info("Using default configuration")

        return default_config

    def train_agent(self, training_image: str,
                   training_timesteps: Optional[int] = None) -> Dict[str, Any]:
        """
        Train PPO agent on a specific image

        Args:
            training_image: Path to image for training
            training_timesteps: Override default training timesteps

        Returns:
            Training results dictionary
        """
        logger.info(f"Starting agent training on: {training_image}")

        # Prepare environment configuration
        env_kwargs = {
            'image_path': training_image,
            'target_quality': self.config['environment']['target_quality'],
            'max_steps': self.config['environment']['max_steps']
        }

        # Prepare training configuration
        training_config = self.config['training'].copy()
        if training_timesteps:
            training_config['total_timesteps'] = training_timesteps

        training_config['model_save_path'] = str(self.model_save_dir)

        # Initialize optimizer
        self.optimizer = PPOVTracerOptimizer(
            env_kwargs=env_kwargs,
            model_config=self.config['model'],
            training_config=training_config
        )

        try:
            # Start training
            start_time = time.time()
            training_results = self.optimizer.train()
            training_time = time.time() - start_time

            # Mark as trained
            self.is_trained = True

            # Save configuration
            config_path = self.model_save_dir / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)

            # Compile final results
            final_results = {
                **training_results,
                'training_image': training_image,
                'training_time_total': training_time,
                'config_saved': str(config_path)
            }

            logger.info(f"Agent training completed in {training_time:.2f} seconds")
            logger.info(f"Best quality achieved: {training_results.get('best_quality', 0.0):.4f}")

            return final_results

        except Exception as e:
            logger.error(f"Agent training failed: {e}")
            raise

    def optimize_image(self, image_path: str,
                      max_episodes: int = 10,
                      use_pretrained: bool = True) -> Dict[str, Any]:
        """
        Optimize VTracer parameters for a specific image

        Args:
            image_path: Path to image to optimize
            max_episodes: Maximum optimization episodes
            use_pretrained: Whether to use pretrained model

        Returns:
            Optimization results
        """
        logger.info(f"Optimizing parameters for: {image_path}")

        if not self.optimizer and use_pretrained:
            # Try to load pretrained model
            model_path = self.model_save_dir / "best_model"
            if model_path.exists():
                self._load_pretrained_model(str(model_path), image_path)
            else:
                raise ValueError("No trained model available. Train first or set use_pretrained=False")

        if not self.optimizer:
            raise ValueError("No optimizer available. Train first or load pretrained model")

        try:
            optimization_results = self.optimizer.optimize_parameters(
                image_path=image_path,
                max_episodes=max_episodes
            )

            logger.info(f"Optimization complete. Best quality: {optimization_results['best_quality']:.4f}")
            return optimization_results

        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            raise

    def _load_pretrained_model(self, model_path: str, image_path: str) -> None:
        """Load a pretrained model for optimization"""
        logger.info(f"Loading pretrained model from: {model_path}")

        # Create environment for the specific image
        env_kwargs = {
            'image_path': image_path,
            'target_quality': self.config['environment']['target_quality'],
            'max_steps': self.config['environment']['max_steps']
        }

        # Initialize optimizer
        self.optimizer = PPOVTracerOptimizer(
            env_kwargs=env_kwargs,
            model_config=self.config['model'],
            training_config=self.config['training']
        )

        # Load the model
        self.optimizer.load_model(model_path)
        self.is_trained = True

    def batch_optimize(self, image_paths: List[str],
                      max_episodes_per_image: int = 10) -> Dict[str, Any]:
        """
        Optimize parameters for multiple images

        Args:
            image_paths: List of image paths to optimize
            max_episodes_per_image: Max episodes per image

        Returns:
            Batch optimization results
        """
        logger.info(f"Starting batch optimization for {len(image_paths)} images")

        batch_results = {}
        total_start_time = time.time()

        for i, image_path in enumerate(image_paths):
            logger.info(f"Optimizing image {i+1}/{len(image_paths)}: {image_path}")

            try:
                result = self.optimize_image(
                    image_path=image_path,
                    max_episodes=max_episodes_per_image
                )
                batch_results[image_path] = result

            except Exception as e:
                logger.error(f"Failed to optimize {image_path}: {e}")
                batch_results[image_path] = {
                    'error': str(e),
                    'best_quality': 0.0,
                    'success': False
                }

        total_time = time.time() - total_start_time

        # Calculate summary statistics
        successful_optimizations = [r for r in batch_results.values() if r.get('success', True)]
        average_quality = sum(r['best_quality'] for r in successful_optimizations) / len(successful_optimizations) if successful_optimizations else 0.0

        summary = {
            'total_images': len(image_paths),
            'successful_optimizations': len(successful_optimizations),
            'average_quality': average_quality,
            'total_time': total_time,
            'results': batch_results
        }

        logger.info(f"Batch optimization complete. Success rate: {len(successful_optimizations)}/{len(image_paths)}")
        logger.info(f"Average quality: {average_quality:.4f}")

        return summary

    def evaluate_performance(self, test_images: List[str],
                           episodes_per_image: int = 5) -> Dict[str, Any]:
        """
        Evaluate agent performance on test images

        Args:
            test_images: List of test image paths
            episodes_per_image: Episodes per test image

        Returns:
            Performance evaluation results
        """
        logger.info(f"Evaluating performance on {len(test_images)} test images")

        if not self.is_trained:
            raise ValueError("Agent must be trained before evaluation")

        evaluation_results = {}

        for image_path in test_images:
            logger.info(f"Evaluating on: {image_path}")

            try:
                result = self.optimize_image(
                    image_path=image_path,
                    max_episodes=episodes_per_image
                )
                evaluation_results[image_path] = result

            except Exception as e:
                logger.error(f"Evaluation failed for {image_path}: {e}")
                evaluation_results[image_path] = {
                    'error': str(e),
                    'best_quality': 0.0
                }

        # Calculate performance metrics
        qualities = [r['best_quality'] for r in evaluation_results.values() if 'error' not in r]
        target_reached = sum(1 for r in evaluation_results.values() if r.get('target_reached', False))

        performance_summary = {
            'test_images_count': len(test_images),
            'successful_evaluations': len(qualities),
            'average_quality': sum(qualities) / len(qualities) if qualities else 0.0,
            'min_quality': min(qualities) if qualities else 0.0,
            'max_quality': max(qualities) if qualities else 0.0,
            'target_reached_count': target_reached,
            'target_reached_rate': target_reached / len(test_images) if test_images else 0.0,
            'detailed_results': evaluation_results
        }

        logger.info(f"Evaluation complete. Average quality: {performance_summary['average_quality']:.4f}")
        logger.info(f"Target reached rate: {performance_summary['target_reached_rate']:.2%}")

        return performance_summary

    def save_config(self, config_path: str) -> None:
        """Save current configuration to file"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to: {config_path}")

    def close(self) -> None:
        """Clean up resources"""
        if self.optimizer:
            self.optimizer.close()
        logger.info("VTracer Agent Interface closed")


# Convenience functions for easy usage
def train_vtracer_agent(training_image: str,
                       save_dir: str = "models/vtracer_ppo",
                       timesteps: int = 50000) -> VTracerAgentInterface:
    """
    Convenience function to train a VTracer agent

    Args:
        training_image: Image to train on
        save_dir: Directory to save model
        timesteps: Training timesteps

    Returns:
        Trained agent interface
    """
    agent = VTracerAgentInterface(model_save_dir=save_dir)
    agent.train_agent(training_image, training_timesteps=timesteps)
    return agent


def optimize_with_pretrained(image_path: str,
                           model_dir: str = "models/vtracer_ppo",
                           max_episodes: int = 10) -> Dict[str, Any]:
    """
    Convenience function to optimize with pretrained model

    Args:
        image_path: Image to optimize
        model_dir: Directory containing pretrained model
        max_episodes: Maximum optimization episodes

    Returns:
        Optimization results
    """
    agent = VTracerAgentInterface(model_save_dir=model_dir)
    return agent.optimize_image(image_path, max_episodes=max_episodes)