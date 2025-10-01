# backend/ai_modules/optimization/rl_optimizer.py
"""Reinforcement learning-based parameter optimization"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import logging
import os
from stable_baselines3 import PPO
from .base_optimizer import BaseOptimizer
from .vtracer_environment import VTracerEnvironment

logger = logging.getLogger(__name__)


class RLOptimizer(BaseOptimizer):
    """Optimize VTracer parameters using reinforcement learning"""

    def __init__(self, model_path: str = None):
        super().__init__("ReinforcementLearning")
        self.model_path = model_path
        self.agent = None
        self.environment = None
        self.training_episodes = 0
        self.total_reward = 0.0

    def _optimize_impl(self, features: Dict[str, float], logo_type: str) -> Dict[str, Any]:
        """Implement RL-based optimization"""
        logger.debug(f"Optimizing parameters for {logo_type} using RL")

        try:
            # Initialize if needed
            if self.agent is None:
                self._initialize_agent(features)

            # Run optimization episode
            optimized_params = self._run_optimization_episode(features, logo_type)

            logger.debug(f"RL optimization result: {optimized_params}")
            return optimized_params

        except Exception as e:
            logger.error(f"RL optimization failed: {e}")
            # Fallback to default parameters
            return self._get_default_parameters(logo_type)

    def _initialize_agent(self, features: Dict[str, float]):
        """Initialize RL agent and environment"""
        try:
            # Create environment
            self.environment = VTracerEnvironment(features)

            # Create or load agent
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading RL agent from {self.model_path}")
                self.agent = PPO.load(self.model_path, env=self.environment)
            else:
                logger.info("Creating new RL agent")
                self.agent = PPO(
                    "MlpPolicy",
                    self.environment,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    verbose=0,
                )

        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise

    def _run_optimization_episode(
        self, features: Dict[str, float], logo_type: str
    ) -> Dict[str, Any]:
        """Run a single optimization episode"""
        try:
            # Reset environment with current features
            self.environment.set_target_features(features, logo_type)
            obs, info = self.environment.reset()

            # Run episode
            done = False
            step_count = 0
            max_steps = 50

            while not done and step_count < max_steps:
                # Get action from agent
                action, _states = self.agent.predict(obs, deterministic=True)

                # Take step
                obs, reward, terminated, truncated, info = self.environment.step(action)
                done = terminated or truncated
                step_count += 1

            # Get final parameters
            final_params = self.environment.get_current_parameters()

            # Update tracking
            self.total_reward += info.get("total_reward", 0)

            logger.debug(f"RL episode completed in {step_count} steps")
            return final_params

        except Exception as e:
            logger.error(f"RL episode failed: {e}")
            return self._get_default_parameters(logo_type)

    def train_agent(self, training_data: Dict[str, list], total_timesteps: int = 10000):
        """Train the RL agent"""
        try:
            if not training_data.get("features"):
                logger.warning("No training data provided for RL training")
                return False

            logger.info(f"Training RL agent for {total_timesteps} timesteps")

            # Initialize with first example
            first_features = training_data["features"][0]
            if self.agent is None:
                self._initialize_agent(first_features)

            # Set up training environment with diverse examples
            self.environment.set_training_data(training_data)

            # Train agent
            self.agent.learn(total_timesteps=total_timesteps)

            # Save model if path provided
            if self.model_path:
                self.agent.save(self.model_path)
                logger.info(f"Saved trained model to {self.model_path}")

            self.training_episodes += total_timesteps // 50  # Approximate episodes
            logger.info("RL training completed successfully")
            return True

        except Exception as e:
            logger.error(f"RL training failed: {e}")
            return False

    def evaluate_agent(self, test_features: list, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance"""
        if self.agent is None:
            logger.warning("No trained agent available for evaluation")
            return {"average_reward": 0.0, "success_rate": 0.0}

        total_rewards = []
        successful_episodes = 0

        for episode in range(num_episodes):
            try:
                # Use random test features
                features = test_features[episode % len(test_features)]
                logo_type = self._infer_logo_type(features)

                # Run episode
                self.environment.set_target_features(features, logo_type)
                obs, info = self.environment.reset()

                episode_reward = 0
                done = False
                steps = 0
                max_steps = 50

                while not done and steps < max_steps:
                    action, _states = self.agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.environment.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                    steps += 1

                total_rewards.append(episode_reward)
                if episode_reward > 0.5:  # Threshold for success
                    successful_episodes += 1

            except Exception as e:
                logger.warning(f"Evaluation episode {episode} failed: {e}")
                total_rewards.append(0.0)

        return {
            "average_reward": np.mean(total_rewards),
            "reward_std": np.std(total_rewards),
            "success_rate": successful_episodes / num_episodes,
            "total_episodes": num_episodes,
        }

    def get_rl_stats(self) -> Dict[str, Any]:
        """Get RL optimization statistics"""
        stats = {
            "training_episodes": self.training_episodes,
            "total_reward": self.total_reward,
            "agent_initialized": self.agent is not None,
            "environment_initialized": self.environment is not None,
        }

        if self.environment:
            stats["environment_stats"] = self.environment.get_stats()

        return stats

    def save_agent(self, path: str):
        """Save trained agent"""
        if self.agent:
            self.agent.save(path)
            logger.info(f"Saved RL agent to {path}")
        else:
            logger.warning("No trained agent to save")

    def load_agent(self, path: str):
        """Load trained agent"""
        try:
            if self.environment is None:
                # Create dummy environment for loading
                dummy_features = {"complexity_score": 0.5, "unique_colors": 16}
                self.environment = VTracerEnvironment(dummy_features)

            self.agent = PPO.load(path, env=self.environment)
            self.model_path = path
            logger.info(f"Loaded RL agent from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load RL agent from {path}: {e}")
            return False
