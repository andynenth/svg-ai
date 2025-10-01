# backend/ai_modules/optimization/ppo_optimizer.py
"""PPO-based optimizer for VTracer parameter optimization"""

import numpy as np
import torch
import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from .vtracer_env import VTracerOptimizationEnv
from .real_time_monitor import RealTimeMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingProgressCallback(BaseCallback):
    """Custom callback to track training progress and best performance"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_qualities = []
        self.best_quality = 0.0
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        """Called after each environment step"""
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'quality' in info:
                    quality = info['quality']
                    self.episode_qualities.append(quality)
                    if quality > self.best_quality:
                        self.best_quality = quality
                        logger.info(f"New best quality achieved: {quality:.4f}")

                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    self.episode_rewards.append(episode_reward)
                    if episode_reward > self.best_reward:
                        self.best_reward = episode_reward
                        logger.info(f"New best reward achieved: {episode_reward:.4f}")


class PPOVTracerOptimizer:
    """PPO-based optimizer for VTracer parameter optimization"""

    def __init__(self,
                 env_kwargs: Dict[str, Any],
                 model_config: Optional[Dict] = None,
                 training_config: Optional[Dict] = None,
                 enable_real_time_monitoring: bool = True):
        """
        Initialize PPO optimizer for VTracer parameter optimization

        Args:
            env_kwargs: Arguments for VTracer environment creation
            model_config: PPO model configuration
            training_config: Training configuration
            enable_real_time_monitoring: Enable real-time monitoring
        """
        self.env_kwargs = env_kwargs
        self.model_config = model_config or self._default_model_config()
        self.training_config = training_config or self._default_training_config()
        self.enable_real_time_monitoring = enable_real_time_monitoring

        # Initialize components
        self.env = None
        self.model = None
        self.vec_env = None
        self.normalized_env = None

        # Training tracking
        self.training_history = []
        self.best_performance = {'reward': -np.inf, 'quality': 0.0}
        self.callbacks = []

        # Real-time monitoring
        self.real_time_monitor = None
        if self.enable_real_time_monitoring:
            self.real_time_monitor = RealTimeMonitor(
                websocket_port=self.training_config.get('websocket_port', 8765),
                save_dir=os.path.join(self.training_config.get('model_save_path', 'models/ppo_vtracer'), 'monitoring')
            )

        # Model storage
        self.model_save_path = self.training_config.get('model_save_path', 'models/ppo_vtracer')
        os.makedirs(self.model_save_path, exist_ok=True)

        logger.info("PPO VTracer Optimizer initialized")

    def _default_model_config(self) -> Dict[str, Any]:
        """Default PPO configuration for VTracer optimization"""
        return {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': {
                'net_arch': [{'pi': [128, 128], 'vf': [128, 128]}],
                'activation_fn': torch.nn.Tanh
            },
            'verbose': 1,
            'device': 'auto'
        }

    def _default_training_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            'total_timesteps': 100000,
            'eval_freq': 10000,
            'n_eval_episodes': 5,
            'deterministic_eval': True,
            'model_save_path': 'models/ppo_vtracer',
            'checkpoint_freq': 25000,
            'early_stopping_patience': 50000,
            'target_quality': 0.85,
            'n_envs': 4  # Number of parallel environments
        }

    def _create_env(self, rank: int = 0):
        """Create a single VTracer environment with monitoring"""
        def _init():
            env = VTracerOptimizationEnv(**self.env_kwargs)
            env = Monitor(env)
            return env
        return _init

    def setup_environment(self) -> None:
        """Setup vectorized environment for training"""
        logger.info("Setting up vectorized environment...")

        # Create vectorized environment
        n_envs = self.training_config['n_envs']

        # Use simple environment creation for compatibility
        try:
            self.vec_env = make_vec_env(
                self._create_env,
                n_envs=n_envs,
                vec_env_cls=SubprocVecEnv,
                vec_env_kwargs={'start_method': 'spawn'}
            )
        except Exception as e:
            logger.warning(f"Failed to create SubprocVecEnv, falling back to single env: {e}")
            # Fallback to single environment
            from stable_baselines3.common.vec_env import DummyVecEnv
            self.vec_env = DummyVecEnv([self._create_env(0)])

        # Apply environment normalization
        self.normalized_env = VecNormalize(
            self.vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=self.model_config['gamma']
        )

        logger.info(f"Created {n_envs} parallel environments with normalization")

    def setup_model(self) -> None:
        """Initialize PPO model"""
        logger.info("Setting up PPO model...")

        if self.normalized_env is None:
            raise ValueError("Environment must be setup before model")

        # Initialize PPO model
        self.model = PPO(
            'MlpPolicy',
            self.normalized_env,
            **self.model_config
        )

        logger.info("PPO model initialized with configuration:")
        for key, value in self.model_config.items():
            if key != 'policy_kwargs':
                logger.info(f"  {key}: {value}")

    def setup_callbacks(self) -> None:
        """Setup training callbacks"""
        self.callbacks = []

        # Training progress callback
        progress_callback = TrainingProgressCallback(verbose=1)
        self.callbacks.append(progress_callback)

        # Real-time monitoring callback
        if self.real_time_monitor:
            monitoring_callback = self.real_time_monitor.create_callback()
            self.callbacks.append(monitoring_callback)

        # Evaluation callback
        eval_callback = EvalCallback(
            self.normalized_env,
            best_model_save_path=os.path.join(self.model_save_path, 'best_model'),
            log_path=os.path.join(self.model_save_path, 'logs'),
            eval_freq=self.training_config['eval_freq'],
            n_eval_episodes=self.training_config['n_eval_episodes'],
            deterministic=self.training_config['deterministic_eval'],
            verbose=1
        )
        self.callbacks.append(eval_callback)

        logger.info("Training callbacks configured")

    async def start_monitoring(self):
        """Start real-time monitoring"""
        if self.real_time_monitor:
            await self.real_time_monitor.start_monitoring()

    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.real_time_monitor:
            await self.real_time_monitor.stop_monitoring()

    def train(self) -> Dict[str, Any]:
        """Train the PPO agent"""
        logger.info("Starting PPO training...")

        # Setup components if not already done
        if self.normalized_env is None:
            self.setup_environment()
        if self.model is None:
            self.setup_model()
        if not self.callbacks:
            self.setup_callbacks()

        # Training configuration
        total_timesteps = self.training_config['total_timesteps']
        checkpoint_freq = self.training_config['checkpoint_freq']

        # Start training
        start_time = time.time()

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.callbacks,
                reset_num_timesteps=False,
                progress_bar=True
            )

            training_time = time.time() - start_time

            # Save final model
            final_model_path = os.path.join(self.model_save_path, 'final_model')
            self.model.save(final_model_path)

            # Save environment normalization statistics
            if self.normalized_env:
                self.normalized_env.save(os.path.join(self.model_save_path, 'env_stats.pkl'))

            # Compile training results
            training_results = {
                'total_timesteps': total_timesteps,
                'training_time': training_time,
                'final_model_path': final_model_path,
                'best_quality': self.callbacks[0].best_quality if self.callbacks else 0.0,
                'best_reward': self.callbacks[0].best_reward if self.callbacks else -np.inf,
                'average_quality': np.mean(self.callbacks[0].episode_qualities) if self.callbacks else 0.0,
                'average_reward': np.mean(self.callbacks[0].episode_rewards) if self.callbacks else 0.0
            }

            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Best quality achieved: {training_results['best_quality']:.4f}")
            logger.info(f"Best reward achieved: {training_results['best_reward']:.4f}")

            return training_results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def optimize_parameters(self, image_path: str, max_episodes: int = 10) -> Dict[str, Any]:
        """
        Use trained model to optimize parameters for a specific image

        Args:
            image_path: Path to image to optimize
            max_episodes: Maximum episodes for optimization

        Returns:
            Optimization results
        """
        if self.model is None:
            raise ValueError("Model must be trained before optimization")

        logger.info(f"Optimizing parameters for: {image_path}")

        # Create single environment for optimization
        env = VTracerOptimizationEnv(image_path, **{k:v for k,v in self.env_kwargs.items() if k != 'image_path'})

        # Run optimization episodes
        best_quality = 0.0
        best_params = None
        best_reward = -np.inf

        for episode in range(max_episodes):
            obs, _ = env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                # Get action from trained model
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward

                if done or truncated:
                    quality = info.get('quality', 0.0)
                    if quality > best_quality:
                        best_quality = quality
                        best_params = env.current_params.copy()
                        best_reward = episode_reward

                    logger.info(f"Episode {episode + 1}: Quality={quality:.4f}, Reward={episode_reward:.2f}")
                    break

        env.close()

        optimization_results = {
            'best_quality': best_quality,
            'best_reward': best_reward,
            'best_parameters': best_params,
            'episodes_run': max_episodes,
            'target_reached': best_quality >= env.target_quality
        }

        logger.info(f"Optimization complete. Best quality: {best_quality:.4f}")
        return optimization_results

    def load_model(self, model_path: str) -> None:
        """Load a trained model"""
        logger.info(f"Loading model from: {model_path}")

        if not self.normalized_env:
            self.setup_environment()

        self.model = PPO.load(model_path, env=self.normalized_env)

        # Load environment normalization if available
        env_stats_path = os.path.join(os.path.dirname(model_path), 'env_stats.pkl')
        if os.path.exists(env_stats_path):
            self.normalized_env = VecNormalize.load(env_stats_path, self.vec_env)

        logger.info("Model loaded successfully")

    def save_model(self, save_path: str) -> None:
        """Save the current model"""
        if self.model is None:
            raise ValueError("No model to save")

        logger.info(f"Saving model to: {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)

        # Save environment normalization
        if self.normalized_env:
            env_stats_path = os.path.join(os.path.dirname(save_path), 'env_stats.pkl')
            self.normalized_env.save(env_stats_path)

        logger.info("Model saved successfully")

    def evaluate_model(self, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        logger.info(f"Evaluating model over {n_episodes} episodes...")

        episode_rewards = []
        episode_qualities = []

        for episode in range(n_episodes):
            obs = self.normalized_env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.normalized_env.step(action)
                episode_reward += reward

                if done:
                    episode_rewards.append(episode_reward)
                    if len(info) > 0 and 'quality' in info[0]:
                        episode_qualities.append(info[0]['quality'])

        evaluation_results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_quality': np.mean(episode_qualities) if episode_qualities else 0.0,
            'std_quality': np.std(episode_qualities) if episode_qualities else 0.0,
            'n_episodes': n_episodes
        }

        logger.info(f"Evaluation complete. Mean reward: {evaluation_results['mean_reward']:.2f}")
        logger.info(f"Mean quality: {evaluation_results['mean_quality']:.4f}")

        return evaluation_results

    def close(self) -> None:
        """Clean up resources"""
        if self.normalized_env:
            self.normalized_env.close()
        if self.vec_env:
            self.vec_env.close()

        # Save monitoring data before closing
        if self.real_time_monitor:
            self.real_time_monitor.save_monitoring_data()

        logger.info("PPO VTracer Optimizer closed")


# Factory function for easy initialization
def create_ppo_optimizer(image_path: str,
                        target_quality: float = 0.85,
                        max_steps: int = 50) -> PPOVTracerOptimizer:
    """
    Factory function to create PPO optimizer with default configuration

    Args:
        image_path: Path to training image
        target_quality: Target SSIM quality
        max_steps: Maximum steps per episode

    Returns:
        Configured PPO optimizer
    """
    env_kwargs = {
        'image_path': image_path,
        'target_quality': target_quality,
        'max_steps': max_steps
    }

    return PPOVTracerOptimizer(env_kwargs)