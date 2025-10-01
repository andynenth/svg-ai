# backend/ai_modules/optimization/vtracer_environment.py
"""Gym environment for VTracer parameter optimization"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from backend.ai_modules.config import VTRACER_PARAM_RANGES, PERFORMANCE_TARGETS

logger = logging.getLogger(__name__)


class VTracerEnvironment(gym.Env):
    """Gym environment for optimizing VTracer parameters using RL"""

    def __init__(self, initial_features: Dict[str, float]):
        super().__init__()

        # Environment configuration
        self.param_names = list(VTRACER_PARAM_RANGES.keys())
        self.param_ranges = VTRACER_PARAM_RANGES
        self.target_features = initial_features
        self.target_logo_type = "simple"

        # RL environment setup
        self._setup_action_space()
        self._setup_observation_space()

        # State tracking
        self.current_parameters = self._get_default_parameters()
        self.step_count = 0
        self.max_steps = 50
        self.episode_reward = 0.0

        # Performance tracking
        self.episode_history = []

        logger.debug("VTracer RL environment initialized")

    def _setup_action_space(self):
        """Setup action space for parameter adjustments"""
        # Action space: continuous adjustments for each parameter
        # Each action is a value between -1 and 1 (percentage change)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.param_names),), dtype=np.float32
        )

    def _setup_observation_space(self):
        """Setup observation space combining features and current parameters"""
        # Observation includes:
        # - Target features (8 features)
        # - Current parameters (8 parameters)
        # - Step count (1 value)
        # - Target logo type one-hot (4 values)
        obs_dim = 8 + 8 + 1 + 4  # 21 total

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Reset state
        self.current_parameters = self._get_default_parameters()
        self.step_count = 0
        self.episode_reward = 0.0

        # Get initial observation
        observation = self._get_observation()
        info = {"step": self.step_count, "parameters": self.current_parameters.copy()}

        logger.debug("Environment reset")
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment"""
        self.step_count += 1

        # Apply action to parameters
        self._apply_action(action)

        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward

        # Check if done
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        # Get new observation
        observation = self._get_observation()

        # Create info
        info = {
            "step": self.step_count,
            "parameters": self.current_parameters.copy(),
            "reward": reward,
            "total_reward": self.episode_reward,
            "quality_score": self._estimate_quality(),
        }

        if done:
            self._record_episode()

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: np.ndarray):
        """Apply action to current parameters"""
        for i, param_name in enumerate(self.param_names):
            if param_name in self.current_parameters:
                # Get current value and range
                current_value = self.current_parameters[param_name]
                min_val, max_val = self.param_ranges[param_name]

                # Calculate adjustment (action is -1 to 1, scale to 10% of range)
                adjustment_factor = action[i] * 0.1  # 10% max change per step
                range_size = max_val - min_val
                adjustment = adjustment_factor * range_size

                # Apply adjustment with clamping
                new_value = current_value + adjustment
                new_value = max(min_val, min(max_val, new_value))

                self.current_parameters[param_name] = new_value

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Feature values (8 features)
        feature_values = [
            self.target_features.get("complexity_score", 0.5),
            self.target_features.get("unique_colors", 16) / 100.0,  # Normalize
            self.target_features.get("edge_density", 0.1),
            self.target_features.get("aspect_ratio", 1.0),
            self.target_features.get("fill_ratio", 0.3),
            self.target_features.get("entropy", 6.0) / 10.0,  # Normalize
            self.target_features.get("corner_density", 0.01) * 100.0,  # Scale up
            self.target_features.get("gradient_strength", 25.0) / 100.0,  # Normalize
        ]

        # Current parameter values (8 parameters, normalized to 0-1)
        param_values = []
        for param_name in self.param_names:
            value = self.current_parameters.get(param_name, 0)
            min_val, max_val = self.param_ranges[param_name]
            normalized_value = (value - min_val) / (max_val - min_val)
            param_values.append(normalized_value)

        # Step count (normalized)
        step_progress = [self.step_count / self.max_steps]

        # Target logo type (one-hot encoding)
        logo_types = ["simple", "text", "gradient", "complex"]
        logo_type_onehot = [1.0 if self.target_logo_type == lt else 0.0 for lt in logo_types]

        # Combine all observations
        observation = np.array(
            feature_values + param_values + step_progress + logo_type_onehot, dtype=np.float32
        )

        return observation

    def _calculate_reward(self) -> float:
        """Calculate reward for current state"""
        try:
            # Estimate quality with current parameters
            quality_score = self._estimate_quality()

            # Base reward from quality
            quality_reward = quality_score

            # Bonus for reaching target thresholds
            target_quality = PERFORMANCE_TARGETS["tier_2"]["target_quality"]  # 0.90
            if quality_score >= target_quality:
                quality_reward += 0.2

            # Penalty for extreme parameter values (encourages balanced solutions)
            extremeness_penalty = self._calculate_extremeness_penalty()

            # Step penalty (encourages faster convergence)
            step_penalty = 0.01 * (self.step_count / self.max_steps)

            total_reward = quality_reward - extremeness_penalty - step_penalty

            return float(np.clip(total_reward, -1.0, 1.0))

        except Exception as e:
            logger.warning(f"Reward calculation failed: {e}")
            return 0.0

    def _estimate_quality(self) -> float:
        """Estimate conversion quality based on parameter appropriateness"""
        try:
            # This is a heuristic quality estimation for RL training
            # In real implementation, this would run actual VTracer conversion

            complexity = self.target_features.get("complexity_score", 0.5)
            unique_colors = self.target_features.get("unique_colors", 16)
            edge_density = self.target_features.get("edge_density", 0.1)

            base_quality = 0.7

            # Color precision appropriateness
            color_prec = self.current_parameters.get("color_precision", 5)
            if unique_colors <= 8 and 2 <= color_prec <= 4:
                base_quality += 0.1
            elif unique_colors > 20 and 6 <= color_prec <= 8:
                base_quality += 0.1
            elif abs(color_prec - unique_colors / 4) > 3:
                base_quality -= 0.1

            # Corner threshold appropriateness
            corner_thresh = self.current_parameters.get("corner_threshold", 50)
            if edge_density > 0.3 and corner_thresh <= 30:
                base_quality += 0.1
            elif edge_density < 0.1 and corner_thresh >= 60:
                base_quality += 0.1

            # Path precision appropriateness
            path_prec = self.current_parameters.get("path_precision", 15)
            if complexity > 0.7 and path_prec >= 20:
                base_quality += 0.1
            elif complexity < 0.3 and path_prec <= 10:
                base_quality += 0.1

            # Add some noise to make learning more robust
            noise = np.random.normal(0, 0.02)

            return float(np.clip(base_quality + noise, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"Quality estimation failed: {e}")
            return 0.5

    def _calculate_extremeness_penalty(self) -> float:
        """Calculate penalty for extreme parameter values"""
        penalty = 0.0

        for param_name, value in self.current_parameters.items():
            if param_name in self.param_ranges:
                min_val, max_val = self.param_ranges[param_name]
                normalized_value = (value - min_val) / (max_val - min_val)

                # Penalty for being too close to extremes
                if normalized_value < 0.1 or normalized_value > 0.9:
                    penalty += 0.05

        return penalty

    def _is_terminated(self) -> bool:
        """Check if episode should terminate early"""
        # Terminate if we achieve very high quality
        quality = self._estimate_quality()
        return quality >= 0.95

    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default parameters for current target type"""
        from backend.ai_modules.config import DEFAULT_VTRACER_PARAMS

        return DEFAULT_VTRACER_PARAMS.get(
            self.target_logo_type, DEFAULT_VTRACER_PARAMS["simple"]
        ).copy()

    def _record_episode(self):
        """Record episode statistics"""
        episode_data = {
            "steps": self.step_count,
            "total_reward": self.episode_reward,
            "final_quality": self._estimate_quality(),
            "target_logo_type": self.target_logo_type,
            "final_parameters": self.current_parameters.copy(),
        }
        self.episode_history.append(episode_data)

        # Keep only recent episodes
        if len(self.episode_history) > 100:
            self.episode_history = self.episode_history[-100:]

    def set_target_features(self, features: Dict[str, float], logo_type: str):
        """Set new target features and logo type"""
        self.target_features = features
        self.target_logo_type = logo_type

    def set_training_data(self, training_data: Dict[str, list]):
        """Set training data for diverse episode generation"""
        self.training_features = training_data.get("features", [])
        self.training_logo_types = []

        # Infer logo types from features if not provided
        for features in self.training_features:
            complexity = features.get("complexity_score", 0.5)
            unique_colors = features.get("unique_colors", 16)

            if complexity < 0.3 and unique_colors <= 8:
                logo_type = "simple"
            elif unique_colors > 30:
                logo_type = "gradient"
            elif complexity > 0.7:
                logo_type = "complex"
            else:
                logo_type = "text"

            self.training_logo_types.append(logo_type)

    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameter values"""
        return self.current_parameters.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics"""
        if not self.episode_history:
            return {"total_episodes": 0}

        rewards = [ep["total_reward"] for ep in self.episode_history]
        qualities = [ep["final_quality"] for ep in self.episode_history]
        steps = [ep["steps"] for ep in self.episode_history]

        return {
            "total_episodes": len(self.episode_history),
            "average_reward": np.mean(rewards),
            "average_quality": np.mean(qualities),
            "average_steps": np.mean(steps),
            "success_rate": sum(1 for q in qualities if q >= 0.85) / len(qualities),
        }
