#!/usr/bin/env python3
"""
VTracer Gymnasium Environment for RL Parameter Optimization
Day 6 Implementation - VTracer Gym Environment
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import tempfile
import cv2
import time
import logging
import os
from pathlib import Path

from ..feature_extraction import ImageFeatureExtractor
from .quality_metrics import OptimizationQualityMetrics
from .parameter_bounds import VTracerParameterBounds
from .reward_functions import MultiObjectiveRewardFunction, ConversionResult


class VTracerOptimizationEnv(gym.Env):
    """Gymnasium environment for VTracer parameter optimization using RL"""

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 image_path: str,
                 target_quality: float = 0.85,
                 max_steps: int = 50):
        super().__init__()

        # Environment setup
        self.image_path = image_path
        self.target_quality = target_quality
        self.max_steps = max_steps
        self.current_step = 0

        # Initialize logger first
        self.logger = logging.getLogger(__name__)

        # Validate image path
        if not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")

        # Initialize components
        self.feature_extractor = ImageFeatureExtractor()
        self.quality_metrics = OptimizationQualityMetrics()
        self.bounds = VTracerParameterBounds
        self.reward_function = MultiObjectiveRewardFunction(target_quality=target_quality)

        # Define action and observation spaces
        self._define_spaces()

        # Environment state
        self.current_params = None
        self.best_quality = 0.0
        self.baseline_quality = None
        self.baseline_params = None
        self.baseline_time = None
        self.baseline_size = None
        self.baseline_result = None

        # Episode tracking
        self.episode_history = []
        self.quality_improvements = []

        # Configuration
        self.config = {
            "quality_weight": 0.6,
            "speed_weight": 0.3,
            "size_weight": 0.1,
            "reward_scaling": 1.0,
            "early_stop_threshold": 0.95  # Stop if we exceed this fraction of target
        }

        self.logger.info(f"VTracer RL Environment initialized for {image_path}")

    def _define_spaces(self):
        """Define action and observation spaces for RL"""
        # Action space: 7 continuous parameters (normalized to [0,1])
        # Parameters: color_precision, layer_difference, corner_threshold,
        #            length_threshold, max_iterations, splice_threshold, path_precision
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Observation space: features + current params + quality metrics
        # - Image features: 6 dimensions (edge_density, unique_colors, entropy,
        #   corner_density, gradient_strength, complexity_score)
        # - Current parameters: 7 dimensions (normalized)
        # - Quality metrics: 2 dimensions (current_quality, quality_improvement)
        # Total: 15-dimensional observation space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(15,), dtype=np.float32
        )

        self.logger.info(f"Action space: {self.action_space}")
        self.logger.info(f"Observation space: {self.observation_space}")

    def _denormalize_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Convert [0,1] actions to actual VTracer parameter ranges"""
        try:
            # Ensure action is in valid range
            action = np.clip(action, 0.0, 1.0)

            # Get parameter bounds
            bounds = self.bounds.get_bounds()

            # Map normalized actions to actual parameter values
            params = {}

            # color_precision: integer 1-32
            params['color_precision'] = int(1 + action[0] * 31)

            # layer_difference: integer 1-32
            params['layer_difference'] = int(1 + action[1] * 31)

            # corner_threshold: integer 0-180 (exponential scaling for fine-tuning at low values)
            params['corner_threshold'] = int(180.0 * (action[2] ** 2))

            # length_threshold: float 0-50 (logarithmic scaling for more options at small values)
            params['length_threshold'] = float(50.0 * (1 - np.exp(-3 * action[3])))

            # max_iterations: integer 1-100
            params['max_iterations'] = int(1 + action[4] * 99)

            # splice_threshold: float 0-180
            params['splice_threshold'] = float(action[5] * 180)

            # path_precision: integer 1-24 (exponential scaling for more precision at high values)
            params['path_precision'] = int(1 + 23 * (action[6] ** 0.5))

            # Validate parameters are within bounds using parameter bounds
            for param_name, value in params.items():
                if param_name in bounds:
                    param_spec = bounds[param_name]
                    min_val = param_spec.get('min', value)
                    max_val = param_spec.get('max', value)
                    params[param_name] = max(min_val, min(max_val, value))

            # Apply type constraints based on parameter bounds
            for param_name, value in params.items():
                if param_name in bounds:
                    param_spec = bounds[param_name]
                    expected_type = param_spec.get('type', type(value))
                    if expected_type == int:
                        params[param_name] = int(value)
                    elif expected_type == float:
                        params[param_name] = float(value)

            # Add mode parameter (not optimized by RL for simplicity)
            params['mode'] = 'spline'

            return params

        except Exception as e:
            self.logger.error(f"Failed to denormalize action: {e}")
            # Return default parameters as fallback
            return {
                'color_precision': 4,
                'layer_difference': 16,
                'corner_threshold': 60.0,
                'length_threshold': 4.0,
                'max_iterations': 10,
                'splice_threshold': 45,
                'path_precision': 8,
                'mode': 'spline'
            }

    def _normalize_parameters(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert VTracer parameters back to normalized [0,1] action space"""
        try:
            action = np.zeros(7, dtype=np.float32)

            # color_precision: 1-32 -> [0,1]
            action[0] = (params['color_precision'] - 1) / 31.0

            # layer_difference: 1-32 -> [0,1]
            action[1] = (params['layer_difference'] - 1) / 31.0

            # corner_threshold: 0-180 -> [0,1] (inverse of exponential scaling)
            action[2] = np.sqrt(params['corner_threshold'] / 180.0)

            # length_threshold: 0-50 -> [0,1] (inverse of logarithmic scaling)
            normalized_length = params['length_threshold'] / 50.0
            action[3] = -np.log(1 - normalized_length) / 3.0 if normalized_length < 1.0 else 1.0

            # max_iterations: 1-100 -> [0,1]
            action[4] = (params['max_iterations'] - 1) / 99.0

            # splice_threshold: 0-180 -> [0,1]
            action[5] = params['splice_threshold'] / 180.0

            # path_precision: 1-24 -> [0,1] (inverse of square root scaling)
            action[6] = ((params['path_precision'] - 1) / 23.0) ** 2

            # Clip to valid range
            action = np.clip(action, 0.0, 1.0)

            return action

        except Exception as e:
            self.logger.error(f"Failed to normalize parameters: {e}")
            return np.array([0.5] * 7, dtype=np.float32)  # Default normalized values

    def _calculate_ssim_quality(self, original_path: str, svg_path: str) -> float:
        """Calculate SSIM quality score between original and SVG"""
        try:
            import cv2
            from skimage.metrics import structural_similarity as ssim
            import cairosvg
            from PIL import Image
            import io

            # Read original image
            original = cv2.imread(original_path)
            if original is None:
                self.logger.error(f"Could not read original image: {original_path}")
                return 0.0

            # Convert SVG to PNG for comparison
            try:
                svg_png_bytes = cairosvg.svg2png(url=svg_path)
                svg_image = Image.open(io.BytesIO(svg_png_bytes))
                svg_array = np.array(svg_image.convert('RGB'))
                svg_bgr = cv2.cvtColor(svg_array, cv2.COLOR_RGB2BGR)
            except Exception as e:
                self.logger.error(f"Could not convert SVG to PNG: {e}")
                return 0.0

            # Resize to same dimensions
            h, w = original.shape[:2]
            svg_resized = cv2.resize(svg_bgr, (w, h))

            # Convert to grayscale for SSIM
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            svg_gray = cv2.cvtColor(svg_resized, cv2.COLOR_BGR2GRAY)

            # Calculate SSIM
            ssim_score = ssim(original_gray, svg_gray, data_range=255)

            return float(max(0.0, min(1.0, ssim_score)))  # Clamp to [0,1]

        except Exception as e:
            self.logger.error(f"Failed to calculate SSIM quality: {e}")
            return 0.0

    def _extract_image_features(self) -> np.ndarray:
        """Extract and normalize image features"""
        try:
            features = self.feature_extractor.extract_features(self.image_path)

            # Expected features: edge_density, unique_colors, entropy, corner_density,
            #                   gradient_strength, complexity_score
            feature_vector = np.array([
                features.get('edge_density', 0.5),
                features.get('unique_colors', 0.5),
                features.get('entropy', 0.5),
                features.get('corner_density', 0.5),
                features.get('gradient_strength', 0.5),
                features.get('complexity_score', 0.5)
            ], dtype=np.float32)

            # Ensure features are normalized to [0,1]
            feature_vector = np.clip(feature_vector, 0.0, 1.0)

            return feature_vector

        except Exception as e:
            self.logger.error(f"Failed to extract image features: {e}")
            # Return default normalized features
            return np.array([0.5] * 6, dtype=np.float32)

    def _generate_observation(self) -> np.ndarray:
        """Generate current observation state"""
        try:
            # Image features (6 dimensions)
            image_features = self._extract_image_features()

            # Current parameters normalized (7 dimensions)
            if self.current_params is not None:
                param_features = self._normalize_parameters(self.current_params)
            else:
                param_features = np.array([0.5] * 7, dtype=np.float32)

            # Quality metrics (2 dimensions)
            current_quality = min(1.0, max(0.0, self.best_quality))
            quality_improvement = 0.0
            if self.baseline_quality is not None:
                quality_improvement = min(1.0, max(0.0,
                    (self.best_quality - self.baseline_quality) / max(0.01, self.baseline_quality)))

            quality_features = np.array([current_quality, quality_improvement], dtype=np.float32)

            # Combine all features (total: 15 dimensions)
            observation = np.concatenate([
                image_features,      # 6 dimensions
                param_features,      # 7 dimensions
                quality_features     # 2 dimensions
            ], dtype=np.float32)

            # Ensure observation is in valid range
            observation = np.clip(observation, 0.0, 1.0)

            return observation

        except Exception as e:
            self.logger.error(f"Failed to generate observation: {e}")
            # Return default observation
            return np.array([0.5] * 15, dtype=np.float32)

    def _calculate_baseline_quality(self) -> float:
        """Calculate baseline quality with default parameters"""
        try:
            # Default VTracer parameters
            default_params = {
                'color_precision': 4,
                'layer_difference': 16,
                'corner_threshold': 60.0,
                'length_threshold': 4.0,
                'max_iterations': 10,
                'splice_threshold': 45,
                'path_precision': 8,
                'mode': 'spline'
            }

            # Convert image and measure quality
            result = self._convert_and_measure(default_params)

            if result['success']:
                self.baseline_quality = result['quality']
                self.baseline_params = default_params
                self.baseline_time = result['processing_time']
                self.baseline_size = result['file_size']

                # Create baseline ConversionResult for reward function
                self.baseline_result = ConversionResult(
                    quality_score=result['quality'],
                    processing_time=result['processing_time'],
                    file_size=result['file_size'],
                    success=result['success'],
                    svg_path=result.get('svg_path', '')
                )

                self.logger.info(f"Baseline quality established: {self.baseline_quality:.3f}")
                return self.baseline_quality
            else:
                self.logger.warning("Failed to establish baseline quality")
                self.baseline_quality = 0.5  # Default baseline
                return self.baseline_quality

        except Exception as e:
            self.logger.error(f"Failed to calculate baseline quality: {e}")
            self.baseline_quality = 0.5  # Default baseline
            return self.baseline_quality

    def _convert_and_measure(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert image with VTracer and measure quality"""
        try:
            import vtracer

            start_time = time.time()

            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_file:
                output_path = tmp_file.name

            try:
                # Convert with VTracer
                vtracer.convert_image_to_svg_py(self.image_path, output_path, **params)

                processing_time = time.time() - start_time

                # Measure file size
                file_size = os.path.getsize(output_path) / 1024.0  # KB

                # Calculate quality metrics using SSIM
                quality_score = self._calculate_ssim_quality(
                    original_path=self.image_path,
                    svg_path=output_path
                )

                return {
                    'success': True,
                    'quality': quality_score,
                    'processing_time': processing_time,
                    'file_size': file_size,
                    'svg_path': output_path
                }

            finally:
                # Clean up temporary file
                if os.path.exists(output_path):
                    try:
                        os.unlink(output_path)
                    except:
                        pass

        except Exception as e:
            self.logger.error(f"VTracer conversion failed: {e}")
            return {
                'success': False,
                'quality': 0.0,
                'processing_time': 1.0,  # Penalty time
                'file_size': 0.0,
                'svg_path': None
            }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        try:
            # Set random seed
            if seed is not None:
                np.random.seed(seed)

            # Reset episode state
            self.current_step = 0
            self.best_quality = 0.0
            self.episode_history = []
            self.quality_improvements = []

            # Reset reward function convergence history
            if hasattr(self.reward_function, 'reset_convergence_history'):
                self.reward_function.reset_convergence_history()

            # Calculate baseline quality if not done yet
            if self.baseline_quality is None:
                self._calculate_baseline_quality()

            # Initialize with random parameters
            initial_action = self.action_space.sample()
            self.current_params = self._denormalize_action(initial_action)

            # Generate initial observation
            observation = self._generate_observation()

            info = {
                'episode': 0,
                'step': self.current_step,
                'baseline_quality': self.baseline_quality,
                'current_params': self.current_params.copy()
            }

            self.logger.info(f"Environment reset - baseline quality: {self.baseline_quality:.3f}")

            return observation, info

        except Exception as e:
            self.logger.error(f"Failed to reset environment: {e}")
            # Return safe defaults
            return np.array([0.5] * 15, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take environment step with given action"""
        try:
            self.current_step += 1

            # Convert action to parameters
            new_params = self._denormalize_action(action)
            self.current_params = new_params

            # Convert and measure quality
            result = self._convert_and_measure(new_params)

            # Calculate reward (will be implemented in Task A6.2)
            reward = self._calculate_reward(result)

            # Update best quality
            if result['success'] and result['quality'] > self.best_quality:
                self.best_quality = result['quality']
                self.quality_improvements.append((self.current_step, self.best_quality))

            # Check episode termination conditions
            terminated = self._check_termination(result)
            truncated = self.current_step >= self.max_steps

            # Generate new observation
            observation = self._generate_observation()

            # Track episode history
            self.episode_history.append({
                'step': self.current_step,
                'action': action.copy(),
                'parameters': new_params.copy(),
                'result': result.copy(),
                'reward': reward,
                'quality': result['quality'],
                'best_quality': self.best_quality
            })

            # Info dictionary
            info = {
                'step': self.current_step,
                'parameters': new_params.copy(),
                'quality': result['quality'],
                'best_quality': self.best_quality,
                'processing_time': result['processing_time'],
                'file_size': result['file_size'],
                'success': result['success'],
                'quality_improvement': result['quality'] - (self.baseline_quality or 0.0),
                'target_reached': result['quality'] >= self.target_quality
            }

            done = terminated or truncated

            if done:
                self.logger.info(f"Episode completed - steps: {self.current_step}, "
                               f"best quality: {self.best_quality:.3f}, "
                               f"target reached: {info['target_reached']}")

            return observation, reward, terminated, truncated, info

        except Exception as e:
            self.logger.error(f"Step failed: {e}")
            # Return safe defaults
            observation = self._generate_observation()
            return observation, -1.0, True, False, {'error': str(e)}

    def _calculate_reward(self, result: Dict[str, Any]) -> float:
        """Calculate reward using multi-objective reward function"""
        try:
            # Create ConversionResult object
            conversion_result = ConversionResult(
                quality_score=result['quality'],
                processing_time=result['processing_time'],
                file_size=result['file_size'],
                success=result['success'],
                svg_path=result.get('svg_path', '')
            )

            # Use baseline result if available
            baseline = self.baseline_result
            if baseline is None:
                # Create dummy baseline if not available
                baseline = ConversionResult(
                    quality_score=self.baseline_quality or 0.5,
                    processing_time=self.baseline_time or 0.1,
                    file_size=self.baseline_size or 10.0,
                    success=True,
                    svg_path=''
                )

            # Calculate reward using multi-objective function
            reward, components = self.reward_function.calculate_reward(
                conversion_result, baseline, self.current_step, self.max_steps
            )

            # Log reward components for debugging
            if self.current_step % 10 == 0:  # Log every 10 steps
                self.logger.debug(f"Reward components: {components}")

            return float(reward)

        except Exception as e:
            self.logger.error(f"Reward calculation failed: {e}")
            # Fallback to basic reward
            if not result['success']:
                return -1.0
            quality_improvement = result['quality'] - (self.baseline_quality or 0.0)
            return float(quality_improvement * 10.0)

    def _check_termination(self, result: Dict[str, Any]) -> bool:
        """Check if episode should terminate"""
        # Terminate if target quality achieved
        if result['success'] and result['quality'] >= self.target_quality:
            return True

        # Terminate if quality improvement has plateaued (basic check)
        if len(self.quality_improvements) >= 5:
            recent_improvements = [qi[1] for qi in self.quality_improvements[-5:]]
            if max(recent_improvements) - min(recent_improvements) < 0.01:
                return True  # Quality plateaued

        # Terminate on conversion failure
        if not result['success']:
            return False  # Don't terminate immediately on failure, allow recovery

        return False

    def render(self, mode: str = 'human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Best Quality: {self.best_quality:.3f}")
            print(f"Target Quality: {self.target_quality:.3f}")
            if self.current_params:
                print(f"Current Parameters: {self.current_params}")
        return None

    def close(self):
        """Clean up environment resources"""
        self.logger.info("VTracer RL Environment closed")
        # Clean up any temporary files or resources
        pass

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of current episode"""
        return {
            'steps': self.current_step,
            'best_quality': self.best_quality,
            'baseline_quality': self.baseline_quality,
            'target_quality': self.target_quality,
            'target_reached': self.best_quality >= self.target_quality,
            'quality_improvements': len(self.quality_improvements),
            'episode_history': self.episode_history.copy()
        }

    def configure(self, **kwargs):
        """Configure environment parameters"""
        for key, value in kwargs.items():
            if key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                self.logger.info(f"Configuration updated: {key} {old_value} → {value}")
            else:
                self.logger.warning(f"Unknown configuration parameter: {key}")