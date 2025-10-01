#!/usr/bin/env python3
"""
Multi-Objective Reward Function for VTracer RL Optimization
Day 6 Implementation - Reward Function Components and Balancing
"""

import numpy as np
import logging
import time
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ConversionResult:
    """Structure for VTracer conversion results"""
    quality_score: float  # SSIM improvement
    processing_time: float  # Conversion time in seconds
    file_size: float  # SVG file size in KB
    success: bool  # Conversion succeeded
    svg_path: str  # Path to generated SVG


class MultiObjectiveRewardFunction:
    """Multi-objective reward function for RL optimization"""

    def __init__(self,
                 quality_weight: float = 0.6,
                 speed_weight: float = 0.3,
                 size_weight: float = 0.1,
                 target_quality: float = 0.85):
        """
        Initialize multi-objective reward function

        Args:
            quality_weight: Weight for quality improvement reward (0.6 default)
            speed_weight: Weight for speed efficiency reward (0.3 default)
            size_weight: Weight for file size optimization reward (0.1 default)
            target_quality: Target quality threshold for bonus rewards
        """
        self.quality_weight = quality_weight
        self.speed_weight = speed_weight
        self.size_weight = size_weight
        self.target_quality = target_quality

        # Ensure weights sum to 1.0
        total_weight = quality_weight + speed_weight + size_weight
        if abs(total_weight - 1.0) > 0.01:
            self.quality_weight = quality_weight / total_weight
            self.speed_weight = speed_weight / total_weight
            self.size_weight = size_weight / total_weight

        # Reward scaling and normalization parameters
        self.reward_scale = 100.0  # Global reward scaling factor
        self.quality_scale = 10.0  # Quality reward scaling
        self.target_bonus_scale = 50.0  # Target achievement bonus scale
        self.failure_penalty = -10.0  # Penalty for conversion failures

        # Convergence tracking
        self.convergence_history = []
        self.convergence_window = 5
        self.oscillation_penalty_scale = 0.1

        # Logger
        self.logger = logging.getLogger(__name__)

        # Reward component history for debugging
        self.reward_history = []

        self.logger.info(f"Multi-objective reward function initialized: "
                        f"quality={self.quality_weight:.2f}, "
                        f"speed={self.speed_weight:.2f}, "
                        f"size={self.size_weight:.2f}")

    def calculate_reward(self,
                        result: ConversionResult,
                        baseline_result: ConversionResult,
                        step: int,
                        max_steps: int) -> Tuple[float, Dict[str, float]]:
        """
        Calculate multi-objective reward with component breakdown

        Args:
            result: Current conversion result
            baseline_result: Baseline conversion result for comparison
            step: Current episode step
            max_steps: Maximum episode steps

        Returns:
            Tuple of (total_reward, component_breakdown)
        """
        try:
            # Initialize component breakdown
            components = {
                'quality_reward': 0.0,
                'speed_reward': 0.0,
                'size_reward': 0.0,
                'target_bonus': 0.0,
                'convergence_reward': 0.0,
                'failure_penalty': 0.0,
                'step_penalty': 0.0
            }

            # Handle conversion failure
            if not result.success:
                components['failure_penalty'] = self.failure_penalty
                total_reward = self.failure_penalty * 0.5  # Reduced penalty to allow learning
                self._log_reward_components(total_reward, components, step)
                return total_reward, components

            # Quality improvement reward
            components['quality_reward'] = self._calculate_quality_reward(
                result, baseline_result
            )

            # Speed efficiency reward
            components['speed_reward'] = self._calculate_speed_reward(
                result, baseline_result
            )

            # File size optimization reward
            components['size_reward'] = self._calculate_size_reward(
                result, baseline_result
            )

            # Target achievement bonus
            components['target_bonus'] = self._calculate_target_bonus(
                result, step, max_steps
            )

            # Convergence encouragement reward
            components['convergence_reward'] = self._calculate_convergence_reward(
                result, step
            )

            # Step penalty to encourage efficiency
            components['step_penalty'] = self._calculate_step_penalty(step, max_steps)

            # Calculate weighted total reward
            quality_component = components['quality_reward'] * self.quality_weight
            speed_component = components['speed_reward'] * self.speed_weight
            size_component = components['size_reward'] * self.size_weight

            total_reward = (
                quality_component +
                speed_component +
                size_component +
                components['target_bonus'] +
                components['convergence_reward'] +
                components['step_penalty']
            ) * self.reward_scale

            # Apply reward normalization and clipping
            total_reward = self._normalize_reward(total_reward, step, max_steps)

            # Log reward components for debugging
            self._log_reward_components(total_reward, components, step)

            return total_reward, components

        except Exception as e:
            self.logger.error(f"Reward calculation failed: {e}")
            # Return safe default reward
            return -1.0, {'error': str(e)}

    def _calculate_quality_reward(self, result: ConversionResult,
                                 baseline_result: ConversionResult) -> float:
        """
        Calculate quality improvement reward
        Formula: quality_reward = (current_ssim - baseline_ssim) * 10
        """
        try:
            # SSIM improvement over baseline
            quality_improvement = result.quality_score - baseline_result.quality_score

            # Exponential scaling for high-quality improvements
            if quality_improvement > 0:
                quality_reward = quality_improvement * self.quality_scale
                # Bonus for significant improvements
                if quality_improvement > 0.1:  # >10% improvement
                    quality_reward *= 1.5
                # Extra bonus for exceptional improvements
                if quality_improvement > 0.2:  # >20% improvement
                    quality_reward *= 2.0
            else:
                # Penalty for quality degradation (less severe to allow exploration)
                quality_reward = quality_improvement * self.quality_scale * 0.5

            return quality_reward

        except Exception as e:
            self.logger.error(f"Quality reward calculation failed: {e}")
            return 0.0

    def _calculate_speed_reward(self, result: ConversionResult,
                               baseline_result: ConversionResult) -> float:
        """
        Calculate speed efficiency reward
        Formula: speed_reward = max(0, (baseline_time - current_time) / baseline_time)
        """
        try:
            if baseline_result.processing_time <= 0:
                return 0.0

            # Speed improvement ratio
            time_improvement = (baseline_result.processing_time - result.processing_time) / baseline_result.processing_time

            # Reward faster conversions
            speed_reward = max(0.0, time_improvement)

            # Penalty for excessive processing time (>2x baseline)
            if result.processing_time > baseline_result.processing_time * 2:
                speed_reward = -0.5  # Moderate penalty

            # Cap the reward to prevent excessive emphasis on speed
            speed_reward = min(speed_reward, 1.0)

            return speed_reward

        except Exception as e:
            self.logger.error(f"Speed reward calculation failed: {e}")
            return 0.0

    def _calculate_size_reward(self, result: ConversionResult,
                              baseline_result: ConversionResult) -> float:
        """
        Calculate file size optimization reward
        Formula: size_reward = max(0, (baseline_size - current_size) / baseline_size)
        """
        try:
            if baseline_result.file_size <= 0:
                return 0.0

            # File size reduction ratio
            size_improvement = (baseline_result.file_size - result.file_size) / baseline_result.file_size

            # Reward smaller file sizes but balance with quality
            size_reward = max(0.0, size_improvement)

            # Prevent over-compression penalty - if size reduction is too aggressive
            # and quality suffers, the quality component will penalize it

            # Bonus for significant size reduction (>20%) with maintained quality
            if size_improvement > 0.2 and result.quality_score >= baseline_result.quality_score:
                size_reward *= 1.3

            # Cap the reward to prevent excessive compression
            size_reward = min(size_reward, 1.0)

            return size_reward

        except Exception as e:
            self.logger.error(f"Size reward calculation failed: {e}")
            return 0.0

    def _calculate_target_bonus(self, result: ConversionResult,
                               step: int, max_steps: int) -> float:
        """
        Calculate target achievement bonus
        - Large bonus for reaching quality target
        - Progressive bonus for approaching target
        - Early termination reward for efficiency
        """
        try:
            target_bonus = 0.0

            # Large bonus for reaching quality target
            if result.quality_score >= self.target_quality:
                target_bonus = self.target_bonus_scale

                # Early termination bonus - reward efficiency
                efficiency_bonus = (max_steps - step) / max_steps
                target_bonus += efficiency_bonus * 20.0

            # Progressive bonus for approaching target (only if quality is improving)
            elif result.quality_score >= self.target_quality * 0.8 and result.quality_score >= baseline_result.quality_score:  # Within 80% of target AND improving
                progress = (result.quality_score - self.target_quality * 0.8) / (self.target_quality * 0.2)
                target_bonus = progress * self.target_bonus_scale * 0.3

            return target_bonus

        except Exception as e:
            self.logger.error(f"Target bonus calculation failed: {e}")
            return 0.0

    def _calculate_convergence_reward(self, result: ConversionResult, step: int) -> float:
        """
        Calculate convergence encouragement reward
        - Reward consistent improvements
        - Penalty for parameter oscillation
        """
        try:
            convergence_reward = 0.0

            # Track quality improvements
            self.convergence_history.append(result.quality_score)

            # Keep only recent history
            if len(self.convergence_history) > self.convergence_window:
                self.convergence_history = self.convergence_history[-self.convergence_window:]

            # Calculate trend if we have enough data
            if len(self.convergence_history) >= 3:
                recent_qualities = np.array(self.convergence_history[-3:])

                # Check for consistent improvement trend
                if np.all(np.diff(recent_qualities) >= 0):  # Non-decreasing trend
                    convergence_reward = 2.0  # Reward consistent improvement
                elif np.all(np.diff(recent_qualities) <= 0):  # Decreasing trend
                    convergence_reward = -1.0  # Penalty for consistent degradation
                else:
                    # Check for oscillation (high variance)
                    if len(self.convergence_history) >= self.convergence_window:
                        variance = np.var(self.convergence_history)
                        if variance > 0.01:  # High oscillation
                            convergence_reward = -variance * self.oscillation_penalty_scale

            return convergence_reward

        except Exception as e:
            self.logger.error(f"Convergence reward calculation failed: {e}")
            return 0.0

    def _calculate_step_penalty(self, step: int, max_steps: int) -> float:
        """
        Calculate step penalty to encourage efficiency
        Progressive reward scaling with episode length
        """
        try:
            # Gentle penalty for longer episodes to encourage efficiency
            step_ratio = step / max_steps
            step_penalty = -step_ratio * 2.0  # Gentle penalty

            return step_penalty

        except Exception as e:
            self.logger.error(f"Step penalty calculation failed: {e}")
            return 0.0

    def _normalize_reward(self, reward: float, step: int, max_steps: int) -> float:
        """
        Normalize and clip reward to prevent gradient explosion
        Handle extreme values gracefully
        """
        try:
            # Clip extreme rewards
            reward = np.clip(reward, -500.0, 1000.0)

            # Apply gentle scaling based on episode progress
            progress_scale = 0.5 + 0.5 * (step / max_steps)  # Scale from 0.5 to 1.0
            reward = reward * progress_scale

            return float(reward)

        except Exception as e:
            self.logger.error(f"Reward normalization failed: {e}")
            return 0.0

    def _log_reward_components(self, total_reward: float,
                              components: Dict[str, float], step: int):
        """Log reward components for debugging and analysis"""
        try:
            # Store reward history
            reward_entry = {
                'step': step,
                'total_reward': total_reward,
                'components': components.copy(),
                'timestamp': time.time()
            }
            self.reward_history.append(reward_entry)

            # Limit history size
            if len(self.reward_history) > 1000:
                self.reward_history = self.reward_history[-500:]

            # Log detailed breakdown (debug level)
            self.logger.debug(f"Step {step}: Total reward = {total_reward:.3f}")
            for component, value in components.items():
                if abs(value) > 0.001:  # Only log significant components
                    self.logger.debug(f"  {component}: {value:.3f}")

        except Exception as e:
            self.logger.error(f"Reward logging failed: {e}")

    def configure(self, **kwargs):
        """
        Configure reward function parameters
        - Support different objective priorities
        - Allow runtime reward weight adjustment
        """
        try:
            updated = False

            if 'quality_weight' in kwargs:
                self.quality_weight = float(kwargs['quality_weight'])
                updated = True

            if 'speed_weight' in kwargs:
                self.speed_weight = float(kwargs['speed_weight'])
                updated = True

            if 'size_weight' in kwargs:
                self.size_weight = float(kwargs['size_weight'])
                updated = True

            if 'target_quality' in kwargs:
                self.target_quality = float(kwargs['target_quality'])
                updated = True

            if 'reward_scale' in kwargs:
                self.reward_scale = float(kwargs['reward_scale'])
                updated = True

            # Renormalize weights
            if updated:
                total_weight = self.quality_weight + self.speed_weight + self.size_weight
                if total_weight > 0:
                    self.quality_weight /= total_weight
                    self.speed_weight /= total_weight
                    self.size_weight /= total_weight

                self.logger.info(f"Reward function reconfigured: "
                                f"quality={self.quality_weight:.2f}, "
                                f"speed={self.speed_weight:.2f}, "
                                f"size={self.size_weight:.2f}")

        except Exception as e:
            self.logger.error(f"Reward function configuration failed: {e}")

    def get_reward_statistics(self) -> Dict[str, Any]:
        """
        Generate reward function analytics
        - Track reward distribution statistics
        - Monitor reward function effectiveness
        """
        try:
            if not self.reward_history:
                return {'error': 'No reward history available'}

            rewards = [entry['total_reward'] for entry in self.reward_history]

            # Component statistics
            component_stats = {}
            for component in ['quality_reward', 'speed_reward', 'size_reward', 'target_bonus']:
                values = [entry['components'].get(component, 0.0) for entry in self.reward_history]
                component_stats[component] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

            return {
                'total_entries': len(self.reward_history),
                'reward_mean': np.mean(rewards),
                'reward_std': np.std(rewards),
                'reward_min': np.min(rewards),
                'reward_max': np.max(rewards),
                'component_stats': component_stats,
                'current_weights': {
                    'quality': self.quality_weight,
                    'speed': self.speed_weight,
                    'size': self.size_weight
                }
            }

        except Exception as e:
            self.logger.error(f"Reward statistics generation failed: {e}")
            return {'error': str(e)}

    def reset_convergence_history(self):
        """Reset convergence tracking for new episode"""
        self.convergence_history = []

    def create_reward_validation_report(self, test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create comprehensive reward function validation report
        Test with known good/bad parameter sets
        """
        try:
            validation_results = []

            for scenario in test_scenarios:
                result = ConversionResult(**scenario['result'])
                baseline = ConversionResult(**scenario['baseline'])

                reward, components = self.calculate_reward(
                    result, baseline, scenario.get('step', 1), scenario.get('max_steps', 50)
                )

                validation_results.append({
                    'scenario_name': scenario.get('name', 'unknown'),
                    'expected_positive': scenario.get('expected_positive', True),
                    'reward': reward,
                    'components': components,
                    'passed': (reward > 0) == scenario.get('expected_positive', True)
                })

            # Summary statistics
            passed_tests = sum(1 for r in validation_results if r['passed'])
            total_tests = len(validation_results)

            return {
                'validation_results': validation_results,
                'tests_passed': passed_tests,
                'tests_total': total_tests,
                'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
                'timestamp': time.time()
            }

        except Exception as e:
            self.logger.error(f"Reward validation report generation failed: {e}")
            return {'error': str(e)}


class AdaptiveRewardWeighting:
    """
    Adaptive reward weighting system
    Adjust weights based on episode progress and performance
    """

    def __init__(self, reward_function: MultiObjectiveRewardFunction):
        self.reward_function = reward_function
        self.original_weights = {
            'quality': reward_function.quality_weight,
            'speed': reward_function.speed_weight,
            'size': reward_function.size_weight
        }
        self.adaptation_history = []
        self.logger = logging.getLogger(__name__)

    def adapt_weights(self, episode_progress: float, quality_progress: float,
                     performance_metrics: Dict[str, float]):
        """
        Adapt reward weights based on episode progress
        - Increase quality weight as target approaches
        - Reduce speed weight for high-quality requirements
        """
        try:
            # Calculate adaptive weights
            new_weights = self.original_weights.copy()

            # Increase quality weight as episode progresses and quality improves
            if quality_progress > 0.5:  # If making good quality progress
                quality_boost = min(0.2, quality_progress - 0.5)
                new_weights['quality'] += quality_boost
                new_weights['speed'] -= quality_boost * 0.5
                new_weights['size'] -= quality_boost * 0.5

            # If speed is becoming an issue, increase speed weight
            if performance_metrics.get('avg_processing_time', 0) > performance_metrics.get('baseline_time', 1) * 2:
                speed_boost = 0.1
                new_weights['speed'] += speed_boost
                new_weights['quality'] -= speed_boost * 0.5
                new_weights['size'] -= speed_boost * 0.5

            # Ensure weights remain positive and sum to 1
            for key in new_weights:
                new_weights[key] = max(0.05, new_weights[key])  # Minimum 5%

            total_weight = sum(new_weights.values())
            for key in new_weights:
                new_weights[key] /= total_weight

            # Apply new weights
            self.reward_function.configure(**{f"{k}_weight": v for k, v in new_weights.items()})

            # Track adaptation
            self.adaptation_history.append({
                'episode_progress': episode_progress,
                'weights': new_weights.copy(),
                'timestamp': time.time()
            })

            self.logger.info(f"Reward weights adapted: {new_weights}")

        except Exception as e:
            self.logger.error(f"Reward weight adaptation failed: {e}")

    def reset_weights(self):
        """Reset weights to original values"""
        try:
            self.reward_function.configure(**{f"{k}_weight": v for k, v in self.original_weights.items()})
            self.logger.info("Reward weights reset to original values")
        except Exception as e:
            self.logger.error(f"Weight reset failed: {e}")


# Factory function for creating reward functions with different configurations
def create_reward_function(config: str = "balanced") -> MultiObjectiveRewardFunction:
    """
    Create reward function with predefined configuration

    Args:
        config: Configuration name ("balanced", "quality_focused", "speed_focused", "size_focused")

    Returns:
        Configured MultiObjectiveRewardFunction
    """
    configs = {
        "balanced": {"quality_weight": 0.6, "speed_weight": 0.3, "size_weight": 0.1},
        "quality_focused": {"quality_weight": 0.8, "speed_weight": 0.15, "size_weight": 0.05},
        "speed_focused": {"quality_weight": 0.4, "speed_weight": 0.5, "size_weight": 0.1},
        "size_focused": {"quality_weight": 0.5, "speed_weight": 0.2, "size_weight": 0.3}
    }

    if config not in configs:
        config = "balanced"

    return MultiObjectiveRewardFunction(**configs[config])