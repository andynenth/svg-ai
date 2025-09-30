# backend/ai_modules/optimization/training_pipeline.py
"""Curriculum-based training pipeline for PPO agent"""

import numpy as np
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from .ppo_optimizer import PPOVTracerOptimizer
from .vtracer_env import VTracerOptimizationEnv
from .real_time_monitor import RealTimeMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingStage:
    """Training curriculum stage configuration"""
    name: str
    image_types: List[str]  # ['simple', 'text', 'gradient', 'complex']
    difficulty: float  # 0.0-1.0
    target_quality: float
    max_episodes: int
    success_threshold: float  # Percentage of successful episodes
    reward_weights: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.reward_weights is None:
            # Default reward weights that adapt to difficulty
            self.reward_weights = {
                'quality': max(0.5, 0.8 - self.difficulty * 0.3),  # Decrease emphasis on quality as difficulty increases
                'speed': min(0.4, 0.2 + self.difficulty * 0.2),     # Increase emphasis on speed
                'size': min(0.3, 0.1 + self.difficulty * 0.2)      # Increase emphasis on size
            }


@dataclass
class StageResult:
    """Results from completing a training stage"""
    stage_name: str
    success: bool
    episodes_completed: int
    average_quality: float
    success_rate: float
    training_time: float
    best_quality: float
    convergence_episodes: int
    stage_metrics: Dict[str, Any]


class CurriculumTrainingPipeline:
    """Curriculum-based training pipeline for PPO agent"""

    def __init__(self,
                 training_images: Dict[str, List[str]],
                 model_config: Optional[Dict] = None,
                 curriculum_config: Optional[Dict] = None,
                 save_dir: str = "models/curriculum_training",
                 enable_real_time_monitoring: bool = True):
        """
        Initialize curriculum training pipeline

        Args:
            training_images: Dict mapping image types to lists of image paths
            model_config: PPO model configuration
            curriculum_config: Curriculum configuration overrides
            save_dir: Directory to save models and results
            enable_real_time_monitoring: Enable real-time monitoring
        """
        self.training_images = training_images
        self.model_config = model_config or self._default_model_config()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.enable_real_time_monitoring = enable_real_time_monitoring

        # Define curriculum stages
        self.curriculum_stages = self._define_curriculum(curriculum_config)
        self.current_stage = 0

        # Training tracking
        self.training_log = []
        self.stage_results = {}
        self.curriculum_metrics = defaultdict(list)

        # Current optimizer
        self.optimizer = None

        # Real-time monitoring
        self.real_time_monitor = None
        if self.enable_real_time_monitoring:
            self.real_time_monitor = RealTimeMonitor(
                websocket_port=8766,  # Different port from individual optimizer
                save_dir=str(self.save_dir / "real_time_monitoring")
            )

        logger.info("Curriculum Training Pipeline initialized")
        logger.info(f"Training images: {[(k, len(v)) for k, v in training_images.items()]}")
        logger.info(f"Curriculum stages: {len(self.curriculum_stages)}")
        logger.info(f"Real-time monitoring: {'enabled' if enable_real_time_monitoring else 'disabled'}")

    def _default_model_config(self) -> Dict[str, Any]:
        """Default model configuration for curriculum training"""
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
            'verbose': 1
        }

    def _define_curriculum(self, config_overrides: Optional[Dict] = None) -> List[TrainingStage]:
        """Define progressive training curriculum"""

        # Default curriculum stages
        default_stages = [
            TrainingStage(
                name="simple_warmup",
                image_types=["simple"],
                difficulty=0.1,
                target_quality=0.75,
                max_episodes=5000,
                success_threshold=0.80
            ),
            TrainingStage(
                name="text_introduction",
                image_types=["simple", "text"],
                difficulty=0.3,
                target_quality=0.80,
                max_episodes=8000,
                success_threshold=0.75
            ),
            TrainingStage(
                name="gradient_challenge",
                image_types=["simple", "text", "gradient"],
                difficulty=0.6,
                target_quality=0.85,
                max_episodes=10000,
                success_threshold=0.70
            ),
            TrainingStage(
                name="complex_mastery",
                image_types=["simple", "text", "gradient", "complex"],
                difficulty=1.0,
                target_quality=0.90,
                max_episodes=15000,
                success_threshold=0.65
            )
        ]

        # Apply configuration overrides if provided
        if config_overrides:
            for i, stage in enumerate(default_stages):
                if f"stage_{i}" in config_overrides:
                    stage_config = config_overrides[f"stage_{i}"]
                    for key, value in stage_config.items():
                        if hasattr(stage, key):
                            setattr(stage, key, value)

        return default_stages

    def _select_training_images(self, stage: TrainingStage, num_images: int = 5) -> List[str]:
        """Select training images for a curriculum stage"""
        selected_images = []

        for image_type in stage.image_types:
            if image_type in self.training_images:
                type_images = self.training_images[image_type]
                # Select images based on stage difficulty
                num_per_type = max(1, num_images // len(stage.image_types))

                if len(type_images) >= num_per_type:
                    # For higher difficulty, prefer more challenging images
                    if stage.difficulty > 0.5:
                        # Take images from the latter half (presumably more complex)
                        start_idx = len(type_images) // 2
                        selected = np.random.choice(
                            type_images[start_idx:],
                            size=min(num_per_type, len(type_images) - start_idx),
                            replace=False
                        ).tolist()
                    else:
                        # Take images from the first half (presumably simpler)
                        end_idx = max(num_per_type, len(type_images) // 2)
                        selected = np.random.choice(
                            type_images[:end_idx],
                            size=min(num_per_type, end_idx),
                            replace=False
                        ).tolist()
                else:
                    selected = type_images

                selected_images.extend(selected)

        return selected_images

    def _create_stage_optimizer(self, stage: TrainingStage, training_image: str) -> PPOVTracerOptimizer:
        """Create optimizer configured for specific curriculum stage"""

        # Environment configuration adapted to stage
        env_kwargs = {
            'image_path': training_image,
            'target_quality': stage.target_quality,
            'max_steps': min(50, int(20 + stage.difficulty * 30))  # Increase max steps with difficulty
        }

        # Training configuration adapted to stage
        training_config = {
            'total_timesteps': stage.max_episodes,
            'eval_freq': max(1000, stage.max_episodes // 10),
            'n_eval_episodes': 5,
            'deterministic_eval': True,
            'n_envs': min(4, max(1, int(2 + stage.difficulty * 2))),  # More envs for harder stages
            'model_save_path': str(self.save_dir / f"stage_{stage.name}"),
            'checkpoint_freq': stage.max_episodes // 4
        }

        # Model configuration adapted to stage
        stage_model_config = self.model_config.copy()

        # Adjust learning rate based on stage difficulty
        stage_model_config['learning_rate'] = self.model_config['learning_rate'] * (1.0 - stage.difficulty * 0.3)

        # Adjust exploration based on stage
        stage_model_config['ent_coef'] = self.model_config['ent_coef'] * (1.0 + stage.difficulty * 0.5)

        return PPOVTracerOptimizer(
            env_kwargs=env_kwargs,
            model_config=stage_model_config,
            training_config=training_config,
            enable_real_time_monitoring=False  # Pipeline handles monitoring
        )

    def _evaluate_stage_performance(self, stage: TrainingStage,
                                  optimizer: PPOVTracerOptimizer) -> StageResult:
        """Evaluate performance on current stage"""
        logger.info(f"Evaluating stage: {stage.name}")

        # Get training images for evaluation
        eval_images = self._select_training_images(stage, num_images=10)

        total_episodes = 0
        total_quality = 0.0
        successful_episodes = 0
        best_quality = 0.0
        stage_start_time = time.time()

        # Evaluate on multiple images from this stage
        for eval_image in eval_images[:5]:  # Limit to 5 for efficiency
            try:
                result = optimizer.optimize_parameters(eval_image, max_episodes=3)

                total_episodes += result['episodes_run']
                total_quality += result['best_quality']
                if result['target_reached']:
                    successful_episodes += 1
                best_quality = max(best_quality, result['best_quality'])

            except Exception as e:
                logger.warning(f"Evaluation failed for {eval_image}: {e}")

        # Calculate metrics
        avg_quality = total_quality / len(eval_images) if eval_images else 0.0
        success_rate = successful_episodes / len(eval_images) if eval_images else 0.0
        evaluation_time = time.time() - stage_start_time

        # Determine if stage passed
        stage_success = success_rate >= stage.success_threshold and avg_quality >= stage.target_quality

        stage_metrics = {
            'eval_images_count': len(eval_images),
            'target_quality': stage.target_quality,
            'success_threshold': stage.success_threshold,
            'difficulty': stage.difficulty,
            'image_types': stage.image_types
        }

        return StageResult(
            stage_name=stage.name,
            success=stage_success,
            episodes_completed=total_episodes,
            average_quality=avg_quality,
            success_rate=success_rate,
            training_time=evaluation_time,
            best_quality=best_quality,
            convergence_episodes=total_episodes,
            stage_metrics=stage_metrics
        )

    def _should_advance_stage(self, stage_result: StageResult) -> bool:
        """Determine if should advance to next curriculum stage"""
        return stage_result.success

    def _advance_to_next_stage(self) -> bool:
        """Advance to the next curriculum stage"""
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            logger.info(f"Advanced to stage {self.current_stage}: {self.curriculum_stages[self.current_stage].name}")
            return True
        return False

    def _is_curriculum_complete(self) -> bool:
        """Check if the curriculum has been completed"""
        return self.current_stage >= len(self.curriculum_stages) - 1

    def _should_repeat_stage(self, stage_result: StageResult, attempt: int) -> bool:
        """Determine if should repeat current stage"""
        # Allow up to 3 attempts per stage
        if attempt >= 3:
            logger.warning(f"Stage {stage_result.stage_name} failed after {attempt} attempts, advancing anyway")
            return False

        # Repeat if performance is significantly below threshold
        if stage_result.success_rate < stage_result.stage_metrics['success_threshold'] * 0.5:
            return True

        return False

    def _adjust_stage_difficulty(self, stage: TrainingStage, stage_result: StageResult) -> TrainingStage:
        """Dynamically adjust stage difficulty based on performance"""

        if stage_result.success_rate > stage.success_threshold * 1.2:
            # Performance is excellent, can increase difficulty
            logger.info(f"Increasing difficulty for stage {stage.name}")
            stage.target_quality = min(0.95, stage.target_quality + 0.02)
            stage.success_threshold = min(0.90, stage.success_threshold + 0.05)

        elif stage_result.success_rate < stage.success_threshold * 0.7:
            # Performance is poor, reduce difficulty
            logger.info(f"Reducing difficulty for stage {stage.name}")
            stage.target_quality = max(0.60, stage.target_quality - 0.02)
            stage.success_threshold = max(0.50, stage.success_threshold - 0.05)

        return stage

    def train_stage(self, stage_idx: int, attempt: int = 1) -> StageResult:
        """Train a single curriculum stage"""
        stage = self.curriculum_stages[stage_idx]
        logger.info(f"Training stage {stage_idx + 1}/{len(self.curriculum_stages)}: {stage.name} (attempt {attempt})")

        # Select training images for this stage
        training_images = self._select_training_images(stage, num_images=3)

        if not training_images:
            raise ValueError(f"No training images available for stage {stage.name}")

        # Use first image as primary training image
        primary_image = training_images[0]
        logger.info(f"Primary training image: {primary_image}")

        # Create stage-specific optimizer
        stage_start_time = time.time()
        self.optimizer = self._create_stage_optimizer(stage, primary_image)

        try:
            # Train the agent
            logger.info(f"Starting training for stage {stage.name}...")
            training_result = self.optimizer.train()

            # Evaluate stage performance
            stage_result = self._evaluate_stage_performance(stage, self.optimizer)
            stage_result.training_time = time.time() - stage_start_time

            # Update curriculum metrics
            self.curriculum_metrics['stage_names'].append(stage.name)
            self.curriculum_metrics['stage_qualities'].append(stage_result.average_quality)
            self.curriculum_metrics['stage_success_rates'].append(stage_result.success_rate)
            self.curriculum_metrics['stage_difficulties'].append(stage.difficulty)

            # Log stage completion
            logger.info(f"Stage {stage.name} completed:")
            logger.info(f"  Success: {stage_result.success}")
            logger.info(f"  Average Quality: {stage_result.average_quality:.4f}")
            logger.info(f"  Success Rate: {stage_result.success_rate:.2%}")
            logger.info(f"  Training Time: {stage_result.training_time:.2f}s")

            # Save stage results
            self.stage_results[stage.name] = stage_result
            self._save_stage_results(stage_result)

            return stage_result

        except Exception as e:
            logger.error(f"Training failed for stage {stage.name}: {e}")
            # Return failed result
            return StageResult(
                stage_name=stage.name,
                success=False,
                episodes_completed=0,
                average_quality=0.0,
                success_rate=0.0,
                training_time=time.time() - stage_start_time,
                best_quality=0.0,
                convergence_episodes=0,
                stage_metrics={'error': str(e)}
            )
        finally:
            if self.optimizer:
                self.optimizer.close()

    def run_curriculum(self) -> Dict[str, Any]:
        """Run complete curriculum training"""
        logger.info("🚀 Starting Curriculum Training")
        logger.info(f"Total stages: {len(self.curriculum_stages)}")

        curriculum_start_time = time.time()
        successful_stages = 0

        stage_idx = 0
        while stage_idx < len(self.curriculum_stages):
            stage = self.curriculum_stages[stage_idx]

            # Train current stage (with retries)
            attempt = 1
            stage_result = None

            while attempt <= 3:  # Max 3 attempts per stage
                stage_result = self.train_stage(stage_idx, attempt)

                if self._should_advance_stage(stage_result):
                    logger.info(f"✅ Stage {stage.name} passed on attempt {attempt}")
                    successful_stages += 1
                    break
                elif self._should_repeat_stage(stage_result, attempt):
                    logger.info(f"🔄 Repeating stage {stage.name} (attempt {attempt + 1})")
                    # Adjust stage difficulty
                    self.curriculum_stages[stage_idx] = self._adjust_stage_difficulty(stage, stage_result)
                    attempt += 1
                else:
                    logger.warning(f"⚠️ Stage {stage.name} failed, advancing anyway")
                    break

            # Move to next stage
            stage_idx += 1

        # Calculate final metrics
        total_time = time.time() - curriculum_start_time

        curriculum_results = {
            'total_stages': len(self.curriculum_stages),
            'successful_stages': successful_stages,
            'success_rate': successful_stages / len(self.curriculum_stages),
            'total_training_time': total_time,
            'stage_results': {name: asdict(result) for name, result in self.stage_results.items()},
            'curriculum_metrics': dict(self.curriculum_metrics),
            'final_performance': self._calculate_final_performance()
        }

        # Save curriculum results
        self._save_curriculum_results(curriculum_results)

        logger.info(f"🎉 Curriculum Training Complete!")
        logger.info(f"Success Rate: {curriculum_results['success_rate']:.2%}")
        logger.info(f"Total Time: {total_time:.2f}s")

        return curriculum_results

    def _calculate_final_performance(self) -> Dict[str, float]:
        """Calculate final curriculum performance metrics"""
        if not self.stage_results:
            return {'average_quality': 0.0, 'overall_success_rate': 0.0}

        qualities = [result.average_quality for result in self.stage_results.values()]
        success_rates = [result.success_rate for result in self.stage_results.values()]

        return {
            'average_quality': np.mean(qualities),
            'overall_success_rate': np.mean(success_rates),
            'quality_std': np.std(qualities),
            'best_stage_quality': max(qualities) if qualities else 0.0,
            'worst_stage_quality': min(qualities) if qualities else 0.0
        }

    def _save_stage_results(self, stage_result: StageResult) -> None:
        """Save results from individual stage"""
        stage_file = self.save_dir / f"stage_{stage_result.stage_name}_results.json"
        with open(stage_file, 'w') as f:
            json.dump(asdict(stage_result), f, indent=2)

    def _save_curriculum_results(self, results: Dict[str, Any]) -> None:
        """Save complete curriculum results"""
        results_file = self.save_dir / "curriculum_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Curriculum results saved to: {results_file}")

    def visualize_curriculum_progress(self, save_path: Optional[str] = None) -> None:
        """Create visualizations of curriculum training progress"""
        if not self.curriculum_metrics['stage_names']:
            logger.warning("No curriculum metrics available for visualization")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        stages = self.curriculum_metrics['stage_names']
        qualities = self.curriculum_metrics['stage_qualities']
        success_rates = self.curriculum_metrics['stage_success_rates']
        difficulties = self.curriculum_metrics['stage_difficulties']

        # Plot 1: Quality progression
        ax1.plot(range(len(stages)), qualities, 'b-o', linewidth=2, markersize=8)
        ax1.set_title('Quality Progression Across Stages', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Curriculum Stage')
        ax1.set_ylabel('Average Quality (SSIM)')
        ax1.set_xticks(range(len(stages)))
        ax1.set_xticklabels(stages, rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Plot 2: Success rate progression
        ax2.plot(range(len(stages)), success_rates, 'g-o', linewidth=2, markersize=8)
        ax2.set_title('Success Rate Progression', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Curriculum Stage')
        ax2.set_ylabel('Success Rate')
        ax2.set_xticks(range(len(stages)))
        ax2.set_xticklabels(stages, rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # Plot 3: Difficulty vs Performance
        ax3.scatter(difficulties, qualities, c=success_rates, cmap='viridis', s=100, alpha=0.8)
        ax3.set_title('Difficulty vs Quality (colored by Success Rate)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Stage Difficulty')
        ax3.set_ylabel('Average Quality')
        cbar = plt.colorbar(ax3.scatter(difficulties, qualities, c=success_rates, cmap='viridis', s=100), ax=ax3)
        cbar.set_label('Success Rate')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Combined metrics
        x = np.arange(len(stages))
        width = 0.35
        ax4.bar(x - width/2, qualities, width, label='Quality', alpha=0.8)
        ax4.bar(x + width/2, success_rates, width, label='Success Rate', alpha=0.8)
        ax4.set_title('Stage Performance Summary', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Curriculum Stage')
        ax4.set_ylabel('Performance Metric')
        ax4.set_xticks(x)
        ax4.set_xticklabels(stages, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Curriculum visualization saved to: {save_path}")
        else:
            plt.savefig(self.save_dir / "curriculum_progress.png", dpi=300, bbox_inches='tight')
            logger.info(f"Curriculum visualization saved to: {self.save_dir / 'curriculum_progress.png'}")

        plt.close()

    def generate_curriculum_report(self) -> str:
        """Generate comprehensive curriculum training report"""
        if not self.stage_results:
            return "No curriculum results available"

        report = []
        report.append("# Curriculum Training Report")
        report.append("=" * 50)
        report.append("")

        # Overall summary
        final_perf = self._calculate_final_performance()
        report.append("## Overall Performance")
        report.append(f"- Average Quality: {final_perf['average_quality']:.4f}")
        report.append(f"- Overall Success Rate: {final_perf['overall_success_rate']:.2%}")
        report.append(f"- Quality Standard Deviation: {final_perf['quality_std']:.4f}")
        report.append(f"- Best Stage Quality: {final_perf['best_stage_quality']:.4f}")
        report.append("")

        # Stage-by-stage breakdown
        report.append("## Stage Results")
        for i, (stage_name, result) in enumerate(self.stage_results.items()):
            report.append(f"### Stage {i+1}: {stage_name}")
            report.append(f"- Success: {'✅' if result.success else '❌'}")
            report.append(f"- Average Quality: {result.average_quality:.4f}")
            report.append(f"- Success Rate: {result.success_rate:.2%}")
            report.append(f"- Best Quality: {result.best_quality:.4f}")
            report.append(f"- Training Time: {result.training_time:.2f}s")
            report.append(f"- Episodes: {result.episodes_completed}")
            report.append("")

        return "\n".join(report)

    async def start_monitoring(self):
        """Start real-time monitoring"""
        if self.real_time_monitor:
            await self.real_time_monitor.start_monitoring()
            logger.info("Pipeline real-time monitoring started")

    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.real_time_monitor:
            await self.real_time_monitor.stop_monitoring()
            logger.info("Pipeline real-time monitoring stopped")

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save training pipeline checkpoint"""
        checkpoint_data = {
            'current_stage': self.current_stage,
            'stage_results': {name: asdict(result) for name, result in self.stage_results.items()},
            'training_log': self.training_log,
            'model_config': self.model_config,
            'training_images': self.training_images,
            'timestamp': time.time()
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training pipeline checkpoint"""
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)

        self.current_stage = checkpoint_data['current_stage']
        self.training_log = checkpoint_data['training_log']

        # Reconstruct stage results
        self.stage_results = {}
        for name, result_dict in checkpoint_data['stage_results'].items():
            self.stage_results[name] = StageResult(**result_dict)

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def save_model(self, model_path: str) -> None:
        """Save trained model"""
        if self.optimizer and hasattr(self.optimizer, 'save_model'):
            self.optimizer.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
        else:
            logger.warning("No trained model available to save")

    def load_model(self, model_path: str):
        """Load trained model"""
        if self.optimizer and hasattr(self.optimizer, 'load_model'):
            return self.optimizer.load_model(model_path)
        else:
            logger.warning("No optimizer available to load model")
            return None

    def close(self) -> None:
        """Clean up resources"""
        if self.optimizer:
            self.optimizer.close()

        # Save monitoring data
        if self.real_time_monitor:
            self.real_time_monitor.save_monitoring_data()

        logger.info("Curriculum Training Pipeline closed")


# Factory function for easy usage
def create_curriculum_pipeline(training_images: Dict[str, List[str]],
                             model_config: Optional[Dict] = None,
                             save_dir: str = "models/curriculum") -> CurriculumTrainingPipeline:
    """
    Factory function to create curriculum training pipeline

    Args:
        training_images: Dict mapping image types to image paths
        model_config: PPO model configuration
        save_dir: Directory to save models and results

    Returns:
        Configured curriculum training pipeline
    """
    return CurriculumTrainingPipeline(
        training_images=training_images,
        model_config=model_config,
        save_dir=save_dir
    )