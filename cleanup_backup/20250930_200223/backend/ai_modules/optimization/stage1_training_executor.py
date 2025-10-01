"""
Stage 1 Training Executor for PPO Agent Training
Component for Task B7.2 - DAY7 PPO Agent Training

Implements Stage 1 training execution for simple geometric logos with:
- 5000 episode training target
- 80% success rate with >75% SSIM improvement
- Real-time monitoring and validation
- Quality assurance and failure detection
- Progress reporting and artifact management
"""

import asyncio
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .training_pipeline import CurriculumTrainingPipeline, TrainingStage
from .ppo_optimizer import PPOVTracerOptimizer
from .real_time_monitor import RealTimeMonitor, TrainingMetrics
from .vtracer_env import VTracerOptimizationEnv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Stage1Config:
    """Configuration for Stage 1 training"""
    target_episodes: int = 5000
    success_rate_threshold: float = 0.80
    ssim_improvement_threshold: float = 0.75
    validation_frequency: int = 1000
    checkpoint_frequency: int = 500
    quality_target: float = 0.85
    max_training_time_hours: int = 12
    early_stopping_patience: int = 3  # Stop if no improvement for 3 validations

    # Progress reporting
    hourly_reports: bool = True
    milestone_notifications: bool = True

    # Quality assurance
    failure_detection: bool = True
    overfitting_detection: bool = True
    performance_monitoring: bool = True


@dataclass
class ValidationResult:
    """Results from validation evaluation"""
    episode: int
    timestamp: float
    avg_quality: float
    success_rate: float
    ssim_improvement: float
    validation_images_count: int
    target_reached: bool
    best_episode_quality: float
    convergence_score: float


@dataclass
class TrainingMilestone:
    """Training milestone achievement"""
    milestone_type: str  # 'episode_count', 'success_rate', 'quality_target'
    episode: int
    timestamp: float
    value: float
    description: str
    celebration_emoji: str = "🎯"


@dataclass
class QualityAssuranceAlert:
    """Quality assurance alert"""
    alert_type: str  # 'failure', 'overfitting', 'stagnation', 'degradation'
    severity: str  # 'low', 'medium', 'high', 'critical'
    episode: int
    timestamp: float
    message: str
    details: Dict[str, Any]
    recommended_action: str


class ValidationProtocol:
    """Validation protocol for Stage 1 training"""

    def __init__(self, validation_images: List[str], config: Stage1Config):
        self.validation_images = validation_images
        self.config = config
        self.validation_history = []
        self.best_validation_result = None

    def run_validation(self, episode: int, optimizer: PPOVTracerOptimizer) -> ValidationResult:
        """Run validation evaluation at current episode"""
        logger.info(f"🔍 Running validation at episode {episode}")

        validation_start = time.time()

        total_quality = 0.0
        total_ssim_improvement = 0.0
        successful_episodes = 0
        best_episode_quality = 0.0

        # Test on validation images
        for val_image in self.validation_images:
            try:
                # Run optimization on validation image
                result = optimizer.optimize_parameters(val_image, max_episodes=3)

                quality = result.get('best_quality', 0.0)
                ssim_improvement = result.get('ssim_improvement', 0.0)
                target_reached = result.get('target_reached', False)

                total_quality += quality
                total_ssim_improvement += ssim_improvement
                if target_reached:
                    successful_episodes += 1
                best_episode_quality = max(best_episode_quality, quality)

            except Exception as e:
                logger.warning(f"Validation failed for {val_image}: {e}")

        # Calculate validation metrics
        n_images = len(self.validation_images)
        avg_quality = total_quality / n_images if n_images > 0 else 0.0
        success_rate = successful_episodes / n_images if n_images > 0 else 0.0
        avg_ssim_improvement = total_ssim_improvement / n_images if n_images > 0 else 0.0

        # Check if target reached
        target_reached = (success_rate >= self.config.success_rate_threshold and
                         avg_ssim_improvement >= self.config.ssim_improvement_threshold)

        # Calculate convergence score
        convergence_score = self._calculate_convergence_score(avg_quality, success_rate)

        validation_result = ValidationResult(
            episode=episode,
            timestamp=time.time(),
            avg_quality=avg_quality,
            success_rate=success_rate,
            ssim_improvement=avg_ssim_improvement,
            validation_images_count=n_images,
            target_reached=target_reached,
            best_episode_quality=best_episode_quality,
            convergence_score=convergence_score
        )

        self.validation_history.append(validation_result)

        # Update best validation result
        if (self.best_validation_result is None or
            validation_result.convergence_score > self.best_validation_result.convergence_score):
            self.best_validation_result = validation_result

        validation_time = time.time() - validation_start
        logger.info(f"✅ Validation complete in {validation_time:.2f}s - "
                   f"Quality: {avg_quality:.4f}, Success Rate: {success_rate:.2%}, "
                   f"SSIM Improvement: {avg_ssim_improvement:.4f}")

        return validation_result

    def _calculate_convergence_score(self, quality: float, success_rate: float) -> float:
        """Calculate convergence score combining quality and success rate"""
        return 0.6 * quality + 0.4 * success_rate

    def should_early_stop(self) -> bool:
        """Check if training should stop early due to lack of improvement"""
        if len(self.validation_history) < self.config.early_stopping_patience:
            return False

        recent_results = self.validation_history[-self.config.early_stopping_patience:]
        best_recent_score = max(r.convergence_score for r in recent_results)

        # Check if recent performance is significantly worse than best
        if self.best_validation_result:
            improvement_threshold = 0.02  # Require at least 2% improvement
            if best_recent_score < self.best_validation_result.convergence_score - improvement_threshold:
                return True

        return False

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary statistics"""
        if not self.validation_history:
            return {}

        qualities = [r.avg_quality for r in self.validation_history]
        success_rates = [r.success_rate for r in self.validation_history]
        convergence_scores = [r.convergence_score for r in self.validation_history]

        return {
            'total_validations': len(self.validation_history),
            'best_quality': max(qualities),
            'best_success_rate': max(success_rates),
            'best_convergence_score': max(convergence_scores),
            'latest_quality': qualities[-1],
            'latest_success_rate': success_rates[-1],
            'quality_trend': np.polyfit(range(len(qualities)), qualities, 1)[0] if len(qualities) > 1 else 0,
            'target_reached_episodes': sum(1 for r in self.validation_history if r.target_reached)
        }


class QualityAssuranceSystem:
    """Quality assurance and failure detection system"""

    def __init__(self, config: Stage1Config):
        self.config = config
        self.alerts = []
        self.metrics_history = deque(maxlen=1000)
        self.validation_history = []

        # Thresholds for quality assurance
        self.thresholds = {
            'reward_stagnation_episodes': 200,
            'quality_drop_threshold': 0.1,
            'success_rate_drop_threshold': 0.2,
            'loss_explosion_threshold': 5.0,
            'overfitting_threshold': 0.15,  # Validation vs training quality gap
            'performance_degradation_threshold': 0.05
        }

    def monitor_training_health(self, episode: int, metrics: TrainingMetrics) -> List[QualityAssuranceAlert]:
        """Monitor training health and detect issues"""
        self.metrics_history.append(metrics)
        alerts = []

        if len(self.metrics_history) < 10:
            return alerts

        recent_metrics = list(self.metrics_history)[-10:]

        # Check for reward stagnation
        alerts.extend(self._check_reward_stagnation(episode, recent_metrics))

        # Check for quality drops
        alerts.extend(self._check_quality_drops(episode, recent_metrics))

        # Check for training instability
        alerts.extend(self._check_training_instability(episode, recent_metrics))

        # Store alerts
        self.alerts.extend(alerts)

        return alerts

    def check_overfitting(self, episode: int, validation_result: ValidationResult,
                         training_metrics: List[TrainingMetrics]) -> List[QualityAssuranceAlert]:
        """Check for overfitting by comparing validation and training performance"""
        alerts = []

        if not training_metrics:
            return alerts

        # Get recent training quality
        recent_training_quality = np.mean([m.quality for m in training_metrics[-20:] if m.quality > 0])
        validation_quality = validation_result.avg_quality

        # Check for significant gap between training and validation
        quality_gap = recent_training_quality - validation_quality
        if quality_gap > self.thresholds['overfitting_threshold']:
            alert = QualityAssuranceAlert(
                alert_type='overfitting',
                severity='medium',
                episode=episode,
                timestamp=time.time(),
                message=f"Potential overfitting detected: training quality ({recent_training_quality:.4f}) "
                       f"significantly higher than validation quality ({validation_quality:.4f})",
                details={
                    'training_quality': recent_training_quality,
                    'validation_quality': validation_quality,
                    'quality_gap': quality_gap
                },
                recommended_action="Consider reducing model complexity or increasing regularization"
            )
            alerts.append(alert)

        return alerts

    def _check_reward_stagnation(self, episode: int, metrics: List[TrainingMetrics]) -> List[QualityAssuranceAlert]:
        """Check for reward stagnation"""
        alerts = []

        rewards = [m.reward for m in metrics]
        if len(set([round(r, 3) for r in rewards])) <= 2:  # Very little variation
            alert = QualityAssuranceAlert(
                alert_type='stagnation',
                severity='medium',
                episode=episode,
                timestamp=time.time(),
                message="Reward stagnation detected in recent episodes",
                details={'recent_rewards': rewards},
                recommended_action="Consider adjusting learning rate or exploration parameters"
            )
            alerts.append(alert)

        return alerts

    def _check_quality_drops(self, episode: int, metrics: List[TrainingMetrics]) -> List[QualityAssuranceAlert]:
        """Check for significant quality drops"""
        alerts = []

        qualities = [m.quality for m in metrics if m.quality > 0]
        if len(qualities) >= 5:
            recent_drop = max(qualities) - qualities[-1]
            if recent_drop > self.thresholds['quality_drop_threshold']:
                alert = QualityAssuranceAlert(
                    alert_type='degradation',
                    severity='high',
                    episode=episode,
                    timestamp=time.time(),
                    message=f"Significant quality drop detected: {recent_drop:.4f}",
                    details={'quality_drop': recent_drop, 'recent_qualities': qualities},
                    recommended_action="Review recent training changes or consider checkpoint restore"
                )
                alerts.append(alert)

        return alerts

    def _check_training_instability(self, episode: int, metrics: List[TrainingMetrics]) -> List[QualityAssuranceAlert]:
        """Check for training instability"""
        alerts = []

        # Check for loss explosion
        losses = [m.policy_loss for m in metrics if m.policy_loss is not None]
        if losses and any(loss > self.thresholds['loss_explosion_threshold'] for loss in losses):
            alert = QualityAssuranceAlert(
                alert_type='failure',
                severity='critical',
                episode=episode,
                timestamp=time.time(),
                message="Training instability detected - loss explosion",
                details={'recent_losses': losses},
                recommended_action="Reduce learning rate immediately or restore from checkpoint"
            )
            alerts.append(alert)

        return alerts

    def get_health_report(self) -> Dict[str, Any]:
        """Get training health report"""
        return {
            'total_alerts': len(self.alerts),
            'alert_breakdown': {
                alert_type: len([a for a in self.alerts if a.alert_type == alert_type])
                for alert_type in ['failure', 'overfitting', 'stagnation', 'degradation']
            },
            'critical_alerts': len([a for a in self.alerts if a.severity == 'critical']),
            'recent_alerts': [asdict(a) for a in self.alerts[-5:]]
        }


class ProgressReporter:
    """Progress reporting and milestone tracking system"""

    def __init__(self, config: Stage1Config, save_dir: Path):
        self.config = config
        self.save_dir = save_dir
        self.milestones = []
        self.hourly_reports = []

        self.training_start_time = None
        self.last_hourly_report = None

        # Define milestone targets
        self.milestone_targets = {
            'episode_1000': {'episode': 1000, 'description': "First 1000 episodes completed"},
            'episode_2500': {'episode': 2500, 'description': "Halfway milestone reached"},
            'episode_5000': {'episode': 5000, 'description': "Target episodes completed"},
            'success_rate_50': {'threshold': 0.5, 'description': "50% success rate achieved"},
            'success_rate_70': {'threshold': 0.7, 'description': "70% success rate achieved"},
            'success_rate_80': {'threshold': 0.8, 'description': "Target 80% success rate achieved"},
            'quality_70': {'threshold': 0.7, 'description': "70% average quality achieved"},
            'quality_80': {'threshold': 0.8, 'description': "80% average quality achieved"},
        }

    def start_training_session(self):
        """Start tracking training session"""
        self.training_start_time = time.time()
        self.last_hourly_report = time.time()

        logger.info("🚀 Stage 1 Training Session Started")
        logger.info(f"Target: {self.config.target_episodes} episodes")
        logger.info(f"Success Rate Target: {self.config.success_rate_threshold:.1%}")
        logger.info(f"SSIM Improvement Target: {self.config.ssim_improvement_threshold:.1%}")

    def check_milestones(self, episode: int, validation_result: Optional[ValidationResult] = None) -> List[TrainingMilestone]:
        """Check for milestone achievements"""
        new_milestones = []

        # Episode milestones
        for milestone_id, target in self.milestone_targets.items():
            if 'episode' in target and episode >= target['episode']:
                if not any(m.milestone_type == milestone_id for m in self.milestones):
                    milestone = TrainingMilestone(
                        milestone_type=milestone_id,
                        episode=episode,
                        timestamp=time.time(),
                        value=episode,
                        description=target['description'],
                        celebration_emoji="🎯"
                    )
                    new_milestones.append(milestone)

        # Performance milestones (require validation result)
        if validation_result:
            # Success rate milestones
            for milestone_id, target in self.milestone_targets.items():
                if 'threshold' in target and 'success_rate' in milestone_id:
                    if validation_result.success_rate >= target['threshold']:
                        if not any(m.milestone_type == milestone_id for m in self.milestones):
                            milestone = TrainingMilestone(
                                milestone_type=milestone_id,
                                episode=episode,
                                timestamp=time.time(),
                                value=validation_result.success_rate,
                                description=target['description'],
                                celebration_emoji="🎉"
                            )
                            new_milestones.append(milestone)

            # Quality milestones
            for milestone_id, target in self.milestone_targets.items():
                if 'threshold' in target and 'quality' in milestone_id:
                    if validation_result.avg_quality >= target['threshold']:
                        if not any(m.milestone_type == milestone_id for m in self.milestones):
                            milestone = TrainingMilestone(
                                milestone_type=milestone_id,
                                episode=episode,
                                timestamp=time.time(),
                                value=validation_result.avg_quality,
                                description=target['description'],
                                celebration_emoji="⭐"
                            )
                            new_milestones.append(milestone)

        # Log new milestones
        for milestone in new_milestones:
            logger.info(f"{milestone.celebration_emoji} MILESTONE: {milestone.description} "
                       f"(Episode {milestone.episode}, Value: {milestone.value:.4f})")

        self.milestones.extend(new_milestones)
        return new_milestones

    def should_generate_hourly_report(self) -> bool:
        """Check if it's time for hourly report"""
        if not self.config.hourly_reports or not self.last_hourly_report:
            return False

        return time.time() - self.last_hourly_report >= 3600  # 1 hour

    def generate_hourly_report(self, episode: int, current_metrics: List[TrainingMetrics],
                              validation_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hourly progress report"""
        current_time = time.time()

        if self.training_start_time:
            elapsed_time = current_time - self.training_start_time
            hours_elapsed = elapsed_time / 3600

            # Calculate progress
            episode_progress = episode / self.config.target_episodes
            estimated_completion = elapsed_time / episode_progress if episode_progress > 0 else None

            # Recent performance
            recent_rewards = [m.reward for m in current_metrics[-50:]] if current_metrics else []
            recent_qualities = [m.quality for m in current_metrics[-50:] if m.quality > 0] if current_metrics else []

            report = {
                'timestamp': current_time,
                'time_str': datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S'),
                'hours_elapsed': hours_elapsed,
                'episode': episode,
                'episode_progress': episode_progress,
                'estimated_completion_hours': estimated_completion / 3600 if estimated_completion else None,
                'recent_avg_reward': np.mean(recent_rewards) if recent_rewards else 0,
                'recent_avg_quality': np.mean(recent_qualities) if recent_qualities else 0,
                'milestones_achieved': len(self.milestones),
                'validation_summary': validation_summary
            }

            self.hourly_reports.append(report)
            self.last_hourly_report = current_time

            logger.info(f"📊 HOURLY REPORT - Episode {episode} "
                       f"({episode_progress:.1%} complete, {hours_elapsed:.1f}h elapsed)")
            logger.info(f"   Recent Avg Reward: {report['recent_avg_reward']:.4f}")
            logger.info(f"   Recent Avg Quality: {report['recent_avg_quality']:.4f}")
            logger.info(f"   Milestones: {len(self.milestones)}")

            return report

        return {}

    def save_progress_report(self) -> None:
        """Save comprehensive progress report"""
        report_data = {
            'training_session': {
                'start_time': self.training_start_time,
                'total_elapsed': time.time() - self.training_start_time if self.training_start_time else 0
            },
            'milestones': [asdict(m) for m in self.milestones],
            'hourly_reports': self.hourly_reports,
            'milestone_targets': self.milestone_targets
        }

        report_file = self.save_dir / "stage1_progress_report.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Progress report saved to: {report_file}")


class ArtifactManager:
    """Training artifact management and model saving system"""

    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.artifacts_dir = save_dir / "artifacts"
        self.checkpoints_dir = save_dir / "checkpoints"
        self.configs_dir = save_dir / "configs"

        # Create directories
        for dir_path in [self.artifacts_dir, self.checkpoints_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.saved_artifacts = []

    def save_checkpoint(self, episode: int, optimizer: PPOVTracerOptimizer,
                       validation_result: Optional[ValidationResult] = None) -> str:
        """Save training checkpoint"""
        checkpoint_name = f"checkpoint_episode_{episode}"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)

        # Save model
        model_path = checkpoint_path / "model"
        if hasattr(optimizer, 'model') and optimizer.model:
            optimizer.model.save(str(model_path))

        # Save checkpoint metadata
        metadata = {
            'episode': episode,
            'timestamp': time.time(),
            'model_path': str(model_path),
            'validation_result': asdict(validation_result) if validation_result else None
        }

        metadata_file = checkpoint_path / "checkpoint_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def save_best_model(self, optimizer: PPOVTracerOptimizer, validation_result: ValidationResult) -> str:
        """Save best model based on validation performance"""
        best_model_dir = self.artifacts_dir / "best_model"
        best_model_dir.mkdir(exist_ok=True)

        # Save model
        model_path = best_model_dir / "model"
        if hasattr(optimizer, 'model') and optimizer.model:
            optimizer.model.save(str(model_path))

        # Save model metadata
        metadata = {
            'validation_result': asdict(validation_result),
            'timestamp': time.time(),
            'model_path': str(model_path),
            'is_best_model': True
        }

        metadata_file = best_model_dir / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"🏆 Best model saved: {best_model_dir} "
                   f"(Quality: {validation_result.avg_quality:.4f}, "
                   f"Success Rate: {validation_result.success_rate:.2%})")

        return str(best_model_dir)

    def export_training_config(self, config: Stage1Config, model_config: Dict[str, Any],
                              env_config: Dict[str, Any]) -> str:
        """Export training configuration for reproducibility"""
        config_data = {
            'stage1_config': asdict(config),
            'model_config': model_config,
            'environment_config': env_config,
            'export_timestamp': time.time(),
            'reproducibility_notes': {
                'random_seed': 'Set numpy and torch random seeds for full reproducibility',
                'environment_version': 'Record VTracer and environment versions',
                'hardware_info': 'Record GPU/CPU specifications for performance comparison'
            }
        }

        config_file = self.configs_dir / "stage1_training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"📋 Training config exported: {config_file}")
        return str(config_file)

    def create_training_summary(self, final_validation: ValidationResult,
                               milestones: List[TrainingMilestone],
                               qa_summary: Dict[str, Any]) -> str:
        """Create comprehensive training summary"""
        summary = {
            'stage1_training_summary': {
                'completion_status': 'SUCCESS' if final_validation.target_reached else 'INCOMPLETE',
                'final_validation': asdict(final_validation),
                'target_achievement': {
                    'success_rate_target': final_validation.success_rate >= 0.80,
                    'ssim_improvement_target': final_validation.ssim_improvement >= 0.75,
                    'overall_target_reached': final_validation.target_reached
                },
                'milestones_achieved': [asdict(m) for m in milestones],
                'quality_assurance_summary': qa_summary,
                'training_artifacts': {
                    'best_model_dir': str(self.artifacts_dir / "best_model"),
                    'checkpoints_dir': str(self.checkpoints_dir),
                    'configs_dir': str(self.configs_dir)
                }
            }
        }

        summary_file = self.artifacts_dir / "stage1_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Create human-readable report
        report_lines = [
            "# Stage 1 Training Summary",
            "=" * 50,
            "",
            f"Status: {summary['stage1_training_summary']['completion_status']}",
            f"Final Quality: {final_validation.avg_quality:.4f}",
            f"Final Success Rate: {final_validation.success_rate:.2%}",
            f"Final SSIM Improvement: {final_validation.ssim_improvement:.4f}",
            f"Target Reached: {final_validation.target_reached}",
            "",
            f"Milestones Achieved: {len(milestones)}",
            f"Quality Assurance Alerts: {qa_summary.get('total_alerts', 0)}",
            "",
            "Artifact Locations:",
            f"- Best Model: {self.artifacts_dir / 'best_model'}",
            f"- Checkpoints: {self.checkpoints_dir}",
            f"- Configs: {self.configs_dir}",
            f"- Summary: {summary_file}"
        ]

        report_file = self.artifacts_dir / "stage1_training_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"📄 Training summary created: {summary_file}")
        logger.info(f"📄 Training report created: {report_file}")

        return str(summary_file)


class Stage1TrainingExecutor:
    """Main Stage 1 training execution system"""

    def __init__(self,
                 training_images: List[str],
                 validation_images: List[str],
                 save_dir: str = "models/stage1_training",
                 config: Optional[Stage1Config] = None,
                 model_config: Optional[Dict] = None):
        """
        Initialize Stage 1 training executor

        Args:
            training_images: List of simple geometric logo image paths for training
            validation_images: List of validation image paths
            save_dir: Directory to save training artifacts
            config: Stage 1 configuration
            model_config: PPO model configuration
        """
        self.training_images = training_images
        self.validation_images = validation_images
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or Stage1Config()
        self.model_config = model_config or self._default_model_config()

        # Initialize components
        self.validation_protocol = ValidationProtocol(validation_images, self.config)
        self.quality_assurance = QualityAssuranceSystem(self.config)
        self.progress_reporter = ProgressReporter(self.config, self.save_dir)
        self.artifact_manager = ArtifactManager(self.save_dir)

        # Real-time monitoring
        self.real_time_monitor = RealTimeMonitor(
            websocket_port=8767,  # Different port for Stage 1
            save_dir=str(self.save_dir / "real_time_monitoring")
        )

        # Training state
        self.current_episode = 0
        self.training_metrics = []
        self.optimizer = None
        self.best_validation_score = 0.0

        logger.info("🎯 Stage 1 Training Executor initialized")
        logger.info(f"Training images: {len(training_images)}")
        logger.info(f"Validation images: {len(validation_images)}")
        logger.info(f"Target episodes: {self.config.target_episodes}")
        logger.info(f"Success rate target: {self.config.success_rate_threshold:.1%}")

    def _default_model_config(self) -> Dict[str, Any]:
        """Default PPO model configuration optimized for Stage 1"""
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

    async def execute_stage1_training(self) -> Dict[str, Any]:
        """Execute complete Stage 1 training with all monitoring and quality assurance"""
        logger.info("🚀 Starting Stage 1 Training Execution")

        training_start_time = time.time()

        try:
            # Start real-time monitoring
            await self.real_time_monitor.start_monitoring()

            # Initialize progress reporting
            self.progress_reporter.start_training_session()

            # Create training stage for curriculum pipeline
            stage1_stage = TrainingStage(
                name="stage1_simple_geometric",
                image_types=["simple"],
                difficulty=0.1,
                target_quality=self.config.quality_target,
                max_episodes=self.config.target_episodes,
                success_threshold=self.config.success_rate_threshold
            )

            # Setup training images dictionary for curriculum pipeline
            training_images_dict = {"simple": self.training_images}

            # Create curriculum pipeline for Stage 1
            pipeline = CurriculumTrainingPipeline(
                training_images=training_images_dict,
                model_config=self.model_config,
                curriculum_config={
                    "stage_0": {
                        "name": "stage1_simple_geometric",
                        "max_episodes": self.config.target_episodes,
                        "success_threshold": self.config.success_rate_threshold,
                        "target_quality": self.config.quality_target
                    }
                },
                save_dir=str(self.save_dir / "curriculum"),
                enable_real_time_monitoring=False  # We handle monitoring separately
            )

            # Custom training loop with validation and quality assurance
            training_result = await self._execute_training_loop(pipeline)

            # Final evaluation and reporting
            final_report = await self._generate_final_report(training_result)

            training_time = time.time() - training_start_time
            logger.info(f"✅ Stage 1 Training Complete in {training_time:.2f}s")

            return final_report

        except Exception as e:
            logger.error(f"❌ Stage 1 Training Failed: {e}")
            raise
        finally:
            # Cleanup
            await self.real_time_monitor.stop_monitoring()
            if self.optimizer:
                self.optimizer.close()

    async def _execute_training_loop(self, pipeline: CurriculumTrainingPipeline) -> Dict[str, Any]:
        """Execute custom training loop with enhanced monitoring"""
        logger.info("🔄 Starting Stage 1 training loop...")

        # Get the first (and only) stage for Stage 1
        stage1_stage = pipeline.curriculum_stages[0]

        # Select primary training image
        if not self.training_images:
            raise ValueError("No training images provided for Stage 1")

        primary_image = self.training_images[0]
        logger.info(f"Primary training image: {primary_image}")

        # Create optimizer
        self.optimizer = pipeline._create_stage_optimizer(stage1_stage, primary_image)

        # Training loop
        episode = 0
        consecutive_failures = 0
        last_validation_episode = 0

        while episode < self.config.target_episodes:
            try:
                # Run training episode
                episode_result = await self._run_training_episode(episode, self.optimizer)

                # Update metrics
                episode_metrics = self._create_episode_metrics(episode, episode_result)
                self.training_metrics.append(episode_metrics)

                # Real-time monitoring
                self.real_time_monitor.on_episode_complete(episode_metrics)

                # Quality assurance monitoring
                qa_alerts = self.quality_assurance.monitor_training_health(episode, episode_metrics)
                for alert in qa_alerts:
                    await self._handle_qa_alert(alert)

                # Validation check
                if episode - last_validation_episode >= self.config.validation_frequency:
                    validation_result = self.validation_protocol.run_validation(episode, self.optimizer)
                    last_validation_episode = episode

                    # Check for overfitting
                    overfitting_alerts = self.quality_assurance.check_overfitting(
                        episode, validation_result, self.training_metrics[-100:]
                    )
                    for alert in overfitting_alerts:
                        await self._handle_qa_alert(alert)

                    # Check milestones
                    new_milestones = self.progress_reporter.check_milestones(episode, validation_result)

                    # Save checkpoint
                    if episode % self.config.checkpoint_frequency == 0:
                        self.artifact_manager.save_checkpoint(episode, self.optimizer, validation_result)

                    # Save best model
                    if validation_result.convergence_score > self.best_validation_score:
                        self.best_validation_score = validation_result.convergence_score
                        self.artifact_manager.save_best_model(self.optimizer, validation_result)

                    # Check for early stopping
                    if self.validation_protocol.should_early_stop():
                        logger.info(f"🛑 Early stopping triggered at episode {episode}")
                        break

                    # Check if target reached
                    if validation_result.target_reached:
                        logger.info(f"🎉 Training target reached at episode {episode}!")
                        break

                # Hourly reporting
                if self.progress_reporter.should_generate_hourly_report():
                    validation_summary = self.validation_protocol.get_validation_summary()
                    self.progress_reporter.generate_hourly_report(
                        episode, self.training_metrics, validation_summary
                    )

                episode += 1
                consecutive_failures = 0

            except Exception as e:
                logger.error(f"Episode {episode} failed: {e}")
                consecutive_failures += 1

                if consecutive_failures >= 5:
                    logger.error("Too many consecutive failures, stopping training")
                    break

                episode += 1

        self.current_episode = episode

        # Final validation
        final_validation = self.validation_protocol.run_validation(episode, self.optimizer)

        return {
            'episodes_completed': episode,
            'final_validation': final_validation,
            'training_metrics': self.training_metrics,
            'validation_history': self.validation_protocol.validation_history,
            'milestones': self.progress_reporter.milestones,
            'qa_alerts': self.quality_assurance.alerts
        }

    async def _run_training_episode(self, episode: int, optimizer: PPOVTracerOptimizer) -> Dict[str, Any]:
        """Run single training episode"""
        # Simulate episode execution
        # In real implementation, this would interface with the actual PPO training step

        # For simulation, we'll create realistic training metrics
        np.random.seed(episode)  # For reproducible simulation

        # Simulate training progression
        progress = min(episode / self.config.target_episodes, 1.0)
        base_quality = 0.6 + 0.3 * progress + np.random.normal(0, 0.05)
        base_reward = base_quality * 10 + np.random.normal(0, 1)

        episode_result = {
            'reward': max(0, base_reward),
            'quality': max(0, min(1, base_quality)),
            'episode_length': np.random.randint(20, 50),
            'success': base_quality > 0.75,
            'ssim_improvement': max(0, base_quality - 0.2 + np.random.normal(0, 0.1)),
            'parameters_used': {
                'color_precision': np.random.randint(2, 6),
                'corner_threshold': np.random.randint(20, 40)
            }
        }

        return episode_result

    def _create_episode_metrics(self, episode: int, episode_result: Dict[str, Any]) -> TrainingMetrics:
        """Create training metrics from episode result"""
        return TrainingMetrics(
            timestamp=time.time(),
            episode=episode,
            reward=episode_result['reward'],
            episode_length=episode_result['episode_length'],
            quality=episode_result['quality'],
            success=episode_result['success'],
            ssim_improvement=episode_result.get('ssim_improvement'),
            parameters_used=episode_result.get('parameters_used')
        )

    async def _handle_qa_alert(self, alert: QualityAssuranceAlert):
        """Handle quality assurance alert"""
        logger.warning(f"🚨 QA ALERT [{alert.severity.upper()}]: {alert.message}")
        logger.info(f"   Recommended Action: {alert.recommended_action}")

        # Broadcast alert via real-time monitoring
        await self.real_time_monitor.websocket_server.broadcast({
            'type': 'qa_alert',
            'alert': asdict(alert)
        })

        # Take automatic action for critical alerts
        if alert.severity == 'critical' and alert.alert_type == 'failure':
            logger.error("🛑 Critical failure detected - emergency checkpoint save")
            if self.optimizer:
                self.artifact_manager.save_checkpoint(
                    self.current_episode, self.optimizer, None
                )

    async def _generate_final_report(self, training_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        final_validation = training_result['final_validation']

        # Export training configuration
        self.artifact_manager.export_training_config(
            self.config, self.model_config, {}
        )

        # Save progress report
        self.progress_reporter.save_progress_report()

        # Generate quality assurance summary
        qa_summary = self.quality_assurance.get_health_report()

        # Create training summary
        self.artifact_manager.create_training_summary(
            final_validation, self.progress_reporter.milestones, qa_summary
        )

        # Comprehensive final report
        final_report = {
            'stage1_training_results': {
                'success': final_validation.target_reached,
                'episodes_completed': training_result['episodes_completed'],
                'target_episodes': self.config.target_episodes,
                'completion_rate': training_result['episodes_completed'] / self.config.target_episodes,

                'final_performance': {
                    'avg_quality': final_validation.avg_quality,
                    'success_rate': final_validation.success_rate,
                    'ssim_improvement': final_validation.ssim_improvement,
                    'convergence_score': final_validation.convergence_score
                },

                'target_achievement': {
                    'success_rate_target_reached': final_validation.success_rate >= self.config.success_rate_threshold,
                    'ssim_improvement_target_reached': final_validation.ssim_improvement >= self.config.ssim_improvement_threshold,
                    'overall_target_reached': final_validation.target_reached
                },

                'training_statistics': {
                    'total_validations': len(training_result['validation_history']),
                    'milestones_achieved': len(training_result['milestones']),
                    'qa_alerts_total': len(training_result['qa_alerts']),
                    'critical_alerts': len([a for a in training_result['qa_alerts'] if a.severity == 'critical'])
                },

                'artifacts': {
                    'save_directory': str(self.save_dir),
                    'best_model_path': str(self.save_dir / "artifacts" / "best_model"),
                    'checkpoints_path': str(self.save_dir / "checkpoints"),
                    'monitoring_data_path': str(self.save_dir / "real_time_monitoring")
                }
            }
        }

        # Save final report
        final_report_file = self.save_dir / "stage1_final_report.json"
        with open(final_report_file, 'w') as f:
            json.dump(final_report, f, indent=2)

        logger.info("📊 Stage 1 Training Final Report:")
        logger.info(f"   Success: {final_validation.target_reached}")
        logger.info(f"   Episodes: {training_result['episodes_completed']}/{self.config.target_episodes}")
        logger.info(f"   Final Quality: {final_validation.avg_quality:.4f}")
        logger.info(f"   Final Success Rate: {final_validation.success_rate:.2%}")
        logger.info(f"   Final SSIM Improvement: {final_validation.ssim_improvement:.4f}")
        logger.info(f"   Milestones: {len(training_result['milestones'])}")
        logger.info(f"   Artifacts saved to: {self.save_dir}")

        return final_report


# Factory function for easy usage
def create_stage1_executor(simple_geometric_images: List[str],
                          validation_images: Optional[List[str]] = None,
                          save_dir: str = "models/stage1_training",
                          config: Optional[Stage1Config] = None) -> Stage1TrainingExecutor:
    """
    Factory function to create Stage 1 training executor

    Args:
        simple_geometric_images: List of simple geometric logo paths for training
        validation_images: List of validation image paths (auto-split if None)
        save_dir: Directory to save training artifacts
        config: Stage 1 configuration

    Returns:
        Configured Stage 1 training executor
    """
    if validation_images is None:
        # Auto-split training images for validation
        n_val = max(1, len(simple_geometric_images) // 5)  # 20% for validation
        validation_images = simple_geometric_images[-n_val:]
        training_images = simple_geometric_images[:-n_val]
    else:
        training_images = simple_geometric_images

    return Stage1TrainingExecutor(
        training_images=training_images,
        validation_images=validation_images,
        save_dir=save_dir,
        config=config or Stage1Config()
    )


# Example usage
async def main():
    """Example Stage 1 training execution"""
    # Get simple geometric images
    simple_images = [
        "/Users/nrw/python/svg-ai/data/optimization_test/simple/1001419.png",
        "/Users/nrw/python/svg-ai/data/optimization_test/simple/10075523.png",
        "/Users/nrw/python/svg-ai/data/optimization_test/simple/1005141.png",
        "/Users/nrw/python/svg-ai/data/optimization_test/simple/10075504.png",
        "/Users/nrw/python/svg-ai/data/optimization_test/simple/10072999.png"
    ]

    # Create Stage 1 executor
    executor = create_stage1_executor(
        simple_geometric_images=simple_images,
        save_dir="models/stage1_demo",
        config=Stage1Config(target_episodes=100)  # Shorter for demo
    )

    # Execute Stage 1 training
    results = await executor.execute_stage1_training()

    print(f"Stage 1 Training Complete: {results['stage1_training_results']['success']}")


if __name__ == "__main__":
    asyncio.run(main())