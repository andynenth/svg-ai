"""Comprehensive training monitoring and visualization system for PPO agent training"""

import numpy as np
import pandas as pd
import json
import csv
import logging
import time
import os
import psutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from threading import Lock

# Optional visualization dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    sns = None

# Optional dependencies for enhanced monitoring
try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@dataclass
class EpisodeMetrics:
    """Single episode metrics container"""
    episode: int
    timestamp: datetime
    reward: float
    length: int
    quality_improvement: float
    quality_final: float
    quality_initial: float
    termination_reason: str
    success: bool

    # Algorithm-specific metrics
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy: Optional[float] = None
    kl_divergence: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None

    # Performance metrics
    episode_time: Optional[float] = None
    memory_usage: Optional[float] = None

    # Additional info
    logo_type: Optional[str] = None
    difficulty: Optional[str] = None
    parameters_explored: Optional[Dict] = None


@dataclass
class TrainingSession:
    """Training session metadata and configuration"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_episodes: int = 0
    total_timesteps: int = 0
    configuration: Optional[Dict] = None
    status: str = "running"  # running, completed, failed, paused


class TrainingMonitor:
    """Comprehensive training monitoring and visualization system for PPO agent training"""

    def __init__(self,
                 log_dir: str = "logs/ppo_training",
                 use_tensorboard: bool = True,
                 use_wandb: bool = False,
                 project_name: str = "vtracer-rl-optimization",
                 session_name: Optional[str] = None,
                 buffer_size: int = 1000):
        """
        Initialize training monitor

        Args:
            log_dir: Directory for storing logs and outputs
            use_tensorboard: Whether to use TensorBoard logging
            use_wandb: Whether to use Weights & Biases logging
            project_name: Project name for experiment tracking
            session_name: Custom session name
            buffer_size: Size of metrics buffer for real-time analysis
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Session management
        self.session_id = session_name or f"session_{datetime.now():%Y%m%d_%H%M%S}"
        self.session = TrainingSession(
            session_id=self.session_id,
            start_time=datetime.now()
        )

        # Logging setup
        self.logger = self._setup_logging()

        # Metrics storage
        self.episodes: List[EpisodeMetrics] = []
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.aggregated_metrics = defaultdict(list)

        # Metrics history for compatibility with integration tests
        self.metrics_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'quality_improvements': [],
            'success_rates': [],
            'training_loss': [],
            'value_loss': [],
            'policy_loss': [],
            'entropy': []
        }

        # Thread safety
        self.lock = Lock()

        # Performance tracking
        self.start_time = time.time()
        self.last_log_time = time.time()

        # External logging setup
        self.tensorboard_writer = None
        self.use_wandb = use_wandb

        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tensorboard_writer = SummaryWriter(
                log_dir=self.log_dir / "tensorboard" / self.session_id
            )
            self.logger.info("TensorBoard logging enabled")

        if use_wandb and WANDB_AVAILABLE:
            wandb.init(project=project_name, name=self.session_id, config={})
            self.logger.info("Weights & Biases logging enabled")
        elif use_wandb:
            self.logger.warning("Weights & Biases requested but not available")

        # File paths
        self.episodes_file = self.log_dir / f"episodes_{self.session_id}.jsonl"
        self.metrics_file = self.log_dir / f"metrics_{self.session_id}.csv"
        self.session_file = self.log_dir / f"session_{self.session_id}.json"

        # Initialize CSV file
        self._init_csv_file()

        self.logger.info(f"Training monitor initialized for session: {self.session_id}")

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger(f"training_monitor_{self.session_id}")
        logger.setLevel(logging.INFO)

        # File handler
        handler = logging.FileHandler(self.log_dir / f"training_{self.session_id}.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _init_csv_file(self):
        """Initialize CSV file with headers"""
        if not self.metrics_file.exists():
            headers = [
                "episode", "timestamp", "reward", "length", "quality_improvement",
                "quality_final", "quality_initial", "termination_reason", "success",
                "policy_loss", "value_loss", "entropy", "kl_divergence",
                "learning_rate", "gradient_norm", "episode_time", "memory_usage",
                "logo_type", "difficulty"
            ]
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_episode(self,
                   episode: int,
                   reward: float,
                   length: int,
                   quality_improvement: float,
                   quality_final: float,
                   quality_initial: float,
                   termination_reason: str = "max_steps",
                   success: bool = False,
                   algorithm_metrics: Optional[Dict[str, float]] = None,
                   performance_metrics: Optional[Dict[str, float]] = None,
                   additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Log comprehensive episode metrics

        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length in steps
            quality_improvement: SSIM quality improvement achieved
            quality_final: Final quality metric
            quality_initial: Initial quality metric
            termination_reason: Reason for episode termination
            success: Whether episode achieved target quality
            algorithm_metrics: PPO algorithm metrics (policy_loss, value_loss, etc.)
            performance_metrics: Performance metrics (time, memory)
            additional_info: Additional episode information
        """
        with self.lock:
            timestamp = datetime.now()

            # Extract algorithm metrics
            alg_metrics = algorithm_metrics or {}
            policy_loss = alg_metrics.get('policy_loss')
            value_loss = alg_metrics.get('value_loss')
            entropy = alg_metrics.get('entropy')
            kl_divergence = alg_metrics.get('kl_divergence')
            learning_rate = alg_metrics.get('learning_rate')
            gradient_norm = alg_metrics.get('gradient_norm')

            # Extract performance metrics
            perf_metrics = performance_metrics or {}
            episode_time = perf_metrics.get('episode_time')
            memory_usage = perf_metrics.get('memory_usage', self._get_memory_usage())

            # Extract additional info
            info = additional_info or {}
            logo_type = info.get('logo_type')
            difficulty = info.get('difficulty')
            parameters_explored = info.get('parameters_explored')

            # Create episode metrics
            episode_metrics = EpisodeMetrics(
                episode=episode,
                timestamp=timestamp,
                reward=reward,
                length=length,
                quality_improvement=quality_improvement,
                quality_final=quality_final,
                quality_initial=quality_initial,
                termination_reason=termination_reason,
                success=success,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                kl_divergence=kl_divergence,
                learning_rate=learning_rate,
                gradient_norm=gradient_norm,
                episode_time=episode_time,
                memory_usage=memory_usage,
                logo_type=logo_type,
                difficulty=difficulty,
                parameters_explored=parameters_explored
            )

            # Store metrics
            self.episodes.append(episode_metrics)
            self.metrics_buffer.append(episode_metrics)

            # Update session
            self.session.total_episodes = len(self.episodes)

            # Write to files
            self._write_episode_json(episode_metrics)
            self._write_episode_csv(episode_metrics)

            # External logging
            self._log_to_tensorboard(episode, episode_metrics)
            self._log_to_wandb(episode, episode_metrics)

            # Log to console periodically
            self._log_episode_summary(episode_metrics)

            self.logger.debug(f"Episode {episode} logged: reward={reward:.2f}, "
                            f"quality_improvement={quality_improvement:.4f}")

    def log_training_step(self,
                         step: int,
                         policy_loss: float,
                         value_loss: float,
                         entropy: float,
                         kl_divergence: Optional[float] = None,
                         learning_rate: Optional[float] = None,
                         gradient_norm: Optional[float] = None) -> None:
        """
        Log training step metrics (called during PPO updates)

        Args:
            step: Training step number
            policy_loss: Policy network loss
            value_loss: Value network loss
            entropy: Action entropy
            kl_divergence: KL divergence between old and new policy
            learning_rate: Current learning rate
            gradient_norm: Gradient norm
        """
        with self.lock:
            # Store aggregated metrics
            self.aggregated_metrics['policy_loss'].append(policy_loss)
            self.aggregated_metrics['value_loss'].append(value_loss)
            self.aggregated_metrics['entropy'].append(entropy)

            if kl_divergence is not None:
                self.aggregated_metrics['kl_divergence'].append(kl_divergence)
            if learning_rate is not None:
                self.aggregated_metrics['learning_rate'].append(learning_rate)
            if gradient_norm is not None:
                self.aggregated_metrics['gradient_norm'].append(gradient_norm)

            # TensorBoard logging
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Train/PolicyLoss', policy_loss, step)
                self.tensorboard_writer.add_scalar('Train/ValueLoss', value_loss, step)
                self.tensorboard_writer.add_scalar('Train/Entropy', entropy, step)
                if kl_divergence is not None:
                    self.tensorboard_writer.add_scalar('Train/KLDivergence', kl_divergence, step)
                if learning_rate is not None:
                    self.tensorboard_writer.add_scalar('Train/LearningRate', learning_rate, step)
                if gradient_norm is not None:
                    self.tensorboard_writer.add_scalar('Train/GradientNorm', gradient_norm, step)

            # Weights & Biases logging
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'train/policy_loss': policy_loss,
                    'train/value_loss': value_loss,
                    'train/entropy': entropy,
                    'train/step': step
                })
                if kl_divergence is not None:
                    wandb.log({'train/kl_divergence': kl_divergence})
                if learning_rate is not None:
                    wandb.log({'train/learning_rate': learning_rate})
                if gradient_norm is not None:
                    wandb.log({'train/gradient_norm': gradient_norm})

    def log_evaluation(self,
                      episode: int,
                      eval_reward: float,
                      eval_quality: float,
                      eval_success_rate: float,
                      eval_episodes: int = 1) -> None:
        """
        Log evaluation metrics

        Args:
            episode: Current training episode
            eval_reward: Average evaluation reward
            eval_quality: Average evaluation quality
            eval_success_rate: Success rate in evaluation
            eval_episodes: Number of evaluation episodes
        """
        with self.lock:
            # TensorBoard logging
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Eval/Reward', eval_reward, episode)
                self.tensorboard_writer.add_scalar('Eval/Quality', eval_quality, episode)
                self.tensorboard_writer.add_scalar('Eval/SuccessRate', eval_success_rate, episode)

            # Weights & Biases logging
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'eval/reward': eval_reward,
                    'eval/quality': eval_quality,
                    'eval/success_rate': eval_success_rate,
                    'eval/episode': episode
                })

            self.logger.info(f"Evaluation at episode {episode}: "
                           f"reward={eval_reward:.2f}, quality={eval_quality:.4f}, "
                           f"success_rate={eval_success_rate:.2%}")

    def _write_episode_json(self, episode_metrics: EpisodeMetrics) -> None:
        """Write episode metrics to JSON lines file"""
        try:
            with open(self.episodes_file, 'a') as f:
                json_data = asdict(episode_metrics)
                json_data['timestamp'] = episode_metrics.timestamp.isoformat()
                f.write(json.dumps(json_data, default=str) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write episode JSON: {e}")

    def _write_episode_csv(self, episode_metrics: EpisodeMetrics) -> None:
        """Write episode metrics to CSV file"""
        try:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [
                    episode_metrics.episode,
                    episode_metrics.timestamp.isoformat(),
                    episode_metrics.reward,
                    episode_metrics.length,
                    episode_metrics.quality_improvement,
                    episode_metrics.quality_final,
                    episode_metrics.quality_initial,
                    episode_metrics.termination_reason,
                    episode_metrics.success,
                    episode_metrics.policy_loss,
                    episode_metrics.value_loss,
                    episode_metrics.entropy,
                    episode_metrics.kl_divergence,
                    episode_metrics.learning_rate,
                    episode_metrics.gradient_norm,
                    episode_metrics.episode_time,
                    episode_metrics.memory_usage,
                    episode_metrics.logo_type,
                    episode_metrics.difficulty
                ]
                writer.writerow(row)
        except Exception as e:
            self.logger.error(f"Failed to write episode CSV: {e}")

    def _log_to_tensorboard(self, episode: int, metrics: EpisodeMetrics) -> None:
        """Log metrics to TensorBoard"""
        if not self.tensorboard_writer:
            return

        try:
            self.tensorboard_writer.add_scalar('Episode/Reward', metrics.reward, episode)
            self.tensorboard_writer.add_scalar('Episode/Length', metrics.length, episode)
            self.tensorboard_writer.add_scalar('Episode/QualityImprovement',
                                             metrics.quality_improvement, episode)
            self.tensorboard_writer.add_scalar('Episode/QualityFinal',
                                             metrics.quality_final, episode)
            self.tensorboard_writer.add_scalar('Episode/Success',
                                             float(metrics.success), episode)

            if metrics.episode_time:
                self.tensorboard_writer.add_scalar('Episode/Time',
                                                 metrics.episode_time, episode)
            if metrics.memory_usage:
                self.tensorboard_writer.add_scalar('Episode/MemoryUsage',
                                                 metrics.memory_usage, episode)

            self.tensorboard_writer.flush()
        except Exception as e:
            self.logger.error(f"Failed to log to TensorBoard: {e}")

    def _log_to_wandb(self, episode: int, metrics: EpisodeMetrics) -> None:
        """Log metrics to Weights & Biases"""
        if not (self.use_wandb and WANDB_AVAILABLE):
            return

        try:
            log_data = {
                'episode/reward': metrics.reward,
                'episode/length': metrics.length,
                'episode/quality_improvement': metrics.quality_improvement,
                'episode/quality_final': metrics.quality_final,
                'episode/success': float(metrics.success),
                'episode/number': episode
            }

            if metrics.episode_time:
                log_data['episode/time'] = metrics.episode_time
            if metrics.memory_usage:
                log_data['episode/memory_usage'] = metrics.memory_usage
            if metrics.logo_type:
                log_data['episode/logo_type'] = metrics.logo_type

            wandb.log(log_data)
        except Exception as e:
            self.logger.error(f"Failed to log to Weights & Biases: {e}")

    def _log_episode_summary(self, metrics: EpisodeMetrics) -> None:
        """Log episode summary to console periodically"""
        current_time = time.time()

        # Log every 10 episodes or every 60 seconds
        if (metrics.episode % 10 == 0 or
            current_time - self.last_log_time > 60):

            self.last_log_time = current_time

            # Calculate recent performance
            recent_episodes = list(self.metrics_buffer)[-10:] if self.metrics_buffer else []

            if recent_episodes:
                avg_reward = np.mean([e.reward for e in recent_episodes])
                avg_quality = np.mean([e.quality_improvement for e in recent_episodes])
                success_rate = np.mean([e.success for e in recent_episodes])

                self.logger.info(
                    f"Episode {metrics.episode}: reward={metrics.reward:.2f} "
                    f"(avg: {avg_reward:.2f}), quality={metrics.quality_improvement:.4f} "
                    f"(avg: {avg_quality:.4f}), success_rate={success_rate:.2%}"
                )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0

    def get_training_statistics(self,
                               window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive training statistics

        Args:
            window_size: Size of window for recent statistics (None for all)

        Returns:
            Dictionary of training statistics
        """
        with self.lock:
            if not self.episodes:
                return {"message": "No episodes recorded yet"}

            # Get episodes for analysis
            episodes = self.episodes
            if window_size:
                episodes = episodes[-window_size:]

            # Calculate statistics
            rewards = [e.reward for e in episodes]
            qualities = [e.quality_improvement for e in episodes]
            lengths = [e.length for e in episodes]
            successes = [e.success for e in episodes]

            # Performance statistics
            training_time = time.time() - self.start_time
            episodes_per_minute = len(self.episodes) / (training_time / 60) if training_time > 0 else 0

            stats = {
                "session_info": {
                    "session_id": self.session_id,
                    "total_episodes": len(self.episodes),
                    "window_episodes": len(episodes),
                    "training_time_minutes": training_time / 60,
                    "episodes_per_minute": episodes_per_minute
                },
                "reward_stats": {
                    "mean": float(np.mean(rewards)),
                    "std": float(np.std(rewards)),
                    "min": float(np.min(rewards)),
                    "max": float(np.max(rewards)),
                    "median": float(np.median(rewards))
                },
                "quality_stats": {
                    "mean": float(np.mean(qualities)),
                    "std": float(np.std(qualities)),
                    "min": float(np.min(qualities)),
                    "max": float(np.max(qualities)),
                    "median": float(np.median(qualities))
                },
                "episode_stats": {
                    "mean_length": float(np.mean(lengths)),
                    "success_rate": float(np.mean(successes)),
                    "total_successful": sum(successes)
                }
            }

            # Algorithm metrics statistics
            if self.aggregated_metrics:
                for metric_name, values in self.aggregated_metrics.items():
                    if values:
                        stats[f"{metric_name}_stats"] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "latest": float(values[-1]) if values else 0.0
                        }

            # Logo type breakdown
            logo_types = defaultdict(list)
            for episode in episodes:
                if episode.logo_type:
                    logo_types[episode.logo_type].append(episode.quality_improvement)

            if logo_types:
                stats["logo_type_performance"] = {}
                for logo_type, improvements in logo_types.items():
                    stats["logo_type_performance"][logo_type] = {
                        "count": len(improvements),
                        "mean_quality": float(np.mean(improvements)),
                        "success_rate": float(np.mean([e.success for e in episodes
                                                     if e.logo_type == logo_type]))
                    }

            return stats

    def get_convergence_analysis(self) -> Dict[str, Any]:
        """
        Analyze training convergence

        Returns:
            Convergence analysis results
        """
        with self.lock:
            if len(self.episodes) < 10:
                return {"message": "Insufficient data for convergence analysis"}

            # Get recent performance trend
            window_size = min(50, len(self.episodes) // 4)
            recent_rewards = [e.reward for e in self.episodes[-window_size:]]
            recent_qualities = [e.quality_improvement for e in self.episodes[-window_size:]]

            # Calculate trends
            x = np.arange(len(recent_rewards))

            # Reward trend
            reward_slope = np.polyfit(x, recent_rewards, 1)[0]
            quality_slope = np.polyfit(x, recent_qualities, 1)[0]

            # Stability analysis
            reward_variance = np.var(recent_rewards)
            quality_variance = np.var(recent_qualities)

            # Moving averages
            ma_window = min(10, len(recent_rewards) // 2)
            reward_ma = np.convolve(recent_rewards, np.ones(ma_window), 'valid') / ma_window
            quality_ma = np.convolve(recent_qualities, np.ones(ma_window), 'valid') / ma_window

            return {
                "convergence_status": {
                    "reward_trend": "improving" if reward_slope > 0.01 else "stable" if abs(reward_slope) < 0.01 else "declining",
                    "quality_trend": "improving" if quality_slope > 0.001 else "stable" if abs(quality_slope) < 0.001 else "declining",
                    "reward_slope": float(reward_slope),
                    "quality_slope": float(quality_slope)
                },
                "stability_metrics": {
                    "reward_variance": float(reward_variance),
                    "quality_variance": float(quality_variance),
                    "stable": reward_variance < 1.0 and quality_variance < 0.01
                },
                "recent_performance": {
                    "episodes_analyzed": len(recent_rewards),
                    "avg_reward": float(np.mean(recent_rewards)),
                    "avg_quality": float(np.mean(recent_qualities)),
                    "reward_ma_latest": float(reward_ma[-1]) if len(reward_ma) > 0 else 0.0,
                    "quality_ma_latest": float(quality_ma[-1]) if len(quality_ma) > 0 else 0.0
                }
            }

    def export_metrics(self,
                      format: str = "json",
                      output_path: Optional[str] = None,
                      include_raw_data: bool = True) -> str:
        """
        Export training metrics in specified format

        Args:
            format: Export format ("json", "csv", "pandas")
            output_path: Custom output path
            include_raw_data: Whether to include raw episode data

        Returns:
            Path to exported file
        """
        with self.lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if output_path is None:
                output_path = self.log_dir / f"export_{self.session_id}_{timestamp}.{format}"
            else:
                output_path = Path(output_path)

            # Prepare export data
            export_data = {
                "session_info": {
                    "session_id": self.session_id,
                    "start_time": self.session.start_time.isoformat(),
                    "export_time": datetime.now().isoformat(),
                    "total_episodes": len(self.episodes)
                },
                "statistics": self.get_training_statistics(),
                "convergence_analysis": self.get_convergence_analysis()
            }

            if include_raw_data:
                export_data["episodes"] = [asdict(e) for e in self.episodes]
                # Convert datetime objects to strings
                for episode in export_data["episodes"]:
                    episode["timestamp"] = episode["timestamp"].isoformat() if isinstance(episode["timestamp"], datetime) else episode["timestamp"]

            # Export based on format
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)

            elif format.lower() == "csv":
                # Export episodes as CSV
                if self.episodes:
                    df = pd.DataFrame([asdict(e) for e in self.episodes])
                    df['timestamp'] = df['timestamp'].astype(str)
                    df.to_csv(output_path, index=False)
                else:
                    # Create empty CSV with headers
                    headers = [field.name for field in EpisodeMetrics.__dataclass_fields__.values()]
                    pd.DataFrame(columns=headers).to_csv(output_path, index=False)

            elif format.lower() == "pandas":
                # Export as pickle for pandas
                if self.episodes:
                    df = pd.DataFrame([asdict(e) for e in self.episodes])
                    df.to_pickle(output_path)
                else:
                    pd.DataFrame().to_pickle(output_path)

            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.logger.info(f"Metrics exported to: {output_path}")
            return str(output_path)

    def validate_metrics(self) -> Dict[str, Any]:
        """
        Validate metrics collection and identify potential issues

        Returns:
            Validation results and health checks
        """
        with self.lock:
            validation_results = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "checks": {}
            }

            # Data completeness checks
            validation_results["checks"]["data_completeness"] = {
                "total_episodes": len(self.episodes),
                "has_episodes": len(self.episodes) > 0,
                "recent_activity": len(self.metrics_buffer) > 0
            }

            if self.episodes:
                # Metric validity checks
                recent_episodes = self.episodes[-10:] if len(self.episodes) >= 10 else self.episodes

                # Check for reasonable values
                rewards = [e.reward for e in recent_episodes]
                qualities = [e.quality_improvement for e in recent_episodes]

                validation_results["checks"]["metric_validity"] = {
                    "reward_range_ok": all(-100 <= r <= 100 for r in rewards),
                    "quality_range_ok": all(-1 <= q <= 1 for q in qualities),
                    "no_nan_rewards": all(not np.isnan(r) for r in rewards),
                    "no_nan_qualities": all(not np.isnan(q) for q in qualities)
                }

                # Check for training progress
                if len(self.episodes) >= 20:
                    early_rewards = [e.reward for e in self.episodes[:10]]
                    late_rewards = [e.reward for e in self.episodes[-10:]]

                    progress_check = np.mean(late_rewards) > np.mean(early_rewards)
                    validation_results["checks"]["training_progress"] = {
                        "improving": progress_check,
                        "early_avg_reward": float(np.mean(early_rewards)),
                        "recent_avg_reward": float(np.mean(late_rewards))
                    }

                # Check for training stability
                if len(recent_episodes) >= 5:
                    reward_std = np.std([e.reward for e in recent_episodes])
                    quality_std = np.std([e.quality_improvement for e in recent_episodes])

                    validation_results["checks"]["training_stability"] = {
                        "stable_rewards": reward_std < 10.0,
                        "stable_quality": quality_std < 0.1,
                        "reward_std": float(reward_std),
                        "quality_std": float(quality_std)
                    }

            # File system checks
            validation_results["checks"]["file_system"] = {
                "log_dir_exists": self.log_dir.exists(),
                "episodes_file_exists": self.episodes_file.exists(),
                "metrics_file_exists": self.metrics_file.exists(),
                "log_dir_writable": os.access(self.log_dir, os.W_OK)
            }

            # External logging checks
            validation_results["checks"]["external_logging"] = {
                "tensorboard_active": self.tensorboard_writer is not None,
                "wandb_active": self.use_wandb and WANDB_AVAILABLE
            }

            # Overall health score
            all_checks = []
            for category, checks in validation_results["checks"].items():
                if isinstance(checks, dict):
                    all_checks.extend([v for k, v in checks.items() if isinstance(v, bool)])

            health_score = sum(all_checks) / len(all_checks) if all_checks else 0.0
            validation_results["health_score"] = health_score
            validation_results["status"] = "healthy" if health_score > 0.8 else "warning" if health_score > 0.5 else "critical"

            return validation_results

    def create_dashboard_data(self) -> Dict[str, Any]:
        """
        Create real-time dashboard data structure

        Returns:
            Dashboard-ready data for web interface
        """
        with self.lock:
            stats = self.get_training_statistics()
            convergence = self.get_convergence_analysis()
            validation = self.validate_metrics()

            # Recent episodes for live plotting
            recent_episodes = self.episodes[-100:] if len(self.episodes) >= 100 else self.episodes

            dashboard_data = {
                "session": {
                    "id": self.session_id,
                    "status": self.session.status,
                    "uptime_minutes": (time.time() - self.start_time) / 60,
                    "last_updated": datetime.now().isoformat()
                },
                "summary": {
                    "total_episodes": len(self.episodes),
                    "avg_reward": stats.get("reward_stats", {}).get("mean", 0),
                    "avg_quality_improvement": stats.get("quality_stats", {}).get("mean", 0),
                    "success_rate": stats.get("episode_stats", {}).get("success_rate", 0),
                    "health_score": validation.get("health_score", 0)
                },
                "charts": {
                    "episode_rewards": [(e.episode, e.reward) for e in recent_episodes],
                    "quality_improvements": [(e.episode, e.quality_improvement) for e in recent_episodes],
                    "episode_lengths": [(e.episode, e.length) for e in recent_episodes],
                    "success_rate_timeline": self._calculate_success_rate_timeline(recent_episodes)
                },
                "convergence": convergence,
                "validation": validation,
                "logo_performance": stats.get("logo_type_performance", {}),
                "algorithm_metrics": {
                    name: {
                        "latest": values[-1] if values else 0,
                        "recent_avg": np.mean(values[-10:]) if len(values) >= 10 else (np.mean(values) if values else 0)
                    }
                    for name, values in self.aggregated_metrics.items()
                }
            }

            return dashboard_data

    def _calculate_success_rate_timeline(self, episodes: List[EpisodeMetrics], window: int = 10) -> List[Tuple[int, float]]:
        """Calculate rolling success rate timeline"""
        if len(episodes) < window:
            return []

        timeline = []
        for i in range(window, len(episodes) + 1):
            window_episodes = episodes[i-window:i]
            success_rate = np.mean([e.success for e in window_episodes])
            timeline.append((episodes[i-1].episode, success_rate))

        return timeline

    def save_session(self) -> None:
        """Save session metadata"""
        with self.lock:
            self.session.end_time = datetime.now()
            self.session.total_episodes = len(self.episodes)

            session_data = asdict(self.session)
            session_data['start_time'] = self.session.start_time.isoformat()
            if self.session.end_time:
                session_data['end_time'] = self.session.end_time.isoformat()

            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)

    def close(self) -> None:
        """Close monitor and clean up resources"""
        with self.lock:
            self.logger.info(f"Closing training monitor for session: {self.session_id}")

            # Close external loggers
            if self.tensorboard_writer:
                self.tensorboard_writer.close()

            if self.use_wandb and WANDB_AVAILABLE:
                wandb.finish()

            # Save final session data
            self.save_session()

            # Final statistics
            final_stats = self.get_training_statistics()
            self.logger.info(f"Final training statistics: {final_stats['session_info']}")

            self.session.status = "completed"


# Convenience functions for integration
def create_training_monitor(log_dir: str = "logs/ppo_training",
                          session_name: Optional[str] = None,
                          use_tensorboard: bool = True,
                          use_wandb: bool = False) -> TrainingMonitor:
    """
    Create a training monitor with sensible defaults

    Args:
        log_dir: Directory for logs
        session_name: Custom session name
        use_tensorboard: Enable TensorBoard logging
        use_wandb: Enable Weights & Biases logging

    Returns:
        Configured TrainingMonitor instance
    """
    return TrainingMonitor(
        log_dir=log_dir,
        session_name=session_name,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb
    )


def validate_training_monitor_installation() -> Dict[str, bool]:
    """
    Validate that all required dependencies are available

    Returns:
        Dictionary of available features
    """
    return {
        "tensorboard": TENSORBOARD_AVAILABLE,
        "wandb": WANDB_AVAILABLE,
        "matplotlib": True,  # Required dependency
        "pandas": True,      # Required dependency
        "numpy": True,       # Required dependency
        "psutil": True       # For memory monitoring
    }