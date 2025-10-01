# backend/ai_modules/optimization/training_orchestrator.py
"""Training orchestration system for comprehensive PPO agent training"""

import os
import json
import time
import logging
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
import numpy as np

from .training_pipeline import CurriculumTrainingPipeline
from .ppo_optimizer import PPOVTracerOptimizer
from .agent_interface import VTracerAgentInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfiguration:
    """Complete training configuration"""
    experiment_name: str
    training_data_path: str
    validation_data_path: str
    output_dir: str

    # Curriculum settings
    use_curriculum: bool = True
    curriculum_config: Optional[Dict] = None

    # Model settings
    model_config: Optional[Dict] = None

    # Training settings
    max_parallel_jobs: int = 2
    enable_hyperparameter_search: bool = False
    hyperparameter_search_trials: int = 10

    # Evaluation settings
    validation_frequency: int = 5000
    evaluation_images_per_type: int = 5

    # Resource management
    memory_limit_gb: int = 8
    gpu_memory_fraction: float = 0.8

    # Monitoring settings
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10000
    monitoring_frequency: int = 1000


@dataclass
class TrainingJobResult:
    """Result from a training job"""
    job_id: str
    experiment_name: str
    success: bool
    training_time: float
    best_quality: float
    final_model_path: str
    metrics: Dict[str, Any]
    error_message: Optional[str] = None


class TrainingDataManager:
    """Manages training data organization and loading"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.image_categories = {}
        self._scan_training_data()

    def _scan_training_data(self) -> None:
        """Scan and categorize training data"""
        logger.info(f"Scanning training data in: {self.data_path}")

        # Expected directory structure:
        # data/
        #   simple/
        #   text/
        #   gradient/
        #   complex/

        categories = ['simple', 'text', 'gradient', 'complex']

        for category in categories:
            category_path = self.data_path / category
            if category_path.exists():
                image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    image_files.extend(category_path.glob(ext))

                self.image_categories[category] = [str(f) for f in image_files]
                logger.info(f"Found {len(image_files)} {category} images")
            else:
                logger.warning(f"Category directory not found: {category_path}")
                self.image_categories[category] = []

    def get_training_images(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """Get training images by category"""
        if category:
            return {category: self.image_categories.get(category, [])}
        return self.image_categories.copy()

    def get_validation_split(self, validation_ratio: float = 0.2) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Split data into training and validation sets"""
        training_data = {}
        validation_data = {}

        for category, images in self.image_categories.items():
            if not images:
                training_data[category] = []
                validation_data[category] = []
                continue

            # Shuffle and split
            shuffled = np.random.permutation(images)
            split_idx = int(len(shuffled) * (1 - validation_ratio))

            training_data[category] = shuffled[:split_idx].tolist()
            validation_data[category] = shuffled[split_idx:].tolist()

        return training_data, validation_data


class TrainingMonitor:
    """Monitors training progress and performance"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_log = []
        self.alerts = []

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log training metrics"""
        timestamp = time.time()
        metric_entry = {
            'timestamp': timestamp,
            'time_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
            **metrics
        }
        self.metrics_log.append(metric_entry)

        # Save metrics to file
        metrics_file = self.output_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)

    def check_training_health(self, recent_metrics: List[Dict[str, Any]]) -> List[str]:
        """Check training health and generate alerts"""
        alerts = []

        if len(recent_metrics) < 5:
            return alerts  # Not enough data

        # Check for stagnant learning
        recent_qualities = [m.get('average_quality', 0) for m in recent_metrics[-5:]]
        if all(abs(q - recent_qualities[0]) < 0.01 for q in recent_qualities):
            alerts.append("Warning: Learning appears stagnant (quality not improving)")

        # Check for performance degradation
        if len(recent_qualities) >= 3:
            trend = np.polyfit(range(len(recent_qualities)), recent_qualities, 1)[0]
            if trend < -0.01:
                alerts.append("Warning: Performance degrading over recent episodes")

        # Check for abnormally low success rates
        recent_success_rates = [m.get('success_rate', 0) for m in recent_metrics[-3:]]
        if all(sr < 0.3 for sr in recent_success_rates):
            alerts.append("Alert: Very low success rates detected")

        self.alerts.extend(alerts)
        return alerts

    def generate_monitoring_report(self) -> str:
        """Generate monitoring report"""
        if not self.metrics_log:
            return "No training metrics available"

        report = []
        report.append("# Training Monitoring Report")
        report.append("=" * 40)

        # Recent performance
        recent_metrics = self.metrics_log[-10:] if len(self.metrics_log) >= 10 else self.metrics_log

        if recent_metrics:
            avg_quality = np.mean([m.get('average_quality', 0) for m in recent_metrics])
            avg_success_rate = np.mean([m.get('success_rate', 0) for m in recent_metrics])

            report.append("## Recent Performance (Last 10 entries)")
            report.append(f"- Average Quality: {avg_quality:.4f}")
            report.append(f"- Average Success Rate: {avg_success_rate:.2%}")
            report.append("")

        # Alerts
        if self.alerts:
            report.append("## Alerts")
            for alert in self.alerts[-5:]:  # Show last 5 alerts
                report.append(f"- {alert}")
            report.append("")

        return "\n".join(report)


class HyperparameterOptimizer:
    """Optimizes hyperparameters for training"""

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.search_space = self._define_search_space()

    def _define_search_space(self) -> Dict[str, List]:
        """Define hyperparameter search space"""
        return {
            'learning_rate': [1e-4, 3e-4, 1e-3],
            'batch_size': [32, 64, 128],
            'n_steps': [1024, 2048, 4096],
            'ent_coef': [0.001, 0.01, 0.1],
            'clip_range': [0.1, 0.2, 0.3]
        }

    def suggest_hyperparameters(self, trial_id: int) -> Dict[str, Any]:
        """Suggest hyperparameters for trial"""
        config = self.base_config.copy()

        # Simple grid search approach
        import itertools

        # Generate all combinations
        keys, values = zip(*self.search_space.items())
        combinations = list(itertools.product(*values))

        # Select combination for this trial
        combination = combinations[trial_id % len(combinations)]

        # Update config
        for key, value in zip(keys, combination):
            config[key] = value

        return config


class TrainingOrchestrator:
    """Orchestrates comprehensive training pipeline"""

    def __init__(self, config: TrainingConfiguration):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.data_manager = TrainingDataManager(config.training_data_path)
        self.monitor = TrainingMonitor(str(self.output_dir / "monitoring"))
        self.hyperparameter_optimizer = HyperparameterOptimizer(config.model_config or {})

        # Training state
        self.active_jobs = {}
        self.completed_jobs = []

        logger.info(f"Training Orchestrator initialized for experiment: {config.experiment_name}")

    def run_training_experiment(self) -> Dict[str, Any]:
        """Run complete training experiment"""
        logger.info(f"🚀 Starting training experiment: {self.config.experiment_name}")

        experiment_start_time = time.time()

        try:
            # Prepare data
            training_data, validation_data = self._prepare_training_data()

            # Run training based on configuration
            if self.config.enable_hyperparameter_search:
                results = self._run_hyperparameter_search(training_data, validation_data)
            else:
                results = self._run_single_training(training_data, validation_data)

            # Evaluate final models
            evaluation_results = self._evaluate_trained_models()

            # Generate final report
            experiment_results = {
                'experiment_name': self.config.experiment_name,
                'total_time': time.time() - experiment_start_time,
                'training_results': results,
                'evaluation_results': evaluation_results,
                'config': asdict(self.config),
                'data_summary': self._get_data_summary()
            }

            # Save experiment results
            self._save_experiment_results(experiment_results)

            logger.info(f"✅ Training experiment completed: {self.config.experiment_name}")
            return experiment_results

        except Exception as e:
            logger.error(f"❌ Training experiment failed: {e}")
            raise

    def _prepare_training_data(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Prepare training and validation data"""
        logger.info("Preparing training data...")

        # Get validation data if specified
        if self.config.validation_data_path and Path(self.config.validation_data_path).exists():
            validation_manager = TrainingDataManager(self.config.validation_data_path)
            training_data = self.data_manager.get_training_images()
            validation_data = validation_manager.get_training_images()
        else:
            # Split existing data
            training_data, validation_data = self.data_manager.get_validation_split(0.2)

        logger.info(f"Training data: {[(k, len(v)) for k, v in training_data.items()]}")
        logger.info(f"Validation data: {[(k, len(v)) for k, v in validation_data.items()]}")

        return training_data, validation_data

    def _run_single_training(self, training_data: Dict[str, List[str]],
                           validation_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """Run single training job"""
        logger.info("Running single training job...")

        if self.config.use_curriculum:
            # Use curriculum training
            pipeline = CurriculumTrainingPipeline(
                training_images=training_data,
                model_config=self.config.model_config,
                save_dir=str(self.output_dir / "curriculum_training")
            )

            results = pipeline.run_curriculum()

            # Generate visualization
            pipeline.visualize_curriculum_progress(
                str(self.output_dir / "curriculum_progress.png")
            )

            pipeline.close()

        else:
            # Use standard training
            results = self._run_standard_training(training_data)

        return results

    def _run_standard_training(self, training_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """Run standard (non-curriculum) training"""
        logger.info("Running standard training...")

        # Select primary training image
        all_images = []
        for images in training_data.values():
            all_images.extend(images)

        if not all_images:
            raise ValueError("No training images available")

        primary_image = all_images[0]

        # Create agent interface
        agent = VTracerAgentInterface(
            model_save_dir=str(self.output_dir / "standard_training"),
            config_file=None
        )

        # Train agent
        training_results = agent.train_agent(
            training_image=primary_image,
            training_timesteps=50000
        )

        agent.close()

        return {
            'training_method': 'standard',
            'primary_image': primary_image,
            'results': training_results
        }

    def _run_hyperparameter_search(self, training_data: Dict[str, List[str]],
                                 validation_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """Run hyperparameter search"""
        logger.info(f"Running hyperparameter search with {self.config.hyperparameter_search_trials} trials...")

        trial_results = []

        for trial_id in range(self.config.hyperparameter_search_trials):
            logger.info(f"Starting hyperparameter trial {trial_id + 1}/{self.config.hyperparameter_search_trials}")

            # Get hyperparameters for this trial
            trial_config = self.hyperparameter_optimizer.suggest_hyperparameters(trial_id)

            try:
                # Run training with these hyperparameters
                trial_result = self._run_hyperparameter_trial(trial_id, trial_config, training_data)
                trial_results.append(trial_result)

                logger.info(f"Trial {trial_id + 1} completed - Quality: {trial_result.get('best_quality', 0):.4f}")

            except Exception as e:
                logger.error(f"Trial {trial_id + 1} failed: {e}")
                trial_results.append({
                    'trial_id': trial_id,
                    'hyperparameters': trial_config,
                    'success': False,
                    'error': str(e)
                })

        # Find best trial
        successful_trials = [t for t in trial_results if t.get('success', False)]

        if successful_trials:
            best_trial = max(successful_trials, key=lambda x: x.get('best_quality', 0))
            logger.info(f"Best trial: {best_trial['trial_id']} with quality {best_trial['best_quality']:.4f}")
        else:
            best_trial = None
            logger.warning("No successful hyperparameter trials")

        return {
            'search_method': 'hyperparameter_search',
            'total_trials': len(trial_results),
            'successful_trials': len(successful_trials),
            'trial_results': trial_results,
            'best_trial': best_trial
        }

    def _run_hyperparameter_trial(self, trial_id: int, hyperparameters: Dict[str, Any],
                                training_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """Run single hyperparameter trial"""

        # Select training image
        all_images = []
        for images in training_data.values():
            all_images.extend(images)

        if not all_images:
            raise ValueError("No training images available")

        primary_image = all_images[trial_id % len(all_images)]  # Rotate through images

        # Create trial-specific directory
        trial_dir = self.output_dir / f"trial_{trial_id}"
        trial_dir.mkdir(exist_ok=True)

        # Create agent with trial hyperparameters
        agent = VTracerAgentInterface(model_save_dir=str(trial_dir))
        agent.config['model'].update(hyperparameters)

        # Run training (shorter for hyperparameter search)
        training_results = agent.train_agent(
            training_image=primary_image,
            training_timesteps=20000  # Reduced for faster trials
        )

        agent.close()

        return {
            'trial_id': trial_id,
            'hyperparameters': hyperparameters,
            'training_image': primary_image,
            'success': True,
            'best_quality': training_results.get('best_quality', 0),
            'training_time': training_results.get('training_time', 0),
            'model_path': str(trial_dir)
        }

    def _evaluate_trained_models(self) -> Dict[str, Any]:
        """Evaluate all trained models"""
        logger.info("Evaluating trained models...")

        # Find all trained models
        model_dirs = []
        for item in self.output_dir.rglob("best_model*"):
            if item.is_file() or item.is_dir():
                model_dirs.append(item.parent)

        if not model_dirs:
            logger.warning("No trained models found for evaluation")
            return {'models_evaluated': 0, 'results': []}

        evaluation_results = []

        for model_dir in model_dirs:
            try:
                result = self._evaluate_single_model(model_dir)
                evaluation_results.append(result)
            except Exception as e:
                logger.error(f"Evaluation failed for {model_dir}: {e}")

        return {
            'models_evaluated': len(evaluation_results),
            'results': evaluation_results,
            'best_model': max(evaluation_results, key=lambda x: x.get('average_quality', 0)) if evaluation_results else None
        }

    def _evaluate_single_model(self, model_dir: Path) -> Dict[str, Any]:
        """Evaluate single trained model"""
        logger.info(f"Evaluating model: {model_dir}")

        # Get validation data
        _, validation_data = self._prepare_training_data()

        # Create agent interface
        agent = VTracerAgentInterface(model_save_dir=str(model_dir))

        # Evaluate on validation images
        eval_images = []
        for category, images in validation_data.items():
            eval_images.extend(images[:self.config.evaluation_images_per_type])

        if eval_images:
            evaluation_result = agent.evaluate_performance(eval_images, episodes_per_image=3)
        else:
            evaluation_result = {'average_quality': 0.0, 'target_reached_rate': 0.0}

        agent.close()

        return {
            'model_dir': str(model_dir),
            'evaluation_images': len(eval_images),
            **evaluation_result
        }

    def _get_data_summary(self) -> Dict[str, Any]:
        """Get summary of training data"""
        return {
            'categories': list(self.data_manager.image_categories.keys()),
            'image_counts': {k: len(v) for k, v in self.data_manager.image_categories.items()},
            'total_images': sum(len(v) for v in self.data_manager.image_categories.values())
        }

    def _save_experiment_results(self, results: Dict[str, Any]) -> None:
        """Save experiment results"""
        results_file = self.output_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save human-readable report
        report_file = self.output_dir / "experiment_report.txt"
        with open(report_file, 'w') as f:
            f.write(self._generate_experiment_report(results))

        logger.info(f"Experiment results saved to: {results_file}")
        logger.info(f"Experiment report saved to: {report_file}")

    def _generate_experiment_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable experiment report"""
        report = []
        report.append(f"# Training Experiment Report: {results['experiment_name']}")
        report.append("=" * 60)
        report.append("")

        # Experiment summary
        report.append("## Experiment Summary")
        report.append(f"- Total Time: {results['total_time']:.2f} seconds")
        report.append(f"- Data Summary: {results['data_summary']}")
        report.append("")

        # Training results
        training_results = results['training_results']
        if 'search_method' in training_results:
            if training_results['search_method'] == 'hyperparameter_search':
                report.append("## Hyperparameter Search Results")
                report.append(f"- Total Trials: {training_results['total_trials']}")
                report.append(f"- Successful Trials: {training_results['successful_trials']}")
                if training_results['best_trial']:
                    best = training_results['best_trial']
                    report.append(f"- Best Trial Quality: {best['best_quality']:.4f}")
                    report.append(f"- Best Hyperparameters: {best['hyperparameters']}")
        else:
            report.append("## Training Results")
            if 'success_rate' in training_results:
                report.append(f"- Success Rate: {training_results['success_rate']:.2%}")
            if 'final_performance' in training_results:
                perf = training_results['final_performance']
                report.append(f"- Average Quality: {perf['average_quality']:.4f}")

        report.append("")

        # Evaluation results
        eval_results = results['evaluation_results']
        report.append("## Evaluation Results")
        report.append(f"- Models Evaluated: {eval_results['models_evaluated']}")
        if eval_results['best_model']:
            best = eval_results['best_model']
            report.append(f"- Best Model Quality: {best['average_quality']:.4f}")
            report.append(f"- Best Model Path: {best['model_dir']}")

        return "\n".join(report)


# Factory function for easy usage
def create_training_orchestrator(experiment_name: str,
                               training_data_path: str,
                               output_dir: str,
                               **kwargs) -> TrainingOrchestrator:
    """
    Factory function to create training orchestrator

    Args:
        experiment_name: Name of the experiment
        training_data_path: Path to training data
        output_dir: Output directory for results
        **kwargs: Additional configuration options

    Returns:
        Configured training orchestrator
    """
    config = TrainingConfiguration(
        experiment_name=experiment_name,
        training_data_path=training_data_path,
        validation_data_path=kwargs.get('validation_data_path', ''),
        output_dir=output_dir,
        **kwargs
    )

    return TrainingOrchestrator(config)