# backend/ai_modules/optimization/validation_framework.py
"""Comprehensive validation framework for periodic model evaluation"""

import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics"""
    quality_score: float
    ssim_score: float
    mse_score: float
    psnr_score: float
    processing_time: float
    target_reached: bool
    convergence_steps: int
    stability_score: float
    consistency_score: float


@dataclass
class ValidationResult:
    """Result from validation evaluation"""
    image_path: str
    category: str
    metrics: ValidationMetrics
    evaluation_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    validation_id: str
    timestamp: float
    model_checkpoint_id: str
    total_images: int
    successful_evaluations: int
    category_results: Dict[str, List[ValidationResult]]
    aggregate_metrics: Dict[str, float]
    performance_trends: Dict[str, List[float]]
    recommendations: List[str]
    validation_time: float


@dataclass
class ValidationConfig:
    """Configuration for validation protocols"""
    validation_frequency: int = 5000  # Validate every N training steps
    validation_images_per_category: int = 5
    evaluation_episodes_per_image: int = 3
    quality_thresholds: Dict[str, float] = None
    stability_test_episodes: int = 5
    consistency_test_runs: int = 3
    timeout_per_image: float = 300.0  # 5 minutes per image
    parallel_evaluations: int = 2
    save_detailed_results: bool = True
    generate_visualizations: bool = True

    def __post_init__(self):
        if self.quality_thresholds is None:
            self.quality_thresholds = {
                'simple': 0.85,
                'text': 0.80,
                'gradient': 0.75,
                'complex': 0.70
            }


class ValidationDatasetManager:
    """Manages validation datasets and sampling strategies"""

    def __init__(self, validation_data: Dict[str, List[str]]):
        """
        Initialize validation dataset manager

        Args:
            validation_data: Dictionary mapping categories to validation image paths
        """
        self.validation_data = validation_data
        self.category_indices = {cat: 0 for cat in validation_data.keys()}
        self.evaluation_history = defaultdict(list)

    def get_validation_batch(self,
                           images_per_category: int = 5,
                           strategy: str = 'round_robin') -> Dict[str, List[str]]:
        """
        Get batch of validation images

        Args:
            images_per_category: Number of images per category
            strategy: Sampling strategy ('round_robin', 'random', 'least_evaluated')

        Returns:
            Dictionary mapping categories to selected image paths
        """
        batch = {}

        for category, images in self.validation_data.items():
            if not images:
                batch[category] = []
                continue

            if strategy == 'round_robin':
                selected = self._round_robin_sample(category, images, images_per_category)
            elif strategy == 'random':
                selected = self._random_sample(images, images_per_category)
            elif strategy == 'least_evaluated':
                selected = self._least_evaluated_sample(category, images, images_per_category)
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")

            batch[category] = selected

        return batch

    def _round_robin_sample(self, category: str, images: List[str], count: int) -> List[str]:
        """Sample images using round-robin strategy"""
        start_idx = self.category_indices[category]
        selected = []

        for i in range(count):
            idx = (start_idx + i) % len(images)
            selected.append(images[idx])

        # Update index for next sampling
        self.category_indices[category] = (start_idx + count) % len(images)
        return selected

    def _random_sample(self, images: List[str], count: int) -> List[str]:
        """Sample images randomly"""
        return list(np.random.choice(images, size=min(count, len(images)), replace=False))

    def _least_evaluated_sample(self, category: str, images: List[str], count: int) -> List[str]:
        """Sample least evaluated images"""
        # Count evaluations per image
        evaluation_counts = defaultdict(int)
        for image_path in self.evaluation_history[category]:
            evaluation_counts[image_path] += 1

        # Sort by evaluation count (ascending)
        sorted_images = sorted(images, key=lambda x: evaluation_counts[x])
        return sorted_images[:count]

    def record_evaluation(self, category: str, image_path: str) -> None:
        """Record that an image was evaluated"""
        self.evaluation_history[category].append(image_path)


class ModelEvaluator:
    """Evaluates model performance on validation data"""

    def __init__(self, agent_interface):
        """
        Initialize model evaluator

        Args:
            agent_interface: Agent interface for model evaluation
        """
        self.agent_interface = agent_interface

    def evaluate_image(self,
                      image_path: str,
                      category: str,
                      episodes: int = 3,
                      stability_test: bool = True) -> ValidationResult:
        """
        Evaluate model on single image

        Args:
            image_path: Path to validation image
            category: Image category
            episodes: Number of evaluation episodes
            stability_test: Whether to perform stability testing

        Returns:
            ValidationResult with comprehensive metrics
        """
        start_time = time.time()

        try:
            # Basic evaluation
            evaluation_results = []
            for episode in range(episodes):
                result = self.agent_interface.optimize_parameters(
                    image_path,
                    max_episodes=1,
                    deterministic=True
                )
                evaluation_results.append(result)

            # Extract metrics from results
            qualities = [r.get('best_quality', 0) for r in evaluation_results]
            processing_times = [r.get('processing_time', 0) for r in evaluation_results]
            targets_reached = [r.get('target_reached', False) for r in evaluation_results]

            # Calculate aggregate metrics
            avg_quality = np.mean(qualities)
            avg_processing_time = np.mean(processing_times)
            target_reached_rate = np.mean(targets_reached)

            # Stability test
            stability_score = 1.0 - np.std(qualities) if len(qualities) > 1 else 1.0

            # Consistency test (multiple independent runs)
            if stability_test:
                consistency_results = []
                for _ in range(3):  # 3 independent runs
                    result = self.agent_interface.optimize_parameters(
                        image_path,
                        max_episodes=1,
                        deterministic=False  # Allow stochasticity
                    )
                    consistency_results.append(result.get('best_quality', 0))

                consistency_score = 1.0 - np.std(consistency_results) if len(consistency_results) > 1 else 1.0
            else:
                consistency_score = 1.0

            # Create metrics object
            metrics = ValidationMetrics(
                quality_score=avg_quality,
                ssim_score=avg_quality,  # Assuming quality is SSIM-based
                mse_score=1.0 - avg_quality,  # Inverse relationship
                psnr_score=20 * np.log10(1.0 / max(1.0 - avg_quality, 1e-8)),
                processing_time=avg_processing_time,
                target_reached=target_reached_rate > 0.5,
                convergence_steps=int(np.mean([r.get('episodes_run', 1) for r in evaluation_results])),
                stability_score=stability_score,
                consistency_score=consistency_score
            )

            evaluation_time = time.time() - start_time

            return ValidationResult(
                image_path=image_path,
                category=category,
                metrics=metrics,
                evaluation_time=evaluation_time,
                success=True
            )

        except Exception as e:
            evaluation_time = time.time() - start_time
            logger.error(f"Evaluation failed for {image_path}: {e}")

            # Return failed result with default metrics
            metrics = ValidationMetrics(
                quality_score=0.0,
                ssim_score=0.0,
                mse_score=1.0,
                psnr_score=0.0,
                processing_time=0.0,
                target_reached=False,
                convergence_steps=0,
                stability_score=0.0,
                consistency_score=0.0
            )

            return ValidationResult(
                image_path=image_path,
                category=category,
                metrics=metrics,
                evaluation_time=evaluation_time,
                success=False,
                error_message=str(e)
            )


class ValidationProtocol:
    """Implements validation protocols and manages validation execution"""

    def __init__(self,
                 agent_interface,
                 validation_data: Dict[str, List[str]],
                 config: Optional[ValidationConfig] = None,
                 save_dir: Optional[str] = None):
        """
        Initialize validation protocol

        Args:
            agent_interface: Agent interface for model evaluation
            validation_data: Validation dataset
            config: Validation configuration
            save_dir: Directory to save validation results
        """
        self.agent_interface = agent_interface
        self.config = config or ValidationConfig()
        self.save_dir = Path(save_dir) if save_dir else Path('validation_results')
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.dataset_manager = ValidationDatasetManager(validation_data)
        self.evaluator = ModelEvaluator(agent_interface)

        # Validation tracking
        self.validation_history: List[ValidationReport] = []
        self.performance_trends = defaultdict(list)

        logger.info(f"ValidationProtocol initialized with {sum(len(imgs) for imgs in validation_data.values())} validation images")

    def run_validation(self,
                      checkpoint_id: str,
                      training_step: int,
                      force_validation: bool = False) -> Optional[ValidationReport]:
        """
        Run comprehensive validation evaluation

        Args:
            checkpoint_id: ID of the model checkpoint being validated
            training_step: Current training step
            force_validation: Force validation even if not due

        Returns:
            ValidationReport if validation was performed, None if skipped
        """
        # Check if validation is due
        if not force_validation and not self._is_validation_due(training_step):
            return None

        logger.info(f"🔍 Starting validation at step {training_step}")
        validation_start_time = time.time()

        # Generate validation ID
        validation_id = f"validation_{training_step}_{int(time.time())}"

        # Get validation batch
        validation_batch = self.dataset_manager.get_validation_batch(
            self.config.validation_images_per_category,
            strategy='round_robin'
        )

        # Run evaluations
        all_results = []
        category_results = {}

        for category, images in validation_batch.items():
            if not images:
                category_results[category] = []
                continue

            logger.info(f"Evaluating {len(images)} {category} images...")

            # Evaluate images in parallel
            category_eval_results = self._evaluate_images_parallel(
                images,
                category,
                self.config.evaluation_episodes_per_image
            )

            category_results[category] = category_eval_results
            all_results.extend(category_eval_results)

            # Record evaluations
            for image_path in images:
                self.dataset_manager.record_evaluation(category, image_path)

        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(all_results)

        # Analyze performance trends
        performance_trends = self._analyze_performance_trends(category_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(category_results, aggregate_metrics)

        # Create validation report
        validation_time = time.time() - validation_start_time

        report = ValidationReport(
            validation_id=validation_id,
            timestamp=time.time(),
            model_checkpoint_id=checkpoint_id,
            total_images=len(all_results),
            successful_evaluations=sum(1 for r in all_results if r.success),
            category_results=category_results,
            aggregate_metrics=aggregate_metrics,
            performance_trends=performance_trends,
            recommendations=recommendations,
            validation_time=validation_time
        )

        # Save validation results
        if self.config.save_detailed_results:
            self._save_validation_report(report)

        # Generate visualizations
        if self.config.generate_visualizations:
            self._generate_validation_visualizations(report)

        # Update tracking
        self.validation_history.append(report)
        self._update_performance_trends(aggregate_metrics)

        logger.info(f"✅ Validation completed in {validation_time:.2f}s")
        logger.info(f"Overall quality: {aggregate_metrics.get('avg_quality', 0):.4f}")
        logger.info(f"Success rate: {aggregate_metrics.get('success_rate', 0):.2%}")

        return report

    def _is_validation_due(self, training_step: int) -> bool:
        """Check if validation is due based on training step"""
        if not self.validation_history:
            return True  # First validation

        last_validation_step = getattr(self.validation_history[-1], 'training_step', 0)
        return training_step - last_validation_step >= self.config.validation_frequency

    def _evaluate_images_parallel(self,
                                 images: List[str],
                                 category: str,
                                 episodes: int) -> List[ValidationResult]:
        """Evaluate images in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=self.config.parallel_evaluations) as executor:
            # Submit evaluation tasks
            future_to_image = {
                executor.submit(
                    self.evaluator.evaluate_image,
                    image,
                    category,
                    episodes
                ): image for image in images
            }

            # Collect results
            for future in as_completed(future_to_image, timeout=self.config.timeout_per_image * len(images)):
                try:
                    result = future.result(timeout=self.config.timeout_per_image)
                    results.append(result)
                except Exception as e:
                    image = future_to_image[future]
                    logger.error(f"Evaluation failed for {image}: {e}")
                    # Create failed result
                    failed_result = ValidationResult(
                        image_path=image,
                        category=category,
                        metrics=ValidationMetrics(
                            quality_score=0.0, ssim_score=0.0, mse_score=1.0,
                            psnr_score=0.0, processing_time=0.0, target_reached=False,
                            convergence_steps=0, stability_score=0.0, consistency_score=0.0
                        ),
                        evaluation_time=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    results.append(failed_result)

        return results

    def _calculate_aggregate_metrics(self, results: List[ValidationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics from validation results"""
        if not results:
            return {}

        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {
                'avg_quality': 0.0,
                'success_rate': 0.0,
                'avg_processing_time': 0.0,
                'stability_score': 0.0,
                'consistency_score': 0.0
            }

        # Calculate averages
        qualities = [r.metrics.quality_score for r in successful_results]
        processing_times = [r.metrics.processing_time for r in successful_results]
        stability_scores = [r.metrics.stability_score for r in successful_results]
        consistency_scores = [r.metrics.consistency_score for r in successful_results]

        return {
            'avg_quality': np.mean(qualities),
            'quality_std': np.std(qualities),
            'quality_min': np.min(qualities),
            'quality_max': np.max(qualities),
            'success_rate': len(successful_results) / len(results),
            'avg_processing_time': np.mean(processing_times),
            'processing_time_std': np.std(processing_times),
            'avg_stability_score': np.mean(stability_scores),
            'avg_consistency_score': np.mean(consistency_scores),
            'target_reached_rate': np.mean([r.metrics.target_reached for r in successful_results])
        }

    def _analyze_performance_trends(self, category_results: Dict[str, List[ValidationResult]]) -> Dict[str, List[float]]:
        """Analyze performance trends across categories"""
        trends = {}

        for category, results in category_results.items():
            successful_results = [r for r in results if r.success]
            if successful_results:
                trends[f'{category}_quality'] = [r.metrics.quality_score for r in successful_results]
                trends[f'{category}_processing_time'] = [r.metrics.processing_time for r in successful_results]
                trends[f'{category}_stability'] = [r.metrics.stability_score for r in successful_results]

        return trends

    def _generate_recommendations(self,
                                category_results: Dict[str, List[ValidationResult]],
                                aggregate_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        # Overall performance check
        avg_quality = aggregate_metrics.get('avg_quality', 0)
        if avg_quality < 0.5:
            recommendations.append("Overall model performance is low - consider extended training")
        elif avg_quality < 0.7:
            recommendations.append("Model performance is moderate - may benefit from hyperparameter tuning")

        # Success rate check
        success_rate = aggregate_metrics.get('success_rate', 0)
        if success_rate < 0.8:
            recommendations.append("Low evaluation success rate - check for implementation issues")

        # Category-specific recommendations
        for category, results in category_results.items():
            successful_results = [r for r in results if r.success]
            if not successful_results:
                recommendations.append(f"No successful evaluations for {category} images - investigate category-specific issues")
                continue

            avg_category_quality = np.mean([r.metrics.quality_score for r in successful_results])
            threshold = self.config.quality_thresholds.get(category, 0.7)

            if avg_category_quality < threshold:
                recommendations.append(f"{category} performance below threshold ({avg_category_quality:.3f} < {threshold}) - consider category-specific training")

            # Stability check
            avg_stability = np.mean([r.metrics.stability_score for r in successful_results])
            if avg_stability < 0.8:
                recommendations.append(f"{category} results show low stability - consider increasing training regularization")

        # Processing time check
        avg_processing_time = aggregate_metrics.get('avg_processing_time', 0)
        if avg_processing_time > 60:  # 1 minute
            recommendations.append("High processing times detected - consider model optimization")

        return recommendations

    def _save_validation_report(self, report: ValidationReport) -> None:
        """Save validation report to disk"""
        report_file = self.save_dir / f"validation_report_{report.validation_id}.json"

        # Convert report to serializable format
        report_dict = asdict(report)

        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Validation report saved to: {report_file}")

    def _generate_validation_visualizations(self, report: ValidationReport) -> None:
        """Generate validation visualization plots"""
        try:
            # Create visualization directory
            viz_dir = self.save_dir / f"visualizations_{report.validation_id}"
            viz_dir.mkdir(exist_ok=True)

            # Plot 1: Quality scores by category
            self._plot_quality_by_category(report, viz_dir / "quality_by_category.png")

            # Plot 2: Performance trends over time
            if len(self.validation_history) > 1:
                self._plot_performance_trends(viz_dir / "performance_trends.png")

            # Plot 3: Processing time distribution
            self._plot_processing_time_distribution(report, viz_dir / "processing_time_dist.png")

            logger.info(f"Validation visualizations saved to: {viz_dir}")

        except Exception as e:
            logger.error(f"Failed to generate validation visualizations: {e}")

    def _plot_quality_by_category(self, report: ValidationReport, save_path: Path) -> None:
        """Plot quality scores by category"""
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = []
        qualities = []
        colors = []

        color_map = {'simple': 'green', 'text': 'blue', 'gradient': 'orange', 'complex': 'red'}

        for category, results in report.category_results.items():
            successful_results = [r for r in results if r.success]
            if successful_results:
                category_qualities = [r.metrics.quality_score for r in successful_results]
                categories.extend([category] * len(category_qualities))
                qualities.extend(category_qualities)
                colors.extend([color_map.get(category, 'gray')] * len(category_qualities))

        if categories and qualities:
            # Create box plot
            category_data = {}
            for cat, qual in zip(categories, qualities):
                if cat not in category_data:
                    category_data[cat] = []
                category_data[cat].append(qual)

            ax.boxplot([category_data.get(cat, []) for cat in ['simple', 'text', 'gradient', 'complex']],
                      labels=['Simple', 'Text', 'Gradient', 'Complex'])

            ax.set_title('Quality Scores by Category', fontsize=14, fontweight='bold')
            ax.set_ylabel('Quality Score (SSIM)')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            # Add threshold lines
            for i, category in enumerate(['simple', 'text', 'gradient', 'complex']):
                threshold = self.config.quality_thresholds.get(category, 0.7)
                ax.axhline(y=threshold, color=color_map.get(category, 'gray'),
                          linestyle='--', alpha=0.7, label=f'{category.title()} threshold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_trends(self, save_path: Path) -> None:
        """Plot performance trends over validation history"""
        if len(self.validation_history) < 2:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Extract data from validation history
        timestamps = [report.timestamp for report in self.validation_history]
        avg_qualities = [report.aggregate_metrics.get('avg_quality', 0) for report in self.validation_history]
        success_rates = [report.aggregate_metrics.get('success_rate', 0) for report in self.validation_history]

        # Plot 1: Average quality over time
        ax1.plot(timestamps, avg_qualities, 'b-o', linewidth=2, markersize=6)
        ax1.set_title('Average Quality Trend', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Quality Score')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Plot 2: Success rate over time
        ax2.plot(timestamps, success_rates, 'g-o', linewidth=2, markersize=6)
        ax2.set_title('Success Rate Trend', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Success Rate')
        ax2.set_xlabel('Validation Timestamp')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_processing_time_distribution(self, report: ValidationReport, save_path: Path) -> None:
        """Plot processing time distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))

        all_times = []
        for results in report.category_results.values():
            successful_results = [r for r in results if r.success]
            times = [r.metrics.processing_time for r in successful_results]
            all_times.extend(times)

        if all_times:
            ax.hist(all_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title('Processing Time Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Processing Time (seconds)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

            # Add mean line
            mean_time = np.mean(all_times)
            ax.axvline(x=mean_time, color='red', linestyle='--',
                      label=f'Mean: {mean_time:.2f}s')
            ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _update_performance_trends(self, aggregate_metrics: Dict[str, float]) -> None:
        """Update performance trends tracking"""
        for metric, value in aggregate_metrics.items():
            self.performance_trends[metric].append(value)

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed"""
        if not self.validation_history:
            return {'total_validations': 0}

        latest_report = self.validation_history[-1]

        return {
            'total_validations': len(self.validation_history),
            'latest_validation_id': latest_report.validation_id,
            'latest_overall_quality': latest_report.aggregate_metrics.get('avg_quality', 0),
            'latest_success_rate': latest_report.aggregate_metrics.get('success_rate', 0),
            'validation_trend': 'improving' if len(self.validation_history) > 1 and
                              latest_report.aggregate_metrics.get('avg_quality', 0) >
                              self.validation_history[-2].aggregate_metrics.get('avg_quality', 0) else 'stable',
            'total_images_evaluated': sum(report.total_images for report in self.validation_history),
            'avg_validation_time': np.mean([report.validation_time for report in self.validation_history])
        }


# Factory function for easy creation
def create_validation_framework(agent_interface,
                              validation_data: Dict[str, List[str]],
                              validation_frequency: int = 5000,
                              images_per_category: int = 5) -> ValidationProtocol:
    """
    Factory function to create validation framework

    Args:
        agent_interface: Agent interface for evaluation
        validation_data: Validation dataset
        validation_frequency: Validation frequency in training steps
        images_per_category: Images per category for validation

    Returns:
        Configured ValidationProtocol
    """
    config = ValidationConfig(
        validation_frequency=validation_frequency,
        validation_images_per_category=images_per_category
    )

    return ValidationProtocol(agent_interface, validation_data, config)