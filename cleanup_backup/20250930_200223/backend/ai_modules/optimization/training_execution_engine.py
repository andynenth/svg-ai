# backend/ai_modules/optimization/training_execution_engine.py
"""Comprehensive training execution engine that integrates all infrastructure components"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Import our infrastructure components
from .training_data_manager import TrainingDataManager, DatasetSplit, BatchConfig
from .checkpoint_manager import CheckpointManager, CheckpointConfig, TrainingState
from .validation_framework import ValidationProtocol, ValidationConfig
from .resource_monitor import ResourceMonitor, ResourceThresholds
from .training_pipeline import CurriculumTrainingPipeline
from .training_orchestrator import TrainingOrchestrator, TrainingConfiguration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Comprehensive execution configuration"""
    # Experiment settings
    experiment_name: str
    data_root_path: str
    output_dir: str

    # Training settings
    use_curriculum: bool = True
    max_training_steps: int = 50000
    validation_frequency: int = 5000
    checkpoint_frequency: int = 1000

    # Resource management
    enable_resource_monitoring: bool = True
    resource_monitoring_interval: float = 10.0
    memory_limit_gb: int = 8
    parallel_workers: int = 2

    # Data management
    validation_split_ratio: float = 0.2
    images_per_category: int = 10
    batch_size: int = 32

    # Quality thresholds
    quality_thresholds: Dict[str, float] = None

    # Advanced settings
    enable_auto_checkpointing: bool = True
    enable_optimization_recommendations: bool = True
    generate_reports: bool = True
    create_visualizations: bool = True

    def __post_init__(self):
        if self.quality_thresholds is None:
            self.quality_thresholds = {
                'simple': 0.85,
                'text': 0.80,
                'gradient': 0.75,
                'complex': 0.70
            }


@dataclass
class ExecutionResult:
    """Result from training execution"""
    execution_id: str
    experiment_name: str
    success: bool
    total_time: float
    training_results: Dict[str, Any]
    validation_results: Dict[str, Any]
    resource_usage_summary: Dict[str, Any]
    checkpoints_created: List[str]
    recommendations: List[str]
    error_message: Optional[str] = None


class TrainingExecutionEngine:
    """
    Comprehensive training execution engine that integrates all infrastructure components
    for robust model training and validation
    """

    def __init__(self, config: ExecutionConfig):
        """
        Initialize training execution engine

        Args:
            config: Execution configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Execution tracking
        self.execution_id = f"execution_{config.experiment_name}_{int(time.time())}"
        self.execution_start_time = None
        self.current_training_step = 0

        # Infrastructure components (will be initialized in setup)
        self.data_manager: Optional[TrainingDataManager] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.validation_protocol: Optional[ValidationProtocol] = None
        self.resource_monitor: Optional[ResourceMonitor] = None
        self.training_orchestrator: Optional[TrainingOrchestrator] = None

        # Training state
        self.training_data: Optional[Dict[str, List[str]]] = None
        self.validation_data: Optional[Dict[str, List[str]]] = None
        self.agent_interface = None

        # Callbacks and hooks
        self.progress_callbacks: List[Callable] = []
        self.checkpoint_callbacks: List[Callable] = []
        self.validation_callbacks: List[Callable] = []

        logger.info(f"TrainingExecutionEngine initialized: {self.execution_id}")

    async def initialize_infrastructure(self) -> None:
        """Initialize all infrastructure components"""
        logger.info("🔧 Initializing training infrastructure...")

        try:
            # 1. Initialize data manager
            await self._initialize_data_manager()

            # 2. Initialize checkpoint manager
            await self._initialize_checkpoint_manager()

            # 3. Initialize resource monitor
            if self.config.enable_resource_monitoring:
                await self._initialize_resource_monitor()

            # 4. Initialize validation protocol (requires agent interface)
            # This will be done during training setup

            logger.info("✅ Infrastructure initialization completed")

        except Exception as e:
            logger.error(f"❌ Infrastructure initialization failed: {e}")
            raise

    async def _initialize_data_manager(self) -> None:
        """Initialize training data manager"""
        logger.info("Initializing data manager...")

        self.data_manager = TrainingDataManager(
            data_root=self.config.data_root_path,
            cache_dir=str(self.output_dir / 'data_cache'),
            enable_caching=True
        )

        # Scan and organize data
        data_summary = self.data_manager.scan_and_organize_data(max_workers=4)
        logger.info(f"Data organized: {data_summary}")

        # Create dataset splits
        split_config = DatasetSplit(
            training_ratio=1.0 - self.config.validation_split_ratio,
            validation_ratio=self.config.validation_split_ratio,
            test_ratio=0.0,
            random_seed=42
        )

        splits = self.data_manager.create_dataset_splits(split_config, 'main')
        self.training_data = splits['train']
        self.validation_data = splits['validation']

        logger.info("Data manager initialized successfully")

    async def _initialize_checkpoint_manager(self) -> None:
        """Initialize checkpoint manager"""
        logger.info("Initializing checkpoint manager...")

        checkpoint_config = CheckpointConfig(
            save_frequency=self.config.checkpoint_frequency,
            max_checkpoints=20,
            save_best_only=False,
            monitor_metric='quality_score',
            save_optimizer_state=True,
            save_training_state=True
        )

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.output_dir / 'checkpoints'),
            config=checkpoint_config
        )

        if self.config.enable_auto_checkpointing:
            self.checkpoint_manager.enable_auto_save()

        logger.info("Checkpoint manager initialized successfully")

    async def _initialize_resource_monitor(self) -> None:
        """Initialize resource monitor"""
        logger.info("Initializing resource monitor...")

        # Set up resource thresholds based on configuration
        thresholds = ResourceThresholds(
            memory_warning=min(80.0, (self.config.memory_limit_gb / 16.0) * 100 * 0.8),
            memory_critical=min(95.0, (self.config.memory_limit_gb / 16.0) * 100 * 0.95)
        )

        self.resource_monitor = ResourceMonitor(
            monitoring_interval=self.config.resource_monitoring_interval,
            thresholds=thresholds,
            save_dir=str(self.output_dir / 'resource_monitoring')
        )

        # Register alert callback
        self.resource_monitor.register_alert_callback(self._handle_resource_alert)

        # Start monitoring
        self.resource_monitor.start_monitoring()

        logger.info("Resource monitor initialized and started")

    async def _initialize_validation_protocol(self, agent_interface) -> None:
        """Initialize validation protocol with agent interface"""
        logger.info("Initializing validation protocol...")

        validation_config = ValidationConfig(
            validation_frequency=self.config.validation_frequency,
            validation_images_per_category=min(5, self.config.images_per_category // 2),
            quality_thresholds=self.config.quality_thresholds,
            generate_visualizations=self.config.create_visualizations
        )

        self.validation_protocol = ValidationProtocol(
            agent_interface=agent_interface,
            validation_data=self.validation_data,
            config=validation_config,
            save_dir=str(self.output_dir / 'validation')
        )

        logger.info("Validation protocol initialized successfully")

    def _handle_resource_alert(self, alert) -> None:
        """Handle resource alerts during training"""
        logger.warning(f"Resource alert: {alert.message}")

        # Take action based on alert type
        if alert.alert_type == 'critical':
            if alert.resource == 'memory':
                logger.critical("Critical memory usage - consider reducing batch size")
                # Could automatically adjust batch size here
            elif alert.resource == 'gpu_temperature':
                logger.critical("Critical GPU temperature - consider pausing training")
                # Could implement automatic training pause here

    async def execute_training(self, agent_interface) -> ExecutionResult:
        """
        Execute comprehensive training with all infrastructure components

        Args:
            agent_interface: Agent interface for training

        Returns:
            ExecutionResult with comprehensive results
        """
        logger.info(f"🚀 Starting training execution: {self.execution_id}")
        self.execution_start_time = time.time()
        self.agent_interface = agent_interface

        try:
            # Initialize validation protocol now that we have agent interface
            await self._initialize_validation_protocol(agent_interface)

            # Setup training orchestrator
            await self._setup_training_orchestrator()

            # Execute main training loop
            training_results = await self._execute_main_training()

            # Perform final validation
            final_validation = await self._perform_final_validation()

            # Generate comprehensive results
            execution_result = await self._generate_execution_results(
                training_results, final_validation
            )

            logger.info(f"✅ Training execution completed successfully: {self.execution_id}")
            return execution_result

        except Exception as e:
            logger.error(f"❌ Training execution failed: {e}")

            # Create failed result
            execution_time = time.time() - self.execution_start_time if self.execution_start_time else 0

            return ExecutionResult(
                execution_id=self.execution_id,
                experiment_name=self.config.experiment_name,
                success=False,
                total_time=execution_time,
                training_results={},
                validation_results={},
                resource_usage_summary={},
                checkpoints_created=[],
                recommendations=[],
                error_message=str(e)
            )

        finally:
            await self._cleanup_resources()

    async def _setup_training_orchestrator(self) -> None:
        """Setup training orchestrator"""
        logger.info("Setting up training orchestrator...")

        # Create training configuration
        training_config = TrainingConfiguration(
            experiment_name=self.config.experiment_name,
            training_data_path=self.config.data_root_path,
            validation_data_path="",  # We handle validation separately
            output_dir=str(self.output_dir / 'training'),
            use_curriculum=self.config.use_curriculum,
            max_parallel_jobs=self.config.parallel_workers,
            enable_hyperparameter_search=False,
            save_checkpoints=True,
            checkpoint_frequency=self.config.checkpoint_frequency
        )

        self.training_orchestrator = TrainingOrchestrator(training_config)

        logger.info("Training orchestrator setup completed")

    async def _execute_main_training(self) -> Dict[str, Any]:
        """Execute main training loop with integrated monitoring"""
        logger.info("🎯 Executing main training loop...")

        # If using curriculum training
        if self.config.use_curriculum:
            results = await self._execute_curriculum_training()
        else:
            results = await self._execute_standard_training()

        return results

    async def _execute_curriculum_training(self) -> Dict[str, Any]:
        """Execute curriculum-based training"""
        logger.info("Starting curriculum training...")

        # Create curriculum pipeline
        pipeline = CurriculumTrainingPipeline(
            training_images=self.training_data,
            save_dir=str(self.output_dir / 'curriculum')
        )

        # Setup progress monitoring
        training_start_time = time.time()

        try:
            # Run curriculum with integrated monitoring
            results = await self._run_monitored_curriculum(pipeline)

            training_time = time.time() - training_start_time
            results['training_time'] = training_time

            return results

        finally:
            pipeline.close()

    async def _run_monitored_curriculum(self, pipeline) -> Dict[str, Any]:
        """Run curriculum training with integrated monitoring"""
        stage_results = {}

        for stage_idx, stage in enumerate(pipeline.curriculum_stages):
            logger.info(f"Training curriculum stage {stage_idx + 1}/{len(pipeline.curriculum_stages)}: {stage.name}")

            # Train stage
            stage_result = pipeline.train_stage(stage_idx)
            stage_results[stage.name] = asdict(stage_result)

            # Update current step for monitoring
            self.current_training_step += stage_result.episodes_completed

            # Perform validation if due
            if self.validation_protocol:
                validation_report = self.validation_protocol.run_validation(
                    checkpoint_id=f"stage_{stage.name}",
                    training_step=self.current_training_step
                )

                if validation_report:
                    stage_results[stage.name]['validation'] = asdict(validation_report)

            # Create checkpoint if needed
            if self.checkpoint_manager and stage_result.success:
                await self._create_stage_checkpoint(stage.name, stage_result)

            # Trigger progress callbacks
            for callback in self.progress_callbacks:
                try:
                    callback(stage_idx + 1, len(pipeline.curriculum_stages), stage_result)
                except Exception as e:
                    logger.error(f"Progress callback failed: {e}")

        return {
            'training_method': 'curriculum',
            'stage_results': stage_results,
            'total_stages': len(pipeline.curriculum_stages),
            'successful_stages': sum(1 for result in stage_results.values() if result['success'])
        }

    async def _execute_standard_training(self) -> Dict[str, Any]:
        """Execute standard (non-curriculum) training"""
        logger.info("Starting standard training...")

        # Use training orchestrator for standard training
        results = self.training_orchestrator.run_training_experiment()

        return {
            'training_method': 'standard',
            'orchestrator_results': results
        }

    async def _create_stage_checkpoint(self, stage_name: str, stage_result) -> Optional[str]:
        """Create checkpoint for training stage"""
        try:
            # Create training state (simplified for demonstration)
            training_state = TrainingState(
                epoch=0,  # Curriculum doesn't use epochs
                step=self.current_training_step,
                model_state_dict={},  # Would need actual model state
                optimizer_state_dict=None,
                scheduler_state_dict=None,
                random_states={'numpy': np.random.get_state()},
                training_metrics=[],
                validation_metrics=[],
                best_metrics={'quality': stage_result.best_quality},
                training_start_time=self.execution_start_time,
                total_training_time=stage_result.training_time,
                curriculum_stage=stage_name
            )

            performance_metrics = {
                'quality_score': stage_result.average_quality,
                'success_rate': stage_result.success_rate,
                'best_quality': stage_result.best_quality
            }

            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                training_state=training_state,
                performance_metrics=performance_metrics,
                notes=f"Curriculum stage: {stage_name}"
            )

            if checkpoint_id:
                logger.info(f"Checkpoint created for stage {stage_name}: {checkpoint_id}")

                # Trigger checkpoint callbacks
                for callback in self.checkpoint_callbacks:
                    try:
                        callback(checkpoint_id, stage_name, performance_metrics)
                    except Exception as e:
                        logger.error(f"Checkpoint callback failed: {e}")

            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to create checkpoint for stage {stage_name}: {e}")
            return None

    async def _perform_final_validation(self) -> Dict[str, Any]:
        """Perform comprehensive final validation"""
        logger.info("🔍 Performing final validation...")

        if not self.validation_protocol:
            return {'final_validation': 'not_available'}

        try:
            # Run comprehensive validation
            final_validation_report = self.validation_protocol.run_validation(
                checkpoint_id="final",
                training_step=self.current_training_step,
                force_validation=True
            )

            if final_validation_report:
                # Generate validation summary
                validation_summary = self.validation_protocol.get_validation_summary()

                return {
                    'final_validation_report': asdict(final_validation_report),
                    'validation_summary': validation_summary
                }
            else:
                return {'final_validation': 'failed'}

        except Exception as e:
            logger.error(f"Final validation failed: {e}")
            return {'final_validation': 'error', 'error': str(e)}

    async def _generate_execution_results(self,
                                        training_results: Dict[str, Any],
                                        validation_results: Dict[str, Any]) -> ExecutionResult:
        """Generate comprehensive execution results"""
        logger.info("📊 Generating execution results...")

        execution_time = time.time() - self.execution_start_time

        # Get resource usage summary
        resource_summary = {}
        if self.resource_monitor:
            resource_summary = self.resource_monitor.get_current_stats()

            # Add optimization recommendations
            recommendations = self.resource_monitor.get_optimization_recommendations()
            resource_summary['optimization_recommendations'] = [
                asdict(rec) for rec in recommendations
            ]

        # Get checkpoint summary
        checkpoints_created = []
        if self.checkpoint_manager:
            recent_checkpoints = self.checkpoint_manager.list_checkpoints(limit=10)
            checkpoints_created = [cp.checkpoint_id for cp in recent_checkpoints]

        # Generate recommendations
        recommendations = self._generate_execution_recommendations(
            training_results, validation_results, resource_summary
        )

        # Determine overall success
        success = self._evaluate_execution_success(training_results, validation_results)

        # Create comprehensive result
        result = ExecutionResult(
            execution_id=self.execution_id,
            experiment_name=self.config.experiment_name,
            success=success,
            total_time=execution_time,
            training_results=training_results,
            validation_results=validation_results,
            resource_usage_summary=resource_summary,
            checkpoints_created=checkpoints_created,
            recommendations=recommendations
        )

        # Save execution results
        if self.config.generate_reports:
            await self._save_execution_results(result)

        return result

    def _generate_execution_recommendations(self,
                                          training_results: Dict[str, Any],
                                          validation_results: Dict[str, Any],
                                          resource_summary: Dict[str, Any]) -> List[str]:
        """Generate execution recommendations based on results"""
        recommendations = []

        # Training performance recommendations
        if 'stage_results' in training_results:
            avg_quality = np.mean([
                result['average_quality'] for result in training_results['stage_results'].values()
                if 'average_quality' in result
            ])

            if avg_quality < 0.7:
                recommendations.append("Consider extended training or hyperparameter tuning - average quality is low")
            elif avg_quality > 0.9:
                recommendations.append("Excellent training performance - consider this configuration for future experiments")

        # Validation recommendations
        if 'final_validation_report' in validation_results:
            final_report = validation_results['final_validation_report']
            success_rate = final_report.get('aggregate_metrics', {}).get('success_rate', 0)

            if success_rate < 0.8:
                recommendations.append("Low validation success rate - investigate model generalization")

        # Resource usage recommendations
        if resource_summary:
            avg_cpu = resource_summary.get('cpu_percent', 0)
            avg_memory = resource_summary.get('memory_percent', 0)

            if avg_cpu < 50:
                recommendations.append("Low CPU utilization - consider increasing parallel workers")
            if avg_memory > 90:
                recommendations.append("High memory usage detected - consider reducing batch size")

        return recommendations

    def _evaluate_execution_success(self,
                                   training_results: Dict[str, Any],
                                   validation_results: Dict[str, Any]) -> bool:
        """Evaluate overall execution success"""
        # Check training success
        training_success = False
        if training_results.get('training_method') == 'curriculum':
            successful_stages = training_results.get('successful_stages', 0)
            total_stages = training_results.get('total_stages', 1)
            training_success = successful_stages / total_stages >= 0.5
        else:
            training_success = training_results.get('success', False)

        # Check validation success
        validation_success = True
        if 'final_validation_report' in validation_results:
            final_report = validation_results['final_validation_report']
            success_rate = final_report.get('aggregate_metrics', {}).get('success_rate', 0)
            validation_success = success_rate >= 0.6

        return training_success and validation_success

    async def _save_execution_results(self, result: ExecutionResult) -> None:
        """Save execution results to files"""
        try:
            # Save JSON results
            results_file = self.output_dir / f"execution_results_{self.execution_id}.json"
            with open(results_file, 'w') as f:
                json.dump(asdict(result), f, indent=2)

            # Save human-readable report
            report_file = self.output_dir / f"execution_report_{self.execution_id}.txt"
            with open(report_file, 'w') as f:
                f.write(self._generate_execution_report(result))

            logger.info(f"Execution results saved to: {results_file}")
            logger.info(f"Execution report saved to: {report_file}")

        except Exception as e:
            logger.error(f"Failed to save execution results: {e}")

    def _generate_execution_report(self, result: ExecutionResult) -> str:
        """Generate human-readable execution report"""
        report = []
        report.append(f"# Training Execution Report: {result.experiment_name}")
        report.append("=" * 80)
        report.append("")

        # Executive summary
        report.append("## Executive Summary")
        report.append(f"- Execution ID: {result.execution_id}")
        report.append(f"- Success: {'✅ Yes' if result.success else '❌ No'}")
        report.append(f"- Total Time: {result.total_time / 3600:.2f} hours")
        report.append(f"- Checkpoints Created: {len(result.checkpoints_created)}")
        report.append("")

        # Training results
        report.append("## Training Results")
        training_method = result.training_results.get('training_method', 'unknown')
        report.append(f"- Training Method: {training_method}")

        if training_method == 'curriculum':
            total_stages = result.training_results.get('total_stages', 0)
            successful_stages = result.training_results.get('successful_stages', 0)
            report.append(f"- Curriculum Stages: {successful_stages}/{total_stages} successful")

        report.append("")

        # Validation results
        report.append("## Validation Results")
        if 'final_validation_report' in result.validation_results:
            final_report = result.validation_results['final_validation_report']
            aggregate_metrics = final_report.get('aggregate_metrics', {})

            avg_quality = aggregate_metrics.get('avg_quality', 0)
            success_rate = aggregate_metrics.get('success_rate', 0)

            report.append(f"- Average Quality: {avg_quality:.4f}")
            report.append(f"- Success Rate: {success_rate:.2%}")
        else:
            report.append("- Final validation not available")

        report.append("")

        # Resource usage
        report.append("## Resource Usage Summary")
        if result.resource_usage_summary:
            summary = result.resource_usage_summary
            report.append(f"- Peak CPU Usage: {summary.get('cpu_percent', 0):.1f}%")
            report.append(f"- Peak Memory Usage: {summary.get('memory_percent', 0):.1f}%")

            if 'gpu_utilization' in summary:
                report.append(f"- GPU Utilization: {summary['gpu_utilization']:.1f}%")

        report.append("")

        # Recommendations
        if result.recommendations:
            report.append("## Recommendations")
            for i, rec in enumerate(result.recommendations, 1):
                report.append(f"{i}. {rec}")

        report.append("")

        return "\n".join(report)

    async def _cleanup_resources(self) -> None:
        """Clean up all resources"""
        logger.info("🧹 Cleaning up resources...")

        try:
            # Stop resource monitoring
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()

                # Save monitoring data
                if self.config.generate_reports:
                    self.resource_monitor.save_monitoring_data()

                    # Create resource plots
                    if self.config.create_visualizations:
                        self.resource_monitor.create_resource_plots()

            # Save checkpoint manager state
            if self.checkpoint_manager:
                if self.config.generate_reports:
                    self.checkpoint_manager.export_checkpoint_info(
                        str(self.output_dir / 'checkpoint_info.json')
                    )

            # Save data manager info
            if self.data_manager and self.config.generate_reports:
                self.data_manager.export_dataset_info(
                    str(self.output_dir / 'dataset_info.json')
                )

            logger.info("✅ Resource cleanup completed")

        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")

    # Callback management methods
    def add_progress_callback(self, callback: Callable) -> None:
        """Add progress callback"""
        self.progress_callbacks.append(callback)

    def add_checkpoint_callback(self, callback: Callable) -> None:
        """Add checkpoint callback"""
        self.checkpoint_callbacks.append(callback)

    def add_validation_callback(self, callback: Callable) -> None:
        """Add validation callback"""
        self.validation_callbacks.append(callback)

    # Status and monitoring methods
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        if not self.execution_start_time:
            return {'status': 'not_started'}

        current_time = time.time()
        elapsed_time = current_time - self.execution_start_time

        status = {
            'execution_id': self.execution_id,
            'status': 'running',
            'elapsed_time_minutes': elapsed_time / 60,
            'current_training_step': self.current_training_step
        }

        # Add resource status if available
        if self.resource_monitor:
            status['resource_status'] = self.resource_monitor.get_current_stats()

        # Add checkpoint status if available
        if self.checkpoint_manager:
            recent_checkpoints = self.checkpoint_manager.list_checkpoints(limit=3)
            status['recent_checkpoints'] = [cp.checkpoint_id for cp in recent_checkpoints]

        return status


# Factory function for easy creation
async def create_training_execution_engine(experiment_name: str,
                                         data_root_path: str,
                                         output_dir: str,
                                         **kwargs) -> TrainingExecutionEngine:
    """
    Factory function to create and initialize training execution engine

    Args:
        experiment_name: Name of the experiment
        data_root_path: Path to training data
        output_dir: Output directory for results
        **kwargs: Additional configuration options

    Returns:
        Initialized TrainingExecutionEngine
    """
    config = ExecutionConfig(
        experiment_name=experiment_name,
        data_root_path=data_root_path,
        output_dir=output_dir,
        **kwargs
    )

    engine = TrainingExecutionEngine(config)
    await engine.initialize_infrastructure()

    return engine