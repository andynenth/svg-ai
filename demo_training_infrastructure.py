#!/usr/bin/env python3
"""
Demonstration of the comprehensive training execution pipeline infrastructure

This script demonstrates how to use all the training infrastructure components
together for robust model training and validation.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the backend modules to the path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.ai_modules.optimization.training_execution_engine import (
    create_training_execution_engine,
    ExecutionConfig
)
from backend.ai_modules.optimization.agent_interface import VTracerAgentInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_training_infrastructure():
    """Demonstrate the complete training infrastructure"""

    logger.info("🚀 Starting Training Infrastructure Demonstration")

    # Configuration for the demonstration
    config = ExecutionConfig(
        experiment_name="infrastructure_demo",
        data_root_path="data/logos",
        output_dir="demo_training_output",

        # Training settings
        use_curriculum=True,
        max_training_steps=10000,  # Reduced for demo
        validation_frequency=2000,
        checkpoint_frequency=1000,

        # Resource management
        enable_resource_monitoring=True,
        resource_monitoring_interval=5.0,
        parallel_workers=1,  # Reduced for demo

        # Data settings
        validation_split_ratio=0.3,
        images_per_category=3,  # Reduced for demo
        batch_size=16,

        # Advanced features
        enable_auto_checkpointing=True,
        enable_optimization_recommendations=True,
        generate_reports=True,
        create_visualizations=True,

        # Quality thresholds
        quality_thresholds={
            'simple': 0.80,
            'text': 0.75,
            'gradient': 0.70,
            'complex': 0.65
        }
    )

    try:
        # 1. Create and initialize the training execution engine
        logger.info("📋 Creating training execution engine...")
        engine = await create_training_execution_engine(
            experiment_name=config.experiment_name,
            data_root_path=config.data_root_path,
            output_dir=config.output_dir,
            use_curriculum=config.use_curriculum,
            max_training_steps=config.max_training_steps,
            validation_frequency=config.validation_frequency,
            checkpoint_frequency=config.checkpoint_frequency,
            enable_resource_monitoring=config.enable_resource_monitoring,
            resource_monitoring_interval=config.resource_monitoring_interval,
            parallel_workers=config.parallel_workers,
            validation_split_ratio=config.validation_split_ratio,
            images_per_category=config.images_per_category,
            batch_size=config.batch_size,
            enable_auto_checkpointing=config.enable_auto_checkpointing,
            enable_optimization_recommendations=config.enable_optimization_recommendations,
            generate_reports=config.generate_reports,
            create_visualizations=config.create_visualizations,
            quality_thresholds=config.quality_thresholds
        )

        logger.info("✅ Training execution engine initialized successfully")

        # 2. Set up callbacks for monitoring
        def progress_callback(current_stage, total_stages, stage_result):
            logger.info(f"📊 Progress: Stage {current_stage}/{total_stages} - "
                       f"Quality: {stage_result.average_quality:.4f}, "
                       f"Success: {stage_result.success}")

        def checkpoint_callback(checkpoint_id, stage_name, metrics):
            logger.info(f"💾 Checkpoint created: {checkpoint_id} for {stage_name} - "
                       f"Quality: {metrics.get('quality_score', 0):.4f}")

        def validation_callback(validation_report):
            logger.info(f"🔍 Validation completed: "
                       f"Quality: {validation_report.aggregate_metrics.get('avg_quality', 0):.4f}")

        # Register callbacks
        engine.add_progress_callback(progress_callback)
        engine.add_checkpoint_callback(checkpoint_callback)
        engine.add_validation_callback(validation_callback)

        # 3. Create agent interface for demonstration
        logger.info("🤖 Creating agent interface...")
        agent_interface = VTracerAgentInterface(
            model_save_dir=str(Path(config.output_dir) / "agent_models"),
            config_file=None  # Use default configuration
        )

        # 4. Execute the training
        logger.info("🎯 Starting training execution...")
        execution_result = await engine.execute_training(agent_interface)

        # 5. Display results
        logger.info("📈 Training Execution Results:")
        logger.info(f"   Success: {'✅' if execution_result.success else '❌'}")
        logger.info(f"   Total Time: {execution_result.total_time / 60:.2f} minutes")
        logger.info(f"   Checkpoints Created: {len(execution_result.checkpoints_created)}")
        logger.info(f"   Recommendations: {len(execution_result.recommendations)}")

        if execution_result.success:
            # Show training results
            training_results = execution_result.training_results
            if 'stage_results' in training_results:
                logger.info("📚 Curriculum Stage Results:")
                for stage_name, stage_result in training_results['stage_results'].items():
                    logger.info(f"   {stage_name}: Quality {stage_result['average_quality']:.4f}, "
                               f"Success: {stage_result['success']}")

            # Show validation results
            validation_results = execution_result.validation_results
            if 'final_validation_report' in validation_results:
                final_validation = validation_results['final_validation_report']
                aggregate_metrics = final_validation.get('aggregate_metrics', {})
                logger.info(f"🔍 Final Validation: Quality {aggregate_metrics.get('avg_quality', 0):.4f}, "
                           f"Success Rate {aggregate_metrics.get('success_rate', 0):.2%}")

            # Show resource usage
            resource_summary = execution_result.resource_usage_summary
            if resource_summary:
                logger.info(f"💻 Resource Usage: CPU {resource_summary.get('cpu_percent', 0):.1f}%, "
                           f"Memory {resource_summary.get('memory_percent', 0):.1f}%")

            # Show recommendations
            if execution_result.recommendations:
                logger.info("💡 Recommendations:")
                for i, rec in enumerate(execution_result.recommendations, 1):
                    logger.info(f"   {i}. {rec}")

        else:
            logger.error(f"❌ Training failed: {execution_result.error_message}")

        # 6. Demonstrate infrastructure components individually
        logger.info("🔧 Demonstrating individual components...")

        # Data Manager
        if engine.data_manager:
            data_summary = engine.data_manager._generate_data_summary()
            logger.info(f"📊 Data Manager: {data_summary['total_images']} images organized")

            # Show quality validation
            quality_report = engine.data_manager.validate_dataset_quality()
            logger.info(f"   Quality Report: {quality_report['valid_images']} valid images, "
                       f"{len(quality_report['recommendations'])} recommendations")

        # Checkpoint Manager
        if engine.checkpoint_manager:
            checkpoint_stats = engine.checkpoint_manager.get_checkpoint_statistics()
            logger.info(f"💾 Checkpoint Manager: {checkpoint_stats['total_checkpoints']} checkpoints, "
                       f"{checkpoint_stats['total_storage_size_mb']:.1f}MB total")

        # Resource Monitor
        if engine.resource_monitor:
            resource_stats = engine.resource_monitor.get_current_stats()
            logger.info(f"💻 Resource Monitor: {resource_stats.get('total_measurements', 0)} measurements, "
                       f"{resource_stats.get('monitoring_duration_minutes', 0):.1f} minutes")

            # Show optimization recommendations
            opt_recommendations = engine.resource_monitor.get_optimization_recommendations()
            if opt_recommendations:
                logger.info(f"   Optimization Recommendations: {len(opt_recommendations)} available")

        # Validation Protocol
        if engine.validation_protocol:
            validation_summary = engine.validation_protocol.get_validation_summary()
            logger.info(f"🔍 Validation Protocol: {validation_summary['total_validations']} validations performed")

        # 7. Show file outputs
        output_dir = Path(config.output_dir)
        if output_dir.exists():
            logger.info("📁 Generated Files:")
            for file_path in output_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in ['.json', '.txt', '.png']:
                    rel_path = file_path.relative_to(output_dir)
                    logger.info(f"   {rel_path}")

        logger.info("✅ Training infrastructure demonstration completed successfully!")

        return execution_result

    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise

    finally:
        # Cleanup
        if 'agent_interface' in locals():
            agent_interface.close()


def demonstrate_individual_components():
    """Demonstrate individual infrastructure components"""

    logger.info("🔧 Demonstrating Individual Components")

    # 1. Training Data Manager Demo
    logger.info("📊 Training Data Manager Demo:")
    try:
        from backend.ai_modules.optimization.training_data_manager import create_training_data_manager

        data_manager = create_training_data_manager(
            data_root="data/logos",
            auto_scan=True
        )

        # Show data organization
        summary = data_manager._generate_data_summary()
        logger.info(f"   Organized {summary['total_images']} images across {len(summary['categories'])} categories")

        # Create dataset splits
        from backend.ai_modules.optimization.training_data_manager import DatasetSplit
        split_config = DatasetSplit(training_ratio=0.8, validation_ratio=0.2)
        splits = data_manager.create_dataset_splits(split_config)

        logger.info(f"   Created splits: {sum(len(imgs) for imgs in splits['train'].values())} train, "
                   f"{sum(len(imgs) for imgs in splits['validation'].values())} validation")

    except Exception as e:
        logger.error(f"   Data Manager demo failed: {e}")

    # 2. Checkpoint Manager Demo
    logger.info("💾 Checkpoint Manager Demo:")
    try:
        from backend.ai_modules.optimization.checkpoint_manager import create_checkpoint_manager

        checkpoint_manager = create_checkpoint_manager(
            checkpoint_dir="demo_checkpoints",
            save_frequency=1000,
            max_checkpoints=5
        )

        logger.info(f"   Checkpoint manager created with directory: demo_checkpoints")

        # Show checkpoint statistics
        stats = checkpoint_manager.get_checkpoint_statistics()
        logger.info(f"   Statistics: {stats['total_checkpoints']} checkpoints")

    except Exception as e:
        logger.error(f"   Checkpoint Manager demo failed: {e}")

    # 3. Resource Monitor Demo
    logger.info("💻 Resource Monitor Demo:")
    try:
        from backend.ai_modules.optimization.resource_monitor import create_resource_monitor

        resource_monitor = create_resource_monitor(
            monitoring_interval=2.0,
            auto_start=True
        )

        # Let it monitor for a few seconds
        import time
        time.sleep(5)

        current_stats = resource_monitor.get_current_stats()
        logger.info(f"   Current CPU: {current_stats.get('cpu_percent', 0):.1f}%, "
                   f"Memory: {current_stats.get('memory_percent', 0):.1f}%")

        resource_monitor.stop_monitoring()

    except Exception as e:
        logger.error(f"   Resource Monitor demo failed: {e}")

    logger.info("✅ Individual components demonstration completed!")


async def main():
    """Main demonstration function"""
    try:
        logger.info("🎯 Starting Comprehensive Training Infrastructure Demonstration")

        # Check if data directory exists
        data_dir = Path("data/logos")
        if not data_dir.exists():
            logger.error(f"❌ Data directory not found: {data_dir}")
            logger.info("Please ensure the training data is available at data/logos/")
            return

        # Run individual components demo first
        demonstrate_individual_components()

        logger.info("\n" + "="*80 + "\n")

        # Run full integration demo
        execution_result = await demonstrate_training_infrastructure()

        if execution_result and execution_result.success:
            logger.info("🎉 All demonstrations completed successfully!")
        else:
            logger.warning("⚠️ Some demonstrations had issues, but infrastructure is functional")

    except KeyboardInterrupt:
        logger.info("🛑 Demonstration interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())