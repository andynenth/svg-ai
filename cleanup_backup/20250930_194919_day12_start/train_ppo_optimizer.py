#!/usr/bin/env python3
"""PPO Training Orchestrator for VTracer parameter optimization

This script orchestrates PPO training with monitoring and validation
as specified in DAY7_PPO_AGENT_TRAINING.md Task B7.2.
"""

import argparse
import yaml
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add backend modules to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_modules.optimization.training_pipeline import CurriculumTrainingPipeline
from backend.ai_modules.optimization.training_orchestrator import TrainingMonitor


class PPOTrainingOrchestrator:
    """Orchestrate PPO training with monitoring and validation"""

    def __init__(self, config_path: str):
        """
        Initialize PPO Training Orchestrator

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Setup logging first
        self.setup_logging()

        # Setup training environment
        self.setup_training_environment()

        # Initialize training pipeline
        self.training_pipeline = CurriculumTrainingPipeline(
            training_images=self.config['training']['training_images'],
            model_config=self.config['training']['model_config'],
            curriculum_config=self.config['training'].get('curriculum_config', None),
            save_dir=self.config['training']['save_dir'],
            enable_real_time_monitoring=self.config['monitoring'].get('enable_real_time_monitoring', True)
        )

        # Initialize monitoring
        self.monitor = TrainingMonitor(
            output_dir=self.config['monitoring']['log_dir']
        )

        # Training state
        self.training_results = {}
        self.training_start_time = None

        self.logger.info(f"PPO Training Orchestrator initialized with config: {config_path}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate YAML configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Validate required sections
            required_sections = ['training', 'environment', 'monitoring', 'logging']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required configuration section: {section}")

            return config
        except Exception as e:
            print(f"ERROR: Failed to load configuration from {config_path}: {e}")
            raise

    def setup_logging(self) -> None:
        """Setup structured logging configuration"""
        log_config = self.config['logging']

        # Create log directory
        log_dir = Path(log_config['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        log_level = getattr(logging, log_config['level'].upper())
        log_format = log_config.get('format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / 'ppo_training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging configuration complete")

    def setup_training_environment(self) -> None:
        """Setup training environment and resource allocation"""
        env_config = self.config['environment']

        self.logger.info("Setting up training environment...")

        # Setup Python environment configuration
        if 'python_path' in env_config:
            python_paths = env_config['python_path']
            if isinstance(python_paths, str):
                python_paths = [python_paths]
            for path in python_paths:
                if path not in sys.path:
                    sys.path.append(path)
                    self.logger.info(f"Added to Python path: {path}")

        # Setup GPU/CPU resource allocation
        if 'cuda_visible_devices' in env_config:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(env_config['cuda_visible_devices'])
            self.logger.info(f"Set CUDA_VISIBLE_DEVICES: {env_config['cuda_visible_devices']}")

        # Memory management
        if 'memory_limit_gb' in env_config:
            # This could be used by TensorFlow/PyTorch for memory management
            self.logger.info(f"Memory limit set to: {env_config['memory_limit_gb']} GB")

        # Set number of threads for CPU operations
        if 'num_threads' in env_config:
            os.environ['OMP_NUM_THREADS'] = str(env_config['num_threads'])
            self.logger.info(f"Set OMP_NUM_THREADS: {env_config['num_threads']}")

        # Create necessary directories
        for dir_key in ['save_dir', 'checkpoint_dir', 'log_dir']:
            if dir_key in self.config['training']:
                dir_path = Path(self.config['training'][dir_key])
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")

        self.logger.info("Training environment setup complete")

    def validate_training_data(self) -> None:
        """Validate training data quality and organization"""
        self.logger.info("Validating training data...")

        training_images = self.config['training']['training_images']
        total_images = 0

        for category, images in training_images.items():
            category_count = len(images) if isinstance(images, list) else 0
            total_images += category_count
            self.logger.info(f"Category '{category}': {category_count} images")

            # Check if image files exist
            if isinstance(images, list):
                missing_files = []
                for img_path in images:
                    if not Path(img_path).exists():
                        missing_files.append(img_path)

                if missing_files:
                    self.logger.warning(f"Missing files in category '{category}': {len(missing_files)}")
                    for missing in missing_files[:5]:  # Show first 5 missing files
                        self.logger.warning(f"  Missing: {missing}")

        if total_images == 0:
            raise ValueError("No training images found in configuration")

        self.logger.info(f"Training data validation complete. Total images: {total_images}")

    def run_training(self) -> Dict[str, Any]:
        """Execute complete training pipeline"""
        self.logger.info("🚀 Starting PPO training pipeline...")
        self.training_start_time = time.time()

        try:
            # Validate training data
            self.validate_training_data()

            # Start monitoring if enabled
            if self.config['monitoring'].get('enable_real_time_monitoring', True):
                self.logger.info("Starting real-time monitoring...")
                # Note: Real-time monitoring is handled by the training pipeline

            # Execute training
            self.logger.info("Executing curriculum training...")
            training_results = self.training_pipeline.run_curriculum()

            # Calculate total training time
            training_time = time.time() - self.training_start_time
            training_results['total_training_time'] = training_time

            # Log training metrics
            self.monitor.log_metrics({
                'training_method': 'curriculum_ppo',
                'total_time': training_time,
                'success_rate': training_results.get('success_rate', 0.0),
                'total_stages': training_results.get('total_stages', 0),
                'successful_stages': training_results.get('successful_stages', 0)
            })

            # Generate training report
            report = self._generate_training_report(training_results)
            self._save_training_report(report)

            # Save training results
            self._save_training_results(training_results)

            self.logger.info(f"✅ PPO training completed successfully in {training_time:.2f} seconds")
            self.logger.info(f"Success rate: {training_results.get('success_rate', 0.0):.2%}")

            return training_results

        except Exception as e:
            self.logger.error(f"❌ PPO training failed: {e}")
            self._handle_training_error(e)
            raise
        finally:
            # Cleanup resources
            self.cleanup()

    def _generate_training_report(self, training_results: Dict[str, Any]) -> str:
        """Generate comprehensive training report"""
        report = []
        report.append("# PPO Training Report")
        report.append("=" * 50)
        report.append("")

        # Training summary
        report.append("## Training Summary")
        report.append(f"- Total Training Time: {training_results.get('total_training_time', 0):.2f} seconds")
        report.append(f"- Total Stages: {training_results.get('total_stages', 0)}")
        report.append(f"- Successful Stages: {training_results.get('successful_stages', 0)}")
        report.append(f"- Success Rate: {training_results.get('success_rate', 0.0):.2%}")
        report.append("")

        # Final performance
        if 'final_performance' in training_results:
            perf = training_results['final_performance']
            report.append("## Final Performance")
            report.append(f"- Average Quality: {perf.get('average_quality', 0.0):.4f}")
            report.append(f"- Overall Success Rate: {perf.get('overall_success_rate', 0.0):.2%}")
            report.append(f"- Best Stage Quality: {perf.get('best_stage_quality', 0.0):.4f}")
            report.append("")

        # Stage details
        if 'stage_results' in training_results:
            report.append("## Stage Results")
            for stage_name, stage_result in training_results['stage_results'].items():
                report.append(f"### {stage_name}")
                report.append(f"- Success: {'✅' if stage_result.get('success', False) else '❌'}")
                report.append(f"- Average Quality: {stage_result.get('average_quality', 0.0):.4f}")
                report.append(f"- Success Rate: {stage_result.get('success_rate', 0.0):.2%}")
                report.append(f"- Training Time: {stage_result.get('training_time', 0.0):.2f}s")
                report.append("")

        # Configuration summary
        report.append("## Configuration")
        report.append(f"- Model Config: {self.config['training']['model_config']}")
        report.append(f"- Save Directory: {self.config['training']['save_dir']}")
        report.append("")

        return "\n".join(report)

    def _save_training_report(self, report: str) -> None:
        """Save training report to file"""
        report_path = Path(self.config['training']['save_dir']) / "training_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        self.logger.info(f"Training report saved to: {report_path}")

    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """Save training results as JSON"""
        results_path = Path(self.config['training']['save_dir']) / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Training results saved to: {results_path}")

    def _handle_training_error(self, error: Exception) -> None:
        """Handle training errors with proper logging and recovery"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': time.time(),
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }

        # Log error details
        error_path = Path(self.config['monitoring']['log_dir']) / "training_error.json"
        with open(error_path, 'w') as f:
            json.dump(error_info, f, indent=2)

        self.logger.error(f"Training error details saved to: {error_path}")

    def cleanup(self) -> None:
        """Clean up resources and close connections"""
        self.logger.info("Cleaning up training resources...")

        try:
            # Close training pipeline
            if hasattr(self.training_pipeline, 'close'):
                self.training_pipeline.close()

            # Save final monitoring data
            if hasattr(self.monitor, 'save_monitoring_data'):
                self.monitor.save_monitoring_data()

            self.logger.info("Resource cleanup complete")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'training_active': self.training_start_time is not None,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0,
            'config_path': self.config_path,
            'save_dir': self.config['training']['save_dir']
        }


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration"""
    return {
        'training': {
            'training_images': {
                'simple': ['data/logos/simple/circle_00.png'],
                'text': ['data/logos/text/text_00.png'],
                'gradient': ['data/logos/gradient/gradient_00.png'],
                'complex': ['data/logos/complex/complex_00.png']
            },
            'model_config': {
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
            },
            'curriculum_config': None,
            'save_dir': 'models/ppo_training',
            'checkpoint_dir': 'models/ppo_training/checkpoints'
        },
        'environment': {
            'num_threads': 4,
            'memory_limit_gb': 8,
            'cuda_visible_devices': '0'
        },
        'monitoring': {
            'enable_real_time_monitoring': True,
            'log_dir': 'logs/ppo_training',
            'checkpoint_frequency': 1000,
            'validation_frequency': 5000
        },
        'logging': {
            'level': 'INFO',
            'log_dir': 'logs/ppo_training',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }


def main():
    """Main entry point for PPO training orchestrator"""
    parser = argparse.ArgumentParser(description='Train PPO optimizer for VTracer parameter optimization')
    parser.add_argument('--config',
                       default='configs/ppo_training.yaml',
                       help='Training configuration file (YAML)')
    parser.add_argument('--create-default-config',
                       action='store_true',
                       help='Create default configuration file and exit')
    parser.add_argument('--validate-config',
                       action='store_true',
                       help='Validate configuration file and exit')

    args = parser.parse_args()

    # Create default config if requested
    if args.create_default_config:
        config_path = Path(args.config)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        default_config = create_default_config()
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, indent=2, default_flow_style=False)

        print(f"Default configuration created at: {config_path}")
        return

    # Validate config if requested
    if args.validate_config:
        try:
            orchestrator = PPOTrainingOrchestrator(args.config)
            print(f"✅ Configuration is valid: {args.config}")
            return
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            sys.exit(1)

    # Run training
    try:
        orchestrator = PPOTrainingOrchestrator(args.config)
        results = orchestrator.run_training()

        print(f"\n🎉 Training completed successfully!")
        print(f"Results saved to: {orchestrator.config['training']['save_dir']}")
        print(f"Success rate: {results.get('success_rate', 0.0):.2%}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()