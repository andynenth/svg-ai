"""
Colab Training Orchestrator - Complete GPU Training System
Orchestrates the complete GPU training pipeline for Colab environment
Integrates all components: model, training, visualization, and persistence
Part of Task 11.2: GPU Model Architecture & Training Pipeline Setup
"""

import torch
import numpy as np
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings

# Import our GPU training components
from .gpu_model_architecture import (
    QualityPredictorGPU,
    GPUFeatureExtractor,
    ColabTrainingConfig,
    ModelOptimizer,
    validate_gpu_setup
)
from .gpu_training_pipeline import (
    GPUTrainingPipeline,
    GPUDataLoader,
    ColabTrainingExample,
    TrainingMetrics,
    load_optimization_data,
    extract_training_examples
)
from .colab_training_visualization import (
    ColabTrainingVisualizer,
    ColabPerformanceMonitor
)
from .colab_persistence_manager import (
    ColabPersistenceManager,
    PersistenceConfig,
    setup_colab_persistence
)

warnings.filterwarnings('ignore')


class ColabTrainingOrchestrator:
    """Complete orchestration of GPU training in Colab environment"""

    def __init__(self, training_config: Optional[ColabTrainingConfig] = None,
                 persistence_config: Optional[PersistenceConfig] = None):

        print("🚀 Initializing Colab Training Orchestrator")
        print("=" * 50)

        # Setup configurations
        self.training_config = training_config or ColabTrainingConfig()
        self.persistence_config = persistence_config or PersistenceConfig()

        # Validate GPU setup
        device, gpu_ready = validate_gpu_setup()
        if not gpu_ready:
            print("⚠️ GPU setup validation failed - switching to CPU")
            self.training_config.device = 'cpu'
            self.training_config.mixed_precision = False

        # Initialize components
        self.model = None
        self.training_pipeline = None
        self.visualizer = ColabTrainingVisualizer()
        self.performance_monitor = ColabPerformanceMonitor()
        self.persistence_manager = ColabPersistenceManager(self.persistence_config)

        # Training state
        self.training_data = []
        self.training_completed = False
        self.final_model_paths = {}

        print(f"✅ Orchestrator initialized")
        print(f"   Device: {self.training_config.device}")
        print(f"   Mixed Precision: {self.training_config.mixed_precision}")
        print(f"   Drive Integration: {self.persistence_manager.drive_mounted}")

    def load_training_data(self, data_sources: List[str]) -> int:
        """Load and prepare training data from multiple sources"""
        print("\n📂 Loading Training Data")
        print("-" * 30)

        # Load data from optimization results
        all_examples = load_optimization_data(data_sources)

        if not all_examples:
            print("❌ No training data found!")
            return 0

        # Filter valid examples
        valid_examples = []
        feature_extractor = GPUFeatureExtractor(self.training_config.device)

        print(f"🔍 Processing {len(all_examples)} examples...")

        for i, example in enumerate(all_examples):
            try:
                # Validate example has required fields
                if (hasattr(example, 'image_path') and
                    hasattr(example, 'vtracer_params') and
                    hasattr(example, 'actual_ssim')):

                    # Extract features if not already done
                    if not hasattr(example, 'image_features') or example.image_features is None:
                        example.image_features = feature_extractor.extract_features_single(example.image_path)

                    # Validate SSIM range
                    if 0 <= example.actual_ssim <= 1:
                        valid_examples.append(example)

                if (i + 1) % 50 == 0:
                    print(f"   Processed {i + 1}/{len(all_examples)} examples")

            except Exception as e:
                print(f"⚠️ Error processing example {i}: {e}")
                continue

        self.training_data = valid_examples
        print(f"✅ Loaded {len(valid_examples)} valid training examples")

        # Analyze data distribution
        self._analyze_training_data()

        return len(valid_examples)

    def _analyze_training_data(self):
        """Analyze training data distribution"""
        if not self.training_data:
            return

        ssim_values = [ex.actual_ssim for ex in self.training_data]
        logo_types = [getattr(ex, 'logo_type', 'unknown') for ex in self.training_data]
        methods = [getattr(ex, 'optimization_method', 'unknown') for ex in self.training_data]

        print(f"\n📊 Data Analysis:")
        print(f"   SSIM Range: {min(ssim_values):.3f} - {max(ssim_values):.3f}")
        print(f"   SSIM Mean: {np.mean(ssim_values):.3f} ± {np.std(ssim_values):.3f}")

        # Logo type distribution
        from collections import Counter
        type_counts = Counter(logo_types)
        print(f"   Logo Types: {dict(type_counts)}")

        method_counts = Counter(methods)
        print(f"   Methods: {dict(method_counts)}")

    def setup_training(self) -> bool:
        """Setup the complete training pipeline"""
        print("\n⚙️ Setting up Training Pipeline")
        print("-" * 30)

        if not self.training_data:
            print("❌ No training data available!")
            return False

        try:
            # Initialize model
            self.model = QualityPredictorGPU(self.training_config)

            # Initialize training pipeline
            self.training_pipeline = GPUTrainingPipeline(self.training_config)

            # Create data loaders
            train_loader, val_loader, statistics = GPUDataLoader.create_dataloaders(
                self.training_data, self.training_config
            )

            self.train_loader = train_loader
            self.val_loader = val_loader
            self.data_statistics = statistics

            print(f"✅ Training pipeline setup complete")
            print(f"   Model parameters: {self.model.count_parameters():,}")
            print(f"   Training batches: {len(train_loader)}")
            print(f"   Validation batches: {len(val_loader)}")

            return True

        except Exception as e:
            print(f"❌ Training setup failed: {e}")
            return False

    def execute_training(self) -> Dict[str, Any]:
        """Execute the complete training process"""
        print("\n🎯 Starting GPU Training")
        print("=" * 50)

        if not self.training_pipeline or not self.model:
            print("❌ Training not setup! Call setup_training() first.")
            return {}

        training_start_time = time.time()

        try:
            # Execute training with real-time monitoring
            training_results = self._training_loop_with_monitoring()

            training_duration = time.time() - training_start_time
            print(f"\n✅ Training completed in {training_duration:.1f}s")

            # Save final model
            self.final_model_paths = self.persistence_manager.save_final_model(
                self.model, training_results
            )

            # Generate comprehensive report
            final_report = self._generate_final_report(training_results, training_duration)

            # Save training summary
            summary_path = self.persistence_manager.create_training_summary(
                training_results['loss_curves'], self.final_model_paths
            )

            self.training_completed = True
            print(f"🎉 Training orchestration complete!")
            print(f"📊 Summary saved: {summary_path}")

            return final_report

        except Exception as e:
            print(f"❌ Training execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _training_loop_with_monitoring(self) -> Dict[str, Any]:
        """Execute training loop with real-time monitoring and visualization"""

        # Training metrics tracking
        metrics = TrainingMetrics()

        for epoch in range(self.training_config.epochs):
            self.performance_monitor.start_epoch()

            # Training step
            train_loss = self._execute_training_epoch()

            # Validation step
            val_loss, val_correlation = self.training_pipeline.optimizer_wrapper.validation_step(
                self.val_loader
            )

            # Update learning rate
            self.training_pipeline.optimizer_wrapper.step_scheduler()
            current_lr = self.training_pipeline.optimizer_wrapper.get_lr()

            # Performance monitoring
            epoch_time = self.performance_monitor.end_epoch()
            memory_info = self.performance_monitor.get_memory_usage()
            gpu_memory = memory_info.get('gpu_allocated', 0)

            # Update metrics
            metrics.update(epoch, train_loss, val_loss, val_correlation,
                         current_lr, epoch_time)

            # Update visualizations
            self.visualizer.update_metrics(
                train_loss, val_loss, val_correlation, current_lr,
                epoch_time, gpu_memory
            )

            # Real-time visualization (every 5 epochs or last epoch)
            if (epoch + 1) % 5 == 0 or epoch == self.training_config.epochs - 1:
                self.visualizer.plot_training_progress()

            # Log performance
            self.performance_monitor.log_performance(
                epoch + 1, train_loss, val_loss, val_correlation
            )

            # Checkpointing
            if (epoch + 1) % self.training_config.checkpoint_freq == 0:
                self.persistence_manager.save_checkpoint(
                    self.model,
                    self.training_pipeline.optimizer_wrapper.optimizer,
                    epoch,
                    metrics.get_summary(),
                    self.training_config.__dict__
                )

            # Save training progress
            if (epoch + 1) % 10 == 0:
                training_history = {
                    'train_losses': metrics.train_losses,
                    'val_losses': metrics.val_losses,
                    'val_correlations': metrics.val_correlations,
                    'learning_rates': metrics.learning_rates
                }
                self.persistence_manager.save_training_progress(training_history)

            # Early stopping check
            if metrics.should_early_stop(self.training_config.early_stopping_patience):
                print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
                break

            # Target achievement check
            if val_correlation >= 0.9:
                print(f"\n🎯 Target correlation achieved: {val_correlation:.4f}")
                break

        # Return complete training results
        return {
            'training_summary': metrics.get_summary(),
            'loss_curves': {
                'train_losses': metrics.train_losses,
                'val_losses': metrics.val_losses,
                'val_correlations': metrics.val_correlations,
                'learning_rates': metrics.learning_rates
            },
            'data_statistics': self.data_statistics,
            'gpu_info': self._get_system_info()
        }

    def _execute_training_epoch(self) -> float:
        """Execute a single training epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_features, batch_targets in self.train_loader:
            loss, _ = self.training_pipeline.optimizer_wrapper.training_step(
                batch_features, batch_targets
            )
            total_loss += loss

        return total_loss / len(self.train_loader)

    def _generate_final_report(self, training_results: Dict[str, Any],
                             training_duration: float) -> Dict[str, Any]:
        """Generate comprehensive final training report"""

        summary = training_results.get('training_summary', {})

        report = {
            'orchestration_summary': {
                'total_training_time': training_duration,
                'training_completed': True,
                'target_achieved': summary.get('best_correlation', 0) >= 0.9,
                'data_quality': len(self.training_data),
                'drive_backup': self.persistence_manager.drive_mounted
            },
            'model_performance': {
                'best_validation_loss': summary.get('best_val_loss', float('inf')),
                'best_correlation': summary.get('best_correlation', 0),
                'final_correlation': summary.get('final_correlation', 0),
                'epochs_completed': summary.get('epochs_completed', 0),
                'convergence_time': summary.get('total_training_time', 0)
            },
            'model_exports': self.final_model_paths,
            'training_efficiency': {
                'avg_epoch_time': summary.get('avg_epoch_time', 0),
                'epochs_per_minute': summary.get('epochs_completed', 0) / (training_duration / 60) if training_duration > 0 else 0,
                'gpu_utilization': 'High' if self.training_config.device == 'cuda' else 'N/A'
            },
            'system_info': self._get_system_info(),
            'configuration': {
                'training_config': self.training_config.__dict__,
                'persistence_config': self.persistence_config.__dict__
            }
        }

        return report

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            'torch_version': torch.__version__,
            'device': self.training_config.device,
            'cuda_available': torch.cuda.is_available()
        }

        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'cuda_version': torch.version.cuda
            })

        return info

    def create_deployment_package(self) -> str:
        """Create complete deployment package"""
        if not self.training_completed:
            print("❌ Training not completed - cannot create deployment package")
            return ""

        print("\n📦 Creating Deployment Package")
        print("-" * 30)

        # Export visualizations
        self.visualizer.export_training_plots("training_plots")

        # Generate final analysis plots
        if self.training_data:
            # Create prediction analysis
            self.model.eval()
            predictions = []
            targets = []
            logo_types = []

            with torch.no_grad():
                for example in self.training_data[:100]:  # Sample for analysis
                    features = np.concatenate([
                        example.image_features,
                        list(example.vtracer_params.values())
                    ])
                    input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.training_config.device)
                    pred = self.model(input_tensor).cpu().item()

                    predictions.append(pred)
                    targets.append(example.actual_ssim)
                    logo_types.append(getattr(example, 'logo_type', 'unknown'))

            self.visualizer.plot_prediction_analysis(predictions, targets, logo_types)

        print("✅ Deployment package created")
        return "deployment_complete"

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'data_loaded': len(self.training_data) > 0,
            'model_initialized': self.model is not None,
            'training_setup': self.training_pipeline is not None,
            'training_completed': self.training_completed,
            'drive_connected': self.persistence_manager.drive_mounted,
            'available_checkpoints': len(self.persistence_manager.list_available_checkpoints())
        }


def create_colab_training_session(
    data_sources: List[str],
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001
) -> ColabTrainingOrchestrator:
    """Create and configure a complete Colab training session"""

    print("🚀 Creating Colab Training Session")
    print("=" * 50)

    # Create training configuration
    training_config = ColabTrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Create orchestrator
    orchestrator = ColabTrainingOrchestrator(training_config)

    # Load training data
    data_count = orchestrator.load_training_data(data_sources)

    if data_count == 0:
        print("❌ No training data loaded - session creation failed")
        return None

    # Setup training
    if not orchestrator.setup_training():
        print("❌ Training setup failed - session creation failed")
        return None

    print(f"✅ Colab training session ready!")
    print(f"   Data: {data_count} examples")
    print(f"   Device: {training_config.device}")
    print(f"   Epochs: {epochs}")

    return orchestrator


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Colab Training Orchestrator")

    # Test configuration
    config = ColabTrainingConfig(epochs=3, batch_size=16)
    orchestrator = ColabTrainingOrchestrator(config)

    # Test status
    status = orchestrator.get_training_status()
    print(f"Status: {status}")

    print("✅ Orchestrator test complete")