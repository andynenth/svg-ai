"""
GPU Training Pipeline for SVG Quality Prediction
Implements complete training pipeline with DataLoader, mixed precision, and monitoring
Part of Task 11.2.2: GPU Training Pipeline Implementation
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import glob
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import time
from collections import defaultdict
import warnings

from .gpu_model_architecture import (
    QualityPredictorGPU,
    GPUFeatureExtractor,
    ColabTrainingConfig,
    ModelOptimizer
)

warnings.filterwarnings('ignore')


@dataclass
class ColabTrainingExample:
    """Training example structure for Colab environment"""
    image_path: str
    image_features: np.ndarray  # 2048 ResNet features (GPU extracted)
    vtracer_params: Dict[str, float]  # 8 normalized parameters
    actual_ssim: float  # Ground truth [0,1]
    logo_type: str  # simple, text, gradient, complex
    optimization_method: str  # method1, method2, method3


class QualityDataset(Dataset):
    """GPU-optimized dataset for Colab training"""

    def __init__(self, training_examples: List[ColabTrainingExample], device='cuda'):
        self.examples = training_examples
        self.device = device

        # Pre-compute and cache all features for GPU efficiency
        self.features = []
        self.targets = []

        print(f"📦 Preparing dataset with {len(training_examples)} examples...")

        for i, example in enumerate(training_examples):
            try:
                # Normalize VTracer parameters
                normalized_params = self._normalize_vtracer_params(example.vtracer_params)

                # Combine image features + parameters
                combined = np.concatenate([
                    example.image_features,  # 2048 dims
                    normalized_params  # 8 dims
                ])

                self.features.append(torch.FloatTensor(combined))
                self.targets.append(torch.FloatTensor([example.actual_ssim]))

                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{len(training_examples)} examples")

            except Exception as e:
                print(f"❌ Error processing example {i}: {e}")
                # Skip invalid examples
                continue

        print(f"✅ Dataset prepared: {len(self.features)} valid examples")

    def _normalize_vtracer_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize VTracer parameters to [0,1] range"""
        normalized = [
            params.get('color_precision', 6.0) / 10.0,  # [0-10] → [0-1]
            params.get('corner_threshold', 60.0) / 100.0,  # [0-100] → [0-1]
            params.get('length_threshold', 4.0) / 10.0,  # [0-10] → [0-1]
            params.get('max_iterations', 10) / 20.0,  # [1-20] → [0-1]
            params.get('splice_threshold', 45.0) / 100.0,  # [0-100] → [0-1]
            params.get('path_precision', 8) / 16.0,  # [1-16] → [0-1]
            params.get('layer_difference', 16.0) / 32.0,  # [0-32] → [0-1]
            params.get('mode', 0) / 1.0  # spline=0, polygon=1 → [0-1]
        ]
        return np.array(normalized, dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def get_statistics(self):
        """Get dataset statistics"""
        if not self.targets:
            return {}

        targets_np = torch.stack(self.targets).numpy().flatten()
        return {
            'count': len(self.targets),
            'ssim_mean': float(np.mean(targets_np)),
            'ssim_std': float(np.std(targets_np)),
            'ssim_min': float(np.min(targets_np)),
            'ssim_max': float(np.max(targets_np))
        }


class GPUDataLoader:
    """GPU-optimized DataLoader factory with validation split"""

    @staticmethod
    def create_dataloaders(
        training_examples: List[ColabTrainingExample],
        config: ColabTrainingConfig
    ) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Create train and validation DataLoaders"""

        print(f"🔄 Creating DataLoaders for {len(training_examples)} examples...")

        # Shuffle for better training
        np.random.shuffle(training_examples)

        # Train/validation split
        split_idx = int(len(training_examples) * (1 - config.validation_split))
        train_data = training_examples[:split_idx]
        val_data = training_examples[split_idx:]

        print(f"   Train: {len(train_data)} examples")
        print(f"   Validation: {len(val_data)} examples")

        # Create datasets
        train_dataset = QualityDataset(train_data, config.device)
        val_dataset = QualityDataset(val_data, config.device)

        # Create DataLoaders with GPU optimization
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=config.pin_memory,
            num_workers=config.num_workers,
            drop_last=True  # For stable batch norm
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=config.pin_memory,
            num_workers=config.num_workers
        )

        # Compile statistics
        train_stats = train_dataset.get_statistics()
        val_stats = val_dataset.get_statistics()

        statistics = {
            'train': train_stats,
            'validation': val_stats,
            'total_examples': len(training_examples),
            'batch_size': config.batch_size,
            'batches_per_epoch': len(train_loader)
        }

        print(f"✅ DataLoaders created:")
        print(f"   Batches per epoch: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")

        return train_loader, val_loader, statistics


class TrainingMetrics:
    """Track and analyze training metrics"""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_correlations = []
        self.learning_rates = []
        self.epoch_times = []
        self.best_val_loss = float('inf')
        self.best_correlation = 0.0
        self.epochs_without_improvement = 0

    def update(self, epoch: int, train_loss: float, val_loss: float,
               val_correlation: float, lr: float, epoch_time: float):
        """Update metrics for current epoch"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_correlations.append(val_correlation)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

        # Track best performance
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if val_correlation > self.best_correlation:
            self.best_correlation = val_correlation

    def should_early_stop(self, patience: int) -> bool:
        """Check if training should stop early"""
        return self.epochs_without_improvement >= patience

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if not self.train_losses:
            return {}

        return {
            'epochs_completed': len(self.train_losses),
            'best_val_loss': self.best_val_loss,
            'best_correlation': self.best_correlation,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_correlation': self.val_correlations[-1],
            'avg_epoch_time': np.mean(self.epoch_times),
            'total_training_time': sum(self.epoch_times),
            'epochs_without_improvement': self.epochs_without_improvement
        }


class GPUTrainingPipeline:
    """Complete GPU training pipeline for Colab environment"""

    def __init__(self, config: ColabTrainingConfig):
        self.config = config
        self.device = config.device
        self.metrics = TrainingMetrics()

        # Initialize model and optimizer
        self.model = QualityPredictorGPU(config)
        self.optimizer_wrapper = ModelOptimizer(self.model, config)

        print(f"🚀 GPU Training Pipeline initialized on {config.device}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              save_checkpoints: bool = True) -> Dict[str, Any]:
        """Execute complete training loop"""

        print(f"\n🎯 Starting GPU Training")
        print(f"   Epochs: {self.config.epochs}")
        print(f"   Batch Size: {self.config.batch_size}")
        print(f"   Learning Rate: {self.config.learning_rate}")
        print(f"   Mixed Precision: {self.config.mixed_precision}")
        print("=" * 50)

        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Training phase
            train_loss = self._train_epoch(train_loader)

            # Validation phase
            val_loss, val_correlation = self.optimizer_wrapper.validation_step(val_loader)

            # Update scheduler
            self.optimizer_wrapper.step_scheduler()
            current_lr = self.optimizer_wrapper.get_lr()

            # Update metrics
            epoch_time = time.time() - epoch_start
            self.metrics.update(epoch, train_loss, val_loss, val_correlation,
                              current_lr, epoch_time)

            # Progress logging
            self._log_progress(epoch, train_loss, val_loss, val_correlation,
                             current_lr, epoch_time)

            # Checkpointing
            if save_checkpoints and (epoch + 1) % self.config.checkpoint_freq == 0:
                self._save_checkpoint(epoch, val_loss, val_correlation)

            # Early stopping check
            if self.metrics.should_early_stop(self.config.early_stopping_patience):
                print(f"\n⏹️ Early stopping triggered after {epoch + 1} epochs")
                print(f"   No improvement for {self.metrics.epochs_without_improvement} epochs")
                break

            # Target correlation check
            if val_correlation >= 0.9:
                print(f"\n🎯 Target correlation achieved: {val_correlation:.4f}")
                break

        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.1f}s")

        return self._generate_training_report(total_time)

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (batch_features, batch_targets) in enumerate(train_loader):
            loss, _ = self.optimizer_wrapper.training_step(batch_features, batch_targets)
            total_loss += loss

            # Optional: Progress within epoch for long epochs
            if batch_idx % 50 == 0 and batch_idx > 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"     Batch {batch_idx}/{len(train_loader)}: Loss {avg_loss:.4f}")

        return total_loss / len(train_loader)

    def _log_progress(self, epoch: int, train_loss: float, val_loss: float,
                     val_correlation: float, lr: float, epoch_time: float):
        """Log training progress"""
        print(f"Epoch {epoch + 1:3d}/{self.config.epochs}: "
              f"Train {train_loss:.4f} | "
              f"Val {val_loss:.4f} | "
              f"Corr {val_correlation:.4f} | "
              f"LR {lr:.2e} | "
              f"Time {epoch_time:.1f}s")

        # Mark best epochs
        if val_loss <= self.metrics.best_val_loss:
            print("     ⭐ Best validation loss!")
        if val_correlation >= self.metrics.best_correlation:
            print("     🎯 Best correlation!")

    def _save_checkpoint(self, epoch: int, val_loss: float, val_correlation: float):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer_wrapper.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_correlation': val_correlation,
            'config': asdict(self.config),
            'metrics': {
                'train_losses': self.metrics.train_losses,
                'val_losses': self.metrics.val_losses,
                'val_correlations': self.metrics.val_correlations
            }
        }

        checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"     💾 Checkpoint saved: {checkpoint_path}")

    def _generate_training_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        summary = self.metrics.get_summary()

        report = {
            'training_config': asdict(self.config),
            'training_summary': summary,
            'performance_metrics': {
                'total_training_time': total_time,
                'average_epoch_time': summary.get('avg_epoch_time', 0),
                'epochs_per_minute': summary.get('epochs_completed', 0) / (total_time / 60),
                'final_correlation': summary.get('final_correlation', 0),
                'target_achieved': summary.get('best_correlation', 0) >= 0.9
            },
            'loss_curves': {
                'train_losses': self.metrics.train_losses,
                'val_losses': self.metrics.val_losses,
                'val_correlations': self.metrics.val_correlations,
                'learning_rates': self.metrics.learning_rates
            },
            'gpu_info': self._get_gpu_info()
        }

        # Save report
        report_path = f"training_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📊 Training report saved: {report_path}")
        return report

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        if self.device == 'cuda' and torch.cuda.is_available():
            return {
                'device_name': torch.cuda.get_device_name(0),
                'memory_allocated_gb': torch.cuda.memory_allocated(0) / 1e9,
                'memory_cached_gb': torch.cuda.memory_reserved(0) / 1e9,
                'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        else:
            return {'device': 'cpu'}

    def export_model(self, export_path: str = "quality_predictor_gpu.pth"):
        """Export trained model"""
        model_export = {
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'training_summary': self.metrics.get_summary(),
            'architecture_info': {
                'input_dim': 2056,
                'hidden_dims': self.config.hidden_dims,
                'output_dim': 1,
                'total_parameters': self.model.count_parameters()
            }
        }

        torch.save(model_export, export_path)
        print(f"📦 Model exported: {export_path}")
        return export_path


def load_optimization_data(data_patterns: List[str]) -> List[ColabTrainingExample]:
    """Load training data from optimization results"""
    training_examples = []
    feature_extractor = None

    print("📂 Loading optimization data...")

    for pattern in data_patterns:
        files = glob.glob(pattern, recursive=True)
        print(f"   Found {len(files)} files matching {pattern}")

        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                examples = extract_training_examples(data, feature_extractor)
                training_examples.extend(examples)

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

    print(f"✅ Loaded {len(training_examples)} training examples")
    return training_examples


def extract_training_examples(data: Dict, feature_extractor: Optional[GPUFeatureExtractor] = None) -> List[ColabTrainingExample]:
    """Extract training examples from optimization results"""
    examples = []

    # Initialize feature extractor if needed
    if feature_extractor is None:
        feature_extractor = GPUFeatureExtractor()

    # Extract examples based on data structure
    if 'optimization_results' in data:
        # Method 1, 2, 3 results
        for result in data['optimization_results']:
            if 'image_path' in result and 'best_params' in result and 'best_ssim' in result:
                try:
                    features = feature_extractor.extract_features_single(result['image_path'])

                    example = ColabTrainingExample(
                        image_path=result['image_path'],
                        image_features=features,
                        vtracer_params=result['best_params'],
                        actual_ssim=result['best_ssim'],
                        logo_type=result.get('logo_type', 'unknown'),
                        optimization_method=result.get('method', 'unknown')
                    )
                    examples.append(example)

                except Exception as e:
                    print(f"❌ Error processing result: {e}")

    return examples


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing GPU Training Pipeline")

    # Create sample config
    config = ColabTrainingConfig(
        epochs=5,  # Short test
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Create pipeline
    pipeline = GPUTrainingPipeline(config)
    print("✅ Pipeline creation successful")