"""
Colab Training Utilities and Visualization
==========================================

Real-time training monitoring, visualization, and persistence utilities
for Google Colab GPU training environment.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import json
import shutil
from datetime import datetime
from IPython.display import clear_output, display
from typing import List, Dict, Any
try:
    import ipywidgets as widgets
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

class ColabTrainingMonitor:
    """Real-time training monitoring for Colab"""

    def __init__(self, save_plots=True):
        self.save_plots = save_plots
        self.plot_history = []

    def plot_training_progress(self, train_losses, val_losses, val_correlations, epoch=None):
        """Real-time training progress visualization"""
        clear_output(wait=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('GPU Training Progress - Real Time', fontsize=16, fontweight='bold')

        # Loss curves
        epochs = range(1, len(train_losses) + 1)

        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')  # Log scale for better visualization

        # Correlation tracking
        axes[0, 1].plot(epochs, val_correlations, 'g-', label='Validation Correlation', linewidth=2)
        axes[0, 1].axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='Target (0.9)')
        axes[0, 1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Excellent (0.95)')
        axes[0, 1].set_title('Validation Correlation Progress')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Pearson Correlation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)

        # Loss difference (overfitting detection)
        if len(train_losses) > 1:
            loss_diff = np.array(val_losses) - np.array(train_losses)
            axes[1, 0].plot(epochs, loss_diff, 'purple', linewidth=2)
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 0].fill_between(epochs, loss_diff, 0,
                                  where=(loss_diff > 0), color='red', alpha=0.3, label='Overfitting')
            axes[1, 0].fill_between(epochs, loss_diff, 0,
                                  where=(loss_diff <= 0), color='green', alpha=0.3, label='Good fit')
            axes[1, 0].set_title('Overfitting Detection (Val - Train Loss)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Difference')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Recent performance window
        window_size = min(10, len(train_losses))
        if window_size > 1:
            recent_epochs = epochs[-window_size:]
            recent_train = train_losses[-window_size:]
            recent_val = val_losses[-window_size:]
            recent_corr = val_correlations[-window_size:]

            ax2 = axes[1, 1]
            ax2_twin = ax2.twinx()

            # Plot recent losses
            line1 = ax2.plot(recent_epochs, recent_train, 'b-', label='Train Loss', linewidth=2)
            line2 = ax2.plot(recent_epochs, recent_val, 'r-', label='Val Loss', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')

            # Plot recent correlation
            line3 = ax2_twin.plot(recent_epochs, recent_corr, 'g-', label='Val Corr', linewidth=2)
            ax2_twin.set_ylabel('Correlation', color='green')
            ax2_twin.tick_params(axis='y', labelcolor='green')

            # Combine legends
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left')

            ax2.set_title(f'Recent Performance (Last {window_size} epochs)')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print current status
        if epoch is not None and len(val_correlations) > 0:
            current_corr = val_correlations[-1]
            current_val_loss = val_losses[-1]
            print(f"📊 Epoch {epoch}: Val Correlation: {current_corr:.4f}, Val Loss: {current_val_loss:.6f}")

            # Performance indicators
            if current_corr > 0.95:
                print("🎯 Excellent correlation achieved!")
            elif current_corr > 0.9:
                print("✅ Good correlation - target reached!")
            elif current_corr > 0.8:
                print("🔄 Decent correlation - improving...")
            else:
                print("⚠️ Low correlation - needs more training")

        if self.save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = f'/content/svg_quality_predictor/training_progress_{timestamp}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.plot_history.append(plot_path)

    def create_training_dashboard(self, config):
        """Create interactive training dashboard"""
        print("🖥️ Training Dashboard")
        print("="*50)

        # GPU Status
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        else:
            print("⚠️ No GPU available - using CPU")

        # Training Configuration
        print(f"⚙️ Configuration:")
        print(f"   Batch Size: {config.batch_size}")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   Max Epochs: {config.epochs}")
        print(f"   Mixed Precision: {config.mixed_precision}")
        print(f"   Early Stopping: {config.early_stopping_patience} epochs")

        # Create progress widgets if available
        if WIDGETS_AVAILABLE:
            self.epoch_progress = widgets.IntProgress(
                value=0,
                min=0,
                max=config.epochs,
                description='Epoch:',
                bar_style='info',
                style={'bar_color': 'lightblue'},
                orientation='horizontal'
            )

            self.loss_display = widgets.HTML(
                value="<b>Loss:</b> Waiting for training..."
            )

            self.correlation_display = widgets.HTML(
                value="<b>Correlation:</b> Waiting for training..."
            )

            self.status_display = widgets.HTML(
                value="<b>Status:</b> Ready to start training"
            )

            # Display widgets
            display(widgets.VBox([
                widgets.HTML("<h3>Training Progress</h3>"),
                self.epoch_progress,
                self.loss_display,
                self.correlation_display,
                self.status_display
            ]))
        else:
            print("⚠️ Interactive widgets not available - using text-based progress monitoring")

    def update_dashboard(self, epoch, train_loss, val_loss, val_corr, status="Training..."):
        """Update training dashboard"""
        if WIDGETS_AVAILABLE:
            if hasattr(self, 'epoch_progress'):
                self.epoch_progress.value = epoch

            if hasattr(self, 'loss_display'):
                self.loss_display.value = f"<b>Loss:</b> Train: {train_loss:.6f}, Val: {val_loss:.6f}"

            if hasattr(self, 'correlation_display'):
                self.correlation_display.value = f"<b>Correlation:</b> {val_corr:.4f}"

            if hasattr(self, 'status_display'):
                self.status_display.value = f"<b>Status:</b> {status}"
        else:
            # Text-based progress update
            print(f"Epoch {epoch}: Loss: {train_loss:.6f}/{val_loss:.6f}, Corr: {val_corr:.4f}, Status: {status}")

def save_training_checkpoint(model, optimizer, epoch, losses, drive_path="/content/drive/MyDrive/svg_quality_predictor_backups"):
    """Save training state to Google Drive"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': losses['train'],
        'val_losses': losses['val'],
        'val_correlations': losses['correlations'],
        'timestamp': str(datetime.now()),
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    }

    # Save to local storage
    local_path = f"/content/svg_quality_predictor/models/checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, local_path)

    # Backup to Google Drive
    drive_checkpoint_path = f"{drive_path}/checkpoint_epoch_{epoch}.pth"
    shutil.copy2(local_path, drive_checkpoint_path)

    print(f"💾 Checkpoint saved: {local_path}")
    print(f"☁️ Backup saved: {drive_checkpoint_path}")

    return local_path

def analyze_model_performance(model, val_loader, device='cuda'):
    """Comprehensive model performance analysis"""
    model.eval()
    predictions = []
    targets = []
    feature_activations = []

    print("🔍 Analyzing model performance...")

    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_features)

            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(batch_targets.cpu().numpy().flatten())

            # Extract intermediate features for analysis
            intermediate = model.feature_network[:-1](batch_features)  # All but final layer
            feature_activations.extend(intermediate.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)
    feature_activations = np.array(feature_activations)

    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    correlation = np.corrcoef(predictions, targets)[0, 1]
    r2_score = 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))

    # Create comprehensive analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')

    # Prediction vs Target scatter
    axes[0, 0].scatter(targets, predictions, alpha=0.6, s=30)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2)
    axes[0, 0].set_xlabel('True SSIM')
    axes[0, 0].set_ylabel('Predicted SSIM')
    axes[0, 0].set_title(f'Predictions vs Targets\nr = {correlation:.4f}')
    axes[0, 0].grid(True, alpha=0.3)

    # Residuals plot
    residuals = predictions - targets
    axes[0, 1].scatter(targets, residuals, alpha=0.6, s=30)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('True SSIM')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Analysis')
    axes[0, 1].grid(True, alpha=0.3)

    # Error distribution
    axes[0, 2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(x=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('Prediction Error')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title(f'Error Distribution\nMAE = {mae:.4f}')
    axes[0, 2].grid(True, alpha=0.3)

    # Feature activation heatmap
    if feature_activations.shape[1] > 1:
        # Sample features for visualization
        sample_features = feature_activations[:50, :min(50, feature_activations.shape[1])]
        im = axes[1, 0].imshow(sample_features.T, aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Feature Activations (Sample)')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Feature Index')
        plt.colorbar(im, ax=axes[1, 0])

    # Prediction confidence analysis
    confidence_scores = 1 - np.abs(residuals)  # Simple confidence metric
    axes[1, 1].scatter(predictions, confidence_scores, alpha=0.6, s=30)
    axes[1, 1].set_xlabel('Predicted SSIM')
    axes[1, 1].set_ylabel('Confidence Score')
    axes[1, 1].set_title('Prediction Confidence')
    axes[1, 1].grid(True, alpha=0.3)

    # Quality binned analysis
    quality_bins = ['Low (<0.7)', 'Medium (0.7-0.9)', 'High (>0.9)']
    bin_mse = []
    bin_counts = []

    for i, (low, high) in enumerate([(0, 0.7), (0.7, 0.9), (0.9, 1.0)]):
        mask = (targets >= low) & (targets < high) if i < 2 else (targets >= low)
        if np.any(mask):
            bin_mse.append(np.mean((predictions[mask] - targets[mask]) ** 2))
            bin_counts.append(np.sum(mask))
        else:
            bin_mse.append(0)
            bin_counts.append(0)

    x_pos = np.arange(len(quality_bins))
    bars = axes[1, 2].bar(x_pos, bin_mse, alpha=0.7, color=['lightcoral', 'gold', 'lightgreen'])
    axes[1, 2].set_xlabel('Quality Range')
    axes[1, 2].set_ylabel('MSE')
    axes[1, 2].set_title('Error by Quality Range')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(quality_bins)

    # Add count labels on bars
    for bar, count in zip(bars, bin_counts):
        height = bar.get_height()
        axes[1, 2].annotate(f'n={count}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Print detailed metrics
    print("\n" + "="*60)
    print("DETAILED PERFORMANCE METRICS")
    print("="*60)
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Pearson Correlation: {correlation:.6f}")
    print(f"R² Score: {r2_score:.6f}")
    print(f"Standard Deviation of Residuals: {np.std(residuals):.6f}")

    # Quality-specific metrics
    print(f"\nQuality Range Analysis:")
    for i, (bin_name, mse_val, count) in enumerate(zip(quality_bins, bin_mse, bin_counts)):
        print(f"  {bin_name}: MSE = {mse_val:.6f}, Count = {count}")

    # Performance rating
    if correlation > 0.95:
        rating = "Excellent 🎯"
    elif correlation > 0.9:
        rating = "Good ✅"
    elif correlation > 0.8:
        rating = "Acceptable 🔄"
    else:
        rating = "Needs Improvement ⚠️"

    print(f"\nOverall Performance Rating: {rating}")
    print("="*60)

    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'r2_score': r2_score,
        'residual_std': np.std(residuals),
        'quality_bin_mse': dict(zip(quality_bins, bin_mse)),
        'quality_bin_counts': dict(zip(quality_bins, bin_counts)),
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'rating': rating
    }

def create_final_report(training_results, performance_analysis, config):
    """Create comprehensive final training report"""
    report = {
        'training_summary': {
            'timestamp': str(datetime.now()),
            'total_epochs': len(training_results['train_losses']),
            'best_validation_loss': training_results['best_val_loss'],
            'final_correlation': training_results['final_correlation'],
            'early_stopping_triggered': len(training_results['train_losses']) < config.epochs
        },
        'model_performance': performance_analysis,
        'training_config': {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'mixed_precision': config.mixed_precision,
            'device': config.device,
            'optimizer': config.optimizer
        },
        'gpu_info': {
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
        },
        'training_curves': {
            'train_losses': training_results['train_losses'],
            'val_losses': training_results['val_losses'],
            'val_correlations': training_results['val_correlations']
        }
    }

    # Save report
    report_path = '/content/svg_quality_predictor/final_training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Backup to Drive
    backup_path = '/content/drive/MyDrive/svg_quality_predictor_backups/final_training_report.json'
    shutil.copy2(report_path, backup_path)

    print(f"📋 Final report saved: {report_path}")
    print(f"☁️ Report backup: {backup_path}")

    return report

def setup_colab_environment():
    """Setup and validate Colab environment"""
    print("🔧 Setting up Colab environment...")

    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU available - training will be slow")

    # Create directories
    directories = [
        '/content/svg_quality_predictor/models',
        '/content/svg_quality_predictor/exports',
        '/content/svg_quality_predictor/plots',
        '/content/drive/MyDrive/svg_quality_predictor_backups'
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created: {dir_path}")

    print("✅ Colab environment setup complete!")

if __name__ == "__main__":
    # Test utilities
    print("Testing Colab training utilities...")
    setup_colab_environment()
    print("✅ All utilities ready!")