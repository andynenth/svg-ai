"""
Colab Training Visualization & Monitoring System
Real-time training progress visualization with matplotlib integration
Part of Task 11.2.3: Colab Training Monitoring & Visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from IPython.display import clear_output, display, HTML
import json
from pathlib import Path
import warnings

# Set matplotlib backend for Colab compatibility
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


class ColabTrainingVisualizer:
    """Real-time training visualization for Colab environment"""

    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        self.figsize = figsize
        self.train_losses = []
        self.val_losses = []
        self.val_correlations = []
        self.learning_rates = []
        self.epoch_times = []
        self.gpu_memory_usage = []

        # Color scheme
        self.colors = {
            'train': '#3498db',  # Blue
            'validation': '#e74c3c',  # Red
            'correlation': '#2ecc71',  # Green
            'target': '#f39c12',  # Orange
            'lr': '#9b59b6',  # Purple
            'memory': '#1abc9c'  # Teal
        }

        # Setup matplotlib for Colab
        plt.ioff()  # Turn off interactive mode
        self.fig = None

    def update_metrics(self, train_loss: float, val_loss: float,
                      val_correlation: float, learning_rate: float,
                      epoch_time: float, gpu_memory_gb: float = 0.0):
        """Update metrics for visualization"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_correlations.append(val_correlation)
        self.learning_rates.append(learning_rate)
        self.epoch_times.append(epoch_time)
        self.gpu_memory_usage.append(gpu_memory_gb)

    def plot_training_progress(self, show_predictions: bool = True):
        """Plot comprehensive training progress"""
        if not self.train_losses:
            print("No training data to plot")
            return

        clear_output(wait=True)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle('🚀 SVG Quality Predictor - GPU Training Progress', fontsize=16, fontweight='bold')

        epochs = range(1, len(self.train_losses) + 1)

        # 1. Loss curves
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.train_losses, 'o-', color=self.colors['train'],
                label='Training Loss', linewidth=2, markersize=4)
        ax1.plot(epochs, self.val_losses, 's-', color=self.colors['validation'],
                label='Validation Loss', linewidth=2, markersize=4)
        ax1.set_title('📈 Training & Validation Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. Correlation progress
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.val_correlations, 'o-', color=self.colors['correlation'],
                label='Validation Correlation', linewidth=2, markersize=4)
        ax2.axhline(y=0.9, color=self.colors['target'], linestyle='--',
                   label='Target (0.9)', linewidth=2)
        ax2.set_title('🎯 Validation Correlation Progress', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Pearson Correlation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # Add correlation status
        current_corr = self.val_correlations[-1] if self.val_correlations else 0
        if current_corr >= 0.9:
            ax2.text(0.02, 0.98, '✅ Target Achieved!', transform=ax2.transAxes,
                    fontsize=12, fontweight='bold', color='green',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))

        # 3. Learning rate schedule
        ax3 = axes[0, 2]
        ax3.plot(epochs, self.learning_rates, 'o-', color=self.colors['lr'],
                linewidth=2, markersize=4)
        ax3.set_title('📊 Learning Rate Schedule', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # 4. Training speed
        ax4 = axes[1, 0]
        if self.epoch_times:
            ax4.plot(epochs, self.epoch_times, 'o-', color=self.colors['train'],
                    linewidth=2, markersize=4)
            ax4.set_title('⏱️ Training Speed', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Time per Epoch (s)')
            ax4.grid(True, alpha=0.3)

            # Add average time annotation
            avg_time = np.mean(self.epoch_times)
            ax4.axhline(y=avg_time, color='red', linestyle='--', alpha=0.7)
            ax4.text(0.02, 0.98, f'Avg: {avg_time:.1f}s', transform=ax4.transAxes,
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        # 5. GPU Memory usage
        ax5 = axes[1, 1]
        if self.gpu_memory_usage and any(mem > 0 for mem in self.gpu_memory_usage):
            ax5.plot(epochs, self.gpu_memory_usage, 'o-', color=self.colors['memory'],
                    linewidth=2, markersize=4)
            ax5.set_title('💾 GPU Memory Usage', fontweight='bold')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Memory (GB)')
            ax5.grid(True, alpha=0.3)

        # 6. Training summary
        ax6 = axes[1, 2]
        ax6.axis('off')

        # Create summary text
        current_epoch = len(self.train_losses)
        best_val_loss = min(self.val_losses) if self.val_losses else 0
        best_correlation = max(self.val_correlations) if self.val_correlations else 0
        total_time = sum(self.epoch_times) if self.epoch_times else 0

        summary_text = [
            "📋 Training Summary",
            "",
            f"Current Epoch: {current_epoch}",
            f"Best Val Loss: {best_val_loss:.4f}",
            f"Best Correlation: {best_correlation:.4f}",
            f"Total Time: {total_time:.1f}s",
            "",
            "🎯 Status:",
            f"{'✅' if best_correlation >= 0.9 else '🔄'} Target Correlation",
            f"{'✅' if best_val_loss < 0.01 else '🔄'} Loss Convergence"
        ]

        y_pos = 0.9
        for line in summary_text:
            fontweight = 'bold' if line.startswith(('📋', '🎯')) else 'normal'
            fontsize = 12 if line.startswith(('📋', '🎯')) else 10
            ax6.text(0.05, y_pos, line, transform=ax6.transAxes,
                    fontsize=fontsize, fontweight=fontweight)
            y_pos -= 0.08

        plt.tight_layout()
        plt.show()

        # Display current metrics
        if len(self.train_losses) > 0:
            self._display_current_metrics()

    def _display_current_metrics(self):
        """Display current training metrics as HTML table"""
        current_idx = len(self.train_losses) - 1

        metrics_html = f"""
        <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">
            <h4 style="color: #343a40; margin-bottom: 10px;">📊 Current Metrics (Epoch {current_idx + 1})</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background-color: #dee2e6;">
                    <td style="padding: 8px; border: 1px solid #adb5bd;"><strong>Metric</strong></td>
                    <td style="padding: 8px; border: 1px solid #adb5bd;"><strong>Value</strong></td>
                    <td style="padding: 8px; border: 1px solid #adb5bd;"><strong>Status</strong></td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #adb5bd;">Train Loss</td>
                    <td style="padding: 8px; border: 1px solid #adb5bd;">{self.train_losses[current_idx]:.4f}</td>
                    <td style="padding: 8px; border: 1px solid #adb5bd;">{'🟢' if self.train_losses[current_idx] < 0.01 else '🟡' if self.train_losses[current_idx] < 0.05 else '🔴'}</td>
                </tr>
                <tr style="background-color: #f8f9fa;">
                    <td style="padding: 8px; border: 1px solid #adb5bd;">Val Loss</td>
                    <td style="padding: 8px; border: 1px solid #adb5bd;">{self.val_losses[current_idx]:.4f}</td>
                    <td style="padding: 8px; border: 1px solid #adb5bd;">{'🟢' if self.val_losses[current_idx] < 0.01 else '🟡' if self.val_losses[current_idx] < 0.05 else '🔴'}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #adb5bd;">Correlation</td>
                    <td style="padding: 8px; border: 1px solid #adb5bd;">{self.val_correlations[current_idx]:.4f}</td>
                    <td style="padding: 8px; border: 1px solid #adb5bd;">{'🟢' if self.val_correlations[current_idx] >= 0.9 else '🟡' if self.val_correlations[current_idx] >= 0.8 else '🔴'}</td>
                </tr>
                <tr style="background-color: #f8f9fa;">
                    <td style="padding: 8px; border: 1px solid #adb5bd;">Learning Rate</td>
                    <td style="padding: 8px; border: 1px solid #adb5bd;">{self.learning_rates[current_idx]:.2e}</td>
                    <td style="padding: 8px; border: 1px solid #adb5bd;">📊</td>
                </tr>
            </table>
        </div>
        """

        display(HTML(metrics_html))

    def plot_prediction_analysis(self, predictions: List[float], targets: List[float],
                               logo_types: Optional[List[str]] = None):
        """Plot prediction vs target analysis"""
        if not predictions or not targets:
            print("No prediction data available")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('🎯 Prediction Analysis', fontsize=14, fontweight='bold')

        # 1. Scatter plot
        ax1 = axes[0]
        ax1.scatter(targets, predictions, alpha=0.6, color=self.colors['correlation'])
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.8)  # Perfect prediction line
        ax1.set_xlabel('Actual SSIM')
        ax1.set_ylabel('Predicted SSIM')
        ax1.set_title('Predictions vs Targets')
        ax1.grid(True, alpha=0.3)

        # Calculate and display correlation
        correlation = np.corrcoef(predictions, targets)[0, 1]
        ax1.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
                fontsize=12, fontweight='bold')

        # 2. Error distribution
        ax2 = axes[1]
        errors = np.array(predictions) - np.array(targets)
        ax2.hist(errors, bins=20, alpha=0.7, color=self.colors['validation'])
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.grid(True, alpha=0.3)

        # Add error statistics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        ax2.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}',
                transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
                fontsize=10)

        # 3. Performance by logo type (if available)
        ax3 = axes[2]
        if logo_types:
            unique_types = list(set(logo_types))
            type_correlations = []

            for logo_type in unique_types:
                type_indices = [i for i, t in enumerate(logo_types) if t == logo_type]
                type_preds = [predictions[i] for i in type_indices]
                type_targets = [targets[i] for i in type_indices]

                if len(type_preds) > 1:
                    type_corr = np.corrcoef(type_preds, type_targets)[0, 1]
                    type_correlations.append(type_corr)
                else:
                    type_correlations.append(0)

            bars = ax3.bar(unique_types, type_correlations, color=sns.color_palette("husl", len(unique_types)))
            ax3.set_ylabel('Correlation')
            ax3.set_title('Performance by Logo Type')
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation=45)

            # Add target line
            ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Logo type data\nnot available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Performance by Logo Type')

        plt.tight_layout()
        plt.show()

    def plot_loss_landscape(self, save_path: Optional[str] = None):
        """Plot loss landscape and convergence analysis"""
        if len(self.train_losses) < 5:
            print("Insufficient data for loss landscape analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('📊 Loss Landscape Analysis', fontsize=14, fontweight='bold')

        epochs = range(1, len(self.train_losses) + 1)

        # 1. Loss convergence with smoothing
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.train_losses, 'o-', alpha=0.6, color=self.colors['train'], label='Train Loss')
        ax1.plot(epochs, self.val_losses, 's-', alpha=0.6, color=self.colors['validation'], label='Val Loss')

        # Add smoothed curves
        if len(epochs) > 5:
            from scipy.ndimage import uniform_filter1d
            smoothed_train = uniform_filter1d(self.train_losses, size=min(5, len(epochs)//3))
            smoothed_val = uniform_filter1d(self.val_losses, size=min(5, len(epochs)//3))
            ax1.plot(epochs, smoothed_train, '-', linewidth=3, color=self.colors['train'], alpha=0.8)
            ax1.plot(epochs, smoothed_val, '-', linewidth=3, color=self.colors['validation'], alpha=0.8)

        ax1.set_title('Loss Convergence')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # 2. Overfitting analysis
        ax2 = axes[0, 1]
        gap = np.array(self.val_losses) - np.array(self.train_losses)
        ax2.plot(epochs, gap, 'o-', color='purple', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Overfitting Gap (Val - Train)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Difference')
        ax2.grid(True, alpha=0.3)

        # Add overfitting warning
        current_gap = gap[-1] if len(gap) > 0 else 0
        if current_gap > 0.02:
            ax2.text(0.02, 0.98, '⚠️ Potential Overfitting', transform=ax2.transAxes,
                    color='red', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))

        # 3. Training dynamics
        ax3 = axes[1, 0]
        if len(epochs) > 1:
            loss_gradient = np.gradient(self.val_losses)
            ax3.plot(epochs, loss_gradient, 'o-', color=self.colors['correlation'], linewidth=2)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_title('Loss Gradient (Rate of Change)')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss Gradient')
            ax3.grid(True, alpha=0.3)

        # 4. Performance heatmap
        ax4 = axes[1, 1]
        if len(epochs) >= 10:
            # Create performance matrix
            metrics_matrix = np.array([
                self.train_losses[-10:],
                self.val_losses[-10:],
                self.val_correlations[-10:]
            ])

            im = ax4.imshow(metrics_matrix, cmap='RdYlGn_r', aspect='auto')
            ax4.set_title('Recent Performance Heatmap')
            ax4.set_ylabel('Metrics')
            ax4.set_xlabel('Recent Epochs')
            ax4.set_yticks([0, 1, 2])
            ax4.set_yticklabels(['Train Loss', 'Val Loss', 'Correlation'])

            # Add colorbar
            plt.colorbar(im, ax=ax4, shrink=0.8)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor heatmap', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)

        plt.tight_layout()
        plt.show()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss landscape saved to {save_path}")

    def export_training_plots(self, export_dir: str = "training_plots"):
        """Export all training plots for documentation"""
        Path(export_dir).mkdir(exist_ok=True)

        # Main progress plot
        self.plot_training_progress(show_predictions=False)
        plt.savefig(f"{export_dir}/training_progress.png", dpi=300, bbox_inches='tight')

        # Loss landscape
        self.plot_loss_landscape(f"{export_dir}/loss_landscape.png")

        print(f"📊 Training plots exported to {export_dir}/")

    def create_training_gif(self, gif_path: str = "training_animation.gif", fps: int = 2):
        """Create animated GIF of training progress (if supported)"""
        # This would require additional setup in Colab
        print("🎬 GIF animation creation would require additional Colab setup")
        print("Consider using plot_training_progress() for real-time updates")


class ColabPerformanceMonitor:
    """Monitor Colab performance and resource usage"""

    def __init__(self):
        self.start_time = time.time()
        self.epoch_start_time = None
        self.memory_history = []

    def start_epoch(self):
        """Mark the start of an epoch"""
        self.epoch_start_time = time.time()

    def end_epoch(self) -> float:
        """Mark the end of an epoch and return duration"""
        if self.epoch_start_time is None:
            return 0.0
        return time.time() - self.epoch_start_time

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        import torch
        memory_info = {}

        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated(0) / 1e9
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved(0) / 1e9
            memory_info['gpu_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            memory_info['gpu_allocated'] = 0
            memory_info['gpu_reserved'] = 0
            memory_info['gpu_total'] = 0

        return memory_info

    def log_performance(self, epoch: int, train_loss: float, val_loss: float, val_corr: float):
        """Log performance metrics"""
        elapsed_time = time.time() - self.start_time
        memory_info = self.get_memory_usage()

        print(f"⚡ Epoch {epoch}: {elapsed_time:.1f}s total | "
              f"GPU: {memory_info['gpu_allocated']:.1f}GB | "
              f"Corr: {val_corr:.4f}")


if __name__ == "__main__":
    # Example usage
    print("🎨 Testing Colab Training Visualization")

    # Create visualizer
    viz = ColabTrainingVisualizer()

    # Simulate some training data
    for epoch in range(10):
        train_loss = 0.1 * np.exp(-epoch * 0.2) + 0.001
        val_loss = 0.12 * np.exp(-epoch * 0.15) + 0.002
        val_corr = min(0.95, 0.5 + epoch * 0.05)
        lr = 0.001 * (0.9 ** epoch)
        epoch_time = 30 + np.random.normal(0, 5)
        gpu_memory = 4.5 + np.random.normal(0, 0.2)

        viz.update_metrics(train_loss, val_loss, val_corr, lr, epoch_time, gpu_memory)

    # Plot results
    viz.plot_training_progress()
    print("✅ Visualization test complete")