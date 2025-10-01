"""
Day 12: Real-time Training Monitoring and Visualization
Advanced visualization and monitoring for GPU training execution
Part of Task 12.1.3: Real-time Training Monitoring and Progress Tracking
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass
from IPython.display import clear_output, display
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class TrainingSnapshot:
    """Single training snapshot for monitoring"""
    epoch: int
    train_loss: float
    val_loss: float
    val_correlation: float
    learning_rate: float
    epoch_time: float
    gpu_memory_gb: float
    timestamp: float


class RealTimeTrainingMonitor:
    """Real-time training monitoring with advanced visualization"""

    def __init__(self, save_dir: str = "/tmp/claude/training_logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.snapshots: List[TrainingSnapshot] = []
        self.best_correlation = 0.0
        self.best_epoch = 0
        self.target_correlation = 0.9

        # Monitoring flags
        self.real_time_enabled = True
        self.save_plots = True
        self.alert_thresholds = {
            'correlation_target': 0.9,
            'overfitting_ratio': 2.0,  # val_loss / train_loss
            'no_improvement_epochs': 5
        }

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  val_correlation: float, learning_rate: float, epoch_time: float,
                  gpu_memory_gb: float = 0.0):
        """Log a training epoch"""

        snapshot = TrainingSnapshot(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_correlation=val_correlation,
            learning_rate=learning_rate,
            epoch_time=epoch_time,
            gpu_memory_gb=gpu_memory_gb,
            timestamp=time.time()
        )

        self.snapshots.append(snapshot)

        # Update best performance tracking
        if val_correlation > self.best_correlation:
            self.best_correlation = val_correlation
            self.best_epoch = epoch

        # Real-time visualization
        if self.real_time_enabled and epoch % 3 == 0:  # Update every 3 epochs
            self.plot_training_progress_realtime()

        # Check for alerts
        self._check_training_alerts(snapshot)

    def plot_training_progress_realtime(self):
        """Real-time training progress visualization"""
        if len(self.snapshots) < 2:
            return

        if self.real_time_enabled:
            clear_output(wait=True)

        fig = plt.figure(figsize=(20, 12))

        # Create a complex dashboard layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Main loss curves (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_loss_curves(ax1)

        # Correlation progress (top right, spans 2 columns)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_correlation_progress(ax2)

        # Learning rate schedule (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_learning_rate(ax3)

        # GPU memory usage (middle center-left)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_gpu_memory(ax4)

        # Epoch timing (middle center-right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_epoch_timing(ax5)

        # Training efficiency (middle right)
        ax6 = fig.add_subplot(gs[1, 3])
        self._plot_training_efficiency(ax6)

        # Performance summary (bottom, spans all columns)
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_performance_summary(ax7)

        plt.suptitle(f'Day 12 GPU Training Dashboard - Epoch {self.snapshots[-1].epoch}',
                    fontsize=16, fontweight='bold')

        if self.save_plots:
            plot_path = self.save_dir / f"training_dashboard_epoch_{self.snapshots[-1].epoch:03d}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')

        plt.show()

    def _plot_loss_curves(self, ax):
        """Plot training and validation loss curves"""
        epochs = [s.epoch for s in self.snapshots]
        train_losses = [s.train_loss for s in self.snapshots]
        val_losses = [s.val_loss for s in self.snapshots]

        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)

        # Highlight best epoch
        if self.best_epoch < len(epochs):
            ax.axvline(x=self.best_epoch, color='green', linestyle='--', alpha=0.7,
                      label=f'Best Epoch ({self.best_epoch})')

        # Add overfitting detection
        if len(train_losses) > 5:
            recent_train = np.mean(train_losses[-5:])
            recent_val = np.mean(val_losses[-5:])
            if recent_val > recent_train * self.alert_thresholds['overfitting_ratio']:
                ax.text(0.7, 0.9, '⚠️ Possible Overfitting', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                       fontsize=10, fontweight='bold')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training & Validation Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better loss visualization

    def _plot_correlation_progress(self, ax):
        """Plot validation correlation progress with target line"""
        epochs = [s.epoch for s in self.snapshots]
        correlations = [s.val_correlation for s in self.snapshots]

        ax.plot(epochs, correlations, 'g-', linewidth=3, alpha=0.8, label='Validation Correlation')

        # Target correlation line
        ax.axhline(y=self.target_correlation, color='orange', linestyle='--',
                  linewidth=2, alpha=0.7, label=f'Target ({self.target_correlation})')

        # Fill area when above target
        target_achieved = [max(c, self.target_correlation) for c in correlations]
        ax.fill_between(epochs, correlations, target_achieved,
                       where=np.array(correlations) >= self.target_correlation,
                       color='green', alpha=0.3, label='Target Achieved')

        # Highlight best correlation
        best_idx = np.argmax(correlations)
        ax.scatter(epochs[best_idx], correlations[best_idx],
                  color='red', s=100, zorder=5, label=f'Best: {correlations[best_idx]:.4f}')

        # Progress indicators
        if correlations[-1] >= self.target_correlation:
            ax.text(0.7, 0.1, '🎯 Target Achieved!', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                   fontsize=12, fontweight='bold')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Correlation')
        ax.set_title('Validation Correlation Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_learning_rate(self, ax):
        """Plot learning rate schedule"""
        epochs = [s.epoch for s in self.snapshots]
        lrs = [s.learning_rate for s in self.snapshots]

        ax.plot(epochs, lrs, 'purple', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    def _plot_gpu_memory(self, ax):
        """Plot GPU memory usage"""
        epochs = [s.epoch for s in self.snapshots]
        memory = [s.gpu_memory_gb for s in self.snapshots]

        if any(m > 0 for m in memory):  # Only plot if we have memory data
            ax.plot(epochs, memory, 'red', linewidth=2, alpha=0.8)
            ax.fill_between(epochs, memory, alpha=0.3, color='red')

            # Memory efficiency warning
            max_memory = max(memory) if memory else 0
            if max_memory > 8.0:  # > 8GB usage warning
                ax.text(0.5, 0.8, '⚠️ High Memory', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                       fontsize=9, ha='center')
        else:
            ax.text(0.5, 0.5, 'Memory monitoring\nnot available', transform=ax.transAxes,
                   ha='center', va='center', fontsize=10, alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('GPU Memory Usage')
        ax.grid(True, alpha=0.3)

    def _plot_epoch_timing(self, ax):
        """Plot epoch timing and efficiency"""
        epochs = [s.epoch for s in self.snapshots]
        times = [s.epoch_time for s in self.snapshots]

        ax.plot(epochs, times, 'brown', linewidth=2, alpha=0.8)

        # Add average line
        if len(times) > 3:
            avg_time = np.mean(times)
            ax.axhline(y=avg_time, color='orange', linestyle='--', alpha=0.7,
                      label=f'Avg: {avg_time:.1f}s')
            ax.legend()

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Epoch Training Time')
        ax.grid(True, alpha=0.3)

    def _plot_training_efficiency(self, ax):
        """Plot training efficiency metrics"""
        if len(self.snapshots) < 3:
            ax.text(0.5, 0.5, 'Efficiency metrics\navailable after\n3+ epochs',
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Training Efficiency')
            return

        epochs = [s.epoch for s in self.snapshots]

        # Calculate efficiency: correlation improvement per time
        efficiency_scores = []
        for i in range(1, len(self.snapshots)):
            corr_improvement = self.snapshots[i].val_correlation - self.snapshots[i-1].val_correlation
            time_taken = self.snapshots[i].epoch_time
            efficiency = corr_improvement / (time_taken + 1e-6)  # Avoid division by zero
            efficiency_scores.append(efficiency)

        ax.plot(epochs[1:], efficiency_scores, 'teal', linewidth=2, marker='s', markersize=4)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Efficiency Score')
        ax.set_title('Training Efficiency\n(Correlation Gain/Time)')
        ax.grid(True, alpha=0.3)

    def _plot_performance_summary(self, ax):
        """Plot comprehensive performance summary"""
        if len(self.snapshots) < 2:
            return

        # Create summary statistics
        current_snapshot = self.snapshots[-1]

        summary_data = {
            'Current Epoch': current_snapshot.epoch,
            'Best Correlation': f"{self.best_correlation:.4f}",
            'Current Correlation': f"{current_snapshot.val_correlation:.4f}",
            'Target Progress': f"{(current_snapshot.val_correlation/self.target_correlation)*100:.1f}%",
            'Training Loss': f"{current_snapshot.train_loss:.4f}",
            'Validation Loss': f"{current_snapshot.val_loss:.4f}",
            'Learning Rate': f"{current_snapshot.learning_rate:.2e}",
            'Avg Epoch Time': f"{np.mean([s.epoch_time for s in self.snapshots]):.1f}s"
        }

        # Create table visualization
        table_data = [[k, v] for k, v in summary_data.items()]

        # Split into two columns for better layout
        mid_point = len(table_data) // 2
        left_data = table_data[:mid_point]
        right_data = table_data[mid_point:]

        ax.axis('off')

        # Left table
        table1 = ax.table(cellText=left_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center left',
                         bbox=[0.0, 0.2, 0.45, 0.6])
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1, 2)

        # Right table
        table2 = ax.table(cellText=right_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center right',
                         bbox=[0.55, 0.2, 0.45, 0.6])
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 2)

        # Add status indicators
        status_text = "🟢 Training On Track" if current_snapshot.val_correlation > 0.8 else "🟡 Needs Improvement"
        if current_snapshot.val_correlation >= self.target_correlation:
            status_text = "🎯 Target Achieved!"

        ax.text(0.5, 0.9, status_text, transform=ax.transAxes, ha='center',
               fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        ax.set_title('Training Performance Summary', fontsize=14, fontweight='bold', pad=20)

    def _check_training_alerts(self, snapshot: TrainingSnapshot):
        """Check for training alerts and warnings"""
        alerts = []

        # Check for target achievement
        if snapshot.val_correlation >= self.alert_thresholds['correlation_target']:
            alerts.append(f"🎯 Target correlation achieved: {snapshot.val_correlation:.4f}")

        # Check for overfitting
        if len(self.snapshots) > 5:
            if snapshot.val_loss > snapshot.train_loss * self.alert_thresholds['overfitting_ratio']:
                alerts.append("⚠️ Possible overfitting detected")

        # Check for no improvement
        if len(self.snapshots) >= self.alert_thresholds['no_improvement_epochs']:
            recent_correlations = [s.val_correlation for s in self.snapshots[-self.alert_thresholds['no_improvement_epochs']:]]
            if max(recent_correlations) <= min(recent_correlations) + 0.001:  # No significant improvement
                alerts.append(f"📈 No improvement for {self.alert_thresholds['no_improvement_epochs']} epochs")

        # Log alerts
        if alerts:
            print("\n" + "="*50)
            for alert in alerts:
                print(f"ALERT: {alert}")
            print("="*50)

    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        if not self.snapshots:
            return {}

        # Calculate summary statistics
        epochs = [s.epoch for s in self.snapshots]
        train_losses = [s.train_loss for s in self.snapshots]
        val_losses = [s.val_loss for s in self.snapshots]
        correlations = [s.val_correlation for s in self.snapshots]
        epoch_times = [s.epoch_time for s in self.snapshots]

        report = {
            'training_summary': {
                'total_epochs': len(self.snapshots),
                'best_correlation': self.best_correlation,
                'best_epoch': self.best_epoch,
                'final_correlation': correlations[-1],
                'target_achieved': correlations[-1] >= self.target_correlation,
                'correlation_improvement': correlations[-1] - correlations[0] if len(correlations) > 1 else 0
            },
            'loss_analysis': {
                'initial_train_loss': train_losses[0],
                'final_train_loss': train_losses[-1],
                'initial_val_loss': val_losses[0],
                'final_val_loss': val_losses[-1],
                'loss_reduction_train': (train_losses[0] - train_losses[-1]) / train_losses[0],
                'loss_reduction_val': (val_losses[0] - val_losses[-1]) / val_losses[0],
                'overfitting_ratio': val_losses[-1] / train_losses[-1]
            },
            'timing_analysis': {
                'total_training_time': sum(epoch_times),
                'average_epoch_time': np.mean(epoch_times),
                'fastest_epoch': min(epoch_times),
                'slowest_epoch': max(epoch_times)
            },
            'correlation_milestones': self._find_correlation_milestones(correlations),
            'recommendations': self._generate_training_recommendations()
        }

        # Save report
        report_path = self.save_dir / "training_monitoring_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def _find_correlation_milestones(self, correlations: List[float]) -> Dict[str, int]:
        """Find epochs where correlation milestones were achieved"""
        milestones = {}
        milestone_values = [0.7, 0.8, 0.85, 0.9, 0.95]

        for milestone in milestone_values:
            for i, corr in enumerate(correlations):
                if corr >= milestone:
                    milestones[f"correlation_{milestone:.2f}"] = i
                    break

        return milestones

    def _generate_training_recommendations(self) -> List[str]:
        """Generate training improvement recommendations"""
        recommendations = []

        if not self.snapshots:
            return recommendations

        current = self.snapshots[-1]

        # Correlation-based recommendations
        if current.val_correlation < 0.8:
            recommendations.append("Consider increasing model capacity or training longer")
        elif current.val_correlation < 0.9:
            recommendations.append("Fine-tune hyperparameters for final performance boost")
        else:
            recommendations.append("Excellent performance - ready for export")

        # Loss-based recommendations
        if len(self.snapshots) > 5:
            recent_val_losses = [s.val_loss for s in self.snapshots[-5:]]
            if np.std(recent_val_losses) > np.mean(recent_val_losses) * 0.1:
                recommendations.append("Consider reducing learning rate for more stable convergence")

        # Timing recommendations
        avg_time = np.mean([s.epoch_time for s in self.snapshots])
        if avg_time > 60:  # More than 1 minute per epoch
            recommendations.append("Consider increasing batch size for faster training")

        return recommendations

    def plot_final_summary(self):
        """Generate final comprehensive training summary visualization"""
        if len(self.snapshots) < 2:
            print("Insufficient data for final summary")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        epochs = [s.epoch for s in self.snapshots]

        # Loss evolution
        train_losses = [s.train_loss for s in self.snapshots]
        val_losses = [s.val_loss for s in self.snapshots]

        ax1.plot(epochs, train_losses, 'b-', label='Training', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Evolution - Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Correlation evolution
        correlations = [s.val_correlation for s in self.snapshots]
        ax2.plot(epochs, correlations, 'g-', linewidth=3)
        ax2.axhline(y=self.target_correlation, color='orange', linestyle='--',
                   label=f'Target ({self.target_correlation})')
        ax2.fill_between(epochs, correlations, alpha=0.3, color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Correlation')
        ax2.set_title('Training Evolution - Correlation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # Learning dynamics
        if len(epochs) > 1:
            correlation_deltas = np.diff(correlations)
            ax3.plot(epochs[1:], correlation_deltas, 'purple', linewidth=2)
            ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Correlation Change')
            ax3.set_title('Learning Dynamics')
            ax3.grid(True, alpha=0.3)

        # Performance distribution
        performance_metrics = [
            ('Best Correlation', self.best_correlation),
            ('Final Correlation', correlations[-1]),
            ('Target', self.target_correlation),
            ('Initial Correlation', correlations[0])
        ]

        metrics, values = zip(*performance_metrics)
        colors = ['gold', 'green', 'orange', 'lightblue']
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_ylabel('Correlation Value')
        ax4.set_title('Performance Summary')
        ax4.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if self.save_plots:
            final_path = self.save_dir / "final_training_summary.png"
            plt.savefig(final_path, dpi=300, bbox_inches='tight')
            print(f"Final summary saved: {final_path}")

        plt.show()


class PerformanceProfiler:
    """Profile training performance and resource usage"""

    def __init__(self):
        self.profiles = []
        self.start_time = None

    def start_profiling(self):
        """Start performance profiling"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def profile_step(self, step_name: str):
        """Profile a training step"""
        if self.start_time is None:
            return

        profile = {
            'step': step_name,
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.start_time,
        }

        if torch.cuda.is_available():
            profile.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,
                'gpu_memory_cached': torch.cuda.memory_reserved() / 1e9,
                'gpu_memory_peak': torch.cuda.max_memory_allocated() / 1e9
            })

        self.profiles.append(profile)

    def get_profile_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        if not self.profiles:
            return {}

        return {
            'total_steps': len(self.profiles),
            'total_time': self.profiles[-1]['elapsed_time'],
            'average_step_time': np.mean([p['elapsed_time'] for p in self.profiles[1:]]) if len(self.profiles) > 1 else 0,
            'peak_gpu_memory': max([p.get('gpu_memory_peak', 0) for p in self.profiles]),
            'profiles': self.profiles
        }


if __name__ == "__main__":
    # Demo the training monitor
    print("🧪 Testing Real-time Training Monitor")

    monitor = RealTimeTrainingMonitor()

    # Simulate training progress
    for epoch in range(1, 11):
        # Simulate realistic training metrics
        train_loss = 0.1 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.001)
        val_loss = train_loss * 1.2 + np.random.normal(0, 0.002)
        val_correlation = min(0.98, 0.5 + epoch * 0.04 + np.random.normal(0, 0.01))
        lr = 0.001 * (0.95 ** epoch)
        epoch_time = 30 + np.random.normal(0, 5)
        gpu_memory = 4.5 + np.random.normal(0, 0.5)

        monitor.log_epoch(epoch, train_loss, val_loss, val_correlation, lr, epoch_time, gpu_memory)
        time.sleep(0.1)  # Brief pause for demo

    # Generate final report
    report = monitor.generate_training_report()
    print(f"\n✅ Training monitoring demo complete!")
    print(f"Report saved in: {monitor.save_dir}")