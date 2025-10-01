#!/usr/bin/env python3
"""
Training Monitoring and Analysis

Monitors training progress, resource usage, and identifies instabilities
as specified in Day 5 Task 5.4.2.
"""

import torch
import os
import sys
import json
import psutil
import time
import subprocess
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

class TrainingMonitor:
    """Comprehensive training monitoring and analysis."""

    def __init__(self, checkpoint_dir: str = 'backend/ai_modules/models/trained'):
        """
        Initialize training monitor.

        Args:
            checkpoint_dir: Directory containing training checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.monitoring_data = {
            'system_resources': {},
            'training_analysis': {},
            'instabilities': [],
            'recommendations': []
        }

    def monitor_system_resources(self) -> Dict[str, Any]:
        """
        Monitor current system resource usage.

        Returns:
            System resource metrics
        """
        print("=== Monitoring System Resources ===")

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / 1024 / 1024
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_gb = disk.used / 1024 / 1024 / 1024
            disk_percent = (disk.used / disk.total) * 100

            # Temperature (if available)
            temperature = self._get_cpu_temperature()

            resources = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'usage_percent': cpu_percent,
                    'core_count': cpu_count,
                    'temperature_c': temperature
                },
                'memory': {
                    'used_mb': memory_mb,
                    'usage_percent': memory_percent,
                    'available_mb': memory.available / 1024 / 1024
                },
                'disk': {
                    'used_gb': disk_gb,
                    'usage_percent': disk_percent,
                    'free_gb': disk.free / 1024 / 1024 / 1024
                }
            }

            print(f"✓ CPU Usage: {cpu_percent:.1f}% ({cpu_count} cores)")
            print(f"✓ Memory Usage: {memory_mb:.0f} MB ({memory_percent:.1f}%)")
            print(f"✓ Disk Usage: {disk_gb:.1f} GB ({disk_percent:.1f}%)")
            if temperature:
                print(f"✓ CPU Temperature: {temperature:.1f}°C")

            self.monitoring_data['system_resources'] = resources
            return resources

        except Exception as e:
            print(f"✗ Failed to monitor system resources: {e}")
            return {}

    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature (macOS specific)."""
        try:
            # Try macOS temperature monitoring
            result = subprocess.run(['sudo', 'powermetrics', '--samplers', 'smc', '-n', '1', '-i', '1'],
                                  capture_output=True, text=True, timeout=5)

            # Parse temperature from output (simplified)
            for line in result.stdout.split('\n'):
                if 'CPU die temperature' in line:
                    temp_str = line.split(':')[1].strip().replace('C', '')
                    return float(temp_str)

            return None

        except Exception:
            return None

    def analyze_training_checkpoints(self) -> Dict[str, Any]:
        """
        Analyze training checkpoints for progress and stability.

        Returns:
            Training analysis results
        """
        print("\n=== Analyzing Training Checkpoints ===")

        try:
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth'))
            checkpoint_files.sort()

            if not checkpoint_files:
                print("✗ No checkpoint files found")
                return {}

            print(f"✓ Found {len(checkpoint_files)} checkpoint files")

            # Analyze checkpoints
            training_metrics = {
                'epochs': [],
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'per_class_acc': [],
                'stages': []
            }

            stage_transitions = []

            for i, checkpoint_path in enumerate(checkpoint_files):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    metrics = checkpoint.get('metrics', {})

                    epoch = checkpoint.get('epoch', i)
                    train_loss = metrics.get('train_loss', 0)
                    train_acc = metrics.get('train_accuracy', 0)
                    val_loss = metrics.get('val_loss', 0)
                    val_acc = metrics.get('val_accuracy', 0)
                    per_class = metrics.get('per_class_accuracy', {})

                    training_metrics['epochs'].append(epoch)
                    training_metrics['train_loss'].append(train_loss)
                    training_metrics['train_acc'].append(train_acc)
                    training_metrics['val_loss'].append(val_loss)
                    training_metrics['val_acc'].append(val_acc)
                    training_metrics['per_class_acc'].append(per_class)

                    # Detect stage transitions by checkpoint size changes
                    file_size = os.path.getsize(checkpoint_path)
                    if i > 0:
                        prev_size = os.path.getsize(checkpoint_files[i-1])
                        if abs(file_size - prev_size) > 1000000:  # 1MB threshold
                            stage_transitions.append(epoch)

                except Exception as e:
                    print(f"✗ Failed to load checkpoint {checkpoint_path}: {e}")
                    continue

            # Calculate training statistics
            if training_metrics['val_acc']:
                best_val_acc = max(training_metrics['val_acc'])
                best_epoch = training_metrics['epochs'][training_metrics['val_acc'].index(best_val_acc)]
                final_val_acc = training_metrics['val_acc'][-1]

                # Detect overfitting
                val_acc_trend = np.diff(training_metrics['val_acc'][-5:]) if len(training_metrics['val_acc']) >= 5 else []
                overfitting_detected = len(val_acc_trend) > 0 and all(x <= 0 for x in val_acc_trend[-3:])

                analysis = {
                    'total_epochs': len(checkpoint_files),
                    'best_validation_accuracy': best_val_acc,
                    'best_epoch': best_epoch,
                    'final_validation_accuracy': final_val_acc,
                    'overfitting_detected': overfitting_detected,
                    'stage_transitions': stage_transitions,
                    'training_metrics': training_metrics
                }

                print(f"✓ Best validation accuracy: {best_val_acc:.1f}% (epoch {best_epoch})")
                print(f"✓ Final validation accuracy: {final_val_acc:.1f}%")
                print(f"✓ Stage transitions detected at epochs: {stage_transitions}")
                print(f"✓ Overfitting detected: {overfitting_detected}")

                self.monitoring_data['training_analysis'] = analysis
                return analysis

        except Exception as e:
            print(f"✗ Failed to analyze checkpoints: {e}")
            return {}

    def detect_training_instabilities(self) -> List[Dict[str, Any]]:
        """
        Detect training instabilities and issues.

        Returns:
            List of detected instabilities
        """
        print("\n=== Detecting Training Instabilities ===")

        instabilities = []

        try:
            training_analysis = self.monitoring_data.get('training_analysis', {})
            if not training_analysis:
                print("✗ No training analysis available")
                return instabilities

            metrics = training_analysis.get('training_metrics', {})
            val_acc = metrics.get('val_acc', [])
            train_acc = metrics.get('train_acc', [])
            val_loss = metrics.get('val_loss', [])

            # 1. Check for validation accuracy plateau
            if len(val_acc) >= 5:
                recent_val_acc = val_acc[-5:]
                if max(recent_val_acc) - min(recent_val_acc) < 5:  # Less than 5% variation
                    instabilities.append({
                        'type': 'validation_plateau',
                        'severity': 'medium',
                        'description': 'Validation accuracy has plateaued in recent epochs',
                        'recommendation': 'Consider reducing learning rate or early stopping'
                    })

            # 2. Check for overfitting
            if len(val_acc) >= 3 and len(train_acc) >= 3:
                recent_train = np.mean(train_acc[-3:])
                recent_val = np.mean(val_acc[-3:])
                overfitting_gap = recent_train - recent_val

                if overfitting_gap > 30:  # More than 30% gap
                    instabilities.append({
                        'type': 'severe_overfitting',
                        'severity': 'high',
                        'description': f'Large gap between train and validation accuracy: {overfitting_gap:.1f}%',
                        'recommendation': 'Increase regularization, reduce model complexity, or add more data'
                    })
                elif overfitting_gap > 15:  # More than 15% gap
                    instabilities.append({
                        'type': 'moderate_overfitting',
                        'severity': 'medium',
                        'description': f'Moderate overfitting detected: {overfitting_gap:.1f}% gap',
                        'recommendation': 'Consider increasing dropout or reducing learning rate'
                    })

            # 3. Check for loss instability
            if len(val_loss) >= 5:
                val_loss_std = np.std(val_loss[-5:])
                val_loss_mean = np.mean(val_loss[-5:])

                if val_loss_std / val_loss_mean > 0.5:  # High relative variance
                    instabilities.append({
                        'type': 'loss_instability',
                        'severity': 'medium',
                        'description': 'High variance in validation loss',
                        'recommendation': 'Reduce learning rate or implement gradient clipping'
                    })

            # 4. Check for class prediction bias
            per_class_acc = metrics.get('per_class_acc', [])
            if per_class_acc:
                latest_per_class = per_class_acc[-1]
                if latest_per_class:
                    active_classes = sum(1 for acc in latest_per_class.values() if acc > 0)
                    total_classes = len(latest_per_class)

                    if active_classes < total_classes:
                        instabilities.append({
                            'type': 'class_prediction_bias',
                            'severity': 'high',
                            'description': f'Model only predicting {active_classes}/{total_classes} classes',
                            'recommendation': 'Increase class weights for underrepresented classes'
                        })

            print(f"✓ Detected {len(instabilities)} training instabilities")
            for instability in instabilities:
                print(f"  - {instability['type']} ({instability['severity']}): {instability['description']}")

            self.monitoring_data['instabilities'] = instabilities
            return instabilities

        except Exception as e:
            print(f"✗ Failed to detect instabilities: {e}")
            return instabilities

    def generate_recommendations(self) -> List[str]:
        """
        Generate training recommendations based on monitoring results.

        Returns:
            List of recommendations
        """
        print("\n=== Generating Training Recommendations ===")

        recommendations = []

        try:
            # System resource recommendations
            resources = self.monitoring_data.get('system_resources', {})
            if resources:
                cpu_usage = resources.get('cpu', {}).get('usage_percent', 0)
                memory_usage = resources.get('memory', {}).get('usage_percent', 0)

                if cpu_usage > 90:
                    recommendations.append("High CPU usage detected - consider reducing batch size or parallel workers")

                if memory_usage > 85:
                    recommendations.append("High memory usage detected - consider reducing batch size or model size")

                temp = resources.get('cpu', {}).get('temperature_c')
                if temp and temp > 80:
                    recommendations.append(f"High CPU temperature ({temp:.1f}°C) - ensure adequate cooling")

            # Training-specific recommendations
            training_analysis = self.monitoring_data.get('training_analysis', {})
            if training_analysis:
                best_val_acc = training_analysis.get('best_validation_accuracy', 0)
                final_val_acc = training_analysis.get('final_validation_accuracy', 0)

                if best_val_acc < 50:
                    recommendations.append("Low validation accuracy - consider adjusting learning rate or model architecture")

                if best_val_acc - final_val_acc > 10:
                    recommendations.append("Performance degraded from best - implement early stopping")

            # Instability-based recommendations
            instabilities = self.monitoring_data.get('instabilities', [])
            high_severity_count = sum(1 for inst in instabilities if inst['severity'] == 'high')

            if high_severity_count > 0:
                recommendations.append(f"Address {high_severity_count} high-severity training issues immediately")

            # Add specific recommendations from Day 5 context
            recommendations.extend([
                "Continue with progressive unfreezing strategy for better transfer learning",
                "Monitor per-class accuracy to ensure balanced predictions across all logo types",
                "Consider ensemble methods if single model accuracy plateaus below target",
                "Implement model quantization for deployment optimization"
            ])

            print(f"✓ Generated {len(recommendations)} recommendations")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

            self.monitoring_data['recommendations'] = recommendations
            return recommendations

        except Exception as e:
            print(f"✗ Failed to generate recommendations: {e}")
            return recommendations

    def create_monitoring_plots(self, save_dir: str = 'training_monitoring_plots'):
        """Create visualization plots for training monitoring."""
        print(f"\n=== Creating Monitoring Plots ===")

        try:
            os.makedirs(save_dir, exist_ok=True)

            training_analysis = self.monitoring_data.get('training_analysis', {})
            metrics = training_analysis.get('training_metrics', {})

            if not metrics.get('epochs'):
                print("✗ No training metrics available for plotting")
                return

            # Training curves plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            epochs = metrics['epochs']

            # Loss curves
            ax1.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss', alpha=0.8)
            ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss', alpha=0.8)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Accuracy curves
            ax2.plot(epochs, metrics['train_acc'], 'b-', label='Training Accuracy', alpha=0.8)
            ax2.plot(epochs, metrics['val_acc'], 'r-', label='Validation Accuracy', alpha=0.8)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Overfitting analysis
            train_val_gap = [t - v for t, v in zip(metrics['train_acc'], metrics['val_acc'])]
            ax3.plot(epochs, train_val_gap, 'g-', label='Train-Val Gap', alpha=0.8)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='Overfitting Threshold')
            ax3.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Severe Overfitting')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy Gap (%)')
            ax3.set_title('Overfitting Analysis (Train - Val Accuracy)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Per-class accuracy (latest epoch)
            if metrics['per_class_acc'] and metrics['per_class_acc'][-1]:
                classes = list(metrics['per_class_acc'][-1].keys())
                accuracies = list(metrics['per_class_acc'][-1].values())

                bars = ax4.bar(classes, accuracies, alpha=0.8, color=['blue', 'green', 'orange', 'red'])
                ax4.set_ylabel('Accuracy (%)')
                ax4.set_title('Per-Class Accuracy (Latest Epoch)')
                ax4.set_ylim(0, 100)

                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{acc:.1f}%', ha='center', va='bottom')

            plt.tight_layout()
            plot_path = os.path.join(save_dir, 'training_monitoring_plots.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ Training plots saved: {plot_path}")

        except Exception as e:
            print(f"✗ Failed to create plots: {e}")

    def save_monitoring_report(self, output_path: str = 'training_monitoring_report.json'):
        """Save comprehensive monitoring report."""
        print(f"\n=== Saving Monitoring Report ===")

        try:
            report = {
                'monitoring_timestamp': datetime.now().isoformat(),
                'monitoring_summary': {
                    'total_instabilities': len(self.monitoring_data.get('instabilities', [])),
                    'high_severity_issues': len([i for i in self.monitoring_data.get('instabilities', [])
                                               if i.get('severity') == 'high']),
                    'total_recommendations': len(self.monitoring_data.get('recommendations', [])),
                    'training_completed': self.monitoring_data.get('training_analysis', {}).get('total_epochs', 0) > 0
                },
                'detailed_analysis': self.monitoring_data
            }

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"✓ Monitoring report saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"✗ Failed to save monitoring report: {e}")
            return None

def run_comprehensive_monitoring():
    """Run comprehensive training monitoring as specified in Day 5."""
    print("Training Monitoring and Analysis (Day 5 Task 5.4.2)")
    print("=" * 60)

    monitor = TrainingMonitor()

    # Monitor system resources
    resources = monitor.monitor_system_resources()

    # Analyze training checkpoints
    training_analysis = monitor.analyze_training_checkpoints()

    # Detect instabilities
    instabilities = monitor.detect_training_instabilities()

    # Generate recommendations
    recommendations = monitor.generate_recommendations()

    # Create visualization plots
    monitor.create_monitoring_plots()

    # Save comprehensive report
    report_path = monitor.save_monitoring_report()

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING MONITORING SUMMARY")
    print("=" * 60)

    print(f"✓ System resources monitored")
    print(f"✓ Training checkpoints analyzed")
    print(f"✓ {len(instabilities)} instabilities detected")
    print(f"✓ {len(recommendations)} recommendations generated")

    if training_analysis:
        best_acc = training_analysis.get('best_validation_accuracy', 0)
        total_epochs = training_analysis.get('total_epochs', 0)
        print(f"\nTraining Results:")
        print(f"  Best validation accuracy: {best_acc:.1f}%")
        print(f"  Total epochs completed: {total_epochs}")
        print(f"  Progressive unfreezing stages detected: {len(training_analysis.get('stage_transitions', []))}")

    high_severity = [i for i in instabilities if i['severity'] == 'high']
    if high_severity:
        print(f"\n⚠️  High-severity issues requiring attention:")
        for issue in high_severity:
            print(f"  - {issue['description']}")
            print(f"    Recommendation: {issue['recommendation']}")

    print(f"\n✓ Training monitoring completed successfully!")
    print(f"✓ Detailed report saved: {report_path}")

    return True

if __name__ == "__main__":
    success = run_comprehensive_monitoring()
    sys.exit(0 if success else 1)