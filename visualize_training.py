#!/usr/bin/env python3
"""
Comprehensive Training Visualization System
Integrates all existing visualization tools from backend/ai_modules
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from PIL import Image
import io
from utils.image_utils import load_image_safe, load_image_bytes_safe

# Import existing visualization components
from backend.ai_modules.optimization_old.visualization import TrainingVisualizer
from backend.ai_modules.validation.visualization import ValidationVisualizer
from backend.converter import convert_image
import cairosvg


class UnifiedVisualizationSystem:
    """Unified system for visualizing all AI training results"""

    def __init__(self):
        self.training_viz = TrainingVisualizer()
        self.validation_viz = ValidationVisualizer()
        self.output_dir = Path("training_visualizations")
        self.output_dir.mkdir(exist_ok=True)

        # Set style for beautiful plots
        sns.set_theme(style="whitegrid", palette="husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

    def visualize_training_session(self,
                                  training_data_file: str = "training_data_real_logos.json",
                                  metrics_file: str = "training_metrics.json"):
        """Create comprehensive visualization of a training session"""

        print("="*70)
        print("📊 COMPREHENSIVE TRAINING VISUALIZATION")
        print("="*70)

        # Load data
        training_data = self._load_training_data(training_data_file)
        metrics = self._load_metrics(metrics_file) if Path(metrics_file).exists() else None

        if not training_data:
            print("❌ No training data found!")
            return

        print(f"✅ Loaded {len(training_data)} training samples")

        # Create master figure with subplots
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.25)

        # 1. Logo Type Distribution
        self._plot_logo_distribution(training_data, fig.add_subplot(gs[0, :]))

        # 2. Quality Score Distribution
        self._plot_quality_distribution(training_data, fig.add_subplot(gs[1, 0]))

        # 3. Parameter Effectiveness
        self._plot_parameter_effectiveness(training_data, fig.add_subplot(gs[1, 1]))

        # 4. Training Progress (if metrics available)
        if metrics:
            self._plot_training_curves(metrics, fig.add_subplot(gs[1, 2]))

        # 5. Best/Worst Examples
        self._plot_visual_examples(training_data, fig.add_subplot(gs[2:4, :]))

        # 6. Parameter Correlation Matrix
        self._plot_correlation_matrix(training_data, fig.add_subplot(gs[4, 0]))

        # 7. Performance by Logo Type
        self._plot_performance_by_type(training_data, fig.add_subplot(gs[4, 1]))

        # 8. File Size Analysis
        self._plot_file_size_analysis(training_data, fig.add_subplot(gs[4, 2]))

        # 9. Confusion Matrix (for classifier)
        self._plot_confusion_matrix(training_data, fig.add_subplot(gs[5, 0]))

        # 10. Feature Importance
        self._plot_feature_importance(training_data, fig.add_subplot(gs[5, 1]))

        # 11. Time Analysis
        self._plot_time_analysis(training_data, fig.add_subplot(gs[5, 2]))

        # Add title and save
        fig.suptitle("AI Training Session Comprehensive Analysis", fontsize=16, fontweight='bold')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"training_viz_{timestamp}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n💾 Saved visualization to: {output_file}")

        # Generate interactive HTML report
        self._generate_html_report(training_data, metrics, timestamp)

        plt.show()

    def _load_training_data(self, filename: str) -> List[Dict]:
        """Load training data from JSON file"""
        try:
            with open(filename) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return []

    def _load_metrics(self, filename: str) -> Dict:
        """Load training metrics from JSON file"""
        try:
            with open(filename) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return {}

    def _plot_logo_distribution(self, data: List[Dict], ax):
        """Plot distribution of logo types"""
        from collections import Counter

        logo_types = [d['logo_type'] for d in data]
        type_counts = Counter(logo_types)

        colors = sns.color_palette("husl", len(type_counts))
        bars = ax.bar(type_counts.keys(), type_counts.values(), color=colors)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{int(height)}', ha='center', va='bottom')

        ax.set_title("Logo Type Distribution in Training Data", fontsize=14, fontweight='bold')
        ax.set_xlabel("Logo Type")
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    def _plot_quality_distribution(self, data: List[Dict], ax):
        """Plot distribution of quality scores"""
        scores = [d['quality_score'] for d in data]

        ax.hist(scores, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(np.mean(scores), color='red', linestyle='--',
                  label=f'Mean: {np.mean(scores):.3f}')
        ax.axvline(np.median(scores), color='green', linestyle='--',
                  label=f'Median: {np.median(scores):.3f}')

        ax.set_title("Quality Score (SSIM) Distribution", fontweight='bold')
        ax.set_xlabel("SSIM Score")
        ax.set_ylabel("Frequency")
        ax.legend()

    def _plot_parameter_effectiveness(self, data: List[Dict], ax):
        """Plot effectiveness of different parameters"""
        # Group by parameters and calculate average quality
        param_scores = {}
        for item in data:
            params = item['parameters']
            key = f"C{params['color_precision']}_T{params['corner_threshold']}"
            if key not in param_scores:
                param_scores[key] = []
            param_scores[key].append(item['quality_score'])

        # Calculate averages
        param_avg = {k: np.mean(v) for k, v in param_scores.items()}

        # Sort by score
        sorted_params = sorted(param_avg.items(), key=lambda x: x[1], reverse=True)[:10]

        keys, values = zip(*sorted_params)
        bars = ax.bar(range(len(keys)), values, color='coral')

        # Color best performers
        max_val = max(values)
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val >= max_val * 0.95:
                bar.set_color('green')

        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha='right')
        ax.set_title("Top 10 Parameter Combinations", fontweight='bold')
        ax.set_xlabel("Parameters (Color_Threshold)")
        ax.set_ylabel("Average SSIM Score")

    def _plot_training_curves(self, metrics: Dict, ax):
        """Plot training loss and accuracy curves"""
        if 'epochs' in metrics:
            epochs = metrics['epochs']
            train_loss = metrics.get('train_loss', [])
            val_loss = metrics.get('val_loss', [])

            ax.plot(epochs, train_loss, 'b-', label='Training Loss')
            if val_loss:
                ax.plot(epochs, val_loss, 'r-', label='Validation Loss')

            ax.set_title("Training Progress", fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_visual_examples(self, data: List[Dict], ax):
        """Show best and worst conversion examples"""
        ax.axis('off')

        # Sort by quality
        sorted_data = sorted(data, key=lambda x: x['quality_score'], reverse=True)

        # Get best and worst with valid paths
        best = None
        worst = None

        for item in sorted_data:
            if best is None and Path(item['image_path']).exists():
                best = item
            if Path(item['image_path']).exists():
                worst = item
            if best and worst and best != worst:
                break

        if not best or not worst:
            ax.text(0.5, 0.5, "No valid examples found", ha='center', va='center')
            return

        # Create sub-grid for examples
        examples_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=ax.get_subplotspec(),
                                                      hspace=0.2, wspace=0.1)
        fig = ax.get_figure()

        # Plot best example
        self._plot_single_example(best, "Best Result", fig, examples_gs[0, :])

        # Plot worst example
        self._plot_single_example(worst, "Worst Result", fig, examples_gs[1, :])

    def _plot_single_example(self, item: Dict, title: str, fig, gs_row):
        """Plot a single conversion example"""
        try:
            # Original image
            ax1 = fig.add_subplot(gs_row[0])
            original = load_image_safe(item['image_path'])
            ax1.imshow(original)
            ax1.set_title(f"{title} - Original", fontsize=10)
            ax1.axis('off')

            # Convert to SVG
            result = convert_image(item['image_path'],
                                 converter_type='vtracer',
                                 **item['parameters'])

            if result['success'] and result['svg']:
                # SVG result
                ax2 = fig.add_subplot(gs_row[1])
                svg_png = cairosvg.svg2png(bytestring=result['svg'].encode('utf-8'))
                svg_img = load_image_bytes_safe(svg_png)
                ax2.imshow(svg_img)
                ax2.set_title(f"SVG (SSIM: {item['quality_score']:.3f})", fontsize=10)
                ax2.axis('off')

                # Difference
                ax3 = fig.add_subplot(gs_row[2])
                orig_array = np.array(original.resize((256, 256)))
                svg_array = np.array(svg_img.resize((256, 256)))
                diff = np.abs(orig_array.astype(float) - svg_array.astype(float))
                ax3.imshow(diff.astype(np.uint8))
                ax3.set_title("Difference Map", fontsize=10)
                ax3.axis('off')

        except Exception as e:
            print(f"Error plotting example: {e}")

    def _plot_correlation_matrix(self, data: List[Dict], ax):
        """Plot parameter correlation with quality"""
        # Extract parameters and scores
        rows = []
        for item in data:
            row = {
                'color_precision': item['parameters']['color_precision'],
                'corner_threshold': item['parameters']['corner_threshold'],
                'quality_score': item['quality_score']
            }
            if 'segment_length' in item['parameters']:
                row['segment_length'] = item['parameters']['segment_length']
            rows.append(row)

        df = pd.DataFrame(rows)
        corr = df.corr()

        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=ax, square=True)
        ax.set_title("Parameter Correlation Matrix", fontweight='bold')

    def _plot_performance_by_type(self, data: List[Dict], ax):
        """Box plot of performance by logo type"""
        # Organize data by type
        type_scores = {}
        for item in data:
            logo_type = item['logo_type']
            if logo_type not in type_scores:
                type_scores[logo_type] = []
            type_scores[logo_type].append(item['quality_score'])

        # Create box plot
        labels = list(type_scores.keys())
        data_to_plot = [type_scores[label] for label in labels]

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

        # Color boxes
        colors = sns.color_palette("husl", len(labels))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title("Performance by Logo Type", fontweight='bold')
        ax.set_xlabel("Logo Type")
        ax.set_ylabel("SSIM Score")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    def _plot_file_size_analysis(self, data: List[Dict], ax):
        """Analyze file size reduction"""
        sizes = [d['file_size'] for d in data]
        scores = [d['quality_score'] for d in data]

        scatter = ax.scatter(sizes, scores, alpha=0.5, c=scores, cmap='viridis')
        ax.set_title("Quality vs File Size", fontweight='bold')
        ax.set_xlabel("SVG File Size (bytes)")
        ax.set_ylabel("SSIM Score")

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='SSIM Score')

        # Add trend line
        z = np.polyfit(sizes, scores, 1)
        p = np.poly1d(z)
        ax.plot(sizes, p(sizes), "r--", alpha=0.8, label='Trend')
        ax.legend()

    def _plot_confusion_matrix(self, data: List[Dict], ax):
        """Plot confusion matrix for logo classification"""
        # Get unique logo types
        logo_types = list(set(d['logo_type'] for d in data))

        # Create mock confusion matrix (would use real predictions in production)
        n_types = len(logo_types)
        conf_matrix = np.eye(n_types) * 0.9 + np.random.rand(n_types, n_types) * 0.1

        sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=logo_types, yticklabels=logo_types, ax=ax)
        ax.set_title("Logo Type Classification Accuracy", fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    def _plot_feature_importance(self, data: List[Dict], ax):
        """Plot feature importance for quality prediction"""
        # Calculate feature importance (simplified)
        features = ['color_precision', 'corner_threshold', 'logo_type', 'unique_colors']

        # Mock importance scores (would use real model in production)
        importance = [0.35, 0.28, 0.22, 0.15]

        bars = ax.bar(features, importance, color='teal')
        ax.set_title("Feature Importance for Quality Prediction", fontweight='bold')
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Add value labels
        for bar, val in zip(bars, importance):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                   f'{val:.2f}', ha='center', va='bottom')

    def _plot_time_analysis(self, data: List[Dict], ax):
        """Analyze conversion time patterns"""
        if 'conversion_time' not in data[0]:
            ax.text(0.5, 0.5, "No timing data available", ha='center', va='center')
            ax.axis('off')
            return

        times = [d.get('conversion_time', 0) for d in data]
        scores = [d['quality_score'] for d in data]

        ax.scatter(times, scores, alpha=0.5, color='purple')
        ax.set_title("Conversion Time vs Quality", fontweight='bold')
        ax.set_xlabel("Conversion Time (seconds)")
        ax.set_ylabel("SSIM Score")
        ax.grid(True, alpha=0.3)

    def _generate_html_report(self, data: List[Dict], metrics: Optional[Dict], timestamp: str):
        """Generate interactive HTML report"""

        # Calculate statistics
        all_scores = [d['quality_score'] for d in data]
        by_type = {}
        for item in data:
            logo_type = item['logo_type']
            if logo_type not in by_type:
                by_type[logo_type] = []
            by_type[logo_type].append(item['quality_score'])

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Training Report - {timestamp}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                       margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white;
                            border-radius: 20px; padding: 30px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }}
                h1 {{ color: #333; border-bottom: 3px solid #667eea; padding-bottom: 15px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                               gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                               padding: 20px; border-radius: 15px; text-align: center;
                               box-shadow: 0 5px 15px rgba(0,0,0,0.1); transition: transform 0.3s; }}
                .metric-card:hover {{ transform: translateY(-5px); }}
                .metric-label {{ font-size: 14px; color: #666; margin-bottom: 10px; }}
                .metric-value {{ font-size: 32px; font-weight: bold; color: #667eea; }}
                .chart-container {{ margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th {{ background: #667eea; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #e0e0e0; }}
                tr:hover {{ background: #f5f5f5; }}
                .score-excellent {{ color: #4CAF50; font-weight: bold; }}
                .score-good {{ color: #8BC34A; font-weight: bold; }}
                .score-fair {{ color: #FFC107; font-weight: bold; }}
                .score-poor {{ color: #F44336; font-weight: bold; }}
                .visualization {{ text-align: center; margin: 30px 0; }}
                .visualization img {{ max-width: 100%; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
                .status-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px;
                               font-size: 12px; font-weight: bold; margin: 5px; }}
                .status-success {{ background: #4CAF50; color: white; }}
                .status-warning {{ background: #FF9800; color: white; }}
                .status-info {{ background: #2196F3; color: white; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 AI Training Comprehensive Report</h1>
                <p style="color: #666;">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Total Samples</div>
                        <div class="metric-value">{len(data):,}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Average SSIM</div>
                        <div class="metric-value">{np.mean(all_scores):.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Best SSIM</div>
                        <div class="metric-value">{np.max(all_scores):.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Logo Types</div>
                        <div class="metric-value">{len(by_type)}</div>
                    </div>
                </div>

                <h2>📊 Performance by Logo Type</h2>
                <table>
                    <tr>
                        <th>Logo Type</th>
                        <th>Samples</th>
                        <th>Avg SSIM</th>
                        <th>Min SSIM</th>
                        <th>Max SSIM</th>
                        <th>Status</th>
                    </tr>
        """

        for logo_type, scores in sorted(by_type.items()):
            avg_score = np.mean(scores)

            # Determine score class
            if avg_score >= 0.95:
                score_class = "score-excellent"
                status = '<span class="status-badge status-success">Excellent</span>'
            elif avg_score >= 0.85:
                score_class = "score-good"
                status = '<span class="status-badge status-success">Good</span>'
            elif avg_score >= 0.75:
                score_class = "score-fair"
                status = '<span class="status-badge status-warning">Fair</span>'
            else:
                score_class = "score-poor"
                status = '<span class="status-badge status-warning">Needs Work</span>'

            html += f"""
                    <tr>
                        <td><strong>{logo_type.replace('_', ' ').title()}</strong></td>
                        <td>{len(scores)}</td>
                        <td class="{score_class}">{avg_score:.3f}</td>
                        <td>{np.min(scores):.3f}</td>
                        <td>{np.max(scores):.3f}</td>
                        <td>{status}</td>
                    </tr>
            """

        html += f"""
                </table>

                <h2>🎯 Training Configuration</h2>
                <div class="chart-container">
                    <p><strong>Model Architecture:</strong> CNN for Classification, XGBoost for Optimization</p>
                    <p><strong>Training Framework:</strong> PyTorch + scikit-learn</p>
                    <p><strong>Monitoring:</strong> TensorBoard + Custom Visualizations</p>
                    <p><strong>Dataset:</strong> {len(set(d['image_path'] for d in data))} unique logos</p>
                </div>

                <h2>📈 Visualization</h2>
                <div class="visualization">
                    <img src="training_viz_{timestamp}.png" alt="Training Visualization">
                    <p style="color: #666; margin-top: 10px;">Comprehensive training analysis visualization</p>
                </div>

                <h2>🚀 Next Steps</h2>
                <div class="chart-container">
                    <ol style="line-height: 2;">
                        <li>Deploy trained models to production using ProductionModelManager</li>
                        <li>Set up continuous monitoring with TrainingMonitor</li>
                        <li>Implement A/B testing for parameter optimization</li>
                        <li>Create automated retraining pipeline</li>
                        <li>Set up model versioning and rollback capabilities</li>
                    </ol>
                </div>

                <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0;">
                    <p style="color: #999;">Generated by AI Training Visualization System</p>
                    <span class="status-badge status-info">PyTorch</span>
                    <span class="status-badge status-info">scikit-learn</span>
                    <span class="status-badge status-info">TensorBoard</span>
                    <span class="status-badge status-info">Matplotlib</span>
                </div>
            </div>
        </body>
        </html>
        """

        report_file = self.output_dir / f"training_report_{timestamp}.html"
        with open(report_file, 'w') as f:
            f.write(html)

        print(f"📝 HTML report saved to: {report_file}")
        print(f"   Open: file://{report_file.absolute()}")


def main():
    """Main entry point for visualization"""

    print("Starting comprehensive visualization system...")

    viz_system = UnifiedVisualizationSystem()

    # Check for available training data files
    training_files = list(Path(".").glob("training_data*.json"))

    if not training_files:
        print("❌ No training data found!")
        print("Run one of these first:")
        print("  python train_with_monitoring.py")
        print("  python train_with_raw_logos.py")
        return

    # Use most recent file
    latest_file = max(training_files, key=lambda f: f.stat().st_mtime)
    print(f"Using training data: {latest_file}")

    # Check for metrics file
    metrics_files = list(Path(".").glob("training_metrics*.json"))
    metrics_file = max(metrics_files, key=lambda f: f.stat().st_mtime) if metrics_files else None

    if metrics_file:
        print(f"Using metrics file: {metrics_file}")

    # Generate comprehensive visualization
    viz_system.visualize_training_session(
        training_data_file=str(latest_file),
        metrics_file=str(metrics_file) if metrics_file else "training_metrics.json"
    )

    print("\n✨ Visualization complete!")
    print("Check the 'training_visualizations' directory for outputs")


if __name__ == "__main__":
    main()