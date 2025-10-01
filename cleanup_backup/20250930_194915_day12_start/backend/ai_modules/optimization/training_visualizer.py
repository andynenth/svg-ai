# backend/ai_modules/optimization/training_visualizer.py
"""Comprehensive training visualization and reporting system for PPO agent training"""

import os
import json
import time
import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict
import math

# Core visualization libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Try to import plotly for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Interactive visualizations will be disabled.")

# Statistical analysis
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrainingDataProcessor:
    """Processes and prepares training data for visualization"""

    def __init__(self):
        self.scaler = StandardScaler()

    def process_metrics_history(self, metrics: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert metrics history to pandas DataFrame with proper indexing"""
        if not metrics:
            return pd.DataFrame()

        df = pd.DataFrame(metrics)

        # Convert timestamps if available
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
        elif 'episode' in df.columns:
            df.set_index('episode', inplace=True)

        # Fill missing values
        df = df.fillna(method='ffill').fillna(0)

        return df

    def calculate_moving_averages(self, data: pd.Series, windows: List[int] = [10, 50, 100]) -> pd.DataFrame:
        """Calculate moving averages for different window sizes"""
        result = pd.DataFrame({'original': data})

        for window in windows:
            if len(data) >= window:
                result[f'ma_{window}'] = data.rolling(window=window, min_periods=1).mean()

        return result

    def detect_anomalies(self, data: pd.Series, method: str = 'iqr', threshold: float = 2.0) -> pd.Series:
        """Detect anomalies in training data"""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            return z_scores > threshold

        return pd.Series([False] * len(data), index=data.index)

    def calculate_convergence_metrics(self, data: pd.Series, window: int = 100) -> Dict[str, float]:
        """Calculate convergence metrics for training data"""
        if len(data) < window:
            return {'convergence_rate': 0.0, 'stability_index': 0.0, 'final_trend': 0.0}

        # Calculate trend in final window
        final_window = data.tail(window)
        x = np.arange(len(final_window))
        slope, _, r_value, _, _ = stats.linregress(x, final_window)

        # Calculate stability (inverse of variance)
        stability = 1.0 / (1.0 + np.var(final_window))

        # Calculate convergence rate (how quickly metrics stabilize)
        rolling_std = data.rolling(window=window//2).std()
        convergence_rate = -np.mean(np.diff(rolling_std.dropna()[-window//2:]))

        return {
            'convergence_rate': convergence_rate,
            'stability_index': stability,
            'final_trend': slope,
            'r_squared': r_value**2
        }


class TrainingPlotter:
    """Creates various training visualization plots"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 10)

    def plot_learning_curves(self, metrics_df: pd.DataFrame,
                           metrics: List[str] = ['episode_rewards', 'quality_improvements'],
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot learning curves for specified metrics"""
        fig, axes = plt.subplots(len(metrics), 1, figsize=self.figsize, dpi=self.dpi)
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            if metric not in metrics_df.columns:
                logger.warning(f"Metric {metric} not found in data")
                continue

            ax = axes[i]
            data = metrics_df[metric].dropna()

            # Plot raw data
            ax.plot(data.index, data, alpha=0.3, color=self.colors[i], label='Raw')

            # Plot moving averages
            ma_data = TrainingDataProcessor().calculate_moving_averages(data)
            for col in ma_data.columns:
                if col.startswith('ma_'):
                    window = int(col.split('_')[1])
                    ax.plot(data.index, ma_data[col], linewidth=2,
                           label=f'MA-{window}', color=self.colors[i % len(self.colors)])

            ax.set_title(f'{metric.replace("_", " ").title()} Learning Curve')
            ax.set_xlabel('Episode' if 'episode' in str(data.index.name) else 'Time')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_loss_curves(self, metrics_df: pd.DataFrame,
                        loss_metrics: List[str] = ['training_loss', 'value_loss', 'policy_loss'],
                        save_path: Optional[str] = None) -> plt.Figure:
        """Plot training loss curves"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        for i, loss_metric in enumerate(loss_metrics):
            if loss_metric not in metrics_df.columns:
                continue

            data = metrics_df[loss_metric].dropna()
            if len(data) == 0:
                continue

            # Smooth the loss curve
            if len(data) > 10:
                smoothed = savgol_filter(data, min(len(data)//5, 51), 3)
                ax.plot(data.index, smoothed, linewidth=2,
                       label=loss_metric.replace('_', ' ').title(),
                       color=self.colors[i])
            else:
                ax.plot(data.index, data, linewidth=2,
                       label=loss_metric.replace('_', ' ').title(),
                       color=self.colors[i])

        ax.set_title('Training Loss Curves')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_reward_distribution(self, metrics_df: pd.DataFrame,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot reward distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)

        if 'episode_rewards' not in metrics_df.columns:
            logger.warning("Episode rewards not found in data")
            return fig

        rewards = metrics_df['episode_rewards'].dropna()

        # Histogram
        axes[0, 0].hist(rewards, bins=50, alpha=0.7, color=self.colors[0])
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].set_xlabel('Episode Reward')
        axes[0, 0].set_ylabel('Frequency')

        # Box plot over time (binned)
        if len(rewards) > 100:
            bins = np.array_split(rewards, 10)
            bin_data = [list(bin_rewards) for bin_rewards in bins if len(bin_rewards) > 0]
            axes[0, 1].boxplot(bin_data)
            axes[0, 1].set_title('Reward Distribution Over Time')
            axes[0, 1].set_xlabel('Training Progress (Deciles)')
            axes[0, 1].set_ylabel('Episode Reward')

        # Cumulative reward
        cumulative_rewards = rewards.cumsum()
        axes[1, 0].plot(cumulative_rewards.index, cumulative_rewards, color=self.colors[1])
        axes[1, 0].set_title('Cumulative Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Cumulative Reward')

        # Reward volatility (rolling standard deviation)
        rolling_std = rewards.rolling(window=min(len(rewards)//10, 100)).std()
        axes[1, 1].plot(rolling_std.index, rolling_std, color=self.colors[2])
        axes[1, 1].set_title('Reward Volatility')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Rolling Std Dev')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_parameter_exploration(self, parameter_history: List[Dict[str, Any]],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Plot parameter exploration heatmaps"""
        if not parameter_history:
            logger.warning("No parameter history available")
            return plt.figure()

        # Convert to DataFrame
        param_df = pd.DataFrame(parameter_history)

        # Identify numeric parameters
        numeric_cols = param_df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            logger.warning("No numeric parameters found for visualization")
            return plt.figure()

        # Create correlation heatmap
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)

        # Parameter correlation heatmap
        corr_matrix = param_df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   ax=axes[0, 0], cbar_kws={'shrink': 0.8})
        axes[0, 0].set_title('Parameter Correlation Matrix')

        # Parameter distribution over time
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            scatter = axes[0, 1].scatter(param_df[col1], param_df[col2],
                                       c=range(len(param_df)),
                                       cmap='viridis', alpha=0.6)
            axes[0, 1].set_xlabel(col1)
            axes[0, 1].set_ylabel(col2)
            axes[0, 1].set_title(f'{col1} vs {col2} Evolution')
            plt.colorbar(scatter, ax=axes[0, 1], label='Time Step')

        # Parameter variance over time
        param_variance = param_df[numeric_cols].rolling(window=min(len(param_df)//5, 50)).var()
        for i, col in enumerate(numeric_cols[:4]):  # Limit to 4 parameters
            if col in param_variance.columns:
                axes[1, 0].plot(param_variance.index, param_variance[col],
                              label=col, color=self.colors[i % len(self.colors)])
        axes[1, 0].set_title('Parameter Variance Over Time')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Variance')
        axes[1, 0].legend()

        # Parameter range exploration
        param_ranges = param_df[numeric_cols].agg(['min', 'max', 'std'])
        x_pos = np.arange(len(numeric_cols))
        axes[1, 1].bar(x_pos, param_ranges.loc['std'], alpha=0.7, color=self.colors[0])
        axes[1, 1].set_xlabel('Parameters')
        axes[1, 1].set_ylabel('Standard Deviation')
        axes[1, 1].set_title('Parameter Exploration Range')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(numeric_cols, rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_training_stability(self, metrics_df: pd.DataFrame,
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot training stability indicators"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)

        if 'episode_rewards' in metrics_df.columns:
            rewards = metrics_df['episode_rewards'].dropna()

            # Gradient (rate of change)
            gradient = np.gradient(rewards)
            axes[0, 0].plot(rewards.index[1:], gradient[1:], alpha=0.7, color=self.colors[0])
            axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('Reward Gradient (Rate of Change)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward Gradient')

            # Rolling coefficient of variation
            rolling_mean = rewards.rolling(window=50).mean()
            rolling_std = rewards.rolling(window=50).std()
            cv = rolling_std / rolling_mean
            axes[0, 1].plot(cv.index, cv, color=self.colors[1])
            axes[0, 1].set_title('Coefficient of Variation (Stability)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('CV (lower = more stable)')

        # Training loss stability
        if 'training_loss' in metrics_df.columns:
            loss = metrics_df['training_loss'].dropna()
            if len(loss) > 10:
                loss_gradient = np.gradient(loss)
                axes[1, 0].plot(loss.index[1:], loss_gradient[1:], color=self.colors[2])
                axes[1, 0].set_title('Training Loss Gradient')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Loss Gradient')

        # Success rate stability
        if 'success_rates' in metrics_df.columns:
            success_rates = metrics_df['success_rates'].dropna()
            if len(success_rates) > 10:
                rolling_success = success_rates.rolling(window=20).mean()
                axes[1, 1].plot(success_rates.index, success_rates, alpha=0.3, color=self.colors[3], label='Raw')
                axes[1, 1].plot(rolling_success.index, rolling_success, linewidth=2, color=self.colors[3], label='Rolling Mean')
                axes[1, 1].set_title('Success Rate Stability')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Success Rate')
                axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig


class InteractiveDashboard:
    """Creates interactive visualizations using Plotly"""

    def __init__(self):
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Interactive features disabled.")

    def create_interactive_training_dashboard(self, metrics_df: pd.DataFrame,
                                           save_path: Optional[str] = None) -> Optional[str]:
        """Create interactive training dashboard"""
        if not PLOTLY_AVAILABLE:
            logger.warning("Cannot create interactive dashboard without Plotly")
            return None

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Learning Curves', 'Loss Curves', 'Reward Distribution',
                          'Training Stability', 'Parameter Evolution', 'Performance Metrics'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Learning curves
        if 'episode_rewards' in metrics_df.columns:
            rewards = metrics_df['episode_rewards'].dropna()
            fig.add_trace(
                go.Scatter(x=rewards.index, y=rewards, mode='lines',
                          name='Episode Rewards', line=dict(color='blue')),
                row=1, col=1
            )

            # Add moving average
            if len(rewards) > 20:
                ma = rewards.rolling(window=20).mean()
                fig.add_trace(
                    go.Scatter(x=ma.index, y=ma, mode='lines',
                              name='MA-20', line=dict(color='red')),
                    row=1, col=1
                )

        # Loss curves
        loss_metrics = ['training_loss', 'value_loss', 'policy_loss']
        colors = ['red', 'green', 'blue']
        for i, loss_metric in enumerate(loss_metrics):
            if loss_metric in metrics_df.columns:
                loss_data = metrics_df[loss_metric].dropna()
                fig.add_trace(
                    go.Scatter(x=loss_data.index, y=loss_data, mode='lines',
                              name=loss_metric.replace('_', ' ').title(),
                              line=dict(color=colors[i])),
                    row=1, col=2
                )

        # Reward distribution
        if 'episode_rewards' in metrics_df.columns:
            rewards = metrics_df['episode_rewards'].dropna()
            fig.add_trace(
                go.Histogram(x=rewards, name='Reward Distribution', nbinsx=30),
                row=2, col=1
            )

        # Training stability (coefficient of variation)
        if 'episode_rewards' in metrics_df.columns:
            rewards = metrics_df['episode_rewards'].dropna()
            if len(rewards) > 50:
                rolling_mean = rewards.rolling(window=50).mean()
                rolling_std = rewards.rolling(window=50).std()
                cv = rolling_std / rolling_mean
                fig.add_trace(
                    go.Scatter(x=cv.index, y=cv, mode='lines',
                              name='Coefficient of Variation',
                              line=dict(color='purple')),
                    row=2, col=2
                )

        # Quality improvements over time
        if 'quality_improvements' in metrics_df.columns:
            quality = metrics_df['quality_improvements'].dropna()
            fig.add_trace(
                go.Scatter(x=quality.index, y=quality, mode='lines',
                          name='Quality Improvements',
                          line=dict(color='orange')),
                row=3, col=1
            )

        # Success rate over time
        if 'success_rates' in metrics_df.columns:
            success = metrics_df['success_rates'].dropna()
            fig.add_trace(
                go.Scatter(x=success.index, y=success, mode='lines',
                          name='Success Rate',
                          line=dict(color='green')),
                row=3, col=2
            )

        # Update layout
        fig.update_layout(
            height=900,
            title_text="Training Dashboard",
            showlegend=True
        )

        # Update x-axis labels
        fig.update_xaxes(title_text="Episode", row=3, col=1)
        fig.update_xaxes(title_text="Episode", row=3, col=2)

        if save_path:
            # Save as HTML
            pyo.plot(fig, filename=save_path, auto_open=False)
            return save_path

        return fig.to_html()

    def create_parameter_exploration_3d(self, parameter_history: List[Dict[str, Any]],
                                      save_path: Optional[str] = None) -> Optional[str]:
        """Create 3D parameter exploration visualization"""
        if not PLOTLY_AVAILABLE or not parameter_history:
            return None

        param_df = pd.DataFrame(parameter_history)
        numeric_cols = param_df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 3:
            logger.warning("Need at least 3 numeric parameters for 3D visualization")
            return None

        # Take first 3 parameters
        x_col, y_col, z_col = numeric_cols[:3]

        fig = go.Figure(data=go.Scatter3d(
            x=param_df[x_col],
            y=param_df[y_col],
            z=param_df[z_col],
            mode='markers+lines',
            marker=dict(
                size=5,
                color=range(len(param_df)),
                colorscale='Viridis',
                colorbar=dict(title="Time Step")
            ),
            line=dict(
                color='darkblue',
                width=2
            ),
            text=[f"Step {i}" for i in range(len(param_df))]
        ))

        fig.update_layout(
            title="3D Parameter Exploration Path",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            width=800,
            height=600
        )

        if save_path:
            pyo.plot(fig, filename=save_path, auto_open=False)
            return save_path

        return fig.to_html()


class ComparisonTools:
    """Tools for comparing training runs and experiments"""

    def __init__(self):
        self.processor = TrainingDataProcessor()

    def compare_experiments(self, experiments: Dict[str, pd.DataFrame],
                          metrics: List[str] = ['episode_rewards', 'quality_improvements'],
                          save_path: Optional[str] = None) -> plt.Figure:
        """Compare multiple training experiments"""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 6*n_metrics), dpi=100)
        if n_metrics == 1:
            axes = [axes]

        colors = sns.color_palette("husl", len(experiments))

        for i, metric in enumerate(metrics):
            ax = axes[i]

            for j, (exp_name, exp_df) in enumerate(experiments.items()):
                if metric not in exp_df.columns:
                    continue

                data = exp_df[metric].dropna()
                if len(data) == 0:
                    continue

                # Plot with moving average
                ma_data = self.processor.calculate_moving_averages(data, [50])

                # Raw data (light)
                ax.plot(data.index, data, alpha=0.2, color=colors[j])

                # Moving average (bold)
                if 'ma_50' in ma_data.columns:
                    ax.plot(data.index, ma_data['ma_50'], linewidth=2,
                           label=f'{exp_name} (MA-50)', color=colors[j])
                else:
                    ax.plot(data.index, data, linewidth=2,
                           label=exp_name, color=colors[j])

            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xlabel('Episode')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        return fig

    def statistical_comparison(self, experiments: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform statistical comparison of experiments"""
        results = {}

        # Common metrics to compare
        common_metrics = ['episode_rewards', 'quality_improvements', 'success_rates']

        for metric in common_metrics:
            metric_results = {}

            # Collect data for each experiment
            experiment_data = {}
            for exp_name, exp_df in experiments.items():
                if metric in exp_df.columns:
                    data = exp_df[metric].dropna()
                    if len(data) > 0:
                        experiment_data[exp_name] = data

            if len(experiment_data) < 2:
                continue

            # Calculate statistics for each experiment
            for exp_name, data in experiment_data.items():
                metric_results[exp_name] = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'median': data.median(),
                    'min': data.min(),
                    'max': data.max(),
                    'final_value': data.iloc[-1] if len(data) > 0 else 0,
                    'samples': len(data)
                }

            # Perform statistical tests
            if len(experiment_data) == 2:
                exp_names = list(experiment_data.keys())
                data1, data2 = experiment_data[exp_names[0]], experiment_data[exp_names[1]]

                # T-test for means
                try:
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    metric_results['statistical_test'] = {
                        'test': 't-test',
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except:
                    metric_results['statistical_test'] = {'error': 'Failed to compute t-test'}

                # Mann-Whitney U test (non-parametric)
                try:
                    u_stat, p_value_mw = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                    metric_results['mann_whitney'] = {
                        'u_statistic': u_stat,
                        'p_value': p_value_mw,
                        'significant': p_value_mw < 0.05
                    }
                except:
                    metric_results['mann_whitney'] = {'error': 'Failed to compute Mann-Whitney U test'}

            results[metric] = metric_results

        return results

    def create_comparison_report(self, statistical_results: Dict[str, Any]) -> str:
        """Create human-readable comparison report"""
        report = []
        report.append("# Training Experiments Statistical Comparison")
        report.append("=" * 50)
        report.append("")

        for metric, results in statistical_results.items():
            report.append(f"## {metric.replace('_', ' ').title()}")
            report.append("-" * 30)

            # Individual experiment statistics
            for exp_name, stats in results.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    report.append(f"**{exp_name}:**")
                    report.append(f"  - Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                    report.append(f"  - Median: {stats['median']:.4f}")
                    report.append(f"  - Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                    report.append(f"  - Final Value: {stats['final_value']:.4f}")
                    report.append(f"  - Samples: {stats['samples']}")
                    report.append("")

            # Statistical tests
            if 'statistical_test' in results:
                test_result = results['statistical_test']
                if 'p_value' in test_result:
                    significance = "significant" if test_result['significant'] else "not significant"
                    report.append(f"**T-test:** p-value = {test_result['p_value']:.4f} ({significance})")

            if 'mann_whitney' in results:
                mw_result = results['mann_whitney']
                if 'p_value' in mw_result:
                    significance = "significant" if mw_result['significant'] else "not significant"
                    report.append(f"**Mann-Whitney U:** p-value = {mw_result['p_value']:.4f} ({significance})")

            report.append("")

        return "\n".join(report)


class AnomalyDetector:
    """Detects anomalies and issues in training"""

    def __init__(self):
        self.processor = TrainingDataProcessor()

    def detect_training_anomalies(self, metrics_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Detect various types of training anomalies"""
        anomalies = defaultdict(list)

        # Reward anomalies
        if 'episode_rewards' in metrics_df.columns:
            rewards = metrics_df['episode_rewards'].dropna()
            reward_anomalies = self.processor.detect_anomalies(rewards, method='iqr')

            for idx in rewards[reward_anomalies].index:
                anomalies['reward_anomalies'].append({
                    'episode': idx,
                    'value': rewards[idx],
                    'type': 'outlier_reward',
                    'severity': 'medium'
                })

        # Training stagnation
        if 'episode_rewards' in metrics_df.columns:
            rewards = metrics_df['episode_rewards'].dropna()
            if len(rewards) > 100:
                # Check for stagnation in last 50 episodes
                recent_rewards = rewards.tail(50)
                if recent_rewards.std() < 0.01:  # Very low variance
                    anomalies['training_issues'].append({
                        'type': 'stagnation',
                        'description': 'Training appears stagnant (low reward variance)',
                        'episodes': f"{recent_rewards.index[0]}-{recent_rewards.index[-1]}",
                        'severity': 'high'
                    })

        # Loss explosion
        loss_metrics = ['training_loss', 'value_loss', 'policy_loss']
        for loss_metric in loss_metrics:
            if loss_metric in metrics_df.columns:
                loss_data = metrics_df[loss_metric].dropna()
                if len(loss_data) > 10:
                    # Check for sudden loss increases
                    loss_diff = loss_data.diff()
                    large_increases = loss_diff > (loss_diff.mean() + 3 * loss_diff.std())

                    for idx in loss_data[large_increases].index:
                        anomalies['loss_anomalies'].append({
                            'episode': idx,
                            'loss_type': loss_metric,
                            'value': loss_data[idx],
                            'increase': loss_diff[idx],
                            'type': 'loss_explosion',
                            'severity': 'high'
                        })

        # Performance degradation
        if 'quality_improvements' in metrics_df.columns:
            quality = metrics_df['quality_improvements'].dropna()
            if len(quality) > 50:
                # Check trend in recent episodes
                recent_quality = quality.tail(30)
                trend_slope = np.polyfit(range(len(recent_quality)), recent_quality, 1)[0]

                if trend_slope < -0.01:  # Negative trend
                    anomalies['performance_issues'].append({
                        'type': 'degradation',
                        'description': f'Quality improvements declining (slope: {trend_slope:.4f})',
                        'episodes': f"{recent_quality.index[0]}-{recent_quality.index[-1]}",
                        'severity': 'medium'
                    })

        # Success rate drops
        if 'success_rates' in metrics_df.columns:
            success_rates = metrics_df['success_rates'].dropna()
            if len(success_rates) > 20:
                # Check for sudden drops
                success_diff = success_rates.diff()
                large_drops = success_diff < -0.2  # Drop of more than 20%

                for idx in success_rates[large_drops].index:
                    anomalies['success_anomalies'].append({
                        'episode': idx,
                        'success_rate': success_rates[idx],
                        'drop': success_diff[idx],
                        'type': 'success_rate_drop',
                        'severity': 'medium'
                    })

        return dict(anomalies)

    def generate_anomaly_report(self, anomalies: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate human-readable anomaly report"""
        report = []
        report.append("# Training Anomaly Detection Report")
        report.append("=" * 40)
        report.append("")

        if not any(anomalies.values()):
            report.append("✅ No significant anomalies detected in training.")
            return "\n".join(report)

        severity_counts = defaultdict(int)

        for category, anomaly_list in anomalies.items():
            if not anomaly_list:
                continue

            report.append(f"## {category.replace('_', ' ').title()}")
            report.append("-" * 30)

            for anomaly in anomaly_list:
                severity = anomaly.get('severity', 'unknown')
                severity_counts[severity] += 1

                if severity == 'high':
                    icon = "🚨"
                elif severity == 'medium':
                    icon = "⚠️"
                else:
                    icon = "ℹ️"

                report.append(f"{icon} **{anomaly.get('type', 'Unknown')}**")

                if 'episode' in anomaly:
                    report.append(f"  - Episode: {anomaly['episode']}")
                if 'episodes' in anomaly:
                    report.append(f"  - Episodes: {anomaly['episodes']}")
                if 'description' in anomaly:
                    report.append(f"  - Description: {anomaly['description']}")
                if 'value' in anomaly:
                    report.append(f"  - Value: {anomaly['value']:.4f}")

                report.append("")

        # Summary
        report.append("## Summary")
        report.append("-" * 10)
        for severity, count in severity_counts.items():
            report.append(f"- {severity.title()} severity: {count} anomalies")

        return "\n".join(report)


class TrainingVisualizer:
    """Main class for comprehensive training visualization and reporting"""

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.processor = TrainingDataProcessor()
        self.plotter = TrainingPlotter()
        self.dashboard = InteractiveDashboard() if PLOTLY_AVAILABLE else None
        self.comparator = ComparisonTools()
        self.anomaly_detector = AnomalyDetector()

        logger.info(f"Training Visualizer initialized. Output directory: {self.output_dir}")

    def generate_comprehensive_report(self, training_data: Dict[str, Any],
                                    experiment_name: str = "training_experiment") -> Dict[str, str]:
        """Generate comprehensive training report with all visualizations"""
        logger.info(f"Generating comprehensive report for {experiment_name}")

        # Process training data
        if 'metrics_history' in training_data:
            metrics_df = self.processor.process_metrics_history(training_data['metrics_history'])
        else:
            logger.warning("No metrics history found in training data")
            metrics_df = pd.DataFrame()

        report_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{experiment_name}_{timestamp}"

        # 1. Learning Curves
        try:
            fig = self.plotter.plot_learning_curves(metrics_df)
            learning_curves_path = self.output_dir / f"{base_name}_learning_curves.png"
            fig.savefig(learning_curves_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            report_files['learning_curves'] = str(learning_curves_path)
            logger.info("✅ Learning curves generated")
        except Exception as e:
            logger.error(f"❌ Failed to generate learning curves: {e}")

        # 2. Loss Curves
        try:
            fig = self.plotter.plot_loss_curves(metrics_df)
            loss_curves_path = self.output_dir / f"{base_name}_loss_curves.png"
            fig.savefig(loss_curves_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            report_files['loss_curves'] = str(loss_curves_path)
            logger.info("✅ Loss curves generated")
        except Exception as e:
            logger.error(f"❌ Failed to generate loss curves: {e}")

        # 3. Reward Distribution
        try:
            fig = self.plotter.plot_reward_distribution(metrics_df)
            reward_dist_path = self.output_dir / f"{base_name}_reward_distribution.png"
            fig.savefig(reward_dist_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            report_files['reward_distribution'] = str(reward_dist_path)
            logger.info("✅ Reward distribution generated")
        except Exception as e:
            logger.error(f"❌ Failed to generate reward distribution: {e}")

        # 4. Parameter Exploration
        if 'parameter_history' in training_data:
            try:
                fig = self.plotter.plot_parameter_exploration(training_data['parameter_history'])
                param_exp_path = self.output_dir / f"{base_name}_parameter_exploration.png"
                fig.savefig(param_exp_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                report_files['parameter_exploration'] = str(param_exp_path)
                logger.info("✅ Parameter exploration generated")
            except Exception as e:
                logger.error(f"❌ Failed to generate parameter exploration: {e}")

        # 5. Training Stability
        try:
            fig = self.plotter.plot_training_stability(metrics_df)
            stability_path = self.output_dir / f"{base_name}_training_stability.png"
            fig.savefig(stability_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            report_files['training_stability'] = str(stability_path)
            logger.info("✅ Training stability analysis generated")
        except Exception as e:
            logger.error(f"❌ Failed to generate training stability analysis: {e}")

        # 6. Interactive Dashboard
        if self.dashboard:
            try:
                dashboard_path = self.output_dir / f"{base_name}_interactive_dashboard.html"
                self.dashboard.create_interactive_training_dashboard(metrics_df, str(dashboard_path))
                report_files['interactive_dashboard'] = str(dashboard_path)
                logger.info("✅ Interactive dashboard generated")
            except Exception as e:
                logger.error(f"❌ Failed to generate interactive dashboard: {e}")

        # 7. Anomaly Detection
        try:
            anomalies = self.anomaly_detector.detect_training_anomalies(metrics_df)
            anomaly_report = self.anomaly_detector.generate_anomaly_report(anomalies)
            anomaly_path = self.output_dir / f"{base_name}_anomaly_report.txt"
            with open(anomaly_path, 'w') as f:
                f.write(anomaly_report)
            report_files['anomaly_report'] = str(anomaly_path)
            logger.info("✅ Anomaly detection report generated")
        except Exception as e:
            logger.error(f"❌ Failed to generate anomaly report: {e}")

        # 8. Summary Statistics
        try:
            summary_stats = self._generate_summary_statistics(metrics_df, training_data)
            summary_path = self.output_dir / f"{base_name}_summary_statistics.json"
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            report_files['summary_statistics'] = str(summary_path)
            logger.info("✅ Summary statistics generated")
        except Exception as e:
            logger.error(f"❌ Failed to generate summary statistics: {e}")

        # 9. Convergence Analysis
        try:
            convergence_analysis = self._analyze_convergence(metrics_df)
            convergence_path = self.output_dir / f"{base_name}_convergence_analysis.json"
            with open(convergence_path, 'w') as f:
                json.dump(convergence_analysis, f, indent=2)
            report_files['convergence_analysis'] = str(convergence_path)
            logger.info("✅ Convergence analysis generated")
        except Exception as e:
            logger.error(f"❌ Failed to generate convergence analysis: {e}")

        # 10. Master Report
        try:
            master_report = self._generate_master_report(training_data, report_files, experiment_name)
            master_path = self.output_dir / f"{base_name}_master_report.md"
            with open(master_path, 'w') as f:
                f.write(master_report)
            report_files['master_report'] = str(master_path)
            logger.info("✅ Master report generated")
        except Exception as e:
            logger.error(f"❌ Failed to generate master report: {e}")

        logger.info(f"📊 Comprehensive report generated with {len(report_files)} components")
        return report_files

    def compare_training_runs(self, training_runs: Dict[str, Dict[str, Any]],
                            output_name: str = "comparison") -> Dict[str, str]:
        """Compare multiple training runs"""
        logger.info(f"Comparing {len(training_runs)} training runs")

        # Process all experiments
        experiments = {}
        for run_name, run_data in training_runs.items():
            if 'metrics_history' in run_data:
                experiments[run_name] = self.processor.process_metrics_history(run_data['metrics_history'])

        if len(experiments) < 2:
            logger.warning("Need at least 2 experiments for comparison")
            return {}

        comparison_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{output_name}_{timestamp}"

        # Comparison plots
        try:
            fig = self.comparator.compare_experiments(experiments)
            comparison_path = self.output_dir / f"{base_name}_comparison.png"
            fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            comparison_files['comparison_plots'] = str(comparison_path)
            logger.info("✅ Comparison plots generated")
        except Exception as e:
            logger.error(f"❌ Failed to generate comparison plots: {e}")

        # Statistical comparison
        try:
            statistical_results = self.comparator.statistical_comparison(experiments)
            comparison_report = self.comparator.create_comparison_report(statistical_results)

            stats_path = self.output_dir / f"{base_name}_statistical_comparison.json"
            with open(stats_path, 'w') as f:
                json.dump(statistical_results, f, indent=2)

            report_path = self.output_dir / f"{base_name}_comparison_report.txt"
            with open(report_path, 'w') as f:
                f.write(comparison_report)

            comparison_files['statistical_comparison'] = str(stats_path)
            comparison_files['comparison_report'] = str(report_path)
            logger.info("✅ Statistical comparison generated")
        except Exception as e:
            logger.error(f"❌ Failed to generate statistical comparison: {e}")

        logger.info(f"📊 Training run comparison completed with {len(comparison_files)} files")
        return comparison_files

    def monitor_training_progress(self, metrics_history: List[Dict[str, Any]],
                                real_time: bool = True) -> Dict[str, Any]:
        """Monitor training progress in real-time or batch mode"""
        metrics_df = self.processor.process_metrics_history(metrics_history)

        # Calculate progress metrics
        progress_metrics = {}

        if 'episode_rewards' in metrics_df.columns:
            rewards = metrics_df['episode_rewards'].dropna()
            if len(rewards) > 0:
                progress_metrics['latest_reward'] = rewards.iloc[-1]
                progress_metrics['average_reward'] = rewards.mean()
                progress_metrics['reward_trend'] = np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) > 1 else 0

                # Calculate improvement
                if len(rewards) > 10:
                    recent_avg = rewards.tail(10).mean()
                    early_avg = rewards.head(10).mean()
                    progress_metrics['improvement_rate'] = (recent_avg - early_avg) / early_avg if early_avg != 0 else 0

        # Convergence analysis
        if len(metrics_df) > 50:
            convergence_metrics = self.processor.calculate_convergence_metrics(
                metrics_df['episode_rewards'] if 'episode_rewards' in metrics_df.columns else pd.Series()
            )
            progress_metrics.update(convergence_metrics)

        # Anomaly detection
        anomalies = self.anomaly_detector.detect_training_anomalies(metrics_df)
        progress_metrics['anomalies_detected'] = sum(len(v) for v in anomalies.values())
        progress_metrics['high_severity_anomalies'] = sum(
            len([a for a in v if a.get('severity') == 'high']) for v in anomalies.values()
        )

        # Training health score
        health_score = self._calculate_training_health_score(metrics_df, anomalies)
        progress_metrics['health_score'] = health_score

        return progress_metrics

    def export_plots(self, figures: List[plt.Figure], output_dir: Optional[str] = None,
                    formats: List[str] = ['png', 'pdf']) -> List[str]:
        """Export plots in multiple formats"""
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, fig in enumerate(figures):
            for fmt in formats:
                filename = output_dir / f"plot_{i}_{timestamp}.{fmt}"
                try:
                    fig.savefig(filename, format=fmt, dpi=150, bbox_inches='tight')
                    exported_files.append(str(filename))
                except Exception as e:
                    logger.error(f"Failed to export plot {i} as {fmt}: {e}")

        logger.info(f"Exported {len(exported_files)} plot files")
        return exported_files

    def _generate_summary_statistics(self, metrics_df: pd.DataFrame,
                                   training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_episodes': len(metrics_df),
            'metrics_available': list(metrics_df.columns)
        }

        # Reward statistics
        if 'episode_rewards' in metrics_df.columns:
            rewards = metrics_df['episode_rewards'].dropna()
            summary['reward_statistics'] = {
                'mean': float(rewards.mean()),
                'std': float(rewards.std()),
                'min': float(rewards.min()),
                'max': float(rewards.max()),
                'median': float(rewards.median()),
                'final_value': float(rewards.iloc[-1]) if len(rewards) > 0 else 0
            }

        # Quality statistics
        if 'quality_improvements' in metrics_df.columns:
            quality = metrics_df['quality_improvements'].dropna()
            summary['quality_statistics'] = {
                'mean': float(quality.mean()),
                'std': float(quality.std()),
                'min': float(quality.min()),
                'max': float(quality.max()),
                'final_value': float(quality.iloc[-1]) if len(quality) > 0 else 0
            }

        # Training efficiency
        if 'training_time' in training_data:
            summary['training_efficiency'] = {
                'total_time': training_data['training_time'],
                'episodes_per_hour': len(metrics_df) / (training_data['training_time'] / 3600) if training_data['training_time'] > 0 else 0
            }

        return summary

    def _analyze_convergence(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze training convergence"""
        convergence_analysis = {}

        if 'episode_rewards' in metrics_df.columns:
            rewards = metrics_df['episode_rewards'].dropna()
            convergence_metrics = self.processor.calculate_convergence_metrics(rewards)
            convergence_analysis['reward_convergence'] = convergence_metrics

            # Estimate convergence point
            if len(rewards) > 100:
                # Use rolling variance to detect convergence
                rolling_var = rewards.rolling(window=50).var()
                threshold = rolling_var.mean() * 0.1  # 10% of mean variance

                convergence_candidates = rolling_var < threshold
                if convergence_candidates.any():
                    convergence_point = convergence_candidates.idxmax()
                    convergence_analysis['estimated_convergence_episode'] = int(convergence_point)
                else:
                    convergence_analysis['estimated_convergence_episode'] = None

        return convergence_analysis

    def _calculate_training_health_score(self, metrics_df: pd.DataFrame,
                                       anomalies: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate overall training health score (0-100)"""
        score = 100.0

        # Penalize for anomalies
        total_anomalies = sum(len(v) for v in anomalies.values())
        high_severity_anomalies = sum(
            len([a for a in v if a.get('severity') == 'high']) for v in anomalies.values()
        )

        score -= high_severity_anomalies * 20  # High severity: -20 points each
        score -= (total_anomalies - high_severity_anomalies) * 5  # Other anomalies: -5 points each

        # Reward for stability
        if 'episode_rewards' in metrics_df.columns:
            rewards = metrics_df['episode_rewards'].dropna()
            if len(rewards) > 50:
                recent_rewards = rewards.tail(50)
                cv = recent_rewards.std() / recent_rewards.mean() if recent_rewards.mean() != 0 else float('inf')
                if cv < 0.1:  # Low coefficient of variation = stable
                    score += 10

        # Reward for positive trend
        if 'episode_rewards' in metrics_df.columns:
            rewards = metrics_df['episode_rewards'].dropna()
            if len(rewards) > 20:
                trend = np.polyfit(range(len(rewards)), rewards, 1)[0]
                if trend > 0:
                    score += min(trend * 1000, 15)  # Up to 15 bonus points for positive trend

        return max(0.0, min(100.0, score))

    def _generate_master_report(self, training_data: Dict[str, Any],
                              report_files: Dict[str, str], experiment_name: str) -> str:
        """Generate master markdown report"""
        report = []
        report.append(f"# Training Visualization Report: {experiment_name}")
        report.append("=" * 80)
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overview
        report.append("## Overview")
        report.append("This report contains comprehensive analysis of the training process,")
        report.append("including learning curves, performance metrics, anomaly detection, and comparisons.")
        report.append("")

        # Generated Files
        report.append("## Generated Files")
        for file_type, file_path in report_files.items():
            filename = Path(file_path).name
            report.append(f"- **{file_type.replace('_', ' ').title()}:** `{filename}`")
        report.append("")

        # Quick Statistics (if available)
        if 'summary_statistics' in report_files:
            try:
                with open(report_files['summary_statistics'], 'r') as f:
                    stats = json.load(f)

                report.append("## Quick Statistics")
                report.append(f"- **Total Episodes:** {stats.get('total_episodes', 'N/A')}")

                if 'reward_statistics' in stats:
                    rs = stats['reward_statistics']
                    report.append(f"- **Average Reward:** {rs['mean']:.4f} ± {rs['std']:.4f}")
                    report.append(f"- **Final Reward:** {rs['final_value']:.4f}")

                if 'training_efficiency' in stats:
                    te = stats['training_efficiency']
                    report.append(f"- **Training Time:** {te['total_time']:.2f} seconds")
                    report.append(f"- **Episodes/Hour:** {te['episodes_per_hour']:.1f}")

                report.append("")
            except:
                pass

        # Recommendations
        report.append("## Recommendations")
        report.append("1. Review learning curves for convergence patterns")
        report.append("2. Check anomaly detection report for training issues")
        report.append("3. Examine parameter exploration for optimization opportunities")
        report.append("4. Compare with baseline methods using comparison tools")
        report.append("")

        # Usage Instructions
        report.append("## Usage Instructions")
        report.append("- **Static Plots:** View PNG files in any image viewer")
        report.append("- **Interactive Dashboard:** Open HTML file in web browser")
        report.append("- **Data Analysis:** Use JSON files for programmatic analysis")
        report.append("- **Reports:** Read TXT/MD files for detailed analysis")

        return "\n".join(report)


# Factory function for easy usage
def create_training_visualizer(output_dir: str = "training_visualizations") -> TrainingVisualizer:
    """
    Factory function to create training visualizer

    Args:
        output_dir: Directory to save visualizations and reports

    Returns:
        Configured training visualizer
    """
    return TrainingVisualizer(output_dir)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    visualizer = create_training_visualizer("example_visualizations")

    # Example training data
    example_data = {
        'metrics_history': [
            {'episode': i, 'episode_rewards': np.random.normal(i*0.1, 0.5),
             'quality_improvements': np.random.normal(0.8 + i*0.01, 0.1),
             'training_loss': np.exp(-i*0.01) + np.random.normal(0, 0.1)}
            for i in range(1000)
        ],
        'parameter_history': [
            {'learning_rate': 0.0003 + np.random.normal(0, 0.0001),
             'batch_size': 64,
             'exploration_rate': max(0.01, 0.9 - i*0.001)}
            for i in range(100)
        ],
        'training_time': 3600  # 1 hour
    }

    # Generate comprehensive report
    try:
        report_files = visualizer.generate_comprehensive_report(example_data, "example_experiment")
        print(f"Generated {len(report_files)} report files:")
        for file_type, path in report_files.items():
            print(f"  - {file_type}: {path}")
    except Exception as e:
        print(f"Error generating report: {e}")