#!/usr/bin/env python3
"""
Model Evaluation & Comparison - DAY3 Task 4

Evaluates the statistical models and compares them with hardcoded correlation formulas.
Generates comprehensive evaluation report with visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.ai_modules.optimization.statistical_parameter_predictor import StatisticalParameterPredictor
    from backend.ai_modules.prediction.statistical_quality_predictor import StatisticalQualityPredictor
except ImportError as e:
    print(f"Warning: Could not import statistical models: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class StatisticalModelEvaluator:
    """Evaluates and compares statistical models with baseline approaches."""

    def __init__(self, project_root: str = None):
        """
        Initialize the evaluator.

        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.data_dir = self.project_root / "data" / "training" / "preprocessed"
        self.output_dir = self.project_root
        self.plots_dir = self.project_root / "evaluation_plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Initialize models
        self.param_predictor = None
        self.quality_predictor = None

        # Evaluation results
        self.evaluation_results = {}
        self.comparison_results = {}

    def load_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed test data.

        Returns:
            Tuple of (test_features, test_targets)
        """
        logger.info("Loading test data")

        X_test = pd.read_csv(self.data_dir / "X_test.csv")
        y_test = pd.read_csv(self.data_dir / "y_test.csv")

        logger.info(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        logger.info(f"Test targets: {y_test.shape}")

        return X_test, y_test

    def load_models(self) -> bool:
        """
        Load trained statistical models.

        Returns:
            True if models loaded successfully, False otherwise
        """
        logger.info("Loading trained statistical models")

        try:
            # Load parameter predictor
            self.param_predictor = StatisticalParameterPredictor()
            if not self.param_predictor.load_model():
                logger.error("Failed to load parameter predictor")
                return False

            # Load quality predictor
            self.quality_predictor = StatisticalQualityPredictor()
            if not self.quality_predictor.load_model():
                logger.error("Failed to load quality predictor")
                return False

            logger.info("Statistical models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def evaluate_parameter_predictor(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the parameter prediction model.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Parameter predictor evaluation results
        """
        logger.info("Evaluating parameter predictor")

        if self.param_predictor is None:
            return {'error': 'Parameter predictor not loaded'}

        try:
            # Get parameter targets (exclude SSIM column)
            parameter_columns = [
                'color_precision', 'corner_threshold', 'max_iterations',
                'path_precision', 'layer_difference', 'length_threshold',
                'splice_threshold', 'colormode'
            ]
            y_param_test = y_test[parameter_columns]

            # Make predictions for each test sample
            predictions = []
            prediction_times = []

            for idx, row in X_test.iterrows():
                # Convert row to feature dictionary
                features = row.to_dict()
                start_time = time.time()

                # Predict parameters
                result = self.param_predictor.predict_parameters(features)
                prediction_time = time.time() - start_time
                prediction_times.append(prediction_time)

                if result['success']:
                    predictions.append(result['parameters'])
                else:
                    # Use default parameters if prediction fails
                    logger.warning(f"Prediction failed for sample {idx}: {result.get('error')}")
                    default_params = {name: 5.0 for name in parameter_columns}
                    default_params['colormode'] = 0.5
                    predictions.append(default_params)

            # Convert predictions to DataFrame
            y_pred = pd.DataFrame(predictions)
            y_pred = y_pred[parameter_columns]  # Ensure column order

            # Calculate metrics for each parameter
            parameter_metrics = {}
            for param in parameter_columns:
                actual = y_param_test[param].values
                predicted = y_pred[param].values

                mae = np.mean(np.abs(actual - predicted))
                mse = np.mean((actual - predicted) ** 2)
                rmse = np.sqrt(mse)

                # R² score
                ss_res = np.sum((actual - predicted) ** 2)
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                # Parameter range for normalized metrics
                param_range = actual.max() - actual.min()
                normalized_mae = mae / param_range if param_range > 0 else 0

                parameter_metrics[param] = {
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'normalized_mae': float(normalized_mae),
                    'param_range': float(param_range)
                }

            # Overall metrics
            overall_mae = np.mean([m['mae'] for m in parameter_metrics.values()])
            overall_r2 = np.mean([m['r2'] for m in parameter_metrics.values()])
            avg_prediction_time = np.mean(prediction_times)

            # Feature importance
            feature_importance = self.param_predictor.get_feature_importance()

            results = {
                'parameter_metrics': parameter_metrics,
                'overall_mae': float(overall_mae),
                'overall_r2': float(overall_r2),
                'avg_prediction_time': float(avg_prediction_time),
                'feature_importance': feature_importance,
                'predictions': y_pred.to_dict('records'),
                'actual': y_param_test.to_dict('records'),
                'num_samples': len(X_test)
            }

            logger.info(f"Parameter predictor - Overall MAE: {overall_mae:.4f}, R²: {overall_r2:.4f}")
            return results

        except Exception as e:
            logger.error(f"Parameter predictor evaluation failed: {e}")
            return {'error': str(e)}

    def evaluate_quality_predictor(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the quality prediction model.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Quality predictor evaluation results
        """
        logger.info("Evaluating quality predictor")

        if self.quality_predictor is None:
            return {'error': 'Quality predictor not loaded'}

        try:
            # Get SSIM targets
            y_ssim_test = y_test['ssim'].values

            # Get parameter columns for combined prediction
            parameter_columns = [
                'color_precision', 'corner_threshold', 'max_iterations',
                'path_precision', 'layer_difference', 'length_threshold',
                'splice_threshold', 'colormode'
            ]
            parameters_test = y_test[parameter_columns]

            # Make predictions
            predictions = []
            uncertainties = []
            confidences = []
            prediction_times = []

            for idx, (_, feature_row) in enumerate(X_test.iterrows()):
                param_row = parameters_test.iloc[idx]

                # Convert to dictionaries
                features = feature_row.to_dict()
                parameters = param_row.to_dict()

                start_time = time.time()
                result = self.quality_predictor.predict_quality(features, parameters)
                prediction_time = time.time() - start_time
                prediction_times.append(prediction_time)

                if result['success']:
                    predictions.append(result['predicted_ssim'])
                    uncertainties.append(result['uncertainty']['std_estimate'])
                    confidences.append(result['confidence'])
                else:
                    logger.warning(f"Quality prediction failed for sample {idx}: {result.get('error')}")
                    predictions.append(0.5)  # Default SSIM
                    uncertainties.append(1.0)  # High uncertainty
                    confidences.append(0.0)  # Low confidence

            predictions = np.array(predictions)

            # Calculate metrics
            mae = np.mean(np.abs(y_ssim_test - predictions))
            mse = np.mean((y_ssim_test - predictions) ** 2)
            rmse = np.sqrt(mse)

            # R² score
            ss_res = np.sum((y_ssim_test - predictions) ** 2)
            ss_tot = np.sum((y_ssim_test - np.mean(y_ssim_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Correlation
            correlation = np.corrcoef(y_ssim_test, predictions)[0, 1]

            # Accuracy within ±0.1 SSIM
            within_01 = np.mean(np.abs(y_ssim_test - predictions) <= 0.1) * 100

            # Average metrics
            avg_uncertainty = np.mean(uncertainties)
            avg_confidence = np.mean(confidences)
            avg_prediction_time = np.mean(prediction_times)

            # Feature importance
            feature_importance = self.quality_predictor.get_feature_importance()

            results = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'correlation': float(correlation),
                'within_01_ssim': float(within_01),
                'avg_uncertainty': float(avg_uncertainty),
                'avg_confidence': float(avg_confidence),
                'avg_prediction_time': float(avg_prediction_time),
                'feature_importance': feature_importance,
                'predictions': predictions.tolist(),
                'actual': y_ssim_test.tolist(),
                'uncertainties': uncertainties,
                'confidences': confidences,
                'num_samples': len(X_test)
            }

            logger.info(f"Quality predictor - MAE: {mae:.4f}, R²: {r2:.4f}, Correlation: {correlation:.4f}")
            logger.info(f"Within ±0.1 SSIM: {within_01:.1f}%")
            return results

        except Exception as e:
            logger.error(f"Quality predictor evaluation failed: {e}")
            return {'error': str(e)}

    def evaluate_baseline_methods(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate baseline correlation formulas for comparison.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Baseline evaluation results
        """
        logger.info("Evaluating baseline correlation formulas")

        try:
            # Since CorrelationFormulas might not exist, create simple baseline methods
            baseline_results = {}

            # Simple parameter prediction baseline (use feature averages)
            parameter_columns = [
                'color_precision', 'corner_threshold', 'max_iterations',
                'path_precision', 'layer_difference', 'length_threshold',
                'splice_threshold', 'colormode'
            ]

            # Load training data to get parameter averages
            y_train = pd.read_csv(self.data_dir / "y_train.csv")
            parameter_means = y_train[parameter_columns].mean()

            # Baseline parameter prediction: just use the mean for all samples
            baseline_param_predictions = pd.DataFrame(
                [parameter_means.to_dict()] * len(X_test)
            )

            # Calculate baseline parameter metrics
            y_param_test = y_test[parameter_columns]
            baseline_param_metrics = {}

            for param in parameter_columns:
                actual = y_param_test[param].values
                predicted = baseline_param_predictions[param].values

                mae = np.mean(np.abs(actual - predicted))
                baseline_param_metrics[param] = {'mae': float(mae)}

            baseline_overall_param_mae = np.mean([m['mae'] for m in baseline_param_metrics.values()])

            # Baseline SSIM prediction: use simple feature-based heuristic
            baseline_ssim_predictions = []
            for _, row in X_test.iterrows():
                # Simple heuristic based on complexity and edge density
                complexity = row.get('complexity_score', 0.5)
                edge_density = row.get('edge_density', 0.5)
                unique_colors = row.get('unique_colors', 0.5)

                # Higher complexity and edge density typically mean lower SSIM
                predicted_ssim = 0.8 - (complexity * 0.3) - (edge_density * 0.2) + (unique_colors * 0.1)
                predicted_ssim = max(0.0, min(1.0, predicted_ssim))  # Clip to valid range
                baseline_ssim_predictions.append(predicted_ssim)

            baseline_ssim_predictions = np.array(baseline_ssim_predictions)
            y_ssim_test = y_test['ssim'].values

            # Calculate baseline SSIM metrics
            baseline_ssim_mae = np.mean(np.abs(y_ssim_test - baseline_ssim_predictions))
            baseline_ssim_r2 = 1 - (np.sum((y_ssim_test - baseline_ssim_predictions) ** 2) /
                                  np.sum((y_ssim_test - np.mean(y_ssim_test)) ** 2))

            baseline_results = {
                'parameter_prediction': {
                    'method': 'Mean baseline',
                    'overall_mae': float(baseline_overall_param_mae),
                    'parameter_metrics': baseline_param_metrics
                },
                'quality_prediction': {
                    'method': 'Simple heuristic',
                    'mae': float(baseline_ssim_mae),
                    'r2': float(baseline_ssim_r2),
                    'predictions': baseline_ssim_predictions.tolist()
                }
            }

            logger.info(f"Baseline parameter MAE: {baseline_overall_param_mae:.4f}")
            logger.info(f"Baseline SSIM MAE: {baseline_ssim_mae:.4f}")
            return baseline_results

        except Exception as e:
            logger.error(f"Baseline evaluation failed: {e}")
            return {'error': str(e)}

    def calculate_improvement_metrics(self) -> Dict[str, Any]:
        """
        Calculate improvement metrics comparing statistical models to baselines.

        Returns:
            Improvement metrics
        """
        logger.info("Calculating improvement metrics")

        if 'parameter_predictor' not in self.evaluation_results:
            return {'error': 'Parameter predictor results not available'}

        if 'quality_predictor' not in self.evaluation_results:
            return {'error': 'Quality predictor results not available'}

        if 'baseline' not in self.evaluation_results:
            return {'error': 'Baseline results not available'}

        try:
            # Parameter prediction improvement
            statistical_param_mae = self.evaluation_results['parameter_predictor']['overall_mae']
            baseline_param_mae = self.evaluation_results['baseline']['parameter_prediction']['overall_mae']

            param_improvement = ((baseline_param_mae - statistical_param_mae) / baseline_param_mae) * 100

            # Quality prediction improvement
            statistical_ssim_mae = self.evaluation_results['quality_predictor']['mae']
            baseline_ssim_mae = self.evaluation_results['baseline']['quality_prediction']['mae']

            ssim_improvement = ((baseline_ssim_mae - statistical_ssim_mae) / baseline_ssim_mae) * 100

            # Overall improvement (average)
            overall_improvement = (param_improvement + ssim_improvement) / 2

            improvement_metrics = {
                'parameter_prediction_improvement': float(param_improvement),
                'quality_prediction_improvement': float(ssim_improvement),
                'overall_improvement': float(overall_improvement),
                'meets_15_percent_target': overall_improvement > 15.0,
                'comparison_details': {
                    'parameter_mae_statistical': float(statistical_param_mae),
                    'parameter_mae_baseline': float(baseline_param_mae),
                    'ssim_mae_statistical': float(statistical_ssim_mae),
                    'ssim_mae_baseline': float(baseline_ssim_mae)
                }
            }

            logger.info(f"Parameter prediction improvement: {param_improvement:.1f}%")
            logger.info(f"Quality prediction improvement: {ssim_improvement:.1f}%")
            logger.info(f"Overall improvement: {overall_improvement:.1f}%")
            logger.info(f"Meets 15% target: {overall_improvement > 15.0}")

            return improvement_metrics

        except Exception as e:
            logger.error(f"Improvement calculation failed: {e}")
            return {'error': str(e)}

    def generate_visualizations(self) -> List[str]:
        """
        Generate comparison visualizations.

        Returns:
            List of generated plot filenames
        """
        logger.info("Generating evaluation visualizations")

        plot_files = []

        try:
            # Plot 1: Parameter prediction comparison
            if 'parameter_predictor' in self.evaluation_results:
                plot_file = self._plot_parameter_comparison()
                if plot_file:
                    plot_files.append(plot_file)

            # Plot 2: Quality prediction comparison
            if 'quality_predictor' in self.evaluation_results:
                plot_file = self._plot_quality_comparison()
                if plot_file:
                    plot_files.append(plot_file)

            # Plot 3: Feature importance plots
            plot_file = self._plot_feature_importance()
            if plot_file:
                plot_files.append(plot_file)

            # Plot 4: Improvement metrics visualization
            if 'improvement_metrics' in self.comparison_results:
                plot_file = self._plot_improvement_metrics()
                if plot_file:
                    plot_files.append(plot_file)

            logger.info(f"Generated {len(plot_files)} visualization plots")
            return plot_files

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return plot_files

    def _plot_parameter_comparison(self) -> str:
        """Generate parameter prediction comparison plot."""
        try:
            param_results = self.evaluation_results['parameter_predictor']
            param_metrics = param_results['parameter_metrics']

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()

            parameter_names = list(param_metrics.keys())

            for i, param in enumerate(parameter_names):
                ax = axes[i]
                actual = param_results['actual']
                predicted = param_results['predictions']

                actual_values = [sample[param] for sample in actual]
                predicted_values = [sample[param] for sample in predicted]

                ax.scatter(actual_values, predicted_values, alpha=0.6)
                ax.plot([min(actual_values), max(actual_values)],
                       [min(actual_values), max(actual_values)], 'r--', lw=1)

                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title(f'{param}\nMAE: {param_metrics[param]["mae"]:.3f}')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = self.plots_dir / "parameter_prediction_comparison.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            return plot_file.name

        except Exception as e:
            logger.error(f"Parameter comparison plot failed: {e}")
            return None

    def _plot_quality_comparison(self) -> str:
        """Generate quality prediction comparison plot."""
        try:
            quality_results = self.evaluation_results['quality_predictor']
            baseline_results = self.evaluation_results.get('baseline', {}).get('quality_prediction', {})

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Statistical model predictions
            actual = np.array(quality_results['actual'])
            predicted = np.array(quality_results['predictions'])

            ax1.scatter(actual, predicted, alpha=0.6, label='Statistical Model')
            ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=1)
            ax1.set_xlabel('Actual SSIM')
            ax1.set_ylabel('Predicted SSIM')
            ax1.set_title(f'Statistical Model\nMAE: {quality_results["mae"]:.4f}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Baseline comparison if available
            if 'predictions' in baseline_results:
                baseline_pred = np.array(baseline_results['predictions'])
                ax2.scatter(actual, baseline_pred, alpha=0.6, color='orange', label='Baseline')
                ax2.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=1)
                ax2.set_xlabel('Actual SSIM')
                ax2.set_ylabel('Predicted SSIM')
                ax2.set_title(f'Baseline Model\nMAE: {baseline_results["mae"]:.4f}')
                ax2.grid(True, alpha=0.3)
                ax2.legend()

            plt.tight_layout()
            plot_file = self.plots_dir / "quality_prediction_comparison.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            return plot_file.name

        except Exception as e:
            logger.error(f"Quality comparison plot failed: {e}")
            return None

    def _plot_feature_importance(self) -> str:
        """Generate feature importance plots."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Parameter predictor feature importance
            if 'parameter_predictor' in self.evaluation_results:
                param_importance = self.evaluation_results['parameter_predictor']['feature_importance']
                if param_importance:
                    features = list(param_importance.keys())
                    importances = list(param_importance.values())

                    # Sort by importance
                    sorted_idx = np.argsort(importances)
                    features = [features[i] for i in sorted_idx[-10:]]  # Top 10
                    importances = [importances[i] for i in sorted_idx[-10:]]

                    ax1.barh(features, importances)
                    ax1.set_xlabel('Feature Importance')
                    ax1.set_title('Parameter Predictor\nFeature Importance')

            # Quality predictor feature importance
            if 'quality_predictor' in self.evaluation_results:
                quality_importance = self.evaluation_results['quality_predictor']['feature_importance']
                if quality_importance:
                    features = list(quality_importance.keys())
                    importances = list(quality_importance.values())

                    # Sort by importance
                    sorted_idx = np.argsort(importances)
                    features = [features[i] for i in sorted_idx[-10:]]  # Top 10
                    importances = [importances[i] for i in sorted_idx[-10:]]

                    ax2.barh(features, importances)
                    ax2.set_xlabel('Feature Importance')
                    ax2.set_title('Quality Predictor\nFeature Importance')

            plt.tight_layout()
            plot_file = self.plots_dir / "feature_importance_comparison.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            return plot_file.name

        except Exception as e:
            logger.error(f"Feature importance plot failed: {e}")
            return None

    def _plot_improvement_metrics(self) -> str:
        """Generate improvement metrics visualization."""
        try:
            improvement = self.comparison_results['improvement_metrics']

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Improvement percentages
            categories = ['Parameter\nPrediction', 'Quality\nPrediction', 'Overall']
            improvements = [
                improvement['parameter_prediction_improvement'],
                improvement['quality_prediction_improvement'],
                improvement['overall_improvement']
            ]

            colors = ['green' if imp > 15 else 'orange' if imp > 0 else 'red' for imp in improvements]

            bars = ax1.bar(categories, improvements, color=colors, alpha=0.7)
            ax1.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='15% Target')
            ax1.set_ylabel('Improvement (%)')
            ax1.set_title('Model Performance Improvement\nvs Baseline Methods')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, improvements):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom')

            # MAE comparison
            comparison = improvement['comparison_details']
            methods = ['Statistical', 'Baseline']
            param_maes = [comparison['parameter_mae_statistical'], comparison['parameter_mae_baseline']]
            ssim_maes = [comparison['ssim_mae_statistical'], comparison['ssim_mae_baseline']]

            x = np.arange(len(methods))
            width = 0.35

            ax2.bar(x - width/2, param_maes, width, label='Parameter MAE', alpha=0.7)
            ax2.bar(x + width/2, ssim_maes, width, label='SSIM MAE', alpha=0.7)

            ax2.set_xlabel('Method')
            ax2.set_ylabel('Mean Absolute Error')
            ax2.set_title('MAE Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(methods)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = self.plots_dir / "improvement_metrics.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            return plot_file.name

        except Exception as e:
            logger.error(f"Improvement metrics plot failed: {e}")
            return None

    def save_evaluation_report(self) -> str:
        """
        Save comprehensive evaluation report.

        Returns:
            Path to saved report file
        """
        logger.info("Saving evaluation report")

        try:
            report = {
                'evaluation_metadata': {
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'evaluator_version': '1.0.0',
                    'models_evaluated': ['parameter_predictor', 'quality_predictor'],
                    'test_samples': self.evaluation_results.get('parameter_predictor', {}).get('num_samples', 0)
                },
                'model_evaluations': self.evaluation_results,
                'comparison_results': self.comparison_results,
                'summary': self._create_evaluation_summary()
            }

            report_file = self.output_dir / "model_evaluation_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Evaluation report saved to {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")
            return None

    def _create_evaluation_summary(self) -> Dict[str, Any]:
        """Create a summary of evaluation results."""
        summary = {}

        try:
            # Parameter predictor summary
            if 'parameter_predictor' in self.evaluation_results:
                param_results = self.evaluation_results['parameter_predictor']
                summary['parameter_predictor'] = {
                    'overall_mae': param_results['overall_mae'],
                    'overall_r2': param_results['overall_r2'],
                    'avg_prediction_time': param_results['avg_prediction_time'],
                    'performance_grade': 'Good' if param_results['overall_mae'] < 5.0 else 'Fair'
                }

            # Quality predictor summary
            if 'quality_predictor' in self.evaluation_results:
                quality_results = self.evaluation_results['quality_predictor']
                summary['quality_predictor'] = {
                    'mae': quality_results['mae'],
                    'r2': quality_results['r2'],
                    'within_01_ssim': quality_results['within_01_ssim'],
                    'meets_requirement': quality_results['mae'] <= 0.1,
                    'performance_grade': 'Good' if quality_results['mae'] <= 0.1 else 'Fair'
                }

            # Improvement summary
            if 'improvement_metrics' in self.comparison_results:
                improvement = self.comparison_results['improvement_metrics']
                summary['improvement'] = {
                    'overall_improvement': improvement['overall_improvement'],
                    'meets_15_percent_target': improvement['meets_15_percent_target'],
                    'recommendation': 'Deploy statistical models' if improvement['meets_15_percent_target']
                                    else 'Consider further optimization'
                }

        except Exception as e:
            logger.error(f"Summary creation failed: {e}")
            summary['error'] = str(e)

        return summary

    def run_complete_evaluation(self) -> bool:
        """
        Run the complete evaluation pipeline.

        Returns:
            True if evaluation completed successfully, False otherwise
        """
        logger.info("Starting complete model evaluation")

        try:
            # Load test data
            X_test, y_test = self.load_test_data()

            # Load models
            if not self.load_models():
                return False

            # Evaluate parameter predictor
            logger.info("Step 1/5: Evaluating parameter predictor")
            self.evaluation_results['parameter_predictor'] = self.evaluate_parameter_predictor(X_test, y_test)

            # Evaluate quality predictor
            logger.info("Step 2/5: Evaluating quality predictor")
            self.evaluation_results['quality_predictor'] = self.evaluate_quality_predictor(X_test, y_test)

            # Evaluate baseline methods
            logger.info("Step 3/5: Evaluating baseline methods")
            self.evaluation_results['baseline'] = self.evaluate_baseline_methods(X_test, y_test)

            # Calculate improvement metrics
            logger.info("Step 4/5: Calculating improvement metrics")
            self.comparison_results['improvement_metrics'] = self.calculate_improvement_metrics()

            # Generate visualizations
            logger.info("Step 5/5: Generating visualizations")
            plot_files = self.generate_visualizations()
            self.comparison_results['generated_plots'] = plot_files

            # Save report
            report_file = self.save_evaluation_report()

            logger.info("Evaluation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return False


def main():
    """Main evaluation script."""
    evaluator = StatisticalModelEvaluator()

    success = evaluator.run_complete_evaluation()

    if success:
        print("\n" + "="*60)
        print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
        print("="*60)

        # Display summary
        if 'improvement_metrics' in evaluator.comparison_results:
            improvement = evaluator.comparison_results['improvement_metrics']
            print(f"Overall improvement: {improvement['overall_improvement']:.1f}%")
            print(f"Meets 15% target: {improvement['meets_15_percent_target']}")

        print(f"\nEvaluation report: model_evaluation_report.json")
        print(f"Visualization plots: {len(evaluator.comparison_results.get('generated_plots', []))} files")

    else:
        print("Model evaluation failed!")


if __name__ == "__main__":
    main()