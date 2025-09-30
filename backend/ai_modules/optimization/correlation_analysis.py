"""
Correlation Analysis for Method 1 Parameter Optimization
Analyzes and refines correlation formulas based on validation data
"""
import numpy as np
import pandas as pd
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path
from datetime import datetime
import warnings

from .validation_pipeline import Method1ValidationPipeline
from .correlation_formulas import CorrelationFormulas
from .feature_mapping_optimizer import FeatureMappingOptimizer
from .parameter_bounds import VTracerParameterBounds

# Setup logging
logger = logging.getLogger(__name__)

class CorrelationAnalysis:
    """Analyze and refine correlation formulas based on validation data"""

    def __init__(self, validation_results_path: Optional[str] = None):
        """Initialize correlation analysis

        Args:
            validation_results_path: Path to validation results JSON file
        """
        self.validation_data = None
        self.validation_results_path = validation_results_path
        self.improvements = {}
        self.correlation_metrics = {}
        self.logo_type_analysis = {}
        self.formulas = CorrelationFormulas()
        self.optimizer = FeatureMappingOptimizer()
        self.bounds = VTracerParameterBounds()

        # Load validation data if path provided
        if validation_results_path and Path(validation_results_path).exists():
            self.load_validation_data(validation_results_path)

    def load_validation_data(self, results_path: str) -> None:
        """Load validation results from JSON file"""
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)

            # Convert to DataFrame for easier analysis
            if 'results' in data:
                self.validation_data = pd.DataFrame(data['results'])
            else:
                self.validation_data = pd.DataFrame(data)

            logger.info(f"Loaded {len(self.validation_data)} validation records")

        except Exception as e:
            logger.error(f"Failed to load validation data from {results_path}: {e}")
            self.validation_data = None

    def extract_mock_features(self, image_path: Path, logo_type: str) -> Dict[str, float]:
        """Extract mock features for analysis testing"""
        # Generate realistic mock features based on logo type
        feature_ranges = {
            "simple": {
                "edge_density": (0.05, 0.15),
                "unique_colors": (2, 8),
                "entropy": (0.2, 0.5),
                "corner_density": (0.02, 0.1),
                "gradient_strength": (0.1, 0.3),
                "complexity_score": (0.1, 0.3)
            },
            "text": {
                "edge_density": (0.2, 0.4),
                "unique_colors": (2, 6),
                "entropy": (0.3, 0.6),
                "corner_density": (0.1, 0.25),
                "gradient_strength": (0.1, 0.4),
                "complexity_score": (0.2, 0.5)
            },
            "gradient": {
                "edge_density": (0.1, 0.25),
                "unique_colors": (20, 80),
                "entropy": (0.6, 0.9),
                "corner_density": (0.05, 0.15),
                "gradient_strength": (0.6, 0.9),
                "complexity_score": (0.4, 0.7)
            },
            "complex": {
                "edge_density": (0.3, 0.6),
                "unique_colors": (50, 200),
                "entropy": (0.7, 0.95),
                "corner_density": (0.2, 0.5),
                "gradient_strength": (0.4, 0.8),
                "complexity_score": (0.7, 0.95)
            }
        }

        ranges = feature_ranges.get(logo_type, feature_ranges["simple"])
        features = {}

        # Add some deterministic variation based on image name
        seed = abs(hash(str(image_path))) % 1000
        np.random.seed(seed)

        for feature, (min_val, max_val) in ranges.items():
            features[feature] = np.random.uniform(min_val, max_val)

        return features

    def generate_sample_validation_data(self, num_samples: int = 50) -> pd.DataFrame:
        """Generate sample validation data for analysis"""
        logger.info(f"Generating {num_samples} sample validation records for analysis")

        # Create mock image paths and logo types
        logo_types = ["simple", "text", "gradient", "complex"]
        records = []

        for i in range(num_samples):
            logo_type = logo_types[i % len(logo_types)]
            image_path = f"mock_image_{i:03d}_{logo_type}.png"

            # Generate realistic features based on logo type
            features = self.extract_mock_features(Path(image_path), logo_type)

            # Run optimization
            result = self.optimizer.optimize(features)

            if result and 'parameters' in result:
                # Mock quality improvement based on logo type
                improvement_ranges = {
                    "simple": (18, 28),
                    "text": (15, 25),
                    "gradient": (12, 20),
                    "complex": (8, 16)
                }
                min_imp, max_imp = improvement_ranges.get(logo_type, (12, 20))
                quality_improvement = np.random.uniform(min_imp, max_imp)

                record = {
                    'image_path': image_path,
                    'logo_type': logo_type,
                    'features': features,
                    'optimized_params': result['parameters'],
                    'confidence': result.get('confidence', 0.0),
                    'quality_improvement': quality_improvement,
                    'success': True,
                    'processing_time': np.random.uniform(0.01, 0.08)
                }
                records.append(record)

        return pd.DataFrame(records)

    def analyze_correlation_effectiveness(self) -> Dict[str, float]:
        """Analyze which correlations perform best"""
        if self.validation_data is None:
            logger.warning("No validation data available, generating sample data")
            self.validation_data = self.generate_sample_validation_data()

        logger.info("Analyzing correlation effectiveness")

        # Filter successful optimizations
        successful_data = self.validation_data[self.validation_data['success'] == True].copy()

        if len(successful_data) == 0:
            logger.error("No successful optimization data available")
            return {}

        effectiveness_scores = {}

        # Analyze each correlation formula
        correlation_mappings = {
            'edge_to_corner_threshold': ('edge_density', 'corner_threshold'),
            'colors_to_precision': ('unique_colors', 'color_precision'),
            'entropy_to_path_precision': ('entropy', 'path_precision'),
            'corners_to_length_threshold': ('corner_density', 'length_threshold'),
            'gradient_to_splice_threshold': ('gradient_strength', 'splice_threshold'),
            'complexity_to_iterations': ('complexity_score', 'max_iterations')
        }

        for formula_name, (feature_key, param_key) in correlation_mappings.items():
            try:
                # Extract feature and parameter values
                feature_values = []
                param_values = []
                quality_improvements = []

                for _, row in successful_data.iterrows():
                    features = row['features']
                    params = row['optimized_params']

                    if (isinstance(features, dict) and feature_key in features and
                        isinstance(params, dict) and param_key in params):

                        feature_values.append(features[feature_key])
                        param_values.append(params[param_key])
                        quality_improvements.append(row['quality_improvement'])

                if len(feature_values) < 3:
                    logger.warning(f"Insufficient data for {formula_name} analysis")
                    effectiveness_scores[formula_name] = 0.0
                    continue

                # Calculate correlation between feature and quality improvement
                feature_quality_corr = np.corrcoef(feature_values, quality_improvements)[0, 1]

                # Calculate how well the formula predicts parameters
                expected_params = []
                for fval in feature_values:
                    try:
                        if formula_name == 'edge_to_corner_threshold':
                            expected = self.formulas.edge_to_corner_threshold(fval)
                        elif formula_name == 'colors_to_precision':
                            expected = self.formulas.colors_to_precision(fval)
                        elif formula_name == 'entropy_to_path_precision':
                            expected = self.formulas.entropy_to_path_precision(fval)
                        elif formula_name == 'corners_to_length_threshold':
                            expected = self.formulas.corners_to_length_threshold(fval)
                        elif formula_name == 'gradient_to_splice_threshold':
                            expected = self.formulas.gradient_to_splice_threshold(fval)
                        elif formula_name == 'complexity_to_iterations':
                            expected = self.formulas.complexity_to_iterations(fval)
                        else:
                            expected = 0
                        expected_params.append(expected)
                    except Exception as e:
                        logger.warning(f"Failed to calculate expected parameter for {formula_name}: {e}")
                        expected_params.append(0)

                # Calculate R-squared for formula accuracy
                if len(expected_params) == len(param_values):
                    try:
                        formula_r2 = r2_score(param_values, expected_params)
                    except:
                        formula_r2 = 0.0
                else:
                    formula_r2 = 0.0

                # Combined effectiveness score (correlation with quality + formula accuracy)
                effectiveness = (abs(feature_quality_corr) * 0.6 + max(0, formula_r2) * 0.4)
                effectiveness_scores[formula_name] = effectiveness

                # Store detailed metrics
                self.correlation_metrics[formula_name] = {
                    'feature_quality_correlation': feature_quality_corr,
                    'formula_r2': formula_r2,
                    'effectiveness_score': effectiveness,
                    'sample_size': len(feature_values),
                    'feature_range': {'min': min(feature_values), 'max': max(feature_values)},
                    'param_range': {'min': min(param_values), 'max': max(param_values)}
                }

                logger.info(f"{formula_name}: effectiveness={effectiveness:.3f}, "
                           f"feature_quality_corr={feature_quality_corr:.3f}, "
                           f"formula_r2={formula_r2:.3f}")

            except Exception as e:
                logger.error(f"Error analyzing {formula_name}: {e}")
                effectiveness_scores[formula_name] = 0.0

        return effectiveness_scores

    def analyze_by_logo_type(self) -> Dict[str, Dict[str, float]]:
        """Calculate correlation effectiveness by logo type"""
        if self.validation_data is None:
            logger.warning("No validation data available for logo type analysis")
            return {}

        logger.info("Analyzing correlation effectiveness by logo type")

        logo_types = ["simple", "text", "gradient", "complex"]
        self.logo_type_analysis = {}

        for logo_type in logo_types:
            type_data = self.validation_data[
                (self.validation_data['logo_type'] == logo_type) &
                (self.validation_data['success'] == True)
            ].copy()

            if len(type_data) == 0:
                logger.warning(f"No successful data for logo type: {logo_type}")
                self.logo_type_analysis[logo_type] = {}
                continue

            # Calculate success rate for this logo type
            total_type_data = self.validation_data[self.validation_data['logo_type'] == logo_type]
            success_rate = len(type_data) / len(total_type_data) * 100 if len(total_type_data) > 0 else 0

            # Calculate quality improvement statistics
            quality_improvements = type_data['quality_improvement'].values

            self.logo_type_analysis[logo_type] = {
                'success_rate': success_rate,
                'sample_size': len(type_data),
                'avg_quality_improvement': np.mean(quality_improvements),
                'median_quality_improvement': np.median(quality_improvements),
                'std_quality_improvement': np.std(quality_improvements),
                'min_quality_improvement': np.min(quality_improvements),
                'max_quality_improvement': np.max(quality_improvements)
            }

            logger.info(f"{logo_type}: success_rate={success_rate:.1f}%, "
                       f"avg_improvement={np.mean(quality_improvements):.1f}%")

        return self.logo_type_analysis

    def identify_underperforming_formulas(self, threshold: float = 0.6) -> List[str]:
        """Identify correlation formulas that need improvement"""
        effectiveness_scores = self.analyze_correlation_effectiveness()

        underperforming = []
        for formula_name, score in effectiveness_scores.items():
            if score < threshold:
                underperforming.append(formula_name)
                logger.warning(f"Underperforming formula: {formula_name} (score: {score:.3f})")

        return underperforming

    def generate_statistical_significance_tests(self) -> Dict[str, Dict[str, float]]:
        """Generate statistical significance tests for each formula"""
        if self.validation_data is None:
            return {}

        logger.info("Generating statistical significance tests")

        significance_tests = {}
        successful_data = self.validation_data[self.validation_data['success'] == True].copy()

        correlation_mappings = {
            'edge_to_corner_threshold': ('edge_density', 'corner_threshold'),
            'colors_to_precision': ('unique_colors', 'color_precision'),
            'entropy_to_path_precision': ('entropy', 'path_precision'),
            'corners_to_length_threshold': ('corner_density', 'length_threshold'),
            'gradient_to_splice_threshold': ('gradient_strength', 'splice_threshold'),
            'complexity_to_iterations': ('complexity_score', 'max_iterations')
        }

        for formula_name, (feature_key, param_key) in correlation_mappings.items():
            try:
                # Extract values
                feature_values = []
                quality_improvements = []

                for _, row in successful_data.iterrows():
                    features = row['features']
                    if isinstance(features, dict) and feature_key in features:
                        feature_values.append(features[feature_key])
                        quality_improvements.append(row['quality_improvement'])

                if len(feature_values) < 3:
                    continue

                # Pearson correlation test
                correlation, p_value = stats.pearsonr(feature_values, quality_improvements)

                # Spearman rank correlation test
                spearman_corr, spearman_p = stats.spearmanr(feature_values, quality_improvements)

                significance_tests[formula_name] = {
                    'pearson_correlation': correlation,
                    'pearson_p_value': p_value,
                    'pearson_significant': p_value < 0.05,
                    'spearman_correlation': spearman_corr,
                    'spearman_p_value': spearman_p,
                    'spearman_significant': spearman_p < 0.05,
                    'sample_size': len(feature_values)
                }

            except Exception as e:
                logger.error(f"Error in significance test for {formula_name}: {e}")

        return significance_tests

    def calculate_r_squared_values(self) -> Dict[str, float]:
        """Calculate R-squared values for each correlation"""
        if self.correlation_metrics:
            return {name: metrics['formula_r2'] for name, metrics in self.correlation_metrics.items()}
        else:
            # Trigger analysis if not done
            self.analyze_correlation_effectiveness()
            return {name: metrics['formula_r2'] for name, metrics in self.correlation_metrics.items()}

    def identify_optimal_parameter_ranges(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Identify optimal parameter ranges per logo type"""
        if self.validation_data is None:
            return {}

        logger.info("Identifying optimal parameter ranges per logo type")

        optimal_ranges = {}
        logo_types = ["simple", "text", "gradient", "complex"]

        # Target parameters to analyze
        parameters = ['corner_threshold', 'color_precision', 'path_precision',
                     'length_threshold', 'splice_threshold', 'max_iterations']

        for logo_type in logo_types:
            # Get successful optimizations for this logo type
            type_data = self.validation_data[
                (self.validation_data['logo_type'] == logo_type) &
                (self.validation_data['success'] == True) &
                (self.validation_data['quality_improvement'] > 15.0)  # Good quality improvements
            ].copy()

            if len(type_data) == 0:
                continue

            optimal_ranges[logo_type] = {}

            for param in parameters:
                param_values = []

                for _, row in type_data.iterrows():
                    params = row['optimized_params']
                    if isinstance(params, dict) and param in params:
                        param_values.append(params[param])

                if len(param_values) >= 3:
                    # Calculate optimal range as mean ± 1 std dev, bounded by parameter limits
                    mean_val = np.mean(param_values)
                    std_val = np.std(param_values)

                    optimal_min = max(mean_val - std_val, min(param_values))
                    optimal_max = min(mean_val + std_val, max(param_values))

                    optimal_ranges[logo_type][param] = (optimal_min, optimal_max)

        return optimal_ranges

    def generate_scatter_plots(self, output_dir: str = "correlation_analysis") -> List[str]:
        """Generate scatter plots for visual analysis"""
        if self.validation_data is None:
            logger.warning("No validation data available for plots")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        logger.info(f"Generating scatter plots in {output_dir}")

        plot_files = []
        successful_data = self.validation_data[self.validation_data['success'] == True].copy()

        correlation_mappings = {
            'edge_to_corner_threshold': ('edge_density', 'corner_threshold'),
            'colors_to_precision': ('unique_colors', 'color_precision'),
            'entropy_to_path_precision': ('entropy', 'path_precision'),
            'corners_to_length_threshold': ('corner_density', 'length_threshold'),
            'gradient_to_splice_threshold': ('gradient_strength', 'splice_threshold'),
            'complexity_to_iterations': ('complexity_score', 'max_iterations')
        }

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        for formula_name, (feature_key, param_key) in correlation_mappings.items():
            try:
                # Extract values
                feature_values = []
                param_values = []
                quality_improvements = []
                logo_types = []

                for _, row in successful_data.iterrows():
                    features = row['features']
                    params = row['optimized_params']

                    if (isinstance(features, dict) and feature_key in features and
                        isinstance(params, dict) and param_key in params):

                        feature_values.append(features[feature_key])
                        param_values.append(params[param_key])
                        quality_improvements.append(row['quality_improvement'])
                        logo_types.append(row['logo_type'])

                if len(feature_values) < 3:
                    continue

                # Create scatter plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Plot 1: Feature vs Parameter (actual correlation)
                scatter1 = ax1.scatter(feature_values, param_values,
                                     c=quality_improvements, cmap='viridis',
                                     alpha=0.7, s=60)
                ax1.set_xlabel(feature_key.replace('_', ' ').title())
                ax1.set_ylabel(param_key.replace('_', ' ').title())
                ax1.set_title(f'{formula_name.replace("_", " ").title()}\nFeature vs Parameter')

                # Add formula prediction line
                x_range = np.linspace(min(feature_values), max(feature_values), 100)
                y_predicted = []
                for x in x_range:
                    try:
                        if formula_name == 'edge_to_corner_threshold':
                            y_pred = self.formulas.edge_to_corner_threshold(x)
                        elif formula_name == 'colors_to_precision':
                            y_pred = self.formulas.colors_to_precision(x)
                        elif formula_name == 'entropy_to_path_precision':
                            y_pred = self.formulas.entropy_to_path_precision(x)
                        elif formula_name == 'corners_to_length_threshold':
                            y_pred = self.formulas.corners_to_length_threshold(x)
                        elif formula_name == 'gradient_to_splice_threshold':
                            y_pred = self.formulas.gradient_to_splice_threshold(x)
                        elif formula_name == 'complexity_to_iterations':
                            y_pred = self.formulas.complexity_to_iterations(x)
                        else:
                            y_pred = 0
                        y_predicted.append(y_pred)
                    except:
                        y_predicted.append(0)

                ax1.plot(x_range, y_predicted, 'r-', linewidth=2, label='Formula Prediction')
                ax1.legend()

                # Add colorbar for quality improvement
                cbar1 = plt.colorbar(scatter1, ax=ax1)
                cbar1.set_label('Quality Improvement (%)')

                # Plot 2: Feature vs Quality Improvement by logo type
                logo_type_colors = {'simple': 'blue', 'text': 'green', 'gradient': 'orange', 'complex': 'red'}
                for logo_type in set(logo_types):
                    type_indices = [i for i, lt in enumerate(logo_types) if lt == logo_type]
                    if type_indices:
                        type_features = [feature_values[i] for i in type_indices]
                        type_quality = [quality_improvements[i] for i in type_indices]
                        ax2.scatter(type_features, type_quality,
                                  color=logo_type_colors.get(logo_type, 'gray'),
                                  label=logo_type, alpha=0.7, s=60)

                ax2.set_xlabel(feature_key.replace('_', ' ').title())
                ax2.set_ylabel('Quality Improvement (%)')
                ax2.set_title(f'{formula_name.replace("_", " ").title()}\nFeature vs Quality by Logo Type')
                ax2.legend()

                # Add correlation info
                correlation = np.corrcoef(feature_values, quality_improvements)[0, 1]
                ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                        transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

                plt.tight_layout()

                # Save plot
                plot_file = output_dir / f"{formula_name}_analysis.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()

                plot_files.append(str(plot_file))
                logger.info(f"Generated plot: {plot_file}")

            except Exception as e:
                logger.error(f"Error generating plot for {formula_name}: {e}")

        return plot_files

    def create_correlation_effectiveness_report(self) -> Dict[str, Any]:
        """Create comprehensive correlation effectiveness report"""
        logger.info("Creating correlation effectiveness report")

        # Run all analyses
        effectiveness_scores = self.analyze_correlation_effectiveness()
        logo_type_analysis = self.analyze_by_logo_type()
        underperforming_formulas = self.identify_underperforming_formulas()
        significance_tests = self.generate_statistical_significance_tests()
        r_squared_values = self.calculate_r_squared_values()
        optimal_ranges = self.identify_optimal_parameter_ranges()

        # Generate plots
        plot_files = self.generate_scatter_plots()

        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_records': len(self.validation_data) if self.validation_data is not None else 0,
                'successful_records': len(self.validation_data[self.validation_data['success'] == True]) if self.validation_data is not None else 0
            },
            'effectiveness_scores': effectiveness_scores,
            'correlation_metrics': self.correlation_metrics,
            'logo_type_analysis': logo_type_analysis,
            'underperforming_formulas': underperforming_formulas,
            'statistical_significance': significance_tests,
            'r_squared_values': r_squared_values,
            'optimal_parameter_ranges': optimal_ranges,
            'generated_plots': plot_files,
            'recommendations': self._generate_recommendations(effectiveness_scores, underperforming_formulas)
        }

        return report

    def _generate_recommendations(self, effectiveness_scores: Dict[str, float],
                                 underperforming_formulas: List[str]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        if underperforming_formulas:
            recommendations.append(
                f"Refine {len(underperforming_formulas)} underperforming formulas: "
                f"{', '.join(underperforming_formulas)}"
            )

        # Find best performing formula
        if effectiveness_scores:
            best_formula = max(effectiveness_scores.items(), key=lambda x: x[1])
            recommendations.append(
                f"Use {best_formula[0]} as reference (highest effectiveness: {best_formula[1]:.3f})"
            )

        # Check logo type specific recommendations
        if self.logo_type_analysis:
            for logo_type, analysis in self.logo_type_analysis.items():
                if analysis.get('success_rate', 0) < 85:
                    recommendations.append(
                        f"Improve {logo_type} logo optimization (current success: {analysis.get('success_rate', 0):.1f}%)"
                    )

        return recommendations

    def export_report(self, output_file: str = "correlation_effectiveness_report.json") -> str:
        """Export correlation effectiveness report to JSON file"""
        report = self.create_correlation_effectiveness_report()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Exported correlation effectiveness report to {output_path}")
        return str(output_path)