"""
A/B Testing Framework - Task 5 Implementation
Framework for comparing different conversion methods and measuring improvements.
"""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import scipy.stats as stats
from enum import Enum

# Import the conversion and quality modules
try:
    from backend.ai_modules.quality.enhanced_metrics import EnhancedQualityMetrics
    from backend.converters.vtracer_converter import VTracerConverter
except ImportError:
    # Fallback for testing
    pass


class ConversionMethod(Enum):
    """Types of conversion methods for A/B testing."""
    BASELINE = "baseline"
    AI_ENHANCED = "ai_enhanced"
    OPTIMIZED = "optimized"
    EXPERIMENTAL = "experimental"


@dataclass
class ConversionResult:
    """Result of a single conversion for A/B testing."""
    method: ConversionMethod
    image_path: str
    output_path: str
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    processing_time: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of comparing two methods."""
    method_a: ConversionMethod
    method_b: ConversionMethod
    sample_size: int
    improvement_metrics: Dict[str, float]
    statistical_significance: Dict[str, Any]
    summary: Dict[str, Any]
    timestamp: datetime


class ABTester:
    """A/B testing framework for conversion methods."""

    def __init__(self, output_dir: str = "data/ab_testing"):
        """
        Initialize A/B tester.

        Args:
            output_dir: Directory for storing test results and outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize quality metrics
        self.quality_metrics = EnhancedQualityMetrics()

        # Default parameters for different methods
        self.method_parameters = {
            ConversionMethod.BASELINE: {
                'color_precision': 4,
                'corner_threshold': 30,
                'path_precision': 8,
                'splice_threshold': 45
            },
            ConversionMethod.AI_ENHANCED: {
                'color_precision': 6,
                'corner_threshold': 20,
                'path_precision': 10,
                'splice_threshold': 35
            },
            ConversionMethod.OPTIMIZED: {
                'color_precision': 8,
                'corner_threshold': 15,
                'path_precision': 12,
                'splice_threshold': 25
            }
        }

        self.test_results: List[ConversionResult] = []

    def compare_methods(self,
                       image_path: str,
                       method_a: ConversionMethod = ConversionMethod.BASELINE,
                       method_b: ConversionMethod = ConversionMethod.AI_ENHANCED) -> ComparisonResult:
        """
        Compare two conversion methods on a single image.

        Args:
            image_path: Path to input image
            method_a: First method to compare
            method_b: Second method to compare

        Returns:
            ComparisonResult: Comparison results with metrics and significance
        """
        # Convert with both methods
        result_a = self._convert_with_method(image_path, method_a)
        result_b = self._convert_with_method(image_path, method_b)

        # Store results
        self.test_results.extend([result_a, result_b])

        # Calculate improvements and significance
        improvement_metrics = self._calculate_improvement(result_a, result_b)
        significance = self._test_statistical_significance([result_a], [result_b])

        # Create summary
        summary = {
            'winner': method_b.value if improvement_metrics.get('composite_score', 0) > 0 else method_a.value,
            'significant_improvement': significance.get('composite_score', {}).get('significant', False),
            'key_improvements': {k: v for k, v in improvement_metrics.items() if abs(v) > 0.05}
        }

        return ComparisonResult(
            method_a=method_a,
            method_b=method_b,
            sample_size=1,
            improvement_metrics=improvement_metrics,
            statistical_significance=significance,
            summary=summary,
            timestamp=datetime.now()
        )

    def batch_compare_methods(self,
                            image_paths: List[str],
                            method_a: ConversionMethod = ConversionMethod.BASELINE,
                            method_b: ConversionMethod = ConversionMethod.AI_ENHANCED) -> ComparisonResult:
        """
        Compare two methods across multiple images for statistical significance.

        Args:
            image_paths: List of image paths to test
            method_a: First method to compare
            method_b: Second method to compare

        Returns:
            ComparisonResult: Aggregated comparison results
        """
        results_a = []
        results_b = []

        print(f"Running batch A/B test: {method_a.value} vs {method_b.value}")
        print(f"Testing {len(image_paths)} images...")

        for i, image_path in enumerate(image_paths):
            print(f"  Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")

            result_a = self._convert_with_method(image_path, method_a)
            result_b = self._convert_with_method(image_path, method_b)

            results_a.append(result_a)
            results_b.append(result_b)

            # Store results
            self.test_results.extend([result_a, result_b])

        # Calculate aggregate improvements and significance
        improvement_metrics = self._calculate_batch_improvement(results_a, results_b)
        significance = self._test_statistical_significance(results_a, results_b)

        # Determine winner and significant improvements
        winner = method_b.value if improvement_metrics.get('composite_score', 0) > 0 else method_a.value
        significant_improvements = {}

        for metric, improvement in improvement_metrics.items():
            if metric in significance and significance[metric].get('significant', False):
                significant_improvements[metric] = improvement

        summary = {
            'winner': winner,
            'significant_improvements': significant_improvements,
            'total_improvements': {k: v for k, v in improvement_metrics.items() if abs(v) > 0.01},
            'confidence_level': 0.95
        }

        return ComparisonResult(
            method_a=method_a,
            method_b=method_b,
            sample_size=len(image_paths),
            improvement_metrics=improvement_metrics,
            statistical_significance=significance,
            summary=summary,
            timestamp=datetime.now()
        )

    def _convert_with_method(self, image_path: str, method: ConversionMethod) -> ConversionResult:
        """Convert image using specified method."""
        start_time = datetime.now()

        try:
            # Get parameters for method
            parameters = self.method_parameters.get(method, self.method_parameters[ConversionMethod.BASELINE])

            # Create output path
            image_name = Path(image_path).stem
            output_path = self.output_dir / f"{image_name}_{method.value}.svg"

            # Simulate conversion (replace with actual converter)
            processing_time, success = self._simulate_conversion(image_path, str(output_path), parameters)

            if success:
                # Calculate quality metrics
                metrics = self.quality_metrics.calculate_metrics(image_path, str(output_path))
            else:
                metrics = {'composite_score': 0.0, 'error': 'conversion_failed'}

            return ConversionResult(
                method=method,
                image_path=image_path,
                output_path=str(output_path),
                metrics=metrics,
                parameters=parameters,
                processing_time=processing_time,
                timestamp=start_time,
                success=success
            )

        except Exception as e:
            return ConversionResult(
                method=method,
                image_path=image_path,
                output_path="",
                metrics={'composite_score': 0.0},
                parameters={},
                processing_time=0.0,
                timestamp=start_time,
                success=False,
                error_message=str(e)
            )

    def _simulate_conversion(self, input_path: str, output_path: str, parameters: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Simulate conversion for testing purposes.
        In production, this would call the actual converter.
        """
        import time
        import random

        # Simulate processing time based on method complexity
        base_time = 0.5
        complexity_factor = parameters.get('color_precision', 4) / 4.0
        processing_time = base_time * complexity_factor + random.uniform(-0.1, 0.1)

        # Simulate success rate (98% for baseline, 95% for enhanced methods)
        success_rate = 0.98 if parameters.get('color_precision', 4) <= 4 else 0.95
        success = random.random() < success_rate

        if success:
            # Create a minimal SVG file for testing
            svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="256" height="256" xmlns="http://www.w3.org/2000/svg">
  <circle cx="128" cy="128" r="100" fill="blue" stroke="black" stroke-width="2"/>
  <!-- Generated with method: {parameters} -->
</svg>'''
            Path(output_path).write_text(svg_content)

        time.sleep(min(processing_time, 0.1))  # Cap simulation time
        return processing_time, success

    def _calculate_improvement(self, result_a: ConversionResult, result_b: ConversionResult) -> Dict[str, float]:
        """Calculate improvement percentages between two results."""
        improvements = {}

        for metric_name in result_a.metrics:
            if metric_name in result_b.metrics and metric_name != 'error':
                value_a = result_a.metrics[metric_name]
                value_b = result_b.metrics[metric_name]

                if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                    if value_a != 0:
                        improvement = ((value_b - value_a) / abs(value_a)) * 100
                        improvements[metric_name] = improvement

        return improvements

    def _calculate_batch_improvement(self,
                                   results_a: List[ConversionResult],
                                   results_b: List[ConversionResult]) -> Dict[str, float]:
        """Calculate average improvement across batch results."""
        if not results_a or not results_b:
            return {}

        # Get all metric names
        all_metrics = set()
        for result in results_a + results_b:
            all_metrics.update(result.metrics.keys())

        improvements = {}

        for metric_name in all_metrics:
            if metric_name == 'error':
                continue

            values_a = [r.metrics.get(metric_name, 0) for r in results_a if metric_name in r.metrics]
            values_b = [r.metrics.get(metric_name, 0) for r in results_b if metric_name in r.metrics]

            if values_a and values_b:
                avg_a = statistics.mean(values_a)
                avg_b = statistics.mean(values_b)

                if avg_a != 0:
                    improvement = ((avg_b - avg_a) / abs(avg_a)) * 100
                    improvements[metric_name] = improvement

        return improvements

    def _test_statistical_significance(self,
                                     results_a: List[ConversionResult],
                                     results_b: List[ConversionResult],
                                     alpha: float = 0.05) -> Dict[str, Any]:
        """Test statistical significance of differences between methods."""
        significance_results = {}

        # Get all metric names
        all_metrics = set()
        for result in results_a + results_b:
            all_metrics.update(result.metrics.keys())

        for metric_name in all_metrics:
            if metric_name == 'error':
                continue

            values_a = [r.metrics.get(metric_name, 0) for r in results_a if metric_name in r.metrics]
            values_b = [r.metrics.get(metric_name, 0) for r in results_b if metric_name in r.metrics]

            if len(values_a) >= 2 and len(values_b) >= 2:
                try:
                    # Perform two-sample t-test
                    t_stat, p_value = stats.ttest_ind(values_a, values_b)

                    significance_results[metric_name] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < alpha,
                        'confidence_level': (1 - alpha) * 100,
                        'sample_size_a': len(values_a),
                        'sample_size_b': len(values_b),
                        'mean_a': statistics.mean(values_a),
                        'mean_b': statistics.mean(values_b),
                        'effect_size': abs(statistics.mean(values_b) - statistics.mean(values_a))
                    }
                except Exception as e:
                    significance_results[metric_name] = {
                        'error': str(e),
                        'significant': False
                    }

        return significance_results

    def generate_comparison_report(self, comparison: ComparisonResult, output_path: str = None) -> str:
        """Generate detailed comparison report."""
        if output_path is None:
            timestamp = comparison.timestamp.strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"ab_test_report_{timestamp}.html"

        # Generate HTML report
        html_content = self._generate_html_report(comparison)

        Path(output_path).write_text(html_content)
        return str(output_path)

    def _generate_html_report(self, comparison: ComparisonResult) -> str:
        """Generate HTML report content."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>A/B Testing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
        .improvement-positive {{ background-color: #d4edda; }}
        .improvement-negative {{ background-color: #f8d7da; }}
        .significant {{ font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>A/B Testing Report</h1>
        <p><strong>Methods:</strong> {comparison.method_a.value} vs {comparison.method_b.value}</p>
        <p><strong>Sample Size:</strong> {comparison.sample_size}</p>
        <p><strong>Test Date:</strong> {comparison.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Winner:</strong> {comparison.summary.get('winner', 'Inconclusive')}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>This A/B test compared {comparison.method_a.value} against {comparison.method_b.value}
           across {comparison.sample_size} test cases.</p>

        <h3>Key Findings:</h3>
        <ul>
"""

        # Add key findings
        for metric, improvement in comparison.summary.get('significant_improvements', {}).items():
            direction = "improved" if improvement > 0 else "decreased"
            html += f"<li><strong>{metric}</strong> {direction} by {abs(improvement):.1f}% (statistically significant)</li>"

        html += """
        </ul>
    </div>

    <div class="section">
        <h2>Improvement Metrics</h2>
        <div>
"""

        # Add improvement metrics
        for metric, improvement in comparison.improvement_metrics.items():
            css_class = "improvement-positive" if improvement > 0 else "improvement-negative"
            significant = comparison.statistical_significance.get(metric, {}).get('significant', False)
            sig_class = "significant" if significant else ""

            html += f"""
            <div class="metric {css_class}">
                <h4 class="{sig_class}">{metric}</h4>
                <p>Improvement: {improvement:.2f}%</p>
                {'<p><strong>Statistically Significant</strong></p>' if significant else ''}
            </div>
"""

        html += """
        </div>
    </div>

    <div class="section">
        <h2>Statistical Analysis</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>P-Value</th>
                <th>Significant</th>
                <th>Effect Size</th>
                <th>Sample Size</th>
            </tr>
"""

        # Add statistical analysis table
        for metric, stats_data in comparison.statistical_significance.items():
            if 'error' not in stats_data:
                html += f"""
            <tr>
                <td>{metric}</td>
                <td>{stats_data.get('p_value', 'N/A'):.4f}</td>
                <td>{'Yes' if stats_data.get('significant', False) else 'No'}</td>
                <td>{stats_data.get('effect_size', 0):.4f}</td>
                <td>{stats_data.get('sample_size_a', 0)} vs {stats_data.get('sample_size_b', 0)}</td>
            </tr>
"""

        html += """
        </table>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ul>
"""

        # Add recommendations
        winner = comparison.summary.get('winner')
        significant_improvements = comparison.summary.get('significant_improvements', {})

        if significant_improvements:
            html += f"<li><strong>Recommended:</strong> Deploy {winner} method based on significant improvements in {len(significant_improvements)} metrics</li>"
        else:
            html += "<li><strong>Inconclusive:</strong> No statistically significant improvements detected. Consider larger sample size or method refinements.</li>"

        html += """
        </ul>
    </div>
</body>
</html>
"""

        return html

    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all A/B tests performed."""
        if not self.test_results:
            return {'total_tests': 0, 'methods_tested': [], 'summary': 'No tests performed'}

        methods = set(result.method for result in self.test_results)
        success_rate = len([r for r in self.test_results if r.success]) / len(self.test_results)

        # Calculate average metrics by method
        method_averages = {}
        for method in methods:
            method_results = [r for r in self.test_results if r.method == method and r.success]
            if method_results:
                avg_score = statistics.mean([r.metrics.get('composite_score', 0) for r in method_results])
                avg_time = statistics.mean([r.processing_time for r in method_results])
                method_averages[method.value] = {
                    'avg_quality': avg_score,
                    'avg_processing_time': avg_time,
                    'sample_size': len(method_results)
                }

        return {
            'total_tests': len(self.test_results),
            'methods_tested': [m.value for m in methods],
            'success_rate': success_rate,
            'method_averages': method_averages,
            'last_test': max(r.timestamp for r in self.test_results).isoformat() if self.test_results else None
        }


def create_sample_test() -> ABTester:
    """Create sample A/B test for demonstration."""
    tester = ABTester()

    # Test with sample image
    test_image = "test_circle.svg"  # This should be a real image path

    # Compare baseline vs AI enhanced
    comparison = tester.compare_methods(
        test_image,
        ConversionMethod.BASELINE,
        ConversionMethod.AI_ENHANCED
    )

    return tester, comparison


if __name__ == "__main__":
    # Test the A/B testing framework
    print("Testing A/B Testing Framework...")

    tester = ABTester("data/test_ab_testing")

    # Create sample test images list (in production, use real images)
    sample_images = ["test_image_1.png", "test_image_2.png", "test_image_3.png"]

    # Run batch comparison
    print("\\nRunning batch comparison...")
    comparison = tester.batch_compare_methods(
        sample_images,
        ConversionMethod.BASELINE,
        ConversionMethod.AI_ENHANCED
    )

    print(f"✓ Batch test completed:")
    print(f"  Winner: {comparison.summary['winner']}")
    print(f"  Sample size: {comparison.sample_size}")
    print(f"  Significant improvements: {len(comparison.summary.get('significant_improvements', {}))}")

    # Generate report
    report_path = tester.generate_comparison_report(comparison)
    print(f"✓ Report generated: {report_path}")

    # Get test summary
    summary = tester.get_test_summary()
    print(f"✓ Test summary: {summary['total_tests']} tests performed")

    print("\\nA/B Testing Framework ready!")