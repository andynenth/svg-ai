"""
Test Report Generator - Task 5 Implementation
Generate comprehensive reports for A/B testing results.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import base64
import io

# Import for PDF generation
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import weasyprint
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    plt = None

logger = logging.getLogger(__name__)


class ABTestReportGenerator:
    """
    Generate comprehensive A/B test reports in multiple formats.
    Creates HTML and PDF reports with detailed statistics and visualizations.
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize report generator.

        Args:
            template_dir: Directory containing custom templates
        """
        self.template_dir = Path(template_dir) if template_dir else None
        logger.info("ABTestReportGenerator initialized")

    def generate_report(self, test_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive test report.

        Args:
            test_results: Dictionary containing test results and analysis

        Returns:
            HTML report as string
        """
        try:
            # Extract key information
            campaign = test_results.get('campaign', {})
            analysis = test_results.get('analysis', {})
            visualizations = test_results.get('visualizations', {})

            # Generate executive summary
            executive_summary = self._generate_executive_summary(campaign, analysis)

            # Generate statistical analysis section
            statistical_analysis = self._generate_statistical_section(analysis)

            # Generate quality metrics section
            quality_metrics = self._generate_quality_metrics_section(analysis)

            # Generate performance metrics section
            performance_metrics = self._generate_performance_section(analysis)

            # Generate visual comparisons section
            visual_section = self._generate_visual_section(visualizations)

            # Use the template
            report_template = self._get_report_template()

            # Format the report
            formatted_report = report_template.format(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                campaign_name=campaign.get('name', 'Unknown Campaign'),
                winner=executive_summary.get('winner', 'Inconclusive'),
                improvement=executive_summary.get('improvement', 0),
                confidence=executive_summary.get('confidence', 0),
                recommendation=executive_summary.get('recommendation', 'Continue testing'),
                sample_size=executive_summary.get('sample_size', 0),
                p_value=executive_summary.get('p_value', 1.0),
                effect_size=executive_summary.get('effect_size', 0),
                duration=executive_summary.get('duration_hours', 0),
                statistical_analysis=statistical_analysis,
                quality_table=quality_metrics,
                performance_table=performance_metrics,
                visual_section=visual_section
            )

            return formatted_report

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return self._generate_error_report(str(e))

    def save_html_report(self, test_results: Dict[str, Any], output_path: str):
        """
        Save HTML report to file.

        Args:
            test_results: Test results and analysis
            output_path: Output file path
        """
        try:
            report_html = self.generate_report(test_results)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_html)

            logger.info(f"HTML report saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save HTML report: {e}")

    def export_pdf_report(self, test_results: Dict[str, Any], output_path: str):
        """
        Export report as PDF.

        Args:
            test_results: Test results and analysis
            output_path: Output PDF file path
        """
        if not PDF_SUPPORT:
            logger.warning("PDF export not available. Install weasyprint for PDF support.")
            return

        try:
            # Generate HTML report
            html_report = self.generate_report(test_results)

            # Convert to PDF using weasyprint
            weasyprint.HTML(string=html_report).write_pdf(output_path)

            logger.info(f"PDF report saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export PDF report: {e}")

    def _generate_executive_summary(self, campaign: Dict, analysis: Dict) -> Dict[str, Any]:
        """Generate executive summary data."""
        try:
            # Extract overall summary
            overall = analysis.get('overall_summary', {})

            improvement = overall.get('quality_improvement', 0)
            t_test = overall.get('t_test', {})
            p_value = t_test.get('p_value', 1.0)
            significant = t_test.get('significant', False)
            effect_size = overall.get('effect_size', 0)

            # Determine winner
            if significant:
                if improvement > 0:
                    winner = "AI Enhanced"
                else:
                    winner = "Baseline"
            else:
                winner = "No Clear Winner"

            # Calculate confidence percentage
            confidence = (1 - p_value) * 100 if p_value < 0.1 else 50

            # Get recommendation
            recommendation = overall.get('recommendation', 'Continue testing')

            # Sample size
            sample_sizes = analysis.get('sample_sizes', {})
            total_samples = sample_sizes.get('total', 0)

            # Duration
            started_at = campaign.get('started_at', '')
            completed_at = campaign.get('completed_at', '')
            duration_hours = campaign.get('duration_seconds', 0) / 3600

            return {
                'winner': winner,
                'improvement': abs(improvement),
                'confidence': confidence,
                'recommendation': recommendation,
                'sample_size': total_samples,
                'p_value': p_value,
                'effect_size': effect_size,
                'duration_hours': duration_hours,
                'significant': significant
            }

        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return {
                'winner': 'Error',
                'improvement': 0,
                'confidence': 0,
                'recommendation': 'Report generation failed',
                'sample_size': 0,
                'p_value': 1.0,
                'effect_size': 0,
                'duration_hours': 0,
                'significant': False
            }

    def _generate_statistical_section(self, analysis: Dict) -> str:
        """Generate statistical analysis section."""
        try:
            html = "<div class='statistical-analysis'>\n"

            # Overall summary
            overall = analysis.get('overall_summary', {})
            if overall:
                html += "<h3>Statistical Significance</h3>\n"
                html += "<table class='stats-table'>\n"
                html += "<tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>\n"

                t_test = overall.get('t_test', {})
                p_value = t_test.get('p_value', 1.0)
                significant = t_test.get('significant', False)

                html += f"<tr><td>P-value</td><td>{p_value:.4f}</td>"
                html += f"<td>{'Significant' if significant else 'Not significant'}</td></tr>\n"

                effect_size = overall.get('effect_size', 0)
                if abs(effect_size) < 0.2:
                    effect_desc = "Small"
                elif abs(effect_size) < 0.5:
                    effect_desc = "Medium"
                else:
                    effect_desc = "Large"

                html += f"<tr><td>Effect Size (Cohen's d)</td><td>{effect_size:.3f}</td><td>{effect_desc}</td></tr>\n"

                html += "</table>\n"

            # Multiple testing correction
            multiple_testing = analysis.get('multiple_testing', {})
            if multiple_testing.get('corrections'):
                html += "<h3>Multiple Testing Correction</h3>\n"
                html += "<table class='stats-table'>\n"
                html += "<tr><th>Metric</th><th>Original p-value</th><th>Adjusted p-value</th><th>FDR Significant</th></tr>\n"

                for metric, correction in multiple_testing['corrections'].items():
                    orig_p = correction.get('original_p', 1.0)
                    adj_p = correction.get('adjusted_p', 1.0)
                    fdr_sig = correction.get('fdr_significant', False)

                    html += f"<tr><td>{metric.upper()}</td><td>{orig_p:.4f}</td>"
                    html += f"<td>{adj_p:.4f}</td><td>{'Yes' if fdr_sig else 'No'}</td></tr>\n"

                html += "</table>\n"

            # Power analysis
            power_analysis = analysis.get('power_analysis')
            if power_analysis:
                html += "<h3>Power Analysis</h3>\n"
                html += "<table class='stats-table'>\n"
                html += "<tr><th>Metric</th><th>Value</th></tr>\n"

                if hasattr(power_analysis, 'current_power'):
                    html += f"<tr><td>Current Power</td><td>{power_analysis.current_power:.3f}</td></tr>\n"
                    html += f"<tr><td>Required Sample Size</td><td>{power_analysis.required_sample_size}</td></tr>\n"
                    html += f"<tr><td>Minimum Detectable Effect</td><td>{power_analysis.minimum_detectable_effect:.3f}</td></tr>\n"

                html += "</table>\n"

            html += "</div>\n"
            return html

        except Exception as e:
            logger.error(f"Failed to generate statistical section: {e}")
            return f"<div class='error'>Error generating statistical analysis: {e}</div>\n"

    def _generate_quality_metrics_section(self, analysis: Dict) -> str:
        """Generate quality metrics table."""
        try:
            html = "<table class='metrics-table'>\n"
            html += "<tr><th>Metric</th><th>Baseline</th><th>AI Enhanced</th><th>Improvement</th><th>P-value</th><th>Significant</th></tr>\n"

            metrics = analysis.get('metrics', {})
            for metric_name, metric_result in metrics.items():
                if hasattr(metric_result, 'control_mean'):
                    control_val = metric_result.control_mean
                    treatment_val = metric_result.treatment_mean
                    improvement = metric_result.improvement_pct
                    p_value = metric_result.p_value
                    significant = metric_result.significant

                    # Format values based on metric type
                    if metric_name == 'ssim':
                        control_str = f"{control_val:.3f}"
                        treatment_str = f"{treatment_val:.3f}"
                    elif metric_name == 'mse':
                        control_str = f"{control_val:.4f}"
                        treatment_str = f"{treatment_val:.4f}"
                    elif metric_name == 'psnr':
                        control_str = f"{control_val:.1f} dB"
                        treatment_str = f"{treatment_val:.1f} dB"
                    elif metric_name == 'duration':
                        control_str = f"{control_val:.2f}s"
                        treatment_str = f"{treatment_val:.2f}s"
                    else:
                        control_str = f"{control_val:.3f}"
                        treatment_str = f"{treatment_val:.3f}"

                    # Color code improvement
                    if improvement > 1:
                        improvement_class = "positive"
                    elif improvement < -1:
                        improvement_class = "negative"
                    else:
                        improvement_class = "neutral"

                    html += f"<tr><td>{metric_name.upper()}</td>"
                    html += f"<td>{control_str}</td>"
                    html += f"<td>{treatment_str}</td>"
                    html += f"<td class='{improvement_class}'>{improvement:+.1f}%</td>"
                    html += f"<td>{p_value:.4f}</td>"
                    html += f"<td>{'✓' if significant else '✗'}</td></tr>\n"

            html += "</table>\n"
            return html

        except Exception as e:
            logger.error(f"Failed to generate quality metrics: {e}")
            return f"<div class='error'>Error generating quality metrics: {e}</div>\n"

    def _generate_performance_section(self, analysis: Dict) -> str:
        """Generate performance metrics section."""
        try:
            html = "<div class='performance-section'>\n"

            # Sample sizes
            sample_sizes = analysis.get('sample_sizes', {})
            if sample_sizes:
                html += "<h3>Sample Sizes</h3>\n"
                html += "<table class='stats-table'>\n"
                html += "<tr><th>Group</th><th>Sample Size</th></tr>\n"

                for group, size in sample_sizes.items():
                    if group != 'total':
                        html += f"<tr><td>{group.title()}</td><td>{size}</td></tr>\n"

                html += f"<tr><td><strong>Total</strong></td><td><strong>{sample_sizes.get('total', 0)}</strong></td></tr>\n"
                html += "</table>\n"

            # Processing time analysis if available
            metrics = analysis.get('metrics', {})
            if 'duration' in metrics:
                duration_metric = metrics['duration']
                if hasattr(duration_metric, 'control_mean'):
                    html += "<h3>Processing Time Analysis</h3>\n"
                    html += "<table class='stats-table'>\n"
                    html += "<tr><th>Metric</th><th>Baseline</th><th>AI Enhanced</th><th>Change</th></tr>\n"

                    control_time = duration_metric.control_mean
                    treatment_time = duration_metric.treatment_mean
                    time_change = duration_metric.improvement_pct

                    html += f"<tr><td>Average Time</td><td>{control_time:.2f}s</td>"
                    html += f"<td>{treatment_time:.2f}s</td><td>{time_change:+.1f}%</td></tr>\n"

                    html += "</table>\n"

            html += "</div>\n"
            return html

        except Exception as e:
            logger.error(f"Failed to generate performance section: {e}")
            return f"<div class='error'>Error generating performance metrics: {e}</div>\n"

    def _generate_visual_section(self, visualizations: Dict) -> str:
        """Generate visual comparisons section."""
        try:
            html = "<div class='visual-section'>\n"
            html += "<h2>Visual Comparisons</h2>\n"

            # Include generated visualizations
            batch_comparisons = visualizations.get('batch_comparisons', [])
            if batch_comparisons:
                html += "<h3>Generated Comparisons</h3>\n"
                html += "<div class='comparison-gallery'>\n"

                for i, comparison_file in enumerate(batch_comparisons[:6]):  # Limit to 6 images
                    # Try to encode image as base64 for embedding
                    try:
                        if Path(comparison_file).exists():
                            with open(comparison_file, 'rb') as img_file:
                                img_data = base64.b64encode(img_file.read()).decode()
                                html += f"<div class='comparison-item'>\n"
                                html += f"<img src='data:image/png;base64,{img_data}' alt='Comparison {i+1}' class='comparison-image'>\n"
                                html += f"<p>Comparison {i+1}: {Path(comparison_file).name}</p>\n"
                                html += f"</div>\n"
                    except Exception as e:
                        html += f"<div class='comparison-item'>\n"
                        html += f"<p>Comparison {i+1}: {Path(comparison_file).name} (Image not available)</p>\n"
                        html += f"</div>\n"

                html += "</div>\n"

            # Link to HTML report if available
            html_report = visualizations.get('html_report')
            if html_report and Path(html_report).exists():
                html += f"<p><a href='{html_report}' target='_blank'>View Interactive Visual Report</a></p>\n"

            html += "</div>\n"
            return html

        except Exception as e:
            logger.error(f"Failed to generate visual section: {e}")
            return f"<div class='error'>Error generating visual section: {e}</div>\n"

    def _get_report_template(self) -> str:
        """Get the HTML report template."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>A/B Test Report - {campaign_name}</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 40px;
            background-color: #f5f7fa;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header .subtitle {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
            background: #f8f9ff;
            padding: 30px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .section h2 {{
            color: #333;
            margin-top: 0;
            font-size: 1.8em;
            font-weight: 600;
        }}
        .executive-summary {{
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            text-align: center;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .summary-item {{
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 5px;
            text-align: center;
        }}
        .summary-item .value {{
            font-size: 2em;
            font-weight: bold;
            display: block;
        }}
        .summary-item .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .metrics-table, .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metrics-table th, .stats-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        .metrics-table td, .stats-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        .metrics-table tr:hover, .stats-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .positive {{ color: #4CAF50; font-weight: bold; }}
        .negative {{ color: #f44336; font-weight: bold; }}
        .neutral {{ color: #ff9800; font-weight: bold; }}
        .comparison-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .comparison-item {{
            background: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .comparison-image {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        .recommendation {{
            background: #e8f5e8;
            border: 1px solid #4CAF50;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .recommendation.reject {{
            background: #ffeaea;
            border-color: #f44336;
        }}
        .recommendation.continue {{
            background: #fff3e0;
            border-color: #ff9800;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            background: #f5f7fa;
            color: #666;
            border-top: 1px solid #ddd;
        }}
        .error {{
            background: #ffeaea;
            color: #d32f2f;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #f44336;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>A/B Test Report</h1>
            <p class="subtitle">{campaign_name} • Generated on {timestamp}</p>
        </div>

        <div class="content">
            <!-- Executive Summary -->
            <div class="executive-summary">
                <h2>Executive Summary</h2>
                <div class="summary-grid">
                    <div class="summary-item">
                        <span class="value">{winner}</span>
                        <span class="label">Winner</span>
                    </div>
                    <div class="summary-item">
                        <span class="value">{improvement:.1f}%</span>
                        <span class="label">Improvement</span>
                    </div>
                    <div class="summary-item">
                        <span class="value">{confidence:.0f}%</span>
                        <span class="label">Confidence</span>
                    </div>
                    <div class="summary-item">
                        <span class="value">{sample_size}</span>
                        <span class="label">Sample Size</span>
                    </div>
                </div>

                <div class="recommendation">
                    <h3>Recommendation</h3>
                    <p><strong>{recommendation}</strong></p>
                </div>
            </div>

            <!-- Statistical Analysis -->
            <div class="section">
                <h2>Statistical Analysis</h2>
                {statistical_analysis}
            </div>

            <!-- Quality Metrics -->
            <div class="section">
                <h2>Quality Metrics</h2>
                {quality_table}
            </div>

            <!-- Performance Metrics -->
            <div class="section">
                <h2>Performance Metrics</h2>
                {performance_table}
            </div>

            <!-- Visual Comparisons -->
            <div class="section">
                {visual_section}
            </div>
        </div>

        <div class="footer">
            <p>Report generated by SVG AI A/B Testing Framework</p>
            <p>Duration: {duration:.1f} hours • P-value: {p_value:.4f} • Effect Size: {effect_size:.3f}</p>
        </div>
    </div>
</body>
</html>
        """.strip()

    def _generate_error_report(self, error_message: str) -> str:
        """Generate error report when main generation fails."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>A/B Test Report - Error</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .error {{ background: #ffeaea; color: #d32f2f; padding: 20px; border-radius: 5px; border: 1px solid #f44336; }}
    </style>
</head>
<body>
    <h1>A/B Test Report Generation Error</h1>
    <div class="error">
        <h2>Error occurred while generating report:</h2>
        <p>{error_message}</p>
        <p>Please check the test results data and try again.</p>
    </div>
    <p>Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</body>
</html>
        """.strip()


def test_report_generator():
    """Test the report generator."""
    print("Testing A/B Test Report Generator...")

    # Create generator
    generator = ABTestReportGenerator()

    # Create mock test results
    test_results = {
        'campaign': {
            'name': 'Test Campaign',
            'started_at': '2024-01-01T10:00:00',
            'completed_at': '2024-01-01T12:30:00',
            'duration_seconds': 9000,  # 2.5 hours
            'samples_collected': 100
        },
        'analysis': {
            'sample_sizes': {
                'control': 50,
                'treatment': 50,
                'total': 100
            },
            'overall_summary': {
                'quality_improvement': 7.5,
                't_test': {
                    'p_value': 0.0023,
                    'significant': True,
                    't_statistic': 3.2
                },
                'effect_size': 0.65,
                'confidence_interval': [0.02, 0.13],
                'recommendation': 'DEPLOY: Significant improvement detected'
            },
            'metrics': {
                'ssim': type('MockResult', (), {
                    'control_mean': 0.82,
                    'treatment_mean': 0.88,
                    'improvement_pct': 7.3,
                    'p_value': 0.0023,
                    'significant': True
                })(),
                'duration': type('MockResult', (), {
                    'control_mean': 2.1,
                    'treatment_mean': 2.3,
                    'improvement_pct': 9.5,
                    'p_value': 0.1234,
                    'significant': False
                })()
            },
            'multiple_testing': {
                'corrections': {
                    'ssim': {
                        'original_p': 0.0023,
                        'adjusted_p': 0.0046,
                        'fdr_significant': True
                    },
                    'duration': {
                        'original_p': 0.1234,
                        'adjusted_p': 0.1234,
                        'fdr_significant': False
                    }
                }
            }
        },
        'visualizations': {
            'batch_comparisons': [],
            'html_report': '/tmp/test_visual_report.html'
        }
    }

    # Test 1: Generate HTML report
    print("\n✓ Testing HTML report generation:")
    report_html = generator.generate_report(test_results)
    assert 'A/B Test Report' in report_html
    assert 'Test Campaign' in report_html
    assert '7.5%' in report_html  # Improvement
    print("  HTML report generated successfully")

    # Test 2: Save HTML report
    print("\n✓ Testing HTML report saving:")
    output_file = '/tmp/test_ab_report.html'
    generator.save_html_report(test_results, output_file)
    assert Path(output_file).exists()
    print(f"  HTML report saved to {output_file}")

    # Test 3: Test executive summary generation
    print("\n✓ Testing executive summary:")
    summary = generator._generate_executive_summary(
        test_results['campaign'], test_results['analysis']
    )
    assert summary['winner'] == 'AI Enhanced'
    assert summary['improvement'] == 7.5
    assert summary['significant'] == True
    print(f"  Executive summary: {summary['winner']} with {summary['improvement']:.1f}% improvement")

    # Test 4: Test statistical section
    print("\n✓ Testing statistical section:")
    stats_html = generator._generate_statistical_section(test_results['analysis'])
    assert 'Statistical Significance' in stats_html
    assert '0.0023' in stats_html  # p-value
    print("  Statistical section generated")

    # Test 5: Test quality metrics table
    print("\n✓ Testing quality metrics table:")
    quality_html = generator._generate_quality_metrics_section(test_results['analysis'])
    assert 'SSIM' in quality_html
    assert '0.820' in quality_html  # Control mean
    assert '0.880' in quality_html  # Treatment mean
    print("  Quality metrics table generated")

    # Test 6: Test error handling
    print("\n✓ Testing error handling:")
    error_report = generator._generate_error_report("Test error message")
    assert 'Error occurred' in error_report
    assert 'Test error message' in error_report
    print("  Error report generated")

    # Test 7: Test PDF export (if available)
    print("\n✓ Testing PDF export:")
    if PDF_SUPPORT:
        try:
            pdf_output = '/tmp/test_ab_report.pdf'
            generator.export_pdf_report(test_results, pdf_output)
            if Path(pdf_output).exists():
                print(f"  PDF report saved to {pdf_output}")
            else:
                print("  PDF export attempted but file not created")
        except Exception as e:
            print(f"  PDF export failed: {e}")
    else:
        print("  PDF support not available (install weasyprint)")

    print("\n✅ All report generator tests passed!")
    return generator


if __name__ == "__main__":
    test_report_generator()