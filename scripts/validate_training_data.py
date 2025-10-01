#!/usr/bin/env python3
"""
Data Validation & Analysis for VTracer Training Data - Day 1 Task 4
Validates and analyzes training data collected from batch parameter testing.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_training_data(data_path: str) -> Dict[str, Any]:
    """
    Load training data from JSON file.

    Args:
        data_path: Path to training data JSON file

    Returns:
        Loaded training data dictionary
    """
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)

        print(f"📊 Loaded training data from: {data_path}")

        results = data.get('results', [])
        metadata = data.get('metadata', {})

        print(f"📋 Data contains {len(results)} conversion results")
        print(f"🕒 Data collected: {metadata.get('timestamp_started', 'Unknown')}")

        return data

    except Exception as e:
        print(f"❌ Failed to load training data: {e}")
        return {}


def validate_data_completeness(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check for missing values and data completeness.

    Args:
        results: List of conversion results

    Returns:
        Validation report dictionary
    """
    print(f"🔍 Validating data completeness...")

    if not results:
        return {
            'total_records': 0,
            'missing_values': {},
            'completeness_rate': 0,
            'valid_records': 0
        }

    # Required fields
    required_fields = [
        'image_path', 'parameters', 'conversion_success',
        'processing_time', 'file_size_ratio', 'ssim', 'mse'
    ]

    missing_counts = {field: 0 for field in required_fields}
    valid_records = 0

    for result in results:
        record_valid = True

        for field in required_fields:
            if field not in result or result[field] is None:
                missing_counts[field] += 1
                record_valid = False

        # Check for reasonable values
        if result.get('conversion_success', False):
            if (result.get('ssim', 0) < 0 or result.get('ssim', 0) > 1 or
                result.get('mse', float('inf')) == float('inf') or
                result.get('processing_time', 0) <= 0):
                record_valid = False

        if record_valid:
            valid_records += 1

    total_records = len(results)
    completeness_rate = valid_records / total_records if total_records > 0 else 0

    validation_report = {
        'total_records': total_records,
        'valid_records': valid_records,
        'completeness_rate': completeness_rate,
        'missing_values': missing_counts,
        'missing_percentages': {
            field: (count / total_records * 100) if total_records > 0 else 0
            for field, count in missing_counts.items()
        }
    }

    print(f"✅ Completeness validation complete:")
    print(f"  📊 Total records: {total_records}")
    print(f"  ✅ Valid records: {valid_records}")
    print(f"  📈 Completeness rate: {completeness_rate:.1%}")

    return validation_report


def detect_outliers(values: List[float], field_name: str) -> Dict[str, Any]:
    """
    Detect outliers using IQR method.

    Args:
        values: List of numeric values
        field_name: Name of the field being analyzed

    Returns:
        Outlier analysis results
    """
    if not values:
        return {'outliers': [], 'outlier_count': 0, 'outlier_rate': 0}

    # Calculate quartiles
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    # Define outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Find outliers
    outliers = [v for v in values if v < lower_bound or v > upper_bound]

    return {
        'field_name': field_name,
        'total_values': len(values),
        'outlier_count': len(outliers),
        'outlier_rate': len(outliers) / len(values) if values else 0,
        'quartiles': {'q1': q1, 'q3': q3, 'iqr': iqr},
        'bounds': {'lower': lower_bound, 'upper': upper_bound},
        'statistics': {
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values)
        }
    }


def analyze_quality_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze quality metrics for outliers and distributions.

    Args:
        results: List of conversion results

    Returns:
        Quality metrics analysis
    """
    print(f"📊 Analyzing quality metrics...")

    # Extract successful conversions only
    successful_results = [r for r in results if r.get('conversion_success', False)]

    if not successful_results:
        return {'error': 'No successful conversions found'}

    # Extract metric values
    ssim_values = [r.get('ssim', 0) for r in successful_results if 'ssim' in r]
    mse_values = [r.get('mse', 0) for r in successful_results if 'mse' in r and r['mse'] != float('inf')]
    processing_times = [r.get('processing_time', 0) for r in successful_results if 'processing_time' in r]
    file_size_ratios = [r.get('file_size_ratio', 0) for r in successful_results if 'file_size_ratio' in r]

    # Analyze each metric for outliers
    quality_analysis = {
        'ssim': detect_outliers(ssim_values, 'SSIM'),
        'mse': detect_outliers(mse_values, 'MSE'),
        'processing_time': detect_outliers(processing_times, 'Processing Time'),
        'file_size_ratio': detect_outliers(file_size_ratios, 'File Size Ratio')
    }

    print(f"✅ Quality metrics analysis complete:")
    for metric, analysis in quality_analysis.items():
        print(f"  {metric}: {analysis['outlier_count']}/{analysis['total_values']} outliers ({analysis['outlier_rate']:.1%})")

    return quality_analysis


def analyze_parameter_distributions(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze parameter value distributions.

    Args:
        results: List of conversion results

    Returns:
        Parameter distribution analysis
    """
    print(f"⚙️  Analyzing parameter distributions...")

    # Extract all parameter combinations
    all_params = []
    for result in results:
        params = result.get('parameters', {})
        if params:
            all_params.append(params)

    if not all_params:
        return {'error': 'No parameter data found'}

    # Get parameter names
    param_names = set()
    for params in all_params:
        param_names.update(params.keys())

    # Analyze distribution of each parameter
    param_distributions = {}

    for param_name in param_names:
        values = [params.get(param_name) for params in all_params if param_name in params]

        # Handle both numeric and categorical parameters
        if values and isinstance(values[0], (int, float)):
            # Numeric parameter
            param_distributions[param_name] = {
                'type': 'numeric',
                'unique_values': list(set(values)),
                'value_counts': {str(v): values.count(v) for v in set(values)},
                'statistics': {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values) if len(set(values)) > 1 else 0
                }
            }
        else:
            # Categorical parameter
            param_distributions[param_name] = {
                'type': 'categorical',
                'unique_values': list(set(values)),
                'value_counts': {str(v): values.count(v) for v in set(values)}
            }

    print(f"✅ Parameter distribution analysis complete:")
    for param_name, dist in param_distributions.items():
        print(f"  {param_name}: {len(dist['unique_values'])} unique values")

    return param_distributions


def create_visualization_plots(results: List[Dict[str, Any]], output_dir: str) -> List[str]:
    """
    Generate visualization plots for the training data.

    Args:
        results: List of conversion results
        output_dir: Directory to save plots

    Returns:
        List of generated plot file paths
    """
    print(f"📈 Creating visualization plots...")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract successful conversions
    successful_results = [r for r in results if r.get('conversion_success', False)]

    if not successful_results:
        print(f"⚠️  No successful conversions to plot")
        return []

    plot_files = []

    # Plot 1: SSIM Distribution Histogram
    ssim_values = [r.get('ssim', 0) for r in successful_results if 'ssim' in r]

    if ssim_values:
        plt.figure(figsize=(10, 6))
        plt.hist(ssim_values, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribution of SSIM Scores', fontsize=14, fontweight='bold')
        plt.xlabel('SSIM Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_ssim = np.mean(ssim_values)
        plt.axvline(mean_ssim, color='red', linestyle='--', label=f'Mean: {mean_ssim:.3f}')
        plt.legend()

        plot_path = os.path.join(output_dir, 'ssim_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_path)
        print(f"  📊 Saved: ssim_distribution.png")

    # Plot 2: Parameter vs SSIM Scatter Plots
    numeric_params = ['color_precision', 'corner_threshold', 'max_iterations',
                     'path_precision', 'layer_difference', 'length_threshold', 'splice_threshold']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Parameter vs SSIM Relationships', fontsize=16, fontweight='bold')

    for i, param in enumerate(numeric_params):
        if i >= 9:  # Only plot first 9 parameters
            break

        row, col = i // 3, i % 3
        ax = axes[row, col]

        param_values = []
        ssim_values_for_param = []

        for result in successful_results:
            params = result.get('parameters', {})
            if param in params and 'ssim' in result:
                param_values.append(params[param])
                ssim_values_for_param.append(result['ssim'])

        if param_values and ssim_values_for_param:
            ax.scatter(param_values, ssim_values_for_param, alpha=0.6)
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel('SSIM Score')
            ax.set_title(f'{param.replace("_", " ").title()} vs SSIM')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{param.replace("_", " ").title()} vs SSIM')

    # Hide empty subplots
    for i in range(len(numeric_params), 9):
        row, col = i // 3, i % 3
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'parameter_ssim_scatter.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(plot_path)
    print(f"  📊 Saved: parameter_ssim_scatter.png")

    # Plot 3: Processing Time vs File Size Ratio
    processing_times = [r.get('processing_time', 0) for r in successful_results if 'processing_time' in r]
    file_size_ratios = [r.get('file_size_ratio', 0) for r in successful_results if 'file_size_ratio' in r]

    if processing_times and file_size_ratios and len(processing_times) == len(file_size_ratios):
        plt.figure(figsize=(10, 6))
        plt.scatter(processing_times, file_size_ratios, alpha=0.6)
        plt.title('Processing Time vs File Size Ratio', fontsize=14, fontweight='bold')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('File Size Ratio (SVG/PNG)')
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(output_dir, 'processing_time_vs_file_size.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_path)
        print(f"  📊 Saved: processing_time_vs_file_size.png")

    print(f"✅ Generated {len(plot_files)} visualization plots")
    return plot_files


def generate_validation_report(data: Dict[str, Any], completeness: Dict[str, Any],
                             quality_analysis: Dict[str, Any], param_distributions: Dict[str, Any],
                             plot_files: List[str], output_path: str) -> Dict[str, Any]:
    """
    Generate comprehensive validation report.

    Args:
        data: Original training data
        completeness: Data completeness analysis
        quality_analysis: Quality metrics analysis
        param_distributions: Parameter distribution analysis
        plot_files: List of generated plot files
        output_path: Path to save the report

    Returns:
        Complete validation report
    """
    print(f"📋 Generating validation report...")

    results = data.get('results', [])
    metadata = data.get('metadata', {})

    # Calculate summary statistics
    successful_conversions = len([r for r in results if r.get('conversion_success', False)])
    failed_conversions = len(results) - successful_conversions

    report = {
        'validation_metadata': {
            'validation_timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': metadata.get('batch_processing_type', 'unknown'),
            'data_collection_started': metadata.get('timestamp_started', 'unknown'),
            'total_data_points': len(results),
            'meets_1000_threshold': len(results) >= 1000
        },
        'data_summary': {
            'total_conversions': len(results),
            'successful_conversions': successful_conversions,
            'failed_conversions': failed_conversions,
            'success_rate': successful_conversions / len(results) if results else 0,
            'logo_categories': metadata.get('logo_selection', {}),
            'parameter_combinations_used': metadata.get('total_parameter_combinations', 0)
        },
        'data_quality': {
            'completeness_analysis': completeness,
            'quality_metrics_analysis': quality_analysis,
            'parameter_distributions': param_distributions
        },
        'visualizations': {
            'plots_generated': len(plot_files),
            'plot_files': [os.path.basename(f) for f in plot_files]
        },
        'validation_summary': {
            'data_validation_passed': True,
            'issues_found': [],
            'recommendations': []
        }
    }

    # Check validation criteria and add issues/recommendations
    if len(results) < 1000:
        report['validation_summary']['issues_found'].append(
            f"Data contains only {len(results)} points, less than target of 1,000"
        )
        report['validation_summary']['recommendations'].append(
            "Run additional batch processing to collect more training data"
        )

    if completeness['completeness_rate'] < 0.95:
        report['validation_summary']['issues_found'].append(
            f"Data completeness rate is {completeness['completeness_rate']:.1%}, below 95% threshold"
        )
        report['validation_summary']['recommendations'].append(
            "Review and fix data collection issues causing missing values"
        )

    if successful_conversions / len(results) < 0.8 if results else False:
        report['validation_summary']['issues_found'].append(
            f"Success rate is {successful_conversions / len(results):.1%}, below 80% threshold"
        )
        report['validation_summary']['recommendations'].append(
            "Review VTracer parameter ranges to improve conversion success rate"
        )

    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✅ Validation report saved to: {output_path}")

    return report


def main():
    """Main entry point for data validation script."""
    parser = argparse.ArgumentParser(description='Validate and analyze VTracer training data')
    parser.add_argument('data_file', help='Path to training data JSON file')
    parser.add_argument('--output-dir', type=str, default='data/training/validation',
                       help='Directory for validation outputs (default: data/training/validation)')
    parser.add_argument('--report-output', type=str, default='data/training/data_validation_report.json',
                       help='Path for validation report (default: data/training/data_validation_report.json)')

    args = parser.parse_args()

    print(f"🔍 Data Validation & Analysis")
    print(f"📁 Input file: {args.data_file}")
    print(f"📊 Output directory: {args.output_dir}")
    print(f"📋 Report output: {args.report_output}")

    # Validate input file
    if not os.path.exists(args.data_file):
        print(f"❌ Error: Training data file not found: {args.data_file}")
        sys.exit(1)

    # Load training data
    data = load_training_data(args.data_file)
    if not data:
        print(f"❌ Error: Failed to load training data")
        sys.exit(1)

    results = data.get('results', [])
    if not results:
        print(f"❌ Error: No results found in training data")
        sys.exit(1)

    print(f"\n🔍 Starting validation analysis...")

    # 1. Validate data completeness
    completeness_analysis = validate_data_completeness(results)

    # 2. Analyze quality metrics for outliers
    quality_analysis = analyze_quality_metrics(results)

    # 3. Analyze parameter distributions
    param_distributions = analyze_parameter_distributions(results)

    # 4. Create visualization plots
    plot_files = create_visualization_plots(results, args.output_dir)

    # 5. Generate comprehensive validation report
    report = generate_validation_report(
        data, completeness_analysis, quality_analysis,
        param_distributions, plot_files, args.report_output
    )

    # Print summary
    print(f"\n📊 Validation Summary:")
    print(f"  🔢 Data points validated: {len(results)}")
    print(f"  ✅ Data completeness: {completeness_analysis['completeness_rate']:.1%}")
    print(f"  📈 Success rate: {report['data_summary']['success_rate']:.1%}")
    print(f"  📊 Plots generated: {len(plot_files)}")
    print(f"  📋 Report saved: {args.report_output}")

    # Check acceptance criteria
    criteria_met = []
    if len(results) >= 1000:
        criteria_met.append("✅ Validates 1,000+ data points")
    else:
        criteria_met.append(f"⚠️  Validates {len(results)} data points (target: 1,000+)")

    if os.path.exists(args.report_output):
        criteria_met.append("✅ Generated validation report")

    if len(plot_files) >= 3:
        criteria_met.append("✅ Created 3+ visualization plots")
    else:
        criteria_met.append(f"⚠️  Created {len(plot_files)} plots (target: 3+)")

    print(f"\n📋 Acceptance Criteria:")
    for criterion in criteria_met:
        print(f"  {criterion}")

    if report['validation_summary']['issues_found']:
        print(f"\n⚠️  Issues Found:")
        for issue in report['validation_summary']['issues_found']:
            print(f"  - {issue}")

    if report['validation_summary']['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in report['validation_summary']['recommendations']:
            print(f"  - {rec}")

    return 0 if len(results) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())