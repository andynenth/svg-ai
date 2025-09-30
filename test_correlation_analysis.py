#!/usr/bin/env python3
"""
Test script for correlation analysis
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from backend.ai_modules.optimization.correlation_analysis import CorrelationAnalysis
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_correlation_analysis():
    """Test the correlation analysis system"""
    print("🔍 Testing Correlation Analysis System")

    # Create analyzer without validation data (will generate sample data)
    analyzer = CorrelationAnalysis()

    print("\n📊 Analyzing correlation effectiveness...")
    effectiveness_scores = analyzer.analyze_correlation_effectiveness()

    print("\n📈 Correlation Effectiveness Scores:")
    for formula, score in effectiveness_scores.items():
        print(f"  - {formula}: {score:.3f}")

    print("\n🎯 Analyzing by logo type...")
    logo_type_analysis = analyzer.analyze_by_logo_type()

    print("\n📋 Logo Type Analysis:")
    for logo_type, metrics in logo_type_analysis.items():
        print(f"  - {logo_type}: {metrics.get('success_rate', 0):.1f}% success, "
              f"{metrics.get('avg_quality_improvement', 0):.1f}% avg improvement")

    print("\n⚠️  Identifying underperforming formulas...")
    underperforming = analyzer.identify_underperforming_formulas()
    print(f"Underperforming formulas: {underperforming}")

    print("\n📊 Generating statistical significance tests...")
    significance_tests = analyzer.generate_statistical_significance_tests()

    print("\n📈 Statistical Significance Results:")
    for formula, test_results in significance_tests.items():
        print(f"  - {formula}: correlation={test_results.get('pearson_correlation', 0):.3f}, "
              f"p-value={test_results.get('pearson_p_value', 1):.3f}, "
              f"significant={'Yes' if test_results.get('pearson_significant', False) else 'No'}")

    print("\n📏 Calculating R-squared values...")
    r_squared_values = analyzer.calculate_r_squared_values()

    print("\n📊 R-squared Values:")
    for formula, r2 in r_squared_values.items():
        print(f"  - {formula}: R² = {r2:.3f}")

    print("\n🎯 Identifying optimal parameter ranges...")
    optimal_ranges = analyzer.identify_optimal_parameter_ranges()

    print("\n📋 Optimal Parameter Ranges by Logo Type:")
    for logo_type, ranges in optimal_ranges.items():
        print(f"  - {logo_type}:")
        for param, (min_val, max_val) in ranges.items():
            print(f"    • {param}: [{min_val:.2f}, {max_val:.2f}]")

    print("\n📊 Generating scatter plots...")
    try:
        plot_files = analyzer.generate_scatter_plots("correlation_analysis_plots")
        print(f"Generated {len(plot_files)} scatter plots")
        for plot_file in plot_files:
            print(f"  - {plot_file}")
    except Exception as e:
        print(f"⚠️  Plot generation failed (expected on headless systems): {e}")

    print("\n📋 Creating comprehensive report...")
    report = analyzer.create_correlation_effectiveness_report()

    print("\n📊 Correlation Effectiveness Report Summary:")
    print(f"  - Total records analyzed: {report['dataset_info']['total_records']}")
    print(f"  - Successful optimizations: {report['dataset_info']['successful_records']}")
    print(f"  - Effectiveness scores calculated: {len(report['effectiveness_scores'])}")
    print(f"  - Underperforming formulas: {len(report['underperforming_formulas'])}")
    print(f"  - Recommendations generated: {len(report['recommendations'])}")

    print("\n🔍 Recommendations:")
    for i, recommendation in enumerate(report['recommendations'], 1):
        print(f"  {i}. {recommendation}")

    print("\n💾 Exporting report...")
    report_file = analyzer.export_report("correlation_effectiveness_report.json")
    print(f"Report exported to: {report_file}")

    print("\n✅ Correlation analysis test completed successfully!")
    return True

if __name__ == "__main__":
    test_correlation_analysis()