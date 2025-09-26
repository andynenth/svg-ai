#!/usr/bin/env python3
"""
Analyze parameter effectiveness and generate sensitivity matrix.

This script tests how each parameter affects quality metrics
to identify the most important parameters for optimization.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from converters.vtracer_converter import VTracerConverter
from utils.image_loader import QualityMetricsWrapper
from utils.ai_detector import create_detector


class ParameterAnalyzer:
    """Analyze parameter impact on conversion quality."""

    def __init__(self):
        """Initialize the analyzer."""
        self.converter = VTracerConverter()
        self.metrics = QualityMetricsWrapper()
        self.detector = create_detector()

        # Base parameters for testing
        self.base_params = {
            'color_precision': 6,
            'layer_difference': 8,
            'corner_threshold': 40,
            'length_threshold': 5.0,
            'max_iterations': 10,
            'splice_threshold': 45,
            'path_precision': 6
        }

        # Parameter variations to test
        self.param_variations = {
            'color_precision': [1, 2, 4, 6, 8, 10, 12],
            'layer_difference': [4, 6, 8, 10, 12, 14, 16],
            'corner_threshold': [10, 20, 30, 40, 50, 60, 70],
            'length_threshold': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'max_iterations': [5, 10, 15, 20],
            'splice_threshold': [20, 30, 40, 50, 60, 70],
            'path_precision': [1, 3, 5, 7, 9]
        }

    def test_parameter(self, image_path: str, param_name: str,
                      param_value: float) -> Dict:
        """
        Test a single parameter value.

        Args:
            image_path: Path to test image
            param_name: Parameter to test
            param_value: Value to test

        Returns:
            Results dictionary
        """
        # Create test parameters
        test_params = self.base_params.copy()
        test_params[param_name] = param_value

        output_path = "temp_analysis.svg"

        try:
            # Convert
            result = self.converter.convert_with_params(
                image_path,
                output_path,
                **test_params
            )

            if not result['success']:
                return {'success': False}

            # Calculate metrics
            ssim = self.metrics.calculate_ssim_from_paths(
                image_path,
                output_path
            )

            # File sizes
            png_size = Path(image_path).stat().st_size
            svg_size = Path(output_path).stat().st_size
            size_ratio = svg_size / png_size

            # Clean up
            Path(output_path).unlink(missing_ok=True)

            return {
                'success': True,
                'ssim': ssim,
                'size_ratio': size_ratio,
                'conversion_time': result.get('conversion_time', 0)
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def analyze_parameter_sensitivity(self, image_path: str) -> Dict:
        """
        Analyze sensitivity of all parameters for an image.

        Args:
            image_path: Path to test image

        Returns:
            Analysis results
        """
        print(f"\nAnalyzing: {Path(image_path).name}")

        # Detect logo type
        logo_type, confidence, _ = self.detector.detect_logo_type(image_path)
        print(f"  Type: {logo_type} (confidence: {confidence:.2f})")

        results = {
            'image': str(image_path),
            'logo_type': logo_type,
            'parameters': {}
        }

        # Test each parameter
        for param_name, values in self.param_variations.items():
            print(f"  Testing {param_name}...", end="")

            param_results = []

            for value in values:
                result = self.test_parameter(image_path, param_name, value)

                if result['success']:
                    param_results.append({
                        'value': value,
                        'ssim': result['ssim'],
                        'size_ratio': result['size_ratio'],
                        'time': result.get('conversion_time', 0)
                    })

            if param_results:
                # Calculate sensitivity metrics
                ssim_values = [r['ssim'] for r in param_results]
                ssim_range = max(ssim_values) - min(ssim_values)
                ssim_std = np.std(ssim_values)

                size_values = [r['size_ratio'] for r in param_results]
                size_range = max(size_values) - min(size_values)

                results['parameters'][param_name] = {
                    'values': param_results,
                    'ssim_range': ssim_range,
                    'ssim_std': ssim_std,
                    'size_range': size_range,
                    'best_value': values[np.argmax(ssim_values)],
                    'best_ssim': max(ssim_values)
                }

                print(f" range={ssim_range:.3f}, best={max(ssim_values):.3f}")
            else:
                print(" failed")

        return results

    def create_effectiveness_matrix(self, test_images: List[str]) -> Dict:
        """
        Create parameter effectiveness matrix across multiple images.

        Args:
            test_images: List of test image paths

        Returns:
            Effectiveness matrix data
        """
        print("="*60)
        print("PARAMETER EFFECTIVENESS ANALYSIS")
        print("="*60)

        all_results = []

        for image_path in test_images:
            if Path(image_path).exists():
                result = self.analyze_parameter_sensitivity(image_path)
                all_results.append(result)

        # Aggregate results
        effectiveness = {}

        for param_name in self.param_variations.keys():
            ssim_impacts = []
            size_impacts = []

            for result in all_results:
                if param_name in result['parameters']:
                    param_data = result['parameters'][param_name]
                    ssim_impacts.append(param_data['ssim_range'])
                    size_impacts.append(param_data['size_range'])

            if ssim_impacts:
                effectiveness[param_name] = {
                    'avg_ssim_impact': np.mean(ssim_impacts),
                    'max_ssim_impact': max(ssim_impacts),
                    'avg_size_impact': np.mean(size_impacts),
                    'importance_score': np.mean(ssim_impacts) * 2 + max(ssim_impacts)
                }

        # Sort by importance
        sorted_params = sorted(
            effectiveness.items(),
            key=lambda x: x[1]['importance_score'],
            reverse=True
        )

        return {
            'effectiveness': effectiveness,
            'sorted_params': sorted_params,
            'detailed_results': all_results
        }

    def generate_report(self, matrix_data: Dict, output_file: str = "parameter_analysis.md"):
        """
        Generate markdown report of parameter effectiveness.

        Args:
            matrix_data: Effectiveness matrix data
            output_file: Output file path
        """
        report = []
        report.append("# Parameter Effectiveness Analysis\n")
        report.append("## Overview\n")
        report.append("Analysis of how each VTracer parameter affects conversion quality.\n")

        # Parameter ranking
        report.append("## Parameter Importance Ranking\n")
        report.append("| Rank | Parameter | SSIM Impact | Size Impact | Importance Score |\n")
        report.append("|------|-----------|-------------|-------------|------------------|\n")

        for i, (param, data) in enumerate(matrix_data['sorted_params'], 1):
            report.append(
                f"| {i} | {param} | {data['avg_ssim_impact']:.4f} | "
                f"{data['avg_size_impact']:.3f} | {data['importance_score']:.3f} |\n"
            )

        # Detailed analysis
        report.append("\n## Detailed Parameter Analysis\n")

        for param, data in matrix_data['sorted_params']:
            report.append(f"\n### {param}\n")
            report.append(f"- **Average SSIM Impact**: {data['avg_ssim_impact']:.4f}\n")
            report.append(f"- **Maximum SSIM Impact**: {data['max_ssim_impact']:.4f}\n")
            report.append(f"- **Average Size Impact**: {data['avg_size_impact']:.3f}x\n")
            report.append(f"- **Importance Score**: {data['importance_score']:.3f}\n")

            # Add interpretation
            if data['importance_score'] > 0.5:
                report.append(f"- **Priority**: HIGH - Critical parameter for quality\n")
            elif data['importance_score'] > 0.2:
                report.append(f"- **Priority**: MEDIUM - Moderate impact on quality\n")
            else:
                report.append(f"- **Priority**: LOW - Minor impact on quality\n")

        # Recommendations
        report.append("\n## Optimization Recommendations\n")

        top_params = matrix_data['sorted_params'][:3]
        report.append("### Focus Areas\n")
        report.append("Based on the analysis, prioritize optimizing these parameters:\n\n")

        for param, data in top_params:
            report.append(f"1. **{param}**: Most impactful with {data['avg_ssim_impact']:.3f} SSIM range\n")

        report.append("\n### Parameter Guidelines\n")
        report.append("- For **quality**: Focus on color_precision and corner_threshold\n")
        report.append("- For **file size**: Adjust path_precision and splice_threshold\n")
        report.append("- For **speed**: Limit max_iterations\n")

        # Save report
        with open(output_file, 'w') as f:
            f.writelines(report)

        print(f"\n✅ Report saved to {output_file}")

    def visualize_sensitivity(self, matrix_data: Dict, output_file: str = "parameter_sensitivity.png"):
        """
        Create visualization of parameter sensitivity.

        Args:
            matrix_data: Effectiveness matrix data
            output_file: Output image file
        """
        # Prepare data for heatmap
        params = list(matrix_data['effectiveness'].keys())
        metrics = ['SSIM Impact', 'Size Impact', 'Importance']

        data = []
        for param in params:
            param_data = matrix_data['effectiveness'][param]
            data.append([
                param_data['avg_ssim_impact'],
                param_data['avg_size_impact'],
                param_data['importance_score'] / 3  # Normalize
            ])

        data = np.array(data).T

        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            data,
            annot=True,
            fmt='.3f',
            xticklabels=params,
            yticklabels=metrics,
            cmap='YlOrRd',
            cbar_kws={'label': 'Impact'}
        )

        plt.title('Parameter Sensitivity Matrix')
        plt.xlabel('Parameters')
        plt.ylabel('Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_file, dpi=100)
        plt.close()

        print(f"✅ Visualization saved to {output_file}")


def main():
    """Main function."""
    # Create analyzer
    analyzer = ParameterAnalyzer()

    # Test images
    test_images = [
        "data/logos/simple_geometric/circle_00.png",
        "data/logos/text_based/text_tech_00.png",
        "data/logos/gradients/gradient_radial_06.png",
        "data/logos/complex/complex_multi_08.png"
    ]

    # Filter existing images
    test_images = [img for img in test_images if Path(img).exists()]

    if not test_images:
        print("❌ No test images found")
        return 1

    # Create effectiveness matrix
    matrix_data = analyzer.create_effectiveness_matrix(test_images)

    # Generate report
    analyzer.generate_report(matrix_data)

    # Create visualization
    try:
        analyzer.visualize_sensitivity(matrix_data)
    except ImportError:
        print("⚠️ Matplotlib/seaborn not installed, skipping visualization")

    # Print summary
    print("\n" + "="*60)
    print("PARAMETER EFFECTIVENESS SUMMARY")
    print("="*60)

    print("\nTop 5 Most Important Parameters:")
    for i, (param, data) in enumerate(matrix_data['sorted_params'][:5], 1):
        print(f"{i}. {param:20} (importance: {data['importance_score']:.3f})")

    # Save JSON data
    with open('parameter_effectiveness.json', 'w') as f:
        # Convert for JSON serialization
        json_data = {
            'effectiveness': matrix_data['effectiveness'],
            'ranking': [(p, d) for p, d in matrix_data['sorted_params']]
        }
        json.dump(json_data, f, indent=2)

    print("\n✅ Analysis complete. See parameter_analysis.md for full report.")

    return 0


if __name__ == "__main__":
    sys.exit(main())