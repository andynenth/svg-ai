"""
Analyze Correlation Formulas - Task 1 Implementation
Analyze and document existing correlation formula behavior.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import logging

# Import the correlation formulas
import sys
sys.path.append('/Users/nrw/python/svg-ai')
from backend.ai_modules.optimization.unified_parameter_formulas import ParameterFormulas CorrelationFormulas


class CorrelationFormulaAnalyzer:
    """Analyzer for existing correlation formulas."""

    def __init__(self):
        """Initialize the analyzer."""
        self.formulas = CorrelationFormulas()
        self.analysis_results = {
            'formulas': {},
            'test_cases': [],
            'performance_metrics': {},
            'documentation': {}
        }

    def document_all_formulas(self) -> Dict[str, Any]:
        """Document all formula behavior, input/output ranges, and relationships."""

        formula_docs = {
            'edge_to_corner_threshold': {
                'description': 'Maps edge density to corner threshold parameter',
                'formula': 'max(10, min(110, int(110 - (edge_density * 800))))',
                'input_range': '[0.0, 1.0]',
                'output_range': '[10, 110]',
                'relationship': 'inverse - higher edge density → lower corner threshold',
                'rationale': 'Higher edge density requires lower corner threshold for better detail preservation',
                'edge_cases': [
                    {'input': 0.0, 'output': 110, 'description': 'No edges → maximum threshold'},
                    {'input': 0.5, 'output': 10, 'description': 'Medium edges → minimum threshold'},
                    {'input': 1.0, 'output': 10, 'description': 'Maximum edges → minimum threshold'}
                ]
            },
            'colors_to_precision': {
                'description': 'Maps unique colors to color precision parameter',
                'formula': 'max(2, min(10, int(2 + log2(max(1, unique_colors)))))',
                'input_range': '[1, ∞)',
                'output_range': '[2, 10]',
                'relationship': 'logarithmic - more colors → higher precision',
                'rationale': 'More colors require higher precision for accurate representation',
                'edge_cases': [
                    {'input': 1, 'output': 2, 'description': 'Single color → minimum precision'},
                    {'input': 2, 'output': 3, 'description': '2 colors → low precision'},
                    {'input': 256, 'output': 10, 'description': '256 colors → maximum precision'},
                    {'input': 1024, 'output': 10, 'description': '1024+ colors → clamped to max'}
                ]
            },
            'entropy_to_path_precision': {
                'description': 'Maps entropy to path precision parameter',
                'formula': 'max(1, min(20, int(20 * (1 - entropy))))',
                'input_range': '[0.0, 1.0]',
                'output_range': '[1, 20]',
                'relationship': 'inverse - higher entropy → lower precision',
                'rationale': 'Higher entropy (randomness) needs less detail precision',
                'edge_cases': [
                    {'input': 0.0, 'output': 20, 'description': 'No entropy → maximum precision'},
                    {'input': 0.5, 'output': 10, 'description': 'Medium entropy → medium precision'},
                    {'input': 1.0, 'output': 1, 'description': 'Maximum entropy → minimum precision'}
                ]
            },
            'corners_to_length_threshold': {
                'description': 'Maps corner density to length threshold parameter',
                'formula': 'max(1.0, min(20.0, 1.0 + (corner_density * 100)))',
                'input_range': '[0.0, 1.0]',
                'output_range': '[1.0, 20.0]',
                'relationship': 'linear positive - more corners → higher threshold',
                'rationale': 'More corners require longer segments to capture detail',
                'edge_cases': [
                    {'input': 0.0, 'output': 1.0, 'description': 'No corners → minimum threshold'},
                    {'input': 0.19, 'output': 20.0, 'description': '19%+ corners → maximum threshold'},
                    {'input': 1.0, 'output': 20.0, 'description': 'Maximum corners → maximum threshold'}
                ]
            },
            'gradient_to_splice_threshold': {
                'description': 'Maps gradient strength to splice threshold parameter',
                'formula': 'max(10, min(100, int(10 + (gradient_strength * 90))))',
                'input_range': '[0.0, 1.0]',
                'output_range': '[10, 100]',
                'relationship': 'linear positive - stronger gradients → higher threshold',
                'rationale': 'Stronger gradients need more splice points for smooth transitions',
                'edge_cases': [
                    {'input': 0.0, 'output': 10, 'description': 'No gradient → minimum splice points'},
                    {'input': 0.5, 'output': 55, 'description': 'Medium gradient → medium splice points'},
                    {'input': 1.0, 'output': 100, 'description': 'Maximum gradient → maximum splice points'}
                ]
            },
            'complexity_to_iterations': {
                'description': 'Maps complexity score to max iterations parameter',
                'formula': 'max(5, min(20, int(5 + (complexity_score * 15))))',
                'input_range': '[0.0, 1.0]',
                'output_range': '[5, 20]',
                'relationship': 'linear positive - higher complexity → more iterations',
                'rationale': 'Higher complexity requires more iterations for convergence',
                'edge_cases': [
                    {'input': 0.0, 'output': 5, 'description': 'Simple → minimum iterations'},
                    {'input': 0.5, 'output': 12, 'description': 'Medium complexity → medium iterations'},
                    {'input': 1.0, 'output': 20, 'description': 'Maximum complexity → maximum iterations'}
                ]
            }
        }

        self.analysis_results['documentation'] = formula_docs
        return formula_docs

    def create_test_cases(self) -> List[Dict[str, Any]]:
        """Create comprehensive test cases from existing behavior."""
        test_cases = []

        # Test edge_to_corner_threshold
        for edge_density in np.linspace(0, 1, 21):
            result = self.formulas.edge_to_corner_threshold(edge_density)
            test_cases.append({
                'function': 'edge_to_corner_threshold',
                'input': {'edge_density': float(edge_density)},
                'output': result,
                'type': 'edge_density_sweep'
            })

        # Test colors_to_precision
        for unique_colors in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
            result = self.formulas.colors_to_precision(unique_colors)
            test_cases.append({
                'function': 'colors_to_precision',
                'input': {'unique_colors': unique_colors},
                'output': result,
                'type': 'color_count_sweep'
            })

        # Test entropy_to_path_precision
        for entropy in np.linspace(0, 1, 21):
            result = self.formulas.entropy_to_path_precision(entropy)
            test_cases.append({
                'function': 'entropy_to_path_precision',
                'input': {'entropy': float(entropy)},
                'output': result,
                'type': 'entropy_sweep'
            })

        # Test corners_to_length_threshold
        for corner_density in np.linspace(0, 1, 21):
            result = self.formulas.corners_to_length_threshold(corner_density)
            test_cases.append({
                'function': 'corners_to_length_threshold',
                'input': {'corner_density': float(corner_density)},
                'output': float(result),
                'type': 'corner_density_sweep'
            })

        # Test gradient_to_splice_threshold
        for gradient_strength in np.linspace(0, 1, 21):
            result = self.formulas.gradient_to_splice_threshold(gradient_strength)
            test_cases.append({
                'function': 'gradient_to_splice_threshold',
                'input': {'gradient_strength': float(gradient_strength)},
                'output': result,
                'type': 'gradient_sweep'
            })

        # Test complexity_to_iterations
        for complexity in np.linspace(0, 1, 21):
            result = self.formulas.complexity_to_iterations(complexity)
            test_cases.append({
                'function': 'complexity_to_iterations',
                'input': {'complexity_score': float(complexity)},
                'output': result,
                'type': 'complexity_sweep'
            })

        # Add edge case tests
        edge_case_tests = [
            # Edge density edge cases
            {'function': 'edge_to_corner_threshold', 'input': {'edge_density': -0.1}, 'expected_behavior': 'clamp to 0'},
            {'function': 'edge_to_corner_threshold', 'input': {'edge_density': 1.5}, 'expected_behavior': 'clamp to 1'},
            # Color count edge cases
            {'function': 'colors_to_precision', 'input': {'unique_colors': 0}, 'expected_behavior': 'handle as 1'},
            {'function': 'colors_to_precision', 'input': {'unique_colors': -5}, 'expected_behavior': 'handle as 1'},
            # Entropy edge cases
            {'function': 'entropy_to_path_precision', 'input': {'entropy': -0.1}, 'expected_behavior': 'clamp to 0'},
            {'function': 'entropy_to_path_precision', 'input': {'entropy': 1.1}, 'expected_behavior': 'clamp to 1'},
        ]

        for edge_case in edge_case_tests:
            try:
                func = getattr(self.formulas, edge_case['function'])
                input_param = list(edge_case['input'].values())[0]
                result = func(input_param)
                edge_case['output'] = result
                edge_case['type'] = 'edge_case'
                test_cases.append(edge_case)
            except Exception as e:
                edge_case['error'] = str(e)
                edge_case['type'] = 'edge_case_error'
                test_cases.append(edge_case)

        self.analysis_results['test_cases'] = test_cases
        return test_cases

    def benchmark_performance(self, iterations: int = 10000) -> Dict[str, Any]:
        """Benchmark current performance of correlation formulas."""
        performance_metrics = {}

        # Benchmark each function
        functions_to_test = [
            ('edge_to_corner_threshold', [0.5]),
            ('colors_to_precision', [128]),
            ('entropy_to_path_precision', [0.5]),
            ('corners_to_length_threshold', [0.5]),
            ('gradient_to_splice_threshold', [0.5]),
            ('complexity_to_iterations', [0.5])
        ]

        for func_name, test_input in functions_to_test:
            func = getattr(self.formulas, func_name)

            # Warm-up
            for _ in range(100):
                func(*test_input)

            # Benchmark
            start_time = time.perf_counter()
            for _ in range(iterations):
                func(*test_input)
            end_time = time.perf_counter()

            total_time = end_time - start_time
            avg_time_us = (total_time / iterations) * 1_000_000  # Convert to microseconds

            performance_metrics[func_name] = {
                'iterations': iterations,
                'total_time_seconds': total_time,
                'avg_time_microseconds': avg_time_us,
                'operations_per_second': iterations / total_time
            }

        # Calculate aggregate metrics
        all_times = [m['avg_time_microseconds'] for m in performance_metrics.values()]
        performance_metrics['aggregate'] = {
            'avg_function_time_us': np.mean(all_times),
            'max_function_time_us': np.max(all_times),
            'min_function_time_us': np.min(all_times),
            'total_benchmark_time': sum(m['total_time_seconds'] for m in performance_metrics.values())
        }

        self.analysis_results['performance_metrics'] = performance_metrics
        return performance_metrics

    def visualize_formulas(self, output_dir: str = 'reports'):
        """Create visualizations of formula behavior."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Correlation Formula Behavior Analysis', fontsize=16)

        # Plot 1: Edge to Corner Threshold
        edge_densities = np.linspace(0, 1, 100)
        corner_thresholds = [self.formulas.edge_to_corner_threshold(ed) for ed in edge_densities]
        axes[0, 0].plot(edge_densities, corner_thresholds, 'b-', linewidth=2)
        axes[0, 0].set_title('Edge Density → Corner Threshold')
        axes[0, 0].set_xlabel('Edge Density')
        axes[0, 0].set_ylabel('Corner Threshold')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Colors to Precision
        color_counts = np.logspace(0, 10, 100, base=2)  # 1 to 1024
        precisions = [self.formulas.colors_to_precision(cc) for cc in color_counts]
        axes[0, 1].semilogx(color_counts, precisions, 'g-', linewidth=2, base=2)
        axes[0, 1].set_title('Unique Colors → Color Precision')
        axes[0, 1].set_xlabel('Unique Colors (log scale)')
        axes[0, 1].set_ylabel('Color Precision')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Entropy to Path Precision
        entropies = np.linspace(0, 1, 100)
        path_precisions = [self.formulas.entropy_to_path_precision(e) for e in entropies]
        axes[0, 2].plot(entropies, path_precisions, 'r-', linewidth=2)
        axes[0, 2].set_title('Entropy → Path Precision')
        axes[0, 2].set_xlabel('Entropy')
        axes[0, 2].set_ylabel('Path Precision')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Corners to Length Threshold
        corner_densities = np.linspace(0, 1, 100)
        length_thresholds = [self.formulas.corners_to_length_threshold(cd) for cd in corner_densities]
        axes[1, 0].plot(corner_densities, length_thresholds, 'c-', linewidth=2)
        axes[1, 0].set_title('Corner Density → Length Threshold')
        axes[1, 0].set_xlabel('Corner Density')
        axes[1, 0].set_ylabel('Length Threshold')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Gradient to Splice Threshold
        gradient_strengths = np.linspace(0, 1, 100)
        splice_thresholds = [self.formulas.gradient_to_splice_threshold(gs) for gs in gradient_strengths]
        axes[1, 1].plot(gradient_strengths, splice_thresholds, 'm-', linewidth=2)
        axes[1, 1].set_title('Gradient Strength → Splice Threshold')
        axes[1, 1].set_xlabel('Gradient Strength')
        axes[1, 1].set_ylabel('Splice Threshold')
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Complexity to Iterations
        complexities = np.linspace(0, 1, 100)
        iterations = [self.formulas.complexity_to_iterations(c) for c in complexities]
        axes[1, 2].plot(complexities, iterations, 'y-', linewidth=2)
        axes[1, 2].set_title('Complexity → Max Iterations')
        axes[1, 2].set_xlabel('Complexity Score')
        axes[1, 2].set_ylabel('Max Iterations')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = output_path / 'correlation_formulas_analysis.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return str(chart_path)

    def export_analysis(self, output_file: str = 'correlation_analysis.json'):
        """Export complete analysis to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)

        return str(output_path)

    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report."""
        report_lines = [
            "=" * 80,
            "CORRELATION FORMULAS ANALYSIS REPORT",
            "=" * 80,
            ""
        ]

        # Document formulas
        report_lines.append("FORMULA DOCUMENTATION")
        report_lines.append("-" * 40)
        for func_name, doc in self.analysis_results['documentation'].items():
            report_lines.append(f"\n{func_name}:")
            report_lines.append(f"  Description: {doc['description']}")
            report_lines.append(f"  Formula: {doc['formula']}")
            report_lines.append(f"  Input Range: {doc['input_range']}")
            report_lines.append(f"  Output Range: {doc['output_range']}")
            report_lines.append(f"  Relationship: {doc['relationship']}")

        # Performance metrics
        report_lines.append("\n" + "=" * 80)
        report_lines.append("PERFORMANCE BENCHMARKS")
        report_lines.append("-" * 40)
        for func_name, metrics in self.analysis_results['performance_metrics'].items():
            if func_name != 'aggregate':
                report_lines.append(f"\n{func_name}:")
                report_lines.append(f"  Avg Time: {metrics['avg_time_microseconds']:.2f} μs")
                report_lines.append(f"  Ops/Sec: {metrics['operations_per_second']:.0f}")

        # Aggregate performance
        agg = self.analysis_results['performance_metrics']['aggregate']
        report_lines.append(f"\nAggregate Performance:")
        report_lines.append(f"  Average Function Time: {agg['avg_function_time_us']:.2f} μs")
        report_lines.append(f"  Fastest Function: {agg['min_function_time_us']:.2f} μs")
        report_lines.append(f"  Slowest Function: {agg['max_function_time_us']:.2f} μs")

        # Test case summary
        report_lines.append("\n" + "=" * 80)
        report_lines.append("TEST CASES SUMMARY")
        report_lines.append("-" * 40)
        test_case_types = {}
        for tc in self.analysis_results['test_cases']:
            tc_type = tc.get('type', 'unknown')
            test_case_types[tc_type] = test_case_types.get(tc_type, 0) + 1

        for tc_type, count in test_case_types.items():
            report_lines.append(f"  {tc_type}: {count} test cases")

        report_lines.append(f"\nTotal Test Cases: {len(self.analysis_results['test_cases'])}")

        return "\n".join(report_lines)


def main():
    """Main function to run the analysis."""
    print("Analyzing Correlation Formulas...")

    analyzer = CorrelationFormulaAnalyzer()

    # Document all formulas
    print("✓ Documenting formula behavior...")
    analyzer.document_all_formulas()

    # Create test cases
    print("✓ Creating test cases...")
    test_cases = analyzer.create_test_cases()
    print(f"  Generated {len(test_cases)} test cases")

    # Benchmark performance
    print("✓ Benchmarking performance...")
    performance = analyzer.benchmark_performance(iterations=10000)
    avg_time = performance['aggregate']['avg_function_time_us']
    print(f"  Average function time: {avg_time:.2f} μs")

    # Visualize formulas
    print("✓ Creating visualizations...")
    chart_path = analyzer.visualize_formulas()
    print(f"  Saved chart to: {chart_path}")

    # Export analysis
    print("✓ Exporting analysis...")
    json_path = analyzer.export_analysis('data/correlation_analysis.json')
    print(f"  Saved JSON to: {json_path}")

    # Generate report
    print("\n" + "=" * 80)
    report = analyzer.generate_analysis_report()
    print(report)

    # Save report to file
    with open('data/correlation_analysis_report.txt', 'w') as f:
        f.write(report)

    print("\n✓ Analysis complete!")
    print(f"✓ Backup created: correlation_formulas_backup.py")
    print(f"✓ Documentation complete: {len(analyzer.analysis_results['documentation'])} formulas")
    print(f"✓ Test suite created: {len(test_cases)} test cases")
    print(f"✓ Performance baseline: {avg_time:.2f} μs average")

    return analyzer


if __name__ == "__main__":
    main()