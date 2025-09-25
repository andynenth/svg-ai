#!/usr/bin/env python3
"""
Comprehensive benchmark system for PNG to SVG conversion.
"""

import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import click
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from converters.vtracer_converter import VTracerConverter
from utils.quality_metrics import ComprehensiveMetrics
from utils.preprocessor import ImagePreprocessor


class BenchmarkRunner:
    """Run benchmarks on PNG to SVG converters."""

    def __init__(self, output_dir: str = 'results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics_calculator = ComprehensiveMetrics()
        self.results = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def run_single_conversion(self, png_path: str, converter,
                            save_output: bool = True) -> Dict[str, Any]:
        """
        Run a single conversion and collect metrics.

        Args:
            png_path: Path to PNG file
            converter: Converter instance
            save_output: Whether to save SVG output

        Returns:
            Dictionary with conversion results
        """
        png_path = Path(png_path)
        category = png_path.parent.name
        filename = png_path.name

        result = {
            'file': str(png_path),
            'filename': filename,
            'category': category,
            'converter': converter.get_name(),
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Run conversion
            start_time = time.time()
            svg_content = converter.convert(str(png_path))
            conversion_time = time.time() - start_time

            # Save output if requested
            if save_output:
                output_path = self.output_dir / f"{png_path.stem}_{converter.__class__.__name__}.svg"
                with open(output_path, 'w') as f:
                    f.write(svg_content)
                result['output_path'] = str(output_path)

            # Calculate metrics
            metrics = self.metrics_calculator.evaluate(
                str(png_path), svg_content, conversion_time
            )

            result.update({
                'status': 'success',
                'metrics': metrics,
                'svg_length': len(svg_content)
            })

        except Exception as e:
            result.update({
                'status': 'failed',
                'error': str(e),
                'metrics': None
            })

        return result

    def run_category(self, category_path: Path, converter) -> List[Dict[str, Any]]:
        """Run benchmark on all files in a category."""
        results = []
        png_files = list(category_path.glob('*.png'))

        with tqdm(total=len(png_files), desc=f"Processing {category_path.name}") as pbar:
            for png_file in png_files:
                result = self.run_single_conversion(png_file, converter, save_output=False)
                results.append(result)
                pbar.update(1)

        return results

    def run_full_benchmark(self, test_dir: str = 'data/logos',
                          converters: Optional[List] = None) -> Dict[str, Any]:
        """
        Run complete benchmark on all test images.

        Args:
            test_dir: Directory containing test images
            converters: List of converter instances to test

        Returns:
            Benchmark results dictionary
        """
        if converters is None:
            converters = [
                VTracerConverter(color_precision=4),
                VTracerConverter(color_precision=6),
                VTracerConverter(color_precision=8),
            ]

        test_path = Path(test_dir)
        all_results = []

        print(f"\n🚀 Running benchmark on {test_dir}")
        print(f"   Converters: {len(converters)}")
        print(f"   Categories: {len(list(test_path.iterdir()))}")
        print("-" * 50)

        for converter in converters:
            print(f"\n📊 Testing {converter.get_name()}")

            for category_dir in sorted(test_path.iterdir()):
                if category_dir.is_dir() and not category_dir.name.startswith('.'):
                    category_results = self.run_category(category_dir, converter)
                    all_results.extend(category_results)

        self.results = all_results
        return self._analyze_results(all_results)

    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results."""
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']

        if not successful:
            return {
                'summary': {
                    'total_tests': len(results),
                    'successful': 0,
                    'failed': len(results),
                    'success_rate': 0
                }
            }

        # Calculate aggregate metrics
        analysis = {
            'summary': {
                'total_tests': len(results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(results)
            },
            'by_category': {},
            'by_converter': {},
            'performance': {
                'avg_conversion_time': sum(r['metrics']['performance']['conversion_time_s']
                                          for r in successful) / len(successful),
                'min_conversion_time': min(r['metrics']['performance']['conversion_time_s']
                                          for r in successful),
                'max_conversion_time': max(r['metrics']['performance']['conversion_time_s']
                                          for r in successful)
            }
        }

        # Analyze by category
        categories = set(r['category'] for r in results)
        for category in categories:
            cat_results = [r for r in successful if r['category'] == category]
            if cat_results:
                visual_metrics = [r['metrics'].get('visual', {}) for r in cat_results
                                 if 'visual' in r['metrics']]

                if visual_metrics and 'ssim' in visual_metrics[0]:
                    analysis['by_category'][category] = {
                        'count': len(cat_results),
                        'avg_ssim': sum(m.get('ssim', 0) for m in visual_metrics) / len(visual_metrics),
                        'avg_time': sum(r['metrics']['performance']['conversion_time_s']
                                      for r in cat_results) / len(cat_results),
                        'success_rate': len(cat_results) / len([r for r in results
                                                               if r['category'] == category])
                    }

        # Analyze by converter
        converters = set(r['converter'] for r in results)
        for converter in converters:
            conv_results = [r for r in successful if r['converter'] == converter]
            if conv_results:
                visual_metrics = [r['metrics'].get('visual', {}) for r in conv_results
                                 if 'visual' in r['metrics']]

                if visual_metrics and 'ssim' in visual_metrics[0]:
                    analysis['by_converter'][converter] = {
                        'count': len(conv_results),
                        'avg_ssim': sum(m.get('ssim', 0) for m in visual_metrics) / len(visual_metrics),
                        'avg_time': sum(r['metrics']['performance']['conversion_time_s']
                                      for r in conv_results) / len(conv_results),
                        'success_rate': len(conv_results) / len([r for r in results
                                                                if r['converter'] == converter])
                    }

        return analysis

    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to file."""
        if filename is None:
            filename = f"benchmark_{self.timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"💾 Results saved to {output_path}")

    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate markdown report from analysis."""
        report = f"""# Benchmark Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Tests**: {analysis['summary']['total_tests']}
- **Successful**: {analysis['summary']['successful']}
- **Failed**: {analysis['summary']['failed']}
- **Success Rate**: {analysis['summary']['success_rate']:.1%}

## Performance
- **Average Conversion Time**: {analysis['performance']['avg_conversion_time']:.3f}s
- **Min Conversion Time**: {analysis['performance']['min_conversion_time']:.3f}s
- **Max Conversion Time**: {analysis['performance']['max_conversion_time']:.3f}s

## Results by Category
| Category | Count | Avg SSIM | Avg Time (s) | Success Rate |
|----------|-------|----------|--------------|--------------|
"""

        for category, stats in sorted(analysis['by_category'].items()):
            if 'avg_ssim' in stats:
                report += f"| {category} | {stats['count']} | {stats['avg_ssim']:.3f} | "
                report += f"{stats['avg_time']:.3f} | {stats['success_rate']:.1%} |\n"

        report += "\n## Results by Converter\n"
        report += "| Converter | Count | Avg SSIM | Avg Time (s) | Success Rate |\n"
        report += "|-----------|-------|----------|--------------|--------------|

        for converter, stats in sorted(analysis['by_converter'].items()):
            if 'avg_ssim' in stats:
                report += f"| {converter} | {stats['count']} | {stats['avg_ssim']:.3f} | "
                report += f"{stats['avg_time']:.3f} | {stats['success_rate']:.1%} |\n"

        return report

    def save_report(self, analysis: Dict[str, Any], filename: Optional[str] = None):
        """Save markdown report."""
        if filename is None:
            filename = f"report_{self.timestamp}.md"

        report = self.generate_report(analysis)
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"📄 Report saved to {output_path}")


@click.command()
@click.option('--test-dir', default='data/logos', help='Directory with test images')
@click.option('--output-dir', default='results', help='Output directory for results')
@click.option('--save-svg', is_flag=True, help='Save converted SVG files')
@click.option('--report', is_flag=True, help='Generate markdown report')
@click.option('--quick', is_flag=True, help='Quick test with fewer images')
def main(test_dir, output_dir, save_svg, report, quick):
    """Run PNG to SVG conversion benchmark."""

    runner = BenchmarkRunner(output_dir)

    # Configure converters to test
    converters = [
        VTracerConverter(color_precision=4, layer_difference=32),  # Fast, lower quality
        VTracerConverter(color_precision=6, layer_difference=16),  # Balanced
        VTracerConverter(color_precision=8, layer_difference=8),   # High quality
    ]

    if quick:
        # Quick test with just one converter and limited images
        converters = [VTracerConverter(color_precision=6)]
        print("⚡ Running quick benchmark (limited dataset)")

    # Run benchmark
    analysis = runner.run_full_benchmark(test_dir, converters)

    # Save results
    runner.save_results()

    # Generate report if requested
    if report:
        runner.save_report(analysis)

    # Print summary
    print("\n" + "=" * 50)
    print("📊 BENCHMARK COMPLETE")
    print("=" * 50)
    print(f"Success Rate: {analysis['summary']['success_rate']:.1%}")
    print(f"Avg Time: {analysis['performance']['avg_conversion_time']:.3f}s")

    if 'by_category' in analysis:
        print("\nTop Performing Categories:")
        sorted_cats = sorted(analysis['by_category'].items(),
                           key=lambda x: x[1].get('avg_ssim', 0), reverse=True)
        for cat, stats in sorted_cats[:3]:
            if 'avg_ssim' in stats:
                print(f"  {cat}: SSIM={stats['avg_ssim']:.3f}")


if __name__ == '__main__':
    main()