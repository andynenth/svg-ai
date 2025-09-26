#!/usr/bin/env python3
"""
Batch optimization of PNG to SVG conversion with quality targets.
"""

import click
import json
import time
from pathlib import Path
from typing import Dict, List
import concurrent.futures
from optimize_iterative import IterativeOptimizer
from utils.visual_compare import VisualComparer
from tqdm import tqdm


class BatchOptimizer:
    """Batch process multiple PNG files with optimization."""

    def __init__(self, target_ssim: float = 0.85, max_iterations: int = 10):
        self.target_ssim = target_ssim
        self.max_iterations = max_iterations
        self.results = []

    def process_file(self, png_path: Path, output_dir: Path,
                    save_comparison: bool = False) -> Dict:
        """Process a single PNG file."""
        start_time = time.time()

        try:
            # Initialize optimizer for this file
            optimizer = IterativeOptimizer(
                target_ssim=self.target_ssim,
                max_iterations=self.max_iterations
            )

            # Run optimization (verbose=False for batch)
            result = optimizer.optimize(str(png_path), verbose=False)

            # Save optimized SVG
            svg_path = output_dir / f"{png_path.stem}.optimized.svg"
            with open(svg_path, 'w') as f:
                f.write(result['best_svg'])

            # Save comparison image if requested
            comparison_path = None
            if save_comparison and result['best_ssim'] > 0:
                try:
                    comparer = VisualComparer()
                    grid = comparer.create_comparison_grid(
                        str(png_path),
                        result['best_svg']
                    )
                    comparison_path = output_dir / f"{png_path.stem}.comparison.png"
                    grid.save(str(comparison_path))
                except Exception as e:
                    print(f"  Warning: Could not create comparison for {png_path.name}: {e}")

            # Compile results
            file_result = {
                'file': str(png_path),
                'success': result['success'],
                'ssim': result['best_ssim'],
                'logo_type': result['logo_type'],
                'iterations': result['iterations'],
                'processing_time': time.time() - start_time,
                'svg_path': str(svg_path),
                'comparison_path': str(comparison_path) if comparison_path else None,
                'best_params': result['best_params'],
                'metrics': result.get('best_metrics', {})
            }

            return file_result

        except Exception as e:
            return {
                'file': str(png_path),
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    def process_directory(self, input_dir: Path, output_dir: Path,
                         pattern: str = "*.png", parallel: int = 1,
                         save_comparisons: bool = False) -> List[Dict]:
        """Process all PNG files in a directory."""
        # Find all PNG files
        png_files = list(input_dir.glob(pattern))
        if not png_files:
            click.echo(f"No files found matching pattern: {pattern}")
            return []

        click.echo(f"Found {len(png_files)} files to process")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process files
        if parallel > 1:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(self.process_file, png_file, output_dir, save_comparisons): png_file
                    for png_file in png_files
                }

                with tqdm(total=len(png_files), desc="Optimizing") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        self.results.append(result)
                        pbar.update(1)

                        # Show quick status
                        if result['success']:
                            pbar.set_postfix({'SSIM': f"{result['ssim']:.3f}"})
        else:
            # Sequential processing
            for png_file in tqdm(png_files, desc="Optimizing"):
                result = self.process_file(png_file, output_dir, save_comparisons)
                self.results.append(result)

        return self.results

    def generate_report(self, output_path: Path):
        """Generate optimization report."""
        if not self.results:
            return

        # Calculate statistics
        successful = [r for r in self.results if r.get('success', False)]
        failed = [r for r in self.results if not r.get('success', False)]

        avg_ssim = sum(r.get('ssim', 0) for r in successful) / len(successful) if successful else 0
        avg_time = sum(r.get('processing_time', 0) for r in successful) / len(successful) if successful else 0
        avg_iterations = sum(r.get('iterations', 0) for r in successful) / len(successful) if successful else 0

        # Group by logo type
        type_stats = {}
        for r in successful:
            logo_type = r.get('logo_type', 'unknown')
            if logo_type not in type_stats:
                type_stats[logo_type] = {'count': 0, 'ssim_sum': 0}
            type_stats[logo_type]['count'] += 1
            type_stats[logo_type]['ssim_sum'] += r.get('ssim', 0)

        # Generate report
        report = {
            'summary': {
                'total_files': len(self.results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(self.results) * 100,
                'target_ssim': self.target_ssim,
                'max_iterations': self.max_iterations
            },
            'performance': {
                'average_ssim': avg_ssim,
                'average_time': avg_time,
                'average_iterations': avg_iterations,
                'total_time': sum(r.get('processing_time', 0) for r in self.results)
            },
            'by_type': {
                logo_type: {
                    'count': stats['count'],
                    'average_ssim': stats['ssim_sum'] / stats['count']
                }
                for logo_type, stats in type_stats.items()
            },
            'quality_distribution': {
                'excellent (>0.95)': len([r for r in successful if r.get('ssim', 0) > 0.95]),
                'good (0.90-0.95)': len([r for r in successful if 0.90 <= r.get('ssim', 0) <= 0.95]),
                'acceptable (0.85-0.90)': len([r for r in successful if 0.85 <= r.get('ssim', 0) < 0.90]),
                'poor (<0.85)': len([r for r in successful if r.get('ssim', 0) < 0.85])
            },
            'failures': [
                {'file': r['file'], 'error': r.get('error', 'Unknown error')}
                for r in failed
            ],
            'detailed_results': self.results
        }

        # Save JSON report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        click.echo("\n" + "="*60)
        click.echo(click.style("BATCH OPTIMIZATION REPORT", fg='cyan', bold=True))
        click.echo("="*60)
        click.echo(f"Total Files: {report['summary']['total_files']}")
        click.echo(f"Successful: {report['summary']['successful']} "
                  f"({report['summary']['success_rate']:.1f}%)")
        click.echo(f"Failed: {report['summary']['failed']}")
        click.echo(f"\nAverage SSIM: {report['performance']['average_ssim']:.4f}")
        click.echo(f"Average Time: {report['performance']['average_time']:.2f}s")
        click.echo(f"Total Time: {report['performance']['total_time']:.2f}s")

        click.echo("\nQuality Distribution:")
        for quality, count in report['quality_distribution'].items():
            click.echo(f"  {quality}: {count}")

        click.echo("\nBy Logo Type:")
        for logo_type, stats in report['by_type'].items():
            click.echo(f"  {logo_type}: {stats['count']} files, "
                      f"avg SSIM: {stats['average_ssim']:.4f}")

        if failed:
            click.echo(click.style(f"\n⚠️  {len(failed)} files failed:", fg='yellow'))
            for failure in report['failures'][:5]:  # Show first 5 failures
                click.echo(f"  - {Path(failure['file']).name}: {failure['error']}")

        click.echo(f"\nFull report saved to: {output_path}")


@click.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--target-ssim', default=0.85, help='Target SSIM quality (0-1)')
@click.option('--max-iterations', default=10, help='Max optimization iterations per file')
@click.option('--parallel', '-p', default=1, help='Number of parallel workers')
@click.option('--pattern', default='*.png', help='File pattern to match')
@click.option('--save-comparisons', is_flag=True, help='Save visual comparison images')
@click.option('--report', type=click.Path(), help='Save detailed report to JSON file')
def main(input_dir, output, target_ssim, max_iterations, parallel, pattern,
         save_comparisons, report):
    """
    Batch optimize PNG to SVG conversions.

    Processes all PNG files in INPUT_DIR and saves optimized SVGs to output directory.
    Each file is automatically optimized to reach the target quality level.

    Examples:

        # Basic batch optimization
        python batch_optimize.py data/logos

        # With quality target and parallel processing
        python batch_optimize.py data/logos --target-ssim 0.90 --parallel 4

        # Save visual comparisons and report
        python batch_optimize.py data/logos --save-comparisons --report results.json
    """

    input_path = Path(input_dir)
    output_path = Path(output) if output else input_path / 'optimized'

    click.echo(f"🚀 Starting batch optimization")
    click.echo(f"   Input: {input_path}")
    click.echo(f"   Output: {output_path}")
    click.echo(f"   Target SSIM: {target_ssim}")
    click.echo(f"   Max iterations: {max_iterations}")
    click.echo(f"   Parallel workers: {parallel}")

    # Initialize batch optimizer
    optimizer = BatchOptimizer(
        target_ssim=target_ssim,
        max_iterations=max_iterations
    )

    # Process directory
    start_time = time.time()
    results = optimizer.process_directory(
        input_path,
        output_path,
        pattern=pattern,
        parallel=parallel,
        save_comparisons=save_comparisons
    )

    total_time = time.time() - start_time

    # Generate report if requested
    if report:
        report_path = Path(report)
        optimizer.generate_report(report_path)
    else:
        # Print basic summary
        successful = len([r for r in results if r.get('success', False)])
        click.echo(f"\n✅ Completed: {successful}/{len(results)} files")
        click.echo(f"   Total time: {total_time:.2f}s")

        if successful > 0:
            avg_ssim = sum(r.get('ssim', 0) for r in results if r.get('success', False)) / successful
            click.echo(f"   Average SSIM: {avg_ssim:.4f}")


if __name__ == '__main__':
    main()