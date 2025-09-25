#!/usr/bin/env python3
"""
Batch conversion tool for PNG to SVG.
"""

import click
import os
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from converters.vtracer_converter import VTracerConverter
from utils.parallel_processor import BatchProcessor, ParallelProcessor
from utils.cache import HybridCache
from utils.quality_metrics import ComprehensiveMetrics


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='output', help='Output directory')
@click.option('--parallel', '-p', default=4, help='Number of parallel workers')
@click.option('--recursive', '-r', is_flag=True, help='Process subdirectories')
@click.option('--pattern', default='*.png', help='File pattern to match')
@click.option('--color-precision', default=6, help='Color precision (1-10)')
@click.option('--optimize-logo', is_flag=True, help='Use logo optimization')
@click.option('--use-cache', is_flag=True, help='Use caching')
@click.option('--chunk-size', default=10, help='Processing chunk size')
@click.option('--report', is_flag=True, help='Generate report')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def batch_convert(input_dir, output_dir, parallel, recursive, pattern,
                 color_precision, optimize_logo, use_cache, chunk_size,
                 report, verbose):
    """
    Batch convert PNG files to SVG format.

    INPUT_DIR: Directory containing PNG files to convert
    """
    # Setup
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Find images
    if recursive:
        image_files = list(input_path.rglob(pattern))
    else:
        image_files = list(input_path.glob(pattern))

    if not image_files:
        click.echo(f"No files matching '{pattern}' found in {input_dir}")
        return

    click.echo(f"🎨 Found {len(image_files)} images to convert")
    click.echo(f"📁 Output directory: {output_dir}")
    click.echo(f"⚡ Using {parallel} parallel workers")

    # Initialize components
    converter = VTracerConverter(color_precision=color_precision)
    cache = HybridCache() if use_cache else None
    metrics_calc = ComprehensiveMetrics()

    # Process images
    if parallel > 1:
        # Use batch processor for parallel processing
        processor = BatchProcessor(chunk_size=chunk_size, max_workers=parallel)

        click.echo("\n🚀 Starting parallel batch conversion...")
        results = processor.process_large_batch(
            [str(f) for f in image_files],
            converter,
            output_dir=output_dir,
            retry_failed=True
        )

        # Get summary
        summary = processor.get_summary()

    else:
        # Sequential processing with progress bar
        click.echo("\n🚀 Starting sequential conversion...")
        results = {}
        successful = []
        failed = []

        with tqdm(total=len(image_files), desc="Converting") as pbar:
            for image_file in image_files:
                try:
                    # Check cache
                    if cache:
                        cached = cache.get(str(image_file), converter.get_name())
                        if cached:
                            svg_content = cached
                            from_cache = True
                        else:
                            svg_content = converter.convert(str(image_file))
                            cache.set(str(image_file), converter.get_name(), svg_content)
                            from_cache = False
                    else:
                        svg_content = converter.convert(str(image_file))
                        from_cache = False

                    # Save output
                    output_file = output_path / f"{image_file.stem}.svg"
                    with open(output_file, 'w') as f:
                        f.write(svg_content)

                    results[str(image_file)] = {
                        'status': 'success',
                        'output': str(output_file),
                        'cached': from_cache
                    }
                    successful.append(str(image_file))

                    if verbose:
                        status = "📦 (cached)" if from_cache else "✅"
                        tqdm.write(f"{status} {image_file.name}")

                except Exception as e:
                    results[str(image_file)] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    failed.append(str(image_file))

                    if verbose:
                        tqdm.write(f"❌ {image_file.name}: {e}")

                pbar.update(1)

        summary = {
            'total_processed': len(image_files),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(image_files)
        }

    # Display summary
    click.echo("\n" + "=" * 50)
    click.echo("📊 CONVERSION COMPLETE")
    click.echo("=" * 50)
    click.echo(f"✅ Successful: {summary['successful']}")
    click.echo(f"❌ Failed: {summary['failed']}")
    click.echo(f"📈 Success Rate: {summary['success_rate']:.1%}")

    if cache:
        cache_stats = cache.get_stats()
        if 'memory' in cache_stats:
            click.echo(f"💾 Cache Hit Rate: {cache_stats['memory']['hit_rate']:.1%}")

    # Generate report if requested
    if report:
        generate_report(results, summary, output_path)


def generate_report(results, summary, output_path):
    """Generate conversion report."""
    report_path = output_path / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    report_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'details': results
    }

    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    click.echo(f"\n📄 Report saved to: {report_path}")


if __name__ == '__main__':
    batch_convert()