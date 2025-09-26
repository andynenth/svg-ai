#!/usr/bin/env python3
"""
PNG to SVG converter using Potrace (easier to install than VTracer).
"""

import click
import time
import os
from pathlib import Path
from converters.potrace_converter import PotraceConverter
from utils.metrics import ConversionMetrics


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default=None, help='Output SVG file path')
@click.option('--threshold', default=128, help='Black/white threshold (0-255)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
def convert(input_path, output, threshold, verbose):
    """Convert PNG image to SVG using Potrace."""

    # Validate input
    if not input_path.lower().endswith('.png'):
        click.echo(click.style("✗ Error: Input must be a PNG file", fg='red'))
        return

    # Determine output path
    if output is None:
        output = input_path.replace('.png', '_potrace.svg')

    click.echo(f"Converting {click.style(input_path, fg='blue')}...")

    # Check if potrace is installed
    import subprocess
    try:
        result = subprocess.run(['which', 'potrace'], capture_output=True)
        if result.returncode != 0:
            click.echo(click.style("⚠️  Potrace not installed!", fg='yellow'))
            click.echo("\nInstall Potrace with:")
            click.echo("  brew install potrace")
            click.echo("\nOr download from: http://potrace.sourceforge.net")
            return
    except:
        pass

    # Initialize converter
    converter = PotraceConverter()

    # Convert
    start_time = time.time()

    try:
        svg_content = converter.convert(input_path, threshold=threshold)
        conversion_time = time.time() - start_time

        # Save output
        with open(output, 'w') as f:
            f.write(svg_content)

        # Calculate metrics
        metrics = ConversionMetrics.calculate_basic(
            input_path, output, conversion_time
        )

        # Display results
        click.echo(click.style(f"✓ Conversion complete!", fg='green'))
        click.echo(f"  → Output: {click.style(output, fg='blue')}")
        click.echo(f"  → Time: {metrics['conversion_time_s']}s")
        click.echo(f"  → Size: {metrics['png_size_kb']}KB → {metrics['svg_size_kb']}KB")

        if verbose:
            click.echo(f"\nNote: Potrace converts to black & white only.")
            click.echo(f"Threshold used: {threshold}")

    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg='red'))
        if verbose:
            import traceback
            click.echo(traceback.format_exc())


if __name__ == '__main__':
    convert()