#!/usr/bin/env python3
"""
PNG to SVG Converter CLI Tool

Usage:
    python convert.py input.png
    python convert.py input.png -o output.svg
    python convert.py input.png --optimize-logo
"""

import click
import time
import os
from pathlib import Path
from converters.vtracer_converter import VTracerConverter
from utils.metrics import ConversionMetrics
from utils.preprocessor import ImagePreprocessor


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default=None, help='Output SVG file path')
@click.option('--optimize-logo', is_flag=True, help='Use logo-optimized settings')
@click.option('--color-precision', default=6, help='Color precision (1-10)')
@click.option('--preprocess', is_flag=True, help='Apply preprocessing')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
def convert(input_path, output, optimize_logo, color_precision, preprocess, verbose):
    """Convert PNG image to SVG format."""

    # Validate input
    if not input_path.lower().endswith('.png'):
        click.echo(click.style("✗ Error: Input must be a PNG file", fg='red'))
        return

    # Determine output path
    if output is None:
        output = input_path.replace('.png', '.svg')

    click.echo(f"Converting {click.style(input_path, fg='blue')}...")

    # Preprocess if requested
    if preprocess:
        click.echo("  → Preprocessing image...")
        img = ImagePreprocessor.prepare_logo(input_path)
        temp_path = "temp_preprocessed.png"
        img.save(temp_path)
        input_to_convert = temp_path
    else:
        input_to_convert = input_path

    # Initialize converter
    converter = VTracerConverter(color_precision=color_precision)

    # Convert
    start_time = time.time()

    try:
        if optimize_logo:
            click.echo("  → Using logo-optimized settings...")
            svg_content = converter.optimize_for_logos(input_to_convert)
        else:
            svg_content = converter.convert(input_to_convert)

        conversion_time = time.time() - start_time

        # Save output
        with open(output, 'w') as f:
            f.write(svg_content)

        # Clean up temp file if used
        if preprocess and os.path.exists("temp_preprocessed.png"):
            os.remove("temp_preprocessed.png")

        # Calculate metrics
        metrics = ConversionMetrics.calculate_basic(
            input_path, output, conversion_time
        )

        # Display results
        click.echo(click.style(f"✓ Converted successfully!", fg='green'))
        click.echo(f"  → Output: {click.style(output, fg='blue')}")
        click.echo(f"  → Time: {metrics['conversion_time_s']}s")
        click.echo(f"  → Size: {metrics['png_size_kb']}KB → {metrics['svg_size_kb']}KB "
                  f"({metrics['size_reduction_pct']:.1f}% reduction)")

        if verbose:
            complexity = ConversionMetrics.estimate_svg_complexity(svg_content)
            click.echo(f"  → SVG Complexity:")
            click.echo(f"    • Paths: {complexity['paths']}")
            click.echo(f"    • Commands: {complexity['commands']}")
            click.echo(f"    • Groups: {complexity['groups']}")
            click.echo(f"    • Colors: {complexity['colors']}")

    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg='red'))
        if verbose:
            import traceback
            click.echo(traceback.format_exc())


if __name__ == '__main__':
    convert()