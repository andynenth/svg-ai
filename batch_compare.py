#!/usr/bin/env python3
"""
Batch comparison tool for PNG to SVG conversions.

This script processes entire directories and generates comprehensive
comparison reports with aggregate statistics.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_visual_comparison import VisualComparisonGenerator
from converters.vtracer_converter import VTracerConverter
from optimize_adaptive import AdaptiveOptimizer


class BatchComparer:
    """Process and compare batches of PNG to SVG conversions."""

    def __init__(self, output_dir: str = "comparison_results"):
        """
        Initialize the batch comparer.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.generator = VisualComparisonGenerator()
        self.converter = VTracerConverter()
        self.optimizer = AdaptiveOptimizer()

    def process_single_file(self, png_path: str, svg_path: Optional[str] = None,
                          generate_visual: bool = False) -> Dict:
        """
        Process a single PNG/SVG pair.

        Args:
            png_path: Path to PNG file
            svg_path: Path to SVG file (will convert if not provided)
            generate_visual: Whether to generate visual comparison

        Returns:
            Comparison results
        """
        png_path = Path(png_path)

        # Convert if SVG not provided
        if svg_path is None:
            svg_path = self.output_dir / f"{png_path.stem}.svg"

            # Use adaptive optimization to convert
            print(f"  Converting {png_path.name}...")
            result = self.optimizer.optimize(str(png_path))

            if result['success']:
                # Convert with optimal parameters
                self.converter.convert_with_params(
                    str(png_path),
                    str(svg_path),
                    **result['final_params']
                )
            else:
                print(f"    ⚠️ Optimization failed, using defaults")
                # Fall back to defaults
                self.converter.convert_with_params(
                    str(png_path),
                    str(svg_path),
                    color_precision=6,
                    layer_difference=8,
                    corner_threshold=40
                )

        # Calculate metrics
        try:
            png_img, svg_img = self.generator.load_images(str(png_path), str(svg_path))
            metrics = self.generator.calculate_all_metrics(
                png_img, svg_img,
                str(png_path), str(svg_path)
            )

            # Generate visual comparison if requested
            if generate_visual:
                visual_path = self.output_dir / f"comparison_{png_path.stem}.png"
                self.generator.generate_comparison(
                    str(png_path), str(svg_path),
                    str(visual_path)
                )
                metrics['visual_path'] = str(visual_path)

            metrics['png_path'] = str(png_path)
            metrics['svg_path'] = str(svg_path)
            metrics['filename'] = png_path.name
            metrics['success'] = True

            return metrics

        except Exception as e:
            print(f"    ❌ Error processing {png_path.name}: {e}")
            return {
                'png_path': str(png_path),
                'svg_path': str(svg_path) if svg_path else None,
                'filename': png_path.name,
                'success': False,
                'error': str(e)
            }

    def process_directory(self, input_dir: str, pattern: str = "*.png",
                        max_files: Optional[int] = None,
                        parallel: int = 1,
                        generate_visuals: bool = False) -> List[Dict]:
        """
        Process all files in a directory.

        Args:
            input_dir: Input directory path
            pattern: File pattern to match
            max_files: Maximum number of files to process
            parallel: Number of parallel workers
            generate_visuals: Whether to generate visual comparisons

        Returns:
            List of comparison results
        """
        input_dir = Path(input_dir)

        if not input_dir.exists():
            print(f"❌ Directory not found: {input_dir}")
            return []

        # Find files
        files = list(input_dir.glob(pattern))

        if max_files:
            files = files[:max_files]

        if not files:
            print(f"❌ No files matching {pattern} in {input_dir}")
            return []

        print(f"Processing {len(files)} files from {input_dir}...")

        results = []

        if parallel > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(
                        self.process_single_file,
                        str(f),
                        None,
                        generate_visuals
                    ): f for f in files
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        if result['success']:
                            print(f"  ✅ {result['filename']}: SSIM={result['ssim']:.3f}")
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
        else:
            # Sequential processing
            for f in files:
                result = self.process_single_file(str(f), None, generate_visuals)
                results.append(result)

        return results

    def generate_report(self, results: List[Dict], output_file: str = None) -> str:
        """
        Generate HTML report from comparison results.

        Args:
            results: List of comparison results
            output_file: Output HTML file path

        Returns:
            HTML content
        """
        if output_file is None:
            output_file = self.output_dir / "comparison_report.html"

        # Calculate aggregate statistics
        successful = [r for r in results if r.get('success', False)]

        if not successful:
            print("❌ No successful comparisons to report")
            return ""

        avg_ssim = sum(r['ssim'] for r in successful) / len(successful)
        avg_psnr = sum(r['psnr'] for r in successful if r['psnr'] != float('inf')) / len(successful)
        avg_unified = sum(r['unified_score'] for r in successful) / len(successful)

        total_png_size = sum(r['png_size'] for r in successful)
        total_svg_size = sum(r['svg_size'] for r in successful)
        overall_size_change = (total_svg_size / total_png_size - 1) * 100

        # Group by quality levels
        excellent = [r for r in successful if r['ssim'] >= 0.98]
        good = [r for r in successful if 0.95 <= r['ssim'] < 0.98]
        fair = [r for r in successful if 0.85 <= r['ssim'] < 0.95]
        poor = [r for r in successful if r['ssim'] < 0.85]

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Comparison Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stat-label {{
            color: #666;
            margin-top: 10px;
        }}
        .quality-distribution {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .quality-bar {{
            display: flex;
            height: 40px;
            border-radius: 20px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .quality-segment {{
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        .excellent {{ background: #10b981; }}
        .good {{ background: #3b82f6; }}
        .fair {{ background: #f59e0b; }}
        .poor {{ background: #ef4444; }}
        .files-table {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #666;
        }}
        td {{
            padding: 12px;
            border-top: 1px solid #e5e7eb;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .metric-badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            color: white;
            margin-top: 40px;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Batch Comparison Report</h1>
            <p>Processed {len(results)} files at {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(successful)}</div>
                <div class="stat-label">Successful Conversions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_ssim:.3f}</div>
                <div class="stat-label">Average SSIM</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_psnr:.1f} dB</div>
                <div class="stat-label">Average PSNR</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{overall_size_change:+.1f}%</div>
                <div class="stat-label">Overall Size Change</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_unified:.1f}</div>
                <div class="stat-label">Average Quality Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(excellent)}</div>
                <div class="stat-label">Excellent Quality (≥0.98)</div>
            </div>
        </div>

        <div class="quality-distribution">
            <h2>Quality Distribution</h2>
            <div class="quality-bar">
                <div class="quality-segment excellent" style="width: {len(excellent)/len(successful)*100}%">
                    {len(excellent)} Excellent
                </div>
                <div class="quality-segment good" style="width: {len(good)/len(successful)*100}%">
                    {len(good)} Good
                </div>
                <div class="quality-segment fair" style="width: {len(fair)/len(successful)*100}%">
                    {len(fair)} Fair
                </div>
                <div class="quality-segment poor" style="width: {len(poor)/len(successful)*100}%">
                    {len(poor)} Poor
                </div>
            </div>
        </div>

        <div class="files-table">
            <h2>File Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>File</th>
                        <th>SSIM</th>
                        <th>PSNR</th>
                        <th>PNG Size</th>
                        <th>SVG Size</th>
                        <th>Size Change</th>
                        <th>Quality Score</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Add file rows
        for r in sorted(successful, key=lambda x: x['ssim'], reverse=True):
            ssim_class = 'excellent' if r['ssim'] >= 0.98 else 'good' if r['ssim'] >= 0.95 else 'fair' if r['ssim'] >= 0.85 else 'poor'

            html += f"""
                    <tr>
                        <td>{r['filename']}</td>
                        <td><span class="metric-badge {ssim_class}">{r['ssim']:.4f}</span></td>
                        <td>{r['psnr']:.1f} dB</td>
                        <td>{r['png_size'] / 1024:.1f} KB</td>
                        <td>{r['svg_size'] / 1024:.1f} KB</td>
                        <td>{r['size_reduction']:+.1f}%</td>
                        <td>{r['unified_score']:.1f}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Generated by SVG-AI Batch Comparison Tool</p>
        </div>
    </div>
</body>
</html>
"""

        # Save report
        with open(output_file, 'w') as f:
            f.write(html)

        print(f"\n✅ Report saved to {output_file}")
        return html


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch comparison tool")
    parser.add_argument('directory', help='Directory containing PNG files')
    parser.add_argument('--pattern', default='*.png', help='File pattern to match')
    parser.add_argument('--max-files', type=int, help='Maximum files to process')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--visuals', action='store_true', help='Generate visual comparisons')
    parser.add_argument('--output-dir', default='comparison_results', help='Output directory')

    args = parser.parse_args()

    # Create comparer
    comparer = BatchComparer(output_dir=args.output_dir)

    # Process directory
    results = comparer.process_directory(
        args.directory,
        pattern=args.pattern,
        max_files=args.max_files,
        parallel=args.parallel,
        generate_visuals=args.visuals
    )

    if results:
        # Generate report
        comparer.generate_report(results)

        # Save JSON data
        json_file = Path(args.output_dir) / 'comparison_data.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✅ Data saved to {json_file}")

        # Print summary
        successful = [r for r in results if r.get('success', False)]
        if successful:
            print(f"\n📊 Summary:")
            print(f"  Processed: {len(results)} files")
            print(f"  Successful: {len(successful)}")
            print(f"  Average SSIM: {sum(r['ssim'] for r in successful) / len(successful):.3f}")
            print(f"  Average unified score: {sum(r['unified_score'] for r in successful) / len(successful):.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())