#!/usr/bin/env python3
"""
Generate visual comparisons between original PNG and converted SVG.

This script creates side-by-side comparison images with quality metrics overlay.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cairosvg
import io

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.quality_metrics import QualityMetrics
from utils.image_loader import ImageLoader


class VisualComparisonGenerator:
    """Generate visual comparisons with metrics overlay."""

    def __init__(self):
        """Initialize the generator."""
        self.metrics = QualityMetrics()
        self.loader = ImageLoader()

    def load_images(self, png_path: str, svg_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load PNG and SVG images.

        Args:
            png_path: Path to PNG file
            svg_path: Path to SVG file

        Returns:
            Tuple of (PNG array, SVG array)
        """
        # Load PNG
        png_img = self.loader.load_image(png_path)

        # Load SVG
        svg_img = self.loader.load_svg_as_image(svg_path, png_img.shape[1], png_img.shape[0])

        return png_img, svg_img

    def create_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Create a difference heatmap.

        Args:
            img1, img2: Input images

        Returns:
            Difference heatmap as RGB image
        """
        # Ensure same shape
        if img1.shape != img2.shape:
            raise ValueError("Images must have same dimensions")

        # Handle RGBA
        if len(img1.shape) == 3 and img1.shape[2] == 4:
            img1 = img1[:, :, :3]
        if len(img2.shape) == 3 and img2.shape[2] == 4:
            img2 = img2[:, :, :3]

        # Calculate absolute difference
        diff = np.abs(img1.astype(float) - img2.astype(float))

        # Convert to single channel
        if len(diff.shape) == 3:
            diff = np.mean(diff, axis=2)

        # Normalize to 0-255
        diff = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)

        # Create heatmap (blue=same, red=different)
        heatmap = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
        heatmap[:, :, 0] = diff  # Red channel for differences
        heatmap[:, :, 2] = 255 - diff  # Blue channel for similarities

        return heatmap

    def calculate_all_metrics(self, img1: np.ndarray, img2: np.ndarray,
                            png_path: str, svg_path: str) -> Dict:
        """
        Calculate all quality metrics.

        Args:
            img1, img2: Images to compare
            png_path, svg_path: File paths for size calculation

        Returns:
            Dictionary of metrics
        """
        # Visual quality metrics
        ssim = self.metrics.calculate_ssim(img1, img2)
        mse = self.metrics.calculate_mse(img1, img2)
        psnr = self.metrics.calculate_psnr(img1, img2)

        # Try to calculate perceptual loss
        try:
            perceptual = self.metrics.calculate_perceptual_loss(img1, img2)
        except ImportError:
            perceptual = 0.0

        # File sizes
        png_size = Path(png_path).stat().st_size
        svg_size = Path(svg_path).stat().st_size
        size_ratio = svg_size / png_size
        size_reduction = (1 - size_ratio) * 100

        # Unified score
        unified = self.metrics.calculate_unified_score(ssim, psnr, perceptual, size_ratio)

        return {
            'ssim': ssim,
            'mse': mse,
            'psnr': psnr,
            'perceptual': perceptual,
            'png_size': png_size,
            'svg_size': svg_size,
            'size_ratio': size_ratio,
            'size_reduction': size_reduction,
            'unified_score': unified
        }

    def create_comparison_grid(self, png_img: np.ndarray, svg_img: np.ndarray,
                             diff_map: np.ndarray, metrics: Dict) -> Image.Image:
        """
        Create a 3-panel comparison grid.

        Args:
            png_img: Original PNG image
            svg_img: Converted SVG image
            diff_map: Difference heatmap
            metrics: Quality metrics dictionary

        Returns:
            PIL Image with comparison grid
        """
        # Convert numpy arrays to PIL Images
        png_pil = Image.fromarray(png_img[:, :, :3] if png_img.shape[2] == 4 else png_img)
        svg_pil = Image.fromarray(svg_img[:, :, :3] if svg_img.shape[2] == 4 else svg_img)
        diff_pil = Image.fromarray(diff_map)

        # Calculate dimensions
        img_width = png_pil.width
        img_height = png_pil.height
        padding = 20
        text_height = 100

        # Create canvas
        canvas_width = img_width * 3 + padding * 4
        canvas_height = img_height + text_height + padding * 3

        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

        # Paste images
        canvas.paste(png_pil, (padding, padding))
        canvas.paste(svg_pil, (img_width + padding * 2, padding))
        canvas.paste(diff_pil, (img_width * 2 + padding * 3, padding))

        # Add labels and metrics
        draw = ImageDraw.Draw(canvas)

        # Try to use a better font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()
            title_font = font

        # Panel titles
        y_text = img_height + padding * 2
        draw.text((padding, y_text), "Original PNG", fill='black', font=title_font)
        draw.text((img_width + padding * 2, y_text), "Converted SVG", fill='black', font=title_font)
        draw.text((img_width * 2 + padding * 3, y_text), "Difference Map", fill='black', font=title_font)

        # Metrics text
        metrics_text = [
            f"SSIM: {metrics['ssim']:.4f}",
            f"PSNR: {metrics['psnr']:.2f} dB" if metrics['psnr'] != float('inf') else "PSNR: ∞",
            f"PNG: {metrics['png_size'] / 1024:.1f} KB",
            f"SVG: {metrics['svg_size'] / 1024:.1f} KB",
            f"Size: {metrics['size_reduction']:+.1f}%",
            f"Score: {metrics['unified_score']:.1f}/100"
        ]

        # Add metrics
        y_metrics = y_text + 25
        for i, text in enumerate(metrics_text):
            x_pos = padding + (i % 3) * (img_width + padding)
            y_pos = y_metrics + (i // 3) * 20
            draw.text((x_pos, y_pos), text, fill='black', font=font)

        return canvas

    def generate_comparison(self, png_path: str, svg_path: str,
                          output_path: str = None) -> Dict:
        """
        Generate visual comparison.

        Args:
            png_path: Path to original PNG
            svg_path: Path to converted SVG
            output_path: Output path for comparison image

        Returns:
            Metrics dictionary
        """
        print(f"Generating comparison for {Path(png_path).name}...")

        # Load images
        png_img, svg_img = self.load_images(png_path, svg_path)

        # Calculate metrics
        metrics = self.calculate_all_metrics(png_img, svg_img, png_path, svg_path)

        # Create difference map
        diff_map = self.create_difference_map(png_img, svg_img)

        # Create comparison grid
        comparison = self.create_comparison_grid(png_img, svg_img, diff_map, metrics)

        # Save if output path provided
        if output_path:
            comparison.save(output_path)
            print(f"  ✅ Saved to {output_path}")

        # Print metrics
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  Size: {metrics['size_reduction']:+.1f}%")
        print(f"  Unified Score: {metrics['unified_score']:.1f}/100")

        return metrics

    def generate_html_comparison(self, png_path: str, svg_path: str,
                                output_path: str = None) -> str:
        """
        Generate HTML comparison page.

        Args:
            png_path: Path to original PNG
            svg_path: Path to converted SVG
            output_path: Output HTML file path

        Returns:
            HTML content
        """
        # Calculate metrics
        png_img, svg_img = self.load_images(png_path, svg_path)
        metrics = self.calculate_all_metrics(png_img, svg_img, png_path, svg_path)

        # Read files for embedding
        import base64

        with open(png_path, 'rb') as f:
            png_data = base64.b64encode(f.read()).decode()

        with open(svg_path, 'r') as f:
            svg_content = f.read()

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVG Conversion Comparison</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .panel {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .panel h2 {{
            margin-top: 0;
            color: #667eea;
        }}
        .image-container {{
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            background: #fafafa;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        .good {{ color: #10b981; }}
        .warning {{ color: #f59e0b; }}
        .bad {{ color: #ef4444; }}
        .unified-score {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin: 20px 0;
        }}
        .unified-score .metric-value {{
            color: white;
            font-size: 3em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 PNG to SVG Conversion Comparison</h1>

        <div class="comparison">
            <div class="panel">
                <h2>Original PNG</h2>
                <div class="image-container">
                    <img src="data:image/png;base64,{png_data}" alt="Original PNG">
                </div>
                <p>File size: {metrics['png_size'] / 1024:.1f} KB</p>
            </div>

            <div class="panel">
                <h2>Converted SVG</h2>
                <div class="image-container">
                    {svg_content}
                </div>
                <p>File size: {metrics['svg_size'] / 1024:.1f} KB
                   ({metrics['size_reduction']:+.1f}%)</p>
            </div>
        </div>

        <div class="unified-score">
            <div class="metric-label">Unified Quality Score</div>
            <div class="metric-value">{metrics['unified_score']:.1f}/100</div>
        </div>

        <div class="metrics">
            <div class="metric">
                <div class="metric-label">SSIM</div>
                <div class="metric-value {('good' if metrics['ssim'] > 0.95 else 'warning' if metrics['ssim'] > 0.85 else 'bad')}">
                    {metrics['ssim']:.4f}
                </div>
            </div>

            <div class="metric">
                <div class="metric-label">PSNR</div>
                <div class="metric-value {('good' if metrics['psnr'] > 35 else 'warning' if metrics['psnr'] > 30 else 'bad')}">
                    {metrics['psnr']:.1f} dB
                </div>
            </div>

            <div class="metric">
                <div class="metric-label">MSE</div>
                <div class="metric-value">
                    {metrics['mse']:.2f}
                </div>
            </div>

            <div class="metric">
                <div class="metric-label">Perceptual Loss</div>
                <div class="metric-value">
                    {metrics['perceptual']:.2f}
                </div>
            </div>

            <div class="metric">
                <div class="metric-label">Size Ratio</div>
                <div class="metric-value {('good' if metrics['size_ratio'] < 0.6 else 'warning' if metrics['size_ratio'] < 1 else 'bad')}">
                    {metrics['size_ratio']:.2f}x
                </div>
            </div>

            <div class="metric">
                <div class="metric-label">File Name</div>
                <div class="metric-value" style="font-size: 1em;">
                    {Path(png_path).name}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
            print(f"  ✅ HTML saved to {output_path}")

        return html


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate visual comparison")
    parser.add_argument('png', help='Original PNG file')
    parser.add_argument('svg', help='Converted SVG file')
    parser.add_argument('--output', help='Output image path')
    parser.add_argument('--html', help='Generate HTML comparison')

    args = parser.parse_args()

    # Check files exist
    if not Path(args.png).exists():
        print(f"❌ PNG file not found: {args.png}")
        return 1

    if not Path(args.svg).exists():
        print(f"❌ SVG file not found: {args.svg}")
        return 1

    # Create generator
    generator = VisualComparisonGenerator()

    # Generate comparison
    output_path = args.output or f"comparison_{Path(args.png).stem}.png"
    metrics = generator.generate_comparison(args.png, args.svg, output_path)

    # Generate HTML if requested
    if args.html:
        html_path = args.html if args.html != True else f"comparison_{Path(args.png).stem}.html"
        generator.generate_html_comparison(args.png, args.svg, html_path)

    print(f"\n✅ Comparison complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())