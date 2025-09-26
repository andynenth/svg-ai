#!/usr/bin/env python3
"""
Quality comparison tool for testing PNG to SVG conversion results.

This script compares original PNG images with converted SVG files,
calculating quality metrics and generating visual comparisons.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")

    try:
        from PIL import Image
    except ImportError:
        missing.append("pillow")

    try:
        import cairosvg
    except ImportError:
        missing.append("cairosvg")

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.info("Install with: pip install " + " ".join(missing))
        return False

    return True


class QualityComparison:
    """Compare quality between original PNG and converted SVG."""

    def __init__(self):
        """Initialize the quality comparison tool."""
        self.metrics = {}
        self.has_dependencies = check_dependencies()

        if self.has_dependencies:
            import numpy as np
            from PIL import Image
            self.np = np
            self.Image = Image

    def load_png(self, png_path: str) -> Optional['np.ndarray']:
        """Load PNG image as numpy array."""
        if not self.has_dependencies:
            return None

        try:
            img = self.Image.open(png_path).convert('RGBA')
            return self.np.array(img)
        except Exception as e:
            logger.error(f"Failed to load PNG {png_path}: {e}")
            return None

    def load_svg(self, svg_path: str, width: int = None, height: int = None) -> Optional['np.ndarray']:
        """Load SVG as numpy array by rendering to PNG."""
        if not self.has_dependencies:
            return None

        try:
            import cairosvg
            from io import BytesIO

            # Render SVG to PNG bytes
            png_bytes = cairosvg.svg2png(
                url=svg_path,
                output_width=width,
                output_height=height
            )

            # Load as PIL Image
            img = self.Image.open(BytesIO(png_bytes)).convert('RGBA')
            return self.np.array(img)

        except Exception as e:
            logger.error(f"Failed to load SVG {svg_path}: {e}")
            return None

    def calculate_mse(self, img1: 'np.ndarray', img2: 'np.ndarray') -> float:
        """Calculate Mean Squared Error between two images."""
        if img1.shape != img2.shape:
            logger.warning(f"Shape mismatch: {img1.shape} vs {img2.shape}")
            return float('inf')

        mse = self.np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        return float(mse)

    def calculate_psnr(self, img1: 'np.ndarray', img2: 'np.ndarray') -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = self.calculate_mse(img1, img2)
        if mse == 0:
            return 100.0
        if mse == float('inf'):
            return 0.0

        max_pixel = 255.0
        psnr = 20 * self.np.log10(max_pixel / self.np.sqrt(mse))
        return float(psnr)

    def calculate_ssim_simple(self, img1: 'np.ndarray', img2: 'np.ndarray') -> float:
        """
        Calculate a simplified SSIM (Structural Similarity Index).

        This is a basic implementation for testing.
        """
        if img1.shape != img2.shape:
            return 0.0

        # Convert to grayscale for simplicity
        if len(img1.shape) == 3:
            gray1 = self.np.mean(img1[:,:,:3], axis=2)  # Ignore alpha
            gray2 = self.np.mean(img2[:,:,:3], axis=2)
        else:
            gray1 = img1
            gray2 = img2

        # Calculate means
        mu1 = self.np.mean(gray1)
        mu2 = self.np.mean(gray2)

        # Calculate variances and covariance
        var1 = self.np.var(gray1)
        var2 = self.np.var(gray2)
        cov = self.np.mean((gray1 - mu1) * (gray2 - mu2))

        # SSIM formula constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        # Calculate SSIM
        numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2)

        if denominator == 0:
            return 1.0 if numerator == 0 else 0.0

        ssim = numerator / denominator
        return float(max(0, min(1, ssim)))

    def compare(self, png_path: str, svg_path: str) -> Dict[str, float]:
        """
        Compare PNG and SVG files and return quality metrics.

        Args:
            png_path: Path to original PNG file
            svg_path: Path to converted SVG file

        Returns:
            Dictionary with quality metrics
        """
        results = {
            'png_path': png_path,
            'svg_path': svg_path,
            'success': False,
            'error': None
        }

        if not self.has_dependencies:
            results['error'] = "Missing dependencies"
            return results

        # Load images
        png_img = self.load_png(png_path)
        if png_img is None:
            results['error'] = "Failed to load PNG"
            return results

        # Get PNG dimensions for SVG rendering
        height, width = png_img.shape[:2]

        svg_img = self.load_svg(svg_path, width=width, height=height)
        if svg_img is None:
            results['error'] = "Failed to load SVG"
            return results

        # Calculate metrics
        try:
            results['mse'] = self.calculate_mse(png_img, svg_img)
            results['psnr'] = self.calculate_psnr(png_img, svg_img)
            results['ssim'] = self.calculate_ssim_simple(png_img, svg_img)

            # File sizes
            results['png_size'] = os.path.getsize(png_path)
            results['svg_size'] = os.path.getsize(svg_path)
            results['size_reduction'] = (1 - results['svg_size'] / results['png_size']) * 100

            # Dimensions
            results['width'] = width
            results['height'] = height

            results['success'] = True

        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Failed to calculate metrics: {e}")

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate a text report from comparison results."""
        if not results.get('success'):
            return f"❌ Comparison failed: {results.get('error', 'Unknown error')}"

        report = []
        report.append("="*60)
        report.append("QUALITY COMPARISON REPORT")
        report.append("="*60)
        report.append(f"\n📁 Files:")
        report.append(f"  Original: {Path(results['png_path']).name}")
        report.append(f"  Converted: {Path(results['svg_path']).name}")

        report.append(f"\n📊 Quality Metrics:")
        report.append(f"  SSIM: {results['ssim']:.4f} (1.0 = perfect)")
        report.append(f"  PSNR: {results['psnr']:.2f} dB (higher is better)")
        report.append(f"  MSE: {results['mse']:.2f} (0 = perfect)")

        report.append(f"\n📦 File Sizes:")
        report.append(f"  PNG: {results['png_size']:,} bytes")
        report.append(f"  SVG: {results['svg_size']:,} bytes")
        report.append(f"  Reduction: {results['size_reduction']:.1f}%")

        report.append(f"\n📐 Dimensions:")
        report.append(f"  Size: {results['width']} x {results['height']} pixels")

        report.append(f"\n✨ Summary:")
        if results['ssim'] >= 0.95:
            report.append("  Excellent quality! Near-perfect conversion.")
        elif results['ssim'] >= 0.85:
            report.append("  Good quality. Minor differences visible.")
        elif results['ssim'] >= 0.75:
            report.append("  Acceptable quality. Some visible differences.")
        else:
            report.append("  Poor quality. Significant differences.")

        if results['size_reduction'] > 0:
            report.append(f"  File size reduced by {results['size_reduction']:.1f}%")
        else:
            report.append(f"  ⚠️ SVG is larger than PNG!")

        report.append("\n" + "="*60)

        return "\n".join(report)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Compare quality between original PNG and converted SVG"
    )
    parser.add_argument("png_path", help="Path to original PNG file")
    parser.add_argument("svg_path", help="Path to converted SVG file")
    parser.add_argument("-o", "--output", help="Save results to JSON file")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check files exist
    if not os.path.exists(args.png_path):
        logger.error(f"PNG file not found: {args.png_path}")
        return 1

    if not os.path.exists(args.svg_path):
        logger.error(f"SVG file not found: {args.svg_path}")
        return 1

    # Run comparison
    logger.info("Starting quality comparison...")
    comparator = QualityComparison()
    results = comparator.compare(args.png_path, args.svg_path)

    # Generate and print report
    report = comparator.generate_report(results)
    print(report)

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")

    # Return success/failure
    return 0 if results.get('success') else 1


if __name__ == "__main__":
    sys.exit(main())