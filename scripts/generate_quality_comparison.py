#!/usr/bin/env python3
"""
Visual Comparison Generator - DAY4 Task 2

Creates side-by-side comparison images and HTML reports for quality evaluation.
Generates visual quality indicators and highlights problem areas.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import sys
import tempfile
import cairosvg
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ai_modules.quality.enhanced_metrics import EnhancedQualityMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('default')


class VisualComparisonGenerator:
    """
    Generates visual comparisons between original PNG and converted SVG files.

    Creates comprehensive comparison grids with metrics overlays, difference maps,
    and quality indicators for detailed quality assessment.
    """

    def __init__(self, output_dir: str = "quality_comparisons"):
        """
        Initialize visual comparison generator.

        Args:
            output_dir: Directory to save comparison outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.metrics_calculator = EnhancedQualityMetrics()
        self.comparison_count = 0

        # Comparison settings
        self.figure_size = (16, 12)
        self.dpi = 150

    def create_comparison_grid(self, original_path: str, converted_path: str,
                             metrics: Dict[str, Any] = None) -> str:
        """
        Create 2x2 comparison grid with metrics overlay.

        Args:
            original_path: Path to original PNG file
            converted_path: Path to converted SVG file
            metrics: Pre-calculated metrics (optional)

        Returns:
            Path to generated comparison image
        """
        logger.info(f"Creating comparison grid for {os.path.basename(original_path)}")

        try:
            # Calculate metrics if not provided
            if metrics is None:
                metrics = self.metrics_calculator.calculate_metrics(original_path, converted_path)

            if not metrics['success']:
                raise ValueError(f"Metrics calculation failed: {metrics.get('error')}")

            # Load images
            original_img = self._load_image(original_path)
            converted_img = self._render_svg_to_image(converted_path, original_img.shape[:2])

            if original_img is None or converted_img is None:
                raise ValueError("Failed to load images")

            # Create difference map
            difference_map = self._create_difference_map(original_img, converted_img)

            # Create the comparison grid
            fig = plt.figure(figsize=self.figure_size)
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

            # Original image (top-left)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original PNG', fontsize=14, fontweight='bold')
            ax1.axis('off')

            # Converted image (top-right)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(cv2.cvtColor(converted_img, cv2.COLOR_BGR2RGB))
            ax2.set_title('Converted SVG', fontsize=14, fontweight='bold')
            ax2.axis('off')

            # Difference map (bottom-left)
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.imshow(difference_map, cmap='hot')
            ax3.set_title('Difference Map', fontsize=14, fontweight='bold')
            ax3.axis('off')

            # Add colorbar for difference map
            plt.colorbar(ax3.images[0], ax=ax3, fraction=0.046, pad=0.04)

            # Metrics overlay (bottom-right)
            ax4 = fig.add_subplot(gs[1, 1])
            self._add_metrics_overlay(ax4, metrics)

            # Add overall title with quality assessment
            composite_score = metrics['composite_score']
            interpretation = metrics['interpretation'].upper()
            fig.suptitle(
                f'Quality Comparison - Score: {composite_score:.3f} ({interpretation})',
                fontsize=16, fontweight='bold'
            )

            # Highlight problem areas if quality is poor
            if composite_score < 0.5:
                self._highlight_problem_areas(ax1, ax2, original_img, converted_img)

            # Save comparison grid
            output_filename = self._generate_output_filename(original_path, 'comparison')
            output_path = self.output_dir / output_filename
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()

            self.comparison_count += 1
            logger.info(f"Comparison grid saved to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create comparison grid: {e}")
            plt.close('all')  # Clean up any open figures
            return None

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image as numpy array."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL as fallback
                pil_image = Image.open(image_path).convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None

    def _render_svg_to_image(self, svg_path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Render SVG to image for comparison."""
        try:
            height, width = target_size

            # Use cairosvg to render SVG to PNG
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                cairosvg.svg2png(
                    url=svg_path,
                    write_to=tmp_file.name,
                    output_width=width,
                    output_height=height
                )

                # Load the rendered image
                rendered_img = cv2.imread(tmp_file.name)

                # Clean up temp file
                os.unlink(tmp_file.name)

                return rendered_img

        except Exception as e:
            logger.error(f"Failed to render SVG {svg_path}: {e}")
            return None

    def _create_difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Create visual difference map between two images."""
        try:
            # Convert to grayscale for difference calculation
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Calculate absolute difference
            diff = np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))

            # Normalize to 0-255 range
            diff_normalized = (diff / diff.max() * 255).astype(np.uint8)

            return diff_normalized
        except Exception as e:
            logger.error(f"Failed to create difference map: {e}")
            return np.zeros_like(img1[:, :, 0])

    def _add_metrics_overlay(self, ax, metrics: Dict[str, Any]) -> None:
        """Add metrics overlay to the subplot."""
        try:
            ax.axis('off')
            ax.set_title('Quality Metrics', fontsize=14, fontweight='bold')

            # Get normalized metrics
            normalized_metrics = metrics['normalized_metrics']
            composite_score = metrics['composite_score']

            # Create text overlay with metrics
            y_position = 0.95
            line_height = 0.08

            # Composite score (highlighted)
            ax.text(0.05, y_position, f'Composite Score:', fontsize=12, fontweight='bold',
                   transform=ax.transAxes, verticalalignment='top')
            ax.text(0.95, y_position, f'{composite_score:.3f}', fontsize=12, fontweight='bold',
                   transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                   color=self._get_score_color(composite_score))

            y_position -= line_height * 1.5

            # Individual metrics
            metric_names = {
                'ssim': 'SSIM',
                'mse': 'MSE',
                'psnr': 'PSNR',
                'edge_preservation': 'Edge Preserv.',
                'color_accuracy': 'Color Accuracy',
                'file_size_ratio': 'File Size',
                'path_complexity': 'Path Complex.'
            }

            for metric_key, metric_label in metric_names.items():
                if metric_key in normalized_metrics:
                    value = normalized_metrics[metric_key]
                    color = self._get_score_color(value)

                    ax.text(0.05, y_position, f'{metric_label}:', fontsize=10,
                           transform=ax.transAxes, verticalalignment='top')
                    ax.text(0.95, y_position, f'{value:.3f}', fontsize=10,
                           transform=ax.transAxes, verticalalignment='top',
                           horizontalalignment='right', color=color)

                    y_position -= line_height

            # Add processing time
            processing_time = metrics.get('processing_time', 0.0)
            ax.text(0.05, 0.05, f'Processing: {processing_time:.3f}s', fontsize=9,
                   transform=ax.transAxes, verticalalignment='bottom', style='italic')

        except Exception as e:
            logger.error(f"Failed to add metrics overlay: {e}")

    def _get_score_color(self, score: float) -> str:
        """Get color based on score value."""
        if score >= 0.8:
            return 'green'
        elif score >= 0.6:
            return 'orange'
        else:
            return 'red'

    def _highlight_problem_areas(self, ax1, ax2, img1: np.ndarray, img2: np.ndarray) -> None:
        """Highlight areas with significant differences."""
        try:
            # Create difference map
            diff_map = self._create_difference_map(img1, img2)

            # Find areas with high difference
            threshold = np.percentile(diff_map, 95)  # Top 5% differences
            problem_areas = diff_map > threshold

            # Find contours of problem areas
            contours, _ = cv2.findContours(
                problem_areas.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Add red rectangles around major problem areas
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small areas
                    x, y, w, h = cv2.boundingRect(contour)

                    # Add rectangle to both original and converted images
                    rect1 = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red',
                                            facecolor='none', alpha=0.7)
                    rect2 = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red',
                                            facecolor='none', alpha=0.7)
                    ax1.add_patch(rect1)
                    ax2.add_patch(rect2)

        except Exception as e:
            logger.error(f"Failed to highlight problem areas: {e}")

    def _generate_output_filename(self, input_path: str, suffix: str) -> str:
        """Generate output filename based on input path."""
        input_name = Path(input_path).stem
        timestamp = int(time.time())
        return f"{input_name}_{suffix}_{timestamp}.png"

    def generate_html_report(self, comparison_results: List[Dict[str, Any]]) -> str:
        """Generate HTML report with all comparisons."""
        logger.info(f"Generating HTML report for {len(comparison_results)} comparisons")

        try:
            html_content = self._create_html_template()

            # Add comparison sections
            comparisons_html = ""
            for i, result in enumerate(comparison_results):
                if result['success']:
                    comparisons_html += self._create_comparison_section(result, i + 1)

            # Insert comparisons into template
            html_content = html_content.replace("{{COMPARISONS}}", comparisons_html)
            html_content = html_content.replace("{{TOTAL_COUNT}}", str(len(comparison_results)))
            html_content = html_content.replace("{{TIMESTAMP}}", time.strftime("%Y-%m-%d %H:%M:%S"))

            # Save HTML report
            report_filename = f"quality_comparison_report_{int(time.time())}.html"
            report_path = self.output_dir / report_filename
            with open(report_path, 'w') as f:
                f.write(html_content)

            logger.info(f"HTML report saved to {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return None

    def _create_html_template(self) -> str:
        """Create HTML template for the report."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .comparison { background-color: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }
        .metric-card { background-color: #f8f9fa; padding: 10px; border-radius: 4px; border-left: 4px solid #007bff; }
        .score-excellent { color: #28a745; font-weight: bold; }
        .score-good { color: #ffc107; font-weight: bold; }
        .score-fair { color: #fd7e14; font-weight: bold; }
        .score-poor { color: #dc3545; font-weight: bold; }
        .comparison-image { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        .summary-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 20px 0; }
        .stat-card { background-color: #e9ecef; padding: 15px; text-align: center; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Quality Comparison Report</h1>
        <p>Generated on {{TIMESTAMP}} | Total Comparisons: {{TOTAL_COUNT}}</p>
    </div>

    {{COMPARISONS}}

    <div class="comparison">
        <h2>Report Summary</h2>
        <p>This report contains detailed quality comparisons between original PNG files and their SVG conversions.
           Each comparison includes visual side-by-side analysis, difference maps, and comprehensive quality metrics.</p>
        <p><strong>Quality Scoring:</strong></p>
        <ul>
            <li><span class="score-excellent">Excellent (≥0.85):</span> High-quality conversion with minimal artifacts</li>
            <li><span class="score-good">Good (≥0.70):</span> Good quality conversion with minor issues</li>
            <li><span class="score-fair">Fair (≥0.50):</span> Acceptable conversion with noticeable differences</li>
            <li><span class="score-poor">Poor (&lt;0.50):</span> Poor conversion quality requiring attention</li>
        </ul>
    </div>
</body>
</html>
        """.strip()

    def _create_comparison_section(self, result: Dict[str, Any], index: int) -> str:
        """Create HTML section for a single comparison."""
        metrics = result['metrics']
        composite_score = metrics['composite_score']
        interpretation = metrics['interpretation']

        # Get score class for styling
        score_class = f"score-{interpretation}"

        # Create metrics cards
        metrics_html = ""
        normalized_metrics = metrics['normalized_metrics']
        for metric_name, value in normalized_metrics.items():
            metric_display = metric_name.replace('_', ' ').title()
            metrics_html += f"""
            <div class="metric-card">
                <strong>{metric_display}</strong><br>
                <span class="{self._get_score_class_for_html(value)}">{value:.3f}</span>
            </div>
            """

        comparison_html = f"""
        <div class="comparison">
            <h2>Comparison #{index}: {result['original_file']}</h2>
            <div class="summary-stats">
                <div class="stat-card">
                    <h3>Composite Score</h3>
                    <span class="{score_class}">{composite_score:.3f}</span>
                </div>
                <div class="stat-card">
                    <h3>Quality</h3>
                    <span class="{score_class}">{interpretation.upper()}</span>
                </div>
                <div class="stat-card">
                    <h3>Processing Time</h3>
                    <span>{metrics['processing_time']:.3f}s</span>
                </div>
            </div>

            <div class="metrics-grid">
                {metrics_html}
            </div>

            <div style="text-align: center; margin: 20px 0;">
                <img src="{os.path.basename(result['comparison_image'])}"
                     alt="Quality Comparison" class="comparison-image">
            </div>
        </div>
        """

        return comparison_html

    def _get_score_class_for_html(self, score: float) -> str:
        """Get CSS class for score styling."""
        if score >= 0.8:
            return 'score-excellent'
        elif score >= 0.6:
            return 'score-good'
        elif score >= 0.4:
            return 'score-fair'
        else:
            return 'score-poor'

    def process_single_comparison(self, original_path: str, converted_path: str) -> Dict[str, Any]:
        """Process a single PNG-SVG comparison."""
        logger.info(f"Processing comparison: {os.path.basename(original_path)}")

        try:
            # Calculate metrics
            start_time = time.time()
            metrics = self.metrics_calculator.calculate_metrics(original_path, converted_path)
            metrics_time = time.time() - start_time

            if not metrics['success']:
                return {
                    'success': False,
                    'error': f"Metrics calculation failed: {metrics.get('error')}",
                    'original_file': os.path.basename(original_path),
                    'converted_file': os.path.basename(converted_path)
                }

            # Create comparison grid
            comparison_image = self.create_comparison_grid(original_path, converted_path, metrics)

            if comparison_image is None:
                return {
                    'success': False,
                    'error': "Failed to create comparison grid",
                    'original_file': os.path.basename(original_path),
                    'converted_file': os.path.basename(converted_path)
                }

            return {
                'success': True,
                'original_file': os.path.basename(original_path),
                'converted_file': os.path.basename(converted_path),
                'metrics': metrics,
                'comparison_image': comparison_image,
                'processing_time': metrics_time
            }

        except Exception as e:
            logger.error(f"Failed to process comparison: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_file': os.path.basename(original_path),
                'converted_file': os.path.basename(converted_path)
            }

    def process_batch_comparisons(self, file_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Process multiple PNG-SVG comparisons."""
        logger.info(f"Processing batch of {len(file_pairs)} comparisons")

        results = []
        for i, (original_path, converted_path) in enumerate(file_pairs):
            logger.info(f"Processing comparison {i + 1}/{len(file_pairs)}")
            result = self.process_single_comparison(original_path, converted_path)
            results.append(result)

        # Generate HTML report
        html_report = self.generate_html_report(results)

        logger.info(f"Batch processing completed: {len(results)} comparisons")
        return results

    def get_generator_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            'comparisons_generated': self.comparison_count,
            'output_directory': str(self.output_dir),
            'metrics_cache_stats': self.metrics_calculator.get_cache_stats()
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Generate visual quality comparisons')
    parser.add_argument('original', help='Path to original PNG file')
    parser.add_argument('converted', help='Path to converted SVG file')
    parser.add_argument('--output-dir', default='quality_comparisons',
                       help='Output directory for comparisons')
    parser.add_argument('--batch', action='store_true',
                       help='Batch mode: original and converted are directories')

    args = parser.parse_args()

    generator = VisualComparisonGenerator(args.output_dir)

    if args.batch:
        # Batch processing mode
        original_dir = Path(args.original)
        converted_dir = Path(args.converted)

        if not original_dir.is_dir() or not converted_dir.is_dir():
            print("Error: Both paths must be directories in batch mode")
            return

        # Find matching PNG-SVG pairs
        file_pairs = []
        for png_file in original_dir.glob("*.png"):
            svg_file = converted_dir / f"{png_file.stem}.svg"
            if svg_file.exists():
                file_pairs.append((str(png_file), str(svg_file)))

        if not file_pairs:
            print("No matching PNG-SVG pairs found")
            return

        print(f"Found {len(file_pairs)} matching pairs")
        results = generator.process_batch_comparisons(file_pairs)

        successful = sum(1 for r in results if r['success'])
        print(f"Processed {successful}/{len(results)} comparisons successfully")

    else:
        # Single file mode
        if not os.path.exists(args.original) or not os.path.exists(args.converted):
            print("Error: Input files not found")
            return

        result = generator.process_single_comparison(args.original, args.converted)

        if result['success']:
            print(f"✓ Comparison generated successfully")
            print(f"  Composite score: {result['metrics']['composite_score']:.3f}")
            print(f"  Quality: {result['metrics']['interpretation']}")
            print(f"  Comparison image: {result['comparison_image']}")
        else:
            print(f"✗ Comparison failed: {result['error']}")


if __name__ == "__main__":
    main()