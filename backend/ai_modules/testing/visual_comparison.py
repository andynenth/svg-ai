"""
Visual Comparison Generator - Task 3 Implementation
Generate visual comparisons for A/B testing results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import cv2
import seaborn as sns

# Import SVG handling
try:
    import cairosvg
    import svglib.svglib as svglib
    from reportlab.graphics import renderPM
    SVG_SUPPORT = True
except ImportError:
    SVG_SUPPORT = False

logger = logging.getLogger(__name__)


class VisualComparisonGenerator:
    """
    Generate visual comparisons for A/B testing results.
    Creates side-by-side comparisons, difference maps, and quality overlays.
    """

    def __init__(self, output_dir: str = 'visual_comparisons'):
        """
        Initialize visual comparison generator.

        Args:
            output_dir: Directory for output images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style settings
        plt.style.use('default')
        sns.set_palette("husl")

        logger.info(f"VisualComparisonGenerator initialized (output_dir={output_dir})")

    def generate_comparison(self, image_path: str, control_result: Dict, treatment_result: Dict) -> plt.Figure:
        """
        Create 4-panel visual comparison.

        Args:
            image_path: Path to original image
            control_result: Control group result dictionary
            treatment_result: Treatment group result dictionary

        Returns:
            Matplotlib figure with comparison
        """
        try:
            # Create 4-panel comparison
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle(f'A/B Test Comparison: {Path(image_path).name}', fontsize=16, fontweight='bold')

            # Load original image
            original_img = self._load_image(image_path)
            if original_img is not None:
                axes[0, 0].imshow(original_img)
                axes[0, 0].set_title('Original', fontsize=14, fontweight='bold')
                axes[0, 0].axis('off')

            # Load and display control result (Baseline)
            control_img = self._load_svg_result(control_result)
            control_quality = control_result.get('quality', {})
            control_ssim = control_quality.get('ssim', 0) if control_quality else 0

            if control_img is not None:
                axes[0, 1].imshow(control_img)
                axes[0, 1].set_title(f"Baseline (SSIM: {control_ssim:.3f})",
                                   fontsize=14, fontweight='bold')
                axes[0, 1].axis('off')

            # Load and display treatment result (AI Enhanced)
            treatment_img = self._load_svg_result(treatment_result)
            treatment_quality = treatment_result.get('quality', {})
            treatment_ssim = treatment_quality.get('ssim', 0) if treatment_quality else 0

            if treatment_img is not None:
                axes[1, 0].imshow(treatment_img)
                axes[1, 0].set_title(f"AI Enhanced (SSIM: {treatment_ssim:.3f})",
                                   fontsize=14, fontweight='bold')
                axes[1, 0].axis('off')

            # Generate difference map
            if control_img is not None and treatment_img is not None:
                diff_img, improvement = self._calculate_difference(control_img, treatment_img)
                im = axes[1, 1].imshow(diff_img, cmap='RdBu', vmin=-1, vmax=1)
                axes[1, 1].set_title(f"Improvement: {improvement:.1%}",
                                   fontsize=14, fontweight='bold')
                axes[1, 1].axis('off')

                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
                cbar.set_label('Difference', fontsize=12)
            else:
                axes[1, 1].text(0.5, 0.5, 'Difference map\nnot available',
                              ha='center', va='center', transform=axes[1, 1].transAxes,
                              fontsize=12)
                axes[1, 1].axis('off')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Failed to generate comparison for {image_path}: {e}")
            return self._create_error_figure(str(e))

    def generate_side_by_side(self, image_path: str, control_result: Dict,
                             treatment_result: Dict) -> plt.Figure:
        """
        Create side-by-side comparison.

        Args:
            image_path: Path to original image
            control_result: Control group result
            treatment_result: Treatment group result

        Returns:
            Matplotlib figure with side-by-side comparison
        """
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Side-by-Side Comparison: {Path(image_path).name}',
                        fontsize=16, fontweight='bold')

            # Original
            original_img = self._load_image(image_path)
            if original_img is not None:
                axes[0].imshow(original_img)
                axes[0].set_title('Original', fontsize=14, fontweight='bold')
                axes[0].axis('off')

            # Control (Baseline)
            control_img = self._load_svg_result(control_result)
            control_quality = control_result.get('quality', {})
            control_ssim = control_quality.get('ssim', 0) if control_quality else 0

            if control_img is not None:
                axes[1].imshow(control_img)
                title = f"Baseline\nSSIM: {control_ssim:.3f}"
                if control_result.get('duration'):
                    title += f"\nTime: {control_result['duration']:.2f}s"
                axes[1].set_title(title, fontsize=14, fontweight='bold')
                axes[1].axis('off')

            # Treatment (AI Enhanced)
            treatment_img = self._load_svg_result(treatment_result)
            treatment_quality = treatment_result.get('quality', {})
            treatment_ssim = treatment_quality.get('ssim', 0) if treatment_quality else 0

            if treatment_img is not None:
                axes[2].imshow(treatment_img)
                title = f"AI Enhanced\nSSIM: {treatment_ssim:.3f}"
                if treatment_result.get('duration'):
                    title += f"\nTime: {treatment_result['duration']:.2f}s"

                # Color code based on improvement
                improvement = treatment_ssim - control_ssim
                if improvement > 0.01:
                    title_color = 'green'
                elif improvement < -0.01:
                    title_color = 'red'
                else:
                    title_color = 'black'

                axes[2].set_title(title, fontsize=14, fontweight='bold', color=title_color)
                axes[2].axis('off')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Failed to generate side-by-side for {image_path}: {e}")
            return self._create_error_figure(str(e))

    def generate_difference_heatmap(self, control_result: Dict, treatment_result: Dict) -> plt.Figure:
        """
        Generate detailed difference heatmap.

        Args:
            control_result: Control group result
            treatment_result: Treatment group result

        Returns:
            Matplotlib figure with difference heatmap
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Difference Analysis Heatmap', fontsize=16, fontweight='bold')

            # Load images
            control_img = self._load_svg_result(control_result)
            treatment_img = self._load_svg_result(treatment_result)

            if control_img is not None and treatment_img is not None:
                # Calculate difference
                diff_img, _ = self._calculate_difference(control_img, treatment_img)

                # Show difference as heatmap
                im1 = axes[0].imshow(diff_img, cmap='RdBu', vmin=-1, vmax=1)
                axes[0].set_title('Pixel-wise Difference', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0], shrink=0.8)

                # Show absolute difference
                abs_diff = np.abs(diff_img)
                im2 = axes[1].imshow(abs_diff, cmap='hot', vmin=0, vmax=1)
                axes[1].set_title('Absolute Difference Magnitude', fontsize=14, fontweight='bold')
                axes[1].axis('off')
                plt.colorbar(im2, ax=axes[1], shrink=0.8)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Failed to generate difference heatmap: {e}")
            return self._create_error_figure(str(e))

    def create_quality_overlay(self, image_path: str, control_result: Dict,
                              treatment_result: Dict) -> plt.Figure:
        """
        Create quality metric overlay visualization.

        Args:
            image_path: Path to original image
            control_result: Control group result
            treatment_result: Treatment group result

        Returns:
            Matplotlib figure with quality overlays
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Quality Metrics Overlay: {Path(image_path).name}',
                        fontsize=16, fontweight='bold')

            # Extract quality metrics
            control_quality = control_result.get('quality', {})
            treatment_quality = treatment_result.get('quality', {})

            metrics = ['ssim', 'mse', 'psnr']
            improvements = {}

            for metric in metrics:
                control_val = control_quality.get(metric, 0)
                treatment_val = treatment_quality.get(metric, 0)

                if metric == 'mse':  # Lower is better for MSE
                    improvement = ((control_val - treatment_val) / control_val * 100) if control_val > 0 else 0
                else:  # Higher is better for SSIM, PSNR
                    improvement = ((treatment_val - control_val) / control_val * 100) if control_val > 0 else 0

                improvements[metric] = improvement

            # Load original image for overlay
            original_img = self._load_image(image_path)

            # Create overlays for each metric
            for i, metric in enumerate(['ssim', 'mse']):
                row, col = i // 2, i % 2
                ax = axes[row, col]

                if original_img is not None:
                    ax.imshow(original_img, alpha=0.7)

                # Create overlay based on improvement
                improvement = improvements.get(metric, 0)
                if improvement > 0:
                    overlay_color = 'green'
                    overlay_alpha = min(0.3, abs(improvement) / 100)
                elif improvement < 0:
                    overlay_color = 'red'
                    overlay_alpha = min(0.3, abs(improvement) / 100)
                else:
                    overlay_color = 'gray'
                    overlay_alpha = 0.1

                # Add colored overlay
                ax.add_patch(mpatches.Rectangle((0, 0), 1, 1,
                                              transform=ax.transAxes,
                                              facecolor=overlay_color,
                                              alpha=overlay_alpha))

                # Add text overlay
                control_val = control_quality.get(metric, 0)
                treatment_val = treatment_quality.get(metric, 0)

                text = f"{metric.upper()}\n"
                text += f"Baseline: {control_val:.3f}\n"
                text += f"AI Enhanced: {treatment_val:.3f}\n"
                text += f"Improvement: {improvement:+.1f}%"

                ax.text(0.02, 0.98, text, transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10, fontweight='bold')

                ax.set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
                ax.axis('off')

            # Summary metrics in remaining subplots
            ax = axes[0, 1]
            if original_img is not None:
                ax.imshow(original_img, alpha=0.5)

            # Create summary text
            summary_text = "Overall Performance Summary\n\n"
            for metric, improvement in improvements.items():
                symbol = "↑" if improvement > 0 else ("↓" if improvement < 0 else "→")
                color = "green" if improvement > 0 else ("red" if improvement < 0 else "gray")
                summary_text += f"{metric.upper()}: {improvement:+.1f}% {symbol}\n"

            ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   fontsize=12, fontweight='bold')

            ax.set_title('Performance Summary', fontsize=12, fontweight='bold')
            ax.axis('off')

            # Processing time comparison
            ax = axes[1, 1]
            control_time = control_result.get('duration', 0)
            treatment_time = treatment_result.get('duration', 0)
            time_change = ((treatment_time - control_time) / control_time * 100) if control_time > 0 else 0

            times = [control_time, treatment_time]
            labels = ['Baseline', 'AI Enhanced']
            colors = ['blue', 'green' if time_change <= 0 else 'orange']

            bars = ax.bar(labels, times, color=colors, alpha=0.7)
            ax.set_ylabel('Processing Time (seconds)', fontsize=10)
            ax.set_title(f'Processing Time\n({time_change:+.1f}% change)', fontsize=12, fontweight='bold')

            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{time_val:.2f}s',
                       ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Failed to create quality overlay: {e}")
            return self._create_error_figure(str(e))

    def export_html_report(self, comparisons: List[Dict[str, Any]], output_path: str):
        """
        Export comparisons as HTML report.

        Args:
            comparisons: List of comparison dictionaries
            output_path: Output HTML file path
        """
        try:
            html_content = self._generate_html_template()

            # Add each comparison
            comparison_html = ""
            for i, comp in enumerate(comparisons):
                comparison_html += self._generate_comparison_html(comp, i)

            # Insert comparisons into template
            html_content = html_content.replace("{{COMPARISONS}}", comparison_html)

            # Write HTML file
            with open(output_path, 'w') as f:
                f.write(html_content)

            logger.info(f"HTML report exported to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export HTML report: {e}")

    def batch_generate_comparisons(self, test_results: List[Dict],
                                 output_dir: Optional[str] = None) -> List[str]:
        """
        Generate comparisons for batch of test results.

        Args:
            test_results: List of A/B test results
            output_dir: Output directory (uses self.output_dir if None)

        Returns:
            List of generated file paths
        """
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.output_dir

        output_path.mkdir(parents=True, exist_ok=True)
        generated_files = []

        try:
            # Group results by image
            image_groups = {}
            for result in test_results:
                image_path = result.get('image', 'unknown')
                if image_path not in image_groups:
                    image_groups[image_path] = {'control': None, 'treatment': None}

                group = result.get('group')
                if group in ['control', 'treatment']:
                    image_groups[image_path][group] = result

            # Generate comparison for each image pair
            for image_path, groups in image_groups.items():
                if groups['control'] and groups['treatment']:
                    # Generate main comparison
                    fig = self.generate_comparison(
                        image_path, groups['control'], groups['treatment']
                    )

                    # Save figure
                    safe_name = Path(image_path).stem.replace(' ', '_')
                    output_file = output_path / f"comparison_{safe_name}.png"
                    fig.savefig(output_file, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    generated_files.append(str(output_file))

                    # Generate side-by-side
                    fig_side = self.generate_side_by_side(
                        image_path, groups['control'], groups['treatment']
                    )
                    output_file_side = output_path / f"sidebyside_{safe_name}.png"
                    fig_side.savefig(output_file_side, dpi=150, bbox_inches='tight')
                    plt.close(fig_side)
                    generated_files.append(str(output_file_side))

            logger.info(f"Generated {len(generated_files)} comparison files")
            return generated_files

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return generated_files

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file path."""
        try:
            if not Path(image_path).exists():
                # Create placeholder image
                return self._create_placeholder_image(f"Image not found:\n{image_path}")

            image = cv2.imread(image_path)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return self._create_placeholder_image(f"Could not load:\n{image_path}")

        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return self._create_placeholder_image(f"Error loading:\n{image_path}")

    def _load_svg_result(self, result: Dict) -> Optional[np.ndarray]:
        """Load SVG result and render as image."""
        try:
            svg_path = result.get('svg_path')
            if not svg_path or not Path(svg_path).exists():
                success = result.get('success', False)
                if not success:
                    error_msg = result.get('error', 'Conversion failed')
                    return self._create_placeholder_image(f"Conversion Error:\n{error_msg}")
                else:
                    return self._create_placeholder_image(f"SVG not found:\n{svg_path}")

            # Try to render SVG
            if SVG_SUPPORT:
                return self._render_svg_with_cairo(svg_path)
            else:
                return self._create_placeholder_image(f"SVG Preview:\n{Path(svg_path).name}")

        except Exception as e:
            logger.warning(f"Failed to load SVG result: {e}")
            return self._create_placeholder_image(f"SVG Load Error:\n{str(e)[:50]}")

    def _render_svg_with_cairo(self, svg_path: str) -> Optional[np.ndarray]:
        """Render SVG using cairosvg."""
        try:
            # Convert SVG to PNG in memory
            png_data = cairosvg.svg2png(url=svg_path)

            # Load PNG data as image
            image = Image.open(io.BytesIO(png_data))
            return np.array(image.convert('RGB'))

        except Exception as e:
            logger.warning(f"Failed to render SVG with cairo: {e}")
            return None

    def _create_placeholder_image(self, text: str, size: Tuple[int, int] = (400, 400)) -> np.ndarray:
        """Create placeholder image with text."""
        # Create image
        img = Image.new('RGB', size, color='lightgray')
        draw = ImageDraw.Draw(img)

        # Try to use a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2

        # Draw text
        draw.text((x, y), text, fill='black', font=font, anchor='mm')

        return np.array(img)

    def _calculate_difference(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate difference between two images.

        Args:
            img1: First image (control)
            img2: Second image (treatment)

        Returns:
            Tuple of (difference_image, improvement_ratio)
        """
        try:
            # Ensure images are same size
            if img1.shape != img2.shape:
                # Resize to match smaller dimension
                min_h = min(img1.shape[0], img2.shape[0])
                min_w = min(img1.shape[1], img2.shape[1])
                img1 = cv2.resize(img1, (min_w, min_h))
                img2 = cv2.resize(img2, (min_w, min_h))

            # Convert to grayscale for difference calculation
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2

            # Normalize to 0-1 range
            gray1 = gray1.astype(np.float32) / 255.0
            gray2 = gray2.astype(np.float32) / 255.0

            # Calculate difference
            diff = gray2 - gray1

            # Calculate improvement ratio
            mse1 = np.mean((gray1 - 0.5) ** 2)  # Baseline MSE from ideal
            mse2 = np.mean((gray2 - 0.5) ** 2)  # Treatment MSE from ideal
            improvement = (mse1 - mse2) / mse1 if mse1 > 0 else 0

            return diff, improvement

        except Exception as e:
            logger.warning(f"Failed to calculate difference: {e}")
            return np.zeros((100, 100)), 0.0

    def _create_error_figure(self, error_message: str) -> plt.Figure:
        """Create error figure when comparison fails."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error generating comparison:\n{error_message}",
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
               fontsize=12)
        ax.set_title('Comparison Error', fontsize=14, fontweight='bold')
        ax.axis('off')
        return fig

    def _generate_html_template(self) -> str:
        """Generate HTML template for report."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>A/B Test Visual Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .header { text-align: center; margin-bottom: 40px; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .comparison { margin-bottom: 40px; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .comparison h2 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        .image-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .image-container { text-align: center; }
        .image-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .metrics { background: #f9f9f9; padding: 15px; border-radius: 5px; margin-top: 10px; }
        .improvement { font-weight: bold; }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        .neutral { color: #ff9800; }
    </style>
</head>
<body>
    <div class="header">
        <h1>A/B Test Visual Comparison Report</h1>
        <p>Generated on: {{TIMESTAMP}}</p>
    </div>

    {{COMPARISONS}}

    <div style="text-align: center; margin-top: 40px; color: #666;">
        <p>Report generated by SVG AI A/B Testing Framework</p>
    </div>
</body>
</html>
        """.strip()

    def _generate_comparison_html(self, comparison: Dict, index: int) -> str:
        """Generate HTML for single comparison."""
        return f"""
    <div class="comparison">
        <h2>Comparison {index + 1}: {comparison.get('image_name', 'Unknown')}</h2>
        <div class="image-grid">
            <div class="image-container">
                <h3>Baseline</h3>
                <img src="{comparison.get('control_image', '')}" alt="Baseline Result">
                <div class="metrics">
                    <p>SSIM: {comparison.get('control_ssim', 0):.3f}</p>
                    <p>Processing Time: {comparison.get('control_time', 0):.2f}s</p>
                </div>
            </div>
            <div class="image-container">
                <h3>AI Enhanced</h3>
                <img src="{comparison.get('treatment_image', '')}" alt="AI Enhanced Result">
                <div class="metrics">
                    <p>SSIM: {comparison.get('treatment_ssim', 0):.3f}</p>
                    <p>Processing Time: {comparison.get('treatment_time', 0):.2f}s</p>
                    <p class="improvement">Improvement: <span class="{self._get_improvement_class(comparison.get('improvement', 0))}">{comparison.get('improvement', 0):+.1f}%</span></p>
                </div>
            </div>
        </div>
    </div>
        """.strip()

    def _get_improvement_class(self, improvement: float) -> str:
        """Get CSS class for improvement value."""
        if improvement > 1:
            return "positive"
        elif improvement < -1:
            return "negative"
        else:
            return "neutral"


def test_visual_comparison_generator():
    """Test the visual comparison generator."""
    print("Testing Visual Comparison Generator...")

    # Create generator
    generator = VisualComparisonGenerator(output_dir='/tmp/visual_comparisons_test')

    # Create mock test results
    control_result = {
        'svg_path': None,  # Will use placeholder
        'quality': {'ssim': 0.82, 'mse': 0.025, 'psnr': 28.5},
        'duration': 2.1,
        'success': False,
        'error': 'Mock baseline conversion'
    }

    treatment_result = {
        'svg_path': None,  # Will use placeholder
        'quality': {'ssim': 0.87, 'mse': 0.018, 'psnr': 31.2},
        'duration': 2.3,
        'success': False,
        'error': 'Mock AI conversion'
    }

    # Test 1: Generate 4-panel comparison
    print("\n✓ Testing 4-panel comparison:")
    fig = generator.generate_comparison('test_image.png', control_result, treatment_result)
    output_file = '/tmp/visual_comparisons_test/test_4panel.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 4-panel comparison to {output_file}")

    # Test 2: Generate side-by-side comparison
    print("\n✓ Testing side-by-side comparison:")
    fig_side = generator.generate_side_by_side('test_image.png', control_result, treatment_result)
    output_file_side = '/tmp/visual_comparisons_test/test_sidebyside.png'
    fig_side.savefig(output_file_side, dpi=150, bbox_inches='tight')
    plt.close(fig_side)
    print(f"  Saved side-by-side comparison to {output_file_side}")

    # Test 3: Generate difference heatmap
    print("\n✓ Testing difference heatmap:")
    fig_diff = generator.generate_difference_heatmap(control_result, treatment_result)
    output_file_diff = '/tmp/visual_comparisons_test/test_heatmap.png'
    fig_diff.savefig(output_file_diff, dpi=150, bbox_inches='tight')
    plt.close(fig_diff)
    print(f"  Saved difference heatmap to {output_file_diff}")

    # Test 4: Create quality overlay
    print("\n✓ Testing quality overlay:")
    fig_overlay = generator.create_quality_overlay('test_image.png', control_result, treatment_result)
    output_file_overlay = '/tmp/visual_comparisons_test/test_overlay.png'
    fig_overlay.savefig(output_file_overlay, dpi=150, bbox_inches='tight')
    plt.close(fig_overlay)
    print(f"  Saved quality overlay to {output_file_overlay}")

    # Test 5: Batch generation
    print("\n✓ Testing batch generation:")
    test_results = []
    for i in range(3):
        # Control results
        test_results.append({
            'image': f'batch_test_{i}.png',
            'group': 'control',
            'quality': {'ssim': 0.80 + i*0.02, 'mse': 0.03 - i*0.005},
            'duration': 2.0 + i*0.1,
            'success': True
        })
        # Treatment results
        test_results.append({
            'image': f'batch_test_{i}.png',
            'group': 'treatment',
            'quality': {'ssim': 0.85 + i*0.02, 'mse': 0.02 - i*0.003},
            'duration': 2.2 + i*0.1,
            'success': True
        })

    generated_files = generator.batch_generate_comparisons(test_results)
    print(f"  Generated {len(generated_files)} files in batch mode")

    # Test 6: HTML export
    print("\n✓ Testing HTML report export:")
    comparisons = [
        {
            'image_name': 'test_1.png',
            'control_ssim': 0.82,
            'treatment_ssim': 0.87,
            'control_time': 2.1,
            'treatment_time': 2.3,
            'improvement': 6.1
        }
    ]
    html_output = '/tmp/visual_comparisons_test/test_report.html'
    generator.export_html_report(comparisons, html_output)
    print(f"  Exported HTML report to {html_output}")

    print("\n✅ All visual comparison tests passed!")
    return generator


if __name__ == "__main__":
    test_visual_comparison_generator()