#!/usr/bin/env python3
"""
View and compare AI training results - see the actual conversions!
"""

import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from backend.converter import convert_image
import cairosvg
import io
import random

def view_training_results(training_file="training_data_real_logos.json", num_samples=10):
    """
    Display visual comparison of training results

    Args:
        training_file: JSON file with training data
        num_samples: Number of examples to show
    """

    # Check if training data exists
    if not Path(training_file).exists():
        print(f"❌ Training data not found: {training_file}")
        print("Run 'python train_with_progress.py' first to generate training data")
        return

    # Load training data
    print(f"Loading training results from {training_file}...")
    with open(training_file) as f:
        data = json.load(f)

    if not data:
        print("No training data found!")
        return

    print(f"Found {len(data)} training samples")

    # Find best results for each unique image
    best_results = {}
    for item in data:
        img_path = item['image_path']
        if img_path not in best_results or item['quality_score'] > best_results[img_path]['quality_score']:
            best_results[img_path] = item

    # Group by logo type
    by_type = {}
    for path, result in best_results.items():
        logo_type = result['logo_type']
        if logo_type not in by_type:
            by_type[logo_type] = []
        by_type[logo_type].append(result)

    # Show summary statistics
    print("\n" + "=" * 70)
    print("📊 TRAINING RESULTS SUMMARY")
    print("=" * 70)

    for logo_type, results in by_type.items():
        scores = [r['quality_score'] for r in results]
        print(f"\n{logo_type.upper()} LOGOS ({len(results)} samples)")
        print(f"  Average SSIM: {np.mean(scores):.3f}")
        print(f"  Best SSIM:    {np.max(scores):.3f}")
        print(f"  Worst SSIM:   {np.min(scores):.3f}")

        # Show best parameters for this type
        best_result = max(results, key=lambda x: x['quality_score'])
        print(f"  Best params:  color={best_result['parameters']['color_precision']}, "
              f"corner={best_result['parameters']['corner_threshold']}")

    # Visual comparison
    print("\n" + "=" * 70)
    print("🖼️  VISUAL COMPARISON")
    print("=" * 70)

    # Select samples to display
    samples_to_show = []
    for logo_type, results in by_type.items():
        # Get best and worst from each type
        if results:
            sorted_results = sorted(results, key=lambda x: x['quality_score'], reverse=True)
            # Best result
            if sorted_results[0]['image_path'] and Path(sorted_results[0]['image_path']).exists():
                samples_to_show.append((sorted_results[0], "best"))
            # Worst result (if different)
            if len(sorted_results) > 1 and sorted_results[-1]['image_path'] and Path(sorted_results[-1]['image_path']).exists():
                samples_to_show.append((sorted_results[-1], "worst"))

    # Limit number of samples
    samples_to_show = samples_to_show[:num_samples]

    if not samples_to_show:
        print("No valid samples to display")
        return

    # Create figure
    fig = plt.figure(figsize=(15, 4 * len(samples_to_show)))
    gs = gridspec.GridSpec(len(samples_to_show), 4, figure=fig, hspace=0.3, wspace=0.2)

    for idx, (result, quality) in enumerate(samples_to_show):
        image_path = result['image_path']

        try:
            # Load original image
            original = Image.open(image_path).convert('RGB')

            # Convert with best parameters
            params = result['parameters']
            conversion = convert_image(image_path, converter_type='vtracer', **params)

            if conversion['success'] and conversion['svg']:
                # Convert SVG to PNG for display
                svg_png_bytes = cairosvg.svg2png(bytestring=conversion['svg'].encode('utf-8'))
                svg_image = Image.open(io.BytesIO(svg_png_bytes)).convert('RGB')

                # Calculate difference
                orig_array = np.array(original.resize((256, 256)))
                svg_array = np.array(svg_image.resize((256, 256)))
                diff_array = np.abs(orig_array.astype(float) - svg_array.astype(float))
                diff_array = (diff_array / diff_array.max() * 255).astype(np.uint8)

                # Plot original
                ax1 = fig.add_subplot(gs[idx, 0])
                ax1.imshow(original)
                ax1.set_title(f"Original\n{result['logo_type']}", fontsize=10)
                ax1.axis('off')

                # Plot SVG result
                ax2 = fig.add_subplot(gs[idx, 1])
                ax2.imshow(svg_image)
                ax2.set_title(f"SVG Result\nSSIM: {result['quality_score']:.3f}", fontsize=10)
                ax2.axis('off')

                # Plot difference
                ax3 = fig.add_subplot(gs[idx, 2])
                ax3.imshow(diff_array)
                ax3.set_title(f"Difference\n({quality} for type)", fontsize=10)
                ax3.axis('off')

                # Plot parameters info
                ax4 = fig.add_subplot(gs[idx, 3])
                ax4.axis('off')

                info_text = f"Parameters:\n"
                info_text += f"Color Precision: {params['color_precision']}\n"
                info_text += f"Corner Threshold: {params['corner_threshold']}\n"
                info_text += f"Segment Length: {params.get('segment_length', 'N/A')}\n"
                info_text += f"\nMetrics:\n"
                info_text += f"SSIM: {result['quality_score']:.3f}\n"
                info_text += f"File Size: {result['file_size']} bytes\n"
                info_text += f"Time: {result.get('conversion_time', 0):.2f}s"

                ax4.text(0.1, 0.5, info_text, fontsize=9, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                print(f"✅ Displayed: {Path(image_path).name} - SSIM: {result['quality_score']:.3f}")

        except Exception as e:
            print(f"❌ Error displaying {image_path}: {e}")

    plt.suptitle("AI Training Results - Visual Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_file = "training_results_comparison.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"\n💾 Saved visual comparison to: {output_file}")

    plt.show()

def generate_html_report(training_file="training_data_real_logos.json", output_file="results_report.html"):
    """Generate an HTML report with all results"""

    print(f"\nGenerating HTML report...")

    with open(training_file) as f:
        data = json.load(f)

    # Find best results per image
    best_results = {}
    for item in data:
        img_path = item['image_path']
        if img_path not in best_results or item['quality_score'] > best_results[img_path]['quality_score']:
            best_results[img_path] = item

    # Group by type
    by_type = {}
    for path, result in best_results.items():
        logo_type = result['logo_type']
        if logo_type not in by_type:
            by_type[logo_type] = []
        by_type[logo_type].append(result)

    # Generate HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Training Results Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
            h2 { color: #666; margin-top: 30px; }
            .summary { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metric { display: inline-block; margin: 10px 20px; }
            .metric-label { color: #888; font-size: 12px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
            table { width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            th { background: #4CAF50; color: white; padding: 12px; text-align: left; }
            td { padding: 10px; border-bottom: 1px solid #eee; }
            tr:hover { background: #f9f9f9; }
            .score-high { color: #4CAF50; font-weight: bold; }
            .score-medium { color: #FF9800; font-weight: bold; }
            .score-low { color: #f44336; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🤖 AI Training Results Report</h1>
    """

    # Summary statistics
    all_scores = [r['quality_score'] for r in best_results.values()]
    html += f"""
        <div class="summary">
            <h2>Overall Performance</h2>
            <div class="metric">
                <div class="metric-label">Total Logos</div>
                <div class="metric-value">{len(best_results)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Average SSIM</div>
                <div class="metric-value">{np.mean(all_scores):.3f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Best SSIM</div>
                <div class="metric-value">{np.max(all_scores):.3f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Training Samples</div>
                <div class="metric-value">{len(data)}</div>
            </div>
        </div>
    """

    # Results by type
    for logo_type in sorted(by_type.keys()):
        results = by_type[logo_type]
        scores = [r['quality_score'] for r in results]

        html += f"""
            <h2>{logo_type.replace('_', ' ').title()} Logos</h2>
            <div class="summary">
                <div class="metric">
                    <div class="metric-label">Count</div>
                    <div class="metric-value">{len(results)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg SSIM</div>
                    <div class="metric-value">{np.mean(scores):.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Best SSIM</div>
                    <div class="metric-value">{np.max(scores):.3f}</div>
                </div>
            </div>

            <table>
                <tr>
                    <th>Logo File</th>
                    <th>SSIM Score</th>
                    <th>Color Precision</th>
                    <th>Corner Threshold</th>
                    <th>File Size</th>
                    <th>Conversion Time</th>
                </tr>
        """

        # Sort by score
        for result in sorted(results, key=lambda x: x['quality_score'], reverse=True)[:10]:
            score = result['quality_score']
            if score >= 0.95:
                score_class = 'score-high'
            elif score >= 0.85:
                score_class = 'score-medium'
            else:
                score_class = 'score-low'

            html += f"""
                <tr>
                    <td>{Path(result['image_path']).name}</td>
                    <td class="{score_class}">{score:.3f}</td>
                    <td>{result['parameters']['color_precision']}</td>
                    <td>{result['parameters']['corner_threshold']}</td>
                    <td>{result['file_size']:,} bytes</td>
                    <td>{result.get('conversion_time', 0):.2f}s</td>
                </tr>
            """

        html += "</table>"

    html += """
        <div class="summary" style="margin-top: 40px;">
            <h3>Next Steps</h3>
            <ol>
                <li>Train classifier model with this data</li>
                <li>Train quality predictor model</li>
                <li>Train parameter optimizer</li>
                <li>Deploy models to production</li>
            </ol>
        </div>
    </body>
    </html>
    """

    # Save HTML
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✅ HTML report saved to: {output_file}")
    print(f"📂 Open in browser: file://{Path(output_file).absolute()}")

def main():
    """Main function to view results"""

    import sys

    print("=" * 70)
    print("🔍 AI TRAINING RESULTS VIEWER")
    print("=" * 70)

    # Check if training data exists
    training_files = list(Path(".").glob("training_data*.json"))

    if not training_files:
        print("❌ No training data found!")
        print("Run one of these first:")
        print("  python train_with_progress.py 100")
        print("  python train_with_raw_logos.py")
        return

    print(f"Found {len(training_files)} training data files:")
    for i, f in enumerate(training_files):
        file_stats = f.stat()
        size_mb = file_stats.st_size / 1024 / 1024
        print(f"  {i+1}. {f.name} ({size_mb:.1f} MB)")

    # Use most recent or specified file
    if len(sys.argv) > 1:
        training_file = sys.argv[1]
    else:
        training_file = max(training_files, key=lambda f: f.stat().st_mtime)
        print(f"\nUsing most recent: {training_file}")

    # View results
    view_training_results(training_file, num_samples=6)

    # Generate HTML report
    generate_html_report(training_file)

    print("\n✨ Done! Check the generated files:")
    print("  • training_results_comparison.png - Visual comparison")
    print("  • results_report.html - Detailed HTML report")

if __name__ == "__main__":
    main()