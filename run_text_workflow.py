#!/usr/bin/env python3
"""Run optimization workflow on text-based logos without click dependency."""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import sys
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iterative_optimizer_standalone import IterativeOptimizer
from utils.visual_compare import VisualComparer
from utils.quality_metrics import ComprehensiveMetrics

def analyze_logo_detection(results: List[Dict]) -> Dict[str, Any]:
    """Analyze logo type detection accuracy."""
    detection_stats = {
        'total': len(results),
        'detected_as': {},
        'should_be': 'text',
        'correct_detections': 0,
        'misclassifications': []
    }

    for result in results:
        logo_type = result.get('logo_type', 'unknown')
        if logo_type not in detection_stats['detected_as']:
            detection_stats['detected_as'][logo_type] = []
        detection_stats['detected_as'][logo_type].append(result['file'])

        if logo_type != 'text':
            detection_stats['misclassifications'].append({
                'file': result['file'],
                'detected': logo_type,
                'expected': 'text'
            })
        else:
            detection_stats['correct_detections'] += 1

    detection_stats['accuracy'] = (detection_stats['correct_detections'] / detection_stats['total']) * 100
    return detection_stats

def run_text_workflow():
    """Run complete optimization workflow on text-based logos."""

    # Setup directories
    input_dir = Path("data/logos/text_based")
    output_dir = Path("data/logos/text_based/optimized_workflow_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    target_ssim = 0.98
    max_iterations = 10

    # Get all PNG files
    png_files = sorted(input_dir.glob("*.png"))
    print(f"\n=== Text-Based Logo Optimization Workflow ===")
    print(f"Found {len(png_files)} text logos to optimize")
    print(f"Target SSIM: {target_ssim:.2%}")
    print(f"Max iterations: {max_iterations}")
    print(f"Output directory: {output_dir}")

    # Results storage
    results = []
    successful = 0
    failed = 0
    total_start = time.time()

    # Process each logo
    for i, png_path in enumerate(png_files, 1):
        print(f"\n[{i}/{len(png_files)}] Processing {png_path.name}")
        print("-" * 50)

        try:
            # Initialize optimizer
            optimizer = IterativeOptimizer(
                input_path=str(png_path),
                output_dir=str(output_dir),
                target_ssim=target_ssim,
                max_iterations=max_iterations
            )

            # Run optimization
            start_time = time.time()
            result = optimizer.optimize()
            processing_time = time.time() - start_time

            # Add processing time to result
            result['processing_time'] = processing_time

            # Print immediate results
            print(f"✓ Logo type detected: {result['logo_type']}")
            print(f"✓ Iterations: {result['iterations']}")
            print(f"✓ Final SSIM: {result['ssim']:.4f}")
            print(f"✓ Success: {result['success']}")
            print(f"✓ Time: {processing_time:.2f}s")

            if result['success']:
                print(f"✅ SUCCESS - Reached target quality")
                successful += 1
            else:
                print(f"⚠️ FAILED - Did not reach target (gap: {target_ssim - result['ssim']:.4f})")
                failed += 1

            # Store result
            results.append(result)

            # Generate visual comparison
            print("Generating visual comparison...")
            comparer = VisualComparer()
            comparison_path = output_dir / f"{png_path.stem}.comparison.png"
            comparer.create_comparison(
                str(png_path),
                result['svg_path'],
                str(comparison_path)
            )
            result['comparison_path'] = str(comparison_path)

        except Exception as e:
            print(f"❌ ERROR processing {png_path.name}: {e}")
            traceback.print_exc()
            failed += 1
            results.append({
                'file': str(png_path),
                'success': False,
                'error': str(e),
                'processing_time': 0
            })

    # Calculate statistics
    total_time = time.time() - total_start
    ssim_values = [r['ssim'] for r in results if 'ssim' in r]
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0

    # Analyze detection accuracy
    detection_analysis = analyze_logo_detection(results)

    # Generate summary report
    summary = {
        'summary': {
            'total_files': len(png_files),
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / len(png_files)) * 100,
            'target_ssim': target_ssim,
            'max_iterations': max_iterations
        },
        'performance': {
            'average_ssim': avg_ssim,
            'average_time': total_time / len(png_files),
            'total_time': total_time
        },
        'detection_analysis': detection_analysis,
        'detailed_results': results
    }

    # Save JSON report
    report_path = output_dir / 'optimization_report.json'
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"✅ Successful: {successful}/{len(png_files)} ({successful/len(png_files)*100:.1f}%)")
    print(f"❌ Failed: {failed}/{len(png_files)}")
    print(f"📊 Average SSIM: {avg_ssim:.4f}")
    print(f"⏱️ Total time: {total_time:.2f}s")
    print(f"📈 Avg time per logo: {total_time/len(png_files):.2f}s")

    print("\n=== DETECTION ANALYSIS ===")
    print(f"Expected type: TEXT")
    print(f"Correct detections: {detection_analysis['correct_detections']}/{detection_analysis['total']} ({detection_analysis['accuracy']:.1f}%)")

    if detection_analysis['detected_as']:
        print("\nDetection breakdown:")
        for logo_type, files in detection_analysis['detected_as'].items():
            print(f"  - {logo_type}: {len(files)} logos")
            if logo_type != 'text' and len(files) <= 3:
                for f in files:
                    print(f"    • {Path(f).name}")

    if detection_analysis['misclassifications']:
        print(f"\n⚠️ MISCLASSIFIED LOGOS ({len(detection_analysis['misclassifications'])})")
        for mis in detection_analysis['misclassifications'][:5]:  # Show first 5
            print(f"  • {Path(mis['file']).name}: detected as '{mis['detected']}' (expected 'text')")

    # Analyze quality distribution
    print("\n=== QUALITY DISTRIBUTION ===")
    quality_bins = {
        'excellent (>99%)': 0,
        'very_good (98-99%)': 0,
        'good (96-98%)': 0,
        'acceptable (94-96%)': 0,
        'poor (<94%)': 0
    }

    for ssim in ssim_values:
        if ssim > 0.99:
            quality_bins['excellent (>99%)'] += 1
        elif ssim >= 0.98:
            quality_bins['very_good (98-99%)'] += 1
        elif ssim >= 0.96:
            quality_bins['good (96-98%)'] += 1
        elif ssim >= 0.94:
            quality_bins['acceptable (94-96%)'] += 1
        else:
            quality_bins['poor (<94%)'] += 1

    for category, count in quality_bins.items():
        bar = '█' * (count * 2)
        print(f"{category:25} {bar} {count}")

    # Analyze parameters used
    print("\n=== PARAMETER ANALYSIS ===")
    params_by_type = {}
    for result in results:
        if 'best_params' in result and result.get('logo_type'):
            logo_type = result['logo_type']
            if logo_type not in params_by_type:
                params_by_type[logo_type] = []
            params_by_type[logo_type].append(result['best_params'])

    for logo_type, params_list in params_by_type.items():
        print(f"\n{logo_type.upper()} preset used for {len(params_list)} logos:")
        if params_list:
            sample_params = params_list[0]
            print(f"  - color_precision: {sample_params.get('color_precision')}")
            print(f"  - corner_threshold: {sample_params.get('corner_threshold')}")
            print(f"  - path_precision: {sample_params.get('path_precision')}")

    print(f"\n📁 Report saved to: {report_path}")

    # Create markdown analysis
    create_markdown_analysis(output_dir, summary, detection_analysis)

def create_markdown_analysis(output_dir: Path, summary: Dict, detection_analysis: Dict):
    """Create detailed markdown analysis report."""

    report_path = output_dir / 'TEXT_WORKFLOW_ANALYSIS.md'

    with open(report_path, 'w') as f:
        f.write("# Text-Based Logos - Optimization Workflow Analysis\n\n")
        f.write(f"## Execution Summary\n")
        f.write(f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total Logos**: {summary['summary']['total_files']}\n")
        f.write(f"- **Success Rate**: {summary['summary']['success_rate']:.1f}%\n")
        f.write(f"- **Average SSIM**: {summary['performance']['average_ssim']:.4f}\n")
        f.write(f"- **Total Time**: {summary['performance']['total_time']:.2f}s\n\n")

        f.write("## Detection Analysis\n\n")
        f.write(f"### Expected vs Actual\n")
        f.write(f"- **Expected Type**: TEXT (for all logos)\n")
        f.write(f"- **Correct Detections**: {detection_analysis['correct_detections']}/{detection_analysis['total']} ")
        f.write(f"({detection_analysis['accuracy']:.1f}%)\n\n")

        if detection_analysis['misclassifications']:
            f.write("### Misclassifications\n\n")
            f.write("| Logo | Detected As | Should Be |\n")
            f.write("|------|------------|----------|\n")
            for mis in detection_analysis['misclassifications']:
                f.write(f"| {Path(mis['file']).name} | {mis['detected']} | {mis['expected']} |\n")
            f.write("\n")

        f.write("## Quality Results\n\n")
        f.write("| Logo | SSIM | Success | Type Detected | Iterations |\n")
        f.write("|------|------|---------|--------------|------------|\n")

        for result in summary['detailed_results']:
            if 'ssim' in result:
                logo_name = Path(result['file']).name
                ssim = result['ssim']
                success = "✅" if result['success'] else "❌"
                logo_type = result.get('logo_type', 'unknown')
                iterations = result.get('iterations', 0)
                f.write(f"| {logo_name} | {ssim:.4f} | {success} | {logo_type} | {iterations} |\n")

        f.write("\n## Key Findings\n\n")

        if detection_analysis['accuracy'] < 50:
            f.write("### ⚠️ Critical Issue: Logo Type Misdetection\n")
            f.write(f"- Only {detection_analysis['accuracy']:.1f}% of text logos correctly identified\n")
            f.write("- Most text logos misclassified as 'gradient' due to anti-aliasing\n")
            f.write("- This causes suboptimal parameter selection\n\n")

        if summary['summary']['success_rate'] >= 90:
            f.write("### ✅ High Quality Achievement\n")
            f.write(f"- {summary['summary']['success_rate']:.1f}% reached target SSIM\n")
            f.write(f"- Average quality: {summary['performance']['average_ssim']:.4f}\n")
            f.write("- Even with wrong parameters, quality remains high\n\n")

        f.write("## Recommendations\n\n")
        f.write("1. **Fix Detection Algorithm**: Improve text recognition for anti-aliased text\n")
        f.write("2. **Force Text Preset**: When processing known text logos\n")
        f.write("3. **Consider OCR**: Use text detection libraries for better classification\n\n")

    print(f"📝 Markdown analysis saved to: {report_path}")

if __name__ == "__main__":
    run_text_workflow()