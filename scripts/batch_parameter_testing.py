#!/usr/bin/env python3
"""
Batch Parameter Testing System for VTracer - Day 1 Task 3
Processes diverse set of logos with parameter variations for training data collection.
"""

import os
import sys
import json
import argparse
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import quality measurement from Task 2
from scripts.measure_conversion_quality import measure_conversion_quality


def load_parameter_grid(grid_path: str) -> List[Dict[str, Any]]:
    """
    Load parameter grid from Task 1.

    Args:
        grid_path: Path to parameter grid JSON file

    Returns:
        List of parameter dictionaries
    """
    try:
        with open(grid_path, 'r') as f:
            data = json.load(f)

        parameter_combinations = data.get('parameter_combinations', [])
        print(f"📊 Loaded {len(parameter_combinations)} parameter combinations")
        return parameter_combinations

    except Exception as e:
        print(f"❌ Failed to load parameter grid: {e}")
        return []


def select_diverse_logo_set(data_dir: str) -> Dict[str, List[str]]:
    """
    Select diverse set of 50 logos for testing.

    Args:
        data_dir: Base data directory path

    Returns:
        Dict with categorized logo paths
    """
    logos_dir = Path(data_dir) / "logos"
    raw_logos_dir = Path(data_dir) / "raw_logos"

    selected_logos = {
        'simple_geometric': [],
        'text_based': [],
        'gradients': [],
        'complex': [],
        'random': []
    }

    # Select 10 from each category
    categories = ['simple_geometric', 'text_based', 'gradients', 'complex']

    for category in categories:
        category_dir = logos_dir / category

        if not category_dir.exists():
            print(f"⚠️  Category directory not found: {category_dir}")
            continue

        # Get all PNG files in category
        png_files = list(category_dir.glob("*.png"))

        if len(png_files) < 10:
            print(f"⚠️  Only {len(png_files)} files in {category}, using all")
            selected_logos[category] = [str(f) for f in png_files]
        else:
            # Randomly select 10
            selected_files = random.sample(png_files, 10)
            selected_logos[category] = [str(f) for f in selected_files]

        print(f"✅ Selected {len(selected_logos[category])} logos from {category}")

    # Select 10 random from raw_logos
    raw_png_files = list(raw_logos_dir.glob("*.png"))
    if len(raw_png_files) >= 10:
        random_files = random.sample(raw_png_files, 10)
        selected_logos['random'] = [str(f) for f in random_files]
        print(f"✅ Selected 10 random logos from raw_logos")
    else:
        print(f"⚠️  Only {len(raw_png_files)} files in raw_logos")
        selected_logos['random'] = [str(f) for f in raw_png_files[:10]]

    # Calculate total
    total_selected = sum(len(logos) for logos in selected_logos.values())
    print(f"📋 Total logos selected: {total_selected}")

    return selected_logos


def load_existing_results(output_path: str) -> Dict[str, Any]:
    """
    Load existing results for crash recovery.

    Args:
        output_path: Path to results JSON file

    Returns:
        Existing results dict or new structure
    """
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            print(f"🔄 Resuming from existing results: {len(data.get('results', []))} conversions already completed")
            return data
        except Exception as e:
            print(f"⚠️  Could not load existing results: {e}")

    # Return new structure
    return {
        'metadata': {
            'batch_processing_type': 'diverse_logo_parameter_testing',
            'timestamp_started': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_logos': 0,
            'total_parameter_combinations': 0,
            'total_planned_conversions': 0,
            'completed_conversions': 0
        },
        'logo_selection': {},
        'results': [],
        'summary_statistics': {}
    }


def save_results_incrementally(results_data: Dict[str, Any], output_path: str):
    """
    Save results incrementally for crash recovery.

    Args:
        results_data: Complete results data structure
        output_path: Path to save JSON file
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Update metadata
    results_data['metadata']['completed_conversions'] = len(results_data['results'])
    results_data['metadata']['timestamp_last_update'] = time.strftime('%Y-%m-%d %H:%M:%S')

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)


def is_already_processed(results_data: Dict[str, Any], image_path: str, params: Dict[str, Any]) -> bool:
    """
    Check if a combination has already been processed.

    Args:
        results_data: Existing results data
        image_path: Path to image
        params: Parameter combination

    Returns:
        True if already processed
    """
    for result in results_data.get('results', []):
        if (result.get('image_path') == image_path and
            result.get('parameters') == params):
            return True
    return False


def generate_summary_statistics(results_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics from results.

    Args:
        results_data: Complete results data

    Returns:
        Summary statistics dictionary
    """
    results = results_data.get('results', [])

    if not results:
        return {}

    # Basic statistics
    successful_conversions = [r for r in results if r.get('conversion_success', False)]
    failed_conversions = [r for r in results if not r.get('conversion_success', False)]

    # Quality metrics for successful conversions
    ssim_scores = [r.get('ssim', 0) for r in successful_conversions if 'ssim' in r]
    mse_scores = [r.get('mse', float('inf')) for r in successful_conversions if 'mse' in r and r['mse'] != float('inf')]
    processing_times = [r.get('processing_time', 0) for r in successful_conversions]
    file_size_ratios = [r.get('file_size_ratio', 0) for r in successful_conversions]

    # Category statistics
    category_stats = {}
    for result in results:
        image_path = result.get('image_path', '')

        # Determine category from path
        if 'simple_geometric' in image_path:
            category = 'simple_geometric'
        elif 'text_based' in image_path:
            category = 'text_based'
        elif 'gradients' in image_path:
            category = 'gradients'
        elif 'complex' in image_path:
            category = 'complex'
        elif 'raw_logos' in image_path:
            category = 'random'
        else:
            category = 'unknown'

        if category not in category_stats:
            category_stats[category] = {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'avg_ssim': 0,
                'avg_processing_time': 0
            }

        category_stats[category]['total'] += 1
        if result.get('conversion_success', False):
            category_stats[category]['successful'] += 1
            if 'ssim' in result:
                category_stats[category]['avg_ssim'] += result['ssim']
            if 'processing_time' in result:
                category_stats[category]['avg_processing_time'] += result['processing_time']
        else:
            category_stats[category]['failed'] += 1

    # Calculate averages
    for category in category_stats:
        if category_stats[category]['successful'] > 0:
            category_stats[category]['avg_ssim'] /= category_stats[category]['successful']
            category_stats[category]['avg_processing_time'] /= category_stats[category]['successful']

    summary = {
        'total_conversions': len(results),
        'successful_conversions': len(successful_conversions),
        'failed_conversions': len(failed_conversions),
        'success_rate': len(successful_conversions) / len(results) if results else 0,
        'quality_metrics': {
            'ssim': {
                'min': min(ssim_scores) if ssim_scores else 0,
                'max': max(ssim_scores) if ssim_scores else 0,
                'avg': sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0,
                'count': len(ssim_scores)
            },
            'mse': {
                'min': min(mse_scores) if mse_scores else 0,
                'max': max(mse_scores) if mse_scores else 0,
                'avg': sum(mse_scores) / len(mse_scores) if mse_scores else 0,
                'count': len(mse_scores)
            },
            'processing_time': {
                'min': min(processing_times) if processing_times else 0,
                'max': max(processing_times) if processing_times else 0,
                'avg': sum(processing_times) / len(processing_times) if processing_times else 0,
                'total': sum(processing_times)
            },
            'file_size_ratio': {
                'min': min(file_size_ratios) if file_size_ratios else 0,
                'max': max(file_size_ratios) if file_size_ratios else 0,
                'avg': sum(file_size_ratios) / len(file_size_ratios) if file_size_ratios else 0
            }
        },
        'category_statistics': category_stats
    }

    return summary


def run_batch_processing(logos: Dict[str, List[str]], parameters: List[Dict[str, Any]],
                        num_params: int, output_path: str, save_frequency: int = 50) -> Dict[str, Any]:
    """
    Run batch processing of logos with parameter combinations.

    Args:
        logos: Categorized logo paths
        parameters: List of parameter combinations
        num_params: Number of parameter combinations to use per logo
        output_path: Path to save results
        save_frequency: Save results every N conversions

    Returns:
        Complete results data
    """
    # Load existing results for crash recovery
    results_data = load_existing_results(output_path)

    # Flatten logo list
    all_logos = []
    for category, logo_paths in logos.items():
        all_logos.extend(logo_paths)

    # Select parameter combinations
    if len(parameters) < num_params:
        print(f"⚠️  Only {len(parameters)} parameters available, using all")
        selected_params = parameters
    else:
        selected_params = random.sample(parameters, num_params)

    # Update metadata
    results_data['metadata'].update({
        'total_logos': len(all_logos),
        'total_parameter_combinations': len(selected_params),
        'total_planned_conversions': len(all_logos) * len(selected_params)
    })
    results_data['logo_selection'] = {k: [os.path.basename(p) for p in v] for k, v in logos.items()}

    print(f"🚀 Starting batch processing:")
    print(f"  📁 Logos: {len(all_logos)}")
    print(f"  ⚙️  Parameter combinations: {len(selected_params)}")
    print(f"  🔢 Total conversions planned: {len(all_logos) * len(selected_params)}")

    # Create progress bar
    total_combinations = len(all_logos) * len(selected_params)
    completed = len(results_data['results'])

    with tqdm(total=total_combinations, initial=completed, desc="Processing", unit="conversions") as pbar:
        conversion_count = 0

        for logo_path in all_logos:
            for params in selected_params:
                # Skip if already processed
                if is_already_processed(results_data, logo_path, params):
                    pbar.update(1)
                    continue

                # Process this combination
                try:
                    result = measure_conversion_quality(logo_path, params)
                    results_data['results'].append(result)

                    # Update progress
                    conversion_count += 1
                    pbar.update(1)

                    # Save incrementally
                    if conversion_count % save_frequency == 0:
                        save_results_incrementally(results_data, output_path)
                        pbar.set_postfix({
                            'saved': f"{len(results_data['results'])} results",
                            'success_rate': f"{sum(1 for r in results_data['results'] if r.get('conversion_success', False)) / len(results_data['results']):.1%}"
                        })

                except Exception as e:
                    # Log error but continue processing
                    error_result = {
                        'image_path': logo_path,
                        'parameters': params,
                        'conversion_success': False,
                        'processing_time': 0.0,
                        'file_size_ratio': 0.0,
                        'ssim': 0.0,
                        'mse': float('inf'),
                        'error': f'Batch processing error: {str(e)}'
                    }
                    results_data['results'].append(error_result)
                    pbar.update(1)
                    conversion_count += 1

    # Generate final summary statistics
    results_data['summary_statistics'] = generate_summary_statistics(results_data)

    # Final save
    save_results_incrementally(results_data, output_path)

    print(f"\n✅ Batch processing completed!")
    print(f"📊 Results saved to: {output_path}")

    return results_data


def main():
    """Main entry point for batch parameter testing."""
    parser = argparse.ArgumentParser(description='Batch parameter testing for VTracer')
    parser.add_argument('--logos', type=int, default=50,
                       help='Number of logos to process (default: 50)')
    parser.add_argument('--params', type=int, default=20,
                       help='Number of parameter combinations per logo (default: 20)')
    parser.add_argument('--grid-path', type=str, default='data/training/parameter_grids.json',
                       help='Path to parameter grid from Task 1')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Base data directory path')
    parser.add_argument('--output', type=str, default='data/training/parameter_quality_data.json',
                       help='Output path for results')
    parser.add_argument('--save-frequency', type=int, default=50,
                       help='Save results every N conversions (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')

    args = parser.parse_args()

    # Set random seed for reproducible results
    random.seed(args.seed)

    print(f"🔧 Batch Parameter Testing Configuration:")
    print(f"  📁 Data directory: {args.data_dir}")
    print(f"  📋 Parameter grid: {args.grid_path}")
    print(f"  🎯 Target logos: {args.logos}")
    print(f"  ⚙️  Parameters per logo: {args.params}")
    print(f"  💾 Output file: {args.output}")
    print(f"  🔢 Random seed: {args.seed}")

    # Validate inputs
    if not os.path.exists(args.grid_path):
        print(f"❌ Parameter grid file not found: {args.grid_path}")
        sys.exit(1)

    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Load parameter grid from Task 1
    parameters = load_parameter_grid(args.grid_path)
    if not parameters:
        print(f"❌ No parameters loaded from grid")
        sys.exit(1)

    # Select diverse logo set
    selected_logos = select_diverse_logo_set(args.data_dir)

    # Validate logo selection
    total_logos = sum(len(logos) for logos in selected_logos.values())
    if total_logos == 0:
        print(f"❌ No logos selected")
        sys.exit(1)

    print(f"\n📋 Logo Selection Summary:")
    for category, logos in selected_logos.items():
        print(f"  {category}: {len(logos)} logos")
    print(f"  Total: {total_logos} logos")

    # Run batch processing
    start_time = time.time()
    results = run_batch_processing(
        selected_logos,
        parameters,
        args.params,
        args.output,
        args.save_frequency
    )
    total_time = time.time() - start_time

    # Print final summary
    summary = results.get('summary_statistics', {})
    print(f"\n📊 Final Summary:")
    print(f"  ⏱️  Total processing time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  🔢 Total conversions: {summary.get('total_conversions', 0)}")
    print(f"  ✅ Successful: {summary.get('successful_conversions', 0)}")
    print(f"  ❌ Failed: {summary.get('failed_conversions', 0)}")
    print(f"  📈 Success rate: {summary.get('success_rate', 0):.1%}")

    if 'quality_metrics' in summary:
        qm = summary['quality_metrics']
        print(f"  🎯 Avg SSIM: {qm.get('ssim', {}).get('avg', 0):.3f}")
        print(f"  ⚡ Avg processing time: {qm.get('processing_time', {}).get('avg', 0):.3f}s")

    # Validate acceptance criteria
    total_conversions = summary.get('total_conversions', 0)
    if total_conversions >= 1000:
        print(f"\n✅ Acceptance criteria met:")
        print(f"  - Processed {total_conversions} conversions (target: 1,000)")
        print(f"  - Results saved to {args.output}")
        print(f"  - Progress tracking with ETA implemented")
        print(f"  - Crash recovery support implemented")
    else:
        print(f"\n⚠️  Acceptance criteria partially met ({total_conversions}/1000 conversions)")

    return 0 if total_conversions > 0 else 1


if __name__ == "__main__":
    sys.exit(main())