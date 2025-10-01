#!/usr/bin/env python3
"""
Parameter Grid Generator for VTracer - Day 1 Task 1
Generates parameter combinations for training data collection.
"""

import json
import itertools
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any

def create_parameter_grid(sampling_strategy='random', num_samples=100):
    """
    Create parameter grid for VTracer based on specifications.

    Args:
        sampling_strategy: 'full' for complete grid, 'random' for random sampling
        num_samples: Number of samples to generate (for random strategy)

    Returns:
        List of parameter dictionaries
    """

    # Define parameter ranges as specified in Day 1 document
    parameter_ranges = {
        'color_precision': [2, 4, 6, 8, 10],
        'corner_threshold': [20, 40, 60, 80],
        'max_iterations': [5, 10, 15, 20],
        'path_precision': [3, 5, 8, 10],
        'layer_difference': [8, 12, 16, 20],
        # Additional parameters found in codebase to make 8 total
        'length_threshold': [1.0, 3.0, 5.0, 8.0],
        'splice_threshold': [30, 45, 60, 75],
        'colormode': ['color', 'binary']
    }

    if sampling_strategy == 'full':
        # Generate full grid (will be very large)
        parameter_combinations = []
        keys = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())

        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            parameter_combinations.append(param_dict)

    elif sampling_strategy == 'random':
        # Random sampling strategy
        parameter_combinations = []

        for _ in range(num_samples):
            param_dict = {}
            for param_name, param_values in parameter_ranges.items():
                param_dict[param_name] = random.choice(param_values)
            parameter_combinations.append(param_dict)

    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    return parameter_combinations

def export_grid_to_json(parameter_grid: List[Dict], output_path: str):
    """
    Export parameter grid to JSON format.

    Args:
        parameter_grid: List of parameter dictionaries
        output_path: Path to save JSON file
    """

    output_data = {
        'metadata': {
            'total_combinations': len(parameter_grid),
            'parameters_per_combination': len(parameter_grid[0]) if parameter_grid else 0,
            'parameter_names': list(parameter_grid[0].keys()) if parameter_grid else [],
            'generation_method': 'random_sampling'
        },
        'parameter_combinations': parameter_grid
    }

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Exported {len(parameter_grid)} parameter combinations to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate VTracer parameter grid')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of parameter combinations to generate (default: 100)')
    parser.add_argument('--strategy', choices=['full', 'random'], default='random',
                       help='Sampling strategy (default: random)')
    parser.add_argument('--output', type=str, default='data/training/parameter_grids.json',
                       help='Output file path (default: data/training/parameter_grids.json)')

    args = parser.parse_args()

    print(f"🔧 Generating parameter grid with {args.strategy} strategy...")
    print(f"📊 Target samples: {args.samples}")

    # Generate parameter grid
    parameter_grid = create_parameter_grid(
        sampling_strategy=args.strategy,
        num_samples=args.samples
    )

    print(f"✅ Generated {len(parameter_grid)} parameter combinations")
    print(f"📋 Each combination has {len(parameter_grid[0])} parameters")

    # Show first few combinations as examples
    print("\n📝 Example combinations:")
    for i, combo in enumerate(parameter_grid[:3]):
        print(f"  {i+1}: {combo}")

    # Export to JSON
    export_grid_to_json(parameter_grid, args.output)

    # Validate output meets acceptance criteria
    if len(parameter_grid) >= 100:
        print("✅ Acceptance criteria met: Generated 100+ parameter combinations")
    else:
        print(f"⚠️  Warning: Only generated {len(parameter_grid)} combinations (target: 100+)")

    if len(parameter_grid[0]) == 8:
        print("✅ Acceptance criteria met: Each combination has all 8 VTracer parameters")
    else:
        print(f"⚠️  Warning: Combinations have {len(parameter_grid[0])} parameters (target: 8)")

if __name__ == "__main__":
    main()