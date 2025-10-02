#!/usr/bin/env python3
"""
Train parameter optimizer that learns best VTracer settings for each logo type
"""

import numpy as np
import json
import joblib
import os
from collections import defaultdict

def train_parameter_optimizer():
    print("=" * 60)
    print("Training Parameter Optimizer")
    print("=" * 60)

    # Load training data
    with open("training_data.json") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} training samples")

    # Group by logo type and find best parameters
    best_params_by_type = defaultdict(lambda: {'params': {}, 'score': 0, 'count': 0})
    param_distributions = defaultdict(lambda: defaultdict(list))

    for item in data:
        logo_type = item['logo_type']
        score = item['quality_score']
        params = item['parameters']

        # Track all parameter values and scores
        for param_name, param_value in params.items():
            param_distributions[logo_type][param_name].append({
                'value': param_value,
                'score': score
            })

        # Keep track of best scoring parameters
        if score > best_params_by_type[logo_type]['score']:
            best_params_by_type[logo_type] = {
                'params': params,
                'score': score,
                'file_size': item['file_size'],
                'mse': item.get('mse', 0),
                'psnr': item.get('psnr', 0)
            }

    # Analyze parameter distributions and create optimization rules
    optimization_rules = {}

    for logo_type in param_distributions:
        print(f"\nAnalyzing {logo_type} logos...")
        rules = {}

        for param_name in param_distributions[logo_type]:
            values_scores = param_distributions[logo_type][param_name]

            # Sort by score to find optimal ranges
            sorted_values = sorted(values_scores, key=lambda x: x['score'], reverse=True)

            # Get top performing values
            top_values = [v['value'] for v in sorted_values[:len(sorted_values)//3]]

            if top_values:
                # Calculate optimal value (weighted by score)
                weights = [v['score'] for v in sorted_values[:len(sorted_values)//3]]
                optimal_value = np.average(top_values, weights=weights)

                rules[param_name] = {
                    'optimal': optimal_value,
                    'min': min(top_values),
                    'max': max(top_values),
                    'best': sorted_values[0]['value'],
                    'avg_score': np.mean([v['score'] for v in sorted_values[:len(sorted_values)//3]])
                }

        optimization_rules[logo_type] = rules

        # Print summary
        best = best_params_by_type[logo_type]
        print(f"  Best SSIM: {best['score']:.3f}")
        print(f"  Best params: {best['params']}")

    # Create the correlation models (parameter recommendations by logo type)
    correlation_models = {}

    for logo_type in optimization_rules:
        recommended_params = {}
        for param_name, stats in optimization_rules[logo_type].items():
            # Use the optimal calculated value
            recommended_params[param_name] = stats['optimal']

        correlation_models[logo_type] = recommended_params

    # Add default parameters for unknown types
    correlation_models['default'] = {
        'color_precision': 6,
        'corner_threshold': 60,
        'segment_length': 4.0,
        'path_precision': 6,
        'layer_difference': 5,
        'filter_speckle': 4,
        'splice_threshold': 45
    }

    # Add analysis metadata
    correlation_models['_metadata'] = {
        'training_samples': len(data),
        'logo_types': list(best_params_by_type.keys()),
        'best_scores': {k: v['score'] for k, v in best_params_by_type.items()},
        'optimization_rules': optimization_rules
    }

    # Save models
    os.makedirs("models/production", exist_ok=True)
    joblib.dump(correlation_models, "models/production/correlation_models.pkl")
    print("\n✅ Saved parameter optimizer to models/production/correlation_models.pkl")

    # Print optimization summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)

    for logo_type, params in correlation_models.items():
        if logo_type.startswith('_'):
            continue
        print(f"\n{logo_type.upper()} logos:")
        if isinstance(params, dict):
            for param_name, value in params.items():
                if isinstance(value, (int, float)):
                    print(f"  {param_name}: {value:.1f}")

    print("\n✅ Parameter optimizer trained successfully!")
    print("The optimizer will now suggest optimal VTracer parameters")
    print("based on the logo type classification.")

if __name__ == "__main__":
    train_parameter_optimizer()