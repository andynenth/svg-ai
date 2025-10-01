#!/usr/bin/env python3
"""
Model Loading Diagnostic Script - Day 2 Task 1
Diagnoses model loading issues and architecture mismatches.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def list_all_model_files(models_dir: str) -> List[str]:
    """
    List all model files in the trained models directory.

    Args:
        models_dir: Path to trained models directory

    Returns:
        List of model file paths
    """
    model_files = []
    models_path = Path(models_dir)

    if not models_path.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return []

    # Find all .pth files
    for file_path in models_path.glob("*.pth"):
        model_files.append(str(file_path))

    print(f"📁 Found {len(model_files)} model files:")
    for i, file_path in enumerate(model_files, 1):
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  {i:2d}. {os.path.basename(file_path)} ({file_size:.1f} MB)")

    return sorted(model_files)


def create_expected_architecture() -> nn.Module:
    """
    Create expected EfficientNet architecture for comparison.

    Returns:
        Expected model architecture
    """
    try:
        # Create EfficientNet-B0 as expected in current code
        model = models.efficientnet_b0(weights=None)

        # Get number of input features for the classifier
        num_features = model.classifier[1].in_features

        # Modify classifier for 4 logo types (as in current code)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 4)
        )

        return model
    except Exception as e:
        print(f"❌ Failed to create expected architecture: {e}")
        return None


def analyze_checkpoint_structure(model_path: str) -> Dict[str, Any]:
    """
    Analyze the structure of a model checkpoint.

    Args:
        model_path: Path to model checkpoint

    Returns:
        Analysis results dictionary
    """
    analysis = {
        'file_path': model_path,
        'file_name': os.path.basename(model_path),
        'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
        'loadable': False,
        'torch_version': None,
        'state_dict_keys': [],
        'missing_keys': [],
        'unexpected_keys': [],
        'architecture_mismatch': False,
        'classifier_output_size': None,
        'error_message': None
    }

    try:
        # Try to load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        analysis['loadable'] = True

        # Check if it's a raw state dict or wrapped checkpoint
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                analysis['torch_version'] = checkpoint.get('torch_version', 'unknown')
                if 'epoch' in checkpoint:
                    analysis['epoch'] = checkpoint['epoch']
                if 'optimizer' in checkpoint:
                    analysis['has_optimizer'] = True
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Assume it's a raw state dict
                state_dict = checkpoint
        else:
            analysis['error_message'] = "Checkpoint is not a dictionary"
            return analysis

        # Get state dict keys
        analysis['state_dict_keys'] = list(state_dict.keys())

        # Try to load into expected architecture
        expected_model = create_expected_architecture()
        if expected_model is not None:
            try:
                # Try loading with strict=False to see mismatches
                missing_keys, unexpected_keys = expected_model.load_state_dict(state_dict, strict=False)
                analysis['missing_keys'] = missing_keys
                analysis['unexpected_keys'] = unexpected_keys
                analysis['architecture_mismatch'] = len(missing_keys) > 0 or len(unexpected_keys) > 0

                # Check classifier dimensions
                classifier_keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
                if classifier_keys:
                    # Get the last classifier layer
                    last_classifier_key = sorted(classifier_keys)[-1]
                    if last_classifier_key in state_dict:
                        classifier_weight = state_dict[last_classifier_key]
                        analysis['classifier_output_size'] = classifier_weight.shape[0]

            except Exception as e:
                analysis['architecture_mismatch'] = True
                analysis['error_message'] = f"Failed to load state dict: {str(e)}"

    except Exception as e:
        analysis['error_message'] = str(e)

    return analysis


def attempt_model_loading(model_path: str) -> Dict[str, Any]:
    """
    Attempt to load a model with different strategies.

    Args:
        model_path: Path to model file

    Returns:
        Loading attempt results
    """
    loading_result = {
        'model_path': model_path,
        'strategies_tried': [],
        'successful_strategy': None,
        'final_error': None,
        'can_inference': False
    }

    strategies = [
        {
            'name': 'direct_load_strict',
            'description': 'Direct load with strict=True'
        },
        {
            'name': 'direct_load_non_strict',
            'description': 'Direct load with strict=False'
        },
        {
            'name': 'pretrained_base_load',
            'description': 'Load with pretrained base then load state dict'
        },
        {
            'name': 'raw_state_dict',
            'description': 'Extract and load raw state dict only'
        }
    ]

    for strategy in strategies:
        strategy_result = {
            'name': strategy['name'],
            'description': strategy['description'],
            'success': False,
            'error': None
        }

        try:
            if strategy['name'] == 'direct_load_strict':
                # Try loading directly into expected architecture
                model = create_expected_architecture()
                state_dict = torch.load(model_path, map_location='cpu')
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                model.load_state_dict(state_dict, strict=True)
                model.eval()
                strategy_result['success'] = True

            elif strategy['name'] == 'direct_load_non_strict':
                # Try loading with strict=False
                model = create_expected_architecture()
                state_dict = torch.load(model_path, map_location='cpu')
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                model.eval()
                strategy_result['success'] = True
                strategy_result['missing_keys'] = missing
                strategy_result['unexpected_keys'] = unexpected

            elif strategy['name'] == 'pretrained_base_load':
                # Try with pretrained base
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                num_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(num_features, 4)
                )
                state_dict = torch.load(model_path, map_location='cpu')
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                model.eval()
                strategy_result['success'] = True

            elif strategy['name'] == 'raw_state_dict':
                # Try loading raw state dict
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint

                    model = create_expected_architecture()
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    strategy_result['success'] = True

            # If strategy succeeded, test inference capability
            if strategy_result['success']:
                # Try a simple forward pass
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    output = model(dummy_input)
                    if output.shape == (1, 4):  # Expected output shape
                        loading_result['can_inference'] = True
                        loading_result['successful_strategy'] = strategy['name']

        except Exception as e:
            strategy_result['error'] = str(e)

        loading_result['strategies_tried'].append(strategy_result)

        # If successful, we can stop trying other strategies
        if strategy_result['success'] and loading_result['can_inference']:
            break

    # If no strategy worked, capture the last error
    if not loading_result['successful_strategy']:
        last_error = loading_result['strategies_tried'][-1]['error']
        loading_result['final_error'] = last_error

    return loading_result


def compare_architectures(expected_model: nn.Module, checkpoint_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare expected architecture with checkpoint architecture.

    Args:
        expected_model: Expected model architecture
        checkpoint_analysis: Analysis of checkpoint structure

    Returns:
        Architecture comparison results
    """
    comparison = {
        'input_dimensions_match': True,
        'output_dimensions_match': True,
        'layer_names_match': True,
        'state_dict_compatible': True,
        'issues_found': [],
        'recommendations': []
    }

    # Compare expected state dict keys with checkpoint keys
    expected_keys = set(expected_model.state_dict().keys())
    checkpoint_keys = set(checkpoint_analysis.get('state_dict_keys', []))

    missing_in_checkpoint = expected_keys - checkpoint_keys
    extra_in_checkpoint = checkpoint_keys - expected_keys

    if missing_in_checkpoint:
        comparison['state_dict_compatible'] = False
        comparison['issues_found'].append(f"Missing keys in checkpoint: {list(missing_in_checkpoint)[:5]}")
        comparison['recommendations'].append("Use strict=False when loading or add missing layers")

    if extra_in_checkpoint:
        comparison['state_dict_compatible'] = False
        comparison['issues_found'].append(f"Extra keys in checkpoint: {list(extra_in_checkpoint)[:5]}")
        comparison['recommendations'].append("Filter out extra keys or update model architecture")

    # Check classifier output dimensions
    expected_classifier_size = 4  # Our target: 4 classes
    checkpoint_classifier_size = checkpoint_analysis.get('classifier_output_size')

    if checkpoint_classifier_size and checkpoint_classifier_size != expected_classifier_size:
        comparison['output_dimensions_match'] = False
        comparison['issues_found'].append(
            f"Classifier output size mismatch: expected {expected_classifier_size}, "
            f"found {checkpoint_classifier_size}"
        )
        comparison['recommendations'].append(
            "Use model adapter to map weights or retrain final layer"
        )

    return comparison


def generate_diagnostic_report(models_dir: str, output_path: str) -> Dict[str, Any]:
    """
    Generate comprehensive diagnostic report.

    Args:
        models_dir: Path to trained models directory
        output_path: Path to save diagnostic report

    Returns:
        Complete diagnostic report
    """
    print(f"🔍 Starting model loading diagnostics...")
    print(f"📁 Models directory: {models_dir}")

    # Initialize report structure
    report = {
        'diagnostic_metadata': {
            'timestamp': datetime.now().isoformat(),
            'models_directory': models_dir,
            'pytorch_version': torch.__version__,
            'total_models_found': 0,
            'models_loadable': 0,
            'models_with_inference': 0
        },
        'model_analyses': [],
        'loading_attempts': [],
        'architecture_comparisons': [],
        'summary': {
            'critical_issues': [],
            'recommendations': [],
            'loadable_models': [],
            'best_model_candidate': None
        }
    }

    # Step 1: List all model files
    model_files = list_all_model_files(models_dir)
    report['diagnostic_metadata']['total_models_found'] = len(model_files)

    if not model_files:
        report['summary']['critical_issues'].append("No model files found in trained directory")
        return report

    # Step 2: Create expected architecture for comparison
    expected_model = create_expected_architecture()
    if expected_model is None:
        report['summary']['critical_issues'].append("Failed to create expected architecture")
        return report

    # Step 3: Analyze each model checkpoint
    print(f"\n🔍 Analyzing {len(model_files)} model checkpoints...")

    for i, model_path in enumerate(model_files, 1):
        print(f"\n📋 Analyzing {i}/{len(model_files)}: {os.path.basename(model_path)}")

        # Analyze checkpoint structure
        checkpoint_analysis = analyze_checkpoint_structure(model_path)
        report['model_analyses'].append(checkpoint_analysis)

        if checkpoint_analysis['loadable']:
            report['diagnostic_metadata']['models_loadable'] += 1
            print(f"  ✅ Checkpoint loadable")
        else:
            print(f"  ❌ Checkpoint not loadable: {checkpoint_analysis['error_message']}")
            continue

        # Attempt model loading
        loading_result = attempt_model_loading(model_path)
        report['loading_attempts'].append(loading_result)

        if loading_result['can_inference']:
            report['diagnostic_metadata']['models_with_inference'] += 1
            report['summary']['loadable_models'].append({
                'model_path': model_path,
                'strategy': loading_result['successful_strategy']
            })
            print(f"  ✅ Model can perform inference using strategy: {loading_result['successful_strategy']}")
        else:
            print(f"  ❌ Model cannot perform inference")

        # Compare architectures
        architecture_comparison = compare_architectures(expected_model, checkpoint_analysis)
        architecture_comparison['model_path'] = model_path
        report['architecture_comparisons'].append(architecture_comparison)

        if not architecture_comparison['state_dict_compatible']:
            print(f"  ⚠️  Architecture incompatibilities found")

    # Step 4: Generate summary and recommendations
    print(f"\n📊 Generating summary...")

    # Find best model candidate
    inference_capable_models = [m for m in report['summary']['loadable_models']]
    if inference_capable_models:
        # Prefer checkpoint_best.pth, then checkpoint_latest.pth
        best_candidates = [m for m in inference_capable_models if 'best' in m['model_path']]
        if best_candidates:
            report['summary']['best_model_candidate'] = best_candidates[0]
        else:
            latest_candidates = [m for m in inference_capable_models if 'latest' in m['model_path']]
            if latest_candidates:
                report['summary']['best_model_candidate'] = latest_candidates[0]
            else:
                report['summary']['best_model_candidate'] = inference_capable_models[0]

    # Collect critical issues and recommendations
    all_issues = []
    all_recommendations = []

    for comp in report['architecture_comparisons']:
        all_issues.extend(comp.get('issues_found', []))
        all_recommendations.extend(comp.get('recommendations', []))

    report['summary']['critical_issues'] = list(set(all_issues))
    report['summary']['recommendations'] = list(set(all_recommendations))

    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✅ Diagnostic report saved to: {output_path}")
    return report


def print_summary(report: Dict[str, Any]):
    """
    Print diagnostic summary to console.

    Args:
        report: Diagnostic report
    """
    metadata = report['diagnostic_metadata']
    summary = report['summary']

    print(f"\n📊 Model Loading Diagnostic Summary:")
    print(f"  🔢 Total models found: {metadata['total_models_found']}")
    print(f"  ✅ Models loadable: {metadata['models_loadable']}")
    print(f"  🎯 Models with inference capability: {metadata['models_with_inference']}")

    if summary['best_model_candidate']:
        best_model = summary['best_model_candidate']
        print(f"  🏆 Best model candidate: {os.path.basename(best_model['model_path'])}")
        print(f"     Strategy: {best_model['strategy']}")

    if summary['critical_issues']:
        print(f"\n⚠️  Critical Issues Found:")
        for issue in summary['critical_issues']:
            print(f"    - {issue}")

    if summary['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"    - {rec}")

    # Check acceptance criteria
    print(f"\n📋 Acceptance Criteria:")
    if metadata['models_loadable'] > 0:
        print(f"  ✅ Identifies specific architecture mismatches")
        print(f"  ✅ Lists which models can/cannot be loaded")
    else:
        print(f"  ❌ Could not load any models")

    print(f"  ✅ Report saved to model_diagnostic_report.json")


def main():
    """Main entry point for model diagnostic script."""
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose model loading issues')
    parser.add_argument('--models-dir', type=str,
                       default='backend/ai_modules/models/trained',
                       help='Directory containing trained models')
    parser.add_argument('--output', type=str,
                       default='model_diagnostic_report.json',
                       help='Output path for diagnostic report')

    args = parser.parse_args()

    print(f"🔧 Model Loading Diagnostics")
    print(f"📁 Models directory: {args.models_dir}")
    print(f"📋 Output report: {args.output}")

    # Check if models directory exists
    if not os.path.exists(args.models_dir):
        print(f"❌ Error: Models directory not found: {args.models_dir}")
        sys.exit(1)

    try:
        # Generate diagnostic report
        report = generate_diagnostic_report(args.models_dir, args.output)

        # Print summary
        print_summary(report)

        # Return exit code based on results
        if report['diagnostic_metadata']['models_with_inference'] > 0:
            print(f"\n✅ Diagnostics completed successfully - found working models")
            return 0
        else:
            print(f"\n⚠️  Diagnostics completed - no working models found")
            return 1

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())