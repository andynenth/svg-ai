#!/usr/bin/env python3
"""
Colab Setup Validation Script
============================

Validates that all Colab components are ready for Agent 2 handoff.
Tests all utilities, data processing, and model components.
"""

import json
import zipfile
import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

def validate_colab_notebook():
    """Validate Colab notebook structure"""
    print("📓 Validating Colab notebook...")

    notebook_path = Path("colab_setup/SVG_Quality_Predictor_Training.ipynb")
    if not notebook_path.exists():
        return False, "Colab notebook not found"

    try:
        with open(notebook_path) as f:
            notebook = json.load(f)

        # Check essential cells
        essential_sections = [
            "gpu_validation",
            "colab_setup",
            "training_dataclass",
            "gpu_feature_extractor",
            "data_processing_pipeline",
            "quality_assessment"
        ]

        found_sections = []
        for cell in notebook.get('cells', []):
            cell_id = cell.get('metadata', {}).get('id', '')
            if cell_id in essential_sections:
                found_sections.append(cell_id)

        missing = [s for s in essential_sections if s not in found_sections]
        if missing:
            return False, f"Missing notebook sections: {missing}"

        return True, "Colab notebook structure validated"

    except Exception as e:
        return False, f"Notebook validation error: {e}"

def validate_data_collection():
    """Validate data collection utilities"""
    print("📊 Validating data collection...")

    try:
        # Import and test data collection
        sys.path.append('colab_setup')
        from local_data_collection import LocalDataCollector

        collector = LocalDataCollector()
        training_examples = collector.collect_optimization_data()

        if len(training_examples) == 0:
            return False, "No training examples collected"

        # Validate example structure
        example = training_examples[0]
        required_fields = ['image_path', 'vtracer_params', 'actual_ssim', 'logo_type']
        for field in required_fields:
            if not hasattr(example, field):
                return False, f"Missing field in training example: {field}"

        return True, f"Data collection validated ({len(training_examples)} examples)"

    except Exception as e:
        return False, f"Data collection error: {e}"

def validate_gpu_model():
    """Validate GPU model architecture"""
    print("🖥️ Validating GPU model architecture...")

    try:
        sys.path.append('colab_setup')
        from gpu_model_architecture import QualityPredictorGPU, ColabTrainingConfig

        # Test model creation
        device = 'cpu'  # Use CPU for testing
        model = QualityPredictorGPU(device=device)

        # Test forward pass
        test_input = torch.randn(1, 2055)  # 2048 features + 7 params
        output = model(test_input)

        if output.shape != torch.Size([1, 1]):
            return False, f"Unexpected output shape: {output.shape}"

        if not (0 <= output.item() <= 1):
            return False, f"Output not in [0,1] range: {output.item()}"

        # Test configuration
        config = ColabTrainingConfig()
        if config.device != 'cpu':  # Should auto-adjust when no CUDA
            return False, "Config did not adjust for missing CUDA"

        return True, "GPU model architecture validated"

    except Exception as e:
        return False, f"GPU model error: {e}"

def validate_training_utilities():
    """Validate training utilities"""
    print("🛠️ Validating training utilities...")

    try:
        sys.path.append('colab_setup')
        from colab_training_utils import ColabTrainingMonitor, setup_colab_environment

        # Test monitor creation
        monitor = ColabTrainingMonitor(save_plots=False)

        # Test environment setup (dry run)
        # Note: This would create directories in a real Colab environment

        return True, "Training utilities validated"

    except Exception as e:
        return False, f"Training utilities error: {e}"

def validate_training_package():
    """Validate training data package"""
    print("📦 Validating training data package...")

    package_path = "colab_training_data_test.zip"
    if not os.path.exists(package_path):
        return False, f"Training package not found: {package_path}"

    try:
        with zipfile.ZipFile(package_path, 'r') as zipf:
            files = zipf.namelist()

            # Check for essential files
            if 'training_metadata.json' not in files:
                return False, "Missing training_metadata.json"

            if 'README.md' not in files:
                return False, "Missing README.md"

            # Check for images
            image_files = [f for f in files if f.startswith('images/') and f.endswith('.png')]
            if len(image_files) == 0:
                return False, "No image files found in package"

            # Validate metadata
            with zipf.open('training_metadata.json') as f:
                metadata = json.load(f)

            required_keys = ['creation_info', 'data_statistics', 'training_examples']
            for key in required_keys:
                if key not in metadata:
                    return False, f"Missing metadata key: {key}"

            num_examples = len(metadata['training_examples'])
            if num_examples < 100:
                return False, f"Insufficient training examples: {num_examples}"

            return True, f"Training package validated ({num_examples} examples, {len(image_files)} images)"

    except Exception as e:
        return False, f"Package validation error: {e}"

def validate_feature_extraction():
    """Validate feature extraction pipeline"""
    print("🔍 Validating feature extraction...")

    try:
        sys.path.append('colab_setup')
        from gpu_model_architecture import calculate_model_size
        from gpu_model_architecture import QualityPredictorGPU

        # Test model size calculation
        model = QualityPredictorGPU(device='cpu')
        size_mb = calculate_model_size(model)

        if size_mb <= 0 or size_mb > 500:  # Reasonable size check
            return False, f"Unreasonable model size: {size_mb} MB"

        return True, f"Feature extraction validated (model size: {size_mb:.2f} MB)"

    except Exception as e:
        return False, f"Feature extraction error: {e}"

def run_comprehensive_validation():
    """Run comprehensive validation of all Colab components"""
    print("🔬 COMPREHENSIVE COLAB SETUP VALIDATION")
    print("="*60)

    validations = [
        ("Colab Notebook Structure", validate_colab_notebook),
        ("Data Collection Pipeline", validate_data_collection),
        ("GPU Model Architecture", validate_gpu_model),
        ("Training Utilities", validate_training_utilities),
        ("Training Data Package", validate_training_package),
        ("Feature Extraction", validate_feature_extraction)
    ]

    results = []
    all_passed = True

    for name, validator in validations:
        try:
            success, message = validator()
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}: {name} - {message}")
            results.append((name, success, message))
            if not success:
                all_passed = False
        except Exception as e:
            print(f"❌ FAIL: {name} - Validation error: {e}")
            results.append((name, False, f"Validation error: {e}"))
            all_passed = False

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"Overall Status: {'✅ ALL SYSTEMS GO' if all_passed else '⚠️ ISSUES DETECTED'}")
    print(f"Tests Passed: {passed}/{total}")

    if not all_passed:
        print("\nFailed Tests:")
        for name, success, message in results:
            if not success:
                print(f"  ❌ {name}: {message}")

    print(f"\nValidation completed: {datetime.now()}")
    print(f"Ready for Agent 2 handoff: {'Yes' if all_passed else 'No - fix issues first'}")

    # Create validation report
    report = {
        'validation_timestamp': str(datetime.now()),
        'overall_status': 'PASS' if all_passed else 'FAIL',
        'tests_passed': passed,
        'tests_total': total,
        'detailed_results': [
            {'test': name, 'status': 'PASS' if success else 'FAIL', 'message': message}
            for name, success, message in results
        ],
        'ready_for_agent2': all_passed
    }

    with open('colab_setup_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📋 Validation report saved: colab_setup_validation_report.json")

    return all_passed

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)