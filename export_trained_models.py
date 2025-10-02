#!/usr/bin/env python3
"""
Export trained models to production format for use with AI endpoints
"""

import torch
import joblib
import json
from pathlib import Path
import sys

def export_models():
    """Export all trained models to production format"""

    print("=" * 70)
    print("🚀 EXPORTING TRAINED MODELS TO PRODUCTION")
    print("=" * 70)

    # Create production directory
    prod_dir = Path("models/production")
    prod_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0

    # 1. Export Logo Classifier
    try:
        if Path("logo_classifier.pth").exists():
            print("\n📦 Exporting Logo Classifier...")

            # Load the model
            checkpoint = torch.load("logo_classifier.pth", map_location='cpu')

            # Create a simple neural network for classification
            import torch.nn as nn

            class LogoClassifier(nn.Module):
                def __init__(self, input_size=7, hidden_size=64, num_classes=5):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden_size, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, num_classes)
                    )

                def forward(self, x):
                    return self.network(x)

            # Initialize model
            model = LogoClassifier()

            # Load state dict if available
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume checkpoint is the state dict itself
                model.load_state_dict(checkpoint)

            model.eval()

            # Convert to TorchScript
            example_input = torch.randn(1, 7)
            traced = torch.jit.trace(model, example_input)

            # Save
            output_path = prod_dir / "logo_classifier.torchscript"
            traced.save(str(output_path))
            print(f"   ✅ Saved to {output_path}")
            success_count += 1

        else:
            print("\n⚠️  logo_classifier.pth not found - skipping")

    except Exception as e:
        print(f"   ❌ Error exporting classifier: {e}")

    # 2. Export Quality Predictor
    try:
        if Path("quality_predictor.pth").exists():
            print("\n📦 Exporting Quality Predictor...")

            # Load the model
            checkpoint = torch.load("quality_predictor.pth", map_location='cpu')

            # Create the quality predictor model
            class QualityPredictor(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Input: image features (10) + vtracer params (8) = 18
                    self.network = nn.Sequential(
                        nn.Linear(18, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1),
                        nn.Sigmoid()  # SSIM is 0-1
                    )

                def forward(self, features, params):
                    x = torch.cat([features, params], dim=1)
                    return self.network(x)

            # Initialize and load model
            model = QualityPredictor()

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()

            # Convert to TorchScript with proper inputs
            example_features = torch.randn(1, 10)
            example_params = torch.randn(1, 8)
            traced = torch.jit.trace(model, (example_features, example_params))

            # Save
            output_path = prod_dir / "quality_predictor.torchscript"
            traced.save(str(output_path))
            print(f"   ✅ Saved to {output_path}")
            success_count += 1

        else:
            print("\n⚠️  quality_predictor.pth not found - skipping")

    except Exception as e:
        print(f"   ❌ Error exporting quality predictor: {e}")

    # 3. Export Parameter Optimizer
    try:
        if Path("parameter_optimizer.pkl").exists():
            print("\n📦 Exporting Parameter Optimizer...")

            # Just copy the pickle file
            import shutil
            output_path = prod_dir / "parameter_optimizer.pkl"
            shutil.copy("parameter_optimizer.pkl", output_path)
            print(f"   ✅ Copied to {output_path}")
            success_count += 1

        else:
            print("\n⚠️  parameter_optimizer.pkl not found - skipping")

    except Exception as e:
        print(f"   ❌ Error exporting optimizer: {e}")

    # 4. Also check for correlation_models.pkl (alternative name)
    try:
        if Path("correlation_models.pkl").exists():
            print("\n📦 Found correlation_models.pkl - copying...")
            import shutil
            output_path = prod_dir / "correlation_models.pkl"
            shutil.copy("correlation_models.pkl", output_path)
            print(f"   ✅ Copied to {output_path}")
            success_count += 1
    except:
        pass

    # 5. Create metadata file
    metadata = {
        "export_date": str(Path.cwd()),
        "models_exported": success_count,
        "model_types": {
            "logo_classifier": "torchscript",
            "quality_predictor": "torchscript",
            "parameter_optimizer": "pickle"
        }
    }

    metadata_path = prod_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ Export complete! {success_count} models exported to {prod_dir}/")
    print("=" * 70)

    if success_count == 0:
        print("\n❌ No models were exported!")
        print("Make sure you have trained models in the current directory:")
        print("  - logo_classifier.pth")
        print("  - quality_predictor.pth")
        print("  - parameter_optimizer.pkl")
        print("\nRun training scripts first to generate these models.")
        return False

    return True

if __name__ == "__main__":
    success = export_models()
    sys.exit(0 if success else 1)