#!/usr/bin/env python3
"""
Machine learning system to predict optimal VTracer parameters.

This script learns from optimization history to predict the best
parameters for new images based on image features.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from utils.ai_detector import create_detector


class ImageFeatureExtractor:
    """Extract features from images for ML prediction."""

    def __init__(self):
        """Initialize the feature extractor."""
        self.detector = create_detector()

    def extract_features(self, image_path: str) -> Dict[str, float]:
        """
        Extract features from an image.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary of features
        """
        try:
            image = Image.open(image_path).convert("RGBA")
            pixels = np.array(image)

            features = {}

            # Basic image properties
            features['width'] = image.width
            features['height'] = image.height
            features['aspect_ratio'] = image.width / image.height
            features['total_pixels'] = image.width * image.height

            # Color analysis
            if pixels.shape[2] == 4:  # Has alpha
                alpha = pixels[:, :, 3]
                rgb = pixels[:, :, :3]

                # Transparency features
                features['has_transparency'] = 1 if np.any(alpha < 255) else 0
                features['transparency_ratio'] = np.sum(alpha < 255) / alpha.size

                # Get non-transparent pixels
                mask = alpha > 0
                if np.any(mask):
                    non_transparent = rgb[mask]
                else:
                    non_transparent = rgb.reshape(-1, 3)
            else:
                features['has_transparency'] = 0
                features['transparency_ratio'] = 0
                non_transparent = pixels.reshape(-1, 3)

            if len(non_transparent) > 0:
                # Color statistics
                features['mean_r'] = np.mean(non_transparent[:, 0])
                features['mean_g'] = np.mean(non_transparent[:, 1])
                features['mean_b'] = np.mean(non_transparent[:, 2])
                features['std_r'] = np.std(non_transparent[:, 0])
                features['std_g'] = np.std(non_transparent[:, 1])
                features['std_b'] = np.std(non_transparent[:, 2])

                # Unique colors (approximation)
                # Reduce color space for faster calculation
                reduced = (non_transparent // 32) * 32
                unique_colors = len(np.unique(reduced.reshape(-1, 3), axis=0))
                features['unique_colors'] = unique_colors
                features['color_complexity'] = unique_colors / len(non_transparent)

                # Edge detection (simple gradient)
                gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
                gy, gx = np.gradient(gray)
                edge_strength = np.sqrt(gx**2 + gy**2)
                features['mean_edge_strength'] = np.mean(edge_strength)
                features['max_edge_strength'] = np.max(edge_strength)
            else:
                # Default values
                for key in ['mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b',
                           'unique_colors', 'color_complexity', 'mean_edge_strength', 'max_edge_strength']:
                    features[key] = 0

            # Logo type detection
            logo_type, confidence, scores = self.detector.detect_logo_type(image_path)

            # One-hot encode logo type
            for t in ['simple', 'text', 'gradient', 'complex']:
                features[f'type_{t}'] = 1 if logo_type == t else 0

            features['detection_confidence'] = confidence

            # Add type scores as features
            for t, score in scores.items():
                features[f'score_{t}'] = score

            return features

        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return {}


class ParameterLearner:
    """Learn optimal parameters from historical data."""

    def __init__(self):
        """Initialize the learner."""
        self.extractor = ImageFeatureExtractor()
        self.models = {}  # One model per parameter
        self.feature_names = None
        self.scaler_X = None
        self.scaler_y = {}

    def prepare_training_data(self, optimization_results: Dict) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare training data from optimization results.

        Args:
            optimization_results: Results from grid search

        Returns:
            X (features) and y (parameter values) arrays
        """
        X = []
        y = {
            'color_precision': [],
            'layer_difference': [],
            'corner_threshold': [],
            'length_threshold': [],
            'max_iterations': [],
            'splice_threshold': [],
            'path_precision': []
        }

        # Extract features and parameters from results
        for category, category_results in optimization_results.items():
            for result in category_results:
                if result.get('best_result'):
                    image_path = result['image']
                    best_params = result['best_result']['params']

                    # Extract features
                    features = self.extractor.extract_features(image_path)

                    if features:
                        # Convert to feature vector
                        if self.feature_names is None:
                            self.feature_names = sorted(features.keys())

                        feature_vector = [features.get(name, 0) for name in self.feature_names]
                        X.append(feature_vector)

                        # Add parameter values
                        for param, value in best_params.items():
                            if param in y:
                                y[param].append(value)

        # Convert to numpy arrays
        X = np.array(X)
        for param in y:
            y[param] = np.array(y[param])

        return X, y

    def train(self, optimization_results_file: str = "parameter_grid_detailed.json"):
        """
        Train models on optimization results.

        Args:
            optimization_results_file: Path to detailed optimization results
        """
        # Load optimization results
        if not Path(optimization_results_file).exists():
            print(f"❌ {optimization_results_file} not found. Run grid search first.")
            return False

        with open(optimization_results_file, 'r') as f:
            results = json.load(f)

        print("Preparing training data...")
        X, y = self.prepare_training_data(results)

        if len(X) == 0:
            print("❌ No training data available")
            return False

        print(f"Training on {len(X)} examples with {X.shape[1]} features")

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)

        # Train a model for each parameter
        for param_name, param_values in y.items():
            if len(param_values) == 0:
                continue

            print(f"\nTraining model for {param_name}...")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, param_values, test_size=0.2, random_state=42
            )

            # Train RandomForest
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"  MSE: {mse:.3f}")
            print(f"  R²: {r2:.3f}")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_features_idx = np.argsort(importances)[-5:]
                print(f"  Top features:")
                for idx in top_features_idx:
                    print(f"    - {self.feature_names[idx]}: {importances[idx]:.3f}")

            self.models[param_name] = model

        return True

    def predict_parameters(self, image_path: str) -> Dict:
        """
        Predict optimal parameters for an image.

        Args:
            image_path: Path to input image

        Returns:
            Dictionary of predicted parameters
        """
        if not self.models:
            print("❌ No trained models available")
            return {}

        # Extract features
        features = self.extractor.extract_features(image_path)

        if not features:
            print("❌ Could not extract features")
            return {}

        # Convert to feature vector
        feature_vector = [features.get(name, 0) for name in self.feature_names]
        X = np.array([feature_vector])

        # Scale features
        if self.scaler_X:
            X = self.scaler_X.transform(X)

        # Predict each parameter
        predictions = {}
        for param_name, model in self.models.items():
            pred = model.predict(X)[0]

            # Round to appropriate precision
            if param_name in ['color_precision', 'layer_difference', 'corner_threshold',
                             'max_iterations', 'splice_threshold', 'path_precision']:
                pred = int(round(pred))
            else:
                pred = round(pred, 1)

            predictions[param_name] = pred

        return predictions

    def save_models(self, output_dir: str = "parameter_models"):
        """Save trained models to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save models
        for param_name, model in self.models.items():
            model_file = output_dir / f"{param_name}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

        # Save feature names and scalers
        metadata = {
            'feature_names': self.feature_names,
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        if self.scaler_X:
            with open(output_dir / 'scaler_X.pkl', 'wb') as f:
                pickle.dump(self.scaler_X, f)

        print(f"✅ Models saved to {output_dir}")

    def load_models(self, model_dir: str = "parameter_models"):
        """Load trained models from disk."""
        model_dir = Path(model_dir)

        if not model_dir.exists():
            print(f"❌ Model directory {model_dir} not found")
            return False

        # Load metadata
        with open(model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            self.feature_names = metadata['feature_names']

        # Load scaler
        scaler_file = model_dir / 'scaler_X.pkl'
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scaler_X = pickle.load(f)

        # Load models
        self.models = {}
        for param_name in ['color_precision', 'layer_difference', 'corner_threshold',
                          'length_threshold', 'max_iterations', 'splice_threshold', 'path_precision']:
            model_file = model_dir / f"{param_name}_model.pkl"
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    self.models[param_name] = pickle.load(f)

        print(f"✅ Loaded {len(self.models)} models from {model_dir}")
        return True


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Learn optimal VTracer parameters")
    parser.add_argument('--train', action='store_true',
                       help='Train models on optimization results')
    parser.add_argument('--predict', help='Predict parameters for an image')
    parser.add_argument('--test-accuracy', action='store_true',
                       help='Test prediction accuracy')

    args = parser.parse_args()

    learner = ParameterLearner()

    if args.train:
        print("="*60)
        print("TRAINING PARAMETER MODELS")
        print("="*60)

        if learner.train():
            learner.save_models()
            print("\n✅ Training complete")

    elif args.predict:
        print("="*60)
        print("PREDICTING PARAMETERS")
        print("="*60)

        # Load models
        if not learner.load_models():
            print("Run with --train first to train models")
            return 1

        # Predict parameters
        params = learner.predict_parameters(args.predict)

        if params:
            print(f"\nPredicted parameters for {args.predict}:")
            for param, value in params.items():
                print(f"  {param}: {value}")

    elif args.test_accuracy:
        print("="*60)
        print("TESTING PREDICTION ACCURACY")
        print("="*60)

        # Load models
        if not learner.load_models():
            print("Run with --train first to train models")
            return 1

        # Test on some images
        test_images = [
            "data/logos/simple_geometric/oval_07.png",
            "data/logos/text_based/text_net_07.png",
            "data/logos/gradients/gradient_radial_04.png"
        ]

        for image_path in test_images:
            if Path(image_path).exists():
                print(f"\n{image_path}:")
                params = learner.predict_parameters(image_path)
                for param, value in params.items():
                    print(f"  {param}: {value}")

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())