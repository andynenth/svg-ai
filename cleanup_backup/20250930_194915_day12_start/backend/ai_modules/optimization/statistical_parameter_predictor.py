"""
Statistical Parameter Predictor - DAY3 Task 2

XGBoost-based model for predicting optimal VTracer parameters based on image features.
Uses multi-output regression to predict all 8 parameters simultaneously.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import time
import sys

# Disable XGBoost due to segmentation fault issues on this system
XGBOOST_AVAILABLE = False
# try:
#     from xgboost import XGBRegressor
#     XGBOOST_AVAILABLE = True
# except ImportError:
#     XGBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalParameterPredictor:
    """XGBoost-based parameter predictor for VTracer optimization."""

    def __init__(self, model_save_path: Optional[str] = None):
        """
        Initialize the parameter predictor.

        Args:
            model_save_path: Path to save/load the trained model
        """
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data" / "training" / "preprocessed"

        # Model save location
        if model_save_path is None:
            self.models_dir = self.project_root / "backend" / "ai_modules" / "models"
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.model_save_path = self.models_dir / "xgb_parameter_predictor.pkl"
        else:
            self.model_save_path = Path(model_save_path)

        # Parameter definitions and bounds
        self.parameter_names = [
            'color_precision', 'corner_threshold', 'max_iterations',
            'path_precision', 'layer_difference', 'length_threshold',
            'splice_threshold', 'colormode'
        ]

        # VTracer parameter bounds for enforcement
        self.parameter_bounds = {
            'color_precision': (1, 10),
            'corner_threshold': (10, 100),
            'max_iterations': (5, 30),
            'path_precision': (1, 10),
            'layer_difference': (1, 30),
            'length_threshold': (1.0, 10.0),
            'splice_threshold': (20, 100),
            'colormode': (0.0, 1.0)  # 0=binary, 1=color
        }

        # Initialize model with fallback options
        try:
            if XGBOOST_AVAILABLE:
                self.base_model = XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )
                self.model_type = "XGBoost"
            else:
                raise ImportError("XGBoost not available")
        except Exception as e:
            logger.warning(f"XGBoost failed to initialize: {e}, falling back to Random Forest")
            self.base_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model_type = "RandomForest"

        self.model = MultiOutputRegressor(self.base_model)
        self.is_trained = False
        self.feature_names = None
        self.scaler = None
        self.training_metrics = {}

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed training data.

        Returns:
            Tuple of (features, parameter_targets)
        """
        logger.info(f"Loading training data from {self.data_dir}")

        # Load training data
        X_train = pd.read_csv(self.data_dir / "X_train.csv")
        y_train = pd.read_csv(self.data_dir / "y_train.csv")

        # Extract parameter targets (exclude SSIM which is the first column)
        parameter_targets = y_train[self.parameter_names]

        logger.info(f"Loaded {X_train.shape[0]} training samples")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Parameter targets: {parameter_targets.shape[1]}")

        self.feature_names = list(X_train.columns)
        return X_train, parameter_targets

    def load_validation_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed validation data.

        Returns:
            Tuple of (features, parameter_targets)
        """
        X_val = pd.read_csv(self.data_dir / "X_val.csv")
        y_val = pd.read_csv(self.data_dir / "y_val.csv")
        parameter_targets = y_val[self.parameter_names]

        return X_val, parameter_targets

    def train_model(self) -> bool:
        """
        Train the XGBoost parameter prediction model.

        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Starting parameter prediction model training")

            # Load training data
            X_train, y_train = self.load_training_data()
            X_val, y_val = self.load_validation_data()

            # Load feature scaler
            scaler_path = self.data_dir / "feature_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            else:
                logger.warning("Feature scaler not found")

            # Train model
            start_time = time.time()
            self.model.fit(X_train, y_train)
            training_time = time.time() - start_time

            self.is_trained = True
            logger.info(f"Model training completed in {training_time:.2f}s")

            # Evaluate on validation set
            self._evaluate_model(X_train, y_train, X_val, y_val)

            # Save trained model
            self.save_model()

            return True

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def _evaluate_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                       X_val: pd.DataFrame, y_val: pd.DataFrame) -> None:
        """
        Evaluate model performance on training and validation sets.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info("Evaluating model performance")

        # Training predictions
        y_train_pred = self.model.predict(X_train)

        # Validation predictions
        y_val_pred = self.model.predict(X_val)

        # Calculate metrics for each parameter
        train_metrics = {}
        val_metrics = {}

        for i, param_name in enumerate(self.parameter_names):
            # Training metrics
            train_mae = mean_absolute_error(y_train.iloc[:, i], y_train_pred[:, i])
            train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])

            # Validation metrics
            val_mae = mean_absolute_error(y_val.iloc[:, i], y_val_pred[:, i])
            val_r2 = r2_score(y_val.iloc[:, i], y_val_pred[:, i])

            train_metrics[param_name] = {'mae': train_mae, 'r2': train_r2}
            val_metrics[param_name] = {'mae': val_mae, 'r2': val_r2}

            logger.info(f"{param_name}: Train MAE={train_mae:.4f}, R²={train_r2:.4f} | "
                       f"Val MAE={val_mae:.4f}, R²={val_r2:.4f}")

        # Overall metrics
        overall_train_mae = np.mean([m['mae'] for m in train_metrics.values()])
        overall_val_mae = np.mean([m['mae'] for m in val_metrics.values()])

        self.training_metrics = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'overall_train_mae': overall_train_mae,
            'overall_val_mae': overall_val_mae,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }

        logger.info(f"Overall Training MAE: {overall_train_mae:.4f}")
        logger.info(f"Overall Validation MAE: {overall_val_mae:.4f}")

    def predict_parameters(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict optimal VTracer parameters for given image features.

        Args:
            features: Dictionary of image features

        Returns:
            Dictionary with predicted parameters and confidence scores
        """
        if not self.is_trained:
            if not self.load_model():
                return self._create_error_result("Model not trained and failed to load")

        try:
            # Convert features to DataFrame
            feature_df = self._prepare_features_for_prediction(features)

            # Make prediction
            start_time = time.time()
            predictions = self.model.predict(feature_df)
            prediction_time = time.time() - start_time

            # Extract single prediction (remove batch dimension)
            pred = predictions[0]

            # Enforce parameter bounds
            bounded_params = self._enforce_parameter_bounds(pred)

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence(feature_df, predictions)

            # Prepare result
            result = {
                'parameters': bounded_params,
                'confidence': confidence_scores,
                'prediction_time': prediction_time,
                'model_type': f'{self.model_type.lower()}_multi_output',
                'success': True
            }

            return result

        except Exception as e:
            logger.error(f"Parameter prediction failed: {e}")
            return self._create_error_result(f"Prediction error: {e}")

    def _prepare_features_for_prediction(self, features: Dict[str, float]) -> pd.DataFrame:
        """
        Prepare features for prediction by ensuring all expected features are present.

        Args:
            features: Input feature dictionary

        Returns:
            Prepared feature DataFrame
        """
        if self.feature_names is None:
            raise ValueError("Feature names not available. Model may not be trained.")

        # Create feature vector with defaults for missing features
        feature_vector = {}

        for feature_name in self.feature_names:
            if feature_name in features:
                feature_vector[feature_name] = features[feature_name]
            else:
                # Provide sensible defaults for missing features
                if 'interaction' in feature_name:
                    feature_vector[feature_name] = 0.25  # Default interaction
                elif 'log_' in feature_name:
                    feature_vector[feature_name] = np.log(256)  # Default log image size
                elif 'ratio' in feature_name:
                    feature_vector[feature_name] = 1.0  # Default ratio
                else:
                    feature_vector[feature_name] = 0.5  # Default mid-range value

                logger.warning(f"Feature {feature_name} missing, using default: {feature_vector[feature_name]}")

        # Convert to DataFrame
        feature_df = pd.DataFrame([feature_vector])

        # Ensure column order matches training
        feature_df = feature_df[self.feature_names]

        return feature_df

    def _enforce_parameter_bounds(self, predictions: np.ndarray) -> Dict[str, float]:
        """
        Enforce VTracer parameter bounds on predictions.

        Args:
            predictions: Raw model predictions

        Returns:
            Dictionary of bounded parameters
        """
        bounded_params = {}

        for i, param_name in enumerate(self.parameter_names):
            raw_value = predictions[i]
            min_val, max_val = self.parameter_bounds[param_name]

            # Clip to bounds
            bounded_value = np.clip(raw_value, min_val, max_val)

            # Special handling for discrete parameters
            if param_name in ['color_precision', 'max_iterations', 'path_precision']:
                bounded_value = int(round(bounded_value))
            elif param_name == 'colormode':
                # Convert to binary decision
                bounded_value = 1.0 if bounded_value > 0.5 else 0.0

            bounded_params[param_name] = bounded_value

        return bounded_params

    def _calculate_confidence(self, features: pd.DataFrame, predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate confidence scores for predictions based on prediction variance.

        Args:
            features: Input features
            predictions: Model predictions

        Returns:
            Dictionary of confidence scores per parameter
        """
        confidence_scores = {}

        try:
            # For XGBoost, we'll use feature importance and prediction consistency
            # as proxies for confidence

            # Base confidence from model training metrics
            base_confidence = 0.7  # Default moderate confidence

            # Adjust confidence based on validation metrics if available
            if hasattr(self, 'training_metrics') and self.training_metrics:
                val_metrics = self.training_metrics.get('val_metrics', {})

                for i, param_name in enumerate(self.parameter_names):
                    param_metrics = val_metrics.get(param_name, {})
                    r2_score = param_metrics.get('r2', 0.5)

                    # Convert R² to confidence (higher R² = higher confidence)
                    param_confidence = max(0.1, min(0.95, base_confidence + r2_score * 0.3))
                    confidence_scores[param_name] = param_confidence
            else:
                # Default confidence if no metrics available
                for param_name in self.parameter_names:
                    confidence_scores[param_name] = base_confidence

            # Overall confidence
            confidence_scores['overall'] = np.mean(list(confidence_scores.values()))

        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            # Fallback to moderate confidence
            for param_name in self.parameter_names:
                confidence_scores[param_name] = 0.6
            confidence_scores['overall'] = 0.6

        return confidence_scores

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        Create standardized error result.

        Args:
            error_message: Error description

        Returns:
            Error result dictionary
        """
        return {
            'parameters': {name: 5.0 for name in self.parameter_names},  # Default parameters
            'confidence': {name: 0.0 for name in self.parameter_names + ['overall']},
            'prediction_time': 0.0,
            'model_type': f'{getattr(self, "model_type", "unknown").lower()}_multi_output',
            'error': error_message,
            'success': False
        }

    def save_model(self) -> bool:
        """
        Save the trained model to disk.

        Returns:
            True if save successful, False otherwise
        """
        if not self.is_trained:
            logger.error("Cannot save untrained model")
            return False

        try:
            # Prepare model data
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'parameter_names': self.parameter_names,
                'parameter_bounds': self.parameter_bounds,
                'training_metrics': self.training_metrics,
                'is_trained': self.is_trained
            }

            # Save model
            joblib.dump(model_data, self.model_save_path)
            logger.info(f"Model saved to {self.model_save_path}")

            # Save model metadata
            metadata = {
                'model_type': f'{self.model_type} MultiOutputRegressor',
                'parameters_predicted': self.parameter_names,
                'training_metrics': self.training_metrics,
                'save_timestamp': pd.Timestamp.now().isoformat(),
                'model_path': str(self.model_save_path)
            }

            metadata_path = self.model_save_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self) -> bool:
        """
        Load trained model from disk.

        Returns:
            True if load successful, False otherwise
        """
        if not self.model_save_path.exists():
            logger.error(f"Model file not found: {self.model_save_path}")
            return False

        try:
            # Load model data
            model_data = joblib.load(self.model_save_path)

            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.parameter_names = model_data['parameter_names']
            self.parameter_bounds = model_data['parameter_bounds']
            self.training_metrics = model_data.get('training_metrics', {})
            self.is_trained = model_data['is_trained']

            logger.info(f"Model loaded from {self.model_save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.

        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_trained:
            logger.error("Model not trained")
            return {}

        try:
            # Average feature importance across all output models
            importance_scores = []

            for estimator in self.model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importance_scores.append(estimator.feature_importances_)

            if importance_scores:
                avg_importance = np.mean(importance_scores, axis=0)
                return dict(zip(self.feature_names, avg_importance))
            else:
                logger.warning("Feature importance not available")
                return {}

        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Model information dictionary
        """
        info = {
            'model_name': 'Statistical Parameter Predictor',
            'algorithm': f'{getattr(self, "model_type", "Unknown")} MultiOutputRegressor',
            'parameters_predicted': self.parameter_names,
            'is_trained': self.is_trained,
            'model_path': str(self.model_save_path)
        }

        if self.feature_names:
            info['feature_names'] = self.feature_names
            info['feature_count'] = len(self.feature_names)

        if self.training_metrics:
            info['training_metrics'] = self.training_metrics

        if self.is_trained:
            info['feature_importance'] = self.get_feature_importance()

        return info


def main():
    """Main training script for parameter predictor."""
    predictor = StatisticalParameterPredictor()

    # Train model
    success = predictor.train_model()

    if success:
        print("\n" + "="*60)
        print("PARAMETER PREDICTOR TRAINING COMPLETED")
        print("="*60)

        # Display model info
        info = predictor.get_model_info()
        print(f"Model: {info['algorithm']}")
        print(f"Parameters predicted: {len(info['parameters_predicted'])}")
        print(f"Features used: {info['feature_count']}")

        if 'training_metrics' in info:
            metrics = info['training_metrics']
            print(f"Overall validation MAE: {metrics['overall_val_mae']:.4f}")

        # Test prediction
        test_features = {
            'edge_density': 0.5,
            'unique_colors': 0.3,
            'entropy': 0.7,
            'complexity_score': 0.4,
            'gradient_strength': 0.2,
            'image_size': 256,
            'aspect_ratio': 1.0
        }

        print("\nTesting prediction with sample features:")
        result = predictor.predict_parameters(test_features)

        if result['success']:
            print("Predicted parameters:")
            for param, value in result['parameters'].items():
                confidence = result['confidence'][param]
                print(f"  {param}: {value:.3f} (confidence: {confidence:.3f})")
        else:
            print(f"Prediction failed: {result.get('error', 'Unknown error')}")

    else:
        print("Model training failed!")


if __name__ == "__main__":
    main()