"""
Statistical Quality Predictor - DAY3 Task 3

Gradient Boosting model for predicting SSIM quality scores based on image features and VTracer parameters.
Includes uncertainty estimation and caching for improved performance.
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
import hashlib
from functools import lru_cache

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalQualityPredictor:
    """Gradient Boosting model for SSIM quality prediction."""

    def __init__(self, model_save_path: Optional[str] = None):
        """
        Initialize the quality predictor.

        Args:
            model_save_path: Path to save/load the trained model
        """
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data" / "training" / "preprocessed"

        # Model save location
        if model_save_path is None:
            self.models_dir = self.project_root / "backend" / "ai_modules" / "models"
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.model_save_path = self.models_dir / "gb_quality_predictor.pkl"
        else:
            self.model_save_path = Path(model_save_path)

        # Feature definitions
        self.image_feature_names = [
            'edge_density', 'unique_colors', 'entropy', 'complexity_score',
            'gradient_strength', 'image_size', 'aspect_ratio'
        ]

        self.parameter_names = [
            'color_precision', 'corner_threshold', 'max_iterations',
            'path_precision', 'layer_difference', 'length_threshold',
            'splice_threshold', 'colormode'
        ]

        # Initialize Gradient Boosting model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            subsample=0.8
        )

        self.is_trained = False
        self.feature_names = None
        self.scaler = None
        self.training_metrics = {}
        self.prediction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed training data with both image features and parameters.

        Returns:
            Tuple of (combined_features, ssim_targets)
        """
        logger.info(f"Loading training data from {self.data_dir}")

        # Load training data
        X_train = pd.read_csv(self.data_dir / "X_train.csv")
        y_train = pd.read_csv(self.data_dir / "y_train.csv")

        # Extract SSIM target (first column in targets)
        ssim_target = y_train[['ssim']]

        # Combine image features with parameters for quality prediction
        # Parameters are also in y_train (columns 1-8)
        parameter_features = y_train[self.parameter_names]

        # Combine all features
        combined_features = pd.concat([X_train, parameter_features], axis=1)

        logger.info(f"Loaded {combined_features.shape[0]} training samples")
        logger.info(f"Combined features: {combined_features.shape[1]} (image: {X_train.shape[1]}, params: {parameter_features.shape[1]})")
        logger.info(f"SSIM target shape: {ssim_target.shape}")

        self.feature_names = list(combined_features.columns)
        return combined_features, ssim_target

    def load_validation_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed validation data.

        Returns:
            Tuple of (combined_features, ssim_targets)
        """
        X_val = pd.read_csv(self.data_dir / "X_val.csv")
        y_val = pd.read_csv(self.data_dir / "y_val.csv")

        ssim_target = y_val[['ssim']]
        parameter_features = y_val[self.parameter_names]
        combined_features = pd.concat([X_val, parameter_features], axis=1)

        return combined_features, ssim_target

    def train_model_with_cv(self) -> bool:
        """
        Train the Gradient Boosting model with cross-validation.

        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Starting quality prediction model training with cross-validation")

            # Load training data
            X_train, y_train = self.load_training_data()
            X_val, y_val = self.load_validation_data()

            # Load feature scaler (for consistency with other models)
            scaler_path = self.data_dir / "feature_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler for image features")

                # Scale only the image features (first 11 columns)
                image_features_train = X_train.iloc[:, :11]
                image_features_val = X_val.iloc[:, :11]

                # Scale image features
                image_features_train_scaled = pd.DataFrame(
                    self.scaler.transform(image_features_train),
                    columns=image_features_train.columns,
                    index=image_features_train.index
                )

                image_features_val_scaled = pd.DataFrame(
                    self.scaler.transform(image_features_val),
                    columns=image_features_val.columns,
                    index=image_features_val.index
                )

                # Combine scaled image features with unscaled parameters
                X_train = pd.concat([
                    image_features_train_scaled,
                    X_train.iloc[:, 11:]  # Parameters (unscaled)
                ], axis=1)

                X_val = pd.concat([
                    image_features_val_scaled,
                    X_val.iloc[:, 11:]  # Parameters (unscaled)
                ], axis=1)

            # Perform cross-validation for model evaluation
            logger.info("Performing 5-fold cross-validation")
            cv_scores = cross_val_score(self.model, X_train, y_train.values.ravel(),
                                      cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()

            logger.info(f"Cross-validation MAE: {cv_mae:.4f} ± {cv_std:.4f}")

            # Train final model on full training set
            start_time = time.time()
            self.model.fit(X_train, y_train.values.ravel())
            training_time = time.time() - start_time

            self.is_trained = True
            logger.info(f"Model training completed in {training_time:.2f}s")

            # Evaluate on training and validation sets
            self._evaluate_model(X_train, y_train, X_val, y_val, cv_mae, cv_std)

            # Save trained model
            self.save_model()

            return True

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def _evaluate_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                       X_val: pd.DataFrame, y_val: pd.DataFrame,
                       cv_mae: float, cv_std: float) -> None:
        """
        Evaluate model performance on training and validation sets.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            cv_mae: Cross-validation MAE
            cv_std: Cross-validation standard deviation
        """
        logger.info("Evaluating model performance")

        # Training predictions
        y_train_pred = self.model.predict(X_train)

        # Validation predictions
        y_val_pred = self.model.predict(X_val)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train.values.ravel(), y_train_pred)
        train_r2 = r2_score(y_train.values.ravel(), y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train.values.ravel(), y_train_pred))

        val_mae = mean_absolute_error(y_val.values.ravel(), y_val_pred)
        val_r2 = r2_score(y_val.values.ravel(), y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val.values.ravel(), y_val_pred))

        # Check ±0.1 SSIM accuracy requirement
        val_within_01 = np.mean(np.abs(y_val.values.ravel() - y_val_pred) <= 0.1) * 100

        logger.info(f"Training - MAE: {train_mae:.4f}, R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
        logger.info(f"Validation - MAE: {val_mae:.4f}, R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
        logger.info(f"Cross-validation - MAE: {cv_mae:.4f} ± {cv_std:.4f}")
        logger.info(f"Predictions within ±0.1 SSIM: {val_within_01:.1f}%")

        # Store metrics
        self.training_metrics = {
            'train_mae': train_mae,
            'train_r2': train_r2,
            'train_rmse': train_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'val_rmse': val_rmse,
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'val_within_01_ssim': val_within_01,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }

    def predict_quality(self, features: Dict[str, float],
                       parameters: Dict[str, float],
                       use_cache: bool = True) -> Dict[str, Any]:
        """
        Predict SSIM quality score for given features and parameters.

        Args:
            features: Dictionary of image features
            parameters: Dictionary of VTracer parameters
            use_cache: Whether to use prediction caching

        Returns:
            Dictionary with predicted SSIM and uncertainty estimates
        """
        if not self.is_trained:
            if not self.load_model():
                return self._create_error_result("Model not trained and failed to load")

        try:
            # Create cache key for this prediction
            cache_key = None
            if use_cache:
                cache_key = self._create_cache_key(features, parameters)
                cached_result = self.prediction_cache.get(cache_key)
                if cached_result is not None:
                    self.cache_hits += 1
                    cached_result['from_cache'] = True
                    return cached_result

            self.cache_misses += 1

            # Prepare features for prediction
            combined_features = self._prepare_features_for_prediction(features, parameters)

            # Make prediction
            start_time = time.time()
            ssim_prediction = self.model.predict(combined_features)[0]
            prediction_time = time.time() - start_time

            # Calculate uncertainty estimates
            uncertainty = self._calculate_uncertainty(combined_features, ssim_prediction)

            # Prepare result
            result = {
                'predicted_ssim': float(ssim_prediction),
                'uncertainty': uncertainty,
                'confidence': max(0.1, min(0.95, 1.0 - uncertainty['std_estimate'])),
                'prediction_time': prediction_time,
                'model_type': 'gradient_boosting',
                'from_cache': False,
                'success': True
            }

            # Cache the result
            if use_cache and cache_key:
                self.prediction_cache[cache_key] = result.copy()

            return result

        except Exception as e:
            logger.error(f"Quality prediction failed: {e}")
            return self._create_error_result(f"Prediction error: {e}")

    def _prepare_features_for_prediction(self, features: Dict[str, float],
                                       parameters: Dict[str, float]) -> pd.DataFrame:
        """
        Prepare combined features for prediction.

        Args:
            features: Image features
            parameters: VTracer parameters

        Returns:
            Prepared feature DataFrame
        """
        if self.feature_names is None:
            raise ValueError("Feature names not available. Model may not be trained.")

        # Combine image features and parameters
        combined_features = {**features, **parameters}

        # Create feature vector with defaults for missing features
        feature_vector = {}

        for feature_name in self.feature_names:
            if feature_name in combined_features:
                feature_vector[feature_name] = combined_features[feature_name]
            else:
                # Provide sensible defaults
                if feature_name in self.parameter_names:
                    if feature_name == 'colormode':
                        feature_vector[feature_name] = 0.5
                    else:
                        feature_vector[feature_name] = 5.0
                elif 'interaction' in feature_name or 'ratio' in feature_name:
                    feature_vector[feature_name] = 0.25
                elif 'log_' in feature_name:
                    feature_vector[feature_name] = np.log(256)
                else:
                    feature_vector[feature_name] = 0.5

                logger.warning(f"Feature {feature_name} missing, using default: {feature_vector[feature_name]}")

        # Convert to DataFrame
        feature_df = pd.DataFrame([feature_vector])

        # Ensure column order matches training
        feature_df = feature_df[self.feature_names]

        # Apply scaling to image features if scaler is available
        if self.scaler is not None:
            # Scale only image features (first 11 columns)
            image_features = feature_df.iloc[:, :11]
            image_features_scaled = pd.DataFrame(
                self.scaler.transform(image_features),
                columns=image_features.columns,
                index=image_features.index
            )

            # Combine scaled image features with unscaled parameters
            feature_df = pd.concat([
                image_features_scaled,
                feature_df.iloc[:, 11:]  # Parameters (unscaled)
            ], axis=1)

        return feature_df

    def _calculate_uncertainty(self, features: pd.DataFrame, prediction: float) -> Dict[str, float]:
        """
        Calculate uncertainty estimates for the prediction.

        Args:
            features: Input features
            prediction: Model prediction

        Returns:
            Dictionary of uncertainty estimates
        """
        try:
            # For Gradient Boosting, we can use staged predictions to estimate uncertainty
            staged_predictions = list(self.model.staged_predict(features))

            if len(staged_predictions) > 10:
                # Use variance of later-stage predictions as uncertainty estimate
                recent_predictions = staged_predictions[-20:]  # Last 20 stages
                prediction_std = np.std(recent_predictions)

                # Normalize std to [0, 1] range based on SSIM range
                normalized_std = min(prediction_std / 0.3, 1.0)  # 0.3 is rough SSIM std

                uncertainty = {
                    'std_estimate': float(prediction_std),
                    'normalized_std': float(normalized_std),
                    'method': 'staged_predictions'
                }
            else:
                # Fallback uncertainty based on training metrics
                val_mae = self.training_metrics.get('val_mae', 0.1)
                uncertainty = {
                    'std_estimate': float(val_mae),
                    'normalized_std': float(min(val_mae / 0.3, 1.0)),
                    'method': 'validation_mae'
                }

            return uncertainty

        except Exception as e:
            logger.warning(f"Uncertainty calculation failed: {e}")
            return {
                'std_estimate': 0.1,
                'normalized_std': 0.33,
                'method': 'default'
            }

    def _create_cache_key(self, features: Dict[str, float], parameters: Dict[str, float]) -> str:
        """
        Create a cache key for the given features and parameters.

        Args:
            features: Image features
            parameters: VTracer parameters

        Returns:
            Cache key string
        """
        # Combine and sort features for consistent key generation
        combined = {**features, **parameters}
        sorted_items = sorted(combined.items())

        # Create hash of the sorted feature string
        feature_string = str(sorted_items)
        return hashlib.md5(feature_string.encode()).hexdigest()

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        Create standardized error result.

        Args:
            error_message: Error description

        Returns:
            Error result dictionary
        """
        return {
            'predicted_ssim': 0.5,  # Default mid-range SSIM
            'uncertainty': {'std_estimate': 1.0, 'normalized_std': 1.0, 'method': 'error'},
            'confidence': 0.0,
            'prediction_time': 0.0,
            'model_type': 'gradient_boosting',
            'from_cache': False,
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
                'image_feature_names': self.image_feature_names,
                'parameter_names': self.parameter_names,
                'training_metrics': self.training_metrics,
                'is_trained': self.is_trained,
                'scaler': self.scaler
            }

            # Save model
            joblib.dump(model_data, self.model_save_path)
            logger.info(f"Model saved to {self.model_save_path}")

            # Save model metadata
            metadata = {
                'model_type': 'Gradient Boosting Regressor',
                'target': 'SSIM quality score',
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
            self.image_feature_names = model_data['image_feature_names']
            self.parameter_names = model_data['parameter_names']
            self.training_metrics = model_data.get('training_metrics', {})
            self.is_trained = model_data['is_trained']
            self.scaler = model_data.get('scaler')

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
            if hasattr(self.model, 'feature_importances_'):
                return dict(zip(self.feature_names, self.model.feature_importances_))
            else:
                logger.warning("Feature importance not available")
                return {}

        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}

    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        self.prediction_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Prediction cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_size': len(self.prediction_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Model information dictionary
        """
        info = {
            'model_name': 'Statistical Quality Predictor',
            'algorithm': 'Gradient Boosting Regressor',
            'target': 'SSIM quality score',
            'is_trained': self.is_trained,
            'model_path': str(self.model_save_path)
        }

        if self.feature_names:
            info['feature_names'] = self.feature_names
            info['feature_count'] = len(self.feature_names)
            info['image_features'] = len(self.image_feature_names)
            info['parameter_features'] = len(self.parameter_names)

        if self.training_metrics:
            info['training_metrics'] = self.training_metrics

        if self.is_trained:
            info['feature_importance'] = self.get_feature_importance()
            info['cache_stats'] = self.get_cache_stats()

        return info


def main():
    """Main training script for quality predictor."""
    predictor = StatisticalQualityPredictor()

    # Train model
    success = predictor.train_model_with_cv()

    if success:
        print("\n" + "="*60)
        print("QUALITY PREDICTOR TRAINING COMPLETED")
        print("="*60)

        # Display model info
        info = predictor.get_model_info()
        print(f"Model: {info['algorithm']}")
        print(f"Target: {info['target']}")
        print(f"Features used: {info['feature_count']}")

        if 'training_metrics' in info:
            metrics = info['training_metrics']
            print(f"Validation MAE: {metrics['val_mae']:.4f}")
            print(f"Validation R²: {metrics['val_r2']:.4f}")
            print(f"Within ±0.1 SSIM: {metrics['val_within_01_ssim']:.1f}%")

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

        test_parameters = {
            'color_precision': 6,
            'corner_threshold': 40,
            'max_iterations': 15,
            'path_precision': 5,
            'layer_difference': 12,
            'length_threshold': 4.0,
            'splice_threshold': 60,
            'colormode': 1.0
        }

        print("\nTesting prediction with sample features and parameters:")
        result = predictor.predict_quality(test_features, test_parameters)

        if result['success']:
            print(f"Predicted SSIM: {result['predicted_ssim']:.4f}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Uncertainty: {result['uncertainty']['std_estimate']:.4f}")
        else:
            print(f"Prediction failed: {result.get('error', 'Unknown error')}")

    else:
        print("Model training failed!")


if __name__ == "__main__":
    main()