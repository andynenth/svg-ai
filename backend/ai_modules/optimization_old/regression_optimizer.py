"""
Regression-based Formula Optimization for Method 1
Implements machine learning-based parameter optimization
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import json

from .correlation_analysis import CorrelationAnalysis
from .refined_correlation_formulas import RefinedCorrelationFormulas

logger = logging.getLogger(__name__)


class RegressionBasedOptimizer:
    """Machine learning-based parameter optimization"""

    def __init__(self):
        """Initialize regression-based optimizer"""
        self.models = {}
        self.scalers = {}
        self.trained = False
        self.feature_importance = {}
        self.model_performance = {}

        # Parameter targets to optimize
        self.target_parameters = [
            'corner_threshold', 'color_precision', 'path_precision',
            'length_threshold', 'splice_threshold', 'max_iterations'
        ]

        # Input features
        self.input_features = [
            'edge_density', 'unique_colors', 'entropy',
            'corner_density', 'gradient_strength', 'complexity_score'
        ]

    def prepare_training_data(self, validation_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training data from validation results"""
        try:
            # Filter successful optimizations
            successful_data = validation_data[validation_data['success'] == True].copy()

            if len(successful_data) < 10:
                raise ValueError("Insufficient training data (need at least 10 successful optimizations)")

            # Extract features
            feature_matrix = []
            for _, row in successful_data.iterrows():
                features = row['features']
                if isinstance(features, dict):
                    feature_row = []
                    for feature_name in self.input_features:
                        value = features.get(feature_name, 0.0)
                        feature_row.append(value)
                    feature_matrix.append(feature_row)

            X = np.array(feature_matrix)

            # Extract target parameters
            y_dict = {}
            for param in self.target_parameters:
                param_values = []
                for _, row in successful_data.iterrows():
                    params = row['optimized_params']
                    if isinstance(params, dict):
                        value = params.get(param, 0)
                        param_values.append(value)
                y_dict[param] = np.array(param_values)

            logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y_dict

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise

    def train_models(self, validation_data: pd.DataFrame) -> Dict[str, Any]:
        """Train regression models for parameter optimization"""
        try:
            logger.info("Training regression models for parameter optimization")

            # Prepare training data
            X, y_dict = self.prepare_training_data(validation_data)

            if X.shape[0] < 10:
                raise ValueError("Insufficient training samples")

            # Train models for each parameter
            training_results = {}

            for param in self.target_parameters:
                if param not in y_dict:
                    logger.warning(f"No target data for parameter: {param}")
                    continue

                y = y_dict[param]

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Try multiple algorithms
                algorithms = {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'ridge': Ridge(alpha=1.0),
                    'linear': LinearRegression()
                }

                best_model = None
                best_score = -np.inf
                best_algorithm = None

                for alg_name, model in algorithms.items():
                    try:
                        # Train model
                        model.fit(X_train_scaled, y_train)

                        # Evaluate
                        y_pred = model.predict(X_test_scaled)
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)

                        # Cross-validation
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                        cv_mean = np.mean(cv_scores)

                        logger.info(f"{param} - {alg_name}: R²={r2:.3f}, MSE={mse:.3f}, CV R²={cv_mean:.3f}")

                        # Select best model
                        if cv_mean > best_score:
                            best_score = cv_mean
                            best_model = model
                            best_algorithm = alg_name

                    except Exception as e:
                        logger.warning(f"Error training {alg_name} for {param}: {e}")

                if best_model is not None:
                    self.models[param] = best_model
                    self.scalers[param] = scaler

                    # Feature importance (if available)
                    if hasattr(best_model, 'feature_importances_'):
                        importance = dict(zip(self.input_features, best_model.feature_importances_))
                        self.feature_importance[param] = importance

                    # Performance metrics
                    y_pred_best = best_model.predict(X_test_scaled)
                    self.model_performance[param] = {
                        'algorithm': best_algorithm,
                        'r2_score': r2_score(y_test, y_pred_best),
                        'mse': mean_squared_error(y_test, y_pred_best),
                        'cv_score': best_score,
                        'training_samples': X_train.shape[0]
                    }

                    training_results[param] = self.model_performance[param]

                else:
                    logger.warning(f"Failed to train model for parameter: {param}")

            self.trained = len(self.models) > 0

            logger.info(f"Successfully trained models for {len(self.models)} parameters")
            return training_results

        except Exception as e:
            logger.error(f"Error training regression models: {e}")
            raise

    def predict_parameters(self, features: Dict[str, float], logo_type: str = 'simple') -> Dict[str, Any]:
        """Predict optimal parameters using trained models"""
        if not self.trained:
            raise ValueError("Models not trained. Call train_models() first.")

        try:
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.input_features:
                value = features.get(feature_name, 0.0)
                feature_vector.append(value)

            X = np.array([feature_vector])

            # Predict parameters
            predictions = {}
            prediction_confidence = {}

            for param in self.target_parameters:
                if param in self.models and param in self.scalers:
                    # Scale features
                    X_scaled = self.scalers[param].transform(X)

                    # Predict
                    pred_value = self.models[param].predict(X_scaled)[0]

                    # Apply bounds based on parameter type
                    if param == 'corner_threshold':
                        pred_value = max(10, min(110, int(pred_value)))
                    elif param == 'color_precision':
                        pred_value = max(2, min(10, int(pred_value)))
                    elif param == 'path_precision':
                        pred_value = max(1, min(20, int(pred_value)))
                    elif param == 'length_threshold':
                        pred_value = max(1.0, min(20.0, float(pred_value)))
                    elif param == 'splice_threshold':
                        pred_value = max(10, min(100, int(pred_value)))
                    elif param == 'max_iterations':
                        pred_value = max(5, min(20, int(pred_value)))

                    predictions[param] = pred_value

                    # Calculate prediction confidence
                    if param in self.model_performance:
                        confidence = max(0.0, min(1.0, self.model_performance[param]['r2_score']))
                        prediction_confidence[param] = confidence

            # Add other parameters
            predictions['layer_difference'] = 10  # Default
            if features.get('complexity_score', 0) > 0.6 or logo_type in ['gradient', 'complex']:
                predictions['mode'] = 'spline'
            else:
                predictions['mode'] = 'polygon'

            # Calculate overall confidence
            param_confidences = list(prediction_confidence.values())
            overall_confidence = np.mean(param_confidences) if param_confidences else 0.5

            return {
                'parameters': predictions,
                'confidence': overall_confidence,
                'parameter_confidences': prediction_confidence,
                'logo_type': logo_type,
                'optimization_method': 'regression_based',
                'model_performance': self.model_performance
            }

        except Exception as e:
            logger.error(f"Error predicting parameters: {e}")
            raise

    def generate_improved_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Generate improved correlation coefficient matrix"""
        try:
            correlation_matrix = {}

            for param in self.target_parameters:
                if param in self.feature_importance:
                    correlation_matrix[param] = {}

                    # Feature importance as correlation strength
                    for feature, importance in self.feature_importance[param].items():
                        correlation_matrix[param][feature] = importance

                    # Add model performance as overall correlation quality
                    if param in self.model_performance:
                        correlation_matrix[param]['model_quality'] = self.model_performance[param]['r2_score']

            logger.info("Generated improved correlation coefficient matrix")
            return correlation_matrix

        except Exception as e:
            logger.error(f"Error generating correlation matrix: {e}")
            return {}

    def save_models(self, model_dir: str = "models/regression_optimizer") -> str:
        """Save trained models to disk"""
        try:
            model_dir = Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save models
            for param, model in self.models.items():
                model_file = model_dir / f"{param}_model.joblib"
                joblib.dump(model, model_file)

            # Save scalers
            for param, scaler in self.scalers.items():
                scaler_file = model_dir / f"{param}_scaler.joblib"
                joblib.dump(scaler, scaler_file)

            # Save metadata
            metadata = {
                'trained': self.trained,
                'target_parameters': self.target_parameters,
                'input_features': self.input_features,
                'model_performance': self.model_performance,
                'feature_importance': self.feature_importance
            }

            metadata_file = model_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Saved regression models to {model_dir}")
            return str(model_dir)

        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise

    def load_models(self, model_dir: str = "models/regression_optimizer") -> bool:
        """Load trained models from disk"""
        try:
            model_dir = Path(model_dir)

            if not model_dir.exists():
                logger.warning(f"Model directory not found: {model_dir}")
                return False

            # Load metadata
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                self.trained = metadata.get('trained', False)
                self.target_parameters = metadata.get('target_parameters', self.target_parameters)
                self.input_features = metadata.get('input_features', self.input_features)
                self.model_performance = metadata.get('model_performance', {})
                self.feature_importance = metadata.get('feature_importance', {})

            # Load models
            for param in self.target_parameters:
                model_file = model_dir / f"{param}_model.joblib"
                if model_file.exists():
                    self.models[param] = joblib.load(model_file)

                scaler_file = model_dir / f"{param}_scaler.joblib"
                if scaler_file.exists():
                    self.scalers[param] = joblib.load(scaler_file)

            logger.info(f"Loaded {len(self.models)} regression models from {model_dir}")
            return len(self.models) > 0

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of trained models"""
        if not self.trained:
            return {"status": "Not trained"}

        return {
            "status": "Trained",
            "models_count": len(self.models),
            "parameters": list(self.models.keys()),
            "performance": self.model_performance,
            "feature_importance": self.feature_importance,
            "input_features": self.input_features
        }