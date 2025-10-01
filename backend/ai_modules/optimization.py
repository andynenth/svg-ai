"""
Unified Optimization Module
Parameter optimization, tuning, and continuous learning
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cachetools
import hashlib
import json
import numpy as np
import pickle
import xgboost as xgb

class OptimizationEngine:
    """Complete optimization system for VTracer parameters"""

    def __init__(self) -> None:
        self.xgb_model = None
        self.parameter_history = []
        self.online_learning_enabled = False
        self.correlation_cache = cachetools.LRUCache(maxsize=1000)

    @staticmethod
    def calculate_base_parameters(features: Dict) -> Dict:
        """Calculate base parameters using formulas"""
        params = {'color_precision': 6, 'layer_difference': 16, 'max_iterations': 10, 'min_area': 10, 'path_precision': 8, 'corner_threshold': 60, 'length_threshold': 4.0, 'splice_threshold': 45}
        if features.get('unique_colors', 0) < 10:
            params['color_precision'] = 2
        elif features.get('unique_colors', 0) > 100:
            params['color_precision'] = 8
        if features.get('has_gradients', False):
            params['layer_difference'] = 8
            params['color_precision'] = max(params['color_precision'], 8)
        if features.get('complexity', 0.5) > 0.7:
            params['max_iterations'] = 20
            params['corner_threshold'] = 30
        return params

    def calculate_color_precision(self, features: Dict) -> int:
        """Calculate optimal color precision"""
        unique_colors = features.get('unique_colors', 10)
        has_gradients = features.get('has_gradients', False)
        if unique_colors < 10:
            return 2
        elif unique_colors < 50:
            return 4
        elif has_gradients:
            return 8
        else:
            return 6

    def calculate_corner_threshold(self, features: Dict) -> float:
        """Calculate optimal corner threshold"""
        edge_density = features.get('edge_density', 0.5)
        complexity = features.get('complexity', 0.5)
        base_threshold = 30.0
        adjustment = (edge_density - 0.5) * 20
        return base_threshold + adjustment

    def load_model(self, model_path: str):
        """Load pre-trained XGBoost model"""
        if Path(model_path).exists():
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(model_path)

    def _prepare_features(self, features: Dict) -> np.ndarray:
        """Prepare feature vector for XGBoost"""
        feature_list = [features.get('unique_colors', 0), features.get('complexity', 0.5), features.get('edge_density', 0.5), features.get('aspect_ratio', 1.0), int(features.get('has_text', False)), int(features.get('has_gradients', False)), features.get('size', [100, 100])[0] if isinstance(features.get('size'), list) else 100, features.get('size', [100, 100])[1] if isinstance(features.get('size'), list) else 100]
        return np.array(feature_list, dtype=np.float32)

    def _params_to_vector(self, params: Dict) -> np.ndarray:
        """Convert parameters to vector for training"""
        return np.array([params.get('color_precision', 6), params.get('layer_difference', 16), params.get('max_iterations', 10), params.get('min_area', 10), params.get('path_precision', 8), params.get('corner_threshold', 60), params.get('length_threshold', 4.0), params.get('splice_threshold', 45)], dtype=np.float32)

    def predict_parameters(self, features: Dict) -> Dict:
        """Predict optimal parameters using ML model"""
        if self.xgb_model is None:
            return self.calculate_base_parameters(features)
        feature_vector = self._prepare_features(features)
        dmatrix = xgb.DMatrix(feature_vector.reshape(1, -1))
        predictions = self.xgb_model.predict(dmatrix)[0]
        params = {'color_precision': int(np.clip(predictions[0], 1, 10)), 'layer_difference': int(np.clip(predictions[1], 1, 32)), 'max_iterations': int(np.clip(predictions[2], 1, 30)), 'min_area': int(np.clip(predictions[3], 1, 100)), 'path_precision': int(np.clip(predictions[4], 1, 15)), 'corner_threshold': int(np.clip(predictions[5], 10, 90)), 'length_threshold': float(np.clip(predictions[6], 1.0, 10.0)), 'splice_threshold': int(np.clip(predictions[7], 10, 90))}
        return params

    def _test_parameters(self, image_path: str, params: Dict) -> float:
        """Test parameters and return quality score"""
        score = 0.8
        if 2 <= params.get('color_precision', 0) <= 8:
            score += 0.1
        if 20 <= params.get('corner_threshold', 0) <= 80:
            score += 0.1
        return min(score, 1.0)

    def fine_tune_parameters(self, image_path: str, base_params: Dict, target_quality: float=0.9) -> Dict:
        """Fine-tune parameters for specific image"""
        best_params = base_params.copy()
        best_quality = 0
        variations = [('color_precision', [-1, 0, 1]), ('corner_threshold', [-10, 0, 10]), ('path_precision', [-2, 0, 2])]
        for (param, deltas) in variations:
            for delta in deltas:
                test_params = best_params.copy()
                test_params[param] = test_params[param] + delta
                if param == 'color_precision':
                    test_params[param] = max(1, min(10, test_params[param]))
                elif param == 'corner_threshold':
                    test_params[param] = max(10, min(90, test_params[param]))
                quality = self._test_parameters(image_path, test_params)
                if quality > best_quality:
                    best_quality = quality
                    best_params = test_params
                if best_quality >= target_quality:
                    break
        return best_params

    def enable_online_learning(self):
        """Enable continuous learning from results"""
        self.online_learning_enabled = True
        self.parameter_history = []

    def record_result(self, features: Dict, params: Dict, quality: float):
        """Record conversion result for learning"""
        if self.online_learning_enabled:
            self.parameter_history.append({'features': features, 'params': params, 'quality': quality, 'timestamp': datetime.now().isoformat()})
            if len(self.parameter_history) >= 100:
                self._update_model()

    def _update_model(self):
        """Update model with recorded results"""
        if len(self.parameter_history) < 50:
            return
        X = []
        y = []
        for record in self.parameter_history[-1000:]:
            feature_vec = self._prepare_features(record['features'])
            param_vec = self._params_to_vector(record['params'])
            X.append(feature_vec)
            y.append(param_vec)
        dtrain = xgb.DMatrix(np.array(X), label=np.array(y))
        params = {'max_depth': 6, 'eta': 0.1, 'objective': 'reg:squarederror'}
        self.xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    def analyze_correlations(self, data: List[Dict]) -> Dict:
        """Analyze parameter-quality correlations"""
        correlations = {}
        if len(data) < 10:
            return correlations
        param_names = ['color_precision', 'corner_threshold', 'path_precision']
        for param in param_names:
            param_values = []
            qualities = []
            for record in data:
                if param in record.get('params', {}):
                    param_values.append(record['params'][param])
                    qualities.append(record.get('quality', 0))
            if len(param_values) > 5:
                correlation = np.corrcoef(param_values, qualities)[0, 1]
                correlations[param] = correlation if not np.isnan(correlation) else 0
        return correlations

    def get_learned_insights(self) -> Dict:
        """Get insights from learned correlations"""
        if len(self.parameter_history) < 20:
            return {'message': 'Not enough data for insights'}
        correlations = self.analyze_correlations(self.parameter_history)
        insights = {'total_conversions': len(self.parameter_history), 'average_quality': np.mean([r.get('quality', 0) for r in self.parameter_history]), 'parameter_correlations': correlations, 'recommendations': []}
        for (param, corr) in correlations.items():
            if abs(corr) > 0.3:
                direction = 'increase' if corr > 0 else 'decrease'
                insights['recommendations'].append(f'Consider {direction} {param} for better quality (correlation: {corr:.3f})')
        return insights

    def optimize(self, image_path: str, features: Dict, use_ml: bool=True, fine_tune: bool=False) -> Dict:
        """Main optimization interface"""
        if use_ml and self.xgb_model is not None:
            params = self.predict_parameters(features)
        else:
            params = self.calculate_base_parameters(features)
        if fine_tune:
            params = self.fine_tune_parameters(image_path, params)
        return params

class LearnedCorrelationsManager:
    """Manages learned parameter correlations"""

    def __init__(self) -> None:
        self.correlations = {}
        self.confidence_scores = {}

    def update_correlation(self, param_name: str, correlation: float, confidence: float):
        """Update a parameter correlation"""
        self.correlations[param_name] = correlation
        self.confidence_scores[param_name] = confidence

    def get_correlation(self, param_name: str) -> Tuple[float, float]:
        """Get correlation and confidence for a parameter"""
        correlation = self.correlations.get(param_name, 0.0)
        confidence = self.confidence_scores.get(param_name, 0.0)
        return (correlation, confidence)

    def get_all_correlations(self) -> Dict:
        """Get all learned correlations"""
        return {'correlations': self.correlations.copy(), 'confidence_scores': self.confidence_scores.copy()}


# Legacy compatibility
PARAMETERFORMULAS = OptimizationEngine
LEARNEDOPTIMIZER = OptimizationEngine
PARAMETERTUNER = OptimizationEngine
ONLINELEARNER = OptimizationEngine
UNIFIEDPARAMETERFORMULAS = OptimizationEngine