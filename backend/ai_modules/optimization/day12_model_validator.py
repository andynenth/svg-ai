"""
Day 12: Comprehensive Model Validation & Quality Assessment
Advanced model validation with statistical analysis, cross-validation, and export readiness assessment
Part of Task 12.2.2: Comprehensive Model Validation and Quality Assessment
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import cosine
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import Day 12 components
from .gpu_model_architecture import QualityPredictorGPU, ColabTrainingConfig
from .gpu_training_pipeline import ColabTrainingExample, QualityDataset

warnings.filterwarnings('ignore')


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics"""
    # Correlation metrics
    pearson_correlation: float
    pearson_p_value: float
    spearman_correlation: float
    kendall_tau: float

    # Error metrics
    mse: float
    rmse: float
    mae: float
    r2_score: float

    # Accuracy metrics
    accuracy_within_0_05: float  # Predictions within 0.05 SSIM
    accuracy_within_0_10: float  # Predictions within 0.10 SSIM
    accuracy_within_0_15: float  # Predictions within 0.15 SSIM

    # Distribution metrics
    mean_prediction: float
    std_prediction: float
    mean_actual: float
    std_actual: float

    # Confidence metrics
    prediction_confidence: float
    calibration_error: float

    # Sample information
    sample_count: int
    prediction_range: Tuple[float, float]
    actual_range: Tuple[float, float]


@dataclass
class CrossValidationResults:
    """Cross-validation results and analysis"""
    cv_method: str
    n_folds: int
    fold_scores: List[float]
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    fold_details: List[Dict[str, Any]]
    stability_score: float
    statistical_significance: float


@dataclass
class LogoTypeValidation:
    """Logo type specific validation results"""
    logo_type: str
    sample_count: int
    metrics: ValidationMetrics
    performance_rank: int
    difficulty_score: float
    recommendations: List[str]


@dataclass
class ExportReadinessAssessment:
    """Model export readiness assessment"""
    export_ready: bool
    readiness_score: float
    requirements_met: List[str]
    requirements_failed: List[str]
    performance_targets: Dict[str, bool]
    recommendations: List[str]
    export_confidence: float


class ModelValidationSuite:
    """Comprehensive model validation suite"""

    def __init__(self, device: str = 'cuda', save_dir: str = "/tmp/claude/model_validation"):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Validation targets
        self.performance_targets = {
            'correlation_target': 0.90,
            'rmse_target': 0.05,
            'accuracy_target': 0.85,
            'r2_target': 0.80,
            'type_consistency_target': 0.85
        }

    def validate_model_comprehensive(
        self,
        model: QualityPredictorGPU,
        test_examples: List[ColabTrainingExample],
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute comprehensive model validation"""

        print("🔍 Comprehensive Model Validation Suite")
        print("=" * 60)

        validation_results = {
            'validation_timestamp': time.time(),
            'model_info': self._get_model_info(model),
            'test_data_info': self._analyze_test_data(test_examples)
        }

        # 1. Overall Performance Validation
        print("\n📊 Overall Performance Validation")
        overall_metrics = self._validate_overall_performance(model, test_examples)
        validation_results['overall_performance'] = overall_metrics

        # 2. Logo Type Specific Validation
        print("\n🎨 Logo Type Specific Validation")
        type_validation = self._validate_by_logo_type(model, test_examples)
        validation_results['logo_type_validation'] = type_validation

        # 3. Cross-Validation Analysis
        print("\n🔄 Cross-Validation Analysis")
        cv_results = self._cross_validate_model(model, test_examples)
        validation_results['cross_validation'] = cv_results

        # 4. Robustness Testing
        print("\n🛡️ Model Robustness Testing")
        robustness_results = self._test_model_robustness(model, test_examples)
        validation_results['robustness'] = robustness_results

        # 5. Prediction Quality Analysis
        print("\n📈 Prediction Quality Analysis")
        quality_analysis = self._analyze_prediction_quality(model, test_examples)
        validation_results['prediction_quality'] = quality_analysis

        # 6. Statistical Significance Testing
        print("\n📊 Statistical Significance Testing")
        significance_results = self._test_statistical_significance(model, test_examples)
        validation_results['statistical_tests'] = significance_results

        # 7. Export Readiness Assessment
        print("\n🎯 Export Readiness Assessment")
        export_assessment = self._assess_export_readiness(validation_results)
        validation_results['export_readiness'] = export_assessment

        # Generate comprehensive report
        self._generate_validation_report(validation_results)

        # Create visualizations
        self._create_validation_visualizations(model, test_examples, validation_results)

        return validation_results

    def _get_model_info(self, model: QualityPredictorGPU) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'model_type': type(model).__name__,
            'total_parameters': model.count_parameters(),
            'architecture': str(model),
            'device': str(next(model.parameters()).device),
            'parameter_summary': self._analyze_model_parameters(model)
        }

    def _analyze_model_parameters(self, model: QualityPredictorGPU) -> Dict[str, Any]:
        """Analyze model parameters"""
        param_info = {}
        total_params = 0

        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count

            param_info[name] = {
                'shape': list(param.shape),
                'parameters': param_count,
                'requires_grad': param.requires_grad,
                'mean': float(param.data.mean()),
                'std': float(param.data.std())
            }

        return {
            'total_parameters': total_params,
            'parameter_details': param_info,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }

    def _analyze_test_data(self, test_examples: List[ColabTrainingExample]) -> Dict[str, Any]:
        """Analyze test data characteristics"""
        if not test_examples:
            return {}

        ssim_values = [ex.actual_ssim for ex in test_examples]
        logo_types = [ex.logo_type for ex in test_examples]
        methods = [ex.optimization_method for ex in test_examples]

        # Count by logo type
        type_counts = {}
        for logo_type in logo_types:
            type_counts[logo_type] = type_counts.get(logo_type, 0) + 1

        # Count by method
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1

        return {
            'total_samples': len(test_examples),
            'ssim_statistics': {
                'mean': np.mean(ssim_values),
                'std': np.std(ssim_values),
                'min': np.min(ssim_values),
                'max': np.max(ssim_values),
                'median': np.median(ssim_values)
            },
            'logo_type_distribution': type_counts,
            'method_distribution': method_counts,
            'data_quality_score': self._calculate_data_quality_score(test_examples)
        }

    def _calculate_data_quality_score(self, test_examples: List[ColabTrainingExample]) -> float:
        """Calculate overall data quality score"""
        if not test_examples:
            return 0.0

        scores = []

        # Diversity score (logo types)
        logo_types = set(ex.logo_type for ex in test_examples)
        diversity_score = min(1.0, len(logo_types) / 4.0)  # Expect 4 types
        scores.append(diversity_score)

        # Range score (SSIM coverage)
        ssim_values = [ex.actual_ssim for ex in test_examples]
        ssim_range = np.max(ssim_values) - np.min(ssim_values)
        range_score = min(1.0, ssim_range / 0.4)  # Expect 0.4 range
        scores.append(range_score)

        # Sample size score
        size_score = min(1.0, len(test_examples) / 100.0)  # Expect 100+ samples
        scores.append(size_score)

        # Balance score
        type_counts = [len([ex for ex in test_examples if ex.logo_type == t]) for t in logo_types]
        if type_counts:
            balance_score = 1.0 - (np.std(type_counts) / np.mean(type_counts))
            balance_score = max(0.0, balance_score)
        else:
            balance_score = 0.0
        scores.append(balance_score)

        return np.mean(scores)

    def _validate_overall_performance(
        self,
        model: QualityPredictorGPU,
        test_examples: List[ColabTrainingExample]
    ) -> ValidationMetrics:
        """Validate overall model performance with comprehensive metrics"""

        model.eval()
        predictions = []
        actuals = []

        print(f"   Evaluating {len(test_examples)} test examples...")

        with torch.no_grad():
            for example in test_examples:
                # Prepare input
                combined_features = np.concatenate([
                    example.image_features,
                    self._normalize_vtracer_params(example.vtracer_params)
                ])

                input_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
                pred = model(input_tensor).cpu().item()

                predictions.append(pred)
                actuals.append(example.actual_ssim)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate comprehensive metrics
        pearson_r, pearson_p = pearsonr(predictions, actuals)
        spearman_r, _ = spearmanr(predictions, actuals)
        kendall_tau, _ = kendalltau(predictions, actuals)

        # Error metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        # Accuracy metrics
        diff = np.abs(predictions - actuals)
        acc_05 = np.mean(diff <= 0.05)
        acc_10 = np.mean(diff <= 0.10)
        acc_15 = np.mean(diff <= 0.15)

        # Distribution metrics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        mean_actual = np.mean(actuals)
        std_actual = np.std(actuals)

        # Confidence and calibration
        prediction_confidence = self._calculate_prediction_confidence(predictions, actuals)
        calibration_error = self._calculate_calibration_error(predictions, actuals)

        metrics = ValidationMetrics(
            pearson_correlation=pearson_r,
            pearson_p_value=pearson_p,
            spearman_correlation=spearman_r,
            kendall_tau=kendall_tau,
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            accuracy_within_0_05=acc_05,
            accuracy_within_0_10=acc_10,
            accuracy_within_0_15=acc_15,
            mean_prediction=mean_pred,
            std_prediction=std_pred,
            mean_actual=mean_actual,
            std_actual=std_actual,
            prediction_confidence=prediction_confidence,
            calibration_error=calibration_error,
            sample_count=len(test_examples),
            prediction_range=(np.min(predictions), np.max(predictions)),
            actual_range=(np.min(actuals), np.max(actuals))
        )

        print(f"   Pearson Correlation: {pearson_r:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   Accuracy (0.1): {acc_10:.1%}")

        return metrics

    def _calculate_prediction_confidence(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate prediction confidence score"""
        # Based on consistency of predictions vs actuals
        residuals = np.abs(predictions - actuals)
        confidence = 1.0 - np.mean(residuals)
        return max(0.0, confidence)

    def _calculate_calibration_error(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate calibration error"""
        # Measure how well prediction confidence matches actual accuracy
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        calibration_errors = []

        for i in range(n_bins):
            bin_mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(np.abs(predictions[bin_mask] - actuals[bin_mask]) <= 0.1)
                bin_confidence = np.mean(predictions[bin_mask])
                calibration_errors.append(abs(bin_accuracy - bin_confidence))

        return np.mean(calibration_errors) if calibration_errors else 0.0

    def _validate_by_logo_type(
        self,
        model: QualityPredictorGPU,
        test_examples: List[ColabTrainingExample]
    ) -> Dict[str, LogoTypeValidation]:
        """Validate model performance by logo type"""

        # Group examples by logo type
        type_groups = defaultdict(list)
        for example in test_examples:
            type_groups[example.logo_type].append(example)

        type_validations = {}
        type_performances = []

        for logo_type, examples in type_groups.items():
            if len(examples) < 5:  # Skip types with too few examples
                continue

            print(f"   Validating {logo_type}: {len(examples)} examples")

            # Calculate metrics for this type
            type_metrics = self._validate_overall_performance(model, examples)

            # Calculate difficulty score
            difficulty_score = self._calculate_type_difficulty(examples)

            # Generate recommendations
            recommendations = self._generate_type_recommendations(type_metrics, difficulty_score)

            type_validation = LogoTypeValidation(
                logo_type=logo_type,
                sample_count=len(examples),
                metrics=type_metrics,
                performance_rank=0,  # Will be set after ranking
                difficulty_score=difficulty_score,
                recommendations=recommendations
            )

            type_validations[logo_type] = type_validation
            type_performances.append((logo_type, type_metrics.pearson_correlation))

        # Rank performance
        type_performances.sort(key=lambda x: x[1], reverse=True)
        for rank, (logo_type, _) in enumerate(type_performances):
            type_validations[logo_type].performance_rank = rank + 1

        return type_validations

    def _calculate_type_difficulty(self, examples: List[ColabTrainingExample]) -> float:
        """Calculate difficulty score for a logo type"""
        ssim_values = [ex.actual_ssim for ex in examples]

        # Difficulty factors
        ssim_variance = np.var(ssim_values)  # Higher variance = harder
        mean_ssim = np.mean(ssim_values)  # Lower mean = harder

        # Normalize and combine
        variance_score = min(1.0, ssim_variance * 20)  # Scale variance
        mean_score = 1.0 - mean_ssim  # Invert mean (lower is harder)

        difficulty = (variance_score + mean_score) / 2.0
        return np.clip(difficulty, 0.0, 1.0)

    def _generate_type_recommendations(self, metrics: ValidationMetrics, difficulty: float) -> List[str]:
        """Generate recommendations for logo type performance"""
        recommendations = []

        if metrics.pearson_correlation < 0.8:
            recommendations.append("Consider type-specific model fine-tuning")

        if metrics.rmse > 0.1:
            recommendations.append("Improve parameter optimization for this type")

        if difficulty > 0.7:
            recommendations.append("Collect more training data for this challenging type")

        if metrics.accuracy_within_0_10 < 0.8:
            recommendations.append("Review feature extraction for this type")

        return recommendations

    def _cross_validate_model(
        self,
        model: QualityPredictorGPU,
        test_examples: List[ColabTrainingExample],
        cv_method: str = 'kfold',
        n_folds: int = 5
    ) -> CrossValidationResults:
        """Perform cross-validation analysis"""

        print(f"   Performing {n_folds}-fold cross-validation...")

        if len(test_examples) < n_folds:
            print(f"   Warning: Insufficient data for {n_folds}-fold CV")
            n_folds = max(2, len(test_examples) // 2)

        # Prepare data for CV
        fold_scores = []
        fold_details = []

        # Simple K-fold without retraining (model already trained)
        fold_size = len(test_examples) // n_folds

        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(test_examples)

            fold_examples = test_examples[start_idx:end_idx]
            fold_metrics = self._validate_overall_performance(model, fold_examples)

            fold_score = fold_metrics.pearson_correlation
            fold_scores.append(fold_score)

            fold_details.append({
                'fold': fold + 1,
                'sample_count': len(fold_examples),
                'correlation': fold_score,
                'rmse': fold_metrics.rmse,
                'accuracy_10': fold_metrics.accuracy_within_0_10
            })

        # Calculate statistics
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        # Confidence interval (assuming normal distribution)
        confidence_interval = (
            mean_score - 1.96 * std_score / np.sqrt(n_folds),
            mean_score + 1.96 * std_score / np.sqrt(n_folds)
        )

        # Stability score (1 - coefficient of variation)
        stability_score = 1.0 - (std_score / mean_score) if mean_score > 0 else 0.0

        # Statistical significance (t-test against baseline)
        baseline_score = 0.5  # Random correlation baseline
        t_stat = (mean_score - baseline_score) / (std_score / np.sqrt(n_folds))
        significance = abs(t_stat)

        cv_results = CrossValidationResults(
            cv_method=cv_method,
            n_folds=n_folds,
            fold_scores=fold_scores,
            mean_score=mean_score,
            std_score=std_score,
            confidence_interval=confidence_interval,
            fold_details=fold_details,
            stability_score=stability_score,
            statistical_significance=significance
        )

        print(f"   CV Mean: {mean_score:.4f} ± {std_score:.4f}")
        print(f"   Stability: {stability_score:.3f}")

        return cv_results

    def _test_model_robustness(
        self,
        model: QualityPredictorGPU,
        test_examples: List[ColabTrainingExample]
    ) -> Dict[str, Any]:
        """Test model robustness with various perturbations"""

        print("   Testing model robustness...")

        robustness_results = {}

        # 1. Input noise robustness
        noise_robustness = self._test_noise_robustness(model, test_examples[:50])  # Use subset for speed
        robustness_results['noise_robustness'] = noise_robustness

        # 2. Parameter perturbation robustness
        param_robustness = self._test_parameter_perturbation_robustness(model, test_examples[:50])
        robustness_results['parameter_robustness'] = param_robustness

        # 3. Feature masking robustness
        masking_robustness = self._test_feature_masking_robustness(model, test_examples[:50])
        robustness_results['feature_masking'] = masking_robustness

        # 4. Adversarial robustness (simple)
        adversarial_robustness = self._test_simple_adversarial_robustness(model, test_examples[:30])
        robustness_results['adversarial_robustness'] = adversarial_robustness

        return robustness_results

    def _test_noise_robustness(self, model: QualityPredictorGPU, examples: List[ColabTrainingExample]) -> Dict[str, float]:
        """Test robustness to input noise"""
        model.eval()

        noise_levels = [0.01, 0.05, 0.1, 0.2]
        robustness_scores = {}

        with torch.no_grad():
            for noise_level in noise_levels:
                correlations = []

                for example in examples:
                    # Original prediction
                    original_input = np.concatenate([
                        example.image_features,
                        self._normalize_vtracer_params(example.vtracer_params)
                    ])
                    original_tensor = torch.FloatTensor(original_input).unsqueeze(0).to(self.device)
                    original_pred = model(original_tensor).cpu().item()

                    # Noisy prediction
                    noise = np.random.normal(0, noise_level, original_input.shape)
                    noisy_input = original_input + noise
                    noisy_tensor = torch.FloatTensor(noisy_input).unsqueeze(0).to(self.device)
                    noisy_pred = model(noisy_tensor).cpu().item()

                    # Correlation between original and noisy predictions
                    correlations.append(abs(original_pred - noisy_pred))

                # Robustness score (1 - average difference)
                avg_difference = np.mean(correlations)
                robustness_score = max(0.0, 1.0 - avg_difference)
                robustness_scores[f'noise_{noise_level}'] = robustness_score

        return robustness_scores

    def _test_parameter_perturbation_robustness(self, model: QualityPredictorGPU, examples: List[ColabTrainingExample]) -> Dict[str, float]:
        """Test robustness to VTracer parameter perturbations"""
        model.eval()

        perturbation_levels = [0.05, 0.1, 0.2]
        robustness_scores = {}

        with torch.no_grad():
            for perturbation in perturbation_levels:
                correlations = []

                for example in examples:
                    # Original prediction
                    original_params = self._normalize_vtracer_params(example.vtracer_params)
                    original_input = np.concatenate([example.image_features, original_params])
                    original_tensor = torch.FloatTensor(original_input).unsqueeze(0).to(self.device)
                    original_pred = model(original_tensor).cpu().item()

                    # Perturbed parameters
                    param_noise = np.random.uniform(-perturbation, perturbation, len(original_params))
                    perturbed_params = np.clip(original_params + param_noise, 0, 1)
                    perturbed_input = np.concatenate([example.image_features, perturbed_params])
                    perturbed_tensor = torch.FloatTensor(perturbed_input).unsqueeze(0).to(self.device)
                    perturbed_pred = model(perturbed_tensor).cpu().item()

                    correlations.append(abs(original_pred - perturbed_pred))

                avg_difference = np.mean(correlations)
                robustness_score = max(0.0, 1.0 - avg_difference)
                robustness_scores[f'param_perturbation_{perturbation}'] = robustness_score

        return robustness_scores

    def _test_feature_masking_robustness(self, model: QualityPredictorGPU, examples: List[ColabTrainingExample]) -> Dict[str, float]:
        """Test robustness to feature masking"""
        model.eval()

        mask_ratios = [0.1, 0.2, 0.3]
        robustness_scores = {}

        with torch.no_grad():
            for mask_ratio in mask_ratios:
                correlations = []

                for example in examples:
                    # Original prediction
                    original_input = np.concatenate([
                        example.image_features,
                        self._normalize_vtracer_params(example.vtracer_params)
                    ])
                    original_tensor = torch.FloatTensor(original_input).unsqueeze(0).to(self.device)
                    original_pred = model(original_tensor).cpu().item()

                    # Masked input (only mask image features, not parameters)
                    masked_features = example.image_features.copy()
                    n_mask = int(len(masked_features) * mask_ratio)
                    mask_indices = np.random.choice(len(masked_features), n_mask, replace=False)
                    masked_features[mask_indices] = 0

                    masked_input = np.concatenate([
                        masked_features,
                        self._normalize_vtracer_params(example.vtracer_params)
                    ])
                    masked_tensor = torch.FloatTensor(masked_input).unsqueeze(0).to(self.device)
                    masked_pred = model(masked_tensor).cpu().item()

                    correlations.append(abs(original_pred - masked_pred))

                avg_difference = np.mean(correlations)
                robustness_score = max(0.0, 1.0 - avg_difference)
                robustness_scores[f'feature_masking_{mask_ratio}'] = robustness_score

        return robustness_scores

    def _test_simple_adversarial_robustness(self, model: QualityPredictorGPU, examples: List[ColabTrainingExample]) -> Dict[str, float]:
        """Test simple adversarial robustness"""
        model.eval()

        epsilon_values = [0.01, 0.05, 0.1]
        robustness_scores = {}

        for epsilon in epsilon_values:
            stable_predictions = 0
            total_predictions = 0

            for example in examples:
                try:
                    # Original input
                    original_input = np.concatenate([
                        example.image_features,
                        self._normalize_vtracer_params(example.vtracer_params)
                    ])

                    # Create adversarial perturbation (simple gradient-free)
                    perturbation = np.random.uniform(-epsilon, epsilon, original_input.shape)
                    adversarial_input = original_input + perturbation

                    # Compare predictions
                    with torch.no_grad():
                        original_tensor = torch.FloatTensor(original_input).unsqueeze(0).to(self.device)
                        adversarial_tensor = torch.FloatTensor(adversarial_input).unsqueeze(0).to(self.device)

                        original_pred = model(original_tensor).cpu().item()
                        adversarial_pred = model(adversarial_tensor).cpu().item()

                        # Consider stable if difference < 0.1
                        if abs(original_pred - adversarial_pred) < 0.1:
                            stable_predictions += 1

                        total_predictions += 1

                except:
                    continue

            robustness_score = stable_predictions / total_predictions if total_predictions > 0 else 0.0
            robustness_scores[f'adversarial_{epsilon}'] = robustness_score

        return robustness_scores

    def _analyze_prediction_quality(self, model: QualityPredictorGPU, test_examples: List[ColabTrainingExample]) -> Dict[str, Any]:
        """Analyze prediction quality and patterns"""

        model.eval()
        predictions = []
        actuals = []
        prediction_details = []

        with torch.no_grad():
            for example in test_examples:
                combined_features = np.concatenate([
                    example.image_features,
                    self._normalize_vtracer_params(example.vtracer_params)
                ])

                input_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
                pred = model(input_tensor).cpu().item()

                predictions.append(pred)
                actuals.append(example.actual_ssim)

                prediction_details.append({
                    'prediction': pred,
                    'actual': example.actual_ssim,
                    'error': abs(pred - example.actual_ssim),
                    'logo_type': example.logo_type,
                    'method': example.optimization_method
                })

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Analyze prediction patterns
        quality_analysis = {
            'prediction_distribution': self._analyze_prediction_distribution(predictions, actuals),
            'error_analysis': self._analyze_prediction_errors(prediction_details),
            'bias_analysis': self._analyze_prediction_bias(predictions, actuals),
            'confidence_analysis': self._analyze_prediction_confidence(predictions, actuals),
            'outlier_analysis': self._analyze_prediction_outliers(prediction_details)
        }

        return quality_analysis

    def _analyze_prediction_distribution(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction distribution characteristics"""
        return {
            'prediction_mean': float(np.mean(predictions)),
            'prediction_std': float(np.std(predictions)),
            'actual_mean': float(np.mean(actuals)),
            'actual_std': float(np.std(actuals)),
            'distribution_similarity': float(1.0 - abs(np.std(predictions) - np.std(actuals))),
            'range_coverage': float(np.max(predictions) - np.min(predictions)) / (np.max(actuals) - np.min(actuals))
        }

    def _analyze_prediction_errors(self, prediction_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prediction error patterns"""
        errors = [detail['error'] for detail in prediction_details]

        # Error by logo type
        type_errors = defaultdict(list)
        for detail in prediction_details:
            type_errors[detail['logo_type']].append(detail['error'])

        type_error_summary = {}
        for logo_type, type_error_list in type_errors.items():
            type_error_summary[logo_type] = {
                'mean_error': np.mean(type_error_list),
                'std_error': np.std(type_error_list),
                'max_error': np.max(type_error_list)
            }

        return {
            'overall_mean_error': np.mean(errors),
            'overall_std_error': np.std(errors),
            'error_percentiles': {
                '50th': np.percentile(errors, 50),
                '75th': np.percentile(errors, 75),
                '90th': np.percentile(errors, 90),
                '95th': np.percentile(errors, 95)
            },
            'type_specific_errors': type_error_summary
        }

    def _analyze_prediction_bias(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction bias patterns"""
        residuals = predictions - actuals

        return {
            'mean_bias': float(np.mean(residuals)),
            'bias_direction': 'overestimate' if np.mean(residuals) > 0 else 'underestimate',
            'bias_magnitude': float(abs(np.mean(residuals))),
            'systematic_bias_score': float(abs(np.mean(residuals)) / np.std(residuals)) if np.std(residuals) > 0 else 0
        }

    def _analyze_prediction_confidence(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence characteristics"""
        # Confidence based on prediction certainty (distance from 0.5)
        confidence_scores = 2 * np.abs(predictions - 0.5)

        # Accuracy for different confidence levels
        confidence_accuracy = {}
        for threshold in [0.1, 0.3, 0.5, 0.7]:
            high_confidence_mask = confidence_scores >= threshold
            if np.sum(high_confidence_mask) > 0:
                high_conf_accuracy = np.mean(np.abs(predictions[high_confidence_mask] - actuals[high_confidence_mask]) <= 0.1)
                confidence_accuracy[f'threshold_{threshold}'] = high_conf_accuracy

        return {
            'mean_confidence': float(np.mean(confidence_scores)),
            'confidence_range': (float(np.min(confidence_scores)), float(np.max(confidence_scores))),
            'confidence_accuracy_relationship': confidence_accuracy
        }

    def _analyze_prediction_outliers(self, prediction_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prediction outliers"""
        errors = [detail['error'] for detail in prediction_details]
        error_threshold = np.percentile(errors, 95)  # Top 5% errors as outliers

        outliers = [detail for detail in prediction_details if detail['error'] > error_threshold]

        if not outliers:
            return {'outlier_count': 0}

        # Analyze outlier characteristics
        outlier_types = defaultdict(int)
        outlier_methods = defaultdict(int)

        for outlier in outliers:
            outlier_types[outlier['logo_type']] += 1
            outlier_methods[outlier['method']] += 1

        return {
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(prediction_details) * 100,
            'outlier_error_threshold': error_threshold,
            'outlier_by_type': dict(outlier_types),
            'outlier_by_method': dict(outlier_methods),
            'worst_outliers': sorted(outliers, key=lambda x: x['error'], reverse=True)[:5]
        }

    def _test_statistical_significance(self, model: QualityPredictorGPU, test_examples: List[ColabTrainingExample]) -> Dict[str, Any]:
        """Test statistical significance of model performance"""

        # Bootstrap confidence intervals
        bootstrap_correlations = []
        n_bootstrap = 100

        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(test_examples), len(test_examples), replace=True)
            bootstrap_examples = [test_examples[i] for i in bootstrap_indices]

            # Calculate correlation for bootstrap sample
            bootstrap_metrics = self._validate_overall_performance(model, bootstrap_examples)
            bootstrap_correlations.append(bootstrap_metrics.pearson_correlation)

        bootstrap_correlations = np.array(bootstrap_correlations)

        # Permutation test
        permutation_correlations = []
        n_permutations = 50  # Limited for speed

        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for example in test_examples:
                combined_features = np.concatenate([
                    example.image_features,
                    self._normalize_vtracer_params(example.vtracer_params)
                ])
                input_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
                pred = model(input_tensor).cpu().item()
                predictions.append(pred)
                actuals.append(example.actual_ssim)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        for _ in range(n_permutations):
            # Randomly permute actual values
            permuted_actuals = np.random.permutation(actuals)
            perm_correlation, _ = pearsonr(predictions, permuted_actuals)
            permutation_correlations.append(perm_correlation)

        permutation_correlations = np.array(permutation_correlations)

        # Calculate p-value
        observed_correlation = pearsonr(predictions, actuals)[0]
        p_value = np.mean(permutation_correlations >= observed_correlation)

        return {
            'bootstrap_analysis': {
                'mean_correlation': float(np.mean(bootstrap_correlations)),
                'std_correlation': float(np.std(bootstrap_correlations)),
                'confidence_interval_95': (
                    float(np.percentile(bootstrap_correlations, 2.5)),
                    float(np.percentile(bootstrap_correlations, 97.5))
                )
            },
            'permutation_test': {
                'observed_correlation': float(observed_correlation),
                'permutation_mean': float(np.mean(permutation_correlations)),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        }

    def _assess_export_readiness(self, validation_results: Dict[str, Any]) -> ExportReadinessAssessment:
        """Assess model readiness for export"""

        overall_metrics = validation_results.get('overall_performance')
        type_validations = validation_results.get('logo_type_validation', {})
        cv_results = validation_results.get('cross_validation')

        requirements_met = []
        requirements_failed = []
        performance_targets = {}

        # Check correlation target
        correlation = overall_metrics.pearson_correlation
        if correlation >= self.performance_targets['correlation_target']:
            requirements_met.append(f"✅ Correlation ≥{self.performance_targets['correlation_target']}: {correlation:.4f}")
            performance_targets['correlation'] = True
        else:
            requirements_failed.append(f"❌ Correlation <{self.performance_targets['correlation_target']}: {correlation:.4f}")
            performance_targets['correlation'] = False

        # Check RMSE target
        rmse = overall_metrics.rmse
        if rmse <= self.performance_targets['rmse_target']:
            requirements_met.append(f"✅ RMSE ≤{self.performance_targets['rmse_target']}: {rmse:.4f}")
            performance_targets['rmse'] = True
        else:
            requirements_failed.append(f"❌ RMSE >{self.performance_targets['rmse_target']}: {rmse:.4f}")
            performance_targets['rmse'] = False

        # Check accuracy target
        accuracy = overall_metrics.accuracy_within_0_10
        if accuracy >= self.performance_targets['accuracy_target']:
            requirements_met.append(f"✅ Accuracy ≥{self.performance_targets['accuracy_target']:.0%}: {accuracy:.1%}")
            performance_targets['accuracy'] = True
        else:
            requirements_failed.append(f"❌ Accuracy <{self.performance_targets['accuracy_target']:.0%}: {accuracy:.1%}")
            performance_targets['accuracy'] = False

        # Check R² target
        r2 = overall_metrics.r2_score
        if r2 >= self.performance_targets['r2_target']:
            requirements_met.append(f"✅ R² ≥{self.performance_targets['r2_target']}: {r2:.4f}")
            performance_targets['r2'] = True
        else:
            requirements_failed.append(f"❌ R² <{self.performance_targets['r2_target']}: {r2:.4f}")
            performance_targets['r2'] = False

        # Check logo type consistency
        if type_validations:
            type_correlations = [tv.metrics.pearson_correlation for tv in type_validations.values()]
            min_type_correlation = min(type_correlations)

            if min_type_correlation >= self.performance_targets['type_consistency_target']:
                requirements_met.append(f"✅ Type consistency ≥{self.performance_targets['type_consistency_target']}: {min_type_correlation:.4f}")
                performance_targets['type_consistency'] = True
            else:
                requirements_failed.append(f"❌ Type consistency <{self.performance_targets['type_consistency_target']}: {min_type_correlation:.4f}")
                performance_targets['type_consistency'] = False

        # Check cross-validation stability
        if cv_results and cv_results.stability_score >= 0.8:
            requirements_met.append(f"✅ CV stability ≥0.8: {cv_results.stability_score:.3f}")
            performance_targets['cv_stability'] = True
        else:
            cv_stability = cv_results.stability_score if cv_results else 0.0
            requirements_failed.append(f"❌ CV stability <0.8: {cv_stability:.3f}")
            performance_targets['cv_stability'] = False

        # Calculate overall readiness
        export_ready = len(requirements_failed) == 0
        readiness_score = len(requirements_met) / (len(requirements_met) + len(requirements_failed))

        # Generate recommendations
        recommendations = []
        if not export_ready:
            if not performance_targets.get('correlation', True):
                recommendations.append("Increase model capacity or extend training")
            if not performance_targets.get('rmse', True):
                recommendations.append("Improve loss function or learning rate tuning")
            if not performance_targets.get('accuracy', True):
                recommendations.append("Enhance feature engineering or data quality")
            if not performance_targets.get('type_consistency', True):
                recommendations.append("Add type-specific training or data balancing")
            if not performance_targets.get('cv_stability', True):
                recommendations.append("Increase regularization or training data size")
        else:
            recommendations.append("Model meets all export requirements")
            recommendations.append("Ready for production deployment")

        export_confidence = readiness_score * correlation  # Weighted by best performance metric

        return ExportReadinessAssessment(
            export_ready=export_ready,
            readiness_score=readiness_score,
            requirements_met=requirements_met,
            requirements_failed=requirements_failed,
            performance_targets=performance_targets,
            recommendations=recommendations,
            export_confidence=export_confidence
        )

    def _normalize_vtracer_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize VTracer parameters to [0,1] range"""
        normalized = [
            params.get('color_precision', 6.0) / 10.0,
            params.get('corner_threshold', 60.0) / 100.0,
            params.get('length_threshold', 4.0) / 10.0,
            params.get('max_iterations', 10) / 20.0,
            params.get('splice_threshold', 45.0) / 100.0,
            params.get('path_precision', 8) / 16.0,
            params.get('layer_difference', 16.0) / 32.0,
            params.get('mode', 0) / 1.0
        ]
        return np.array(normalized, dtype=np.float32)

    def _generate_validation_report(self, validation_results: Dict[str, Any]):
        """Generate comprehensive validation report"""

        report_path = self.save_dir / "comprehensive_validation_report.json"

        # Convert numpy types to JSON serializable
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj

        serializable_results = convert_numpy_types(validation_results)

        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"📋 Comprehensive validation report saved: {report_path}")

    def _create_validation_visualizations(self, model: QualityPredictorGPU, test_examples: List[ColabTrainingExample], validation_results: Dict[str, Any]):
        """Create comprehensive validation visualizations"""

        # Get predictions for visualization
        model.eval()
        predictions = []
        actuals = []
        types = []

        with torch.no_grad():
            for example in test_examples:
                combined_features = np.concatenate([
                    example.image_features,
                    self._normalize_vtracer_params(example.vtracer_params)
                ])
                input_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
                pred = model(input_tensor).cpu().item()

                predictions.append(pred)
                actuals.append(example.actual_ssim)
                types.append(example.logo_type)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. Predictions vs Actuals scatter plot
        ax1 = fig.add_subplot(gs[0, :2])
        scatter = ax1.scatter(actuals, predictions, alpha=0.6, c=predictions, cmap='viridis')
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
        ax1.set_xlabel('Actual SSIM')
        ax1.set_ylabel('Predicted SSIM')
        ax1.set_title('Predictions vs Actuals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Predicted SSIM')

        # 2. Residuals plot
        ax2 = fig.add_subplot(gs[0, 2:])
        residuals = predictions - actuals
        ax2.scatter(actuals, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Actual SSIM')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Analysis')
        ax2.grid(True, alpha=0.3)

        # 3. Error distribution
        ax3 = fig.add_subplot(gs[1, 0])
        errors = np.abs(predictions - actuals)
        ax3.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Absolute Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
        ax3.legend()

        # 4. Performance by logo type
        ax4 = fig.add_subplot(gs[1, 1])
        unique_types = list(set(types))
        type_correlations = []
        type_names = []

        for logo_type in unique_types:
            type_mask = np.array(types) == logo_type
            if np.sum(type_mask) > 1:
                type_pred = predictions[type_mask]
                type_actual = actuals[type_mask]
                corr, _ = pearsonr(type_pred, type_actual)
                type_correlations.append(corr)
                type_names.append(logo_type)

        bars = ax4.bar(type_names, type_correlations)
        ax4.set_ylabel('Correlation')
        ax4.set_title('Performance by Logo Type')
        ax4.set_ylim(0, 1)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        # Add correlation values on bars
        for bar, corr in zip(bars, type_correlations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')

        # 5. Cross-validation results
        ax5 = fig.add_subplot(gs[1, 2])
        cv_results = validation_results.get('cross_validation')
        if cv_results and cv_results.fold_scores:
            fold_numbers = list(range(1, len(cv_results.fold_scores) + 1))
            ax5.plot(fold_numbers, cv_results.fold_scores, 'bo-', linewidth=2, markersize=8)
            ax5.axhline(y=cv_results.mean_score, color='r', linestyle='--',
                       label=f'Mean: {cv_results.mean_score:.3f}')
            ax5.fill_between(fold_numbers,
                            cv_results.mean_score - cv_results.std_score,
                            cv_results.mean_score + cv_results.std_score,
                            alpha=0.3, color='red')
            ax5.set_xlabel('Fold')
            ax5.set_ylabel('Correlation')
            ax5.set_title('Cross-Validation Results')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. Export readiness assessment
        ax6 = fig.add_subplot(gs[1, 3])
        export_assessment = validation_results.get('export_readiness')
        if export_assessment:
            targets = list(export_assessment.performance_targets.keys())
            target_met = [export_assessment.performance_targets[t] for t in targets]

            colors = ['green' if met else 'red' for met in target_met]
            bars = ax6.bar(targets, [1] * len(targets), color=colors, alpha=0.7)
            ax6.set_ylabel('Target Met')
            ax6.set_title('Export Readiness')
            ax6.set_ylim(0, 1)
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

        # 7. Model robustness summary
        ax7 = fig.add_subplot(gs[2, :2])
        robustness_results = validation_results.get('robustness', {})
        if robustness_results:
            all_robustness_scores = []
            robustness_labels = []

            for test_type, scores in robustness_results.items():
                if isinstance(scores, dict):
                    for score_name, score_value in scores.items():
                        all_robustness_scores.append(score_value)
                        robustness_labels.append(f"{test_type}_{score_name}")

            if all_robustness_scores:
                bars = ax7.bar(range(len(all_robustness_scores)), all_robustness_scores)
                ax7.set_ylabel('Robustness Score')
                ax7.set_title('Model Robustness Summary')
                ax7.set_xticks(range(len(robustness_labels)))
                ax7.set_xticklabels(robustness_labels, rotation=45, ha='right')
                ax7.set_ylim(0, 1)

        # 8. Performance metrics summary table
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('off')

        overall_metrics = validation_results.get('overall_performance')
        if overall_metrics:
            summary_data = [
                ['Pearson Correlation', f"{overall_metrics.pearson_correlation:.4f}"],
                ['RMSE', f"{overall_metrics.rmse:.4f}"],
                ['R² Score', f"{overall_metrics.r2_score:.4f}"],
                ['Accuracy (0.1)', f"{overall_metrics.accuracy_within_0_10:.1%}"],
                ['Sample Count', f"{overall_metrics.sample_count}"],
                ['Export Ready', "✅" if export_assessment.export_ready else "❌"]
            ]

            table = ax8.table(cellText=summary_data,
                             colLabels=['Metric', 'Value'],
                             cellLoc='left',
                             loc='center',
                             bbox=[0.0, 0.0, 1.0, 1.0])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            ax8.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

        # 9. Prediction quality analysis
        ax9 = fig.add_subplot(gs[3, :])

        # Create a comprehensive quality score visualization
        quality_scores = []
        quality_labels = []

        # Overall correlation
        quality_scores.append(overall_metrics.pearson_correlation)
        quality_labels.append('Overall\nCorrelation')

        # RMSE (inverted and normalized)
        rmse_score = max(0, 1 - overall_metrics.rmse / 0.2)  # Normalize RMSE
        quality_scores.append(rmse_score)
        quality_labels.append('RMSE\n(inverted)')

        # Accuracy
        quality_scores.append(overall_metrics.accuracy_within_0_10)
        quality_labels.append('Accuracy\n(0.1 threshold)')

        # Cross-validation stability
        if cv_results:
            quality_scores.append(cv_results.stability_score)
            quality_labels.append('CV\nStability')

        # Export readiness
        quality_scores.append(export_assessment.readiness_score)
        quality_labels.append('Export\nReadiness')

        # Create radar-like bar chart
        bars = ax9.bar(quality_labels, quality_scores, color=plt.cm.RdYlGn([score for score in quality_scores]))
        ax9.set_ylabel('Quality Score')
        ax9.set_title('Comprehensive Model Quality Assessment')
        ax9.set_ylim(0, 1)

        # Add score labels on bars
        for bar, score in zip(bars, quality_scores):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.suptitle('Day 12: Comprehensive Model Validation Results',
                    fontsize=16, fontweight='bold')

        # Save visualization
        viz_path = self.save_dir / "comprehensive_validation_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Comprehensive validation visualization saved: {viz_path}")


if __name__ == "__main__":
    # Demo the validation suite
    print("🧪 Testing Comprehensive Model Validation Suite")

    validator = ModelValidationSuite()

    # This would normally use real model and test data
    print("✅ Validation suite initialization complete!")
    print(f"Save directory: {validator.save_dir}")
    print("Ready for comprehensive model validation!")