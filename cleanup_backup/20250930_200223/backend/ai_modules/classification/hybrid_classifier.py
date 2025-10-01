#!/usr/bin/env python3
"""
Hybrid Classification System for Logo Type Classification
Combines rule-based and neural network classifiers with intelligent routing
"""

import time
import logging
import os
import hashlib
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
import torch
import numpy as np

# Import our existing classifiers
from .rule_based_classifier import RuleBasedClassifier
from .efficientnet_classifier import EfficientNetClassifier
from ..feature_extraction import ImageFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Routing decision matrix as specified in Day 6 plan
ROUTING_STRATEGY = {
    'rule_confidence_high': {
        'threshold': 0.85,
        'action': 'use_rule_based',
        'expected_time': '0.1-0.5s'
    },
    'rule_confidence_medium': {
        'threshold': 0.65,
        'complexity_check': True,
        'action': 'conditional_neural',
        'expected_time': '0.5-5s'
    },
    'rule_confidence_low': {
        'threshold': 0.45,
        'action': 'use_neural_network',
        'expected_time': '2-5s'
    },
    'fallback': {
        'action': 'use_ensemble',
        'expected_time': '3-6s'
    }
}

class FeatureCache:
    """Feature extraction caching for performance optimization"""

    def __init__(self, max_size: int = 500):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}

    def _get_image_hash(self, image_path: str) -> str:
        """Generate hash for image file for caching"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return hashlib.md5(image_path.encode()).hexdigest()

    def get_cached_features(self, image_path: str) -> Optional[Dict]:
        """Get cached features if available"""
        image_hash = self._get_image_hash(image_path)
        if image_hash in self.cache:
            self.access_count[image_hash] = self.access_count.get(image_hash, 0) + 1
            logger.debug(f"Feature cache hit for image: {image_path}")
            return self.cache[image_hash]
        return None

    def cache_features(self, image_path: str, features: Dict):
        """Cache extracted features"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_count[lru_key]

        image_hash = self._get_image_hash(image_path)
        self.cache[image_hash] = features
        self.access_count[image_hash] = 1

class ClassificationCache:
    """Classification result caching for performance optimization"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}

    def _get_image_hash(self, image_path: str) -> str:
        """Generate hash for image file for caching"""
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return hashlib.md5(image_path.encode()).hexdigest()

    def get_cached_result(self, image_path: str) -> Optional[Dict]:
        """Get cached classification result if available"""
        image_hash = self._get_image_hash(image_path)
        if image_hash in self.cache:
            self.access_count[image_hash] = self.access_count.get(image_hash, 0) + 1
            logger.debug(f"Cache hit for image: {image_path}")
            return self.cache[image_hash]
        return None

    def cache_result(self, image_path: str, result: Dict):
        """Cache classification result"""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_count[lru_key]

        image_hash = self._get_image_hash(image_path)
        self.cache[image_hash] = result
        self.access_count[image_hash] = 1

class ConfidenceCalibrator:
    """Confidence calibration system for consistent confidence scales across methods"""

    def __init__(self):
        self.calibration_history = {
            'rule_based': {'predictions': [], 'confidences': [], 'correct': []},
            'neural_network': {'predictions': [], 'confidences': [], 'correct': []},
            'ensemble': {'predictions': [], 'confidences': [], 'correct': []}
        }
        self.method_accuracies = {
            'rule_based': {'high_conf': 0.85, 'medium_conf': 0.70, 'low_conf': 0.55},
            'neural_network': {'high_conf': 0.90, 'medium_conf': 0.75, 'low_conf': 0.60},
            'ensemble': {'high_conf': 0.95, 'medium_conf': 0.85, 'low_conf': 0.70}
        }
        # Calibration parameters updated during learning
        self.calibration_params = {
            'rule_based': {'scale': 1.0, 'shift': 0.0},
            'neural_network': {'scale': 1.0, 'shift': 0.0},
            'ensemble': {'scale': 1.0, 'shift': 0.0}
        }

    def calibrate_confidence(self, confidence: float, method: str,
                           context: Dict = None) -> float:
        """
        Calibrate confidence score based on method and historical performance

        Args:
            confidence: Raw confidence score
            method: Classification method used
            context: Optional context for calibration

        Returns:
            Calibrated confidence score [0, 1]
        """
        try:
            # Apply method-specific calibration
            if method in self.calibration_params:
                params = self.calibration_params[method]
                calibrated = confidence * params['scale'] + params['shift']
            else:
                calibrated = confidence

            # Apply context-based adjustments
            if context:
                calibrated = self._apply_context_adjustments(calibrated, method, context)

            # Apply historical accuracy adjustments
            calibrated = self._apply_accuracy_adjustments(calibrated, method)

            # Ensure valid range
            return float(np.clip(calibrated, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"Confidence calibration failed: {e}")
            return float(np.clip(confidence, 0.0, 1.0))

    def _apply_context_adjustments(self, confidence: float, method: str,
                                  context: Dict) -> float:
        """Apply context-based confidence adjustments"""

        # Image complexity adjustment
        complexity = context.get('complexity_score', 0.5)

        if method == 'rule_based':
            # Rule-based confidence decreases with complexity
            complexity_adjustment = -0.1 * (complexity - 0.5)
        elif method == 'neural_network':
            # Neural network more reliable for complex images
            complexity_adjustment = 0.05 * (complexity - 0.5)
        else:  # ensemble
            # Ensemble benefits from high complexity (where methods disagree)
            complexity_adjustment = 0.1 * max(0, complexity - 0.7)

        # Agreement adjustment for ensemble
        if method.startswith('ensemble'):
            agreement = context.get('agreement', True)
            if agreement:
                confidence += 0.1  # Boost for agreement
            else:
                confidence -= 0.05  # Slight reduction for disagreement

        return confidence + complexity_adjustment

    def _apply_accuracy_adjustments(self, confidence: float, method: str) -> float:
        """Apply historical accuracy-based adjustments"""

        if method not in self.method_accuracies:
            return confidence

        accuracies = self.method_accuracies[method]

        # Determine confidence tier
        if confidence >= 0.8:
            expected_accuracy = accuracies['high_conf']
            tier = 'high'
        elif confidence >= 0.6:
            expected_accuracy = accuracies['medium_conf']
            tier = 'medium'
        else:
            expected_accuracy = accuracies['low_conf']
            tier = 'low'

        # If we have historical data, use it for adjustment
        if method in self.calibration_history:
            history = self.calibration_history[method]
            if len(history['confidences']) > 10:  # Need sufficient data
                # Calculate actual accuracy for this confidence tier
                tier_mask = self._get_confidence_tier_mask(history['confidences'], tier)
                if np.sum(tier_mask) > 0:
                    actual_accuracy = np.mean(np.array(history['correct'])[tier_mask])

                    # Adjust confidence based on accuracy difference
                    accuracy_diff = actual_accuracy - expected_accuracy
                    adjustment = accuracy_diff * 0.2  # Scale adjustment
                    confidence += adjustment

        return confidence

    def _get_confidence_tier_mask(self, confidences: list, tier: str) -> np.ndarray:
        """Get boolean mask for confidence tier"""
        confidences = np.array(confidences)

        if tier == 'high':
            return confidences >= 0.8
        elif tier == 'medium':
            return (confidences >= 0.6) & (confidences < 0.8)
        else:  # low
            return confidences < 0.6

    def update_calibration_history(self, prediction: str, confidence: float,
                                 method: str, is_correct: bool):
        """Update calibration history with new prediction"""
        if method in self.calibration_history:
            history = self.calibration_history[method]
            history['predictions'].append(prediction)
            history['confidences'].append(confidence)
            history['correct'].append(is_correct)

            # Keep only recent history (sliding window)
            max_history = 1000
            if len(history['predictions']) > max_history:
                history['predictions'] = history['predictions'][-max_history:]
                history['confidences'] = history['confidences'][-max_history:]
                history['correct'] = history['correct'][-max_history:]

    def update_calibration_parameters(self, method: str):
        """Update calibration parameters based on historical performance"""
        if method not in self.calibration_history:
            return

        history = self.calibration_history[method]
        if len(history['confidences']) < 20:  # Need sufficient data
            return

        try:
            # Use Platt scaling approach
            confidences = np.array(history['confidences'])
            correct = np.array(history['correct'])

            # Simple linear calibration: confidence = scale * raw_conf + shift
            # Fit to minimize calibration error
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV

            # Reshape for sklearn
            X = confidences.reshape(-1, 1)
            y = correct.astype(int)

            # Fit calibration
            calibrator = LogisticRegression()
            calibrator.fit(X, y)

            # Extract parameters (simplified)
            scale = calibrator.coef_[0][0]
            shift = calibrator.intercept_[0]

            # Update parameters with smoothing
            params = self.calibration_params[method]
            params['scale'] = 0.9 * params['scale'] + 0.1 * scale
            params['shift'] = 0.9 * params['shift'] + 0.1 * shift

        except Exception as e:
            logger.warning(f"Calibration parameter update failed for {method}: {e}")

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics and reliability metrics"""
        stats = {}

        for method, history in self.calibration_history.items():
            if len(history['confidences']) == 0:
                stats[method] = {'samples': 0}
                continue

            confidences = np.array(history['confidences'])
            correct = np.array(history['correct'])

            method_stats = {
                'samples': int(len(confidences)),
                'accuracy': float(np.mean(correct)),
                'average_confidence': float(np.mean(confidences)),
                'calibration_error': self._calculate_calibration_error(confidences, correct),
                'reliability_bins': self._calculate_reliability_bins(confidences, correct)
            }

            stats[method] = method_stats

        return stats

    def _calculate_calibration_error(self, confidences: np.ndarray,
                                   correct: np.ndarray) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        try:
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    accuracy_in_bin = correct[in_bin].mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            return ece

        except Exception:
            return 0.0

    def _calculate_reliability_bins(self, confidences: np.ndarray,
                                  correct: np.ndarray) -> Dict[str, float]:
        """Calculate reliability for different confidence bins"""
        bins = {
            'high_confidence': {'threshold': 0.8, 'accuracy': 0.0, 'count': 0},
            'medium_confidence': {'threshold': 0.6, 'accuracy': 0.0, 'count': 0},
            'low_confidence': {'threshold': 0.0, 'accuracy': 0.0, 'count': 0}
        }

        # High confidence bin
        high_mask = confidences >= 0.8
        if np.sum(high_mask) > 0:
            bins['high_confidence']['accuracy'] = float(np.mean(correct[high_mask]))
            bins['high_confidence']['count'] = int(np.sum(high_mask))

        # Medium confidence bin
        medium_mask = (confidences >= 0.6) & (confidences < 0.8)
        if np.sum(medium_mask) > 0:
            bins['medium_confidence']['accuracy'] = float(np.mean(correct[medium_mask]))
            bins['medium_confidence']['count'] = int(np.sum(medium_mask))

        # Low confidence bin
        low_mask = confidences < 0.6
        if np.sum(low_mask) > 0:
            bins['low_confidence']['accuracy'] = float(np.mean(correct[low_mask]))
            bins['low_confidence']['count'] = int(np.sum(low_mask))

        return bins

class MemoryOptimizedClassifier:
    """Memory-efficient model loading with lazy initialization"""

    def __init__(self):
        self.neural_model = None
        self.neural_model_path = 'day6_exports/neural_network_traced.pt'  # ULTRATHINK TorchScript
        self.model_loaded = False
        self.memory_threshold = 200  # MB
        self.logger = logging.getLogger(__name__)

    def _load_neural_model_if_needed(self):
        """Load neural model only when needed with memory management"""
        if not self.model_loaded:
            import psutil
            memory_usage = psutil.virtual_memory().used / (1024 * 1024)

            if memory_usage > self.memory_threshold:
                # Memory pressure - unload other models if possible
                self._cleanup_memory()

            try:
                # Try ULTRATHINK TorchScript first
                if os.path.exists(self.neural_model_path):
                    self.neural_model = torch.jit.load(self.neural_model_path)
                    self.logger.info(f"Loaded TorchScript model from {self.neural_model_path}")
                else:
                    # Fallback to regular PyTorch model
                    fallback_path = "day6_exports/efficientnet_logo_classifier_best.pth"
                    if os.path.exists(fallback_path):
                        from .efficientnet_classifier import EfficientNetClassifier
                        temp_classifier = EfficientNetClassifier(model_path=fallback_path)
                        self.neural_model = temp_classifier.model
                        self.logger.info(f"Loaded PyTorch model from {fallback_path}")
                    else:
                        raise FileNotFoundError("No neural network model found")

                self.model_loaded = True

            except Exception as e:
                self.logger.error(f"Failed to load neural model: {e}")
                self.neural_model = None
                self.model_loaded = False

    def _cleanup_memory(self):
        """Clean up memory to make space for model loading"""
        import gc
        import torch

        self.logger.info("Cleaning up memory before model loading...")

        # Force garbage collection
        gc.collect()

        # Clear PyTorch cache if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Unload model if loaded and not recently used
        if self.model_loaded and hasattr(self, '_last_used_time'):
            import time
            if time.time() - self._last_used_time > 300:  # 5 minutes
                self.unload_model()

    def unload_model(self):
        """Explicitly unload the neural model to free memory"""
        if self.model_loaded:
            self.neural_model = None
            self.model_loaded = False
            self._cleanup_memory()
            self.logger.info("Neural model unloaded to free memory")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024),
            'model_loaded': self.model_loaded
        }

class HybridClassifier:
    """
    Intelligent hybrid logo classification system
    Combines rule-based and neural network classifiers with smart routing
    """

    def __init__(self, neural_model_path: str = None, enable_caching: bool = True):
        """
        Initialize hybrid classifier with intelligent routing

        Args:
            neural_model_path: Path to neural network model (optional)
            enable_caching: Enable result caching for performance
        """
        self.logger = logging.getLogger(__name__)

        # Initialize classifiers
        self.feature_extractor = ImageFeatureExtractor()
        self.rule_classifier = RuleBasedClassifier()

        # Initialize neural network classifier
        if neural_model_path is None:
            # Use the Day 6 exported model
            neural_model_path = "day6_exports/efficientnet_logo_classifier_best.pth"

        try:
            self.neural_classifier = EfficientNetClassifier(model_path=neural_model_path)
            self.neural_available = True
            self.logger.info("Neural network classifier initialized successfully")
        except Exception as e:
            self.logger.warning(f"Neural network classifier failed to initialize: {e}")
            self.neural_classifier = None
            self.neural_available = False

        # Routing configuration
        self.routing_config = ROUTING_STRATEGY

        # Performance tracking
        self.performance_stats = {
            'total_classifications': 0,
            'rule_based_used': 0,
            'neural_network_used': 0,
            'ensemble_used': 0,
            'average_time': 0.0,
            'cache_hits': 0
        }

        # Confidence calibration system
        self.confidence_calibrator = ConfidenceCalibrator()

        # Caching systems
        if enable_caching:
            self.cache = ClassificationCache()
            self.feature_cache = FeatureCache()
        else:
            self.cache = None
            self.feature_cache = None

        # Memory optimization
        self.memory_optimizer = MemoryOptimizedClassifier()

    def classify(self, image_path: str, time_budget: Optional[float] = None) -> Dict[str, Any]:
        """
        Intelligent classification with method routing

        Args:
            image_path: Path to image file
            time_budget: Maximum time allowed (seconds)

        Returns:
            Classification result with metadata
        """
        start_time = time.time()

        try:
            # Check cache first
            if self.cache:
                cached_result = self.cache.get_cached_result(image_path)
                if cached_result:
                    self.performance_stats['cache_hits'] += 1
                    cached_result['cache_hit'] = True
                    cached_result['processing_time'] = time.time() - start_time
                    return cached_result

            # Phase 1: Feature extraction (with caching)
            features = self._get_cached_features(image_path)

            # Phase 2: Rule-based classification
            rule_result = self._get_rule_classification(features)

            # Phase 3: Intelligent routing decision
            routing_decision = self._determine_routing(
                rule_result, features, time_budget
            )

            # Phase 4: Execute routing decision
            final_result = self._execute_classification(
                image_path, rule_result, routing_decision
            )

            # Phase 5: Add metadata and return
            processing_time = time.time() - start_time
            final_result = self._format_result(final_result, processing_time, routing_decision)

            # Cache result if caching is enabled
            if self.cache:
                self.cache.cache_result(image_path, final_result)

            # Update performance stats
            self._update_performance_stats(final_result, processing_time)

            return final_result

        except Exception as e:
            self.logger.error(f"Hybrid classification failed for {image_path}: {e}")
            return self._create_fallback_result(str(e))

    def _get_cached_features(self, image_path: str) -> Dict[str, float]:
        """Get features with caching for performance optimization"""
        # Check feature cache first
        if self.feature_cache:
            cached_features = self.feature_cache.get_cached_features(image_path)
            if cached_features:
                self.performance_stats.setdefault('feature_cache_hits', 0)
                self.performance_stats['feature_cache_hits'] += 1
                return cached_features

        # Extract features if not cached
        start_time = time.time()
        features = self.feature_extractor.extract_features(image_path)
        extraction_time = time.time() - start_time

        # Cache the features for future use
        if self.feature_cache:
            self.feature_cache.cache_features(image_path, features)

        # Track feature extraction time
        self.performance_stats.setdefault('feature_extraction_time', [])
        self.performance_stats['feature_extraction_time'].append(extraction_time)

        return features

    def classify_batch(self, image_paths: list, time_budget_per_image: Optional[float] = None) -> list:
        """
        Optimized batch classification for multiple images

        Args:
            image_paths: List of image file paths
            time_budget_per_image: Maximum time allowed per image (seconds)

        Returns:
            List of classification results
        """
        start_time = time.time()
        self.logger.info(f"Starting batch classification of {len(image_paths)} images")

        # Extract features for all images (with caching)
        features_batch = []
        for path in image_paths:
            try:
                features = self._get_cached_features(path)
                features_batch.append((path, features))
            except Exception as e:
                self.logger.error(f"Feature extraction failed for {path}: {e}")
                features_batch.append((path, None))

        # Rule-based classification for all
        rule_results = []
        for path, features in features_batch:
            if features is not None:
                rule_result = self._get_rule_classification(features)
                rule_results.append((path, rule_result, features))
            else:
                # Create error result for failed feature extraction
                error_result = {
                    'logo_type': 'unknown',
                    'confidence': 0.0,
                    'reasoning': 'Feature extraction failed',
                    'features_used': {}
                }
                rule_results.append((path, error_result, {}))

        # Determine routing for each image
        neural_indices = []
        results = [None] * len(image_paths)  # Pre-allocate results list

        for i, (path, rule_result, features) in enumerate(rule_results):
            if features:
                routing = self._determine_routing(rule_result, features, time_budget_per_image)

                if routing['use_neural'] and self.neural_available:
                    neural_indices.append(i)
                else:
                    # Use rule-based result directly
                    results[i] = self._format_result(rule_result, 0.1, routing)
            else:
                # Error case - use fallback result
                results[i] = self._create_fallback_result("Feature extraction failed")

        # Batch neural network inference for selected images
        if neural_indices and self.neural_available:
            neural_paths = [image_paths[i] for i in neural_indices]

            try:
                if hasattr(self.neural_classifier, 'classify_batch'):
                    # Use batch inference if available
                    neural_results = self.neural_classifier.classify_batch(neural_paths)
                else:
                    # Fall back to individual classification
                    neural_results = []
                    for path in neural_paths:
                        try:
                            result = self.neural_classifier.classify(path)
                            neural_results.append(result)
                        except Exception as e:
                            self.logger.error(f"Neural classification failed for {path}: {e}")
                            neural_results.append(('unknown', 0.0))

                # Insert neural results in correct positions
                for neural_idx, original_idx in enumerate(neural_indices):
                    if neural_idx < len(neural_results):
                        neural_result = neural_results[neural_idx]
                        if isinstance(neural_result, tuple):
                            logo_type, confidence = neural_result
                            formatted_result = {
                                'logo_type': logo_type,
                                'confidence': confidence,
                                'method_used': 'neural_network',
                                'reasoning': 'Neural network classification',
                                'processing_time': 2.0  # Approximate
                            }
                        else:
                            formatted_result = neural_result

                        routing_info = {'method': 'neural_network', 'use_neural': True, 'reasoning': 'Batch neural inference'}
                        results[original_idx] = self._format_result(formatted_result, 2.0, routing_info)

            except Exception as e:
                self.logger.error(f"Batch neural inference failed: {e}")
                # Fall back to rule-based for all neural candidates
                for i in neural_indices:
                    if results[i] is None:
                        path, rule_result, features = rule_results[i]
                        routing = {'method': 'rule_based_fallback', 'reasoning': 'Neural batch failed'}
                        results[i] = self._format_result(rule_result, 0.1, routing)

        # Ensure all results are filled
        for i, result in enumerate(results):
            if result is None:
                results[i] = self._create_fallback_result("Classification failed")

        total_time = time.time() - start_time
        self.logger.info(f"Batch classification completed in {total_time:.3f}s "
                        f"({total_time/len(image_paths):.3f}s per image)")

        return results

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics"""
        memory_stats = self.memory_optimizer.get_memory_usage()

        # Add cache memory estimates
        if self.cache:
            memory_stats['classification_cache_size'] = len(self.cache.cache)
        if self.feature_cache:
            memory_stats['feature_cache_size'] = len(self.feature_cache.cache)

        # Add performance stats
        memory_stats['performance_stats'] = self.performance_stats

        return memory_stats

    def cleanup_memory(self):
        """Clean up memory and optimize usage"""
        self.logger.info("Starting memory cleanup...")

        # Clean up memory optimizer
        self.memory_optimizer._cleanup_memory()

        # Clean up caches if they're getting large
        if self.cache and len(self.cache.cache) > 500:
            # Clear half the cache (LRU)
            items_to_remove = len(self.cache.cache) // 2
            lru_keys = sorted(self.cache.access_count.items(), key=lambda x: x[1])[:items_to_remove]
            for key, _ in lru_keys:
                if key in self.cache.cache:
                    del self.cache.cache[key]
                del self.cache.access_count[key]
            self.logger.info(f"Cleared {items_to_remove} items from classification cache")

        if self.feature_cache and len(self.feature_cache.cache) > 250:
            # Clear half the feature cache
            items_to_remove = len(self.feature_cache.cache) // 2
            lru_keys = sorted(self.feature_cache.access_count.items(), key=lambda x: x[1])[:items_to_remove]
            for key, _ in lru_keys:
                if key in self.feature_cache.cache:
                    del self.feature_cache.cache[key]
                del self.feature_cache.access_count[key]
            self.logger.info(f"Cleared {items_to_remove} items from feature cache")

    def monitor_memory_usage(self) -> bool:
        """Monitor memory usage and take action if needed"""
        memory_stats = self.get_memory_usage()

        # Check if memory usage is high
        if memory_stats.get('percent', 0) > 80:
            self.logger.warning(f"High memory usage detected: {memory_stats['percent']:.1f}%")
            self.cleanup_memory()
            return True

        return False

    def classify_with_fallbacks(self, image_path: str) -> Dict[str, Any]:
        """Classification with comprehensive error handling"""

        try:
            # Primary classification attempt
            return self.classify(image_path)

        except FileNotFoundError:
            return self._create_error_result('image_not_found', f"Image file not found: {image_path}")

        except Exception as e:
            # Check for PIL image errors
            import PIL
            if isinstance(e, (PIL.UnidentifiedImageError, OSError)):
                return self._create_error_result('invalid_image', f"Invalid or corrupted image: {image_path}")

            # Check for PyTorch memory errors
            if 'out of memory' in str(e).lower() or isinstance(e, RuntimeError) and 'memory' in str(e).lower():
                # Fallback to rule-based only
                try:
                    self.logger.warning(f"Neural network out of memory, falling back to rule-based for {image_path}")
                    features = self._get_cached_features(image_path)
                    rule_result = self._get_rule_classification(features)
                    rule_result['method_used'] = 'rule_based_fallback'
                    rule_result['reasoning'] = 'Neural network unavailable due to memory constraints'
                    rule_result['processing_time'] = 0.1
                    return rule_result

                except Exception as fallback_error:
                    return self._create_error_result('classification_failed',
                                                   f"All classification methods failed: {str(fallback_error)}")

            # Log unexpected errors
            self.logger.error(f"Unexpected classification error for {image_path}: {e}")
            return self._create_error_result('unexpected_error', f"Unexpected error: {str(e)}")

    def _create_error_result(self, error_type: str, message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'logo_type': 'unknown',
            'confidence': 0.0,
            'raw_confidence': 0.0,
            'method_used': 'error_fallback',
            'reasoning': message,
            'error': True,
            'error_type': error_type,
            'error_message': message,
            'processing_time': 0.0,
            'timestamp': time.time(),
            'routing_decision': {
                'method': 'error_fallback',
                'reasoning': message,
                'use_neural': False,
                'use_ensemble': False
            },
            'calibration_applied': False
        }

    def validate_input(self, image_path: str) -> bool:
        """Validate input image before processing"""

        # Check file existence
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Check file size (reasonable limits)
        file_size = os.path.getsize(image_path)
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError(f"Image file too large: {file_size / (1024*1024):.1f}MB")

        if file_size < 100:  # 100 bytes minimum
            raise ValueError(f"Image file too small: {file_size} bytes")

        # Check file format
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                if img.format not in ['PNG', 'JPEG', 'JPG', 'BMP', 'GIF']:
                    raise ValueError(f"Unsupported image format: {img.format}")

                # Check image dimensions
                width, height = img.size
                if width < 10 or height < 10:
                    raise ValueError(f"Image too small: {width}x{height}")

                if width > 5000 or height > 5000:
                    raise ValueError(f"Image too large: {width}x{height}")

                # Check for suspicious file patterns
                if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                    self.logger.warning(f"Unusual image mode: {img.mode} for {image_path}")

        except Exception as e:
            if isinstance(e, ValueError):
                raise  # Re-raise validation errors
            raise ValueError(f"Invalid image file: {str(e)}")

        return True

    def classify_safe(self, image_path: str, time_budget: Optional[float] = None) -> Dict[str, Any]:
        """Safe classification with input validation and error handling"""
        try:
            # Validate input first
            self.validate_input(image_path)

            # Monitor memory usage
            self.monitor_memory_usage()

            # Perform classification with fallbacks
            return self.classify_with_fallbacks(image_path)

        except Exception as e:
            self.logger.error(f"Safe classification failed for {image_path}: {e}")
            return self._create_error_result('validation_failed', f"Input validation failed: {str(e)}")

    def _get_rule_classification(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Get rule-based classification with confidence"""
        try:
            logo_type, confidence = self.rule_classifier.classify(features)

            # Get detailed explanation if available
            if hasattr(self.rule_classifier, 'classify_with_explanation'):
                detailed = self.rule_classifier.classify_with_explanation(features)
                reasoning = detailed.get('reasoning', f'Rule-based classification: {logo_type}')
            else:
                reasoning = f'Rule-based classification: {logo_type} (confidence: {confidence:.2f})'

            return {
                'logo_type': logo_type,
                'confidence': confidence,
                'reasoning': reasoning,
                'features_used': features
            }

        except Exception as e:
            self.logger.error(f"Rule-based classification failed: {e}")
            return {
                'logo_type': 'simple',
                'confidence': 0.5,
                'reasoning': f'Rule-based fallback due to error: {e}',
                'features_used': features
            }

    def _determine_routing(self, rule_result: Dict, features: Dict,
                          time_budget: Optional[float]) -> Dict[str, Any]:
        """Determine which classification method(s) to use"""

        confidence = rule_result['confidence']
        complexity = features.get('complexity_score', 0.5)

        routing_decision = {
            'method': 'rule_based',  # default
            'use_neural': False,
            'use_ensemble': False,
            'reasoning': '',
            'estimated_time': 0.1
        }

        # High confidence rule-based result
        if confidence >= self.routing_config['rule_confidence_high']['threshold']:
            routing_decision.update({
                'method': 'rule_based',
                'reasoning': f'High confidence rule-based result: {confidence:.2f}',
                'estimated_time': 0.1
            })

        # Medium confidence - check complexity and time budget
        elif confidence >= self.routing_config['rule_confidence_medium']['threshold']:
            if complexity > 0.7 or (time_budget and time_budget > 3.0):
                if self.neural_available:
                    routing_decision.update({
                        'method': 'neural_network',
                        'use_neural': True,
                        'reasoning': f'Medium confidence with high complexity: {complexity:.2f}',
                        'estimated_time': 3.0
                    })
                else:
                    routing_decision.update({
                        'method': 'rule_based',
                        'reasoning': f'Neural network unavailable, using rule-based: {complexity:.2f}',
                        'estimated_time': 0.1
                    })
            else:
                routing_decision.update({
                    'method': 'rule_based',
                    'reasoning': f'Medium confidence, low complexity: {complexity:.2f}',
                    'estimated_time': 0.1
                })

        # Low confidence - use neural network
        elif confidence >= self.routing_config['rule_confidence_low']['threshold']:
            if self.neural_available:
                routing_decision.update({
                    'method': 'neural_network',
                    'use_neural': True,
                    'reasoning': f'Low rule confidence: {confidence:.2f}',
                    'estimated_time': 3.0
                })
            else:
                routing_decision.update({
                    'method': 'rule_based',
                    'reasoning': f'Low confidence but neural unavailable: {confidence:.2f}',
                    'estimated_time': 0.1
                })

        # Very low confidence - use ensemble
        else:
            if self.neural_available:
                routing_decision.update({
                    'method': 'ensemble',
                    'use_neural': True,
                    'use_ensemble': True,
                    'reasoning': f'Very low confidence, using ensemble: {confidence:.2f}',
                    'estimated_time': 4.0
                })
            else:
                routing_decision.update({
                    'method': 'rule_based',
                    'reasoning': f'Very low confidence but neural unavailable: {confidence:.2f}',
                    'estimated_time': 0.1
                })

        # Time budget override
        if time_budget and routing_decision['estimated_time'] > time_budget:
            routing_decision.update({
                'method': 'rule_based',
                'use_neural': False,
                'use_ensemble': False,
                'reasoning': f'Time budget constraint: {time_budget}s',
                'estimated_time': 0.1
            })

        return routing_decision

    def _execute_classification(self, image_path: str, rule_result: Dict,
                              routing_decision: Dict) -> Dict[str, Any]:
        """Execute the chosen classification method(s)"""

        if routing_decision['method'] == 'rule_based':
            return {
                'logo_type': rule_result['logo_type'],
                'confidence': rule_result['confidence'],
                'method_used': 'rule_based',
                'reasoning': rule_result['reasoning']
            }

        elif routing_decision['method'] == 'neural_network':
            if self.neural_available:
                try:
                    neural_result = self.neural_classifier.classify(image_path)
                    return {
                        'logo_type': neural_result['logo_type'],
                        'confidence': neural_result['confidence'],
                        'method_used': 'neural_network',
                        'reasoning': 'Neural network classification',
                        'all_probabilities': neural_result.get('all_probabilities', {})
                    }
                except Exception as e:
                    self.logger.error(f"Neural network classification failed: {e}")
                    return {
                        'logo_type': rule_result['logo_type'],
                        'confidence': rule_result['confidence'] * 0.8,  # Reduce confidence due to fallback
                        'method_used': 'rule_based_fallback',
                        'reasoning': f'Neural network failed, using rule-based fallback: {e}'
                    }
            else:
                return {
                    'logo_type': rule_result['logo_type'],
                    'confidence': rule_result['confidence'],
                    'method_used': 'rule_based_fallback',
                    'reasoning': 'Neural network not available, using rule-based'
                }

        elif routing_decision['method'] == 'ensemble':
            return self._ensemble_classify(image_path, rule_result)

        else:
            # Fallback to rule-based
            return {
                'logo_type': rule_result['logo_type'],
                'confidence': rule_result['confidence'],
                'method_used': 'rule_based_fallback',
                'reasoning': 'Fallback to rule-based classification'
            }

    def _ensemble_classify(self, image_path: str, rule_result: Dict) -> Dict[str, Any]:
        """Combine rule-based and neural network results"""

        if not self.neural_available:
            return {
                'logo_type': rule_result['logo_type'],
                'confidence': rule_result['confidence'],
                'method_used': 'rule_based_only',
                'reasoning': 'Neural network not available for ensemble'
            }

        try:
            # Get neural network result
            neural_result = self.neural_classifier.classify(image_path)
            neural_type = neural_result['logo_type']
            neural_confidence = neural_result['confidence']

            # Extract rule-based results
            rule_type = rule_result['logo_type']
            rule_confidence = rule_result['confidence']

            # Agreement case - both methods agree
            if rule_type == neural_type:
                # Weighted confidence (higher weight for more confident method)
                if rule_confidence > neural_confidence:
                    final_confidence = 0.7 * rule_confidence + 0.3 * neural_confidence
                else:
                    final_confidence = 0.3 * rule_confidence + 0.7 * neural_confidence

                return {
                    'logo_type': rule_type,
                    'confidence': min(0.95, final_confidence + 0.1),  # Boost for agreement
                    'method_used': 'ensemble_agreement',
                    'reasoning': f'Both methods agree: rule={rule_confidence:.2f}, neural={neural_confidence:.2f}',
                    'agreement': True,
                    'rule_result': {'type': rule_type, 'confidence': rule_confidence},
                    'neural_result': {'type': neural_type, 'confidence': neural_confidence}
                }

            # Disagreement case - methods disagree
            else:
                # Use the more confident prediction
                if rule_confidence > neural_confidence:
                    final_type = rule_type
                    final_confidence = rule_confidence * 0.8  # Reduce confidence due to disagreement
                    winning_method = 'rule_based'
                    alternative_type = neural_type
                    alternative_confidence = neural_confidence
                else:
                    final_type = neural_type
                    final_confidence = neural_confidence * 0.8
                    winning_method = 'neural_network'
                    alternative_type = rule_type
                    alternative_confidence = rule_confidence

                return {
                    'logo_type': final_type,
                    'confidence': final_confidence,
                    'method_used': f'ensemble_disagreement_{winning_method}',
                    'reasoning': f'Disagreement resolved by confidence: rule={rule_type}({rule_confidence:.2f}) vs neural={neural_type}({neural_confidence:.2f})',
                    'agreement': False,
                    'alternative_prediction': {
                        'logo_type': alternative_type,
                        'confidence': alternative_confidence
                    },
                    'rule_result': {'type': rule_type, 'confidence': rule_confidence},
                    'neural_result': {'type': neural_type, 'confidence': neural_confidence}
                }

        except Exception as e:
            self.logger.error(f"Ensemble classification failed: {e}")
            return {
                'logo_type': rule_result['logo_type'],
                'confidence': rule_result['confidence'] * 0.9,
                'method_used': 'ensemble_fallback',
                'reasoning': f'Ensemble failed, using rule-based: {e}'
            }

    def _format_result(self, result: Dict[str, Any], processing_time: float,
                      routing_decision: Dict) -> Dict[str, Any]:
        """Format final classification result with metadata"""

        # Extract base information
        raw_confidence = result.get('confidence', 0.0)
        method_used = result.get('method_used', 'unknown')

        # Prepare calibration context
        calibration_context = {
            'complexity_score': routing_decision.get('complexity_score', 0.5),
            'processing_time': processing_time,
            'agreement': result.get('agreement', True)
        }

        # Apply confidence calibration
        calibrated_confidence = self.confidence_calibrator.calibrate_confidence(
            confidence=raw_confidence,
            method=method_used.split('_')[0],  # Get base method (rule, neural, ensemble)
            context=calibration_context
        )

        formatted_result = {
            'logo_type': result.get('logo_type', 'unknown'),
            'confidence': calibrated_confidence,
            'raw_confidence': raw_confidence,  # Keep original for analysis
            'method_used': method_used,
            'reasoning': result.get('reasoning', ''),
            'processing_time': processing_time,
            'routing_decision': routing_decision,
            'cache_hit': result.get('cache_hit', False),
            'calibration_applied': True,
            'timestamp': time.time()
        }

        # Add optional fields if they exist
        optional_fields = ['all_probabilities', 'agreement', 'alternative_prediction',
                          'rule_result', 'neural_result']
        for field in optional_fields:
            if field in result:
                formatted_result[field] = result[field]

        return formatted_result

    def _create_fallback_result(self, error_message: str) -> Dict[str, Any]:
        """Create fallback result for error cases"""
        return {
            'logo_type': 'simple',  # Safe default
            'confidence': 0.3,
            'method_used': 'fallback',
            'reasoning': f'Classification failed: {error_message}',
            'processing_time': 0.0,
            'error': True,
            'error_message': error_message,
            'timestamp': time.time()
        }

    def _update_performance_stats(self, result: Dict, processing_time: float):
        """Update performance statistics"""
        self.performance_stats['total_classifications'] += 1

        # Update method usage counts
        method = result.get('method_used', 'unknown')
        if 'rule_based' in method:
            self.performance_stats['rule_based_used'] += 1
        elif 'neural_network' in method:
            self.performance_stats['neural_network_used'] += 1
        elif 'ensemble' in method:
            self.performance_stats['ensemble_used'] += 1

        # Update average time (rolling average)
        total = self.performance_stats['total_classifications']
        current_avg = self.performance_stats['average_time']
        self.performance_stats['average_time'] = ((current_avg * (total - 1)) + processing_time) / total

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = self.performance_stats.copy()

        if stats['total_classifications'] > 0:
            total = stats['total_classifications']
            stats['method_distribution'] = {
                'rule_based_percent': (stats['rule_based_used'] / total) * 100,
                'neural_network_percent': (stats['neural_network_used'] / total) * 100,
                'ensemble_percent': (stats['ensemble_used'] / total) * 100
            }

            if self.cache:
                stats['cache_hit_rate'] = (stats['cache_hits'] / total) * 100

        return stats

    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'total_classifications': 0,
            'rule_based_used': 0,
            'neural_network_used': 0,
            'ensemble_used': 0,
            'average_time': 0.0,
            'cache_hits': 0
        }

    def classify_batch(self, image_paths: list, time_budget_per_image: Optional[float] = None) -> list:
        """Classify multiple images with batch processing optimization"""
        results = []

        for image_path in image_paths:
            try:
                result = self.classify(image_path, time_budget_per_image)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch classification failed for {image_path}: {e}")
                results.append(self._create_fallback_result(str(e)))

        return results

    def update_calibration_feedback(self, prediction_result: Dict, ground_truth: str):
        """Update calibration system with ground truth feedback

        Args:
            prediction_result: Result from classify() method
            ground_truth: True label for the image
        """
        try:
            predicted_label = prediction_result.get('logo_type', '')
            raw_confidence = prediction_result.get('raw_confidence', prediction_result.get('confidence', 0.0))
            method_used = prediction_result.get('method_used', 'unknown')
            is_correct = predicted_label == ground_truth

            # Extract base method name
            base_method = method_used.split('_')[0]
            if base_method not in ['rule', 'neural', 'ensemble']:
                base_method = 'rule'  # Default fallback

            # Map method names for calibrator
            method_mapping = {
                'rule': 'rule_based',
                'neural': 'neural_network',
                'ensemble': 'ensemble'
            }

            calibrator_method = method_mapping.get(base_method, 'rule_based')

            # Update calibration history
            self.confidence_calibrator.update_calibration_history(
                prediction=predicted_label,
                confidence=raw_confidence,
                method=calibrator_method,
                is_correct=is_correct
            )

            # Update calibration parameters if we have enough data
            self.confidence_calibrator.update_calibration_parameters(calibrator_method)

        except Exception as e:
            self.logger.error(f"Calibration feedback update failed: {e}")

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get confidence calibration statistics"""
        return self.confidence_calibrator.get_calibration_stats()

    def validate_calibration_quality(self) -> Dict[str, Any]:
        """Validate the quality of confidence calibration across methods"""
        calibration_stats = self.get_calibration_stats()
        validation_results = {
            'overall_quality': 'good',
            'method_quality': {},
            'recommendations': []
        }

        quality_issues = 0

        for method, stats in calibration_stats.items():
            if stats.get('samples', 0) < 50:
                validation_results['method_quality'][method] = 'insufficient_data'
                validation_results['recommendations'].append(
                    f"Collect more data for {method} calibration (current: {stats.get('samples', 0)})"
                )
                continue

            calibration_error = stats.get('calibration_error', 0.0)
            accuracy = stats.get('accuracy', 0.0)
            avg_confidence = stats.get('average_confidence', 0.0)

            # Quality criteria
            if calibration_error > 0.15:  # High calibration error
                validation_results['method_quality'][method] = 'poor_calibration'
                validation_results['recommendations'].append(
                    f"{method} has high calibration error ({calibration_error:.3f})"
                )
                quality_issues += 1
            elif abs(accuracy - avg_confidence) > 0.2:  # Large confidence-accuracy gap
                validation_results['method_quality'][method] = 'confidence_mismatch'
                validation_results['recommendations'].append(
                    f"{method} confidence doesn't match accuracy (acc: {accuracy:.3f}, conf: {avg_confidence:.3f})"
                )
                quality_issues += 1
            else:
                validation_results['method_quality'][method] = 'good'

        # Overall quality assessment
        if quality_issues == 0:
            validation_results['overall_quality'] = 'excellent'
        elif quality_issues == 1:
            validation_results['overall_quality'] = 'good'
        elif quality_issues == 2:
            validation_results['overall_quality'] = 'fair'
        else:
            validation_results['overall_quality'] = 'poor'

        return validation_results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the hybrid classifier system"""
        return {
            'system_type': 'HybridClassifier',
            'components': {
                'rule_based': 'RuleBasedClassifier',
                'neural_network': 'EfficientNetClassifier' if self.neural_available else 'Not Available',
                'feature_extractor': 'ImageFeatureExtractor',
                'confidence_calibrator': 'ConfidenceCalibrator'
            },
            'neural_available': self.neural_available,
            'caching_enabled': self.cache is not None,
            'calibration_enabled': True,
            'routing_strategy': self.routing_config,
            'performance_stats': self.get_performance_stats(),
            'calibration_stats': self.get_calibration_stats()
        }

# Main function for testing
if __name__ == "__main__":
    # Test the hybrid classifier
    hybrid = HybridClassifier()
    print("Hybrid classifier initialized successfully")
    print("Model info:", hybrid.get_model_info())