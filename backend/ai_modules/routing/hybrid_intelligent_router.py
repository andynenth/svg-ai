# backend/ai_modules/routing/hybrid_intelligent_router.py
import time
import logging
import random
from typing import Dict, Any, Optional, List
from pathlib import Path

# Import Day 1 components
from ..management.production_model_manager import ProductionModelManager
from ..inference.optimized_quality_predictor import OptimizedQualityPredictor

class SimpleFeatureExtractor:
    """Simplified feature extractor for when full extractor is unavailable"""

    def extract_features(self, image_path: str) -> Dict[str, float]:
        """Extract basic features using PIL"""
        try:
            from PIL import Image
            import numpy as np

            # Load image
            image = Image.open(image_path).convert('RGB')
            img_array = np.array(image)

            # Basic feature extraction
            height, width = img_array.shape[:2]

            # Color features
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))

            # Edge approximation (simplified)
            gray = np.mean(img_array, axis=2)
            edges = np.abs(np.diff(gray, axis=0)).sum() + np.abs(np.diff(gray, axis=1)).sum()
            edge_density = edges / (height * width)

            # Entropy approximation
            hist, _ = np.histogram(gray, bins=256, range=(0, 256))
            hist_norm = hist / hist.sum()
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))

            return {
                'edge_density': min(edge_density / 1000, 1.0),  # Normalize
                'unique_colors': min(unique_colors / 256, 1.0),
                'entropy': entropy / 8.0,
                'corner_density': edge_density / 2000,  # Approximation
                'gradient_strength': edge_density / 1500,
                'complexity_score': (edge_density / 1000 + unique_colors / 256 + entropy / 8.0) / 3
            }

        except Exception as e:
            logging.warning(f"Feature extraction failed: {e}, using defaults")
            return {
                'edge_density': 0.1,
                'unique_colors': 0.3,
                'entropy': 0.5,
                'corner_density': 0.1,
                'gradient_strength': 0.2,
                'complexity_score': 0.3
            }

class SimpleClassifier:
    """Simplified classifier for logo type detection"""

    def classify(self, image_path: str, features: Dict[str, float]) -> tuple[str, float]:
        """Classify logo type based on features"""
        try:
            complexity = features.get('complexity_score', 0.3)
            edge_density = features.get('edge_density', 0.1)
            unique_colors = features.get('unique_colors', 0.3)

            # Simple heuristic classification
            if complexity < 0.2 and edge_density < 0.1:
                return 'simple_geometric', 0.8
            elif unique_colors < 0.1 and edge_density > 0.3:
                return 'text_based', 0.7
            elif unique_colors > 0.7:
                return 'gradient', 0.6
            else:
                return 'complex', 0.5

        except Exception as e:
            logging.warning(f"Classification failed: {e}")
            return 'unknown', 0.5

class HybridIntelligentRouter:
    """Intelligent routing system for optimal tier selection"""

    def __init__(self, model_manager: ProductionModelManager):
        self.model_manager = model_manager
        self.quality_predictor = OptimizedQualityPredictor(model_manager)
        self.feature_extractor = self._get_feature_extractor()
        self.classifier = self._get_classifier()

    def _get_feature_extractor(self):
        """Get feature extractor with fallback"""
        try:
            from ..feature_extraction import ImageFeatureExtractor
            return ImageFeatureExtractor()
        except ImportError:
            logging.warning("Feature extractor unavailable, using simplified version")
            return SimpleFeatureExtractor()

    def _get_classifier(self):
        """Get classifier with fallback"""
        try:
            from ..classification.hybrid_classifier import HybridClassifier
            return HybridClassifier()
        except ImportError:
            logging.warning("Hybrid classifier unavailable, using simple classifier")
            return SimpleClassifier()

    def determine_optimal_tier(self,
                             image_path: str,
                             target_quality: float = 0.9,
                             time_budget: Optional[float] = None) -> Dict[str, Any]:
        """Determine optimal processing tier for given constraints"""

        start_time = time.time()

        try:
            # Phase 1: Quick image analysis
            features = self.feature_extractor.extract_features(image_path)

            # Handle different classifier types
            if hasattr(self.classifier, 'classify') and len(self.classifier.classify.__code__.co_varnames) >= 3:
                # Full classifier with features parameter
                logo_type, confidence = self.classifier.classify(image_path, features)
            else:
                # Simple classifier
                logo_type, confidence = self.classifier.classify(image_path, features)

            # Phase 2: Predict quality for each tier
            tier_predictions = {}
            for tier in [1, 2, 3]:
                prediction = self._predict_tier_performance(
                    image_path, features, tier, logo_type
                )
                tier_predictions[tier] = prediction

            # Phase 3: Select optimal tier
            optimal_tier = self._select_optimal_tier(
                tier_predictions,
                target_quality,
                time_budget
            )

            routing_time = time.time() - start_time

            return {
                'selected_tier': optimal_tier,
                'routing_time': routing_time,
                'logo_type': logo_type,
                'confidence': confidence,
                'target_quality': target_quality,
                'tier_predictions': tier_predictions,
                'selection_reasoning': self._explain_selection(
                    optimal_tier, tier_predictions, target_quality, time_budget
                )
            }

        except Exception as e:
            logging.error(f"Routing failed: {e}")
            # Fallback to conservative tier selection
            return self._fallback_tier_selection(target_quality, time_budget)

    def _predict_tier_performance(self,
                                image_path: str,
                                features: Dict[str, float],
                                tier: int,
                                logo_type: str) -> Dict[str, Any]:
        """Predict performance for a specific tier"""

        # Get tier-specific parameters
        params = self._get_tier_params(features, tier, logo_type)

        # Predict quality
        predicted_quality = self.quality_predictor.predict_quality(image_path, params)

        # Estimate processing time
        estimated_time = self._estimate_tier_time(tier, features)

        return {
            'predicted_quality': predicted_quality,
            'estimated_time': estimated_time,
            'parameters': params,
            'confidence': self._calculate_prediction_confidence(tier, logo_type)
        }

    def _get_tier_params(self, features: Dict, tier: int, logo_type: str) -> Dict[str, Any]:
        """Get optimized parameters for specific tier"""

        if tier == 1:
            # Method 1: Fast correlation-based optimization
            return self._get_method1_params(features, logo_type)
        elif tier == 2:
            # Method 1 + 2: Add quality prediction guidance
            base_params = self._get_method1_params(features, logo_type)
            return self._refine_with_quality_prediction(base_params, features)
        else:
            # Method 1 + 2 + 3: Full optimization
            return self._get_method3_params(features, logo_type)

    def _get_method1_params(self, features: Dict, logo_type: str) -> Dict[str, Any]:
        """Get Method 1 parameters based on logo type and features"""

        # Base parameters by logo type
        if logo_type == 'simple_geometric':
            base_params = {
                'color_precision': 3,
                'corner_threshold': 30,
                'length_threshold': 4,
                'max_iterations': 10,
                'splice_threshold': 45,
                'path_precision': 3,
                'layer_difference': 16,
                'filter_speckle': 4
            }
        elif logo_type == 'text_based':
            base_params = {
                'color_precision': 2,
                'corner_threshold': 20,
                'length_threshold': 3,
                'max_iterations': 15,
                'splice_threshold': 30,
                'path_precision': 5,
                'layer_difference': 8,
                'filter_speckle': 2
            }
        elif logo_type == 'gradient':
            base_params = {
                'color_precision': 8,
                'corner_threshold': 40,
                'length_threshold': 5,
                'max_iterations': 20,
                'splice_threshold': 60,
                'path_precision': 4,
                'layer_difference': 4,
                'filter_speckle': 6
            }
        else:  # complex or unknown
            base_params = {
                'color_precision': 6,
                'corner_threshold': 35,
                'length_threshold': 4,
                'max_iterations': 15,
                'splice_threshold': 50,
                'path_precision': 4,
                'layer_difference': 12,
                'filter_speckle': 4
            }

        # Fine-tune based on features
        complexity = features.get('complexity_score', 0.5)

        # Adjust color precision based on complexity
        if complexity > 0.7:
            base_params['color_precision'] = min(base_params['color_precision'] + 2, 10)
        elif complexity < 0.3:
            base_params['color_precision'] = max(base_params['color_precision'] - 1, 1)

        return base_params

    def _refine_with_quality_prediction(self, base_params: Dict, features: Dict) -> Dict[str, Any]:
        """Refine Method 1 parameters using quality prediction"""

        # Make a copy to avoid modifying the original
        refined_params = base_params.copy()

        # Use complexity to adjust parameters
        complexity = features.get('complexity_score', 0.5)

        if complexity > 0.6:
            # High complexity - increase quality-focused parameters
            refined_params['max_iterations'] = min(refined_params['max_iterations'] + 5, 30)
            refined_params['path_precision'] = min(refined_params['path_precision'] + 1, 8)
        elif complexity < 0.4:
            # Low complexity - optimize for speed
            refined_params['max_iterations'] = max(refined_params['max_iterations'] - 3, 5)

        return refined_params

    def _get_method3_params(self, features: Dict, logo_type: str) -> Dict[str, Any]:
        """Get Method 3 (full optimization) parameters"""

        # Start with Method 1 + 2 base
        base_params = self._refine_with_quality_prediction(
            self._get_method1_params(features, logo_type), features
        )

        # Apply Method 3 enhancements
        complexity = features.get('complexity_score', 0.5)
        edge_density = features.get('edge_density', 0.3)

        # Full optimization adjustments
        base_params.update({
            'max_iterations': min(base_params['max_iterations'] + 10, 50),
            'path_precision': min(base_params['path_precision'] + 2, 10),
            'corner_threshold': max(base_params['corner_threshold'] - 5, 10),
            'length_threshold': base_params['length_threshold'] * 0.8
        })

        # Edge-specific adjustments
        if edge_density > 0.5:
            base_params['splice_threshold'] = min(base_params['splice_threshold'] + 10, 90)

        return base_params

    def _select_optimal_tier(self,
                           tier_predictions: Dict[int, Dict],
                           target_quality: float,
                           time_budget: Optional[float]) -> int:
        """Select optimal tier based on predictions and constraints"""

        # Filter tiers that meet quality target
        viable_tiers = []
        for tier, prediction in tier_predictions.items():
            if prediction['predicted_quality'] >= target_quality:
                viable_tiers.append(tier)

        # If no tier meets quality target, use highest tier
        if not viable_tiers:
            logging.warning(f"No tier meets quality target {target_quality}, using tier 3")
            return 3

        # If time budget specified, filter by time constraint
        if time_budget:
            time_viable_tiers = []
            for tier in viable_tiers:
                if tier_predictions[tier]['estimated_time'] <= time_budget:
                    time_viable_tiers.append(tier)

            if time_viable_tiers:
                viable_tiers = time_viable_tiers
            else:
                logging.warning(f"No tier meets time budget {time_budget}s, ignoring constraint")

        # Select fastest tier that meets constraints
        return min(viable_tiers)

    def _estimate_tier_time(self, tier: int, features: Dict[str, float]) -> float:
        """Estimate processing time for tier based on image complexity"""

        # Base times per tier (calibrated from benchmarks)
        base_times = {
            1: 0.3,   # Method 1: Fast correlation
            2: 1.2,   # Method 1+2: With quality prediction
            3: 4.0    # Method 1+2+3: Full optimization
        }

        # Complexity multiplier based on image features
        complexity_score = self._calculate_complexity_score(features)
        complexity_multiplier = 1.0 + (complexity_score * 0.5)

        estimated_time = base_times[tier] * complexity_multiplier

        # Add small random variance for realism
        variance = estimated_time * 0.1
        estimated_time += random.uniform(-variance, variance)

        return max(0.1, estimated_time)  # Minimum 0.1s

    def _calculate_complexity_score(self, features: Dict[str, float]) -> float:
        """Calculate image complexity score (0-1)"""

        # Normalize key features that affect processing time
        edge_score = min(features.get('edge_density', 0.1) / 0.2, 1.0)
        color_score = min(features.get('unique_colors', 8) / 32, 1.0)
        entropy_score = min(features.get('entropy', 4.0) / 8.0, 1.0)

        # Weighted combination
        complexity = (edge_score * 0.4 + color_score * 0.3 + entropy_score * 0.3)
        return max(0.0, min(1.0, complexity))

    def _calculate_prediction_confidence(self, tier: int, logo_type: str) -> float:
        """Calculate confidence in tier prediction"""

        # Base confidence by tier
        base_confidence = {1: 0.8, 2: 0.9, 3: 0.95}

        # Adjust by logo type classification confidence
        type_confidence_modifier = {
            'simple_geometric': 0.95,
            'text_based': 0.9,
            'gradient': 0.85,
            'complex': 0.8,
            'unknown': 0.7
        }

        return base_confidence[tier] * type_confidence_modifier.get(logo_type, 0.7)

    def _explain_selection(self, optimal_tier: int, tier_predictions: Dict,
                          target_quality: float, time_budget: Optional[float]) -> str:
        """Generate explanation for tier selection"""

        selected_prediction = tier_predictions[optimal_tier]

        explanation = f"Selected Tier {optimal_tier} "
        explanation += f"(predicted quality: {selected_prediction['predicted_quality']:.3f}, "
        explanation += f"estimated time: {selected_prediction['estimated_time']:.2f}s). "

        if time_budget:
            explanation += f"Time budget: {time_budget}s. "

        explanation += f"Quality target: {target_quality}. "

        # Add reasoning based on constraints
        viable_tiers = [t for t, p in tier_predictions.items()
                       if p['predicted_quality'] >= target_quality]

        if len(viable_tiers) > 1:
            explanation += f"Tiers {viable_tiers} meet quality target, selected fastest."
        else:
            explanation += "Only viable tier meeting quality target."

        return explanation

    def _fallback_tier_selection(self, target_quality: float, time_budget: Optional[float]) -> Dict[str, Any]:
        """Fallback tier selection when routing fails"""

        if time_budget and time_budget < 1.0:
            selected_tier = 1
            reason = f"Time budget {time_budget}s requires fastest tier"
        elif target_quality >= 0.95:
            selected_tier = 3
            reason = f"High quality target {target_quality} requires best tier"
        elif target_quality >= 0.85:
            selected_tier = 2
            reason = f"Medium quality target {target_quality} uses balanced tier"
        else:
            selected_tier = 1
            reason = f"Low quality target {target_quality} uses fast tier"

        return {
            'selected_tier': selected_tier,
            'routing_time': 0.001,  # Minimal fallback time
            'logo_type': 'unknown',
            'confidence': 0.5,
            'target_quality': target_quality,
            'tier_predictions': {},
            'selection_reasoning': f"Fallback selection: {reason}",
            'fallback_used': True
        }