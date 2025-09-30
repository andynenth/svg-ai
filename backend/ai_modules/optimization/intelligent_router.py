#!/usr/bin/env python3
"""
Intelligent Routing System - ML-Based Method Selection
Routes optimization requests to the most effective method based on image characteristics,
system state, and historical performance data.
"""

import time
import json
import logging
import pickle
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import threading
from datetime import datetime, timedelta

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

# Local imports
from .base_optimizer import BaseOptimizer
from .feature_mapping import FeatureMappingOptimizer
from .regression_optimizer import RegressionBasedOptimizer
from .ppo_optimizer import PPOVTracerOptimizer as PPOOptimizer
from .performance_optimizer import Method1PerformanceOptimizer as PerformanceOptimizer
from .error_handler import OptimizationErrorHandler
from ..feature_extraction import ImageFeatureExtractor
from .resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Complete routing decision with confidence and reasoning"""
    primary_method: str
    fallback_methods: List[str]
    confidence: float
    reasoning: str
    estimated_time: float
    estimated_quality: float
    system_load_factor: float
    resource_availability: Dict[str, float]
    decision_timestamp: float
    cache_key: Optional[str] = None


@dataclass
class MethodPerformance:
    """Track performance metrics for each optimization method"""
    method_name: str
    success_count: int = 0
    failure_count: int = 0
    total_time: float = 0.0
    total_quality_improvement: float = 0.0
    avg_confidence: float = 0.0
    last_used: float = 0.0
    reliability_score: float = 1.0


class IntelligentRouter:
    """ML-based intelligent routing system for optimization method selection"""

    def __init__(self, model_path: Optional[str] = None, cache_size: int = 10000):
        """Initialize the intelligent routing system"""

        # Core ML model and data processing
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path or "/tmp/claude/intelligent_router_model.pkl"
        self.training_data = []
        self.feature_columns = []
        self.model_trained = False
        self.model_last_updated = 0.0

        # Method registry and performance tracking
        self.available_methods = {
            'feature_mapping': FeatureMappingOptimizer(),
            'regression': RegressionBasedOptimizer(),
            'ppo': PPOOptimizer(),
            'performance': PerformanceOptimizer()
        }

        self.method_performance = {
            name: MethodPerformance(method_name=name)
            for name in self.available_methods.keys()
        }

        # System monitoring and resource management
        self.resource_monitor = ResourceMonitor()
        self.feature_extractor = ImageFeatureExtractor()
        self.error_handler = OptimizationErrorHandler()

        # Decision caching for performance optimization
        self.decision_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

        # Routing history and analytics
        self.routing_history = []
        self.routing_analytics = {
            "total_decisions": 0,
            "method_selections": defaultdict(int),
            "confidence_history": [],
            "decision_times": [],
            "cache_hit_rate": 0.0
        }

        # User preferences and adaptive learning
        self.user_preferences = {
            "quality_weight": 0.7,
            "speed_weight": 0.3,
            "preferred_methods": [],
            "avoided_methods": []
        }

        # Thread safety
        self._lock = threading.RLock()

        # Initialize the system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize the routing system components"""
        try:
            # Load existing model if available
            if Path(self.model_path).exists():
                self._load_model()
                logger.info("Loaded existing routing model")
            else:
                # Initialize with basic model
                self._initialize_base_model()
                logger.info("Initialized new routing model")

            # Load historical performance data
            self._load_performance_history()

            logger.info("Intelligent routing system initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize routing system: {e}")
            self._initialize_base_model()

    def _initialize_base_model(self):
        """Initialize a basic RandomForest model"""
        self.model = CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            method='sigmoid',
            cv=3
        )

        # Define basic feature set
        self.feature_columns = [
            'complexity_score', 'unique_colors', 'edge_density', 'aspect_ratio',
            'file_size', 'image_area', 'color_variance', 'gradient_strength',
            'text_probability', 'geometric_score', 'system_load', 'memory_usage',
            'gpu_available', 'recent_failures', 'time_constraint'
        ]

        # Create synthetic training data for initial model
        self._create_initial_training_data()

    def _create_initial_training_data(self):
        """Create initial synthetic training data based on domain knowledge"""
        np.random.seed(42)

        methods = list(self.available_methods.keys())
        initial_data = []

        # Generate synthetic samples for each method's optimal scenarios
        for _ in range(200):  # 50 samples per method

            # Feature mapping - good for simple geometric logos
            features = {
                'complexity_score': np.random.uniform(0.1, 0.4),
                'unique_colors': np.random.randint(2, 6),
                'edge_density': np.random.uniform(0.1, 0.3),
                'aspect_ratio': np.random.uniform(0.8, 1.2),
                'file_size': np.random.uniform(1000, 10000),
                'image_area': np.random.uniform(10000, 100000),
                'color_variance': np.random.uniform(0.1, 0.3),
                'gradient_strength': np.random.uniform(0.0, 0.2),
                'text_probability': np.random.uniform(0.0, 0.3),
                'geometric_score': np.random.uniform(0.7, 1.0),
                'system_load': np.random.uniform(0.1, 0.6),
                'memory_usage': np.random.uniform(0.2, 0.7),
                'gpu_available': 1.0,
                'recent_failures': 0,
                'time_constraint': np.random.uniform(0.5, 1.0)
            }
            initial_data.append((features, 'feature_mapping', 0.95, 0.15))

            # Regression - good for text and medium complexity
            features = {
                'complexity_score': np.random.uniform(0.3, 0.7),
                'unique_colors': np.random.randint(4, 12),
                'edge_density': np.random.uniform(0.4, 0.8),
                'aspect_ratio': np.random.uniform(0.5, 2.0),
                'file_size': np.random.uniform(5000, 50000),
                'image_area': np.random.uniform(20000, 200000),
                'color_variance': np.random.uniform(0.3, 0.6),
                'gradient_strength': np.random.uniform(0.1, 0.4),
                'text_probability': np.random.uniform(0.4, 0.9),
                'geometric_score': np.random.uniform(0.3, 0.7),
                'system_load': np.random.uniform(0.2, 0.8),
                'memory_usage': np.random.uniform(0.3, 0.8),
                'gpu_available': np.random.choice([0.0, 1.0]),
                'recent_failures': np.random.randint(0, 2),
                'time_constraint': np.random.uniform(0.3, 0.8)
            }
            initial_data.append((features, 'regression', 0.90, 0.30))

            # PPO - good for complex logos with high quality requirements
            features = {
                'complexity_score': np.random.uniform(0.6, 1.0),
                'unique_colors': np.random.randint(8, 25),
                'edge_density': np.random.uniform(0.5, 1.0),
                'aspect_ratio': np.random.uniform(0.3, 3.0),
                'file_size': np.random.uniform(20000, 200000),
                'image_area': np.random.uniform(50000, 500000),
                'color_variance': np.random.uniform(0.5, 1.0),
                'gradient_strength': np.random.uniform(0.3, 1.0),
                'text_probability': np.random.uniform(0.0, 0.6),
                'geometric_score': np.random.uniform(0.1, 0.5),
                'system_load': np.random.uniform(0.1, 0.7),
                'memory_usage': np.random.uniform(0.4, 0.9),
                'gpu_available': 1.0,
                'recent_failures': 0,
                'time_constraint': np.random.uniform(0.1, 0.6)
            }
            initial_data.append((features, 'ppo', 0.85, 0.60))

            # Performance - good for high-load scenarios and speed requirements
            features = {
                'complexity_score': np.random.uniform(0.2, 0.8),
                'unique_colors': np.random.randint(3, 15),
                'edge_density': np.random.uniform(0.2, 0.6),
                'aspect_ratio': np.random.uniform(0.6, 1.6),
                'file_size': np.random.uniform(2000, 50000),
                'image_area': np.random.uniform(15000, 150000),
                'color_variance': np.random.uniform(0.2, 0.8),
                'gradient_strength': np.random.uniform(0.1, 0.5),
                'text_probability': np.random.uniform(0.0, 0.8),
                'geometric_score': np.random.uniform(0.2, 0.8),
                'system_load': np.random.uniform(0.7, 1.0),
                'memory_usage': np.random.uniform(0.7, 1.0),
                'gpu_available': np.random.choice([0.0, 1.0]),
                'recent_failures': np.random.randint(0, 3),
                'time_constraint': np.random.uniform(0.8, 1.0)
            }
            initial_data.append((features, 'performance', 0.80, 0.10))

        self.training_data = initial_data
        self._train_model()

    def route_optimization(self, image_path: str, features: Optional[Dict[str, Any]] = None,
                          quality_target: float = 0.85, time_constraint: float = 30.0,
                          user_preferences: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Main routing function - selects optimal optimization method

        Args:
            image_path: Path to the image to optimize
            features: Pre-extracted image features (optional)
            quality_target: Target quality score (0.0-1.0)
            time_constraint: Maximum time allowed (seconds)
            user_preferences: User-specific preferences

        Returns:
            RoutingDecision with method selection and reasoning
        """
        start_time = time.time()

        with self._lock:
            try:
                # Extract or use provided features
                if features is None:
                    features = self._extract_enhanced_features(image_path, quality_target, time_constraint)
                else:
                    features = self._enhance_features(features, quality_target, time_constraint)

                # Check cache first
                cache_key = self._generate_cache_key(features)
                cached_decision = self._get_cached_decision(cache_key)
                if cached_decision:
                    self.cache_hits += 1
                    logger.debug(f"Cache hit for routing decision")
                    return cached_decision

                self.cache_misses += 1

                # Update user preferences if provided
                if user_preferences:
                    self._update_user_preferences(user_preferences)

                # Get system state
                system_state = self._get_system_state()

                # Make ML-based decision
                if self.model_trained:
                    decision = self._ml_route_decision(features, system_state, quality_target, time_constraint)
                else:
                    decision = self._rule_based_fallback(features, system_state, quality_target, time_constraint)

                # Apply multi-criteria decision framework
                decision = self._apply_multi_criteria_framework(decision, features, system_state)

                # Add intelligent fallback strategies
                decision.fallback_methods = self._generate_fallback_strategies(
                    decision.primary_method, features, system_state
                )

                # Cache the decision
                decision.cache_key = cache_key
                self._cache_decision(cache_key, decision)

                # Record routing decision
                decision_time = time.time() - start_time
                self._record_routing_decision(decision, features, decision_time)

                logger.info(f"Routing decision made in {decision_time:.3f}s: {decision.primary_method} "
                           f"(confidence: {decision.confidence:.3f})")

                return decision

            except Exception as e:
                logger.error(f"Routing failed: {e}")
                return self._emergency_fallback(features or {})

    def _extract_enhanced_features(self, image_path: str, quality_target: float,
                                  time_constraint: float) -> Dict[str, Any]:
        """Extract comprehensive features for routing decision"""
        try:
            # Basic image features
            features = self.feature_extractor.extract_features(image_path)

            # Add system and context features
            features.update(self._enhance_features(features, quality_target, time_constraint))

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._get_default_features(quality_target, time_constraint)

    def _enhance_features(self, features: Dict[str, Any], quality_target: float,
                         time_constraint: float) -> Dict[str, Any]:
        """Enhance features with system state and context"""
        enhanced = features.copy()

        # Add system state
        system_state = self._get_system_state()
        enhanced.update(system_state)

        # Add context
        enhanced['quality_target'] = quality_target
        enhanced['time_constraint'] = time_constraint
        enhanced['time_constraint_normalized'] = min(time_constraint / 30.0, 1.0)

        # Add recent performance metrics
        enhanced['recent_failures'] = sum(
            perf.failure_count for perf in self.method_performance.values()
            if time.time() - perf.last_used < 3600  # last hour
        )

        return enhanced

    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for routing decisions"""
        try:
            resource_info = self.resource_monitor.get_current_resources()

            return {
                'system_load': resource_info.get('cpu_usage', 0.5),
                'memory_usage': resource_info.get('memory_usage', 0.5),
                'gpu_available': 1.0 if resource_info.get('gpu_available', False) else 0.0,
                'disk_io_load': resource_info.get('disk_io', 0.3),
                'network_load': resource_info.get('network_io', 0.2),
                'concurrent_jobs': resource_info.get('active_jobs', 0),
                'system_temperature': resource_info.get('temperature', 50.0) / 100.0
            }
        except Exception as e:
            logger.warning(f"Could not get system state: {e}")
            return {
                'system_load': 0.5, 'memory_usage': 0.5, 'gpu_available': 0.0,
                'disk_io_load': 0.3, 'network_load': 0.2, 'concurrent_jobs': 0,
                'system_temperature': 0.5
            }

    def _ml_route_decision(self, features: Dict[str, Any], system_state: Dict[str, Any],
                          quality_target: float, time_constraint: float) -> RoutingDecision:
        """Make ML-based routing decision"""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)

            # Get method probabilities
            method_probs = self.model.predict_proba([feature_vector])[0]
            method_names = list(self.available_methods.keys())

            # Calculate scores for each method
            method_scores = {}
            for i, method in enumerate(method_names):
                base_confidence = method_probs[i]

                # Adjust for system state and constraints
                adjusted_score = self._adjust_score_for_constraints(
                    method, base_confidence, features, system_state, quality_target, time_constraint
                )

                method_scores[method] = adjusted_score

            # Select best method
            best_method = max(method_scores.items(), key=lambda x: x[1])
            primary_method = best_method[0]
            confidence = best_method[1]

            # Estimate performance
            estimated_time, estimated_quality = self._estimate_performance(
                primary_method, features, system_state
            )

            # Generate reasoning
            reasoning = self._generate_reasoning(
                primary_method, confidence, features, system_state, method_scores
            )

            return RoutingDecision(
                primary_method=primary_method,
                fallback_methods=[],  # Will be filled later
                confidence=confidence,
                reasoning=reasoning,
                estimated_time=estimated_time,
                estimated_quality=estimated_quality,
                system_load_factor=system_state.get('system_load', 0.5),
                resource_availability=system_state,
                decision_timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"ML routing failed: {e}")
            return self._rule_based_fallback(features, system_state, quality_target, time_constraint)

    def _prepare_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector for ML model"""
        vector = []

        for col in self.feature_columns:
            value = features.get(col, 0.0)

            # Handle different data types
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            elif not isinstance(value, (int, float)):
                value = 0.0

            vector.append(float(value))

        return np.array(vector).reshape(1, -1)

    def _adjust_score_for_constraints(self, method: str, base_confidence: float,
                                    features: Dict[str, Any], system_state: Dict[str, Any],
                                    quality_target: float, time_constraint: float) -> float:
        """Adjust confidence score based on constraints and system state"""

        adjusted_score = base_confidence

        # Adjust for method reliability
        method_perf = self.method_performance[method]
        if method_perf.success_count + method_perf.failure_count > 0:
            reliability = method_perf.success_count / (method_perf.success_count + method_perf.failure_count)
            adjusted_score *= reliability

        # Adjust for system load
        system_load = system_state.get('system_load', 0.5)
        if method in ['ppo', 'regression'] and system_load > 0.8:
            adjusted_score *= 0.7  # Penalize resource-intensive methods under high load
        elif method in ['feature_mapping', 'performance'] and system_load > 0.8:
            adjusted_score *= 1.2  # Favor lightweight methods under high load

        # Adjust for GPU availability
        gpu_available = system_state.get('gpu_available', 0.0)
        if method == 'ppo' and gpu_available < 0.5:
            adjusted_score *= 0.5  # PPO needs GPU

        # Adjust for time constraints
        time_factor = min(time_constraint / 30.0, 1.0)
        if method in ['ppo', 'regression'] and time_factor < 0.3:
            adjusted_score *= 0.6  # Penalize slow methods for tight constraints
        elif method in ['feature_mapping', 'performance'] and time_factor < 0.3:
            adjusted_score *= 1.3  # Favor fast methods for tight constraints

        # Adjust for quality requirements
        if quality_target > 0.9 and method in ['ppo', 'regression']:
            adjusted_score *= 1.2  # Favor high-quality methods for high targets
        elif quality_target < 0.7 and method in ['feature_mapping', 'performance']:
            adjusted_score *= 1.2  # Favor fast methods for lower quality targets

        # Adjust for user preferences
        if method in self.user_preferences.get('preferred_methods', []):
            adjusted_score *= 1.15
        elif method in self.user_preferences.get('avoided_methods', []):
            adjusted_score *= 0.85

        return max(0.0, min(1.0, adjusted_score))

    def _estimate_performance(self, method: str, features: Dict[str, Any],
                            system_state: Dict[str, Any]) -> Tuple[float, float]:
        """Estimate time and quality for the selected method"""

        # Base estimates per method
        base_estimates = {
            'feature_mapping': (0.1, 0.85),
            'regression': (0.3, 0.88),
            'ppo': (0.6, 0.92),
            'performance': (0.05, 0.82)
        }

        base_time, base_quality = base_estimates.get(method, (0.2, 0.85))

        # Adjust for image complexity
        complexity = features.get('complexity_score', 0.5)
        time_multiplier = 1.0 + complexity * 2.0
        quality_adjustment = complexity * 0.1  # More complex images may have lower quality

        # Adjust for system load
        system_load = system_state.get('system_load', 0.5)
        time_multiplier *= (1.0 + system_load * 0.5)

        # Adjust for GPU availability
        if method == 'ppo' and system_state.get('gpu_available', 0.0) < 0.5:
            time_multiplier *= 3.0  # Much slower without GPU

        estimated_time = base_time * time_multiplier
        estimated_quality = max(0.5, base_quality - quality_adjustment)

        return estimated_time, estimated_quality

    def _generate_reasoning(self, method: str, confidence: float, features: Dict[str, Any],
                           system_state: Dict[str, Any], method_scores: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the routing decision"""

        reasons = []

        # Primary reason based on method characteristics
        method_reasons = {
            'feature_mapping': "optimal for simple geometric logos with low complexity",
            'regression': "best for text-based logos and medium complexity images",
            'ppo': "ideal for complex logos requiring highest quality optimization",
            'performance': "selected for speed optimization under high system load"
        }

        primary_reason = method_reasons.get(method, f"selected based on ML model prediction")
        reasons.append(f"Method '{method}' {primary_reason}")

        # Add confidence level
        if confidence > 0.9:
            reasons.append(f"high confidence ({confidence:.2f})")
        elif confidence > 0.7:
            reasons.append(f"moderate confidence ({confidence:.2f})")
        else:
            reasons.append(f"low confidence ({confidence:.2f})")

        # Add feature-based reasoning
        complexity = features.get('complexity_score', 0.5)
        if complexity < 0.3:
            reasons.append("low image complexity detected")
        elif complexity > 0.7:
            reasons.append("high image complexity detected")

        # Add system state reasoning
        system_load = system_state.get('system_load', 0.5)
        if system_load > 0.8:
            reasons.append("high system load detected")
        elif system_load < 0.3:
            reasons.append("low system load - can use intensive methods")

        # Add alternative methods consideration
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_methods) > 1:
            second_best = sorted_methods[1]
            score_diff = sorted_methods[0][1] - second_best[1]
            if score_diff < 0.1:
                reasons.append(f"close alternative: {second_best[0]} (score: {second_best[1]:.2f})")

        return "; ".join(reasons)

    def _apply_multi_criteria_framework(self, decision: RoutingDecision,
                                       features: Dict[str, Any],
                                       system_state: Dict[str, Any]) -> RoutingDecision:
        """Apply multi-criteria decision framework to refine the decision"""

        # Balance quality vs speed based on user preferences
        quality_weight = self.user_preferences.get('quality_weight', 0.7)
        speed_weight = self.user_preferences.get('speed_weight', 0.3)

        # Adjust confidence based on quality/speed trade-off
        if decision.estimated_time > features.get('time_constraint', 30.0):
            # Time constraint violation - reduce confidence
            time_penalty = min(0.3, (decision.estimated_time - features.get('time_constraint', 30.0)) / 30.0)
            decision.confidence *= (1.0 - time_penalty)
            decision.reasoning += f"; time constraint concern (penalty: {time_penalty:.2f})"

        # Consider method reliability
        method_perf = self.method_performance[decision.primary_method]
        if method_perf.reliability_score < 0.8:
            reliability_penalty = (0.8 - method_perf.reliability_score) * 0.5
            decision.confidence *= (1.0 - reliability_penalty)
            decision.reasoning += f"; reliability concern ({method_perf.reliability_score:.2f})"

        # Factor in recent success rates
        recent_failures = sum(
            perf.failure_count for perf in self.method_performance.values()
            if time.time() - perf.last_used < 1800  # last 30 minutes
        )

        if recent_failures > 3:
            decision.confidence *= 0.9
            decision.reasoning += "; recent system instability detected"

        return decision

    def _generate_fallback_strategies(self, primary_method: str, features: Dict[str, Any],
                                    system_state: Dict[str, Any]) -> List[str]:
        """Generate intelligent fallback strategies"""

        all_methods = list(self.available_methods.keys())
        fallbacks = []

        # Remove primary method from consideration
        available_fallbacks = [m for m in all_methods if m != primary_method]

        # Strategy 1: Performance-based fallback ordering
        method_performance_scores = {}
        for method in available_fallbacks:
            perf = self.method_performance[method]
            if perf.success_count + perf.failure_count > 0:
                reliability = perf.success_count / (perf.success_count + perf.failure_count)
                avg_quality = perf.total_quality_improvement / max(perf.success_count, 1)
                score = reliability * 0.6 + avg_quality * 0.4
            else:
                score = 0.5  # Default for unknown methods
            method_performance_scores[method] = score

        # Sort by performance score
        sorted_fallbacks = sorted(
            available_fallbacks,
            key=lambda m: method_performance_scores[m],
            reverse=True
        )

        # Strategy 2: Context-aware fallback selection
        complexity = features.get('complexity_score', 0.5)
        system_load = system_state.get('system_load', 0.5)

        # For high system load, prioritize lightweight methods
        if system_load > 0.8:
            lightweight_methods = ['feature_mapping', 'performance']
            for method in lightweight_methods:
                if method in sorted_fallbacks and method not in fallbacks:
                    fallbacks.append(method)

        # For complex images, ensure we have a high-quality fallback
        if complexity > 0.7:
            quality_methods = ['ppo', 'regression']
            for method in quality_methods:
                if method in sorted_fallbacks and method not in fallbacks:
                    fallbacks.append(method)

        # Add remaining methods by performance score
        for method in sorted_fallbacks:
            if method not in fallbacks:
                fallbacks.append(method)

        # Always ensure we have an emergency fallback
        if 'feature_mapping' not in fallbacks:
            fallbacks.append('feature_mapping')

        return fallbacks[:3]  # Limit to top 3 fallbacks

    def _rule_based_fallback(self, features: Dict[str, Any], system_state: Dict[str, Any],
                            quality_target: float, time_constraint: float) -> RoutingDecision:
        """Rule-based fallback when ML model is not available"""

        complexity = features.get('complexity_score', 0.5)
        unique_colors = features.get('unique_colors', 16)
        system_load = system_state.get('system_load', 0.5)
        gpu_available = system_state.get('gpu_available', 0.0)

        # Rule-based decision logic
        if system_load > 0.8 or time_constraint < 5.0:
            # High load or tight time constraints - use fast methods
            method = 'performance'
            confidence = 0.8
            reasoning = "rule-based: high system load or tight time constraint"

        elif complexity < 0.3 and unique_colors <= 6:
            # Simple geometric logos
            method = 'feature_mapping'
            confidence = 0.85
            reasoning = "rule-based: simple geometric logo detected"

        elif complexity > 0.7 and quality_target > 0.9 and gpu_available > 0.5:
            # Complex logos with high quality requirements and GPU available
            method = 'ppo'
            confidence = 0.75
            reasoning = "rule-based: complex logo with high quality target and GPU available"

        else:
            # Default to regression for medium complexity
            method = 'regression'
            confidence = 0.7
            reasoning = "rule-based: medium complexity default"

        estimated_time, estimated_quality = self._estimate_performance(method, features, system_state)

        return RoutingDecision(
            primary_method=method,
            fallback_methods=[],
            confidence=confidence,
            reasoning=reasoning,
            estimated_time=estimated_time,
            estimated_quality=estimated_quality,
            system_load_factor=system_load,
            resource_availability=system_state,
            decision_timestamp=time.time()
        )

    def _emergency_fallback(self, features: Dict[str, Any]) -> RoutingDecision:
        """Emergency fallback when all else fails"""
        return RoutingDecision(
            primary_method='feature_mapping',  # Most reliable method
            fallback_methods=['performance'],  # Fastest backup
            confidence=0.5,
            reasoning="emergency fallback: system error occurred",
            estimated_time=0.2,
            estimated_quality=0.8,
            system_load_factor=0.5,
            resource_availability={},
            decision_timestamp=time.time()
        )

    def _generate_cache_key(self, features: Dict[str, Any]) -> str:
        """Generate cache key for routing decisions"""
        # Use key features that affect routing decisions
        key_features = {
            'complexity': round(features.get('complexity_score', 0.5), 2),
            'colors': features.get('unique_colors', 16),
            'edges': round(features.get('edge_density', 0.1), 2),
            'aspect': round(features.get('aspect_ratio', 1.0), 1),
            'size_class': 'small' if features.get('file_size', 10000) < 10000 else 'large',
            'system_load_class': 'high' if features.get('system_load', 0.5) > 0.7 else 'normal',
            'quality_target': round(features.get('quality_target', 0.85), 1),
            'time_class': 'urgent' if features.get('time_constraint', 30.0) < 10.0 else 'normal'
        }

        key_string = json.dumps(key_features, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def _get_cached_decision(self, cache_key: str) -> Optional[RoutingDecision]:
        """Retrieve cached routing decision if available and valid"""
        if cache_key in self.decision_cache:
            cached_decision, timestamp = self.decision_cache[cache_key]

            # Check if cache entry is still valid (1 hour expiration)
            if time.time() - timestamp < 3600:
                # Update timestamp for recent access
                cached_decision.decision_timestamp = time.time()
                return cached_decision
            else:
                # Remove expired entry
                del self.decision_cache[cache_key]

        return None

    def _cache_decision(self, cache_key: str, decision: RoutingDecision):
        """Cache routing decision for future use"""
        # Implement LRU cache behavior
        if len(self.decision_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.decision_cache.keys(),
                key=lambda k: self.decision_cache[k][1]
            )
            del self.decision_cache[oldest_key]

        self.decision_cache[cache_key] = (decision, time.time())

    def _record_routing_decision(self, decision: RoutingDecision, features: Dict[str, Any],
                               decision_time: float):
        """Record routing decision for analytics and learning"""

        self.routing_analytics['total_decisions'] += 1
        self.routing_analytics['method_selections'][decision.primary_method] += 1
        self.routing_analytics['confidence_history'].append(decision.confidence)
        self.routing_analytics['decision_times'].append(decision_time)

        # Update cache hit rate
        total_requests = self.cache_hits + self.cache_misses
        if total_requests > 0:
            self.routing_analytics['cache_hit_rate'] = self.cache_hits / total_requests

        # Store for training data
        self.routing_history.append({
            'timestamp': decision.decision_timestamp,
            'features': features,
            'decision': asdict(decision),
            'decision_time': decision_time
        })

        # Trigger model retraining if we have enough new data
        if len(self.routing_history) % 100 == 0:
            self._retrain_model_async()

    def record_optimization_result(self, decision: RoutingDecision, success: bool,
                                 actual_time: float, actual_quality: float):
        """Record the actual optimization result for learning"""

        with self._lock:
            method_perf = self.method_performance[decision.primary_method]
            method_perf.last_used = time.time()

            if success:
                method_perf.success_count += 1
                method_perf.total_time += actual_time
                method_perf.total_quality_improvement += actual_quality

                # Update reliability score (exponential moving average)
                method_perf.reliability_score = (
                    method_perf.reliability_score * 0.9 +
                    1.0 * 0.1
                )
            else:
                method_perf.failure_count += 1

                # Decrease reliability score
                method_perf.reliability_score = (
                    method_perf.reliability_score * 0.9 +
                    0.0 * 0.1
                )

            # Add to training data for model improvement
            if len(self.routing_history) > 0:
                # Find corresponding routing decision
                for record in reversed(self.routing_history[-10:]):  # Check last 10 decisions
                    if abs(record['timestamp'] - decision.decision_timestamp) < 1.0:
                        record['actual_result'] = {
                            'success': success,
                            'actual_time': actual_time,
                            'actual_quality': actual_quality,
                            'quality_delta': actual_quality - decision.estimated_quality,
                            'time_delta': actual_time - decision.estimated_time
                        }
                        break

            logger.info(f"Recorded result for {decision.primary_method}: "
                       f"success={success}, time={actual_time:.3f}s, quality={actual_quality:.3f}")

    def _retrain_model_async(self):
        """Asynchronously retrain the ML model with new data"""
        try:
            # Only retrain if enough time has passed since last training
            if time.time() - self.model_last_updated < 3600:  # 1 hour minimum
                return

            # Prepare training data from routing history with actual results
            training_features = []
            training_labels = []

            for record in self.routing_history:
                if 'actual_result' in record and record['actual_result']['success']:
                    features = record['features']
                    method = record['decision']['primary_method']

                    # Add this as a positive example
                    feature_vector = []
                    for col in self.feature_columns:
                        feature_vector.append(features.get(col, 0.0))

                    training_features.append(feature_vector)
                    training_labels.append(method)

            # Only retrain if we have sufficient new data
            if len(training_features) >= 50:
                self._train_model_with_data(training_features, training_labels)
                self.model_last_updated = time.time()
                logger.info(f"Model retrained with {len(training_features)} samples")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

    def _train_model_with_data(self, features: List[List[float]], labels: List[str]):
        """Train the ML model with provided data"""
        try:
            X = np.array(features)
            y = np.array(labels)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train the model
            self.model.fit(X_scaled, y)
            self.model_trained = True

            # Evaluate model performance
            if len(set(y)) > 1 and len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )

                if len(X_test) > 0:
                    self.model.fit(X_train, y_train)
                    y_pred = self.model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    logger.info(f"Model accuracy: {accuracy:.3f}")

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self.model_trained = False

    def _train_model(self):
        """Train the ML model with current training data"""
        if not self.training_data:
            logger.warning("No training data available")
            return

        try:
            # Prepare training data
            features = []
            labels = []

            for feature_dict, method, confidence, time_est in self.training_data:
                feature_vector = []
                for col in self.feature_columns:
                    feature_vector.append(feature_dict.get(col, 0.0))

                features.append(feature_vector)
                labels.append(method)

            self._train_model_with_data(features, labels)

        except Exception as e:
            logger.error(f"Initial model training failed: {e}")
            self.model_trained = False

    def _load_model(self):
        """Load existing model from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.method_performance = model_data.get('method_performance', self.method_performance)
            self.model_trained = True
            self.model_last_updated = model_data.get('last_updated', time.time())

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._initialize_base_model()

    def save_model(self):
        """Save current model to disk"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'method_performance': self.method_performance,
                'last_updated': time.time()
            }

            # Ensure directory exists
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _load_performance_history(self):
        """Load historical performance data"""
        try:
            history_path = self.model_path.replace('.pkl', '_history.json')
            if Path(history_path).exists():
                with open(history_path, 'r') as f:
                    history_data = json.load(f)

                self.routing_history = history_data.get('routing_history', [])
                self.routing_analytics = history_data.get('routing_analytics', self.routing_analytics)

                logger.info(f"Loaded {len(self.routing_history)} historical routing decisions")

        except Exception as e:
            logger.warning(f"Could not load performance history: {e}")

    def save_performance_history(self):
        """Save performance history to disk"""
        try:
            history_path = self.model_path.replace('.pkl', '_history.json')

            # Keep only recent history (last 1000 decisions)
            recent_history = self.routing_history[-1000:] if len(self.routing_history) > 1000 else self.routing_history

            history_data = {
                'routing_history': recent_history,
                'routing_analytics': self.routing_analytics,
                'last_saved': time.time()
            }

            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)

            logger.info(f"Performance history saved to {history_path}")

        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")

    def _update_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences for adaptive learning"""
        self.user_preferences.update(preferences)

        # Validate and constrain preferences
        if 'quality_weight' in preferences and 'speed_weight' in preferences:
            total_weight = preferences['quality_weight'] + preferences['speed_weight']
            if total_weight > 0:
                self.user_preferences['quality_weight'] = preferences['quality_weight'] / total_weight
                self.user_preferences['speed_weight'] = preferences['speed_weight'] / total_weight

    def _get_default_features(self, quality_target: float, time_constraint: float) -> Dict[str, Any]:
        """Get default features when extraction fails"""
        return {
            'complexity_score': 0.5,
            'unique_colors': 8,
            'edge_density': 0.3,
            'aspect_ratio': 1.0,
            'file_size': 10000,
            'image_area': 50000,
            'color_variance': 0.4,
            'gradient_strength': 0.2,
            'text_probability': 0.3,
            'geometric_score': 0.5,
            'quality_target': quality_target,
            'time_constraint': time_constraint,
            'time_constraint_normalized': min(time_constraint / 30.0, 1.0),
            'recent_failures': 0
        }

    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics"""
        with self._lock:
            analytics = self.routing_analytics.copy()

            # Add method performance statistics
            method_stats = {}
            for method, perf in self.method_performance.items():
                total_attempts = perf.success_count + perf.failure_count
                method_stats[method] = {
                    'success_rate': perf.success_count / max(total_attempts, 1),
                    'avg_time': perf.total_time / max(perf.success_count, 1),
                    'avg_quality': perf.total_quality_improvement / max(perf.success_count, 1),
                    'reliability_score': perf.reliability_score,
                    'total_uses': total_attempts,
                    'last_used': perf.last_used
                }

            analytics['method_performance'] = method_stats

            # Add cache statistics
            total_requests = self.cache_hits + self.cache_misses
            analytics['cache_statistics'] = {
                'hit_rate': self.cache_hits / max(total_requests, 1),
                'total_requests': total_requests,
                'cache_size': len(self.decision_cache)
            }

            # Add model statistics
            analytics['model_status'] = {
                'trained': self.model_trained,
                'last_updated': self.model_last_updated,
                'training_samples': len(self.training_data),
                'feature_count': len(self.feature_columns)
            }

            return analytics

    def optimize_routing_performance(self):
        """Optimize routing performance and cleanup"""
        with self._lock:
            try:
                # Clean up old cache entries
                current_time = time.time()
                expired_keys = []

                for cache_key, (decision, timestamp) in self.decision_cache.items():
                    if current_time - timestamp > 3600:  # 1 hour expiration
                        expired_keys.append(cache_key)

                for key in expired_keys:
                    del self.decision_cache[key]

                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

                # Pre-compute common scenarios if system is idle
                system_state = self._get_system_state()
                if system_state.get('system_load', 1.0) < 0.3:
                    self._precompute_common_scenarios()

                # Save model and history periodically
                if current_time - self.model_last_updated > 7200:  # 2 hours
                    self.save_model()
                    self.save_performance_history()

            except Exception as e:
                logger.error(f"Performance optimization failed: {e}")

    def _precompute_common_scenarios(self):
        """Pre-compute routing decisions for common scenarios"""
        try:
            common_scenarios = [
                # Simple logos
                {'complexity_score': 0.2, 'unique_colors': 3, 'edge_density': 0.2},
                # Text logos
                {'complexity_score': 0.4, 'unique_colors': 2, 'edge_density': 0.7},
                # Complex logos
                {'complexity_score': 0.8, 'unique_colors': 15, 'edge_density': 0.6},
                # Gradient logos
                {'complexity_score': 0.6, 'unique_colors': 20, 'edge_density': 0.3}
            ]

            precomputed_count = 0
            for scenario in common_scenarios:
                # Add default system state
                enhanced_scenario = self._enhance_features(scenario, 0.85, 30.0)
                cache_key = self._generate_cache_key(enhanced_scenario)

                if cache_key not in self.decision_cache:
                    # Generate decision and cache it
                    if self.model_trained:
                        decision = self._ml_route_decision(enhanced_scenario, self._get_system_state(), 0.85, 30.0)
                    else:
                        decision = self._rule_based_fallback(enhanced_scenario, self._get_system_state(), 0.85, 30.0)

                    self._cache_decision(cache_key, decision)
                    precomputed_count += 1

            if precomputed_count > 0:
                logger.info(f"Pre-computed {precomputed_count} common routing scenarios")

        except Exception as e:
            logger.error(f"Pre-computation failed: {e}")

    def get_method_recommendation(self, image_features: Dict[str, Any]) -> str:
        """Get a quick method recommendation without full routing"""
        try:
            decision = self.route_optimization("", features=image_features)
            return decision.primary_method
        except Exception as e:
            logger.error(f"Method recommendation failed: {e}")
            return 'feature_mapping'  # Safe default

    def shutdown(self):
        """Gracefully shutdown the routing system"""
        logger.info("Shutting down intelligent routing system...")

        try:
            # Save current state
            self.save_model()
            self.save_performance_history()

            # Clear caches
            self.decision_cache.clear()

            logger.info("Intelligent routing system shutdown complete")

        except Exception as e:
            logger.error(f"Shutdown error: {e}")


# Factory function for easy instantiation
def create_intelligent_router(model_path: Optional[str] = None,
                            cache_size: int = 10000) -> IntelligentRouter:
    """Create and initialize an intelligent router instance"""
    return IntelligentRouter(model_path=model_path, cache_size=cache_size)


# Usage example and testing
if __name__ == "__main__":
    # Example usage
    router = create_intelligent_router()

    # Example routing decision
    test_features = {
        'complexity_score': 0.3,
        'unique_colors': 4,
        'edge_density': 0.2,
        'aspect_ratio': 1.0,
        'file_size': 5000
    }

    decision = router.route_optimization(
        image_path="test_image.png",
        features=test_features,
        quality_target=0.9,
        time_constraint=15.0
    )

    print(f"Routing Decision:")
    print(f"  Primary Method: {decision.primary_method}")
    print(f"  Confidence: {decision.confidence:.3f}")
    print(f"  Estimated Time: {decision.estimated_time:.3f}s")
    print(f"  Estimated Quality: {decision.estimated_quality:.3f}")
    print(f"  Reasoning: {decision.reasoning}")
    print(f"  Fallbacks: {decision.fallback_methods}")