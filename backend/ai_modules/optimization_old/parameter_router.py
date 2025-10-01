#!/usr/bin/env python3
"""
Intelligent Parameter Router - Method 1 Integration
Routes images to optimal optimization methods based on characteristics
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

from .feature_mapping import FeatureMappingOptimizer
from ..feature_extraction import ImageFeatureExtractor
from .error_handler import OptimizationErrorHandler


class OptimizationMethod(Enum):
    METHOD_1_CORRELATION = "method_1_correlation"
    DEFAULT_PARAMETERS = "default_parameters"
    CONSERVATIVE_FALLBACK = "conservative_fallback"


@dataclass
class RoutingDecision:
    """Structure for optimization routing decision"""
    method: OptimizationMethod
    confidence: float
    reasoning: str
    expected_improvement: float
    processing_time_estimate: float


class ParameterRouter:
    """Route images to optimal optimization method based on characteristics"""

    def __init__(self):
        # Core components
        self.method1_optimizer = FeatureMappingOptimizer()
        self.feature_extractor = ImageFeatureExtractor()
        self.error_handler = OptimizationErrorHandler()

        # Routing history and analytics
        self.routing_history = []
        self.routing_analytics = {
            "total_routes": 0,
            "method_usage": {method.value: 0 for method in OptimizationMethod},
            "success_rates": {method.value: {"attempts": 0, "successes": 0} for method in OptimizationMethod},
            "average_improvements": {method.value: [] for method in OptimizationMethod},
            "processing_times": {method.value: [] for method in OptimizationMethod}
        }

        # Routing configuration
        self.routing_rules = {
            "feature_confidence_threshold": 0.7,
            "method1_confidence_threshold": 0.6,
            "complexity_threshold": 0.8,
            "edge_density_threshold": 0.5,
            "unique_colors_threshold": 0.6,
            "entropy_threshold": 0.7,
            "enable_adaptive_routing": True,
            "enable_ab_testing": False,
            "ab_test_split": 0.5
        }

        # Performance tracking
        self.performance_thresholds = {
            "speed_priority_time_limit": 0.05,  # 50ms for speed priority
            "quality_priority_improvement_target": 0.2,  # 20% improvement for quality
            "balanced_time_limit": 0.1,  # 100ms for balanced priority
            "balanced_improvement_target": 0.15  # 15% improvement for balanced
        }

        # A/B testing framework
        self.ab_testing = {
            "enabled": False,
            "experiments": {},
            "current_split": 0.5
        }

        # Logger
        self.logger = logging.getLogger(__name__)

        # Initialize conservative fallback systems
        self.__init_conservative_fallback_parameters()

        self.logger.info("Parameter router initialized with fallback and recovery systems")

    def route_optimization(self, image_path: str, features: Dict[str, float],
                          requirements: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """Determine optimal optimization method for image"""
        start_time = time.time()

        try:
            # Default requirements
            if requirements is None:
                requirements = {"speed_priority": "balanced", "quality_target": 0.85}

            # Extract additional context
            context = {
                "image_path": image_path,
                "features": features,
                "requirements": requirements,
                "timestamp": time.time()
            }

            # Perform routing decision
            decision = self._make_routing_decision(features, requirements, context)

            # Track routing decision
            self._track_routing_decision(decision, context, time.time() - start_time)

            self.logger.info(f"Routing decision: {decision.method.value} "
                           f"(confidence: {decision.confidence:.3f}, "
                           f"reasoning: {decision.reasoning})")

            return decision

        except Exception as e:
            # Fallback routing decision
            self.logger.error(f"Routing failed: {e}")
            return RoutingDecision(
                method=OptimizationMethod.CONSERVATIVE_FALLBACK,
                confidence=0.5,
                reasoning=f"Routing error: {str(e)}",
                expected_improvement=0.1,
                processing_time_estimate=0.1
            )

    def _make_routing_decision(self, features: Dict[str, float],
                              requirements: Dict[str, Any],
                              context: Dict[str, Any]) -> RoutingDecision:
        """Core routing logic implementation"""

        # Extract key features
        edge_density = features.get("edge_density", 0.5)
        unique_colors = features.get("unique_colors", 0.5)
        entropy = features.get("entropy", 0.5)
        complexity_score = features.get("complexity_score", 0.5)
        gradient_strength = features.get("gradient_strength", 0.5)
        corner_density = features.get("corner_density", 0.5)

        # Extract requirements
        speed_priority = requirements.get("speed_priority", "balanced")
        quality_target = requirements.get("quality_target", 0.85)

        # Calculate feature quality/confidence
        feature_confidence = self._calculate_feature_confidence(features)

        # Logo type inference for routing
        logo_type = self._infer_logo_type_for_routing(features)

        # Performance-based routing
        if speed_priority == "fast":
            return self._route_for_speed(features, feature_confidence, logo_type)
        elif speed_priority == "quality":
            return self._route_for_quality(features, feature_confidence, logo_type, quality_target)
        else:  # balanced
            return self._route_for_balanced(features, feature_confidence, logo_type, quality_target)

    def _route_for_speed(self, features: Dict[str, float],
                        feature_confidence: float, logo_type: str) -> RoutingDecision:
        """Route with speed priority"""

        # High confidence features with simple types → Method 1 (fast)
        if feature_confidence >= 0.8 and logo_type in ["simple", "text"]:
            return RoutingDecision(
                method=OptimizationMethod.METHOD_1_CORRELATION,
                confidence=0.9,
                reasoning=f"High confidence {logo_type} logo with speed priority",
                expected_improvement=0.12,
                processing_time_estimate=0.02
            )

        # Medium confidence → Conservative approach for speed
        elif feature_confidence >= 0.6:
            return RoutingDecision(
                method=OptimizationMethod.CONSERVATIVE_FALLBACK,
                confidence=0.7,
                reasoning="Medium confidence with speed priority - using conservative approach",
                expected_improvement=0.08,
                processing_time_estimate=0.01
            )

        # Low confidence → Default parameters (fastest)
        else:
            return RoutingDecision(
                method=OptimizationMethod.DEFAULT_PARAMETERS,
                confidence=0.5,
                reasoning="Low confidence with speed priority - using default parameters",
                expected_improvement=0.05,
                processing_time_estimate=0.005
            )

    def _route_for_quality(self, features: Dict[str, float],
                          feature_confidence: float, logo_type: str,
                          quality_target: float) -> RoutingDecision:
        """Route with quality priority"""

        # Any decent confidence → Method 1 for best quality
        if feature_confidence >= 0.5:
            # Estimate improvement based on logo type and complexity
            complexity = features.get("complexity_score", 0.5)
            expected_improvement = 0.15 + (complexity * 0.1)  # More complex = more improvement potential

            return RoutingDecision(
                method=OptimizationMethod.METHOD_1_CORRELATION,
                confidence=feature_confidence,
                reasoning=f"Quality priority for {logo_type} logo - using Method 1",
                expected_improvement=expected_improvement,
                processing_time_estimate=0.08
            )

        # Very low confidence → Conservative fallback
        else:
            return RoutingDecision(
                method=OptimizationMethod.CONSERVATIVE_FALLBACK,
                confidence=0.6,
                reasoning="Very low feature confidence - conservative approach for quality",
                expected_improvement=0.10,
                processing_time_estimate=0.05
            )

    def _route_for_balanced(self, features: Dict[str, float],
                           feature_confidence: float, logo_type: str,
                           quality_target: float) -> RoutingDecision:
        """Route with balanced speed/quality priority"""

        # High confidence → Method 1
        if feature_confidence >= self.routing_rules["method1_confidence_threshold"]:
            return RoutingDecision(
                method=OptimizationMethod.METHOD_1_CORRELATION,
                confidence=feature_confidence,
                reasoning=f"High confidence {logo_type} logo - balanced optimization",
                expected_improvement=0.15,
                processing_time_estimate=0.05
            )

        # Medium confidence with simple types → Method 1
        elif feature_confidence >= 0.5 and logo_type in ["simple", "text"]:
            return RoutingDecision(
                method=OptimizationMethod.METHOD_1_CORRELATION,
                confidence=feature_confidence,
                reasoning=f"Medium confidence {logo_type} logo suitable for Method 1",
                expected_improvement=0.12,
                processing_time_estimate=0.06
            )

        # Complex images with medium confidence → Conservative
        elif logo_type in ["complex", "gradient"] and feature_confidence >= 0.4:
            return RoutingDecision(
                method=OptimizationMethod.CONSERVATIVE_FALLBACK,
                confidence=0.7,
                reasoning=f"Complex {logo_type} logo - conservative approach",
                expected_improvement=0.10,
                processing_time_estimate=0.03
            )

        # Low confidence → Default parameters
        else:
            return RoutingDecision(
                method=OptimizationMethod.DEFAULT_PARAMETERS,
                confidence=0.5,
                reasoning="Low confidence features - using default parameters",
                expected_improvement=0.05,
                processing_time_estimate=0.01
            )

    def _calculate_feature_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in extracted features"""
        try:
            # Check for valid feature ranges
            valid_features = 0
            total_features = 0

            expected_features = [
                "edge_density", "unique_colors", "entropy",
                "corner_density", "gradient_strength", "complexity_score"
            ]

            for feature_name in expected_features:
                total_features += 1
                value = features.get(feature_name, -1)

                # Valid if in [0, 1] range and not extreme values
                if 0.0 <= value <= 1.0 and not (value == 0.0 or value == 1.0):
                    valid_features += 1

            base_confidence = valid_features / total_features if total_features > 0 else 0.0

            # Adjust based on feature diversity
            feature_values = [features.get(f, 0.5) for f in expected_features]
            std_dev = self._calculate_std_dev(feature_values)

            # Higher standard deviation indicates more diverse/informative features
            diversity_bonus = min(0.2, std_dev * 0.5)

            final_confidence = min(1.0, base_confidence + diversity_bonus)

            return final_confidence

        except Exception:
            return 0.5  # Default confidence

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of values"""
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _infer_logo_type_for_routing(self, features: Dict[str, float]) -> str:
        """Infer logo type for routing decisions"""
        edge_density = features.get("edge_density", 0.5)
        unique_colors = features.get("unique_colors", 0.5)
        entropy = features.get("entropy", 0.5)
        complexity = features.get("complexity_score", 0.5)
        gradient_strength = features.get("gradient_strength", 0.5)

        # More sophisticated classification than simple heuristics

        # Simple geometric logos
        if (complexity < 0.3 and edge_density < 0.3 and
            unique_colors < 0.4 and gradient_strength < 0.3):
            return "simple"

        # Text-based logos
        elif (entropy > 0.7 and edge_density > 0.3 and
              unique_colors < 0.3 and complexity < 0.5):
            return "text"

        # Gradient logos
        elif (gradient_strength > 0.5 or
              (unique_colors > 0.6 and complexity < 0.7)):
            return "gradient"

        # Complex logos (default)
        else:
            return "complex"

    def _track_routing_decision(self, decision: RoutingDecision,
                               context: Dict[str, Any], routing_time: float):
        """Track routing decision for analytics"""
        try:
            # Create routing record
            record = {
                "timestamp": time.time(),
                "decision": asdict(decision),
                "context": context,
                "routing_time": routing_time
            }

            # Add to history
            self.routing_history.append(record)

            # Limit history size
            if len(self.routing_history) > 1000:
                self.routing_history = self.routing_history[-500:]

            # Update analytics
            self.routing_analytics["total_routes"] += 1
            self.routing_analytics["method_usage"][decision.method.value] += 1
            self.routing_analytics["processing_times"][decision.method.value].append(routing_time)

            # Update success rates (will be updated after optimization completes)
            self.routing_analytics["success_rates"][decision.method.value]["attempts"] += 1

        except Exception as e:
            self.logger.warning(f"Failed to track routing decision: {e}")

    def update_routing_success(self, decision: RoutingDecision,
                              actual_improvement: float, actual_time: float,
                              success: bool = True):
        """Update routing analytics with actual results"""
        try:
            method_name = decision.method.value

            if success:
                self.routing_analytics["success_rates"][method_name]["successes"] += 1
                self.routing_analytics["average_improvements"][method_name].append(actual_improvement)

            # Track actual processing time
            if actual_time > 0:
                # Replace estimate with actual time in recent records
                for record in reversed(self.routing_history[-10:]):
                    if record["decision"]["method"] == method_name:
                        record["actual_time"] = actual_time
                        record["actual_improvement"] = actual_improvement
                        record["success"] = success
                        break

        except Exception as e:
            self.logger.warning(f"Failed to update routing success: {e}")

    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get comprehensive routing analytics"""
        analytics = self.routing_analytics.copy()

        # Calculate derived statistics
        for method_name in OptimizationMethod:
            method_val = method_name.value
            attempts = analytics["success_rates"][method_val]["attempts"]
            successes = analytics["success_rates"][method_val]["successes"]

            # Success rate
            analytics["success_rates"][method_val]["rate"] = (
                successes / attempts if attempts > 0 else 0.0
            )

            # Average improvement
            improvements = analytics["average_improvements"][method_val]
            analytics["average_improvements"][method_val] = {
                "mean": sum(improvements) / len(improvements) if improvements else 0.0,
                "count": len(improvements),
                "recent": improvements[-10:] if improvements else []
            }

            # Average processing time
            times = analytics["processing_times"][method_val]
            analytics["processing_times"][method_val] = {
                "mean": sum(times) / len(times) if times else 0.0,
                "count": len(times),
                "recent": times[-10:] if times else []
            }

        return analytics

    def configure_routing_rules(self, **rules):
        """Update routing rule configuration"""
        for key, value in rules.items():
            if key in self.routing_rules:
                old_value = self.routing_rules[key]
                self.routing_rules[key] = value
                self.logger.info(f"Routing rule updated: {key} {old_value} → {value}")
            else:
                self.logger.warning(f"Unknown routing rule: {key}")

    def enable_ab_testing(self, experiment_name: str, split_ratio: float = 0.5):
        """Enable A/B testing for routing strategies"""
        self.ab_testing["enabled"] = True
        self.ab_testing["current_split"] = split_ratio
        self.ab_testing["experiments"][experiment_name] = {
            "split_ratio": split_ratio,
            "start_time": time.time(),
            "group_a_results": [],
            "group_b_results": []
        }

        self.routing_rules["enable_ab_testing"] = True
        self.routing_rules["ab_test_split"] = split_ratio

        self.logger.info(f"A/B testing enabled: {experiment_name} (split: {split_ratio})")

    def disable_ab_testing(self):
        """Disable A/B testing"""
        self.ab_testing["enabled"] = False
        self.routing_rules["enable_ab_testing"] = False
        self.logger.info("A/B testing disabled")

    def get_ab_test_results(self, experiment_name: str) -> Dict[str, Any]:
        """Get A/B testing results for analysis"""
        if experiment_name not in self.ab_testing["experiments"]:
            return {"error": "Experiment not found"}

        experiment = self.ab_testing["experiments"][experiment_name]

        group_a = experiment["group_a_results"]
        group_b = experiment["group_b_results"]

        def calculate_stats(results):
            if not results:
                return {"count": 0, "mean_improvement": 0, "mean_time": 0}

            improvements = [r.get("improvement", 0) for r in results]
            times = [r.get("time", 0) for r in results]

            return {
                "count": len(results),
                "mean_improvement": sum(improvements) / len(improvements),
                "mean_time": sum(times) / len(times),
                "success_rate": sum(1 for r in results if r.get("success", False)) / len(results)
            }

        return {
            "experiment_name": experiment_name,
            "duration": time.time() - experiment["start_time"],
            "group_a": calculate_stats(group_a),
            "group_b": calculate_stats(group_b),
            "split_ratio": experiment["split_ratio"]
        }

    def create_routing_diagnostic_report(self, output_path: str = None) -> str:
        """Generate comprehensive routing diagnostic report"""
        try:
            analytics = self.get_routing_analytics()

            report = {
                "routing_summary": {
                    "total_routes": analytics["total_routes"],
                    "method_distribution": analytics["method_usage"],
                    "timestamp": time.time()
                },
                "performance_metrics": {
                    "success_rates": analytics["success_rates"],
                    "average_improvements": analytics["average_improvements"],
                    "processing_times": analytics["processing_times"]
                },
                "routing_rules": self.routing_rules.copy(),
                "ab_testing_status": self.ab_testing.copy(),
                "recent_routing_history": self.routing_history[-20:] if self.routing_history else []
            }

            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                self.logger.info(f"Routing diagnostic report saved to: {output_path}")
                return output_path
            else:
                return json.dumps(report, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to generate routing diagnostic report: {e}")
            return f"Error generating report: {e}"

    def validate_routing_decision(self, decision: RoutingDecision,
                                 actual_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate routing decision against actual results"""
        try:
            actual_improvement = actual_result.get("improvement", 0)
            actual_time = actual_result.get("processing_time", 0)
            actual_success = actual_result.get("success", False)

            # Calculate prediction accuracy
            improvement_accuracy = 1.0 - abs(decision.expected_improvement - actual_improvement) / max(0.01, decision.expected_improvement)
            time_accuracy = 1.0 - abs(decision.processing_time_estimate - actual_time) / max(0.01, decision.processing_time_estimate)

            # Overall routing accuracy
            routing_accuracy = (improvement_accuracy + time_accuracy) / 2

            validation_result = {
                "routing_accuracy": max(0.0, routing_accuracy),
                "improvement_prediction_accuracy": max(0.0, improvement_accuracy),
                "time_prediction_accuracy": max(0.0, time_accuracy),
                "decision_appropriate": actual_success and actual_improvement >= decision.expected_improvement * 0.7,
                "predicted_vs_actual": {
                    "improvement": {"predicted": decision.expected_improvement, "actual": actual_improvement},
                    "time": {"predicted": decision.processing_time_estimate, "actual": actual_time},
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning
                }
            }

            # Update routing success tracking
            self.update_routing_success(decision, actual_improvement, actual_time, actual_success)

            return validation_result

        except Exception as e:
            self.logger.error(f"Failed to validate routing decision: {e}")
            return {"error": str(e)}

    def get_routing_recommendations(self) -> List[str]:
        """Generate routing optimization recommendations"""
        recommendations = []

        try:
            analytics = self.get_routing_analytics()

            # Analyze success rates
            for method_name, stats in analytics["success_rates"].items():
                if stats["attempts"] >= 10:  # Sufficient data
                    success_rate = stats["rate"]
                    if success_rate < 0.7:
                        recommendations.append(
                            f"Consider adjusting routing rules for {method_name} "
                            f"(current success rate: {success_rate:.1%})"
                        )

            # Analyze processing times
            for method_name, stats in analytics["processing_times"].items():
                if isinstance(stats, dict) and stats["count"] >= 5:
                    avg_time = stats["mean"]
                    if avg_time > self.performance_thresholds.get("balanced_time_limit", 0.1):
                        recommendations.append(
                            f"Processing time for {method_name} may be too high "
                            f"(average: {avg_time*1000:.1f}ms)"
                        )

            # Check routing distribution
            total_routes = analytics["total_routes"]
            if total_routes >= 20:
                method_usage = analytics["method_usage"]
                default_usage_rate = method_usage.get("default_parameters", 0) / total_routes

                if default_usage_rate > 0.5:
                    recommendations.append(
                        "High usage of default parameters suggests feature extraction "
                        "or routing logic may need improvement"
                    )

            # Generic recommendations if no specific issues found
            if not recommendations:
                recommendations.append("Routing performance appears optimal")

        except Exception as e:
            recommendations.append(f"Unable to generate recommendations: {e}")

        return recommendations

    # ===== FALLBACK AND RECOVERY SYSTEMS =====

    def __init_conservative_fallback_parameters(self):
        """Initialize conservative parameter sets for edge cases"""
        self.conservative_parameters = {
            "safe_edge_cases": {
                # Ultra-safe parameters for problematic images
                "color_precision": 2,
                "corner_threshold": 60,
                "length_threshold": 4.0,
                "max_iterations": 10,
                "splice_threshold": 90,
                "path_precision": 8,
                "layer_difference": 16,
                "mode": "spline"
            },
            "compatibility_mode": {
                # High compatibility with various VTracer versions
                "color_precision": 4,
                "corner_threshold": 40,
                "length_threshold": 8.0,
                "max_iterations": 15,
                "splice_threshold": 70,
                "path_precision": 6,
                "layer_difference": 12,
                "mode": "polygon"
            },
            "degraded_mode": {
                # Minimal processing for system issues
                "color_precision": 1,
                "corner_threshold": 80,
                "length_threshold": 2.0,
                "max_iterations": 5,
                "splice_threshold": 100,
                "path_precision": 12,
                "layer_difference": 20,
                "mode": "spline"
            }
        }

        # Initialize conservative fallback in routing rules
        self.routing_rules.update({
            "enable_conservative_fallback": True,
            "conservative_failure_threshold": 3,  # Failed attempts before degraded mode
            "system_stress_threshold": 0.8,  # Memory/CPU usage threshold
        })

    def get_conservative_parameters(self, fallback_type: str = "safe_edge_cases",
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get conservative parameter set for fallback scenarios"""
        try:
            # Initialize if not already done
            if not hasattr(self, 'conservative_parameters'):
                self.__init_conservative_fallback_parameters()

            base_params = self.conservative_parameters.get(fallback_type,
                                                          self.conservative_parameters["safe_edge_cases"])

            # Adjust based on context if provided
            if context:
                adjusted_params = base_params.copy()

                # If system stress is high, use degraded mode
                system_stress = context.get("system_stress", 0.0)
                if system_stress > self.routing_rules["system_stress_threshold"]:
                    adjusted_params.update(self.conservative_parameters["degraded_mode"])
                    self.logger.warning(f"High system stress ({system_stress:.1%}), using degraded mode")

                # If previous failures detected, be more conservative
                failure_count = context.get("previous_failures", 0)
                if failure_count >= self.routing_rules["conservative_failure_threshold"]:
                    adjusted_params["max_iterations"] = min(5, adjusted_params["max_iterations"])
                    adjusted_params["color_precision"] = max(1, adjusted_params["color_precision"] - 1)
                    self.logger.warning(f"Multiple failures ({failure_count}), extra conservative parameters")

                return adjusted_params

            return base_params

        except Exception as e:
            self.logger.error(f"Failed to get conservative parameters: {e}")
            # Return absolute minimum parameters
            return {
                "color_precision": 1,
                "corner_threshold": 100,
                "length_threshold": 1.0,
                "max_iterations": 3,
                "splice_threshold": 120,
                "path_precision": 16,
                "layer_difference": 24,
                "mode": "spline"
            }

    def handle_routing_failure(self, error: Exception, context: Dict[str, Any]) -> RoutingDecision:
        """Handle routing system failures with graceful degradation"""
        try:
            failure_info = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": time.time(),
                "context": context
            }

            # Track failure for analysis
            if not hasattr(self, 'routing_failures'):
                self.routing_failures = []

            self.routing_failures.append(failure_info)

            # Limit failure history
            if len(self.routing_failures) > 100:
                self.routing_failures = self.routing_failures[-50:]

            self.logger.error(f"Routing failure handled: {failure_info['error_type']} - {failure_info['error_message']}")

            # Determine appropriate fallback strategy
            failure_count = len([f for f in self.routing_failures[-10:]
                               if time.time() - f["timestamp"] < 300])  # Last 5 minutes

            if failure_count >= 5:
                # Too many recent failures - use degraded mode
                return RoutingDecision(
                    method=OptimizationMethod.CONSERVATIVE_FALLBACK,
                    confidence=0.3,
                    reasoning=f"Multiple routing failures ({failure_count}) - degraded mode",
                    expected_improvement=0.05,
                    processing_time_estimate=0.02
                )
            elif failure_count >= 2:
                # Some recent failures - use conservative approach
                return RoutingDecision(
                    method=OptimizationMethod.CONSERVATIVE_FALLBACK,
                    confidence=0.5,
                    reasoning=f"Recent routing failures ({failure_count}) - conservative approach",
                    expected_improvement=0.08,
                    processing_time_estimate=0.03
                )
            else:
                # Single failure - fallback to default parameters
                return RoutingDecision(
                    method=OptimizationMethod.DEFAULT_PARAMETERS,
                    confidence=0.6,
                    reasoning="Single routing failure - fallback to default parameters",
                    expected_improvement=0.10,
                    processing_time_estimate=0.01
                )

        except Exception as fallback_error:
            self.logger.critical(f"Routing failure handler failed: {fallback_error}")
            # Ultimate fallback
            return RoutingDecision(
                method=OptimizationMethod.DEFAULT_PARAMETERS,
                confidence=0.2,
                reasoning="Critical routing system failure - ultimate fallback",
                expected_improvement=0.02,
                processing_time_estimate=0.005
            )

    def implement_adaptive_routing_learning(self):
        """Implement adaptive routing learning based on historical performance"""
        try:
            if not hasattr(self, 'adaptive_learning'):
                self.adaptive_learning = {
                    "enabled": True,
                    "learning_rate": 0.05,  # How quickly to adapt thresholds
                    "min_samples": 20,  # Minimum samples before adapting
                    "adaptation_history": [],
                    "last_adaptation": time.time()
                }

            analytics = self.get_routing_analytics()
            adaptations_made = []

            # Adapt confidence thresholds based on success rates
            for method_name, stats in analytics["success_rates"].items():
                if stats["attempts"] >= self.adaptive_learning["min_samples"]:
                    success_rate = stats["rate"]

                    # If method is underperforming, raise its confidence threshold
                    if success_rate < 0.7 and method_name == "method_1_correlation":
                        old_threshold = self.routing_rules["method1_confidence_threshold"]
                        adaptation = self.adaptive_learning["learning_rate"] * (0.7 - success_rate)
                        new_threshold = min(0.9, old_threshold + adaptation)

                        if abs(new_threshold - old_threshold) > 0.01:  # Meaningful change
                            self.routing_rules["method1_confidence_threshold"] = new_threshold
                            adaptations_made.append(f"Method 1 confidence threshold: {old_threshold:.3f} → {new_threshold:.3f}")

                    # If method is overperforming, potentially lower threshold
                    elif success_rate > 0.9 and method_name == "method_1_correlation":
                        old_threshold = self.routing_rules["method1_confidence_threshold"]
                        adaptation = self.adaptive_learning["learning_rate"] * (success_rate - 0.9)
                        new_threshold = max(0.3, old_threshold - adaptation)

                        if abs(new_threshold - old_threshold) > 0.01:  # Meaningful change
                            self.routing_rules["method1_confidence_threshold"] = new_threshold
                            adaptations_made.append(f"Method 1 confidence threshold: {old_threshold:.3f} → {new_threshold:.3f}")

            # Adapt other thresholds based on performance patterns
            total_routes = analytics["total_routes"]
            if total_routes >= 50:  # Sufficient data
                method_usage = analytics["method_usage"]

                # If too many routes go to default parameters, lower feature confidence threshold
                default_rate = method_usage.get("default_parameters", 0) / total_routes
                if default_rate > 0.4:
                    old_threshold = self.routing_rules["feature_confidence_threshold"]
                    new_threshold = max(0.4, old_threshold - 0.05)
                    if new_threshold != old_threshold:
                        self.routing_rules["feature_confidence_threshold"] = new_threshold
                        adaptations_made.append(f"Feature confidence threshold: {old_threshold:.3f} → {new_threshold:.3f}")

            # Record adaptations
            if adaptations_made:
                adaptation_record = {
                    "timestamp": time.time(),
                    "adaptations": adaptations_made,
                    "analytics_snapshot": analytics
                }
                self.adaptive_learning["adaptation_history"].append(adaptation_record)
                self.adaptive_learning["last_adaptation"] = time.time()

                # Limit history
                if len(self.adaptive_learning["adaptation_history"]) > 20:
                    self.adaptive_learning["adaptation_history"] = self.adaptive_learning["adaptation_history"][-10:]

                self.logger.info(f"Adaptive learning made {len(adaptations_made)} adjustments: {'; '.join(adaptations_made)}")

        except Exception as e:
            self.logger.error(f"Adaptive routing learning failed: {e}")

    def add_user_override_capability(self, image_path: str, forced_method: str,
                                   user_id: Optional[str] = None,
                                   override_reason: Optional[str] = None) -> RoutingDecision:
        """Allow users to override routing decisions"""
        try:
            # Validate forced method
            try:
                method_enum = OptimizationMethod(forced_method)
            except ValueError:
                self.logger.warning(f"Invalid forced method: {forced_method}")
                method_enum = OptimizationMethod.DEFAULT_PARAMETERS

            # Create override decision
            override_decision = RoutingDecision(
                method=method_enum,
                confidence=1.0,  # User override has maximum confidence
                reasoning=f"User override: {override_reason or 'No reason provided'}",
                expected_improvement=0.15,  # Default expectation
                processing_time_estimate=0.05
            )

            # Track user override for analytics
            override_record = {
                "timestamp": time.time(),
                "user_id": user_id,
                "image_path": image_path,
                "forced_method": forced_method,
                "override_reason": override_reason,
                "decision": asdict(override_decision)
            }

            if not hasattr(self, 'user_overrides'):
                self.user_overrides = []

            self.user_overrides.append(override_record)

            # Limit override history
            if len(self.user_overrides) > 200:
                self.user_overrides = self.user_overrides[-100:]

            self.logger.info(f"User override applied: {forced_method} for {image_path} by {user_id or 'anonymous'}")

            return override_decision

        except Exception as e:
            self.logger.error(f"User override failed: {e}")
            # Fallback to default
            return RoutingDecision(
                method=OptimizationMethod.DEFAULT_PARAMETERS,
                confidence=0.5,
                reasoning=f"User override error: {str(e)}",
                expected_improvement=0.05,
                processing_time_estimate=0.01
            )

    def create_enhanced_performance_monitoring(self) -> Dict[str, Any]:
        """Create comprehensive performance monitoring dashboard"""
        try:
            analytics = self.get_routing_analytics()

            # Calculate advanced metrics
            current_time = time.time()

            # Recent performance (last hour)
            recent_routes = [r for r in self.routing_history
                           if current_time - r["timestamp"] < 3600]

            recent_performance = {
                "routes_last_hour": len(recent_routes),
                "methods_used_recently": {},
                "average_confidence_recent": 0,
                "routing_errors_recent": 0
            }

            if recent_routes:
                for route in recent_routes:
                    method = route["decision"]["method"]
                    # Ensure method is a string value for dictionary operations
                    if hasattr(method, 'value'):
                        method = method.value
                    recent_performance["methods_used_recently"][method] = recent_performance["methods_used_recently"].get(method, 0) + 1
                    recent_performance["average_confidence_recent"] += route["decision"]["confidence"]

                recent_performance["average_confidence_recent"] /= len(recent_routes)

            # System health metrics
            system_health = {
                "routing_failure_rate": 0,
                "adaptive_learning_active": hasattr(self, 'adaptive_learning') and self.adaptive_learning.get("enabled", False),
                "conservative_fallback_usage": 0,
                "user_override_rate": 0
            }

            # Calculate failure rate
            if hasattr(self, 'routing_failures') and self.routing_failures:
                recent_failures = [f for f in self.routing_failures
                                 if current_time - f["timestamp"] < 3600]
                total_recent = len(recent_routes) + len(recent_failures)
                if total_recent > 0:
                    system_health["routing_failure_rate"] = len(recent_failures) / total_recent

            # Conservative fallback usage
            conservative_usage = analytics["method_usage"].get("conservative_fallback", 0)
            total_routes = analytics["total_routes"]
            if total_routes > 0:
                system_health["conservative_fallback_usage"] = conservative_usage / total_routes

            # User override rate
            if hasattr(self, 'user_overrides') and self.user_overrides:
                recent_overrides = [o for o in self.user_overrides
                                  if current_time - o["timestamp"] < 3600]
                if len(recent_routes) > 0:
                    system_health["user_override_rate"] = len(recent_overrides) / len(recent_routes)

            # Performance trending
            trend_analysis = self._analyze_performance_trends()

            monitoring_dashboard = {
                "timestamp": current_time,
                "overall_analytics": analytics,
                "recent_performance": recent_performance,
                "system_health": system_health,
                "trend_analysis": trend_analysis,
                "recommendations": self.get_routing_recommendations(),
                "adaptive_learning_status": getattr(self, 'adaptive_learning', {"enabled": False}),
                "routing_rules_current": self.routing_rules.copy()
            }

            return monitoring_dashboard

        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        try:
            if len(self.routing_history) < 10:
                return {"insufficient_data": True}

            # Split history into recent and older for comparison
            mid_point = len(self.routing_history) // 2
            older_routes = self.routing_history[:mid_point]
            recent_routes = self.routing_history[mid_point:]

            def calculate_period_stats(routes):
                if not routes:
                    return {"confidence": 0, "routing_time": 0, "method_distribution": {}}

                confidences = [r["decision"]["confidence"] for r in routes]
                routing_times = [r["routing_time"] for r in routes]

                methods = {}
                for r in routes:
                    method = r["decision"]["method"]
                    # Ensure method is a string value for dictionary operations
                    if hasattr(method, 'value'):
                        method = method.value
                    methods[method] = methods.get(method, 0) + 1

                return {
                    "confidence": sum(confidences) / len(confidences),
                    "routing_time": sum(routing_times) / len(routing_times),
                    "method_distribution": methods
                }

            older_stats = calculate_period_stats(older_routes)
            recent_stats = calculate_period_stats(recent_routes)

            # Calculate trends
            confidence_trend = recent_stats["confidence"] - older_stats["confidence"]
            time_trend = recent_stats["routing_time"] - older_stats["routing_time"]

            return {
                "confidence_trend": confidence_trend,
                "time_trend": time_trend,
                "confidence_improving": confidence_trend > 0.01,
                "time_improving": time_trend < -0.001,  # Negative is better (faster)
                "older_period_stats": older_stats,
                "recent_period_stats": recent_stats
            }

        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return {"error": str(e)}

    def get_routing_failure_analysis(self) -> Dict[str, Any]:
        """Analyze routing failures for patterns and insights"""
        try:
            if not hasattr(self, 'routing_failures') or not self.routing_failures:
                return {"no_failures": True, "message": "No routing failures recorded"}

            current_time = time.time()

            # Recent failures (last 24 hours)
            recent_failures = [f for f in self.routing_failures
                             if current_time - f["timestamp"] < 86400]

            failure_analysis = {
                "total_failures": len(self.routing_failures),
                "recent_failures": len(recent_failures),
                "failure_types": {},
                "common_patterns": [],
                "recovery_effectiveness": 0
            }

            # Analyze failure types
            for failure in self.routing_failures:
                error_type = failure["error_type"]
                failure_analysis["failure_types"][error_type] = failure_analysis["failure_types"].get(error_type, 0) + 1

            # Find common patterns
            if len(recent_failures) >= 3:
                # Look for patterns in recent failures
                error_types = [f["error_type"] for f in recent_failures]
                most_common_error = max(set(error_types), key=error_types.count)

                if error_types.count(most_common_error) >= len(recent_failures) * 0.5:
                    failure_analysis["common_patterns"].append(f"Frequent {most_common_error} errors")

            return failure_analysis

        except Exception as e:
            self.logger.error(f"Failure analysis failed: {e}")
            return {"error": str(e)}