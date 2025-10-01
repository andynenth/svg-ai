#!/usr/bin/env python3
"""
4-Tier System Orchestrator - Complete Integration Architecture
Coordinates all optimization methods with intelligent routing and quality prediction
"""

import time
import json
import logging
import asyncio
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum
import concurrent.futures

# Core system imports
from .intelligent_router import IntelligentRouter, RoutingDecision
from .enhanced_router_integration import get_enhanced_router, EnhancedRoutingDecision
from .feature_mapping import FeatureMappingOptimizer
from .regression_optimizer import RegressionBasedOptimizer
from .ppo_optimizer import PPOOptimizer
from .performance_optimizer import PerformanceOptimizer
from .error_handler import OptimizationErrorHandler
from ..feature_extraction import ImageFeatureExtractor
from ...utils.quality_metrics import ComprehensiveMetrics

logger = logging.getLogger(__name__)


class OptimizationTier(Enum):
    """4-Tier System Architecture Levels"""
    TIER_1_CLASSIFICATION = "classification"  # Image analysis and feature extraction
    TIER_2_ROUTING = "routing"               # Intelligent method selection
    TIER_3_OPTIMIZATION = "optimization"     # Parameter optimization methods
    TIER_4_QUALITY_PREDICTION = "prediction" # Quality prediction and validation


@dataclass
class SystemExecutionContext:
    """Complete context for 4-tier system execution"""
    request_id: str
    image_path: str
    user_requirements: Dict[str, Any]
    system_state: Dict[str, Any]
    tier_results: Dict[OptimizationTier, Dict[str, Any]]
    execution_timeline: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    error_log: List[str]
    start_time: float

    def add_tier_result(self, tier: OptimizationTier, result: Dict[str, Any]):
        """Add result from a tier execution"""
        self.tier_results[tier] = result
        self.execution_timeline.append({
            "tier": tier.value,
            "timestamp": time.time(),
            "result_summary": {k: v for k, v in result.items() if k in ["success", "confidence", "method", "execution_time"]},
            "duration": time.time() - self.start_time
        })

    def add_error(self, tier: OptimizationTier, error: str):
        """Add error to the execution log"""
        self.error_log.append(f"[{tier.value}] {error}")


@dataclass
class SystemPerformanceMetrics:
    """System-wide performance tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    tier_performance: Dict[str, Dict[str, float]] = None
    method_effectiveness: Dict[str, Dict[str, float]] = None
    average_execution_time: float = 0.0
    quality_improvement_rate: float = 0.0
    system_reliability: float = 0.0
    cache_hit_rates: Dict[str, float] = None

    def __post_init__(self):
        if self.tier_performance is None:
            self.tier_performance = {tier.value: {"avg_time": 0.0, "success_rate": 1.0} for tier in OptimizationTier}
        if self.method_effectiveness is None:
            self.method_effectiveness = {}
        if self.cache_hit_rates is None:
            self.cache_hit_rates = {}


class Tier4SystemOrchestrator:
    """
    Complete 4-Tier System Orchestrator
    Integrates Classification → Routing → Optimization → Quality Prediction
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the 4-tier system orchestrator"""

        # System configuration
        self.config = config or self._get_default_config()

        # Core system components
        self.feature_extractor = ImageFeatureExtractor()
        self.intelligent_router = IntelligentRouter()
        self.enhanced_router = get_enhanced_router()  # Agent 1 integration point
        self.error_handler = OptimizationErrorHandler()
        self.quality_metrics = ComprehensiveMetrics()

        # Optimization methods registry (Tier 3)
        self.optimization_methods = {
            'feature_mapping': FeatureMappingOptimizer(),
            'regression': RegressionBasedOptimizer(),
            'ppo': PPOOptimizer(),
            'performance': PerformanceOptimizer()
        }

        # System state and monitoring
        self.system_metrics = SystemPerformanceMetrics()
        self.active_contexts: Dict[str, SystemExecutionContext] = {}
        self.execution_history: List[SystemExecutionContext] = []

        # Threading and concurrency
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.get("max_concurrent_requests", 10))
        self._lock = threading.RLock()

        # System health monitoring
        self.system_health = {
            "status": "initializing",
            "last_health_check": time.time(),
            "component_status": {},
            "performance_alerts": []
        }

        # Initialize system
        self._initialize_system()

        logger.info("4-Tier System Orchestrator initialized successfully")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
            "max_concurrent_requests": 10,
            "enable_async_processing": True,
            "enable_caching": True,
            "cache_ttl": 3600,
            "quality_prediction_threshold": 0.8,
            "fallback_timeout": 30.0,
            "performance_monitoring": True,
            "error_recovery": True,
            "tier_timeouts": {
                "classification": 5.0,
                "routing": 2.0,
                "optimization": 60.0,
                "prediction": 10.0
            },
            "quality_targets": {
                "simple": 0.95,
                "text": 0.92,
                "gradient": 0.88,
                "complex": 0.85
            },
            "production_mode": False
        }

    def _initialize_system(self):
        """Initialize all system components"""
        try:
            # Initialize components
            components = {
                "feature_extractor": self.feature_extractor,
                "intelligent_router": self.intelligent_router,
                "error_handler": self.error_handler,
                "quality_metrics": self.quality_metrics
            }

            # Add optimization methods
            components.update(self.optimization_methods)

            # Health check each component
            for name, component in components.items():
                try:
                    if hasattr(component, 'health_check'):
                        status = component.health_check()
                    else:
                        status = "operational"

                    self.system_health["component_status"][name] = status

                except Exception as e:
                    self.system_health["component_status"][name] = f"error: {e}"
                    logger.warning(f"Component {name} failed health check: {e}")

            self.system_health["status"] = "operational"
            logger.info("All system components initialized")

        except Exception as e:
            self.system_health["status"] = "error"
            logger.error(f"System initialization failed: {e}")
            raise

    async def execute_4tier_optimization(
        self,
        image_path: str,
        user_requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute complete 4-tier optimization pipeline

        Args:
            image_path: Path to image file
            user_requirements: User-specified requirements (quality target, time budget, etc.)

        Returns:
            Complete optimization result with metadata from all tiers
        """

        # Create execution context
        request_id = f"4tier_{int(time.time() * 1000)}"
        context = SystemExecutionContext(
            request_id=request_id,
            image_path=image_path,
            user_requirements=user_requirements or {},
            system_state=self._get_system_state(),
            tier_results={},
            execution_timeline=[],
            performance_metrics={},
            error_log=[],
            start_time=time.time()
        )

        # Store active context
        with self._lock:
            self.active_contexts[request_id] = context

        try:
            # Execute all 4 tiers sequentially with error handling
            tier_1_result = await self._execute_tier_1_classification(context)
            if not tier_1_result["success"]:
                return self._create_error_response(context, "Tier 1 (Classification) failed")

            tier_2_result = await self._execute_tier_2_routing(context)
            if not tier_2_result["success"]:
                return self._create_error_response(context, "Tier 2 (Routing) failed")

            tier_3_result = await self._execute_tier_3_optimization(context)
            if not tier_3_result["success"]:
                return self._create_error_response(context, "Tier 3 (Optimization) failed")

            tier_4_result = await self._execute_tier_4_quality_prediction(context)
            # Note: Tier 4 failure is not critical - we can still return optimization results

            # Compile final results
            final_result = self._compile_final_results(context)

            # Update system metrics
            self._update_system_metrics(context, success=True)

            logger.info(f"4-Tier optimization completed: {request_id} in {time.time() - context.start_time:.3f}s")
            return final_result

        except Exception as e:
            context.add_error(OptimizationTier.TIER_1_CLASSIFICATION, f"System error: {e}")
            self._update_system_metrics(context, success=False)
            return self._create_error_response(context, f"System execution failed: {e}")

        finally:
            # Cleanup active context
            with self._lock:
                if request_id in self.active_contexts:
                    self.execution_history.append(self.active_contexts[request_id])
                    del self.active_contexts[request_id]

                    # Limit history size
                    if len(self.execution_history) > 1000:
                        self.execution_history = self.execution_history[-500:]

    async def _execute_tier_1_classification(self, context: SystemExecutionContext) -> Dict[str, Any]:
        """
        Tier 1: Image Classification and Feature Extraction
        """
        tier_start = time.time()

        try:
            logger.debug(f"Executing Tier 1 (Classification) for {context.request_id}")

            # Extract comprehensive image features
            features = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.feature_extractor.extract_features,
                context.image_path
            )

            # Classify image type
            image_type = self._classify_image_type(features)

            # Determine complexity level
            complexity_level = self._determine_complexity_level(features)

            # Assess image characteristics
            image_characteristics = self._assess_image_characteristics(features)

            tier_result = {
                "success": True,
                "execution_time": time.time() - tier_start,
                "features": features,
                "image_type": image_type,
                "complexity_level": complexity_level,
                "characteristics": image_characteristics,
                "feature_confidence": self._calculate_feature_confidence(features),
                "tier": "classification"
            }

            context.add_tier_result(OptimizationTier.TIER_1_CLASSIFICATION, tier_result)
            logger.debug(f"Tier 1 completed: {image_type} image, complexity: {complexity_level}")

            return tier_result

        except Exception as e:
            error_msg = f"Tier 1 classification failed: {e}"
            context.add_error(OptimizationTier.TIER_1_CLASSIFICATION, error_msg)
            logger.error(error_msg)

            return {
                "success": False,
                "execution_time": time.time() - tier_start,
                "error": error_msg,
                "tier": "classification"
            }

    async def _execute_tier_2_routing(self, context: SystemExecutionContext) -> Dict[str, Any]:
        """
        Tier 2: Intelligent Method Selection and Routing
        """
        tier_start = time.time()

        try:
            logger.debug(f"Executing Tier 2 (Routing) for {context.request_id}")

            # Get classification results
            tier_1_result = context.tier_results[OptimizationTier.TIER_1_CLASSIFICATION]
            features = tier_1_result["features"]

            # Prepare routing parameters
            routing_params = {
                **context.user_requirements,
                "image_type": tier_1_result["image_type"],
                "complexity_level": tier_1_result["complexity_level"]
            }

            # Execute enhanced intelligent routing (Agent 1 integration)
            routing_decision = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.enhanced_router.route_with_quality_prediction,
                context.image_path,
                features,
                **routing_params
            )

            # Validate routing decision
            if not self._validate_routing_decision(routing_decision, routing_params):
                raise ValueError("Routing decision validation failed")

            # Extract enhanced routing features
            enhanced_features = {}
            if isinstance(routing_decision, EnhancedRoutingDecision):
                enhanced_features = {
                    "predicted_qualities": routing_decision.predicted_qualities,
                    "quality_confidence": routing_decision.quality_confidence,
                    "prediction_time": routing_decision.prediction_time,
                    "ml_confidence": routing_decision.ml_confidence,
                    "feature_importance": routing_decision.feature_importance,
                    "alternative_methods": routing_decision.alternative_methods,
                    "enhanced_reasoning": routing_decision.enhanced_reasoning,
                    "prediction_model_version": routing_decision.prediction_model_version
                }

            tier_result = {
                "success": True,
                "execution_time": time.time() - tier_start,
                "primary_method": routing_decision.primary_method,
                "fallback_methods": routing_decision.fallback_methods,
                "confidence": routing_decision.confidence,
                "reasoning": routing_decision.reasoning,
                "estimated_time": routing_decision.estimated_time,
                "estimated_quality": routing_decision.estimated_quality,
                "routing_decision": asdict(routing_decision),
                "enhanced_features": enhanced_features,
                "tier": "routing"
            }

            context.add_tier_result(OptimizationTier.TIER_2_ROUTING, tier_result)
            logger.debug(f"Tier 2 completed: Selected method '{routing_decision.primary_method}' with confidence {routing_decision.confidence:.3f}")

            return tier_result

        except Exception as e:
            error_msg = f"Tier 2 routing failed: {e}"
            context.add_error(OptimizationTier.TIER_2_ROUTING, error_msg)
            logger.error(error_msg)

            return {
                "success": False,
                "execution_time": time.time() - tier_start,
                "error": error_msg,
                "tier": "routing"
            }

    async def _execute_tier_3_optimization(self, context: SystemExecutionContext) -> Dict[str, Any]:
        """
        Tier 3: Parameter Optimization Execution
        """
        tier_start = time.time()

        try:
            logger.debug(f"Executing Tier 3 (Optimization) for {context.request_id}")

            # Get previous tier results
            tier_1_result = context.tier_results[OptimizationTier.TIER_1_CLASSIFICATION]
            tier_2_result = context.tier_results[OptimizationTier.TIER_2_ROUTING]

            features = tier_1_result["features"]
            primary_method = tier_2_result["primary_method"]
            fallback_methods = tier_2_result["fallback_methods"]

            # Execute primary optimization method
            optimization_result = await self._execute_optimization_method(
                primary_method, features, tier_1_result["image_type"], context
            )

            # If primary method fails, try fallback methods
            if not optimization_result["success"] and fallback_methods:
                logger.warning(f"Primary method {primary_method} failed, trying fallbacks")

                for fallback_method in fallback_methods:
                    logger.debug(f"Trying fallback method: {fallback_method}")

                    fallback_result = await self._execute_optimization_method(
                        fallback_method, features, tier_1_result["image_type"], context
                    )

                    if fallback_result["success"]:
                        optimization_result = fallback_result
                        optimization_result["fallback_used"] = fallback_method
                        optimization_result["primary_method_failed"] = primary_method
                        break

            # Validate optimization result
            if not optimization_result["success"]:
                raise ValueError("All optimization methods failed")

            tier_result = {
                "success": True,
                "execution_time": time.time() - tier_start,
                "method_used": optimization_result["method"],
                "optimized_parameters": optimization_result["parameters"],
                "optimization_confidence": optimization_result.get("confidence", 0.0),
                "optimization_metadata": optimization_result.get("metadata", {}),
                "fallback_used": optimization_result.get("fallback_used"),
                "tier": "optimization"
            }

            context.add_tier_result(OptimizationTier.TIER_3_OPTIMIZATION, tier_result)
            logger.debug(f"Tier 3 completed: Method '{optimization_result['method']}' optimization successful")

            return tier_result

        except Exception as e:
            error_msg = f"Tier 3 optimization failed: {e}"
            context.add_error(OptimizationTier.TIER_3_OPTIMIZATION, error_msg)
            logger.error(error_msg)

            return {
                "success": False,
                "execution_time": time.time() - tier_start,
                "error": error_msg,
                "tier": "optimization"
            }

    async def _execute_tier_4_quality_prediction(self, context: SystemExecutionContext) -> Dict[str, Any]:
        """
        Tier 4: Quality Prediction and Validation
        """
        tier_start = time.time()

        try:
            logger.debug(f"Executing Tier 4 (Quality Prediction) for {context.request_id}")

            # Get previous tier results
            tier_1_result = context.tier_results[OptimizationTier.TIER_1_CLASSIFICATION]
            tier_3_result = context.tier_results[OptimizationTier.TIER_3_OPTIMIZATION]

            features = tier_1_result["features"]
            optimized_parameters = tier_3_result["optimized_parameters"]
            method_used = tier_3_result["method_used"]

            # Predict quality using multiple approaches
            quality_predictions = await self._predict_optimization_quality(
                features, optimized_parameters, method_used, context
            )

            # Validate quality predictions
            quality_validation = self._validate_quality_predictions(quality_predictions, context)

            # Generate quality assurance recommendations
            qa_recommendations = self._generate_qa_recommendations(quality_predictions, context)

            tier_result = {
                "success": True,
                "execution_time": time.time() - tier_start,
                "quality_predictions": quality_predictions,
                "quality_validation": quality_validation,
                "qa_recommendations": qa_recommendations,
                "prediction_confidence": quality_predictions.get("confidence", 0.0),
                "expected_quality_score": quality_predictions.get("predicted_ssim", 0.0),
                "tier": "prediction"
            }

            context.add_tier_result(OptimizationTier.TIER_4_QUALITY_PREDICTION, tier_result)
            logger.debug(f"Tier 4 completed: Predicted quality {quality_predictions.get('predicted_ssim', 0.0):.3f}")

            return tier_result

        except Exception as e:
            error_msg = f"Tier 4 quality prediction failed: {e}"
            context.add_error(OptimizationTier.TIER_4_QUALITY_PREDICTION, error_msg)
            logger.warning(error_msg)  # Warning since this tier is not critical

            return {
                "success": False,
                "execution_time": time.time() - tier_start,
                "error": error_msg,
                "tier": "prediction"
            }

    async def _execute_optimization_method(
        self,
        method_name: str,
        features: Dict[str, float],
        image_type: str,
        context: SystemExecutionContext
    ) -> Dict[str, Any]:
        """Execute specific optimization method"""

        if method_name not in self.optimization_methods:
            return {
                "success": False,
                "error": f"Unknown optimization method: {method_name}",
                "method": method_name
            }

        try:
            optimizer = self.optimization_methods[method_name]

            # Execute optimization
            optimization_result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                optimizer.optimize,
                features,
                image_type
            )

            return {
                "success": True,
                "method": method_name,
                "parameters": optimization_result,
                "confidence": getattr(optimizer, 'last_confidence', 0.8),
                "metadata": {
                    "optimizer_stats": optimizer.get_optimization_stats() if hasattr(optimizer, 'get_optimization_stats') else {},
                    "execution_timestamp": time.time()
                }
            }

        except Exception as e:
            logger.error(f"Optimization method {method_name} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": method_name
            }

    async def _predict_optimization_quality(
        self,
        features: Dict[str, float],
        optimized_parameters: Dict[str, Any],
        method_used: str,
        context: SystemExecutionContext
    ) -> Dict[str, Any]:
        """Predict optimization quality using available models"""

        # This is where Agent 1's enhanced router with quality prediction would be integrated
        # For now, we'll use a simplified prediction model

        try:
            # Basic quality prediction based on features and parameters
            base_quality = self._estimate_base_quality(features)
            parameter_quality_boost = self._estimate_parameter_boost(optimized_parameters, method_used)
            method_effectiveness = self._get_method_effectiveness(method_used, features)

            predicted_ssim = min(0.99, base_quality + parameter_quality_boost * method_effectiveness)

            # Add confidence estimation
            confidence = self._calculate_prediction_confidence(features, method_used)

            return {
                "predicted_ssim": predicted_ssim,
                "confidence": confidence,
                "base_quality": base_quality,
                "parameter_boost": parameter_quality_boost,
                "method_effectiveness": method_effectiveness,
                "prediction_method": "simplified_model",
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Quality prediction failed: {e}")
            return {
                "predicted_ssim": 0.8,  # Conservative estimate
                "confidence": 0.3,
                "error": str(e),
                "prediction_method": "fallback"
            }

    def _compile_final_results(self, context: SystemExecutionContext) -> Dict[str, Any]:
        """Compile final results from all tiers"""

        total_time = time.time() - context.start_time

        # Get key results from each tier
        classification_result = context.tier_results.get(OptimizationTier.TIER_1_CLASSIFICATION, {})
        routing_result = context.tier_results.get(OptimizationTier.TIER_2_ROUTING, {})
        optimization_result = context.tier_results.get(OptimizationTier.TIER_3_OPTIMIZATION, {})
        prediction_result = context.tier_results.get(OptimizationTier.TIER_4_QUALITY_PREDICTION, {})

        # Compile comprehensive result
        final_result = {
            "success": True,
            "request_id": context.request_id,
            "total_execution_time": total_time,

            # Core optimization results
            "optimized_parameters": optimization_result.get("optimized_parameters", {}),
            "method_used": optimization_result.get("method_used", "unknown"),
            "optimization_confidence": optimization_result.get("optimization_confidence", 0.0),

            # Quality prediction
            "predicted_quality": prediction_result.get("expected_quality_score", 0.8),
            "quality_confidence": prediction_result.get("prediction_confidence", 0.0),

            # Image analysis
            "image_type": classification_result.get("image_type", "unknown"),
            "complexity_level": classification_result.get("complexity_level", "medium"),
            "image_features": classification_result.get("features", {}),

            # Routing decision
            "routing_decision": {
                "primary_method": routing_result.get("primary_method", "unknown"),
                "confidence": routing_result.get("confidence", 0.0),
                "reasoning": routing_result.get("reasoning", ""),
                "fallback_methods": routing_result.get("fallback_methods", [])
            },

            # System performance
            "tier_performance": {
                tier.value: result.get("execution_time", 0.0)
                for tier, result in context.tier_results.items()
            },

            # Execution timeline
            "execution_timeline": context.execution_timeline,

            # Metadata
            "metadata": {
                "system_version": "4-tier-v1.0",
                "timestamp": datetime.now().isoformat(),
                "user_requirements": context.user_requirements,
                "system_state": context.system_state,
                "tier_results_summary": {
                    tier.value: {"success": result.get("success", False), "time": result.get("execution_time", 0.0)}
                    for tier, result in context.tier_results.items()
                },
                "error_log": context.error_log
            }
        }

        # Add quality assurance recommendations if available
        if prediction_result.get("qa_recommendations"):
            final_result["qa_recommendations"] = prediction_result["qa_recommendations"]

        return final_result

    def _create_error_response(self, context: SystemExecutionContext, error_message: str) -> Dict[str, Any]:
        """Create error response with partial results"""

        return {
            "success": False,
            "request_id": context.request_id,
            "error": error_message,
            "total_execution_time": time.time() - context.start_time,
            "partial_results": context.tier_results,
            "execution_timeline": context.execution_timeline,
            "error_log": context.error_log,
            "metadata": {
                "system_version": "4-tier-v1.0",
                "timestamp": datetime.now().isoformat(),
                "failure_point": error_message
            }
        }

    # Helper methods for tier execution

    def _classify_image_type(self, features: Dict[str, float]) -> str:
        """Classify image type based on features"""
        edge_density = features.get("edge_density", 0.0)
        unique_colors = features.get("unique_colors", 0.0)
        text_probability = features.get("text_probability", 0.0)
        gradient_strength = features.get("gradient_strength", 0.0)

        if text_probability > 0.7:
            return "text"
        elif gradient_strength > 0.6:
            return "gradient"
        elif edge_density < 0.2 and unique_colors < 8:
            return "simple"
        else:
            return "complex"

    def _determine_complexity_level(self, features: Dict[str, float]) -> str:
        """Determine complexity level"""
        complexity_score = features.get("complexity_score", 0.5)

        if complexity_score < 0.3:
            return "low"
        elif complexity_score < 0.7:
            return "medium"
        else:
            return "high"

    def _assess_image_characteristics(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Assess image characteristics"""
        return {
            "has_gradients": features.get("gradient_strength", 0.0) > 0.3,
            "is_geometric": features.get("geometric_score", 0.0) > 0.6,
            "has_text": features.get("text_probability", 0.0) > 0.5,
            "is_monochrome": features.get("unique_colors", 10) < 3,
            "is_detailed": features.get("edge_density", 0.0) > 0.5
        }

    def _calculate_feature_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in feature extraction"""
        # Simple confidence based on feature completeness and consistency
        expected_features = ["edge_density", "unique_colors", "complexity_score", "entropy"]
        present_features = sum(1 for feat in expected_features if feat in features and features[feat] is not None)

        base_confidence = present_features / len(expected_features)

        # Adjust based on feature values consistency
        if all(0.0 <= features.get(feat, 0.5) <= 1.0 for feat in expected_features if feat in features):
            base_confidence *= 1.1  # Boost for reasonable values

        return min(1.0, base_confidence)

    def _validate_routing_decision(self, decision: RoutingDecision, params: Dict[str, Any]) -> bool:
        """Validate routing decision"""
        if not decision.primary_method or decision.primary_method not in self.optimization_methods:
            return False

        if decision.confidence < 0.1:  # Very low confidence
            return False

        return True

    def _estimate_base_quality(self, features: Dict[str, float]) -> float:
        """Estimate base quality from image features"""
        complexity = features.get("complexity_score", 0.5)
        edge_density = features.get("edge_density", 0.3)

        # Simpler images typically get higher quality
        base_quality = 0.7 + 0.2 * (1.0 - complexity) + 0.1 * min(edge_density, 0.5)
        return min(0.9, base_quality)

    def _estimate_parameter_boost(self, parameters: Dict[str, Any], method: str) -> float:
        """Estimate quality boost from optimized parameters"""
        # Method-specific quality boost estimation
        method_boosts = {
            "feature_mapping": 0.1,
            "regression": 0.12,
            "ppo": 0.15,
            "performance": 0.08
        }

        return method_boosts.get(method, 0.1)

    def _get_method_effectiveness(self, method: str, features: Dict[str, float]) -> float:
        """Get method effectiveness for given features"""
        complexity = features.get("complexity_score", 0.5)

        # Method effectiveness based on complexity
        if method == "feature_mapping":
            return 1.2 - complexity  # Better for simple images
        elif method == "ppo":
            return 0.8 + complexity * 0.4  # Better for complex images
        elif method == "regression":
            return 1.0  # Consistent across complexities
        else:
            return 0.9

    def _calculate_prediction_confidence(self, features: Dict[str, float], method: str) -> float:
        """Calculate confidence in quality prediction"""
        feature_confidence = self._calculate_feature_confidence(features)
        method_confidence = 0.8  # Base method confidence

        return (feature_confidence + method_confidence) / 2

    def _validate_quality_predictions(self, predictions: Dict[str, Any], context: SystemExecutionContext) -> Dict[str, Any]:
        """Validate quality predictions"""
        predicted_quality = predictions.get("predicted_ssim", 0.0)
        confidence = predictions.get("confidence", 0.0)

        validation_result = {
            "is_valid": True,
            "validation_score": 1.0,
            "warnings": [],
            "recommendations": []
        }

        # Check prediction bounds
        if predicted_quality < 0.0 or predicted_quality > 1.0:
            validation_result["is_valid"] = False
            validation_result["warnings"].append("Predicted quality out of valid range")

        # Check confidence
        if confidence < 0.3:
            validation_result["warnings"].append("Low prediction confidence")
            validation_result["recommendations"].append("Consider using fallback method")

        return validation_result

    def _generate_qa_recommendations(self, predictions: Dict[str, Any], context: SystemExecutionContext) -> List[str]:
        """Generate quality assurance recommendations"""
        recommendations = []

        predicted_quality = predictions.get("predicted_ssim", 0.0)
        confidence = predictions.get("confidence", 0.0)

        # Quality-based recommendations
        if predicted_quality < 0.7:
            recommendations.append("Consider using a more sophisticated optimization method")

        if confidence < 0.5:
            recommendations.append("Prediction confidence is low - validate results manually")

        # User requirement-based recommendations
        target_quality = context.user_requirements.get("quality_target", 0.85)
        if predicted_quality < target_quality:
            recommendations.append(f"Predicted quality ({predicted_quality:.3f}) below target ({target_quality:.3f})")

        return recommendations

    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        return {
            "active_requests": len(self.active_contexts),
            "system_health": self.system_health["status"],
            "component_status": self.system_health["component_status"],
            "timestamp": time.time()
        }

    def _update_system_metrics(self, context: SystemExecutionContext, success: bool):
        """Update system performance metrics"""
        with self._lock:
            self.system_metrics.total_requests += 1
            if success:
                self.system_metrics.successful_requests += 1

            # Update tier performance metrics
            for tier, result in context.tier_results.items():
                if tier.value not in self.system_metrics.tier_performance:
                    self.system_metrics.tier_performance[tier.value] = {"avg_time": 0.0, "success_rate": 1.0}

                # Update average time (simple moving average)
                current_avg = self.system_metrics.tier_performance[tier.value]["avg_time"]
                new_time = result.get("execution_time", 0.0)
                self.system_metrics.tier_performance[tier.value]["avg_time"] = (current_avg + new_time) / 2

            # Update system reliability
            self.system_metrics.system_reliability = (
                self.system_metrics.successful_requests / max(self.system_metrics.total_requests, 1)
            )

    # Public API methods

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_health": self.system_health,
            "performance_metrics": asdict(self.system_metrics),
            "active_requests": len(self.active_contexts),
            "configuration": self.config,
            "component_versions": {
                "orchestrator": "1.0.0",
                "intelligent_router": "1.0.0",
                "optimization_methods": len(self.optimization_methods)
            }
        }

    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        recent_history = self.execution_history[-limit:] if limit else self.execution_history

        return [
            {
                "request_id": ctx.request_id,
                "execution_time": time.time() - ctx.start_time if ctx.request_id in self.active_contexts else
                                 ctx.execution_timeline[-1]["duration"] if ctx.execution_timeline else 0.0,
                "success": len(ctx.error_log) == 0,
                "tiers_completed": len(ctx.tier_results),
                "image_type": ctx.tier_results.get(OptimizationTier.TIER_1_CLASSIFICATION, {}).get("image_type", "unknown"),
                "method_used": ctx.tier_results.get(OptimizationTier.TIER_3_OPTIMIZATION, {}).get("method_used", "unknown"),
                "timestamp": ctx.start_time
            }
            for ctx in recent_history
        ]

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_start = time.time()

        health_results = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "check_duration": 0.0,
            "components": {},
            "performance": {},
            "alerts": []
        }

        try:
            # Check each component
            for name, component in {**{"router": self.intelligent_router}, **self.optimization_methods}.items():
                try:
                    if hasattr(component, 'health_check'):
                        component_status = await asyncio.get_event_loop().run_in_executor(
                            self.executor, component.health_check
                        )
                    else:
                        component_status = "operational"

                    health_results["components"][name] = component_status

                except Exception as e:
                    health_results["components"][name] = f"error: {e}"
                    health_results["alerts"].append(f"Component {name} health check failed: {e}")

            # Performance checks
            health_results["performance"] = {
                "active_requests": len(self.active_contexts),
                "total_requests": self.system_metrics.total_requests,
                "success_rate": self.system_metrics.system_reliability,
                "average_execution_time": self.system_metrics.average_execution_time
            }

            # Overall status determination
            failed_components = [name for name, status in health_results["components"].items() if "error" in str(status)]
            if failed_components:
                health_results["overall_status"] = "degraded"
                health_results["alerts"].append(f"Failed components: {failed_components}")

            health_results["check_duration"] = time.time() - health_start

        except Exception as e:
            health_results["overall_status"] = "error"
            health_results["alerts"].append(f"Health check failed: {e}")

        # Update system health
        self.system_health.update({
            "status": health_results["overall_status"],
            "last_health_check": time.time(),
            "component_status": health_results["components"]
        })

        return health_results

    def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("Shutting down 4-Tier System Orchestrator...")

        try:
            # Shutdown executor
            self.executor.shutdown(wait=True)

            # Save system metrics
            if hasattr(self, 'intelligent_router'):
                self.intelligent_router.save_model()
                self.intelligent_router.save_performance_history()

            logger.info("4-Tier System Orchestrator shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Factory function
def create_4tier_orchestrator(config: Optional[Dict[str, Any]] = None) -> Tier4SystemOrchestrator:
    """Create and initialize 4-tier system orchestrator"""
    return Tier4SystemOrchestrator(config)


# Example usage
if __name__ == "__main__":
    async def main():
        # Create orchestrator
        orchestrator = create_4tier_orchestrator()

        # Perform health check
        health = await orchestrator.health_check()
        print(f"System Health: {health['overall_status']}")

        # Example optimization
        test_image = "data/logos/simple_geometric/circle_00.png"
        user_requirements = {
            "quality_target": 0.9,
            "time_constraint": 30.0,
            "speed_priority": "balanced"
        }

        if Path(test_image).exists():
            result = await orchestrator.execute_4tier_optimization(test_image, user_requirements)
            print(f"Optimization Result: {result['success']}")
            print(f"Method Used: {result.get('method_used', 'unknown')}")
            print(f"Predicted Quality: {result.get('predicted_quality', 0.0):.3f}")
            print(f"Total Time: {result.get('total_execution_time', 0.0):.3f}s")

        # Shutdown
        orchestrator.shutdown()

    asyncio.run(main())