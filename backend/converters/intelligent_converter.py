#!/usr/bin/env python3
"""
Intelligent Converter - Complete Multi-Method Integration
Integrates all three optimization methods with intelligent routing
"""

from typing import Dict, Any, Optional, List
import time
import logging
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass

from .ai_enhanced_converter import AIEnhancedConverter
from ..ai_modules.optimization_old.adaptive_optimizer import AdaptiveOptimizer
from ..ai_modules.optimization_old.parameter_router import ParameterRouter

try:
    from ..ai_modules.optimization_old.ppo_optimizer import PPOVTracerOptimizer
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    logging.getLogger(__name__).warning("PPO optimizer not available - Method 2 disabled")


@dataclass
class MethodPerformanceStats:
    """Statistics for a single optimization method"""
    count: int = 0
    avg_quality: float = 0.0
    avg_time: float = 0.0
    total_quality: float = 0.0
    total_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0

    def update(self, quality_improvement: float, processing_time: float, success: bool):
        """Update performance statistics"""
        self.count += 1
        self.total_quality += quality_improvement
        self.total_time += processing_time
        self.avg_quality = self.total_quality / self.count
        self.avg_time = self.total_time / self.count

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.success_count / max(1, self.count)


@dataclass
class RoutingDecision:
    """Routing decision with method and reasoning"""
    method: str
    confidence: float
    reasoning: str
    expected_improvement: float
    processing_time_estimate: float
    fallback_methods: List[str]


class IntelligentConverter(AIEnhancedConverter):
    """Intelligent converter with all three optimization methods"""

    def __init__(self):
        super().__init__()

        # Initialize all optimization methods
        self.method1_optimizer = self.optimizer  # From AIEnhancedConverter
        self.method2_optimizer = None  # PPO (loaded when model available)
        self.method3_optimizer = AdaptiveOptimizer()

        # Initialize PPO optimizer if available
        if PPO_AVAILABLE:
            try:
                self.method2_optimizer = PPOVTracerOptimizer()
                self.logger.info("PPO optimizer (Method 2) initialized successfully")
            except Exception as e:
                self.logger.warning(f"PPO optimizer initialization failed: {e}")
                self.method2_optimizer = None

        # Intelligent routing system
        self.router = ParameterRouter()

        # Performance tracking with detailed statistics
        self.method_performance = {
            'method1': MethodPerformanceStats(),
            'method2': MethodPerformanceStats(),
            'method3': MethodPerformanceStats()
        }

        # Method availability tracking
        self.method_availability = {
            'method1': True,  # Always available (mathematical correlation)
            'method2': self.method2_optimizer is not None,
            'method3': True   # Adaptive optimizer always available
        }

        # Configuration for method selection
        self.routing_config = {
            'complexity_thresholds': {
                'simple': 0.3,      # <0.3 complexity -> Method 1
                'medium': 0.7       # 0.3-0.7 -> Method 2 or 1, >0.7 -> Method 3
            },
            'quality_mode': 'balanced',  # fast, balanced, quality
            'enable_learning': True,
            'enable_fallbacks': True,
            'max_processing_time': 60.0,
            'min_quality_improvement': 0.10
        }

        # Learning and adaptation
        self.routing_history = []
        self.method_preferences = {
            'simple': ['method1', 'method3', 'method2'],
            'text': ['method1', 'method2', 'method3'],
            'gradient': ['method3', 'method2', 'method1'],
            'complex': ['method3', 'method2', 'method1']
        }

        # Update converter name
        self.name = "Intelligent Converter (All Methods)"

        self.logger = logging.getLogger(__name__)
        self.logger.info("Intelligent Converter initialized with all optimization methods")

    def convert(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Intelligent conversion using optimal method selection"""
        start_time = time.time()

        try:
            # Extract features and analyze image
            features = self.feature_extractor.extract_features(image_path)
            logo_type = self._classify_image(image_path)

            # Determine optimal optimization method
            routing_decision = self._select_optimization_method(
                image_path, features, logo_type, kwargs
            )

            # Execute optimization using selected method
            result = self._execute_optimization(
                image_path, routing_decision, features, **kwargs
            )

            # Update performance tracking
            processing_time = time.time() - start_time
            self._update_method_performance(routing_decision, result, processing_time)

            # Add intelligent routing metadata
            result.update({
                'routing_decision': routing_decision.__dict__,
                'intelligent_routing': True,
                'total_processing_time': processing_time,
                'method_used': routing_decision.method
            })

            self.logger.info(f"Intelligent conversion completed using {routing_decision.method} in {processing_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Intelligent conversion failed: {e}")
            return self._fallback_conversion(image_path, **kwargs)

    def _select_optimization_method(self, image_path: str, features: Dict[str, float],
                                  logo_type: str, kwargs: Dict[str, Any]) -> RoutingDecision:
        """Intelligent method selection with learning and adaptation"""

        # Calculate complexity score
        complexity = self._calculate_complexity_score(features)

        # Get user preferences from kwargs
        preferred_method = kwargs.get('force_method')
        quality_mode = kwargs.get('quality_mode', self.routing_config['quality_mode'])
        max_time = kwargs.get('max_processing_time', self.routing_config['max_processing_time'])

        # If user forces a specific method, validate and use it
        if preferred_method and self._is_method_available(preferred_method):
            return RoutingDecision(
                method=preferred_method,
                confidence=0.9,
                reasoning=f"User-forced method selection",
                expected_improvement=self._estimate_quality_improvement(preferred_method, features),
                processing_time_estimate=self._estimate_processing_time(preferred_method, features),
                fallback_methods=self._get_fallback_methods(preferred_method)
            )

        # Intelligent routing based on complexity and preferences
        if quality_mode == 'fast':
            selected_method = self._select_fastest_method(complexity, logo_type)
        elif quality_mode == 'quality':
            selected_method = self._select_highest_quality_method(complexity, logo_type)
        else:  # balanced
            selected_method = self._select_balanced_method(complexity, logo_type, max_time)

        # Learn from historical performance
        if self.routing_config['enable_learning']:
            selected_method = self._apply_learned_preferences(selected_method, logo_type, features)

        # Validate method availability
        if not self._is_method_available(selected_method):
            selected_method = self._select_fallback_method(selected_method, complexity, logo_type)

        # Calculate confidence based on method performance history
        confidence = self._calculate_routing_confidence(selected_method, features, logo_type)

        return RoutingDecision(
            method=selected_method,
            confidence=confidence,
            reasoning=self._generate_routing_reasoning(selected_method, complexity, logo_type, quality_mode),
            expected_improvement=self._estimate_quality_improvement(selected_method, features),
            processing_time_estimate=self._estimate_processing_time(selected_method, features),
            fallback_methods=self._get_fallback_methods(selected_method)
        )

    def _execute_optimization(self, image_path: str, routing_decision: RoutingDecision,
                            features: Dict[str, float], **kwargs) -> Dict[str, Any]:
        """Execute optimization using the selected method"""

        method = routing_decision.method
        start_time = time.time()

        try:
            if method == 'method1':
                result = self._execute_method1(image_path, features, **kwargs)
            elif method == 'method2' and self.method2_optimizer:
                result = self._execute_method2(image_path, features, **kwargs)
            elif method == 'method3':
                result = self._execute_method3(image_path, features, **kwargs)
            else:
                # Fallback to method1
                self.logger.warning(f"Method {method} not available, falling back to method1")
                result = self._execute_method1(image_path, features, **kwargs)
                routing_decision.method = 'method1'

            # Add execution metadata
            execution_time = time.time() - start_time
            result.update({
                'method_execution_time': execution_time,
                'method_used': routing_decision.method,
                'success': True
            })

            return result

        except Exception as e:
            self.logger.error(f"Method {method} execution failed: {e}")

            # Try fallback methods
            for fallback_method in routing_decision.fallback_methods:
                if self._is_method_available(fallback_method):
                    try:
                        self.logger.info(f"Attempting fallback to {fallback_method}")
                        if fallback_method == 'method1':
                            result = self._execute_method1(image_path, features, **kwargs)
                        elif fallback_method == 'method3':
                            result = self._execute_method3(image_path, features, **kwargs)

                        result.update({
                            'method_execution_time': time.time() - start_time,
                            'method_used': fallback_method,
                            'success': True,
                            'fallback_used': True,
                            'original_method_failed': method
                        })
                        return result
                    except Exception as fallback_error:
                        self.logger.warning(f"Fallback method {fallback_method} also failed: {fallback_error}")
                        continue

            # If all methods fail, use the base converter's error handling
            raise RuntimeError(f"All optimization methods failed for {image_path}")

    def _execute_method1(self, image_path: str, features: Dict[str, float], **kwargs) -> Dict[str, Any]:
        """Execute Method 1 (Mathematical Correlation)"""
        # Use the inherited AIEnhancedConverter functionality
        svg_content = super().convert(image_path, **kwargs)

        return {
            'svg_content': svg_content,
            'method': 'method1',
            'optimization_type': 'mathematical_correlation'
        }

    def _execute_method2(self, image_path: str, features: Dict[str, float], **kwargs) -> Dict[str, Any]:
        """Execute Method 2 (PPO Reinforcement Learning)"""
        if not self.method2_optimizer:
            raise RuntimeError("PPO optimizer not available")

        # Use PPO optimizer for parameter optimization
        optimization_result = self.method2_optimizer.optimize_parameters(image_path, features)

        # Convert using optimized parameters
        svg_content = self._convert_with_optimized_params(
            image_path, optimization_result['parameters'], **kwargs
        )

        return {
            'svg_content': svg_content,
            'method': 'method2',
            'optimization_type': 'ppo_reinforcement_learning',
            'optimization_result': optimization_result
        }

    def _execute_method3(self, image_path: str, features: Dict[str, float], **kwargs) -> Dict[str, Any]:
        """Execute Method 3 (Adaptive Spatial Optimization)"""
        # Use adaptive optimizer for spatial optimization
        optimization_result = self.method3_optimizer.optimize(image_path, features)

        # Convert using spatially optimized parameters
        svg_content = self._convert_with_optimized_params(
            image_path, optimization_result['parameters'], **kwargs
        )

        return {
            'svg_content': svg_content,
            'method': 'method3',
            'optimization_type': 'adaptive_spatial',
            'optimization_result': optimization_result
        }

    def _calculate_complexity_score(self, features: Dict[str, float]) -> float:
        """Calculate normalized complexity score from features"""
        # Weighted combination of complexity indicators
        complexity = (
            features.get('edge_density', 0.5) * 0.3 +
            features.get('unique_colors', 0.5) * 0.2 +
            features.get('entropy', 0.5) * 0.2 +
            features.get('corner_density', 0.5) * 0.2 +
            features.get('gradient_strength', 0.5) * 0.1
        )
        return max(0.0, min(1.0, complexity))

    def _select_fastest_method(self, complexity: float, logo_type: str) -> str:
        """Select method optimized for speed"""
        # Method 1 is always fastest
        return 'method1'

    def _select_highest_quality_method(self, complexity: float, logo_type: str) -> str:
        """Select method optimized for quality"""
        thresholds = self.routing_config['complexity_thresholds']

        if complexity < thresholds['simple']:
            return 'method1'  # Simple images work well with method1
        elif complexity > thresholds['medium']:
            return 'method3'  # Complex images need spatial optimization
        else:
            # Medium complexity - prefer method2 if available, else method3
            return 'method2' if self._is_method_available('method2') else 'method3'

    def _select_balanced_method(self, complexity: float, logo_type: str, max_time: float) -> str:
        """Select method with optimal quality/time balance"""
        thresholds = self.routing_config['complexity_thresholds']

        # Consider time constraints
        if max_time < 1.0:
            return 'method1'  # Very tight time constraint
        elif max_time < 10.0:
            # Moderate time constraint - avoid method3 for complex images
            if complexity < thresholds['medium']:
                return 'method1'
            else:
                return 'method2' if self._is_method_available('method2') else 'method1'
        else:
            # Generous time allowance - optimize for quality
            return self._select_highest_quality_method(complexity, logo_type)

    def _apply_learned_preferences(self, selected_method: str, logo_type: str,
                                 features: Dict[str, float]) -> str:
        """Apply learned preferences based on historical performance"""

        # Get performance statistics for this logo type
        logo_performance = self._get_logo_type_performance(logo_type)

        # If we have enough data and current selection isn't performing well
        if (logo_performance and
            selected_method in logo_performance and
            logo_performance[selected_method]['avg_quality'] < self.routing_config['min_quality_improvement']):

            # Try to find a better performing method for this logo type
            best_method = max(logo_performance.keys(),
                            key=lambda m: logo_performance[m]['avg_quality'])

            if (self._is_method_available(best_method) and
                logo_performance[best_method]['avg_quality'] > logo_performance[selected_method]['avg_quality']):

                self.logger.info(f"Learning override: switching from {selected_method} to {best_method} for {logo_type}")
                return best_method

        return selected_method

    def _is_method_available(self, method: str) -> bool:
        """Check if a method is available for use"""
        return self.method_availability.get(method, False)

    def _select_fallback_method(self, unavailable_method: str, complexity: float, logo_type: str) -> str:
        """Select fallback method when preferred method is unavailable"""

        # Get fallback preferences for this logo type
        preferences = self.method_preferences.get(logo_type, ['method1', 'method3', 'method2'])

        # Remove the unavailable method from preferences
        available_preferences = [m for m in preferences if self._is_method_available(m) and m != unavailable_method]

        if available_preferences:
            return available_preferences[0]

        # Final fallback - always available
        return 'method1'

    def _calculate_routing_confidence(self, method: str, features: Dict[str, float], logo_type: str) -> float:
        """Calculate confidence in routing decision based on historical performance"""

        stats = self.method_performance.get(method)
        if not stats or stats.count < 5:
            return 0.7  # Default confidence for new/untested methods

        # Base confidence on success rate and quality performance
        confidence = (stats.success_rate * 0.6) + (min(stats.avg_quality, 0.5) * 0.4)

        # Adjust based on complexity matching
        complexity = self._calculate_complexity_score(features)
        if method == 'method1' and complexity < 0.3:
            confidence += 0.1
        elif method == 'method3' and complexity > 0.7:
            confidence += 0.1
        elif method == 'method2' and 0.3 <= complexity <= 0.7:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _generate_routing_reasoning(self, method: str, complexity: float, logo_type: str, quality_mode: str) -> str:
        """Generate human-readable reasoning for routing decision"""

        reasons = []

        # Complexity-based reasoning
        if complexity < 0.3:
            reasons.append(f"Low complexity ({complexity:.2f}) suits mathematical correlation")
        elif complexity > 0.7:
            reasons.append(f"High complexity ({complexity:.2f}) benefits from spatial optimization")
        else:
            reasons.append(f"Medium complexity ({complexity:.2f}) allows flexible method selection")

        # Logo type reasoning
        if logo_type == 'simple' and method == 'method1':
            reasons.append("Simple geometric shapes work well with correlation mapping")
        elif logo_type == 'complex' and method == 'method3':
            reasons.append("Complex designs benefit from adaptive spatial analysis")
        elif logo_type == 'gradient' and method == 'method3':
            reasons.append("Gradient handling requires spatial optimization")

        # Quality mode reasoning
        if quality_mode == 'fast' and method == 'method1':
            reasons.append("Fast processing mode prioritizes Method 1")
        elif quality_mode == 'quality' and method in ['method2', 'method3']:
            reasons.append("Quality mode prioritizes advanced optimization methods")

        # Method availability
        if method == 'method1' and not self._is_method_available('method2'):
            reasons.append("PPO method unavailable, using reliable correlation method")

        return "; ".join(reasons) if reasons else f"Default routing to {method}"

    def _estimate_quality_improvement(self, method: str, features: Dict[str, float]) -> float:
        """Estimate expected quality improvement for method"""

        stats = self.method_performance.get(method)
        if stats and stats.count > 0:
            return stats.avg_quality

        # Default estimates based on method characteristics
        complexity = self._calculate_complexity_score(features)

        if method == 'method1':
            return 0.15 + (0.1 * (1 - complexity))  # Better on simpler images
        elif method == 'method2':
            return 0.25  # Consistent performance target
        elif method == 'method3':
            return 0.35 + (0.1 * complexity)  # Better on complex images

        return 0.10  # Conservative default

    def _estimate_processing_time(self, method: str, features: Dict[str, float]) -> float:
        """Estimate processing time for method"""

        stats = self.method_performance.get(method)
        if stats and stats.count > 0:
            return stats.avg_time

        # Default time estimates
        complexity = self._calculate_complexity_score(features)

        if method == 'method1':
            return 0.05 + (0.05 * complexity)
        elif method == 'method2':
            return 2.0 + (3.0 * complexity)
        elif method == 'method3':
            return 10.0 + (20.0 * complexity)

        return 1.0  # Conservative default

    def _get_fallback_methods(self, primary_method: str) -> List[str]:
        """Get ordered list of fallback methods"""

        all_methods = ['method1', 'method2', 'method3']

        # Remove primary method
        fallbacks = [m for m in all_methods if m != primary_method and self._is_method_available(m)]

        # Ensure method1 is always last fallback (most reliable)
        if 'method1' in fallbacks:
            fallbacks.remove('method1')
            fallbacks.append('method1')

        return fallbacks

    def _update_method_performance(self, routing_decision: RoutingDecision,
                                 result: Dict[str, Any], processing_time: float):
        """Update performance tracking for the used method"""

        method = routing_decision.method
        success = result.get('success', False)

        # Estimate quality improvement (would need actual measurement in practice)
        estimated_improvement = routing_decision.expected_improvement

        # Update method statistics
        if method in self.method_performance:
            self.method_performance[method].update(estimated_improvement, processing_time, success)

        # Store routing history for learning
        routing_record = {
            'timestamp': time.time(),
            'method': method,
            'success': success,
            'processing_time': processing_time,
            'routing_decision': routing_decision.__dict__,
            'result_summary': {k: v for k, v in result.items() if k not in ['svg_content']}
        }

        self.routing_history.append(routing_record)

        # Limit history size
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-500:]

    def _get_logo_type_performance(self, logo_type: str) -> Dict[str, Dict[str, float]]:
        """Get performance statistics by logo type from routing history"""

        type_performance = {}

        # Filter history by logo type (would need logo type tracking in practice)
        relevant_records = [r for r in self.routing_history if r.get('logo_type') == logo_type]

        # Calculate performance by method
        for method in ['method1', 'method2', 'method3']:
            method_records = [r for r in relevant_records if r['method'] == method]

            if method_records:
                avg_quality = sum(r.get('estimated_improvement', 0) for r in method_records) / len(method_records)
                avg_time = sum(r['processing_time'] for r in method_records) / len(method_records)
                success_rate = sum(1 for r in method_records if r['success']) / len(method_records)

                type_performance[method] = {
                    'avg_quality': avg_quality,
                    'avg_time': avg_time,
                    'success_rate': success_rate,
                    'count': len(method_records)
                }

        return type_performance

    def _fallback_conversion(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Final fallback conversion when all intelligent methods fail"""

        try:
            # Use the base AIEnhancedConverter as final fallback
            svg_content = super().convert(image_path, **kwargs)

            return {
                'svg_content': svg_content,
                'method': 'fallback_method1',
                'success': True,
                'routing_decision': {
                    'method': 'fallback',
                    'reasoning': 'All intelligent routing failed, using base converter'
                },
                'intelligent_routing': False
            }

        except Exception as e:
            self.logger.error(f"Final fallback conversion failed: {e}")
            raise RuntimeError(f"Complete conversion failure for {image_path}: {e}")

    def _classify_image(self, image_path: str) -> str:
        """Classify image type for routing decisions"""
        # Use inherited classification logic from AIEnhancedConverter
        try:
            features = self.feature_extractor.extract_features(image_path)
            return self._infer_logo_type(features)
        except Exception as e:
            self.logger.warning(f"Image classification failed: {e}")
            return 'complex'  # Conservative default

    # Additional methods for comprehensive API

    def get_method_availability(self) -> Dict[str, bool]:
        """Get current method availability status"""
        return self.method_availability.copy()

    def get_method_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed performance statistics for all methods"""
        stats = {}
        for method, perf in self.method_performance.items():
            stats[method] = {
                'count': perf.count,
                'avg_quality': perf.avg_quality,
                'avg_time': perf.avg_time,
                'success_rate': perf.success_rate,
                'success_count': perf.success_count,
                'failure_count': perf.failure_count
            }
        return stats

    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get routing analytics and insights"""
        if not self.routing_history:
            return {'message': 'No routing history available'}

        # Calculate method usage distribution
        method_usage = {}
        for record in self.routing_history:
            method = record['method']
            method_usage[method] = method_usage.get(method, 0) + 1

        # Calculate average performance by method
        method_performance = {}
        for method in method_usage.keys():
            method_records = [r for r in self.routing_history if r['method'] == method]
            if method_records:
                avg_time = sum(r['processing_time'] for r in method_records) / len(method_records)
                success_rate = sum(1 for r in method_records if r['success']) / len(method_records)
                method_performance[method] = {
                    'avg_time': avg_time,
                    'success_rate': success_rate,
                    'usage_count': len(method_records)
                }

        return {
            'total_conversions': len(self.routing_history),
            'method_usage_distribution': method_usage,
            'method_performance': method_performance,
            'recent_success_rate': sum(1 for r in self.routing_history[-100:] if r['success']) / min(100, len(self.routing_history))
        }

    def configure_routing(self, **config_updates):
        """Update routing configuration"""
        for key, value in config_updates.items():
            if key in self.routing_config:
                old_value = self.routing_config[key]
                self.routing_config[key] = value
                self.logger.info(f"Routing configuration updated: {key} = {value} (was {old_value})")
            else:
                self.logger.warning(f"Unknown routing configuration key: {key}")

    def reset_method_performance(self):
        """Reset method performance tracking"""
        for method in self.method_performance:
            self.method_performance[method] = MethodPerformanceStats()
        self.routing_history.clear()
        self.logger.info("Method performance tracking reset")

    def export_routing_history(self) -> List[Dict[str, Any]]:
        """Export routing history for analysis"""
        return self.routing_history.copy()

    # Batch processing with intelligent routing
    def batch_convert_intelligent(self, image_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Convert multiple images with intelligent routing optimization"""
        results = []

        # Analyze all images first for batch optimization
        self.logger.info(f"Starting intelligent batch conversion of {len(image_paths)} images")

        for i, image_path in enumerate(image_paths):
            try:
                self.logger.info(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
                result = self.convert(image_path, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch conversion failed for {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'success': False,
                    'error': str(e),
                    'method_used': 'none'
                })

        # Generate batch summary
        batch_summary = self._generate_batch_summary(results)

        return {
            'results': results,
            'batch_summary': batch_summary
        }

    def _generate_batch_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for batch processing"""

        successful_results = [r for r in results if r.get('success', False)]

        if not successful_results:
            return {
                'total_images': len(results),
                'successful_conversions': 0,
                'success_rate': 0.0,
                'message': 'No successful conversions'
            }

        # Method usage statistics
        method_usage = {}
        total_time = 0

        for result in successful_results:
            method = result.get('method_used', 'unknown')
            method_usage[method] = method_usage.get(method, 0) + 1
            total_time += result.get('total_processing_time', 0)

        return {
            'total_images': len(results),
            'successful_conversions': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'total_processing_time': total_time,
            'avg_processing_time': total_time / len(successful_results),
            'method_usage': method_usage,
            'most_used_method': max(method_usage.keys(), key=method_usage.get) if method_usage else 'none'
        }