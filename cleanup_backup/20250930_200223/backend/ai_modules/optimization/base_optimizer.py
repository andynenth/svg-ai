# backend/ai_modules/optimization/base_optimizer.py
"""Base class for parameter optimization"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
import time
from backend.ai_modules.config import VTRACER_PARAM_RANGES, DEFAULT_VTRACER_PARAMS

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """Base class for VTracer parameter optimization"""

    def __init__(self, name: str):
        self.name = name
        self.optimization_history = []
        self.param_ranges = VTRACER_PARAM_RANGES
        self.default_params = DEFAULT_VTRACER_PARAMS

    @abstractmethod
    def _optimize_impl(self, features: Dict[str, float], logo_type: str) -> Dict[str, Any]:
        """Implement actual optimization logic"""
        pass

    def optimize(self, features: Dict[str, float], logo_type: str = None) -> Dict[str, Any]:
        """Optimize VTracer parameters with error handling"""
        start_time = time.time()

        try:
            # Infer logo type if not provided
            if logo_type is None:
                logo_type = self._infer_logo_type(features)

            # Run optimization
            optimized_params = self._optimize_impl(features, logo_type)

            # Validate parameters
            validated_params = self._validate_parameters(optimized_params)

            # Record optimization
            optimization_time = time.time() - start_time
            self.optimization_history.append(
                {
                    "timestamp": time.time(),
                    "features": features,
                    "logo_type": logo_type,
                    "parameters": validated_params,
                    "optimization_time": optimization_time,
                    "optimizer": self.name,
                }
            )

            logger.info(f"{self.name} optimization completed in {optimization_time:.3f}s")
            return validated_params

        except Exception as e:
            logger.error(f"Optimization failed with {self.name}: {e}")
            # Return default parameters for the logo type
            return self._get_default_parameters(logo_type or "simple")

    def _infer_logo_type(self, features: Dict[str, float]) -> str:
        """Simple logo type inference from features"""
        complexity = features.get("complexity_score", 0.5)
        unique_colors = features.get("unique_colors", 16)
        edge_density = features.get("edge_density", 0.1)

        if complexity < 0.3 and unique_colors <= 4:
            return "simple"
        elif edge_density > 0.5 and unique_colors <= 8:
            return "text"
        elif unique_colors > 20:
            return "gradient"
        else:
            return "complex"

    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and constrain parameters to valid ranges"""
        validated = {}

        for param_name, value in params.items():
            if param_name in self.param_ranges:
                min_val, max_val = self.param_ranges[param_name]
                # Constrain to valid range
                constrained_value = max(min_val, min(max_val, value))
                validated[param_name] = constrained_value

                if constrained_value != value:
                    logger.warning(
                        f"Parameter {param_name} constrained from {value} to {constrained_value}"
                    )
            else:
                validated[param_name] = value

        return validated

    def _get_default_parameters(self, logo_type: str) -> Dict[str, Any]:
        """Get default parameters for a logo type"""
        if logo_type in self.default_params:
            return self.default_params[logo_type].copy()
        else:
            logger.warning(f"Unknown logo type {logo_type}, using 'simple' defaults")
            return self.default_params["simple"].copy()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.optimization_history:
            return {"total_optimizations": 0}

        times = [opt["optimization_time"] for opt in self.optimization_history]
        logo_types = [opt["logo_type"] for opt in self.optimization_history]

        return {
            "total_optimizations": len(self.optimization_history),
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "logo_type_distribution": {
                logo_type: logo_types.count(logo_type) for logo_type in set(logo_types)
            },
        }
