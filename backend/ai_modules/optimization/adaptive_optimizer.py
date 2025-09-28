# backend/ai_modules/optimization/adaptive_optimizer.py
"""Adaptive parameter optimization using multiple strategies"""

import numpy as np
from typing import Dict, Any
import logging
from .base_optimizer import BaseOptimizer
from .feature_mapping import FeatureMappingOptimizer
from backend.ai_modules.config import GA_CONFIG

logger = logging.getLogger(__name__)


class AdaptiveOptimizer(BaseOptimizer):
    """Adaptive optimizer that combines multiple optimization strategies"""

    def __init__(self):
        super().__init__("Adaptive")
        self.feature_mapper = FeatureMappingOptimizer()
        self.strategy_performance = {
            "feature_mapping": {"total_uses": 0, "avg_quality": 0.0},
            "genetic_algorithm": {"total_uses": 0, "avg_quality": 0.0},
            "grid_search": {"total_uses": 0, "avg_quality": 0.0},
            "random_search": {"total_uses": 0, "avg_quality": 0.0},
        }
        self.optimization_history = []

    def _optimize_impl(self, features: Dict[str, float], logo_type: str) -> Dict[str, Any]:
        """Implement adaptive optimization by selecting best strategy"""
        logger.debug(f"Running adaptive optimization for {logo_type}")

        try:
            # Select optimization strategy based on performance and context
            strategy = self._select_optimization_strategy(features, logo_type)

            # Run selected strategy
            if strategy == "feature_mapping":
                result = self._run_feature_mapping(features, logo_type)
            elif strategy == "genetic_algorithm":
                result = self._run_genetic_algorithm(features, logo_type)
            elif strategy == "grid_search":
                result = self._run_grid_search(features, logo_type)
            else:  # random_search
                result = self._run_random_search(features, logo_type)

            # Record strategy use
            self.strategy_performance[strategy]["total_uses"] += 1

            logger.debug(f"Adaptive optimization used {strategy}: {result}")
            return result

        except Exception as e:
            logger.error(f"Adaptive optimization failed: {e}")
            return self._get_default_parameters(logo_type)

    def _select_optimization_strategy(self, features: Dict[str, float], logo_type: str) -> str:
        """Select the best optimization strategy based on context and performance"""
        complexity = features.get("complexity_score", 0.5)
        unique_colors = features.get("unique_colors", 16)

        # Strategy selection logic
        if complexity < 0.3 and unique_colors <= 8:
            # Simple images: use feature mapping (fast and effective)
            return "feature_mapping"
        elif complexity > 0.7 or unique_colors > 30:
            # Complex images: use genetic algorithm (thorough search)
            return "genetic_algorithm"
        elif self.strategy_performance["feature_mapping"]["total_uses"] < 5:
            # Not enough data: try feature mapping first
            return "feature_mapping"
        else:
            # Choose based on historical performance
            best_strategy = max(
                self.strategy_performance.keys(),
                key=lambda s: self.strategy_performance[s]["avg_quality"],
            )
            return best_strategy

    def _run_feature_mapping(self, features: Dict[str, float], logo_type: str) -> Dict[str, Any]:
        """Run feature mapping optimization"""
        return self.feature_mapper._optimize_impl(features, logo_type)

    def _run_genetic_algorithm(self, features: Dict[str, float], logo_type: str) -> Dict[str, Any]:
        """Run genetic algorithm optimization"""
        try:
            from deap import base, creator, tools, algorithms
            import random

            # Create fitness and individual classes (avoid duplicate creation)
            if not hasattr(creator, "FitnessMax"):
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.FitnessMax)

            # Setup GA toolbox
            toolbox = base.Toolbox()

            # Individual creation
            def create_individual():
                return [
                    random.uniform(*self.param_ranges["color_precision"]),
                    random.uniform(*self.param_ranges["corner_threshold"]),
                    random.uniform(*self.param_ranges["path_precision"]),
                    random.uniform(*self.param_ranges["layer_difference"]),
                    random.uniform(*self.param_ranges["splice_threshold"]),
                    random.uniform(*self.param_ranges["filter_speckle"]),
                    random.uniform(*self.param_ranges["segment_length"]),
                    random.uniform(*self.param_ranges["max_iterations"]),
                ]

            def evaluate_individual(individual):
                """Fitness function based on feature compatibility"""
                params = {
                    "color_precision": individual[0],
                    "corner_threshold": individual[1],
                    "path_precision": individual[2],
                    "layer_difference": individual[3],
                    "splice_threshold": individual[4],
                    "filter_speckle": individual[5],
                    "segment_length": individual[6],
                    "max_iterations": individual[7],
                }

                # Simple fitness based on parameter appropriateness
                fitness = self._evaluate_parameter_fitness(params, features, logo_type)
                return (fitness,)

            # Register functions
            toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", evaluate_individual)
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Run GA
            population = toolbox.population(n=GA_CONFIG["population_size"])

            # Evaluate initial population
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Evolution
            for generation in range(min(10, GA_CONFIG["generations"])):  # Limited for Phase 1
                # Select parents
                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))

                # Crossover and mutation
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < GA_CONFIG["crossover_prob"]:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if random.random() < GA_CONFIG["mutation_prob"]:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluate invalid individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Replace population
                population[:] = offspring

            # Get best solution
            best_individual = tools.selBest(population, k=1)[0]

            # Convert to parameter dictionary
            param_names = [
                "color_precision",
                "corner_threshold",
                "path_precision",
                "layer_difference",
                "splice_threshold",
                "filter_speckle",
                "segment_length",
                "max_iterations",
            ]

            best_params = dict(zip(param_names, best_individual))
            return self._validate_parameters(best_params)

        except Exception as e:
            logger.error(f"Genetic algorithm optimization failed: {e}")
            return self._get_default_parameters(logo_type)

    def _run_grid_search(self, features: Dict[str, float], logo_type: str) -> Dict[str, Any]:
        """Run grid search optimization (simplified for Phase 1)"""
        try:
            # Start with defaults
            base_params = self._get_default_parameters(logo_type)
            best_params = base_params.copy()
            best_fitness = self._evaluate_parameter_fitness(best_params, features, logo_type)

            # Test variations of key parameters
            key_params = ["color_precision", "corner_threshold", "path_precision"]

            for param_name in key_params:
                if param_name in self.param_ranges:
                    min_val, max_val = self.param_ranges[param_name]

                    # Test 3 values: low, medium, high
                    test_values = [
                        min_val + 0.2 * (max_val - min_val),
                        min_val + 0.5 * (max_val - min_val),
                        min_val + 0.8 * (max_val - min_val),
                    ]

                    for test_val in test_values:
                        test_params = best_params.copy()
                        test_params[param_name] = test_val

                        fitness = self._evaluate_parameter_fitness(test_params, features, logo_type)

                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_params[param_name] = test_val

            return best_params

        except Exception as e:
            logger.error(f"Grid search optimization failed: {e}")
            return self._get_default_parameters(logo_type)

    def _run_random_search(self, features: Dict[str, float], logo_type: str) -> Dict[str, Any]:
        """Run random search optimization"""
        try:
            import random

            base_params = self._get_default_parameters(logo_type)
            best_params = base_params.copy()
            best_fitness = self._evaluate_parameter_fitness(best_params, features, logo_type)

            # Try random variations
            for _ in range(20):  # Limited iterations for Phase 1
                test_params = {}

                for param_name, (min_val, max_val) in self.param_ranges.items():
                    if param_name in base_params:
                        test_params[param_name] = random.uniform(min_val, max_val)

                fitness = self._evaluate_parameter_fitness(test_params, features, logo_type)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = test_params.copy()

            return best_params

        except Exception as e:
            logger.error(f"Random search optimization failed: {e}")
            return self._get_default_parameters(logo_type)

    def _evaluate_parameter_fitness(
        self, params: Dict[str, Any], features: Dict[str, float], logo_type: str
    ) -> float:
        """Evaluate fitness of parameter set (heuristic for Phase 1)"""
        try:
            fitness = 0.7  # Base fitness

            complexity = features.get("complexity_score", 0.5)
            unique_colors = features.get("unique_colors", 16)
            edge_density = features.get("edge_density", 0.1)

            # Color precision fitness
            color_prec = params.get("color_precision", 5)
            if unique_colors <= 8 and color_prec <= 4:
                fitness += 0.1  # Good for simple images
            elif unique_colors > 20 and color_prec >= 6:
                fitness += 0.1  # Good for complex images

            # Corner threshold fitness
            corner_thresh = params.get("corner_threshold", 50)
            if edge_density > 0.3 and corner_thresh <= 30:
                fitness += 0.1  # Good for high edge density
            elif edge_density < 0.1 and corner_thresh >= 60:
                fitness += 0.1  # Good for low edge density

            # Path precision fitness
            path_prec = params.get("path_precision", 15)
            if complexity > 0.7 and path_prec >= 20:
                fitness += 0.1  # Good for complex images
            elif complexity < 0.3 and path_prec <= 10:
                fitness += 0.1  # Good for simple images

            # Add some randomness to avoid local optima
            fitness += np.random.normal(0, 0.05)

            return max(0.0, min(1.0, fitness))

        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return 0.5

    def update_strategy_performance(self, strategy: str, quality: float):
        """Update performance tracking for a strategy"""
        if strategy in self.strategy_performance:
            current_avg = self.strategy_performance[strategy]["avg_quality"]
            total_uses = self.strategy_performance[strategy]["total_uses"]

            # Update running average
            new_avg = (current_avg * total_uses + quality) / (total_uses + 1)
            self.strategy_performance[strategy]["avg_quality"] = new_avg

    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptive optimization"""
        return {
            "strategy_performance": self.strategy_performance,
            "total_optimizations": sum(
                stats["total_uses"] for stats in self.strategy_performance.values()
            ),
            "best_strategy": (
                max(
                    self.strategy_performance.keys(),
                    key=lambda s: self.strategy_performance[s]["avg_quality"],
                )
                if any(s["total_uses"] > 0 for s in self.strategy_performance.values())
                else None
            ),
        }
