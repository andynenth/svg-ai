#!/usr/bin/env python3
"""Genetic Algorithm Setup Verification"""

import random
from deap import base, creator, tools, algorithms
import numpy as np

def test_ga_setup():
    """Test basic GA functionality"""
    print("🧬 Testing GA setup...")

    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    print("✅ DEAP classes created")

    # Initialize toolbox
    toolbox = base.Toolbox()
    print("✅ Toolbox initialized")

    # Test parameter optimization simulation (VTracer parameters)
    # VTracer has 8 main parameters, we'll optimize a subset
    def create_individual():
        """Create individual representing VTracer parameters"""
        return [
            random.uniform(1, 10),    # color_precision (1-10)
            random.uniform(10, 100),  # corner_threshold (10-100)
            random.uniform(5, 50),    # path_precision (5-50)
            random.uniform(1, 10)     # layer_difference (1-10)
        ]

    def evaluate_individual(individual):
        """Dummy fitness function (would be SVG quality in real implementation)"""
        # Simulate quality score based on parameter balance
        color_prec, corner_thresh, path_prec, layer_diff = individual

        # Dummy fitness: penalize extreme values, reward balance
        fitness = 1.0
        if color_prec < 2 or color_prec > 8:
            fitness -= 0.2
        if corner_thresh < 20 or corner_thresh > 80:
            fitness -= 0.2
        if path_prec < 10 or path_prec > 40:
            fitness -= 0.2
        if layer_diff < 2 or layer_diff > 8:
            fitness -= 0.2

        # Add some noise to simulate real quality measurement
        fitness += random.uniform(-0.1, 0.1)
        return (max(0.0, fitness),)  # Ensure non-negative

    # Register functions
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    print("✅ GA operators registered")

    # Test population creation
    population = toolbox.population(n=10)
    print(f"✅ Population created: {len(population)} individuals")

    # Test evaluation
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    print(f"✅ Population evaluated: avg fitness = {np.mean([ind.fitness.values[0] for ind in population]):.3f}")

    # Test genetic operations
    offspring = toolbox.select(population, k=5)
    offspring = list(map(toolbox.clone, offspring))

    # Test crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Test mutation
    for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    print("✅ Genetic operations (crossover, mutation) tested")

    # Test evolution step
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print("✅ Evolution step completed")

    # Test that we can find best individual
    best_ind = tools.selBest(population, k=1)[0]
    print(f"✅ Best individual: {best_ind} (fitness: {best_ind.fitness.values[0]:.3f})")

    return True

if __name__ == "__main__":
    try:
        success = test_ga_setup()
        if success:
            print("🎉 GA setup test completed successfully!")
        else:
            print("❌ GA setup test failed")
            exit(1)
    except Exception as e:
        print(f"❌ GA setup test error: {e}")
        exit(1)