## File: meta_snake.py
import numpy as np
import random
import contextlib
import io
from snake.policy_iter.main import run_genetic_snake

PARAM_BOUNDS = {
    "MUTATION_RATE": (0.01, 0.3),
    "FOOD_MULTIPLE": (1, 10),
    "STEPS_MULTIPLE": (0.001, 0.1),
    "VISITED_MULTIPLE": (0.001, 0.1)
}

POP_SIZE = 10
GENERATIONS = 5


def random_config():
    return {
        "POP_SIZE": 100,
        "GENS": 5,
        "MUTATION_RATE": random.uniform(*PARAM_BOUNDS["MUTATION_RATE"]),
        "FOOD_MULTIPLE": random.uniform(*PARAM_BOUNDS["FOOD_MULTIPLE"]),
        "STEPS_MULTIPLE": random.uniform(*PARAM_BOUNDS["STEPS_MULTIPLE"]),
        "VISITED_MULTIPLE": random.uniform(*PARAM_BOUNDS["VISITED_MULTIPLE"])
    }


def mutate_config(config):
    mutated = config.copy()
    for key in PARAM_BOUNDS:
        if random.random() < 0.5:
            lb, ub = PARAM_BOUNDS[key]
            mutated[key] = np.clip(mutated[key] + random.uniform(-0.1, 0.1) * (ub - lb), lb, ub)
    return mutated


def crossover_configs(c1, c2):
    return {
        k: random.choice([c1[k], c2[k]])
        for k in c1
    }


def normalize_fitness(fitnesses):
    fitnesses = np.array(fitnesses)
    return (fitnesses - np.min(fitnesses)) / (np.max(fitnesses) - np.min(fitnesses) + 1e-8)


def run_silently(config):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _, hist, _, _, _, _ = run_genetic_snake(config)
    # Use the average change in fitness to define performance
    diffs = np.diff(hist)
    trend_score = np.mean(diffs)
    return trend_score


def meta_optimize():
    population = [random_config() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        print(f"\n--- Meta Generation {gen+1}/{GENERATIONS} ---")
        results = []

        for cfg in population:
            trend_score = run_silently(cfg)
            results.append((cfg, trend_score))

        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:POP_SIZE // 2]

        print(f"Top Config Trend Score: {top[0][1]:.4f}")
        for key in top[0][0]:
            print(f"  {key}: {top[0][0][key]:.4f}")

        fitness_scores = normalize_fitness([fit for _, fit in top])

        new_population = [top[0][0]]  # Elitism
        while len(new_population) < POP_SIZE:
            parents = random.choices([cfg for cfg, _ in top], weights=fitness_scores, k=2)
            child = mutate_config(crossover_configs(parents[0], parents[1]))
            new_population.append(child)

        population = new_population

    return top[0][0]  # Return best config


if __name__ == "__main__":
    best_config = meta_optimize()
    print("\nBest Config After Meta-Optimization:")
    for k, v in best_config.items():
        print(f"{k}: {v:.4f}")
