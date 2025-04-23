## File: main.py
from snake.policy_iter.policy_utils import random_policy, evaluate_policy
from snake.policy_iter.constants import *
from snake.policy_iter.snake_game import SnakeGame
from mamal import mutate, crossover
from snake.policy_iter.visualize import plot_fitness, plot_top_individuals, plot_exploration, plot_statistics
from rich.console import Console
from rich.progress import track
from rich.table import Table
import numpy as np
import pygame  # type: ignore

def run_genetic_snake(config):
    POP_SIZE = config["POP_SIZE"]
    GENS = config["GENS"]
    MUTATION_RATE = config["MUTATION_RATE"]
    FOOD_MULTIPLE = config["FOOD_MULTIPLE"]
    STEPS_MULTIPLE = config["STEPS_MULTIPLE"]
    VISITED_MULTIPLE = config["VISITED_MULTIPLE"]

    console = Console()
    population = []
    visited_sets = []
    for _ in range(POP_SIZE):
        policy, visited = random_policy()
        population.append(policy)
        visited_sets.append(visited)

    history = []
    history_top10 = []
    hist_explored_states = []
    hist_avg = []
    hist_std = []
    best_policy = None
    best_fitness = -float("inf")

    for gen in range(GENS):
        console.rule(f"[bold cyan]Generation {gen + 1}/{GENS}")
        fitnesses = []
        updated_visited_sets = []
        for p in track(population, description="Evaluating..."):
            game = SnakeGame()
            state = game.reset()
            visited = set([state])
            food_collected = 0
            for _ in range(STEPS_PER_GAME):
                prev_score = game.score
                action = p[state]
                state, done = game.step(action)
                visited.add(state)
                if game.score > prev_score:
                    food_collected += 1
                if done:
                    break
            fitness = (
                STEPS_MULTIPLE * game.steps +
                FOOD_MULTIPLE * food_collected +
                VISITED_MULTIPLE * len(visited)
            )
            fitnesses.append(fitness)
            updated_visited_sets.append(visited)

        elite_idx = np.argmax(fitnesses)
        best_policy = population[elite_idx]
        best_fitness = fitnesses[elite_idx]
        avg_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)

        new_population = [best_policy]
        new_visited_sets = [visited_sets[elite_idx]]

        console.log(f":trophy: [bold green]Best Fitness: {best_fitness:.4f}")
        console.log(f"[bold yellow]Best Policy Index: {elite_idx}")
        console.log(f"[bold blue]Explored States: {len(visited_sets[elite_idx])}")
        console.log(f":bar_chart: [bold cyan]Average Fitness: {avg_fitness:.4f}")
        console.log(f":chart_with_upwards_trend: [bold magenta]Std Dev Fitness: {std_fitness:.4f}")

        hist_explored_states.append(len(visited_sets[elite_idx]))
        hist_avg.append(avg_fitness)
        hist_std.append(std_fitness)

        ranked_indices = np.argsort(fitnesses)[::-1]
        used_partners = set()
        total_pair_fitness = 0
        pair_fitness_scores = []

        for i in ranked_indices:
            if i in used_partners:
                continue
            best_mate = max(
                (j for j in ranked_indices if j != i and j not in used_partners),
                key=lambda j: fitnesses[i] + fitnesses[j] + len(visited_sets[i] ^ visited_sets[j]),
                default=None
            )
            if best_mate is None:
                continue
            used_partners.update([i, best_mate])
            score = fitnesses[i] + fitnesses[best_mate]
            pair_fitness_scores.append((score, i, best_mate))
            total_pair_fitness += score

        for score, i, j in pair_fitness_scores:
            n_kids = max(1, round((score / total_pair_fitness) * POP_SIZE))
            for _ in range(n_kids):
                child = mutate(crossover(population[i], population[j], ACTIONS), ACTIONS)
                visited_union = visited_sets[i].union(visited_sets[j])
                new_population.append(child)
                new_visited_sets.append(visited_union)
                if len(new_population) >= POP_SIZE:
                    break
            if len(new_population) >= POP_SIZE:
                break

        while len(new_population) < POP_SIZE:
            policy, visited = random_policy()
            new_population.append(policy)
            new_visited_sets.append(visited)

        population = new_population
        visited_sets = new_visited_sets
        top_10 = sorted(fitnesses, reverse=True)[:10]
        history_top10.append(top_10)
        history.append(best_fitness)

    table = Table(title="Top Performer Policy (first 50 entries)")
    table.add_column("State", style="dim", overflow="fold")
    table.add_column("Action", justify="center")
    for i, (state, action) in enumerate(best_policy.items()):
        table.add_row(str(state), str(action))
        if i >= 49:
            break
    console.print(table)

    return best_policy, history, history_top10, hist_explored_states, hist_avg, hist_std

def simulate(policy):
    console = Console()
    console.print("\n[bold cyan]Simulating best policy...")
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    clock = pygame.time.Clock()
    game = SnakeGame()
    state = game.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        screen.fill((0, 0, 0))
        fx, fy = game.food
        pygame.draw.rect(screen, (255, 0, 0), (fx * TILE_SIZE, fy * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        for x, y in game.snake:
            pygame.draw.rect(screen, (0, 255, 0), (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        action = policy[state]
        state, done = game.step(action)

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
    print(f"Final score: {game.score}, Duration: {game.steps} steps")

if __name__ == "__main__":
    default_config = {
        "POP_SIZE": 1000,
        "GENS": 20,
        "MUTATION_RATE": 0.1,
        "FOOD_MULTIPLE": 5,
        "STEPS_MULTIPLE": 0.01,
        "VISITED_MULTIPLE": 0.02
    }
    best, hist, hist_top, explored_states, avg_hist, std_hist = run_genetic_snake(default_config)

    plot_fitness(hist)
    plot_top_individuals(hist_top)
    plot_exploration(explored_states)
    plot_statistics(avg_hist, std_hist)

    simulate(best)