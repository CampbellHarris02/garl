import pygame # type: ignore
import numpy as np
import random
import sys
import math
from collections import defaultdict, deque
import matplotlib.pyplot as plt # type: ignore
from rich.progress import track
from rich.console import Console
from rich.table import Table

from snake.mamal import mutate, crossover

GRID_SIZE = 8
SCREEN_SIZE = 600
TILE_SIZE = SCREEN_SIZE // GRID_SIZE
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACTIONS = [-1, 0, 1]

POP_SIZE = 1000
GENS = 20
MUTATION_RATE = 0.1
STEPS_PER_GAME = 1000
FOOD_VALUE = 5

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = deque([(5, 5)])
        self.direction = random.randint(0, 3)
        self.place_food()
        self.score = 0
        self.steps = 0
        self.head_history = deque(maxlen=24)
        return self.get_state()

    def place_food(self):
        while True:
            self.food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if self.food not in self.snake:
                break

    def step(self, action):
        self.direction = (self.direction + action) % 4
        dx, dy = DIRS[self.direction]
        head = (self.snake[0][0] + dx, self.snake[0][1] + dy)

        self.steps += 1
        self.head_history.appendleft(head)

        if len(self.snake) < 3 and len(self.head_history) >= 12:
            recent = list(self.head_history)
            for window in [4, 6]:
                if len(recent) >= 2 * window and recent[:window] == recent[window:2*window]:
                    return self.get_state(), True

        if head[0] < 0 or head[0] >= GRID_SIZE or head[1] < 0 or head[1] >= GRID_SIZE or head in self.snake:
            return self.get_state(), True

        self.snake.appendleft(head)
        if head == self.food:
            self.score += FOOD_VALUE
            self.place_food()
        else:
            self.snake.pop()

        return self.get_state(), False

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_dx = np.sign(self.food[0] - head_x)
        food_dy = np.sign(self.food[1] - head_y)

        def danger(offset):
            dx, dy = DIRS[(self.direction + offset) % 4]
            nx, ny = head_x + dx, head_y + dy
            return int(nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE or (nx, ny) in self.snake)

        return (
            food_dx, food_dy,
            self.direction,
            danger(0), danger(1), danger(-1),
            min(len(self.snake), 20),  # Discretized snake length
            head_x, head_y
        )

def random_policy():
    policy = defaultdict(lambda: random.choice(ACTIONS))
    game = SnakeGame()
    state = game.reset()
    visited_states = set()

    for _ in range(STEPS_PER_GAME):
        visited_states.add(state)
        action = random.choice(ACTIONS)
        policy[state] = action
        state, done = game.step(action)
        if done:
            break

    return policy, visited_states

def evaluate_policy(policy):
    game = SnakeGame()
    state = game.reset()
    visited = set([state])
    food_collected = 0
    for _ in range(STEPS_PER_GAME):
        prev_score = game.score
        action = policy[state]
        state, done = game.step(action)
        visited.add(state)
        if game.score > prev_score:
            food_collected += 1
        if done:
            break
    fitness = 0.01 * game.steps + 5 * food_collected
    return fitness, visited


def policy_distance(p1, p2):
    keys = set(p1.keys()).union(set(p2.keys()))
    diff = sum(p1[k] != p2[k] for k in keys)
    return diff / len(keys) if keys else 0.0

def run_genetic_snake():
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
            f, visited = evaluate_policy(p)
            fitnesses.append(f)
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

# Run everything
if __name__ == "__main__":
    best, hist, hist_top, explored_states, avg_hist, std_hist = run_genetic_snake()

    # Plot best fitness per generation
    plt.plot(hist)
    plt.title("Best Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("log(steps) + score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot top 10 individuals' fitness history
    hist_top_array = np.array(hist_top)
    for i in range(10):
        plt.plot(hist_top_array[:, i], label=f'Top {i+1}', alpha=0.8)
    plt.title("Top 10 Individual Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot explored states by best policy
    plt.plot(explored_states)
    plt.title("Best Individual's Explored States per Generation")
    plt.xlabel("Generation")
    plt.ylabel("# Unique States Visited")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot average and standard deviation of fitness
    plt.figure(figsize=(10, 5))
    plt.plot(avg_hist, label="Average Fitness", color="steelblue", linewidth=2)
    plt.fill_between(range(len(std_hist)), np.array(avg_hist) - np.array(std_hist), np.array(avg_hist) + np.array(std_hist), color="skyblue", alpha=0.3, label="±1 Std Dev")
    plt.title("Population Fitness Statistics", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    simulate(best)
