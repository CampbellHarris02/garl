import pygame # type: ignore
import numpy as np
import random
import sys
import math
from collections import defaultdict, deque
import matplotlib.pyplot as plt # type: ignore
# make pretty
from rich.progress import track
from rich.console import Console
from rich.table import Table


# Game constants
GRID_SIZE = 8
SCREEN_SIZE = 600
TILE_SIZE = SCREEN_SIZE // GRID_SIZE
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACTIONS = [-1, 0, 1]  # left, forward, right

# GA params
POP_SIZE = 10000
GENS = 100
MUTATION_RATE = 0.01
ELITE_CUTOFF = 0.1
STEPS_PER_GAME = 10000

FOOD_VALUE = 1

# Game logic
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = deque([(5, 5)])
        self.direction = random.randint(0, 3)
        self.place_food()
        self.score = 0
        self.steps = 0
        self.head_history = deque(maxlen=24)  # track last head positions
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

        # Infinite loop detection (broader pattern check)
        if len(self.snake) < 3 and len(self.head_history) >= 12:
            recent = list(self.head_history)
            for window in [4, 6]:
                if len(recent) >= 2 * window and recent[:window] == recent[window:2*window]:
                    return self.get_state(), True  # Loop detected


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
            len(self.snake),
            head_x, head_y
        )


# Genetic structure
def random_policy():
    return defaultdict(lambda: random.choice(ACTIONS))

def evaluate_policy(policy):
    game = SnakeGame()
    state = game.reset()
    console = Console()
    for step in range(STEPS_PER_GAME):
        action = policy[state]
        state, done = game.step(action)
        if done:
            break

    return math.log(game.steps + 1) + game.score


def mutate(policy):
    new_policy = defaultdict(lambda: random.choice(ACTIONS))
    for k in policy:
        if random.random() < MUTATION_RATE:
            new_policy[k] = random.choice(ACTIONS)
        else:
            new_policy[k] = policy[k]
    return new_policy

def crossover(p1, p2):
    child = defaultdict(lambda: random.choice(ACTIONS))
    for k in set(p1.keys()).union(p2.keys()):
        child[k] = p1[k] if random.random() < 0.5 else p2[k]
    return child

def run_genetic_snake():
    console = Console()
    population = [random_policy() for _ in range(POP_SIZE)]
    history = []
    history_top10 = []
    best_policy = None
    best_fitness = -float("inf")
    

    for gen in range(GENS):
        console.rule(f"[bold cyan]Generation {gen + 1}/{GENS}")
        fitnesses = []
        for policy in track(population, description="Evaluating..."):
            fitnesses.append(evaluate_policy(policy))

        sorted_indices = np.argsort(fitnesses)[-int(ELITE_CUTOFF * POP_SIZE):]
        elites = [population[i] for i in sorted_indices]

        top_10 = sorted(fitnesses, reverse=True)[:10]
        history_top10.append(top_10)
        
        console.log(f"[bold green]Best fitness: {top_10[0]:.2f}")
        console.log(f"Top 10: {[f'{x:.2f}' for x in top_10]}")

        if top_10[0] > best_fitness:
            best_fitness = top_10[0]
            best_policy = population[fitnesses.index(best_fitness)]

        children = []
        while len(children) < POP_SIZE:
            parents = random.sample(elites, 2)
            child = mutate(crossover(parents[0], parents[1]))
            children.append(child)

        population = children
        history.append(best_fitness)

    # Log top policy entries
    table = Table(title="Top Performer Policy (first 50 entries)")
    table.add_column("State", style="dim", overflow="fold")
    table.add_column("Action", justify="center")

    for i, (state, action) in enumerate(best_policy.items()):
        table.add_row(str(state), str(action))
        if i >= 49:
            break
    console.print(table)

    return best_policy, history, history_top10


# Run best policy
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
    best, hist, hist_top = run_genetic_snake()

    # Plot
    plt.plot(hist)
    plt.title("Best Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("log(steps) + score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # plot top 10 history
    for i in range(10):
        plt.plot(hist_top[:, i], label=f'Top {i+1}', alpha=0.8)
    plt.title("Top 10 Individual Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    simulate(best)
