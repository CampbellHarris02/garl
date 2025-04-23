import pygame # type: ignore
import numpy as np
import random
import math
from collections import deque, defaultdict
import matplotlib.pyplot as plt # type: ignore
from rich.progress import track
from rich.console import Console
from rich.table import Table

# Constants
GRID_SIZE = 8
SCREEN_SIZE = 600
TILE_SIZE = SCREEN_SIZE // GRID_SIZE
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACTIONS = [-1, 0, 1]
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.1
EPISODES = 2000
STEPS_PER_GAME = 10000
FOOD_VALUE = 1

# Environment
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = deque([(5, 5)])
        self.direction = random.randint(0, 3)
        self.place_food()
        self.score = 0
        self.steps = 0
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

        if head[0] < 0 or head[0] >= GRID_SIZE or head[1] < 0 or head[1] >= GRID_SIZE or head in self.snake:
            return self.get_state(), -10, True

        self.snake.appendleft(head)
        if head == self.food:
            self.score += FOOD_VALUE
            self.place_food()
            reward = 10
        else:
            self.snake.pop()
            reward = -0.1

        return self.get_state(), reward, False

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
            len(self.snake), head_x, head_y
        )

# Q-Learning
q_table = defaultdict(lambda: [0.0 for _ in ACTIONS])
console = Console()
best_rewards = []

for episode in track(range(EPISODES), description="Training..."):
    game = SnakeGame()
    state = game.reset()
    total_reward = 0

    for _ in range(STEPS_PER_GAME):
        if random.random() < EPSILON:
            action_idx = random.randint(0, len(ACTIONS) - 1)
        else:
            action_idx = np.argmax(q_table[state])

        action = ACTIONS[action_idx]
        next_state, reward, done = game.step(action)
        total_reward += reward

        next_max = max(q_table[next_state])
        q_table[state][action_idx] += ALPHA * (reward + GAMMA * next_max - q_table[state][action_idx])

        state = next_state
        if done:
            break

    best_rewards.append(total_reward)

# Display top learned policy
table = Table(title="Top Learned Policy (50 states)")
table.add_column("State", style="dim", overflow="fold")
table.add_column("Best Action", justify="center")

for i, (state, actions) in enumerate(list(q_table.items())[:50]):
    best_action = ACTIONS[np.argmax(actions)]
    table.add_row(str(state), str(best_action))
console.print(table)

# Plot
plt.plot(best_rewards)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.show()

# Simulation
def simulate(policy):
    console.print("\n[bold cyan]Simulating learned policy...")
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

        action_idx = np.argmax(policy[state])
        action = ACTIONS[action_idx]
        state, _, done = game.step(action)

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
    print(f"Final score: {game.score}, Duration: {game.steps} steps")

simulate(q_table)