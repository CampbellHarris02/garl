import pygame
import numpy as np
import random
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

# Grid and snake setup
GRID_SIZE = 8
SCREEN_SIZE = 600
TILE_SIZE = SCREEN_SIZE // GRID_SIZE

# Directions: up, right, down, left
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

episodes = 300000
min_epsilon = 0.001


class SnakeGame:
    def __init__(self, r_step, r_food, r_death):
        self.r_step = r_step
        self.r_food = r_food
        self.r_death = r_death
        self.reset()

    def reset(self):
        self.snake = [(5, 5)]
        self.direction = random.randint(0, 3)
        self.place_food()
        self.score = 0
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

        if (
            head[0] < 0 or head[0] >= GRID_SIZE or
            head[1] < 0 or head[1] >= GRID_SIZE or
            head in self.snake
        ):
            return self.get_state(), self.r_death, True

        self.snake.insert(0, head)
        if head == self.food:
            self.score += 1
            self.place_food()
            return self.get_state(), self.r_food, False
        else:
            self.snake.pop()
            return self.get_state(), self.r_step, False

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_dx = np.sign(self.food[0] - head_x)
        food_dy = np.sign(self.food[1] - head_y)

        def is_danger(dir_offset):
            dx, dy = DIRS[(self.direction + dir_offset) % 4]
            nx, ny = head_x + dx, head_y + dy
            return (
                nx < 0 or nx >= GRID_SIZE or
                ny < 0 or ny >= GRID_SIZE or
                (nx, ny) in self.snake
            )

        return (
            food_dx, food_dy,
            self.direction,
            int(is_danger(0)),   # straight
            int(is_danger(1)),   # right
            int(is_danger(-1)),  # left
            len(self.snake)      # snake size
        )



def choose_action(q_table, state, epsilon):
    if random.random() < epsilon:
        return random.choice([-1, 0, 1])
    return np.argmax(q_table[state]) - 1



def evaluate_fitness(alpha, gamma, epsilon_decay, r_step, r_food, r_death, episodes=5000):
    game = SnakeGame(r_step, r_food, r_death)
    q_table = defaultdict(lambda: np.zeros(3))
    epsilon = 1.0
    scores = []

    for _ in range(episodes):
        state = game.reset()
        while True:
            action = choose_action(q_table, state, epsilon)
            next_state, reward, done = game.step(action)
            a_idx = action + 1
            q_table[state][a_idx] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][a_idx])
            state = next_state
            if done:
                scores.append(game.score)
                break
        epsilon = max(0.01, epsilon * epsilon_decay)

    return np.mean(scores)



def genetic_optimization(generations=10, pop_size=30):
    def random_individual():
        return {
            "alpha": np.random.uniform(0.01, 0.5),
            "gamma": np.random.uniform(0.8, 0.99),
            "epsilon_decay": np.random.uniform(0.999, 0.99999),
            "r_step": np.random.uniform(-0.5, 0.5),
            "r_food": np.random.uniform(0.5, 20.0),
            "r_death": np.random.uniform(-10.0, -0.1)
        }

    def mutate(ind):
        for key, std in {
            "alpha": 0.01, "gamma": 0.01, "epsilon_decay": 0.00001,
            "r_step": 0.01, "r_food": 0.1, "r_death": 0.1
        }.items():
            ind[key] += np.random.normal(0, std)
        ind["alpha"] = np.clip(ind["alpha"], 0.01, 0.5)
        ind["gamma"] = np.clip(ind["gamma"], 0.8, 0.99)
        ind["epsilon_decay"] = np.clip(ind["epsilon_decay"], 0.999, 0.99999)
        ind["r_step"] = np.clip(ind["r_step"], -0.1, 0.0)
        ind["r_food"] = np.clip(ind["r_food"], 0.1, 3.0)
        ind["r_death"] = np.clip(ind["r_death"], -5.0, -0.01)
        return ind

    population = [random_individual() for _ in range(pop_size)]
    best_fitnesses, avg_fitnesses = [], []

    for gen in range(generations):
        fitnesses = [evaluate_fitness(**ind) for ind in population]
        ranked = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
        best_fitness, avg_fitness = ranked[0][0], np.mean(fitnesses)
        best_fitnesses.append(best_fitness)
        avg_fitnesses.append(avg_fitness)

        print(f"Gen {gen:>2}: Best Score = {best_fitness:.2f} | Avg Score = {avg_fitness:.2f}")

        # Elitism and mutation
        population = [ranked[0][1], ranked[1][1]]
        while len(population) < pop_size:
            parent = random.choice(ranked[:5])[1]
            child = mutate(parent.copy())
            population.append(child)

    # Plot fitness
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitnesses, label="Best Fitness")
    plt.plot(avg_fitnesses, label="Average Fitness", linestyle="--")
    plt.xlabel("Generation")
    plt.ylabel("Average Game Score")
    plt.title("GA Optimization of Reward Function and Learning Parameters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return ranked[0][1]


def train(alpha, gamma, epsilon_decay, r_step, r_food, r_death):
    print(f"alpha: {alpha}, gamma: {gamma}, epsilon_decay: {epsilon_decay}, r_step: {r_step}, r_food: {r_food}, r_death: {r_death}")
    game = SnakeGame(r_food=r_food, r_step=r_step, r_death=r_death)
    q_table = defaultdict(lambda: np.zeros(3))
    episode_rewards = []
    episode_scores = []
    epsilon = 1.0

    for ep in range(episodes):
        state = game.reset()
        total_reward = 0
        total_score = 0

        while True:
            action = choose_action(q_table, state, epsilon)
            next_state, reward, done = game.step(action)
            total_reward += reward

            best_next = np.max(q_table[next_state])
            a_idx = action + 1
            q_table[state][a_idx] += alpha * (reward + gamma * best_next - q_table[state][a_idx])

            episode_score = game.score
            total_score += episode_score
            state = next_state
            if done:
                break
        
        
        episode_scores.append(total_score)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if ep % 2000 == 0:
            print(f"Episode {ep}, Reward: {total_reward}, Score: {total_score} Epsilon: {epsilon:.5f}")

    return q_table, episode_rewards, episode_scores



def simulate(q_table, alpha, gamma, epsilon_decay, r_step, r_food, r_death):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    clock = pygame.time.Clock()
    game = SnakeGame(r_food=r_food, r_step=r_step, r_death=r_death)
    state = game.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break

        screen.fill((0, 0, 0))
        fx, fy = game.food
        pygame.draw.rect(screen, (255, 0, 0), (fx * TILE_SIZE, fy * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        for x, y in game.snake:
            pygame.draw.rect(screen, (0, 255, 0), (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        action = np.argmax(q_table[state]) - 1
        state, _, done = game.step(action)

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
    print(f"Final score: {game.score}")


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

if __name__ == "__main__":
    print("Running Genetic Algorithm Optimization...")
    best_params = genetic_optimization()
    print(f"\nBest hyperparameters: {best_params}\n")
    
    print("Training with default parameters...")
    q1, rewards1, score1 = train(alpha=0.1, gamma=0.9, epsilon_decay=0.999, r_food=1, r_step=-0.01, r_death=-1)

    print("Training with best GA parameters...")
    q2, rewards2, score2 = train(**best_params)

    # Smooth with moving average (window size can be adjusted)
    smooth_sc1 = moving_average(score1, window_size=1000)
    smooth_sc2 = moving_average(score2, window_size=1000)
    smooth_rew1 = moving_average(rewards1, window_size=1000)
    smooth_rew2 = moving_average(rewards2, window_size=1000)

    # Plot smoothed scores
    plt.figure(figsize=(10, 4))
    plt.plot(smooth_sc1, label="Manual optimal Parameters", color='blue')
    plt.plot(smooth_sc2, label="GA optimal Parameters", color='orange')
    plt.title("Smoothed Score per Episode Comparison")
    plt.xlabel("Episode (×1000)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot smoothed rewards
    plt.figure(figsize=(10, 4))
    plt.plot(smooth_rew1, label="Manual optimal Parameters", color='blue')
    plt.plot(smooth_rew2, label="GA optimal Parameters", color='orange')
    plt.title("Smoothed Reward per Episode Comparison")
    plt.xlabel("Episode (×1000)")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    simulate(q1, **best_params)

