# final_snake.py
    """ 

    Returns:
        _type_: _description_
    """
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

episodes = 50000
min_epsilon = 0.001


# --- Snake game class ---
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


def train(alpha, gamma, epsilon_decay, r_food, r_step, r_death):
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

        if ep % 1000 == 0:
            print(f"Episode {ep}, Reward: {total_reward}, Score: {total_score} Epsilon: {epsilon:.5f}")

    return q_table, episode_rewards, episode_scores

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def simulate(q_table):
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

# Run
if __name__ == "__main__":
    print("Training with default parameters...")
    q1, rewards1, score1 = train(alpha=0.1, gamma=0.9, epsilon_decay=0.999, r_food=1, r_step=-0.01, r_death=-1)

    print("Training with custom parameters (gamma=0.85, alpha=0.01, epsilon_decay=0.99999)...")
    q2, rewards2, score2 = train(alpha=0.323043223295110567, gamma=0.858392013828569, epsilon_decay=0.9990327683881194, r_food=2.69, r_step=0, r_death=-4.11)


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

    simulate(q1)
