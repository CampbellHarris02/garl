import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from snake_game import SnakeGame
from model import DQN
from default_config import default_config
from visualize_dqn_plots import plot_rewards, plot_epsilon, plot_loss
import pygame  # type: ignore

def simulate_trained_model(model, config):
    """Simulate the Snake game using a trained DQN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()
    game = SnakeGame()
    state = game.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        state_tensor = torch.tensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            action = torch.argmax(model(state_tensor)).item() - 1

        state, done = game.step(action)

        # Draw
        screen.fill((0, 0, 0))
        fx, fy = game.food
        pygame.draw.rect(screen, (255, 0, 0), (fx * 75, fy * 75, 75, 75))  # tile size = 600/8 = 75
        for x, y in game.snake:
            pygame.draw.rect(screen, (0, 255, 0), (x * 75, y * 75, 75, 75))
        pygame.display.flip()
        clock.tick(10)

    print(f"Final score: {game.score}, Duration: {game.steps} steps")
    pygame.quit()


# DQN Training Function
def train_dqn(config, episodes=500, max_steps=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    game = SnakeGame()
    state_size = len(game.get_state())
    action_size = 3  # [-1, 0, 1]

    model = DQN(state_size, action_size, config).to(device)
    target_model = DQN(state_size, action_size, config).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    memory = deque(maxlen=config["replay_buffer_size"])
    batch_size = int(config["batch_size"])
    gamma = config["gamma"]

    epsilon = config["epsilon_start"]
    epsilon_min = config["epsilon_min"]
    epsilon_decay = config["epsilon_decay"]

    total_rewards = []
    epsilon_history = []
    loss_history = []

    for episode in range(episodes):
        state = game.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        episode_loss = 0

        for _ in range(max_steps):
            if random.random() < epsilon:
                action = random.choice([-1, 0, 1])
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
                    action_values = model(state_tensor)
                    action = torch.argmax(action_values).item() - 1

            next_state, done = game.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            reward = 1 if game.score > total_reward else -0.01
            total_reward = game.score
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                break

            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                states_tensor = torch.tensor(np.array(states)).float().to(device)
                actions_tensor = torch.tensor([a+1 for a in actions]).unsqueeze(1).to(device)
                rewards_tensor = torch.tensor(rewards).float().to(device)
                next_states_tensor = torch.tensor(np.array(next_states)).float().to(device)
                dones_tensor = torch.tensor(dones).float().to(device)

                q_values = model(states_tensor).gather(1, actions_tensor).squeeze()
                next_q_values = target_model(next_states_tensor).max(1)[0]
                target_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

                loss = loss_fn(q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()

        if episode % config["target_update_interval"] == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        epsilon_history.append(epsilon)
        loss_history.append(episode_loss)
        total_rewards.append(game.score + np.log(game.steps + 1))
        
        torch.save(model.state_dict(), "trained_dqn.pth")

    return np.mean(total_rewards[-10:]), total_rewards, epsilon_history, loss_history


if __name__ == "__main__":
    avg, total_rewards, epsilon_history, loss_history = train_dqn(default_config)
    plot_rewards(total_rewards)
    plot_epsilon(epsilon_history)
    plot_loss(loss_history)
    
    # Simulate with the trained model
    model = DQN(len(SnakeGame().get_state()), 3, default_config)
    model.load_state_dict(torch.load("trained_dqn.pth"))
    simulate_trained_model(model, default_config)